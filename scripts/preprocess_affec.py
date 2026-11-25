"""Turn the raw AFFEC BIDS files into model-ready tensors (one .npz).

For every trial of the facial-emotion-recognition task:
  1. find the video onset in `sub-*_task-fer_run-*_events.tsv` (flag == "video")
  2. take a `--duration` second window from that onset (default 5 s = video + last frame)
     and split it into `--timesteps` frames (default 50, as in the project slides)
  3. extract EEG band DE maps, physio (GSR/temp/accel) and gaze heatmaps per frame (src/features.py)
  4. baseline-correct EEG and physio with the 1 s fixation-cross period before the video
  5. z-score every EEG/physio feature per subject (x' = (x - mu_f) / sigma_f)
and attach the labels from `beh/*_beh.tsv` (felt/perceived valence & arousal, 1-9) and the
stimulus emotion category from `trial_type` (angry, disgust, fear, happy, neutral, sad).

    python scripts/preprocess_affec.py --raw data/raw/affec --out data/processed/affec_T50.npz
"""
import argparse
import glob
import gzip
import io
import json
import os
import sys
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from features import (PHYSIO_FEATURES, eeg_band_filter, eeg_de_frames, frame_centres,  # noqa: E402
                      gaze_heatmaps, physio_frames, to_grid)

ARGS = None


def read_physio(tsv_gz: str):
    """Headerless BIDS physio file; column names live in the JSON sidecar."""
    meta = json.load(open(tsv_gz.replace(".tsv.gz", ".json")))
    with gzip.open(tsv_gz, "rt") as f:
        txt = f.read()
    df = pd.read_csv(io.StringIO(txt), sep="\t", header=None, low_memory=False)
    if not np.issubdtype(df.dtypes.iloc[0], np.number):            # file carries its own header row
        df = pd.read_csv(io.StringIO(txt), sep="\t")
    else:
        df.columns = meta["Columns"][: df.shape[1]]
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.dropna(subset=["onset"]).sort_values("onset").reset_index(drop=True)


def process_run(raw, sub, run, ev, beh):
    import mne
    T, dur, base = ARGS.timesteps, ARGS.duration, ARGS.baseline
    trials = []
    for trial, g in ev.groupby("trial"):
        vid = g[g.flag == "video"]
        if vid.empty:
            continue
        b = beh[(beh.run == run) & (beh.trial == trial)]
        trials.append(dict(trial=int(trial), t0=float(vid.onset.iloc[0]),
                           trial_type=str(vid.trial_type.iloc[0]), stim_file=str(vid.stim_file.iloc[0]),
                           **{k: (float(b[k].iloc[0]) if len(b) else np.nan)
                              for k in ["f_emotion_v", "f_emotion_a", "p_emotion_v", "p_emotion_a"]}))
    if not trials:
        return []
    out = [dict(subject=sub, run=run, **t,
                eeg=np.zeros((T, 5, 9, 9), np.float32), physio=np.zeros((T, len(PHYSIO_FEATURES)), np.float32),
                eye=np.zeros((T, 1, ARGS.eye_hw, ARGS.eye_hw), np.float32), mask=np.zeros(3, bool))
           for t in trials]
    stem = f"{raw}/{sub}/%s/{sub}_task-fer_run-{run}_%s"

    # ---- EEG
    edf = stem % ("eeg", "eeg.edf")
    if os.path.exists(edf):
        try:
            r = mne.io.read_raw_edf(edf, preload=True, verbose=False)
            sf, data = r.info["sfreq"], r.get_data() * 1e6           # volts -> microvolts
            banded = eeg_band_filter(data, sf)
            for o in out:
                if o["t0"] + dur > data.shape[1] / sf:
                    continue
                stim = eeg_de_frames(banded, sf, frame_centres(o["t0"], dur, T))
                if base > 0:
                    stim -= eeg_de_frames(banded, sf, frame_centres(o["t0"] - base, base, 10)).mean(0)
                o["eeg"] = to_grid(stim, r.ch_names)
                o["mask"][0] = np.isfinite(o["eeg"]).all()
        except Exception as e:
            warnings.warn(f"{sub} run {run} EEG: {e}")

    # ---- physio (Shimmer GSR + temperature + accelerometer)
    gsr = stem % ("beh", "recording-gsr_physio.tsv.gz")
    if os.path.exists(gsr):
        try:
            df = read_physio(gsr)
            on, cols = df.onset.to_numpy(), {c: df[c].to_numpy(float) for c in df.columns}
            for o in out:
                if o["t0"] + dur > on[-1] or o["t0"] - base < on[0]:
                    continue
                stim = physio_frames(on, cols, frame_centres(o["t0"], dur, T))
                if base > 0:
                    stim -= physio_frames(on, cols, frame_centres(o["t0"] - base, base, 10)).mean(0)
                o["physio"] = stim
                o["mask"][1] = np.isfinite(stim).all()
        except Exception as e:
            warnings.warn(f"{sub} run {run} GSR: {e}")

    # ---- eye tracking (Gazepoint, best point-of-gaze)
    gaze = stem % ("beh", "recording-gaze_physio.tsv.gz")
    if os.path.exists(gaze):
        try:
            df = read_physio(gaze)
            on = df.onset.to_numpy()
            for o in out:
                if o["t0"] + dur > on[-1]:
                    continue
                hm = gaze_heatmaps(on, df.BPOGX.to_numpy(), df.BPOGY.to_numpy(), df.BPOGV.fillna(0).to_numpy(),
                                   frame_centres(o["t0"], dur, T), hw=ARGS.eye_hw)
                o["eye"] = hm
                # require valid gaze in at least half of the frames
                o["mask"][2] = (hm.reshape(T, -1).sum(1) > 0).mean() >= 0.5
        except Exception as e:
            warnings.warn(f"{sub} run {run} gaze: {e}")
    return out


def process_subject(sub):
    raw = ARGS.raw
    evs = sorted(glob.glob(f"{raw}/{sub}/{sub}_task-fer_run-*_events.tsv"))
    behs = sorted(glob.glob(f"{raw}/{sub}/beh/{sub}_task-fer_run-*_beh.tsv"))
    if not evs:
        return []
    beh = pd.concat([pd.read_csv(b, sep="\t") for b in behs]) if behs else \
        pd.DataFrame(columns=["run", "trial", "f_emotion_v", "f_emotion_a", "p_emotion_v", "p_emotion_a"])
    rows = []
    for e in evs:
        ev = pd.read_csv(e, sep="\t")
        run = int(e.split("_run-")[1].split("_")[0])
        rows += process_run(raw, sub, run, ev, beh)
    if not rows:
        return rows
    # per-subject z-score (only over trials where the modality is present)
    for key, m in (("eeg", 0), ("physio", 1)):
        ok = [r for r in rows if r["mask"][m]]
        if len(ok) < 2:
            continue
        x = np.stack([r[key] for r in ok])                      # (n, T, ...)
        mu, sd = x.mean(axis=(0, 1)), x.std(axis=(0, 1)) + 1e-6  # per feature (empty grid cells stay 0)
        for r in ok:
            r[key] = ((r[key] - mu) / sd).astype(np.float32)
    print(f"  {sub}: {len(rows)} trials, modality coverage {np.mean([r['mask'] for r in rows], 0).round(2)}",
          flush=True)
    return rows


def main():
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw/affec")
    ap.add_argument("--out", default="data/processed/affec_T50.npz")
    ap.add_argument("--timesteps", type=int, default=50)
    ap.add_argument("--duration", type=float, default=5.0, help="seconds from video onset")
    ap.add_argument("--baseline", type=float, default=1.0, help="pre-video baseline seconds (0 = off)")
    ap.add_argument("--eye-hw", type=int, default=16)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--include-excluded", action="store_true",
                    help="keep participants flagged ex=1 in participants.tsv")
    ARGS = ap.parse_args()

    subs = sorted(os.path.basename(p) for p in glob.glob(f"{ARGS.raw}/sub-*"))
    pt = os.path.join(ARGS.raw, "participants.tsv")
    if os.path.exists(pt) and not ARGS.include_excluded:
        part = pd.read_csv(pt, sep="\t")
        excluded = set(part.loc[part["ex"] == 1, "participant_id"])
        if excluded:
            print("excluding participants flagged ex=1:", sorted(excluded))
        subs = [s for s in subs if s not in excluded]
    print(f"{len(subs)} subjects", flush=True)

    with Pool(ARGS.workers) as pool:
        rows = [r for rs in pool.map(process_subject, subs) for r in rs]
    if not rows:
        sys.exit("no trials found - check --raw")

    os.makedirs(os.path.dirname(ARGS.out) or ".", exist_ok=True)
    np.savez_compressed(
        ARGS.out,
        eeg=np.stack([r["eeg"] for r in rows]).astype(np.float16),
        physio=np.stack([r["physio"] for r in rows]).astype(np.float16),
        eye=np.stack([r["eye"] for r in rows]).astype(np.float16),
        mask=np.stack([r["mask"] for r in rows]),
        subject=np.array([r["subject"] for r in rows]),
        run=np.array([r["run"] for r in rows]),
        trial=np.array([r["trial"] for r in rows]),
        trial_type=np.array([r["trial_type"] for r in rows]),
        stim_file=np.array([r["stim_file"] for r in rows]),
        f_valence=np.array([r["f_emotion_v"] for r in rows], np.float32),
        f_arousal=np.array([r["f_emotion_a"] for r in rows], np.float32),
        p_valence=np.array([r["p_emotion_v"] for r in rows], np.float32),
        p_arousal=np.array([r["p_emotion_a"] for r in rows], np.float32),
        physio_features=np.array(PHYSIO_FEATURES),
        config=json.dumps(vars(ARGS)),
    )
    m = np.stack([r["mask"] for r in rows])
    print(f"saved {len(rows)} trials -> {ARGS.out}")
    print(f"  EEG {m[:, 0].mean():.1%}  physio {m[:, 1].mean():.1%}  eye {m[:, 2].mean():.1%}  "
          f"all three {m.all(1).mean():.1%}")
    print("  stimulus classes:", pd.Series([r["trial_type"] for r in rows]).value_counts().to_dict())


if __name__ == "__main__":
    main()
