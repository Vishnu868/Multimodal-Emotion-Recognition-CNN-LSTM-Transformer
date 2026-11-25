"""Figures for the report: dataset statistics and what each modality looks like after preprocessing.

    python scripts/make_figures.py --data data/processed/affec_T50.npz --out docs/figures
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
ORDER = ["neutral", "happy", "sad", "angry", "fear", "disgust"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/affec_T50.npz")
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    z = np.load(args.data)
    tt, mask = z["trial_type"], z["mask"]
    ok = mask.all(1)

    # 1. dataset overview
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    counts = [int((tt == c).sum()) for c in ORDER]
    ax[0].bar(ORDER, counts, color="#4C72B0")
    ax[0].set(title=f"Stimulus emotion per trial (n={len(tt)})", ylabel="trials")
    ax[1].bar(["EEG", "Physio", "Eye", "All three"], [*mask.mean(0) * 100, ok.mean() * 100], color="#55A868")
    ax[1].set(title="Modality availability", ylabel="% of trials", ylim=(0, 105))
    fv, fa = z["f_valence"], z["f_arousal"]
    for c, col in zip(ORDER, plt.cm.tab10.colors):
        m = (tt == c) & ~np.isnan(fv)
        ax[2].scatter(fv[m] + np.random.uniform(-.25, .25, m.sum()), fa[m] + np.random.uniform(-.25, .25, m.sum()),
                      s=4, alpha=.4, color=col, label=c)
    ax[2].axhline(5, color="grey", lw=.8)
    ax[2].axvline(5, color="grey", lw=.8)
    ax[2].set(title="Felt valence vs arousal (self-report, 1-9)", xlabel="valence", ylabel="arousal")
    ax[2].legend(markerscale=3, fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{args.out}/dataset_overview.png", dpi=130)
    plt.close(fig)

    # 2. modality examples (class averages over trials with all modalities)
    eeg = z["eeg"].astype(np.float32)
    phy = z["physio"].astype(np.float32)
    eye = z["eye"].astype(np.float32)
    cls = ORDER[:4]
    fig, ax = plt.subplots(3, 4, figsize=(15, 10.5))
    for k, c in enumerate(cls):
        m = ok & (tt == c)
        im = ax[0, k].imshow(eeg[m][:, :, 2].mean((0, 1)), cmap="RdBu_r", vmin=-.3, vmax=.3)
        ax[0, k].set(title=f"EEG alpha DE map - {c}", xticks=[], yticks=[])
        ax[1, k].imshow(eye[m][:, :, 0].mean((0, 1)), cmap="magma")
        ax[1, k].set(title=f"Mean gaze heatmap - {c}", xticks=[], yticks=[])
    fig.colorbar(im, ax=ax[0, 3], fraction=.046)
    t = np.arange(phy.shape[1]) * 0.1
    for j, (fi, name) in enumerate([(0, "GSR conductance (z)"), (2, "Skin temperature (z)")]):
        for c in cls:
            m = ok & (tt == c)
            ax[2, j].plot(t, phy[m][:, :, fi].mean(0), label=c)
        ax[2, j].set(title=name + " vs time from video onset", xlabel="s")
        ax[2, j].legend(fontsize=8)
    # one trial: EEG band power over time, and its gaze path
    i = int(np.flatnonzero(ok)[0])
    for b in range(5):
        ax[2, 2].plot(t, eeg[i, :, b].reshape(len(t), -1).mean(1), label=BANDS[b])
    ax[2, 2].set(title=f"Single trial: mean band DE ({tt[i]})", xlabel="s")
    ax[2, 2].legend(fontsize=8)
    ax[2, 3].imshow(eye[i, :, 0].sum(0), cmap="magma")
    ax[2, 3].set(title="Single trial: gaze over 5 s", xticks=[], yticks=[])
    fig.tight_layout()
    fig.savefig(f"{args.out}/modality_examples.png", dpi=130)
    plt.close(fig)
    print("figures ->", args.out)


if __name__ == "__main__":
    main()
