"""Download the AFFEC multimodal emotion dataset from Zenodo.

AFFEC (Advancing Face-to-Face Emotion Communication), brAIn lab, IT University of Copenhagen.
BIDS layout, 72 participants, ~5k trials, CC licence. Only the modalities this project uses are
fetched: EEG (63 ch @ 256 Hz), eye-tracking gaze (16 ch @ 62 Hz), GSR/physio (Shimmer @ ~52 Hz)
and the core metadata (events, felt/perceived ratings, participants).

The archives are large (EEG alone is ~10 GB). Instead of downloading whole zips, individual
members are pulled with HTTP range requests (remotezip), so you can fetch a subset of subjects:

    python scripts/download_affec.py --out data/raw/affec                  # everything (~12 GB)
    python scripts/download_affec.py --out data/raw/affec --subjects 10    # first 10 subjects
    python scripts/download_affec.py --out data/raw/affec --modalities core gaze gsr

Re-running skips files that already exist with the correct size.
"""
import argparse
import concurrent.futures as cf
import os
import sys
import threading
import time
import urllib.request
import zipfile

from remotezip import RemoteZip

ZENODO = "https://zenodo.org/api/records/{rec}/files/{name}/content"
LATEST = "19408373"   # v2 (April 2026): core, eeg, gaze, pupil, cursor, videostream
V01 = "14794876"      # v0.1 (Feb 2025): the only version that ships gsr.zip

ARCHIVES = {
    "core": (LATEST, "core.zip"),
    "eeg": (LATEST, "eeg.zip"),
    "gaze": (LATEST, "gaze.zip"),
    "gsr": (V01, "gsr.zip"),
    "pupil": (LATEST, "pupil.zip"),
}
ROOT_FILES = ["participants.tsv", "participants.json", "dataset_description.json",
              "task-fer_events.json", "task-fer_beh.json", "README.md"]


_local = threading.local()


def _remote(url):
    """One RemoteZip per worker thread, so the central directory is read only once per thread."""
    cache = getattr(_local, "zips", None)
    if cache is None:
        cache = _local.zips = {}
    if url not in cache:
        cache[url] = RemoteZip(url)
    return cache[url]


def fetch_member(url, info, out_dir, retries=5):
    dest = os.path.join(out_dir, info.filename)
    if os.path.exists(dest) and os.path.getsize(dest) == info.file_size:
        return 0
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for attempt in range(retries):
        try:
            try:
                data = _remote(url).read(info.filename)
            except Exception:
                _local.zips.pop(url, None)                      # drop a broken handle, then retry
                raise
            tmp = dest + ".part"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, dest)
            return len(data)
        except Exception as e:  # network hiccup: back off and retry
            if attempt == retries - 1:
                print(f"  FAILED {info.filename}: {e}", file=sys.stderr)
                return 0
            time.sleep(2 ** attempt)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/raw/affec")
    ap.add_argument("--modalities", nargs="+", default=["core", "eeg", "gaze", "gsr"],
                    choices=list(ARCHIVES))
    ap.add_argument("--subjects", type=int, default=0, help="only the first N subjects (0 = all)")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for name in ROOT_FILES:
        dest = os.path.join(args.out, name)
        if not os.path.exists(dest):
            urllib.request.urlretrieve(ZENODO.format(rec=LATEST, name=name), dest)

    subjects = None
    for mod in args.modalities:
        rec, zname = ARCHIVES[mod]
        url = ZENODO.format(rec=rec, name=zname)
        if mod == "core":                                       # small (6 MB): fetch whole zip
            tmp = os.path.join(args.out, zname)
            urllib.request.urlretrieve(url, tmp)
            with zipfile.ZipFile(tmp) as z:
                members = [n for n in z.namelist() if n.startswith("sub-")]
                if args.subjects:
                    subjects = sorted({n.split("/")[0] for n in members})[: args.subjects]
                    members = [n for n in members if n.split("/")[0] in subjects]
                z.extractall(args.out, members)
            os.remove(tmp)
            print(f"[core] {len(members)} files extracted", flush=True)
            continue
        with RemoteZip(url) as rz:
            infos = [i for i in rz.infolist() if not i.is_dir() and i.filename.startswith("sub-")]
        if args.subjects:
            if subjects is None:
                subjects = sorted({i.filename.split("/")[0] for i in infos})[: args.subjects]
            infos = [i for i in infos if i.filename.split("/")[0] in subjects]
        total = sum(i.file_size for i in infos)
        print(f"[{mod}] {len(infos)} files, {total / 1e9:.2f} GB from {zname} (record {rec})", flush=True)
        done, t0 = 0, time.time()
        with cf.ThreadPoolExecutor(args.workers) as ex:
            futs = {ex.submit(fetch_member, url, i, args.out): i for i in infos}
            for k, fut in enumerate(cf.as_completed(futs), 1):
                done += futs[fut].file_size
                if k % 20 == 0 or k == len(infos):
                    rate = done / max(time.time() - t0, 1e-6) / 1e6
                    print(f"  {k}/{len(infos)} files  {done / 1e9:.2f}/{total / 1e9:.2f} GB  {rate:.1f} MB/s",
                          flush=True)
    print("done ->", args.out)


if __name__ == "__main__":
    main()
