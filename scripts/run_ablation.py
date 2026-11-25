"""Modality ablation: train every modality subset and tabulate test performance.

    python scripts/run_ablation.py --data data/processed/affec_T50.npz --labels stimulus4 --split subject --folds 5

Writes outputs/ablation_<tag>/results.md with mean +- std accuracy / macro-F1 per subset.
Any extra arguments are passed straight to src/train.py.
"""
import argparse
import itertools
import json
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
MODS = ["eeg", "physio", "eye"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="synthetic")
    ap.add_argument("--labels", default="stimulus4")
    ap.add_argument("--split", default="subject")
    ap.add_argument("--folds", type=int, default=5)
    args, extra = ap.parse_known_args()
    if args.data != "synthetic":
        args.data = os.path.abspath(args.data)

    tag = "synthetic" if args.data == "synthetic" else f"affec_{args.labels}"
    out_root = os.path.join(ROOT, "outputs", f"ablation_{tag}_{args.split}")
    os.makedirs(out_root, exist_ok=True)
    rows = []
    subsets = [c for r in (1, 2, 3) for c in itertools.combinations(MODS, r)]
    for sub in subsets:
        out = os.path.join(out_root, "+".join(sub))
        cmd = [sys.executable, os.path.join(ROOT, "src", "train.py"), "--data", args.data, "--labels", args.labels,
               "--split", args.split, "--folds", str(args.folds), "--modalities", *sub, "--out", out, *extra]
        print(">>", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        s = json.load(open(os.path.join(out, "metrics.json")))["summary"]
        rows.append((" + ".join(m.upper() if m == "eeg" else m.capitalize() for m in sub), s))

    chance = rows[0][1]["chance"]
    lines = [f"# Modality ablation ({tag}, {args.split} split, {args.folds}-fold)", "",
             f"Chance level: {chance:.3f}", "",
             "| Modalities | Accuracy | Macro-F1 | Balanced acc. |", "|---|---|---|---|"]
    for name, s in rows:
        lines.append(f"| {name} | {s['acc'][0]:.3f} ± {s['acc'][1]:.3f} | {s['f1'][0]:.3f} ± {s['f1'][1]:.3f} | "
                     f"{s['balanced_acc'][0]:.3f} ± {s['balanced_acc'][1]:.3f} |")
    open(os.path.join(out_root, "results.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
