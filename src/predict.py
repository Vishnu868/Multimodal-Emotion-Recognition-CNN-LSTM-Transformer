"""Run a trained model on samples and show emotion probabilities + cross-modal attention.

    python src/predict.py --checkpoint outputs/<run>/model_fold0.pt --data data/processed/affec_T50.npz --n 5
    python src/predict.py --checkpoint outputs/<run>/model_fold0.pt --data synthetic --n 5

Output mirrors the slides: p = [0.05, 0.12, 0.75, 0.08] -> predicted emotion = "sad" (index 2).
"""
import argparse

import numpy as np
import torch

from data import load_affec, make_synthetic
from model import MultimodalEmotionNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", default="synthetic")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu")
    cfg, classes = ck["config"], ck["classes"]
    if args.data == "synthetic":
        data = make_synthetic(n=200, noise=cfg.get("noise", 3.0), seed=args.seed)
    else:
        data = load_affec(args.data, cfg["labels"])
    model = MultimodalEmotionNet(len(classes), cfg["modalities"], data["physio"].shape[-1],
                                 d_model=cfg["d_model"], n_heads=cfg["heads"], n_layers=cfg["layers"],
                                 dropout=cfg["dropout"])
    model.load_state_dict(ck["state_dict"])
    model.eval()

    idx = np.random.default_rng(args.seed).choice(len(data["labels"]), args.n, replace=False)
    t = lambda k: torch.from_numpy(data[k][idx])  # noqa: E731
    with torch.no_grad():
        logits, attn = model(t("eeg"), t("physio"), t("eye"), t("mask"), return_attention=True)
    probs = torch.softmax(logits, -1).numpy()
    names = model.modalities
    for k, i in enumerate(idx):
        p = probs[k]
        print(f"sample {i}: p = {np.round(p, 2).tolist()} -> predicted '{classes[p.argmax()]}' "
              f"(index {p.argmax()}), true '{classes[data['labels'][i]]}'")
        a = attn[k, -1].numpy()                       # last fusion layer, (tokens, tokens)
        rows = ", ".join(f"{q}->[" + " ".join(f"{names[j]}:{a[r, j]:.2f}" for j in range(len(names))) + "]"
                         for r, q in enumerate(names))
        print(f"    cross-modal attention: {rows}")


if __name__ == "__main__":
    main()
