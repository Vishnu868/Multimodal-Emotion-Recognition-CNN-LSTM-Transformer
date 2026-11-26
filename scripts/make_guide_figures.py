"""Explanatory diagrams for the project guide (architecture, trial timeline, pipeline, ablation chart).

    python scripts/make_guide_figures.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = "docs/figures"
os.makedirs(OUT, exist_ok=True)
C = {"eeg": "#4C72B0", "phy": "#DD8452", "eye": "#55A868", "fus": "#8172B3", "cls": "#C44E52", "grey": "#8C8C8C"}


def box(ax, x, y, w, h, text, color, fs=9, alpha=0.18):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec=color, alpha=alpha, lw=0))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc="none", ec=color, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color="#222")


def arrow(ax, x1, y1, x2, y2, color="#555"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, color=color, lw=1.2))


def architecture():
    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    rows = [("eeg", 4.7, "EEG\n63 ch · 256 Hz", "5-band DE maps\n(50, 5, 9, 9)", "2D CNN\n3×3 conv ×2 + pool", "LSTM\n128", "f_EEG"),
            ("phy", 2.8, "GSR · temp · accel\n~52 Hz", "6 features/frame\n(50, 6)", "(normalised\ninput)", "2-layer LSTM\n128", "f_phys"),
            ("eye", 0.9, "Eye tracking\n62 Hz", "gaze heatmaps\n(50, 1, 16, 16)", "2D CNN\nconv+ReLU+maxpool", "GRU\n128", "f_eye")]
    for key, y, a, b, c, d, f in rows:
        col = C[key]
        box(ax, 0.1, y, 1.7, 1.0, a, col)
        box(ax, 2.2, y, 1.9, 1.0, b, col)
        box(ax, 4.5, y, 1.9, 1.0, c, col)
        box(ax, 6.8, y, 1.4, 1.0, d, col)
        for x1, x2 in ((1.8, 2.2), (4.1, 4.5), (6.4, 6.8)):
            arrow(ax, x1 + .02, y + .5, x2 - .02, y + .5)
        ax.text(8.45, y + .5, f, fontsize=9, va="center", color=col, fontweight="bold")
        arrow(ax, 8.95, y + .5, 9.35, 3.3)
    box(ax, 9.4, 2.3, 1.6, 2.0, "Transformer\nfusion\n\nmulti-head\nself-attention\nacross 3 tokens", C["fus"])
    box(ax, 11.4, 2.75, 1.5, 1.1, "FC → 256-D F′\nFC → 4\nsoftmax", C["cls"])
    arrow(ax, 11.0, 3.3, 11.4, 3.3)
    ax.text(12.15, 2.4, "neutral · happy\nsad · angry", ha="center", va="top", fontsize=9)
    for x, t in [(0.95, "1. Signals"), (3.15, "2. Preprocessing"), (5.45, "3. Spatial (CNN)"),
                 (7.5, "4. Temporal (RNN)"), (10.2, "5. Fusion"), (12.15, "6. Decision")]:
        ax.text(x, 6.0, t, ha="center", fontsize=10, fontweight="bold", color="#333")
    fig.tight_layout()
    fig.savefig(f"{OUT}/guide_architecture.png", dpi=150)
    plt.close(fig)


def timeline():
    fig, ax = plt.subplots(figsize=(12, 2.6))
    segs = [("fixation", 0.7, "#BBBBBB"), ("scenario text (≈17–35 s)", 6, "#D9D9D9"), ("fixation", 0.7, "#BBBBBB"),
            ("VIDEO\n≈3–4 s", 3.8, C["cls"]), ("last\nframe 1 s", 1.0, "#E8A0A0"),
            ("rate FELT\nvalence/arousal", 3, C["eeg"]), ("rate PERCEIVED\nvalence/arousal", 3, C["eye"])]
    x = 0
    for name, w, col in segs:
        ax.barh(0, w, left=x, color=col, edgecolor="white", height=0.6)
        ax.text(x + w / 2, 0, name, ha="center", va="center", fontsize=8.5,
                color="white" if col in (C["cls"], C["eeg"], C["eye"]) else "#222")
        x += w
    v0 = 0.7 + 6 + 0.7
    ax.annotate("", xy=(v0 + 4.8, -0.55), xytext=(v0, -0.55), arrowprops=dict(arrowstyle="<->", color="#222"))
    ax.text(v0 + 2.4, -0.75, "analysis window: 5 s from video onset = 50 steps × 0.1 s", ha="center", fontsize=9)
    ax.annotate("", xy=(v0, 0.5), xytext=(v0 - 0.7, 0.5), arrowprops=dict(arrowstyle="<->", color="#555"))
    ax.text(v0 - 0.35, 0.62, "1 s baseline", ha="center", fontsize=8)
    ax.set_xlim(-0.2, x + 0.2)
    ax.set_ylim(-1, 0.9)
    ax.axis("off")
    ax.set_title("One AFFEC trial (not to scale)", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT}/guide_trial_timeline.png", dpi=150)
    plt.close(fig)


def pipeline():
    fig, ax = plt.subplots(figsize=(13, 2.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 2)
    ax.axis("off")
    steps = ["Zenodo\n(AFFEC, 12 GB)", "download_affec.py\nraw BIDS files", "preprocess_affec.py\n→ affec_T50.npz",
             "train.py\nk-fold CV", "outputs/ results/\nmetrics · plots · models", "predict.py\nprobabilities"]
    cols = [C["grey"], C["eeg"], C["phy"], C["fus"], C["eye"], C["cls"]]
    for i, (s, c) in enumerate(zip(steps, cols)):
        box(ax, 0.1 + i * 2.17, 0.4, 1.85, 1.2, s, c, fs=9)
        if i:
            arrow(ax, i * 2.17 - 0.2, 1.0, i * 2.17 + 0.08, 1.0)
    fig.tight_layout()
    fig.savefig(f"{OUT}/guide_pipeline.png", dpi=150)
    plt.close(fig)


def ablation():
    names = ["eeg", "physio", "eye", "eeg+physio", "eeg+eye", "physio+eye", "eeg+physio+eye"]
    labels = ["EEG", "Physio", "Eye", "EEG+\nPhysio", "EEG+\nEye", "Physio+\nEye", "All\nthree"]
    base = "results/ablation_affec_stimulus4_subject"
    m = [json.load(open(f"{base}/{n}/metrics.json"))["summary"]["acc"] for n in names]
    fig, ax = plt.subplots(figsize=(9, 4))
    cols = [C["eeg"], C["phy"], C["eye"], C["grey"], C["grey"], C["fus"], C["cls"]]
    ax.bar(labels, [a[0] * 100 for a in m], yerr=[a[1] * 100 for a in m], color=cols, capsize=4)
    ax.axhline(25, ls="--", color="#333", lw=1)
    ax.text(6.45, 25.6, "chance 25%", ha="right", fontsize=9)
    for i, a in enumerate(m):
        ax.text(i, a[0] * 100 + a[1] * 100 + 0.6, f"{a[0] * 100:.1f}", ha="center", fontsize=9)
    ax.set(ylabel="test accuracy on unseen people (%)", ylim=(20, 40),
           title="Which signals help? (4 emotions, subject-independent, 3-fold)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/guide_ablation.png", dpi=150)
    plt.close(fig)


def eeg_grid():
    from features import EEG_GRID  # noqa: E402
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    for r, row in enumerate(EEG_GRID):
        for c, ch in enumerate(row):
            ax.add_patch(plt.Rectangle((c, 8 - r), 1, 1, fc="#DCE6F2" if ch else "#F5F5F5", ec="white", lw=2))
            if ch:
                ax.text(c + .5, 8 - r + .5, ch, ha="center", va="center", fontsize=8)
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 9)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("EEG electrodes placed on a 9×9 grid\n(top = forehead, bottom = back of head)", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{OUT}/guide_eeg_grid.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    architecture()
    timeline()
    pipeline()
    ablation()
    eeg_grid()
    print("guide figures ->", OUT)
