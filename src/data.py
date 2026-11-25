"""Datasets: synthetic multimodal generator + the preprocessed AFFEC dataset.

Tensor shapes (per sample, T = 50 timesteps):
    eeg    (T, 5, 9, 9)     differential entropy in 5 bands on a 9x9 electrode grid
    physio (T, 6)           GSR conductance, GSR slope, temperature, accelerometer x/y/z
    eye    (T, 1, 16, 16)   gaze heatmaps
    mask   (3,)             which modalities are present (EEG, physio, eye)
"""
import numpy as np
import torch
from torch.utils.data import Dataset

# Class order follows the project slides: p = [0.05, 0.12, 0.75, 0.08] -> "Sad" (index 2).
EMOTIONS4 = ["neutral", "happy", "sad", "angry"]
EMOTIONS6 = ["neutral", "happy", "sad", "angry", "fear", "disgust"]
MODALITIES = ("eeg", "physio", "eye")

LABEL_SCHEMES = {
    # stimulus emotion category of the video (AFFEC `trial_type`), restricted to the 4 slide classes
    "stimulus4": EMOTIONS4,
    # all six stimulus categories
    "stimulus6": EMOTIONS6,
    # self-reported valence/arousal (1-9) mapped to 4 quadrants named after their prototypical emotion:
    # high-valence/low-arousal = neutral (calm), HV/HA = happy, LV/LA = sad, LV/HA = angry
    "felt_va4": EMOTIONS4,
    "perceived_va4": EMOTIONS4,
}


def va_quadrant(valence, arousal, threshold=5.0):
    """neutral(HV-LA)=0, happy(HV-HA)=1, sad(LV-LA)=2, angry(LV-HA)=3; NaN ratings -> -1."""
    hv, ha = valence > threshold, arousal > threshold
    q = np.where(hv, np.where(ha, 1, 0), np.where(ha, 3, 2))
    return np.where(np.isnan(valence) | np.isnan(arousal), -1, q)


def load_affec(path: str, scheme: str = "stimulus4", require_all: bool = True):
    """Load the .npz written by scripts/preprocess_affec.py and build labels for `scheme`.
    Returns a dict with eeg, physio, eye, mask, labels, groups (subject ids), classes."""
    z = np.load(path, allow_pickle=False)
    classes = LABEL_SCHEMES[scheme]
    if scheme.startswith("stimulus"):
        lut = {c: i for i, c in enumerate(classes)}
        y = np.array([lut.get(t, -1) for t in z["trial_type"]])
    elif scheme == "felt_va4":
        y = va_quadrant(z["f_valence"], z["f_arousal"])
    elif scheme == "perceived_va4":
        y = va_quadrant(z["p_valence"], z["p_arousal"])
    else:
        raise ValueError(f"unknown label scheme {scheme!r}; choose from {list(LABEL_SCHEMES)}")

    mask = z["mask"]
    keep = y >= 0
    keep &= mask.all(1) if require_all else mask.any(1)
    return dict(
        eeg=z["eeg"][keep].astype(np.float32), physio=z["physio"][keep].astype(np.float32),
        eye=z["eye"][keep].astype(np.float32), mask=mask[keep], labels=y[keep].astype(np.int64),
        groups=z["subject"][keep], classes=classes,
    )


def make_synthetic(n: int = 2000, T: int = 50, n_classes: int = 4, physio_f: int = 6, eye_hw: int = 16,
                   noise: float = 1.0, seed: int = 0):
    """Synthetic data with the exact tensor shapes of the real pipeline (slides, objective 4).
    Each emotion gets its own frontal-alpha asymmetry / band-power pattern, arousal-driven GSR rise
    and a preferred gaze region, all buried in noise."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_classes, n)
    t = np.linspace(0, 1, T)

    eeg = rng.normal(0, noise, (n, T, 5, 9, 9)).astype(np.float32)
    band_pattern = rng.normal(0, 0.6, (n_classes, 5, 9, 9)).astype(np.float32)
    eeg += band_pattern[y][:, None] * (0.5 + t)[None, :, None, None, None]

    arousal = np.linspace(-1, 1.5, n_classes)[rng.permutation(n_classes)][y]
    physio = rng.normal(0, noise, (n, T, physio_f)).astype(np.float32)
    physio[:, :, 0] += arousal[:, None] * np.linspace(0.2, 1.5, T)   # GSR rises with arousal
    physio[:, :, 1] += arousal[:, None] * 0.5                          # its slope
    physio[:, :, 2] += 0.3 * arousal[:, None]                          # temperature

    yy, xx = np.mgrid[0:eye_hw, 0:eye_hw]
    centres = rng.uniform(3, eye_hw - 3, (n_classes, 2))
    eye = np.zeros((n, T, 1, eye_hw, eye_hw), dtype=np.float32)
    for i in range(n):
        cy, cx = centres[y[i]] + rng.normal(0, 1.5, 2)
        jy, jx = rng.normal(0, 1.5, (2, T))
        d = (yy[None] - (cy + jy)[:, None, None]) ** 2 + (xx[None] - (cx + jx)[:, None, None]) ** 2
        hm = np.exp(-d / 4.0) + np.abs(rng.normal(0, 0.15 * noise, (T, eye_hw, eye_hw)))
        eye[i, :, 0] = hm / hm.sum(axis=(1, 2), keepdims=True)
    groups = rng.integers(0, 40, n).astype(str)                        # fake "subjects"
    return dict(eeg=eeg, physio=physio, eye=eye, mask=np.ones((n, 3), bool),
                labels=y.astype(np.int64), groups=groups, classes=EMOTIONS4[:n_classes]
                if n_classes <= 4 else [f"class{i}" for i in range(n_classes)])


class MultimodalDataset(Dataset):
    def __init__(self, data: dict, idx=None, modality_dropout: float = 0.0):
        idx = np.arange(len(data["labels"])) if idx is None else np.asarray(idx)
        self.eeg = torch.from_numpy(data["eeg"][idx])
        self.physio = torch.from_numpy(data["physio"][idx])
        self.eye = torch.from_numpy(data["eye"][idx])
        self.mask = torch.from_numpy(data["mask"][idx].astype(bool))
        self.y = torch.from_numpy(data["labels"][idx])
        self.modality_dropout = modality_dropout

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        mask = self.mask[i].clone()
        if self.modality_dropout > 0:                      # randomly hide one modality (robustness)
            drop = torch.rand(3) < self.modality_dropout
            if (mask & ~drop).any():
                mask &= ~drop
        return self.eeg[i], self.physio[i], self.eye[i], mask, self.y[i]
