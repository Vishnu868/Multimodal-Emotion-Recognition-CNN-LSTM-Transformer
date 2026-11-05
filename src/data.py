"""Synthetic multimodal data generator and dataset utilities."""
import numpy as np
import torch
from torch.utils.data import Dataset

EMOTIONS = ["neutral", "happy", "sad", "angry", "fear"]


def make_synthetic(n: int = 2000, T: int = 50, eeg_ch: int = 63, physio_f: int = 5,
                   eye_hw: int = 16, n_classes: int = len(EMOTIONS), seed: int = 0):
    """Each emotion gets its own EEG band power, arousal level and gaze focus, plus noise."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_classes, n)
    t = np.linspace(0, 1, T)

    freqs = np.linspace(4, 30, n_classes)                  # theta ... beta
    eeg = rng.normal(0, 1, (n, T, eeg_ch)).astype(np.float32)
    eeg += np.sin(2 * np.pi * freqs[y][:, None, None] * t[None, :, None]) * \
        rng.uniform(0.5, 1.0, (n, 1, eeg_ch))

    arousal = np.linspace(-1, 1.5, n_classes)[y]
    physio = rng.normal(0, 1, (n, T, physio_f)).astype(np.float32)
    physio[:, :, 0] += arousal[:, None] * np.linspace(0.5, 1.5, T)  # GSR rises with arousal
    physio[:, :, 1] += 0.3 * arousal[:, None]                         # temperature

    yy, xx = np.mgrid[0:eye_hw, 0:eye_hw]
    centres = rng.uniform(3, eye_hw - 3, (n_classes, 2))
    eye = np.zeros((n, T, 1, eye_hw, eye_hw), dtype=np.float32)
    for i in range(n):
        cy, cx = centres[y[i]] + rng.normal(0, 1, 2)
        jitter = rng.normal(0, 0.8, (T, 2))
        d = (yy[None] - (cy + jitter[:, 0, None, None])) ** 2 + \
            (xx[None] - (cx + jitter[:, 1, None, None])) ** 2
        eye[i, :, 0] = np.exp(-d / 8.0) + rng.normal(0, 0.1, (T, eye_hw, eye_hw))
    return eeg, physio, eye, y.astype(np.int64)


def load_real_dataset(path: str):
    """Template: return eeg (N,T,63), physio (N,T,F), eye (N,T,1,H,W), labels (N,)."""
    raise NotImplementedError("Implement loading/segmentation for your recorded dataset here.")


class MultimodalDataset(Dataset):
    def __init__(self, eeg, physio, eye, labels):
        self.eeg = torch.from_numpy(eeg)
        self.physio = torch.from_numpy(physio)
        self.eye = torch.from_numpy(eye)
        self.y = torch.from_numpy(labels)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.eeg[i], self.physio[i], self.eye[i], self.y[i]
