"""Signal processing for the three modalities.

Every trial is cut into the same window (default: 5 s from video onset, i.e. the emotional video clip
plus its last frame), split into T = 50 frames of 0.1 s. All modalities share the BIDS `onset`
clock (seconds since run start), so the same frame centres are used for EEG, physio and eye.

EEG    -> per-frame differential entropy (DE) in 5 frequency bands, placed on a 9x9 scalp grid
          (T, 5, 9, 9). The 2D CNN convolves 3x3 neighbourhoods of electrodes across bands, which is
          the convolution in the slides: Y_ijk = sum_mnc W_mnck X_(i+m)(j+n)c + b_k.
Physio -> GSR conductance, its first derivative, skin temperature, 3-axis accelerometer (T, 6).
Eye    -> gaze heatmap per frame from the "best point of gaze" (T, 1, 16, 16).
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt

BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}

# 9x9 approximation of the 10-10 montage (top = frontal, bottom = occipital, left = left hemisphere).
# AFFEC's 63 channels: 62 scalp electrodes land on this grid; A1 (left earlobe = recording reference)
# is deliberately left out.
EEG_GRID = [
    [None, None, None, "Fp1", "Fpz", "Fp2", None, None, None],
    ["F9", "AF7", None, "AF3", "AFz", "AF4", None, "AF8", "F10"],
    ["F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8"],
    ["FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8"],
    ["T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8"],
    ["TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8"],
    ["P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8"],
    ["PO9", "PO7", "PO5", "PO3", "POz", "PO4", "PO6", "PO8", "PO10"],
    [None, None, "O9", "O1", "Oz", "O2", "O10", None, None],
]
GRID_POS = {ch.lower(): (r, c) for r, row in enumerate(EEG_GRID) for c, ch in enumerate(row) if ch}

PHYSIO_FEATURES = ["gsr_conductance", "gsr_slope", "temperature", "acc_x", "acc_y", "acc_z"]


def frame_centres(start: float, duration: float, T: int) -> np.ndarray:
    step = duration / T
    return start + step * (np.arange(T) + 0.5)


# ----------------------------------------------------------------------------------------- EEG
def eeg_band_filter(data: np.ndarray, sfreq: float) -> np.ndarray:
    """Zero-phase band-pass of a continuous run. data (C, N) -> (B, C, N)."""
    out = np.empty((len(BANDS),) + data.shape, dtype=np.float32)
    for b, (lo, hi) in enumerate(BANDS.values()):
        sos = butter(4, [lo, hi], btype="bandpass", fs=sfreq, output="sos")
        out[b] = sosfiltfilt(sos, data, axis=-1)
    return out


def eeg_de_frames(banded: np.ndarray, sfreq: float, centres: np.ndarray, win: float = 0.5) -> np.ndarray:
    """Differential entropy 0.5*log(2*pi*e*var) in a sliding window around each frame centre.
    banded (B, C, N) -> (T, B, C)."""
    half = int(round(win * sfreq / 2))
    n = banded.shape[-1]
    feats = np.empty((len(centres), banded.shape[0], banded.shape[1]), dtype=np.float32)
    for t, c in enumerate(centres):
        i = int(round(c * sfreq))
        seg = banded[..., max(i - half, 0): min(i + half, n)]
        var = seg.var(axis=-1) if seg.shape[-1] > 1 else np.zeros(banded.shape[:2])
        feats[t] = 0.5 * np.log(2 * np.pi * np.e * (var + 1e-12))
    return feats


def to_grid(x: np.ndarray, ch_names) -> np.ndarray:
    """(..., C) electrode vector -> (..., 9, 9) topographic map; unplaced cells stay 0."""
    out = np.zeros(x.shape[:-1] + (9, 9), dtype=np.float32)
    for k, ch in enumerate(ch_names):
        pos = GRID_POS.get(ch.lower())
        if pos is not None:
            out[..., pos[0], pos[1]] = x[..., k]
    return out


# -------------------------------------------------------------------------------------- physio
def physio_frames(onset: np.ndarray, cols: dict, centres: np.ndarray) -> np.ndarray:
    """Average each signal within its 0.1 s frame (falls back to interpolation for empty frames)."""
    gsr = cols["GSR_Conductance_cal"]
    slope = np.gradient(gsr, onset) if len(onset) > 1 else np.zeros_like(gsr)
    signals = [gsr, slope, cols["Temperature_cal"], cols["Low_Noise_Accelerometer_X_cal"],
               cols["Low_Noise_Accelerometer_Y_cal"], cols["Low_Noise_Accelerometer_Z_cal"]]
    half = (centres[1] - centres[0]) / 2 if len(centres) > 1 else 0.05
    out = np.empty((len(centres), len(signals)), dtype=np.float32)
    for f, s in enumerate(signals):
        for t, c in enumerate(centres):
            m = (onset >= c - half) & (onset < c + half)
            out[t, f] = s[m].mean() if m.any() else np.interp(c, onset, s)
    return out


# ----------------------------------------------------------------------------------------- eye
def gaze_heatmaps(onset, x, y, valid, centres, hw: int = 16, sigma: float = 1.0,
                  crop=(0.0, 1.0)) -> np.ndarray:
    """Gaussian-splatted gaze density per frame, normalised to sum 1 (all-zero when no valid gaze)."""
    half = (centres[1] - centres[0]) / 2 if len(centres) > 1 else 0.05
    lo, hi = crop
    grid = (np.arange(hw) + 0.5) / hw
    out = np.zeros((len(centres), 1, hw, hw), dtype=np.float32)
    for t, c in enumerate(centres):
        m = (onset >= c - half) & (onset < c + half) & (valid > 0)
        if not m.any():
            continue
        gx = np.clip((x[m] - lo) / (hi - lo), 0, 1)
        gy = np.clip((y[m] - lo) / (hi - lo), 0, 1)
        dx = (grid[None, :] - gx[:, None]) * hw          # (S, W)
        dy = (grid[None, :] - gy[:, None]) * hw          # (S, H)
        h = np.einsum("sh,sw->hw", np.exp(-dy ** 2 / (2 * sigma ** 2)), np.exp(-dx ** 2 / (2 * sigma ** 2)))
        out[t, 0] = h / max(h.sum(), 1e-8)
    return out
