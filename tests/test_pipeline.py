"""Fast checks for shapes, masking and feature extraction.  Run: pytest -q"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data import MultimodalDataset, make_synthetic, va_quadrant  # noqa: E402
from features import (GRID_POS, eeg_band_filter, eeg_de_frames, frame_centres, gaze_heatmaps,  # noqa: E402
                      physio_frames, to_grid)
from model import MultimodalEmotionNet  # noqa: E402

AFFEC_CHANNELS = ("Fp1 Fpz Fp2 AF7 AF3 AF4 AF8 F7 F5 F3 F1 Fz F2 F4 F6 F8 FT7 FC5 FC3 FC1 FCz FC2 FC4 FC6 FT8 "
                  "T7 C5 C3 C1 Cz C2 C4 C6 T8 TP7 CP5 CP3 CP1 CPz CP2 CP4 CP6 TP8 P7 P5 P3 P1 Pz P2 P4 P6 P8 "
                  "PO7 PO3 POz PO4 PO8 O1 Oz O2 F9 F10").split()   # + A1 (reference, not placed)


def test_model_shapes_and_attention():
    d = make_synthetic(n=8, seed=0)
    model = MultimodalEmotionNet(n_classes=4).eval()
    t = lambda k: torch.from_numpy(d[k])  # noqa: E731
    logits, attn = model(t("eeg"), t("physio"), t("eye"), t("mask"), return_attention=True)
    assert logits.shape == (8, 4)
    assert attn.shape == (8, 2, 3, 3)
    assert torch.allclose(attn.sum(-1), torch.ones(8, 2, 3), atol=1e-5)


def test_missing_modality_is_ignored():
    d = make_synthetic(n=4, seed=0)
    model = MultimodalEmotionNet(n_classes=4).eval()
    t = lambda k: torch.from_numpy(d[k])  # noqa: E731
    mask = torch.tensor([[True, True, False]] * 4)
    a = model(t("eeg"), t("physio"), t("eye"), mask)
    b = model(t("eeg"), t("physio"), torch.randn_like(t("eye")), mask)  # eye content must not matter
    assert torch.allclose(a, b, atol=1e-5)


def test_unimodal_model():
    d = make_synthetic(n=4, seed=0)
    model = MultimodalEmotionNet(n_classes=4, modalities=["physio"])
    out = model(*(torch.from_numpy(d[k]) for k in ("eeg", "physio", "eye", "mask")))
    assert out.shape == (4, 4)


def test_dataset_item():
    d = make_synthetic(n=10, seed=0)
    item = MultimodalDataset(d, modality_dropout=0.5)[0]
    assert item[0].shape == (50, 5, 9, 9) and item[1].shape == (50, 6) and item[2].shape == (50, 1, 16, 16)
    assert item[3].any()


def test_all_affec_channels_on_grid():
    missing = [c for c in AFFEC_CHANNELS if c.lower() not in GRID_POS]
    assert not missing, missing
    assert len({GRID_POS[c.lower()] for c in AFFEC_CHANNELS}) == len(AFFEC_CHANNELS)


def test_eeg_features():
    sf, n = 256.0, 256 * 10
    t = np.arange(n) / sf
    data = np.stack([np.sin(2 * np.pi * 10 * t), np.sin(2 * np.pi * 20 * t)])  # alpha vs beta
    de = eeg_de_frames(eeg_band_filter(data, sf), sf, frame_centres(2.0, 5.0, 50))
    assert de.shape == (50, 5, 2)
    assert de[:, 2, 0].mean() > de[:, 2, 1].mean()      # channel 0 dominates alpha
    assert de[:, 3, 1].mean() > de[:, 3, 0].mean()      # channel 1 dominates beta
    assert to_grid(de, ["Cz", "Oz"]).shape == (50, 5, 9, 9)


def test_physio_and_gaze_frames():
    on = np.arange(0, 10, 1 / 52)
    cols = {k: np.random.rand(len(on)) for k in ["GSR_Conductance_cal", "Temperature_cal",
                                                  "Low_Noise_Accelerometer_X_cal", "Low_Noise_Accelerometer_Y_cal",
                                                  "Low_Noise_Accelerometer_Z_cal"]}
    c = frame_centres(1.0, 5.0, 50)
    assert physio_frames(on, cols, c).shape == (50, 6)
    on = np.arange(0, 10, 1 / 62)
    hm = gaze_heatmaps(on, np.full(len(on), 0.25), np.full(len(on), 0.75), np.ones(len(on)), c)
    assert hm.shape == (50, 1, 16, 16)
    assert np.allclose(hm.sum((1, 2, 3)), 1)
    r, col = np.unravel_index(hm[0, 0].argmax(), (16, 16))
    assert r > 8 and col < 8                              # y=0.75 -> lower half, x=0.25 -> left half


def test_va_quadrant():
    q = va_quadrant(np.array([8, 8, 2, 2, np.nan]), np.array([2, 8, 2, 8, 5]))
    assert q.tolist() == [0, 1, 2, 3, -1]
