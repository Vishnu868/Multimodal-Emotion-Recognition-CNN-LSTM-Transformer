"""Hybrid multimodal emotion recognition network (CNN + LSTM/GRU + Transformer).

    EEG    (B,T,5,9,9)    -> per-frame 2D CNN over the electrode grid  -> LSTM -> f_EEG  (128)
    Physio (B,T,6)        -> LSTM                                       -> f_phys (128)
    Eye    (B,T,1,16,16)  -> per-frame 2D CNN + ReLU + max-pool         -> GRU  -> f_eye  (128)

    F  = [f_EEG; f_phys; f_eye] + modality embeddings          (3 tokens)
    F' = TransformerFusion(F)  with softmax(QK^T / sqrt(d_k)) V  (multi-head self-attention)
    z  = W_c F' + b_c,  p = softmax(z)                          (F' projected to 256-D)
"""
import torch
import torch.nn as nn

MODALITIES = ("eeg", "physio", "eye")


class EEGBranch(nn.Module):
    """Spatial CNN over the 9x9 scalp map (5 frequency bands as input channels), then LSTM over time."""

    def __init__(self, n_bands: int = 5, d_model: int = 128, dropout: float = 0.2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(n_bands, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                                    # 9x9 -> 4x4
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(64 * 4 * 4, d_model), nn.ReLU(),
        )
        self.lstm = nn.LSTM(d_model, d_model, batch_first=True)

    def forward(self, x):                                       # (B, T, 5, 9, 9)
        B, T = x.shape[:2]
        h = self.cnn(x.flatten(0, 1)).view(B, T, -1)
        _, (h_n, _) = self.lstm(h)
        return h_n[-1]                                          # f_EEG = h_T


class PhysioBranch(nn.Module):
    """Two-layer LSTM over normalised GSR / temperature / accelerometer frames."""

    def __init__(self, n_features: int = 6, d_model: int = 128, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, d_model, num_layers=2, batch_first=True, dropout=dropout)

    def forward(self, x):                                       # (B, T, F)
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]                                          # f_phys = h_T in R^128


class EyeBranch(nn.Module):
    """Per-frame CNN over gaze heatmaps (conv -> ReLU -> max-pool -> flatten = g_t), then GRU."""

    def __init__(self, d_model: int = 128, dropout: float = 0.2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveMaxPool2d(4),
            nn.Flatten(), nn.Dropout(dropout),                  # g_t in R^(32*4*4 = 512)
        )
        self.gru = nn.GRU(32 * 4 * 4, d_model, batch_first=True)

    def forward(self, x):                                       # (B, T, 1, H, W)
        B, T = x.shape[:2]
        x = x / (x.flatten(2).amax(-1)[..., None, None, None] + 1e-6)  # per-frame scale to [0, 1]
        g = self.cnn(x.flatten(0, 1)).view(B, T, -1)
        _, h_n = self.gru(g)
        return h_n[-1]                                          # f_eye = h_T in R^128


class FusionLayer(nn.Module):
    """Pre-norm Transformer encoder layer that also returns the attention map (for interpretability)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(4 * d_model, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        h = self.norm1(x)
        a, w = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=True,
                         average_attn_weights=True)
        x = x + self.drop(a)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x, w                                             # w: (B, tokens, tokens)


class MultimodalEmotionNet(nn.Module):
    def __init__(self, n_classes: int = 4, modalities=MODALITIES, physio_features: int = 6,
                 d_model: int = 128, n_heads: int = 4, n_layers: int = 2, fused_dim: int = 256,
                 dropout: float = 0.3):
        super().__init__()
        self.modalities = tuple(m for m in MODALITIES if m in modalities)
        branches = {"eeg": lambda: EEGBranch(5, d_model, dropout),
                    "physio": lambda: PhysioBranch(physio_features, d_model, dropout),
                    "eye": lambda: EyeBranch(d_model, dropout)}
        self.branches = nn.ModuleDict({m: branches[m]() for m in self.modalities})
        self.modality_emb = nn.Parameter(torch.randn(len(self.modalities), d_model) * 0.02)
        self.layers = nn.ModuleList(FusionLayer(d_model, n_heads, dropout) for _ in range(n_layers))
        self.norm = nn.LayerNorm(d_model)
        self.project = nn.Sequential(nn.Linear(len(self.modalities) * d_model, fused_dim), nn.ReLU(),
                                     nn.Dropout(dropout))      # F' in R^256
        self.classifier = nn.Linear(fused_dim, n_classes)      # z = W_c F' + b_c

    def forward(self, eeg, physio, eye, mask=None, return_attention: bool = False):
        inputs = {"eeg": eeg, "physio": physio, "eye": eye}
        tokens = torch.stack([self.branches[m](inputs[m]) for m in self.modalities], 1) + self.modality_emb
        present = None
        if mask is not None:                                    # mask: (B, 3) in eeg/physio/eye order
            present = mask[:, [MODALITIES.index(m) for m in self.modalities]].bool()
            present = present | ~present.any(1, keepdim=True)   # never mask every token
            tokens = tokens * present[..., None]
        attn = []
        x = tokens
        for layer in self.layers:
            x, w = layer(x, key_padding_mask=None if present is None else ~present)
            attn.append(w)
        x = self.norm(x)
        if present is not None:
            x = x * present[..., None]
        fused = self.project(x.flatten(1))
        logits = self.classifier(fused)                         # softmax lives in the loss / predict()
        return (logits, torch.stack(attn, 1)) if return_attention else logits

    @torch.no_grad()
    def predict_proba(self, eeg, physio, eye, mask=None):
        return torch.softmax(self.forward(eeg, physio, eye, mask), dim=-1)
