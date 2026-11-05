"""Hybrid multimodal emotion recognition model: EEG (CNN+LSTM), Physio (LSTM), Eye (CNN+GRU),
fused with a Transformer encoder."""
import torch
import torch.nn as nn


class EEGBranch(nn.Module):
    """1D convolution across channels at each timestep (spatial), then LSTM over time."""

    def __init__(self, n_channels: int = 63, d_model: int = 128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.lstm = nn.LSTM(64, d_model, batch_first=True)

    def forward(self, x):                        # x: (B, T, C)
        h = self.cnn(x.transpose(1, 2))          # (B, 64, T)
        _, (h_n, _) = self.lstm(h.transpose(1, 2))
        return h_n[-1]                           # (B, d_model)


class PhysioBranch(nn.Module):
    def __init__(self, n_features: int = 5, d_model: int = 128):
        super().__init__()
        self.norm = nn.LayerNorm(n_features)
        self.lstm = nn.LSTM(n_features, d_model, num_layers=2, batch_first=True, dropout=0.2)

    def forward(self, x):                        # x: (B, T, F)
        _, (h_n, _) = self.lstm(self.norm(x))
        return h_n[-1]


class EyeBranch(nn.Module):
    """Per-frame 2D CNN over gaze heatmaps, then GRU over time."""

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),                        # 32 * 4 * 4 = 512
        )
        self.gru = nn.GRU(512, d_model, batch_first=True)

    def forward(self, x):                        # x: (B, T, 1, H, W)
        B, T = x.shape[:2]
        g = self.cnn(x.flatten(0, 1)).view(B, T, -1)
        _, h_n = self.gru(g)
        return h_n[-1]


class MultimodalEmotionNet(nn.Module):
    def __init__(self, n_classes: int, eeg_channels: int = 63, physio_features: int = 5,
                 d_model: int = 128, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.eeg = EEGBranch(eeg_channels, d_model)
        self.physio = PhysioBranch(physio_features, d_model)
        self.eye = EyeBranch(d_model)
        self.modality_emb = nn.Parameter(torch.randn(3, d_model) * 0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model,
                                           dropout=dropout, batch_first=True)
        self.fusion = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout),
                                  nn.Linear(d_model, n_classes))

    def forward(self, eeg, physio, eye, return_attention_tokens: bool = False):
        tokens = torch.stack([self.eeg(eeg), self.physio(physio), self.eye(eye)], dim=1)
        tokens = tokens + self.modality_emb
        x = torch.cat([self.cls.expand(tokens.size(0), -1, -1), tokens], dim=1)
        fused = self.fusion(x)
        logits = self.head(fused[:, 0])          # softmax applied in the loss / at inference
        return (logits, fused) if return_attention_tokens else logits
