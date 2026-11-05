"""Neural Network Architectures for CSI Respiration Sensing (v2).

Implements a hybrid Dual-Branch Spatial-Temporal-Spectral model:
1. Temporal Branch: Global window normalization + 1D-CNN + BiLSTM over 46 active
   subcarriers across a 16-second window (360 timesteps at 22.5 Hz).
2. Spectral & Periodicity Branch: Normalized breathing-band PSD (0.08-0.80 Hz) +
   autocorrelation peak & spectral SNR features from PCA-fused subcarriers.
3. Multi-task Heads:
   - 4-class respiratory state classification (Apnea, Slow, Normal, Fast)
   - Continuous breathing rate regression (BPM)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """Self-attention pooling layer across the temporal sequence dimension."""

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.Tanh(),
            nn.Linear(in_features // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.attention(x), dim=1)
        return torch.sum(x * weights, dim=1)


class CSI_RespirationNet(nn.Module):
    """Dual-Branch 1D-CNN + BiLSTM + Spectral/Autocorrelation Multi-task Network."""

    def __init__(
        self,
        in_channels: int = 46,
        spec_dim: int = 131,
        aux_dim: int = 6,
        num_classes: int = 4,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.spec_dim = spec_dim
        self.aux_dim = aux_dim

        # Branch 1: Spatio-Temporal 1D-CNN + BiLSTM
        self.temp_conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(3),  # 360 -> 120
            nn.Dropout(dropout),
            nn.Conv1d(32, 48, kernel_size=5, padding=2),
            nn.BatchNorm1d(48),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(3),  # 120 -> 40
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=48,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.attention_pool = AttentionPooling(64)

        # Branch 2: Spectral Density & Autocorrelation Periodicity Encoder
        self.spec_fc = nn.Sequential(
            nn.Linear(spec_dim + aux_dim, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(64, 48),
            nn.LeakyReLU(0.1),
        )

        fusion_dim = 64 + 48

        # Head 1: 4-Class Respiratory State Classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 48),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(48, num_classes),
        )

        # Head 2: Continuous Breathing Rate Regressor (BPM)
        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        x_wave: torch.Tensor,
        x_spec: torch.Tensor | None = None,
        x_aux: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x_wave : (batch_size, 46, 360)
        x_spec : (batch_size, 131) or None
        x_aux  : (batch_size, 6) or None
        """
        # Global window normalization preserves inter-subcarrier variance ratios
        mean = x_wave.mean(dim=(-2, -1), keepdim=True)
        std = x_wave.std(dim=(-2, -1), keepdim=True) + 1e-6
        xw = (x_wave - mean) / std

        t_conv = self.temp_conv(xw).permute(0, 2, 1)  # (B, 40, 48)
        lstm_out, _ = self.lstm(t_conv)
        t_feat = 0.5 * (lstm_out.mean(dim=1) + self.attention_pool(lstm_out))  # (B, 64)

        if x_spec is None:
            x_spec = torch.zeros(x_wave.size(0), self.spec_dim, device=x_wave.device, dtype=x_wave.dtype)
        if x_aux is None:
            x_aux = torch.zeros(x_wave.size(0), self.aux_dim, device=x_wave.device, dtype=x_wave.dtype)

        s_feat = self.spec_fc(torch.cat([x_spec, x_aux], dim=-1))  # (B, 48)
        fused = torch.cat([t_feat, s_feat], dim=-1)

        logits_class = self.classifier(fused)
        pred_bpm = self.regressor(fused).squeeze(-1) * 30.0
        return logits_class, pred_bpm
