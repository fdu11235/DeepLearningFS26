"""FraudLSTM model.

Input:  (B, L, F) padded windows, plus (B,) lengths giving the count of
        valid (non-padding) timesteps in each window.
Output: (B,) logit per window — fraud probability for the most recent tx.
"""
from __future__ import annotations

import torch
from torch import nn


class FraudLSTM(nn.Module):
    def __init__(
        self,
        in_features: int,
        proj_dim: int = 64,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.in_features = in_features
        self.input_proj = nn.Linear(in_features, proj_dim)
        self.lstm = nn.LSTM(
            input_size=proj_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F_in). Sequences are LEFT-padded so the last valid timestep
        # is always at position L-1, regardless of length. We can therefore index
        # the last position safely.
        proj = self.input_proj(x)
        out, _ = self.lstm(proj)
        last = out[:, -1, :]
        logits = self.head(last).squeeze(-1)
        return logits
