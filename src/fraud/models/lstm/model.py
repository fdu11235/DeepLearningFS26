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
        # x: (B, L, F_in), right-padded. lengths: (B,) int, valid timesteps.
        # pack_padded_sequence skips padded positions so the LSTM cell state
        # is only updated by real transactions.
        proj = self.input_proj(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            proj, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out_packed, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            out_packed, batch_first=True, total_length=x.size(1)
        )
        idx = (lengths - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, out.size(-1))
        last = out.gather(1, idx).squeeze(1)
        return self.head(last).squeeze(-1)
