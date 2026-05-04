"""Per-card sliding-window construction for the LSTM.

For each card (cc_num), transactions are sorted by time and we emit one
sliding window per transaction. Window i = the last L transactions ending at
i (inclusive); the label is is_fraud of transaction i (predict the most
recent transaction given its card history).

Cards with fewer than L history are left-padded with zeros and a length mask
records the number of valid timesteps.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class SequenceArrays:
    X: np.ndarray  # (N, L, F) float32, left-padded
    y: np.ndarray  # (N,)      float32, label of last tx in window
    lengths: np.ndarray  # (N,) int64, valid history per window (1..L)


def build_sequences(
    features: np.ndarray,
    labels: np.ndarray,
    card_ids: np.ndarray,
    timestamps: np.ndarray,
    seq_len: int = 20,
) -> SequenceArrays:
    """Build left-padded sliding windows grouped by card.

    Parameters
    ----------
    features : (N, F) float array of preprocessed transaction features.
    labels   : (N,) int array, is_fraud per transaction.
    card_ids : (N,) array, cc_num per transaction (used only for grouping).
    timestamps : (N,) array of sortable timestamps (datetime64 or int).
    seq_len  : window length L.
    """
    if features.ndim != 2:
        raise ValueError(f"features must be 2D, got {features.shape}")
    n, n_feat = features.shape
    if labels.shape != (n,) or card_ids.shape != (n,) or timestamps.shape != (n,):
        raise ValueError("features/labels/card_ids/timestamps must have matching length")

    # Stable sort by (card_id, timestamp) so transactions of each card are contiguous and ordered.
    order = np.lexsort((timestamps, card_ids))
    sorted_features = features[order]
    sorted_labels = labels[order]
    sorted_cards = card_ids[order]

    # Group boundaries via run-length on sorted_cards.
    change = np.concatenate(([True], sorted_cards[1:] != sorted_cards[:-1]))
    group_starts = np.flatnonzero(change)
    group_ends = np.concatenate((group_starts[1:], [n]))

    X = np.zeros((n, seq_len, n_feat), dtype=np.float32)
    y = sorted_labels.astype(np.float32)
    lengths = np.zeros(n, dtype=np.int64)

    for g_start, g_end in zip(group_starts, group_ends):
        for i in range(g_start, g_end):
            history_start = max(g_start, i - seq_len + 1)
            window = sorted_features[history_start : i + 1]
            valid = window.shape[0]
            # left-pad: window goes into the last `valid` positions of the L-length slot.
            X[i, seq_len - valid :] = window
            lengths[i] = valid

    # Restore original row ordering so callers can join back to the dataframe.
    inverse = np.empty_like(order)
    inverse[order] = np.arange(n)
    return SequenceArrays(X=X[inverse], y=y[inverse], lengths=lengths[inverse])
