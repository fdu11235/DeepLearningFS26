"""Per-card sliding-window construction for the LSTM.

For each card (cc_num), transactions are sorted by time and we emit one
sliding window per transaction. Window i = the last L transactions ending at
i (inclusive); the label is is_fraud of transaction i (predict the most
recent transaction given its card history).

Cards with fewer than L history are right-padded with zeros and a length
mask records the number of valid timesteps. Right padding lets the LSTM
use ``pack_padded_sequence`` so padded steps are not processed at all.

The optional ``anchor_mask`` argument lets callers concatenate older
"history" rows in front of the rows they actually want windows for, so
that val/test sequences can use the train history of the same card
without propagating split boundaries into the windowing logic.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class SequenceArrays:
    X: np.ndarray  # (N, L, F) float32, right-padded
    y: np.ndarray  # (N,)      float32, label of last tx in window
    lengths: np.ndarray  # (N,) int64, valid history per window (1..L)


def build_sequences(
    features: np.ndarray,
    labels: np.ndarray,
    card_ids: np.ndarray,
    timestamps: np.ndarray,
    seq_len: int = 20,
    anchor_mask: np.ndarray | None = None,
) -> SequenceArrays:
    """Build right-padded sliding windows grouped by card.

    Parameters
    ----------
    features : (N, F) float array of preprocessed transaction features.
    labels   : (N,) int array, is_fraud per transaction.
    card_ids : (N,) array, cc_num per transaction (used only for grouping).
    timestamps : (N,) array of sortable timestamps (datetime64 or int).
    seq_len  : window length L.
    anchor_mask : (N,) bool array. If provided, only rows where
        ``anchor_mask[i]`` is True become emitted windows; the others are
        used purely as history for later anchors of the same card. If
        None, every row is an anchor.
    """
    if features.ndim != 2:
        raise ValueError(f"features must be 2D, got {features.shape}")
    n, n_feat = features.shape
    if labels.shape != (n,) or card_ids.shape != (n,) or timestamps.shape != (n,):
        raise ValueError("features/labels/card_ids/timestamps must have matching length")
    if anchor_mask is None:
        anchor_mask = np.ones(n, dtype=bool)
    elif anchor_mask.shape != (n,):
        raise ValueError("anchor_mask must have shape (N,)")

    order = np.lexsort((timestamps, card_ids))
    sorted_features = features[order]
    sorted_labels = labels[order]
    sorted_cards = card_ids[order]
    sorted_anchor = anchor_mask[order]

    change = np.concatenate(([True], sorted_cards[1:] != sorted_cards[:-1]))
    group_starts = np.flatnonzero(change)
    group_ends = np.concatenate((group_starts[1:], [n]))

    n_anchors = int(sorted_anchor.sum())
    X = np.zeros((n_anchors, seq_len, n_feat), dtype=np.float32)
    y = np.zeros(n_anchors, dtype=np.float32)
    lengths = np.zeros(n_anchors, dtype=np.int64)
    anchor_original_pos = np.empty(n_anchors, dtype=np.int64)

    out_idx = 0
    for g_start, g_end in zip(group_starts, group_ends):
        for i in range(g_start, g_end):
            if not sorted_anchor[i]:
                continue
            history_start = max(g_start, i - seq_len + 1)
            window = sorted_features[history_start : i + 1]
            valid = window.shape[0]
            X[out_idx, :valid] = window
            y[out_idx] = sorted_labels[i]
            lengths[out_idx] = valid
            anchor_original_pos[out_idx] = order[i]
            out_idx += 1

    # Sort outputs back to the original row order of the anchor rows so
    # callers can join on row index. Non-anchor rows are simply absent.
    permute = np.argsort(anchor_original_pos)
    return SequenceArrays(X=X[permute], y=y[permute], lengths=lengths[permute])


def last_n_per_card(
    features: np.ndarray,
    labels: np.ndarray,
    card_ids: np.ndarray,
    timestamps: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the last ``n`` rows per card by timestamp.

    Used to pull a tail of train (or train+val) history into val/test
    sequence construction so windows that span the split boundary can
    see real prior transactions instead of zero padding.
    """
    order = np.lexsort((timestamps, card_ids))
    sorted_c = card_ids[order]
    change = np.concatenate(([True], sorted_c[1:] != sorted_c[:-1]))
    group_starts = np.flatnonzero(change)
    group_ends = np.concatenate((group_starts[1:], [len(card_ids)]))
    keep_sorted_idx: list[int] = []
    for gs, ge in zip(group_starts, group_ends):
        keep_sorted_idx.extend(range(max(gs, ge - n), ge))
    keep = order[np.asarray(keep_sorted_idx, dtype=np.int64)]
    return features[keep], labels[keep], card_ids[keep], timestamps[keep]
