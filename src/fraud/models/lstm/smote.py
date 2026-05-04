"""Sequence-level SMOTE for the LSTM+SMOTE configuration.

We flatten each (L, F) window into an L*F vector, run SMOTE in that flat
space, and reshape synthetic vectors back to (L, F). Length masks for
synthetic windows are set to L (full history) since the synthesised values
fill all positions.

Caveat: interpolation in flat-window space can produce temporally
implausible sequences (e.g., merchant identity drifting mid-sequence). This
is the LSTM analogue of the same caveat that already shows up in the
literature for "RNN + SMOTE", and the legacy notebook found SMOTE actively
hurt classic models. Running this configuration tests whether the same
finding holds for the LSTM.
"""
from __future__ import annotations

import gc

import numpy as np
from imblearn.over_sampling import SMOTE


def sequence_smote(
    X: np.ndarray,
    y: np.ndarray,
    lengths: np.ndarray,
    random_state: int = 42,
    k_neighbors: int = 5,
    sampling_strategy: float | str = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Oversample positive sequences via SMOTE on flattened windows.

    `sampling_strategy=0.1` means: generate enough synthetic minority
    samples so that the minority count equals 10% of the majority count.
    Use `"auto"` for full 1:1 balancing (much heavier on memory — peak RAM
    can hit ~4-5x the original train tensor on long sequences).

    Returns (X_resampled, y_resampled, lengths_resampled). Original samples
    are preserved; synthetic samples are appended.
    """
    if X.ndim != 3:
        raise ValueError(f"expected X of shape (N, L, F), got {X.shape}")
    n, L, F = X.shape
    flat = X.reshape(n, L * F)

    sampler = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors,
        random_state=random_state,
    )
    flat_res, y_res = sampler.fit_resample(flat, y.astype(int))

    # Free the flattened original before allocating the reshaped result.
    del flat
    gc.collect()

    X_res = flat_res.reshape(-1, L, F).astype(np.float32, copy=False)
    y_res = y_res.astype(np.float32)
    del flat_res
    gc.collect()

    n_synth = X_res.shape[0] - n
    lengths_res = np.concatenate([lengths, np.full(n_synth, L, dtype=lengths.dtype)])
    return X_res, y_res, lengths_res
