"""torch Dataset wrapping pre-built sequence arrays."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, lengths: np.ndarray):
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        if y.dtype != np.float32:
            y = y.astype(np.float32)
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.lengths = torch.from_numpy(lengths.astype(np.int64))

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.lengths[idx], self.y[idx]


def collate_fn(batch):
    xs, lens, ys = zip(*batch)
    return torch.stack(xs, 0), torch.stack(lens, 0), torch.stack(ys, 0)
