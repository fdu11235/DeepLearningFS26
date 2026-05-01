"""F2-optimal threshold sweep on validation probabilities.

F2 weights recall twice as much as precision — appropriate for fraud detection
where missed fraud is much costlier than false alarms.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve


def tune_threshold_f2(y_true: np.ndarray, y_proba: np.ndarray, beta: float = 2.0) -> tuple[float, float]:
    """Return (best_threshold, best_fbeta) maximizing F-beta on the PR curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # precision/recall arrays are length N+1, thresholds length N. Trim.
    precision = precision[:-1]
    recall = recall[:-1]
    if len(thresholds) == 0:
        return 0.5, 0.0
    beta2 = beta * beta
    denom = beta2 * precision + recall
    fbeta = np.zeros_like(precision)
    np.divide(
        (1 + beta2) * precision * recall,
        denom,
        out=fbeta,
        where=denom > 0,
    )
    best_idx = int(np.argmax(fbeta))
    return float(thresholds[best_idx]), float(fbeta[best_idx])
