"""McNemar test for paired model predictions on the same test set.

Ported from the legacy notebook; used to compare LSTM vs each classic
baseline (or LSTM vs LSTM+SMOTE) for statistical significance.
"""
from __future__ import annotations

from typing import Any
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar


def mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, Any]:
    """Return p-value and contingency-table breakdown of where models disagree.

    Convention used in the legacy notebook: tabulate (correct_a, correct_b)
    where correct = (pred == y_true).
    """
    correct_a = (pred_a == y_true).astype(int)
    correct_b = (pred_b == y_true).astype(int)
    n00 = int(((correct_a == 0) & (correct_b == 0)).sum())
    n01 = int(((correct_a == 0) & (correct_b == 1)).sum())
    n10 = int(((correct_a == 1) & (correct_b == 0)).sum())
    n11 = int(((correct_a == 1) & (correct_b == 1)).sum())
    table = np.array([[n11, n10], [n01, n00]])
    result = mcnemar(table, exact=False, correction=True)
    return {
        "statistic": float(result.statistic),
        "pvalue": float(result.pvalue),
        "table": table.tolist(),
        "a_only_correct": n10,
        "b_only_correct": n01,
    }
