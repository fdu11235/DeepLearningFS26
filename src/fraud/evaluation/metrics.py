"""Classification metrics matching the legacy notebook (ROC-AUC, PR-AUC, P/R/F1)."""
from __future__ import annotations

from typing import Any
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = (y_proba >= threshold).astype(int)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    # classification_report keys are stringified labels; "1" for ints, "1.0" for floats.
    fraud = report.get("1", report.get("1.0", {}))
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "fraud_precision": float(fraud.get("precision", 0.0)),
        "fraud_recall": float(fraud.get("recall", 0.0)),
        "fraud_f1": float(fraud.get("f1-score", 0.0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "support_pos": int((y_true == 1).sum()),
        "support_neg": int((y_true == 0).sum()),
    }


def format_report(metrics: dict[str, Any], name: str = "model") -> str:
    cm = metrics["confusion_matrix"]
    return (
        f"=== {name} (threshold={metrics['threshold']:.4f}) ===\n"
        f"ROC-AUC: {metrics['roc_auc']:.4f}   PR-AUC: {metrics['pr_auc']:.4f}\n"
        f"Fraud  P: {metrics['fraud_precision']:.3f}  "
        f"R: {metrics['fraud_recall']:.3f}  F1: {metrics['fraud_f1']:.3f}\n"
        f"Confusion matrix [[TN, FP], [FN, TP]]: {cm}\n"
        f"Support: pos={metrics['support_pos']}  neg={metrics['support_neg']}"
    )
