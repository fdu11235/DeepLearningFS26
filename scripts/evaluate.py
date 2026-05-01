"""Compare LSTM and LSTM+SMOTE against the four classic baselines.

Loads:
  - models/lstm/test_predictions.npz, models/lstm/threshold.json, metrics.json
  - models/lstm_smote/... (same)
  - data/processed/legacy_predictions.parquet (one-time export from the legacy
    notebook with columns y_true, svm, rf, svm_smote, rf_smote — predictions
    on fraudTest.csv; same row ordering as fraudTest.csv)

Prints a side-by-side metrics table and runs McNemar tests on the most
informative pairings.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fraud.evaluation import compute_metrics, mcnemar_test  # noqa: E402
from fraud.utils import load_json  # noqa: E402


def _load_lstm(model_dir: Path):
    pred_path = model_dir / "test_predictions.npz"
    thr_path = model_dir / "threshold.json"
    if not pred_path.exists():
        return None
    npz = np.load(pred_path)
    threshold = float(load_json(thr_path)["threshold"]) if thr_path.exists() else 0.5
    return {
        "y_true": npz["y_true"].astype(int),
        "y_proba": npz["y_proba"],
        "threshold": threshold,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lstm", type=Path, default=ROOT / "models/lstm")
    parser.add_argument("--lstm-smote", type=Path, default=ROOT / "models/lstm_smote")
    parser.add_argument(
        "--legacy-predictions",
        type=Path,
        default=ROOT / "data/processed/legacy_predictions.parquet",
    )
    args = parser.parse_args()

    lstm = _load_lstm(args.lstm)
    lstm_smote = _load_lstm(args.lstm_smote)

    print("\n=== Test-set metrics ===")
    rows = []
    if lstm is not None:
        m_default = compute_metrics(lstm["y_true"], lstm["y_proba"], threshold=0.5)
        m_tuned = compute_metrics(lstm["y_true"], lstm["y_proba"], threshold=lstm["threshold"])
        rows.append(("LSTM (thr=0.5)", m_default))
        rows.append(("LSTM (tuned)", m_tuned))
    if lstm_smote is not None:
        m_default = compute_metrics(lstm_smote["y_true"], lstm_smote["y_proba"], threshold=0.5)
        m_tuned = compute_metrics(
            lstm_smote["y_true"], lstm_smote["y_proba"], threshold=lstm_smote["threshold"]
        )
        rows.append(("LSTM+SMOTE (thr=0.5)", m_default))
        rows.append(("LSTM+SMOTE (tuned)", m_tuned))

    if not rows:
        print("No LSTM predictions found. Run `python scripts/train_lstm.py` first.")
        return

    print(f"{'Model':<30} {'ROC-AUC':>8} {'PR-AUC':>8} {'P':>6} {'R':>6} {'F1':>6}")
    for name, m in rows:
        print(
            f"{name:<30} {m['roc_auc']:>8.4f} {m['pr_auc']:>8.4f} "
            f"{m['fraud_precision']:>6.3f} {m['fraud_recall']:>6.3f} {m['fraud_f1']:>6.3f}"
        )

    if not args.legacy_predictions.exists():
        print(
            f"\n(Legacy predictions not found at {args.legacy_predictions}. "
            "Run the export cell at the end of notebooks/legacy/fraud_detection_notebook.ipynb "
            "to enable McNemar comparison against SVM/RF/SVM+SMOTE/RF+SMOTE.)"
        )
        return

    legacy = pd.read_parquet(args.legacy_predictions)
    print(
        f"\n=== Legacy classic baselines (from {args.legacy_predictions.name}) ==="
    )
    for col in ["svm", "rf", "svm_smote", "rf_smote"]:
        if col not in legacy.columns:
            continue
        m = compute_metrics(
            legacy["y_true"].to_numpy(),
            legacy[col].to_numpy().astype(float),
            threshold=0.5,
        )
        print(
            f"{col:<30} {m['roc_auc']:>8.4f} {m['pr_auc']:>8.4f} "
            f"{m['fraud_precision']:>6.3f} {m['fraud_recall']:>6.3f} {m['fraud_f1']:>6.3f}"
        )

    if lstm is None:
        return

    # McNemar comparisons require aligned predictions on the same rows.
    if len(legacy) != len(lstm["y_true"]):
        print(
            f"\nSkipping McNemar — legacy predictions have {len(legacy)} rows but "
            f"LSTM predictions have {len(lstm['y_true'])}. They must align row-for-row."
        )
        return

    print("\n=== McNemar tests ===")
    y_true = lstm["y_true"]
    lstm_pred = (lstm["y_proba"] >= lstm["threshold"]).astype(int)

    if "rf" in legacy.columns:
        rf_pred = legacy["rf"].to_numpy().astype(int)
        result = mcnemar_test(y_true, lstm_pred, rf_pred)
        print(f"LSTM vs RF        p={result['pvalue']:.3e}  (LSTM-only-correct={result['a_only_correct']}, RF-only-correct={result['b_only_correct']})")

    if lstm_smote is not None:
        smote_pred = (lstm_smote["y_proba"] >= lstm_smote["threshold"]).astype(int)
        result = mcnemar_test(y_true, lstm_pred, smote_pred)
        print(f"LSTM vs LSTM+SMOTE p={result['pvalue']:.3e}")

        if "rf_smote" in legacy.columns:
            rf_smote_pred = legacy["rf_smote"].to_numpy().astype(int)
            result = mcnemar_test(y_true, smote_pred, rf_smote_pred)
            print(f"LSTM+SMOTE vs RF+SMOTE p={result['pvalue']:.3e}")


if __name__ == "__main__":
    main()
