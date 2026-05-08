"""Small hyperparameter sweep driver around train_lstm().

Usage:
    python scripts/tune_lstm.py --sweep-config configs/tune_lstm.yaml

Reads a sweep YAML that declares a base config and a parameter grid,
Cartesian-products the grid, runs each trial as a normal train_lstm()
call into a unique out_dir under <out_root>/<sweep_name>/, generates
per-trial training-curve and confusion-matrix PNGs, and aggregates
all trial metrics into a single tuning_results.csv at the sweep root.

Resumable: trials whose out_dir already contains metrics.json are skipped.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fraud.evaluation import plot_confusion_matrix, plot_training_curves  # noqa: E402
from fraud.models.lstm.trainer import TrainConfig, train_lstm  # noqa: E402

logger = logging.getLogger("tune_lstm")

CSV_COLUMNS = [
    "trial_idx",
    "trial_tag",
    "lr",
    "dropout",
    "best_val_pr_auc",
    "test_roc_auc_default",
    "test_pr_auc_default",
    "test_f1_default",
    "tn_default",
    "fp_default",
    "fn_default",
    "tp_default",
    "test_roc_auc_tuned",
    "test_pr_auc_tuned",
    "test_precision_tuned",
    "test_recall_tuned",
    "test_f1_tuned",
    "tn_tuned",
    "fp_tuned",
    "fn_tuned",
    "tp_tuned",
    "tuned_threshold",
    "completed_at",
]


def _format_val(v: Any) -> str:
    """Compact, filename-safe formatting for hyperparameter values."""
    if isinstance(v, float):
        if v == 0 or (1e-3 <= abs(v) < 1e3):
            s = f"{v:g}"
        else:
            s = f"{v:.0e}"
        return s.replace(".", "p")
    return str(v)


def _trial_tag(combo: dict[str, Any]) -> str:
    return "_".join(f"{k}{_format_val(v)}" for k, v in combo.items())


def _row_from_metrics(idx: int, tag: str, combo: dict[str, Any], metrics: dict) -> dict:
    default = metrics["test_default_threshold"]
    tuned = metrics["test_tuned_threshold"]
    cm_d = default["confusion_matrix"]
    cm_t = tuned["confusion_matrix"]
    return {
        "trial_idx": idx,
        "trial_tag": tag,
        "lr": combo.get("lr", ""),
        "dropout": combo.get("dropout", ""),
        "best_val_pr_auc": metrics["best_val_pr_auc"],
        "test_roc_auc_default": default["roc_auc"],
        "test_pr_auc_default": default["pr_auc"],
        "test_f1_default": default["fraud_f1"],
        "tn_default": cm_d[0][0],
        "fp_default": cm_d[0][1],
        "fn_default": cm_d[1][0],
        "tp_default": cm_d[1][1],
        "test_roc_auc_tuned": tuned["roc_auc"],
        "test_pr_auc_tuned": tuned["pr_auc"],
        "test_precision_tuned": tuned["fraud_precision"],
        "test_recall_tuned": tuned["fraud_recall"],
        "test_f1_tuned": tuned["fraud_f1"],
        "tn_tuned": cm_t[0][0],
        "fp_tuned": cm_t[0][1],
        "fn_tuned": cm_t[1][0],
        "tp_tuned": cm_t[1][1],
        "tuned_threshold": tuned["threshold"],
        "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _append_csv_row(csv_path: Path, row: dict) -> None:
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def _generate_plots(trial_dir: Path, metrics: dict, trial_tag: str) -> None:
    """Best-effort: failures here must not abort the sweep."""
    log_csv = trial_dir / "training_log.csv"
    try:
        if log_csv.exists():
            curves_png = trial_dir / "training_curves.png"
            if not curves_png.exists():
                plot_training_curves(log_csv, curves_png, title=trial_tag)
    except Exception as e:
        logger.warning("plot_training_curves failed for %s: %s", trial_tag, e)

    try:
        cm_d_png = trial_dir / "confusion_matrix_default.png"
        if not cm_d_png.exists():
            plot_confusion_matrix(
                metrics["test_default_threshold"]["confusion_matrix"],
                cm_d_png,
                title=f"{trial_tag} — test @ thr=0.5",
            )
    except Exception as e:
        logger.warning("plot_confusion_matrix (default) failed for %s: %s", trial_tag, e)

    try:
        thr = metrics["test_tuned_threshold"]["threshold"]
        cm_t_png = trial_dir / "confusion_matrix_tuned.png"
        if not cm_t_png.exists():
            plot_confusion_matrix(
                metrics["test_tuned_threshold"]["confusion_matrix"],
                cm_t_png,
                title=f"{trial_tag} — test @ thr={thr:.4f}",
            )
    except Exception as e:
        logger.warning("plot_confusion_matrix (tuned) failed for %s: %s", trial_tag, e)


def _read_existing_rows(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        return {row["trial_tag"] for row in reader if row.get("trial_tag")}


def main() -> None:
    parser = argparse.ArgumentParser(description="LSTM hyperparameter sweep")
    parser.add_argument("--sweep-config", required=True, help="path to sweep YAML")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    with open(args.sweep_config) as f:
        sweep = yaml.safe_load(f)

    base_config_path = Path(sweep["base_config"])
    if not base_config_path.is_absolute():
        base_config_path = ROOT / base_config_path
    with open(base_config_path) as f:
        base_cfg = yaml.safe_load(f)

    sweep_name = sweep["sweep_name"]
    out_root = Path(sweep["out_root"])
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    sweep_root = out_root / sweep_name
    sweep_root.mkdir(parents=True, exist_ok=True)

    sweep_seed = sweep.get("seed", base_cfg.get("seed", 42))
    grid = sweep["grid"]
    grid_keys = sorted(grid.keys())
    combos = [dict(zip(grid_keys, values)) for values in itertools.product(*(grid[k] for k in grid_keys))]

    csv_path = sweep_root / "tuning_results.csv"
    already_logged_tags = _read_existing_rows(csv_path)
    logger.info(
        "Sweep '%s': %d trials over grid keys %s -> %s",
        sweep_name, len(combos), grid_keys, sweep_root,
    )

    summaries: list[tuple[int, str, dict]] = []

    for idx, combo in enumerate(combos):
        tag = _trial_tag(combo)
        trial_dir = sweep_root / f"trial_{idx:03d}_{tag}"
        metrics_path = trial_dir / "metrics.json"

        if metrics_path.exists():
            logger.info("[%d/%d] %s — skipping (metrics.json already exists)", idx + 1, len(combos), tag)
            with open(metrics_path) as f:
                metrics = json.load(f)
            _generate_plots(trial_dir, metrics, tag)
            if tag not in already_logged_tags:
                _append_csv_row(csv_path, _row_from_metrics(idx, tag, combo, metrics))
                already_logged_tags.add(tag)
            summaries.append((idx, tag, metrics))
            continue

        trial_cfg_dict = {**base_cfg, **combo, "out_dir": str(trial_dir), "seed": sweep_seed}
        cfg = TrainConfig(**trial_cfg_dict)
        logger.info(
            "[%d/%d] %s — training (lr=%g, dropout=%g, out=%s)",
            idx + 1, len(combos), tag, cfg.lr, cfg.dropout, trial_dir,
        )
        train_lstm(cfg)

        with open(metrics_path) as f:
            metrics = json.load(f)
        _generate_plots(trial_dir, metrics, tag)
        _append_csv_row(csv_path, _row_from_metrics(idx, tag, combo, metrics))
        already_logged_tags.add(tag)
        summaries.append((idx, tag, metrics))

    if not summaries:
        logger.warning("No trials produced metrics; nothing to summarise.")
        return

    best_pr = max(summaries, key=lambda t: t[2]["test_default_threshold"]["pr_auc"])
    best_f1 = max(summaries, key=lambda t: t[2]["test_tuned_threshold"]["fraud_f1"])
    logger.info("=" * 70)
    logger.info(
        "Best test PR-AUC: trial_%03d %s -> %.4f",
        best_pr[0], best_pr[1], best_pr[2]["test_default_threshold"]["pr_auc"],
    )
    logger.info(
        "Best tuned F1:    trial_%03d %s -> %.4f (thr=%.4f)",
        best_f1[0], best_f1[1],
        best_f1[2]["test_tuned_threshold"]["fraud_f1"],
        best_f1[2]["test_tuned_threshold"]["threshold"],
    )
    logger.info("Aggregated results: %s", csv_path)


if __name__ == "__main__":
    main()
