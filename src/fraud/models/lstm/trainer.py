"""Training loop for FraudLSTM.

Drives one full run: load → split → fit preprocessor → build sequences →
optional SMOTE → train with early stop on val PR-AUC → tune F2 threshold →
evaluate on the held-out test CSV → save artifacts.
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from fraud.data import feature_engineering, load_raw_csv, temporal_split
from fraud.evaluation import compute_metrics, format_report, tune_threshold_f2
from fraud.preprocessing import (
    FEATURE_COLS,
    build_preprocessor,
    save_preprocessor,
)
from fraud.utils import save_json, set_seed

from .dataset import SequenceDataset, collate_fn
from .model import FraudLSTM
from .sequences import build_sequences
from .smote import sequence_smote

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    train_csv: str
    test_csv: str
    out_dir: str
    seq_len: int = 20
    proj_dim: int = 64
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    batch_size: int = 256
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    early_stop_patience: int = 5
    use_smote: bool = False
    smote_sampling_strategy: float | str = 0.1
    val_fraction: float = 0.15
    seed: int = 42
    device: str = "auto"
    subset_frac: float = 1.0


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _prepare_features(
    df: pd.DataFrame, preprocessor, fit: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_features, y, card_ids, timestamps_int) for sequence building."""
    if fit:
        X = preprocessor.fit_transform(df[FEATURE_COLS], df["is_fraud"])
    else:
        X = preprocessor.transform(df[FEATURE_COLS])
    X = np.asarray(X, dtype=np.float32)
    y = df["is_fraud"].to_numpy(dtype=np.int64)
    card_ids = df["cc_num"].to_numpy()
    ts = df["trans_date_trans_time"].astype("int64").to_numpy()
    return X, y, card_ids, ts


def _epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss = 0.0
    total = 0
    all_y = []
    all_p = []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, lengths, y in loader:
            x = x.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x, lengths)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total_loss += loss.item() * y.size(0)
            total += y.size(0)
            all_y.append(y.detach().cpu().numpy())
            all_p.append(torch.sigmoid(logits).detach().cpu().numpy())
    y_arr = np.concatenate(all_y)
    p_arr = np.concatenate(all_p)
    pr_auc = average_precision_score(y_arr, p_arr) if y_arr.sum() > 0 else float("nan")
    return total_loss / max(total, 1), pr_auc, y_arr, p_arr


def train_lstm(cfg: TrainConfig) -> dict[str, Any]:
    set_seed(cfg.seed)
    device = _resolve_device(cfg.device)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Using device: %s", device)

    # ---- Load + feature engineer ----
    raw_train = load_raw_csv(cfg.train_csv)
    if cfg.subset_frac < 1.0:
        raw_train = raw_train.sample(frac=cfg.subset_frac, random_state=cfg.seed).reset_index(drop=True)
    df_train_full = feature_engineering(raw_train)
    raw_test = load_raw_csv(cfg.test_csv)
    df_test = feature_engineering(raw_test)

    # ---- Temporal train/val split ----
    df_train, df_val = temporal_split(df_train_full, val_fraction=cfg.val_fraction)
    logger.info("Train rows: %d  Val rows: %d  Test rows: %d", len(df_train), len(df_val), len(df_test))

    # ---- Preprocessor: fit on train slice only ----
    preprocessor = build_preprocessor()
    X_train, y_train, c_train, t_train = _prepare_features(df_train, preprocessor, fit=True)
    X_val, y_val, c_val, t_val = _prepare_features(df_val, preprocessor, fit=False)
    X_test, y_test, c_test, t_test = _prepare_features(df_test, preprocessor, fit=False)
    n_features = X_train.shape[1]
    logger.info("Feature dim: %d", n_features)

    # ---- Build sequences (per-card sliding windows) ----
    train_seq = build_sequences(X_train, y_train, c_train, t_train, seq_len=cfg.seq_len)
    val_seq = build_sequences(X_val, y_val, c_val, t_val, seq_len=cfg.seq_len)
    test_seq = build_sequences(X_test, y_test, c_test, t_test, seq_len=cfg.seq_len)
    logger.info(
        "Sequences -> train: %d  val: %d  test: %d  (positives: %d / %d / %d)",
        len(train_seq.y), len(val_seq.y), len(test_seq.y),
        int(train_seq.y.sum()), int(val_seq.y.sum()), int(test_seq.y.sum()),
    )

    # ---- Optional SMOTE on train sequences ----
    if cfg.use_smote:
        logger.info(
            "Applying sequence-level SMOTE to train set (sampling_strategy=%s)",
            cfg.smote_sampling_strategy,
        )
        train_X, train_y, train_lengths = sequence_smote(
            train_seq.X,
            train_seq.y,
            train_seq.lengths,
            random_state=cfg.seed,
            sampling_strategy=cfg.smote_sampling_strategy,
        )
        pos_weight = torch.tensor([1.0], device=device)
    else:
        train_X, train_y, train_lengths = train_seq.X, train_seq.y, train_seq.lengths
        n_pos = int(train_y.sum())
        n_neg = int(len(train_y) - n_pos)
        pos_weight_value = (n_neg / max(n_pos, 1)) if n_pos > 0 else 1.0
        pos_weight = torch.tensor([pos_weight_value], device=device)
        logger.info("pos_weight = %.2f (neg=%d, pos=%d)", pos_weight_value, n_neg, n_pos)

    train_ds = SequenceDataset(train_X, train_y, train_lengths)
    val_ds = SequenceDataset(val_seq.X, val_seq.y, val_seq.lengths)
    test_ds = SequenceDataset(test_seq.X, test_seq.y, test_seq.lengths)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    # ---- Model + optimizer ----
    model = FraudLSTM(
        in_features=n_features,
        proj_dim=cfg.proj_dim,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # ---- Training loop with early stopping on val PR-AUC ----
    log_path = out_dir / "training_log.csv"
    best_state = None
    best_val_pr = -1.0
    epochs_since_improvement = 0
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_pr_auc", "val_loss", "val_pr_auc"])
        for epoch in range(1, cfg.epochs + 1):
            train_loss, train_pr, _, _ = _epoch(model, train_loader, criterion, optimizer, device, train=True)
            val_loss, val_pr, _, _ = _epoch(model, val_loader, criterion, optimizer, device, train=False)
            writer.writerow([epoch, train_loss, train_pr, val_loss, val_pr])
            f.flush()
            logger.info(
                "Epoch %02d  train_loss=%.4f train_PR=%.4f  val_loss=%.4f val_PR=%.4f",
                epoch, train_loss, train_pr, val_loss, val_pr,
            )
            if val_pr > best_val_pr:
                best_val_pr = val_pr
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
                if epochs_since_improvement >= cfg.early_stop_patience:
                    logger.info("Early stopping at epoch %d (best val PR-AUC=%.4f)", epoch, best_val_pr)
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- Threshold tuning on val ----
    _, _, y_val_arr, p_val_arr = _epoch(model, val_loader, criterion, optimizer, device, train=False)
    best_thr, best_f2 = tune_threshold_f2(y_val_arr, p_val_arr)
    logger.info("Tuned threshold: %.4f  (F2=%.4f)", best_thr, best_f2)

    # ---- Test evaluation ----
    _, _, y_test_arr, p_test_arr = _epoch(model, test_loader, criterion, optimizer, device, train=False)
    test_metrics_default = compute_metrics(y_test_arr, p_test_arr, threshold=0.5)
    test_metrics_tuned = compute_metrics(y_test_arr, p_test_arr, threshold=best_thr)
    logger.info("\n%s", format_report(test_metrics_default, name="LSTM (thr=0.5)"))
    logger.info("\n%s", format_report(test_metrics_tuned, name="LSTM (tuned thr)"))

    # ---- Save artifacts ----
    torch.save(model.state_dict(), out_dir / "model.pt")
    save_preprocessor(preprocessor, out_dir / "preprocessor.pkl")
    save_json({"threshold": best_thr, "f2_on_val": best_f2}, out_dir / "threshold.json")
    save_json(
        {
            "config": cfg.__dict__,
            "n_features": int(n_features),
            "best_val_pr_auc": float(best_val_pr),
            "test_default_threshold": test_metrics_default,
            "test_tuned_threshold": test_metrics_tuned,
        },
        out_dir / "metrics.json",
    )
    # Persist test probabilities for downstream McNemar comparisons.
    np.savez(
        out_dir / "test_predictions.npz",
        y_true=y_test_arr,
        y_proba=p_test_arr,
        threshold=np.asarray([best_thr], dtype=np.float64),
    )
    return {
        "best_val_pr_auc": float(best_val_pr),
        "test_default_threshold": test_metrics_default,
        "test_tuned_threshold": test_metrics_tuned,
        "threshold": best_thr,
    }
