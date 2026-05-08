"""Lightweight plotting helpers for training diagnostics.

Used by the hyperparameter sweep driver to drop per-trial PNGs alongside
the existing CSV/JSON artifacts. Re-exported from `fraud.evaluation`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # headless-safe; must be set before pyplot import.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_training_curves(log_csv_path: str | Path, out_png_path: str | Path, title: str | None = None) -> None:
    """Render a 1x2 figure: (train_loss vs val_loss) and (train_pr_auc vs val_pr_auc).

    The vertical dashed line marks the epoch with the best validation PR-AUC,
    which is the checkpoint train_lstm() actually saves.
    """
    df = pd.read_csv(log_csv_path)
    best_epoch = int(df.loc[df["val_pr_auc"].idxmax(), "epoch"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(df["epoch"], df["train_loss"], label="train", marker="o", markersize=3)
    axes[0].plot(df["epoch"], df["val_loss"], label="val", marker="o", markersize=3)
    axes[0].axvline(best_epoch, linestyle="--", color="gray", alpha=0.7, label=f"best epoch ({best_epoch})")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("BCE loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(df["epoch"], df["train_pr_auc"], label="train", marker="o", markersize=3)
    axes[1].plot(df["epoch"], df["val_pr_auc"], label="val", marker="o", markersize=3)
    axes[1].axvline(best_epoch, linestyle="--", color="gray", alpha=0.7)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("PR-AUC")
    axes[1].set_title("PR-AUC")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png_path, dpi=110)
    plt.close(fig)


def plot_confusion_matrix(
    cm: Sequence[Sequence[int]],
    out_png_path: str | Path,
    title: str,
    class_names: Sequence[str] = ("legit", "fraud"),
) -> None:
    """Render a 2x2 confusion-matrix heatmap with raw counts.

    `cm` must be the [[tn, fp], [fn, tp]] shape that `compute_metrics`
    already produces and writes into metrics.json.
    """
    cm_arr = np.asarray(cm, dtype=np.int64)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm_arr, cmap="Blues")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(title)

    threshold = cm_arr.max() / 2.0 if cm_arr.size else 0
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(
                j,
                i,
                f"{cm_arr[i, j]:,}",
                ha="center",
                va="center",
                color="white" if cm_arr[i, j] > threshold else "black",
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png_path, dpi=110)
    plt.close(fig)
