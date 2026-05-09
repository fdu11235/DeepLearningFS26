"""
Generates all additional plots for the Deep Learning FS26 fraud detection paper.

Reads from:
  models/lstm/test_predictions.npz, threshold.json, training_log.csv
  models/lstm_smote/test_predictions.npz, threshold.json, training_log.csv
  models/lstm_no_imbalance/test_predictions.npz, threshold.json, training_log.csv
  models/tuning/lstm_lr_dropout_v1/tuning_results.csv

Outputs to:
  images/graphs/fig_pr_curves_lstm.png       (PR curves with thresholds)
  images/graphs/fig_roc_curves_lstm.png      (ROC curves overlaid)
  images/graphs/fig_score_distribution.png   (histograms of predicted scores)
  images/graphs/fig_threshold_sweep.png      (P, R, F1, F2 vs threshold)
  images/graphs/fig_hyperparam_heatmap.png   (lr x dropout heatmap)
  images/graphs/fig_training_curves_all.png  (3 variants overlaid)
  images/graphs/fig_confusion_matrices.png   (3 variants x 2 thresholds)
  images/graphs/fig_calibration.png          (reliability diagram)

Usage:
    python generate_plots.py

Run from the repository root (where 'models/' lives).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, roc_auc_score, precision_recall_curve, average_precision_score,
    confusion_matrix, fbeta_score, precision_score, recall_score, f1_score,
)

# ---------- Configuration ----------

VARIANTS = [
    ("lstm_no_imbalance", "Plain BCE",  "#1f77b4"),
    ("lstm",              "pos_weight≈172", "#d62728"),
    ("lstm_smote",        "SMOTE",     "#2ca02c"),
]

ROOT      = Path(".")
MODELS    = ROOT / "models"
OUT_DIR   = ROOT / "images" / "graphs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


# ---------- Helpers ----------

def load_variant(name):
    npz = np.load(MODELS / name / "test_predictions.npz")
    with open(MODELS / name / "threshold.json") as f:
        thr = json.load(f)["threshold"]
    log = pd.read_csv(MODELS / name / "training_log.csv")
    return npz["y_true"], npz["y_proba"], thr, log


# ---------- 1. PR curves overlaid with threshold markers ----------

def plot_pr_curves():
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for name, label, color in VARIANTS:
        y, p, thr, _ = load_variant(name)
        precision, recall, ths = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)

        ax.plot(recall, precision, label=f"{label} (AP={ap:.3f})", color=color, lw=1.8)

        # mark default threshold (0.5)
        idx_def = np.argmin(np.abs(ths - 0.5)) if (ths.size and ths.min() <= 0.5 <= ths.max()) else None
        if idx_def is not None:
            ax.scatter(recall[idx_def], precision[idx_def],
                       marker="o", color=color, edgecolor="white", s=80, zorder=5)

        # mark tuned threshold
        idx_tun = np.argmin(np.abs(ths - thr)) if ths.size else None
        if idx_tun is not None:
            ax.scatter(recall[idx_tun], precision[idx_tun],
                       marker="*", color=color, edgecolor="white", s=200, zorder=5)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_title("Precision-Recall curves on test set\n(circle = default 0.5, star = F2-tuned)")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.savefig(OUT_DIR / "fig_pr_curves_lstm.png")
    plt.close(fig)


# ---------- 2. ROC curves overlaid ----------

def plot_roc_curves():
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for name, label, color in VARIANTS:
        y, p, _, _ = load_variant(name)
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.4f})", color=color, lw=1.8)

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves on test set")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.savefig(OUT_DIR / "fig_roc_curves_lstm.png")
    plt.close(fig)


# ---------- 3. Score distribution histograms ----------

def plot_score_distribution():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, (name, label, color) in zip(axes, VARIANTS):
        y, p, thr, _ = load_variant(name)
        bins = np.linspace(0, 1, 51)
        ax.hist(p[y == 0], bins=bins, alpha=0.6, color="#888888",
                label="Non-fraud", density=True)
        ax.hist(p[y == 1], bins=bins, alpha=0.7, color=color,
                label="Fraud", density=True)
        ax.axvline(thr, color="black", ls="--", lw=1.2, label=f"Tuned τ={thr:.3f}")
        ax.set_yscale("log")
        ax.set_title(label)
        ax.set_xlabel("Predicted probability")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Density (log)")
    fig.suptitle("Predicted score distributions on test set", y=1.02)
    fig.savefig(OUT_DIR / "fig_score_distribution.png")
    plt.close(fig)


# ---------- 4. Threshold sweep ----------

def plot_threshold_sweep():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, (name, label, color) in zip(axes, VARIANTS):
        y, p, thr, _ = load_variant(name)
        thr_grid = np.linspace(0.001, 0.999, 200)
        precs, recs, f1s, f2s = [], [], [], []
        for t in thr_grid:
            yhat = (p >= t).astype(int)
            precs.append(precision_score(y, yhat, zero_division=0))
            recs.append(recall_score(y, yhat, zero_division=0))
            f1s.append(f1_score(y, yhat, zero_division=0))
            f2s.append(fbeta_score(y, yhat, beta=2.0, zero_division=0))

        ax.plot(thr_grid, precs, label="Precision", color="#1f77b4", lw=1.5)
        ax.plot(thr_grid, recs,  label="Recall",    color="#d62728", lw=1.5)
        ax.plot(thr_grid, f1s,   label="F1",        color="#2ca02c", lw=1.2, ls=":")
        ax.plot(thr_grid, f2s,   label="F2",        color="#9467bd", lw=1.5, ls="--")
        ax.axvline(thr, color="black", lw=1.0, alpha=0.6, label=f"Tuned τ={thr:.3f}")
        ax.set_title(label)
        ax.set_xlabel("Threshold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower center", fontsize=8, ncol=2)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Score")
    fig.suptitle("Threshold sweep (P, R, F1, F2 vs decision threshold)", y=1.03)
    fig.savefig(OUT_DIR / "fig_threshold_sweep.png")
    plt.close(fig)


# ---------- 5. Hyperparameter heatmap ----------

def plot_hyperparam_heatmap():
    # The plain-BCE / no_imbalance arm is the paper's baseline, so we visualise
    # the grid that was actually run on it. The earlier `lstm_lr_dropout_v1`
    # sweep (cost-sensitive base) lives at models/tuning/ but is no longer the
    # one whose picked config drives the comparison in Section 5.
    df = pd.read_csv(MODELS / "tuning" / "lstm_no_imbalance_lr_dropout_v1" / "tuning_results.csv")
    pivot = df.pivot(index="dropout", columns="lr", values="best_val_pr_auc")

    fig, ax = plt.subplots(figsize=(6.5, 4))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.0e}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{d}" for d in pivot.index])
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Dropout")
    ax.set_title("Hyperparameter grid: validation PR-AUC")

    for i, dr in enumerate(pivot.index):
        for j, lr in enumerate(pivot.columns):
            v = pivot.iloc[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    color="white" if v < pivot.values.mean() else "black",
                    fontsize=10, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Val PR-AUC")
    fig.savefig(OUT_DIR / "fig_hyperparam_heatmap.png")
    plt.close(fig)


# ---------- 6. Training curves of all 3 variants overlaid ----------

def plot_training_curves_all():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for name, label, color in VARIANTS:
        _, _, _, log = load_variant(name)
        axes[0].plot(log["epoch"], log["train_loss"], color=color, ls="-", lw=1.5,
                     label=f"{label} (train)")
        axes[0].plot(log["epoch"], log["val_loss"], color=color, ls="--", lw=1.5,
                     label=f"{label} (val)")
        axes[1].plot(log["epoch"], log["val_pr_auc"], color=color, lw=1.8,
                     label=label, marker="o", markersize=4)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss curves")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation PR-AUC")
    axes[1].set_title("Validation PR-AUC by epoch")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.savefig(OUT_DIR / "fig_training_curves_all.png")
    plt.close(fig)


# ---------- 7. Confusion matrices grid (3 variants x 2 thresholds) ----------

def plot_confusion_matrices():
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for col, (name, label, color) in enumerate(VARIANTS):
        y, p, thr, _ = load_variant(name)
        for row, (t_name, t_val) in enumerate([("Default τ=0.5", 0.5),
                                               (f"Tuned τ={thr:.3f}", thr)]):
            yhat = (p >= t_val).astype(int)
            cm = confusion_matrix(y, yhat)
            ax = axes[row, col]
            im = ax.imshow(cm, cmap="Blues", aspect="auto")
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(["Pred 0", "Pred 1"])
            ax.set_yticklabels(["True 0", "True 1"])
            ax.set_title(f"{label}\n{t_name}", fontsize=10)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                            color="white" if cm[i, j] > cm.max() / 2 else "black",
                            fontsize=10, fontweight="bold")
    fig.suptitle("Confusion matrices: 3 LSTM variants × 2 thresholds", y=1.02)
    fig.savefig(OUT_DIR / "fig_confusion_matrices.png")
    plt.close(fig)


# ---------- 8. Calibration / reliability plot ----------

def plot_calibration():
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    bins = np.linspace(0, 1, 11)
    for name, label, color in VARIANTS:
        y, p, _, _ = load_variant(name)
        bin_idx = np.digitize(p, bins) - 1
        bin_idx = np.clip(bin_idx, 0, len(bins) - 2)
        bin_centers, true_freq = [], []
        for b in range(len(bins) - 1):
            mask = bin_idx == b
            if mask.sum() > 100:
                bin_centers.append(p[mask].mean())
                true_freq.append(y[mask].mean())
        ax.plot(bin_centers, true_freq, "o-", color=color, label=label, lw=1.5, ms=6)

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraud frequency")
    ax.set_title("Calibration (reliability diagram)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(OUT_DIR / "fig_calibration.png")
    plt.close(fig)


# ---------- Main ----------

if __name__ == "__main__":
    print("Generating plots...")
    plot_pr_curves();             print("  ✓ fig_pr_curves_lstm.png")
    plot_roc_curves();            print("  ✓ fig_roc_curves_lstm.png")
    plot_score_distribution();    print("  ✓ fig_score_distribution.png")
    plot_threshold_sweep();       print("  ✓ fig_threshold_sweep.png")
    plot_hyperparam_heatmap();    print("  ✓ fig_hyperparam_heatmap.png")
    plot_training_curves_all();   print("  ✓ fig_training_curves_all.png")
    plot_confusion_matrices();    print("  ✓ fig_confusion_matrices.png")
    plot_calibration();           print("  ✓ fig_calibration.png")
    print(f"\nAll plots saved to {OUT_DIR}/")
