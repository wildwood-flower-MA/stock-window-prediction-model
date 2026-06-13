"""
Evaluation & Scoring Pipeline — LOB Price Movement Prediction
==============================================================
Loads a trained model checkpoint and produces a full classification report
including per-class precision/recall/F1, Matthews Correlation Coefficient,
Cohen's Kappa, and a confusion matrix plot.

Expects:
    data/
        X_test.npy   shape (N_test, 100, 40)   float32
        y_test.npy   shape (N_test,)             int64   values in {0,1,2}
    checkpoints/
        best_model.pt

Usage:
    conda activate ML-pytorch
    python evaluate.py --checkpoint checkpoints/best_model.pt --data_dir data
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")          # headless — no display required
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)

from model import build_model
from train import LOBDataset   # reuse Dataset defined in train.py

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

CLASS_NAMES = ["DOWN", "FLAT", "UP"]


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference over the full DataLoader.

    Returns:
        y_true  : (N,)    int64   ground-truth labels
        y_pred  : (N,)    int64   argmax predictions
        y_proba : (N, 3)  float32 softmax probabilities
    """
    model.eval()
    all_true  = []
    all_logits = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        logits  = model(X_batch)                        # (B, 3)
        all_logits.append(logits.cpu())
        all_true.append(y_batch)

    logits_cat = torch.cat(all_logits, dim=0)           # (N, 3)
    y_true     = torch.cat(all_true,  dim=0).numpy()    # (N,)
    y_proba    = F.softmax(logits_cat, dim=-1).numpy()  # (N, 3)
    y_pred     = logits_cat.argmax(dim=-1).numpy()      # (N,)

    return y_true, y_pred, y_proba


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def compute_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute a comprehensive set of classification metrics.

    Returns a dict with scalar metrics and per-class F1.
    """
    return {
        "accuracy"  : accuracy_score(y_true, y_pred),
        "f1_macro"  : f1_score(y_true, y_pred, average="macro",   zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_per_class": f1_score(y_true, y_pred, average=None,    zero_division=0),
        "mcc"        : matthews_corrcoef(y_true, y_pred),
        "kappa"      : cohen_kappa_score(y_true, y_pred),
        "conf_matrix": confusion_matrix(y_true, y_pred),
    }


def print_report(scores: dict, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Pretty-print all metrics to stdout."""
    sep = "-" * 52

    print(sep)
    print(f"  Accuracy          : {scores['accuracy']:.4f}")
    print(f"  F1 (macro)        : {scores['f1_macro']:.4f}")
    print(f"  F1 (weighted)     : {scores['f1_weighted']:.4f}")
    print(f"  Matthews CC       : {scores['mcc']:.4f}")
    print(f"  Cohen's Kappa     : {scores['kappa']:.4f}")
    print(sep)
    for i, name in enumerate(CLASS_NAMES):
        print(f"  F1 [{name:^4s}]        : {scores['f1_per_class'][i]:.4f}")
    print(sep)
    print("\nFull classification report:\n")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                                zero_division=0))


# ---------------------------------------------------------------------------
# Confusion matrix plot
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    output_path: str,
    normalise: bool = True,
) -> None:
    """
    Save a confusion-matrix heatmap to `output_path`.

    Args:
        cm          : raw confusion matrix from sklearn
        output_path : file path for the saved PNG
        normalise   : if True, show row-normalised values (recall per class)
    """
    if normalise:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot  = cm.astype(np.float64) / np.where(row_sums == 0, 1, row_sums)
        fmt      = ".2f"
        title    = "Confusion Matrix (row-normalised)"
    else:
        cm_plot = cm.astype(np.float64)
        fmt     = "d"
        title   = "Confusion Matrix (counts)"

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_plot, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set(
        xticks=range(len(CLASS_NAMES)),
        yticks=range(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted label",
        ylabel="True label",
        title=title,
    )

    thresh = cm_plot.max() / 2.0
    for i in range(cm_plot.shape[0]):
        for j in range(cm_plot.shape[1]):
            val  = cm_plot[i, j]
            text = f"{val:{fmt}}" if fmt == ".2f" else f"{int(val)}"
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if val > thresh else "black", fontsize=11)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> dict:
    # ── Device ──────────────────────────────────────────────────────────────
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"Device     : {device}")

    # ── Data ────────────────────────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    X_test   = np.load(data_dir / "X_test.npy", mmap_mode="r")
    y_test   = np.load(data_dir / "y_test.npy", mmap_mode="r")
    print(f"Test samples: {len(X_test):,}")

    test_ds     = LOBDataset(X_test, y_test)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model_name = getattr(args, "model_name", "LOBModel")
    model = build_model(
        model_name=model_name,
        in_channels=X_test.shape[2],
        num_classes=3,
    ).to(device)
    print(f"Model      : {model_name}")

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    print(f"Checkpoint : {checkpoint}")

    # ── Inference ────────────────────────────────────────────────────────────
    y_true, y_pred, y_proba = predict(model, test_loader, device)

    # ── Scores ───────────────────────────────────────────────────────────────
    scores = compute_scores(y_true, y_pred)
    print_report(scores, y_true, y_pred)

    # ── Outputs ──────────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Confusion matrix (both normalised and raw)
    plot_confusion_matrix(
        scores["conf_matrix"],
        output_path=str(out_dir / "confusion_matrix_normalised.png"),
        normalise=True,
    )
    plot_confusion_matrix(
        scores["conf_matrix"],
        output_path=str(out_dir / "confusion_matrix_counts.png"),
        normalise=False,
    )

    # Save predictions and probabilities for downstream analysis
    np.save(out_dir / "y_pred.npy",  y_pred)
    np.save(out_dir / "y_proba.npy", y_proba)

    # Save scalar metrics
    scalar_scores = {
        k: v for k, v in scores.items()
        if not isinstance(v, np.ndarray)
    }
    scalar_scores["f1_DOWN"] = float(scores["f1_per_class"][0])
    scalar_scores["f1_FLAT"] = float(scores["f1_per_class"][1])
    scalar_scores["f1_UP"]   = float(scores["f1_per_class"][2])

    metrics_path = out_dir / "metrics.txt"
    with open(metrics_path, "w") as f:
        for key, val in scalar_scores.items():
            f.write(f"{key}: {val:.6f}\n")

    print(f"Predictions  saved → {out_dir / 'y_pred.npy'}")
    print(f"Probabilities saved → {out_dir / 'y_proba.npy'}")
    print(f"Metrics      saved → {metrics_path}")

    return {
        "model_name": model_name,
        "scores": scores,
        "metrics_path": str(metrics_path),
        "output_dir": str(out_dir),
        "checkpoint": str(checkpoint),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained LOB model")

    p.add_argument("--checkpoint",  type=str, default="checkpoints/best_model.pt",
                   help="Path to saved model weights (.pt)")
    p.add_argument("--data_dir",    type=str, default="data",
                   help="Directory with X_test.npy and y_test.npy")
    p.add_argument("--output_dir",  type=str, default="results",
                   help="Directory to write metrics, plots, and predictions")
    p.add_argument("--batch_size",  type=int, default=128)
    p.add_argument("--cpu",         action="store_true",
                   help="Force CPU even if CUDA is available")
    p.add_argument("--model_name",  type=str, default="LOBModel",
                   help="Model architecture to evaluate: LOBModel or LSTMModel")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
