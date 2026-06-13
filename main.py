"""
main.py — Full Pipeline
========================
Runs all three stages in sequence:
  1. prepare_data  — parse FI-2010 .txt files → .npy arrays
    2. train         — train selected model architectures, save best checkpoints
    3. evaluate      — score them on test set, save metrics + comparison plots
Expected data layout:
    raw_data/
      BenchmarkDatasets/
        NoAuction/
          1.NoAuction_Zscore/
            NoAuction_Zscore_Training/  Train_Dst_NoAuction_ZScore_CF_*.txt
            NoAuction_Zscore_Testing/   Test_Dst_NoAuction_ZScore_CF_*.txt
Edit the CONFIG block below to match your setup, then run:
    conda activate ML-pytorch
    python main.py
"""

import types
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from prepare_data import prepare
from train import train
from evaluate import evaluate

# ===========================================================================
# CONFIG — edit these values
# ===========================================================================

CFG = types.SimpleNamespace(

    # ── Paths ────────────────────────────────────────────────────────────────
    raw_data_dir  = "raw_data",       # root folder containing BenchmarkDatasets/
    data_dir      = "data",           # output of prepare_data, input of train/eval
    output_dir    = "checkpoints",    # best_model.pt + history.npz
    results_dir   = "results",        # confusion matrix PNGs + metrics.txt

    # ── Dataset ──────────────────────────────────────────────────────────────
    horizon       = 4,                # 0=k1  1=k2  2=k3  3=k5  4=k10
    norm          = "ZScore",         # "ZScore" | "MinMax" | "DecPre"
    mode          = "NoAuction",      # "NoAuction" | "Auction"
    train_folds   = [1, 2, 3, 4, 5, 6, 7],
    val_folds     = [8],
    test_folds    = [9],

    # ── Training ─────────────────────────────────────────────────────────────
    epochs        = 10,                # quick test (was 50)
    batch_size    = 64,
    lr            = 1e-3,
    weight_decay  = 1e-4,
    patience      = 10,                # quick test (was 10)
    num_workers   = 0,
    cpu           = False,            # True → force CPU even if CUDA is present
    max_samples   = 5000,             # None → full dataset; int → cap train size

    # ── Evaluation ───────────────────────────────────────────────────────────
    eval_batch_size = 256,
    models          = ["LOBModel", "LSTMModel"],
)

# ===========================================================================

def _header(title: str) -> None:
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def _plot_history_comparison(histories: dict[str, dict[str, np.ndarray]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
    for model_name, history in histories.items():
        epochs = np.arange(1, len(history["train_acc"]) + 1)
        axes[0].plot(epochs, history["train_acc"], label=f"{model_name} train")
        axes[0].plot(epochs, history["test_acc"], linestyle="--", label=f"{model_name} test")

        axes[1].plot(epochs, history["train_loss"], label=f"{model_name} train")
        axes[1].plot(epochs, history["test_loss"], linestyle="--", label=f"{model_name} test")

    axes[0].set_title("Accuracy over epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Loss over epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Cross-entropy loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "train_test_comparison.png", dpi=150)
    plt.close(fig)


def _plot_confusion_matrix_comparison(results: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    class_names = ["DOWN", "FLAT", "UP"]
    for ax, result in zip(axes, results):
        cm = result["scores"]["conf_matrix"]
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot = cm.astype(np.float64) / np.where(row_sums == 0, 1, row_sums)

        im = ax.imshow(cm_plot, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set(
            xticks=range(len(class_names)),
            yticks=range(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
            xlabel="Predicted label",
            ylabel="True label",
            title=f"{result['model_name']}\nconfusion matrix",
        )

        thresh = cm_plot.max() / 2.0
        for i in range(cm_plot.shape[0]):
            for j in range(cm_plot.shape[1]):
                val = cm_plot[i, j]
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color="white" if val > thresh else "black",
                    fontsize=10,
                )

    fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix_comparison.png", dpi=150)
    plt.close(fig)


def _write_comparison_summary(
    histories: dict[str, dict[str, np.ndarray]],
    eval_results: list[dict],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "comparison_summary.txt"

    metrics_by_model = {result["model_name"]: result["scores"] for result in eval_results}

    with open(summary_path, "w", encoding="utf-8") as f:
        for model_name, history in histories.items():
            scores = metrics_by_model[model_name]
            best_epoch = int(history["best_epoch"][0]) if history["best_epoch"].size else -1
            last_idx = -1

            f.write(f"{model_name}\n")
            f.write(f"  best_epoch: {best_epoch}\n")
            f.write(f"  final_train_acc: {history['train_acc'][last_idx]:.6f}\n")
            f.write(f"  final_test_acc: {history['test_acc'][last_idx]:.6f}\n")
            f.write(f"  final_train_loss: {history['train_loss'][last_idx]:.6f}\n")
            f.write(f"  final_test_loss: {history['test_loss'][last_idx]:.6f}\n")
            f.write(f"  test_accuracy: {scores['accuracy']:.6f}\n")
            f.write(f"  test_f1_macro: {scores['f1_macro']:.6f}\n")
            f.write(f"  test_mcc: {scores['mcc']:.6f}\n")
            f.write(f"  test_kappa: {scores['kappa']:.6f}\n")
            f.write("\n")


def main() -> None:

    # # ── Stage 1: Data preparation ────────────────────────────────────────────
    # _header("STAGE 1 / 3 — Data Preparation")

    # prepare(types.SimpleNamespace(
    #     data_dir    = CFG.raw_data_dir,
    #     output_dir  = CFG.data_dir,
    #     horizon     = CFG.horizon,
    #     norm        = CFG.norm,
    #     mode        = CFG.mode,
    #     train_folds = CFG.train_folds,
    #     val_folds   = CFG.val_folds,
    #     test_folds  = CFG.test_folds,
    # ))

    training_runs = []
    histories = {}

    # ── Stage 2: Training ────────────────────────────────────────────────────
    _header("STAGE 2 / 3 — Training")

    for model_name in CFG.models:
        _header(f"Training {model_name}")
        run = train(types.SimpleNamespace(
            data_dir     = CFG.data_dir,
            output_dir   = str(Path(CFG.output_dir) / model_name),
            epochs       = CFG.epochs,
            batch_size   = CFG.batch_size,
            lr           = CFG.lr,
            weight_decay = CFG.weight_decay,
            patience     = CFG.patience,
            num_workers  = CFG.num_workers,
            cpu          = CFG.cpu,
            max_samples  = CFG.max_samples,
            model_name   = model_name,
        ))
        training_runs.append(run)
        histories[model_name] = {
            key: value for key, value in np.load(run["history_path"]).items()
        }

    # ── Stage 3: Evaluation ──────────────────────────────────────────────────
    _header("STAGE 3 / 3 — Evaluation")

    eval_results = []
    for run in training_runs:
        model_name = run["model_name"]
        _header(f"Evaluating {model_name}")
        result = evaluate(types.SimpleNamespace(
            checkpoint  = run["checkpoint"],
            data_dir    = CFG.data_dir,
            output_dir  = str(Path(CFG.results_dir) / model_name),
            batch_size  = CFG.eval_batch_size,
            cpu         = CFG.cpu,
            model_name  = model_name,
        ))
        eval_results.append(result)

    comparison_dir = Path(CFG.results_dir) / "comparison"
    _plot_history_comparison(histories, comparison_dir)
    _plot_confusion_matrix_comparison(eval_results, comparison_dir)
    _write_comparison_summary(histories, eval_results, comparison_dir)

    _header("DONE")
    for run in training_runs:
        print(f"  {run['model_name']} checkpoint  : {run['checkpoint']}")
    print(f"  Comparison plots : {comparison_dir / 'train_test_comparison.png'}")
    print(f"  Conf. matrices   : {comparison_dir / 'confusion_matrix_comparison.png'}")
    print(f"  Summary          : {comparison_dir / 'comparison_summary.txt'}")


if __name__ == "__main__":
    main()
