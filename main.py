"""
main.py — Full Pipeline
========================
Runs all three stages in sequence:
  1. prepare_data  — parse FI-2010 .txt files → .npy arrays
  2. train         — train LOBModel, save best checkpoint
  3. evaluate      — score on test set, save metrics + confusion matrix plots
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
    epochs        = 50,
    batch_size    = 64,
    lr            = 1e-3,
    weight_decay  = 1e-4,
    patience      = 10,               # early-stopping patience
    num_workers   = 0,
    cpu           = False,            # True → force CPU even if CUDA is present

    # ── Evaluation ───────────────────────────────────────────────────────────
    eval_batch_size = 256,
)

# ===========================================================================

def _header(title: str) -> None:
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


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

    # ── Stage 2: Training ────────────────────────────────────────────────────
    _header("STAGE 2 / 3 — Training")

    train(types.SimpleNamespace(
        data_dir     = CFG.data_dir,
        output_dir   = CFG.output_dir,
        epochs       = CFG.epochs,
        batch_size   = CFG.batch_size,
        lr           = CFG.lr,
        weight_decay = CFG.weight_decay,
        patience     = CFG.patience,
        num_workers  = CFG.num_workers,
        cpu          = CFG.cpu,
    ))

    # ── Stage 3: Evaluation ──────────────────────────────────────────────────
    _header("STAGE 3 / 3 — Evaluation")

    checkpoint = str(Path(CFG.output_dir) / "best_model.pt")

    evaluate(types.SimpleNamespace(
        checkpoint  = checkpoint,
        data_dir    = CFG.data_dir,
        output_dir  = CFG.results_dir,
        batch_size  = CFG.eval_batch_size,
        cpu         = CFG.cpu,
    ))

    _header("DONE")
    print(f"  Checkpoint  : {checkpoint}")
    print(f"  Metrics     : {Path(CFG.results_dir) / 'metrics.txt'}")
    print(f"  Conf. matrix: {Path(CFG.results_dir) / 'confusion_matrix_normalised.png'}")


if __name__ == "__main__":
    main()
