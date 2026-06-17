"""
Data Preparation — FI-2010 LOB Dataset
========================================
Parses the raw FI-2010 .txt files and produces train/val/test numpy arrays
ready for train.py and evaluate.py.

Verified dataset structure (shape confirmed from actual files):
  - Each .txt file shape : (149, N_events)   e.g. (149, 39512)
  - Rows 0..39   : 40 raw LOB features (bid/ask price+vol, 10 levels each)
  - Rows 40..142 : engineered time-sensitive / time-insensitive features
  - Row  143     : zeros — separator row
  - Rows 144..148: labels for horizons k = 1, 2, 3, 5, 10

Original labels: 1=UP  2=FLAT  3=DOWN
Our convention : 0=DOWN  1=FLAT  2=UP

Actual directory layout on disk:

    raw_data/
      BenchmarkDatasets/
        NoAuction/
          1.NoAuction_Zscore/
            NoAuction_Zscore_Training/
              Train_Dst_NoAuction_ZScore_CF_1.txt  ...  CF_9.txt
            NoAuction_Zscore_Testing/
              Test_Dst_NoAuction_ZScore_CF_1.txt   ...  CF_9.txt
          2.NoAuction_MinMax/  ...
          3.NoAuction_DecPre/  ...
        Auction/  ...

Usage:
    conda activate ML-pytorch
    python prepare_data.py --data_dir raw_data --output_dir data --horizon 4
                                                                  # horizon index:
                                                                  # 0=k1 1=k2 2=k3
                                                                  # 3=k5  4=k10
Output (in --output_dir):
    X_train.npy   (N_train, 100, 40)  float32
    y_train.npy   (N_train,)           int64
    X_val.npy     (N_val,   100, 40)  float32
    y_val.npy     (N_val,  )           int64
    X_test.npy    (N_test,  100, 40)  float32
    y_test.npy    (N_test, )           int64
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_LOB_FEATURES = 40        # raw order book features (rows 0–39)
N_TOTAL_ROWS   = 149       # total rows per .txt file
LABEL_ROW_OFFSET = 144     # rows 144–148 are labels (row 143 = zero separator)
WINDOW_SIZE    = 100       # sequence length fed to the model

# FI-2010 label → our label
LABEL_MAP = {1: 2, 2: 1, 3: 0}   # 1=UP→2,  2=FLAT→1,  3=DOWN→0

# Maps (mode, norm) → (numbered_dir, training_subdir, testing_subdir, file_norm_token)
# Directory names use "Zscore" (lowercase z); file names use "ZScore" (uppercase Z).
PATH_MAP: dict[tuple[str, str], tuple[str, str, str, str]] = {
    ("NoAuction", "ZScore"): (
        "1.NoAuction_Zscore",
        "NoAuction_Zscore_Training",
        "NoAuction_Zscore_Testing",
        "ZScore",
    ),
    ("NoAuction", "MinMax"): (
        "2.NoAuction_MinMax",
        "NoAuction_MinMax_Training",
        "NoAuction_MinMax_Testing",
        "MinMax",
    ),
    ("NoAuction", "DecPre"): (
        "3.NoAuction_DecPre",
        "NoAuction_DecPre_Training",
        "NoAuction_DecPre_Testing",
        "DecPre",
    ),
    ("Auction", "ZScore"): (
        "1.Auction_Zscore",
        "Auction_Zscore_Training",
        "Auction_Zscore_Testing",
        "ZScore",
    ),
    ("Auction", "MinMax"): (
        "2.Auction_MinMax",
        "Auction_MinMax_Training",
        "Auction_MinMax_Testing",
        "MinMax",
    ),
    ("Auction", "DecPre"): (
        "3.Auction_DecPre",
        "Auction_DecPre_Training",
        "Auction_DecPre_Testing",
        "DecPre",
    ),
}


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

def load_txt(path: str) -> np.ndarray:
    """
    Load a single FI-2010 .txt file.

    Uses pandas.read_csv (space-delimited, no header) — significantly faster
    than numpy.loadtxt for large files.

    Returns
    -------
    data : np.ndarray  shape (149, N_events)  float64
    """
    data = pd.read_csv(path, sep=r"\s+", header=None).to_numpy()  # (149, N_events)
    assert data.shape[0] == N_TOTAL_ROWS, (
        f"Expected {N_TOTAL_ROWS} rows, got {data.shape[0]} in {path}"
    )
    return data


def load_files(file_paths: list[str]) -> np.ndarray:
    """
    Load and horizontally concatenate a list of FI-2010 files.

    Returns
    -------
    data : np.ndarray  shape (149, N_total_events)
    """
    parts = [load_txt(p) for p in sorted(file_paths)]
    return np.concatenate(parts, axis=1)      # concatenate along time axis


def resolve_paths(
    data_dir: str,
    mode: str,
    norm: str,
    split: str,           # "train" or "test"
    folds: list[int],
) -> list[str]:
    """
    Build absolute file paths for the requested folds.

    data_dir is the root that contains BenchmarkDatasets/.
    split must be "train" or "test".
    """
    key = (mode, norm)
    if key not in PATH_MAP:
        raise ValueError(
            f"Unknown (mode, norm) combination: {key}. "
            f"Valid options: {list(PATH_MAP.keys())}"
        )
    numbered_dir, train_sub, test_sub, file_norm = PATH_MAP[key]

    subdir    = train_sub if split == "train" else test_sub
    prefix    = "Train"   if split == "train" else "Test"
    base      = Path(data_dir) / "BenchmarkDatasets" / mode / numbered_dir / subdir

    paths = []
    for fold in folds:
        fname = f"{prefix}_Dst_{mode}_{file_norm}_CF_{fold}.txt"
        full  = base / fname
        if not full.exists():
            raise FileNotFoundError(
                f"File not found: {full}\n"
                f"Check --data_dir (currently: {data_dir}), --mode and --norm."
            )
        paths.append(str(full))
    return paths


# ---------------------------------------------------------------------------
# Window extraction
# ---------------------------------------------------------------------------

def make_windows(
    data: np.ndarray,
    horizon_idx: int,
    window: int = WINDOW_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build sliding windows from a (149, N) data array.

    Parameters
    ----------
    data        : (149, N)  raw file data
    horizon_idx : which label row to use  (0=k1, 1=k2, 2=k3, 3=k5, 4=k10)
    window      : number of time steps per sample

    Returns
    -------
    X : (M, window, 40)  float32
    y : (M,)              int64     values in {0, 1, 2}
    """
    features = data[:N_LOB_FEATURES, :].T.astype(np.float32)  # (N, 40)
    raw_labels = data[LABEL_ROW_OFFSET + horizon_idx, :]       # (N,)

    # Map labels: 1→2 (UP), 2→1 (FLAT), 3→0 (DOWN)
    labels = np.vectorize(LABEL_MAP.__getitem__)(raw_labels.astype(int))

    n_samples = len(features) - window + 1
    if n_samples <= 0:
        raise ValueError(
            f"Not enough events ({len(features)}) for window size {window}"
        )

    # Use the label of the LAST event in each window (DeepLOB convention)
    X = np.lib.stride_tricks.sliding_window_view(
        features, window_shape=(window, N_LOB_FEATURES)
    ).reshape(n_samples, window, N_LOB_FEATURES)

    y = labels[window - 1:]   # label of the last step in each window

    return X, y.astype(np.int64)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def prepare(args: argparse.Namespace) -> None:
    data_dir   = args.data_dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    horizon_names = {0: "k=1", 1: "k=2", 2: "k=3", 3: "k=5", 4: "k=10"}
    print(f"Mode        : {args.mode}")
    print(f"Norm        : {args.norm}")
    print(f"Horizon     : {horizon_names[args.horizon]}  (index {args.horizon})")

    # ── Discover files ───────────────────────────────────────────────────────
    train_files = resolve_paths(data_dir, args.mode, args.norm, "train", args.train_folds)
    val_files   = resolve_paths(data_dir, args.mode, args.norm, "train", args.val_folds)
    test_files  = resolve_paths(data_dir, args.mode, args.norm, "test",  args.test_folds)

    print(f"Train files : {[Path(p).name for p in train_files]}")
    print(f"Val files   : {[Path(p).name for p in val_files]}")
    print(f"Test files  : {[Path(p).name for p in test_files]}")

    # ── Build arrays ─────────────────────────────────────────────────────────
    print("\nLoading and windowing data...")

    X_train, y_train = make_windows(load_files(train_files), args.horizon)
    X_val,   y_val   = make_windows(load_files(val_files),   args.horizon)
    X_test,  y_test  = make_windows(load_files(test_files),  args.horizon)

    # ── Class distribution ────────────────────────────────────────────────────
    def dist(y):
        c = np.bincount(y, minlength=3)
        total = c.sum()
        return "  ".join(
            f"{name}={c[i]} ({100*c[i]/total:.1f}%)"
            for i, name in enumerate(["DOWN", "FLAT", "UP"])
        )

    print(f"\nTrain : {len(X_train):>7,} samples  |  {dist(y_train)}")
    print(f"Val   : {len(X_val):>7,} samples  |  {dist(y_val)}")
    print(f"Test  : {len(X_test):>7,} samples  |  {dist(y_test)}")

    # ── Save ─────────────────────────────────────────────────────────────────
    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "X_val.npy",   X_val)
    np.save(output_dir / "y_val.npy",   y_val)
    np.save(output_dir / "X_test.npy",  X_test)
    np.save(output_dir / "y_test.npy",  y_test)

    print(f"\nSaved to: {output_dir.resolve()}")
    print(f"  X_train.npy : {X_train.shape}  dtype={X_train.dtype}")
    print(f"  X_test.npy  : {X_test.shape}   dtype={X_test.dtype}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare FI-2010 LOB data for training"
    )
    p.add_argument("--data_dir",    type=str, default="raw_data",
                   help="Directory containing the raw FI-2010 .txt files")
    p.add_argument("--output_dir",  type=str, default="data",
                   help="Output directory for .npy files")
    p.add_argument("--horizon",     type=int, default=4, choices=[0, 1, 2, 3, 4],
                   help="Label horizon index: 0=k1 1=k2 2=k3 3=k5 4=k10 (default: 4)")
    p.add_argument("--norm",        type=str, default="ZScore",
                   choices=["ZScore", "MinMax", "DecPre"],
                   help="Normalisation scheme in file names (default: ZScore)")
    p.add_argument("--mode",        type=str, default="NoAuction",
                   choices=["NoAuction", "Auction"],
                   help="Dataset variant (default: NoAuction)")
    p.add_argument("--train_folds", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7],
                   help="Fold numbers to use for training (default: 1–7)")
    p.add_argument("--val_folds",   type=int, nargs="+", default=[8],
                   help="Fold number(s) to use for validation (default: 8)")
    p.add_argument("--test_folds",  type=int, nargs="+", default=[9],
                   help="Fold number(s) to use for testing (default: 9)")
    return p.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
