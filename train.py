"""
Training Pipeline — LOB Price Movement Prediction
===================================================
Expects pre-processed numpy arrays in the following layout:

    data/
        X_train.npy   shape (N_train, 100, 40)   float32
        y_train.npy   shape (N_train,)             int64   values in {0,1,2}
        X_val.npy     shape (N_val,   100, 40)   float32
        y_val.npy     shape (N_val,  )             int64

Usage:
    conda activate ML-pytorch
    python train.py --data_dir data --epochs 50 --batch_size 64 --lr 1e-3
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model import LOBModel

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LOBDataset(Dataset):
    """
    Thin wrapper around pre-processed numpy arrays.

    Args:
        X : np.ndarray  (N, 100, 40)  float32  — LOB windows
        y : np.ndarray  (N,)          int64    — class labels {0, 1, 2}
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        assert len(X) == len(y), "X and y must have the same number of samples"
        # Keep as numpy arrays (possibly memory-mapped) — tensors are created
        # per-sample in __getitem__ to avoid allocating the full dataset in RAM.
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(np.array(self.X[idx], dtype=np.float32))
        y = torch.tensor(int(self.y[idx]), dtype=torch.int64)
        return x, y


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Stops training when validation loss has not improved for `patience` epochs
    and saves the best checkpoint.

    Args:
        patience   : epochs to wait before stopping
        min_delta  : minimum improvement to count as an improvement
        checkpoint : path to save the best model weights
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4,
                 checkpoint: str = "best_model.pt") -> None:
        self.patience   = patience
        self.min_delta  = min_delta
        self.checkpoint = checkpoint
        self.best_loss  = float("inf")
        self.counter    = 0
        self.stop       = False

    def __call__(self, val_loss: float, model: nn.Module) -> None:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
            torch.save(model.state_dict(), self.checkpoint)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


# ---------------------------------------------------------------------------
# Train / Validate helpers
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """
    Single training epoch.

    Returns:
        avg_loss : float
        accuracy : float  (0–1)
    """
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(X_batch)                     # (B, 3)
        loss   = criterion(logits, y_batch)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        correct    += (logits.argmax(dim=1) == y_batch).sum().item()
        total      += len(y_batch)

    return total_loss / total, correct / total


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Validation pass — no gradient computation.

    Returns:
        avg_loss : float
        accuracy : float  (0–1)
    """
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        logits = model(X_batch)
        loss   = criterion(logits, y_batch)

        total_loss += loss.item() * len(y_batch)
        correct    += (logits.argmax(dim=1) == y_batch).sum().item()
        total      += len(y_batch)

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    # ── Device ──────────────────────────────────────────────────────────────
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"Device: {device}")

    # ── Data ────────────────────────────────────────────────────────────────
    data_dir = Path(args.data_dir)

    X_train = np.load(data_dir / "X_train.npy", mmap_mode="r")
    y_train = np.load(data_dir / "y_train.npy", mmap_mode="r")
    X_val   = np.load(data_dir / "X_val.npy",   mmap_mode="r")
    y_val   = np.load(data_dir / "y_val.npy",   mmap_mode="r")

    print(f"Train samples : {len(X_train):,}  |  Val samples : {len(X_val):,}")
    print(f"Class distribution (train): {np.bincount(y_train.astype(int))}")

    train_ds = LOBDataset(X_train, y_train)
    val_ds   = LOBDataset(X_val,   y_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = LOBModel(
        in_channels=X_train.shape[2],
        cnn_channels=64,
        gru_hidden=128,
        num_classes=3,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters    : {total_params:,}")

    # ── Loss — class-balanced weights ────────────────────────────────────────
    counts  = np.bincount(y_train.astype(int), minlength=3).astype(np.float32)
    weights = torch.tensor(1.0 / (counts + 1e-6), dtype=torch.float32).to(device)
    weights = weights / weights.sum() * 3          # normalise to sum=3
    criterion = nn.CrossEntropyLoss(weight=weights)

    # ── Optimiser & Scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2
    )

    # ── Early stopping & checkpoint ──────────────────────────────────────────
    checkpoint_path = Path(args.output_dir) / "best_model.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    early_stop = EarlyStopping(
        patience=args.patience, checkpoint=str(checkpoint_path)
    )

    # ── Training loop ────────────────────────────────────────────────────────
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\n{'Epoch':>6}  {'Train Loss':>11}  {'Train Acc':>10}  "
          f"{'Val Loss':>9}  {'Val Acc':>8}  {'LR':>10}  {'Time':>6}")
    print("-" * 75)

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()

        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        lr  = optimizer.param_groups[0]["lr"]
        dt  = time.perf_counter() - t0

        print(f"{epoch:>6}  {tr_loss:>11.4f}  {tr_acc:>9.4f}  "
              f"{vl_loss:>9.4f}  {vl_acc:>8.4f}  {lr:>10.2e}  {dt:>5.1f}s")

        early_stop(vl_loss, model)
        if early_stop.stop:
            print(f"\nEarly stopping triggered at epoch {epoch}.")
            break

    # ── Save history ─────────────────────────────────────────────────────────
    history_path = Path(args.output_dir) / "history.npz"
    np.savez(history_path, **{k: np.array(v) for k, v in history.items()})
    print(f"\nBest model   → {checkpoint_path}")
    print(f"Training log → {history_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LOB price movement model")

    p.add_argument("--data_dir",     type=str,   default="data",
                   help="Directory with X_train/y_train/X_val/y_val .npy files")
    p.add_argument("--output_dir",   type=str,   default="checkpoints",
                   help="Directory to write best model and history")
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience",     type=int,   default=10,
                   help="Early stopping patience (epochs)")
    p.add_argument("--num_workers",  type=int,   default=0,
                   help="DataLoader worker processes (0 = main process)")
    p.add_argument("--cpu",          action="store_true",
                   help="Force CPU even if CUDA is available")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
