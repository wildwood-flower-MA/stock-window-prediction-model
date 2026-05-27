import time
from types import SimpleNamespace
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from src.data_preprocessing import LOBDataset
from src.GRU import LOBModel

args = SimpleNamespace(
    data_path = "lob_data.h5",
    output_dir = "checkpoints",
    epochs = 50,
    batch_size = 64,
    lr = 1e-3,
    weight_decay = 1e-4,
    patience = 10,
    cpu = False,

    seq_len = 100,
    horizon = 4,
    features = [0, 1, 2, 3]
)

class EarlyStopping:
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, checkpoint: str = "best_model.pt") -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint = checkpoint
        self.best_loss = float("inf")
        self.counter = 0
        self.stop = False

    def __call__(self, val_loss: float, model: nn.Module) -> None:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.checkpoint)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

def train_epoch(model, loader, criterion, optimizer, device):
    
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()*len(y_batch)
        correct += (logits.argmax(dim=1) == y_batch).sum().item()
        total += len(y_batch)

    return total_loss/total, correct/total

@torch.no_grad()
def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        total_loss += loss.item()*len(y_batch)
        correct += (logits.argmax(dim=1) == y_batch).sum().item()
        total += len(y_batch)

    return total_loss/total, correct/total

def train(cfg: SimpleNamespace) -> None:
    
    device = torch.device("cuda" if torch.cuda.is_available() and not cfg.cpu else "cpu")
    print(f"Device: {device}")

    features_idx = cfg.features if len(cfg.features) > 0 else None
    in_channels = len(features_idx) if features_idx else 144

    if features_idx:
        print(f"Wybrane cechy (indeksy): {features_idx}")
    else:
        print("Użyto wszystkich 144 cech wejściowych.")

    full_dataset = LOBDataset(
        file_path=cfg.data_path,
        sequence_length=cfg.seq_len,
        horizon_idx=cfg.horizon,
        features_idx=features_idx
    )

    dataset_length = len(full_dataset)
    train_size = int(0.8*dataset_length)

    train_indices = list(range(train_size))
    val_indices = list(range(train_size, dataset_length))

    train_ds = Subset(full_dataset, train_indices)
    val_ds = Subset(full_dataset, val_indices)

    print(f"Training samples: {len(train_ds):,};  Validation samples : {len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size * 2, shuffle=False, pin_memory=True)

    model = LOBModel(
        in_channels=in_channels,
        cnn_channels=64,
        gru_hidden=128,
        num_classes=3
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of model parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr*1e-2)

    checkpoint_path = Path(cfg.output_dir) / "best_model.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    early_stop = EarlyStopping(patience=cfg.patience, checkpoint=str(checkpoint_path))

    print(f"\n{'Epoch':>6}  {'Loss(Tr)':>11}  {'Acc.(Tr)':>11}  {'Loss(Val)':>11}  {'Acc.(Val)':>12}  {'lr':>10}  {'Time':>6}")
    print("-"*80)

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.perf_counter()

        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        dt = time.perf_counter() - t0

        print(f"{epoch:>6}  {tr_loss:>11.4f}  {tr_acc:>11.4f}  {vl_loss:>11.4f}  {vl_acc:>12.4f}  {lr:>10.2e}  {dt:>5.1f}s")

        early_stop(vl_loss, model)
        if early_stop.stop:
            print(f"\n Early stopping at epoch {epoch}.")
            break

    print(f"\n Best model at {checkpoint_path}")

if __name__ == "__main__":
    train(args)