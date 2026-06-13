import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import torch

N_LOB_FEATURES = 40
N_TOTAL_ROWS = 149
LABEL_ROW_OFFSET = 144
WINDOW_SIZE = 100

LABEL_MAP = {1: 2, 2: 1, 3: 0}

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

def load_txt(path: str) -> np.ndarray:
    data = pd.read_csv(path, sep=r"\s+", header=None).to_numpy()
    return data

def load_files(file_paths: list[str]) -> np.ndarray:
    parts = [load_txt(p) for p in sorted(file_paths)]
    return np.concatenate(parts, axis=1)

def resolve_paths(
    data_dir: str,
    mode: str,
    norm: str,
    split: str,
    folds: list[int],
) -> list[str]:
    key = (mode, norm)
    numbered_dir, train_sub, test_sub, file_norm = PATH_MAP[key]

    subdir = train_sub if split == "train" else test_sub
    prefix = "Train" if split == "train" else "Test"
    base = Path(data_dir) / "BenchmarkDatasets" / mode / numbered_dir / subdir

    paths = []
    for fold in folds:
        fname = f"{prefix}_Dst_{mode}_{file_norm}_CF_{fold}.txt"
        full = base / fname
        paths.append(str(full))
    return paths

class LOBDataset(torch.utils.data.Dataset):
    def __init__(self, h5_path: str, split: str, sequence_length: int, horizon_idx: int = 0):
        self.h5_path = h5_path
        self.split = split
        self.sequence_length = sequence_length
        self.horizon_idx = horizon_idx
        self.dataset = None
        
        with h5py.File(self.h5_path, 'r') as f:
            self.length = len(f[self.split]) - self.sequence_length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.dataset is None:
            self.file = h5py.File(self.h5_path, 'r')
            self.dataset = self.file[self.split]
            
        window = self.dataset[idx : idx + self.sequence_length]

        x = window[:, :N_LOB_FEATURES]
        y_raw = window[-1, LABEL_ROW_OFFSET + self.horizon_idx]
        y = LABEL_MAP[int(y_raw)]
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

def prepare(args: argparse.Namespace) -> None:
    data_dir = args.data_dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_h5_path = output_dir / "lob_dataset.h5"

    train_files = resolve_paths(data_dir, args.mode, args.norm, "train", args.train_folds)
    val_files   = resolve_paths(data_dir, args.mode, args.norm, "train", args.val_folds)
    test_files  = resolve_paths(data_dir, args.mode, args.norm, "test",  args.test_folds)

    with h5py.File(output_h5_path, 'w') as h5f:
        for split_name, files in zip(["train", "val", "test"], [train_files, val_files, test_files]):
            data = load_files(files)
            data = data.T 
            
            h5f.create_dataset(
                split_name,
                data=data,
                chunks=True,
                compression='gzip',
                dtype='float32'
            )

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="raw_data")
    p.add_argument("--output_dir", type=str, default="data")
    p.add_argument("--horizon", type=int, default=4, choices=[0, 1, 2, 3, 4])
    p.add_argument("--norm", type=str, default="ZScore", choices=["ZScore", "MinMax", "DecPre"])
    p.add_argument("--mode", type=str, default="NoAuction", choices=["NoAuction", "Auction"])
    p.add_argument("--train_folds", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7])
    p.add_argument("--val_folds", type=int, nargs="+", default=[8])
    p.add_argument("--test_folds", type=int, nargs="+", default=[9])
    return p.parse_args()

if __name__ == "__main__":
    prepare(parse_args())