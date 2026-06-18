import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import torch

#N_LOB_FEATURES = 40
#SELECTED_FEATURES = [i for i in range(40)]
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

def load_txt(zf: zipfile.ZipFile, internal_path: str) -> np.ndarray:
    
    with zf.open(internal_path) as f:
        data = pd.read_csv(f, sep=r"\s+", header=None).to_numpy()
    return data

def load_files(zf: zipfile.ZipFile, file_paths: list[str]) -> np.ndarray:
    parts = [load_txt(zf, p) for p in sorted(file_paths)]
    return np.concatenate(parts, axis=1)

def resolve_paths(
    mode: str,
    norm: str,
    split: str,
    folds: list[int],
) -> list[str]:

    key = (mode, norm)
    numbered_dir, train_sub, test_sub, file_norm = PATH_MAP[key]

    subdir = train_sub if split == "train" else test_sub
    prefix = "Train" if split == "train" else "Test"
    base = f"BenchmarkDatasets/{mode}/{numbered_dir}/{subdir}"

    paths = []
    for fold in folds:
        fname = f"{prefix}_Dst_{mode}_{file_norm}_CF_{fold}.txt"
        paths.append(f"{base}/{fname}")
    return paths


class LOBDataset(torch.utils.data.Dataset):
    def __init__(self, h5_path: str, split: str, sequence_length: int, horizon_idx: int = 0, selected_features: list = None):
        self.h5_path = h5_path
        self.split = split
        self.sequence_length = sequence_length
        
        self.selected_features = selected_features if selected_features is not None else list(range(40))
        
        with h5py.File(self.h5_path, 'r') as f:
            raw_data = f[self.split][:]

        self.X = raw_data[:, self.selected_features].astype(np.float32)
        raw_labels = raw_data[:, LABEL_ROW_OFFSET + horizon_idx]
        map_func = np.vectorize(LABEL_MAP.__getitem__)
        self.Y = map_func(raw_labels).astype(np.int64)
        self.length = len(self.X) - self.sequence_length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x_window = self.X[idx : idx + self.sequence_length]
        y_label = self.Y[idx + self.sequence_length - 1]
        return torch.from_numpy(x_window), torch.as_tensor(y_label)
        
def prepare(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_h5_path = output_dir / "lob_dataset.h5"

    zip_path = data_dir / "BenchmarkDatasets.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"Nie znaleziono {zip_path}")

    train_files = resolve_paths(args.mode, args.norm, "train", args.train_folds)
    val_files   = resolve_paths(args.mode, args.norm, "train", args.val_folds)
    test_files  = resolve_paths(args.mode, args.norm, "test",  args.test_folds)

    with h5py.File(output_h5_path, 'w') as h5f:
        with zipfile.ZipFile(zip_path, 'r') as zf:

            available_files = zf.namelist()
            if train_files[0] not in available_files:
                raise FileNotFoundError(
                    f"Błąd, oczekiwano '{train_files[0]}', "
                    f"Czy istnieje BenchmarkDatasets?"
                )

            for split_name, files in zip(["train", "val", "test"], [train_files, val_files, test_files]):
                print(f" Now: {split_name}")
                data = load_files(zf, files)
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