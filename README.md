# stock-window-prediction-model

AGH University — Advanced Machine Learning in HEP project

**Task:** 3-class mid-price movement prediction (DOWN / FLAT / UP) on Limit Order Book data using a DeepLOB-inspired deep learning model in PyTorch.

**Dataset:** [FI-2010 Benchmark LOB Dataset](http://urn.fi/urn:nbn:fi:csc-kata20170601153214969115)  
Ntakaris et al., 2017 — 5 Finnish stocks, 10-day horizon, Z-Score normalised, NoAuction variant.

---

## Project Structure

```
.
├── main.py           ← single entry point — runs full pipeline
├── model.py          ← LOBModel architecture
├── prepare_data.py   ← FI-2010 parser → .npy arrays
├── train.py          ← training loop + early stopping
├── evaluate.py       ← scoring + confusion matrix plots
├── environment.yml   ← conda environment definition
│
├── raw_data/                          ← original FI-2010 archive (unzipped)
│   └── BenchmarkDatasets/
│       ├── NoAuction/
│       │   ├── 1.NoAuction_Zscore/
│       │   │   ├── NoAuction_Zscore_Training/
│       │   │   │   ├── Train_Dst_NoAuction_ZScore_CF_1.txt
│       │   │   │   └── ...  CF_9.txt
│       │   │   └── NoAuction_Zscore_Testing/
│       │   │       ├── Test_Dst_NoAuction_ZScore_CF_1.txt
│       │   │       └── ...  CF_9.txt
│       │   ├── 2.NoAuction_MinMax/    (same layout)
│       │   └── 3.NoAuction_DecPre/   (same layout)
│       └── Auction/                  (same layout)
│
├── data/             ← generated: X_train/val/test.npy, y_*.npy
├── checkpoints/      ← generated: best_model.pt, history.npz
└── results/          ← generated: metrics.txt, confusion matrix PNGs
```

---

## Architecture — `model.py`

```
Input (batch, 100, 40)
  │
  ▼
CNNFeatureExtractor          Conv1D(40→64) + BN + GELU  ×2
  │                          + residual projection (1×1 conv)
  ▼
GRUEncoder                   GRU(64→128, batch_first) + LayerNorm
  │
  ▼
TemporalAttention            learnable softmax weights over time
  │                          → weighted sum → context vector (128,)
  ▼
ClassificationHead           Linear(128→64) → ReLU → Dropout(0.2)
  │                          → Linear(64→3)
  ▼
Output (batch, 3)  — raw logits
```

| Module | Purpose |
|---|---|
| `CNNFeatureExtractor` | local microstructure: spread, imbalance, liquidity patterns |
| `GRUEncoder` | temporal market dynamics over the 100-event window |
| `TemporalAttention` | selects the most informative time steps automatically |
| `ClassificationHead` | maps context vector to 3-class logits |

Total parameters: **~106 K** (all trainable)

---

## Data Preparation — `prepare_data.py`

Parses raw FI-2010 `.txt` files. Verified file shape: `(149, ~39 000)` — 149 rows, ~39 k events per file.

| Rows | Content |
|---|---|
| `0–39` | 40 raw LOB features: bid/ask price + volume, 10 levels each |
| `40–142` | engineered time-sensitive / time-insensitive features (unused) |
| `143` | zeros — separator row |
| `144–148` | labels for horizons k = 1, 2, 3, 5, 10 |

Label remapping: `1 (UP) → 2`, `2 (FLAT) → 1`, `3 (DOWN) → 0`  
Window: 100-event sliding window, label = last event in window.  
File loading uses `pandas.read_csv` (significantly faster than `numpy.loadtxt`).

---

## Training — `train.py`

| Component | Detail |
|---|---|
| Loss | `CrossEntropyLoss` with class-balanced weights |
| Optimiser | `AdamW` (lr=1e-3, weight_decay=1e-4) |
| Scheduler | `CosineAnnealingLR` |
| Regularisation | gradient clipping (`max_norm=1.0`), Dropout(0.2) |
| Early stopping | patience=10, saves `best_model.pt` on val loss improvement |
| Output | `checkpoints/best_model.pt`, `checkpoints/history.npz` |

---

## Evaluation — `evaluate.py`

Metrics computed on the test set:

- Accuracy, F1 (macro, weighted, per-class)
- Matthews Correlation Coefficient (MCC)
- Cohen's Kappa
- Confusion matrix (row-normalised + raw counts, saved as PNG)

Output saved to `results/`: `metrics.txt`, `y_pred.npy`, `y_proba.npy`, confusion matrix plots.

---

## Quick Start

### 1. Create environment

```bash
conda env create -f environment.yml
conda activate ML-pytorch
```

### 2. Download dataset

Download the FI-2010 archive from the link above and unzip it into `raw_data/`.  
The expected structure after unzipping:

```
raw_data/
  BenchmarkDatasets/
    NoAuction/
      1.NoAuction_Zscore/
        NoAuction_Zscore_Training/   Train_Dst_NoAuction_ZScore_CF_1..9.txt
        NoAuction_Zscore_Testing/    Test_Dst_NoAuction_ZScore_CF_1..9.txt
      2.NoAuction_MinMax/
      3.NoAuction_DecPre/
    Auction/
```

### 3. Run full pipeline

```bash
python main.py
```

This runs all three stages automatically:
1. **prepare_data** — parse `.txt` → `.npy` arrays in `data/`
2. **train** — train model, save best checkpoint to `checkpoints/`
3. **evaluate** — compute metrics, save plots to `results/`

### Run stages individually

```bash
# Data preparation only
python prepare_data.py --data_dir raw_data --output_dir data --horizon 4

# Training only
python train.py --data_dir data --epochs 50 --batch_size 64 --lr 1e-3

# Evaluation only
python evaluate.py --checkpoint checkpoints/best_model.pt --data_dir data
```

---

## Configuration

All pipeline settings are in the `CFG` block at the top of `main.py`:

```python
CFG = types.SimpleNamespace(
    raw_data_dir = "raw_data",
    horizon      = 4,       # 0=k1  1=k2  2=k3  3=k5  4=k10
    epochs       = 50,
    batch_size   = 64,
    lr           = 1e-3,
    patience     = 10,
    ...
)
```