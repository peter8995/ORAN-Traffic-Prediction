# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

O-RAN traffic prediction using ViT + LSTM/BiLSTM architectures for 5G network slice types (eMBB, mMTC, uRLLC). Five progressive model iterations exist, each in its own directory.

## Training Commands

Models 1-3 have hardcoded parameters; run directly with `python train.py` from the model directory.

Models 4-5 use CLI arguments (run from model directory):
```bash
python train.py --train_dirs tr0-4 tr10 --test_dirs tr5-6 \
  --slice_type embb \
  --epochs 500 \
  --batch_size 1024 \
  --learning_rate 1e-6 \
  --sequence_length 15 \
  --val_split 0.2
```

Model 5 adds `--lambda_detect 0.5` for spike detection BCE loss weight.

Directory range notation: `tr0-4` expands to `tr0, tr1, tr2, tr3, tr4`.

## Architecture

Each model directory contains three files with the same interaction pattern:

- **`classicModel.py`** — Model definitions: `vitModel` (feature extraction) → `lstmModel`/`biLSTMModel` (sequence) → `TrafficModel` (assembled pipeline with attention + prediction heads)
- **`DataProcessor.py`** — CSV loading, MinMaxScaler transforms, sliding window creation. Returns `TrafficDataset` for DataLoader.
- **`train.py`** — Training loop, loss computation, evaluation, matplotlib plotting to `results/`.

Forward pass: Input `(batch, seq_len, 11)` → ViT `(batch, seq_len+1, 768)` → LSTM `(batch, seq_len+1, 128)` → MultiheadAttention → last token → FF → prediction.

## Model Differences

| Model | Key Change |
|-------|-----------|
| model1 | Unified multi-task: all 3 slices + global traffic jointly, MultiheadAttention fusion |
| model2 | Independent per-slice models, no cross-slice attention |
| model3 | Slice-specific losses: quantile loss (q=0.7) for mMTC, MSE for others |
| model4 | Multi-directory Colosseum dataset support via argparse, `parse_directory_args()` |
| model5 | **SpikeAwareLSTM**: dual-head (regression + spike detection), adaptive threshold labels, BCEWithLogitsLoss, gradient clipping |

## Model5 Spike Detection Details

- `fit_spike_params()` must be called before dataset creation to compute global `xi_max`
- Spike labels computed on **raw (unscaled)** target values using adaptive threshold: `τ = Q_0.9 × (1 + 0.05 × ξ_m/ξ_max)`
- Parameters: L=1200 (10-min Q0.9 window), m=120 (1-min volatility), ψ=0.05
- `process_file()` returns `(X, y, spike_labels)` triplets
- `forward()` returns `(pred, spike_logits)` — no Sigmoid in model, handled by BCEWithLogitsLoss

## Datasets

- **TRACTOR**: `Dataset/Tractor/Trial{5,7}/` — used by models 1-3
- **Colosseum**: `Dataset/colosseum-oran-coloran-dataset/tr{N}/` — used by models 4-5, organized as `tr{N}/exp{}/bs{}/{embb|mtc|urllc}/*metrics.csv`

11 input features (dl_buffer, ul_buffer, tx/rx brate, ul_sinr, dl/ul_mcs, phr, tx/rx_pkts, ul_rssi). Target: `sum_granted_prbs` (model5) or `sum_requested_prbs` (others).

Note: `mmtc` slice type maps to folder name `mtc` in the dataset.

## Dataset Utilities

- `clean_dataset.py` — Flattens nested CSV structure, removes non-metrics files (has dry-run mode)
- `group_by_slice.py` — Reorganizes metrics.csv by slice type using IMSI UE ID mapping

## Dependencies

PyTorch, torchvision, pandas, numpy, scikit-learn (MinMaxScaler), matplotlib. No requirements.txt exists.