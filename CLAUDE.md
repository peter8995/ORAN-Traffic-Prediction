# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

O-RAN traffic prediction using ViT + LSTM/BiLSTM architectures for 5G network slice types (eMBB, mMTC, uRLLC). Six progressive model iterations exist, each in its own directory.

## Training Commands

Models 1-3 have hardcoded parameters; run directly with `python train.py` from the model directory.

Models 4-6 use CLI arguments (run from model directory):
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

Model 6 adds `--weight_decay 1e-4`, `--patience 30` (early stopping), and `--max_pos_weight 7.0` (clamp spike BCE pos_weight) on top of model 5 args.

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
| model6 | **Anti-overfitting**: ViT backbone frozen (last 2 blocks unfrozen), Dropout(0.3) in FF/spike heads, weight decay, early stopping with best model restore, spike head gradient detach, pos_weight clamp |

## Model5/6 Spike Detection Details

- `fit_spike_params()` must be called before dataset creation to compute global `xi_max`
- Spike labels computed on **raw (unscaled)** target values using adaptive threshold: `τ = Q_0.9 × (1 + 0.05 × ξ_m/ξ_max)`
- Parameters: L=1200 (10-min Q0.9 window), m=120 (1-min volatility), ψ=0.05
- `process_file()` returns `(X, y, spike_labels)` triplets
- `forward()` returns `(pred, spike_logits)` — no Sigmoid in model, handled by BCEWithLogitsLoss
- **Model6 only**: spike head uses `context_vector.detach()` to prevent spike gradients from biasing the shared backbone/regression pathway
- **Model6 only**: `pos_weight` clamped to `--max_pos_weight` (default 7.0) to prevent extreme class imbalance from causing all-positive predictions

## Datasets

- **TRACTOR**: `Dataset/Tractor/Trial{5,7}/` — used by models 1-3
- **Colosseum**: `Dataset/colosseum-oran-coloran-dataset/tr{N}/` — used by models 4-6, organized as `tr{N}/exp{}/bs{}/{embb|mtc|urllc}/*metrics.csv`

11 input features (dl_buffer, ul_buffer, tx/rx brate, ul_sinr, dl/ul_mcs, phr, tx/rx_pkts, ul_rssi). Target: `sum_granted_prbs` (model5) or `sum_requested_prbs` (others).

Note: `mmtc` slice type maps to folder name `mtc` in the dataset.

## Dataset Utilities

- `clean_dataset.py` — Flattens nested CSV structure, removes non-metrics files (has dry-run mode)
- `group_by_slice.py` — Reorganizes metrics.csv by slice type using IMSI UE ID mapping

## Model6 Known Issues & Improvement Plans

### Plan A 已套用 (2026-03-20)
- [x] 移除 `context_vector.detach()`，恢復 spike head 梯度流
- [x] Dropout 0.3 → 0.1（ff 層和 spike head）
- [x] `lambda_detect` 預設改為 `1.0`
- [x] `max_pos_weight` 預設改為 `7.0`（原 5.0，為提高 spike recall）
- 結果：過擬合無復發，spike 開始有選擇性預測，但 Recall 仍偏低 (0.15)、F1=0.22

### Plan A2 已套用 (2026-03-22)
- [x] ViT 改為 `pretrained=False, freeze=False`（全解凍、不用 ImageNet 權重）
- **原因：** ImageNet 預訓練對 15×11 時序輸入無意義，凍結層產生雜訊特徵導致預測只能輸出均值

### Spike Recall 調優備選方案（依序嘗試）
若 `max_pos_weight=7.0` 仍不足，依序嘗試：
1. **降低推論門檻**：logit threshold 從 0.0 改為 -1.0（sigmoid ~0.27），不改訓練只改推論
2. **Focal Loss 取代 BCEWithLogitsLoss**：讓模型更關注難分類樣本，減少 easy-negative 主導
3. **Huber Loss (SmoothL1Loss) 取代 MSE**：減少迴歸預測的平滑化傾向

### 架構備選方案（若以上皆無效）

#### Plan B: 自訂小型 ViT（Custom Tiny ViT）
- 用 `nn.TransformerEncoder` 自建 2-4 層、128-256 維的小型 ViT
- 保留 ViT 架構精神（patch embedding + cls token + transformer encoder + positional encoding）
- 參數量從 86M 降至 ~1-2M，全部可訓練
- 不依賴 torchvision 預訓練模型，輸入直接為 `(batch, seq_len, 11)`
- **實作要點：**
  - `nn.Linear(inFeatures, d_model)` 取代 `conv_proj`（不需要假裝是影像）
  - `nn.TransformerEncoderLayer(d_model=128 or 256, nhead=4, dim_feedforward=512, dropout=0.1)`
  - 可學習的 positional embedding `nn.Parameter(torch.randn(1, seq_len+1, d_model))`
  - CLS token 取最後輸出送入 LSTM
  - LSTM `inputSize` 從 768 改為 d_model
- **優點：** 容量匹配資料規模，論文仍可寫「基於 Vision Transformer 架構」
- **風險：** 需調整 d_model/層數等超參數

#### Plan C: 分階段訓練 (Two-phase Training)
- Phase 1（前 N epochs）：只訓練迴歸，`lambda_detect=0`，spike head 凍結
- Phase 2（剩餘 epochs）：解凍 spike head，`lambda_detect=1.0`，降低 LR
- Dropout 維持 0.2
- **優點：** 迴歸品質有保障，spike head 在穩定 backbone 上學習
- **風險：** 多一個超參數（phase 切換點），調參複雜度增加

#### Plan D: Gradient Scaling 取代 detach
- 自訂 GradScaler layer，spike head 梯度乘以 0.1 再回傳 backbone
- Dropout 降到 0.15
- **優點：** 折衷方案，spike 能學但不主導 backbone
- **風險：** 多一個 scaling 超參數，實作較複雜，效果不確定

## model_detect_peak: Spike Predictability Baselines

Independent spike detection baselines (Autoencoder, Isolation Forest) to evaluate spike predictability without the full ViT+LSTM pipeline.

### Architecture

- **`DataProcessor.py`** — Shared data loading. Reuses model5/6 spike label formula (`τ = Q_0.9 × (1 + ψ × ξ_m/ξ_max)`). Adds 6 rolling temporal features on raw data before scaling, then creates sliding windows like model6.
- **`autoencoder_detect.py`** — PyTorch linear autoencoder. Flattened sliding window `(seq_len × 17)` → encoder (128→64→latent) → decoder → reconstruct. High reconstruction error = spike.
- **`isolation_forest_detect.py`** — Sklearn IsolationForest on flattened sliding window `(seq_len × 17)`. Includes threshold sweep for best F1.

### Input Features (17 per timestep)

11 original features + 6 rolling temporal features:
- `target_rolling_mean_m` / `target_rolling_std_m` / `target_rolling_max_m` — rolling stats over m=120 window (~30s)
- `target_rolling_mean_L` / `target_rolling_q90_L` — rolling stats over L=1200 window (~5min)
- `target_diff` — first-order difference of target

Rolling features bridge the gap between short sliding window (seq_len=15, ~3.75s) and the long-range spike definition (L=1200, ~5min).

### Training Commands

```bash
cd model_detect_peak

# Autoencoder
python autoencoder_detect.py --train_dirs tr0-4 tr10 --test_dirs tr5-6 \
  --slice_type embb --epochs 100 --sequence_length 15

# Isolation Forest
python isolation_forest_detect.py --train_dirs tr0-4 tr10 --test_dirs tr5-6 \
  --slice_type embb --sequence_length 15
```

### Key Arguments

| Arg | AE Default | IF Default | Description |
|-----|-----------|-----------|-------------|
| `--sequence_length` | 15 | 15 | Sliding window length |
| `--threshold_percentile` | 88 | — | Percentile of train recon error for spike threshold |
| `--contamination` | — | 0.11 | Expected anomaly proportion |
| `--latent_dim` | 16 | — | Autoencoder bottleneck size |
| `--n_estimators` | — | 200 | Number of IF trees |

### Initial Results (without rolling features, without sliding window)

| Metric | Autoencoder | Isolation Forest |
|--------|------------|-----------------|
| F1 | 0.176 | 0.188 |
| Precision | 0.155 | 0.183 |
| Recall | 0.204 | 0.192 |
| Accuracy | 0.769 | 0.798 |

These serve as baselines to evaluate spike predictability from raw features alone.

## Dependencies

PyTorch, torchvision, pandas, numpy, scikit-learn (MinMaxScaler, IsolationForest), matplotlib. No requirements.txt exists.