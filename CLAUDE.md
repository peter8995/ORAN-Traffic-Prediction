# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

O-RAN traffic prediction using ViT + LSTM/BiLSTM architectures for 5G network slice types (eMBB, mMTC, uRLLC). Eight model directories exist (model1–model7, model7_1).

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

Model 7 adds Tiny ViT hyperparameters on top of model 6 args: `--d_model 128`, `--n_heads 4`, `--n_layers 3`, `--dim_feedforward 512`, `--vit_dropout 0.1`.

Directory range notation: `tr0-4` expands to `tr0, tr1, tr2, tr3, tr4`.

## Architecture

Each model directory contains three files with the same interaction pattern:

- **`classicModel.py`** — Model definitions: `vitModel`/`TinyViT` (feature extraction) → `lstmModel`/`biLSTMModel` (sequence) → `TrafficModel` (assembled pipeline with attention + prediction heads)
- **`DataProcessor.py`** — CSV loading, MinMaxScaler transforms, sliding window creation. Returns `TrafficDataset` for DataLoader.
- **`train.py`** — Training loop, loss computation, evaluation, matplotlib plotting to `results/`.

Forward pass (model1-6): Input `(batch, seq_len, 11)` → ViT `(batch, seq_len+1, 768)` → LSTM `(batch, seq_len+1, 128)` → MultiheadAttention → last token → FF → prediction.

Forward pass (model7): Input `(batch, seq_len, 11)` → TinyViT `(batch, seq_len+1, d_model)` → LSTM `(batch, seq_len+1, 128)` → MultiheadAttention → last token → FF → prediction.

## Model Differences

| Model | Key Change |
|-------|-----------|
| model1 | Unified multi-task: all 3 slices + global traffic jointly, MultiheadAttention fusion |
| model2 | Independent per-slice models, no cross-slice attention |
| model3 | Slice-specific losses: quantile loss (q=0.7) for mMTC, MSE for others |
| model4 | Multi-directory Colosseum dataset support via argparse, `parse_directory_args()` |
| model5 | **SpikeAwareLSTM**: dual-head (regression + spike detection), adaptive threshold labels, BCEWithLogitsLoss, gradient clipping |
| model6 | **Anti-overfitting**: ViT backbone frozen (last 2 blocks unfrozen), Dropout(0.3) in FF/spike heads, weight decay, early stopping with best model restore, spike head gradient detach, pos_weight clamp |
| model7 | **Tiny ViT (Plan B)**: Custom lightweight ViT (`nn.TransformerEncoder`, ~0.8-2.4M params) replaces torchvision vit_b_16 (86M params). `nn.Linear` input projection, learnable CLS token + positional embedding, configurable d_model/n_heads/n_layers. No torchvision dependency. |
| model7_1 | **Regression-only ablation**: Identical to model7 but with spike head removed. Used to verify whether spike head actually improves regression, especially at spike timesteps. performance_matrix.txt outputs MAE/RMSE/R² split by spike vs normal regions + MAE ratio. |

## Model7 Tiny ViT Details

- `TinyViT` class in `classicModel.py` replaces `vitModel` (torchvision vit_b_16)
- Input projection: `nn.Linear(inFeatures, d_model)` — no Conv2d, no image pretense
- Learnable CLS token `nn.Parameter(1, 1, d_model)` + positional embedding `nn.Parameter(1, seq_len+1, d_model)`
- `nn.TransformerEncoder` with configurable layers, GELU activation
- `LayerNorm` on output, Xavier/truncated normal initialization
- LSTM `inputSize` = `d_model` (not 768)
- Default config: `d_model=128, n_heads=4, n_layers=3, dim_feedforward=512` → embb ~805K params, mmtc/urllc ~1.1M params
- Alternative: `d_model=256, n_layers=4` → ~2.4M params

## Model5/6/7 Spike Detection Details

- `fit_spike_params()` must be called before dataset creation to compute global `xi_max`
- Spike labels computed on **raw (unscaled)** target values using adaptive threshold: `τ = Q_0.9 × (1 + 0.05 × ξ_m/ξ_max)`
- Parameters: L=1200 (10-min Q0.9 window), m=120 (1-min volatility), ψ=0.05
- `process_file()` returns `(X, y, spike_labels)` triplets
- `forward()` returns `(pred, spike_logits)` — no Sigmoid in model, handled by BCEWithLogitsLoss
- **Model6 only**: spike head uses `context_vector.detach()` to prevent spike gradients from biasing the shared backbone/regression pathway
- **Model6/7**: `pos_weight` clamped to `--max_pos_weight` (default 7.0) to prevent extreme class imbalance from causing all-positive predictions

## Datasets

- **TRACTOR**: `Dataset/Tractor/Trial{5,7}/` — used by models 1-3
- **Colosseum**: `Dataset/colosseum-oran-coloran-dataset/tr{N}/` — used by models 4-7, organized as `tr{N}/exp{}/bs{}/{embb|mtc|urllc}/*metrics.csv`

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
- 結果（embb）：過擬合無復發，spike 開始有選擇性預測，但 Recall 仍偏低 (0.15)、F1=0.22

### Plan A2 已套用 (2026-03-22)
- [x] ViT 改為 `pretrained=False, freeze=False`（全解凍、不用 ImageNet 權重）
- **原因：** ImageNet 預訓練對 15×11 時序輸入無意義，凍結層產生雜訊特徵導致預測只能輸出均值
- 結果（embb）：Spike F1=0.23, R=0.37（Recall 翻倍），但 regression 嚴重低估
- 結果（mmtc）：Spike 崩潰全 positive（F1=0.25, R=0.9963, FP Rate=0.8776），regression 方塊波

### Plan B 已實作為 Model7 (2026-03-24)
- [x] 自訂 `TinyViT`（`nn.TransformerEncoder`）取代 `vit_b_16`
- 詳見 Model7 Tiny ViT Details 段落

## Model7 實驗結果

### Round 1 (2026-03-24): vit_dropout=0.1, weight_decay=0, patience=20
結果存放：`model7/results/`

**embb:**

| Metric | Model6 A2 | Model7 R1 |
|--------|-----------|-----------|
| F1 | 0.2316 | 0.2350 |
| Precision | 0.1689 | 0.1373 |
| Recall | 0.3683 | 0.8168 |
| Accuracy | 0.7193 | 0.3809 |
| FP Rate | 0.2351 | 0.6765 |
| Spike Rate | 11.49% | 11.64% |

- Spike head 偏向全 positive（FP Rate 67.6%）
- Regression 大幅改善：紅藍線貼合度高，不再嚴重低估
- Loss curve overfitting：val loss ~epoch 25 後持續上升，early stopping ~epoch 80

**mmtc:**

| Metric | Model6 | Model7 R1 |
|--------|--------|-----------|
| F1 | 0.2520 | **0.4674** |
| Precision | 0.1443 | **0.3427** |
| Recall | 0.9963 | 0.7346 |
| Accuracy | 0.2354 | **0.7877** |
| FP Rate | 0.8776 | **0.2046** |
| Spike Rate | 12.93% | 12.68% |

- Spike head 不再崩潰，有選擇性預測（F1 +85%, FP Rate -76%）
- Regression 能追蹤波形（model6 只有方塊波），但有單一異常尖刺（timestep ~7500）
- Loss curve overfitting：val loss ~epoch 30 後上升，early stopping ~epoch 78

### Round 2 (2026-03-24): vit_dropout=0.2, weight_decay=1e-5, patience=50
結果存放：`model7/results/2/`

**embb:**

| Metric | R1 (wd=0, dp=0.1) | R2 (wd=1e-5, dp=0.2) |
|--------|-------------------|----------------------|
| F1 | 0.2350 | 0.2327 |
| Precision | 0.1373 | 0.1386 |
| Recall | 0.8168 | 0.7257 |
| Accuracy | 0.3809 | 0.4429 |
| FP Rate | 0.6765 | 0.5943 |
| Spike Rate | 11.64% | 11.64% |

- FP Rate 下降 8%（0.68→0.59），Accuracy 上升（0.38→0.44），但 F1 持平
- Regression 比 R1 更穩，segment 間跳轉更準確
- Loss curve 大幅改善：train/val gap ~0.005，val loss ~epoch 50 才開始緩慢上升，early stopping ~epoch 100

**mmtc:**

| Metric | R1 (wd=0, dp=0.1) | R2 (wd=1e-5, dp=0.2) |
|--------|-------------------|----------------------|
| F1 | 0.4674 | 0.4742 |
| Precision | 0.3427 | 0.3504 |
| Recall | 0.7346 | 0.7335 |
| Accuracy | 0.7877 | 0.7938 |
| FP Rate | 0.2046 | 0.1974 |
| Spike Rate | 12.68% | 12.68% |

- 數字上 R2 微幅改善（F1 +0.007, FP Rate -0.007），幾乎持平
- 但圖上 spike detection 看起來仍大面積紅色
- Regression 有追蹤波形但低估高值段，有幾根異常尖刺
- Loss curve：val loss ~epoch 35 後上升，weight_decay=1e-5 對 mmtc 不夠

### Round 3 (2026-03-24): vit_dropout=0.2, weight_decay=0.0, patience=50
結果存放：`model7/results/embb_20260324_224612/`、`model7/results/mmtc_20260324_184109/`
「dropout 單獨效果測試」（R2 去掉 weight_decay）

**embb:**

| Metric | R1 | R2 | R3 |
|--------|----|----|-----|
| R² | — | — | **0.9739** |
| F1 | 0.2350 | 0.2327 | 0.2328 |
| FP Rate | 0.6765 | **0.5943** | 0.6062 |
| Early stop | ~ep80 | ~ep100 | ~ep80 |

**mmtc:**

| Metric | R1 | R2 | R3 |
|--------|----|----|-----|
| R² | — | — | **0.3894** ✗ |
| F1 | 0.4674 | **0.4742** | 0.4615 |
| FP Rate | 0.2046 | **0.1974** | 0.2211 |

- embb：dropout=0.2 有效（FP 從 0.68→0.61），但 weight_decay 能再改善（R2=0.59）
- mmtc：R3 是三輪中最差；R²=0.39 是最大待解問題（幾乎等同猜均值）
- 結論：embb/mmtc 最佳配置均為 R2（vit_dropout=0.2, weight_decay=1e-5, patience=50）

### 觀察總結
- Tiny ViT regression 品質全面優於 86M vit_b_16（不再低估、不再方塊波）
- embb R²=0.97（健康）；mmtc R²=0.39（根本問題，待解）
- embb R2 正則化有效抑制 overfitting（early stopping 從 ep80 延到 ep100）
- mmtc spike F1 在三輪中 R2 最佳（0.474），但整體仍不理想
- 兩個 slice 的最佳正則化策略相同（R2），但 mmtc 需要更多改進
- Spike head 真正目的是輔助 regression（讓 backbone 對突峰敏感），F1 高低不是主要指標
- Spike label 仍使用 rolling Q90 公式（~11-13% spike rate），label 品質是根本問題

### 待嘗試方案
- ~~**model7_1 ablation**：先跑純 regression baseline，比較 spike 時段 MAE ratio~~ ✅ 已完成，結論見下方
- Plan C: 分階段訓練（Phase 1 只訓練 regression，Phase 2 解凍 spike head）
- Plan D: Gradient Scaling（spike head 梯度乘以 0.1 再回傳 backbone）
- Focal Loss 取代 BCEWithLogitsLoss
- Huber Loss (SmoothL1Loss) 取代 MSE
- 降低推論門檻：logit threshold 從 0.0 改為 -1.0

## model7_1: Regression-Only Ablation

與 model7 架構完全相同，移除 spike_head，用於驗證 spike head 是否改善 regression。
所有 slice 均使用 TinyViT + BiLSTM。eMBB 額外跑了一版 LSTM 作比較。

### 關鍵差異
- `classicModel.py`：無 `spike_head`，`forward()` 只回傳 `pred`，所有 slice 使用 `biLSTMModel`
- `train.py`：loss 純 MSE，無 `lambda_detect`/`max_pos_weight` 參數
- `performance_matrix.txt` 新增：
  - Spike timesteps MAE/RMSE/R²
  - Normal timesteps MAE/RMSE/R²
  - MAE ratio（spike/normal）— 越高代表 spike 時段誤差越大
- Prediction plot：黃色背景標出 ground truth spike 區域

### Training Command
```bash
cd model7_1
python train.py --train_dirs tr0-26 --test_dirs tr27 --slice_type embb \
  --epochs 500 --patience 25 --vit_dropout 0.2 --weight_decay 0.0
```

### Ablation 結果 (2026-03-26)

結果存放：`model7_1/results/`

#### 結論一：Spike head 對 regression 無幫助

Model7 R3（有 spike head）vs Model7_1（純 regression），相同超參數（vit_dropout=0.2, wd=0）：

| Metric | eMBB M7 | eMBB M7_1 | mMTC M7 | mMTC M7_1 |
|--------|---------|-----------|---------|-----------|
| MAE | 0.0046 | **0.0043** | 0.0024 | **0.0023** |
| RMSE | 0.0060 | **0.0054** | 0.0036 | 0.0036 |
| R² | 0.9739 | **0.9780** | 0.3894 | **0.3958** |

Model7_1 在所有 slice 上 regression 微幅優於 model7。spike head 未能讓 backbone 對突變更敏感，反而分散學習能力。

#### 結論二：LSTM vs BiLSTM（eMBB）

| Metric | LSTM | BiLSTM |
|--------|------|--------|
| R²(all) | 0.9789 | 0.9780 |
| Spike MAE | 0.0039 | **0.0034** |
| Spike R² | 0.9678 | **0.9743** |
| MAE ratio | 0.8874 | **0.7527** |

整體持平，BiLSTM 在 spike 時段稍優（雙向特性對突變時序捕捉較好）。

#### 結論三：mMTC / uRLLC regression 根本性困難

| | eMBB | mMTC | uRLLC |
|--|------|------|-------|
| R²(all) | **0.9780** | 0.3958 | 0.4140 |
| Spike R² | **0.9743** | -0.6150 | -0.3230 |
| Normal R² | 0.9798 | 0.4966 | 0.5270 |
| MAE ratio | **0.75** | 3.38 | 3.68 |
| Spike Rate | 11.64% | 12.68% | 13.05% |

- mMTC/uRLLC 連 normal 時段 R² 只有 ~0.5，spike 時段 R² 為負（比猜均值差）
- uRLLC 首次有結果，表現與 mMTC 同等困難

#### 結論四：Prediction Plot 與 Loss Curve 視覺分析

**eMBB（LSTM & BiLSTM）：**
- Prediction：紅藍線貼合，但每個 segment 內紅藍線都近乎水平 — 模型實質上是在預測**每個 segment（experiment）的均值水平**，而非 segment 內的 timestep-level 時序波動
- **eMBB R²=0.978 有灌水成分**：不同 experiment 的流量等級差異大（如某段 ~0.02、某段 ~0.05），只要正確預測等級就能解釋大部分變異。segment 內變異極小，模型未被真正考驗
- Loss curve 差異：LSTM 收斂慢（~epoch 80 才接近 val loss），BiLSTM 收斂快（~epoch 15），但最終 val loss 都在 ~0.0004-0.0005
- BiLSTM train/val gap 更小，泛化更好

**mMTC：**
- Prediction：紅線能追蹤藍線的整體趨勢，但**高值段嚴重低估**，黃色 spike 區域幾乎都有明顯落差。部分 segment 紅線只輸出「平均帶」，無法追蹤快速波動
- mMTC 有大量 segment 內高頻波動和尖刺，這才是真正考驗時序預測能力的場景
- Loss curve 非常健康：train/val 幾乎重疊，~epoch 105 early stop，無 overfitting
- **模型已充分收斂但 R²=0.40 → 不是訓練不夠，是模型/特徵的表達能力不足以捕捉 mMTC 波動模式**

**uRLLC：**
- Prediction：與 mMTC 類似但更嚴重 — 高值段大幅低估，spike 區域紅線幾乎全被壓低，部分 segment 紅線波幅只有藍線一半
- Loss curve 健康：train/val gap 小，~epoch 190 early stop
- 同樣是**已收斂但表達能力不足**

**跨 slice 比較：**

| | eMBB | mMTC | uRLLC |
|--|------|------|-------|
| Prediction 品質 | 紅藍貼合（但 segment 內近乎平的） | 趨勢可追蹤，高值低估 | 趨勢勉強，高值嚴重低估 |
| Loss curve | 健康，BiLSTM 微 overfitting | 非常健康，無 overfitting | 健康，無 overfitting |
| 瓶頸 | R² 被 segment 間差異灌水 | 模型表達能力不足 | 模型表達能力不足 |

**綜合判斷：**
- eMBB 在 Colosseum 模擬環境中流量本身平穩，segment 內變異小，高 R² 不代表真正的時序預測能力
- mMTC/uRLLC 的 loss 已收斂且無 overfitting → 加 epoch、調正則化無幫助
- mMTC/uRLLC 的瓶頸可能在於：sequence_length=15（3.75 秒）太短、11 個特徵不夠描述流量動態、或這兩種 slice 本身可預測性低

#### 各實驗詳細結果

**embb (LSTM)** — `results/embb_20260325_234312/`（patience=25）
- All: MAE=0.0043, RMSE=0.0054, R²=0.9789
- Spike(N=16460): MAE=0.0039, RMSE=0.0058, R²=0.9678
- Normal(N=124921): MAE=0.0044, RMSE=0.0054, R²=0.9798
- MAE ratio: 0.8874

**embb (BiLSTM)** — `results/embb_20260326_111857(biLSTM)/`（patience=25）
- All: MAE=0.0044, RMSE=0.0055, R²=0.9780
- Spike(N=16460): MAE=0.0034, RMSE=0.0052, R²=0.9743
- Normal(N=124921): MAE=0.0045, RMSE=0.0056, R²=0.9781
- MAE ratio: 0.7527

**mmtc (BiLSTM)** — `results/mmtc_20260325_222018/`（patience=25）
- All: MAE=0.0023, RMSE=0.0036, R²=0.3958
- Spike(N=17873): MAE=0.0061, RMSE=0.0077, R²=-0.6150
- Normal(N=123106): MAE=0.0018, RMSE=0.0025, R²=0.4966
- MAE ratio: 3.3794

**urllc (BiLSTM)** — `results/urllc_20260326_041826/`（patience=25）
- All: MAE=0.0079, RMSE=0.0135, R²=0.4140
- Spike(N=17896): MAE=0.0216, RMSE=0.0305, R²=-0.3230
- Normal(N=119277): MAE=0.0059, RMSE=0.0085, R²=0.5270
- MAE ratio: 3.6843

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

### Results with Rolling Features + Sliding Window (2026-03-23)

| Metric | Autoencoder | Isolation Forest |
|--------|------------|-----------------|
| F1 | 0.169 | 0.207 |
| Precision | 0.137 | 0.167 |
| Recall | 0.219 | 0.271 |
| Accuracy | 0.737 | 0.745 |
| FP Rate | 0.191 | 0.189 |
| FN Rate | 0.781 | 0.729 |

IF threshold sweep best: P90 → P=0.170, R=0.249, F1=0.202. Rolling features + sliding window 幾乎沒有改善。

### 診斷結論：Spike Label 定義缺陷 + 特徵不足 (2026-03-23)

**Spike Label 問題：**
- 公式 `τ = Q_0.9(last 1200) × (1 + 0.05 × ξ_m/ξ_max)` 中 ψ=0.05 幾乎無作用
- Threshold 實質上就是 local Q90 → 任何 stationary window 永遠有 ~10% spike
- Test set 12.21% spike rate 印證此問題 — 標的不是「異常突刺」而是「local 前 10%」
- Unsupervised 模型無法區分正常高值 vs 真正異常

**修復方案（三選一，待 Peter 決定）：**

#### 方案 A: Global Threshold
```python
global_mean = target_all_train.mean()
global_std = target_all_train.std()
tau = global_mean + k * global_std  # k=2 or 3
```
- 只有偏離全局分佈的值才算 spike，spike rate 預期降到 2-5%

#### 方案 B: Global Threshold + 突變條件
```python
diff = target.diff().abs()
spike = (target > global_threshold) & (diff > diff_threshold)
```
- 值高且來得快才算 spike，更符合「突刺」語義

#### 方案 C: 保留 Rolling 架構但調高 ψ
- ψ 從 0.05 提到 0.3~0.5，讓波動率調節真正生效
- 最保守的修改，但可能仍不夠

**特徵工程待加入：**

| 特徵 | 公式 | 用途 |
|------|------|------|
| `z_score_m` | `(target - rolling_mean_m) / rolling_std_m` | 短期異常程度 |
| `z_score_L` | `(target - rolling_mean_L) / rolling_std_L` | 長期異常程度 |
| `ratio_to_q90` | `target / rolling_q90_L` | 與歷史高值比值 |
| `diff_abs` | `abs(target.diff())` | 突變幅度 |
| `diff_pct` | `target.pct_change().abs()` | 百分比突變 |
| `rolling_std_L` | L window std | 長期波動率 |

另外 Autoencoder flatten 15×17=255 維丟失時序結構，可考慮改用 Conv1D/LSTM Autoencoder。

### Plan B 實作與首次結果 (2026-03-23)

已在 model_detect_peak 實作方案 B（global threshold + 突變條件），Peter 選擇 B 因為主要目標是 **regression（traffic prediction）**，spike head 是輔助工具讓 regression 在突變時段更準確。

**為何論文原公式在 Colosseum 資料上失效：**
- 論文用 Barcelona 5G 資料，有事件驅動的劇烈 spike（足球賽），遠超 Q90
- Colosseum 是模擬環境，流量相對均勻，「高值」與「正常值」差距不大
- 同一公式在不同資料特性下效果天差地遠

**方案 B 公式：**
```python
spike = (target > global_mean + k * global_std) & (|diff| > diff_mean + j * diff_std)
```
- `--k`: 值門檻（越大越嚴格）
- `--j`: 突變門檻（越大越嚴格）

**首次結果（k=2, j=2）— 過嚴：**

| Metric | Autoencoder | Isolation Forest |
|--------|------------|-----------------|
| F1 | 0.009 | 0.047 |
| Precision | 0.005 | 0.024 |
| Recall | 0.157 | 0.902 |
| Spike Rate | 0.54% | 0.54% |

- k=2 篩掉 ~97.5% 值，j=2 篩掉 ~97.5% diff，AND 後只剩 0.54% spike
- 目標 spike rate 應在 3-5%，需放寬參數
- 下一步：試 `--k 2 --j 1` 或 `--k 1.5 --j 1.5`

## Dependencies

PyTorch, torchvision, pandas, numpy, scikit-learn (MinMaxScaler, IsolationForest), matplotlib. No requirements.txt exists.