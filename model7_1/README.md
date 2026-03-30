# Model7_1: Regression-Only Traffic Prediction (Ablation Baseline)

## Overview

Model7_1 是 Model7 的**消融實驗版本**，移除了 Spike 偵測頭，僅保留回歸任務。用於驗證多任務學習 (spike detection) 是否對回歸性能有幫助。

**與 Model7 的差異:**
- 移除 Spike 偵測頭 (無 BCEWithLogitsLoss)
- **所有切片均使用 BiLSTM** (Model7 中 eMBB 使用 LSTM)
- Loss 僅為 MSE

---

## Model Architecture

```
Input: (batch, 15, 11)
         │
         ▼
┌─────────────────────────┐
│       TinyViT           │
│  Linear(11 → 128)       │
│  + CLS Token            │
│  + Positional Embedding │
│  TransformerEncoder ×3  │
│  LayerNorm              │
└─────────┬───────────────┘
          │ (batch, 16, 128)
          ▼
┌─────────────────────────┐
│   BiLSTM (all slices)   │
│  BiLSTM(128→128) →256   │
│  Dropout(0.2)           │
│  BiLSTM(256→64) →128    │
└─────────┬───────────────┘
          │ (batch, 16, 128)
          ▼
┌─────────────────────────┐
│  MultiheadAttention     │
│  (4 heads, self-attn)   │
│  → 取最後一個 token      │
└─────────┬───────────────┘
          │ (batch, 128)
          ▼
┌─────────────────────────┐
│   Regression Head       │
│  Linear(128→64) → ReLU  │
│  Dropout(0.1)           │
│  Linear(64→32) → ReLU   │
│  Dropout(0.1)           │
│  Linear(32→1)           │
└─────────────────────────┘
          │
          ▼
    Output: (batch, 1)
```

### Component Details

#### 1. TinyViT (輕量 Vision Transformer)

與 Model7 相同的自定義輕量 ViT (~0.8M params)。

| 參數 | 值 |
|------|-----|
| d_model | 128 |
| n_heads | 4 |
| n_layers | 3 |
| dim_feedforward | 512 |
| dropout | 0.1 |
| Activation | GELU |

- **Input Projection**: `Linear(11, 128)`, Xavier Uniform 初始化
- **CLS Token**: Learnable `Parameter(1, 1, 128)`, Truncated Normal (std=0.02)
- **Positional Embedding**: Learnable `Parameter(1, 16, 128)`, Truncated Normal (std=0.02)
- **TransformerEncoder**: 3 層, 每層含 MultiheadAttention + FFN + LayerNorm + Residual

#### 2. BiLSTM (所有切片統一使用)

```
BiLSTM Layer 1: input=128, hidden=128, bidirectional → output=256
  ↓ Dropout(0.2)
BiLSTM Layer 2: input=256, hidden=64, bidirectional → output=128
```

與 Model7 不同：**eMBB 也使用 BiLSTM** (Model7 中 eMBB 使用單向 LSTM)。

#### 3. Attention Fusion

- `MultiheadAttention(embed_dim=128, num_heads=4, dropout=0.1)`
- Self-attention (Q=K=V)
- 取最後一個 token 作為 context vector

#### 4. Regression Head

```
Linear(128, 64) → ReLU → Dropout(0.1)
Linear(64, 32) → ReLU → Dropout(0.1)
Linear(32, 1)  # 無激活函數
```

**Total Parameters: ~532K**

---

## Input / Output

### Input Features (11 維)

| # | Feature | 說明 |
|---|---------|------|
| 1 | dl_buffer [bytes] | 下行緩衝區 |
| 2 | ul_buffer [bytes] | 上行緩衝區 |
| 3 | tx_brate downlink [Mbps] | 下行傳輸速率 |
| 4 | rx_brate uplink [Mbps] | 上行接收速率 |
| 5 | ul_sinr | 上行信噪比 |
| 6 | dl_mcs | 下行調變編碼 |
| 7 | ul_mcs | 上行調變編碼 |
| 8 | phr | 功率餘裕報告 |
| 9 | tx_pkts downlink | 下行封包數 |
| 10 | rx_pkts uplink | 上行封包數 |
| 11 | ul_rssi | 上行接收訊號強度 |

- **Sequence Length**: 15 timesteps (每步 500ms → 7.5 秒 context)
- **Scaling**: MinMaxScaler [0, 1]

### Output

- **回歸**: `(batch, 1)` — 預測 granted PRBs (scaled)
- 無 Spike 偵測輸出

---

## Training Setup

### Loss Function

```
Loss = MSE(pred, target)
```

純回歸損失，無 spike detection 輔助任務。

### Hyperparameters

| 參數 | 值 |
|------|-----|
| Epochs | 500 |
| Batch Size | 1024 |
| Learning Rate | 1e-4 |
| Optimizer | Adam (weight_decay=0.0) |
| LR Scheduler | CosineAnnealingLR (T_max=500) |
| Gradient Clipping | max_norm=1.0 |
| Early Stopping | patience=25 |
| Val Split | 0.2 |

### Spike Label (僅用於分析)

Spike labels 仍會計算 (使用與 Model7 相同的 adaptive threshold)，但**不參與訓練**，僅用於測試時分析 spike/normal 區域的預測表現差異。

---

## Data Pipeline

```
Raw CSV (Colosseum O-RAN Dataset)
  → Drop all-NA columns, fill NA with 0
  → Fit MinMaxScaler on training data
  → Compute spike labels (for analysis only)
  → Scale features
  → Sliding window (size=15, stride=1)
  → Train/Val split (80/20, seed=42)
  → DataLoader (batch=1024)
```

- **Train**: tr0–tr26 (27 trial directories)
- **Test**: tr27

---

## Performance

### Overall Regression

| Slice | MAE | RMSE | R² |
|-------|-----|------|----|
| eMBB (LSTM) | 0.0043 | 0.0054 | **0.9789** |
| eMBB (BiLSTM) | 0.0044 | 0.0055 | **0.9780** |
| mMTC (BiLSTM) | 0.0023 | 0.0036 | 0.3958 |
| uRLLC (BiLSTM) | 0.0079 | 0.0135 | 0.4140 |

### Spike vs Normal 區域分析

| Slice | Spike MAE | Normal MAE | MAE Ratio | Spike R² | Normal R² |
|-------|-----------|------------|-----------|----------|-----------|
| eMBB (LSTM) | 0.0039 | 0.0044 | 0.89 | 0.9678 | 0.9798 |
| eMBB (BiLSTM) | 0.0034 | 0.0045 | 0.75 | 0.9743 | 0.9781 |
| mMTC | 0.0061 | 0.0018 | 3.38 | -0.6150 | 0.4966 |
| uRLLC | 0.0216 | 0.0059 | 3.68 | -0.3230 | 0.5270 |

### Key Observations

#### eMBB
- R²≈0.978，但有**灌水成分**：不同 experiment 的流量等級差異大（如某段 ~0.02、某段 ~0.05），模型實質上是在預測**每個 segment（experiment）的均值水平**，而非 segment 內的 timestep-level 時序波動。segment 內紅藍線近乎水平，高 R² 不代表真正的時序預測能力。
- BiLSTM 在 spike 區域表現優於 LSTM（MAE ratio 0.75 vs 0.89）
- Loss curve：LSTM 收斂慢（~epoch 80），BiLSTM 收斂快（~epoch 15），但最終 val loss 相近（~0.0004-0.0005）；BiLSTM train/val gap 更小，泛化更好

#### mMTC / uRLLC
- R²≈0.4，spike 區域 R² 為負值（比預測平均值還差），MAE ratio 達 3.38 / 3.68
- 高值段嚴重低估，spike 區域幾乎全被壓低
- **Loss curve 非常健康**（train/val 幾乎重疊，無 overfitting）→ 瓶頸不是訓練不足，而是**模型/特徵表達能力不足以捕捉 mMTC/uRLLC 波動模式**
- 可能原因：sequence_length=15（~3.75 秒）太短、11 個特徵不足以描述流量動態、或 mMTC/uRLLC 本身可預測性較低

---

## Model7 vs Model7_1 Comparison

> 比較基準為 Model7 Round 3（vit_dropout=0.2, weight_decay=0, patience=50）

| 指標 | Model7 (Multi-task) | Model7_1 (Regression-only) |
|------|---------------------|---------------------------|
| eMBB MAE | 0.0046 | **0.0043** |
| eMBB RMSE | 0.0060 | **0.0054** |
| eMBB R² | 0.9739 | **0.9780** |
| mMTC MAE | 0.0024 | **0.0023** |
| mMTC RMSE | 0.0036 | 0.0036 |
| mMTC R² | 0.3894 | **0.3958** |
| uRLLC R² | — | **0.4140** |
| Spike Detection | 有 (F1: 0.24~0.48) | 無 |
| eMBB Seq Model | LSTM | BiLSTM |

**結論**：Spike detection 輔助任務未提升回歸性能，Model7_1 在所有 slice 上 regression 微幅優於 Model7。spike head 未能讓 backbone 對突變更敏感，反而分散了學習能力。

---

## File Structure

```
model7_1/
├── classicModel.py    # Model definitions (TinyViT, BiLSTM, TrafficModel)
├── DataProcessor.py   # Data loading, scaling, spike labeling, windowing
├── train.py           # Training loop, evaluation, CLI arguments
└── results/
    ├── embb_20260325_234312/          # eMBB with LSTM
    ├── embb_20260326_111857(biLSTM)/  # eMBB with BiLSTM
    ├── mmtc_20260325_222018/
    └── urllc_20260326_041826/
```
