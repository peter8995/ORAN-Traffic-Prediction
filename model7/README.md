# Model7: Spike-Aware Multi-Task Traffic Prediction

## Overview

Model7 是一個**多任務學習模型**，同時執行：
1. **回歸任務** — 預測 O-RAN 網路流量 (granted PRBs)
2. **Spike 偵測任務** — 偵測流量異常峰值 (二元分類)

針對三種 5G 網路切片 (eMBB / mMTC / uRLLC) 使用不同的序列模型架構。

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
│   Sequence Model        │
│  eMBB:  LSTM (→ 64)     │
│  mMTC:  BiLSTM (→ 128)  │
│  uRLLC: BiLSTM (→ 128)  │
└─────────┬───────────────┘
          │ (batch, 16, output_dim)
          ▼
┌─────────────────────────┐
│  MultiheadAttention     │
│  (4 heads, self-attn)   │
│  → 取最後一個 token      │
└─────────┬───────────────┘
          │ (batch, output_dim)
          ▼
    ┌─────┴─────┐
    ▼           ▼
┌────────┐ ┌──────────┐
│ 回歸頭  │ │ Spike 頭 │
│ FC→64  │ │ FC→32    │
│ FC→32  │ │ FC→1     │
│ FC→1   │ │ (logits) │
└────────┘ └──────────┘
```

### Component Details

#### 1. TinyViT (輕量 Vision Transformer)

自定義輕量 ViT，取代 torchvision vit_b_16 (86M params)，僅 ~0.8M params。

| 參數 | 值 |
|------|-----|
| d_model | 128 |
| n_heads | 4 |
| n_layers | 3 |
| dim_feedforward | 512 |
| dropout | 0.1 (default) / 0.2 (eMBB) |
| Activation | GELU |

- **Input Projection**: `Linear(11, 128)`, Xavier Uniform 初始化
- **CLS Token**: Learnable `Parameter(1, 1, 128)`, Truncated Normal (std=0.02)
- **Positional Embedding**: Learnable `Parameter(1, 16, 128)`, Truncated Normal (std=0.02)

#### 2. Sequence Models (依切片類型)

| Slice | Model | 結構 | Output Dim |
|-------|-------|------|-----------|
| eMBB | LSTM | LSTM(128→128) → Dropout(0.2) → LSTM(128→64) | 64 |
| mMTC | BiLSTM | BiLSTM(128→128) → Dropout(0.2) → BiLSTM(256→64) | 128 |
| uRLLC | BiLSTM | BiLSTM(128→128) → Dropout(0.2) → BiLSTM(256→64) | 128 |

#### 3. Attention Fusion

- `MultiheadAttention(embed_dim=output_dim, num_heads=4, dropout=0.1)`
- Self-attention (Q=K=V)
- 取最後一個 token 作為 context vector

#### 4. Prediction Heads

**回歸頭:**
```
Linear(output_dim, 64) → ReLU → Dropout(0.1)
Linear(64, 32) → ReLU → Dropout(0.1)
Linear(32, 1)
```

**Spike 偵測頭:**
```
Linear(output_dim, 32) → ReLU → Dropout(0.1)
Linear(32, 1)  # logits (BCEWithLogitsLoss)
```

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
- **Spike 偵測**: `(batch, 1)` — logits (> 0 表示 spike)

---

## Training Setup

### Loss Function

```
Total Loss = MSE(pred, target) + λ_detect × BCEWithLogitsLoss(spike_logits, spike_labels)
```

- `λ_detect = 1.0`
- `pos_weight = min(n_neg / n_pos, 7.0)` — 處理正負樣本不平衡

### Spike Label 生成 (Adaptive Threshold)

```
τ(t) = Q_0.9(Y[t-L:t]) × (1 + ψ × ξ_m(t) / ξ_max)
```

| 參數 | 值 | 說明 |
|------|-----|------|
| L | 1200 | Rolling Q90 窗口 (~10 分鐘) |
| m | 120 | Rolling std 窗口 (~1 分鐘) |
| ψ | 0.05 | 動態調整係數 |
| ξ_max | global | 訓練集最大波動度 |

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

### Regularization

- Dropout: 0.1~0.2
- Gradient Clipping: max_norm=1.0
- Early Stopping: patience=25

---

## Data Pipeline

```
Raw CSV (Colosseum O-RAN Dataset)
  → Drop all-NA columns, fill NA with 0
  → Fit MinMaxScaler on training data
  → Compute spike labels (adaptive threshold, on raw data)
  → Scale features
  → Sliding window (size=15, stride=1)
  → Train/Val split (80/20, seed=42)
  → DataLoader (batch=1024)
```

- **Train**: tr0–tr26 (27 trial directories)
- **Test**: tr27

---

## Performance (2026-03-26)

### Regression

| Slice | MAE | RMSE | R² |
|-------|-----|------|----|
| eMBB | 0.0048 | 0.0064 | **0.9702** |
| mMTC | 0.0024 | 0.0036 | 0.4031 |
| uRLLC | 0.0083 | 0.0142 | 0.3577 |

### Spike Detection

| Slice | Accuracy | Precision | Recall | F1 | FP Rate | FN Rate |
|-------|----------|-----------|--------|-----|---------|---------|
| eMBB | 39.12% | 13.79% | **80.50%** | 0.2354 | 66.33% | 19.50% |
| mMTC | 79.55% | 35.34% | 73.85% | 0.4780 | 19.62% | 26.15% |
| uRLLC | 78.57% | 34.18% | 69.42% | 0.4581 | 20.06% | 30.58% |

---

## File Structure

```
model7/
├── classicModel.py    # Model definitions (TinyViT, LSTM, BiLSTM, TrafficModel)
├── DataProcessor.py   # Data loading, scaling, spike labeling, windowing
├── train.py           # Training loop, evaluation, CLI arguments
└── results/
    ├── embb_20260326_155447/
    ├── mmtc_20260326_170316/
    └── urllc_20260326_200044/
```
