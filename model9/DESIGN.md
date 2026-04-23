# Model9: Multi-Domain Prediction + Chebyshev Anomaly Detection

## Status
Design phase (2026-04-23)。尚未實作 code。

## Purpose
重現論文 *A Prediction-Based Anomaly Detection Method for Traffic Flow with Multi-Domain Feature Extraction* 的 pseudo code 架構（BiLSTM + GAT + FFT-dual-Transformer，per-node prediction，Chebyshev threshold），但**預測目標沿用 model8**：

- Target：`sum_granted_prbs`，horizon=1
- Sequence length：T=15
- Per-slice 獨立訓練（embb / mmtc / urllc）
- 沿用 model8 的 data pipeline (RobustScaler, chunk shuffle sampler, per-CSV val split)

## Why
- Model8 第三輪結果：mMTC R²=0.13 / uRLLC R²=0.09，weighted MSE 改善 tail bias 但拖壞 bulk，整體 R² 反而退步
- 結論：特徵資訊不足是中長期瓶頸（memory: project_model8_status.md）
- 預期 multi-domain 特徵（time + space + frequency）能為 target 提供更豐富 context signal，突破 mean regression
- Paper 本身是 *Prediction-Based **Anomaly Detection***，連帶重現 Chebyshev threshold 作為下游輸出

## For
Peter 的 O-RAN traffic prediction 研究，與 model8 / model8_1 做架構對等比較。

## 關鍵約束
1. Per-slice 獨立訓練（三 slice 採集時段不同，無 cross-slice）
2. Colosseum 資料無 anomaly label；既有 spike 定義（model5 spike head、model_detect_peak 的 isolation tree）已驗證在此 dataset 上失效
3. CLI args 以 model8 為基礎，data/training/loss 部分 100% 沿用或擴充；arch 部分按 branch 重新組織；且要確認argu的內容有寫進去
4. Architecture-agnostic 的 results artifact（prediction plots / metrics / history）全部沿用 model8 樣式

## Non-goals
- Cross-slice unified training
- Synthetic anomaly 注入
- Anomaly detection 量化評估（precision / recall / F1）— 僅做視覺化 + flag rate 統計
- Real-time inference 或 deployment 最佳化

## Assumptions
1. `sequence_length=15` 對 rfft 產生 `T//2+1 = 8` 個 unique frequency bin，足夠支撐 frequency branch 表達
2. Colosseum metrics.csv 含 `sum_granted_prbs` 欄位，可供 N=16 primary 將 target history 當作第 16 個 feature node
3. Feature-as-node 的 adjacency 用 fully-connected + self-loop 時，GAT 的 attention 可自行學出 feature 間關聯（實質上是 cross-feature learnable attention）
4. N=16 small graph + per-node 共享 BiLSTM/Transformer 的計算成本可接受（相較 model8 約 ~16x batch pass）
5. Model8 的 `DataProcessor.py` 可擴充為 model9 版本，不需要完全重寫
6. Spike pseudo-label 不可用於 anomaly eval，因此本設計不考慮 detection metric

---

## 架構總覽

```text
Input window:   x_hist ∈ R^{B × T × N}      (T=15, N=16 primary / N=15 ablation)
Adjacency:      A ∈ {0,1}^{N × N}           (binary fully-connected + self-loop, primary)

┌──────────────────────────────────────────────────────────────────────────────┐
│ TemporalBranch (per-node 共享 BiLSTM)                                        │
│   x.permute(0,2,1).reshape(B·N, T, 1)                                        │
│     → Linear(1 → bilstm_input_proj_dim=16)                                   │
│     → BiLSTM(hidden=lstm_hidden=64, layers=lstm_layers=2)                    │
│     → take last step [B·N, 2·hidden]                                         │
│     → Linear(2·hidden → H=64) + ReLU + Dropout(lstm_dropout)                 │
│     → reshape to [B, N, H]                                                   │
└──────────────────────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────────────────────┐
│ SpatialBranch (GAT)                                                          │
│   x.permute(0,2,1)                                # [B, N, T]                │
│     → Linear(T=15 → gat_input_proj_dim=32) + Dropout(gat_dropout)            │
│     → GATLayer(in=32, out=gat_hidden=64, heads=4)  → ELU                     │
│     → GATLayer(in=64, out=H=64, heads=4)                                     │
│   heads merge: mean (primary) / concat (ablation，per-layer 可分開切換)      │
│   adjacency: binary fully-connected + self-loop (primary)                    │
└──────────────────────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────────────────────────────┐
│ FrequencyBranch (FFT + dual Transformer)                                             │
│   x.permute(0,2,1)                                # [B, N, T]                        │
│   f = torch.fft.rfft(x, dim=-1)                   # [B, N, F] complex (F=T//2+1=8)   │
│   mag   = |f|          ,  phase = angle(f)        # each real, [B, N, F]             │
│   for spec in {mag, phase}:                                                          │
│     reshape [B·N, F, 1]                                                              │
│     → Linear(1 → fft_hidden=64)                                                      │
│     → + learnable positional embedding (shape [1, token_count, fft_hidden])          │
│     → TransformerEncoder(n_heads=4, n_layers=2, ff=256, dropout=0.1, norm_first=True)│
│     → mean pool over F (primary; CLS / last 為 ablation)                             │
│     → Linear(fft_hidden → fft_hidden) + ReLU + Dropout(fft_dropout)                  │
│     → reshape [B, N, fft_hidden]                                                     │
│   concat [h_mag, h_phase]                                                            │
│     → Linear(2·fft_hidden → H=64) + ReLU + Dropout(fft_dropout)                      │
│   → [B, N, H]                                                                        │
└──────────────────────────────────────────────────────────────────────────────────────┘

Fusion: concat(h_temp, h_spat, h_freq) → [B, N, 3H]
        → Linear(3H → H) + ReLU + Dropout(head_dropout)     # head_dropout = max(lstm_dropout, gat_dropout, fft_dropout)
        → Linear(H → horizon=1) per-node → [B, N, 1] → squeeze → [B, N]

Readout over N (mean pool primary):
        [B, N] → reduce(dim=-1) → [B]       (ablation: attention, gated weighted sum)

Loss: base_loss(y_hat, y_true) [+ λ_smooth · smooth_diff_consistency]
      base_loss ∈ {MSE, Huber, Weighted MSE, Weighted Huber}

Anomaly (post-training, val-set calibrated):
      e_val   = y_hat_val - y_true_val                      # residual
      μ_s, σ_s = mean(e_val), std(e_val)                    # signed
      μ_a, σ_a = mean(|e_val|), std(|e_val|)                # abs
      on test: flag_signed[t] = (e_t < μ_s - kσ_s) or (e_t > μ_s + kσ_s)
               flag_abs[t]    = (|e_t| > μ_a + kσ_a)
```

---

## 模組細節

### 1. Input & Features (N=16 primary)

**Feature-as-node 對應**：把 model8 的 15 個 input feature **加上** `sum_granted_prbs` 歷史，當作 N=16 個 virtual nodes；每個 node 的序列是 T=15 步歷史。

- N=16（primary）：`FEATURE_ORDER` (model8 15 個) ∪ `{sum_granted_prbs}`
- N=15（ablation）：與 model8 input 完全對齊，做 fairness 比較（透過 `--include_target_history False` 切換）

資料流入模型前 per-feature 做 RobustScaler（train-only fit，沿用 model8）。

### 2. TemporalBranch — per-node 共享 BiLSTM

- 每個 feature node 的 T=15 scalar 序列獨立送進共享 BiLSTM（parameter sharing across N）
- Input projection `Linear(1 → 16)` 防止 `input_size=1` 造成的訓練不穩（pseudo code 工程建議）
- Take last timestep 的 hidden state（符合 pseudo code 寫法；因 T=15 較短，額外 pooling 收益不大）
- Output projection：`Linear(2·hidden → H=64) → ReLU → Dropout(lstm_dropout)`，統一 fusion 維度並加非線性 + 正則化

### 3. SpatialBranch — GAT

- 輸入：每個 node 的 T=15 長度歷史當作 node feature vector
- `Linear(T=15 → 32) → Dropout(gat_dropout)` 先投影再進 GAT（pseudo code 在 GAT 小節補充說明的工程做法；Dropout 做輸入 regularization）
- 兩層 multi-head GAT（4 heads）累積兩跳 message passing；層間以 ELU 非線性銜接
- Heads merge：primary `mean`；ablation `concat`（需 `output_dim % heads == 0`）
  - `--gat_head_merge` 控制非最後層，`--gat_final_head_merge` 控制最後層（對應原論文「中間 concat、最後 mean」可由 CLI 組合）
- Adjacency primary：binary fully-connected with self-loop（`A[i,j]=1` for all i,j）
  - GAT 的 attention 自行學 feature 間重要性；self-loop 讓 target node 可保留自身 signal
  - ablation: binary no self-loop（`A[i,j]=1 if i≠j`）、correlation-based（train set abs-corr > `--adj_corr_threshold`）

### 4. FrequencyBranch — FFT + dual Transformer

- `torch.fft.rfft` 對每個 node 的 T=15 序列做 FFT → 8 個 unique complex bin
- 拆成 magnitude + phase 兩條**獨立** Transformer 路徑（Figure 1(b) 與 pseudo code）
- 每個 Transformer：
  - `Linear(1 → fft_hidden=64)` 投影為 token
  - **加 learnable positional embedding**（shape `[1, token_count, fft_hidden]`；token_count = F，CLS 模式下為 F+1）
  - `TransformerEncoder(n_heads=4, n_layers=2, ff=256, dropout=0.1, norm_first=True)`（pre-LN）
  - Readout：mean pool（primary）/ CLS / last bin（ablation）
  - `Linear(fft_hidden → fft_hidden) → ReLU → Dropout(fft_dropout)` 做 output projection
- DC bin (index 0) 保留（scaling 後接近 0，對模型貢獻微弱但無害）
- `fuse = Linear(2·fft_hidden → H) → ReLU → Dropout(fft_dropout)` 融合 magnitude 與 phase representation

### 5. PredictionHead + Readout over N

- Fusion：`concat([h_temp, h_spat, h_freq], dim=-1)` → `[B, N, 3H]`
- Per-node MLP：
  - `node_head_hidden = Linear(3H → H) → ReLU → Dropout(head_dropout)`
    - `head_dropout = max(lstm_dropout, gat_dropout, fft_dropout)`（取最大保守值；三 branch 的 dropout 強度不一致時由 head 統一吸收）
  - `node_head_out = Linear(H → horizon)` → `[B, N, horizon]`
- 最後 readout over N：mean pool（primary）→ `[B, horizon]` → scalar target
- Ablation readout：
  - `attention`：learnable query vector 對 `node_head_hidden` 輸出（`[B, N, H]`）做 attention，score softmax 後加權 `node_pred`
  - `gated`：learnable weight per node，softmax 後加權平均 `node_pred`

**備忘**：N 軸 readout 會稀釋 target node 信號（選 A 選項時已知的 tradeoff）。若 mean pool 效果不佳，ablation 會切 attention 觀察是否 target-related node 權重自動變大。

---

## Anomaly Detection — Chebyshev Threshold

### 流程
1. **訓練完成後**，用 val loader 一 pass 計算 `y_hat_val, y_true_val`
2. 計算 residual `e = y_hat - y_true`（scaled space 與 original space 各輸出一份）
3. 估計：
   - Signed：`μ_s = mean(e)`, `σ_s = std(e)` → threshold `[μ_s − kσ_s, μ_s + kσ_s]`
   - Abs：`μ_a = mean(|e|)`, `σ_a = std(|e|)` → threshold `μ_a + kσ_a`
4. 在 test 上逐 sample 比對，輸出 binary flag（signed + abs 兩份）

### k 值
- Default `k=3`（pseudo code / Chebyshev inequality 常用值）
- 額外在 `k=2, 3` 下都輸出 flag rate，便於視覺化比較（`--anomaly_extra_k 2.0 3.0`）

### Per-slice implicit
每個 slice 有自己的 model，μ/σ/threshold 自然 per-slice。N 維度在 readout 後已消失 → 無 per-node threshold 概念。

### 無量化 eval
Dataset 無 anomaly label；既有 spike 定義在此 dataset 上失效（memory 明確記錄）。Anomaly 部分僅輸出：
- `anomaly_flags.npy` — test set binary mask
- `anomaly_summary.txt` — μ/σ/threshold/flag_rate/k
- `anomaly_overlay.png` — `y_true / y_pred / flag` 三線疊圖（sanity check）

---

## Training Pipeline（繼承 model8）

- Optimizer：Adam（沿用 model8）
- LR scheduler：cosine（default）或 none
- Early stopping：`patience` epoch 無 val 改善即停（`patience=0` 關閉）
- Smooth diff consistency loss：`λ × mean(|Δŷ − Δy|)` on chunk-internal pairs（chunk 邊界 mask）
- Scale-aware λ：`effective λ = base × sqrt(lambda_ref_iqr / target_iqr)`（沿用 model8）
- **Model9 default `--lambda_smooth=0.0`**（關閉 smooth loss；要用再顯式開，與 model8 default=0.1 不同）
- Weighted MSE / Huber：`weight(y) = 1 + α × clamp((y − q_low)/(q_high − q_low), 0, 1)`，base loss 可換成 MSE 或 Huber

---

## Data Pipeline（繼承 model8 + 擴充）

### 繼承
- RobustScaler per-slice global，train-only fit（features + target 各一個 scaler）
- Chunk shuffle sampler：CSV 為最小時間單位，chunk size = `--chunk_size=64`
- Val split：`--val_dirs` 顯式 leave-trial-out 優先，`--val_split` per-CSV 比例 fallback
- Sequence length=15, horizon=1, 記住不可跨CSV

### 擴充
- `--include_target_history` (default `True`)：
  - `True` → FEATURE_ORDER 加入 `sum_granted_prbs`（N=16）
  - `False` → 沿用 model8 的 15-feature FEATURE_ORDER（N=15，fairness ablation）
- Adjacency 建構（`--adj_type`）：
  - `binary_selfloop` → `A = ones(N, N)`
  - `binary_noselfloop` → `A = ones(N, N) - eye(N)`
  - `correlation` → `A = (|corr_train| > adj_corr_threshold).float()` + self-loop

---

## Output Artifacts

每個 run 產生 `model9/results/{slice}_{timestamp}/`：

```
results/{slice}_{timestamp}/
├── config.json                    ← CLI args dump（所有 args 自動 include）
├── model_best.pth                 ← 最佳 checkpoint
├── training.log                   ← stdout log
├── history.csv                    ← per-epoch train/val loss
├── metrics.csv                    ← test MAE / R² / MSE 等
├── anomaly_summary.txt            ← 新：μ/σ/threshold/flag_rate for k=2, 3
├── anomaly_flags.npy              ← 新：test set binary flag (signed + abs)
└── plots/
    ├── history.png                ← 訓練曲線
    ├── prediction.png             ← y_true vs y_pred 時序
    ├── prediction_combined.png    ← 多 csv 合併
    ├── scatter.png                ← y_true vs y_pred 散點
    ├── error_hist.png             ← residual 直方圖
    ├── error_box.png              ← residual box plot
    ├── residuals.png              ← residual 隨時間
    └── anomaly_overlay.png        ← 新：y_true / y_pred / flag 疊圖
```

---

## File Structure

```
model9/
├── DataProcessor.py     ← 繼承 model8 邏輯 + --include_target_history 擴充
├── model.py             ← TemporalBranch + SpatialBranch + FrequencyBranch + PredictionHead + Readout
├── anomaly.py           ← Chebyshev threshold 估計 + flag 計算
├── train.py             ← 繼承 model8 訓練迴圈 + 訓練後呼叫 anomaly pipeline
├── DESIGN.md            ← 本文件
├── README.md            ← usage 範例
└── results/             ← auto-created per run
```

GAT 自寫 `GATLayer`（放在 `model.py` 內），不引入 `torch_geometric` / `dgl`。

---

## CLI Arguments

### 🟦 Group 1: Data 與 split（100% 沿用 model8）

| Arg | Default | 說明 |
|---|---|---|
| `--data_root` | `Dataset/colosseum-oran-coloran-dataset` | 資料根目錄 |
| `--train_dirs` | required | 訓練 trial dirs (e.g. `tr0-24`) |
| `--val_dirs` | `None` | 驗證 trial dirs；設了就 ignore `--val_split` |
| `--test_dirs` | required | 測試 trial dirs |
| `--slice_type` | required | `embb / mmtc / urllc` |
| `--val_split` | `0.2` | fallback per-CSV 切分比例 |
| `--sequence_length` | `15` | T |
| `--horizon` | `1` | H_out |
| `--chunk_size` | `64` | chunk sampler |

### 🟩 Group 2: Training loop（100% 沿用 model8）

| Arg | Default | 說明 |
|---|---|---|
| `--batch_size` | `1024` | |
| `--epochs` | `500` | |
| `--learning_rate` | `1e-4` | |
| `--weight_decay` | `0.0` | |
| `--scheduler` | `cosine` | `none / cosine` |
| `--patience` | `50` | 0 = disable early stopping |
| `--seed` | `42` | |
| `--num_workers` | `0` | |
| `--device` | `auto` | |

### 🟨 Group 3: Loss（擴充 Huber / Weighted Huber）

| Arg | Default | 說明 |
|---|---|---|
| `--loss_type` | `mse` | **擴充為** `mse / huber / weighted_mse / weighted_huber` |
| `--huber_delta` | `1.0` | **新增**（δ for Huber / Weighted Huber） |
| `--weight_alpha` | `4.0` | 沿用 |
| `--weight_quantile` | `0.9` | 沿用 |
| `--weight_upper_quantile` | `0.99` | 沿用 |
| `--lambda_smooth` | `0.0` | **default 改為 0**（model8 是 0.1） |
| `--disable_scale_aware_lambda` | `False` | 沿用 |
| `--lambda_ref_iqr` | `104.0` | 沿用 |

### 🟪 Group 4: Architecture

**沿用 model8 舊名（default 調整）：**

| Arg | model8 default | model9 default | 語意變化 |
|---|---|---|---|
| `--rnn_type` | `lstm` | `bilstm` | paper 用 BiLSTM |
| `--lstm_hidden` | `128` | `64` | model9 為 per-node 共享，小一點即可 |
| `--lstm_layers` | `2` | `2` | 不變 |
| `--lstm_dropout` | `0.2` | `0.2` | 不變 |

**從 model8 移除（model9 無對應分支）：**
`--d_model --n_heads --n_layers --dim_feedforward --vit_dropout`

**Model9 新增：**

| Arg | Default | 歸屬 | 說明 |
|---|---|---|---|
| `--hidden_dim` | `64` | 融合 | 三 branch 共用的 H，fusion 後維度 |
| `--include_target_history` | `True` | 輸入 | `True`=N=16 primary / `False`=N=15 fairness ablation |
| `--readout` | `mean` | Fusion pool | `mean / attention / gated` |
| `--bilstm_input_proj_dim` | `16` | BiLSTM | 1=無 projection，忠於 pseudo code |
| `--fft_hidden` | `64` | FFT | FFT Transformer d_model |
| `--fft_n_heads` | `4` | FFT | |
| `--fft_n_layers` | `2` | FFT | |
| `--fft_dim_feedforward` | `256` | FFT | |
| `--fft_dropout` | `0.1` | FFT | |
| `--fft_readout` | `mean` | FFT | `mean / cls / last` |
| `--adj_type` | `binary_selfloop` | GAT | `binary_selfloop / binary_noselfloop / correlation` |
| `--adj_corr_threshold` | `0.3` | GAT | 只在 `adj_type=correlation` 生效 |
| `--gat_hidden` | `64` | GAT | |
| `--gat_heads` | `4` | GAT | |
| `--gat_layers` | `2` | GAT | |
| `--gat_dropout` | `0.1` | GAT | |
| `--gat_input_proj_dim` | `32` | GAT | `Linear(T → gat_input_proj_dim)` 再進 GAT |
| `--gat_head_merge` | `concat` | GAT | 非最後層 heads 合併方式；`mean / concat` |
| `--gat_final_head_merge` | `mean` | GAT | 最後層 heads 合併方式；`mean / concat` |

### 🟥 Group 5: Anomaly detection（全新）

| Arg | Default | 說明 |
|---|---|---|
| `--chebyshev_k` | `3.0` | threshold = μ + kσ (abs) 或 μ ± kσ (signed) |
| `--anomaly_error_mode` | `both` | `signed / abs / both` |
| `--anomaly_extra_k` | `[2.0, 3.0]` | 額外輸出這些 k 值的 flag rate |

### Argparse 驗證
- `--weight_alpha >= 0`
- `0 ≤ weight_quantile < weight_upper_quantile ≤ 1`
- `0 < val_split < 1`（僅在未指定 `--val_dirs` 時需要）
- `huber_delta > 0`
- `chebyshev_k > 0`
- `0 < adj_corr_threshold < 1`（僅在 `adj_type=correlation` 時）
- `train_dirs / val_dirs / test_dirs` 三組交集為空
- `--gat_head_merge=concat` 時 `gat_hidden % gat_heads == 0`
- `--gat_final_head_merge=concat` 時 `hidden_dim % gat_heads == 0`（最後層 output_dim 為 `hidden_dim`）

---

## Decision Log

| # | 項目 | Primary | Ablation / 備選 | Excluded | 理由 |
|---|---|---|---|---|---|
| 1 | Scope | Prediction + Chebyshev anomaly flag (qualitative) | — | Synthetic injection / quantitative anomaly eval | 無可靠 label，既有 spike 定義失效 |
| 2 | Node 對應 | Feature-as-node (A) | — | Univariate+side-branch (B), 只取 target node (C) | 最貼近 pseudo code 三 branch 結構 |
| 3 | N 大小 | 16 (含 `sum_granted_prbs` 歷史) | 15 (fairness ablation) | — | 保留 target autocorrelation signal |
| 4 | Readout over N | Mean pool | Attention pool, gated weighted sum | Max pool, concat flatten | 最簡無額外參數；max 易被噪音主宰；concat 爆參數 |
| 5 | GAT adjacency | Binary fully-connected w/ self-loop | Binary no self-loop, correlation-based | Learnable, distance-weighted | self-loop 保留 target 自身 signal；fully-connected 讓 GAT attention 自學 |
| 6 | FFT type | `torch.fft.rfft` | — | `fft` | 實數序列無需 conjugate 冗餘 |
| 7 | FFT readout | Mean pool over bins | CLS token | Last bin | pseudo code 作者自推薦；無額外參數 |
| 8 | DC bin | 保留 | — | 丟 DC | Scaling 後接近 0，保留無害 |
| 9 | BiLSTM input | `input_size=1 → Linear(1, 16) → LSTM` | `input_size=1` 無 projection | — | 短序列 scalar 輸入，projection 穩定訓練 |
| 10 | Base loss | MSE | Huber, Weighted MSE, Weighted Huber（可 CLI 切換） | — | 先排除 loss 變數看架構效果，其他版本為 ablation |
| 11 | smooth_loss default | `0.0` | `0.1` (沿用 model8 表現) | — | 先看純 base loss；與 model8 比較時再開 |
| 12 | Anomaly error mode | Signed + Abs 都輸出 | — | 只 signed 或只 abs | 資訊量完整 |
| 13 | Threshold estimation | Val set μ, σ | — | Test leakage | pseudo code 建議、避免 leakage |
| 14 | Chebyshev k | 3.0 (同時輸出 k=2, 3 flag rate) | — | — | 常用 default；無 label 不能自動調 |
| 15 | Pipeline inheritance | RobustScaler / chunk sampler / per-CSV val / smooth loss / scale-aware λ 全沿用 | — | — | 與 model8 對等比較，架構差異為唯一變數 |
| 16 | CLI arg 舊名 | 沿用 model8 `--lstm_*` `--rnn_type` 名稱 | — | 移除 `--d_model --n_heads --n_layers --dim_feedforward --vit_dropout` | LSTM args 語意一致僅調 default；Transformer 原屬 temporal ViT 無 model9 對應 |
| 17 | GAT 實作 | 自寫 `GATLayer` | — | torch_geometric / dgl | 零新 dependency；小圖 fully-connected 無 sparse 需求 |
| 18 | 檔案結構 | 鏡像 model8 + `anomaly.py` | — | — | 與既有 repo 風格一致；anomaly 獨立模組便於 reuse |
| 19 | Branch output 是否加 ReLU+Dropout | 全部加（TemporalBranch output_proj、SpatialBranch input_proj 後的 Dropout、FrequencyBranch per-encoder output_proj 與 fuse、PredictionHead hidden） | 純 Linear（嚴格 pseudo code） | — | 三 branch 輸出都進同一個 fusion，統一非線性 + 正則化避免 co-adaptation；head_dropout 取三 branch dropout 最大值 |
| 20 | FFT Transformer 位置資訊 | Learnable positional embedding（shape `[1, token_count, fft_hidden]`，token_count 在 CLS readout 下為 F+1） | 無 pos / sinusoidal | — | bin 的序順有意義（頻率由低到高），需給 Transformer 位置訊號；learnable 不額外假設分佈 |
| 21 | GAT heads merge | primary mean / mean（全部層） | `--gat_head_merge` + `--gat_final_head_merge` 可組合成「中間 concat、最後 mean」等 paper 默認 | — | mean 維度一致簡單；提供 CLI 切換以複現 paper 或做 ablation |

---

## Open Questions
無。所有設計選擇已於 brainstorming session 2026-04-23 鎖定。

---

## Implementation Plan（待後續啟動）

### Phase 1 — DataProcessor
- Fork `model8/DataProcessor.py` → `model9/DataProcessor.py`
- 新增 `FEATURE_ORDER_WITH_TARGET = FEATURE_ORDER + ["sum_granted_prbs"]`
- 依 `--include_target_history` 切換 FEATURE_ORDER
- 新增 `build_adjacency(adj_type, corr_threshold, train_features)` 工具函式
- 輸出 `adjacency.npy` 供 GAT 使用

### Phase 2 — Model
- `GATLayer` 自寫 multi-head attention on adjacency
- `TemporalBranch / SpatialBranch / FrequencyBranch / PredictionHead / Readout`
- `MultiDomainTrafficPredictor` 整合

### Phase 3 — Anomaly
- `anomaly.py`：`chebyshev_threshold_signed / abs`、`detect_anomaly_*`、`compute_flag_summary`

### Phase 4 — train.py
- Fork model8/train.py
- 加入新 CLI args（Group 4 / 5）
- Training loop 與 model8 幾乎一致，loss 擴充 Huber 支援
- 訓練完成後呼叫 anomaly pipeline 並輸出 artifact

### Phase 5 — Run & Compare
- 三 slice 各跑一次 primary config
- 與 model8 第三輪、model8_1 對照 test MAE / R²
- 驗收：mmtc / uRLLC R² 是否突破 0.13 / 0.09 的地板

---

## References
- `peter/A Prediction-Based Anomaly Detection Method for Traffic Flow with Multi-Domain Feature Extraction - pseudo code.md` — 架構 pseudo code
- `peter/A Prediction-Based Anomaly Detection Method for Traffic Flow with Multi-Domain Feature Extraction.pdf` — 原論文
- `ORAN-Traffic-Prediction/model8/DESIGN.md` — model8 完整設計與 Decision Log
- `ORAN-Traffic-Prediction/docs/model8.md` — model8 三輪實驗結果
- `ORAN-Traffic-Prediction/model8_1/` — Transformer ablation 對照組
