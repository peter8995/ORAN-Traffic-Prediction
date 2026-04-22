# Model8 Design Document

> Status: Design locked, awaiting implementation
> Date: 2026-04-13
> Owner: Peter

---

## 1. Goal

解決 mMTC / uRLLC 在 PRB regression 任務上的品質不足問題（過往 R² ≈ 0.4）。
首要評估指標為 **per-slice 的 MAE / RMSE / R² / MAPE**，目標欄位為 `sum_granted_prbs`。

## 2. Scope

- 三個 5G slice（embb / mmtc / urllc）**獨立訓練**、獨立 checkpoint、獨立報表。
- CLI `--slice_type` 切換，每次只跑一個 slice。
- **明確排除** cross-slice 聯合訓練與 cross-slice attention（資料時間段未對齊）。

## 3. Non-goals

- 不做 cross-slice 聯合或 attention
- 不做機率預測 / 分位數預測 / spike detection head
- 首版不實際跑 multi-horizon（架構 ready，CLI 預設 horizon=1）
- 不支援中斷續跑、多卡訓練
- 不整合 TensorBoard / wandb

---

## 4. Architecture

### 4.1 Forward Pass

```
Input  (batch, seq_len, 15)
  │
  ├─► Linear projection (15 → d_model)
  │
  ├─► Positional embedding (learnable, length seq_len)
  │
  ├─► TinyViT: nn.TransformerEncoder (no CLS token)
  │     output: (batch, seq_len, d_model)
  │
  ├─► LSTM 或 BiLSTM  (rnn_type 可切換)
  │     output: (batch, seq_len, lstm_hidden * dirs)
  │
  ├─► Take last timestep
  │     output: (batch, lstm_hidden * dirs)
  │
  ├─► FF head: hidden → hidden/2 → horizon
  │
  └─► Output (batch, horizon)   # horizon=1 首版
```

### 4.2 模組職責

| 模組 | 職責 |
|---|---|
| `DataProcessor.py` | CSV 載入、特徵組裝、global per-slice RobustScaler fit/transform、sliding window（不跨 csv）、按 experiment 切 train/val、回傳 `TrafficDataset` |
| `model.py` | TinyViT + RNN + FF 預測頭組成 `TrafficModel`，forward 回傳 `(batch, horizon)` |
| `train.py` | Chunk shuffle sampler、訓練迴圈、loss 計算（MSE + smooth）、early stopping、評估、視覺化、輸出檔案結構 |

---

## 5. Data Pipeline

### 5.0 Dataset 格式

**根目錄**：`Dataset/colosseum-oran-coloran-dataset/`

**目錄階層**：
```
colosseum-oran-coloran-dataset/
└── tr{N}/                    # N = 0..27，共 28 個 trial（不同實驗設定）
    └── exp{M}/               # M = 1..k，每個 trial 內多個 experiment
        └── bs{B}/            # B = 1..7，每個 experiment 內多個 base station
            ├── embb/         # eMBB slice
            │   └── {IMSI}_metrics.csv
            ├── mtc/          # mMTC slice（注意：資料夾叫 mtc 不是 mmtc）
            │   └── {IMSI}_metrics.csv
            └── urllc/        # uRLLC slice
                └── {IMSI}_metrics.csv
```

**CSV 命名**：`{IMSI}_metrics.csv`，IMSI 是 15 位數 UE 識別碼，每個 slice 資料夾下可能有 1 個或多個 UE 的 csv（典型 1-2 個）。

**單一 csv 屬性**：
- 行數：約 **2,000-2,100 列**（含 header）
- 採樣間隔：約 **250 ms**（從 Timestamp 欄位推算）
- 一個 csv ≈ **8-9 分鐘**連續時序
- 編碼：UTF-8，逗號分隔

**CSV 欄位**（36 個，含空欄位）：

```
Timestamp, num_ues, IMSI, RNTI, <empty>, slicing_enabled, slice_id, slice_prb,
power_multiplier, scheduling_policy, <empty>, dl_mcs, dl_n_samples,
dl_buffer [bytes], tx_brate downlink [Mbps], tx_pkts downlink,
tx_errors downlink (%), dl_cqi, <empty>, ul_mcs, ul_n_samples,
ul_buffer [bytes], rx_brate uplink [Mbps], rx_pkts uplink,
rx_errors uplink (%), ul_rssi, ul_sinr, phr, <empty>,
sum_requested_prbs, sum_granted_prbs, <empty>, dl_pmi, dl_ri, ul_n, ul_turbo_iters
```

**欄位分類**：

| 類別 | 欄位 |
|---|---|
| 時間 | `Timestamp`（毫秒整數） |
| 識別 | `IMSI`, `RNTI`, `num_ues`, `slice_id` |
| 配置（常數）| `slicing_enabled`, `slice_prb`, `power_multiplier`, `scheduling_policy` |
| 輸入特徵（15 KPI） | `dl_buffer`, `ul_buffer`, `tx_brate`, `rx_brate`, `ul_sinr`, `dl_mcs`, `ul_mcs`, `phr`, `tx_pkts`, `rx_pkts`, `ul_rssi`  `dl_cqi`, `rx_errors_ul`, `dl_n_samples`, `ul_n_samples` |
| 預測目標 | `sum_granted_prbs` |
| 不使用 | `sum_requested_prbs`（eMBB 幾乎全 0）、`tx_errors downlink`（全 0）、`dl_pmi`, `dl_ri`（全 0）、`ul_turbo_iters`、`ul_n`、空欄位 |

**Slice 名稱對應**：
- CLI `--slice_type embb` → 讀取 `*/bs*/embb/*.csv`
- CLI `--slice_type mmtc` → 讀取 `*/bs*/mtc/*.csv`（**注意資料夾為 mtc**）
- CLI `--slice_type urllc` → 讀取 `*/bs*/urllc/*.csv`

**Train dirs 範圍記號**：CLI 接受 `tr0-26` 自動展開為 `tr0, tr1, ..., tr26`，也接受多參數 `tr0-4 tr10`。

**資料量估計**（單一 slice，27 個 trial）：
- 每 csv ≈ 2,000 列
- 每 bs/slice ≈ 1-2 csv → 每 bs ≈ 2,000-4,000 列
- 每 exp ≈ 7 bs → 每 exp ≈ 14,000-28,000 列
- 每 trial ≈ 5 exp → 每 trial ≈ 70,000-140,000 列
- 27 trial total ≈ **2-4 百萬 timesteps per slice**

### 5.1 輸入特徵（15 個，寫死於 DataProcessor）

**原 15 個 KPI**：
`dl_buffer`, `ul_buffer`, `tx_brate`, `rx_brate`, `ul_sinr`, `dl_mcs`, `ul_mcs`, `phr`, `tx_pkts`, `rx_pkts`, `ul_rssi`
`dl_cqi`, `rx_errors_ul`, `dl_n_samples`, `ul_n_samples`

### 5.2 預測目標

`sum_granted_prbs`

### 5.3 Scaler

- **Global per-slice RobustScaler**（一個 slice 一個 scaler）
- 只在 train set 上 fit
- features 與 target 各自獨立 scaler
- 評估時 inverse_transform 回原尺度

### 5.4 Sliding Window 與邊界規則

- **csv 是時序連續性的最小單位** — sliding window、chunk、train/val 切分都不能跨 csv 邊界
- 對每個 csv 獨立做 sliding window，再 concat
- Sequence length 由 `--sequence_length` 控制
- horizon 由 `--horizon` 控制（首版預設 1）

### 5.5 Train / Val / Test 切分

兩種 val 策略擇一：

**(A) 顯式 leave-trial-out（推薦，對齊 test 分佈）**
- `--val_dirs` 指定完整的 val trial（沿用 `tr0-26` 範圍記號），例如 `--val_dirs tr25 tr26`
- val 與 train 同為 leave-trial-out，`val_total_loss` 可直接作為 test 效能代理（early stopping 不會被「同 trial 不同 bs」的高相關性樣本誤導）
- 指定 `--val_dirs` 時 `--val_split` 會被忽略

**(B) 自動隨機 CSV 切分（舊行為）**
- 未指定 `--val_dirs` 時啟用
- val 從 train 切：`--val_split 0.2`，**按 csv 為單位** shuffle 後切，不在 csv 內部切
- ⚠️ 同 `(tr, exp)` 的 7 個 bs 可能被分到兩側，泛化風險高

**三組 dirs 強制不交集**
- `DataProcessor.prepare()` 檢查 train∩test、train∩val、test∩val，任一有交集直接 `ValueError`
- 失敗點在讀 CSV 之前，使用者不會白等 I/O

### 5.6 Chunk Shuffle Sampler

- chunk_size 由 `--chunk_size` 控制（預設 64）
- 每個 csv 內先切成連續 chunks，chunk 不跨 csv
- chunks 之間 shuffle，chunk 內保持時序
- mini-batch 由連續 chunks 串成

---

## 6. Loss Function

```
L_total = L_main + λ_smooth × L_smooth
```

- **L_main**: MSE（在 scaled 空間計算）
- **L_smooth**: 差分一致性
  ```
  L_smooth = mean(|(ŷ_t - ŷ_{t-1}) - (y_t - y_{t-1})|)
  ```
  - chunk 邊界（每個 chunk 第一個 sample）mask 掉
  - csv 邊界因為 chunk 不跨 csv 已自動處理
- **λ_smooth**: CLI `--lambda_smooth`，預設 0.1

---

## 7. Training Configuration

| 項目 | 設定 |
|---|---|
| Optimizer | AdamW (lr=1e-3) |
| Weight decay | CLI `--weight_decay`，預設 0 |
| LR scheduler | CLI `--scheduler {none, cosine}`，預設 none |
| Batch size | 1024 |
| Epochs | 500 |
| Early stopping | patience=50，monitor val_loss，restore best |
| Random seed | CLI `--seed`，預設 42（torch / numpy / random 全設） |
| Checkpoint | 只存 `model_best.pth`（val_loss 最低） |

### Model Hyperparameters（preset A，輕量 ≈0.8M）

| 參數 | 預設 |
|---|---|
| `d_model` | 128 |
| `n_heads` | 4 |
| `n_layers` | 3 |
| `dim_feedforward` | 512 |
| `vit_dropout` | 0.2 |
| `rnn_type` | lstm |
| `lstm_hidden` | 128 |
| `lstm_layers` | 2 |
| `lstm_dropout` | 0.2 |

---

## 8. CLI預設

```bash
python train.py \
  --train_dirs tr0-24 \
  --val_dirs tr25 tr26 \
  --test_dirs tr27 \
  --slice_type embb \
  --sequence_length 15 \
  --horizon 1 \
  --batch_size 1024 \
  --chunk_size 64 \
  --epochs 500 \
  --learning_rate 1e-3 \
  --weight_decay 0.0 \
  --scheduler none \
  --rnn_type lstm \
  --d_model 128 \
  --n_heads 4 \
  --n_layers 3 \
  --dim_feedforward 512 \
  --vit_dropout 0.2 \
  --lstm_hidden 128 \
  --lstm_layers 2 \
  --lstm_dropout 0.2 \
  --lambda_smooth 0.1 \
  --patience 50 \
  --seed 42
```

> 舊行為（自動 CSV 切分）：把 `--val_dirs tr25 tr26` 換成 `--val_split 0.2`。

### 參數分組

| 群組 | 參數 |
|---|---|
| 資料 | `--train_dirs`, `--val_dirs`, `--test_dirs`, `--slice_type`, `--val_split` |
| 序列 | `--sequence_length`, `--horizon`, `--chunk_size` |
| 訓練 | `--batch_size`, `--epochs`, `--learning_rate`, `--weight_decay`, `--scheduler`, `--patience`, `--seed` |
| 模型-RNN | `--rnn_type {lstm,bilstm}`, `--lstm_hidden`, `--lstm_layers`, `--lstm_dropout` |
| 模型-ViT | `--d_model`, `--n_heads`, `--n_layers`, `--dim_feedforward`, `--vit_dropout` |
| Loss | `--lambda_smooth` |

---

## 9. Evaluation & Outputs

### 9.1 評估指標（test set）

- MAE, RMSE, R², MAPE（target=0 的 sample mask 掉）
- 在原始尺度（inverse_transform 後）計算

### 9.2 視覺化（7 種，README 對齊）

1. Per-slice predictions（紅藍線）
2. Combined predictions（all slices）
3. Error distributions（histogram）
4. Error box plots
5. Residuals over time
6. Training history curves（train/val loss + main/smooth 分項）
7. Scatter plots with R² + perfect prediction diagonal

### 9.3 輸出結構

```
model8/results/{slice_type}_{YYYYMMDD_HHMMSS}/
├── config.json            # 完整 CLI args 快照
├── model_best.pth         # best checkpoint
├── metrics.csv            # MAE/RMSE/R²/MAPE
├── metrics.tex            # LaTeX 表格
├── history.csv            # 每 epoch loss/lr/time
├── training.log           # stdout 全文
└── plots/
    ├── prediction.png
    ├── prediction_combined.png
    ├── error_hist.png
    ├── error_box.png
    ├── residuals.png
    ├── history.png
    └── scatter.png
```

---

## 10. Non-Functional Requirements

| 項目 | 設定 |
|---|---|
| 計算資源 | 單張 GPU |
| 訓練時間 | 每 slice < 2 小時（輕量 + early stop） |
| 可重現性 | 固定 seed，同 args 同結果 |
| Logging | print + `training.log`，無 TensorBoard / wandb |
| Checkpoint | best only，無中斷續跑 |

---

## 11. Decision Log

| # | 決策 | 替代方案 | 選擇理由 |
|---|---|---|---|
| 1 | 主目標：解決 mMTC/uRLLC R² 不足 | multi-horizon / multi-slice 聯合 / 全部 | 聚焦 model7_1 揭示的最大瓶頸 |
| 2 | 排除 cross-slice attention | model1 式聯合 | 三 slice 資料時間段未對齊，物理上無意義 |
| 3 | Per-slice 獨立執行 train.py | 一鍵跑三 slice 的 loop | 易 debug，易針對單 slice 調參 |
| 4 | Horizon=1 為首版，但全鏈路 horizon-aware | 寫死 1 / 完整 multi-horizon | YAGNI 與 ready 的折衷，零成本未來擴充 |
| 5 | 無 CLS token 的 TinyViT | 保留 CLS / TinyViT 內部用 CLS 但不傳給 LSTM | 下游是 LSTM 不是分類，CLS 冗餘且破壞時序語義 |
| 6 | rnn_type 可切換 LSTM/BiLSTM | 寫死其一 | model7_1 顯示兩者在不同 slice 表現有差 |
| 7 | Smooth loss = 預測差分 vs 真實差分一致性 | 純預測差分 / hidden 平滑 / 權重平滑 | 對 horizon=1 適用，且直接對應「突變段也要準」目標 |
| 8 | Chunk shuffle sampler（chunk_size=64） | 純隨機 / 完全連續 / 雙 dataloader | 兼顧梯度多樣性與差分有效性 |
| 9 | 15 features = 11 原 KPI + dl_cqi + rx_errors + dl/ul_n_samples | 加 sum_requested_prbs / 全 HIGH+MAYBE / 維持 11 個 | sum_requested_prbs 在 eMBB 幾乎全 0；其他 4 個跨 slice 都有訊號 |
| 10 | Target = sum_granted_prbs | sum_requested_prbs | 反映實際分配，與調度行為對齊 |
| 11 | RobustScaler（per-slice global） | MinMax / Standard | PRB 有突峰 outlier，MinMax 會壓縮正常值解析度 |
| 12 | Train/val 按 csv（experiment）切，不按 timestep | 隨機 timestep 切 | 防止時序洩漏 |
| 12a | 新增 `--val_dirs` 顯式 leave-trial-out，取代預設的隨機 CSV 切分 | 只保留隨機 CSV / 分層抽 exp | val 要和 test（leave-trial-out）同分佈，才能讓 early stopping 的選擇對 test 有意義；舊的隨機 CSV 切會把同 (tr, exp) 不同 bs 分兩側，造成 val_loss 過度樂觀 |
| 12b | train / val / test dirs 交集直接 raise | 自動剔除交集 / 印 warning | 自動剔除會讓 CLI 寫的 train 數量與實際不符，日後追訓練條件混亂；raise 逼使用者明寫 `tr0-24` |
| 13 | csv 是時序最小單位（sliding window/chunk/切分都不跨） | 跨 csv 拼接 | 不同 csv 屬不同實驗，跨界違反物理 |
| 14 | Loss = MSE + 0.1 × smooth | SmoothL1 / 不同 λ | 選擇符合使用者偏好 |
| 15 | AdamW + cosine 為 CLI option，預設 weight_decay=0、scheduler=none | 寫死配方 | 與 early stopping 不完全相容，先 baseline 再決定 |
| 16 | Patience=50 | 25 / 不用 | 給 smooth loss 收斂時間 |
| 17 | 預設 model preset = 輕量 A (≈0.8M) | 中型 B / 大型 C | 從輕量驗證 pipeline，避免 underfit/overfit 與 bug 混淆 |
| 18 | 評估指標：MAE/RMSE/R²/MAPE | 加分區段指標 | 分區段指標需定義「高值區門檻」，YAGNI |
| 19 | 全 7 種視覺化 | 精簡 4 種 | README 既然列了，照寫 |
| 20 | 輸出路徑：`results/{slice}_{timestamp}/` | 其他結構 | 沿用前面 model 慣例 |
| 21 | NFR：單卡、固定 seed、only best ckpt、純文字 log | 多卡 / TensorBoard / 續跑 | YAGNI，論文需求已足夠 |

---

## 12. Implementation Risks & Mitigations

| 風險 | Mitigation |
|---|---|
| 新增 4 特徵某 slice 全零 → 噪聲 | DataProcessor 載入後 log 各特徵的 non-zero ratio，發現問題回頭討論 |
| RobustScaler 在某 slice 上 IQR=0 → 除零 | sklearn 會自動回退，但實作時需 catch warning |
| MAPE 在 target≈0 處爆炸 | mask `y_true == 0` 的 sample；輸出時註記 mask 比例 |
| chunk_size 若大於某些 csv 長度 → 該 csv 變成不足一塊的尾巴 | 接受，尾巴單獨成一塊或丟掉，看資料分佈決定 |
| smooth loss 主導梯度 → regression 退化 | λ=0.1 為保守起點；history.csv 分項記錄 main/smooth loss 便於診斷 |
| TinyViT 0.8M 太小 underfit mMTC/uRLLC | 預設輕量做 baseline，必要時切換 preset B/C |