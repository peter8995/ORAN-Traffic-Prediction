# Model9 Brainstorming Session Log

**Date**: 2026-04-23
**Skill**: `brainstorming` (one question per message, Understanding Lock before design)
**Outcome**: 架構與設計全數鎖定 → `DESIGN.md`
**Facilitator**: Claude (Opus 4.7)

---

## Session Goal

使用者請求：「使用 `A Prediction-Based Anomaly Detection Method for Traffic Flow with Multi-Domain Feature Extraction - pseudo code.md` 的架構生出一個 model9，但預測的東西跟我本來的 model8 一樣」。

### 關鍵架構 Gap（開場分析）

- Pseudo code 假設 `B × T × N`（N 個 node/sensor，per-node 預測 flow scalar），需 `N × N` adjacency
- model8 是**單一 slice 的多特徵時間序列**：input `(B, 15, 15_features)`，output `sum_granted_prbs` 單一 scalar，per-slice 獨立訓練

→ 「N 個 node」在 model8 的設定下沒有天然對應，必須先決定 mapping，才能進後續設計。

---

## Q1 — 「N 個 node」該怎麼對應到 model8 的設定？

### 選項
- **(A) Feature-as-node**：把 15 features 當作 N=15 virtual nodes，adjacency 用 feature correlation。BiLSTM per-feature 獨立跑，GAT 在 feature 間 message passing，FFT per-feature。最後 node 軸 pool 或取 target node → scalar。
- **(B) Univariate + 多特徵 concat**：target 當 N=1 node 走 BiLSTM/FFT，其他 14 features 另走 side branch（MLP）加進 fusion。GAT 因 N=1 退化，移除。
- **(C) Multi-feature-as-node + 只預測 target**：與 (A) 類似，但 prediction head 只從 target node 的 embedding 輸出 scalar，不做 node pool。

### 對話過程

**User**: 「C 只有這個缺點嗎? 這兩個選項差別是甚麼?」

**Claude**: 解釋 (A) 與 (C) 只差在 readout。若 (A) 的 readout 選「只取 target node」就等於 (C)。列出 (C) 的其他隱性缺點：BiLSTM 丟失跨 feature temporal interaction、adjacency 語意薄弱、FFT per-feature 語意不強、計算成本 ~15x。

**User**: 「那A的缺點呢 請詳述」

**Claude**: 列出 (A) 獨有缺點（pooling 引入的）：target 信號稀釋、readout 無 principled 選擇、與論文 per-node prediction 精神矛盾、GAT message passing 失去明確目標、與 target semantic 不直覺。

### Decision
**✅ (A) Feature-as-node**

---

## Q2 — Scope：要不要包含 anomaly detection？

### 選項
- **(i) 只做 prediction**：跟 model8 對等比較
- **(ii) Prediction + Chebyshev anomaly detection**：完整重現 pseudo code pipeline
- **(iii) 先做 (i)，之後再加 (ii)**

### Decision
**✅ (ii) Prediction + Chebyshev anomaly detection 都做**

---

## Q3 — Anomaly detection 的 ground truth 從哪來？

### 選項
- **(i) 不評估，只輸出 flag**
- **(ii) 用 spike 當 pseudo-label 評估**（沿用 model_detect_peak / model5 spike 定義）
- **(iii) 人工注入 synthetic anomaly**

### 對話過程

**User**: 「i 有缺點嗎? 我沒有要注入」

**Claude**: 列出 (i) 的缺點：
1. 無法驗證 anomaly detection 的品質（paper/thesis 講不出準確率）
2. k 值沒法調
3. 與 model5 / model_detect_peak 無法橫向比較
4. 隱性 bug 容易被蓋掉
5. 視覺化可部分 mitigate 但非量化

提出折衷方案 (i-b)：用既有 spike 定義當 pseudo-label。

**User**: 「model5裡面的spike 公式在我這個dataset內是沒有作用的 連簡單的isolation tree都無法正確預測」

→ 既有 spike 定義不可靠，pseudo-label 不可行 → 回到 (i-a)。

### Decision
**✅ (i-a) 純 prediction + Chebyshev flag 輸出 + 視覺化 sanity check，不算量化 metric**

關鍵約束入 Decision Log：「dataset 無可靠 anomaly label，既有 spike 定義（model5 / isolation tree）已驗證在 Colosseum 上失效」。

---

## Q4 — N 軸的 readout 怎麼做？

### 選項
- **(a) Mean pool** — 最簡，無參數
- **(b) Max pool** — 無參數，易被噪音 feature 主宰
- **(c) Attention pool（learnable query）** — 有參數，可學 feature 重要性
- **(d) Gated / weighted sum** — 輕量 learnable weight
- **(e) Concat 全部 flatten** — 參數爆炸，不建議

### 對話過程

**User**: 「先跑a 把BC留著當備選」 → 後修正：「把CD當備選我打錯了」

### Decision
- Primary: **(a) Mean pool**
- Ablation: **(c) Attention pool、(d) Gated weighted sum**
- Excluded: **(b) Max pool、(e) Concat flatten**

---

## Q5 — GAT 的 adjacency matrix 怎麼建？

### 選項
- **(a) Binary fully-connected no self-loop**
- **(b) Binary fully-connected with self-loop**
- **(c) Correlation-based**（train set abs-corr threshold）
- **(d) Learnable adjacency**
- **(e) Distance-weighted (Gaussian kernel)**

### 對話過程

**User**: 「b當primary ac 當correlation」 → 解讀為 (b) primary，(a) 與 (c) 當 ablation。

### Decision
- Primary: **(b) Binary fully-connected with self-loop**
- Ablation: **(a) No self-loop、(c) Correlation-based**
- Excluded: **(d) Learnable、(e) Distance-weighted**

---

## Q6 — `sum_granted_prbs` 歷史要不要當第 16 個 feature node？

### 選項
- **(i) N=16**：加入 target 歷史，更貼近 pseudo code「用 flow 歷史預測 flow」精神
- **(ii) N=15**：沿用 model8 FEATURE_ORDER，fair comparison
- **(iii) 雙版本**：(i) primary + (ii) fairness ablation

### Decision
**✅ (iii) 雙版本**
- Primary: N=16（含 `sum_granted_prbs` 歷史）
- Fairness ablation: N=15
- 透過 `--include_target_history` 切換

---

## Q7 — FFT branch 細節

### 7a: rfft vs fft
- (i) `torch.fft.rfft` — 8 unique bins
- (ii) `torch.fft.fft` — 15 bins with conjugate redundancy

### 7b: Frequency domain readout
- (i) Last bin — 忠於 pseudo code 字面
- (ii) Mean pool over bins — pseudo code 作者自推薦
- (iii) CLS token — learnable，多一組參數

### 7c: DC component (bin 0)
- (i) 保留
- (ii) 丟掉

### User Response: 「7a-i / 7b-ii,iii當備選 / 7c-i」

### Decision
- **7a**: (i) `rfft`
- **7b**: (ii) Mean pool primary，(iii) CLS token 為 ablation；excluded (i) last bin
- **7c**: (i) 保留 DC bin

---

## Q8 — BiLSTM branch 輸入維度

### 選項
- (i) `input_size=1`（忠於 pseudo code，scalar 直接進 LSTM）
- (ii) `input_size=1 → Linear(1→d_proj)`（工程穩定版本）

### User Response: 「Q8-ii」

### Decision
**✅ (ii) with `--bilstm_input_proj_dim=16` default**，(i) 留做 ablation（設 `--bilstm_input_proj_dim=1`）

---

## Q9 — inherit model8 pipeline 哪些？

| 項目 | 是否沿用 |
|---|---|
| RobustScaler per-slice global, train-only fit | ✅ |
| `--val_dirs` 顯式 leave-trial-out | ✅ |
| Chunk shuffle sampler (chunk size=64) | ✅ |
| Sequence length=15 | ✅ |
| Horizon=1 | ✅ |
| Smooth diff consistency loss | ✅（default 關閉；見 Q10） |
| Per-slice 獨立訓練 | ✅ |

### User Response: 「Q9想問之前有argument可供更改的可以保留嗎?」

### Decision
- **全部沿用**，model8 既有 CLI args 100% 保留，model9 為 superset
- 舊 model8 命令幾乎可直接跑（除了 arch-specific args）

---

## Q10 — Base loss 怎麼選？

### 選項
- (i) 純 MSE
- (ii) 純 Huber
- (iii) Weighted MSE（繼承 model8 第三輪未竟嘗試）

### User Response: 「i primary, huber, weight MSE, weight huber為可調整argument, smooth一樣留著」

### Decision
- Primary: **MSE**
- CLI-configurable: **Huber、Weighted MSE、Weighted Huber**
- `--loss_type` 擴充為 `{mse, huber, weighted_mse, weighted_huber}`
- 新增 `--huber_delta`（default 1.0）
- `+ λ × smooth diff consistency` 沿用（但 Q11 後續將 default 改 0）

---

## Q11 — Anomaly detection 細節

### 11a: Signed vs Absolute error
- (i) Signed `[μ±kσ]`
- (ii) Absolute `|e| > μ_|e| + kσ_|e|`
- (iii) 兩個都輸出

### 11b: Threshold 估計資料
- (i) Val set μ/σ（pseudo code 建議，避免 leakage）

### 11c: k default
- `--chebyshev_k` default `3.0`
- 額外輸出 `--anomaly_extra_k` 列表（default `[2.0, 3.0]`）

### 11d: Output 內容
- `anomaly_flags.npy`
- `anomaly_summary.txt`
- `anomaly_overlay.png`

### User Response: 「11a-iii / 11b ok / 11c ok / 11d 我本來Model8還有輸出很多其餘東西 都可以保留嗎? 還會有意義嗎?」

**Claude**: 檢查 model8 train.py，確認所有既有輸出（config.json / model_best.pth / training.log / history.csv / metrics.csv / plots/* 七張圖）都是 architecture-agnostic，在 model9 仍然有效。Anomaly 輸出為**加在**既有結構上的新項。

### Decision
- **11a**: (iii) Signed + Abs 兩個都輸出
- **11b**: (i) Val set μ/σ
- **11c**: k=3.0 default，同時輸出 k=2, 3 flag rate
- **11d**: model8 所有既有輸出保留 + anomaly 三個新項

---

## Q12 — Hyperparameter defaults

### 建議
| 超參 | Default | 說明 |
|---|---|---|
| `--hidden_dim` | 64 | 三 branch 共用 H |
| `--bilstm_input_proj_dim` | 16 | |
| `--gat_hidden` | 64 | |
| `--gat_heads` | 4 | |
| `--gat_layers` | 2 | |
| `--fft_n_heads` | 4 | |
| `--fft_n_layers` | 2 | |
| `--fft_dim_feedforward` | 256 | |
| `--fft_dropout` | 0.1 | |

重點：`hidden_dim=64`（比 model8 的 128 小），因 per-node 共享會被 N=16 倍放大。

### User Response: 「ok」

---

## Q13 — 檔案結構 + GAT 實作

### 13a: File layout
```
model9/
├── DataProcessor.py
├── model.py
├── train.py
├── anomaly.py         ← 新增，便於 reuse
├── DESIGN.md
├── README.md
└── results/
```

### 13b: GAT 實作
- (i) 自寫 GATLayer — 無新 dependency，pseudo code 使用此抽象
- (ii) `torch_geometric.nn.GATv2Conv` — 大 dep
- (iii) `dgl` — 大 dep

### User Response: 「13a ok 但先不要實作code生成design.md即可 / 13b i」

### Decision
- **13a**: ✅ 鏡像 model8 + `anomaly.py`；**code 暫不實作，只生 DESIGN.md**
- **13b**: (i) 自寫 `GATLayer`；excluded torch_geometric / dgl

---

## Arg 結構談判（Q13 後）

### Round 1 — 最終 arg 列表（Claude 提案）

分五組：
- 🟦 Group 1: Data 與 split（100% 沿用 model8）
- 🟩 Group 2: Training loop（100% 沿用 model8）
- 🟨 Group 3: Loss（擴充 Huber / Weighted Huber）
- 🟪 Group 4: Architecture（大改，因三 branch 結構不同）
- 🟥 Group 5: Anomaly detection（全新）

提出兩問：
1. Group 4 要不要把 `--lstm_*` rename 成 `--bilstm_*`？
2. 其他 default 有要改的嗎？

### User Response: 「1.保留舊名 2.lambda_smooth 預設改成 0」

### Round 2 — Claude 誤套用到 Transformer args

Claude 嘗試把 `--d_model --n_heads --n_layers --dim_feedforward --vit_dropout`（原為 model8 temporal ViT Transformer 用）repurpose 到 FFT branch。

### User Response: 「不對 FFT保留妳原本的寫法就好 不用套用transformer」

### Round 3 — 最終定案

| 處置 | Args |
|---|---|
| **✅ 保留 model8 舊名（default 調整）** | `--rnn_type --lstm_hidden --lstm_layers --lstm_dropout` |
| **❌ 從 model8 移除** | `--d_model --n_heads --n_layers --dim_feedforward --vit_dropout`（model9 無對應分支） |
| **🆕 FFT 用新名** | `--fft_hidden --fft_n_heads --fft_n_layers --fft_dim_feedforward --fft_dropout --fft_readout` |
| **🆕 GAT 用新名** | `--gat_hidden --gat_heads --gat_layers --gat_dropout --gat_input_proj_dim --adj_type --adj_corr_threshold` |
| **🆕 其他新增** | `--hidden_dim --include_target_history --readout --bilstm_input_proj_dim` |
| **🆕 Loss 擴充** | `--huber_delta`；`--loss_type` 擴充為四選項 |
| **🆕 Anomaly** | `--chebyshev_k --anomaly_error_mode --anomaly_extra_k` |
| **📝 Default 改** | `--lambda_smooth 0.1 → 0.0`、`--rnn_type lstm → bilstm`、`--lstm_hidden 128 → 64` |

---

## Understanding Lock（Confirmed 2026-04-23）

Claude 做 Understanding Summary（18 條 Decision Log）並要求確認。

### User Response: 「預計的argument有哪些呢? 讓我確認」

Claude 列出完整五組 arg 表格後，User 做 Round 1/2/3 arg 定案（見上）。

### User Response: 「ok 請把這個args結構也一並寫到design.md內」

→ Claude 生成 `DESIGN.md`（含完整 CLI args + 18 條 Decision Log）。

---

## Final Decision Log（18 條）

見 `DESIGN.md` 的 Decision Log 章節。核心：

1. Feature-as-node (A)
2. Mean pool primary
3. Binary self-loop adjacency primary
4. N=16 primary + N=15 ablation
5. rfft + mean pool + 保留 DC
6. BiLSTM with input projection 1→16
7. MSE primary, Huber / Weighted 配置化
8. `--lambda_smooth` default `0.0`
9. Chebyshev k=3 signed+abs, val-set estimation
10. 無 anomaly 量化 eval（dataset 限制）
11. 沿用 model8 pipeline 全部
12. model8 舊 `--lstm_*` arg 名保留
13. model8 Transformer args 移除
14. FFT Transformer 用新 `--fft_*` arg
15. 自寫 GATLayer，零新 dep
16. File 結構鏡像 model8 + `anomaly.py`
17. Output artifact 繼承 model8 + 加三個 anomaly 項
18. Per-slice 獨立訓練，絕不 cross-slice

---

## Rejected Ideas

- Univariate+side-branch (Q1-B)
- Target-node-only readout as separate option (Q1-C，後與 A 在 readout 選擇上統一處理)
- Max pool readout (Q4-b)
- Concat flatten readout (Q4-e)
- Learnable adjacency (Q5-d)
- Distance-weighted adjacency (Q5-e)
- FFT last bin readout (Q7b-i)
- Synthetic anomaly 注入 (Q3-iii)
- Spike-based pseudo-label anomaly eval (Q3-ii，已驗證不可行)
- `torch_geometric` / `dgl` 作為 GAT 實作 (Q13b-ii/iii)
- Rename `--lstm_*` 為 `--bilstm_*`
- Repurpose model8 Transformer args 到 FFT branch

---

## Deliverables

- ✅ `model9/DESIGN.md`（完整設計 + 18 條 Decision Log + 全部 CLI args）
- ✅ `model9/brainstorming.md`（本文件，session log）
- ✅ Memory：`project_model9_status.md` + `MEMORY.md` 更新
- ⬜ `model9/DataProcessor.py`（Phase 1，未實作）
- ⬜ `model9/model.py`（Phase 2，未實作）
- ⬜ `model9/anomaly.py`（Phase 3，未實作）
- ⬜ `model9/train.py`（Phase 4，未實作）
- ⬜ `model9/README.md`（未實作）
