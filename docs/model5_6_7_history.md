# Model5 / Model6 / Model7 — Spike-Aware 與 Tiny ViT 歷史

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
- 詳見上方 Model7 Tiny ViT Details 段落

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

### 待嘗試方案（歷史紀錄）
- ~~**model7_1 ablation**：先跑純 regression baseline~~ ✅ 已完成，見 [model7_1_ablation.md](model7_1_ablation.md)
- Plan C: 分階段訓練（Phase 1 只訓練 regression，Phase 2 解凍 spike head）
- Plan D: Gradient Scaling（spike head 梯度乘以 0.1 再回傳 backbone）
- Focal Loss 取代 BCEWithLogitsLoss
- Huber Loss (SmoothL1Loss) 取代 MSE
- 降低推論門檻：logit threshold 從 0.0 改為 -1.0
