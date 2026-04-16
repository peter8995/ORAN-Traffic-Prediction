# model8: Regression-First Redesign (2026-04-13+)

全新設計，專攻 model7_1 留下的 mMTC/uRLLC regression 瓶頸（R²≈0.4）。
依 `model8/DESIGN.md` 實作，不參考前面 model 的 code。

## 核心決策
- **Per-slice 獨立訓練**（拒絕 cross-slice unified／cross-slice attention，因三個 slice 資料採集時段不同）
- 移除 spike head（model7_1 已證明無幫助）
- RobustScaler 取代 MinMaxScaler（per-slice global，train-only fit）
- Chunk shuffle sampler（csv 內切 chunk，chunk 內順序保留，跨 csv 不混）
- Multi-horizon 架構已就緒，第一版 horizon=1
- Loss = MSE + λ × smooth diff consistency；smooth loss 在 chunk/experiment 邊界要 mask
- Val split 以 csv（experiment）為單位，非 random timestep

## 檔案
- `model8/DataProcessor.py` — RobustScaler + ChunkShuffleSampler + per-CSV sliding window
- `model8/model.py` — TinyViT + LSTM/BiLSTM + FF head
- `model8/train.py` — AdamW + 可選 cosine + early stopping + 7 visualizations
- `model8/DESIGN.md` — 完整設計與 21-row Decision Log
- `model8/README.md` — 簡述三檔功能

## Training Command
```bash
cd model8
python train.py --train_dirs tr0-26 --test_dirs tr27 --slice_type mmtc \
  --epochs 500 --patience 50 --batch_size 1024 \
  --learning_rate 1e-4 --scheduler cosine \
  --rnn_type bilstm --lambda_smooth 0.1
```

## 特徵（13 維，2026-04-14 調整）
原 15 維扣掉兩個常數特徵：
- **Dropped**: `ul_rssi`（全 train set 近乎全 0）、`ul_buffer`（<1% 非零）
- **Kept (13)**: dl_buffer, tx_brate, rx_brate, ul_sinr, dl_mcs, ul_mcs, phr, tx_pkts, rx_pkts, dl_cqi, rx_errors_ul, dl_n_samples, ul_n_samples
- Target: `sum_granted_prbs`（原 model1-7 用 `sum_requested_prbs`）

## 第一輪結果 (2026-04-13，有 bug，已回溯)

`model8/results/{embb,mmtc,urllc}_20260413_*`：

| Slice | rnn | Best epoch | Test MAE | R² |
|-------|-----|-----------|----------|-----|
| embb | lstm | 5 | 179.14 | **0.7632** |
| mmtc | bilstm | 2 | 31.28 | **0.0385** |
| urllc | bilstm | 15 | 65.07 | **0.1690** |

**診斷發現的 5 個問題：**
1. `patience=500`（config）→ early stopping 實質停用，跑滿 500 epoch
2. mmtc best epoch=2 + 後期多次 val loss 爆炸（311, 63, 33）
3. uRLLC val loss 低於 train loss（反向 gap）
4. eMBB ~epoch 100 後 overfitting
5. `ul_rssi` 在三個 slice 的 train set 都是常數 0

## 第二輪修復（2026-04-14，待跑）

已套用修正：
1. **lambda_smooth scale-aware**（train.py:130）
   - `effective = base × sqrt(lambda_ref_iqr / target_scaler.scale_[0])`
   - `--lambda_ref_iqr` CLI 參數，預設 `104.0`（來自 urllc train-target IQR baseline）
   - `--disable_scale_aware_lambda` 可完全關閉 scale-aware 調整
   - ⚠️ 方向性尚待驗證：若 scaled space 下 main/smooth 比值已接近，則應改回固定 lambda
2. **Drop 常數特徵** → 13 維（DataProcessor.py:22）
3. **預設 lr=1e-4 + scheduler=cosine**（train.py:69, 71）
4. **patience=0 代表關閉 early stopping**（train.py:677）
   - `--patience` 預設 50；設為 `0` 時 early stopping 邏輯完全跳過，跑滿 `--epochs`

## 第二輪結果 (2026-04-14)

| Slice | best_epoch | Test MAE | Test R² |
|-------|-----------|----------|---------|
| embb  | 1  | 52.66 | **0.9672** |
| mmtc  | 27 | 29.43 | **0.1450** |
| urllc | 15 | 64.21 | **0.2193** |

**關鍵觀察：**
- embb 第 1 epoch 就收斂，cosine 後續 499 epoch 無效
- mmtc/urllc 在 ~15-27 epoch 後 val_main 單調爬升 → overfit
- 三 slice 全部 **mean regression**（低值過高估、高值過低估）
  - mmtc high-tail (true≥138)：`neg_ratio=0.981`, mean_resid≈-85
  - urllc high-tail (true≥321)：`neg_ratio=0.993`, mean_resid≈-215
  - embb low-tail mean_resid≈+17.8；high-tail mean_resid≈-32.0, neg_ratio=0.925

## 第三輪改動（2026-04-15，待跑）

### Weighted MSE Loss（train.py）
對 target 尾端樣本加權，直接對抗 mean regression：

```
weight(y) = 1 + α × clamp((y - q_low) / (q_high - q_low), 0, 1)
```
- `y ≤ q_low` → weight=1
- `y ≥ q_high` → weight=1+α（預設 α=4，即上限 5）
- q_low/q_high 在 **scaled space** 由 train_loader 一次 pass 算出，per-slice 自適應
- 實作：`WeightedMSELoss` (train.py:177) + `collect_train_target_quantiles` (train.py:196)

### 新 CLI 參數
| 參數 | 預設 | 說明 |
|------|------|------|
| `--loss_type` | `mse` | `mse` / `weighted_mse` |
| `--weight_alpha` | `4.0` | 最大額外權重，必須 ≥ 0 |
| `--weight_quantile` | `0.9` | ramp 起點分位數 |
| `--weight_upper_quantile` | `0.99` | ramp 飽和分位數 |

argparse 會驗證：`0 ≤ q_low < q_high ≤ 1` 且 `alpha ≥ 0`。

### 建議試跑命令

```bash
# mmtc / urllc：短 epoch + dropout↑ + 單向 LSTM + 低 smooth + weighted loss
python train.py --train_dirs tr0-26 --test_dirs tr27 --slice_type mmtc \
  --epochs 80 --patience 20 --batch_size 1024 \
  --learning_rate 1e-4 --scheduler cosine \
  --rnn_type lstm --lstm_dropout 0.35 \
  --lambda_smooth 0.03 --disable_scale_aware_lambda \
  --loss_type weighted_mse --weight_alpha 4

# embb：已收斂，只調細節
python train.py ... --slice_type embb --epochs 30 --patience 10 \
  --learning_rate 5e-5 --loss_type weighted_mse --weight_alpha 2
```

## 第三輪結果 (2026-04-15)

**Config 差異**：
- embb: `loss_type=mse`（未改）, lr=5e-5, cosine, epochs=80, patience=0
- mmtc/urllc: `loss_type=weighted_mse` α=4, scheduler=none, lstm, dropout=0.3, lambda_smooth=0.03, disable_scale_aware_lambda, patience=0

| Slice | best_epoch | Test MAE | Test R² | vs Round 2 |
|-------|-----------|----------|---------|------------|
| embb  | 2  | 52.39 | **0.9673** | 持平 |
| mmtc  | 34 | 32.53 | **0.1265** | R² -13% |
| urllc | 49 | 78.48 | **0.0873** | R² -60% |

**Weighted MSE 細部分析（非全面失敗）**：
- weighted_mse 改善了高流量低估，但惡化了低流量高估：
  - mmtc top10% bias: -85→-65（改善）, bottom10% bias: +17→+27（惡化）
  - urllc top10% bias: -215→-148（改善）, bottom10% bias: +32→+51（惡化）
- 整體 R²/MAE 變差的主因是 bulk 區間被拉壞
- train_loss >> val_loss 是可預期現象（train/val target 分布不同 + 加權放大差距），非崩潰證據

**配置問題**：
- α=4 過重 + scheduler=none + patience=0（無 early stop）→ 高噪聲梯度下 bulk 偏差被推向整體偏正
- embb 本輪未使用 weighted_mse，等於只是短 epoch 重跑

**結論**：
- embb 已穩定，維持 mse
- mmtc/urllc 本輪配置不佳；weighted loss 概念有效（改善 tail），但需降 α + 加 scheduler + early stop
- 特徵資訊不足是中長期瓶頸

## 尚未解決
- Weighted MSE 方向正確（改善 tail bias）但 α=4 過重，需調降 + 搭配 scheduler/early stop
- 特徵資訊不足是 mmtc/urllc 的中長期瓶頸（13 維 input 無法區分 spike context）
- 下一步：model8_1 ablation（移除 Transformer，純 LSTM）驗證 Transformer 是否對 mmtc/urllc 有幫助
