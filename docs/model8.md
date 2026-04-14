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

## 尚未解決
- `T_max=args.epochs` 若 early stop 較早觸發，cosine 衰減幾乎不生效 → 可能需縮小 T_max
- 第二輪三 slice 結果待跑驗證
