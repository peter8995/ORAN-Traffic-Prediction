# model7_1: Regression-Only Ablation

與 model7 架構完全相同，移除 spike_head，用於驗證 spike head 是否改善 regression。
所有 slice 均使用 TinyViT + BiLSTM。eMBB 額外跑了一版 LSTM 作比較。

## 關鍵差異
- `classicModel.py`：無 `spike_head`，`forward()` 只回傳 `pred`，所有 slice 使用 `biLSTMModel`
- `train.py`：loss 純 MSE，無 `lambda_detect`/`max_pos_weight` 參數
- `performance_matrix.txt` 新增：
  - Spike timesteps MAE/RMSE/R²
  - Normal timesteps MAE/RMSE/R²
  - MAE ratio（spike/normal）— 越高代表 spike 時段誤差越大
- Prediction plot：黃色背景標出 ground truth spike 區域

## Training Command
```bash
cd model7_1
python train.py --train_dirs tr0-26 --test_dirs tr27 --slice_type embb \
  --epochs 500 --patience 25 --vit_dropout 0.2 --weight_decay 0.0
```

## Ablation 結果 (2026-03-26)

結果存放：`model7_1/results/`

### 結論一：Spike head 對 regression 無幫助

Model7 R3（有 spike head）vs Model7_1（純 regression），相同超參數（vit_dropout=0.2, wd=0）：

| Metric | eMBB M7 | eMBB M7_1 | mMTC M7 | mMTC M7_1 |
|--------|---------|-----------|---------|-----------|
| MAE | 0.0046 | **0.0043** | 0.0024 | **0.0023** |
| RMSE | 0.0060 | **0.0054** | 0.0036 | 0.0036 |
| R² | 0.9739 | **0.9780** | 0.3894 | **0.3958** |

Model7_1 在所有 slice 上 regression 微幅優於 model7。spike head 未能讓 backbone 對突變更敏感，反而分散學習能力。

### 結論二：LSTM vs BiLSTM（eMBB）

| Metric | LSTM | BiLSTM |
|--------|------|--------|
| R²(all) | 0.9789 | 0.9780 |
| Spike MAE | 0.0039 | **0.0034** |
| Spike R² | 0.9678 | **0.9743** |
| MAE ratio | 0.8874 | **0.7527** |

整體持平，BiLSTM 在 spike 時段稍優（雙向特性對突變時序捕捉較好）。

### 結論三：mMTC / uRLLC regression 根本性困難

| | eMBB | mMTC | uRLLC |
|--|------|------|-------|
| R²(all) | **0.9780** | 0.3958 | 0.4140 |
| Spike R² | **0.9743** | -0.6150 | -0.3230 |
| Normal R² | 0.9798 | 0.4966 | 0.5270 |
| MAE ratio | **0.75** | 3.38 | 3.68 |
| Spike Rate | 11.64% | 12.68% | 13.05% |

- mMTC/uRLLC 連 normal 時段 R² 只有 ~0.5，spike 時段 R² 為負（比猜均值差）
- uRLLC 首次有結果，表現與 mMTC 同等困難

### 結論四：Prediction Plot 與 Loss Curve 視覺分析

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

### 各實驗詳細結果

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
