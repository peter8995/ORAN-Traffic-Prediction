# model_detect_peak: Spike Predictability Baselines

Independent spike detection baselines (Autoencoder, Isolation Forest) to evaluate spike predictability without the full ViT+LSTM pipeline.

## Architecture

- **`DataProcessor.py`** — Shared data loading. Reuses model5/6 spike label formula (`τ = Q_0.9 × (1 + ψ × ξ_m/ξ_max)`). Adds 6 rolling temporal features on raw data before scaling, then creates sliding windows like model6.
- **`autoencoder_detect.py`** — PyTorch linear autoencoder. Flattened sliding window `(seq_len × 17)` → encoder (128→64→latent) → decoder → reconstruct. High reconstruction error = spike.
- **`isolation_forest_detect.py`** — Sklearn IsolationForest on flattened sliding window `(seq_len × 17)`. Includes threshold sweep for best F1.

## Input Features (17 per timestep)

11 original features + 6 rolling temporal features:
- `target_rolling_mean_m` / `target_rolling_std_m` / `target_rolling_max_m` — rolling stats over m=120 window (~30s)
- `target_rolling_mean_L` / `target_rolling_q90_L` — rolling stats over L=1200 window (~5min)
- `target_diff` — first-order difference of target

Rolling features bridge the gap between short sliding window (seq_len=15, ~3.75s) and the long-range spike definition (L=1200, ~5min).

## Training Commands

```bash
cd model_detect_peak

# Autoencoder
python autoencoder_detect.py --train_dirs tr0-4 tr10 --test_dirs tr5-6 \
  --slice_type embb --epochs 100 --sequence_length 15

# Isolation Forest
python isolation_forest_detect.py --train_dirs tr0-4 tr10 --test_dirs tr5-6 \
  --slice_type embb --sequence_length 15
```

## Key Arguments

| Arg | AE Default | IF Default | Description |
|-----|-----------|-----------|-------------|
| `--sequence_length` | 15 | 15 | Sliding window length |
| `--threshold_percentile` | 88 | — | Percentile of train recon error for spike threshold |
| `--contamination` | — | 0.11 | Expected anomaly proportion |
| `--latent_dim` | 16 | — | Autoencoder bottleneck size |
| `--n_estimators` | — | 200 | Number of IF trees |

## Initial Results (without rolling features, without sliding window)

| Metric | Autoencoder | Isolation Forest |
|--------|------------|-----------------|
| F1 | 0.176 | 0.188 |
| Precision | 0.155 | 0.183 |
| Recall | 0.204 | 0.192 |
| Accuracy | 0.769 | 0.798 |

## Results with Rolling Features + Sliding Window (2026-03-23)

| Metric | Autoencoder | Isolation Forest |
|--------|------------|-----------------|
| F1 | 0.169 | 0.207 |
| Precision | 0.137 | 0.167 |
| Recall | 0.219 | 0.271 |
| Accuracy | 0.737 | 0.745 |
| FP Rate | 0.191 | 0.189 |
| FN Rate | 0.781 | 0.729 |

IF threshold sweep best: P90 → P=0.170, R=0.249, F1=0.202. Rolling features + sliding window 幾乎沒有改善。

## 診斷結論：Spike Label 定義缺陷 + 特徵不足 (2026-03-23)

**Spike Label 問題：**
- 公式 `τ = Q_0.9(last 1200) × (1 + 0.05 × ξ_m/ξ_max)` 中 ψ=0.05 幾乎無作用
- Threshold 實質上就是 local Q90 → 任何 stationary window 永遠有 ~10% spike
- Test set 12.21% spike rate 印證此問題 — 標的不是「異常突刺」而是「local 前 10%」
- Unsupervised 模型無法區分正常高值 vs 真正異常

**修復方案（三選一，待 Peter 決定）：**

### 方案 A: Global Threshold
```python
global_mean = target_all_train.mean()
global_std = target_all_train.std()
tau = global_mean + k * global_std  # k=2 or 3
```
- 只有偏離全局分佈的值才算 spike，spike rate 預期降到 2-5%

### 方案 B: Global Threshold + 突變條件
```python
diff = target.diff().abs()
spike = (target > global_threshold) & (diff > diff_threshold)
```
- 值高且來得快才算 spike，更符合「突刺」語義

### 方案 C: 保留 Rolling 架構但調高 ψ
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

## Plan B 實作與首次結果 (2026-03-23)

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
