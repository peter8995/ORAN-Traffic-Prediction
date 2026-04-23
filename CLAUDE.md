# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

O-RAN traffic prediction using ViT / BiLSTM / GAT / FFT-Transformer architectures for 5G network slice types (eMBB, mMTC, uRLLC). Eleven model directories exist (model1–model7, model7_1, model8, model8_1, model9).

**Current focus**: model9（Multi-Domain：BiLSTM + GAT + FFT-dual-Transformer + Chebyshev anomaly），code 實作完成、DESIGN 與 code 已 1:1 對齊、review 通過，待跑三 slice 實驗。model8_1（Transformer ablation）已完成三 slice 比較。

## 詳細文件 (docs/)

| 主題 | 路徑 |
|------|------|
| Model5/6/7 spike-aware 歷史、Tiny ViT 細節、model6 改進計畫、model7 三輪實驗結果 | `docs/model5_6_7_history.md` |
| model7_1 regression-only ablation 完整結果（證實 spike head 無幫助；mMTC/uRLLC 瓶頸根因分析） | `docs/model7_1_ablation.md` |
| model_detect_peak — Autoencoder / Isolation Forest baselines、spike label 診斷、方案 B 結果 | `docs/model_detect_peak.md` |
| model8 regression-first 重設計進度、三輪實驗結果與 weighted MSE 分析 | `docs/model8.md` |
| model8 完整設計與 Decision Log | `model8/DESIGN.md` |
| model8_1 Transformer ablation 設計與 decision log | `docs/model8_1.md` |
| model9 Multi-Domain 完整設計 + 21 條 Decision Log + 全部 CLI args（含 GAT head merge ablation） | `model9/DESIGN.md` |
| model9 brainstorming session 對話紀錄 (Q1-Q13 + arg 三輪談判) | `model9/brainstorming.md` |

## Architecture（models 1-7 共通）

Each model directory contains three files:

- **`classicModel.py`** — `vitModel`/`TinyViT` → `lstmModel`/`biLSTMModel` → `TrafficModel` (attention + prediction heads)
- **`DataProcessor.py`** — CSV loading, MinMaxScaler, sliding window → `TrafficDataset`
- **`train.py`** — Training loop, loss, evaluation, plotting to `results/`

Forward pass (model1-6): `(B, seq_len, 11)` → ViT `(B, seq_len+1, 768)` → LSTM `(B, seq_len+1, 128)` → MHA → last token → FF → pred

Forward pass (model7/7_1): 同上但 ViT 換成 TinyViT，輸出 `(B, seq_len+1, d_model)`

**model8** 採新架構（RobustScaler、chunk sampler、per-CSV 切分），檔案結構為 `DataProcessor.py` / `model.py` / `train.py`。細節見 `docs/model8.md` 與 `model8/DESIGN.md`。

**model9** 採三 branch multi-domain 架構（BiLSTM + GAT + FFT-dual-Transformer → fusion → mean pool over N nodes → scalar）。檔案結構為 `DataProcessor.py` / `model.py` / `anomaly.py` / `train.py`（自寫 GATLayer，不引入 `torch_geometric`）。細節見 `model9/DESIGN.md` 與 `model9/brainstorming.md`。

## Model Differences

| Model | Key Change |
|-------|-----------|
| model1 | Unified multi-task: 3 slices + global traffic jointly, MHA fusion |
| model2 | Independent per-slice, no cross-slice attention |
| model3 | Slice-specific losses: quantile (q=0.7) for mMTC, MSE for others |
| model4 | Multi-directory Colosseum dataset via argparse |
| model5 | **SpikeAwareLSTM**: dual-head (regression + spike), BCEWithLogits, grad clip |
| model6 | **Anti-overfitting**: ViT freeze, Dropout, weight decay, early stopping, pos_weight clamp |
| model7 | **Tiny ViT**: custom `nn.TransformerEncoder` (~1M params) replaces vit_b_16 (86M) |
| model7_1 | **Regression-only ablation**: model7 minus spike_head. 證實 spike head 無助於 regression |
| model8 | **Regression-first redesign**: RobustScaler, chunk shuffle sampler, smooth loss, per-CSV val split |
| model8_1 | **Transformer ablation**: model8 minus TransformerEncoder, pure LSTM baseline for comparison |
| model9 | **Multi-Domain**: BiLSTM + GAT + FFT-dual-Transformer (feature-as-node, N=16/15), Chebyshev anomaly flag, same target (`sum_granted_prbs`) as model8 |

## Training Commands

### Models 1-3
Hardcoded parameters; run `python train.py` directly.

### Models 4-7 (pattern)
```bash
python train.py --train_dirs tr0-4 tr10 --test_dirs tr5-6 \
  --slice_type embb --epochs 500 --batch_size 1024 \
  --learning_rate 1e-6 --sequence_length 15 --val_split 0.2
```

Per-model additions:
- **model5**: `--lambda_detect 0.5`
- **model6**: `--weight_decay 1e-4 --patience 30 --max_pos_weight 7.0`
- **model7/7_1**: 加上 Tiny ViT 超參 `--d_model 128 --n_heads 4 --n_layers 3 --dim_feedforward 512 --vit_dropout 0.2`

Directory range: `tr0-4` → `tr0..tr4`.

### model8
```bash
cd model8
# 推薦：顯式 leave-trial-out val（2026-04-21+）
python train.py --train_dirs tr0-24 --val_dirs tr25 tr26 --test_dirs tr27 \
  --slice_type mmtc \
  --epochs 500 --patience 50 --batch_size 1024 \
  --learning_rate 1e-4 --scheduler cosine \
  --rnn_type bilstm --lambda_smooth 0.1

# 舊行為：自動隨機切 CSV 20% 當 val（fallback）
python train.py --train_dirs tr0-26 --test_dirs tr27 --slice_type mmtc \
  --val_split 0.2 ...
```
- `--val_dirs` 指定時 `--val_split` 被忽略
- `train_dirs` / `val_dirs` / `test_dirs` 三組交集會 `ValueError`（必須分離）
- 細節見 `docs/model8.md`

### model8_1
```bash
cd model8_1
python train.py --train_dirs tr0-24 --val_dirs tr25 tr26 --test_dirs tr27 \
  --slice_type mmtc \
  --epochs 500 --patience 0 --batch_size 1024 \
  --learning_rate 1e-4 --scheduler cosine \
  --rnn_type bilstm --lambda_smooth 0.05
```
和 model8 完全相同設定，唯一差異是 model.py 移除了 TransformerEncoder（DataProcessor 同步支援 `--val_dirs`）。細節見 `docs/model8_1.md`。

### model9（code 已實作）
目標：用論文 *A Prediction-Based Anomaly Detection Method for Traffic Flow with Multi-Domain Feature Extraction* pseudo code 的三 branch 架構（BiLSTM + GAT + FFT-dual-Transformer），預測目標沿用 model8（`sum_granted_prbs`、per-slice、horizon=1、T=15）。採 feature-as-node 對應（N=16 primary 含 `sum_granted_prbs` 歷史 / N=15 fairness ablation）。

Primary 訓練命令：
```bash
cd model9
python train.py --train_dirs tr0-24 --val_dirs tr25 tr26 --test_dirs tr27 \
  --slice_type mmtc \
  --epochs 500 --patience 50 --batch_size 1024 \
  --learning_rate 1e-4 --scheduler cosine \
  --rnn_type bilstm --lambda_smooth 0.0 \
  --include_target_history True --readout mean \
  --adj_type binary_selfloop \
  --gat_head_merge mean --gat_final_head_merge mean \
  --chebyshev_k 3.0 --anomaly_error_mode both
```

Paper-style GAT ablation（中間層 concat heads、最後層 mean）：
```bash
python train.py ... \
  --gat_head_merge concat --gat_final_head_merge mean
```
（`concat` 時 `gat_hidden % gat_heads == 0` 與 `hidden_dim % gat_heads == 0` 由 argparse 檢查）

Model8 的 `--d_model / --n_heads / --n_layers / --dim_feedforward / --vit_dropout` 在 model9 **移除**（無 temporal ViT 對應），改用 `--fft_*` 系列 args 控制 FFT Transformer，另加 `--gat_*` 系列 args（含 `--gat_head_merge` / `--gat_final_head_merge`）。完整 args 與 21 條 Decision Log 見 `model9/DESIGN.md`。

## Datasets

- **TRACTOR**: `Dataset/Tractor/Trial{5,7}/` — used by models 1-3
- **Colosseum**: `Dataset/colosseum-oran-coloran-dataset/tr{N}/exp{}/bs{}/{embb|mtc|urllc}/*metrics.csv` — used by models 4-8

11 input features (model1-7)：dl_buffer, ul_buffer, tx/rx brate, ul_sinr, dl/ul_mcs, phr, tx/rx_pkts, ul_rssi
15 input features (model8/8_1 current, 2026-04-22+)：扣掉 `ul_rssi`，加上 `dl_cqi`, `rx_errors_ul`, `dl_n_samples`, `ul_n_samples`, `ul_buffer`, `sum_requested_prbs`（前期 2026-04-14 曾降至 13 維，後續為補 mmtc/urllc 資訊加回 `ul_buffer` + `sum_requested_prbs`）
16 input features (model9 primary)：上述 15 維 + `sum_granted_prbs` 歷史當第 16 個 feature node；N=15 ablation 透過 `--include_target_history False` 切換

Target：`sum_requested_prbs`（model1-7_1）/ `sum_granted_prbs`（model5, model8/8_1, model9）

Note：`mmtc` slice type → 資料夾名稱 `mtc`。

## Dataset Utilities

- `clean_dataset.py` — Flattens nested CSV structure (dry-run mode 可用)
- `group_by_slice.py` — Reorganizes metrics.csv by slice type via IMSI UE mapping

## Dependencies

PyTorch, torchvision, pandas, numpy, scikit-learn (MinMaxScaler, RobustScaler, IsolationForest), matplotlib. No requirements.txt exists.
