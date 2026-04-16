# model8_1: Transformer Ablation (2026-04-16)

移除 model8 的 TransformerEncoder，其餘完全相同。純 ablation 實驗，比較有無 Transformer 的差異。

## 核心設計

- Forward flow: `input → Linear(13→d_model) → +pos_embedding → LSTM → last_step → FF head`
- 和 model8 唯一差異：移除 3 層 TransformerEncoder
- 保留 input_projection（13→128）+ pos_embedding + LSTM + head
- DataProcessor.py / train.py 完全不改

## 檔案結構

- `model8_1/model.py` — TrafficModel without Transformer
- `model8_1/DataProcessor.py` — 複製自 model8（完全相同）
- `model8_1/train.py` — 複製自 model8（完全相同）

## 第一跑設定（純 ablation，對照 model8 round 2）

```bash
# 三個 slice 各跑一次，設定和 model8 round 2 完全相同
python train.py --train_dirs tr0-26 --test_dirs tr27 --slice_type embb \
  --epochs 500 --patience 0 --batch_size 1024 \
  --learning_rate 1e-4 --scheduler cosine \
  --rnn_type bilstm --lambda_smooth 0.05

python train.py --train_dirs tr0-26 --test_dirs tr27 --slice_type mmtc \
  --epochs 500 --patience 0 --batch_size 1024 \
  --learning_rate 1e-4 --scheduler cosine \
  --rnn_type bilstm --lambda_smooth 0.05

python train.py --train_dirs tr0-26 --test_dirs tr27 --slice_type urllc \
  --epochs 500 --patience 0 --batch_size 1024 \
  --learning_rate 1e-4 --scheduler cosine \
  --rnn_type bilstm --lambda_smooth 0.05
```

## 對照基準（model8 round 2 結果）

| Slice | best_epoch | Test MAE | Test R² |
|-------|-----------|----------|---------|
| embb  | 1  | 52.66 | 0.9672 |
| mmtc  | 27 | 29.43 | 0.1450 |
| urllc | 15 | 64.21 | 0.2193 |

## Decision Log

| # | Decision | Alternatives | Reason |
|---|----------|-------------|--------|
| 1 | 保留 input_projection + pos_embedding | 也拿掉，LSTM 吃 raw 13 維 | 乾淨 ablation，只移除一個變因 |
| 2 | 目標：比較有無 Transformer 差異 | 設定 R² 硬門檻 | ablation 重點在歸因 |
| 3 | 獨立目錄 model8_1/ | 在 model8 加 --no_vit flag | 避免污染 model8 code |
| 4 | 先跑純 ablation，再根據結果調參 | 同時調參 | 控制變因 |
| 5 | 用 round 2 完整設定 (epochs=500) | 縮短 epochs | 嚴格對照 |
