#!/usr/bin/env bash
set -euo pipefail

LRS=("1e-5" "3e-5" "1e-4")
BATCH_SIZES=(512 256)

BASE_ARGS=(
  --train_dirs tr0-9 tr11-14 tr16-27
  --val_dirs tr10
  --test_dirs tr15
  --epochs 100
  --slice_type mmtc
  --rnn_type bilstm
  --patience 20
  --sequence_length 30
  --lambda_smooth 0
)

echo "[INFO] Starting sweep in $(pwd)"
echo "[INFO] Batch sizes: ${BATCH_SIZES[*]}"
echo "[INFO] Learning rates: ${LRS[*]}"

for bs in "${BATCH_SIZES[@]}"; do
  for lr in "${LRS[@]}"; do
    echo
    echo "============================================================"
    echo "[RUN] batch_size=${bs} | learning_rate=${lr}"
    echo "============================================================"
    python train.py "${BASE_ARGS[@]}" --batch_size "${bs}" --learning_rate "${lr}"
  done
done

echo
echo "[INFO] Sweep completed."
