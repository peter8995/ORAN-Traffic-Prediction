# Model9

Multi-domain traffic prediction model with:

- Per-node shared BiLSTM temporal branch
- GAT spatial branch on feature-as-node graph
- FFT magnitude/phase dual-transformer frequency branch
- Chebyshev residual thresholding for anomaly flags

## Files

- `DataProcessor.py`: model8-compatible data pipeline with `include_target_history` and adjacency building
- `model.py`: multi-branch predictor
- `anomaly.py`: Chebyshev threshold calibration and overlay plot helpers
- `train.py`: training, evaluation, plotting, and anomaly artifact export

## Example

```bash
python model9/train.py \
  --slice_type embb \
  --train_dirs tr0-24 \
  --val_dirs tr25 tr26 \
  --test_dirs tr27-29 \
  --loss_type mse \
  --lambda_smooth 0.0 \
  --include_target_history true \
  --adj_type binary_selfloop \
  --readout mean
```

## Fairness Ablation

Disable target history to match model8 input width:

```bash
python model9/train.py \
  --slice_type embb \
  --train_dirs tr0-24 \
  --val_dirs tr25 tr26 \
  --test_dirs tr27-29 \
  --include_target_history false
```

## GAT Head Merge Ablation

Keep the current design with mean-merged heads:

```bash
python model9/train.py \
  --slice_type embb \
  --train_dirs tr0-24 \
  --val_dirs tr25 tr26 \
  --test_dirs tr27-29 \
  --gat_head_merge mean \
  --gat_final_head_merge mean
```

Try a more standard GAT-style intermediate concat:

```bash
python model9/train.py \
  --slice_type embb \
  --train_dirs tr0-24 \
  --val_dirs tr25 tr26 \
  --test_dirs tr27-29 \
  --gat_head_merge concat \
  --gat_final_head_merge mean
```
