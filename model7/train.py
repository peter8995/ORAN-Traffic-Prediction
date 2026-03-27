import argparse
import math
import os
from datetime import datetime

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, random_split

from classicModel import *
from DataProcessor import *

SEGMENT_SIZE = 10000

def quantile_loss(preds, targets, quantile: float):
    errors = targets - preds
    loss = torch.max((quantile - 1) * errors, quantile * errors)
    return torch.abs(loss).mean()

def parse_directory_args(args_list, base_path):
    """
    Takes a list of strings like ['tr0-4', 'tr11']
    and returns full paths: ['/base/tr0', '/base/tr1', ... '/base/tr4', '/base/tr11']
    """
    parsed_dirs = []
    for item in args_list:
        if '-' in item:
            # e.g item = "tr0-4"
            prefix = ''.join([c for c in item.split('-')[0] if not c.isdigit()]) # gets "tr"
            start = int(''.join([c for c in item.split('-')[0] if c.isdigit()])) # gets 0
            end = int(''.join([c for c in item.split('-')[1] if c.isdigit()]))   # gets 4
            for i in range(start, end + 1):
                parsed_dirs.append(os.path.join(base_path, f"{prefix}{i}"))
        else:
            # e.g item = "tr10"
            parsed_dirs.append(os.path.join(base_path, item))
    return parsed_dirs

def create_run_directory(slice_type: str, base_dir: str = './results') -> str:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(base_dir, f'{slice_type}_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    print(f"Created run directory: {run_dir}")
    return run_dir


def plot_loss_curve(run_dir, slice_type, train_losses, val_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.title(f'{slice_type.upper()} Slice Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(run_dir, f'{slice_type}_loss_curve.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Loss curve saved to: {save_path}")


def plot_segmented_prediction(run_dir, slice_type, target, prediction):
    total = len(target)
    n_segments = math.ceil(total / SEGMENT_SIZE)

    fig, axes = plt.subplots(n_segments, 1, figsize=(12, 4 * n_segments))
    if n_segments == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        start = i * SEGMENT_SIZE
        end = min(start + SEGMENT_SIZE, total)
        x_range = range(start, end)
        ax.plot(x_range, target[start:end], label='Actual', color='blue', alpha=0.6)
        ax.plot(x_range, prediction[start:end], label='Predicted', color='orange', alpha=0.8)
        ax.set_ylabel('Traffic (Granted PRBs)')
        ax.set_title(f'Steps {start}–{end - 1}')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time Steps')
    fig.suptitle(f'{slice_type.upper()} Prediction vs Target ({total} steps)', fontsize=14)
    plt.tight_layout()

    save_path = os.path.join(run_dir, f'{slice_type}_prediction.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Prediction plot saved to: {save_path}")


def plot_segmented_spike(run_dir, slice_type, target, prediction, spike_gt, spike_pred):
    total = len(target)
    n_segments = math.ceil(total / SEGMENT_SIZE)

    height_ratios = [3, 1] * n_segments
    fig, axes = plt.subplots(n_segments * 2, 1, figsize=(12, 5 * n_segments),
                             gridspec_kw={'height_ratios': height_ratios, 'hspace': 0.15})
    if n_segments == 1:
        axes = list(axes)

    for i in range(n_segments):
        start = i * SEGMENT_SIZE
        end = min(start + SEGMENT_SIZE, total)
        x_range = np.arange(start, end)

        ax_demand = axes[i * 2]
        ax_spike = axes[i * 2 + 1]

        ax_demand.plot(x_range, target[start:end], label='Actual RAN Demand',
                       color='blue', alpha=0.7, linewidth=0.8)
        ax_demand.plot(x_range, prediction[start:end], label='Predicted RAN Demand',
                       color='red', linestyle='--', alpha=0.7, linewidth=0.8)
        ax_demand.set_ylabel('Traffic (Granted PRBs)')
        ax_demand.set_title(f'Steps {start}–{end - 1}')
        ax_demand.legend(loc='upper right', fontsize=8)
        ax_demand.grid(True, alpha=0.3)

        gt_seg = spike_gt[start:end].flatten().astype(int)
        pred_seg = spike_pred[start:end].flatten().astype(int)

        ax_spike.fill_between(x_range, 0, 1, where=(gt_seg == 1),
                              color='green', alpha=0.3, label='Ground Truth Spike', step='mid')
        ax_spike.fill_between(x_range, 0, 1, where=(pred_seg == 1),
                              color='red', alpha=0.4, label='Predicted Spike', step='mid')
        ax_spike.set_yticks([0, 1])
        ax_spike.set_yticklabels(['Normal', 'Peak'])
        ax_spike.legend(loc='upper right', fontsize=8)
        ax_spike.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time Steps')
    fig.suptitle(f'{slice_type.upper()} Resource Demand & Spike Detection ({total} steps)', fontsize=14)
    plt.tight_layout()

    save_path = os.path.join(run_dir, f'{slice_type}_spike_detection.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Spike detection plot saved to: {save_path}")


def save_performance_matrix(run_dir, args, target, prediction, spike_gt, spike_pred):
    # Regression metrics
    mae = np.mean(np.abs(target - prediction))
    rmse = np.sqrt(np.mean((target - prediction) ** 2))
    ss_res = np.sum((target - prediction) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Spike metrics
    tp = np.sum((spike_pred == 1) & (spike_gt == 1))
    fp = np.sum((spike_pred == 1) & (spike_gt == 0))
    fn = np.sum((spike_pred == 0) & (spike_gt == 1))
    tn = np.sum((spike_pred == 0) & (spike_gt == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    spike_rate = np.mean(spike_gt) * 100

    filepath = os.path.join(run_dir, 'performance_matrix.txt')
    with open(filepath, 'w') as f:
        f.write('=' * 40 + '\n')
        f.write('Training Parameters\n')
        f.write('=' * 40 + '\n')
        for key, value in vars(args).items():
            f.write(f'{key}: {value}\n')

        f.write('\n' + '=' * 40 + '\n')
        f.write('Regression Metrics\n')
        f.write('=' * 40 + '\n')
        f.write(f'MAE:    {mae:.4f}\n')
        f.write(f'RMSE:   {rmse:.4f}\n')
        f.write(f'R²:     {r2:.4f}\n')

        f.write('\n' + '=' * 40 + '\n')
        f.write('Spike Detection Metrics\n')
        f.write('=' * 40 + '\n')
        f.write(f'Accuracy:   {accuracy:.4f}\n')
        f.write(f'Precision:  {precision:.4f}\n')
        f.write(f'Recall:     {recall:.4f}\n')
        f.write(f'F1 Score:   {f1:.4f}\n')
        f.write(f'FP Rate:    {fp_rate:.4f}\n')
        f.write(f'FN Rate:    {fn_rate:.4f}\n')
        f.write(f'Spike Rate: {spike_rate:.2f}%\n')

    print(f"Performance matrix saved to: {filepath}")

    # Also print to console
    print(f"\n{'=' * 40}")
    print("Spike Detection Performance Metrics:")
    print('=' * 40)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"FP Rate:   {fp_rate:.4f}")
    print(f"FN Rate:   {fn_rate:.4f}")
    print(f"Spike Rate: {spike_rate:.2f}% of test samples")
    print(f"\nRegression Metrics:")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")
    print('=' * 40 + '\n')


def main():
    parser = argparse.ArgumentParser(description="Train Traffic Model (Tiny ViT) across multiple directories.")

    # Paths Options
    parser.add_argument('--base_path', type=str, default='/home/cislab301b/peter/ORAN-Traffic-Prediction/Dataset/colosseum-oran-coloran-dataset', help="Base directory where tr* folders exist.")
    parser.add_argument('--train_dirs', type=str, nargs='+', required=True, help="List of folder ranges or names (e.g. --train_dirs tr0-4 tr10)")
    parser.add_argument('--test_dirs', type=str, nargs='+', required=True, help="List of folder ranges or names (e.g. --test_dirs tr5-6)")

    # Tunables
    parser.add_argument('--slice_type', type=str, default='embb', choices=['embb', 'mmtc', 'urllc'], help="Type of the slice (embb, mmtc, urllc)")
    parser.add_argument('--epochs', type=int, default=500, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=1024, help="Batch size")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--sequence_length', type=int, default=15, help="Sequence length for the LSTM")
    parser.add_argument('--val_split', type=float, default=0.2, help="Percentage of training data to use for validation")
    parser.add_argument('--lambda_detect', type=float, default=1.0, help="Weight for the spike detection BCE loss")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="L2 regularization weight decay")
    parser.add_argument('--patience', type=int, default=30, help="Early stopping patience (0 to disable)")
    parser.add_argument('--max_pos_weight', type=float, default=7.0, help="Clamp upper bound for spike BCE pos_weight")

    # Tiny ViT hyperparameters
    parser.add_argument('--d_model', type=int, default=128, help="Tiny ViT embedding dimension")
    parser.add_argument('--n_heads', type=int, default=4, help="Number of attention heads in Tiny ViT")
    parser.add_argument('--n_layers', type=int, default=3, help="Number of transformer encoder layers in Tiny ViT")
    parser.add_argument('--dim_feedforward', type=int, default=512, help="Feedforward dimension in Tiny ViT encoder layers")
    parser.add_argument('--vit_dropout', type=float, default=0.1, help="Dropout rate in Tiny ViT encoder layers")

    args = parser.parse_args()

    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_epochs = args.epochs
    sequenceLength = args.sequence_length
    slice_type = args.slice_type
    lambda_detect = args.lambda_detect

    # Parse String Arrays into Full Absolute Paths
    train_paths = parse_directory_args(args.train_dirs, args.base_path)
    test_paths = parse_directory_args(args.test_dirs, args.base_path)

    # Dataset
    processor = DataProcessor(sequenceLength)

    print(f"\n{'='*20} Analzing Training Directories {'='*20}")
    print(f"Base Path: {args.base_path}")
    print(f"Parsed Train Paths: {train_paths}")

    train_files = processor.accumulate_files(train_paths, slice_type)
    if not train_files:
        raise ValueError("No metrics.csv files found in the provided --train_dirs!")

    # Compute global xi_max across all training files BEFORE building any Dataset
    # so spike thresholds are consistent (not per-file relative)
    processor.fit_spike_params(train_files)

    print(f"Discovered {len(train_files)} files. Building global MinMaxScaler from ALL training files:")

    # Global fit: scan all training files to establish consistent scaling
    all_X_raw, all_Y_raw = [], []
    for f in train_files:
        df = processor.load_and_clean(f)
        if len(df) <= sequenceLength:
            continue
        all_X_raw.append(df[processor.features].values)
        all_Y_raw.append(df[[processor.target]].values)

    if not all_X_raw:
        raise ValueError("No valid training files found after filtering by sequence length!")

    scalerX = MinMaxScaler()
    scalerY = MinMaxScaler()
    scalerX.fit(np.concatenate(all_X_raw, axis=0))
    scalerY.fit(np.concatenate(all_Y_raw, axis=0))
    del all_X_raw, all_Y_raw  # Free memory after fitting

    print(f"Global scalers fitted on {len(train_files)} files.")

    # Load combined dataset across all train_dirs
    print(f"\nBuilding comprehensive Training & Validation Dataset...")
    full_train_ds = TrafficDataset(train_paths, processor, scalerX, scalerY, slice_type)

    # Split dataset into Training and Validation
    val_len = int(len(full_train_ds) * args.val_split)
    train_len = len(full_train_ds) - val_len

    print(f"Found {len(full_train_ds)} valid chunked windows across all train files.")
    print(f"Splitting -> Train Size: {train_len} | Validation Size: {val_len}")

    train_ds, val_ds = random_split(full_train_ds, [train_len, val_len], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Building comprehensive Testing Dataset from {test_paths}...")
    test_ds = TrafficDataset(test_paths, processor, scalerX, scalerY, slice_type)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    print(f"Test dataset holds {len(test_ds)} sequence windows.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"---------------------------------------")
    if torch.cuda.is_available():
        print(f"GPU is available.")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"GPU is NOT available. Using CPU.")
    print(f"Current device object: {device}")
    print(f"---------------------------------------")

    print(f"\nTiny ViT config: d_model={args.d_model}, n_heads={args.n_heads}, "
          f"n_layers={args.n_layers}, dim_ff={args.dim_feedforward}, dropout={args.vit_dropout}")

    model = TrafficModel(
        sequenceLength=sequenceLength,
        inFeatures=len(processor.features),
        sliceType=slice_type,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dim_feedforward=args.dim_feedforward,
        vit_dropout=args.vit_dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,} | Trainable: {trainable_count:,} ({100*trainable_count/total_params:.1f}%)")

    opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)

    criterion_mse = nn.MSELoss()
    # Compute pos_weight from training spike ratio to handle class imbalance
    all_spike_labels = []
    for _, _, s in train_loader:
        all_spike_labels.append(s)
    all_spike_labels = torch.cat(all_spike_labels)
    n_pos = all_spike_labels.sum().item()
    n_neg = len(all_spike_labels) - n_pos
    if n_pos > 0:
        raw_pw = n_neg / n_pos
        clamped_pw = min(raw_pw, args.max_pos_weight)
        pos_weight = torch.tensor([clamped_pw]).to(device)
    else:
        raw_pw = 1.0
        clamped_pw = 1.0
        pos_weight = torch.tensor([1.0]).to(device)
    print(f"Spike pos/neg ratio: {n_pos:.0f}/{n_neg:.0f}, raw_pos_weight={raw_pw:.2f}, clamped={clamped_pw:.2f}")
    # BCEWithLogitsLoss = Sigmoid + BCE in one op, numerically more stable than BCELoss + Sigmoid
    criterion_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    patience = args.patience

    for epoch in range(num_epochs):
        model.train()
        epochs_loss = 0.0

        for x, y, s in train_loader:
            x, y, s = x.to(device), y.to(device), s.to(device)
            # Add an extra dimension to s to match pred_spike shape (batch_size, 1)
            s = s.unsqueeze(-1)
            opt.zero_grad()

            pred, pred_spike = model(x)

            loss_reg = criterion_mse(pred, y)
            loss_spike = criterion_bce(pred_spike, s)
            loss = loss_reg + lambda_detect * loss_spike

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            epochs_loss += loss.item()

        epochs_loss /= len(train_loader)
        train_losses.append(epochs_loss)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x, y, s in val_loader:
                x, y, s = x.to(device), y.to(device), s.to(device)
                s = s.unsqueeze(-1)

                pred, pred_spike = model(x)

                loss_reg = criterion_mse(pred, y)
                loss_spike = criterion_bce(pred_spike, s)
                loss = loss_reg + lambda_detect * loss_spike

                val_loss += loss.item()

        # Catch if validation loader is empty directly
        if len(val_loader) > 0:
            val_loss /= len(val_loader)
        else:
            val_loss = 0.0

        val_losses.append(val_loss)

        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {epochs_loss:.5f} | Val Loss: {val_loss:.5f} | LR: {current_lr:.2e}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience > 0 and patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            print(f"Best val loss: {best_val_loss:.5f}")
            break

    # Restore best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model (val_loss={best_val_loss:.5f})")

    # Testing
    test_pred = []
    test_target = []
    test_spike_pred = []
    test_spike_target = []

    if len(test_loader) == 0:
        print("Warning: No testing directories matched validation criteria / files were too small.")
        return

    model.eval()
    with torch.no_grad():
        for x, y, s in test_loader:
            x, y, s = x.to(device), y.to(device), s.to(device)
            s = s.unsqueeze(-1)

            pred, pred_spike = model(x)
            test_pred.append(pred.cpu().numpy())
            test_target.append(y.cpu().numpy())
            test_spike_pred.append(pred_spike.cpu().numpy())
            test_spike_target.append(s.cpu().numpy())

    print("Testing collection complete!")

    test_prediction = np.concatenate(test_pred, axis=0)
    test_target = np.concatenate(test_target, axis=0)
    test_spike_prediction = np.concatenate(test_spike_pred, axis=0)
    test_spike_targets = np.concatenate(test_spike_target, axis=0)

    # Binarize spike predictions (logits > 0.0 == sigmoid > 0.5)
    spike_preds_binary = (test_spike_prediction > 0.0).astype(int)

    # Output
    run_dir = create_run_directory(slice_type)
    plot_loss_curve(run_dir, slice_type, train_losses, val_losses)
    plot_segmented_prediction(run_dir, slice_type, test_target, test_prediction)
    plot_segmented_spike(run_dir, slice_type, test_target, test_prediction,
                         test_spike_targets, spike_preds_binary)
    save_performance_matrix(run_dir, args, test_target, test_prediction,
                            test_spike_targets, spike_preds_binary)

if __name__ == "__main__":
    main()
