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

def parse_directory_args(args_list, base_path):
    """
    Takes a list of strings like ['tr0-4', 'tr11']
    and returns full paths: ['/base/tr0', '/base/tr1', ... '/base/tr4', '/base/tr11']
    """
    parsed_dirs = []
    for item in args_list:
        if '-' in item:
            prefix = ''.join([c for c in item.split('-')[0] if not c.isdigit()])
            start = int(''.join([c for c in item.split('-')[0] if c.isdigit()]))
            end = int(''.join([c for c in item.split('-')[1] if c.isdigit()]))
            for i in range(start, end + 1):
                parsed_dirs.append(os.path.join(base_path, f"{prefix}{i}"))
        else:
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
    fig.suptitle(f'{slice_type.upper()} Prediction vs Target ({total} steps) — Regression Only', fontsize=14)
    plt.tight_layout()

    save_path = os.path.join(run_dir, f'{slice_type}_prediction.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Prediction plot saved to: {save_path}")


def save_performance_matrix(run_dir, args, target, prediction, spike_gt):
    target = target.flatten()
    prediction = prediction.flatten()
    spike_gt = spike_gt.flatten().astype(int)

    # Overall regression metrics
    mae = np.mean(np.abs(target - prediction))
    rmse = np.sqrt(np.mean((target - prediction) ** 2))
    ss_res = np.sum((target - prediction) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Regression metrics split by spike / normal regions
    spike_mask = spike_gt == 1
    normal_mask = spike_gt == 0
    n_spike = spike_mask.sum()
    n_normal = normal_mask.sum()

    if n_spike > 0:
        mae_spike = np.mean(np.abs(target[spike_mask] - prediction[spike_mask]))
        rmse_spike = np.sqrt(np.mean((target[spike_mask] - prediction[spike_mask]) ** 2))
        ss_res_s = np.sum((target[spike_mask] - prediction[spike_mask]) ** 2)
        ss_tot_s = np.sum((target[spike_mask] - np.mean(target[spike_mask])) ** 2)
        r2_spike = 1 - ss_res_s / ss_tot_s if ss_tot_s > 0 else 0.0
    else:
        mae_spike = rmse_spike = r2_spike = float('nan')

    if n_normal > 0:
        mae_normal = np.mean(np.abs(target[normal_mask] - prediction[normal_mask]))
        rmse_normal = np.sqrt(np.mean((target[normal_mask] - prediction[normal_mask]) ** 2))
        ss_res_n = np.sum((target[normal_mask] - prediction[normal_mask]) ** 2)
        ss_tot_n = np.sum((target[normal_mask] - np.mean(target[normal_mask])) ** 2)
        r2_normal = 1 - ss_res_n / ss_tot_n if ss_tot_n > 0 else 0.0
    else:
        mae_normal = rmse_normal = r2_normal = float('nan')

    mae_ratio = mae_spike / mae_normal if (n_normal > 0 and n_spike > 0 and mae_normal > 0) else float('nan')
    spike_rate = np.mean(spike_gt) * 100

    filepath = os.path.join(run_dir, 'performance_matrix.txt')
    with open(filepath, 'w') as f:
        f.write('=' * 40 + '\n')
        f.write('Training Parameters\n')
        f.write('=' * 40 + '\n')
        for key, value in vars(args).items():
            f.write(f'{key}: {value}\n')

        f.write('\n' + '=' * 40 + '\n')
        f.write('Regression Metrics (All)\n')
        f.write('=' * 40 + '\n')
        f.write(f'MAE:    {mae:.4f}\n')
        f.write(f'RMSE:   {rmse:.4f}\n')
        f.write(f'R²:     {r2:.4f}\n')

        f.write('\n' + '=' * 40 + '\n')
        f.write('Regression Metrics by Region\n')
        f.write('=' * 40 + '\n')
        f.write(f'Spike Rate: {spike_rate:.2f}%\n')
        f.write(f'\nSpike timesteps (N={n_spike}):\n')
        f.write(f'  MAE:  {mae_spike:.4f}\n')
        f.write(f'  RMSE: {rmse_spike:.4f}\n')
        f.write(f'  R²:   {r2_spike:.4f}\n')
        f.write(f'\nNormal timesteps (N={n_normal}):\n')
        f.write(f'  MAE:  {mae_normal:.4f}\n')
        f.write(f'  RMSE: {rmse_normal:.4f}\n')
        f.write(f'  R²:   {r2_normal:.4f}\n')
        f.write(f'\nMAE ratio (spike/normal): {mae_ratio:.4f}\n')

    print(f"Performance matrix saved to: {filepath}")

    print(f"\n{'=' * 40}")
    print("Regression Metrics (All):")
    print(f"  MAE:  {mae:.4f}  RMSE: {rmse:.4f}  R²: {r2:.4f}")
    print(f"\nSpike timesteps (N={n_spike}, {spike_rate:.2f}%):")
    print(f"  MAE:  {mae_spike:.4f}  RMSE: {rmse_spike:.4f}  R²: {r2_spike:.4f}")
    print(f"\nNormal timesteps (N={n_normal}):")
    print(f"  MAE:  {mae_normal:.4f}  RMSE: {rmse_normal:.4f}  R²: {r2_normal:.4f}")
    print(f"\nMAE ratio (spike/normal): {mae_ratio:.4f}")
    print('=' * 40 + '\n')


def main():
    parser = argparse.ArgumentParser(description="Train Regression-Only Traffic Model (Tiny ViT, no spike head).")

    # Paths Options
    parser.add_argument('--base_path', type=str, default='/home/cislab301b/peter/ORAN-Traffic-Prediction/Dataset/colosseum-oran-coloran-dataset')
    parser.add_argument('--train_dirs', type=str, nargs='+', required=True)
    parser.add_argument('--test_dirs', type=str, nargs='+', required=True)

    # Tunables
    parser.add_argument('--slice_type', type=str, default='embb', choices=['embb', 'mmtc', 'urllc'])
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--sequence_length', type=int, default=15)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--patience', type=int, default=50)

    # Tiny ViT hyperparameters
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--dim_feedforward', type=int, default=512)
    parser.add_argument('--vit_dropout', type=float, default=0.2)

    args = parser.parse_args()

    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_epochs = args.epochs
    sequenceLength = args.sequence_length
    slice_type = args.slice_type

    train_paths = parse_directory_args(args.train_dirs, args.base_path)
    test_paths = parse_directory_args(args.test_dirs, args.base_path)

    processor = DataProcessor(sequenceLength)

    print(f"\n{'='*20} Analzing Training Directories {'='*20}")
    print(f"Parsed Train Paths: {train_paths}")

    train_files = processor.accumulate_files(train_paths, slice_type)
    if not train_files:
        raise ValueError("No metrics.csv files found in the provided --train_dirs!")

    processor.fit_spike_params(train_files)

    print(f"Discovered {len(train_files)} files. Building global MinMaxScaler from ALL training files:")

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
    del all_X_raw, all_Y_raw

    print(f"Global scalers fitted on {len(train_files)} files.")

    print(f"\nBuilding comprehensive Training & Validation Dataset...")
    full_train_ds = TrafficDataset(train_paths, processor, scalerX, scalerY, slice_type)

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
        print(f"GPU is available: {torch.cuda.get_device_name(0)}")
    else:
        print(f"GPU is NOT available. Using CPU.")
    print(f"Device: {device}")
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

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    patience = args.patience

    for epoch in range(num_epochs):
        model.train()
        epochs_loss = 0.0

        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()

            pred = model(x)
            loss = criterion_mse(pred, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            epochs_loss += loss.item()

        epochs_loss /= len(train_loader)
        train_losses.append(epochs_loss)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += criterion_mse(pred, y).item()

        val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        val_losses.append(val_loss)

        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {epochs_loss:.5f} | Val Loss: {val_loss:.5f} | LR: {current_lr:.2e}")

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

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model (val_loss={best_val_loss:.5f})")

    # Testing
    test_pred = []
    test_target = []
    test_spike_target = []

    if len(test_loader) == 0:
        print("Warning: No testing directories matched validation criteria / files were too small.")
        return

    model.eval()
    with torch.no_grad():
        for x, y, s in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_pred.append(pred.cpu().numpy())
            test_target.append(y.cpu().numpy())
            test_spike_target.append(s.numpy())

    print("Testing collection complete!")

    test_prediction = np.concatenate(test_pred, axis=0)
    test_target = np.concatenate(test_target, axis=0)
    test_spike_targets = np.concatenate(test_spike_target, axis=0)

    run_dir = create_run_directory(slice_type)
    plot_loss_curve(run_dir, slice_type, train_losses, val_losses)
    plot_segmented_prediction(run_dir, slice_type, test_target, test_prediction)
    save_performance_matrix(run_dir, args, test_target, test_prediction, test_spike_targets)

if __name__ == "__main__":
    main()
