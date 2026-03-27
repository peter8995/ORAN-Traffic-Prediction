"""
Autoencoder-based spike detection for O-RAN traffic data.

Trains an autoencoder to reconstruct normal traffic patterns.
High reconstruction error indicates anomalous spikes.
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from matplotlib import pyplot as plt

from DataProcessor import DataProcessor, parse_directory_args, build_scalers


class TrafficAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon


def main():
    parser = argparse.ArgumentParser(description="Autoencoder spike detection for O-RAN traffic.")
    parser.add_argument('--base_path', type=str,
                        default='/home/cislab301b/peter/ORAN-Traffic-Prediction/Dataset/colosseum-oran-coloran-dataset')
    parser.add_argument('--train_dirs', type=str, nargs='+', required=True)
    parser.add_argument('--test_dirs', type=str, nargs='+', required=True)
    parser.add_argument('--slice_type', type=str, default='embb', choices=['embb', 'mmtc', 'urllc'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--sequence_length', type=int, default=15, help="Sliding window length")
    parser.add_argument('--latent_dim', type=int, default=16)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--patience', type=int, default=15, help="Early stopping patience (0 to disable)")
    parser.add_argument('--threshold_percentile', type=float, default=88,
                        help="Percentile of train reconstruction error to set as spike threshold")
    parser.add_argument('--k', type=float, default=2.0,
                        help="Spike value threshold: global_mean + k*global_std")
    parser.add_argument('--j', type=float, default=2.0,
                        help="Spike diff threshold: diff_mean + j*diff_std")
    args = parser.parse_args()

    train_paths = parse_directory_args(args.train_dirs, args.base_path)
    test_paths = parse_directory_args(args.test_dirs, args.base_path)

    # Data loading
    processor = DataProcessor(sequenceLength=args.sequence_length)
    processor.k = args.k
    processor.j = args.j
    train_files = processor.accumulate_files(train_paths, args.slice_type)
    if not train_files:
        raise ValueError("No metrics.csv files found in --train_dirs!")

    processor.fit_spike_params(train_files)
    scalerX, scalerY = build_scalers(processor, train_files)

    print(f"\nLoading training data...")
    train_X, train_Y, train_S = processor.process_directories(train_paths, scalerX, scalerY, args.slice_type)
    print(f"Loading test data...")
    test_X, test_Y, test_S = processor.process_directories(test_paths, scalerX, scalerY, args.slice_type)

    if len(train_X) == 0 or len(test_X) == 0:
        raise ValueError("No valid data found!")

    # train_X: (N, seq_len, 17), flatten sliding window for autoencoder
    seq_len = train_X.shape[1]
    n_features = train_X.shape[2]
    train_input = train_X.reshape(len(train_X), -1)  # (N, seq_len * 17)
    test_input = test_X.reshape(len(test_X), -1)
    input_dim = train_input.shape[1]

    print(f"Train: {len(train_input)} windows | Test: {len(test_input)} windows")
    print(f"Window shape: ({seq_len}, {n_features}) → flattened: {input_dim}")
    print(f"Train spike rate: {train_S.mean()*100:.2f}% | Test spike rate: {test_S.mean()*100:.2f}%")

    # PyTorch datasets
    train_tensor = torch.tensor(train_input, dtype=torch.float32)
    train_labels = torch.tensor(train_S, dtype=torch.float32)
    full_ds = TensorDataset(train_tensor, train_labels)

    val_len = int(len(full_ds) * args.val_split)
    train_len = len(full_ds) - val_len
    train_ds, val_ds = random_split(full_ds, [train_len, val_len], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    test_tensor = torch.tensor(test_input, dtype=torch.float32)
    test_label_tensor = torch.tensor(test_S, dtype=torch.float32)
    test_loader = DataLoader(TensorDataset(test_tensor, test_label_tensor),
                             batch_size=args.batch_size, shuffle=False)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = TrafficAutoencoder(input_dim, args.latent_dim).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Autoencoder params: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    # Training
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for x_batch, _ in train_loader:
            x_batch = x_batch.to(device)
            recon = model(x_batch)
            loss = criterion(recon, x_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, _ in val_loader:
                x_batch = x_batch.to(device)
                recon = model(x_batch)
                val_loss += criterion(recon, x_batch).item()
        val_loss /= max(len(val_loader), 1)
        val_losses.append(val_loss)

        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch+1}/{args.epochs}] Train: {epoch_loss:.6f} | Val: {val_loss:.6f} | LR: {lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if args.patience > 0 and patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {args.patience} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best model (val_loss={best_val_loss:.6f})")

    # Compute reconstruction errors on training set for threshold calibration
    model.eval()
    train_errors = []
    with torch.no_grad():
        for x_batch, _ in train_loader:
            x_batch = x_batch.to(device)
            recon = model(x_batch)
            errors = ((recon - x_batch) ** 2).mean(dim=1)  # per-sample MSE
            train_errors.append(errors.cpu().numpy())
    train_errors = np.concatenate(train_errors)

    threshold = np.percentile(train_errors, args.threshold_percentile)
    print(f"\nReconstruction error threshold (P{args.threshold_percentile:.0f}): {threshold:.6f}")

    # Test evaluation
    test_errors = []
    with torch.no_grad():
        for x_batch, _ in test_loader:
            x_batch = x_batch.to(device)
            recon = model(x_batch)
            errors = ((recon - x_batch) ** 2).mean(dim=1)
            test_errors.append(errors.cpu().numpy())
    test_errors = np.concatenate(test_errors)

    spike_preds = (test_errors > threshold).astype(int)

    # Metrics
    tp = np.sum((spike_preds == 1) & (test_S == 1))
    fp = np.sum((spike_preds == 1) & (test_S == 0))
    fn = np.sum((spike_preds == 0) & (test_S == 1))
    tn = np.sum((spike_preds == 0) & (test_S == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0

    print("\n" + "=" * 50)
    print("Autoencoder Spike Detection Results")
    print("=" * 50)
    print(f"Accuracy:   {accuracy:.4f}")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}")
    print(f"F1 Score:   {f1:.4f}")
    print(f"FP Rate:    {fp_rate:.4f}")
    print(f"FN Rate:    {fn_rate:.4f}")
    print(f"Threshold:  {threshold:.6f} (P{args.threshold_percentile:.0f} of train errors)")
    print(f"Test spikes: {test_S.sum():.0f}/{len(test_S)} ({test_S.mean()*100:.2f}%)")
    print(f"Predicted:   {spike_preds.sum()}/{len(spike_preds)} ({spike_preds.mean()*100:.2f}%)")
    print("=" * 50)

    # Save results
    result_dir = './results'
    os.makedirs(result_dir, exist_ok=True)
    slice_type = args.slice_type

    # 1. Loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.title(f'Autoencoder Loss - {slice_type.upper()}')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f'ae_{slice_type}_loss.png'), dpi=300)
    plt.close()

    # 2. Reconstruction error distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(train_errors, bins=100, alpha=0.7, label='Train', density=True)
    axes[0].axvline(threshold, color='r', linestyle='--', label=f'Threshold (P{args.threshold_percentile:.0f})')
    axes[0].set_title('Train Reconstruction Error')
    axes[0].set_xlabel('MSE')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    test_normal_errors = test_errors[test_S == 0]
    test_spike_errors = test_errors[test_S == 1]
    axes[1].hist(test_normal_errors, bins=100, alpha=0.6, label='Normal', density=True, color='blue')
    if len(test_spike_errors) > 0:
        axes[1].hist(test_spike_errors, bins=100, alpha=0.6, label='Spike', density=True, color='red')
    axes[1].axvline(threshold, color='r', linestyle='--', label=f'Threshold')
    axes[1].set_title('Test Reconstruction Error by Class')
    axes[1].set_xlabel('MSE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f'ae_{slice_type}_error_dist.png'), dpi=300)
    plt.close()

    # 3. Spike detection timeline
    max_pts = min(len(test_S), 10000)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})

    ax1.plot(test_errors[:max_pts], label='Reconstruction Error', color='purple', alpha=0.7, linewidth=0.5)
    ax1.axhline(threshold, color='r', linestyle='--', alpha=0.8, label=f'Threshold')
    ax1.set_ylabel('Reconstruction Error')
    ax1.set_title(f'Autoencoder Spike Detection - {slice_type.upper()}')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    x_axis = np.arange(max_pts)
    gt = test_S[:max_pts].astype(int)
    pred = spike_preds[:max_pts]
    ax2.fill_between(x_axis, 0, 1, where=(gt == 1), color='green', alpha=0.3, label='Ground Truth', step='mid')
    ax2.fill_between(x_axis, 0, 1, where=(pred == 1), color='red', alpha=0.4, label='Predicted', step='mid')
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Normal', 'Peak'])
    ax2.set_xlabel('Time Steps')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f'ae_{slice_type}_spike_detection.png'), dpi=300)
    plt.close()

    print(f"\nPlots saved to {result_dir}/")


if __name__ == "__main__":
    main()