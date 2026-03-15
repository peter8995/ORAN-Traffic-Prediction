import argparse
from classicModel import *
from DataProcessor import *
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
import torch

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

def main():
    parser = argparse.ArgumentParser(description="Train Traffic Model across multiple directories.")
    
    # Paths Options
    parser.add_argument('--base_path', type=str, default='/home/cislab301b/peter/ORAN-Traffic-Prediction/Dataset/colosseum-oran-coloran-dataset', help="Base directory where tr* folders exist.")
    parser.add_argument('--train_dirs', type=str, nargs='+', required=True, help="List of folder ranges or names (e.g. --train_dirs tr0-4 tr10)")
    parser.add_argument('--test_dirs', type=str, nargs='+', required=True, help="List of folder ranges or names (e.g. --test_dirs tr5-6)")
    
    # Tunables
    parser.add_argument('--slice_type', type=str, default='embb', choices=['embb', 'mmtc', 'urllc'], help="Type of the slice (embb, mmtc, urllc)")
    parser.add_argument('--epochs', type=int, default=500, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=1024, help="Batch size")
    parser.add_argument('--learning_rate', type=float, default=1e-6, help="Learning rate")
    parser.add_argument('--sequence_length', type=int, default=15, help="Sequence length for the LSTM")
    parser.add_argument('--val_split', type=float, default=0.2, help="Percentage of training data to use for validation")
    parser.add_argument('--lambda_detect', type=float, default=0.5, help="Weight for the spike detection BCE loss")
    
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

    model = TrafficModel(
        sequenceLength=sequenceLength,
        inFeatures=len(processor.features), sliceType=slice_type
    ).to(device)
    
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion_mse = nn.MSELoss()
    # BCEWithLogitsLoss = Sigmoid + BCE in one op, numerically more stable than BCELoss + Sigmoid
    criterion_bce = nn.BCEWithLogitsLoss()

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        epochs_loss = 0.0

        for x, y, s in train_loader:
            x, y, s = x.to(device), y.to(device), s.to(device)
            # Add an extra dimension to s to match pred_spike shape (batch_size, 1)
            s = s.unsqueeze(-1)
            opt.zero_grad()
            
            pred, pred_spike = model(x)
            
            if slice_type == 'mmtc':
                loss_reg = quantile_loss(pred, y, 0.7)
            else:
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
                
                if slice_type == 'mmtc':
                    loss_reg = quantile_loss(pred, y, 0.7)
                else:
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

        # Print every single epoch instead of every 10 epochs
        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {epochs_loss:.5f} | Val Loss: {val_loss:.5f}")
    
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
    
    # Calculate Spike Detection Metrics
    # spike_head outputs logits (no Sigmoid), so threshold is 0.0 (== sigmoid > 0.5)
    spike_preds_binary = (test_spike_prediction > 0.0).astype(int)
    
    # Avoid division by zero warnings
    tp = np.sum((spike_preds_binary == 1) & (test_spike_targets == 1))
    fp = np.sum((spike_preds_binary == 1) & (test_spike_targets == 0))
    fn = np.sum((spike_preds_binary == 0) & (test_spike_targets == 1))
    tn = np.sum((spike_preds_binary == 0) & (test_spike_targets == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    print("\n" + "="*40)
    print("Spike Detection Performance Metrics:")
    print("="*40)
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"FP Rate:   {fp_rate:.4f}")
    print(f"FN Rate:   {fn_rate:.4f}")
    print(f"Spike Rate: {np.mean(test_spike_targets)*100:.2f}% of test samples")
    print("="*40 + "\n")

    result_dir = './results'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        print(f"Created directory: {result_dir}")
    
    # Plot Loss Curve
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.title(f'{slice_type.upper()} Slice Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    loss_save_path = os.path.join(result_dir, f'{slice_type}_multi_dir_loss_curve.png')
    plt.savefig(loss_save_path, dpi=300)
    print(f"✅ Loss curve saved to: {loss_save_path}")
    
    # Plot Prediction Comparison (Only plot first 2000 points if massive to prevent crash)
    max_plot_pts = min(len(test_target), 10000)
    plt.figure(figsize=(10, 5))
    plt.plot(test_target[:max_plot_pts], label='Target (Actual)', color='blue', alpha=0.6)
    plt.plot(test_prediction[:max_plot_pts], label='Prediction', color='orange', alpha=0.8)
    plt.title(f'{slice_type.upper()} Multi-Directory Prediction vs Target (Showing {max_plot_pts} pts)')
    plt.xlabel('Time Steps')
    plt.ylabel('Traffic (Granted PRBs)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    pred_save_path = os.path.join(result_dir, f'{slice_type}_multi_dir_prediction.png')
    plt.savefig(pred_save_path, dpi=300)
    print(f"✅ Image saved to: {pred_save_path}")

    # Plot Spike Detection Visualization (similar to paper Figures 3-5)
    max_spike_pts = min(len(test_target), 10000)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})

    # Top: Resource demand prediction
    ax1.plot(test_target[:max_spike_pts], label='Actual RAN Demand', color='blue', alpha=0.7, linewidth=0.8)
    ax1.plot(test_prediction[:max_spike_pts], label='Predicted RAN Demand', color='red', linestyle='--', alpha=0.7, linewidth=0.8)
    ax1.set_ylabel('Traffic (Granted PRBs)')
    ax1.set_title(f'{slice_type.upper()} Resource Demand Prediction & Spike Detection')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Bottom: Spike detection bar (green = spike detected, white = normal)
    spike_binary_plot = spike_preds_binary[:max_spike_pts].flatten()
    spike_gt_plot = test_spike_targets[:max_spike_pts].flatten().astype(int)
    x_axis = np.arange(max_spike_pts)

    ax2.fill_between(x_axis, 0, 1, where=(spike_gt_plot == 1),
                     color='green', alpha=0.3, label='Ground Truth Spike', step='mid')
    ax2.fill_between(x_axis, 0, 1, where=(spike_binary_plot == 1),
                     color='red', alpha=0.4, label='Predicted Spike', step='mid')
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Normal', 'Peak'])
    ax2.set_xlabel('Time Steps')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    spike_save_path = os.path.join(result_dir, f'{slice_type}_spike_detection.png')
    plt.savefig(spike_save_path, dpi=300)
    print(f"✅ Spike detection plot saved to: {spike_save_path}")

if __name__ == "__main__":
    main()