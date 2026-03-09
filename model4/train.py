import argparse
from classicModel import *
from DataProcessor import *
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
import os
import torch

def quantile_loss(preds, targets, quantile: float):
    errors = targets - preds
    loss = torch.max((quantile - 1) * errors, quantile * errors)
    return torch.abs(loss).mean()

def main():
    parser = argparse.ArgumentParser(description="Train Traffic Model for a specific slice.")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the dataset CSV file")
    parser.add_argument('--slice_type', type=str, default='embb', choices=['embb', 'mmtc', 'urllc'], help="Type of the slice (embb, mmtc, urllc)")
    parser.add_argument('--epochs', type=int, default=500, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=1024, help="Batch size")
    parser.add_argument('--learning_rate', type=float, default=1e-6, help="Learning rate")
    parser.add_argument('--sequence_length', type=int, default=15, help="Sequence length for the LSTM")
    
    args = parser.parse_args()

    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_epochs = args.epochs
    sequenceLength = args.sequence_length
    slice_type = args.slice_type
    data_path = args.data_path

    # Dataset
    processor = DataProcessor(sequenceLength)

    print(f"\n{'='*20} Checking data: {data_path} {'='*20}") 
    
    df = processor.load_and_clean(data_path)
    
    # 1. 顯示前 5 筆資料
    print(">> Head:")
    print(df.head())

    # 2. 檢查有無異常極端值
    print("\n>> Describe:")
    print(df.describe())

    # 3. 檢查是否有 NaN
    print("\n>> NaN Check:")
    nan_counts = df.isnull().sum()
    print(nan_counts[nan_counts > 0])
    
    if df.isnull().values.any():
        print("Found NaN!.")
    else:
        print("Data check OK: No any NaN found.")

    print(f"{'='*60}\n")
    
    train_len = int(len(df) * 0.8)
    df_train = df.iloc[:train_len]
    
    train_X = df_train[processor.features].values
    train_Y = df_train[[processor.target]].values
    
    scalerX = MinMaxScaler()
    scalerY = MinMaxScaler()
    
    scalerX.fit(train_X)
    scalerY.fit(train_Y)
    
    train_ds = TrafficDataset(data_path, processor, scalerX, scalerY, mode='train')
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_ds = TrafficDataset(data_path, processor, scalerX, scalerY, mode='val')
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    test_ds = TrafficDataset(data_path, processor, scalerX, scalerY, mode='test')
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

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
    criterion_huber = nn.SmoothL1Loss()

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        epochs_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            pred = model(x)
            
            if slice_type == 'mmtc':
                loss = quantile_loss(pred, y, 0.7)
            else:
                loss = criterion_mse(pred, y)
                
            loss.backward()
            opt.step()
            epochs_loss += loss.item()

        epochs_loss /= len(train_loader)
        train_losses.append(epochs_loss)
               
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                
                if slice_type == 'mmtc':
                    loss = quantile_loss(pred, y, 0.7)
                else:
                    loss = criterion_mse(pred, y)
                    
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {epochs_loss:.5f} | Val Loss: {val_loss:.5f}")
    
    # Testing
    test_pred = []
    test_target = []

    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_pred.append(pred.cpu().numpy())
            test_target.append(y.cpu().numpy())

    print("Testing collection complete!")

    test_prediction = np.concatenate(test_pred, axis=0)
    test_target = np.concatenate(test_target, axis=0)

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
    
    slice_basename = os.path.basename(data_path).replace('.csv', '')
    loss_save_path = os.path.join(result_dir, f'{slice_type}_{slice_basename}_loss_curve.png')
    plt.savefig(loss_save_path, dpi=300)
    print(f"✅ Loss curve saved to: {loss_save_path}")
    
    # Plot Prediction Comparison
    plt.figure(figsize=(10, 5))
    plt.plot(test_target, label='Target (Actual)', color='blue', alpha=0.6)
    plt.plot(test_prediction, label='Prediction', color='orange', alpha=0.8)
    plt.title(f'{slice_type.upper()} Slice Prediction vs Target')
    plt.xlabel('Time Steps')
    plt.ylabel('Traffic (Granted PRBs)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    pred_save_path = os.path.join(result_dir, f'{slice_type}_{slice_basename}_prediction.png')
    plt.savefig(pred_save_path, dpi=300)
    print(f"✅ Image saved to: {pred_save_path}")

if __name__ == "__main__":
    main()