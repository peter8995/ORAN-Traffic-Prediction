from classicModel import *
from DataProcessor import *
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import numpy as np

def quantile_loss(preds, targets, quantile: float):
    errors = targets - preds
    loss = torch.max((quantile - 1) * errors, quantile * errors)
    return torch.abs(loss).mean()

def main():
    batch_size = 1024
    learning_rate = 1e-6
    num_epochs = 500
    sequenceLength = 15

    #Dataset
    processor = DataProcessor(sequenceLength)

    embb_path = '../Dataset/Tractor/Trial7/embb_03_03c.csv'
    mmtc_path = '../Dataset/Tractor/Trial7/mmtc_2.csv'
    urllc_path = '../Dataset/Tractor/Trial7/Raw/urll_11_18.csv'

    def get_fitted_scalers(path):
        print(f"\n{'='*20} Checking data: {path} {'='*20}") 
        
        df = processor.load_and_clean(path)
        
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
        
        return scalerX, scalerY

    scalers_embb = get_fitted_scalers(embb_path)
    scalers_mmtc = get_fitted_scalers(mmtc_path)
    scalers_urllc = get_fitted_scalers(urllc_path)

    train_ds_embb = TrafficDataset(embb_path, processor, scalers_embb[0], scalers_embb[1], mode='train')
    train_ds_mmtc = TrafficDataset(mmtc_path, processor, scalers_mmtc[0], scalers_mmtc[1], mode='train')
    train_ds_urllc = TrafficDataset(urllc_path, processor, scalers_urllc[0], scalers_urllc[1], mode='train')
    
    train_loader_embb = DataLoader(train_ds_embb, batch_size=batch_size, shuffle=True)
    train_loader_mmtc = DataLoader(train_ds_mmtc, batch_size=batch_size, shuffle=True)
    train_loader_urllc = DataLoader(train_ds_urllc, batch_size=batch_size, shuffle=True)

    val_ds_embb = TrafficDataset(embb_path, processor, scalers_embb[0], scalers_embb[1], mode='val')
    val_ds_mmtc = TrafficDataset(mmtc_path, processor, scalers_mmtc[0], scalers_mmtc[1], mode='val')
    val_ds_urllc = TrafficDataset(urllc_path, processor, scalers_urllc[0], scalers_urllc[1], mode='val')

    val_loader_embb = DataLoader(val_ds_embb, batch_size=batch_size, shuffle=False)
    val_loader_mmtc = DataLoader(val_ds_mmtc, batch_size=batch_size, shuffle=False)
    val_loader_urllc = DataLoader(val_ds_urllc, batch_size=batch_size, shuffle=False)

    test_ds_embb = TrafficDataset(embb_path, processor, scalers_embb[0], scalers_embb[1], mode='test')
    test_ds_mmtc = TrafficDataset(mmtc_path, processor, scalers_mmtc[0], scalers_mmtc[1], mode='test')
    test_ds_urllc = TrafficDataset(urllc_path, processor, scalers_urllc[0], scalers_urllc[1], mode='test')

    test_loader_embb = DataLoader(test_ds_embb, batch_size=batch_size, shuffle=False)
    test_loader_mmtc = DataLoader(test_ds_mmtc, batch_size=batch_size, shuffle=False)
    test_loader_urllc = DataLoader(test_ds_urllc, batch_size=batch_size, shuffle=False)



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"---------------------------------------")
    if torch.cuda.is_available():
        print(f"GPU is available.")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"GPU is NOT available. Using CPU.")
    print(f"Current device object: {device}")
    print(f"---------------------------------------")

    model_embb = TrafficModel(
        sequenceLength=sequenceLength,
        inFeatures=len(processor.features), sliceType= 'embb'
    ).to(device)
    model_mmtc = TrafficModel(
        sequenceLength=sequenceLength,
        inFeatures=len(processor.features), sliceType= 'mmtc'
    ).to(device)
    model_urllc = TrafficModel(
        sequenceLength=sequenceLength,
        inFeatures=len(processor.features), sliceType= 'urllc'
    ).to(device)
    
    opt_embb = torch.optim.Adam(model_embb.parameters(), lr=learning_rate)
    opt_mmtc = torch.optim.Adam(model_mmtc.parameters(), lr=learning_rate)
    opt_urllc = torch.optim.Adam(model_urllc.parameters(), lr=learning_rate)

    criterion_mse = nn.MSELoss()
    criterion_huber = nn.SmoothL1Loss()


    train_losses_embb = []
    train_losses_mmtc = []
    train_losses_urllc = []

    val_losses_embb = []
    val_losses_mmtc = []
    val_losses_urllc = []


    for epoch in range(num_epochs):
        model_embb.train()
        model_mmtc.train()
        model_urllc.train()

        epochs_loss_embb = 0.0
        epochs_loss_mmtc = 0.0
        epochs_loss_urllc = 0.0
        for x, y in train_loader_embb:
            x, y = x.to(device), y.to(device)
            opt_embb.zero_grad()
            pred = model_embb(x)
            loss = criterion_mse(pred, y)
            loss.backward()
            opt_embb.step()
            epochs_loss_embb += loss.item()
        
        for x, y in train_loader_mmtc:
            x, y = x.to(device), y.to(device)
            opt_mmtc.zero_grad()
            pred = model_mmtc(x)
            loss = quantile_loss(pred, y, 0.7)
            loss.backward()
            opt_mmtc.step()
            epochs_loss_mmtc += loss.item()
        
        for x, y in train_loader_urllc:
            x, y = x.to(device), y.to(device)
            opt_urllc.zero_grad()
            pred = model_urllc(x)
            loss = criterion_mse(pred, y)
            loss.backward()
            opt_urllc.step()
            epochs_loss_urllc += loss.item()

        epochs_loss_embb /= len(train_loader_embb)
        epochs_loss_mmtc /= len(train_loader_mmtc)
        epochs_loss_urllc /= len(train_loader_urllc)

        train_losses_embb.append(epochs_loss_embb)
        train_losses_mmtc.append(epochs_loss_mmtc)
        train_losses_urllc.append(epochs_loss_urllc)
               
        model_embb.eval()
        model_mmtc.eval()
        model_urllc.eval()

        val_loss_embb = 0.0
        val_loss_mmtc = 0.0
        val_loss_urllc = 0.0

        with torch.no_grad():
            for x, y in val_loader_embb:
                x, y = x.to(device), y.to(device)
                pred = model_embb(x)
                loss = criterion_mse(pred, y)
                val_loss_embb += loss.item()
            
            for x, y in val_loader_mmtc:
                x, y = x.to(device), y.to(device)
                pred = model_mmtc(x)
                loss = quantile_loss(pred, y, 0.7)
                val_loss_mmtc += loss.item()

            for x, y in val_loader_urllc:
                x, y = x.to(device), y.to(device)
                pred = model_urllc(x)
                loss = criterion_mse(pred, y)
                val_loss_urllc += loss.item()

        val_loss_embb /= len(val_loader_embb)
        val_loss_mmtc /= len(val_loader_mmtc)
        val_loss_urllc /= len(val_loader_urllc)

        val_losses_embb.append(val_loss_embb)
        val_losses_mmtc.append(val_loss_mmtc)
        val_losses_urllc.append(val_loss_urllc)

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"   eMBB  | Train: {epochs_loss_embb:.5f} | Val: {val_loss_embb:.5f}")
        print(f"   mMTC  | Train: {epochs_loss_mmtc:.5f} | Val: {val_loss_mmtc:.5f}")
        print(f"   uRLLC | Train: {epochs_loss_urllc:.5f} | Val: {val_loss_urllc:.5f}")
        print("-" * 60)
    
    embb_pred = []
    embb_target = []
    mmtc_pred = []
    mmtc_target = []
    urllc_pred = []
    urllc_target = []   

    model_embb.eval()
    model_mmtc.eval()
    model_urllc.eval()
    with torch.no_grad():
        for x, y in test_loader_embb:
            x, y = x.to(device), y.to(device)
            pred = model_embb(x)
            p_np = pred.cpu().numpy()
            y_np = y.cpu().numpy()
            embb_pred.append(p_np)
            embb_target.append(y_np)

        for x, y in test_loader_mmtc:
            x, y = x.to(device), y.to(device)
            pred = model_mmtc(x)
            p_np = pred.cpu().numpy()
            y_np = y.cpu().numpy()
            mmtc_pred.append(p_np)
            mmtc_target.append(y_np)

        for x, y in test_loader_urllc:
            x, y = x.to(device), y.to(device)
            pred = model_urllc(x)
            p_np = pred.cpu().numpy()
            y_np = y.cpu().numpy()
            urllc_pred.append(p_np)
            urllc_target.append(y_np)    
    print("Data collect complete!")

    embb_prediction = np.concatenate(embb_pred, axis=0)
    embb_target = np.concatenate(embb_target, axis=0)
    mmtc_prediction = np.concatenate(mmtc_pred, axis=0)
    mmtc_target = np.concatenate(mmtc_target, axis=0)
    urllc_prediction = np.concatenate(urllc_pred, axis=0)
    urllc_target = np.concatenate(urllc_target, axis=0)

    result_dir = './results'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        print(f"Created directory: {result_dir}")
    
    # loss curve
    plt.figure(figsize=(18, 5))

    # eMBB Loss
    plt.subplot(1, 3, 1)
    plt.plot(train_losses_embb, label='Train')
    plt.plot(val_losses_embb, label='Val')
    plt.title('eMBB Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # mMTC Loss
    plt.subplot(1, 3, 2)
    plt.plot(train_losses_mmtc, label='Train')
    plt.plot(val_losses_mmtc, label='Val')
    plt.title('mMTC Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # uRLLC Loss
    plt.subplot(1, 3, 3)
    plt.plot(train_losses_urllc, label='Train')
    plt.plot(val_losses_urllc, label='Val')
    plt.title('uRLLC Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    loss_save_path = os.path.join(result_dir, f'loss_curve_lr{learning_rate}_ep{num_epochs}_seq{sequenceLength}.png')
    plt.savefig(loss_save_path, dpi=300)
    print(f"✅ Loss curve saved to: {loss_save_path}")
    
    # prediction comparison
    plt.figure(figsize=(18, 5))

    # embb pred fig
    plt.subplot(1, 3, 1)
    plt.plot(embb_target, label='Target', color='blue', alpha=0.6)
    plt.plot(embb_prediction, label='Prediction', color='orange', alpha=0.8)
    plt.title('eMBB Slice Prediction')
    plt.xlabel('Time Steps')
    plt.ylabel('Traffic')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # mMTC pred fig
    plt.subplot(1, 3, 2)
    plt.plot(mmtc_target, label='Target', color='blue', alpha=0.6)
    plt.plot(mmtc_prediction, label='Prediction', color='orange', alpha=0.8)
    plt.title('mMTC Slice Prediction')
    plt.xlabel('Time Steps')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # uRLLC pred fig
    plt.subplot(1, 3, 3)
    plt.plot(urllc_target, label='Target', color='blue', alpha=0.6)
    plt.plot(urllc_prediction, label='Prediction', color='orange', alpha=0.8)
    plt.title('uRLLC Slice Prediction')
    plt.xlabel('Time Steps')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(result_dir, f'prediction_comparison_lr{learning_rate}_ep{num_epochs}_seq{sequenceLength}.png')
    plt.savefig(save_path, dpi=300)
    print(f"✅ Image saved to: {save_path}")

if __name__ == "__main__":
    main()