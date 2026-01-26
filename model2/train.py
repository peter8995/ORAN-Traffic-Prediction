from classicModel import *
from DataProcessor import *
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import numpy as np


def main():
    batch_size = 1024
    learning_rate = 1e-6
    num_epochs = 500
    sequenceLength = 15

    #Dataset
    processor = DataProcessor(sequenceLength)

    embb_path = '../Dataset/Trial7/Raw/embb_11_18.csv'
    mmtc_path = '../Dataset/Trial7/Raw/mmtc_11_18.csv'
    urllc_path = '../Dataset/Trial7/Raw/urll_11_18.csv'

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

    train_dataset = MultiTaskDataset(
        embb_path=embb_path, 
        mmtc_path=mmtc_path, 
        urllc_path=urllc_path, 
        processor=processor,
        scalers_embb=scalers_embb,
        scalers_mmtc=scalers_mmtc,
        scalers_urllc=scalers_urllc,
        mode='train'
    )
    val_dataset = MultiTaskDataset(
        embb_path=embb_path, 
        mmtc_path=mmtc_path, 
        urllc_path=urllc_path, 
        processor=processor,
        scalers_embb=scalers_embb,
        scalers_mmtc=scalers_mmtc,
        scalers_urllc=scalers_urllc,
        mode='val'
    )

    test_dataset = MultiTaskDataset(
        embb_path=embb_path, 
        mmtc_path=mmtc_path, 
        urllc_path=urllc_path, 
        processor=processor,
        scalers_embb=scalers_embb,
        scalers_mmtc=scalers_mmtc,
        scalers_urllc=scalers_urllc,
        mode='test'
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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

        for i, (x1, x2, x3, y1, y2, y3) in enumerate(train_loader):
            x_embb, y_embb = x1.to(device), y1.to(device)
            x_mmtc, y_mmtc = x2.to(device), y2.to(device)
            x_urllc, y_urllc = x3.to(device), y3.to(device)

            # Train embb model
            opt_embb.zero_grad()
            pred_embb = model_embb(x_embb)
            loss_embb = criterion_mse(pred_embb, y_embb)
            loss_embb.backward()
            opt_embb.step()
            epochs_loss_embb += loss_embb.item()

            # Train mmtc model
            opt_mmtc.zero_grad()
            pred_mmtc = model_mmtc(x_mmtc)
            loss_mmtc = criterion_mse(pred_mmtc, y_mmtc)
            loss_mmtc.backward()
            opt_mmtc.step()
            epochs_loss_mmtc += loss_mmtc.item()

            # Train urllc model
            opt_urllc.zero_grad()
            pred_urllc = model_urllc(x_urllc)
            loss_urllc = criterion_mse(pred_urllc, y_urllc)
            loss_urllc.backward()
            opt_urllc.step()
            epochs_loss_urllc += loss_urllc.item()

        

        epochs_loss_embb /= len(train_loader)
        epochs_loss_mmtc /= len(train_loader)
        epochs_loss_urllc /= len(train_loader)

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
            for i, (x1, x2, x3, y1, y2, y3) in enumerate(val_loader):
                x_embb, y_embb = x1.to(device), y1.to(device)
                x_mmtc, y_mmtc = x2.to(device), y2.to(device)
                x_urllc, y_urllc = x3.to(device), y3.to(device)

                # embb val
                pred_embb = model_embb(x_embb)
                val_loss_embb += criterion_mse(pred_embb, y_embb).item()

                # mmtc val
                pred_mmtc = model_mmtc(x_mmtc)
                val_loss_mmtc += criterion_mse(pred_mmtc, y_mmtc).item()
                
                # uRLLC Val
                pred_urllc = model_urllc(x_urllc)
                val_loss_urllc += criterion_mse(pred_urllc, y_urllc).item() 
        val_loss_embb /= len(val_loader)
        val_loss_mmtc /= len(val_loader)
        val_loss_urllc /= len(val_loader)

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
        for i, (x1, x2, x3, y1, y2, y3) in enumerate(test_loader):
            x_embb, y_embb = x1.to(device), y1.to(device)
            x_mmtc, y_mmtc = x2.to(device), y2.to(device)
            x_urllc, y_urllc = x3.to(device), y3.to(device)

            pred_embb = model_embb(x_embb)
            pred_mmtc = model_mmtc(x_mmtc)
            pred_urllc = model_urllc(x_urllc)

            embb_p_np = pred_embb.cpu().numpy()
            embb_y_np = y_embb.cpu().numpy()
            mmtc_p_np = pred_mmtc.cpu().numpy()
            mmtc_y_np = y_mmtc.cpu().numpy()
            urllc_p_np = pred_urllc.cpu().numpy()
            urllc_y_np = y_urllc.cpu().numpy()

            embb_pred.append(embb_p_np)
            embb_target.append(embb_y_np)
            mmtc_pred.append(mmtc_p_np)
            mmtc_target.append(mmtc_y_np)
            urllc_pred.append(urllc_p_np)
            urllc_target.append(urllc_y_np)
    
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
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # mMTC Loss
    plt.subplot(1, 3, 2)
    plt.plot(train_losses_mmtc, label='Train')
    plt.plot(val_losses_mmtc, label='Val')
    plt.title('mMTC Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Huber)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # uRLLC Loss
    plt.subplot(1, 3, 3)
    plt.plot(train_losses_urllc, label='Train')
    plt.plot(val_losses_urllc, label='Val')
    plt.title('uRLLC Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Huber)')
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