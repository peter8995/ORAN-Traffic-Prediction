from classicModel import *
from DataProcessor import *
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import numpy as np


def main():
    batch_size = 32
    learning_rate = 1e-6
    num_epochs = 150
    sequenceLength = 30

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

    model = TrafficModel(
        sequenceLength=sequenceLength,
        inFeatures=len(processor.features)
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    criterion_mse = nn.MSELoss()
    criterion_huber = nn.SmoothL1Loss()

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for i, (x1, x2, x3, y1, y2, y3) in enumerate(train_loader):
            x_embb = x1.to(device)
            x_mmtc = x2.to(device)
            x_urllc = x3.to(device)

            y_embb = y1.to(device)
            y_mmtc = y2.to(device)
            y_urllc = y3.to(device)

            optimizer.zero_grad()

            pred_global, pred_embb, pred_mmtc, pred_urllc = model(x_embb, x_mmtc, x_urllc)
            
            y_global = y_embb + y_mmtc + y_urllc
            loss_embb = criterion_mse(pred_embb, y_embb)
            loss_mmtc = criterion_huber(pred_mmtc, y_mmtc)
            loss_urllc = criterion_huber(pred_urllc, y_urllc)
            loss_global = criterion_mse(pred_global, y_global)            

            loss = loss_embb + loss_mmtc + loss_urllc + loss_global

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, (x1, x2, x3, y1, y2, y3) in enumerate(val_loader):
                x_embb = x1.to(device)
                x_mmtc = x2.to(device)
                x_urllc = x3.to(device)

                y_embb = y1.to(device)
                y_mmtc = y2.to(device)
                y_urllc = y3.to(device)

                pred_global, pred_embb, pred_mmtc, pred_urllc = model(x_embb, x_mmtc, x_urllc)
            
                y_global = y_embb + y_mmtc + y_urllc
                loss_embb = criterion_mse(pred_embb, y_embb)
                loss_mmtc = criterion_huber(pred_mmtc, y_mmtc)
                loss_urllc = criterion_huber(pred_urllc, y_urllc)
                loss_global = criterion_mse(pred_global, y_global)            

                pred_t = pred_global[:, 1:]
                pred_t_minus_1 = pred_global[:, 0:-1]

                loss_smooth = criterion_mse(pred_t, pred_t_minus_1)
                loss = loss_embb + loss_mmtc + loss_urllc + loss_global

                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss:.4f}")

    all_pred = []
    all_target = []
    embb_pred = []
    embb_target = []
    mmtc_pred = []
    mmtc_target = []
    urllc_pred = []
    urllc_target = []   

    model.eval()
    with torch.no_grad():
        for i, (x1, x2, x3, y1, y2, y3) in enumerate(test_loader):
            x_embb = x1.to(device)
            x_mmtc = x2.to(device)
            x_urllc = x3.to(device)

            y_embb = y1.to(device)
            y_mmtc = y2.to(device)
            y_urllc = y3.to(device)

            pred_global, pred_embb, pred_mmtc, pred_urllc = model(x_embb, x_mmtc, x_urllc)
            y_global = y1.to(device) + y2.to(device) + y3.to(device)

            p_np = pred_global.cpu().numpy()
            y_np = y_global.cpu().numpy()

            embb_p_np = pred_embb.cpu().numpy()
            embb_y_np = y_embb.cpu().numpy()
            mmtc_p_np = pred_mmtc.cpu().numpy()
            mmtc_y_np = y_mmtc.cpu().numpy()
            urllc_p_np = pred_urllc.cpu().numpy()
            urllc_y_np = y_urllc.cpu().numpy()


            all_pred.append(p_np)
            all_target.append(y_np)
            embb_pred.append(embb_p_np)
            embb_target.append(embb_y_np)
            mmtc_pred.append(mmtc_p_np)
            mmtc_target.append(mmtc_y_np)
            urllc_pred.append(urllc_p_np)
            urllc_target.append(urllc_y_np)
    
    print("Data collect complete!")

    prediction = np.concatenate(all_pred, axis=0)
    target = np.concatenate(all_target, axis=0)
    embb_prediction = np.concatenate(embb_pred, axis=0)
    embb_target = np.concatenate(embb_target, axis=0)
    mmtc_prediction = np.concatenate(mmtc_pred, axis=0)
    mmtc_target = np.concatenate(mmtc_target, axis=0)
    urllc_prediction = np.concatenate(urllc_pred, axis=0)
    urllc_target = np.concatenate(urllc_target, axis=0)

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs[0, 0].plot(target, label='Target')
    axs[0, 0].plot(prediction, label='Prediction', alpha=0.7)
    axs[0, 0].set_title('Global')
    axs[0, 0].legend()

    axs[0, 1].plot(embb_target, label='Target')
    axs[0, 1].plot(embb_prediction, label='Prediction', alpha=0.7)
    axs[0, 1].set_title('eMBB Slice')
    axs[0, 1].legend()

    axs[1, 0].plot(mmtc_target, label='Target')
    axs[1, 0].plot(mmtc_prediction, label='Prediction', alpha=0.7)
    axs[1, 0].set_title('mMTC Slice')
    axs[1, 0].legend()

    axs[1, 1].plot(urllc_target, label='Target')
    axs[1, 1].plot(urllc_prediction, label='Prediction', alpha=0.7)
    axs[1, 1].set_title('uRLLLC Slice')
    axs[1, 1].legend()

    plt.tight_layout()

    result_dir = './results'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        print(f"Created directory: {result_dir}")

    save_path = os.path.join(result_dir, f'prediction_comparison_lr{learning_rate}_ep{num_epochs}_seq{sequenceLength}.png')
    
    plt.savefig(save_path, dpi=300)
    print(f"✅ Image saved to: {save_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    loss_save_path = os.path.join(result_dir, f'loss_curve_lr{learning_rate}_ep{num_epochs}_seq{sequenceLength}.png')
    plt.savefig(loss_save_path, dpi=300)
    print(f"✅ Loss curve saved to: {loss_save_path}")

    plt.show()

if __name__ == "__main__":
    main()