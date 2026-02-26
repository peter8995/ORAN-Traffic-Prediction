import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# 設定 CSV 檔案所在的資料夾路徑
csv_folder = "./Trial7"  # 請修改成你的資料夾路徑
output_folder = "./output_plots"  # 輸出圖片的資料夾

# 建立輸出資料夾
os.makedirs(output_folder, exist_ok=True)

# 取得所有 CSV 檔案
csv_files = glob.glob(os.path.join(csv_folder, "*.csv"))

for csv_path in csv_files:
    filename = os.path.basename(csv_path)
    name_without_ext = os.path.splitext(filename)[0]
    
    try:
        df = pd.read_csv(csv_path)
        
        if "sum_requested_prbs" not in df.columns:
            print(f"⚠️  {filename} 中找不到 'sum_requested_prbs' 欄位，跳過。")
            continue
        
        x = range(len(df))  # 第幾筆資料（0, 1, 2, ...）
        
        fig, ax = plt.subplots(figsize=(20, 4))  # 寬20、高4，可自行調整
        ax.plot(x, df["sum_requested_prbs"], linewidth=0.8)
        ax.set_title(name_without_ext, fontsize=12)
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("sum_requested_prbs")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = os.path.join(output_folder, f"{name_without_ext}.png")
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        print(f"✅ 已輸出：{output_path}")
    
    except Exception as e:
        print(f"❌ 處理 {filename} 時發生錯誤：{e}")

print("\n全部完成！")