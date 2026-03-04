import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# 設定包含所有資料的根目錄
root_folder = "./tr0"  
output_folder = "./output_plots"  # 輸出圖片的資料夾

# 建立輸出資料夾
os.makedirs(output_folder, exist_ok=True)

# 遞迴取得所有 CSV 檔案
csv_files = glob.glob(os.path.join(root_folder, "**", "*.csv"), recursive=True)

# 要繪製的特徵指標
metrics_to_plot = [
    "tx_brate downlink [Mbps]",
    "tx_pkts downlink",
    "sum_requested_prbs",
    "sum_granted_prbs"
]

for csv_path in csv_files:
    filename = os.path.basename(csv_path)
    name_without_ext = os.path.splitext(filename)[0]
    
    # 保持原本的資料夾結構
    rel_dir = os.path.dirname(os.path.relpath(csv_path, root_folder))
    target_dir = os.path.join(output_folder, rel_dir)
    os.makedirs(target_dir, exist_ok=True)
    
    try:
        df = pd.read_csv(csv_path)
        x = range(len(df))  # 第幾筆資料（0, 1, 2, ...）

        # 檢查檔案是否為空
        if df.empty:
            print(f"⚠️  {filename} 是空檔案，跳過。")
            continue

        fig, axs = plt.subplots(len(metrics_to_plot), 1, figsize=(20, 4 * len(metrics_to_plot)))
        fig.suptitle(f"Metrics for {name_without_ext}", fontsize=16)

        plotted_any = False

        for i, metric in enumerate(metrics_to_plot):
            ax = axs[i]
            if metric in df.columns:
                ax.plot(x, df[metric], linewidth=1.2, color=f"C{i}")
                ax.set_ylabel(metric)
                ax.grid(True, alpha=0.3)
                plotted_any = True
            else:
                ax.text(0.5, 0.5, f"Metric '{metric}' not found", ha='center', va='center', transform=ax.transAxes)
                ax.set_ylabel(metric)
                ax.grid(True, alpha=0.3)
            
            if i == len(metrics_to_plot) - 1:
                ax.set_xlabel("Sample Index")

        if plotted_any:
            plt.tight_layout()
            fig.subplots_adjust(top=0.95) # 留空間給 suptitle
            output_path = os.path.join(target_dir, f"{name_without_ext}_metrics.png")
            plt.savefig(output_path, dpi=150)
            print(f"✅ 已輸出：{output_path}")
        else:
            print(f"⚠️  {filename} 找不到任何指定的指標欄位，跳過繪圖。")

        plt.close(fig) # 確保關閉圖表釋放記憶體
    
    except Exception as e:
        print(f"❌ 處理 {csv_path} 時發生錯誤：{e}")

print("\n全部完成！")