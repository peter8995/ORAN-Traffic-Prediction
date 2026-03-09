import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# 設定包含所有資料的根目錄 (上一層目錄)
base_folder = "."  
output_folder = "./output_plots"  # 輸出圖片的資料夾

# 建立輸出資料夾
os.makedirs(output_folder, exist_ok=True)

# 要繪製的特徵指標
metrics_to_plot = [
    "tx_brate downlink [Mbps]",
    "tx_pkts downlink",
    "sum_requested_prbs",
    "sum_granted_prbs"
]

slice_map = {
    0.0: "eMBB",
    1.0: "MTC",
    2.0: "URLLC"
}

# 處理 tr0 到 tr27
for tr_id in range(28):
    tr_folder = f"tr{tr_id}"
    
    # 尋找特定結構下的 CSV: tr*/exp*/bs*/slices_bs*/*.csv
    search_pattern = os.path.join(base_folder, tr_folder, "exp*", "bs*", "slices_bs*", "*.csv")
    csv_files = glob.glob(search_pattern)
    
    for csv_path in csv_files:
        filename = os.path.basename(csv_path)
        name_without_ext = os.path.splitext(filename)[0]
        
        # 拆解路徑以建立預期的輸出結構
        # 例如: ./tr0/exp1/bs1/slices_bs1/xxx.csv
        path_parts = csv_path.split(os.sep)
        
        # 找出 trX 和 expX 和 slices_bsX 的名稱
        try:
            tr_name = [p for p in path_parts if p.startswith('tr')][0]
            exp_name = [p for p in path_parts if p.startswith('exp')][0]
            slices_name = [p for p in path_parts if p.startswith('slices_bs')][0]
        except IndexError:
            # 如果路徑結構不符合預期，套用預設的相對路徑保留方式
            rel_dir = os.path.dirname(os.path.relpath(csv_path, base_folder))
            target_dir = os.path.join(output_folder, rel_dir)
        else:
            # 建立目標資料夾結構: output_plots/tr0/exp1/slices_bs1/
            target_dir = os.path.join(output_folder, tr_name, exp_name, slices_name)
        
        os.makedirs(target_dir, exist_ok=True)
        
        try:
            df = pd.read_csv(csv_path)
            x = range(len(df))  # 第幾筆資料（0, 1, 2, ...）

            # 檢查檔案是否為空
            if df.empty:
                print(f"⚠️  {filename} 是空檔案，跳過。")
                continue

            # 判斷 Slice 類型
            slice_prefix = ""
            if "slice_id" in df.columns:
                unique_slices = df["slice_id"].dropna().unique()
                if len(unique_slices) > 0:
                    sid = float(unique_slices[0])
                    slice_prefix = slice_map.get(sid, f"Slice{sid}") + "_"
            
            output_filename = f"{slice_prefix}{name_without_ext}_metrics.png"

            fig, axs = plt.subplots(len(metrics_to_plot), 1, figsize=(20, 4 * len(metrics_to_plot)))
            fig.suptitle(f"Metrics for {slice_prefix}{name_without_ext}", fontsize=16)

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
                output_path = os.path.join(target_dir, output_filename)
                plt.savefig(output_path, dpi=150)
                print(f"✅ 已輸出：{output_path}")
            else:
                print(f"⚠️  {filename} 找不到任何指定的指標欄位，跳過繪圖。")

            plt.close(fig) # 確保關閉圖表釋放記憶體
        
        except Exception as e:
            print(f"❌ 處理 {csv_path} 時發生錯誤：{e}")

print("\n全部完成！")