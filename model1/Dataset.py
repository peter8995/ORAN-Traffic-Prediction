import os
import pandas as pd

# === 設定你的目錄 ===
# 依實際路徑修改（Trial5 資料夾下）
trial5_path = "./Dataset/Trial7"

paths = {
    "RAW": os.path.join(trial5_path, "Raw"),
    "CLEAN": os.path.join(trial5_path),
}

datasets = ["embb", "urllc", "mmtc", "null"]

# 儲存各子資料夾的 KPI 結構
kpi_columns = {}

for dtype, dpath in paths.items():
    if not os.path.exists(dpath):
        print(f"[!] 找不到資料夾：{dpath}")
        continue

    print(f"\n=== 📂 資料來源: {dtype} ({dpath}) ===")

    for name in datasets:
        csv_files = [f for f in os.listdir(dpath) if name.lower() in f.lower() and f.endswith(".csv")]
        if not csv_files:
            print(f"[!] {dtype} 資料夾沒找到與 {name} 相關的 CSV 檔")
            continue

        file_path = os.path.join(dpath, csv_files[0])
        df = pd.read_csv(file_path, nrows=1)
        key = f"{dtype}_{name.upper()}"
        kpi_columns[key] = list(df.columns)

        print(f"\n--- {key} ---")
        print(f"檔案：{csv_files[0]}")
        print(f"KPI 數量：{len(df.columns)}")
        print(df.columns.tolist())

# === KPI 結構比較 ===
print("\n\n=======================")
print("📊 KPI 結構比較")
print("=======================")

# 找出 RAW 與 LOGS 各自的所有 KPI 集合
raw_kpis = set().union(*[set(v) for k, v in kpi_columns.items() if k.startswith("RAW")])
clean_kpis = set().union(*[set(v) for k, v in kpi_columns.items() if k.startswith("CLEAN")])

common_kpis = raw_kpis & clean_kpis
raw_only = raw_kpis - clean_kpis
clean_only = clean_kpis - raw_kpis

print(f"\n🔸 共同 KPI ({len(common_kpis)}):")
print(sorted(common_kpis))

print(f"\n🔹 RAW 專屬 KPI ({len(raw_only)}):")
print(sorted(raw_only))

print(f"\n🔸 CLEAN 專屬 KPI ({len(clean_only)}):")
print(sorted(clean_only))
