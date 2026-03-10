import pandas as pd
import os
import torch
from torch.utils.data import Dataset
import numpy as np


class DataProcessor:
    def __init__(self, sequenceLength):
        self.sequenceLength = sequenceLength

        self.features = ['dl_buffer [bytes]', 'ul_buffer [bytes]', 
            'tx_brate downlink [Mbps]', 'rx_brate uplink [Mbps]', 
            'ul_sinr', 'dl_mcs', 'ul_mcs', 'phr', 
            'tx_pkts downlink', 'rx_pkts uplink', 'ul_rssi']
        self.target = 'sum_granted_prbs'
    
    def load_and_clean(self, file_path):
        df = pd.read_csv(file_path)
        df = df.dropna(axis=1, how='all')
        df = df.fillna(0)
        
        #sampling_interval = 0.25
        #df['Relative_Time'] = np.arange(len(df)) * sampling_interval
        
        return df

    def process_file(self, file_path, scalerX, scalerY):
        df = self.load_and_clean(file_path)
        
        # Avoid crashing when processing extremely small files that can't fulfill 1 sequence length sliding window.
        if len(df) <= self.sequenceLength:
            return np.array([]), np.array([])
            
        dataX = df[self.features].values
        dataY = df[[self.target]].values

        dataXScaled = scalerX.transform(dataX)
        dataYScaled = scalerY.transform(dataY)

        X_windows, y_windows = [], []
        for i in range(len(dataXScaled) - self.sequenceLength):
            window_X = dataXScaled[i:i+self.sequenceLength]
            window_y = dataYScaled[i+self.sequenceLength]

            X_windows.append(window_X)
            y_windows.append(window_y)

        return np.array(X_windows), np.array(y_windows)

    def accumulate_files(self, directories, slice_type=None):
        """Recursively fetches all *metrics.csv filepaths given a specific list of directory."""
        csv_files = []
        
        target_folder = slice_type
        if slice_type == 'mmtc':
            target_folder = 'mtc'
            
        for directory in directories:
            # os.walk is handy for exploring nested subdirectories
            for root, dirs, files in os.walk(directory):
                # Filter by slice_type if provided by matching the current parent folder name
                path_parts = root.split(os.sep)
                if target_folder and target_folder not in path_parts:
                    continue
                    
                for file in files:
                    if file.endswith("metrics.csv"):
                        csv_files.append(os.path.join(root, file))
        return csv_files

    def process_directories(self, directories, scalerX, scalerY, slice_type=None):
        files = self.accumulate_files(directories, slice_type)
        print(f"Discovered {len(files)} metrics.csv files for slice '{slice_type}' in provided directories.")
        all_x, all_y = [], []

        for f in files:
            file_x, file_y = self.process_file(f, scalerX, scalerY)
            if len(file_x) > 0:
                all_x.append(file_x)
                all_y.append(file_y)

        if len(all_x) == 0:
            return np.array([]), np.array([])

        final_x = np.concatenate(all_x, axis=0)
        final_y = np.concatenate(all_y, axis=0)
        return final_x, final_y

class TrafficDataset(Dataset):
    def __init__(self, directories, processor, scalerX, scalerY, slice_type=None):
        # 透過 process_directories 可以確保獨立檔案處理不會有 Time-Series Bleeding 交疊的情形
        self.X, self.y = processor.process_directories(directories, scalerX, scalerY, slice_type)
        self.data_len = len(self.X)
        
        # Train / Validation splitting is now dynamically generated via torch slicing within train.py, 
        # testing relies exclusively on this root instantiation.

    def __len__(self):
        return self.data_len
    
    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, y