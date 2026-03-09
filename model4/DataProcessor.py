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

class TrafficDataset(Dataset):
    def __init__(self, file_path, processor, scalerX, scalerY, mode='train'):
        # 獨立處理單一檔案
        self.X, self.y = processor.process_file(file_path, scalerX, scalerY)
        self.data_len = len(self.X)

        # 定義切割比例
        train_ratio = 0.8
        val_ratio = 0.1
        
        if mode == 'train':
            self.start_idx = 0
            self.end_idx = int(self.data_len * train_ratio)
        elif mode == 'val':
            self.start_idx = int(self.data_len * train_ratio)
            self.end_idx = int(self.data_len * (train_ratio + val_ratio))
        elif mode == 'test':
            self.start_idx = int(self.data_len * (train_ratio + val_ratio))
            self.end_idx = self.data_len
        else:
            raise ValueError("Invalid mode. Must be 'train', 'val', or 'test'.")

    def __len__(self):
        return self.end_idx - self.start_idx
    
    def __getitem__(self, idx):
        real_idx = self.start_idx + idx
        x = torch.tensor(self.X[real_idx], dtype=torch.float32)
        y = torch.tensor(self.y[real_idx], dtype=torch.float32)
        return x, y