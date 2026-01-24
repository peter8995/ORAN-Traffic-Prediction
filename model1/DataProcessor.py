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
        self.target = 'sum_requested_prbs'
    
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
    
class MultiTaskDataset(Dataset):
    def __init__(self, embb_path, mmtc_path, urllc_path, processor, scalers_embb, scalers_mmtc, scalers_urllc, mode = 'train'):
        self.X_embb, self.y_embb = processor.process_file(embb_path, scalers_embb[0], scalers_embb[1])
        self.X_mmtc, self.y_mmtc = processor.process_file(mmtc_path, scalers_mmtc[0], scalers_mmtc[1])
        self.X_urllc, self.y_urllc = processor.process_file(urllc_path, scalers_urllc[0], scalers_urllc[1])
        
        self.min_len = min(len(self.X_embb), len(self.X_mmtc), len(self.X_urllc))

        self.X_embb = self.X_embb[:self.min_len]
        self.y_embb = self.y_embb[:self.min_len]
        self.X_mmtc = self.X_mmtc[:self.min_len]
        self.y_mmtc = self.y_mmtc[:self.min_len]
        self.X_urllc = self.X_urllc[:self.min_len]
        self.y_urllc = self.y_urllc[:self.min_len]

        train_ratio = 0.8
        val_ratio = 0.1
        test_ratio = 0.1

        if mode == 'train':
            self.start_idx = 0
            self.end_idx = int(self.min_len * train_ratio)
        elif mode == 'val':
            self.start_idx = int(self.min_len * train_ratio)
            self.end_idx = int(self.min_len * (train_ratio + val_ratio))
        elif mode == 'test':
            self.start_idx = int(self.min_len * (train_ratio + val_ratio))
            self.end_idx = self.min_len
        else:
            raise ValueError("Invalid mode. Must be 'train', 'val', or 'test'.")

    def __len__(self):
        return self.end_idx - self.start_idx
    
    def __getitem__(self, idx):
        real_idx = self.start_idx + idx
        
        x1 = torch.tensor(self.X_embb[real_idx], dtype=torch.float32)
        x2 = torch.tensor(self.X_mmtc[real_idx], dtype=torch.float32)
        x3 = torch.tensor(self.X_urllc[real_idx], dtype=torch.float32)

        y1 = torch.tensor(self.y_embb[real_idx], dtype=torch.float32)
        y2 = torch.tensor(self.y_mmtc[real_idx], dtype=torch.float32)
        y3 = torch.tensor(self.y_urllc[real_idx], dtype=torch.float32)

        return x1, x2, x3, y1, y2, y3