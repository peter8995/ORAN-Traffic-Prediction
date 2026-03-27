import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class DataProcessor:
    def __init__(self, sequenceLength=15):
        self.sequenceLength = sequenceLength

        self.features = ['dl_buffer [bytes]', 'ul_buffer [bytes]',
            'tx_brate downlink [Mbps]', 'rx_brate uplink [Mbps]',
            'ul_sinr', 'dl_mcs', 'ul_mcs', 'phr',
            'tx_pkts downlink', 'rx_pkts uplink', 'ul_rssi']
        self.target = 'sum_granted_prbs'

        # Spike detection parameters
        self.m = 120       # short-term volatility window (~1 min)
        self.L = 1200      # long-term baseline window (~10 min)

        # Plan B: global threshold + sudden-change condition
        self.k = 2.0       # global value threshold: mean + k*std
        self.j = 2.0       # diff threshold: mean + j*std
        self.global_mean = 0.0
        self.global_std = 1.0
        self.diff_mean = 0.0
        self.diff_std = 1.0

        # Rolling temporal features
        self.rolling_features = [
            'target_rolling_mean_m', 'target_rolling_std_m', 'target_rolling_max_m',
            'target_rolling_mean_L', 'target_rolling_q90_L', 'target_diff'
        ]

    def fit_spike_params(self, train_files):
        """Compute global target stats and diff stats from all training files (Plan B)."""
        all_targets = []
        all_diffs = []
        for f in train_files:
            df = self.load_and_clean(f)
            if len(df) < 2:
                continue
            vals = df[self.target].values.astype(float)
            all_targets.append(vals)
            all_diffs.append(np.abs(np.diff(vals)))

        all_targets = np.concatenate(all_targets)
        all_diffs = np.concatenate(all_diffs)

        self.global_mean = float(np.mean(all_targets))
        self.global_std = float(np.std(all_targets))
        self.diff_mean = float(np.mean(all_diffs))
        self.diff_std = float(np.std(all_diffs))

        tau_value = self.global_mean + self.k * self.global_std
        tau_diff = self.diff_mean + self.j * self.diff_std

        print(f"[SpikeDet-PlanB] global_mean={self.global_mean:.2f}, global_std={self.global_std:.2f}")
        print(f"[SpikeDet-PlanB] value threshold (k={self.k}): {tau_value:.2f}")
        print(f"[SpikeDet-PlanB] diff_mean={self.diff_mean:.2f}, diff_std={self.diff_std:.2f}")
        print(f"[SpikeDet-PlanB] diff threshold (j={self.j}): {tau_diff:.2f}")
        print(f"[SpikeDet-PlanB] Computed from {len(all_targets)} data points across {len(train_files)} files.")

    def calculate_spike_labels(self, target_data):
        """Plan B: spike = value exceeds global threshold AND sudden change."""
        vals = target_data.flatten().astype(float)

        tau_value = self.global_mean + self.k * self.global_std
        tau_diff = self.diff_mean + self.j * self.diff_std

        high_value = vals > tau_value
        diff = np.abs(np.diff(vals, prepend=vals[0]))
        sudden_change = diff > tau_diff

        s_labels = (high_value & sudden_change).astype(int)
        return s_labels

    def load_and_clean(self, file_path):
        df = pd.read_csv(file_path)
        df = df.dropna(axis=1, how='all')
        df = df.fillna(0)
        return df

    def add_rolling_features(self, df):
        """
        Add rolling temporal features computed on RAW data before scaling.
        Bridges the gap between short-term input and long-term spike definition.
        """
        target = df[self.target]

        # Short-term volatility window (m=120, ~30s)
        df['target_rolling_mean_m'] = target.rolling(window=self.m, min_periods=1).mean()
        df['target_rolling_std_m'] = target.rolling(window=self.m, min_periods=1).std().fillna(0)
        df['target_rolling_max_m'] = target.rolling(window=self.m, min_periods=1).max()

        # Long-term baseline window (L=1200, ~5min)
        df['target_rolling_mean_L'] = target.rolling(window=self.L, min_periods=1).mean()
        df['target_rolling_q90_L'] = target.rolling(window=self.L, min_periods=1).quantile(0.9)

        # First-order difference
        df['target_diff'] = target.diff().fillna(0)

        return df

    def process_file(self, file_path, scalerX, scalerY):
        """Load file, add rolling features, create sliding windows (same as model6)."""
        df = self.load_and_clean(file_path)
        if len(df) <= self.sequenceLength:
            return np.array([]), np.array([]), np.array([])

        dataY = df[[self.target]].values

        # Spike labels on RAW target before scaling
        s_labels = self.calculate_spike_labels(dataY)

        # Add rolling features (computed on raw data before scaling)
        df = self.add_rolling_features(df)

        all_features = self.features + self.rolling_features
        dataX = df[all_features].values

        dataXScaled = scalerX.transform(dataX)
        dataYScaled = scalerY.transform(dataY)

        X_windows, y_windows, s_windows = [], [], []
        for i in range(len(dataXScaled) - self.sequenceLength):
            X_windows.append(dataXScaled[i:i+self.sequenceLength])
            y_windows.append(dataYScaled[i+self.sequenceLength])
            s_windows.append(s_labels[i+self.sequenceLength])

        return np.array(X_windows), np.array(y_windows), np.array(s_windows)

    def accumulate_files(self, directories, slice_type=None):
        """Recursively find all *metrics.csv files, optionally filtered by slice."""
        csv_files = []
        target_folder = slice_type
        if slice_type == 'mmtc':
            target_folder = 'mtc'

        for directory in directories:
            for root, dirs, files in os.walk(directory):
                path_parts = root.split(os.sep)
                if target_folder and target_folder not in path_parts:
                    continue
                for file in files:
                    if file.endswith("metrics.csv"):
                        csv_files.append(os.path.join(root, file))
        return csv_files

    def process_directories(self, directories, scalerX, scalerY, slice_type=None):
        """Load all files, return concatenated windowed arrays."""
        files = self.accumulate_files(directories, slice_type)
        print(f"Discovered {len(files)} metrics.csv files for slice '{slice_type}'.")

        all_x, all_y, all_s = [], [], []
        total_raw_rows = 0
        for f in files:
            fx, fy, fs = self.process_file(f, scalerX, scalerY)
            if len(fx) > 0:
                all_x.append(fx)
                all_y.append(fy)
                all_s.append(fs)
                total_raw_rows += len(fx) + self.sequenceLength

        print(f"-> Total raw data points: {total_raw_rows}")
        if not all_x:
            return np.array([]), np.array([]), np.array([])

        return np.concatenate(all_x), np.concatenate(all_y), np.concatenate(all_s)


def parse_directory_args(args_list, base_path):
    """Expand 'tr0-4' into [base/tr0, ..., base/tr4]."""
    parsed = []
    for item in args_list:
        if '-' in item:
            prefix = ''.join(c for c in item.split('-')[0] if not c.isdigit())
            start = int(''.join(c for c in item.split('-')[0] if c.isdigit()))
            end = int(''.join(c for c in item.split('-')[1] if c.isdigit()))
            for i in range(start, end + 1):
                parsed.append(os.path.join(base_path, f"{prefix}{i}"))
        else:
            parsed.append(os.path.join(base_path, item))
    return parsed


def build_scalers(processor, train_files):
    """Fit global MinMaxScalers on all training files (including rolling features)."""
    all_X, all_Y = [], []
    all_features = processor.features + processor.rolling_features

    for f in train_files:
        df = processor.load_and_clean(f)
        if len(df) <= processor.sequenceLength:
            continue
        df = processor.add_rolling_features(df)
        all_X.append(df[all_features].values)
        all_Y.append(df[[processor.target]].values)

    scalerX = MinMaxScaler()
    scalerY = MinMaxScaler()
    scalerX.fit(np.concatenate(all_X, axis=0))
    scalerY.fit(np.concatenate(all_Y, axis=0))
    del all_X, all_Y
    return scalerX, scalerY
