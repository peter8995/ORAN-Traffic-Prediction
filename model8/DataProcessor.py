from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
from torch.utils.data import Dataset, Sampler


SLICE_DIR_MAP: Dict[str, str] = {
    "embb": "embb",
    "mmtc": "mtc",
    "urllc": "urllc",
}

FEATURE_ORDER: List[str] = [
    "dl_buffer",
    "tx_brate",
    "rx_brate",
    "ul_sinr",
    "dl_mcs",
    "ul_mcs",
    "phr",
    "tx_pkts",
    "rx_pkts",
    "dl_cqi",
    "rx_errors_ul",
    "dl_n_samples",
    "ul_n_samples",
]

FEATURE_CANDIDATES: Dict[str, List[str]] = {
    "dl_buffer": ["dlbufferbytes", "dlbuffer"],
    "ul_buffer": ["ulbufferbytes", "ulbuffer"],
    "tx_brate": ["txbratedownlinkmbps", "txbrate", "txbrateMbps"],
    "rx_brate": ["rxbrateuplinkmbps", "rxbrate", "rxbrateMbps"],
    "ul_sinr": ["ulsinr"],
    "dl_mcs": ["dlmcs"],
    "ul_mcs": ["ulmcs"],
    "phr": ["phr"],
    "tx_pkts": ["txpktsdownlink", "txpkts"],
    "rx_pkts": ["rxpktsuplink", "rxpkts"],
    "ul_rssi": ["ulrssi"],
    "dl_cqi": ["dlcqi"],
    "rx_errors_ul": ["rxerrorsuplink", "rxerrorsuplinkpercent"],
    "dl_n_samples": ["dlnsamples"],
    "ul_n_samples": ["ulnsamples"],
}

TARGET_CANDIDATES: List[str] = ["sumgrantedprbs"]
TIMESTAMP_CANDIDATES: List[str] = ["timestamp"]

TRIAL_RANGE_PATTERN = re.compile(r"^tr(\d+)-(\d+)$")
TRIAL_SINGLE_PATTERN = re.compile(r"^tr(\d+)$")


def _normalize_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


@dataclass
class CsvRecord:
    csv_path: Path
    features_raw: np.ndarray
    target_raw: np.ndarray
    features_scaled: np.ndarray | None = None
    target_scaled: np.ndarray | None = None


class TrafficDataset(Dataset):
    def __init__(
        self,
        records: Sequence[CsvRecord],
        sequence_length: int,
        horizon: int,
        chunk_size: int,
    ) -> None:
        self.sequence_length = sequence_length
        self.horizon = horizon
        self.chunk_size = chunk_size

        self.features_per_csv: List[np.ndarray] = []
        self.target_per_csv: List[np.ndarray] = []
        self.csv_paths: List[str] = []
        self.window_counts: List[int] = []
        self.csv_offsets: List[int] = []

        offset = 0
        for record in records:
            if record.features_scaled is None or record.target_scaled is None:
                raise ValueError(f"Record has not been scaled: {record.csv_path}")
            num_windows = record.features_scaled.shape[0] - sequence_length - horizon + 1
            if num_windows <= 0:
                continue
            self.features_per_csv.append(record.features_scaled.astype(np.float32))
            self.target_per_csv.append(record.target_scaled.astype(np.float32))
            self.csv_paths.append(str(record.csv_path))
            self.window_counts.append(num_windows)
            self.csv_offsets.append(offset)
            offset += num_windows

        self.total_windows = offset
        if self.total_windows <= 0:
            raise ValueError("No valid sliding windows were generated.")

        self.index_to_csv = np.empty(self.total_windows, dtype=np.int32)
        self.index_to_start = np.empty(self.total_windows, dtype=np.int32)

        cursor = 0
        for csv_idx, count in enumerate(self.window_counts):
            self.index_to_csv[cursor : cursor + count] = csv_idx
            self.index_to_start[cursor : cursor + count] = np.arange(count, dtype=np.int32)
            cursor += count

        self.chunk_start_flags = (self.index_to_start % self.chunk_size) == 0
        self.chunk_ranges = self._build_chunk_ranges()

    def _build_chunk_ranges(self) -> List[Tuple[int, int]]:
        ranges: List[Tuple[int, int]] = []
        for offset, count in zip(self.csv_offsets, self.window_counts):
            for start in range(0, count, self.chunk_size):
                global_start = offset + start
                global_end = offset + min(start + self.chunk_size, count)
                ranges.append((global_start, global_end))
        return ranges

    def __len__(self) -> int:
        return self.total_windows

    def __getitem__(self, index: int):
        csv_idx = int(self.index_to_csv[index])
        window_start = int(self.index_to_start[index])

        seq_start = window_start
        seq_end = seq_start + self.sequence_length
        target_start = seq_end
        target_end = target_start + self.horizon

        features = self.features_per_csv[csv_idx][seq_start:seq_end]
        target = self.target_per_csv[csv_idx][target_start:target_end, 0]
        chunk_start = self.chunk_start_flags[index]

        x = torch.from_numpy(np.ascontiguousarray(features)).float()
        y = torch.from_numpy(np.ascontiguousarray(target)).float()
        flag = torch.tensor(chunk_start, dtype=torch.bool)
        return x, y, flag


class ChunkShuffleSampler(Sampler[int]):
    def __init__(self, dataset: TrafficDataset, shuffle: bool = True, seed: int = 42) -> None:
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        chunks = list(self.dataset.chunk_ranges)
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(chunks)
        self.epoch += 1
        for start, end in chunks:
            for idx in range(start, end):
                yield idx

    def __len__(self) -> int:
        return len(self.dataset)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


@dataclass
class PreparedData:
    train_dataset: TrafficDataset
    val_dataset: TrafficDataset
    test_dataset: TrafficDataset
    feature_scaler: RobustScaler
    target_scaler: RobustScaler
    train_csv_paths: List[str]
    val_csv_paths: List[str]
    test_csv_paths: List[str]


class DataProcessor:
    def __init__(
        self,
        data_root: str | Path,
        slice_type: str,
        sequence_length: int,
        horizon: int,
        chunk_size: int,
        seed: int = 42,
    ) -> None:
        if slice_type not in SLICE_DIR_MAP:
            raise ValueError(f"Unsupported slice_type: {slice_type}")
        self.data_root = Path(data_root)
        self.slice_type = slice_type
        self.sequence_length = sequence_length
        self.horizon = horizon
        self.chunk_size = chunk_size
        self.seed = seed

    @staticmethod
    def expand_trial_dirs(tokens: Sequence[str]) -> List[str]:
        expanded: List[str] = []
        for token in tokens:
            range_match = TRIAL_RANGE_PATTERN.match(token)
            single_match = TRIAL_SINGLE_PATTERN.match(token)
            if range_match:
                start = int(range_match.group(1))
                end = int(range_match.group(2))
                if start > end:
                    raise ValueError(f"Invalid trial range: {token}")
                expanded.extend([f"tr{i}" for i in range(start, end + 1)])
            elif single_match:
                expanded.append(token)
            else:
                raise ValueError(f"Invalid trial token: {token}")

        unique_trials = sorted(set(expanded), key=lambda x: int(x[2:]))
        return unique_trials

    def _resolve_columns(self, df: pd.DataFrame) -> Tuple[List[str], str, str | None]:
        normalized: Dict[str, List[str]] = {}
        for col in df.columns:
            key = _normalize_col(col)
            normalized.setdefault(key, []).append(col)

        def pick_column(candidates: Sequence[str], name_for_error: str) -> str:
            for cand in candidates:
                if cand in normalized and normalized[cand]:
                    return normalized[cand][0]
            for norm_name, cols in normalized.items():
                if any(cand in norm_name for cand in candidates):
                    return cols[0]
            raise ValueError(f"Column for `{name_for_error}` not found.")

        feature_cols = [
            pick_column(FEATURE_CANDIDATES[name], name_for_error=name) for name in FEATURE_ORDER
        ]
        target_col = pick_column(TARGET_CANDIDATES, name_for_error="sum_granted_prbs")
        timestamp_col = None
        for cand in TIMESTAMP_CANDIDATES:
            if cand in normalized and normalized[cand]:
                timestamp_col = normalized[cand][0]
                break
        return feature_cols, target_col, timestamp_col

    @staticmethod
    def _to_float_array(series: pd.Series) -> np.ndarray:
        arr = pd.to_numeric(series, errors="coerce").astype(np.float32)
        if arr.isna().any():
            arr = arr.ffill().bfill().fillna(0.0)
        return arr.to_numpy(dtype=np.float32)

    def _load_single_csv(self, csv_path: Path) -> CsvRecord:
        df = pd.read_csv(csv_path, skipinitialspace=True)
        feature_cols, target_col, timestamp_col = self._resolve_columns(df)

        if timestamp_col is not None:
            df = df.sort_values(timestamp_col, kind="stable")

        feature_arrays = [self._to_float_array(df[col]) for col in feature_cols]
        target_array = self._to_float_array(df[target_col]).reshape(-1, 1)
        features = np.stack(feature_arrays, axis=1)

        return CsvRecord(
            csv_path=csv_path,
            features_raw=features.astype(np.float32),
            target_raw=target_array.astype(np.float32),
        )

    def collect_csv_paths(self, trial_dirs: Sequence[str]) -> List[Path]:
        slice_dir = SLICE_DIR_MAP[self.slice_type]
        csv_paths: List[Path] = []

        for trial_dir in trial_dirs:
            trial_path = self.data_root / trial_dir
            if not trial_path.exists():
                print(f"[WARN] Trial directory not found, skip: {trial_path}")
                continue
            pattern = f"exp*/bs*/{slice_dir}/*_metrics.csv"
            found = sorted(trial_path.glob(pattern))
            csv_paths.extend(found)

        csv_paths = sorted(csv_paths)
        if not csv_paths:
            raise FileNotFoundError(
                f"No CSV found for slice `{self.slice_type}` in trials: {trial_dirs}"
            )
        return csv_paths

    @staticmethod
    def _fit_scalers(train_records: Sequence[CsvRecord]) -> Tuple[RobustScaler, RobustScaler]:
        if not train_records:
            raise ValueError("Empty train_records.")
        train_features = np.concatenate([r.features_raw for r in train_records], axis=0)
        train_target = np.concatenate([r.target_raw for r in train_records], axis=0)

        feature_scaler = RobustScaler()
        target_scaler = RobustScaler()
        feature_scaler.fit(train_features)
        target_scaler.fit(train_target)
        return feature_scaler, target_scaler

    @staticmethod
    def _transform_records(
        records: Sequence[CsvRecord],
        feature_scaler: RobustScaler,
        target_scaler: RobustScaler,
    ) -> None:
        for record in records:
            record.features_scaled = feature_scaler.transform(record.features_raw).astype(np.float32)
            record.target_scaled = target_scaler.transform(record.target_raw).astype(np.float32)

    @staticmethod
    def _split_train_val(
        train_records: Sequence[CsvRecord], val_split: float, seed: int
    ) -> Tuple[List[CsvRecord], List[CsvRecord]]:
        if not 0.0 < val_split < 1.0:
            raise ValueError("--val_split must be in (0, 1).")
        if len(train_records) < 2:
            raise ValueError("Need at least two train CSV files to split train/val by CSV.")

        records = list(train_records)
        rng = random.Random(seed)
        rng.shuffle(records)

        val_count = max(1, int(len(records) * val_split))
        if val_count >= len(records):
            val_count = len(records) - 1

        val_records = records[:val_count]
        new_train_records = records[val_count:]
        return new_train_records, val_records

    @staticmethod
    def _log_non_zero_ratios(records: Sequence[CsvRecord]) -> None:
        non_zero = np.zeros(len(FEATURE_ORDER), dtype=np.float64)
        total = 0
        for record in records:
            values = record.features_raw
            non_zero += np.count_nonzero(values, axis=0)
            total += values.shape[0]
        if total == 0:
            return
        print("[INFO] Train feature non-zero ratio:")
        for idx, name in enumerate(FEATURE_ORDER):
            ratio = non_zero[idx] / total
            print(f"  - {name}: {ratio:.4f}")

    def prepare(
        self,
        train_dirs: Sequence[str],
        test_dirs: Sequence[str],
        val_split: float,
    ) -> PreparedData:
        expanded_train_dirs = self.expand_trial_dirs(train_dirs)
        expanded_test_dirs = self.expand_trial_dirs(test_dirs)

        train_csv_paths = self.collect_csv_paths(expanded_train_dirs)
        test_csv_paths = self.collect_csv_paths(expanded_test_dirs)

        print(
            f"[INFO] Slice={self.slice_type} | "
            f"train_csv={len(train_csv_paths)} | test_csv={len(test_csv_paths)}"
        )

        print("[INFO] Loading train CSV files...")
        train_records_all = [self._load_single_csv(path) for path in train_csv_paths]
        print("[INFO] Loading test CSV files...")
        test_records = [self._load_single_csv(path) for path in test_csv_paths]

        train_records, val_records = self._split_train_val(train_records_all, val_split, self.seed)
        print(f"[INFO] Split train/val by CSV: train={len(train_records)} val={len(val_records)}")

        self._log_non_zero_ratios(train_records)

        feature_scaler, target_scaler = self._fit_scalers(train_records)
        self._transform_records(train_records, feature_scaler, target_scaler)
        self._transform_records(val_records, feature_scaler, target_scaler)
        self._transform_records(test_records, feature_scaler, target_scaler)

        train_dataset = TrafficDataset(
            train_records,
            sequence_length=self.sequence_length,
            horizon=self.horizon,
            chunk_size=self.chunk_size,
        )
        val_dataset = TrafficDataset(
            val_records,
            sequence_length=self.sequence_length,
            horizon=self.horizon,
            chunk_size=self.chunk_size,
        )
        test_dataset = TrafficDataset(
            test_records,
            sequence_length=self.sequence_length,
            horizon=self.horizon,
            chunk_size=self.chunk_size,
        )

        return PreparedData(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            feature_scaler=feature_scaler,
            target_scaler=target_scaler,
            train_csv_paths=[str(r.csv_path) for r in train_records],
            val_csv_paths=[str(r.csv_path) for r in val_records],
            test_csv_paths=[str(r.csv_path) for r in test_records],
        )
