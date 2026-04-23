from __future__ import annotations

import argparse
import io
import json
import random
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from anomaly import evaluate_anomalies, save_anomaly_overlay
from DataProcessor import ChunkShuffleSampler, DataProcessor, TrafficDataset
from model import TrafficModel


class Tee(io.TextIOBase):
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "t", "yes", "y"}:
        return True
    if lowered in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    default_data_root = (
        Path(__file__).resolve().parents[1] / "Dataset" / "colosseum-oran-coloran-dataset"
    )

    parser = argparse.ArgumentParser(description="model9 training script")

    parser.add_argument("--data_root", type=str, default=str(default_data_root))
    parser.add_argument("--train_dirs", nargs="+", required=True)
    parser.add_argument(
        "--val_dirs",
        nargs="+",
        default=None,
        help="Validation trial dirs (e.g., tr25 tr26 or tr25-26). If set, --val_split is ignored.",
    )
    parser.add_argument("--test_dirs", nargs="+", required=True)
    parser.add_argument("--slice_type", type=str, choices=["embb", "mmtc", "urllc"], required=True)
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Validation split ratio used only when --val_dirs is not provided.",
    )

    parser.add_argument("--sequence_length", type=int, default=15)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--chunk_size", type=int, default=64)

    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--scheduler", type=str, choices=["none", "cosine"], default="cosine")
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stopping patience. Set 0 to disable early stopping.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument(
        "--loss_type",
        type=str,
        choices=["mse", "huber", "weighted_mse", "weighted_huber"],
        default="mse",
    )
    parser.add_argument("--huber_delta", type=float, default=1.0)
    parser.add_argument("--weight_alpha", type=float, default=4.0)
    parser.add_argument("--weight_quantile", type=float, default=0.9)
    parser.add_argument("--weight_upper_quantile", type=float, default=0.99)
    parser.add_argument("--lambda_smooth", type=float, default=0.0)
    parser.add_argument("--lambda_ref_iqr", type=float, default=104.0)
    parser.add_argument("--disable_scale_aware_lambda", action="store_true")

    parser.add_argument("--rnn_type", type=str, choices=["lstm", "bilstm"], default="bilstm")
    parser.add_argument("--lstm_hidden", type=int, default=64)
    parser.add_argument("--lstm_layers", type=int, default=2)
    parser.add_argument("--lstm_dropout", type=float, default=0.2)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument(
        "--include_target_history",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
    )
    parser.add_argument(
        "--readout",
        type=str,
        choices=["mean", "attention", "gated"],
        default="mean",
    )
    parser.add_argument("--bilstm_input_proj_dim", type=int, default=16)

    parser.add_argument("--fft_hidden", type=int, default=64)
    parser.add_argument("--fft_n_heads", type=int, default=4)
    parser.add_argument("--fft_n_layers", type=int, default=2)
    parser.add_argument("--fft_dim_feedforward", type=int, default=256)
    parser.add_argument("--fft_dropout", type=float, default=0.1)
    parser.add_argument("--fft_readout", type=str, choices=["mean", "cls", "last"], default="mean")

    parser.add_argument(
        "--adj_type",
        type=str,
        choices=["binary_selfloop", "binary_noselfloop", "correlation"],
        default="binary_selfloop",
    )
    parser.add_argument("--adj_corr_threshold", type=float, default=0.3)
    parser.add_argument("--gat_hidden", type=int, default=64)
    parser.add_argument("--gat_heads", type=int, default=4)
    parser.add_argument("--gat_layers", type=int, default=2)
    parser.add_argument("--gat_dropout", type=float, default=0.1)
    parser.add_argument("--gat_input_proj_dim", type=int, default=32)
    parser.add_argument(
        "--gat_head_merge",
        type=str,
        choices=["mean", "concat"],
        default="concat",
        help="How non-final GAT layers merge attention heads.",
    )
    parser.add_argument(
        "--gat_final_head_merge",
        type=str,
        choices=["mean", "concat"],
        default="mean",
        help="How the final GAT layer merges attention heads.",
    )

    parser.add_argument("--chebyshev_k", type=float, default=3.0)
    parser.add_argument(
        "--anomaly_error_mode",
        type=str,
        choices=["signed", "abs", "both"],
        default="both",
    )
    parser.add_argument("--anomaly_extra_k", nargs="*", type=float, default=[2.0, 3.0])

    args = parser.parse_args()
    if args.weight_alpha < 0:
        parser.error("--weight_alpha must be >= 0.")
    if not (0.0 <= args.weight_quantile <= 1.0):
        parser.error("--weight_quantile must be in [0, 1].")
    if not (0.0 <= args.weight_upper_quantile <= 1.0):
        parser.error("--weight_upper_quantile must be in [0, 1].")
    if args.weight_quantile >= args.weight_upper_quantile:
        parser.error("--weight_quantile must be < --weight_upper_quantile.")
    if args.val_dirs is None and not (0.0 < args.val_split < 1.0):
        parser.error("--val_split must be in (0, 1) when --val_dirs is not provided.")
    if args.huber_delta <= 0:
        parser.error("--huber_delta must be > 0.")
    if args.chebyshev_k <= 0:
        parser.error("--chebyshev_k must be > 0.")
    if args.adj_type == "correlation" and not (0.0 < args.adj_corr_threshold < 1.0):
        parser.error("--adj_corr_threshold must be in (0, 1) when --adj_type=correlation.")
    if args.sequence_length <= 0:
        parser.error("--sequence_length must be > 0.")
    if args.horizon <= 0:
        parser.error("--horizon must be > 0.")
    if args.gat_head_merge == "concat" and args.gat_hidden % args.gat_heads != 0:
        parser.error("--gat_hidden must be divisible by --gat_heads when --gat_head_merge=concat.")
    if args.gat_final_head_merge == "concat" and args.hidden_dim % args.gat_heads != 0:
        parser.error(
            "--hidden_dim must be divisible by --gat_heads when --gat_final_head_merge=concat."
        )

    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def smooth_consistency_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    chunk_start_flags: torch.Tensor,
) -> torch.Tensor:
    if y_pred.shape[0] < 2:
        return y_pred.new_tensor(0.0)

    pred_diff = y_pred[1:] - y_pred[:-1]
    true_diff = y_true[1:] - y_true[:-1]
    valid_pairs = ~chunk_start_flags[1:]

    if not torch.any(valid_pairs):
        return y_pred.new_tensor(0.0)

    valid_pairs = valid_pairs.unsqueeze(-1).expand_as(pred_diff)
    diff = torch.abs(pred_diff - true_diff)
    return diff[valid_pairs].mean()


def resolve_lambda_smooth(
    lambda_base: float,
    target_scaler,
    disable_scale_aware: bool,
    reference_iqr: float,
) -> tuple[float, Dict[str, float]]:
    if disable_scale_aware:
        return float(lambda_base), {
            "enabled": 0.0,
            "base_lambda": float(lambda_base),
            "effective_lambda": float(lambda_base),
            "target_iqr": float(target_scaler.scale_[0]),
            "reference_iqr": float(reference_iqr),
            "multiplier": 1.0,
        }

    reference_iqr = float(reference_iqr)
    target_iqr = float(target_scaler.scale_[0])
    multiplier = float(np.sqrt(reference_iqr / max(target_iqr, 1e-8)))
    effective = float(lambda_base * multiplier)

    return effective, {
        "enabled": 1.0,
        "base_lambda": float(lambda_base),
        "effective_lambda": effective,
        "target_iqr": target_iqr,
        "reference_iqr": reference_iqr,
        "multiplier": multiplier,
    }


class RegressionLoss(nn.Module):
    def __init__(self, base_loss: str, huber_delta: float = 1.0) -> None:
        super().__init__()
        if base_loss not in {"mse", "huber"}:
            raise ValueError(f"Unsupported base_loss: {base_loss}")
        self.base_loss = base_loss
        self.huber_delta = float(huber_delta)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.base_loss == "mse":
            return F.mse_loss(pred, target)
        return F.huber_loss(pred, target, delta=self.huber_delta)


class WeightedRegressionLoss(nn.Module):
    def __init__(
        self,
        base_loss: str,
        q_low: float,
        q_high: float,
        alpha: float,
        huber_delta: float = 1.0,
    ) -> None:
        super().__init__()
        if base_loss not in {"mse", "huber"}:
            raise ValueError(f"Unsupported base_loss: {base_loss}")
        span = max(float(q_high - q_low), 1e-8)
        self.base_loss = base_loss
        self.huber_delta = float(huber_delta)
        self.alpha = float(alpha)
        self.register_buffer("q_low", torch.tensor(float(q_low)))
        self.register_buffer("span", torch.tensor(span))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ramp = torch.clamp((target - self.q_low) / self.span, min=0.0, max=1.0)
        weight = 1.0 + self.alpha * ramp

        if self.base_loss == "mse":
            base = (pred - target) ** 2
        else:
            base = F.huber_loss(pred, target, delta=self.huber_delta, reduction="none")

        return (weight * base).mean()


def collect_train_target_quantiles(
    loader: DataLoader,
    q_low: float,
    q_high: float,
) -> Tuple[float, float]:
    buf: List[np.ndarray] = []
    for _, y, _ in loader:
        buf.append(y.reshape(-1).numpy())
    flat = np.concatenate(buf, axis=0) if buf else np.array([0.0], dtype=np.float32)
    lo = float(np.quantile(flat, q_low))
    hi = float(np.quantile(flat, q_high))
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    main_loss_fn: nn.Module,
    lambda_smooth: float,
    optimizer: torch.optim.Optimizer | None = None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss_sum = 0.0
    main_loss_sum = 0.0
    smooth_loss_sum = 0.0
    sample_count = 0

    for x, y, chunk_start in loader:
        x = x.to(device)
        y = y.to(device)
        chunk_start = chunk_start.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        pred = model(x)
        main_loss = main_loss_fn(pred, y)
        smooth_loss = smooth_consistency_loss(pred, y, chunk_start)
        total_loss = main_loss + (lambda_smooth * smooth_loss)

        if is_train:
            total_loss.backward()
            optimizer.step()

        batch_size = x.shape[0]
        total_loss_sum += float(total_loss.item()) * batch_size
        main_loss_sum += float(main_loss.item()) * batch_size
        smooth_loss_sum += float(smooth_loss.item()) * batch_size
        sample_count += batch_size

    if sample_count == 0:
        raise RuntimeError("Empty loader: no samples in epoch.")

    return {
        "total_loss": total_loss_sum / sample_count,
        "main_loss": main_loss_sum / sample_count,
        "smooth_loss": smooth_loss_sum / sample_count,
    }


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    pred_batches: List[np.ndarray] = []
    true_batches: List[np.ndarray] = []

    for x, y, _ in loader:
        x = x.to(device)
        pred = model(x).cpu().numpy()
        pred_batches.append(pred)
        true_batches.append(y.numpy())

    y_pred = np.concatenate(pred_batches, axis=0)
    y_true = np.concatenate(true_batches, axis=0)
    return y_pred, y_true


def inverse_transform(target_scaler, values: np.ndarray) -> np.ndarray:
    shape = values.shape
    inv = target_scaler.inverse_transform(values.reshape(-1, 1))
    return inv.reshape(shape)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)

    mask = y_true_flat > 0.0
    if not np.any(mask):
        return {
            "MAE": float("nan"),
            "RMSE": float("nan"),
            "R2": float("nan"),
            "MAPE": float("nan"),
            "mask_ratio": 0.0,
            "num_samples": float(y_true_flat.size),
            "num_valid": 0.0,
        }

    yt = y_true_flat[mask]
    yp = y_pred_flat[mask]
    err = yp - yt

    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(np.square(err))))
    denom = np.maximum(np.abs(yt), 1e-8)
    mape = float(np.mean(np.abs(err) / denom) * 100.0)

    ss_res = float(np.sum(np.square(err)))
    ss_tot = float(np.sum(np.square(yt - np.mean(yt))))
    r2 = float("nan") if ss_tot == 0.0 else float(1.0 - (ss_res / ss_tot))

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
        "mask_ratio": float(np.mean(mask)),
        "num_samples": float(y_true_flat.size),
        "num_valid": float(yt.size),
    }


def _downsample_pair(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    max_points: int = 5000,
) -> Tuple[np.ndarray, np.ndarray]:
    n = y_true.shape[0]
    if n <= max_points:
        return y_true, y_pred
    idx = np.linspace(0, n - 1, max_points, dtype=np.int64)
    return y_true[idx], y_pred[idx]


def _extract_experiment_label(csv_path: str) -> str:
    path = Path(csv_path)
    trial = "tr?"
    exp = "exp?"
    for part in path.parts:
        if part.startswith("tr") and part[2:].isdigit():
            trial = part
        if part.startswith("exp") and part[3:].isdigit():
            exp = part
    return f"{trial}/{exp}"


def build_experiment_segments(dataset: TrafficDataset) -> List[Tuple[str, int, int]]:
    segments: List[Tuple[str, int, int]] = []
    current_label = None
    current_start = 0
    cursor = 0

    for csv_path, count in zip(dataset.csv_paths, dataset.window_counts):
        if count <= 0:
            continue

        label = _extract_experiment_label(csv_path)
        if current_label is None:
            current_label = label
            current_start = cursor
        elif label != current_label:
            segments.append((current_label, current_start, cursor))
            current_label = label
            current_start = cursor
        cursor += count

    if current_label is not None and cursor > current_start:
        segments.append((current_label, current_start, cursor))

    if not segments:
        segments = [("all", 0, len(dataset))]
    return segments


def save_plots(
    plots_dir: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    history_df: pd.DataFrame,
    r2_score: float,
    main_loss_plot_title: str,
    exp_segments: List[Tuple[str, int, int]] | None = None,
) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    y_true_1d = y_true.reshape(-1)
    y_pred_1d = y_pred.reshape(-1)
    residuals = y_pred_1d - y_true_1d
    abs_error = np.abs(residuals)

    p_true, p_pred = _downsample_pair(y_true_1d, y_pred_1d)
    if exp_segments:
        n_exp = len(exp_segments)
        fig, axes = plt.subplots(n_exp, 1, figsize=(12, max(3 * n_exp, 4)), squeeze=False)
        axes_list = axes.reshape(-1)

        for idx, (label, start, end) in enumerate(exp_segments):
            ax = axes_list[idx]
            seg_true = y_true_1d[start:end]
            seg_pred = y_pred_1d[start:end]
            if seg_true.size == 0:
                ax.set_title(f"{label} (empty)")
                ax.axis("off")
                continue
            seg_true_ds, seg_pred_ds = _downsample_pair(seg_true, seg_pred, max_points=1500)
            ax.plot(seg_true_ds, color="tab:blue", linewidth=1.0, label="True")
            ax.plot(seg_pred_ds, color="tab:red", linewidth=1.0, alpha=0.8, label="Pred")
            ax.set_title(f"{label} (n={seg_true.size})")
            ax.set_ylabel("PRBs")
            ax.grid(alpha=0.2)
            if idx == 0:
                ax.legend(loc="upper right")
        axes_list[-1].set_xlabel("Sample (within experiment)")
        plt.tight_layout()
        plt.savefig(plots_dir / "prediction.png", dpi=180)
        plt.close(fig)
    else:
        plt.figure(figsize=(12, 4))
        plt.plot(p_true, color="tab:blue", linewidth=1.2, label="True")
        plt.plot(p_pred, color="tab:red", linewidth=1.2, alpha=0.8, label="Pred")
        plt.title("Per-slice Prediction")
        plt.xlabel("Sample")
        plt.ylabel("sum_granted_prbs")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "prediction.png", dpi=180)
        plt.close()

    plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(p_true, color="tab:blue", linewidth=1.0, label="True")
    ax1.plot(p_pred, color="tab:red", linewidth=1.0, alpha=0.8, label="Pred")
    ax1.set_title("Combined Prediction View")
    ax1.set_ylabel("PRBs")
    ax1.legend()

    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(p_pred - p_true, color="tab:orange", linewidth=1.0, label="Residual")
    ax2.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("Sample")
    ax2.set_ylabel("Residual")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "prediction_combined.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.hist(residuals, bins=60, color="tab:purple", alpha=0.85)
    plt.title("Residual Distribution")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(plots_dir / "error_hist.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.boxplot(
        [residuals, abs_error],
        tick_labels=["Residual", "Absolute Error"],
        showfliers=False,
    )
    plt.title("Error Box Plot")
    plt.tight_layout()
    plt.savefig(plots_dir / "error_box.png", dpi=180)
    plt.close()

    r_plot, _ = _downsample_pair(residuals, residuals)
    plt.figure(figsize=(12, 4))
    plt.plot(r_plot, color="tab:green", linewidth=1.0)
    plt.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    plt.title("Residuals Over Time")
    plt.xlabel("Sample")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.savefig(plots_dir / "residuals.png", dpi=180)
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(history_df["epoch"], history_df["train_total_loss"], label="train")
    axes[0, 0].plot(history_df["epoch"], history_df["val_total_loss"], label="val")
    axes[0, 0].set_title("Total Loss")
    axes[0, 0].legend()

    axes[0, 1].plot(history_df["epoch"], history_df["train_main_loss"], label="train_main")
    axes[0, 1].plot(history_df["epoch"], history_df["val_main_loss"], label="val_main")
    axes[0, 1].set_title(main_loss_plot_title)
    axes[0, 1].legend()

    axes[1, 0].plot(history_df["epoch"], history_df["train_smooth_loss"], label="train_smooth")
    axes[1, 0].plot(history_df["epoch"], history_df["val_smooth_loss"], label="val_smooth")
    axes[1, 0].set_title("Smooth Loss")
    axes[1, 0].legend()

    axes[1, 1].plot(history_df["epoch"], history_df["lr"], color="tab:brown")
    axes[1, 1].set_title("Learning Rate")

    for ax in axes.flat:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(plots_dir / "history.png", dpi=180)
    plt.close(fig)

    s_true, s_pred = _downsample_pair(y_true_1d, y_pred_1d)
    min_v = float(min(np.min(s_true), np.min(s_pred)))
    max_v = float(max(np.max(s_true), np.max(s_pred)))
    plt.figure(figsize=(6, 6))
    plt.scatter(s_true, s_pred, s=8, alpha=0.4, color="tab:blue")
    plt.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=1.0)
    plt.title(f"Prediction Scatter (R2={r2_score:.4f})")
    plt.xlabel("True")
    plt.ylabel("Pred")
    plt.tight_layout()
    plt.savefig(plots_dir / "scatter.png", dpi=180)
    plt.close()


def save_metrics(metrics: Dict[str, float], slice_type: str, output_dir: Path) -> None:
    row = {
        "slice_type": slice_type,
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "R2": metrics["R2"],
        "MAPE": metrics["MAPE"],
        "mask_ratio": metrics["mask_ratio"],
        "num_samples": metrics["num_samples"],
        "num_valid": metrics["num_valid"],
    }
    metrics_df = pd.DataFrame([row])
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    metrics_df.to_latex(output_dir / "metrics.tex", index=False, float_format="%.6f")


def create_output_dir(slice_type: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).resolve().parent / "results" / f"{slice_type}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)
    return output_dir


def build_main_loss(
    args: argparse.Namespace,
    train_loader: DataLoader,
    device: torch.device,
) -> tuple[nn.Module, str]:
    if args.loss_type == "mse":
        print("[INFO] loss_type=mse (unweighted)")
        return RegressionLoss(base_loss="mse", huber_delta=args.huber_delta).to(device), "Main (MSE) Loss"

    if args.loss_type == "huber":
        print(f"[INFO] loss_type=huber | delta={args.huber_delta:.6f}")
        return (
            RegressionLoss(base_loss="huber", huber_delta=args.huber_delta).to(device),
            "Main (Huber) Loss",
        )

    q_low, q_high = collect_train_target_quantiles(
        train_loader, args.weight_quantile, args.weight_upper_quantile
    )
    base_loss = "mse" if args.loss_type == "weighted_mse" else "huber"
    main_loss = WeightedRegressionLoss(
        base_loss=base_loss,
        q_low=q_low,
        q_high=q_high,
        alpha=args.weight_alpha,
        huber_delta=args.huber_delta,
    ).to(device)
    print(
        "[INFO] weighted_loss | "
        f"type={args.loss_type}, "
        f"q_low(q{args.weight_quantile:.2f})={q_low:.6f}, "
        f"q_high(q{args.weight_upper_quantile:.2f})={q_high:.6f}, "
        f"alpha={args.weight_alpha:.2f}, delta={args.huber_delta:.6f}"
    )
    title = "Main (Weighted MSE) Loss" if args.loss_type == "weighted_mse" else "Main (Weighted Huber) Loss"
    return main_loss, title


def main_train(args: argparse.Namespace, output_dir: Path) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"[INFO] Device: {device}")

    data_processor = DataProcessor(
        data_root=args.data_root,
        slice_type=args.slice_type,
        sequence_length=args.sequence_length,
        horizon=args.horizon,
        chunk_size=args.chunk_size,
        seed=args.seed,
        include_target_history=args.include_target_history,
        adj_type=args.adj_type,
        adj_corr_threshold=args.adj_corr_threshold,
    )
    prepared = data_processor.prepare(
        train_dirs=args.train_dirs,
        test_dirs=args.test_dirs,
        val_dirs=args.val_dirs,
        val_split=args.val_split,
    )

    np.save(output_dir / "adjacency.npy", prepared.adjacency)

    pin_memory = device.type == "cuda"
    train_sampler = ChunkShuffleSampler(prepared.train_dataset, shuffle=True, seed=args.seed)

    train_loader = DataLoader(
        prepared.train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        prepared.val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        prepared.test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    print(
        "[INFO] Windows | "
        f"train={len(prepared.train_dataset)} "
        f"val={len(prepared.val_dataset)} "
        f"test={len(prepared.test_dataset)}"
    )
    print(
        f"[INFO] Input feature dimension: {len(prepared.feature_names)} "
        f"({prepared.feature_names})"
    )
    print(
        f"[INFO] Adjacency | type={args.adj_type}, "
        f"shape={prepared.adjacency.shape}, density={prepared.adjacency.mean():.4f}"
    )

    lambda_smooth_eff, lambda_info = resolve_lambda_smooth(
        lambda_base=args.lambda_smooth,
        target_scaler=prepared.target_scaler,
        disable_scale_aware=args.disable_scale_aware_lambda,
        reference_iqr=args.lambda_ref_iqr,
    )
    print(
        "[INFO] lambda_smooth | "
        f"base={lambda_info['base_lambda']:.6f}, "
        f"effective={lambda_info['effective_lambda']:.6f}, "
        f"target_iqr={lambda_info['target_iqr']:.6f}, "
        f"multiplier={lambda_info['multiplier']:.6f}, "
        f"scale_aware={'off' if args.disable_scale_aware_lambda else 'on'}"
    )

    model = TrafficModel(
        input_dim=len(prepared.feature_names),
        sequence_length=args.sequence_length,
        adjacency=prepared.adjacency,
        horizon=args.horizon,
        hidden_dim=args.hidden_dim,
        rnn_type=args.rnn_type,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        lstm_dropout=args.lstm_dropout,
        bilstm_input_proj_dim=args.bilstm_input_proj_dim,
        readout=args.readout,
        fft_hidden=args.fft_hidden,
        fft_n_heads=args.fft_n_heads,
        fft_n_layers=args.fft_n_layers,
        fft_dim_feedforward=args.fft_dim_feedforward,
        fft_dropout=args.fft_dropout,
        fft_readout=args.fft_readout,
        gat_hidden=args.gat_hidden,
        gat_heads=args.gat_heads,
        gat_layers=args.gat_layers,
        gat_dropout=args.gat_dropout,
        gat_input_proj_dim=args.gat_input_proj_dim,
        gat_head_merge=args.gat_head_merge,
        gat_final_head_merge=args.gat_final_head_merge,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    main_loss_fn, main_loss_plot_title = build_main_loss(args, train_loader, device)

    history: List[Dict[str, float]] = []
    best_val = float("inf")
    best_epoch = 0
    patience_count = 0
    best_ckpt_path = output_dir / "model_best.pth"

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        train_stats = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            main_loss_fn=main_loss_fn,
            lambda_smooth=lambda_smooth_eff,
            optimizer=optimizer,
        )
        val_stats = run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            main_loss_fn=main_loss_fn,
            lambda_smooth=lambda_smooth_eff,
            optimizer=None,
        )

        if scheduler is not None:
            scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - epoch_start

        row = {
            "epoch": epoch,
            "train_total_loss": train_stats["total_loss"],
            "train_main_loss": train_stats["main_loss"],
            "train_smooth_loss": train_stats["smooth_loss"],
            "val_total_loss": val_stats["total_loss"],
            "val_main_loss": val_stats["main_loss"],
            "val_smooth_loss": val_stats["smooth_loss"],
            "lr": lr,
            "epoch_time_sec": elapsed,
        }
        history.append(row)

        print(
            f"[Epoch {epoch:03d}] "
            f"train={train_stats['total_loss']:.6f} "
            f"(main={train_stats['main_loss']:.6f}, smooth={train_stats['smooth_loss']:.6f}) | "
            f"val={val_stats['total_loss']:.6f} "
            f"(main={val_stats['main_loss']:.6f}, smooth={val_stats['smooth_loss']:.6f}) | "
            f"lr={lr:.6e} | time={elapsed:.2f}s"
        )

        if val_stats["total_loss"] < best_val:
            best_val = val_stats["total_loss"]
            best_epoch = epoch
            patience_count = 0
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"[INFO] Saved new best checkpoint at epoch {epoch}.")
        else:
            patience_count += 1
            if args.patience > 0 and patience_count >= args.patience:
                print(f"[INFO] Early stopping triggered at epoch {epoch}.")
                break

    print(f"[INFO] Best epoch={best_epoch}, best_val_loss={best_val:.6f}")
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))

    y_pred_val_scaled, y_true_val_scaled = collect_predictions(model, val_loader, device)
    y_pred_test_scaled, y_true_test_scaled = collect_predictions(model, test_loader, device)

    y_pred_val = inverse_transform(prepared.target_scaler, y_pred_val_scaled)
    y_true_val = inverse_transform(prepared.target_scaler, y_true_val_scaled)
    y_pred = inverse_transform(prepared.target_scaler, y_pred_test_scaled)
    y_true = inverse_transform(prepared.target_scaler, y_true_test_scaled)

    metrics = compute_metrics(y_true=y_true, y_pred=y_pred)
    print(
        "[INFO] Test metrics | "
        f"MAE={metrics['MAE']:.6f}, RMSE={metrics['RMSE']:.6f}, "
        f"R2={metrics['R2']:.6f}, MAPE={metrics['MAPE']:.6f}, "
        f"valid_ratio={metrics['mask_ratio']:.4f}"
    )

    anomaly_results = evaluate_anomalies(
        y_true_val_scaled=y_true_val_scaled,
        y_pred_val_scaled=y_pred_val_scaled,
        y_true_test_scaled=y_true_test_scaled,
        y_pred_test_scaled=y_pred_test_scaled,
        y_true_val=y_true_val,
        y_pred_val=y_pred_val,
        y_true_test=y_true,
        y_pred_test=y_pred,
        chebyshev_k=args.chebyshev_k,
        anomaly_extra_k=args.anomaly_extra_k,
        anomaly_error_mode=args.anomaly_error_mode,
    )

    with open(output_dir / "anomaly_summary.txt", "w", encoding="utf-8") as f:
        f.write(str(anomaly_results["summary_text"]))
    np.save(output_dir / "anomaly_flags.npy", anomaly_results["flag_tensor"])
    save_anomaly_overlay(
        plot_path=output_dir / "plots" / "anomaly_overlay.png",
        y_true=y_true,
        y_pred=y_pred,
        flags=np.asarray(anomaly_results["overlay_flags"]),
    )

    save_metrics(metrics, args.slice_type, output_dir)
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "history.csv", index=False)
    exp_segments = build_experiment_segments(prepared.test_dataset)
    print(f"[INFO] Prediction subplot groups (experiments): {len(exp_segments)}")
    save_plots(
        plots_dir=output_dir / "plots",
        y_true=y_true,
        y_pred=y_pred,
        history_df=history_df,
        r2_score=metrics["R2"],
        main_loss_plot_title=main_loss_plot_title,
        exp_segments=exp_segments,
    )


def main() -> None:
    args = parse_args()
    output_dir = create_output_dir(args.slice_type)

    config = vars(args).copy()
    config["output_dir"] = str(output_dir)
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    log_path = output_dir / "training.log"
    with open(log_path, "w", encoding="utf-8") as log_file:
        tee_out = Tee(sys.stdout, log_file)
        tee_err = Tee(sys.stderr, log_file)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            print("[INFO] Start training with config:")
            print(json.dumps(config, indent=2, ensure_ascii=False))
            main_train(args, output_dir)
            print("[INFO] Training finished.")


if __name__ == "__main__":
    main()
