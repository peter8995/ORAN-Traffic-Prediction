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
from torch.utils.data import DataLoader

from DataProcessor import (
    ChunkShuffleSampler,
    DataProcessor,
    FEATURE_ORDER,
    TrafficDataset,
)
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


def parse_args() -> argparse.Namespace:
    default_data_root = (
        Path(__file__).resolve().parents[1]
        / "Dataset"
        / "colosseum-oran-coloran-dataset"
    )

    parser = argparse.ArgumentParser(description="model8 training script")

    parser.add_argument("--data_root", type=str, default=str(default_data_root))
    parser.add_argument("--train_dirs", nargs="+", required=True)
    parser.add_argument("--test_dirs", nargs="+", required=True)
    parser.add_argument("--slice_type", type=str, choices=["embb", "mmtc", "urllc"], required=True)
    parser.add_argument("--val_split", type=float, default=0.2)

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

    parser.add_argument("--rnn_type", type=str, choices=["lstm", "bilstm"], default="lstm")
    parser.add_argument("--lstm_hidden", type=int, default=128)
    parser.add_argument("--lstm_layers", type=int, default=2)
    parser.add_argument("--lstm_dropout", type=float, default=0.2)

    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--dim_feedforward", type=int, default=512)
    parser.add_argument("--vit_dropout", type=float, default=0.2)

    parser.add_argument("--lambda_smooth", type=float, default=0.1)
    parser.add_argument(
        "--lambda_ref_iqr",
        type=float,
        default=104.0,
        help="Reference target IQR for scale-aware lambda (104 from current urllc baseline).",
    )
    parser.add_argument(
        "--disable_scale_aware_lambda",
        action="store_true",
        help="Disable slice scale-aware lambda_smooth adjustment.",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


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
    y_pred: torch.Tensor, y_true: torch.Tensor, chunk_start_flags: torch.Tensor
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

    # Scale-aware smoothing:
    # Use sqrt(reference_iqr / target_iqr) to keep adjustment moderate.
    # Default reference_iqr=104 comes from the current urllc train-target IQR.
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


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mse_loss: nn.Module,
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
        main_loss = mse_loss(pred, y)
        smooth_loss = smooth_consistency_loss(pred, y, chunk_start)
        total_loss = main_loss + lambda_smooth * smooth_loss

        if is_train:
            total_loss.backward()
            optimizer.step()

        bs = x.shape[0]
        total_loss_sum += float(total_loss.item()) * bs
        main_loss_sum += float(main_loss.item()) * bs
        smooth_loss_sum += float(smooth_loss.item()) * bs
        sample_count += bs

    if sample_count == 0:
        raise RuntimeError("Empty loader: no samples in epoch.")

    return {
        "total_loss": total_loss_sum / sample_count,
        "main_loss": main_loss_sum / sample_count,
        "smooth_loss": smooth_loss_sum / sample_count,
    }


@torch.no_grad()
def collect_predictions(
    model: nn.Module, loader: DataLoader, device: torch.device
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

    # PRB target is non-negative by definition. After inverse transform,
    # zeros can become tiny negatives (e.g., -1e-4) due to float precision.
    # Use strictly positive mask to avoid MAPE explosion near zero.
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
    y_true: np.ndarray, y_pred: np.ndarray, max_points: int = 5000
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
    exp_segments: List[Tuple[str, int, int]] | None = None,
) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    y_true_1d = y_true.reshape(-1)
    y_pred_1d = y_pred.reshape(-1)
    residuals = y_pred_1d - y_true_1d
    abs_error = np.abs(residuals)

    # 1. Per-experiment prediction subplots
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

    # 2. Combined predictions
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

    # 3. Error distribution histogram
    plt.figure(figsize=(8, 4))
    plt.hist(residuals, bins=60, color="tab:purple", alpha=0.85)
    plt.title("Residual Distribution")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(plots_dir / "error_hist.png", dpi=180)
    plt.close()

    # 4. Error box plot
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

    # 5. Residuals over time
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

    # 6. Training history
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(history_df["epoch"], history_df["train_total_loss"], label="train")
    axes[0, 0].plot(history_df["epoch"], history_df["val_total_loss"], label="val")
    axes[0, 0].set_title("Total Loss")
    axes[0, 0].legend()

    axes[0, 1].plot(history_df["epoch"], history_df["train_main_loss"], label="train_main")
    axes[0, 1].plot(history_df["epoch"], history_df["val_main_loss"], label="val_main")
    axes[0, 1].set_title("Main (MSE) Loss")
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

    # 7. Scatter with diagonal
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
    )
    prepared = data_processor.prepare(
        train_dirs=args.train_dirs,
        test_dirs=args.test_dirs,
        val_split=args.val_split,
    )

    pin_memory = device.type == "cuda"
    train_sampler = ChunkShuffleSampler(
        prepared.train_dataset,
        shuffle=True,
        seed=args.seed,
    )

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

    input_dim = len(FEATURE_ORDER)
    print(f"[INFO] Input feature dimension: {input_dim} ({FEATURE_ORDER})")

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
        input_dim=input_dim,
        sequence_length=args.sequence_length,
        horizon=args.horizon,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dim_feedforward=args.dim_feedforward,
        vit_dropout=args.vit_dropout,
        rnn_type=args.rnn_type,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        lstm_dropout=args.lstm_dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    mse_loss = nn.MSELoss()
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
            mse_loss=mse_loss,
            lambda_smooth=lambda_smooth_eff,
            optimizer=optimizer,
        )
        val_stats = run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            mse_loss=mse_loss,
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

    y_pred_scaled, y_true_scaled = collect_predictions(model, test_loader, device)
    y_pred = inverse_transform(prepared.target_scaler, y_pred_scaled)
    y_true = inverse_transform(prepared.target_scaler, y_true_scaled)

    metrics = compute_metrics(y_true=y_true, y_pred=y_pred)
    print(
        "[INFO] Test metrics | "
        f"MAE={metrics['MAE']:.6f}, RMSE={metrics['RMSE']:.6f}, "
        f"R2={metrics['R2']:.6f}, MAPE={metrics['MAPE']:.6f}, "
        f"valid_ratio={metrics['mask_ratio']:.4f}"
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
