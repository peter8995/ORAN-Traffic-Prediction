from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODE_ORDER = ("signed", "abs")


@dataclass(frozen=True)
class ThresholdRecord:
    k: float
    mean: float
    std: float
    lower: float
    upper: float
    flag_rate: float


def build_k_values(primary_k: float, extra_ks: Sequence[float]) -> List[float]:
    values = {float(primary_k)}
    values.update(float(k) for k in extra_ks)
    return sorted(values)


def resolve_modes(error_mode: str) -> List[str]:
    if error_mode == "both":
        return list(MODE_ORDER)
    if error_mode in MODE_ORDER:
        return [error_mode]
    raise ValueError(f"Unsupported anomaly_error_mode: {error_mode}")


def _compute_mode_records(
    val_residuals: np.ndarray,
    test_residuals: np.ndarray,
    ks: Sequence[float],
) -> Dict[str, List[ThresholdRecord]]:
    val_residuals = val_residuals.reshape(-1).astype(np.float64)
    test_residuals = test_residuals.reshape(-1).astype(np.float64)

    signed_mean = float(np.mean(val_residuals))
    signed_std = float(np.std(val_residuals))

    abs_val = np.abs(val_residuals)
    abs_test = np.abs(test_residuals)
    abs_mean = float(np.mean(abs_val))
    abs_std = float(np.std(abs_val))

    signed_records: List[ThresholdRecord] = []
    abs_records: List[ThresholdRecord] = []

    for k in ks:
        signed_lower = signed_mean - (float(k) * signed_std)
        signed_upper = signed_mean + (float(k) * signed_std)
        signed_flags = (test_residuals < signed_lower) | (test_residuals > signed_upper)
        signed_records.append(
            ThresholdRecord(
                k=float(k),
                mean=signed_mean,
                std=signed_std,
                lower=float(signed_lower),
                upper=float(signed_upper),
                flag_rate=float(np.mean(signed_flags)),
            )
        )

        abs_upper = abs_mean + (float(k) * abs_std)
        abs_flags = abs_test > abs_upper
        abs_records.append(
            ThresholdRecord(
                k=float(k),
                mean=abs_mean,
                std=abs_std,
                lower=0.0,
                upper=float(abs_upper),
                flag_rate=float(np.mean(abs_flags)),
            )
        )

    return {"signed": signed_records, "abs": abs_records}


def _build_flag_tensor(
    test_residuals: np.ndarray,
    records: Dict[str, List[ThresholdRecord]],
    ks: Sequence[float],
) -> np.ndarray:
    test_residuals = test_residuals.reshape(-1).astype(np.float64)
    flag_tensor = np.zeros((len(ks), len(MODE_ORDER), test_residuals.size), dtype=np.uint8)

    for k_idx, k in enumerate(ks):
        signed_record = records["signed"][k_idx]
        abs_record = records["abs"][k_idx]

        signed_flags = (test_residuals < signed_record.lower) | (test_residuals > signed_record.upper)
        abs_flags = np.abs(test_residuals) > abs_record.upper

        flag_tensor[k_idx, 0, :] = signed_flags.astype(np.uint8)
        flag_tensor[k_idx, 1, :] = abs_flags.astype(np.uint8)

    return flag_tensor


def _select_overlay_flags(
    flag_tensor: np.ndarray,
    ks: Sequence[float],
    primary_k: float,
    error_mode: str,
) -> np.ndarray:
    primary_idx = ks.index(float(primary_k))
    modes = resolve_modes(error_mode)
    selected = np.zeros(flag_tensor.shape[-1], dtype=bool)
    for mode in modes:
        selected |= flag_tensor[primary_idx, MODE_ORDER.index(mode), :].astype(bool)
    return selected.astype(np.uint8)


def _format_records(space_name: str, records: Dict[str, List[ThresholdRecord]]) -> str:
    lines = [f"[{space_name}]"]
    for mode in MODE_ORDER:
        lines.append(f"{mode}:")
        for record in records[mode]:
            if mode == "signed":
                lines.append(
                    "  "
                    f"k={record.k:.3f} mean={record.mean:.6f} std={record.std:.6f} "
                    f"lower={record.lower:.6f} upper={record.upper:.6f} "
                    f"flag_rate={record.flag_rate:.6f}"
                )
            else:
                lines.append(
                    "  "
                    f"k={record.k:.3f} mean={record.mean:.6f} std={record.std:.6f} "
                    f"threshold={record.upper:.6f} flag_rate={record.flag_rate:.6f}"
                )
    return "\n".join(lines)


def evaluate_anomalies(
    y_true_val_scaled: np.ndarray,
    y_pred_val_scaled: np.ndarray,
    y_true_test_scaled: np.ndarray,
    y_pred_test_scaled: np.ndarray,
    y_true_val: np.ndarray,
    y_pred_val: np.ndarray,
    y_true_test: np.ndarray,
    y_pred_test: np.ndarray,
    chebyshev_k: float,
    anomaly_extra_k: Sequence[float],
    anomaly_error_mode: str,
) -> Dict[str, object]:
    ks = build_k_values(chebyshev_k, anomaly_extra_k)

    val_residual_scaled = (y_pred_val_scaled - y_true_val_scaled).reshape(-1)
    test_residual_scaled = (y_pred_test_scaled - y_true_test_scaled).reshape(-1)
    scaled_records = _compute_mode_records(val_residual_scaled, test_residual_scaled, ks)

    val_residual = (y_pred_val - y_true_val).reshape(-1)
    test_residual = (y_pred_test - y_true_test).reshape(-1)
    original_records = _compute_mode_records(val_residual, test_residual, ks)

    flag_tensor = _build_flag_tensor(test_residual, original_records, ks)
    overlay_flags = _select_overlay_flags(flag_tensor, ks, float(chebyshev_k), anomaly_error_mode)

    summary_lines = [
        f"primary_k={float(chebyshev_k):.6f}",
        f"error_mode={anomaly_error_mode}",
        f"k_values={ks}",
        f"flag_tensor_shape={tuple(flag_tensor.shape)}  # (num_k, mode, num_test)",
        "mode_index={0: signed, 1: abs}",
        "",
        _format_records("original_space", original_records),
        "",
        _format_records("scaled_space", scaled_records),
    ]

    return {
        "k_values": ks,
        "flag_tensor": flag_tensor,
        "overlay_flags": overlay_flags,
        "summary_text": "\n".join(summary_lines) + "\n",
    }


def _downsample_for_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    flags: np.ndarray,
    max_points: int = 5000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    flags = flags.reshape(-1)

    if y_true.size <= max_points:
        return y_true, y_pred, flags

    idx = np.linspace(0, y_true.size - 1, max_points, dtype=np.int64)
    return y_true[idx], y_pred[idx], flags[idx]


def save_anomaly_overlay(
    plot_path: str | Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    flags: np.ndarray,
) -> None:
    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    y_true_ds, y_pred_ds, flags_ds = _downsample_for_plot(y_true, y_pred, flags)
    x = np.arange(y_true_ds.size)

    fig, ax1 = plt.subplots(figsize=(12, 4.5))
    true_line = ax1.plot(x, y_true_ds, color="tab:blue", linewidth=1.0, label="True")[0]
    pred_line = ax1.plot(x, y_pred_ds, color="tab:orange", linewidth=1.0, alpha=0.85, label="Pred")[0]
    ax1.set_xlabel("Sample")
    ax1.set_ylabel("sum_granted_prbs")
    ax1.set_title("Prediction + Anomaly Flags")
    ax1.grid(alpha=0.2)

    ax2 = ax1.twinx()
    flag_line = ax2.step(
        x,
        flags_ds.astype(np.float32),
        where="mid",
        color="tab:red",
        linewidth=1.0,
        alpha=0.8,
        label="Flag",
    )[0]
    ax2.set_ylabel("Flag")
    ax2.set_ylim(-0.05, 1.05)

    ax1.legend([true_line, pred_line, flag_line], ["True", "Pred", "Flag"], loc="upper right")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=180)
    plt.close(fig)
