"""
Isolation Forest spike detection for O-RAN traffic data.

Uses sklearn's IsolationForest to detect anomalous traffic spikes.
"""

import argparse
import os
import numpy as np
from sklearn.ensemble import IsolationForest
from matplotlib import pyplot as plt

from DataProcessor import DataProcessor, parse_directory_args, build_scalers


def main():
    parser = argparse.ArgumentParser(description="Isolation Forest spike detection for O-RAN traffic.")
    parser.add_argument('--base_path', type=str,
                        default='/home/cislab301b/peter/ORAN-Traffic-Prediction/Dataset/colosseum-oran-coloran-dataset')
    parser.add_argument('--train_dirs', type=str, nargs='+', required=True)
    parser.add_argument('--test_dirs', type=str, nargs='+', required=True)
    parser.add_argument('--slice_type', type=str, default='embb', choices=['embb', 'mmtc', 'urllc'])
    parser.add_argument('--sequence_length', type=int, default=15, help="Sliding window length")
    parser.add_argument('--contamination', type=float, default=0.11,
                        help="Expected proportion of anomalies in training data")
    parser.add_argument('--n_estimators', type=int, default=200, help="Number of trees")
    parser.add_argument('--max_samples', type=str, default='auto',
                        help="Number of samples per tree ('auto' or integer)")
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--k', type=float, default=2.0,
                        help="Spike value threshold: global_mean + k*global_std")
    parser.add_argument('--j', type=float, default=2.0,
                        help="Spike diff threshold: diff_mean + j*diff_std")
    args = parser.parse_args()

    train_paths = parse_directory_args(args.train_dirs, args.base_path)
    test_paths = parse_directory_args(args.test_dirs, args.base_path)

    # Data loading
    processor = DataProcessor(sequenceLength=args.sequence_length)
    processor.k = args.k
    processor.j = args.j
    train_files = processor.accumulate_files(train_paths, args.slice_type)
    if not train_files:
        raise ValueError("No metrics.csv files found in --train_dirs!")

    processor.fit_spike_params(train_files)
    scalerX, scalerY = build_scalers(processor, train_files)

    print(f"\nLoading training data...")
    train_X, train_Y, train_S = processor.process_directories(train_paths, scalerX, scalerY, args.slice_type)
    print(f"Loading test data...")
    test_X, test_Y, test_S = processor.process_directories(test_paths, scalerX, scalerY, args.slice_type)

    if len(train_X) == 0 or len(test_X) == 0:
        raise ValueError("No valid data found!")

    # train_X: (N, seq_len, 17), flatten sliding window for Isolation Forest
    seq_len = train_X.shape[1]
    n_features = train_X.shape[2]
    train_input = train_X.reshape(len(train_X), -1)  # (N, seq_len * 17)
    test_input = test_X.reshape(len(test_X), -1)

    print(f"Train: {len(train_input)} windows | Test: {len(test_input)} windows")
    print(f"Window shape: ({seq_len}, {n_features}) → flattened: {train_input.shape[1]}")
    print(f"Train spike rate: {train_S.mean()*100:.2f}% | Test spike rate: {test_S.mean()*100:.2f}%")

    # Parse max_samples
    max_samples = args.max_samples
    if max_samples != 'auto':
        max_samples = int(max_samples)

    # Train Isolation Forest
    print(f"\nFitting Isolation Forest (n_estimators={args.n_estimators}, contamination={args.contamination})...")
    clf = IsolationForest(
        n_estimators=args.n_estimators,
        contamination=args.contamination,
        max_samples=max_samples,
        random_state=args.random_state,
        n_jobs=-1,
    )
    clf.fit(train_input)
    print("Fitting complete.")

    # Predict on test set
    # IsolationForest: -1 = anomaly, 1 = normal
    test_labels = clf.predict(test_input)
    spike_preds = (test_labels == -1).astype(int)

    # Anomaly scores (lower = more anomalous)
    test_scores = clf.decision_function(test_input)
    train_scores = clf.decision_function(train_input)

    # Metrics
    tp = np.sum((spike_preds == 1) & (test_S == 1))
    fp = np.sum((spike_preds == 1) & (test_S == 0))
    fn = np.sum((spike_preds == 0) & (test_S == 1))
    tn = np.sum((spike_preds == 0) & (test_S == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0

    print("\n" + "=" * 50)
    print("Isolation Forest Spike Detection Results")
    print("=" * 50)
    print(f"Accuracy:      {accuracy:.4f}")
    print(f"Precision:     {precision:.4f}")
    print(f"Recall:        {recall:.4f}")
    print(f"F1 Score:      {f1:.4f}")
    print(f"FP Rate:       {fp_rate:.4f}")
    print(f"FN Rate:       {fn_rate:.4f}")
    print(f"Contamination: {args.contamination}")
    print(f"Test spikes:   {test_S.sum():.0f}/{len(test_S)} ({test_S.mean()*100:.2f}%)")
    print(f"Predicted:     {spike_preds.sum()}/{len(spike_preds)} ({spike_preds.mean()*100:.2f}%)")
    print("=" * 50)

    # Sweep contamination / threshold for best F1
    print("\n--- Threshold Sweep (on test set) ---")
    percentiles = [90, 92, 94, 95, 96, 97, 98, 99]
    best_f1, best_pct = 0, 95
    for pct in percentiles:
        thr = np.percentile(train_scores, 100 - pct)
        preds_sweep = (test_scores < thr).astype(int)
        tp_s = np.sum((preds_sweep == 1) & (test_S == 1))
        fp_s = np.sum((preds_sweep == 1) & (test_S == 0))
        fn_s = np.sum((preds_sweep == 0) & (test_S == 1))
        p_s = tp_s / (tp_s + fp_s) if (tp_s + fp_s) > 0 else 0
        r_s = tp_s / (tp_s + fn_s) if (tp_s + fn_s) > 0 else 0
        f1_s = 2 * p_s * r_s / (p_s + r_s) if (p_s + r_s) > 0 else 0
        flag = " <-- best" if f1_s > best_f1 else ""
        if f1_s > best_f1:
            best_f1 = f1_s
            best_pct = pct
        print(f"  Top {100-pct:2d}% anomaly (P{pct}): P={p_s:.3f} R={r_s:.3f} F1={f1_s:.3f}{flag}")

    print(f"\nBest threshold: top {100-best_pct}% (F1={best_f1:.3f})")

    # Save results
    result_dir = './results'
    os.makedirs(result_dir, exist_ok=True)
    slice_type = args.slice_type

    # 1. Anomaly score distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(train_scores, bins=100, alpha=0.7, density=True)
    axes[0].set_title('Train Anomaly Scores')
    axes[0].set_xlabel('Decision Function Score (lower = more anomalous)')
    axes[0].grid(True, alpha=0.3)

    test_normal_scores = test_scores[test_S == 0]
    test_spike_scores = test_scores[test_S == 1]
    axes[1].hist(test_normal_scores, bins=100, alpha=0.6, label='Normal', density=True, color='blue')
    if len(test_spike_scores) > 0:
        axes[1].hist(test_spike_scores, bins=100, alpha=0.6, label='Spike', density=True, color='red')
    axes[1].set_title('Test Anomaly Scores by Class')
    axes[1].set_xlabel('Decision Function Score')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f'if_{slice_type}_score_dist.png'), dpi=300)
    plt.close()

    # 2. Spike detection timeline
    max_pts = min(len(test_S), 10000)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})

    ax1.plot(-test_scores[:max_pts], label='Anomaly Score (negated)', color='purple', alpha=0.7, linewidth=0.5)
    ax1.set_ylabel('Anomaly Score')
    ax1.set_title(f'Isolation Forest Spike Detection - {slice_type.upper()}')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    x_axis = np.arange(max_pts)
    gt = test_S[:max_pts].astype(int)
    pred = spike_preds[:max_pts]
    ax2.fill_between(x_axis, 0, 1, where=(gt == 1), color='green', alpha=0.3, label='Ground Truth', step='mid')
    ax2.fill_between(x_axis, 0, 1, where=(pred == 1), color='red', alpha=0.4, label='Predicted', step='mid')
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Normal', 'Peak'])
    ax2.set_xlabel('Time Steps')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f'if_{slice_type}_spike_detection.png'), dpi=300)
    plt.close()

    print(f"\nPlots saved to {result_dir}/")


if __name__ == "__main__":
    main()