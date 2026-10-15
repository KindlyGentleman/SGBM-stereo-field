"""Educational tiny convolutional neural network for stereo patch correspondence.

Implements:
1. Synthetic 15x15 patch dataset generator with independent scene splits.
2. Two-layer tiny CNN (1,054 parameters) using PyTorch on CPU.
3. Training loop with validation-based model selection.
4. Classical Mean Absolute Difference (SAD) baseline with train/val threshold tuning.
5. Final evaluation on untouched held-out test set (accuracy, F1, latency, confusion matrix).
6. Single-patch candidate disparity sweep comparing CNN confidence with classical SAD.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import stereo


class TinyStereoNet(nn.Module):
    """A tiny 2-layer CNN patch classifier with 1,054 parameters.

    Input: (B, 2, 15, 15) stacked left and candidate right patches.
    Architecture:
        Conv2d(2, 8, kernel_size=3, padding=1) -> ReLU
        Conv2d(8, 12, kernel_size=3, padding=1) -> ReLU
        AdaptiveAvgPool2d((1, 1)) -> Global Average Pooling
        Linear(12, 2) -> Match (1) / Non-match (0) logits
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(2, 8, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 12, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(12, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.relu1(self.conv1(x))
        h = self.relu2(self.conv2(h))
        pooled = self.gap(h)
        flat = pooled.view(pooled.size(0), -1)
        return self.fc(flat)


def normalize_patches(patches: np.ndarray) -> np.ndarray:
    """Normalize each patch independently to zero mean and unit variance.

    Matches standard normalized cross-correlation (NCC) photometric preprocessing.
    Args:
        patches: Shape (N, 2, H, W)
    Returns:
        Normalized float32 array of shape (N, 2, H, W).
    """
    mean = np.mean(patches, axis=(2, 3), keepdims=True)
    std = np.std(patches, axis=(2, 3), keepdims=True) + 1e-5
    return ((patches - mean) / std).astype(np.float32)


def extract_patches_from_scene(
    left: np.ndarray,
    right: np.ndarray,
    disp_gt: np.ndarray,
    valid_mask: np.ndarray,
    n_samples: int,
    seed: int,
    patch_size: int = 15
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract balanced positive and negative patch pairs from a stereo scene.

    Args:
        left: Left grayscale image (H, W).
        right: Right grayscale image (H, W).
        disp_gt: Ground truth disparity map (H, W).
        valid_mask: Non-occluded visibility mask (H, W).
        n_samples: Total number of patch pairs to extract (half pos, half neg).
        seed: Random seed for deterministic extraction.
        patch_size: Square patch dimensions (default 15).

    Returns:
        patches: Array of shape (n_samples, 2, patch_size, patch_size), float32 in [0, 1].
        labels: Array of shape (n_samples,), int64 with 1 (match) or 0 (nonmatch).
    """
    rng = np.random.default_rng(seed)
    rad = patch_size // 2
    h, w = left.shape[:2]

    # Valid candidate center pixels in left image
    inner_mask = np.zeros_like(valid_mask, dtype=bool)
    inner_mask[rad:h - rad, rad:w - rad] = True
    candidates = valid_mask & inner_mask & (disp_gt > 0)

    ys, xs = np.where(candidates)
    if len(ys) == 0:
        raise ValueError("No valid candidates found for patch extraction.")

    n_pos = n_samples // 2
    n_neg = n_samples - n_pos

    idx_pos = rng.choice(len(ys), size=n_pos, replace=(len(ys) < n_pos))
    patches_pos = []
    for idx in idx_pos:
        y, xl = ys[idx], xs[idx]
        d = int(round(float(disp_gt[y, xl])))
        xr = xl - d
        if rad <= xr < w - rad:
            pl = left[y - rad:y + rad + 1, xl - rad:xl + rad + 1].astype(np.float32) / 255.0
            pr = right[y - rad:y + rad + 1, xr - rad:xr + rad + 1].astype(np.float32) / 255.0
            patches_pos.append(np.stack([pl, pr], axis=0))

    # Negative pairs: sample candidate right patches with incorrect disparity or location
    idx_neg = rng.choice(len(ys), size=n_neg * 3, replace=True)
    patches_neg = []
    shifts = [-12, -8, -6, -4, -3, -2, 2, 3, 4, 6, 8, 12]
    for idx in idx_neg:
        if len(patches_neg) >= n_neg:
            break
        y, xl = ys[idx], xs[idx]
        d_true = int(round(float(disp_gt[y, xl])))
        delta = rng.choice(shifts)
        xr_fake = xl - d_true + delta

        if rad <= xr_fake < w - rad:
            pl = left[y - rad:y + rad + 1, xl - rad:xl + rad + 1].astype(np.float32) / 255.0
            pr = right[y - rad:y + rad + 1, xr_fake - rad:xr_fake + rad + 1].astype(np.float32) / 255.0
            patches_neg.append(np.stack([pl, pr], axis=0))

    # Combine and balance
    actual_pos = len(patches_pos)
    actual_neg = len(patches_neg)
    min_count = min(actual_pos, actual_neg)

    p_pos = np.array(patches_pos[:min_count], dtype=np.float32)
    p_neg = np.array(patches_neg[:min_count], dtype=np.float32)

    all_patches = np.concatenate([p_pos, p_neg], axis=0)
    all_labels = np.concatenate([
        np.ones(min_count, dtype=np.int64),
        np.zeros(min_count, dtype=np.int64)
    ], axis=0)

    perm = rng.permutation(len(all_labels))
    return all_patches[perm], all_labels[perm]


def build_split_datasets(
    train_samples: int = 1000,
    val_samples: int = 200,
    test_samples: int = 250,
    patch_size: int = 15
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Build strictly partitioned train, validation, and test datasets.

    Independent scene seeds ensure zero data leakage across splits.
    """
    # 1. Training dataset from seeds 100 to 109
    train_p_list, train_l_list = [], []
    samples_per_train_scene = max(50, train_samples // 10)
    for seed in range(100, 110):
        l, r, d_gt, v_mask = stereo.generate_layered_synthetic_scene(320, 240, seed=seed)
        p, y = extract_patches_from_scene(l, r, d_gt, v_mask, samples_per_train_scene, seed=seed, patch_size=patch_size)
        train_p_list.append(p)
        train_l_list.append(y)

    x_train = np.concatenate(train_p_list, axis=0)[:train_samples]
    y_train = np.concatenate(train_l_list, axis=0)[:train_samples]

    # 2. Validation dataset from seeds 200 to 203
    val_p_list, val_l_list = [], []
    samples_per_val_scene = max(50, val_samples // 4)
    for seed in range(200, 204):
        l, r, d_gt, v_mask = stereo.generate_layered_synthetic_scene(320, 240, seed=seed)
        p, y = extract_patches_from_scene(l, r, d_gt, v_mask, samples_per_val_scene, seed=seed, patch_size=patch_size)
        val_p_list.append(p)
        val_l_list.append(y)

    x_val = np.concatenate(val_p_list, axis=0)[:val_samples]
    y_val = np.concatenate(val_l_list, axis=0)[:val_samples]

    # 3. Held-out test dataset from seeds 300 to 304
    test_p_list, test_l_list = [], []
    samples_per_test_scene = max(50, test_samples // 5)
    for seed in range(300, 305):
        l, r, d_gt, v_mask = stereo.generate_layered_synthetic_scene(320, 240, seed=seed)
        p, y = extract_patches_from_scene(l, r, d_gt, v_mask, samples_per_test_scene, seed=seed, patch_size=patch_size)
        test_p_list.append(p)
        test_l_list.append(y)

    x_test = np.concatenate(test_p_list, axis=0)[:test_samples]
    y_test = np.concatenate(test_l_list, axis=0)[:test_samples]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def evaluate_classical_baseline(
    x_data: np.ndarray,
    y_data: np.ndarray,
    threshold: float
) -> Dict[str, Any]:
    """Evaluate Mean Absolute Difference (SAD) baseline at a given threshold.

    Decision rule: if mean(|P_L - P_R|) < threshold, predict match (1), else nonmatch (0).
    """
    diffs = np.mean(np.abs(x_data[:, 0] - x_data[:, 1]), axis=(1, 2))
    preds = (diffs < threshold).astype(np.int64)

    acc = float(np.mean(preds == y_data) * 100.0)
    tp = int(np.sum((preds == 1) & (y_data == 1)))
    fp = int(np.sum((preds == 1) & (y_data == 0)))
    fn = int(np.sum((preds == 0) & (y_data == 1)))
    tn = int(np.sum((preds == 0) & (y_data == 0)))

    prec = float(tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return {
        "threshold": threshold,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": {"TP": tp, "FP": fp, "FN": fn, "TN": tn}
    }


def find_optimal_classical_threshold(
    x_val: np.ndarray,
    y_val: np.ndarray
) -> float:
    """Find the MAD threshold that maximizes classification accuracy on validation data."""
    best_thresh = 0.30
    best_acc = -1.0

    # Grid search across reasonable normalized intensity differences in [0.05, 1.20]
    for thresh in np.linspace(0.05, 1.20, 100):
        res = evaluate_classical_baseline(x_val, y_val, thresh)
        if res["accuracy"] > best_acc:
            best_acc = res["accuracy"]
            best_thresh = float(thresh)

    return best_thresh


def train_cnn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 0.008,
    seed: int = 42,
    num_threads: int = 2
) -> Tuple[TinyStereoNet, pd.DataFrame, float]:
    """Train TinyStereoNet on CPU using validation accuracy for model selection.

    Returns:
        best_model: Model weights that achieved highest validation accuracy.
        history_df: DataFrame with epoch-level train/val loss and accuracy.
        train_time_s: Total training duration in seconds.
    """
    torch.manual_seed(seed)
    torch.set_num_threads(num_threads)

    model = TinyStereoNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    tx_train = torch.from_numpy(x_train).float()
    ty_train = torch.from_numpy(y_train).long()
    tx_val = torch.from_numpy(x_val).float()
    ty_val = torch.from_numpy(y_val).long()

    n_train = len(tx_train)
    history = []
    best_val_acc = -1.0
    best_state = None

    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        correct_train = 0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            bx = tx_train[idx]
            by = ty_train[idx]

            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(bx)
            preds = torch.argmax(logits, dim=1)
            correct_train += int(torch.sum(preds == by).item())

        train_loss = epoch_loss / n_train
        train_acc = (correct_train / n_train) * 100.0

        # Validation evaluation
        model.eval()
        with torch.no_grad():
            val_logits = model(tx_val)
            val_loss = criterion(val_logits, ty_val).item()
            val_preds = torch.argmax(val_logits, dim=1)
            val_acc = (torch.sum(val_preds == ty_val).item() / len(ty_val)) * 100.0

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    train_time_s = time.perf_counter() - t0

    if best_state is not None:
        model.load_state_dict(best_state)

    history_df = pd.DataFrame(history)
    return model, history_df, train_time_s


def evaluate_cnn_test(
    model: TinyStereoNet,
    x_test: np.ndarray,
    y_test: np.ndarray
) -> Tuple[Dict[str, Any], float]:
    """Evaluate final trained model on untouched held-out test data."""
    model.eval()
    tx_test = torch.from_numpy(x_test).float()

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(tx_test)
        preds = torch.argmax(logits, dim=1).numpy()
    inf_time_s = time.perf_counter() - t0

    acc = float(np.mean(preds == y_test) * 100.0)
    tp = int(np.sum((preds == 1) & (y_test == 1)))
    fp = int(np.sum((preds == 1) & (y_test == 0)))
    fn = int(np.sum((preds == 0) & (y_test == 1)))
    tn = int(np.sum((preds == 0) & (y_test == 0)))

    prec = float(tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
        "samples_evaluated": len(y_test),
        "inference_time_s": inf_time_s,
        "inference_ms_per_patch": (inf_time_s / len(y_test)) * 1000.0
    }
    return metrics, inf_time_s


def sweep_patch_candidate_disparities(
    model: TinyStereoNet,
    left_img: np.ndarray,
    right_img: np.ndarray,
    x0: int,
    y0: int,
    patch_size: int = 15,
    max_disp: int = 48
) -> Dict[str, np.ndarray]:
    """Scan candidate horizontal disparities for one patch, comparing CNN probability with SAD.

    Demonstrates matching scores along an epipolar line for educational visualization.
    """
    model.eval()
    rad = patch_size // 2
    h, w = left_img.shape[:2]

    disps = np.arange(max_disp, dtype=int)
    sad_costs = np.full(max_disp, np.nan, dtype=np.float32)
    cnn_probs = np.full(max_disp, np.nan, dtype=np.float32)

    p_left = left_img[y0 - rad:y0 + rad + 1, x0 - rad:x0 + rad + 1].astype(np.float32) / 255.0

    batch_patches = []
    valid_d_indices = []

    for d in disps:
        xr = x0 - d
        if rad <= xr < w - rad:
            p_right = right_img[y0 - rad:y0 + rad + 1, xr - rad:xr + rad + 1].astype(np.float32) / 255.0
            sad = float(np.sum(np.abs(p_left - p_right)))
            sad_costs[d] = sad
            pair = np.stack([p_left, p_right], axis=0)
            batch_patches.append(pair)
            valid_d_indices.append(d)

    if len(batch_patches) > 0:
        raw_arr = np.array(batch_patches, dtype=np.float32)
        norm_arr = normalize_patches(raw_arr)
        bx = torch.from_numpy(norm_arr).float()
        with torch.no_grad():
            logits = model(bx)
            probs = torch.softmax(logits, dim=1)[:, 1].numpy()

        for idx, d in enumerate(valid_d_indices):
            cnn_probs[d] = float(probs[idx])

    return {
        "disparities": disps,
        "sad_costs": sad_costs,
        "cnn_probs": cnn_probs
    }


def run_cnn_experiment(
    epochs: int = 15,
    num_threads: int = 2,
    results_dir: str = "results",
    save_json: bool = True
) -> Dict[str, Any]:
    """Execute complete CNN educational patch correspondence experiment."""
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("EXPERIMENT 3: TINY CNN PATCH CORRESPONDENCE CLASSIFIER")
    print("=" * 72)
    print("Building independent datasets (Train: 1000, Val: 200, Test: 250)...")

    (x_train_raw, y_train), (x_val_raw, y_val), (x_test_raw, y_test) = build_split_datasets()

    # Apply per-patch zero-mean unit-variance normalization
    x_train = normalize_patches(x_train_raw)
    x_val = normalize_patches(x_val_raw)
    x_test = normalize_patches(x_test_raw)

    print(f"  Training samples:   {len(y_train)} (Pos: {np.sum(y_train == 1)}, Neg: {np.sum(y_train == 0)})")
    print(f"  Validation samples: {len(y_val)} (Pos: {np.sum(y_val == 1)}, Neg: {np.sum(y_val == 0)})")
    print(f"  Test samples:       {len(y_test)} (Pos: {np.sum(y_test == 1)}, Neg: {np.sum(y_test == 0)})")

    # 1. Classical baseline selection on train/val
    print("\n[Step 1] Selecting classical MAD threshold on validation set...")
    opt_thresh = find_optimal_classical_threshold(x_val, y_val)
    val_baseline = evaluate_classical_baseline(x_val, y_val, opt_thresh)
    print(f"  Optimal MAD threshold: {opt_thresh:.3f} | Val Accuracy: {val_baseline['accuracy']:.1f}%")

    # 2. Train tiny CNN
    print(f"\n[Step 2] Training TinyStereoNet ({epochs} epochs, {num_threads} CPU threads)...")
    model, history_df, train_time_s = train_cnn(
        x_train, y_train, x_val, y_val,
        epochs=epochs, lr=0.008, num_threads=num_threads
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameter count: {n_params} parameters")
    print(f"  CPU Training time:     {train_time_s:.2f} s ({train_time_s / epochs * 1000.0:.1f} ms/epoch)")
    print(f"  Final Val Accuracy:    {history_df['val_acc'].iloc[-1]:.1f}% (Best: {history_df['val_acc'].max():.1f}%)")

    # 3. Evaluate both on untouched held-out test data
    print("\n[Step 3] Evaluating on untouched held-out test set...")
    cnn_test_metrics, inf_time_s = evaluate_cnn_test(model, x_test, y_test)
    classical_test_metrics = evaluate_classical_baseline(x_test, y_test, opt_thresh)

    print("-" * 72)
    print(f"{'Method':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
    print("-" * 72)
    print(f"{'Classical (MAD)':<20} {classical_test_metrics['accuracy']:<12.1f} {classical_test_metrics['precision']:<12.1f} {classical_test_metrics['recall']:<12.1f} {classical_test_metrics['f1']:<12.1f}")
    print(f"{'Tiny CNN (Learned)':<20} {cnn_test_metrics['accuracy']:<12.1f} {cnn_test_metrics['precision']:<12.1f} {cnn_test_metrics['recall']:<12.1f} {cnn_test_metrics['f1']:<12.1f}")
    print("-" * 72)
    print(f"  CNN Test Confusion Matrix:  TP={cnn_test_metrics['confusion_matrix']['TP']}, FP={cnn_test_metrics['confusion_matrix']['FP']}, FN={cnn_test_metrics['confusion_matrix']['FN']}, TN={cnn_test_metrics['confusion_matrix']['TN']}")
    print(f"  CNN CPU Inference latency:  {cnn_test_metrics['inference_ms_per_patch']:.3f} ms/patch")

    # 4. Save results
    history_csv = out_dir / "cnn_history.csv"
    history_df.to_csv(history_csv, index=False)

    summary = {
        "model_parameters": n_params,
        "epochs": epochs,
        "cpu_threads": num_threads,
        "training_time_seconds": train_time_s,
        "optimal_classical_threshold": opt_thresh,
        "classical_baseline": classical_test_metrics,
        "tiny_cnn": cnn_test_metrics
    }
    if save_json:
        metrics_json = out_dir / "cnn_metrics.json"
        with open(metrics_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    # Save model weights and sample test patches for visualization script
    torch.save(model.state_dict(), str(out_dir / "tiny_cnn.pt"))
    np.savez(
        str(out_dir / "cnn_test_samples.npz"),
        x_test=x_test[:20],
        y_test=y_test[:20]
    )

    print(f"\nSaved training history to: {history_csv}")
    print(f"Saved test metrics to:     {metrics_json}")
    print("=" * 72)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate tiny CNN patch correspondence classifier.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs (default: 15)")
    parser.add_argument("--threads", type=int, default=2, help="Number of CPU threads for PyTorch (default: 2)")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to save metrics (default: results)")
    args = parser.parse_args()

    run_cnn_experiment(epochs=args.epochs, num_threads=args.threads, results_dir=args.results_dir)


if __name__ == "__main__":
    main()
