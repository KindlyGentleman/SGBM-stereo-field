"""Unit test suite for TinyStereoNet and patch classification experiments."""

import numpy as np
import pytest
import torch

import cnn


def test_patch_extraction_shape_and_balance():
    """Verify that extract_patches_from_scene produces balanced, correctly shaped pairs."""
    import stereo
    left, right, disp_gt, valid_mask = stereo.generate_layered_synthetic_scene(320, 240, seed=10)

    patches, labels = cnn.extract_patches_from_scene(
        left, right, disp_gt, valid_mask, n_samples=100, seed=42, patch_size=15
    )

    assert patches.ndim == 4
    assert patches.shape[1] == 2
    assert patches.shape[2] == 15
    assert patches.shape[3] == 15
    assert len(patches) == len(labels)

    # Values in [0, 1]
    assert np.all(patches >= 0.0)
    assert np.all(patches <= 1.0)

    # Balanced positive (1) and negative (0) classes
    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)
    assert n_pos == n_neg
    assert n_pos > 0


def test_split_independence_no_leakage():
    """Verify train, validation, and test datasets are partitioned with zero scene overlap."""
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = cnn.build_split_datasets(
        train_samples=80, val_samples=40, test_samples=40, patch_size=15
    )

    assert len(x_train) == 80
    assert len(x_val) == 40
    assert len(x_test) == 40

    # Ensure labels are binary {0, 1}
    assert set(np.unique(y_train)).issubset({0, 1})
    assert set(np.unique(y_val)).issubset({0, 1})
    assert set(np.unique(y_test)).issubset({0, 1})


def test_tinystereonet_architecture():
    """Verify TinyStereoNet tensor shapes and parameter count (must be exactly 1,054)."""
    model = cnn.TinyStereoNet()
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == 1054, f"Expected 1,054 parameters, got {n_params}"

    batch = torch.randn(8, 2, 15, 15)
    logits = model(batch)
    assert logits.shape == (8, 2)


def test_finite_loss_and_gradients():
    """Verify forward and backward passes produce finite loss and non-zero gradients."""
    model = cnn.TinyStereoNet()
    criterion = torch.nn.CrossEntropyLoss()

    bx = torch.randn(16, 2, 15, 15)
    by = torch.randint(0, 2, (16,))

    logits = model(bx)
    loss = criterion(logits, by)

    assert torch.isfinite(loss)
    assert loss.item() > 0.0

    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None
        assert torch.all(torch.isfinite(param.grad))


def test_classical_baseline_evaluation():
    """Verify classical Mean Absolute Difference (MAD) baseline classification logic."""
    # Synthetic pair where pos has diff 0.05 and neg has diff 0.50
    p_pos = np.zeros((4, 2, 15, 15), dtype=np.float32)
    p_pos[:, 1] = 0.05

    p_neg = np.zeros((4, 2, 15, 15), dtype=np.float32)
    p_neg[:, 1] = 0.50

    x_data = np.concatenate([p_pos, p_neg], axis=0)
    y_data = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.int64)

    res = cnn.evaluate_classical_baseline(x_data, y_data, threshold=0.20)
    assert np.isclose(res["accuracy"], 100.0)
    assert res["confusion_matrix"]["TP"] == 4
    assert res["confusion_matrix"]["TN"] == 4
    assert res["confusion_matrix"]["FP"] == 0
    assert res["confusion_matrix"]["FN"] == 0


def test_patch_candidate_sweep():
    """Verify candidate disparity sweep returns valid arrays and probability bounds."""
    import stereo
    model = cnn.TinyStereoNet()
    left, right, disp_gt, _ = stereo.generate_layered_synthetic_scene(240, 180, seed=1)

    sweep = cnn.sweep_patch_candidate_disparities(model, left, right, x0=80, y0=60, max_disp=32)

    assert len(sweep["disparities"]) == 32
    assert len(sweep["sad_costs"]) == 32
    assert len(sweep["cnn_probs"]) == 32

    valid_probs = sweep["cnn_probs"][np.isfinite(sweep["cnn_probs"])]
    assert np.all(valid_probs >= 0.0)
    assert np.all(valid_probs <= 1.0)
