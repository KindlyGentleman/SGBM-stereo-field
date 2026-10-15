"""Unit test suite for classical stereo matching, calibration, and synthetic scenes."""

import tempfile
from pathlib import Path
import cv2
import numpy as np
import pytest

import stereo


def test_constant_disparity_scene_recovery():
    """Verify known synthetic disparity recovery on a planar scene."""
    shift_px = 16
    left, right, disp_gt, valid_mask = stereo.generate_constant_disparity_scene(
        width=240, height=180, disparity=shift_px, seed=42
    )

    assert left.shape == (180, 240)
    assert right.shape == (180, 240)
    assert disp_gt.shape == (180, 240)
    assert valid_mask.shape == (180, 240)
    assert np.all(disp_gt == float(shift_px))

    # Evaluate SGBM on the constant disparity scene
    matcher = stereo.create_sgbm_matcher(num_disparities=32, block_size=7)
    disp, _ = stereo.compute_disparity_sgbm(left, right, matcher)

    # Evaluate interior non-border valid pixels
    eval_region = valid_mask.copy()
    eval_region[:15, :] = False
    eval_region[-15:, :] = False
    eval_region[:, :15] = False
    eval_region[:, -15:] = False

    valid_pts = disp[eval_region & (disp > 0)]
    assert len(valid_pts) > 0
    median_disp = np.median(valid_pts)
    assert np.isclose(median_disp, shift_px, atol=1.0)

    bad2 = stereo.compute_bad_pixel_rate(disp, disp_gt, eval_region, threshold=2.0)
    assert bad2 < 1.0


def test_layered_scene_geometry_and_visibility():
    """Verify geometry, occlusion handling, and non-occluded mask in layered scene."""
    left, right, disp_gt, valid_mask = stereo.generate_layered_synthetic_scene(
        width=320, height=240, seed=42
    )

    assert left.shape == (240, 320)
    assert right.shape == (240, 320)
    assert disp_gt.shape == (240, 320)
    assert valid_mask.shape == (240, 320)

    # Disparity layers must be 8, 18, and 32
    unique_disps = set(np.unique(disp_gt))
    assert unique_disps == {8.0, 18.0, 32.0}

    # Points on the left view with valid_mask must satisfy: xr = xl - d is inside right image
    ys, xs = np.where(valid_mask)
    for y, xl in zip(ys[::50], xs[::50]):
        d = int(disp_gt[y, xl])
        xr = xl - d
        assert 0 <= xr < 320, f"Projected coordinate {xr} out of bounds"

    # Coverage must be non-trivial (above 80% and below 98% due to real occlusions)
    cov = np.mean(valid_mask) * 100.0
    assert 80.0 < cov < 98.0


def test_deterministic_generation():
    """Verify that identical random seeds produce identical images and disparities."""
    l1, r1, gt1, m1 = stereo.generate_layered_synthetic_scene(320, 240, seed=123)
    l2, r2, gt2, m2 = stereo.generate_layered_synthetic_scene(320, 240, seed=123)

    assert np.array_equal(l1, l2)
    assert np.array_equal(r1, r2)
    assert np.array_equal(gt1, gt2)
    assert np.array_equal(m1, m2)


def test_sad_matching_cost_profile():
    """Verify that educational SAD scanline matching identifies true disparity."""
    left, right, disp_gt, _ = stereo.generate_layered_synthetic_scene(320, 240, seed=42)

    # Pick a point in the textured foreground box (d=32)
    x0, y0 = 110, 100
    true_d = int(disp_gt[y0, x0])
    assert true_d == 32

    disps, costs, best_d, min_cost = stereo.compute_sad_cost_profile(
        left, right, x0, y0, patch_size=9, max_disp=48
    )

    assert len(disps) == 48
    assert len(costs) == 48
    assert best_d == true_d
    assert min_cost >= 0.0


def test_disparity_scaling_and_invalid_handling():
    """Verify OpenCV fixed-point scaling (/ 16.0) and negative sentinel clamping."""
    left, right, _, _ = stereo.generate_constant_disparity_scene(160, 120, disparity=8)

    disp_bm, _ = stereo.compute_disparity_bm(left, right)
    disp_sgbm, _ = stereo.compute_disparity_sgbm(left, right)

    assert disp_bm.dtype == np.float32
    assert disp_sgbm.dtype == np.float32
    assert np.all(disp_bm >= 0.0)
    assert np.all(disp_sgbm >= 0.0)


def test_parameter_constraints():
    """Verify constraints: numDisparities multiple of 16, odd block size."""
    bm = stereo.create_bm_matcher(num_disparities=35, block_size=8)
    assert bm.getNumDisparities() % 16 == 0
    assert bm.getBlockSize() % 2 == 1

    sgbm = stereo.create_sgbm_matcher(num_disparities=45, block_size=6)
    assert sgbm.getNumDisparities() % 16 == 0
    assert sgbm.getBlockSize() % 2 == 1


def test_calibration_validation_and_failures():
    """Verify calibration loading and strict rejection of corrupted archives."""
    with tempfile.TemporaryDirectory() as tmpdir:
        valid_npz = Path(tmpdir) / "valid.npz"
        np.savez(
            valid_npz,
            mtx_left=np.eye(3, dtype=np.float64),
            dist_left=np.zeros((1, 5), dtype=np.float64),
            mtx_right=np.eye(3, dtype=np.float64),
            dist_right=np.zeros((1, 5), dtype=np.float64),
            R=np.eye(3, dtype=np.float64),
            T=np.array([[-0.1], [0.0], [0.0]], dtype=np.float64)
        )
        calib = stereo.load_calibration(str(valid_npz))
        assert "mtx_left" in calib
        assert "T" in calib

        # Missing key must raise KeyError
        bad_npz = Path(tmpdir) / "bad.npz"
        np.savez(
            bad_npz,
            mtx_left=np.eye(3),
            dist_left=np.zeros((1, 5)),
            R=np.eye(3)
        )
        with pytest.raises(KeyError):
            stereo.load_calibration(str(bad_npz))

        # Non-existent file must raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            stereo.load_calibration(str(Path(tmpdir) / "nonexistent.npz"))


def test_disparity_to_depth():
    """Verify metric depth triangulation formula: Z = (f * B) / d."""
    disp = np.array([[10.0, 20.0], [0.0, -2.0]], dtype=np.float32)
    fx = 800.0       # pixels
    baseline = 0.1   # meters (100 mm)

    # Expected: 800 * 0.1 / 10 = 8.0 m, 800 * 0.1 / 20 = 4.0 m, invalid = 0.0
    depth = stereo.disparity_to_depth(disp, fx, baseline, min_disp=0.5)
    assert np.isclose(depth[0, 0], 8.0)
    assert np.isclose(depth[0, 1], 4.0)
    assert depth[1, 0] == 0.0
    assert depth[1, 1] == 0.0


def test_bad_pixel_rate_and_completeness():
    """Verify standard and completeness-aware bad-pixel metric calculation."""
    gt = np.array([
        [10.0, 10.0, 10.0],
        [10.0, 10.0, 10.0]
    ], dtype=np.float32)
    mask = np.ones((2, 3), dtype=bool)

    # Pred:
    # (0,0): 10.0 (err 0, correct)
    # (0,1): 11.5 (err 1.5 <= 2.0, correct)
    # (0,2): 13.5 (err 3.5 > 2.0, wrong)
    # (1,0): 10.0 (err 0, correct)
    # (1,1): 0.0  (missing)
    # (1,2): 0.0  (missing)
    pred = np.array([
        [10.0, 11.5, 13.5],
        [10.0, 0.0, 0.0]
    ], dtype=np.float32)

    # Standard Bad@2: 1 wrong out of 4 valid predictions = 25.0%
    bad_std = stereo.compute_bad_pixel_rate(pred, gt, mask, threshold=2.0, completeness_aware=False)
    assert np.isclose(bad_std, 25.0)

    # Completeness-aware Bad@2: (1 wrong + 2 missing) / 6 total = 50.0%
    bad_ca = stereo.compute_bad_pixel_rate(pred, gt, mask, threshold=2.0, completeness_aware=True)
    assert np.isclose(bad_ca, 50.0)


def test_perturbations():
    """Verify controlled photometric perturbation functions."""
    img = np.full((50, 50), 120, dtype=np.uint8)

    b_up = stereo.apply_perturbation(img, "brightness_up")
    assert np.all(b_up == 150)

    b_down = stereo.apply_perturbation(img, "brightness_down")
    assert np.all(b_down == 90)

    noise = stereo.apply_perturbation(img, "gaussian_noise", seed=42)
    assert noise.shape == img.shape
    assert not np.all(noise == 120)

    gamma = stereo.apply_perturbation(img, "gamma_high")
    assert gamma.shape == img.shape
