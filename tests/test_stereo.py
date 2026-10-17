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
    """Verify OpenCV fixed-point scaling (/ 16.0) and negative invalid sentinel preservation."""
    left, right, _, _ = stereo.generate_constant_disparity_scene(160, 120, disparity=8)

    disp_bm, _ = stereo.compute_disparity_bm(left, right)
    disp_sgbm, _ = stereo.compute_disparity_sgbm(left, right)

    assert disp_bm.dtype == np.float32
    assert disp_sgbm.dtype == np.float32

    # Border or unmatched pixels have negative invalid sentinels (-1.0)
    # Valid matches have disp >= 0.0 (near 8.0 px)
    assert np.any(disp_bm < 0.0), "BM should preserve invalid sentinel for unmatched border pixels"
    assert np.any(disp_sgbm < 0.0), "SGBM should preserve invalid sentinel for unmatched border pixels"
    assert np.any(np.isclose(disp_sgbm[disp_sgbm >= 0.0], 8.0, atol=1.5)), "Valid matches must be near 8.0 px"

    # Test detailed container
    res = stereo.compute_disparity_sgbm_detailed(left, right)
    assert isinstance(res, stereo.DisparityResult)
    assert res.raw.dtype == np.int16
    assert res.disp.dtype == np.float32
    assert res.valid_mask.dtype == bool
    assert res.valid_mask.shape == (120, 160)


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
    """Verify metric depth triangulation formula: Z = (f * B) / (d + doffs)."""
    disp = np.array([[10.0, 20.0], [-1.0, 0.0]], dtype=np.float32)
    fx = 800.0       # pixels
    baseline = 0.1   # meters (100 mm)

    # Expected: 800 * 0.1 / 10 = 8.0 m, 800 * 0.1 / 20 = 4.0 m
    # With min_disp=0.5:
    # disp -1.0 -> 0.0 (invalid)
    # disp 0.0 -> 0.0 (below min_disp)
    depth = stereo.disparity_to_depth(disp, fx, baseline, min_disp=0.5)
    assert np.isclose(depth[0, 0], 8.0)
    assert np.isclose(depth[0, 1], 4.0)
    assert depth[1, 0] == 0.0
    assert depth[1, 1] == 0.0

    # Test with doffs: Z = 800 * 0.1 / (10.0 + 10.0) = 4.0 m
    depth_doffs = stereo.disparity_to_depth(disp, fx, baseline, min_disp=0.5, doffs=10.0)
    assert np.isclose(depth_doffs[0, 0], 4.0)


def test_comprehensive_hand_calculated_metrics():
    """Verify all Part 4.4 hand-calculated metric conditions and invariants.

    Conditions tested:
    1. Fully correct predictions
    2. Fully incorrect predictions
    3. Fully missing predictions (empty valid predictions)
    4. Mixed missing and incorrect predictions
    5. Valid zero disparity (must be recognized as valid, not missing)
    6. Negative invalid sentinels (-1.0, -16.0)
    7. Nonfinite ground truth (NaN, Inf)
    8. Non-occluded masks
    9. Empty valid reference region
    10. Invariance to unrelated pixels outside evaluation mask
    """
    # 1. Fully correct predictions
    gt = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    pred = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    mask = np.ones(3, dtype=bool)

    m1 = stereo.compute_full_disparity_metrics(pred, gt, mask, thresholds=(1.0, 2.0))
    assert m1["eval_pixels"] == 3
    assert m1["valid_predictions"] == 3
    assert np.isclose(m1["coverage_pct"], 100.0)
    assert np.isclose(m1["invalid_pct"], 0.0)
    assert np.isclose(m1["bad2_pct"], 0.0)
    assert np.isclose(m1["bad2_ca_pct"], 0.0)
    assert np.isclose(m1["disp_mae_px"], 0.0)
    assert np.isclose(m1["disp_rmse_px"], 0.0)
    assert np.isclose(m1["disp_median_ae_px"], 0.0)

    # 2. Fully incorrect predictions
    pred_bad = np.array([15.0, 25.0, 35.0], dtype=np.float32)  # all errors = 5.0 px > 2.0
    m2 = stereo.compute_full_disparity_metrics(pred_bad, gt, mask, thresholds=(1.0, 2.0))
    assert np.isclose(m2["bad2_pct"], 100.0)
    assert np.isclose(m2["bad2_ca_pct"], 100.0)
    assert np.isclose(m2["disp_mae_px"], 5.0)
    assert np.isclose(m2["disp_rmse_px"], 5.0)
    assert np.isclose(m2["disp_median_ae_px"], 5.0)

    # 3. Fully missing predictions (all sentinels)
    pred_missing = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    m3 = stereo.compute_full_disparity_metrics(pred_missing, gt, mask, thresholds=(1.0, 2.0))
    assert m3["eval_pixels"] == 3
    assert m3["valid_predictions"] == 0
    assert np.isclose(m3["coverage_pct"], 0.0)
    assert np.isclose(m3["invalid_pct"], 100.0)
    assert np.isnan(m3["bad2_pct"]), "Conditional Bad@T must be NaN when no valid predictions exist"
    assert np.isclose(m3["bad2_ca_pct"], 100.0), "Completeness-aware Bad@T must be 100% when all missing"
    assert np.isnan(m3["disp_mae_px"])
    assert np.isnan(m3["disp_rmse_px"])
    assert np.isnan(m3["disp_median_ae_px"])

    # 4. Mixed: 4 pixels total
    # pixel 0: gt 10, pred 10.0 (error 0.0, correct)
    # pixel 1: gt 10, pred 11.5 (error 1.5 <= 2.0, correct at bad@2, wrong at bad@1)
    # pixel 2: gt 10, pred 14.0 (error 4.0 > 2.0, wrong at bad@2)
    # pixel 3: gt 10, pred -1.0 (missing)
    gt4 = np.array([10.0, 10.0, 10.0, 10.0], dtype=np.float32)
    pred4 = np.array([10.0, 11.5, 14.0, -1.0], dtype=np.float32)
    mask4 = np.ones(4, dtype=bool)

    m4 = stereo.compute_full_disparity_metrics(pred4, gt4, mask4, thresholds=(1.0, 2.0))
    assert m4["eval_pixels"] == 4
    assert m4["valid_predictions"] == 3
    assert np.isclose(m4["coverage_pct"], 75.0)  # 3/4
    assert np.isclose(m4["invalid_pct"], 25.0)   # 1/4

    # Valid errors: 0.0, 1.5, 4.0.
    # At Bad@2: 1 wrong out of 3 valid = 33.333%
    assert np.isclose(m4["bad2_pct"], 100.0 / 3.0)
    # At Bad@2 Completeness-Aware: (1 wrong + 1 missing) / 4 total = 50.0%
    assert np.isclose(m4["bad2_ca_pct"], 50.0)
    # Check formula equality: (1 - 0.75) + 0.75 * (1/3) = 0.50
    assert np.isclose(m4["bad2_ca_pct"], (1.0 - 0.75 + 0.75 * (1.0 / 3.0)) * 100.0)

    # MAE over 3 valid = (0.0 + 1.5 + 4.0) / 3 = 5.5 / 3 = 1.8333
    assert np.isclose(m4["disp_mae_px"], 5.5 / 3.0)
    # Median over 3 valid = 1.5
    assert np.isclose(m4["disp_median_ae_px"], 1.5)

    # 5. Valid zero disparity
    gt_zero = np.array([0.0, 10.0], dtype=np.float32)
    pred_zero = np.array([0.0, 10.0], dtype=np.float32)
    m_zero = stereo.compute_full_disparity_metrics(pred_zero, gt_zero, np.ones(2, dtype=bool))
    assert m_zero["eval_pixels"] == 2
    assert m_zero["valid_predictions"] == 2, "Disparity 0.0 must be recognized as valid match"
    assert np.isclose(m_zero["coverage_pct"], 100.0)
    assert np.isclose(m_zero["bad2_pct"], 0.0)

    # 6. Nonfinite ground truth (NaN, Inf) and negative invalid sentinels
    gt_nan = np.array([np.nan, np.inf, 10.0, 10.0], dtype=np.float32)
    pred_nan = np.array([10.0, 10.0, 10.0, -16.0], dtype=np.float32)
    m_nan = stereo.compute_full_disparity_metrics(pred_nan, gt_nan, np.ones(4, dtype=bool))
    # Only 2 finite reference pixels exist (pixels 2 and 3)
    assert m_nan["eval_pixels"] == 2
    # Pixel 2 is valid (10), Pixel 3 is invalid sentinel (-16)
    assert m_nan["valid_predictions"] == 1
    assert np.isclose(m_nan["coverage_pct"], 50.0)
    assert np.isclose(m_nan["bad2_pct"], 0.0)
    assert np.isclose(m_nan["bad2_ca_pct"], 50.0)

    # 7. Empty valid reference region
    gt_empty = np.array([np.nan, -5.0], dtype=np.float32)
    m_empty = stereo.compute_full_disparity_metrics(np.zeros(2), gt_empty, np.array([False, False]))
    assert m_empty["eval_pixels"] == 0
    assert m_empty["coverage_pct"] == 0.0
    assert np.isnan(m_empty["bad2_pct"])
    assert np.isnan(m_empty["bad2_ca_pct"])

    # 8. Invariance to unrelated pixels outside evaluation mask
    gt_inv = np.array([10.0, 999.0], dtype=np.float32)
    pred_inv = np.array([10.0, -999.0], dtype=np.float32)
    mask_inv = np.array([True, False], dtype=bool)
    m_inv = stereo.compute_full_disparity_metrics(pred_inv, gt_inv, mask_inv)
    assert m_inv["eval_pixels"] == 1
    assert m_inv["valid_predictions"] == 1
    assert np.isclose(m_inv["coverage_pct"], 100.0)
    assert np.isclose(m_inv["bad2_pct"], 0.0)
    assert np.isclose(m_inv["disp_mae_px"], 0.0)



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


def test_depth_sensitivity_and_physics_scaling():
    """Verify depth sensitivity across distances and resolution scaling invariance."""
    f_px = 964.68366
    b_m = 0.097497
    fb = f_px * b_m
    assert np.isclose(fb, 94.0537, atol=0.001)

    # Test 1m, 2m, 5m, 10m, 15m theoretical sensitivities for 1 px error
    sens_1m = stereo.compute_depth_sensitivity(1.0, f_px, b_m, delta_disp_px=1.0)
    assert np.isclose(sens_1m["delta_z_first_order_m"], 0.01063, atol=1e-4)

    sens_2m = stereo.compute_depth_sensitivity(2.0, f_px, b_m, delta_disp_px=1.0)
    assert np.isclose(sens_2m["delta_z_first_order_m"], 0.04253, atol=1e-4)

    sens_5m = stereo.compute_depth_sensitivity(5.0, f_px, b_m, delta_disp_px=1.0)
    assert np.isclose(sens_5m["delta_z_first_order_m"], 0.2658, atol=1e-3)

    sens_10m = stereo.compute_depth_sensitivity(10.0, f_px, b_m, delta_disp_px=1.0)
    assert np.isclose(sens_10m["delta_z_first_order_m"], 1.0632, atol=1e-3)

    sens_15m = stereo.compute_depth_sensitivity(15.0, f_px, b_m, delta_disp_px=1.0)
    assert np.isclose(sens_15m["delta_z_first_order_m"], 2.3922, atol=1e-3)

    # Verify exact non-linear asymmetry: farther error > closer error
    assert sens_5m["delta_z_exact_farther_m"] > sens_5m["delta_z_exact_closer_m"]

    # Resolution scaling invariance:
    # If image is scaled by factor s, focal length is f*s, disparity is d*s
    # Depth Z = (f*s * B) / (d*s) = fB / d
    s = 0.5
    d_full = np.array([20.0, 40.0], dtype=np.float32)
    d_half = d_full * s
    z_full = stereo.disparity_to_depth(d_full, f_px, b_m)
    z_half = stereo.disparity_to_depth(d_half, f_px * s, b_m)
    assert np.allclose(z_full, z_half)


def test_constant_disparity_visibility_mask_boundary():
    """Verify constant scene visibility mask correctly identifies left unmatchable border."""
    for d in [8, 16, 24]:
        for w, h in [(160, 120), (240, 180), (320, 240)]:
            left, right, disp_gt, valid_mask = stereo.generate_constant_disparity_scene(
                width=w, height=h, disparity=d, seed=42
            )
            # Left border pixels [0, d) in left view project to x_R = x_L - d < 0 (out of bounds)
            assert np.all(~valid_mask[:, :d]), f"Left border [0, {d}) must be invalid"
            # Pixels [d, w) in left view project to x_R = x_L - d in [0, w - d) (in bounds)
            assert np.all(valid_mask[:, d:]), f"Interior region [{d}, {w}) must be valid"

            # Check that every valid pixel has exact correspondence: left(y, x) == right(y, x - d)
            for y in range(0, h, 20):
                for x in range(d, w, 20):
                    assert left[y, x] == right[y, x - d], f"Pixel mismatch at ({x}, {y})"


def test_lr_consistency_geometry():
    """Verify left-right consistency geometry, out-of-bounds rejection, and sentinel handling."""
    h, w = 60, 80
    disp_l = np.full((h, w), 10.0, dtype=np.float32)
    # Right matcher convention: matching disparity magnitude is 10.0
    disp_r = np.full((h, w), -10.0, dtype=np.float32)

    # 1. Left border pixels x < 10 project to xr = x - 10 < 0 (out of bounds)
    pct, mask = stereo.compute_lr_consistency(disp_l, disp_r, threshold=1.0)
    assert np.all(~mask[:, :10]), "Out of bounds left border must fail consistency"
    assert np.all(mask[:, 10:]), "Valid in-bounds matches must pass consistency"
    expected_pct = ((w - 10) / w) * 100.0
    assert np.isclose(pct, expected_pct)

    # 2. Corrupt some right disparities with invalid sentinel -1.0
    disp_r_corrupt = disp_r.copy()
    disp_r_corrupt[:, 20:30] = -1.0
    pct2, mask2 = stereo.compute_lr_consistency(disp_l, disp_r_corrupt, threshold=1.0)
    # Left pixels that project into [20, 30) are [30, 40)
    assert np.all(~mask2[:, 30:40]), "Pixels matching invalid right disparity must fail consistency"

    # 3. Disparity discrepancy exceeding threshold fails
    disp_r_bad = disp_r.copy()
    disp_r_bad[:, 50:] = -15.0  # discrepancy = |10 - 15| = 5.0 > 1.0
    pct3, mask3 = stereo.compute_lr_consistency(disp_l, disp_r_bad, threshold=1.0)
    # Left pixels that project into [50, w) are [60, w)
    assert np.all(~mask3[:, 60:]), "Pixels with discrepancy > threshold must fail consistency"


def test_photometric_warp_mae_geometry():
    """Verify photometric warp MAE excludes out-of-bounds coordinates and handles sentinels."""
    h, w = 50, 60
    # Clean pattern
    rng = np.random.default_rng(42)
    left = rng.integers(30, 220, (h, w), dtype=np.uint8)
    disp = 12.0
    disp_map = np.full((h, w), disp, dtype=np.float32)

    # Construct right view shifted by exact disparity
    right = np.zeros_like(left)
    right[:, :w - int(disp)] = left[:, int(disp):]

    mae = stereo.compute_photometric_warp_mae(left, right, disp_map)
    # Since left[:, 12:] matches right[:, :48], warp MAE over valid region must be 0.0
    assert np.isclose(mae, 0.0, atol=1e-5)

    # All-invalid disparity map must return NaN, not a magic sentinel like 999.0
    disp_invalid = np.full((h, w), -1.0, dtype=np.float32)
    mae_nan = stereo.compute_photometric_warp_mae(left, right, disp_invalid)
    assert np.isnan(mae_nan)


def test_reproject_to_pointcloud_geometry():
    """Verify Q-matrix pointcloud reprojection with known metric coordinates."""
    # Construct an ideal disparity-to-depth Q matrix
    # [1  0   0    -cx  ]
    # [0  1   0    -cy  ]
    # [0  0   0     f   ]
    # [0  0  -1/B  (cx-cx')/B]
    f = 500.0
    B = 0.1
    cx = 32.0
    cy = 24.0
    Q = np.array([
        [1.0, 0.0, 0.0, -cx],
        [0.0, 1.0, 0.0, -cy],
        [0.0, 0.0, 0.0, f],
        [0.0, 0.0, 1.0 / B, 0.0]
    ], dtype=np.float32)

    disp = np.full((48, 64), 10.0, dtype=np.float32)
    # Invalidate a pixel
    disp[0, 0] = -1.0

    pts, _ = stereo.reproject_to_pointcloud(disp, Q, min_disp=0.5, max_depth=50.0)
    assert len(pts) == 48 * 64 - 1
    assert np.all(np.isfinite(pts))
    # At d=10.0, Z = fB / d = 500 * 0.1 / 10 = 5.0 m
    assert np.allclose(pts[:, 2], 5.0, atol=0.1)


