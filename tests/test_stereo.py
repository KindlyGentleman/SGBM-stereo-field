"""Unit test suite for stereo matching, calibration, and metric calculations."""

import tempfile
from pathlib import Path
import cv2
import numpy as np
import pytest

import stereo


def test_synthetic_calibration_loading():
    """Verify safe calibration loading from NPZ archive."""
    with tempfile.TemporaryDirectory() as tmpdir:
        calib_file = Path(tmpdir) / "test_calib.npz"
        np.savez(
            calib_file,
            mtx_left=np.eye(3, dtype=np.float64),
            dist_left=np.zeros((1, 5), dtype=np.float64),
            mtx_right=np.eye(3, dtype=np.float64),
            dist_right=np.zeros((1, 5), dtype=np.float64),
            R=np.eye(3, dtype=np.float64),
            T=np.array([[-97.5], [0.0], [0.0]], dtype=np.float64)
        )
        
        calib = stereo.load_calibration(str(calib_file))
        assert "mtx_left" in calib
        assert "T" in calib
        assert np.isclose(calib["T"][0, 0], -97.5)


def test_rectification_maps_generation():
    """Verify stereoRectify and remap map creation."""
    calib = {
        "mtx_left": np.array([[800.0, 0, 320.0], [0, 800.0, 240.0], [0, 0, 1.0]]),
        "dist_left": np.zeros((1, 5)),
        "mtx_right": np.array([[800.0, 0, 320.0], [0, 800.0, 240.0], [0, 0, 1.0]]),
        "dist_right": np.zeros((1, 5)),
        "R": np.eye(3),
        "T": np.array([[-100.0], [0.0], [0.0]])
    }
    maps_l, maps_r, Q = stereo.build_rectification_maps(calib, (640, 480))
    assert maps_l[0].shape == (480, 640)
    assert maps_r[0].shape == (480, 640)
    assert Q.shape == (4, 4)


def test_synthetic_stereo_shift():
    """Test SGBM disparity estimation on a known horizontal pixel shift."""
    h, w = 180, 240
    rng = np.random.default_rng(42)
    
    # Create random textured base image
    base = rng.integers(30, 220, size=(h, w), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (5, 5), 1.5)
    
    # Shift right image horizontally by shift_px
    shift_px = 16
    left = base.copy()
    right = np.zeros_like(base)
    right[:, :-shift_px] = base[:, shift_px:]
    
    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=32,
        blockSize=7,
        P1=100,
        P2=800,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    
    disp = matcher.compute(left, right).astype(np.float32) / 16.0
    
    # Evaluate interior region away from borders
    interior = disp[30:-30, 40:-40]
    valid_pts = interior[interior > 0]
    
    assert len(valid_pts) > 0
    median_disp = np.median(valid_pts)
    assert np.isclose(median_disp, shift_px, atol=2.0)


def test_completeness_aware_bad_pixel():
    """Verify that missing predictions are penalized in completeness-aware metric."""
    gt = np.array([
        [10.0, 10.0, 10.0],
        [10.0, 10.0, 10.0]
    ], dtype=np.float32)
    
    mask = np.ones((2, 3), dtype=bool)
    
    # Pred has:
    # 2 correct (error = 0)
    # 2 wrong (error = 4.0 > 2.0)
    # 2 missing (disp <= 0)
    pred = np.array([
        [10.0, 10.0, 14.0],
        [14.0, 0.0, -1.0]
    ], dtype=np.float32)
    
    # Total GT valid = 6
    # Failures = 2 wrong + 2 missing = 4
    # Expected bad rate = 4 / 6 * 100 = 66.67%
    bad_rate = stereo.compute_completeness_aware_bad_pixel(pred, gt, mask, threshold=2.0)
    assert np.isclose(bad_rate, 66.6667, atol=0.01)


def test_valid_coverage():
    """Verify coverage calculation."""
    disp = np.array([
        [12.0, 0.0],
        [5.0, -1.0]
    ], dtype=np.float32)
    # 2 out of 4 pixels valid -> 50%
    cov = stereo.compute_valid_coverage(disp)
    assert np.isclose(cov, 50.0)


def test_lr_consistency():
    """Verify Left-Right consistency check logic."""
    disp_l = np.array([
        [10.0, 10.0, 10.0],
        [10.0, 10.0, 10.0]
    ], dtype=np.float32)
    
    disp_r = np.array([
        [10.0, 10.0, 10.0],
        [10.0, 10.0, 10.0]
    ], dtype=np.float32)
    
    cons_pct, mask = stereo.compute_lr_consistency(disp_l, disp_r, threshold=1.5)
    assert np.isclose(cons_pct, 100.0)


def test_reproject_to_pointcloud():
    """Verify 3D metric reprojection with Q matrix."""
    calib = {
        "mtx_left": np.array([[800.0, 0, 320.0], [0, 800.0, 240.0], [0, 0, 1.0]]),
        "dist_left": np.zeros((1, 5)),
        "mtx_right": np.array([[800.0, 0, 320.0], [0, 800.0, 240.0], [0, 0, 1.0]]),
        "dist_right": np.zeros((1, 5)),
        "R": np.eye(3),
        "T": np.array([[-0.100], [0.0], [0.0]])  # 100 mm in meters
    }
    _, _, Q = stereo.build_rectification_maps(calib, (640, 480))
    disp = np.ones((10, 10), dtype=np.float32) * 20.0
    
    pts, _ = stereo.reproject_to_pointcloud(disp, Q, min_disp=1.0, max_depth=50.0)
    assert len(pts) == 100
    assert pts.shape[1] == 3
    assert np.all(np.isfinite(pts))


def test_disparity_to_depth():
    """Verify disparity to metric depth conversion: Z = (fx * B) / d."""
    disp = np.array([[10.0, 20.0], [0.0, -5.0]], dtype=np.float32)
    fx = 1000.0
    baseline = 0.1  # 100 mm in meters
    # Expected: 1000 * 0.1 / 10.0 = 10.0 m, 1000 * 0.1 / 20.0 = 5.0 m, others 0.0
    depth = stereo.disparity_to_depth(disp, fx, baseline, min_disp=0.5)
    assert np.isclose(depth[0, 0], 10.0)
    assert np.isclose(depth[0, 1], 5.0)
    assert depth[1, 0] == 0.0
    assert depth[1, 1] == 0.0


def test_compute_metric_depth_errors():
    """Verify metric depth statistics calculation."""
    gt = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    # pred has small error
    pred = np.array([1.1, 1.9, 3.1, 4.2], dtype=np.float32)
    metrics = stereo.compute_metric_depth_errors(pred, gt, min_z=0.5, max_z=4.5)
    assert metrics["eval_pixels"] == 4
    assert np.isclose(metrics["ref_coverage_pct"], 100.0)
    assert np.isclose(metrics["mae_m"], 0.125, atol=0.01)
    assert metrics["delta_125_pct"] == 100.0


def test_apply_filter_suite():
    """Verify filter execution across candidate algorithms."""
    disp = np.array([
        [10.0, 12.0, 0.0],
        [11.0, 10.0, 0.0],
        [9.0,  11.0, 0.0]
    ], dtype=np.float32)
    left_img = np.zeros((3, 3, 3), dtype=np.uint8)

    f0 = stereo.apply_filter("f0_raw", disp, left_img)
    assert f0.shape == disp.shape
    assert f0[0, 2] == 0.0

    f2 = stereo.apply_filter("f2_median_valid", disp, left_img)
    assert f2.shape == disp.shape
    assert f2[0, 2] == 0.0

    f3 = stereo.apply_filter("f3_guided", disp, left_img)
    assert f3.shape == disp.shape
    assert f3[0, 2] == 0.0


def test_rectification_cache():
    """Verify rectification map caching and cache invalidation."""
    stereo.clear_rectification_cache()
    calib = {
        "mtx_left": np.eye(3, dtype=np.float64),
        "dist_left": np.zeros((1, 5), dtype=np.float64),
        "mtx_right": np.eye(3, dtype=np.float64),
        "dist_right": np.zeros((1, 5), dtype=np.float64),
        "R": np.eye(3, dtype=np.float64),
        "T": np.array([[-0.1], [0.0], [0.0]], dtype=np.float64)
    }
    maps1, _, _ = stereo.build_rectification_maps(calib, (320, 240), use_cache=True)
    maps2, _, _ = stereo.build_rectification_maps(calib, (320, 240), use_cache=True)
    assert maps1[0] is maps2[0]
    stereo.clear_rectification_cache()
    maps3, _, _ = stereo.build_rectification_maps(calib, (320, 240), use_cache=True)
    assert maps1[0] is not maps3[0]


def test_run_stereo_pipeline_presets():
    """Verify execution of FAST, BALANCED, and QUALITY pipeline presets."""
    rng = np.random.default_rng(123)
    base = rng.integers(40, 210, size=(120, 160), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (3, 3), 1.0)
    left = base.copy()
    right = np.roll(base, -4, axis=1)

    for preset in ["FAST", "BALANCED", "QUALITY"]:
        disp, stats = stereo.run_stereo_pipeline(left, right, preset=preset)
        assert disp.shape == (120, 160)
        assert disp.dtype == np.float32
        assert stats["preset"] == preset
        assert stats["wall_ms"] > 0.0


