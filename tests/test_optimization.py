"""Unit test suite for filter suite, pipeline presets, and CPU optimization."""

import cv2
import numpy as np
import pytest

import optimize
import stereo


def test_filter_suite_application():
    """Verify application of all post-processing filter modes."""
    left, right, _, _ = stereo.generate_constant_disparity_scene(160, 120, disparity=16)
    m = stereo.create_sgbm_matcher(num_disparities=32, block_size=5)
    rm = cv2.ximgproc.createRightMatcher(m) if hasattr(cv2, "ximgproc") else None

    dl_raw = m.compute(left, right)
    disp_l = dl_raw.astype(np.float32) / 16.0
    disp_l[dl_raw <= -16] = -1.0

    dr_raw = rm.compute(right, left) if rm is not None else dl_raw

    # Test raw, median_valid, guided, joint_bilateral, wls, confidence_edge
    f_raw = optimize.apply_filter("raw", disp_l, left)
    assert f_raw.shape == disp_l.shape
    assert np.any(np.isclose(f_raw[f_raw >= 0], 16.0, atol=2.0))

    f_med = optimize.apply_filter("median_valid", disp_l, left)
    assert f_med.shape == disp_l.shape

    f_guided = optimize.apply_filter("guided", disp_l, left)
    assert f_guided.shape == disp_l.shape

    left_bgr = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
    f_jb = optimize.apply_filter("joint_bilateral", disp_l, left_bgr)
    assert f_jb.shape == disp_l.shape

    f_wls = optimize.apply_filter("wls", disp_l, left, disp_r_raw=dr_raw, matcher=m)
    assert f_wls.shape == disp_l.shape

    f_conf = optimize.apply_filter("confidence_edge", disp_l, left, disp_r_raw=dr_raw, matcher=m)
    assert f_conf.shape == disp_l.shape



def test_pipeline_presets_execution():
    """Verify FAST, BALANCED, and QUALITY presets execute and return positive metrics."""
    # Verify presets have distinct, strictly monotonic scales and disparity ranges
    scales = [optimize.PIPELINE_PRESETS[p]["scale"] for p in ["FAST", "BALANCED", "QUALITY"]]
    ndisps = [optimize.PIPELINE_PRESETS[p]["numDisparities"] for p in ["FAST", "BALANCED", "QUALITY"]]
    assert scales == [0.5, 0.75, 1.0], f"Presets must use distinct scales: {scales}"
    assert ndisps == [80, 128, 160], f"Presets must have strictly increasing disparities: {ndisps}"
    # Verify block size is 5 (Pareto-dominant over bs=7)
    for p in ["FAST", "BALANCED", "QUALITY"]:
        assert optimize.PIPELINE_PRESETS[p]["blockSize"] == 5

    img_l = np.random.randint(40, 200, (180, 240, 3), dtype=np.uint8)
    img_r = np.roll(img_l, -8, axis=1)

    for preset_name in ["FAST", "BALANCED", "QUALITY"]:
        disp, stats = optimize.run_stereo_pipeline(img_l, img_r, preset=preset_name)

        assert disp.shape == (180, 240)
        assert stats["preset"] == preset_name
        assert stats["wall_ms"] > 0.0
        assert stats["cpu_ms"] >= 0.0
        assert 0.0 <= stats["valid_coverage_pct"] <= 100.0


def test_invalid_filter_and_preset_exceptions():
    """Verify exceptions on unknown filter names or presets."""
    dummy = np.zeros((50, 50), dtype=np.float32)
    img = np.zeros((50, 50), dtype=np.uint8)

    with pytest.raises(ValueError, match="Unknown filter name"):
        optimize.apply_filter("non_existent_filter", dummy, img)

    with pytest.raises(ValueError, match="Unknown preset"):
        optimize.run_stereo_pipeline(img, img, preset="SUPER_FAST")


def test_median_valid_no_zero_bleed_at_borders():
    """Verify that median_valid does not collapse boundary valid disparities to 0.0."""
    # Create 10x10 map: left 5 columns valid with disparity 24.0, right 5 columns invalid (-1.0)
    disp = np.full((10, 10), -1.0, dtype=np.float32)
    disp[:, :5] = 24.0
    img = np.zeros((10, 10), dtype=np.uint8)

    f_med = optimize.apply_filter("median_valid", disp, img)

    # Column 4 is right at the border of invalid pixels.
    # Under old zero-fill, more than half of the 3x3 window was 0.0, collapsing column 4 to 0.0.
    # Under dilate-fill, column 4 must remain valid with disparity 24.0.
    assert np.all(f_med[:, :5] == 24.0)
    assert np.all(f_med[:, 5:] == -1.0)


def test_compute_pareto_frontier():
    """Verify Pareto frontier correctly identifies non-dominated configurations."""
    import pandas as pd
    df = pd.DataFrame([
        {"name": "A", "total_ms": 20.0, "bad2_pct": 10.0},  # Pareto optimal (fastest)
        {"name": "B", "total_ms": 30.0, "bad2_pct": 5.0},   # Pareto optimal (most accurate)
        {"name": "C", "total_ms": 25.0, "bad2_pct": 12.0},  # Dominated by A (slower and higher error)
        {"name": "D", "total_ms": 35.0, "bad2_pct": 8.0},   # Dominated by B (slower and higher error)
    ])

    is_pareto = optimize.compute_pareto_frontier(df, latency_col="total_ms", error_col="bad2_pct")
    assert list(is_pareto) == [True, True, False, False]


def test_cpu_profiling_runs_and_returns_structure():
    """Verify run_cpu_profiling executes and records resolutions, threads, and presets."""
    prof = optimize.run_cpu_profiling(iterations=1, save_json=False)
    assert "resolutions" in prof
    assert "threads" in prof
    assert "presets" in prof
    assert "720p" in prof["resolutions"]
    assert "360p" in prof["resolutions"]
    assert "180p" in prof["resolutions"]
    assert "FAST" in prof["presets"]
    assert "BALANCED" in prof["presets"]
    assert "QUALITY" in prof["presets"]

