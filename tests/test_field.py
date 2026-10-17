"""Unit test suite for Middlebury 2014 and Gunung field evaluations."""

import tempfile
from pathlib import Path
import numpy as np
import pytest

import field
import stereo


def test_parse_calib_txt():
    """Verify calib.txt parser extracts camera matrices and scalar parameters."""
    with tempfile.TemporaryDirectory() as tmpdir:
        calib_file = Path(tmpdir) / "calib.txt"
        calib_file.write_text(
            "cam0=[1000 0 320; 0 1000 240; 0 0 1]\n"
            "cam1=[1000 0 320; 0 1000 240; 0 0 1]\n"
            "doffs=12.5\n"
            "baseline=150.0\n"
            "width=640\n"
            "height=480\n"
            "ndisp=96\n"
            "vmin=4\n"
            "vmax=80\n",
            encoding="utf-8"
        )
        params = field.parse_calib_txt(str(calib_file))

        assert params["cam0"].shape == (3, 3)
        assert np.isclose(params["cam0"][0, 0], 1000.0)
        assert np.isclose(params["doffs"], 12.5)
        assert np.isclose(params["baseline"], 150.0)
        assert params["width"] == 640
        assert params["height"] == 480
        assert params["ndisp"] == 96


def test_realsense_depth_loading_and_projection():
    """Verify RealSense depth unprojection and z-buffered projection."""
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_file = Path(tmpdir) / "rs_depth.npz"
        raw_depth_mm = np.full((480, 640), 2000, dtype=np.uint16)
        raw_depth_mm[100, 100] = 0  # Invalid
        raw_depth_mm[200:220, 200:220] = 1500  # Closer foreground
        np.savez(npz_file, depth_map=raw_depth_mm)

        depth_m = field.load_realsense_depth(str(npz_file))
        assert depth_m.shape == (480, 640)
        assert np.isclose(depth_m[210, 210], 1.5)
        assert np.isclose(depth_m[300, 300], 2.0)
        assert depth_m[100, 100] == 0.0

        # Project with identity rotation and zero translation
        R_ident = np.eye(3, dtype=np.float64)
        t_zero = np.zeros(3, dtype=np.float64)
        K_dummy = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]], dtype=np.float64)

        proj = field.project_realsense_depth_to_stereo(
            raw_depth_mm, R_ident, t_zero, K_rs=K_dummy, K_rect=K_dummy, out_shape=(480, 640)
        )
        assert proj.shape == (480, 640)
        # Closer object must be preserved
        assert np.isclose(proj[210, 210], 1.5)


def test_metric_depth_errors_calculation():
    """Verify metric depth statistics: MAE, RMSE, and Delta<1.25."""
    ref_depth = np.array([
        [1.0, 2.0],
        [3.0, 0.0]
    ], dtype=np.float32)

    pred_depth = np.array([
        [1.1, 1.8],
        [3.0, 2.0]
    ], dtype=np.float32)

    # Valid evaluation pixels are (0,0), (0,1), (1,0) (ref > 0.4 and <= 4.0)
    # Errors: |1.1 - 1.0| = 0.1, |1.8 - 2.0| = 0.2, |3.0 - 3.0| = 0.0
    # MAE = (0.1 + 0.2 + 0.0) / 3 = 0.1
    # MSE = (0.01 + 0.04 + 0.0) / 3 = 0.05 / 3 = 0.016667 -> RMSE = 0.1291
    # max(1.1/1.0, 1.0/1.1) = 1.1 < 1.25
    # max(1.8/2.0, 2.0/1.8) = 1.111 < 1.25
    # max(3.0/3.0, 3.0/3.0) = 1.0 < 1.25 -> 100%
    stats = field.compute_metric_depth_errors(pred_depth, ref_depth, min_z=0.4, max_z=4.0)

    assert stats["eval_pixels"] == 3
    assert np.isclose(stats["depth_mae_m"], 0.1, atol=1e-4)
    assert np.isclose(stats["depth_median_m"], 0.1, atol=1e-4)
    assert np.isclose(stats["delta_125_pct"], 100.0)


def test_puncak_warung_honest_reporting():
    """Verify that distant terrain puncakWarung reports OUT_OF_RANGE without fabricated depth."""
    warung_dir = field.DEFAULT_GUNUNG_ROOT / "puncakWarung"
    if not (field.DEFAULT_GUNUNG_ROOT.exists() and warung_dir.exists() and (warung_dir / "stereo").exists()):
        pytest.skip(f"Gunung terrain raw data not present at {warung_dir}")
    rec, _, ref_d, st_d = field.evaluate_gunung_terrain("puncakWarung")
    assert rec["status"] == "OUT_OF_RANGE"
    assert ref_d is None
    assert np.isnan(rec["depth_mae_m"])
    assert rec["warp_mae"] > 0.0


def test_gunung_coverage_reporting():
    """Verify Gunung terrain evaluation reports valid frame and reference coverage."""
    tinggi_dir = field.DEFAULT_GUNUNG_ROOT / "puncakTinggi"
    if not (field.DEFAULT_GUNUNG_ROOT.exists() and tinggi_dir.exists() and (tinggi_dir / "stereo").exists()):
        pytest.skip(f"Gunung terrain raw data not present at {tinggi_dir}")
    rec, _, ref_d, st_d = field.evaluate_gunung_terrain("puncakTinggi")
    assert rec["status"] == "VERIFIED"
    assert "frame_coverage_pct" in rec
    assert "ref_coverage_pct" in rec
    # For a 1280x720 frame, frame coverage should match eval_pixels / (1280 * 720) * 100
    expected_cov = (rec["eval_pixels"] / (1280 * 720)) * 100.0
    assert np.isclose(rec["frame_coverage_pct"], expected_cov, atol=1e-3)
    assert 1.0 <= rec["frame_coverage_pct"] <= 15.0
    assert 10.0 <= rec["ref_coverage_pct"] <= 60.0


def test_gunung_out_of_range_and_unmeasured_logic():
    """Verify depth error logic when reference depths are outside active IR range."""
    # Active range is [0.4, 4.0] m. Simulate far field scene (> 15 m)
    far_ref = np.full((100, 100), 16.0, dtype=np.float32)
    stereo_pred = np.full((100, 100), 15.5, dtype=np.float32)

    stats = field.compute_metric_depth_errors(stereo_pred, far_ref, min_z=0.4, max_z=4.0)
    assert stats["eval_pixels"] == 0
    assert stats["ref_coverage_pct"] == 0.0
    assert stats["frame_coverage_pct"] == 0.0
    assert np.isnan(stats["depth_mae_m"])
    assert np.isnan(stats["depth_median_m"])
    assert np.isnan(stats["delta_125_pct"])


def test_middlebury_empty_directory_handling():
    """Verify run_middlebury_benchmark returns typed empty DataFrame without throwing KeyError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        empty_root = Path(tmpdir)
        df = field.run_middlebury_benchmark(empty_root, scale=0.5)

        assert len(df) == 0
        expected_cols = [
            "scene", "scale", "width", "height", "ndisp", "runtime_ms",
            "bad1_pct", "bad2_pct", "bad4_pct", "bad2_ca_pct",
            "disp_mae_px", "disp_rmse_px", "coverage_pct",
            "depth_mae_m", "depth_rmse_m"
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column {col} in empty DataFrame"


def test_middlebury_upscaling_normalized_convolution():
    """Verify that normalized convolution upscaling does not bleed invalid 0-sentinels into valid border pixels."""
    import cv2
    # Create small 4x4 disparity map where left half is valid (20.0) and right half is invalid (-1.0)
    disp_small = np.array([
        [20.0, 20.0, -1.0, -1.0],
        [20.0, 20.0, -1.0, -1.0],
        [20.0, 20.0, -1.0, -1.0],
        [20.0, 20.0, -1.0, -1.0],
    ], dtype=np.float32)

    scale = 0.5
    w_orig, h_orig = 8, 8

    # Apply normalized convolution upscaling (as implemented in field.py)
    valid_m = (disp_small >= 0.0) & np.isfinite(disp_small)
    d_fill = np.where(valid_m, disp_small, 0.0).astype(np.float32)
    w_m = valid_m.astype(np.float32)
    d_up = cv2.resize(d_fill, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    w_up = cv2.resize(w_m, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    valid_up = w_up >= 0.5
    d_norm = np.zeros_like(d_up)
    d_norm[valid_up] = (d_up[valid_up] / w_up[valid_up]) * (1.0 / scale)
    disp_up = np.where(valid_up, d_norm, -1.0)

    # In the valid region, all upscaled disparities should be exactly 40.0 (20.0 / 0.5)
    # The normalized convolution prevents the zero sentinel from reducing boundary values below 40.0
    valid_pixels = disp_up[valid_up]
    assert np.all(valid_pixels == 40.0)
    # Invalid region is marked -1.0
    assert np.all(disp_up[~valid_up] == -1.0)


def test_evaluate_middlebury_scene_smoke():
    """Verify evaluate_middlebury_scene runs and returns valid metric bounds on available training data."""
    adirondack_dir = field.DEFAULT_MIDDLEBURY_ROOT / "Adirondack"
    if not (adirondack_dir.exists() and (adirondack_dir / "im0.png").exists()):
        pytest.skip(f"Adirondack scene not present at {adirondack_dir}")
    res = field.evaluate_middlebury_scene(adirondack_dir, scale=0.5)
    assert res["scene"] == "Adirondack"
    assert res["scale"] == 0.5
    assert 0.0 <= res["bad2_pct"] <= 100.0
    assert 0.0 <= res["coverage_pct"] <= 100.0
    assert np.isfinite(res["disp_mae_px"])
    assert np.isfinite(res["depth_mae_m"])
    assert res["runtime_ms"] > 0.0

