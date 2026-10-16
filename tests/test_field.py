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
    if field.DEFAULT_GUNUNG_ROOT.exists() and (field.DEFAULT_GUNUNG_ROOT / "puncakWarung").exists():
        rec, _, ref_d, st_d = field.evaluate_gunung_terrain("puncakWarung")
        assert rec["status"] == "OUT_OF_RANGE"
        assert ref_d is None
        assert np.isnan(rec["depth_mae_m"])
        assert rec["warp_mae"] > 0.0


def test_gunung_coverage_reporting():
    """Verify Gunung terrain evaluation reports valid frame and reference coverage."""
    if field.DEFAULT_GUNUNG_ROOT.exists() and (field.DEFAULT_GUNUNG_ROOT / "puncakTinggi").exists():
        rec, _, ref_d, st_d = field.evaluate_gunung_terrain("puncakTinggi")
        assert rec["status"] == "VERIFIED"
        assert "frame_coverage_pct" in rec
        assert "ref_coverage_pct" in rec
        # For a 1280x720 frame, frame coverage should match eval_pixels / (1280 * 720) * 100
        expected_cov = (rec["eval_pixels"] / (1280 * 720)) * 100.0
        assert np.isclose(rec["frame_coverage_pct"], expected_cov, atol=1e-3)
        assert 1.0 <= rec["frame_coverage_pct"] <= 15.0
        assert 10.0 <= rec["ref_coverage_pct"] <= 60.0

