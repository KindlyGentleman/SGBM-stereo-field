"""
Unit tests for datasets and PFM I/O.
"""

from pathlib import Path
import numpy as np
import pytest

from stereo_robustness.datasets import (
    read_pfm,
    write_pfm,
    parse_middlebury_calib,
    discover_middlebury_scenes,
    discover_gunung_scenes,
)


def test_pfm_roundtrip(tmp_path: Path):
    test_array = np.random.uniform(0.0, 150.0, size=(120, 160)).astype(np.float32)
    pfm_file = tmp_path / "test.pfm"
    write_pfm(pfm_file, test_array)

    loaded, scale = read_pfm(pfm_file)
    assert loaded.shape == test_array.shape
    assert np.allclose(loaded, test_array, atol=1e-5)


def test_middlebury_loading():
    scenes = discover_middlebury_scenes("data/middlebury/MiddEval3/trainingQ")
    assert len(scenes) >= 15
    assert "Adirondack" in scenes

    scene = scenes["Adirondack"]
    left, right = scene.load_images()
    assert left.shape == right.shape
    assert left.ndim == 3

    disp_gt, mask_nocc = scene.load_ground_truth()
    assert disp_gt is not None
    assert mask_nocc is not None
    assert disp_gt.shape == left.shape[:2]
    assert mask_nocc.shape == left.shape[:2]

    # Verify quarter resolution ground truth disparity range
    valid_disps = disp_gt[mask_nocc]
    assert len(valid_disps) > 0
    assert valid_disps.min() >= 0.0
    assert valid_disps.max() < 300.0


def test_gunung_loading():
    scenes = discover_gunung_scenes(r"D:\02_Riset\Research Paper\source_code\comparison\gunung")
    assert len(scenes) == 4
    for s_name in ["puncakTinggi", "puncakWarung", "tanggaBiru", "tanggaKuning"]:
        assert s_name in scenes
        s = scenes[s_name]
        l, r = s.load_images()
        assert l.shape == (720, 1280, 3)
        assert r.shape == (720, 1280, 3)
