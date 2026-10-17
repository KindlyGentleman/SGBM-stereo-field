"""Regression tests for scientific visualization integrity and reproduction safeguards."""

import os
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

import field
import reproduce
import visualize


def test_figure_06_rejects_missing_metrics(monkeypatch, tmp_path):
    """Verify Figure 6 skips cleanly when verified metrics are missing."""
    empty_results = tmp_path / "results"
    empty_results.mkdir()
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()

    import cnn
    monkeypatch.setattr(cnn, "run_cnn_experiment", lambda save_json=True: None)
    monkeypatch.setattr(visualize, "RESULTS_DIR", empty_results)
    monkeypatch.setattr(visualize, "ASSETS_DIR", assets_dir)

    status = visualize.generate_figure_06_cnn()
    assert status == "SKIPPED_MISSING_METRICS"


def test_figure_06_rejects_missing_history(monkeypatch, tmp_path):
    """Verify Figure 6 skips cleanly with status string when training history is missing."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()

    # Provide valid verified metrics so execution reaches history check
    import json
    metrics = {
        "classical_baseline": {"accuracy": 91.06},
        "tiny_cnn": {"accuracy": 95.53},
        "multi_seed_analysis": {"cnn_accuracy_mean": 95.53, "cnn_accuracy_std": 0.70},
        "paired_comparison": {"mcnemar_exact_pvalue": 0.0074},
        "selected_seed": 202
    }
    with open(results_dir / "cnn_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f)

    monkeypatch.setattr(visualize, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(visualize, "ASSETS_DIR", assets_dir)

    status = visualize.generate_figure_06_cnn()
    assert status == "SKIPPED_MISSING_HISTORY"


def test_figure_06_rejects_missing_weights(monkeypatch, tmp_path):
    """Verify Figure 6 skips cleanly when trained weights checkpoint is missing."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()

    import json
    metrics = {
        "classical_baseline": {"accuracy": 91.06},
        "tiny_cnn": {"accuracy": 95.53},
        "multi_seed_analysis": {"cnn_accuracy_mean": 95.53, "cnn_accuracy_std": 0.70},
        "paired_comparison": {"mcnemar_exact_pvalue": 0.0074},
        "selected_seed": 202
    }
    with open(results_dir / "cnn_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f)

    # Create dummy valid history CSV but NO .pt model weights
    history_df = pd.DataFrame({
        "epoch": [1, 2, 3],
        "train_loss": [0.5, 0.4, 0.3],
        "val_loss": [0.45, 0.35, 0.25],
        "train_acc": [80.0, 85.0, 90.0],
        "val_acc": [82.0, 86.0, 89.0]
    })
    history_df.to_csv(results_dir / "cnn_history.csv", index=False)

    monkeypatch.setattr(visualize, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(visualize, "ASSETS_DIR", assets_dir)

    status = visualize.generate_figure_06_cnn()
    assert status == "SKIPPED_MISSING_TRAINED_MODEL"


def test_figure_07_preserves_assets_when_field_data_absent(monkeypatch, tmp_path):
    """Verify Figure 7 skips without overwriting existing figure when field data is absent."""
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    sentinel_file = assets_dir / "fig07_cross_dataset.png"
    sentinel_file.write_bytes(b"EXISTING_HISTORICAL_FIGURE_BYTES")

    # Point field paths to empty directories
    empty_gunung = tmp_path / "gunung"
    empty_gunung.mkdir()
    empty_calib = tmp_path / "calib.npz"

    monkeypatch.setattr(field, "DEFAULT_GUNUNG_ROOT", empty_gunung)
    monkeypatch.setattr(field, "DEFAULT_CALIB_PATH", empty_calib)
    monkeypatch.setattr(visualize, "ASSETS_DIR", assets_dir)

    status = visualize.generate_figure_07_cross_dataset()
    assert status == "SKIPPED_MISSING_FIELD_DATA"
    # Verify existing file content was preserved and not overwritten
    assert sentinel_file.read_bytes() == b"EXISTING_HISTORICAL_FIGURE_BYTES"


def test_reproduce_runner_handles_missing_middlebury(monkeypatch, tmp_path):
    """Verify reproduce.run_middlebury_benchmark handles empty scene directory without KeyError."""
    empty_mid = tmp_path / "middlebury_empty"
    empty_mid.mkdir()

    results_dir = tmp_path / "results"
    results_dir.mkdir()

    monkeypatch.setattr(field, "DEFAULT_MIDDLEBURY_ROOT", empty_mid)
    monkeypatch.setattr(reproduce, "RESULTS_DIR", results_dir)

    # Calling run_middlebury_benchmark with save_csv=False
    df = reproduce.run_middlebury_benchmark(save_csv=False)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert "bad2_pct" in df.columns
    assert "scene" in df.columns


def test_reproduce_run_full_suite_structure(monkeypatch, tmp_path):
    """Verify reproduce.run_full_suite returns structured dictionary ledger."""
    # Mock individual experiment runners to return quickly
    monkeypatch.setattr(reproduce, "run_synthetic_benchmark", lambda save_csv=True: pd.DataFrame([{"scene": "test"}]))
    monkeypatch.setattr(reproduce, "run_middlebury_benchmark", lambda save_csv=True: pd.DataFrame([{"scene": "test"}]))
    monkeypatch.setattr(reproduce, "run_gunung_benchmark", lambda save_csv=True: pd.DataFrame([{"terrain": "test"}]))
    monkeypatch.setattr(reproduce, "run_optimization_benchmark", lambda save_csv=True: pd.DataFrame([{"filter": "test"}]))
    monkeypatch.setattr(reproduce, "run_profiling_benchmark", lambda save_json=True: {"resolutions": {}})
    monkeypatch.setattr(reproduce, "run_cnn_benchmark", lambda save_json=True: {"tiny_cnn": {"accuracy": 90.0}})
    monkeypatch.setattr(visualize, "generate_all_figures", lambda: {"fig01": "GENERATED", "fig07": "SKIPPED_MISSING_FIELD_DATA"})

    mock_results = tmp_path / "results"
    mock_results.mkdir()
    monkeypatch.setattr(reproduce, "RESULTS_DIR", mock_results)

    suite_res = reproduce.run_full_suite()
    assert isinstance(suite_res, dict)
    assert "pass_count" in suite_res
    assert "records" in suite_res
    assert suite_res["has_fail"] is False
    assert suite_res["pass_count"] >= 5
