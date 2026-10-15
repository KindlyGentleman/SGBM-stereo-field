"""
Stereo evaluation metrics with completeness awareness.
Evaluates disparity predictions against ground truth masks and measures computational performance.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

import numpy as np


@dataclass
class StereoMetrics:
    total_eval_pixels: int
    valid_pred_pixels: int
    coverage: float  # valid_pred / total_eval
    bad_1_0: float  # error > 1.0 px or invalid
    bad_2_0: float  # error > 2.0 px or invalid
    bad_3_0: float  # error > 3.0 px or invalid
    conditional_bad_1_0: float  # bad rate only on valid predictions
    conditional_bad_2_0: float
    conditional_bad_3_0: float
    mae: float  # EPE on mutually valid pixels
    rmse: float  # RMSE on mutually valid pixels
    invalid_fraction: float
    runtime_ms: float
    runtime_sec_per_mp: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_stereo_metrics(
    disp_pred: np.ndarray,
    disp_gt: np.ndarray,
    mask_eval: Optional[np.ndarray] = None,
    pred_valid_mask: Optional[np.ndarray] = None,
    runtime_sec: float = 0.0,
    bad_thresholds: tuple = (1.0, 2.0, 3.0),
) -> StereoMetrics:
    """
    Calculate completeness-aware stereo matching metrics.
    
    If pred_valid_mask is None, non-zero positive finite values are considered valid.
    In completeness-aware bad-pixel calculations, pixels that are valid in GT but
    missing in prediction are penalized as bad pixels.
    """
    if disp_pred.shape != disp_gt.shape:
        raise ValueError(f"Shape mismatch: pred {disp_pred.shape} vs gt {disp_gt.shape}")

    # Ground truth evaluation mask
    if mask_eval is None:
        mask_gt_valid = np.isfinite(disp_gt) & (disp_gt > 0)
    else:
        mask_gt_valid = mask_eval & np.isfinite(disp_gt) & (disp_gt > 0)

    total_eval = int(np.count_nonzero(mask_gt_valid))
    if total_eval == 0:
        return StereoMetrics(
            total_eval_pixels=0,
            valid_pred_pixels=0,
            coverage=0.0,
            bad_1_0=0.0,
            bad_2_0=0.0,
            bad_3_0=0.0,
            conditional_bad_1_0=0.0,
            conditional_bad_2_0=0.0,
            conditional_bad_3_0=0.0,
            mae=0.0,
            rmse=0.0,
            invalid_fraction=1.0,
            runtime_ms=runtime_sec * 1000.0,
            runtime_sec_per_mp=0.0,
        )

    # Prediction validity mask
    if pred_valid_mask is None:
        pred_valid = np.isfinite(disp_pred) & (disp_pred > 0)
    else:
        pred_valid = pred_valid_mask & np.isfinite(disp_pred) & (disp_pred > 0)

    # Mutually valid pixels
    mutually_valid = mask_gt_valid & pred_valid
    n_mutually_valid = int(np.count_nonzero(mutually_valid))
    coverage = float(n_mutually_valid / total_eval)
    invalid_fraction = 1.0 - coverage

    # Absolute errors on mutually valid pixels
    abs_diff = np.abs(disp_pred[mutually_valid] - disp_gt[mutually_valid])

    if n_mutually_valid > 0:
        mae = float(np.mean(abs_diff))
        rmse = float(np.sqrt(np.mean(abs_diff ** 2)))
        cond_bad_1 = float(np.count_nonzero(abs_diff > 1.0) / n_mutually_valid * 100.0)
        cond_bad_2 = float(np.count_nonzero(abs_diff > 2.0) / n_mutually_valid * 100.0)
        cond_bad_3 = float(np.count_nonzero(abs_diff > 3.0) / n_mutually_valid * 100.0)
    else:
        mae = float("nan")
        rmse = float("nan")
        cond_bad_1 = 100.0
        cond_bad_2 = 100.0
        cond_bad_3 = 100.0

    # Completeness-aware bad pixels: penalize unpredicted eval pixels as errors
    n_missing = total_eval - n_mutually_valid
    bad_1_count = int(np.count_nonzero(abs_diff > 1.0)) + n_missing
    bad_2_count = int(np.count_nonzero(abs_diff > 2.0)) + n_missing
    bad_3_count = int(np.count_nonzero(abs_diff > 3.0)) + n_missing

    bad_1 = float(bad_1_count / total_eval * 100.0)
    bad_2 = float(bad_2_count / total_eval * 100.0)
    bad_3 = float(bad_3_count / total_eval * 100.0)

    # Runtime per megapixel
    total_pixels = disp_pred.size
    mp = total_pixels / 1e6
    runtime_per_mp = runtime_sec / mp if mp > 0 else 0.0

    return StereoMetrics(
        total_eval_pixels=total_eval,
        valid_pred_pixels=n_mutually_valid,
        coverage=coverage,
        bad_1_0=bad_1,
        bad_2_0=bad_2,
        bad_3_0=bad_3,
        conditional_bad_1_0=cond_bad_1,
        conditional_bad_2_0=cond_bad_2,
        conditional_bad_3_0=cond_bad_3,
        mae=mae,
        rmse=rmse,
        invalid_fraction=invalid_fraction,
        runtime_ms=runtime_sec * 1000.0,
        runtime_sec_per_mp=runtime_per_mp,
    )
