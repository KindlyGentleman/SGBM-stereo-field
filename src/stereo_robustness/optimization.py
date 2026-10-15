"""
Optimization framework for classical stereo matching parameters.
Implements staged search, Pareto objective ranking, trial logging,
and validation freezing.
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Callable

import numpy as np
import pandas as pd

from stereo_robustness.datasets import StereoScene
from stereo_robustness.stereo import (
    StereoBMConfig,
    StereoSGBMConfig,
    compute_disparity_bm,
    compute_disparity_sgbm,
)
from stereo_robustness.metrics import compute_stereo_metrics, StereoMetrics
from stereo_robustness.preprocessing import preprocess_pair


@dataclass
class TrialRecord:
    trial_id: int
    algorithm: str
    params: Dict[str, Any]
    preprocessing: str
    avg_dev_bad_2_0: float
    avg_dev_bad_1_0: float
    avg_dev_coverage: float
    avg_dev_mae: float
    avg_dev_rmse: float
    avg_dev_runtime_ms: float
    per_scene_bad_2_0: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_bm_config_on_scenes(
    cfg: StereoBMConfig,
    scenes: List[StereoScene],
    prep_method: str = "gray",
    **prep_kwargs: Any,
) -> Tuple[float, float, float, float, float, float, Dict[str, float]]:
    """Evaluate a StereoBM parameter configuration across a list of scenes."""
    bad2_list, bad1_list, cov_list, mae_list, rmse_list, time_list = [], [], [], [], [], []
    per_scene = {}

    for sc in scenes:
        left_bgr, right_bgr = sc.load_images()
        disp_gt, mask_nocc = sc.load_ground_truth()
        calib = sc.load_calib()

        # Update numDisparities according to geometry
        ndisp = int(np.ceil(calib.ndisp / 16.0) * 16) if calib and calib.ndisp > 0 else 80
        curr_cfg = StereoBMConfig(
            numDisparities=ndisp,
            blockSize=cfg.blockSize,
            preFilterCap=cfg.preFilterCap,
            textureThreshold=cfg.textureThreshold,
            uniquenessRatio=cfg.uniquenessRatio,
            speckleWindowSize=cfg.speckleWindowSize,
            speckleRange=cfg.speckleRange,
            disp12MaxDiff=cfg.disp12MaxDiff,
            minDisparity=cfg.minDisparity,
        )

        l_prep, r_prep = preprocess_pair(left_bgr, right_bgr, method=prep_method, **prep_kwargs)
        res = compute_disparity_bm(l_prep, r_prep, curr_cfg)
        m = compute_stereo_metrics(
            res.disparity, disp_gt, mask_eval=mask_nocc, pred_valid_mask=res.valid_mask, runtime_sec=res.runtime_sec
        )

        bad2_list.append(m.bad_2_0)
        bad1_list.append(m.bad_1_0)
        cov_list.append(m.coverage)
        if not np.isnan(m.mae):
            mae_list.append(m.mae)
            rmse_list.append(m.rmse)
        time_list.append(m.runtime_ms)
        per_scene[sc.scene_id] = m.bad_2_0

    return (
        float(np.mean(bad2_list)),
        float(np.mean(bad1_list)),
        float(np.mean(cov_list)),
        float(np.mean(mae_list)) if mae_list else float("nan"),
        float(np.mean(rmse_list)) if rmse_list else float("nan"),
        float(np.mean(time_list)),
        per_scene,
    )


def evaluate_sgbm_config_on_scenes(
    cfg: StereoSGBMConfig,
    scenes: List[StereoScene],
    prep_method: str = "gray",
    **prep_kwargs: Any,
) -> Tuple[float, float, float, float, float, float, Dict[str, float]]:
    """Evaluate a StereoSGBM parameter configuration across a list of scenes."""
    bad2_list, bad1_list, cov_list, mae_list, rmse_list, time_list = [], [], [], [], [], []
    per_scene = {}

    for sc in scenes:
        left_bgr, right_bgr = sc.load_images()
        disp_gt, mask_nocc = sc.load_ground_truth()
        calib = sc.load_calib()

        ndisp = int(np.ceil(calib.ndisp / 16.0) * 16) if calib and calib.ndisp > 0 else 80
        curr_cfg = StereoSGBMConfig(
            minDisparity=cfg.minDisparity,
            numDisparities=ndisp,
            blockSize=cfg.blockSize,
            P1=cfg.P1,
            P2=max(cfg.P2, cfg.P1 + 1),
            disp12MaxDiff=cfg.disp12MaxDiff,
            preFilterCap=cfg.preFilterCap,
            uniquenessRatio=cfg.uniquenessRatio,
            speckleWindowSize=cfg.speckleWindowSize,
            speckleRange=cfg.speckleRange,
            mode=cfg.mode,
        )

        l_prep, r_prep = preprocess_pair(left_bgr, right_bgr, method=prep_method, **prep_kwargs)
        res = compute_disparity_sgbm(l_prep, r_prep, curr_cfg)
        m = compute_stereo_metrics(
            res.disparity, disp_gt, mask_eval=mask_nocc, pred_valid_mask=res.valid_mask, runtime_sec=res.runtime_sec
        )

        bad2_list.append(m.bad_2_0)
        bad1_list.append(m.bad_1_0)
        cov_list.append(m.coverage)
        if not np.isnan(m.mae):
            mae_list.append(m.mae)
            rmse_list.append(m.rmse)
        time_list.append(m.runtime_ms)
        per_scene[sc.scene_id] = m.bad_2_0

    return (
        float(np.mean(bad2_list)),
        float(np.mean(bad1_list)),
        float(np.mean(cov_list)),
        float(np.mean(mae_list)) if mae_list else float("nan"),
        float(np.mean(rmse_list)) if rmse_list else float("nan"),
        float(np.mean(time_list)),
        per_scene,
    )
