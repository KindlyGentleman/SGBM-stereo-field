"""
Milestone 1: Middlebury baseline evaluation script.
Runs uncalibrated and calibrated baseline StereoBM and StereoSGBM on clean dev and val scenes.
Computes completeness-aware bad-pixel rates, valid coverage, MAE, RMSE, and runtime.
"""

from __future__ import annotations
import json
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import yaml

from stereo_robustness.datasets import discover_middlebury_scenes, write_pfm
from stereo_robustness.stereo import (
    StereoBMConfig,
    StereoSGBMConfig,
    compute_disparity_bm,
    compute_disparity_sgbm,
)
from stereo_robustness.metrics import compute_stereo_metrics
from stereo_robustness.preprocessing import to_gray
from stereo_robustness.visualization import colorize_disparity, colorize_error_map, save_image_rgb

OUT_DIR = Path("results/middlebury/baseline")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DISP_DIR = OUT_DIR / "disparities"
DISP_DIR.mkdir(parents=True, exist_ok=True)

with open("configs/middlebury_baseline.yaml", "r") as f:
    config = yaml.safe_load(f)

scenes_dict = discover_middlebury_scenes("data/middlebury/MiddEval3/trainingQ")

eval_scenes = config["splits"]["dev_scenes"] + config["splits"]["val_scenes"]

records = []

print(f"Running Middlebury Baseline on {len(eval_scenes)} scenes (Dev + Val)...")

for scene_name in eval_scenes:
    if scene_name not in scenes_dict:
        print(f"Warning: Scene {scene_name} not found in dataset!")
        continue

    scene = scenes_dict[scene_name]
    left_bgr, right_bgr = scene.load_images()
    left_gray = to_gray(left_bgr)
    right_gray = to_gray(right_bgr)
    disp_gt, mask_nocc = scene.load_ground_truth()
    calib = scene.load_calib()

    # Geometry-derived disparity search range
    if calib and calib.ndisp > 0:
        num_disp = int(np.ceil(calib.ndisp / 16.0) * 16)
    else:
        num_disp = 80

    # 1. Baseline StereoBM
    bm_cfg = StereoBMConfig(
        numDisparities=num_disp,
        blockSize=int(config["stereo_bm"]["blockSize"]),
        preFilterCap=int(config["stereo_bm"]["preFilterCap"]),
        textureThreshold=int(config["stereo_bm"]["textureThreshold"]),
        uniquenessRatio=int(config["stereo_bm"]["uniquenessRatio"]),
        speckleWindowSize=int(config["stereo_bm"]["speckleWindowSize"]),
        speckleRange=int(config["stereo_bm"]["speckleRange"]),
        disp12MaxDiff=int(config["stereo_bm"]["disp12MaxDiff"]),
    )
    bm_res = compute_disparity_bm(left_gray, right_gray, bm_cfg)

    # 2. Baseline StereoSGBM
    sgbm_cfg = StereoSGBMConfig(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=int(config["stereo_sgbm"]["blockSize"]),
        P1=int(config["stereo_sgbm"]["P1"]),
        P2=int(config["stereo_sgbm"]["P2"]),
        disp12MaxDiff=int(config["stereo_sgbm"]["disp12MaxDiff"]),
        preFilterCap=int(config["stereo_sgbm"]["preFilterCap"]),
        uniquenessRatio=int(config["stereo_sgbm"]["uniquenessRatio"]),
        speckleWindowSize=int(config["stereo_sgbm"]["speckleWindowSize"]),
        speckleRange=int(config["stereo_sgbm"]["speckleRange"]),
        mode=int(config["stereo_sgbm"]["mode"]),
    )
    sgbm_res = compute_disparity_sgbm(left_bgr, right_bgr, sgbm_cfg)

    # Evaluate on non-occluded mask and all-valid GT mask
    mask_all_gt = np.isfinite(disp_gt) & (disp_gt > 0)

    for alg_name, res in [("StereoBM", bm_res), ("StereoSGBM", sgbm_res)]:
        # Non-occluded metrics
        m_nocc = compute_stereo_metrics(
            res.disparity, disp_gt, mask_eval=mask_nocc, pred_valid_mask=res.valid_mask, runtime_sec=res.runtime_sec
        )
        # All-valid GT metrics
        m_all = compute_stereo_metrics(
            res.disparity, disp_gt, mask_eval=mask_all_gt, pred_valid_mask=res.valid_mask, runtime_sec=res.runtime_sec
        )

        rec = {
            "scene": scene_name,
            "algorithm": alg_name,
            "split": "dev" if scene_name in config["splits"]["dev_scenes"] else "val",
            "num_disparities": num_disp,
            "runtime_ms": m_nocc.runtime_ms,
            "runtime_sec_per_mp": m_nocc.runtime_sec_per_mp,
            # Non-occluded primary
            "nocc_coverage": m_nocc.coverage,
            "nocc_bad_1_0": m_nocc.bad_1_0,
            "nocc_bad_2_0": m_nocc.bad_2_0,
            "nocc_bad_3_0": m_nocc.bad_3_0,
            "nocc_cond_bad_2_0": m_nocc.conditional_bad_2_0,
            "nocc_mae": m_nocc.mae,
            "nocc_rmse": m_nocc.rmse,
            # All valid primary
            "all_coverage": m_all.coverage,
            "all_bad_1_0": m_all.bad_1_0,
            "all_bad_2_0": m_all.bad_2_0,
            "all_bad_3_0": m_all.bad_3_0,
            "all_cond_bad_2_0": m_all.conditional_bad_2_0,
            "all_mae": m_all.mae,
            "all_rmse": m_all.rmse,
        }
        records.append(rec)

        # Save disparity visualizations
        disp_vis = colorize_disparity(res.disparity, res.valid_mask, vmin=0, vmax=float(num_disp))
        save_image_rgb(DISP_DIR / f"{scene_name}_{alg_name}_disp.png", disp_vis)

        # Save error map visualization on non-occluded mask
        err_vis = colorize_error_map(res.disparity, disp_gt, mask_nocc, res.valid_mask, max_error=5.0)
        save_image_rgb(DISP_DIR / f"{scene_name}_{alg_name}_err.png", err_vis)

        # Save raw PFM prediction
        write_pfm(DISP_DIR / f"{scene_name}_{alg_name}_disp.pfm", res.disparity)

    print(f"Processed {scene_name:12s} | StereoBM bad2: {records[-2]['nocc_bad_2_0']:.1f}% cov: {records[-2]['nocc_coverage']*100:.1f}% | SGBM bad2: {records[-1]['nocc_bad_2_0']:.1f}% cov: {records[-1]['nocc_coverage']*100:.1f}%")

# Save summary tables
df = pd.DataFrame(records)
df.to_csv(OUT_DIR / "baseline_metrics.csv", index=False)
with open(OUT_DIR / "baseline_metrics.json", "w") as f:
    json.dump(records, f, indent=2)

print("\n--- BASELINE SUMMARY (Average across Dev + Val Scenes) ---")
summary = df.groupby("algorithm")[["nocc_bad_1_0", "nocc_bad_2_0", "nocc_bad_3_0", "nocc_coverage", "nocc_mae", "nocc_rmse", "runtime_ms"]].mean()
print(summary.to_string())
summary.to_csv(OUT_DIR / "baseline_summary.csv")
print("\nBaseline run completed successfully!")
