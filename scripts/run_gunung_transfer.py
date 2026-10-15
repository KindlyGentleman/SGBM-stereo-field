"""
Milestone 5 (Phase E & F): Transfer of frozen Middlebury configurations to Gunung outdoor stereo imagery.
Evaluates G0 (Default), G1 (Optimized), and G2 (Optimized + Preprocessing) across multiple resolutions
(1280x720, 640x360, 320x180) using ground-truth-free diagnostics:
1. Left-Right round-trip consistency
2. Valid prediction coverage
3. Photometric warp residual
4. Disparity spatial gradient and edge alignment
5. Specific ROI inspection (sky, grass, gravel, building).
"""

from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import yaml

from stereo_robustness.calibration import StereoCalibrationParams, compute_rectification_models
from stereo_robustness.stereo import (
    StereoBMConfig,
    StereoSGBMConfig,
    compute_disparity_bm,
    compute_disparity_sgbm,
    compute_right_disparity,
)
from stereo_robustness.preprocessing import preprocess_pair
from stereo_robustness.consistency import compute_unsupervised_diagnostics, warp_right_to_left
from stereo_robustness.visualization import colorize_disparity, save_image_rgb

OUT_DIR = Path("results/gunung/transfer")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DISP_DIR = OUT_DIR / "disparities"
DISP_DIR.mkdir(parents=True, exist_ok=True)

with open("configs/gunung_dataset.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load frozen Middlebury configs
with open("results/middlebury/best_configs/bm_frozen_config.json", "r") as f:
    bm_frozen = json.load(f)
with open("results/middlebury/best_configs/sgbm_frozen_config.json", "r") as f:
    sgbm_frozen = json.load(f)

calib_path = Path(config["calibration"]["params_file"])
calib_params = StereoCalibrationParams.from_npz(calib_path)

scenes = config["dataset"]["scenes"]
base_dir = Path(config["dataset"]["root_dir"])

resolutions = [
    (1280, 720, 128),
    (640, 360, 64),
    (320, 180, 32),
]

def clean_dict(d: dict, cls) -> dict:
    return {k: v for k, v in d.items() if k in cls.__annotations__}

records = []

print(f"Running Cross-Domain Transfer on {len(scenes)} Gunung scenes across 3 resolutions...")

for scene_name in scenes:
    s_dir = base_dir / scene_name / "stereo"
    l_raw = cv2.imread(str(s_dir / "left_0000.png"))
    r_raw = cv2.imread(str(s_dir / "right_0000.png"))

    for width, height, num_disp in resolutions:
        res_tag = f"{width}x{height}"
        # Resize raw images if scaling down
        if (width, height) != (1280, 720):
            l_curr = cv2.resize(l_raw, (width, height), interpolation=cv2.INTER_AREA)
            r_curr = cv2.resize(r_raw, (width, height), interpolation=cv2.INTER_AREA)
            # Scale camera calibration parameters to match resolution
            scale_x = width / 1280.0
            scale_y = height / 720.0
            mtx_l_scaled = calib_params.mtx_left.copy()
            mtx_l_scaled[0, :] *= scale_x
            mtx_l_scaled[1, :] *= scale_y
            mtx_r_scaled = calib_params.mtx_right.copy()
            mtx_r_scaled[0, :] *= scale_x
            mtx_r_scaled[1, :] *= scale_y
            scaled_calib = StereoCalibrationParams(
                mtx_left=mtx_l_scaled,
                dist_left=calib_params.dist_left,
                mtx_right=mtx_r_scaled,
                dist_right=calib_params.dist_right,
                R=calib_params.R,
                T=calib_params.T,
                image_size=(width, height),
                baseline_mm=calib_params.baseline_mm,
            )
            rect_models = compute_rectification_models(scaled_calib, alpha=0.0, new_image_size=(width, height))
        else:
            l_curr = l_raw
            r_curr = r_raw
            rect_models = compute_rectification_models(calib_params, alpha=0.0, new_image_size=(1280, 720))

        # Apply calibrated rectification
        rect_l, rect_r = rect_models.rectify(l_curr, r_curr)

        # Build model suites
        models = [
            ("G0_StereoBM_Default", "StereoBM", clean_dict(bm_frozen["G0_default"], StereoBMConfig), "gray"),
            ("G1_StereoBM_Optimized", "StereoBM", clean_dict(bm_frozen["G1_optimized"], StereoBMConfig), "gray"),
            (f"G2_StereoBM_Opt_{bm_frozen['G2_preprocessing']}", "StereoBM", clean_dict(bm_frozen["G1_optimized"], StereoBMConfig), bm_frozen["G2_preprocessing"]),
            ("G0_StereoSGBM_Default", "StereoSGBM", clean_dict(sgbm_frozen["G0_default"], StereoSGBMConfig), "gray"),
            ("G1_StereoSGBM_Optimized", "StereoSGBM", clean_dict(sgbm_frozen["G1_optimized"], StereoSGBMConfig), "gray"),
            (f"G2_StereoSGBM_Opt_{sgbm_frozen['G2_preprocessing']}", "StereoSGBM", clean_dict(sgbm_frozen["G1_optimized"], StereoSGBMConfig), sgbm_frozen["G2_preprocessing"]),
        ]

        for model_name, alg, cfg_dict, prep in models:
            cfg_dict["numDisparities"] = num_disp
            l_p, r_p = preprocess_pair(rect_l, rect_r, method=prep)

            if alg == "StereoBM":
                bm_cfg = StereoBMConfig(**cfg_dict)
                res_l = compute_disparity_bm(l_p, r_p, bm_cfg)
                # Compute right disparity for consistency
                # Flip for right view
                l_flip = cv2.flip(r_p, 1)
                r_flip = cv2.flip(l_p, 1)
                res_r_flip = compute_disparity_bm(l_flip, r_flip, bm_cfg)
                disp_r = cv2.flip(res_r_flip.disparity, 1)
                valid_r = cv2.flip(res_r_flip.valid_mask.astype(np.uint8), 1) > 0
            else:
                sgbm_cfg = StereoSGBMConfig(**cfg_dict)
                res_l = compute_disparity_sgbm(l_p, r_p, sgbm_cfg)
                res_r = compute_right_disparity(l_p, r_p, sgbm_cfg)
                disp_r = res_r.disparity
                valid_r = res_r.valid_mask

            # Evaluate unsupervised proxy diagnostics
            diag = compute_unsupervised_diagnostics(
                rect_l, rect_r, res_l.disparity, disp_r, res_l.valid_mask, valid_r, lr_threshold=1.5
            )

            rec = {
                "scene": scene_name,
                "resolution": res_tag,
                "model": model_name,
                "algorithm": alg,
                "preprocessing": prep,
                "runtime_ms": res_l.runtime_sec * 1000.0,
                "valid_coverage_pct": diag.valid_disparity_pct,
                "lr_consistent_pct_valid": diag.lr_consistent_pct_of_valid,
                "lr_consistent_pct_total": diag.lr_consistent_pct_of_total,
                "photometric_mae": diag.photometric_mae,
                "photometric_rmse": diag.photometric_rmse,
                "disparity_gradient_mean": diag.disparity_gradient_mean,
                "edge_alignment_score": diag.edge_alignment_score,
            }
            records.append(rec)

            # Save full native resolution disparity visualization for primary scene
            if scene_name == "puncakTinggi" and res_tag == "1280x720":
                disp_vis = colorize_disparity(res_l.disparity, res_l.valid_mask, vmin=0, vmax=float(num_disp))
                save_image_rgb(DISP_DIR / f"{scene_name}_{model_name}_{res_tag}_disp.png", disp_vis)
                # Save consistency map: 1 for consistent, 0 for inconsistent
                cons_vis = np.zeros((*res_l.disparity.shape, 3), dtype=np.uint8)
                cons_mask = diag.lr_consistent_pixels
                # Draw consistent pixels in green, valid inconsistent in red, invalid in black
                # Re-compute consistency mask for visualization
                from stereo_robustness.consistency import check_left_right_consistency
                c_mask, _, _ = check_left_right_consistency(res_l.disparity, disp_r, res_l.valid_mask, valid_r, 1.5)
                cons_vis[res_l.valid_mask & (~c_mask)] = [220, 20, 60]  # Red for inconsistent
                cons_vis[c_mask] = [34, 139, 34]  # Forest green for consistent
                save_image_rgb(DISP_DIR / f"{scene_name}_{model_name}_{res_tag}_consistency.png", cons_vis)

        print(f"Processed {scene_name:12s} | Resolution: {res_tag}")

df_transfer = pd.DataFrame(records)
df_transfer.to_csv(OUT_DIR / "gunung_transfer_metrics.csv", index=False)
with open(OUT_DIR / "gunung_transfer_metrics.json", "w") as f:
    json.dump(records, f, indent=2)

print("\n--- GUNUNG TRANSFER SUMMARY (Native 1280x720, Averaged across All 4 Scenes) ---")
summary_native = df_transfer[df_transfer["resolution"] == "1280x720"].groupby("model")[
    ["valid_coverage_pct", "lr_consistent_pct_valid", "lr_consistent_pct_total", "photometric_mae", "runtime_ms"]
].mean()
print(summary_native.to_string())
summary_native.to_csv(OUT_DIR / "gunung_transfer_summary_native.csv")

print("\n--- MULTI-RESOLUTION SCALING SUMMARY (puncakTinggi, G1_StereoSGBM_Optimized) ---")
scaling_summary = df_transfer[(df_transfer["scene"] == "puncakTinggi") & (df_transfer["model"] == "G1_StereoSGBM_Optimized")][
    ["resolution", "runtime_ms", "valid_coverage_pct", "lr_consistent_pct_valid", "photometric_mae"]
]
print(scaling_summary.to_string(index=False))
