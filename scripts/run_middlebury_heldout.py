"""
Milestone 3: Held-out benchmark evaluation script.
Evaluates frozen configurations (G0 default, G1 optimized, G2 +preprocessing)
ONCE on held-out benchmark scenes (Jadeplant, Shelves, ArtL).
Strictly zero post-hoc tuning.
"""

from __future__ import annotations
import json
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
from stereo_robustness.preprocessing import preprocess_pair, to_gray
from stereo_robustness.visualization import colorize_disparity, colorize_error_map, save_image_rgb

OUT_DIR = Path("results/middlebury/heldout")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DISP_DIR = OUT_DIR / "disparities"
DISP_DIR.mkdir(parents=True, exist_ok=True)

with open("configs/middlebury_baseline.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load frozen configurations
BEST_CFG_DIR = Path("results/middlebury/best_configs")
with open(BEST_CFG_DIR / "bm_frozen_config.json", "r") as f:
    bm_frozen = json.load(f)
with open(BEST_CFG_DIR / "sgbm_frozen_config.json", "r") as f:
    sgbm_frozen = json.load(f)

scenes_dict = discover_middlebury_scenes("data/middlebury/MiddEval3/trainingQ")
heldout_scenes = config["splits"]["heldout_scenes"]

print(f"Running Held-out Benchmark Verification on {len(heldout_scenes)} scenes: {heldout_scenes}...")

records = []

# Define configurations to evaluate
# Clean dictionaries to match dataclass fields
def clean_sgbm_dict(d: dict) -> dict:
    return {k: v for k, v in d.items() if k in StereoSGBMConfig.__annotations__}

def clean_bm_dict(d: dict) -> dict:
    return {k: v for k, v in d.items() if k in StereoBMConfig.__annotations__}

models = [
    ("G0_StereoBM_Default", "StereoBM", StereoBMConfig(**clean_bm_dict(bm_frozen["G0_default"])), "gray"),
    ("G1_StereoBM_Optimized", "StereoBM", StereoBMConfig(**clean_bm_dict(bm_frozen["G1_optimized"])), "gray"),
    (f"G2_StereoBM_Opt_{bm_frozen['G2_preprocessing']}", "StereoBM", StereoBMConfig(**clean_bm_dict(bm_frozen["G1_optimized"])), bm_frozen["G2_preprocessing"]),
    ("G0_StereoSGBM_Default", "StereoSGBM", StereoSGBMConfig(**clean_sgbm_dict(sgbm_frozen["G0_default"])), "gray"),
    ("G1_StereoSGBM_Optimized", "StereoSGBM", StereoSGBMConfig(**clean_sgbm_dict(sgbm_frozen["G1_optimized"])), "gray"),
    (f"G2_StereoSGBM_Opt_{sgbm_frozen['G2_preprocessing']}", "StereoSGBM", StereoSGBMConfig(**clean_sgbm_dict(sgbm_frozen["G1_optimized"])), sgbm_frozen["G2_preprocessing"]),
]


for scene_name in heldout_scenes:
    scene = scenes_dict[scene_name]
    left_bgr, right_bgr = scene.load_images()
    disp_gt, mask_nocc = scene.load_ground_truth()
    calib = scene.load_calib()
    num_disp = int(np.ceil(calib.ndisp / 16.0) * 16) if calib and calib.ndisp > 0 else 80

    for model_name, alg, cfg, prep in models:
        # Clone cfg with scene-specific numDisparities
        if alg == "StereoBM":
            cur_cfg = StereoBMConfig(
                numDisparities=num_disp,
                blockSize=cfg.blockSize,
                preFilterCap=cfg.preFilterCap,
                textureThreshold=cfg.textureThreshold,
                uniquenessRatio=cfg.uniquenessRatio,
                speckleWindowSize=cfg.speckleWindowSize,
                speckleRange=cfg.speckleRange,
                disp12MaxDiff=cfg.disp12MaxDiff,
            )
            l_p, r_p = preprocess_pair(left_bgr, right_bgr, method=prep)
            res = compute_disparity_bm(l_p, r_p, cur_cfg)
        else:
            cur_cfg = StereoSGBMConfig(
                minDisparity=0,
                numDisparities=num_disp,
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
            l_p, r_p = preprocess_pair(left_bgr, right_bgr, method=prep)
            res = compute_disparity_sgbm(l_p, r_p, cur_cfg)

        m = compute_stereo_metrics(
            res.disparity, disp_gt, mask_eval=mask_nocc, pred_valid_mask=res.valid_mask, runtime_sec=res.runtime_sec
        )

        records.append({
            "scene": scene_name,
            "model": model_name,
            "algorithm": alg,
            "preprocessing": prep,
            "num_disparities": num_disp,
            "bad_1_0": m.bad_1_0,
            "bad_2_0": m.bad_2_0,
            "bad_3_0": m.bad_3_0,
            "cond_bad_2_0": m.conditional_bad_2_0,
            "coverage": m.coverage,
            "mae": m.mae,
            "rmse": m.rmse,
            "runtime_ms": m.runtime_ms,
        })

        # Save visualizations
        disp_vis = colorize_disparity(res.disparity, res.valid_mask, vmin=0, vmax=float(num_disp))
        save_image_rgb(DISP_DIR / f"{scene_name}_{model_name}_disp.png", disp_vis)
        err_vis = colorize_error_map(res.disparity, disp_gt, mask_nocc, res.valid_mask, max_error=5.0)
        save_image_rgb(DISP_DIR / f"{scene_name}_{model_name}_err.png", err_vis)
        write_pfm(DISP_DIR / f"{scene_name}_{model_name}_disp.pfm", res.disparity)

        print(f"Heldout {scene_name:10s} | {model_name:25s} | Bad2: {m.bad_2_0:.2f}% | Cov: {m.coverage*100:.1f}% | MAE: {m.mae:.2f}px | Time: {m.runtime_ms:.1f}ms")

df_heldout = pd.DataFrame(records)
df_heldout.to_csv(OUT_DIR / "heldout_metrics.csv", index=False)
with open(OUT_DIR / "heldout_metrics.json", "w") as f:
    json.dump(records, f, indent=2)

print("\n--- HELDOUT SUMMARY ---")
summary = df_heldout.groupby("model")[["bad_1_0", "bad_2_0", "bad_3_0", "coverage", "mae", "rmse", "runtime_ms"]].mean()
print(summary.to_string())
summary.to_csv(OUT_DIR / "heldout_summary.csv")
