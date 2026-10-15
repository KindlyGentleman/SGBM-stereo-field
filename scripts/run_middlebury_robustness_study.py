"""
Phase B: Middlebury photometric robustness evaluation under controlled domain shifts.
Evaluates baseline StereoBM and StereoSGBM under asymmetric brightness scaling,
gamma adjustments, and Gaussian noise.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from stereo_robustness.datasets import discover_middlebury_scenes
from stereo_robustness.stereo import (
    StereoBMConfig,
    StereoSGBMConfig,
    compute_disparity_bm,
    compute_disparity_sgbm,
)
from stereo_robustness.metrics import compute_stereo_metrics
from stereo_robustness.perturbations import (
    apply_brightness_scaling,
    apply_gamma_correction,
    apply_gaussian_noise,
    STANDARD_BRIGHTNESS_FACTORS,
    STANDARD_GAMMA_VALUES,
    STANDARD_NOISE_SIGMAS,
    NOISE_REPETITION_SEEDS,
)
from stereo_robustness.preprocessing import to_gray

OUT_DIR = Path("results/middlebury/robustness")
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open("configs/middlebury_baseline.yaml", "r") as f:
    config = yaml.safe_load(f)

scenes_dict = discover_middlebury_scenes("data/middlebury/MiddEval3/trainingQ")

# Select representative development and validation scenes for the robustness grid
test_scenes = ["Adirondack", "Motorcycle", "MotorcycleE", "Pipes", "Piano", "PianoL"]

records = []

print(f"Running Phase B Photometric Robustness on {len(test_scenes)} scenes...")

for scene_name in test_scenes:
    scene = scenes_dict[scene_name]
    left_bgr, right_bgr = scene.load_images()
    disp_gt, mask_nocc = scene.load_ground_truth()
    calib = scene.load_calib()
    num_disp = int(np.ceil(calib.ndisp / 16.0) * 16) if calib and calib.ndisp > 0 else 80

    bm_cfg = StereoBMConfig(numDisparities=num_disp, blockSize=15)
    sgbm_cfg = StereoSGBMConfig(numDisparities=num_disp, blockSize=5, P1=600, P2=2400)

    # 1. Brightness scaling
    for alpha in STANDARD_BRIGHTNESS_FACTORS:
        r_pert, meta = apply_brightness_scaling(right_bgr, alpha)

        # StereoBM
        bm_res = compute_disparity_bm(to_gray(left_bgr), to_gray(r_pert), bm_cfg)
        m_bm = compute_stereo_metrics(bm_res.disparity, disp_gt, mask_eval=mask_nocc, pred_valid_mask=bm_res.valid_mask, runtime_sec=bm_res.runtime_sec)
        records.append({
            "scene": scene_name,
            "perturbation_type": "brightness",
            "param_value": alpha,
            "seed": 0,
            "algorithm": "StereoBM",
            "bad_1_0": m_bm.bad_1_0,
            "bad_2_0": m_bm.bad_2_0,
            "bad_3_0": m_bm.bad_3_0,
            "coverage": m_bm.coverage,
            "mae": m_bm.mae,
            "rmse": m_bm.rmse,
            "runtime_ms": m_bm.runtime_ms,
            "sat_low_pct": meta.saturation_low_pct,
            "sat_high_pct": meta.saturation_high_pct,
        })

        # StereoSGBM
        sgbm_res = compute_disparity_sgbm(left_bgr, r_pert, sgbm_cfg)
        m_sgbm = compute_stereo_metrics(sgbm_res.disparity, disp_gt, mask_eval=mask_nocc, pred_valid_mask=sgbm_res.valid_mask, runtime_sec=sgbm_res.runtime_sec)
        records.append({
            "scene": scene_name,
            "perturbation_type": "brightness",
            "param_value": alpha,
            "seed": 0,
            "algorithm": "StereoSGBM",
            "bad_1_0": m_sgbm.bad_1_0,
            "bad_2_0": m_sgbm.bad_2_0,
            "bad_3_0": m_sgbm.bad_3_0,
            "coverage": m_sgbm.coverage,
            "mae": m_sgbm.mae,
            "rmse": m_sgbm.rmse,
            "runtime_ms": m_sgbm.runtime_ms,
            "sat_low_pct": meta.saturation_low_pct,
            "sat_high_pct": meta.saturation_high_pct,
        })

    # 2. Gamma correction
    for gamma in STANDARD_GAMMA_VALUES:
        r_pert, meta = apply_gamma_correction(right_bgr, gamma)

        bm_res = compute_disparity_bm(to_gray(left_bgr), to_gray(r_pert), bm_cfg)
        m_bm = compute_stereo_metrics(bm_res.disparity, disp_gt, mask_eval=mask_nocc, pred_valid_mask=bm_res.valid_mask, runtime_sec=bm_res.runtime_sec)
        records.append({
            "scene": scene_name,
            "perturbation_type": "gamma",
            "param_value": gamma,
            "seed": 0,
            "algorithm": "StereoBM",
            "bad_1_0": m_bm.bad_1_0,
            "bad_2_0": m_bm.bad_2_0,
            "bad_3_0": m_bm.bad_3_0,
            "coverage": m_bm.coverage,
            "mae": m_bm.mae,
            "rmse": m_bm.rmse,
            "runtime_ms": m_bm.runtime_ms,
            "sat_low_pct": meta.saturation_low_pct,
            "sat_high_pct": meta.saturation_high_pct,
        })

        sgbm_res = compute_disparity_sgbm(left_bgr, r_pert, sgbm_cfg)
        m_sgbm = compute_stereo_metrics(sgbm_res.disparity, disp_gt, mask_eval=mask_nocc, pred_valid_mask=sgbm_res.valid_mask, runtime_sec=sgbm_res.runtime_sec)
        records.append({
            "scene": scene_name,
            "perturbation_type": "gamma",
            "param_value": gamma,
            "seed": 0,
            "algorithm": "StereoSGBM",
            "bad_1_0": m_sgbm.bad_1_0,
            "bad_2_0": m_sgbm.bad_2_0,
            "bad_3_0": m_sgbm.bad_3_0,
            "coverage": m_sgbm.coverage,
            "mae": m_sgbm.mae,
            "rmse": m_sgbm.rmse,
            "runtime_ms": m_sgbm.runtime_ms,
            "sat_low_pct": meta.saturation_low_pct,
            "sat_high_pct": meta.saturation_high_pct,
        })

    # 3. Gaussian noise with 3 independent seeds
    for sigma in STANDARD_NOISE_SIGMAS:
        for seed in (NOISE_REPETITION_SEEDS if sigma > 0 else [42]):
            r_pert, meta = apply_gaussian_noise(right_bgr, sigma, seed=seed)

            bm_res = compute_disparity_bm(to_gray(left_bgr), to_gray(r_pert), bm_cfg)
            m_bm = compute_stereo_metrics(bm_res.disparity, disp_gt, mask_eval=mask_nocc, pred_valid_mask=bm_res.valid_mask, runtime_sec=bm_res.runtime_sec)
            records.append({
                "scene": scene_name,
                "perturbation_type": "gaussian_noise",
                "param_value": sigma,
                "seed": seed,
                "algorithm": "StereoBM",
                "bad_1_0": m_bm.bad_1_0,
                "bad_2_0": m_bm.bad_2_0,
                "bad_3_0": m_bm.bad_3_0,
                "coverage": m_bm.coverage,
                "mae": m_bm.mae,
                "rmse": m_bm.rmse,
                "runtime_ms": m_bm.runtime_ms,
                "sat_low_pct": meta.saturation_low_pct,
                "sat_high_pct": meta.saturation_high_pct,
            })

            sgbm_res = compute_disparity_sgbm(left_bgr, r_pert, sgbm_cfg)
            m_sgbm = compute_stereo_metrics(sgbm_res.disparity, disp_gt, mask_eval=mask_nocc, pred_valid_mask=sgbm_res.valid_mask, runtime_sec=sgbm_res.runtime_sec)
            records.append({
                "scene": scene_name,
                "perturbation_type": "gaussian_noise",
                "param_value": sigma,
                "seed": seed,
                "algorithm": "StereoSGBM",
                "bad_1_0": m_sgbm.bad_1_0,
                "bad_2_0": m_sgbm.bad_2_0,
                "bad_3_0": m_sgbm.bad_3_0,
                "coverage": m_sgbm.coverage,
                "mae": m_sgbm.mae,
                "rmse": m_sgbm.rmse,
                "runtime_ms": m_sgbm.runtime_ms,
                "sat_low_pct": meta.saturation_low_pct,
                "sat_high_pct": meta.saturation_high_pct,
            })

    print(f"Completed perturbations for {scene_name:12s}")

df = pd.DataFrame(records)
df.to_csv(OUT_DIR / "robustness_results.csv", index=False)
with open(OUT_DIR / "robustness_results.json", "w") as f:
    json.dump(records, f, indent=2)

print(f"\nRobustness experiments completed! Saved {len(records)} test condition records.")
