"""
Milestone 6 (Phase G): Gunung-specific sequential target-domain adaptation.
Tests single-variable interventions on Gunung outdoor imagery:
1. Radiometric adaptation (Histogram Matching, CLAHE)
2. Matching window scaling (blockSize 5 vs 7 vs 9)
3. Extended disparity search range (128 vs 160)
4. Left-Right consistency outlier invalidation
5. Edge-preserving disparity post-filtering (WLS / Guided)
Outputs auditable hypothesis-evidence table to docs/GUNUNG_ADAPTATION_AUDIT.md.
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
    StereoSGBMConfig,
    compute_disparity_sgbm,
    compute_right_disparity,
)
from stereo_robustness.preprocessing import preprocess_pair
from stereo_robustness.consistency import compute_unsupervised_diagnostics, check_left_right_consistency
from stereo_robustness.postprocessing import apply_speckle_filter, apply_median_filter, apply_wls_filter
from stereo_robustness.visualization import colorize_disparity, save_image_rgb

OUT_DIR = Path("results/gunung/adaptation")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DISP_DIR = OUT_DIR / "disparities"
DISP_DIR.mkdir(parents=True, exist_ok=True)

with open("configs/gunung_dataset.yaml", "r") as f:
    config = yaml.safe_load(f)

calib_path = Path(config["calibration"]["params_file"])
calib_params = StereoCalibrationParams.from_npz(calib_path)
rect_models = compute_rectification_models(calib_params, alpha=0.0, new_image_size=(1280, 720))

s_dir = Path(config["dataset"]["root_dir"]) / "puncakTinggi" / "stereo"
l_raw = cv2.imread(str(s_dir / "left_0000.png"))
r_raw = cv2.imread(str(s_dir / "right_0000.png"))
rect_l, rect_r = rect_models.rectify(l_raw, r_raw)

# Load frozen SGBM config G1 as unchanged transfer baseline
with open("results/middlebury/best_configs/sgbm_frozen_config.json", "r") as f:
    sgbm_frozen = json.load(f)

def clean_dict(d: dict, cls) -> dict:
    return {k: v for k, v in d.items() if k in cls.__annotations__}

base_cfg_dict = clean_dict(sgbm_frozen["G1_optimized"], StereoSGBMConfig)
base_cfg_dict["numDisparities"] = 128

interventions = [
    # 0. Unchanged transfer baseline
    {
        "id": "A0_Transfer_Baseline",
        "hypothesis": "Frozen Middlebury G1 configuration applied directly without adaptation.",
        "prep": "gray",
        "cfg_delta": {},
        "post_filter": "none",
    },
    # 1. Radiometric Adaptation: CLAHE
    {
        "id": "A1_Prep_CLAHE",
        "hypothesis": "Local contrast enhancement via CLAHE recovers low-contrast gravel texture in shadow.",
        "prep": "clahe",
        "cfg_delta": {},
        "post_filter": "none",
    },
    # 2. Radiometric Adaptation: Histogram Matching
    {
        "id": "A2_Prep_HistMatch",
        "hypothesis": "Histogram matching compensates dynamic range divergence caused by direct sunlight.",
        "prep": "hist_match",
        "cfg_delta": {},
        "post_filter": "none",
    },
    # 3. Matching Window Expansion: blockSize = 9
    {
        "id": "A3_Window_Size_9",
        "hypothesis": "Larger 9x9 correlation window provides sufficient context to resolve repetitive volcanic gravel.",
        "prep": "gray",
        "cfg_delta": {"blockSize": 9},
        "post_filter": "none",
    },
    # 4. Extended Disparity Range: numDisparities = 160
    {
        "id": "A4_Search_Range_160",
        "hypothesis": "Expanding search range from 128 to 160 pixels captures near-field foreground rocks (<0.6m).",
        "prep": "gray",
        "cfg_delta": {"numDisparities": 160},
        "post_filter": "none",
    },
    # 5. Left-Right Consistency Filtering
    {
        "id": "A5_LR_Consistency_Filter",
        "hypothesis": "Invalidating pixels failing round-trip correspondence eliminates false matches in thin grass.",
        "prep": "gray",
        "cfg_delta": {},
        "post_filter": "lr_mask",
    },
    # 6. Edge-Preserving Disparity Post-filtering (Bilateral/WLS)
    {
        "id": "A6_Edge_Preserving_Filter",
        "hypothesis": "Guided edge-preserving smoothing fills planar cavities while strictly preserving building edges.",
        "prep": "gray",
        "cfg_delta": {},
        "post_filter": "edge_guided",
    },
]

records = []
print("Executing Sequential Target-Domain Adaptation on puncakTinggi...")

for item in interventions:
    cur_cfg_dict = base_cfg_dict.copy()
    cur_cfg_dict.update(item["cfg_delta"])
    sgbm_cfg = StereoSGBMConfig(**cur_cfg_dict)

    l_p, r_p = preprocess_pair(rect_l, rect_r, method=item["prep"])
    res_l = compute_disparity_sgbm(l_p, r_p, sgbm_cfg)
    res_r = compute_right_disparity(l_p, r_p, sgbm_cfg)

    disp_final = res_l.disparity.copy()
    valid_final = res_l.valid_mask.copy()

    # Apply isolated post-filter if specified
    if item["post_filter"] == "lr_mask":
        cons_mask, _, _ = check_left_right_consistency(disp_final, res_r.disparity, valid_final, res_r.valid_mask, threshold=1.5)
        valid_final = cons_mask
        disp_final[~valid_final] = 0.0
    elif item["post_filter"] == "edge_guided":
        disp_filtered, val_filtered = apply_wls_filter(res_l, res_r, rect_l, wls_lambda=8000.0, wls_sigma=1.5)
        disp_final = disp_filtered
        valid_final = val_filtered

    diag = compute_unsupervised_diagnostics(
        rect_l, rect_r, disp_final, res_r.disparity, valid_final, res_r.valid_mask, lr_threshold=1.5
    )

    records.append({
        "intervention_id": item["id"],
        "hypothesis": item["hypothesis"],
        "runtime_ms": res_l.runtime_sec * 1000.0,
        "valid_coverage_pct": diag.valid_disparity_pct,
        "lr_consistent_pct_valid": diag.lr_consistent_pct_of_valid,
        "lr_consistent_pct_total": diag.lr_consistent_pct_of_total,
        "photometric_mae": diag.photometric_mae,
        "photometric_rmse": diag.photometric_rmse,
        "disparity_gradient_mean": diag.disparity_gradient_mean,
        "edge_alignment_score": diag.edge_alignment_score,
    })

    # Save disparity visualization
    disp_vis = colorize_disparity(disp_final, valid_final, vmin=0, vmax=float(cur_cfg_dict["numDisparities"]))
    save_image_rgb(DISP_DIR / f"{item['id']}_disp.png", disp_vis)

    print(f"Intervention {item['id']:25s} | Cov: {diag.valid_disparity_pct:5.1f}% | LR-Cons(Valid): {diag.lr_consistent_pct_of_valid:5.1f}% | Photo MAE: {diag.photometric_mae:5.2f} | EdgeScore: {diag.edge_alignment_score:5.3f}")

df_adapt = pd.DataFrame(records)
df_adapt.to_csv(OUT_DIR / "adaptation_metrics.csv", index=False)
with open(OUT_DIR / "adaptation_metrics.json", "w") as f:
    json.dump(records, f, indent=2)

print("\n--- TARGET-DOMAIN ADAPTATION SUMMARY ---")
print(df_adapt[["intervention_id", "valid_coverage_pct", "lr_consistent_pct_valid", "photometric_mae", "edge_alignment_score", "runtime_ms"]].to_string(index=False))
