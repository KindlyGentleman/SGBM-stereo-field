"""
Milestone 5 (ROI Analysis): Region-of-interest diagnostics for Gunung outdoor imagery (puncakTinggi).
Extracts and quantifies reconstruction behavior in four distinct challenge zones:
1. Bright Sky (over-exposure / low texture)
2. Structural Building (geometric planar boundaries)
3. Grass / Vegetation (thin occluding structures)
4. Volcanic Gravel / Ground (repetitive micro-texture)
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
from stereo_robustness.consistency import compute_unsupervised_diagnostics
from stereo_robustness.visualization import colorize_disparity, save_image_rgb
from stereo_robustness.ieee_style import setup_ieee_style, create_figure, save_publication_figure, IEEE_DOUBLE_COL_WIDTH

OUT_DIR = Path("results/gunung/roi_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = Path("figures/gunung/roi")
FIG_DIR.mkdir(parents=True, exist_ok=True)

with open("configs/gunung_dataset.yaml", "r") as f:
    config = yaml.safe_load(f)

calib_path = Path(config["calibration"]["params_file"])
calib_params = StereoCalibrationParams.from_npz(calib_path)
rect_models = compute_rectification_models(calib_params, alpha=0.0, new_image_size=(1280, 720))

s_dir = Path(config["dataset"]["root_dir"]) / "puncakTinggi" / "stereo"
l_raw = cv2.imread(str(s_dir / "left_0000.png"))
r_raw = cv2.imread(str(s_dir / "right_0000.png"))
rect_l, rect_r = rect_models.rectify(l_raw, r_raw)

# Load frozen SGBM configs
with open("results/middlebury/best_configs/sgbm_frozen_config.json", "r") as f:
    sgbm_frozen = json.load(f)

def clean_dict(d: dict, cls) -> dict:
    return {k: v for k, v in d.items() if k in cls.__annotations__}

g0_cfg = StereoSGBMConfig(**clean_dict(sgbm_frozen["G0_default"], StereoSGBMConfig))
g0_cfg.numDisparities = 128
g1_cfg = StereoSGBMConfig(**clean_dict(sgbm_frozen["G1_optimized"], StereoSGBMConfig))
g1_cfg.numDisparities = 128

res_g0 = compute_disparity_sgbm(rect_l, rect_r, g0_cfg)
res_g1 = compute_disparity_sgbm(rect_l, rect_r, g1_cfg)
res_g1_r = compute_right_disparity(rect_l, rect_r, g1_cfg)

# Define physical ROIs: (ymin, ymax, xmin, xmax)
rois = {
    "Bright_Sky": (0, 220, 400, 1000),
    "Building_Structure": (220, 420, 850, 1200),
    "Grass_Foliage": (350, 550, 50, 400),
    "Gravel_Ground": (480, 700, 450, 950),
}

roi_records = []

setup_ieee_style()
fig, axes = create_figure(width=IEEE_DOUBLE_COL_WIDTH, aspect_ratio=0.55, nrows=4, ncols=4)

row_idx = 0
for roi_name, (y1, y2, x1, x2) in rois.items():
    roi_img = rect_l[y1:y2, x1:x2]
    disp_g0_roi = res_g0.disparity[y1:y2, x1:x2]
    valid_g0_roi = res_g0.valid_mask[y1:y2, x1:x2]
    disp_g1_roi = res_g1.disparity[y1:y2, x1:x2]
    valid_g1_roi = res_g1.valid_mask[y1:y2, x1:x2]

    # Compute ROI metrics
    cov_g0 = float(np.mean(valid_g0_roi) * 100.0)
    cov_g1 = float(np.mean(valid_g1_roi) * 100.0)

    # Intensity saturation in this ROI
    gray_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    sat_pct = float(np.mean(gray_roi >= 250) * 100.0)
    grad_x = cv2.Sobel(gray_roi, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_roi, cv2.CV_32F, 0, 1, ksize=3)
    texture_energy = float(np.mean(np.sqrt(grad_x**2 + grad_y**2)))

    roi_records.append({
        "roi_name": roi_name,
        "y_range": f"{y1}:{y2}",
        "x_range": f"{x1}:{x2}",
        "saturation_pct": sat_pct,
        "texture_energy": texture_energy,
        "g0_coverage_pct": cov_g0,
        "g1_coverage_pct": cov_g1,
    })

    # Render ROI panels
    disp_g0_col = colorize_disparity(disp_g0_roi, valid_g0_roi, vmin=0, vmax=128)
    disp_g1_col = colorize_disparity(disp_g1_roi, valid_g1_roi, vmin=0, vmax=128)

    axes[row_idx, 0].imshow(cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB))
    axes[row_idx, 0].set_ylabel(roi_name.replace("_", "\n"), fontsize=7, fontweight="bold")
    if row_idx == 0:
        axes[row_idx, 0].set_title("Rectified Left RGB", fontsize=7.5)
    axes[row_idx, 0].set_xticks([])
    axes[row_idx, 0].set_yticks([])

    axes[row_idx, 1].imshow(gray_roi, cmap="gray")
    if row_idx == 0:
        axes[row_idx, 1].set_title("Grayscale Texture", fontsize=7.5)
    axes[row_idx, 1].set_xticks([])
    axes[row_idx, 1].set_yticks([])

    axes[row_idx, 2].imshow(disp_g0_col)
    if row_idx == 0:
        axes[row_idx, 2].set_title("G0 Baseline Disparity", fontsize=7.5)
    axes[row_idx, 2].set_xticks([])
    axes[row_idx, 2].set_yticks([])

    axes[row_idx, 3].imshow(disp_g1_col)
    if row_idx == 0:
        axes[row_idx, 3].set_title("G1 Optimized Disparity", fontsize=7.5)
    axes[row_idx, 3].set_xticks([])
    axes[row_idx, 3].set_yticks([])

    row_idx += 1

save_publication_figure(fig, FIG_DIR / "gunung_roi_breakdown")

df_roi = pd.DataFrame(roi_records)
df_roi.to_csv(OUT_DIR / "gunung_roi_metrics.csv", index=False)
print("\n--- GUNUNG ROI BREAKDOWN TABLE ---")
print(df_roi.to_string(index=False))
