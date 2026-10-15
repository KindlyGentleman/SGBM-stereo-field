"""
Milestone 4 (Phase D): Gunung camera geometry diagnostics and rectification evaluation.
Evaluates raw epipolar geometry and compares three rectification pathways:
1. Calibrated rectification (from calibrated intrinsics and extrinsics)
2. Uncalibrated rectification (Hartley's fundamental matrix homographies)
3. Provisional vertical shift alignment
Generates epipolar overlays, correspondence plots, residual histograms, and diagnostic reports.
"""

from __future__ import annotations
import json
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from stereo_robustness.calibration import StereoCalibrationParams, compute_rectification_models
from stereo_robustness.geometry_checks import detect_and_match_features, evaluate_epipolar_geometry
from stereo_robustness.rectification import (
    rectify_calibrated,
    rectify_uncalibrated,
    rectify_provisional_shift,
)
from stereo_robustness.ieee_style import setup_ieee_style, create_figure, save_publication_figure, IEEE_SINGLE_COL_WIDTH, IEEE_DOUBLE_COL_WIDTH

OUT_DIR = Path("results/gunung/rectification")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = Path("figures/gunung/rectification")
FIG_DIR.mkdir(parents=True, exist_ok=True)

with open("configs/gunung_dataset.yaml", "r") as f:
    config = yaml.safe_load(f)

calib_path = Path(config["calibration"]["params_file"])
calib_params = StereoCalibrationParams.from_npz(calib_path)

scenes = config["dataset"]["scenes"]
base_dir = Path(config["dataset"]["root_dir"])

records = []

print(f"Running Geometry Diagnostics on {len(scenes)} Gunung scenes...")

for scene_name in scenes:
    s_dir = base_dir / scene_name / "stereo"
    l_path = s_dir / "left_0000.png"
    r_path = s_dir / "right_0000.png"

    l_img = cv2.imread(str(l_path))
    r_img = cv2.imread(str(r_path))
    h, w = l_img.shape[:2]

    # 1. Raw correspondence diagnostics
    pts1_raw, pts2_raw, raw_matches = detect_and_match_features(l_img, r_img, method="orb", max_features=3000)
    raw_diag, inlier_mask_raw, F_raw = evaluate_epipolar_geometry(pts1_raw, pts2_raw, (h, w), ransac_thresh=1.5)

    print(f"\n--- Scene: {scene_name} ---")
    print(f"Raw matches: {raw_diag.total_matches} | Inliers: {raw_diag.inlier_matches} ({raw_diag.inlier_ratio*100:.1f}%)")
    print(f"Raw Median Vertical Displacement: {raw_diag.raw_vertical_disp_median:.2f} px (P90: {raw_diag.raw_vertical_disp_p90:.2f} px)")

    # 2. Test Pathway 1: Calibrated rectification
    res_calib = rectify_calibrated(l_img, r_img, calib_params, alpha=0.0)
    print(f"Calibrated Rectification   | Median Vertical Disp: {res_calib.after_diag.raw_vertical_disp_median:.2f} px | P90: {res_calib.after_diag.raw_vertical_disp_p90:.2f} px | Inliers: {res_calib.after_diag.inlier_matches}")

    # 3. Test Pathway 2: Uncalibrated rectification
    try:
        res_uncalib = rectify_uncalibrated(l_img, r_img, ransac_thresh=1.5)
        uncalib_med = res_uncalib.after_diag.raw_vertical_disp_median
        uncalib_p90 = res_uncalib.after_diag.raw_vertical_disp_p90
        uncalib_inl = res_uncalib.after_diag.inlier_matches
        print(f"Uncalibrated Rectification | Median Vertical Disp: {uncalib_med:.2f} px | P90: {uncalib_p90:.2f} px | Inliers: {uncalib_inl}")
    except Exception as e:
        print(f"Uncalibrated Rectification failed: {e}")
        res_uncalib = None
        uncalib_med = uncalib_p90 = uncalib_inl = float("nan")

    # 4. Test Pathway 3: Provisional shift alignment
    res_shift = rectify_provisional_shift(l_img, r_img)
    print(f"Provisional Shift Alignment| Median Vertical Disp: {res_shift.after_diag.raw_vertical_disp_median:.2f} px | P90: {res_shift.after_diag.raw_vertical_disp_p90:.2f} px | Inliers: {res_shift.after_diag.inlier_matches}")

    # Record metrics
    records.append({
        "scene": scene_name,
        "pathway": "Raw_Unrectified",
        "matches": raw_diag.total_matches,
        "inliers": raw_diag.inlier_matches,
        "inlier_ratio": raw_diag.inlier_ratio,
        "vert_disp_median": raw_diag.raw_vertical_disp_median,
        "vert_disp_mean": raw_diag.raw_vertical_disp_mean,
        "vert_disp_std": raw_diag.raw_vertical_disp_std,
        "vert_disp_p90": raw_diag.raw_vertical_disp_p90,
        "horiz_disp_median": raw_diag.raw_horizontal_disp_median,
        "sampson_error_mean": raw_diag.epipolar_error_mean,
        "spatial_coverage_pct": raw_diag.spatial_coverage_pct,
    })
    records.append({
        "scene": scene_name,
        "pathway": "Calibrated_Rectify",
        "matches": res_calib.after_diag.total_matches,
        "inliers": res_calib.after_diag.inlier_matches,
        "inlier_ratio": res_calib.after_diag.inlier_ratio,
        "vert_disp_median": res_calib.after_diag.raw_vertical_disp_median,
        "vert_disp_mean": res_calib.after_diag.raw_vertical_disp_mean,
        "vert_disp_std": res_calib.after_diag.raw_vertical_disp_std,
        "vert_disp_p90": res_calib.after_diag.raw_vertical_disp_p90,
        "horiz_disp_median": res_calib.after_diag.raw_horizontal_disp_median,
        "sampson_error_mean": res_calib.after_diag.epipolar_error_mean,
        "spatial_coverage_pct": res_calib.after_diag.spatial_coverage_pct,
    })
    if res_uncalib:
        records.append({
            "scene": scene_name,
            "pathway": "Uncalibrated_Hartley",
            "matches": res_uncalib.after_diag.total_matches,
            "inliers": res_uncalib.after_diag.inlier_matches,
            "inlier_ratio": res_uncalib.after_diag.inlier_ratio,
            "vert_disp_median": res_uncalib.after_diag.raw_vertical_disp_median,
            "vert_disp_mean": res_uncalib.after_diag.raw_vertical_disp_mean,
            "vert_disp_std": res_uncalib.after_diag.raw_vertical_disp_std,
            "vert_disp_p90": res_uncalib.after_diag.raw_vertical_disp_p90,
            "horiz_disp_median": res_uncalib.after_diag.raw_horizontal_disp_median,
            "sampson_error_mean": res_uncalib.after_diag.epipolar_error_mean,
            "spatial_coverage_pct": res_uncalib.after_diag.spatial_coverage_pct,
        })
    records.append({
        "scene": scene_name,
        "pathway": "Provisional_Shift",
        "matches": res_shift.after_diag.total_matches,
        "inliers": res_shift.after_diag.inlier_matches,
        "inlier_ratio": res_shift.after_diag.inlier_ratio,
        "vert_disp_median": res_shift.after_diag.raw_vertical_disp_median,
        "vert_disp_mean": res_shift.after_diag.raw_vertical_disp_mean,
        "vert_disp_std": res_shift.after_diag.raw_vertical_disp_std,
        "vert_disp_p90": res_shift.after_diag.raw_vertical_disp_p90,
        "horiz_disp_median": res_shift.after_diag.raw_horizontal_disp_median,
        "sampson_error_mean": res_shift.after_diag.epipolar_error_mean,
        "spatial_coverage_pct": res_shift.after_diag.spatial_coverage_pct,
    })

    # Save rectified images for the primary scene (puncakTinggi)
    if scene_name == "puncakTinggi":
        cv2.imwrite(str(OUT_DIR / "puncakTinggi_calibrated_left.png"), res_calib.rect_left)
        cv2.imwrite(str(OUT_DIR / "puncakTinggi_calibrated_right.png"), res_calib.rect_right)
        if res_uncalib:
            cv2.imwrite(str(OUT_DIR / "puncakTinggi_uncalib_left.png"), res_uncalib.rect_left)
            cv2.imwrite(str(OUT_DIR / "puncakTinggi_uncalib_right.png"), res_uncalib.rect_right)

        # Generate diagnostic comparison figures
        setup_ieee_style()

        # Figure 1: Vertical displacement residual histogram
        pts1_c, pts2_c, _ = detect_and_match_features(res_calib.rect_left, res_calib.rect_right)
        _, m_c, _ = evaluate_epipolar_geometry(pts1_c, pts2_c, (h, w))
        dy_raw = np.abs(pts2_raw[inlier_mask_raw, 1] - pts1_raw[inlier_mask_raw, 1])
        dy_calib = np.abs(pts2_c[m_c, 1] - pts1_c[m_c, 1]) if np.count_nonzero(m_c) > 0 else np.array([0])

        fig, ax = create_figure(width=IEEE_SINGLE_COL_WIDTH, aspect_ratio=0.7)
        ax.hist(dy_raw, bins=30, range=(0, 40), alpha=0.6, color="red", label=f"Raw (Med: {raw_diag.raw_vertical_disp_median:.1f}px)")
        ax.hist(dy_calib, bins=30, range=(0, 40), alpha=0.7, color="blue", label=f"Calibrated (Med: {res_calib.after_diag.raw_vertical_disp_median:.1f}px)")
        ax.axvline(1.5, color="green", linestyle="--", lw=1.0, label="Acceptance Gate (1.5px)")
        ax.set_xlabel("Absolute Vertical Displacement (px)", fontsize=7)
        ax.set_ylabel("Inlier Match Count", fontsize=7)
        ax.set_title("Vertical Misalignment Residual (puncakTinggi)", fontsize=8)
        ax.legend(loc="upper right", fontsize=6.5)
        save_publication_figure(fig, FIG_DIR / "vertical_residual_histogram")

        # Figure 2: Epipolar line overlay on stereo pair before and after
        fig, axes = create_figure(width=IEEE_DOUBLE_COL_WIDTH, aspect_ratio=0.5, nrows=2, ncols=2)
        # Raw Left & Right
        axes[0, 0].imshow(cv2.cvtColor(l_img, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("Raw Left View", fontsize=7.5)
        axes[0, 0].axis("off")
        axes[0, 1].imshow(cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title(f"Raw Right View (Median $\\Delta y = {raw_diag.raw_vertical_disp_median:.1f}$px)", fontsize=7.5)
        axes[0, 1].axis("off")
        # Overlay horizontal guide lines
        for y_line in [150, 300, 450, 600]:
            axes[0, 0].axhline(y_line, color="yellow", lw=0.7, alpha=0.8)
            axes[0, 1].axhline(y_line, color="yellow", lw=0.7, alpha=0.8)

        # Calibrated Rectified Left & Right
        axes[1, 0].imshow(cv2.cvtColor(res_calib.rect_left, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title("Rectified Left View", fontsize=7.5)
        axes[1, 0].axis("off")
        axes[1, 1].imshow(cv2.cvtColor(res_calib.rect_right, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title(f"Rectified Right View (Median $\\Delta y = {res_calib.after_diag.raw_vertical_disp_median:.1f}$px)", fontsize=7.5)
        axes[1, 1].axis("off")
        for y_line in [150, 300, 450, 600]:
            axes[1, 0].axhline(y_line, color="lime", lw=0.7, alpha=0.8)
            axes[1, 1].axhline(y_line, color="lime", lw=0.7, alpha=0.8)

        save_publication_figure(fig, FIG_DIR / "epipolar_overlay_comparison")

# Save diagnostic results table
df_rect = pd.DataFrame(records)
df_rect.to_csv(OUT_DIR / "geometry_diagnostics.csv", index=False)
with open(OUT_DIR / "geometry_diagnostics.json", "w") as f:
    json.dump(records, f, indent=2)

print("\n--- GEOMETRY DIAGNOSTICS SUMMARY ---")
print(df_rect[["scene", "pathway", "vert_disp_median", "vert_disp_p90", "inliers", "sampson_error_mean"]].to_string(index=False))
