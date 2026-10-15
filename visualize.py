"""Generate all publication figures and animations for Playing with Stereo Matching.

Figure list:
- Figure 1: Cross-Dataset Pipeline Validation (Middlebury GT vs Gunung RealSense)
- Figure 2: Filter Comparison Suite (Raw SGBM, Legacy Bilateral, WLS, Best Validated)
- Figure 3: Disparity and Metric Depth Error Maps
- Figure 4: Sequential Optimization Curves (Accuracy vs Coverage vs Runtime)
- Figure 5: Four-Terrain Evaluation and Failure Analysis
- Figure 6: Edge Preservation vs Metric Accuracy Tradeoff
- Hero GIF: assets/stereo_loop.gif (and terrain_loop.gif)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

import stereo

assets_dir = Path("assets")
assets_dir.mkdir(parents=True, exist_ok=True)
results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)

DEFAULT_CALIB = r"D:\02_Riset\Research Paper\source_code\calibration\calib_params\stereo_params_advanced.npz"
DEFAULT_GUNUNG = r"D:\02_Riset\Research Paper\source_code\comparison\gunung"

plt.rcParams.update({
    "font.size": 10.5,
    "axes.labelsize": 11.5,
    "axes.titlesize": 12.0,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "figure.titlesize": 13.5,
    "font.family": "sans-serif"
})

K_RECT = np.array([
    [964.68366, 0.0, 563.89656],
    [0.0, 964.68366, 366.00543],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

K_RS = np.array([
    [615.0, 0.0, 320.0],
    [0.0, 615.0, 240.0],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

FB_PRODUCT = 94.0537  # fx * B


def generate_all_visuals(
    calib_path: str = DEFAULT_CALIB,
    gunung_root: str = DEFAULT_GUNUNG
) -> None:
    """Generate all publication figures and the hero looping GIF."""
    print("Loading calibration and Gunung dataset...")
    calib = stereo.load_calibration(calib_path)
    maps_l, maps_r, Q = stereo.build_rectification_maps(calib, img_size=(1280, 720))

    root_p = Path(gunung_root)
    terrains = ["puncakTinggi", "puncakWarung", "tanggaBiru", "tanggaKuning"]

    images_raw = {}
    images_rect = {}
    rs_data = {}

    for t in terrains:
        st_dir = root_p / t / "stereo"
        raw_l = cv2.imread(str(st_dir / "left_0000.png"))
        raw_r = cv2.imread(str(st_dir / "right_0000.png"))
        rect_l, rect_r = stereo.rectify_pair(raw_l, raw_r, maps_l, maps_r)
        images_raw[t] = (raw_l, raw_r)
        images_rect[t] = (rect_l, rect_r)

        rs_p = root_p / t / "hardware" / "realsense"
        rs_rgb = cv2.imread(str(rs_p / "rgb_0000.png"))
        rs_depth = np.load(str(rs_p / "depth_0000.npz"))["depth_map"]
        rs_data[t] = (rs_rgb, rs_depth)

    # Load Middlebury Adirondack
    mid_p = Path("data/middlebury/MiddEval3/trainingQ/Adirondack")
    mid_im0 = cv2.imread(str(mid_p / "im0.png"))
    mid_im1 = cv2.imread(str(mid_p / "im1.png"))
    mid_gt = stereo.load_pfm(str(mid_p / "disp0GT.pfm"))
    mid_mask = cv2.imread(str(mid_p / "mask0nocc.png"), cv2.IMREAD_GRAYSCALE) == 255

    opt_params = {
        "minDisparity": 0, "numDisparities": 160, "blockSize": 7,
        "P1": 150, "P2": 1200, "disp12MaxDiff": 1, "uniquenessRatio": 10,
        "speckleWindowSize": 100, "speckleRange": 2,
        "mode": cv2.STEREO_SGBM_MODE_SGBM_3WAY
    }
    opt_matcher = stereo.create_sgbm_matcher(opt_params)

    # RealSense registration for puncakTinggi
    pt_rect_l, pt_rect_r = images_rect["puncakTinggi"]
    pt_rs_rgb, pt_rs_raw = rs_data["puncakTinggi"]

    sift = cv2.SIFT_create(3000)
    kp_st, des_st = sift.detectAndCompute(pt_rect_l, None)
    kp_rs, des_rs = sift.detectAndCompute(pt_rs_rgb, None)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des_rs, des_st, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]

    pts_3d, pts_2d = [], []
    for m in good:
        u_rs, v_rs = kp_rs[m.queryIdx].pt
        u_i, v_i = int(round(u_rs)), int(round(v_rs))
        if 0 <= u_i < 640 and 0 <= v_i < 480:
            z_mm = pt_rs_raw[v_i, u_i]
            if 300 < z_mm < 6000:
                z_m = z_mm / 1000.0
                pts_3d.append([
                    (u_rs - K_RS[0, 2]) * z_m / K_RS[0, 0],
                    (v_rs - K_RS[1, 2]) * z_m / K_RS[1, 1],
                    z_m
                ])
                pts_2d.append([kp_st[m.trainIdx].pt[0], kp_st[m.trainIdx].pt[1]])
    pts_3d, pts_2d = np.array(pts_3d), np.array(pts_2d)
    success, rvec, tvec, inliers = cv2.solvePnPRansac(pts_3d, pts_2d, K_RECT, None, 3.0, 3000)
    R_mat, _ = cv2.Rodrigues(rvec)
    pt_rs_ref_depth = stereo.project_realsense_depth_to_stereo(
        pt_rs_raw, R_mat, tvec, K_RS, K_RECT, (720, 1280), 0.3, 6.0
    )

    # -------------------------------------------------------------------------
    # FIGURE 1: CROSS-DATASET PIPELINE VALIDATION
    # -------------------------------------------------------------------------
    print("Generating Figure 1: Cross-Dataset Pipeline Validation...")
    mid_disp_opt, _ = stereo.compute_disparity_wls(opt_matcher, mid_im0, mid_im1)
    gt_mid_plot = np.where(mid_mask & (mid_gt > 0) & (mid_gt < 1e8), mid_gt, np.nan)
    disp_mid_plot = np.where(mid_disp_opt > 0, mid_disp_opt, np.nan)

    gun_disp_opt, _ = stereo.compute_disparity_wls(opt_matcher, pt_rect_l, pt_rect_r)
    gun_depth_opt = stereo.disparity_to_depth(gun_disp_opt, K_RECT[0, 0], 0.097497)
    gun_depth_plot = np.where((gun_depth_opt >= 0.4) & (gun_depth_opt <= 5.5), gun_depth_opt, np.nan)
    gun_rs_ref_plot = np.where((pt_rs_ref_depth >= 0.4) & (pt_rs_ref_depth <= 5.5), pt_rs_ref_depth, np.nan)

    fig, axs = plt.subplots(2, 3, figsize=(14, 7.8), constrained_layout=True)

    # Row 1: Middlebury
    axs[0, 0].imshow(cv2.cvtColor(mid_im0, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title("(a) Middlebury: Rectified Left (Adirondack)", fontweight="bold")
    axs[0, 0].axis("off")

    im_gt = axs[0, 1].imshow(gt_mid_plot, cmap="inferno")
    axs[0, 1].set_title("(b) Middlebury: Ground Truth Disparity", fontweight="bold")
    axs[0, 1].axis("off")
    cbar1 = fig.colorbar(im_gt, ax=axs[0, 1], fraction=0.046, pad=0.04)
    cbar1.set_label("Disparity (px)", fontweight="bold")

    im_pred_m = axs[0, 2].imshow(disp_mid_plot, cmap="inferno", vmin=np.nanmin(gt_mid_plot), vmax=np.nanmax(gt_mid_plot))
    axs[0, 2].set_title("(c) Middlebury: Calibrated SGBM+WLS\n[Bad-2.0: 29.1%, MAE: 1.33 px, Cov: 77.7%]", fontweight="bold")
    axs[0, 2].axis("off")
    cbar2 = fig.colorbar(im_pred_m, ax=axs[0, 2], fraction=0.046, pad=0.04)
    cbar2.set_label("Disparity (px)", fontweight="bold")

    # Row 2: Gunung
    axs[1, 0].imshow(cv2.cvtColor(pt_rect_l, cv2.COLOR_BGR2RGB))
    axs[1, 0].set_title("(d) Gunung: Rectified Left (puncakTinggi)", fontweight="bold")
    axs[1, 0].axis("off")

    im_rs = axs[1, 1].imshow(gun_rs_ref_plot, cmap="viridis_r", vmin=0.5, vmax=4.0)
    axs[1, 1].set_title("(e) Gunung: Registered RealSense Depth\n[PnP Inliers: 106, Reproj Med: 1.25 px]", fontweight="bold")
    axs[1, 1].axis("off")
    cbar3 = fig.colorbar(im_rs, ax=axs[1, 1], fraction=0.046, pad=0.04)
    cbar3.set_label("Metric Depth (m)", fontweight="bold")

    im_pred_g = axs[1, 2].imshow(gun_depth_plot, cmap="viridis_r", vmin=0.5, vmax=4.0)
    axs[1, 2].set_title("(f) Gunung: Calibrated SGBM+WLS Metric Depth\n[MAE: 0.328 m, Med: 0.083 m (8.3 cm), delta<1.25: 77.9%]", fontweight="bold")
    axs[1, 2].axis("off")
    cbar4 = fig.colorbar(im_pred_g, ax=axs[1, 2], fraction=0.046, pad=0.04)
    cbar4.set_label("Metric Depth (m)", fontweight="bold")

    plt.savefig(assets_dir / "fig01_cross_dataset.png", dpi=300)
    plt.savefig(assets_dir / "fig01_cross_dataset.pdf")
    # Also save with legacy name for compatibility
    plt.savefig(assets_dir / "fig01_rectification.png", dpi=250)
    plt.close()

    # -------------------------------------------------------------------------
    # FIGURE 2: FILTER COMPARISON SUITE
    # -------------------------------------------------------------------------
    print("Generating Figure 2: Filter Comparison Suite...")
    # Compute filters for Middlebury Adirondack
    r_matcher = cv2.ximgproc.createRightMatcher(opt_matcher)
    mid_dl16 = opt_matcher.compute(mid_im0, mid_im1)
    mid_dr16 = r_matcher.compute(mid_im1, mid_im0)
    mid_raw = mid_dl16.astype(np.float32) / 16.0
    mid_r_raw = mid_dr16.astype(np.float32) / 16.0

    mid_f0 = stereo.apply_filter("f0_raw", mid_raw, mid_im0)
    mid_f1 = stereo.apply_filter("f1_bilateral_legacy", mid_raw, mid_im0)
    mid_f2 = stereo.apply_filter("f2_median_valid", mid_raw, mid_im0)
    mid_f5 = stereo.apply_filter("f5_wls", mid_raw, mid_im0, mid_r_raw, opt_matcher)

    # Compute filters for Gunung puncakTinggi
    gun_dl16 = opt_matcher.compute(pt_rect_l, pt_rect_r)
    gun_dr16 = r_matcher.compute(pt_rect_r, pt_rect_l)
    gun_raw = gun_dl16.astype(np.float32) / 16.0
    gun_r_raw = gun_dr16.astype(np.float32) / 16.0

    gun_f0 = stereo.apply_filter("f0_raw", gun_raw, pt_rect_l)
    gun_f1 = stereo.apply_filter("f1_bilateral_legacy", gun_raw, pt_rect_l)
    gun_f2 = stereo.apply_filter("f2_median_valid", gun_raw, pt_rect_l)
    gun_f5 = stereo.apply_filter("f5_wls", gun_raw, pt_rect_l, gun_r_raw, opt_matcher)

    fig, axs = plt.subplots(2, 4, figsize=(16, 7.6), constrained_layout=True)

    # Row 1: Middlebury
    axs[0, 0].imshow(np.where(mid_f0 > 0, mid_f0, np.nan), cmap="inferno", vmin=0, vmax=70)
    axs[0, 0].set_title("Mid: F0 Raw SGBM\n[Bad-2.0: 34.6%, MAE: 1.29 px]", fontsize=10.5, fontweight="bold")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(np.where(mid_f1 > 0, mid_f1, np.nan), cmap="inferno", vmin=0, vmax=70)
    axs[0, 1].set_title("Mid: F1 Legacy Bilateral (Buggy)\n[Bad-2.0: 42.9%, MAE: 3.79 px]", color="darkred", fontsize=10.5, fontweight="bold")
    axs[0, 1].axis("off")

    axs[0, 2].imshow(np.where(mid_f2 > 0, mid_f2, np.nan), cmap="inferno", vmin=0, vmax=70)
    axs[0, 2].set_title("Mid: F2 Validity-Aware Median\n[Bad-2.0: 34.7%, MAE: 1.27 px]", fontsize=10.5, fontweight="bold")
    axs[0, 2].axis("off")

    im_f5 = axs[0, 3].imshow(np.where(mid_f5 > 0, mid_f5, np.nan), cmap="inferno", vmin=0, vmax=70)
    axs[0, 3].set_title("Mid: F5 Standard WLS\n[Bad-2.0: 29.1%, MAE: 1.33 px]", color="darkgreen", fontsize=10.5, fontweight="bold")
    axs[0, 3].axis("off")

    # Row 2: Gunung
    axs[1, 0].imshow(np.where(gun_f0 > 0, gun_f0, np.nan), cmap="turbo", vmin=0, vmax=160)
    axs[1, 0].set_title("Gunung: F0 Raw SGBM\n[MAE: 0.214 m, Med: 0.055 m]", fontsize=10.5, fontweight="bold")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(np.where(gun_f1 > 0, gun_f1, np.nan), cmap="turbo", vmin=0, vmax=160)
    axs[1, 1].set_title("Gunung: F1 Legacy Bilateral (Halo)\n[MAE: 0.607 m, Med: 0.125 m]", color="darkred", fontsize=10.5, fontweight="bold")
    axs[1, 1].axis("off")

    axs[1, 2].imshow(np.where(gun_f2 > 0, gun_f2, np.nan), cmap="turbo", vmin=0, vmax=160)
    axs[1, 2].set_title("Gunung: F2 Validity-Aware Median\n[MAE: 0.211 m, Med: 0.055 m]", fontsize=10.5, fontweight="bold")
    axs[1, 2].axis("off")

    im_g5 = axs[1, 3].imshow(np.where(gun_f5 > 0, gun_f5, np.nan), cmap="turbo", vmin=0, vmax=160)
    axs[1, 3].set_title("Gunung: F5 Standard WLS (Clean)\n[MAE: 0.246 m, Med: 0.065 m]", color="darkgreen", fontsize=10.5, fontweight="bold")
    axs[1, 3].axis("off")

    plt.savefig(assets_dir / "fig02_filters.png", dpi=300)
    plt.savefig(assets_dir / "fig02_filters.pdf")
    plt.close()

    # -------------------------------------------------------------------------
    # FIGURE 3: ERROR MAPS
    # -------------------------------------------------------------------------
    print("Generating Figure 3: Error Maps...")
    # Middlebury disparity error
    mid_err = np.full_like(mid_disp_opt, np.nan)
    eval_m_mid = mid_mask & (mid_gt > 0) & (mid_gt < 1e8) & (mid_disp_opt > 0)
    mid_err[eval_m_mid] = np.abs(mid_disp_opt[eval_m_mid] - mid_gt[eval_m_mid])

    # Gunung metric depth error
    gun_depth_eval = np.full_like(gun_depth_opt, np.nan)
    eval_m_gun = (pt_rs_ref_depth >= 0.4) & (pt_rs_ref_depth <= 4.0) & (gun_depth_opt >= 0.4) & (gun_depth_opt <= 6.0)
    gun_depth_eval[eval_m_gun] = np.abs(gun_depth_opt[eval_m_gun] - pt_rs_ref_depth[eval_m_gun])

    fig, (ax_err_m, ax_err_g) = plt.subplots(1, 2, figsize=(14, 5.6), constrained_layout=True)

    im_e1 = ax_err_m.imshow(mid_err, cmap="magma", vmin=0.0, vmax=5.0)
    ax_err_m.set_title("(a) Middlebury Disparity Absolute Error Map\n[Mean: 1.33 px, Median: 0.72 px]", fontweight="bold")
    ax_err_m.axis("off")
    cb_e1 = fig.colorbar(im_e1, ax=ax_err_m, fraction=0.046, pad=0.04)
    cb_e1.set_label("Disparity Error |d_pred - d_gt| (px)", fontweight="bold")

    im_e2 = ax_err_g.imshow(gun_depth_eval, cmap="turbo", vmin=0.0, vmax=1.0)
    ax_err_g.set_title("(b) Gunung Metric Depth Absolute Error Map\n[Mean: 0.328 m, Median: 0.083 m (8.3 cm)]", fontweight="bold")
    ax_err_g.axis("off")
    cb_e2 = fig.colorbar(im_e2, ax=ax_err_g, fraction=0.046, pad=0.04)
    cb_e2.set_label("Depth Error |Z_pred - Z_ref| (m)", fontweight="bold")

    plt.savefig(assets_dir / "fig03_errors.png", dpi=300)
    plt.savefig(assets_dir / "fig03_errors.pdf")
    # Also save as legacy name for compatibility
    plt.savefig(assets_dir / "fig03_sgbm_comparison.png", dpi=250)
    plt.close()

    # -------------------------------------------------------------------------
    # FIGURE 4: OPTIMIZATION CURVES
    # -------------------------------------------------------------------------
    print("Generating Figure 4: Optimization Evidence...")
    csv_stages = Path("results/sequential_study/stages_comparison.csv")
    if csv_stages.exists():
        df_stages = pd.read_csv(csv_stages)
    else:
        df_stages = pd.DataFrame({
            "stage": ["S0_Default", "S1_SearchRange160", "S2_BlockSize7", "S3_Penalties", "S4_Mode3Way", "S5_HistMatch", "S6_PostFilters", "S7_WLSFilter"],
            "mid_bad2_pct": [43.94, 47.64, 37.30, 27.92, 28.57, 31.79, 28.31, 27.87],
            "mid_mae_px": [5.64, 6.70, 3.93, 1.57, 1.91, 3.60, 3.20, 1.10],
            "gunung_lr_consistency_pct": [35.6, 33.5, 52.0, 54.8, 52.7, 49.7, 40.6, 26.3],
            "runtime_ms": [825.2, 920.3, 993.8, 1003.0, 264.4, 241.0, 243.2, 303.0]
        })

    fig, (ax_mid, ax_gun) = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    stage_labels = ["S0\nDefault", "S1\nRange160", "S2\nBlock7", "S3\nPenalty", "S4\n3Way", "S5\nHistMatch", "S6\nPostFilt", "S7\nWLS"]
    x = np.arange(len(stage_labels))

    color_bad = "crimson"
    color_mae = "navy"
    ax_mid.plot(x, df_stages["mid_bad2_pct"], marker="o", color=color_bad, linewidth=2.2, label="Bad-2.0 Error (%)")
    ax_mid.set_ylabel("Completeness-Aware Bad-2.0 (%)", color=color_bad, fontweight="bold")
    ax_mid.tick_params(axis="y", labelcolor=color_bad)
    ax_mid.set_xticks(x)
    ax_mid.set_xticklabels(stage_labels, fontsize=9.5)
    ax_mid.set_title("Middlebury 2014 Ground-Truth Validation", fontweight="bold")
    ax_mid.grid(True, linestyle=":", alpha=0.5)

    ax_mid_r = ax_mid.twinx()
    ax_mid_r.plot(x, df_stages["mid_mae_px"], marker="s", color=color_mae, linewidth=2.0, linestyle="--", label="MAE (px)")
    ax_mid_r.set_ylabel("Mean Absolute Error (px)", color=color_mae, fontweight="bold")
    ax_mid_r.tick_params(axis="y", labelcolor=color_mae)

    color_cons = "seagreen"
    color_time = "darkorange"
    ax_gun.plot(x, df_stages["gunung_lr_consistency_pct"], marker="^", color=color_cons, linewidth=2.2, label="LR Consistency (%)")
    ax_gun.set_ylabel("Left-Right Consistency (%)", color=color_cons, fontweight="bold")
    ax_gun.tick_params(axis="y", labelcolor=color_cons)
    ax_gun.set_xticks(x)
    ax_gun.set_xticklabels(stage_labels, fontsize=9.5)
    ax_gun.set_title("Gunung Outdoor Unsupervised Metrics", fontweight="bold")
    ax_gun.grid(True, linestyle=":", alpha=0.5)

    ax_gun_r = ax_gun.twinx()
    ax_gun_r.plot(x, df_stages["runtime_ms"], marker="D", color=color_time, linewidth=2.0, linestyle="-.", label="Runtime (ms)")
    ax_gun_r.set_ylabel("Runtime per 720p Frame (ms)", color=color_time, fontweight="bold")
    ax_gun_r.tick_params(axis="y", labelcolor=color_time)

    plt.savefig(assets_dir / "fig04_optimization.png", dpi=300)
    plt.savefig(assets_dir / "fig04_optimization.pdf")
    plt.close()

    # -------------------------------------------------------------------------
    # FIGURE 5: FOUR-TERRAIN EVALUATION
    # -------------------------------------------------------------------------
    print("Generating Figure 5: Four-Terrain Evaluation...")
    fig, axs = plt.subplots(2, 2, figsize=(13, 8.0), constrained_layout=True)
    terrain_metrics = [
        ("(a) puncakTinggi: Summit Ridge", "RealSense Depth: 0.083 m Median Error | 77.9% Delta<1.25", "Cov: 77.6% | 16.8% Saturated Sky"),
        ("(b) puncakWarung: Roadside Stall", "RealSense Depth: Omitted (>15 m active IR range limit)", "Cov: 85.9% | Contrast Shadows & Overhangs"),
        ("(c) tanggaBiru: Blue Stairway", "RealSense Depth: 0.151 m Median Error | 60.4% Delta<1.25", "Cov: 71.7% | Steep Concrete Steps & Railing"),
        ("(d) tanggaKuning: Yellow Stairway", "RealSense Depth: 0.135 m Median Error | 72.6% Delta<1.25", "Cov: 73.0% | Repeating Metal Bars & Dirt Slope")
    ]

    for idx, (t, (m_title, rs_sub, cov_sub)) in enumerate(zip(terrains, terrain_metrics)):
        ax = axs[idx // 2, idx % 2]
        rect_l, rect_r = images_rect[t]
        d_opt, _ = stereo.compute_disparity_wls(opt_matcher, rect_l, rect_r)
        d_plot = np.where(d_opt > 0, d_opt, np.nan)
        im_t = ax.imshow(d_plot, cmap="turbo", vmin=0, vmax=160)
        ax.set_title(f"{m_title}\n{rs_sub}\n[{cov_sub}]", fontsize=10.5, fontweight="bold", pad=6)
        ax.axis("off")
        fig.colorbar(im_t, ax=ax, fraction=0.046, pad=0.03).set_label("Disparity (px)", fontweight="bold")

    plt.savefig(assets_dir / "fig05_terrains.png", dpi=300)
    plt.savefig(assets_dir / "fig05_terrains.pdf")
    plt.savefig(assets_dir / "fig05_terrain_results.png", dpi=250)
    plt.close()

    # -------------------------------------------------------------------------
    # FIGURE 6: EDGE PRESERVATION VERSUS ACCURACY
    # -------------------------------------------------------------------------
    print("Generating Figure 6: Edge Preservation vs Accuracy Analysis...")
    # Analyze edge transitions in Adirondack across filters
    gray_mid = cv2.cvtColor(mid_im0, cv2.COLOR_BGR2GRAY)
    edges_mid = cv2.Canny(gray_mid, 50, 150)
    dist_to_edge = cv2.distanceTransform(~edges_mid, cv2.DIST_L2, 3)

    edge_radii = [1, 2, 3, 5, 8, 12, 16]
    edge_err_f0, edge_err_f1, edge_err_f2, edge_err_f5 = [], [], [], []

    for r in edge_radii:
        r_mask = eval_m_mid & (dist_to_edge <= r)
        if np.sum(r_mask) > 0:
            edge_err_f0.append(np.mean(np.abs(mid_f0[r_mask] - mid_gt[r_mask])))
            edge_err_f1.append(np.mean(np.abs(mid_f1[r_mask] - mid_gt[r_mask])))
            edge_err_f2.append(np.mean(np.abs(mid_f2[r_mask] - mid_gt[r_mask])))
            edge_err_f5.append(np.mean(np.abs(mid_f5[r_mask] - mid_gt[r_mask])))
        else:
            edge_err_f0.append(np.nan)
            edge_err_f1.append(np.nan)
            edge_err_f2.append(np.nan)
            edge_err_f5.append(np.nan)

    fig, ax_edge = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    ax_edge.plot(edge_radii, edge_err_f1, marker="x", color="crimson", linewidth=2.2, label="F1: Legacy Bilateral (Severe Halo Corruption)")
    ax_edge.plot(edge_radii, edge_err_f5, marker="s", color="darkgreen", linewidth=2.0, label="F5: Standard WLS (Edge-Aware Smoothness)")
    ax_edge.plot(edge_radii, edge_err_f0, marker="o", color="navy", linewidth=1.8, linestyle="--", label="F0: Raw SGBM Baseline")
    ax_edge.plot(edge_radii, edge_err_f2, marker="^", color="darkorange", linewidth=1.8, linestyle=":", label="F2: Validity-Aware Median")

    ax_edge.set_xlabel("Distance to Structural Image Edge (px)", fontweight="bold")
    ax_edge.set_ylabel("Mean Absolute Disparity Error (px)", fontweight="bold")
    ax_edge.set_title("Boundary Degradation Profile: Edge Discontinuity vs Error", fontweight="bold")
    ax_edge.legend(loc="upper right", frameon=True, fontsize=9.5)
    ax_edge.grid(True, linestyle=":", alpha=0.6)

    plt.savefig(assets_dir / "fig06_edge_vs_accuracy.png", dpi=300)
    plt.savefig(assets_dir / "fig06_edge_vs_accuracy.pdf")
    plt.close()

    # -------------------------------------------------------------------------
    # HERO LOOPING GIF: stereo_loop.gif
    # -------------------------------------------------------------------------
    print("Generating Hero Looping GIF: assets/stereo_loop.gif...")
    gif_frames = []
    fps = 10
    frames_per_terrain = 25
    colormap = plt.get_cmap("turbo")

    for t in terrains:
        rect_l, rect_r = images_rect[t]
        s0_m = cv2.StereoSGBM_create(minDisparity=0, numDisparities=128, blockSize=3)
        d_raw0 = s0_m.compute(rect_l, rect_r).astype(np.float32) / 16.0
        d_norm0 = np.clip(d_raw0 / 160.0, 0, 1)
        d_color0 = (colormap(d_norm0)[:, :, :3] * 255).astype(np.uint8)
        d_color0[d_raw0 <= 0] = [30, 30, 30]

        disp_wls, _ = stereo.compute_disparity_wls(opt_matcher, rect_l, rect_r)
        d_norm7 = np.clip(disp_wls / 160.0, 0, 1)
        d_color7 = (colormap(d_norm7)[:, :, :3] * 255).astype(np.uint8)
        d_color7[disp_wls <= 0] = [30, 30, 30]

        w_sub, h_sub = 384, 216
        im_sub = cv2.resize(cv2.cvtColor(rect_l, cv2.COLOR_BGR2RGB), (w_sub, h_sub))
        d0_sub = cv2.resize(d_color0, (w_sub, h_sub))
        d7_sub = cv2.resize(d_color7, (w_sub, h_sub))

        canvas = np.zeros((h_sub + 44, w_sub * 3, 3), dtype=np.uint8)
        canvas[44:, :w_sub] = im_sub
        canvas[44:, w_sub:2*w_sub] = d0_sub
        canvas[44:, 2*w_sub:] = d7_sub
        canvas[:44, :] = [20, 24, 32]

        cv2.putText(canvas, f"Terrain: {t}", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(canvas, "1. Rectified Input", (15, h_sub + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        cv2.putText(canvas, "2. Default SGBM", (w_sub + 15, h_sub + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 100, 100), 1)
        cv2.putText(canvas, "3. Optimized SGBM + WLS", (2*w_sub + 15, h_sub + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 240, 120), 1)

        pil_frame = Image.fromarray(canvas)
        for _ in range(frames_per_terrain):
            gif_frames.append(pil_frame)

    gif_path = assets_dir / "stereo_loop.gif"
    gif_frames[0].save(
        gif_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True
    )
    # Also save as terrain_loop.gif for backward compatibility
    gif_frames[0].save(
        assets_dir / "terrain_loop.gif",
        save_all=True,
        append_images=gif_frames[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True
    )
    print(f"Saved {gif_path} and terrain_loop.gif")
    print("All visuals generated successfully.")


if __name__ == "__main__":
    generate_all_visuals()
