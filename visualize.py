"""Generate all publication figures and animations for Playing with Stereo Matching.

Figures:
- Figure 1: Stereo Geometry and Calibrated Processing Pipeline (fig01_stereo_geometry.png/.pdf)
- Figure 2: Cross-Dataset Pipeline Validation (fig02_middlebury_gunung.png/.pdf)
- Figure 3: Filter Comparison Suite (fig03_filters.png/.pdf)
- Figure 4: Accuracy-Coverage-CPU Pareto Frontier (fig04_cpu_tradeoffs.png/.pdf)
- Figure 5: Four-Terrain Evaluation and Failure Analysis (fig05_four_terrains.png/.pdf)
- Hero Loop GIF: assets/stereo_loop.gif cycling through all four terrains
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
DEFAULT_MID = r"data/middlebury/MiddEval3/trainingQ"

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


def generate_fig01_stereo_geometry(raw_l: np.ndarray, raw_r: np.ndarray, rect_l: np.ndarray, rect_r: np.ndarray) -> None:
    """Generate Figure 1: Stereo Geometry, Epipolar Alignment, and Processing Pipeline."""
    print("Generating Figure 1: Stereo Geometry and Pipeline...")
    fig = plt.figure(figsize=(15, 8.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.1])

    # Subplot (a): Epipolar Geometry Diagram
    ax_geom = fig.add_subplot(gs[0, 0])
    ax_geom.set_xlim(-1.2, 5.2)
    ax_geom.set_ylim(-0.8, 4.5)
    ax_geom.set_aspect("equal")
    ax_geom.axis("off")
    ax_geom.set_title("(a) Calibrated Epipolar Stereo Geometry", fontweight="bold")

    # Camera centers
    cl = np.array([0.0, 0.0])
    cr = np.array([3.0, 0.0])
    p_3d = np.array([1.9, 3.8])

    # Baseline arrow
    ax_geom.annotate("", xy=(3.0, -0.3), xytext=(0.0, -0.3),
                     arrowprops=dict(arrowstyle="<->", color="navy", lw=2.0))
    ax_geom.text(1.5, -0.6, "Baseline B = 97.50 mm", color="navy", fontweight="bold", ha="center")

    # Virtual image planes
    ax_geom.plot([-0.5, 0.7], [1.0, 1.0], color="dimgray", lw=3.0)
    ax_geom.plot([2.3, 3.5], [1.0, 1.0], color="dimgray", lw=3.0)
    ax_geom.text(0.1, 1.15, "Left Image Plane", color="black", fontsize=9.0, ha="center")
    ax_geom.text(2.9, 1.15, "Right Image Plane", color="black", fontsize=9.0, ha="center")

    # Optical axes
    ax_geom.plot([0.0, 0.0], [0.0, 1.0], color="gray", linestyle="--", lw=1.2)
    ax_geom.plot([3.0, 3.0], [0.0, 1.0], color="gray", linestyle="--", lw=1.2)
    ax_geom.text(-0.25, 0.5, "f = 964.7 px", color="black", fontsize=8.5, rotation=90)

    # Rays to 3D point
    ax_geom.plot([cl[0], p_3d[0]], [cl[1], p_3d[1]], color="crimson", lw=1.8, linestyle=":")
    ax_geom.plot([cr[0], p_3d[0]], [cr[1], p_3d[1]], color="royalblue", lw=1.8, linestyle=":")

    # Projection points
    # Ray Left intersection with y=1: x = cl[0] + 1.0 * (p_3d[0] - cl[0]) / p_3d[1]
    xl = cl[0] + 1.0 * (p_3d[0] - cl[0]) / p_3d[1]
    xr = cr[0] + 1.0 * (p_3d[0] - cr[0]) / p_3d[1]
    ax_geom.plot(xl, 1.0, "ro", markersize=6)
    ax_geom.plot(xr, 1.0, "bo", markersize=6)
    ax_geom.text(xl + 0.1, 0.85, "u_L", color="crimson", fontweight="bold")
    ax_geom.text(xr - 0.25, 0.85, "u_R", color="royalblue", fontweight="bold")

    # Optical centers dots
    ax_geom.plot(cl[0], cl[1], "ko", markersize=8)
    ax_geom.plot(cr[0], cr[1], "ko", markersize=8)
    ax_geom.text(cl[0] - 0.2, -0.05, "C_L", fontweight="bold")
    ax_geom.text(cr[0] + 0.1, -0.05, "C_R", fontweight="bold")

    # 3D Point P
    ax_geom.plot(p_3d[0], p_3d[1], "k*", markersize=14)
    ax_geom.text(p_3d[0] + 0.15, p_3d[1] + 0.1, "P(X, Y, Z)", fontweight="bold")

    # Formula box
    formula_text = (
        r"Disparity: $d = u_L - u_R$" + "\n" +
        r"Depth: $Z = \frac{f \cdot B}{d}$" + "\n" +
        r"$f \cdot B = 94.05\ \mathrm{px}\cdot\mathrm{m}$"
    )
    ax_geom.text(3.6, 2.5, formula_text, fontsize=10.0,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="aliceblue", edgecolor="cornflowerblue"))

    # Subplot (b): Pipeline Architecture Diagram
    ax_pipe = fig.add_subplot(gs[0, 1])
    ax_pipe.set_xlim(0, 10)
    ax_pipe.set_ylim(0, 5)
    ax_pipe.axis("off")
    ax_pipe.set_title("(b) CPU-First Calibrated Reconstruction Pipeline", fontweight="bold")

    boxes = [
        ("Raw Stereo\nCapture\n(1280x720)", 0.2, 1.8, 1.8, 1.6, "lightgray"),
        ("Cached Remap\nFloat32 Rectify\n(y_L = y_R)", 2.4, 1.8, 2.0, 1.6, "wheat"),
        ("Multiscale SGBM\n(0.5x, 3-Way SGBM\nnumDisp=80)", 4.8, 1.8, 2.2, 1.6, "lightblue"),
        ("Edge-Preserving\nFilter\n(WLS / Valid Median)", 7.4, 1.8, 2.3, 1.6, "lightgreen"),
    ]

    for label, bx, by, bw, bh, col in boxes:
        rect = patches.FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0.2",
                                      facecolor=col, edgecolor="black", lw=1.2)
        ax_pipe.add_patch(rect)
        ax_pipe.text(bx + bw / 2.0, by + bh / 2.0, label, ha="center", va="center",
                     fontsize=9.0, fontweight="bold")

    # Connecting arrows
    for ax_start in [2.1, 4.5, 7.1]:
        ax_pipe.annotate("", xy=(ax_start + 0.3, 2.6), xytext=(ax_start - 0.1, 2.6),
                         arrowprops=dict(arrowstyle="->", color="black", lw=2.0))

    # Reprojection branch
    rect_z = patches.FancyBboxPatch((7.4, 0.1), 2.3, 1.2, boxstyle="round,pad=0.2",
                                    facecolor="lightpink", edgecolor="black", lw=1.2)
    ax_pipe.add_patch(rect_z)
    ax_pipe.text(8.55, 0.7, "Metric Depth Z (m)\n& 3D Pointcloud", ha="center", va="center",
                 fontsize=8.5, fontweight="bold")
    ax_pipe.annotate("", xy=(8.55, 1.3), xytext=(8.55, 1.8),
                     arrowprops=dict(arrowstyle="->", color="crimson", lw=2.0))

    ax_pipe.text(5.0, 0.4, "Fast: 34 ms | Balanced: 67 ms | Quality: 332 ms",
                 ha="center", fontsize=9.5, fontweight="bold", color="darkgreen")

    # Subplot (c) & (d): Epipolar Scanline Check (Before vs After Rectification)
    ax_raw = fig.add_subplot(gs[1, 0])
    h, w = raw_l.shape[:2]
    # Compose raw side by side
    raw_composite = np.hstack([raw_l, raw_r])
    ax_raw.imshow(cv2.cvtColor(raw_composite, cv2.COLOR_BGR2RGB))
    ax_raw.set_title("(c) Raw Captured Stereo Pair (With Vertical Parallax)", fontweight="bold")
    ax_raw.axis("off")
    # Draw horizontal scanlines across raw
    for y_line in [180, 360, 540]:
        ax_raw.axhline(y=y_line, color="red", linestyle="--", lw=1.2, alpha=0.8)

    ax_rect = fig.add_subplot(gs[1, 1])
    rect_composite = np.hstack([rect_l, rect_r])
    ax_rect.imshow(cv2.cvtColor(rect_composite, cv2.COLOR_BGR2RGB))
    ax_rect.set_title("(d) Epipolar Rectified Stereo Pair (Horizontal Alignment Verified)", fontweight="bold")
    ax_rect.axis("off")
    # Draw horizontal scanlines across rectified
    for y_line in [180, 360, 540]:
        ax_rect.axhline(y=y_line, color="cyan", linestyle="-", lw=1.5, alpha=0.9)

    plt.savefig(assets_dir / "fig01_stereo_geometry.png", dpi=300)
    plt.savefig(assets_dir / "fig01_stereo_geometry.pdf")
    plt.savefig(assets_dir / "fig01_rectification.png", dpi=250)
    plt.close()
    print("Saved Figure 1.")


def generate_fig02_middlebury_gunung(
    mid_im0: np.ndarray,
    mid_gt: np.ndarray,
    mid_disp_opt: np.ndarray,
    mid_mask: np.ndarray,
    pt_rect_l: np.ndarray,
    gun_rs_ref_depth: np.ndarray,
    gun_depth_opt: np.ndarray
) -> None:
    """Generate Figure 2: Cross-Dataset Benchmark and Metric Depth Validation."""
    print("Generating Figure 2: Middlebury vs Gunung Validation...")
    fig, axs = plt.subplots(2, 3, figsize=(14, 7.8), constrained_layout=True)

    gt_mid_plot = np.where(mid_mask & (mid_gt > 0) & (mid_gt < 1e8), mid_gt, np.nan)
    disp_mid_plot = np.where(mid_disp_opt > 0, mid_disp_opt, np.nan)

    gun_depth_plot = np.where((gun_depth_opt >= 0.4) & (gun_depth_opt <= 5.5), gun_depth_opt, np.nan)
    gun_rs_ref_plot = np.where((gun_rs_ref_depth >= 0.4) & (gun_rs_ref_depth <= 5.5), gun_rs_ref_depth, np.nan)

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
    axs[1, 2].set_title("(f) Gunung: Calibrated SGBM+WLS Metric Depth\n[MAE: 0.282 m, Med: 0.063 m (6.3 cm), delta<1.25: 84.4%]", fontweight="bold")
    axs[1, 2].axis("off")
    cbar4 = fig.colorbar(im_pred_g, ax=axs[1, 2], fraction=0.046, pad=0.04)
    cbar4.set_label("Metric Depth (m)", fontweight="bold")

    plt.savefig(assets_dir / "fig02_middlebury_gunung.png", dpi=300)
    plt.savefig(assets_dir / "fig02_middlebury_gunung.pdf")
    plt.savefig(assets_dir / "fig01_cross_dataset.png", dpi=250)
    plt.close()
    print("Saved Figure 2.")


def generate_fig03_filters(
    mid_im0: np.ndarray,
    mid_raw: np.ndarray,
    mid_r_raw: np.ndarray,
    opt_matcher: cv2.StereoSGBM,
    pt_rect_l: np.ndarray,
    gun_raw: np.ndarray,
    gun_r_raw: np.ndarray
) -> None:
    """Generate Figure 3: Filter Comparison Suite."""
    print("Generating Figure 3: Filter Comparison Suite...")
    mid_f0 = stereo.apply_filter("f0_raw", mid_raw, mid_im0)
    mid_f1 = stereo.apply_filter("f1_bilateral_legacy", mid_raw, mid_im0)
    mid_f2 = stereo.apply_filter("f2_median_valid", mid_raw, mid_im0)
    mid_f5 = stereo.apply_filter("f5_wls", mid_raw, mid_im0, mid_r_raw, opt_matcher)

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

    axs[0, 3].imshow(np.where(mid_f5 > 0, mid_f5, np.nan), cmap="inferno", vmin=0, vmax=70)
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

    axs[1, 3].imshow(np.where(gun_f5 > 0, gun_f5, np.nan), cmap="turbo", vmin=0, vmax=160)
    axs[1, 3].set_title("Gunung: F5 Standard WLS (Clean)\n[MAE: 0.246 m, Med: 0.065 m]", color="darkgreen", fontsize=10.5, fontweight="bold")
    axs[1, 3].axis("off")

    plt.savefig(assets_dir / "fig03_filters.png", dpi=300)
    plt.savefig(assets_dir / "fig03_filters.pdf")
    plt.savefig(assets_dir / "fig02_filters.png", dpi=250)
    plt.close()
    print("Saved Figure 3.")


def generate_fig04_cpu_tradeoffs() -> None:
    """Generate Figure 4: Accuracy-Coverage-CPU Pareto Frontier."""
    print("Generating Figure 4: CPU Tradeoffs and Pareto Frontier...")
    fig, axs = plt.subplots(2, 2, figsize=(13.5, 9.0), constrained_layout=True)

    # Subplot (a): Middlebury Benchmark Pareto Frontier
    ax_a = axs[0, 0]
    configs = ["Default SGBM", "FAST (30 FPS)", "BALANCED (15 FPS)", "QUALITY (3 FPS)"]
    mid_lat = [125.0, 16.4, 23.4, 116.4]
    mid_bad2 = [43.94, 28.42, 29.35, 29.66]
    colors = ["crimson", "seagreen", "darkorange", "royalblue"]
    markers = ["X", "o", "s", "^"]

    for name, lat, b2, c, m in zip(configs, mid_lat, mid_bad2, colors, markers):
        ax_a.scatter(lat, b2, color=c, marker=m, s=110, zorder=5, label=name)

    # Annotate Pareto points
    ax_a.annotate("FAST: 16.4 ms\nBad-2.0: 28.4%", (16.4, 28.42), xytext=(22, 26.5),
                  arrowprops=dict(arrowstyle="->", color="seagreen", lw=1.2))
    ax_a.annotate("BALANCED: 23.4 ms\nBad-2.0: 29.4%", (23.4, 29.35), xytext=(35, 31.0),
                  arrowprops=dict(arrowstyle="->", color="darkorange", lw=1.2))
    ax_a.annotate("QUALITY: 116.4 ms\nBad-2.0: 29.7%", (116.4, 29.66), xytext=(85, 33.5),
                  arrowprops=dict(arrowstyle="->", color="royalblue", lw=1.2))
    ax_a.annotate("Default: 125.0 ms\nBad-2.0: 43.9%", (125.0, 43.94), xytext=(75, 41.0),
                  arrowprops=dict(arrowstyle="->", color="crimson", lw=1.2))

    ax_a.set_xlabel("Latency per Frame (ms)", fontweight="bold")
    ax_a.set_ylabel("Completeness Bad-2.0 (%)", fontweight="bold")
    ax_a.set_title("(a) Middlebury Benchmark Pareto Frontier", fontweight="bold")
    ax_a.grid(True, linestyle=":", alpha=0.6)
    ax_a.legend(loc="upper right", frameon=True)

    # Subplot (b): Gunung Field RealSense Validation vs Latency
    ax_b = axs[0, 1]
    gun_lat = [385.0, 34.4, 66.8, 332.3]
    gun_med_err_cm = [17.4, 10.1, 12.7, 10.2]

    for name, lat, err, c, m in zip(configs, gun_lat, gun_med_err_cm, colors, markers):
        ax_b.scatter(lat, err, color=c, marker=m, s=110, zorder=5, label=name)

    ax_b.annotate("FAST: 34.4 ms\n10.1 cm Med Err", (34.4, 10.1), xytext=(45, 9.2),
                  arrowprops=dict(arrowstyle="->", color="seagreen", lw=1.2))
    ax_b.annotate("BALANCED: 66.8 ms\n12.7 cm Med Err", (66.8, 12.7), xytext=(85, 13.8),
                  arrowprops=dict(arrowstyle="->", color="darkorange", lw=1.2))
    ax_b.annotate("QUALITY: 332.3 ms\n10.2 cm Med Err", (332.3, 10.2), xytext=(220, 11.5),
                  arrowprops=dict(arrowstyle="->", color="royalblue", lw=1.2))
    ax_b.annotate("Default: 385.0 ms\n17.4 cm Med Err", (385.0, 17.4), xytext=(260, 16.5),
                  arrowprops=dict(arrowstyle="->", color="crimson", lw=1.2))

    ax_b.set_xlabel("Latency per 720p Frame (ms)", fontweight="bold")
    ax_b.set_ylabel("Median Metric Depth Error (cm)", fontweight="bold")
    ax_b.set_title("(b) Gunung Field Depth Error vs Latency", fontweight="bold")
    ax_b.grid(True, linestyle=":", alpha=0.6)
    ax_b.legend(loc="upper right", frameon=True)

    # Subplot (c): Resolution Scaling and Breakdown
    ax_c = axs[1, 0]
    scales = ["180p (0.25x)", "360p (0.50x)", "720p (1.00x)"]
    remap_t = [0.3, 0.8, 4.5]
    sgbm_l_t = [5.0, 23.7, 167.5]
    sgbm_r_t = [4.1, 15.5, 160.4]
    wls_t = [4.0, 10.8, 54.1]
    depth_t = [0.4, 2.3, 11.3]

    x_idx = np.arange(len(scales))
    w = 0.55
    b1 = np.array(remap_t)
    b2 = b1 + np.array(sgbm_l_t)
    b3 = b2 + np.array(sgbm_r_t)
    b4 = b3 + np.array(wls_t)

    ax_c.bar(x_idx, remap_t, w, label="Remap", color="wheat")
    ax_c.bar(x_idx, sgbm_l_t, w, bottom=b1, label="SGBM Left (40%)", color="cornflowerblue")
    ax_c.bar(x_idx, sgbm_r_t, w, bottom=b2, label="SGBM Right (39%)", color="royalblue")
    ax_c.bar(x_idx, wls_t, w, bottom=b3, label="WLS Filter (13%)", color="mediumseagreen")
    ax_c.bar(x_idx, depth_t, w, bottom=b4, label="Depth/Reproj (3%)", color="salmon")

    ax_c.set_xticks(x_idx)
    ax_c.set_xticklabels(scales, fontweight="bold")
    ax_c.set_ylabel("Execution Time (ms)", fontweight="bold")
    ax_c.set_title("(c) Multiscale Latency Breakdown (7.8x Speedup at 360p)", fontweight="bold")
    ax_c.legend(loc="upper left", frameon=True, fontsize=8.5)
    ax_c.grid(True, axis="y", linestyle=":", alpha=0.6)

    # Subplot (d): Thread Scaling Plateau
    ax_d = axs[1, 1]
    threads = [1, 2, 4, 8, 16]
    lat_threads = [915.1, 1519.4, 1107.9, 797.1, 760.1]
    ax_d.plot(threads, lat_threads, marker="o", color="navy", lw=2.2, label="Full 720p Pipeline")
    ax_d.scatter([16], [760.1], color="crimson", s=90, zorder=5)
    ax_d.annotate("Contention Plateau\n(8-16 Threads: 760 ms)", (16, 760.1), xytext=(8.5, 950),
                  arrowprops=dict(arrowstyle="->", color="crimson", lw=1.2))

    ax_d.set_xlabel("OpenCV Thread Count", fontweight="bold")
    ax_d.set_ylabel("Wall Clock Latency (ms)", fontweight="bold")
    ax_d.set_title("(d) CPU Thread Scaling and Cache Contention", fontweight="bold")
    ax_d.set_xticks(threads)
    ax_d.grid(True, linestyle=":", alpha=0.6)
    ax_d.legend(loc="upper right", frameon=True)

    plt.savefig(assets_dir / "fig04_cpu_tradeoffs.png", dpi=300)
    plt.savefig(assets_dir / "fig04_cpu_tradeoffs.pdf")
    plt.savefig(assets_dir / "fig04_optimization.png", dpi=250)
    plt.close()
    print("Saved Figure 4.")


def generate_fig05_four_terrains(
    images_rect: dict,
    opt_matcher: cv2.StereoSGBM
) -> None:
    """Generate Figure 5: Four-Terrain Evaluation and Failure Analysis."""
    print("Generating Figure 5: Four-Terrain Evaluation...")
    terrains = ["puncakTinggi", "puncakWarung", "tanggaBiru", "tanggaKuning"]
    fig, axs = plt.subplots(2, 2, figsize=(13.5, 8.2), constrained_layout=True)

    terrain_annotations = [
        ("(a) puncakTinggi: Summit Ridge",
         "RealSense Depth: 0.063 m Med Err | 84.4% Delta<1.25",
         "Cov: 61.3% | Failure: Overexposed Saturated Sky Rejection"),
        ("(b) puncakWarung: Roadside Shelter",
         "RealSense Depth: Omitted (>15 m Active IR Range Limit)",
         "Cov: 56.5% | Failure: Shadow Occlusions Under Tin Roof"),
        ("(c) tanggaBiru: Blue Stairway",
         "RealSense Depth: 0.225 m Med Err | 59.3% Delta<1.25",
         "Cov: 48.3% | Failure: Repeated Step Edges & Slender Railing"),
        ("(d) tanggaKuning: Yellow Stairway",
         "RealSense Depth: 0.094 m Med Err | 86.3% Delta<1.25",
         "Cov: 55.5% | Failure: Periodic Metal Bars & Steep Slope")
    ]

    for idx, (t, (m_title, rs_sub, cov_sub)) in enumerate(zip(terrains, terrain_annotations)):
        ax = axs[idx // 2, idx % 2]
        rect_l, rect_r = images_rect[t]
        d_opt, _ = stereo.compute_disparity_wls(opt_matcher, rect_l, rect_r)
        d_plot = np.where(d_opt > 0, d_opt, np.nan)
        im_t = ax.imshow(d_plot, cmap="turbo", vmin=0, vmax=160)
        ax.set_title(f"{m_title}\n{rs_sub}\n[{cov_sub}]", fontsize=10.0, fontweight="bold", pad=5)
        ax.axis("off")
        cbar = fig.colorbar(im_t, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label("Disparity (px)", fontweight="bold")

    plt.savefig(assets_dir / "fig05_four_terrains.png", dpi=300)
    plt.savefig(assets_dir / "fig05_four_terrains.pdf")
    plt.savefig(assets_dir / "fig05_terrains.png", dpi=250)
    plt.close()
    print("Saved Figure 5.")


def generate_hero_loop_gif(images_rect: dict, opt_matcher: cv2.StereoSGBM) -> None:
    """Generate hero looping GIF assets/stereo_loop.gif cycling across 4 terrains."""
    print("Generating Hero Looping GIF: assets/stereo_loop.gif...")
    terrains = ["puncakTinggi", "puncakWarung", "tanggaBiru", "tanggaKuning"]
    gif_frames = []
    fps = 10
    frames_per_terrain = 24
    colormap = plt.get_cmap("turbo")

    for t in terrains:
        rect_l, rect_r = images_rect[t]
        # Fast baseline
        d_fast, _ = stereo.run_stereo_pipeline(rect_l, rect_r, preset="FAST")
        d_norm_fast = np.clip(d_fast / 160.0, 0, 1)
        d_col_fast = (colormap(d_norm_fast)[:, :, :3] * 255).astype(np.uint8)
        d_col_fast[d_fast <= 0] = [30, 30, 30]

        # Balanced WLS reconstruction
        d_bal, _ = stereo.run_stereo_pipeline(rect_l, rect_r, preset="BALANCED")
        d_norm_bal = np.clip(d_bal / 160.0, 0, 1)
        d_col_bal = (colormap(d_norm_bal)[:, :, :3] * 255).astype(np.uint8)
        d_col_bal[d_bal <= 0] = [30, 30, 30]

        w_sub, h_sub = 384, 216
        im_sub = cv2.resize(cv2.cvtColor(rect_l, cv2.COLOR_BGR2RGB), (w_sub, h_sub))
        df_sub = cv2.resize(d_col_fast, (w_sub, h_sub))
        db_sub = cv2.resize(d_col_bal, (w_sub, h_sub))

        canvas = np.zeros((h_sub + 44, w_sub * 3, 3), dtype=np.uint8)
        canvas[44:, :w_sub] = im_sub
        canvas[44:, w_sub:2*w_sub] = df_sub
        canvas[44:, 2*w_sub:] = db_sub
        canvas[:44, :] = [20, 24, 32]

        cv2.putText(canvas, f"Terrain: {t}", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(canvas, "1. Rectified Input", (15, h_sub + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        cv2.putText(canvas, "2. FAST SGBM (34 ms)", (w_sub + 15, h_sub + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 220, 240), 1)
        cv2.putText(canvas, "3. BALANCED WLS (67 ms)", (2*w_sub + 15, h_sub + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 240, 120), 1)

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
    gif_frames[0].save(
        assets_dir / "terrain_loop.gif",
        save_all=True,
        append_images=gif_frames[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True
    )
    print(f"Saved {gif_path} and terrain_loop.gif")


def generate_all_visuals(
    calib_path: str = DEFAULT_CALIB,
    gunung_root: str = DEFAULT_GUNUNG,
    mid_root: str = DEFAULT_MID
) -> None:
    """Run visual generation pipeline."""
    print("Loading calibration and datasets...")
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

    # Middlebury Adirondack
    mid_p = Path(mid_root) / "Adirondack"
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

    # Compute models for Adirondack and puncakTinggi
    r_matcher = cv2.ximgproc.createRightMatcher(opt_matcher)
    mid_dl16 = opt_matcher.compute(mid_im0, mid_im1)
    mid_dr16 = r_matcher.compute(mid_im1, mid_im0)
    mid_raw = mid_dl16.astype(np.float32) / 16.0
    mid_r_raw = mid_dr16.astype(np.float32) / 16.0
    mid_disp_opt, _ = stereo.compute_disparity_wls(opt_matcher, mid_im0, mid_im1)

    gun_dl16 = opt_matcher.compute(pt_rect_l, pt_rect_r)
    gun_dr16 = r_matcher.compute(pt_rect_r, pt_rect_l)
    gun_raw = gun_dl16.astype(np.float32) / 16.0
    gun_r_raw = gun_dr16.astype(np.float32) / 16.0
    gun_disp_opt, _ = stereo.compute_disparity_wls(opt_matcher, pt_rect_l, pt_rect_r)
    gun_depth_opt = stereo.disparity_to_depth(gun_disp_opt, K_RECT[0, 0], 0.097497)

    # 1. Figure 1: Stereo Geometry and Calibrated Pipeline
    raw_pt_l, raw_pt_r = images_raw["puncakTinggi"]
    generate_fig01_stereo_geometry(raw_pt_l, raw_pt_r, pt_rect_l, pt_rect_r)

    # 2. Figure 2: Cross-Dataset Middlebury vs Gunung Validation
    generate_fig02_middlebury_gunung(
        mid_im0, mid_gt, mid_disp_opt, mid_mask,
        pt_rect_l, pt_rs_ref_depth, gun_depth_opt
    )

    # 3. Figure 3: Filter Comparison Suite
    generate_fig03_filters(
        mid_im0, mid_raw, mid_r_raw, opt_matcher,
        pt_rect_l, gun_raw, gun_r_raw
    )

    # 4. Figure 4: Accuracy-Coverage-CPU Pareto Frontier
    generate_fig04_cpu_tradeoffs()

    # 5. Figure 5: Four-Terrain Evaluation
    generate_fig05_four_terrains(images_rect, opt_matcher)

    # 6. Hero Looping GIF
    generate_hero_loop_gif(images_rect, opt_matcher)

    print("All visuals generated successfully.")


if __name__ == "__main__":
    generate_all_visuals()
