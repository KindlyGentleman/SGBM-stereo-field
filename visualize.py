"""Scientific visualization suite for stereo matching and tiny CNN experiments.

Generates IEEE-style publication figures and animations:
- fig01_stereo_geometry: Vector illustration of rectified stereo geometry and depth.
- fig02_classical_matching: Layered scene matching (BM vs SGBM vs GT vs Error + SAD profile).
- fig03_photometric_robustness: Reconstruction accuracy and coverage under perturbations.
- fig04_cnn_architecture: Educational diagram of TinyStereoNet and convolution mechanics.
- fig05_cnn_learning: Training loss and accuracy progression vs classical baseline.
- fig06_classical_vs_cnn: Patch-level comparison, disparity scan, and confusion matrices.
- stereo_loop.gif: Looping animation of stereo reconstruction across changing conditions.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import stereo
import cnn

# Styling defaults
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "figure.titlesize": 11,
    "lines.linewidth": 1.2,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

COLOR_BM = "#4575b4"       # Blue
COLOR_SGBM = "#d73027"     # Red
COLOR_SGBM_CLAHE = "#2ca02c"  # Green
COLOR_CNN = "#7570b3"      # Purple
COLOR_BASELINE = "#e7298a" # Magenta


def generate_figure_01_geometry(output_dir: Path) -> None:
    """Generate Figure 1: Rectified stereo camera geometry and metric depth."""
    fig, ax = plt.subplots(figsize=(7.16, 4.4), dpi=300)
    ax.set_xlim(-2.0, 9.0)
    ax.set_ylim(-1.0, 7.5)
    ax.axis("off")

    # Optical centers
    ol_x, ol_y = 0.0, 0.0
    or_x, or_y = 5.0, 0.0

    # Baseline
    ax.plot([ol_x, or_x], [ol_y, or_y], "k-", lw=1.8)
    ax.plot(ol_x, ol_y, "ks", markersize=7)
    ax.plot(or_x, or_y, "ks", markersize=7)
    ax.text(ol_x, -0.45, r"$\mathbf{O}_L\,(0, 0)$" + "\nLeft Camera", ha="center", fontsize=9.5)
    ax.text(or_x, -0.45, r"$\mathbf{O}_R\,(B, 0)$" + "\nRight Camera", ha="center", fontsize=9.5)
    ax.annotate("", xy=(or_x, -0.15), xytext=(ol_x, -0.15),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.2))
    ax.text(2.5, -0.32, r"Baseline $B$", ha="center", fontsize=9.5, fontweight="bold")

    # Optical axes
    ax.plot([ol_x, ol_x], [ol_y, 6.5], "k--", lw=0.9, alpha=0.5)
    ax.plot([or_x, or_x], [or_y, 6.5], "k--", lw=0.9, alpha=0.5)

    # Image planes at Z = f (f_y = 1.6)
    f_y = 1.6
    ax.plot([-1.1, 1.1], [f_y, f_y], color="#1f77b4", lw=2.4)
    ax.plot([3.9, 6.1], [f_y, f_y], color="#1f77b4", lw=2.4)
    ax.text(0.0, f_y + 0.18, "Left Image Plane", ha="center", fontsize=9, color="#1f77b4", fontweight="bold")
    ax.text(5.0, f_y + 0.18, "Right Image Plane", ha="center", fontsize=9, color="#1f77b4", fontweight="bold")

    # Scene point P = (1.8, 5.8)
    px, py = 1.8, 5.8
    ax.plot(px, py, "ro", markersize=8)
    ax.text(px + 0.18, py + 0.15, r"Scene Point $P(X, Y, Z)$", fontsize=10, fontweight="bold", color="darkred")

    # Projection rays from optical centers through image planes to P
    ax.plot([ol_x, px], [ol_y, py], "r-", lw=1.3, alpha=0.85)
    ax.plot([or_x, px], [or_y, py], "r-", lw=1.3, alpha=0.85)

    # Projected image coordinates
    xl_proj = (f_y / py) * px
    xr_proj = or_x + (f_y / py) * (px - or_x)

    ax.plot(xl_proj, f_y, "bo", markersize=6)
    ax.plot(xr_proj, f_y, "bo", markersize=6)
    ax.text(xl_proj, f_y - 0.35, r"$x_L$", ha="center", fontsize=9.5, color="darkblue", fontweight="bold")
    ax.text(xr_proj, f_y - 0.35, r"$x_R$", ha="center", fontsize=9.5, color="darkblue", fontweight="bold")

    # Projected displacement (disparity d)
    ax.plot([xl_proj, xl_proj], [f_y, f_y - 0.55], "b:", lw=0.8)
    ax.plot([xr_proj, xr_proj], [f_y, f_y - 0.55], "b:", lw=0.8)

    # Focal length f bracket
    ax.annotate("", xy=(6.5, f_y), xytext=(6.5, 0.0),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.1))
    ax.text(6.65, f_y / 2.0, r"Focal length $f$", va="center", ha="left", fontsize=9)

    # Depth Z bracket
    ax.annotate("", xy=(-1.5, py), xytext=(-1.5, 0.0),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.1))
    ax.text(-1.65, py / 2.0, r"Metric Depth $Z$", va="center", ha="right", fontsize=9.5, rotation=90)

    # Equations box
    box_lines = [
        r"$\mathbf{Rectified\,Scanline\,Geometry}$",
        r"Horizontal correspondence: $y_L = y_R$",
        r"Disparity: $d = x_L - x_R$",
        r"Triangulation: $Z = \frac{f \cdot B}{d}$"
    ]
    ax.text(2.5, 3.2, "\n".join(box_lines), ha="center", va="center", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa", edgecolor="#b0bec5", lw=1.2))

    fig.savefig(output_dir / "fig01_stereo_geometry.png")
    fig.savefig(output_dir / "fig01_stereo_geometry.pdf")
    plt.close(fig)
    print("Generated Figure 1: fig01_stereo_geometry.png/.pdf")


def generate_figure_02_classical(output_dir: Path, data: Dict[str, Any], sad_data: Dict[str, Any]) -> None:
    """Generate Figure 2: Layered scene classical stereo matching and SAD profile."""
    fig = plt.figure(figsize=(7.16, 5.2), dpi=300)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.9], hspace=0.32, wspace=0.25)

    left = data["layered_left"]
    right = data["layered_right_clean"]
    gt = data["layered_gt"]
    mask = data["layered_mask"]
    disp_bm = data["layered_clean_stereobm_disp"]
    disp_sgbm = data["layered_clean_stereosgbm_disp"]

    # Mask invalid regions for GT
    gt_masked = np.where(mask, gt, np.nan)
    err_sgbm = np.where(mask & (disp_sgbm > 0), np.abs(disp_sgbm - gt), np.nan)

    # Shared disparity scale [0, 35]
    vmin, vmax = 0.0, 35.0

    # Panel (a): Left image
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.imshow(left, cmap="gray")
    ax_a.set_title("(a) Left View $I_L$", fontsize=9.5)
    ax_a.axis("off")
    # Mark reference point for SAD profile
    rx, ry = sad_data["ref_x"], sad_data["ref_y"]
    ax_a.plot(rx, ry, "r+", markersize=8, markeredgewidth=1.8)
    ax_a.text(rx + 8, ry - 8, f"P({rx},{ry})", color="red", fontsize=8, fontweight="bold")

    # Panel (b): Right image
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.imshow(right, cmap="gray")
    ax_b.set_title("(b) Right View $I_R$", fontsize=9.5)
    ax_b.axis("off")

    # Panel (c): Ground truth disparity
    ax_c = fig.add_subplot(gs[0, 2])
    im_c = ax_c.imshow(gt_masked, cmap="viridis", vmin=vmin, vmax=vmax)
    ax_c.set_title(r"(c) Reference Disparity $d_{\mathrm{gt}}$", fontsize=9.5)
    ax_c.axis("off")

    # Panel (d): StereoBM disparity
    ax_d = fig.add_subplot(gs[1, 0])
    disp_bm_plot = np.where(disp_bm > 0, disp_bm, np.nan)
    ax_d.imshow(disp_bm_plot, cmap="viridis", vmin=vmin, vmax=vmax)
    ax_d.set_title("(d) StereoBM Disparity", fontsize=9.5)
    ax_d.axis("off")

    # Panel (e): StereoSGBM disparity
    ax_e = fig.add_subplot(gs[1, 1])
    disp_sgbm_plot = np.where(disp_sgbm > 0, disp_sgbm, np.nan)
    ax_e.imshow(disp_sgbm_plot, cmap="viridis", vmin=vmin, vmax=vmax)
    ax_e.set_title("(e) StereoSGBM Disparity", fontsize=9.5)
    ax_e.axis("off")

    # Panel (f): StereoSGBM absolute error map
    ax_f = fig.add_subplot(gs[1, 2])
    im_f = ax_f.imshow(err_sgbm, cmap="magma", vmin=0.0, vmax=5.0)
    ax_f.set_title(r"(f) Spatial Error $|\hat{d} - d_{\mathrm{gt}}|$", fontsize=9.5)
    ax_f.axis("off")

    # Colorbars
    cb_disp = fig.colorbar(im_c, ax=[ax_c, ax_d, ax_e], orientation="vertical", fraction=0.02, pad=0.03)
    cb_disp.set_label("Disparity (px)", fontsize=8.5)

    cb_err = fig.colorbar(im_f, ax=ax_f, orientation="vertical", fraction=0.046, pad=0.04)
    cb_err.set_label("Error (px)", fontsize=8.5)

    # Panel (g): Educational SAD scanline profile across bottom span
    ax_g = fig.add_subplot(gs[2, :])
    disps = np.array(sad_data["disparities"])
    costs = np.array(sad_data["costs"])
    true_d = sad_data["true_disparity"]
    best_d = sad_data["best_disparity"]

    valid_c = [c for c in costs if c is not None]
    ax_g.plot(disps, costs, color="#2b5c8f", lw=1.6, label="Sum of Absolute Differences (SAD)")
    ax_g.axvline(true_d, color="red", linestyle="--", lw=1.4, label=f"True Disparity ($d={true_d}$ px)")
    ax_g.plot(best_d, sad_data["min_cost"], "ro", markersize=6, label=f"SAD Minimum ($d^*={best_d}$ px)")

    ax_g.set_xlabel("Candidate Disparity $d$ (pixels)", fontsize=9.5)
    ax_g.set_ylabel("SAD Matching Cost", fontsize=9.5)
    ax_g.set_title(f"(g) Educational SAD Cost Profile for Pixel ({rx}, {ry}) along Scanline", fontsize=9.5)
    ax_g.grid(True, linestyle=":", alpha=0.6)
    ax_g.legend(loc="upper right", frameon=True, fontsize=8.5)
    ax_g.set_xlim(0, 48)

    fig.savefig(output_dir / "fig02_classical_matching.png")
    fig.savefig(output_dir / "fig02_classical_matching.pdf")
    plt.close(fig)
    print("Generated Figure 2: fig02_classical_matching.png/.pdf")


def generate_figure_03_robustness(output_dir: Path, metrics_df: pd.DataFrame) -> None:
    """Generate Figure 3: Photometric robustness comparison under controlled perturbations."""
    df_layer = metrics_df[metrics_df["scene"] == "layered"].copy()

    pert_order = ["clean", "brightness_up", "brightness_down", "contrast_low", "gaussian_noise", "gamma_high"]
    pert_labels = ["Clean", "Brightness +30", "Brightness -30", "Contrast 0.5x", "Gaussian Noise", "Gamma 1.6"]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.16, 2.7), dpi=300)

    x = np.arange(len(pert_order))
    width = 0.26

    # Extract metrics for each matcher
    bm_bad = [df_layer[(df_layer["perturbation"] == p) & (df_layer["matcher"] == "StereoBM")]["bad_2.0_pct"].values[0] for p in pert_order]
    sgbm_bad = [df_layer[(df_layer["perturbation"] == p) & (df_layer["matcher"] == "StereoSGBM")]["bad_2.0_pct"].values[0] for p in pert_order]
    clahe_bad = [df_layer[(df_layer["perturbation"] == p) & (df_layer["matcher"] == "StereoSGBM+CLAHE")]["bad_2.0_pct"].values[0] for p in pert_order]

    bm_cov = [df_layer[(df_layer["perturbation"] == p) & (df_layer["matcher"] == "StereoBM")]["coverage_pct"].values[0] for p in pert_order]
    sgbm_cov = [df_layer[(df_layer["perturbation"] == p) & (df_layer["matcher"] == "StereoSGBM")]["coverage_pct"].values[0] for p in pert_order]
    clahe_cov = [df_layer[(df_layer["perturbation"] == p) & (df_layer["matcher"] == "StereoSGBM+CLAHE")]["coverage_pct"].values[0] for p in pert_order]

    bm_mae = [df_layer[(df_layer["perturbation"] == p) & (df_layer["matcher"] == "StereoBM")]["mae_px"].values[0] for p in pert_order]
    sgbm_mae = [df_layer[(df_layer["perturbation"] == p) & (df_layer["matcher"] == "StereoSGBM")]["mae_px"].values[0] for p in pert_order]
    clahe_mae = [df_layer[(df_layer["perturbation"] == p) & (df_layer["matcher"] == "StereoSGBM+CLAHE")]["mae_px"].values[0] for p in pert_order]

    # Subplot 1: Bad@2.0 error rate
    ax1.bar(x - width, bm_bad, width, label="StereoBM", color=COLOR_BM, alpha=0.9)
    ax1.bar(x, sgbm_bad, width, label="StereoSGBM", color=COLOR_SGBM, alpha=0.9)
    ax1.bar(x + width, clahe_bad, width, label="SGBM+CLAHE", color=COLOR_SGBM_CLAHE, alpha=0.9)
    ax1.set_ylabel("Bad@2.0 Error Rate (%)", fontsize=9)
    ax1.set_title("(a) Disparity Error (Lower is better)", fontsize=9.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(pert_labels, rotation=35, ha="right", fontsize=8)
    ax1.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax1.legend(loc="upper left", frameon=True, fontsize=7.5)

    # Subplot 2: Valid coverage %
    ax2.bar(x - width, bm_cov, width, color=COLOR_BM, alpha=0.9)
    ax2.bar(x, sgbm_cov, width, color=COLOR_SGBM, alpha=0.9)
    ax2.bar(x + width, clahe_cov, width, color=COLOR_SGBM_CLAHE, alpha=0.9)
    ax2.set_ylabel("Valid Coverage (%)", fontsize=9)
    ax2.set_title("(b) Match Coverage (Higher is better)", fontsize=9.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(pert_labels, rotation=35, ha="right", fontsize=8)
    ax2.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax2.set_ylim(0, 100)

    # Subplot 3: MAE in pixels
    ax3.plot(x, bm_mae, marker="o", color=COLOR_BM, label="StereoBM", lw=1.4)
    ax3.plot(x, sgbm_mae, marker="s", color=COLOR_SGBM, label="StereoSGBM", lw=1.4)
    ax3.plot(x, clahe_mae, marker="^", color=COLOR_SGBM_CLAHE, label="SGBM+CLAHE", lw=1.4)
    ax3.set_ylabel("MAE (pixels)", fontsize=9)
    ax3.set_title("(c) Mean Absolute Error", fontsize=9.5)
    ax3.set_xticks(x)
    ax3.set_xticklabels(pert_labels, rotation=35, ha="right", fontsize=8)
    ax3.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    fig.savefig(output_dir / "fig03_photometric_robustness.png")
    fig.savefig(output_dir / "fig03_photometric_robustness.pdf")
    plt.close(fig)
    print("Generated Figure 3: fig03_photometric_robustness.png/.pdf")


def generate_figure_04_cnn_architecture(output_dir: Path) -> None:
    """Generate Figure 4: Educational diagram of TinyStereoNet and convolution mechanics."""
    fig, ax = plt.subplots(figsize=(7.16, 3.8), dpi=300)
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 5.0)
    ax.axis("off")

    # Helper to draw colored boxes
    def draw_box(x, y, w, h, title, subtitle, color="#e3f2fd", border="#1565c0"):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                                      facecolor=color, edgecolor=border, lw=1.2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2 + 0.15, title, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color="#0d47a1")
        ax.text(x + w / 2, y + h / 2 - 0.22, subtitle, ha="center", va="center",
                fontsize=7.5, color="#37474f")

    # 1. Input Patches
    draw_box(0.0, 1.8, 1.4, 1.2, "Input Patches", "2 x 15 x 15\n(Left + Right)", color="#f5f5f5", border="#757575")

    # Arrow 1
    ax.annotate("", xy=(1.9, 2.4), xytext=(1.45, 2.4), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.text(1.68, 2.65, "Conv 3x3\npad=1", ha="center", fontsize=7.5)

    # 2. Conv1 Feature Maps
    draw_box(2.0, 1.7, 1.5, 1.4, "Conv1 + ReLU", "8 maps\n8 x 15 x 15", color="#e8f5e9", border="#2e7d32")

    # Arrow 2
    ax.annotate("", xy=(4.0, 2.4), xytext=(3.55, 2.4), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.text(3.78, 2.65, "Conv 3x3\npad=1", ha="center", fontsize=7.5)

    # 3. Conv2 Feature Maps
    draw_box(4.1, 1.6, 1.5, 1.6, "Conv2 + ReLU", "12 maps\n12 x 15 x 15", color="#fff3e0", border="#ef6c00")

    # Arrow 3
    ax.annotate("", xy=(6.1, 2.4), xytext=(5.65, 2.4), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.text(5.88, 2.65, "Global Avg\nPooling", ha="center", fontsize=7.5)

    # 4. Global Average Pooling
    draw_box(6.2, 1.8, 1.3, 1.2, "GAP", "12 x 1 x 1\n(Vector)", color="#ede7f6", border="#512da8")

    # Arrow 4
    ax.annotate("", xy=(8.0, 2.4), xytext=(7.55, 2.4), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.text(7.78, 2.65, "Linear\n(12 -> 2)", ha="center", fontsize=7.5)

    # 5. Output Logits
    draw_box(8.1, 1.8, 1.6, 1.2, "Output Logits", "Match: P(1)\nNonmatch: P(0)", color="#fce4ec", border="#c2185b")

    # Convolution mechanics callout below
    callout_box = patches.FancyBboxPatch((0.2, 0.1), 9.6, 1.2, boxstyle="round,pad=0.15",
                                         facecolor="#fafafa", edgecolor="#cfd8dc", lw=1.0)
    ax.add_patch(callout_box)
    desc_lines = [
        r"$\mathbf{Total\,Model\,Parameters:\,1,054}$ (Conv1: 152, Conv2: 876, Linear: 26).",
        r"$\mathbf{2D\,Convolution\,Mechanics:}\quad (I * K)(u, v) = \sum_{c=0}^1 \sum_{i=-1}^1 \sum_{j=-1}^1 I_c(u+i, v+j) \cdot K_c(i, j) + b$",
        r"The 2 input channels allow learned kernels to compute spatial cross-correlation between dual patches."
    ]
    ax.text(0.4, 0.7, "\n".join(desc_lines), fontsize=8.2, va="center")

    fig.savefig(output_dir / "fig04_cnn_architecture.png")
    fig.savefig(output_dir / "fig04_cnn_architecture.pdf")
    plt.close(fig)
    print("Generated Figure 4: fig04_cnn_architecture.png/.pdf")


def generate_figure_05_cnn_learning(output_dir: Path, cnn_history: pd.DataFrame, cnn_metrics: Dict[str, Any]) -> None:
    """Generate Figure 5: CNN training loss and validation accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.0), dpi=300)

    epochs = cnn_history["epoch"].values
    tr_loss = cnn_history["train_loss"].values
    val_loss = cnn_history["val_loss"].values
    tr_acc = cnn_history["train_acc"].values
    val_acc = cnn_history["val_acc"].values

    base_acc = cnn_metrics["classical_baseline"]["accuracy"]
    test_acc = cnn_metrics["tiny_cnn"]["accuracy"]
    n_params = cnn_metrics["model_parameters"]

    # Panel (a): Loss curves
    ax1.plot(epochs, tr_loss, "b-o", markersize=4, lw=1.4, label="Training Loss")
    ax1.plot(epochs, val_loss, "r--s", markersize=4, lw=1.4, label="Validation Loss")
    ax1.set_xlabel("Epoch", fontsize=9.5)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=9.5)
    ax1.set_title("(a) Loss Progression across Epochs", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True, fontsize=8.5)
    ax1.set_xticks(epochs[::2] if len(epochs) > 8 else epochs)

    # Panel (b): Accuracy curves
    ax2.plot(epochs, tr_acc, "b-o", markersize=4, lw=1.4, label="Training Accuracy")
    ax2.plot(epochs, val_acc, "r--s", markersize=4, lw=1.4, label="Validation Accuracy")
    ax2.axhline(base_acc, color=COLOR_BASELINE, linestyle=":", lw=1.6,
                label=f"Classical MAD Baseline ({base_acc:.1f}%)")

    # Annotation of final held-out test score
    ax2.annotate(
        f"Final Held-Out Test:\nCNN: {test_acc:.1f}% ({n_params} params)",
        xy=(epochs[-1], val_acc[-1]),
        xytext=(epochs[-4] if len(epochs) >= 4 else 1, val_acc[-1] - 12),
        arrowprops=dict(arrowstyle="->", lw=1.1, color="black"),
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff9c4", edgecolor="#fbc02d", lw=1.0)
    )

    ax2.set_xlabel("Epoch", fontsize=9.5)
    ax2.set_ylabel("Accuracy (%)", fontsize=9.5)
    ax2.set_title("(b) Classification Accuracy vs Classical Baseline", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right", frameon=True, fontsize=8.5)
    ax2.set_ylim(40, 102)
    ax2.set_xticks(epochs[::2] if len(epochs) > 8 else epochs)

    plt.tight_layout()
    fig.savefig(output_dir / "fig05_cnn_learning.png")
    fig.savefig(output_dir / "fig05_cnn_learning.pdf")
    plt.close(fig)
    print("Generated Figure 5: fig05_cnn_learning.png/.pdf")


def generate_figure_06_comparison(
    output_dir: Path,
    cnn_metrics: Dict[str, Any],
    sample_patches_path: Path
) -> None:
    """Generate Figure 6: Classical MAD vs Tiny CNN patch matching comparison."""
    fig = plt.figure(figsize=(7.16, 5.0), dpi=300)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.2], hspace=0.35, wspace=0.28)

    # 1. Sample patches visualization
    samples = np.load(str(sample_patches_path))
    x_test = samples["x_test"]
    y_test = samples["y_test"]

    # Find one positive and one negative sample
    pos_idx = np.where(y_test == 1)[0][0]
    neg_idx = np.where(y_test == 0)[0][0]

    p_pos_l = x_test[pos_idx, 0]
    p_pos_r = x_test[pos_idx, 1]
    diff_pos = np.abs(p_pos_l - p_pos_r)

    p_neg_l = x_test[neg_idx, 0]
    p_neg_r = x_test[neg_idx, 1]
    diff_neg = np.abs(p_neg_l - p_neg_r)

    # Subplot (a): Patch pair comparisons
    ax_a = fig.add_subplot(gs[0, 0])
    # Composite 3x2 image for patches
    row1 = np.hstack([p_pos_l, p_pos_r, diff_pos])
    row2 = np.hstack([p_neg_l, p_neg_r, diff_neg])
    grid = np.vstack([row1, row2])
    ax_a.imshow(grid, cmap="gray")
    ax_a.set_title("(a) Patch Pairs: Left | Right | |Diff|\nTop: Match (1), Bottom: Nonmatch (0)", fontsize=9)
    ax_a.axis("off")

    # Subplot (b): Candidate Disparity Sweep for a test patch
    # Load model and sweep
    ax_b = fig.add_subplot(gs[0, 1])
    try:
        import torch
        model = cnn.TinyStereoNet()
        model.load_state_dict(torch.load("results/tiny_cnn.pt", weights_only=True))
        l_img, r_img, gt_img, _ = stereo.generate_layered_synthetic_scene(320, 240, seed=300)
        sweep = cnn.sweep_patch_candidate_disparities(model, l_img, r_img, x0=110, y0=100, max_disp=48)
        d_arr = sweep["disparities"]
        cnn_p = sweep["cnn_probs"]
        sad_c = sweep["sad_costs"]

        # Dual axis: CNN probability (left) and SAD cost (right)
        ax_b.plot(d_arr, cnn_p, color=COLOR_CNN, lw=1.6, label="CNN Match Probability")
        ax_b.axvline(32, color="red", linestyle="--", lw=1.2, label="True Disparity (32 px)")
        ax_b.set_xlabel("Candidate Disparity $d$ (pixels)", fontsize=9)
        ax_b.set_ylabel("CNN Probability $P(1)$", color=COLOR_CNN, fontsize=9)
        ax_b.tick_params(axis="y", labelcolor=COLOR_CNN)
        ax_b.set_ylim(-0.05, 1.05)

        ax_b_right = ax_b.twinx()
        ax_b_right.plot(d_arr, sad_c, color="#666666", linestyle=":", lw=1.3, label="SAD Cost")
        ax_b_right.set_ylabel("SAD Matching Cost", color="#666666", fontsize=9)
        ax_b_right.tick_params(axis="y", labelcolor="#666666")

        ax_b.set_title("(b) Disparity Sweep: CNN Confidence vs SAD", fontsize=9)
        ax_b.grid(True, linestyle=":", alpha=0.5)
    except Exception as e:
        ax_b.text(0.5, 0.5, f"Sweep plot: {e}", ha="center", va="center", fontsize=8)

    # Subplot (c): Confusion Matrices
    ax_c = fig.add_subplot(gs[1, 0])
    cm_base = cnn_metrics["classical_baseline"]["confusion_matrix"]
    cm_cnn = cnn_metrics["tiny_cnn"]["confusion_matrix"]

    mat_base = np.array([[cm_base["TP"], cm_base["FN"]], [cm_base["FP"], cm_base["TN"]]])
    im_cm = ax_c.imshow(mat_base, cmap="Blues", alpha=0.85)
    ax_c.set_xticks([0, 1])
    ax_c.set_yticks([0, 1])
    ax_c.set_xticklabels(["Pred Match", "Pred Nonmatch"], fontsize=8.5)
    ax_c.set_yticklabels(["True Match", "True Nonmatch"], fontsize=8.5)
    for i in range(2):
        for j in range(2):
            ax_c.text(j, i, str(mat_base[i, j]), ha="center", va="center", fontsize=11, fontweight="bold")
    ax_c.set_title(f"(c) Classical MAD Confusion Matrix\nAccuracy = {cnn_metrics['classical_baseline']['accuracy']:.1f}%", fontsize=9)

    ax_d = fig.add_subplot(gs[1, 1])
    mat_cnn = np.array([[cm_cnn["TP"], cm_cnn["FN"]], [cm_cnn["FP"], cm_cnn["TN"]]])
    im_cm2 = ax_d.imshow(mat_cnn, cmap="Purples", alpha=0.85)
    ax_d.set_xticks([0, 1])
    ax_d.set_yticks([0, 1])
    ax_d.set_xticklabels(["Pred Match", "Pred Nonmatch"], fontsize=8.5)
    ax_d.set_yticklabels(["True Match", "True Nonmatch"], fontsize=8.5)
    for i in range(2):
        for j in range(2):
            ax_d.text(j, i, str(mat_cnn[i, j]), ha="center", va="center", fontsize=11, fontweight="bold")
    ax_d.set_title(f"(d) Tiny CNN Confusion Matrix\nAccuracy = {cnn_metrics['tiny_cnn']['accuracy']:.1f}%", fontsize=9)

    fig.savefig(output_dir / "fig06_classical_vs_cnn.png")
    fig.savefig(output_dir / "fig06_classical_vs_cnn.pdf")
    plt.close(fig)
    print("Generated Figure 6: fig06_classical_vs_cnn.png/.pdf")


def generate_stereo_loop_gif(output_dir: Path, data: Dict[str, Any], metrics_df: pd.DataFrame) -> None:
    """Generate hero stereo looping GIF from evaluated conditions."""
    conditions = [
        ("clean", "Condition: Clean Baseline"),
        ("brightness_up", "Condition: Brightness (+30)"),
        ("contrast_low", "Condition: Low Contrast (0.5x)"),
        ("gaussian_noise", "Condition: Gaussian Noise (\u03c3=15)"),
        ("gamma_high", "Condition: Non-linear Gamma (1.6)")
    ]

    left = data["layered_left"]
    df_layer = metrics_df[metrics_df["scene"] == "layered"]

    frames: List[Image.Image] = []

    for pert, title in conditions:
        right = data.get(f"layered_right_{pert}", data["layered_right_clean"])
        disp_key = f"layered_{pert}_stereosgbm_disp"
        disp = data.get(disp_key, data["layered_clean_stereosgbm_disp"])

        row = df_layer[(df_layer["perturbation"] == pert) & (df_layer["matcher"] == "StereoSGBM")]
        bad_rate = row["bad_2.0_pct"].values[0] if len(row) > 0 else 0.0
        cov = row["coverage_pct"].values[0] if len(row) > 0 else 0.0

        # Create colorized disparity using viridis
        disp_clean = np.where(disp > 0, disp, 0.0)
        disp_norm = np.clip(disp_clean / 35.0 * 255.0, 0, 255).astype(np.uint8)
        disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_VIRIDIS)
        disp_color[disp <= 0] = [30, 30, 30] # dark gray invalid

        left_bgr = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
        right_bgr = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)

        # Composite side-by-side: Left | Right | StereoSGBM Disparity
        h, w = left.shape
        header_h = 42
        comp = np.zeros((h + header_h, w * 3, 3), dtype=np.uint8)
        comp[:header_h, :] = [245, 245, 245]
        comp[header_h:, :w] = left_bgr
        comp[header_h:, w:2*w] = right_bgr
        comp[header_h:, 2*w:] = disp_color

        # Convert to PIL for text annotations
        pil_img = Image.fromarray(comp)
        draw = ImageDraw.Draw(pil_img)

        # Labels
        draw.text((15, 12), f"{title}   |   StereoSGBM Bad@2.0: {bad_rate:.2f}%   |   Coverage: {cov:.1f}%", fill=(20, 20, 20))
        draw.text((15, header_h + 8), "Left Image", fill=(255, 255, 0))
        draw.text((w + 15, header_h + 8), "Right Image", fill=(255, 255, 0))
        draw.text((2 * w + 15, header_h + 8), "StereoSGBM Disparity", fill=(255, 255, 0))

        frames.append(pil_img)

    gif_path = output_dir / "stereo_loop.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=1400,
        loop=0
    )
    print(f"Generated hero GIF: {gif_path}")


def main() -> None:
    assets_dir = Path("assets")
    assets_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path("results")

    print("=" * 72)
    print("SCIENTIFIC VISUALIZATION PIPELINE")
    print("=" * 72)

    # 1. Figure 1: Geometry illustration
    generate_figure_01_geometry(assets_dir)

    # Load saved experimental data
    disp_path = results_dir / "sample_disparities.npz"
    sad_path = results_dir / "sad_profile.json"
    metrics_path = results_dir / "metrics.csv"
    cnn_hist_path = results_dir / "cnn_history.csv"
    cnn_met_path = results_dir / "cnn_metrics.json"
    cnn_samples_path = results_dir / "cnn_test_samples.npz"

    if not disp_path.exists() or not metrics_path.exists():
        print("Missing classical results. Run 'python reproduce.py --full' first.")
        return

    disp_data = np.load(str(disp_path))
    metrics_df = pd.read_csv(metrics_path)

    with open(sad_path, "r", encoding="utf-8") as f:
        sad_data = json.load(f)

    # 2. Figure 2: Classical matching composite
    generate_figure_02_classical(assets_dir, disp_data, sad_data)

    # 3. Figure 3: Photometric robustness
    generate_figure_03_robustness(assets_dir, metrics_df)

    # 4. Figure 4: CNN architecture
    generate_figure_04_cnn_architecture(assets_dir)

    # 5. Figures 5 and 6: CNN experiments
    if cnn_hist_path.exists() and cnn_met_path.exists():
        cnn_hist = pd.read_csv(cnn_hist_path)
        with open(cnn_met_path, "r", encoding="utf-8") as f:
            cnn_metrics = json.load(f)

        generate_figure_05_cnn_learning(assets_dir, cnn_hist, cnn_metrics)
        if cnn_samples_path.exists():
            generate_figure_06_comparison(assets_dir, cnn_metrics, cnn_samples_path)
    else:
        print("CNN results missing. Run 'python cnn.py --epochs 15' to generate Figures 5 and 6.")

    # 6. Hero animated GIF
    generate_stereo_loop_gif(assets_dir, disp_data, metrics_df)

    print("\n" + "=" * 72)
    print("All scientific figures and animations generated successfully.")
    print("=" * 72)


if __name__ == "__main__":
    main()
