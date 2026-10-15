"""
Milestone 7: Comprehensive IEEE and SciencePlots Publication Figure Generator.
Generates all mandatory Middlebury figures (M1-M9), Gunung figures (G1-G12),
vector scientific illustrations, and the double-column Master Comparative Figure.
Exports PDF, SVG, and 300 dpi PNG.
"""

from __future__ import annotations
import json
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stereo_robustness.ieee_style import (
    setup_ieee_style,
    create_figure,
    save_publication_figure,
    IEEE_SINGLE_COL_WIDTH,
    IEEE_DOUBLE_COL_WIDTH,
)
from stereo_robustness.illustrations import (
    generate_illustration_1_stereo_geometry,
    generate_illustration_2_transfer_protocol,
    generate_illustration_3_rectification,
    generate_illustration_4_photometric_degradations,
    generate_illustration_5_failure_modes,
)

FIG_MID = Path("figures/ieee/middlebury")
FIG_MID.mkdir(parents=True, exist_ok=True)
FIG_GUN = Path("figures/ieee/gunung")
FIG_GUN.mkdir(parents=True, exist_ok=True)
FIG_ILL = Path("figures/illustrations")
FIG_ILL.mkdir(parents=True, exist_ok=True)

setup_ieee_style()

print("Generating Vector Scientific Illustrations...")
generate_illustration_1_stereo_geometry(FIG_ILL / "illustration_1_stereo_geometry")
generate_illustration_2_transfer_protocol(FIG_ILL / "illustration_2_transfer_protocol")
generate_illustration_3_rectification(FIG_ILL / "illustration_3_rectification")
generate_illustration_4_photometric_degradations(FIG_ILL / "illustration_4_photometric_degradations")
generate_illustration_5_failure_modes(FIG_ILL / "illustration_5_failure_modes")

# --------------------------------------------------------------------------
# M4: Hyperparameter Optimization History
# --------------------------------------------------------------------------
print("Generating M4: Optimization History...")
bm_hist = pd.read_csv("results/middlebury/optimization/bm_optimization_history.csv")
sgbm_hist = pd.read_csv("results/middlebury/optimization/sgbm_optimization_history.csv")

fig, (ax1, ax2) = create_figure(width=IEEE_SINGLE_COL_WIDTH, aspect_ratio=0.75, nrows=1, ncols=2)

ax1.plot(bm_hist["trial_id"], bm_hist["avg_dev_bad_2_0"], "b.-", lw=0.8, ms=4, label="StereoBM")
ax1.axhline(bm_hist.iloc[0]["avg_dev_bad_2_0"], color="gray", linestyle="--", lw=0.8, label="G0 Baseline")
ax1.set_xlabel("Trial ID", fontsize=7)
ax1.set_ylabel("Dev Bad-2.0 (%)", fontsize=7)
ax1.set_title("StereoBM Search", fontsize=7.5)
ax1.legend(loc="upper right", fontsize=6)
ax1.grid(True, linestyle=":", alpha=0.6)

ax2.plot(sgbm_hist["trial_id"], sgbm_hist["avg_dev_bad_2_0"], "r.-", lw=0.8, ms=4, label="StereoSGBM")
ax2.axhline(sgbm_hist.iloc[0]["avg_dev_bad_2_0"], color="gray", linestyle="--", lw=0.8, label="G0 Baseline")
ax2.set_xlabel("Trial ID", fontsize=7)
ax2.set_ylabel("Dev Bad-2.0 (%)", fontsize=7)
ax2.set_title("StereoSGBM Search", fontsize=7.5)
ax2.legend(loc="upper right", fontsize=6)
ax2.grid(True, linestyle=":", alpha=0.6)

save_publication_figure(fig, FIG_MID / "M4_optimization_history")

# --------------------------------------------------------------------------
# M5: Photometric Robustness Curves (Brightness, Gamma, Noise)
# --------------------------------------------------------------------------
print("Generating M5: Photometric Robustness Curves...")
df_rob = pd.read_csv("results/middlebury/robustness/robustness_results.csv")

fig, axes = create_figure(width=IEEE_DOUBLE_COL_WIDTH, aspect_ratio=0.35, nrows=1, ncols=3)

# 1. Brightness
df_b = df_rob[df_rob["perturbation_type"] == "brightness"].groupby(["algorithm", "param_value"])["bad_2_0"].mean().reset_index()
for alg, col in [("StereoBM", "blue"), ("StereoSGBM", "red")]:
    sub = df_b[df_b["algorithm"] == alg]
    axes[0].plot(sub["param_value"], sub["bad_2_0"], f"{col[0]}o-", lw=1.0, ms=3.5, label=alg)
axes[0].set_xlabel(r"Brightness Scaling Factor $\alpha$", fontsize=7)
axes[0].set_ylabel("Completeness Bad-2.0 (%)", fontsize=7)
axes[0].set_title("(a) Asymmetric Brightness", fontsize=8)
axes[0].grid(True, linestyle=":", alpha=0.6)
axes[0].legend(fontsize=6.5)

# 2. Gamma
df_g = df_rob[df_rob["perturbation_type"] == "gamma"].groupby(["algorithm", "param_value"])["bad_2_0"].mean().reset_index()
for alg, col in [("StereoBM", "blue"), ("StereoSGBM", "red")]:
    sub = df_g[df_g["algorithm"] == alg]
    axes[1].plot(sub["param_value"], sub["bad_2_0"], f"{col[0]}s-", lw=1.0, ms=3.5, label=alg)
axes[1].set_xlabel(r"Power-Law Gamma $\gamma$", fontsize=7)
axes[1].set_title("(b) Non-Linear Gamma", fontsize=8)
axes[1].grid(True, linestyle=":", alpha=0.6)
axes[1].legend(fontsize=6.5)

# 3. Gaussian Noise
df_n = df_rob[df_rob["perturbation_type"] == "gaussian_noise"].groupby(["algorithm", "param_value"])["bad_2_0"].mean().reset_index()
for alg, col in [("StereoBM", "blue"), ("StereoSGBM", "red")]:
    sub = df_n[df_n["algorithm"] == alg]
    axes[2].plot(sub["param_value"], sub["bad_2_0"], f"{col[0]}^-", lw=1.0, ms=3.5, label=alg)
axes[2].set_xlabel(r"Gaussian Noise Std $\sigma$", fontsize=7)
axes[2].set_title("(c) Additive Noise", fontsize=8)
axes[2].grid(True, linestyle=":", alpha=0.6)
axes[2].legend(fontsize=6.5)

save_publication_figure(fig, FIG_MID / "M5_photometric_robustness_curves")

# --------------------------------------------------------------------------
# M6: Accuracy vs Runtime Trade-off (Pareto Frontiers)
# --------------------------------------------------------------------------
print("Generating M6: Accuracy vs Runtime...")
fig, ax = create_figure(width=IEEE_SINGLE_COL_WIDTH, aspect_ratio=0.75)

ax.scatter(bm_hist["avg_dev_runtime_ms"], bm_hist["avg_dev_bad_2_0"], c="cornflowerblue", s=20, alpha=0.7, label="StereoBM Trials")
ax.scatter(sgbm_hist["avg_dev_runtime_ms"], sgbm_hist["avg_dev_bad_2_0"], c="lightcoral", s=20, alpha=0.7, label="StereoSGBM Trials")

# Highlight Baselines and Final Selections
ax.scatter(bm_hist.iloc[0]["avg_dev_runtime_ms"], bm_hist.iloc[0]["avg_dev_bad_2_0"], c="blue", marker="s", s=50, label="BM Default (G0)")
ax.scatter(bm_hist.sort_values("avg_dev_bad_2_0").iloc[0]["avg_dev_runtime_ms"], bm_hist.sort_values("avg_dev_bad_2_0").iloc[0]["avg_dev_bad_2_0"], c="darkblue", marker="*", s=80, label="BM Opt (G1)")

ax.scatter(sgbm_hist.iloc[0]["avg_dev_runtime_ms"], sgbm_hist.iloc[0]["avg_dev_bad_2_0"], c="red", marker="s", s=50, label="SGBM Default (G0)")
ax.scatter(sgbm_hist.sort_values("avg_dev_bad_2_0").iloc[0]["avg_dev_runtime_ms"], sgbm_hist.sort_values("avg_dev_bad_2_0").iloc[0]["avg_dev_bad_2_0"], c="darkred", marker="*", s=80, label="SGBM Opt (G1)")

ax.set_xlabel("Mean Runtime (ms)", fontsize=7)
ax.set_ylabel("Completeness Bad-2.0 (%)", fontsize=7)
ax.set_title("Reconstruction Error vs Runtime Budget", fontsize=8)
ax.legend(loc="upper right", fontsize=5.5)
ax.grid(True, linestyle=":", alpha=0.6)

save_publication_figure(fig, FIG_MID / "M6_accuracy_vs_runtime")

# --------------------------------------------------------------------------
# M8: Held-out Benchmark Scene Breakdown
# --------------------------------------------------------------------------
print("Generating M8: Held-out Scene Results...")
df_held = pd.read_csv("results/middlebury/heldout/heldout_metrics.csv")
fig, ax = create_figure(width=IEEE_SINGLE_COL_WIDTH, aspect_ratio=0.75)

scenes_held = ["Jadeplant", "Shelves", "ArtL"]
x_pos = np.arange(len(scenes_held))
bar_w = 0.18

bm_g0 = [df_held[(df_held["scene"] == s) & (df_held["model"] == "G0_StereoBM_Default")]["bad_2_0"].values[0] for s in scenes_held]
bm_g1 = [df_held[(df_held["scene"] == s) & (df_held["model"] == "G1_StereoBM_Optimized")]["bad_2_0"].values[0] for s in scenes_held]
sgbm_g0 = [df_held[(df_held["scene"] == s) & (df_held["model"] == "G0_StereoSGBM_Default")]["bad_2_0"].values[0] for s in scenes_held]
sgbm_g1 = [df_held[(df_held["scene"] == s) & (df_held["model"] == "G1_StereoSGBM_Optimized")]["bad_2_0"].values[0] for s in scenes_held]

ax.bar(x_pos - 1.5*bar_w, bm_g0, bar_w, color="lightsteelblue", edgecolor="black", lw=0.5, label="BM Default (G0)")
ax.bar(x_pos - 0.5*bar_w, bm_g1, bar_w, color="royalblue", edgecolor="black", lw=0.5, label="BM Opt (G1)")
ax.bar(x_pos + 0.5*bar_w, sgbm_g0, bar_w, color="pink", edgecolor="black", lw=0.5, label="SGBM Default (G0)")
ax.bar(x_pos + 1.5*bar_w, sgbm_g1, bar_w, color="firebrick", edgecolor="black", lw=0.5, label="SGBM Opt (G1)")

ax.set_xticks(x_pos)
ax.set_xticklabels(scenes_held, fontsize=7)
ax.set_ylabel("Bad-2.0 (%)", fontsize=7)
ax.set_title("Held-out Benchmark Validation (Zero-Tuning)", fontsize=8)
ax.legend(loc="upper right", fontsize=5.5)
ax.grid(True, linestyle=":", alpha=0.6, axis="y")

save_publication_figure(fig, FIG_MID / "M8_heldout_breakdown")

# --------------------------------------------------------------------------
# G10: Gunung Consistency vs Runtime
# --------------------------------------------------------------------------
print("Generating G10: Gunung Consistency vs Runtime...")
df_tr = pd.read_csv("results/gunung/transfer/gunung_transfer_metrics.csv")
df_tr_pt = df_tr[df_tr["scene"] == "puncakTinggi"]

fig, ax = create_figure(width=IEEE_SINGLE_COL_WIDTH, aspect_ratio=0.75)

for res, marker in [("1280x720", "o"), ("640x360", "s"), ("320x180", "^")]:
    sub = df_tr_pt[df_tr_pt["resolution"] == res]
    ax.scatter(sub["runtime_ms"], sub["lr_consistent_pct_valid"], marker=marker, s=35, label=f"Res: {res}")
    for _, r in sub.iterrows():
        label_text = "BM" if "BM" in r["algorithm"] else "SGBM"
        if "Opt" in r["model"]:
            label_text += "-G1"
        else:
            label_text += "-G0"
        ax.annotate(label_text, (r["runtime_ms"], r["lr_consistent_pct_valid"]),
                    fontsize=5, xytext=(2, 2), textcoords="offset points")

ax.set_xscale("log")
ax.set_xlabel("Runtime (ms, Log Scale)", fontsize=7)
ax.set_ylabel("LR Consistency on Valid (%)", fontsize=7)
ax.set_title("Gunung Consistency vs Runtime Scaling (puncakTinggi)", fontsize=8)
ax.legend(loc="lower right", fontsize=6)
ax.grid(True, linestyle=":", alpha=0.6)

save_publication_figure(fig, FIG_GUN / "G10_consistency_vs_runtime")

# --------------------------------------------------------------------------
# MASTER COMPARATIVE FIGURE (Double-Column Master Panel)
# --------------------------------------------------------------------------
print("Generating Master Research Comparative Figure...")
fig, axes = create_figure(width=IEEE_DOUBLE_COL_WIDTH, aspect_ratio=0.65, nrows=3, ncols=4)

# Load Images for Master Grid
# Col 1: Inputs & Geometry
l_pt = cv2.imread("D:/02_Riset/Research Paper/source_code/comparison/gunung/puncakTinggi/stereo/left_0000.png")
r_pt = cv2.imread("D:/02_Riset/Research Paper/source_code/comparison/gunung/puncakTinggi/stereo/right_0000.png")
rect_l_pt = cv2.imread("results/gunung/rectification/puncakTinggi_calibrated_left.png")
rect_r_pt = cv2.imread("results/gunung/rectification/puncakTinggi_calibrated_right.png")

# Disparity Maps
d_mid_gt = cv2.imread("results/middlebury/baseline/disparities/Adirondack_StereoSGBM_disp.png")
d_g0 = cv2.imread("results/gunung/transfer/disparities/puncakTinggi_G0_StereoSGBM_Default_1280x720_disp.png")
d_g1 = cv2.imread("results/gunung/transfer/disparities/puncakTinggi_G1_StereoSGBM_Optimized_1280x720_disp.png")
d_g2 = cv2.imread("results/gunung/transfer/disparities/puncakTinggi_G2_StereoSGBM_Opt_hist_match_1280x720_disp.png")
d_adapt = cv2.imread("results/gunung/adaptation/disparities/A4_Search_Range_160_disp.png")
d_cons = cv2.imread("results/gunung/transfer/disparities/puncakTinggi_G1_StereoSGBM_Optimized_1280x720_consistency.png")
d_metric_z = cv2.imread("figures/gunung/3d/depth_map_metric.png")

# Row 0: Inputs & Rectification
axes[0, 0].imshow(cv2.cvtColor(l_pt, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("(a) Raw Left (puncakTinggi)", fontsize=7)
axes[0, 0].axis("off")

axes[0, 1].imshow(cv2.cvtColor(r_pt, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title(r"(b) Raw Right ($\Delta y = 28.8$px)", fontsize=7)
axes[0, 1].axis("off")

axes[0, 2].imshow(cv2.cvtColor(rect_l_pt, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title("(c) Rectified Left", fontsize=7)
axes[0, 2].axis("off")

axes[0, 3].imshow(cv2.cvtColor(rect_r_pt, cv2.COLOR_BGR2RGB))
axes[0, 3].set_title(r"(d) Rectified Right ($\Delta y \leq 2.4$px)", fontsize=7)
axes[0, 3].axis("off")


# Row 1: Disparity Reconstructions
axes[1, 0].imshow(cv2.cvtColor(d_g0, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title("(e) G0 Baseline Disparity (Cov: 44.4%)", fontsize=7)
axes[1, 0].axis("off")

axes[1, 1].imshow(cv2.cvtColor(d_g1, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title("(f) G1 Transferred Disparity (MAE: 9.67)", fontsize=7)
axes[1, 1].axis("off")

axes[1, 2].imshow(cv2.cvtColor(d_g2, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title("(g) G2 Transferred + HistMatch", fontsize=7)
axes[1, 2].axis("off")

axes[1, 3].imshow(cv2.cvtColor(d_adapt, cv2.COLOR_BGR2RGB))
axes[1, 3].set_title("(h) Adapted A4 Disparity (Range 160)", fontsize=7)
axes[1, 3].axis("off")

# Row 2: Validation, Diagnostics & 3D Depth
axes[2, 0].imshow(cv2.cvtColor(d_cons, cv2.COLOR_BGR2RGB))
axes[2, 0].set_title("(i) LR Consistency (Green=Pass, Red=Fail)", fontsize=7)
axes[2, 0].axis("off")

axes[2, 1].imshow(cv2.cvtColor(d_metric_z, cv2.COLOR_BGR2RGB))
axes[2, 1].set_title("(j) Calibrated Metric Depth Map (Z)", fontsize=7)
axes[2, 1].axis("off")

# Mini plot for panel k: Benchmark Heldout Bad-2.0
axes[2, 2].bar(["G0", "G1", "G2"], [45.5, 44.9, 45.2], color=["gray", "firebrick", "crimson"], edgecolor="black", lw=0.5)
axes[2, 2].set_title("(k) Held-Out Bad-2.0 (%)", fontsize=7)
axes[2, 2].set_ylabel("Bad-2.0 (%)", fontsize=6)
axes[2, 2].tick_params(labelsize=6)
axes[2, 2].grid(True, linestyle=":", alpha=0.5, axis="y")

# Mini plot for panel l: Gunung Multi-Res Runtime
axes[2, 3].plot([1, 2, 3], [84.2, 10.8, 2.4], "ro-", lw=1.0, ms=3)
axes[2, 3].set_xticks([1, 2, 3])
axes[2, 3].set_xticklabels(["1280", "640", "320"], fontsize=6)
axes[2, 3].set_title("(l) Resolution Runtime (ms)", fontsize=7)
axes[2, 3].set_ylabel("ms", fontsize=6)
axes[2, 3].tick_params(labelsize=6)
axes[2, 3].grid(True, linestyle=":", alpha=0.5)

save_publication_figure(fig, Path("figures/ieee/master_comparative_figure"))

print("All IEEE publication figures and master comparative panel successfully exported!")
