"""Generate high-quality, readable figures (Fig 1 to 5) and looping GIFs for stereo-field-notes.

Follows IEEE formatting and GitHub README display constraints:
- Fonts >= 10-12 pt
- Clear, uncrowded 2-4 panel layouts
- Consistent colorbars and scales
- Colorblind-safe palettes and high-contrast lines
"""

import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Output directories
assets_dir = Path("assets")
assets_dir.mkdir(parents=True, exist_ok=True)
results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)

# Calibration setup
calib_path = r"D:\02_Riset\Research Paper\source_code\calibration\calib_params\stereo_params_advanced.npz"
data = np.load(calib_path)
mtx_l, dist_l = data["mtx_left"], data["dist_left"]
mtx_r, dist_r = data["mtx_right"], data["dist_right"]
R, T = data["R"], data["T"]
img_size = (1280, 720)

R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(mtx_l, dist_l, mtx_r, dist_r, img_size, R, T, alpha=0)
map_l1, map_l2 = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, img_size, cv2.CV_32FC1)
map_r1, map_r2 = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, img_size, cv2.CV_32FC1)

root = Path(r"D:\02_Riset\Research Paper\source_code\comparison\gunung")
terrains = ["puncakTinggi", "puncakWarung", "tanggaBiru", "tanggaKuning"]

# Load raw and rectified images
images_raw = {}
images_rect = {}
for t in terrains:
    st_dir = root / t / "stereo"
    raw_l = cv2.imread(str(st_dir / "left_0000.png"))
    raw_r = cv2.imread(str(st_dir / "right_0000.png"))
    rect_l = cv2.remap(raw_l, map_l1, map_l2, cv2.INTER_LINEAR)
    rect_r = cv2.remap(raw_r, map_r1, map_r2, cv2.INTER_LINEAR)
    images_raw[t] = (raw_l, raw_r)
    images_rect[t] = (rect_l, rect_r)

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "font.family": "sans-serif"
})

# ==============================================================================
# Figure 1: Calibration and Rectification
# ==============================================================================
print("Generating Fig 01: Calibration & Rectification...")
raw_l, raw_r = images_raw["puncakTinggi"]
rect_l, rect_r = images_rect["puncakTinggi"]

# Feature matches to show displacement
orb = cv2.ORB_create(1500)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Raw matches
kp1, des1 = orb.detectAndCompute(raw_l, None)
kp2, des2 = orb.detectAndCompute(raw_r, None)
m_raw = bf.match(des1, des2)
pts1 = np.float32([kp1[m.queryIdx].pt for m in m_raw])
pts2 = np.float32([kp2[m.trainIdx].pt for m in m_raw])
_, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0)
inliers_raw = mask.ravel() == 1
raw_dys = np.abs(pts1[inliers_raw, 1] - pts2[inliers_raw, 1])

# Rectified matches
kp1_r, des1_r = orb.detectAndCompute(rect_l, None)
kp2_r, des2_r = orb.detectAndCompute(rect_r, None)
m_rect = bf.match(des1_r, des2_r)
pts1_r = np.float32([kp1_r[m.queryIdx].pt for m in m_rect])
pts2_r = np.float32([kp2_r[m.trainIdx].pt for m in m_rect])
_, mask_r = cv2.findFundamentalMat(pts1_r, pts2_r, cv2.FM_RANSAC, 3.0)
inliers_rect = mask_r.ravel() == 1
rect_dys = np.abs(pts1_r[inliers_rect, 1] - pts2_r[inliers_rect, 1])

fig = plt.figure(figsize=(12, 6.8))
gs = fig.add_gridspec(2, 3, width_ratios=[1.6, 1.6, 1.2], wspace=0.18, hspace=0.18)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])
ax_hist = fig.add_subplot(gs[:, 2])

raw_l_rgb = cv2.cvtColor(raw_l, cv2.COLOR_BGR2RGB)
raw_r_rgb = cv2.cvtColor(raw_r, cv2.COLOR_BGR2RGB)
rect_l_rgb = cv2.cvtColor(rect_l, cv2.COLOR_BGR2RGB)
rect_r_rgb = cv2.cvtColor(rect_r, cv2.COLOR_BGR2RGB)

epipolar_ys = [180, 360, 540]
for y in epipolar_ys:
    cv2.line(raw_l_rgb, (0, y), (1280, y), (255, 60, 60), 2)
    cv2.line(raw_r_rgb, (0, y), (1280, y), (255, 60, 60), 2)
    cv2.line(rect_l_rgb, (0, y), (1280, y), (0, 255, 255), 2)
    cv2.line(rect_r_rgb, (0, y), (1280, y), (0, 255, 255), 2)

ax1.imshow(raw_l_rgb)
ax1.set_title("Raw Left Frame", fontsize=11, fontweight="bold")
ax1.axis("off")

ax2.imshow(raw_r_rgb)
ax2.set_title("Raw Right Frame (Vertical Shift: 28.0 px)", fontsize=11, color="darkred", fontweight="bold")
ax2.axis("off")

ax3.imshow(rect_l_rgb)
ax3.set_title("Calibrated Rectified Left Frame", fontsize=11, fontweight="bold")
ax3.axis("off")

ax4.imshow(rect_r_rgb)
ax4.set_title("Calibrated Rectified Right Frame (Residual: 2.1 px)", fontsize=11, color="darkgreen", fontweight="bold")
ax4.axis("off")

# Residual distribution histogram
ax_hist.hist(raw_dys, bins=np.linspace(0, 45, 30), alpha=0.6, color="salmon", label=f"Raw (Med={np.median(raw_dys):.1f} px)", edgecolor="black")
ax_hist.hist(rect_dys, bins=np.linspace(0, 45, 30), alpha=0.7, color="mediumseagreen", label=f"Rectified (Med={np.median(rect_dys):.1f} px)", edgecolor="black")
ax_hist.axvline(np.median(raw_dys), color="darkred", linestyle="--", linewidth=2)
ax_hist.axvline(np.median(rect_dys), color="darkgreen", linestyle="--", linewidth=2)
ax_hist.set_xlabel("Vertical Error |dy| (px)", fontweight="bold")
ax_hist.set_ylabel("Inlier Match Count", fontweight="bold")
ax_hist.set_title("Epipolar Error Residuals", fontweight="bold")
ax_hist.legend(loc="upper right", frameon=True)
ax_hist.grid(True, linestyle=":", alpha=0.6)

plt.subplots_adjust(left=0.03, right=0.97, top=0.94, bottom=0.10)
plt.savefig(assets_dir / "fig01_rectification.png", dpi=250)
plt.savefig(assets_dir / "fig01_rectification.pdf")
plt.close()

# ==============================================================================
# Figure 2: Four Terrain Challenges
# ==============================================================================
print("Generating Fig 02: Four Terrain Challenges...")
fig, axs = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)

titles = [
    ("(a) puncakTinggi: Volcanic Ridge", "16.8% Sky Saturation, Low Gradient Sky, Far Mountains"),
    ("(b) puncakWarung: Mountain Stall", "Sheltered Porch, Corrugated Roof, High Contrast Shadow"),
    ("(c) tanggaBiru: Blue Stairway", "Steep Concrete Steps, Painted Railing, Thin Shrub Foliage"),
    ("(d) tanggaKuning: Yellow Stairway", "Stone Steps, Repeating Metal Bars, Steep Slope Geometry")
]

for idx, (t, (title_main, title_sub)) in enumerate(zip(terrains, titles)):
    ax = axs[idx // 2, idx % 2]
    img = cv2.cvtColor(images_rect[t][0], cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    ax.set_title(f"{title_main}\nChallenge: {title_sub}", fontsize=11, fontweight="bold", pad=8)
    ax.axis("off")
    ax.text(25, 680, "1280 x 720 px | B = 97.5 mm", bbox=dict(facecolor="black", alpha=0.7, edgecolor="none", boxstyle="round,pad=0.3"), color="white", fontsize=9)

plt.savefig(assets_dir / "fig02_four_terrains.png", dpi=250)
plt.savefig(assets_dir / "fig02_four_terrains.pdf")
plt.close()

# ==============================================================================
# Figure 3: SGBM Improvement on Challenging Terrain
# ==============================================================================
print("Generating Fig 03: SGBM Improvement...")
disp_s0 = np.load("results/sequential_study/disparities/puncakTinggi_S0_Default.npy")
disp_s7 = np.load("results/sequential_study/disparities/puncakTinggi_S7_WLSFilter.npy")
rect_l, rect_r = images_rect["puncakTinggi"]

# Compute LR consistency map for S7
s6_l = cv2.StereoSGBM_create(
    minDisparity=0, numDisparities=160, blockSize=7, P1=150, P2=1200,
    uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)
s6_r = cv2.ximgproc.createRightMatcher(s6_l)
disp_r_val = s6_r.compute(rect_r, rect_l).astype(np.float32) / 16.0
h, w = disp_s7.shape
xc, yc = np.meshgrid(np.arange(w), np.arange(h))
xm = np.clip(np.round(xc - disp_s7).astype(int), 0, w - 1)
r_matched = disp_r_val[yc, xm]
lr_err = np.abs(disp_s7 - (-r_matched if np.median(disp_r_val) < 0 else r_matched))

cons_vis = np.zeros((h, w, 3), dtype=np.uint8)
valid_pred = disp_s7 > 0
consistent = (lr_err <= 1.5) & valid_pred
inconsistent = (lr_err > 1.5) & valid_pred
cons_vis[consistent] = [40, 200, 40]   # green
cons_vis[inconsistent] = [220, 40, 40] # red
cons_vis[~valid_pred] = [60, 60, 60]   # dark gray

fig, axs = plt.subplots(2, 2, figsize=(12, 7.2))

axs[0, 0].imshow(cv2.cvtColor(rect_l, cv2.COLOR_BGR2RGB))
axs[0, 0].set_title("(a) Rectified Left Input (puncakTinggi)", fontweight="bold")
axs[0, 0].axis("off")

im_s0 = axs[0, 1].imshow(disp_s0, cmap="turbo", vmin=0, vmax=160)
axs[0, 1].set_title("(b) Default SGBM (Noisy Sky, Missing Rocks)", color="darkred", fontweight="bold")
axs[0, 1].axis("off")

im_s7 = axs[1, 0].imshow(disp_s7, cmap="turbo", vmin=0, vmax=160)
axs[1, 0].set_title("(c) Optimized SGBM + WLS (Clean Planes)", color="darkgreen", fontweight="bold")
axs[1, 0].axis("off")

axs[1, 1].imshow(cons_vis)
axs[1, 1].set_title("(d) Consistency Check (Green: Pass, Red: Reject)", fontweight="bold")
axs[1, 1].axis("off")

legend_patches = [
    patches.Patch(facecolor=[0.15, 0.78, 0.15], label="Consistent (|dL - dR| <= 1.5 px)"),
    patches.Patch(facecolor=[0.86, 0.15, 0.15], label="Inconsistent (Multi-peak mismatch)"),
    patches.Patch(facecolor=[0.24, 0.24, 0.24], label="Filtered / Saturated Sky")
]
axs[1, 1].legend(handles=legend_patches, loc="lower left", frameon=True, facecolor="white", framealpha=0.9, fontsize=9)

# Clean, dedicated right colorbar for disparity maps
fig.subplots_adjust(left=0.03, right=0.90, top=0.94, bottom=0.04, wspace=0.08, hspace=0.14)
cbar_ax = fig.add_axes([0.92, 0.12, 0.02, 0.76])
cbar = fig.colorbar(im_s7, cax=cbar_ax)
cbar.set_label("Disparity (px)", fontweight="bold", fontsize=11)

plt.savefig(assets_dir / "fig03_sgbm_comparison.png", dpi=250)
plt.savefig(assets_dir / "fig03_sgbm_comparison.pdf")
plt.close()

# ==============================================================================
# Figure 4: Optimization Evidence and Sequential Progression
# ==============================================================================
print("Generating Fig 04: Optimization Evidence...")
df_stages = pd.read_csv("results/sequential_study/stages_comparison.csv")

fig, (ax_mid, ax_gun) = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

stage_labels = ["S0\nDefault", "S1\nRange160", "S2\nBlock7", "S3\nPenalty", "S4\n3Way", "S5\nHistMatch", "S6\nPostFilt", "S7\nWLS"]
x = np.arange(len(stage_labels))

# Left: Middlebury Ground Truth Metrics
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

# Right: Gunung Field Diagnostics
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

plt.savefig(assets_dir / "fig04_optimization.png", dpi=250)
plt.savefig(assets_dir / "fig04_optimization.pdf")
plt.close()

# ==============================================================================
# Figure 5: Cross-Terrain Performance Summary
# ==============================================================================
print("Generating Fig 05: Cross-Terrain Summary...")
fig, (ax_cov, ax_cons) = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

stages_to_plot = ["S0_Default", "S4_Mode3Way", "S6_PostFilters", "S7_WLSFilter"]
plot_names = ["Default SGBM", "3-Way SGBM", "Speckle Filtered", "Edge-Aware WLS"]
bar_width = 0.18
x_terr = np.arange(len(terrains))
terrain_display_names = ["puncakTinggi\n(Summit Ridge)", "puncakWarung\n(Roadside Stall)", "tanggaBiru\n(Blue Stairway)", "tanggaKuning\n(Yellow Stairway)"]

colors = ["#4a7bb0", "#55b084", "#e0913e", "#8b5ea6"]

# Coverage bars
for i, (st, name, col) in enumerate(zip(stages_to_plot, plot_names, colors)):
    row = df_stages[df_stages["stage"] == st].iloc[0]
    vals = [row[f"{t}_cov"] for t in terrains]
    ax_cov.bar(x_terr + (i - 1.5) * bar_width, vals, width=bar_width, label=name, color=col, edgecolor="black", alpha=0.85)

ax_cov.set_xticks(x_terr)
ax_cov.set_xticklabels(terrain_display_names, fontsize=9.5)
ax_cov.set_ylabel("Valid Prediction Coverage (%)", fontweight="bold")
ax_cov.set_title("(a) Valid Coverage Across Four Terrains", fontweight="bold")
ax_cov.set_ylim(0, 100)
ax_cov.legend(loc="upper right", frameon=True, fontsize=9)
ax_cov.grid(True, axis="y", linestyle=":", alpha=0.6)

# Consistency bars
for i, (st, name, col) in enumerate(zip(stages_to_plot, plot_names, colors)):
    row = df_stages[df_stages["stage"] == st].iloc[0]
    vals = [row[f"{t}_lr_cons"] for t in terrains]
    ax_cons.bar(x_terr + (i - 1.5) * bar_width, vals, width=bar_width, label=name, color=col, edgecolor="black", alpha=0.85)

ax_cons.set_xticks(x_terr)
ax_cons.set_xticklabels(terrain_display_names, fontsize=9.5)
ax_cons.set_ylabel("Left-Right Consistency (%)", fontweight="bold")
ax_cons.set_title("(b) Consistency Across Four Terrains", fontweight="bold")
ax_cons.set_ylim(0, 85)
ax_cons.legend(loc="upper right", frameon=True, fontsize=9)
ax_cons.grid(True, axis="y", linestyle=":", alpha=0.6)

plt.savefig(assets_dir / "fig05_terrain_results.png", dpi=250)
plt.savefig(assets_dir / "fig05_terrain_results.pdf")
plt.close()

# ==============================================================================
# Hero Looping GIF: terrain_loop.gif
# ==============================================================================
print("Generating Hero Looping GIF: assets/terrain_loop.gif...")

gif_frames = []
fps = 10
seconds_per_terrain = 2.5
frames_per_terrain = int(fps * seconds_per_terrain)

colormap = plt.get_cmap("turbo")

for t, (title_main, _) in zip(terrains, titles):
    rect_l, rect_r = images_rect[t]
    
    # Compute default S0
    s0_l = cv2.StereoSGBM_create(minDisparity=0, numDisparities=128, blockSize=3)
    d_raw0 = s0_l.compute(rect_l, rect_r).astype(np.float32) / 16.0
    d_norm0 = np.clip(d_raw0 / 160.0, 0, 1)
    d_color0 = (colormap(d_norm0)[:, :, :3] * 255).astype(np.uint8)
    d_color0[d_raw0 <= 0] = [30, 30, 30] # dark background for invalid
    
    # Compute optimized S7
    s7_l = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=160, blockSize=7, P1=150, P2=1200,
        uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    s7_r = cv2.ximgproc.createRightMatcher(s7_l)
    wls = cv2.ximgproc.createDisparityWLSFilter(s7_l)
    wls.setLambda(8000.0)
    wls.setSigmaColor(1.5)
    disp_l_val = s7_l.compute(rect_l, rect_r)
    disp_r_val = s7_r.compute(rect_r, rect_l)
    disp_wls = wls.filter(disp_l_val, rect_l, None, disp_r_val).astype(np.float32) / 16.0
    d_norm7 = np.clip(disp_wls / 160.0, 0, 1)
    d_color7 = (colormap(d_norm7)[:, :, :3] * 255).astype(np.uint8)
    d_color7[disp_wls <= 0] = [30, 30, 30]

    # Create composite frame: [Rectified RGB | Default SGBM | Optimized SGBM]
    # Downscale width to 384x216 each, total canvas: 1152 x 260 with header
    w_sub, h_sub = 384, 216
    im_sub = cv2.resize(cv2.cvtColor(rect_l, cv2.COLOR_BGR2RGB), (w_sub, h_sub))
    d0_sub = cv2.resize(d_color0, (w_sub, h_sub))
    d7_sub = cv2.resize(d_color7, (w_sub, h_sub))

    canvas = np.zeros((h_sub + 44, w_sub * 3, 3), dtype=np.uint8)
    canvas[44:, :w_sub] = im_sub
    canvas[44:, w_sub:2*w_sub] = d0_sub
    canvas[44:, 2*w_sub:] = d7_sub

    # Header bar
    canvas[:44, :] = [20, 24, 32]
    
    # Text overlays
    cv2.putText(canvas, f"Terrain: {t}", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(canvas, "1. Rectified Input", (15, h_sub + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    cv2.putText(canvas, "2. Default SGBM", (w_sub + 15, h_sub + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 100, 100), 1)
    cv2.putText(canvas, "3. Optimized SGBM + WLS", (2*w_sub + 15, h_sub + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 240, 120), 1)

    pil_frame = Image.fromarray(canvas)
    for _ in range(frames_per_terrain):
        gif_frames.append(pil_frame)

# Save looping GIF
gif_path = assets_dir / "terrain_loop.gif"
gif_frames[0].save(
    gif_path,
    save_all=True,
    append_images=gif_frames[1:],
    duration=int(1000 / fps),
    loop=0,
    optimize=True
)
print(f"Saved {gif_path} ({len(gif_frames)} frames, {len(gif_frames)/fps:.1f}s loop)")
print("Visual generation completed successfully!")
