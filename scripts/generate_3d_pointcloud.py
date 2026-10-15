"""
3D Point Cloud Reconstruction and Metric Depth Projection.
Uses calibrated stereo parameters and Q reprojection matrix to reconstruct
a dense 3D point cloud of puncakTinggi.
Exports standard PLY point cloud, depth distribution, and colored previews.
"""

from __future__ import annotations
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml

from stereo_robustness.calibration import StereoCalibrationParams, compute_rectification_models, reproject_to_3d
from stereo_robustness.stereo import StereoSGBMConfig, compute_disparity_sgbm, compute_right_disparity
from stereo_robustness.consistency import check_left_right_consistency
from stereo_robustness.visualization import save_image_rgb
from stereo_robustness.ieee_style import setup_ieee_style, create_figure, save_publication_figure, IEEE_SINGLE_COL_WIDTH, IEEE_DOUBLE_COL_WIDTH

OUT_DIR = Path("results/gunung/3d")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = Path("figures/gunung/3d")
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

# SGBM with extended search range (numDisparities = 160)
cfg = StereoSGBMConfig(
    minDisparity=0,
    numDisparities=160,
    blockSize=7,
    P1=150,
    P2=1200,
    disp12MaxDiff=1,
    preFilterCap=63,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=2,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
)

res_l = compute_disparity_sgbm(rect_l, rect_r, cfg)
res_r = compute_right_disparity(rect_l, rect_r, cfg)

# Apply strict Left-Right consistency mask to eliminate spurious foliage and sky
cons_mask, _, _ = check_left_right_consistency(res_l.disparity, res_r.disparity, res_l.valid_mask, res_r.valid_mask, threshold=1.5)

# Reproject to 3D metric coordinates (meters)
# Physical bounds: depth between 0.5m and 35.0m
points_3d, valid_3d = reproject_to_3d(res_l.disparity, rect_models.Q, min_depth=0.5, max_depth=35.0, valid_mask=cons_mask)

colors_rgb = cv2.cvtColor(rect_l, cv2.COLOR_BGR2RGB)

# Extract valid 3D points and colors
valid_pts = points_3d[valid_3d]
valid_cols = colors_rgb[valid_3d]

print(f"Reconstructed {len(valid_pts):,} valid 3D metric points.")
print(f"X range: [{valid_pts[:, 0].min():.2f}, {valid_pts[:, 0].max():.2f}] m")
print(f"Y range: [{valid_pts[:, 1].min():.2f}, {valid_pts[:, 1].max():.2f}] m")
print(f"Z (Depth) range: [{valid_pts[:, 2].min():.2f}, {valid_pts[:, 2].max():.2f}] m")
print(f"Median Depth: {np.median(valid_pts[:, 2]):.2f} m")

# Export to ASCII PLY file
ply_file = OUT_DIR / "puncakTinggi_pointcloud.ply"
with open(ply_file, "w") as f:
    f.write("ply\n")
    f.write("format ascii 1.0\n")
    f.write(f"element vertex {len(valid_pts)}\n")
    f.write("property float x\n")
    f.write("property float y\n")
    f.write("property float z\n")
    f.write("property uchar red\n")
    f.write("property uchar green\n")
    f.write("property uchar blue\n")
    f.write("end_header\n")
    for pt, col in zip(valid_pts, valid_cols):
        f.write(f"{pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f} {col[0]} {col[1]} {col[2]}\n")

print(f"Saved PLY point cloud to: {ply_file}")

# 1. Depth map visualization
setup_ieee_style()
z_map = np.zeros(res_l.disparity.shape, dtype=np.float32)
z_map[valid_3d] = points_3d[:, :, 2][valid_3d]

fig, ax = create_figure(width=IEEE_SINGLE_COL_WIDTH, aspect_ratio=0.6)
im = ax.imshow(np.where(valid_3d, z_map, np.nan), cmap="plasma_r", vmin=1.0, vmax=25.0)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Depth $Z$ (meters)", fontsize=7)
cbar.ax.tick_params(labelsize=6)
ax.set_title("Calibrated Metric Depth Map (puncakTinggi)", fontsize=8)
ax.axis("off")
save_publication_figure(fig, FIG_DIR / "depth_map_metric")

# 2. Depth distribution histogram
fig, ax = create_figure(width=IEEE_SINGLE_COL_WIDTH, aspect_ratio=0.7)
ax.hist(valid_pts[:, 2], bins=50, range=(0.5, 30.0), color="royalblue", edgecolor="black", lw=0.5, alpha=0.8)
ax.axvline(np.median(valid_pts[:, 2]), color="red", linestyle="--", lw=1.0, label=f"Median Depth ({np.median(valid_pts[:, 2]):.1f}m)")
ax.set_xlabel("Reconstructed Depth $Z$ (m)", fontsize=7)
ax.set_ylabel("3D Point Count", fontsize=7)
ax.set_title("Metric Depth Distribution (puncakTinggi)", fontsize=8)
ax.legend(loc="upper right", fontsize=6.5)
save_publication_figure(fig, FIG_DIR / "depth_distribution_histogram")

# 3. 3D point cloud orthographic bird's-eye-view preview (X vs Z)
fig, ax = create_figure(width=IEEE_SINGLE_COL_WIDTH, aspect_ratio=0.8)
subsample = np.random.choice(len(valid_pts), size=min(15000, len(valid_pts)), replace=False)
pts_sub = valid_pts[subsample]
cols_sub = valid_cols[subsample] / 255.0

ax.scatter(pts_sub[:, 0], pts_sub[:, 2], c=cols_sub, s=0.8, alpha=0.7)
ax.set_xlabel("Lateral Position $X$ (m)", fontsize=7)
ax.set_ylabel("Forward Depth $Z$ (m)", fontsize=7)
ax.set_title("Top-Down 3D Point Cloud Preview (X-Z Plane)", fontsize=8)
ax.set_xlim(-6.0, 6.0)
ax.set_ylim(0.0, 30.0)
ax.grid(True, linestyle=":", alpha=0.6)
save_publication_figure(fig, FIG_DIR / "pointcloud_topdown_preview")

print("3D reconstruction and scientific depth figures generated successfully!")
