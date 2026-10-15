"""Sequential SGBM Optimization Experiment across Middlebury and All Four Gunung Terrains.

Stages evaluated:
- S0: Calibrated Default SGBM
- S1: Disparity Search Range (minDisparity=0, numDisparities=160 vs 128)
- S2: Matching Block Size (blockSize in [3, 5, 7, 9, 11])
- S3: Smoothness Penalties (P1=150, P2=1200 vs P1=8*3*s^2, P2=32*3*s^2)
- S4: Matching Modes (MODE_SGBM, MODE_SGBM_3WAY, MODE_HH)
- S5: Radiometric Normalization (None, CLAHE, Histogram Matching)
- S6: Invalid Match Filtering (uniquenessRatio=10, speckleWindowSize=100, speckleRange=2)
- S7: Edge-Aware Disparity WLS Filtering (lambda=8000, sigma=1.5)
"""

import time
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

# 1. Calibration loading & rectification setup
calib_path = r"D:\02_Riset\Research Paper\source_code\calibration\calib_params\stereo_params_advanced.npz"
data = np.load(calib_path)
mtx_l, dist_l = data["mtx_left"], data["dist_left"]
mtx_r, dist_r = data["mtx_right"], data["dist_right"]
R, T = data["R"], data["T"]
img_size = (1280, 720)

R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(mtx_l, dist_l, mtx_r, dist_r, img_size, R, T, alpha=0)
map_l1, map_l2 = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, img_size, cv2.CV_32FC1)
map_r1, map_r2 = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, img_size, cv2.CV_32FC1)

# 2. Load and rectify all four terrains
gunung_root = Path(r"D:\02_Riset\Research Paper\source_code\comparison\gunung")
terrains = ["puncakTinggi", "puncakWarung", "tanggaBiru", "tanggaKuning"]
rectified_terrains = {}

for t in terrains:
    st_dir = gunung_root / t / "stereo"
    raw_l = cv2.imread(str(st_dir / "left_0000.png"))
    raw_r = cv2.imread(str(st_dir / "right_0000.png"))
    rect_l = cv2.remap(raw_l, map_l1, map_l2, cv2.INTER_LINEAR)
    rect_r = cv2.remap(raw_r, map_r1, map_r2, cv2.INTER_LINEAR)
    rectified_terrains[t] = (rect_l, rect_r)

# 3. Middlebury loading helper
def load_pfm(file_path):
    with open(file_path, "rb") as f:
        header = f.readline().decode("latin-1").strip()
        color = (header == "PF")
        dim_line = f.readline().decode("latin-1").strip()
        while dim_line.startswith("#"):
            dim_line = f.readline().decode("latin-1").strip()
        width, height = map(int, dim_line.split())
        scale = float(f.readline().decode("latin-1").strip())
        endian = "<" if scale < 0 else ">"
        data = np.fromfile(f, endian + "f")
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)
        return data

mid_scenes = ["Adirondack", "Motorcycle", "Playtable"]
middlebury_pairs = {}
for sc in mid_scenes:
    sc_dir = Path(f"data/middlebury/MiddEval3/trainingQ/{sc}")
    if sc_dir.exists():
        im0 = cv2.imread(str(sc_dir / "im0.png"))
        im1 = cv2.imread(str(sc_dir / "im1.png"))
        gt = load_pfm(str(sc_dir / "disp0GT.pfm"))
        mask = cv2.imread(str(sc_dir / "mask0nocc.png"), cv2.IMREAD_GRAYSCALE)
        middlebury_pairs[sc] = (im0, im1, gt, mask)

def eval_middlebury_stage(matcher, right_matcher=None, wls_filter=None, preproc_fn=None):
    bad2_list, cov_list, mae_list = [], [], []
    for sc, (im0, im1, gt, mask) in middlebury_pairs.items():
        p0, p1 = (im0.copy(), im1.copy()) if preproc_fn is None else preproc_fn(im0, im1)
        # Compute disparity
        if wls_filter is not None and right_matcher is not None:
            disp_l = matcher.compute(p0, p1)
            disp_r = right_matcher.compute(p1, p0)
            disp_filt = wls_filter.filter(disp_l, p0, None, disp_r)
            disp = disp_filt.astype(np.float32) / 16.0
        else:
            disp_raw = matcher.compute(p0, p1)
            disp = disp_raw.astype(np.float32) / 16.0
        
        valid_pred = disp > 0
        gt_valid = (mask == 255) & (gt < 1e8) & (gt > 0)
        
        # Completeness-aware Bad-2.0
        err = np.abs(disp - gt)
        bad_err = (err > 2.0) & gt_valid & valid_pred
        missed = gt_valid & (~valid_pred)
        bad_complete = (np.sum(bad_err) + np.sum(missed)) / np.sum(gt_valid) * 100.0
        cov = np.sum(valid_pred & gt_valid) / np.sum(gt_valid) * 100.0
        mae = np.mean(err[gt_valid & valid_pred]) if np.sum(gt_valid & valid_pred) > 0 else 999.0
        
        bad2_list.append(bad_complete)
        cov_list.append(cov)
        mae_list.append(mae)
        
    return float(np.mean(bad2_list)), float(np.mean(cov_list)), float(np.mean(mae_list))

def eval_gunung_stage(matcher, right_matcher=None, wls_filter=None, preproc_fn=None):
    results_per_terrain = {}
    for t, (im0, im1) in rectified_terrains.items():
        t0 = time.perf_counter()
        p0, p1 = (im0.copy(), im1.copy()) if preproc_fn is None else preproc_fn(im0, im1)
        
        if wls_filter is not None and right_matcher is not None:
            disp_l = matcher.compute(p0, p1)
            disp_r = right_matcher.compute(p1, p0)
            disp_filt = wls_filter.filter(disp_l, p0, None, disp_r)
            disp = disp_filt.astype(np.float32) / 16.0
            r_disp = disp_r.astype(np.float32) / 16.0
        else:
            disp_raw = matcher.compute(p0, p1)
            disp = disp_raw.astype(np.float32) / 16.0
            # compute right for LR consistency check
            if right_matcher is not None:
                r_disp = right_matcher.compute(p1, p0).astype(np.float32) / 16.0
            else:
                r_disp = None

        dt_ms = (time.perf_counter() - t0) * 1000.0
        h, w = disp.shape
        valid_mask = disp > 0
        coverage = float(np.mean(valid_mask) * 100.0)
        
        # Left-Right consistency if available
        if r_disp is not None:
            x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
            x_match = np.clip(np.round(x_coords - disp).astype(int), 0, w - 1)
            r_disp_matched = r_disp[y_coords, x_match]
            lr_diff = np.abs(disp - (-r_disp_matched if np.median(r_disp) < 0 else r_disp_matched))
            lr_consistent = (lr_diff <= 1.5) & valid_mask
            lr_cons_pct = float(np.sum(lr_consistent) / (np.sum(valid_mask) + 1e-8) * 100.0)
        else:
            lr_cons_pct = 0.0

        # Photometric warp MAE
        gray_l = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray_r = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY).astype(np.float32)
        x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
        x_warp = np.clip(x_coords - disp, 0, w - 1).astype(np.float32)
        warped_r = cv2.remap(gray_r, x_warp, y_coords.astype(np.float32), cv2.INTER_LINEAR)
        warp_mae = float(np.mean(np.abs(gray_l[valid_mask] - warped_r[valid_mask]))) if np.sum(valid_mask) > 0 else 999.0

        results_per_terrain[t] = {
            "coverage": coverage,
            "lr_consistency": lr_cons_pct,
            "warp_mae": warp_mae,
            "runtime_ms": dt_ms,
            "disparity": disp
        }
    return results_per_terrain

# Build stages
stages = {}

# S0: Rectified Default SGBM
s0_l = cv2.StereoSGBM_create(minDisparity=0, numDisparities=128, blockSize=3)
s0_r = cv2.ximgproc.createRightMatcher(s0_l)
stages["S0_Default"] = {"matcher": s0_l, "right_matcher": s0_r, "filter": None, "preproc": None}

# S1: Disparity Search Range (160 px for near-field rocks)
s1_l = cv2.StereoSGBM_create(minDisparity=0, numDisparities=160, blockSize=3)
s1_r = cv2.ximgproc.createRightMatcher(s1_l)
stages["S1_SearchRange160"] = {"matcher": s1_l, "right_matcher": s1_r, "filter": None, "preproc": None}

# S2: Matching Block Size (blockSize=7)
s2_l = cv2.StereoSGBM_create(minDisparity=0, numDisparities=160, blockSize=7)
s2_r = cv2.ximgproc.createRightMatcher(s2_l)
stages["S2_BlockSize7"] = {"matcher": s2_l, "right_matcher": s2_r, "filter": None, "preproc": None}

# S3: Smoothness Penalties (P1=150, P2=1200)
s3_l = cv2.StereoSGBM_create(minDisparity=0, numDisparities=160, blockSize=7, P1=150, P2=1200)
s3_r = cv2.ximgproc.createRightMatcher(s3_l)
stages["S3_Penalties"] = {"matcher": s3_l, "right_matcher": s3_r, "filter": None, "preproc": None}

# S4: Matching Modes (MODE_SGBM_3WAY)
s4_l = cv2.StereoSGBM_create(minDisparity=0, numDisparities=160, blockSize=7, P1=150, P2=1200, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
s4_r = cv2.ximgproc.createRightMatcher(s4_l)
stages["S4_Mode3Way"] = {"matcher": s4_l, "right_matcher": s4_r, "filter": None, "preproc": None}

# S5: Radiometric Normalization (Histogram Matching)
def hist_match_fn(im0, im1):
    h0 = cv2.cvtColor(im0, cv2.COLOR_BGR2YCrCb)
    h1 = cv2.cvtColor(im1, cv2.COLOR_BGR2YCrCb)
    y0, y1 = h0[:, :, 0], h1[:, :, 0]
    hist0, _ = np.histogram(y0.flatten(), 256, [0, 256])
    hist1, _ = np.histogram(y1.flatten(), 256, [0, 256])
    cdf0 = hist0.cumsum() / (hist0.sum() + 1e-8)
    cdf1 = hist1.cumsum() / (hist1.sum() + 1e-8)
    lut = np.zeros(256, dtype=np.uint8)
    g_j = 0
    for g_i in range(256):
        while g_j < 255 and cdf1[g_j] < cdf0[g_i]:
            g_j += 1
        lut[g_i] = g_j
    h1_matched = cv2.LUT(y1, lut)
    h1[:, :, 0] = h1_matched
    return im0, cv2.cvtColor(h1, cv2.COLOR_YCrCb2BGR)

stages["S5_HistMatch"] = {"matcher": s4_l, "right_matcher": s4_r, "filter": None, "preproc": hist_match_fn}

# S6: Invalid Match Filtering (uniqueness=10, speckle=100)
s6_l = cv2.StereoSGBM_create(
    minDisparity=0, numDisparities=160, blockSize=7, P1=150, P2=1200,
    uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)
s6_r = cv2.ximgproc.createRightMatcher(s6_l)
stages["S6_PostFilters"] = {"matcher": s6_l, "right_matcher": s6_r, "filter": None, "preproc": None}

# S7: Edge-Aware Disparity WLS Filtering
wls_filt = cv2.ximgproc.createDisparityWLSFilter(s6_l)
wls_filt.setLambda(8000.0)
wls_filt.setSigmaColor(1.5)
stages["S7_WLSFilter"] = {"matcher": s6_l, "right_matcher": s6_r, "filter": wls_filt, "preproc": None}

print("=" * 105)
print("SEQUENTIAL SGBM OPTIMIZATION STUDY ACROSS MIDDLEBURY AND FOUR GUNUNG TERRAINS")
print("=" * 105)

header = f"{'Stage':<18} | {'Mid Bad-2':<9} | {'Mid Cov':<7} | {'Mid MAE':<7} | {'Gun Cov':<7} | {'Gun LR Cons':<11} | {'Gun Warp MAE':<12} | {'Time ms':<7}"
print(header)
print("-" * 105)

all_rows = []
saved_disparities = {}

for st_name, cfg in stages.items():
    # Evaluate Middlebury
    mid_bad, mid_cov, mid_mae = eval_middlebury_stage(
        cfg["matcher"], cfg["right_matcher"], cfg["filter"], cfg["preproc"]
    )
    # Evaluate Gunung all 4 terrains
    gun_res = eval_gunung_stage(
        cfg["matcher"], cfg["right_matcher"], cfg["filter"], cfg["preproc"]
    )
    
    # Average Gunung metrics
    gun_cov = float(np.mean([gun_res[t]["coverage"] for t in terrains]))
    gun_lr = float(np.mean([gun_res[t]["lr_consistency"] for t in terrains]))
    gun_warp = float(np.mean([gun_res[t]["warp_mae"] for t in terrains]))
    gun_time = float(np.mean([gun_res[t]["runtime_ms"] for t in terrains]))
    
    print(f"{st_name:<18} | {mid_bad:<9.2f}% | {mid_cov:<7.1f}% | {mid_mae:<7.2f} | {gun_cov:<7.1f}% | {gun_lr:<11.1f}% | {gun_warp:<12.2f} | {gun_time:<7.1f}")
    
    row = {
        "stage": st_name,
        "mid_bad2_pct": mid_bad,
        "mid_cov_pct": mid_cov,
        "mid_mae_px": mid_mae,
        "gunung_cov_pct": gun_cov,
        "gunung_lr_consistency_pct": gun_lr,
        "gunung_warp_mae": gun_warp,
        "runtime_ms": gun_time
    }
    # Per-terrain breakdown
    for t in terrains:
        row[f"{t}_cov"] = gun_res[t]["coverage"]
        row[f"{t}_lr_cons"] = gun_res[t]["lr_consistency"]
        row[f"{t}_warp"] = gun_res[t]["warp_mae"]
        row[f"{t}_time"] = gun_res[t]["runtime_ms"]
        # Save disparity for visualization
        if st_name in ["S0_Default", "S4_Mode3Way", "S6_PostFilters", "S7_WLSFilter"]:
            saved_disparities[f"{t}_{st_name}"] = gun_res[t]["disparity"]
            
    all_rows.append(row)

df = pd.DataFrame(all_rows)
out_dir = Path("results/sequential_study")
out_dir.mkdir(parents=True, exist_ok=True)
df.to_csv(out_dir / "stages_comparison.csv", index=False)
print("=" * 105)
print(f"Results saved to {out_dir / 'stages_comparison.csv'}")

# Save sample disparities as .npy for plotting
disp_dir = out_dir / "disparities"
disp_dir.mkdir(exist_ok=True)
for k, v in saved_disparities.items():
    np.save(disp_dir / f"{k}.npy", v)

print(f"Saved {len(saved_disparities)} disparity fields for publication figures.")
