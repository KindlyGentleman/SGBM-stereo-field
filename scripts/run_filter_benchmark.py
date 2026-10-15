"""Systematic Filter Benchmark across Middlebury and Gunung Terrains.

Evaluates candidate filters under identical frozen SGBM inputs:
- F0: Raw SGBM (Baseline)
- F1: Previous Bilateral Filter (Legacy implementation)
- F2: Validity-Aware Median Filter
- F3: Guided Filter
- F4: Joint Bilateral Filter
- F5: Standard WLS Filter
- F6: Confidence-Aware Edge Preservation

Evaluated against Middlebury ground truth (Bad-2.0, MAE, Coverage) and
Gunung registered Intel RealSense reference depth (MAE, Median, RMSE, Delta<1.25).
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pandas as pd
import stereo

CALIB_PATH = r"D:\02_Riset\Research Paper\source_code\calibration\calib_params\stereo_params_advanced.npz"
GUNUNG_ROOT = Path(r"D:\02_Riset\Research Paper\source_code\comparison\gunung")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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


def run_middlebury_filter_benchmark():
    """Benchmark all filter candidates across Middlebury benchmark scenes."""
    print("Running Middlebury Filter Benchmark...")
    scenes = ["Adirondack", "Motorcycle", "Playtable"]
    records = []

    matcher = stereo.create_sgbm_matcher({
        "minDisparity": 0, "numDisparities": 160, "blockSize": 7,
        "P1": 150, "P2": 1200, "disp12MaxDiff": 1, "uniquenessRatio": 10
    })
    right_matcher = cv2.ximgproc.createRightMatcher(matcher)

    filters = [
        "f0_raw",
        "f1_bilateral_legacy",
        "f2_median_valid",
        "f3_guided",
        "f4_joint_bilateral",
        "f5_wls",
        "f6_confidence_edge"
    ]

    for sc in scenes:
        sc_dir = Path(f"data/middlebury/MiddEval3/trainingQ/{sc}")
        if not sc_dir.exists():
            continue
        im0 = cv2.imread(str(sc_dir / "im0.png"))
        im1 = cv2.imread(str(sc_dir / "im1.png"))
        gt = stereo.load_pfm(str(sc_dir / "disp0GT.pfm"))
        mask = cv2.imread(str(sc_dir / "mask0nocc.png"), cv2.IMREAD_GRAYSCALE) == 255

        t0 = time.perf_counter()
        d_l_16 = matcher.compute(im0, im1)
        d_r_16 = right_matcher.compute(im1, im0)
        match_time = (time.perf_counter() - t0) * 1000.0

        disp_raw = d_l_16.astype(np.float32) / 16.0
        disp_r_raw = d_r_16.astype(np.float32) / 16.0

        for f_name in filters:
            t0 = time.perf_counter()
            f_disp = stereo.apply_filter(f_name, disp_raw, im0, disp_r_raw, matcher)
            filt_time = (time.perf_counter() - t0) * 1000.0

            bad_2 = stereo.compute_completeness_aware_bad_pixel(f_disp, gt, mask, threshold=2.0)
            eval_m = mask & (gt > 0) & (f_disp > 0)
            mae = float(np.mean(np.abs(f_disp[eval_m] - gt[eval_m]))) if np.any(eval_m) else 999.0
            cov = stereo.compute_valid_coverage(f_disp)

            records.append({
                "scene": sc,
                "filter": f_name,
                "bad_2_0_pct": bad_2,
                "mae_px": mae,
                "coverage_pct": cov,
                "runtime_ms": match_time + filt_time
            })

    df = pd.DataFrame(records)
    out_csv = RESULTS_DIR / "filter_benchmark_middlebury.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved Middlebury filter benchmark to {out_csv}")
    print(df.groupby("filter")[["bad_2_0_pct", "mae_px", "coverage_pct", "runtime_ms"]].mean().round(2))
    return df


def run_gunung_filter_benchmark():
    """Benchmark all filter candidates on Gunung against registered RealSense depth."""
    print("\nRunning Gunung RealSense Reference Depth Filter Benchmark...")
    calib = stereo.load_calibration(CALIB_PATH)
    maps_l, maps_r, _ = stereo.build_rectification_maps(calib, (1280, 720))

    terrains = ["puncakTinggi", "tanggaBiru", "tanggaKuning"]
    filters = [
        "f0_raw",
        "f1_bilateral_legacy",
        "f2_median_valid",
        "f3_guided",
        "f4_joint_bilateral",
        "f5_wls",
        "f6_confidence_edge"
    ]

    records = []

    for t in terrains:
        st_dir = GUNUNG_ROOT / t / "stereo"
        rs_dir = GUNUNG_ROOT / t / "hardware" / "realsense"

        raw_l = cv2.imread(str(st_dir / "left_0000.png"))
        raw_r = cv2.imread(str(st_dir / "right_0000.png"))
        rect_l, rect_r = stereo.rectify_pair(raw_l, raw_r, maps_l, maps_r)

        rs_rgb = cv2.imread(str(rs_dir / "rgb_0000.png"))
        rs_depth_raw = np.load(str(rs_dir / "depth_0000.npz"))["depth_map"]

        # Feature matching and PnP
        sift = cv2.SIFT_create(nfeatures=3000)
        kp_st, des_st = sift.detectAndCompute(rect_l, None)
        kp_rs, des_rs = sift.detectAndCompute(rs_rgb, None)
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(des_rs, des_st, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]

        pts_3d, pts_2d = [], []
        for m in good:
            u_rs, v_rs = kp_rs[m.queryIdx].pt
            u_i, v_i = int(round(u_rs)), int(round(v_rs))
            if 0 <= u_i < 640 and 0 <= v_i < 480:
                z_mm = rs_depth_raw[v_i, u_i]
                if 300 < z_mm < 6000:
                    z_m = z_mm / 1000.0
                    pts_3d.append([
                        (u_rs - K_RS[0, 2]) * z_m / K_RS[0, 0],
                        (v_rs - K_RS[1, 2]) * z_m / K_RS[1, 1],
                        z_m
                    ])
                    pts_2d.append([kp_st[m.trainIdx].pt[0], kp_st[m.trainIdx].pt[1]])
        pts_3d = np.array(pts_3d, dtype=np.float64)
        pts_2d = np.array(pts_2d, dtype=np.float64)

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d, pts_2d, K_RECT, None, reprojectionError=3.0, iterationsCount=3000
        )
        if not success or inliers is None:
            continue

        R_mat, _ = cv2.Rodrigues(rvec)
        rs_ref_depth = stereo.project_realsense_depth_to_stereo(
            rs_depth_raw, R_mat, tvec, K_RS, K_RECT, out_shape=(720, 1280), min_depth=0.3, max_depth=6.0
        )

        matcher = stereo.create_sgbm_matcher({
            "minDisparity": 0, "numDisparities": 160, "blockSize": 7,
            "P1": 150, "P2": 1200, "disp12MaxDiff": 1, "uniquenessRatio": 10
        })
        right_matcher = cv2.ximgproc.createRightMatcher(matcher)

        t0 = time.perf_counter()
        d_l_16 = matcher.compute(rect_l, rect_r)
        d_r_16 = right_matcher.compute(rect_r, rect_l)
        match_time = (time.perf_counter() - t0) * 1000.0

        disp_raw = d_l_16.astype(np.float32) / 16.0
        disp_r_raw = d_r_16.astype(np.float32) / 16.0

        for f_name in filters:
            t0 = time.perf_counter()
            f_disp = stereo.apply_filter(f_name, disp_raw, rect_l, disp_r_raw, matcher)
            filt_time = (time.perf_counter() - t0) * 1000.0

            pred_depth = stereo.disparity_to_depth(f_disp, focal_length_px=K_RECT[0, 0], baseline_m=0.097497)
            errs = stereo.compute_metric_depth_errors(pred_depth, rs_ref_depth, min_z=0.4, max_z=4.0)

            records.append({
                "terrain": t,
                "filter": f_name,
                "depth_mae_m": errs["mae_m"],
                "depth_median_m": errs["median_m"],
                "depth_rmse_m": errs["rmse_m"],
                "delta_125_pct": errs["delta_125_pct"],
                "ref_coverage_pct": errs["ref_coverage_pct"],
                "runtime_ms": match_time + filt_time
            })

    df = pd.DataFrame(records)
    out_csv = RESULTS_DIR / "filter_benchmark_gunung.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved Gunung filter benchmark to {out_csv}")
    print(df.groupby("filter")[["depth_mae_m", "depth_median_m", "depth_rmse_m", "delta_125_pct", "ref_coverage_pct", "runtime_ms"]].mean().round(3))
    return df


if __name__ == "__main__":
    run_middlebury_filter_benchmark()
    run_gunung_filter_benchmark()
