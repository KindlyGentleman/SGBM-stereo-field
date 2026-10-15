"""Audit and recompute V5 baseline metrics under a strictly fixed protocol.

Ensures transparent accounting for:
- Middlebury: fixed non-occluded mask (mask0nocc == 255 & gt > 0 & gt < 1e8)
- Gunung: fixed RealSense reference mask (0.4 m <= Z_ref <= 4.0 m)
- Both conditional error on valid predictions and completeness-aware error
- Mean vs Median metrics
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
WORK_DIR = Path("work")

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


def audit_middlebury():
    print("=" * 80)
    print("AUDITING V5 BASELINE: MIDDLEBURY BENCHMARK (FIXED NON-OCCLUDED MASK)")
    print("=" * 80)
    scenes = ["Adirondack", "Motorcycle", "Playtable"]
    filters = ["f0_raw", "f1_bilateral_legacy", "f2_median_valid", "f3_guided", "f4_joint_bilateral", "f5_wls", "f6_confidence_edge"]

    matcher = stereo.create_sgbm_matcher({
        "minDisparity": 0, "numDisparities": 160, "blockSize": 7,
        "P1": 150, "P2": 1200, "disp12MaxDiff": 1, "uniquenessRatio": 10
    })
    r_matcher = cv2.ximgproc.createRightMatcher(matcher)

    rows = []
    for sc in scenes:
        sc_dir = Path(f"data/middlebury/MiddEval3/trainingQ/{sc}")
        im0 = cv2.imread(str(sc_dir / "im0.png"))
        im1 = cv2.imread(str(sc_dir / "im1.png"))
        gt = stereo.load_pfm(str(sc_dir / "disp0GT.pfm"))
        mask = cv2.imread(str(sc_dir / "mask0nocc.png"), cv2.IMREAD_GRAYSCALE) == 255
        gt_valid = mask & (gt > 0) & (gt < 1e8)
        n_gt = np.sum(gt_valid)

        d_l_16 = matcher.compute(im0, im1)
        d_r_16 = r_matcher.compute(im1, im0)
        disp_raw = d_l_16.astype(np.float32) / 16.0
        disp_r_raw = d_r_16.astype(np.float32) / 16.0

        for f_name in filters:
            f_disp = stereo.apply_filter(f_name, disp_raw, im0, disp_r_raw, matcher)
            pred_valid = f_disp > 0
            joint = gt_valid & pred_valid
            n_joint = np.sum(joint)

            cov_on_gt = n_joint / n_gt * 100.0
            overall_cov = np.sum(pred_valid) / f_disp.size * 100.0

            # Completeness-aware Bad-2.0
            err_all = np.abs(f_disp - gt)
            bad_err = joint & (err_all > 2.0)
            missed = gt_valid & (~pred_valid)
            bad_complete = (np.sum(bad_err) + np.sum(missed)) / n_gt * 100.0

            # Conditional metrics on joint
            if n_joint > 0:
                cond_err = err_all[joint]
                cond_mae = float(np.mean(cond_err))
                cond_med = float(np.median(cond_err))
                cond_bad2 = float(np.sum(cond_err > 2.0) / n_joint * 100.0)
            else:
                cond_mae = cond_med = cond_bad2 = np.nan

            rows.append({
                "scene": sc,
                "filter": f_name,
                "n_gt": n_gt,
                "n_joint": n_joint,
                "cov_on_gt_pct": cov_on_gt,
                "overall_cov_pct": overall_cov,
                "bad_complete_pct": bad_complete,
                "cond_bad2_pct": cond_bad2,
                "cond_mae_px": cond_mae,
                "cond_med_px": cond_med
            })

    df = pd.DataFrame(rows)
    print("\nSummary by Filter (Middlebury Dev Average):")
    summary = df.groupby("filter")[[
        "cov_on_gt_pct", "bad_complete_pct", "cond_bad2_pct", "cond_mae_px", "cond_med_px"
    ]].mean().round(2)
    print(summary)
    df.to_csv(RESULTS_DIR / "audit_middlebury_fixed_protocol.csv", index=False)
    return df


def audit_gunung():
    print("\n" + "=" * 80)
    print("AUDITING V5 BASELINE: GUNUNG TERRAINS (FIXED REALSENSE REFERENCE MASK)")
    print("=" * 80)
    calib = stereo.load_calibration(CALIB_PATH)
    maps_l, maps_r, _ = stereo.build_rectification_maps(calib, (1280, 720))

    terrains = ["puncakTinggi", "tanggaBiru", "tanggaKuning", "puncakWarung"]
    filters = ["f0_raw", "f1_bilateral_legacy", "f2_median_valid", "f3_guided", "f4_joint_bilateral", "f5_wls", "f6_confidence_edge"]

    matcher = stereo.create_sgbm_matcher({
        "minDisparity": 0, "numDisparities": 160, "blockSize": 7,
        "P1": 150, "P2": 1200, "disp12MaxDiff": 1, "uniquenessRatio": 10
    })
    r_matcher = cv2.ximgproc.createRightMatcher(matcher)

    rows = []
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

        if len(pts_3d) < 6:
            print(f"[{t}] RealSense reference registration omitted (insufficient near-field features).")
            for f_name in filters:
                rows.append({
                    "terrain": t,
                    "filter": f_name,
                    "ref_status": "OMITTED_OUT_OF_RANGE",
                    "n_ref_eval": 0,
                    "n_joint": 0,
                    "ref_cov_pct": 0.0,
                    "complete_mae_m": np.nan,
                    "cond_mae_m": np.nan,
                    "cond_med_m": np.nan,
                    "cond_rmse_m": np.nan,
                    "delta_125_pct": np.nan
                })
            continue

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d, pts_2d, K_RECT, None, reprojectionError=3.0, iterationsCount=3000
        )
        if not success or inliers is None:
            continue

        R_mat, _ = cv2.Rodrigues(rvec)
        rs_ref_depth = stereo.project_realsense_depth_to_stereo(
            rs_depth_raw, R_mat, tvec, K_RS, K_RECT, (720, 1280), 0.3, 6.0
        )

        # Fixed evaluation reference mask on RealSense (0.4 m <= Z <= 4.0 m)
        ref_eval_mask = (rs_ref_depth >= 0.4) & (rs_ref_depth <= 4.0)
        n_ref_eval = np.sum(ref_eval_mask)

        d_l_16 = matcher.compute(rect_l, rect_r)
        d_r_16 = r_matcher.compute(rect_r, rect_l)
        disp_raw = d_l_16.astype(np.float32) / 16.0
        disp_r_raw = d_r_16.astype(np.float32) / 16.0

        for f_name in filters:
            f_disp = stereo.apply_filter(f_name, disp_raw, rect_l, disp_r_raw, matcher)
            pred_depth = stereo.disparity_to_depth(f_disp, focal_length_px=K_RECT[0, 0], baseline_m=0.097497)

            pred_valid_on_ref = (pred_depth >= 0.4) & (pred_depth <= 6.0)
            joint = ref_eval_mask & pred_valid_on_ref
            n_joint = np.sum(joint)
            ref_cov = n_joint / n_ref_eval * 100.0

            # Completeness-aware error: missing predictions assigned a fixed penalty threshold (1.0 m)
            err_m = np.abs(pred_depth - rs_ref_depth)
            err_complete = np.where(joint, err_m, 1.0)
            complete_mae = float(np.mean(err_complete[ref_eval_mask]))

            if n_joint > 0:
                cond_err = err_m[joint]
                cond_mae = float(np.mean(cond_err))
                cond_med = float(np.median(cond_err))
                cond_rmse = float(np.sqrt(np.mean(cond_err ** 2)))
                z_p = pred_depth[joint]
                z_r = rs_ref_depth[joint]
                d125 = float(np.mean(np.maximum(z_p / z_r, z_r / z_p) < 1.25) * 100.0)
            else:
                cond_mae = cond_med = cond_rmse = d125 = np.nan

            rows.append({
                "terrain": t,
                "filter": f_name,
                "ref_status": "VERIFIED",
                "n_ref_eval": n_ref_eval,
                "n_joint": n_joint,
                "ref_cov_pct": ref_cov,
                "complete_mae_m": complete_mae,
                "cond_mae_m": cond_mae,
                "cond_med_m": cond_med,
                "cond_rmse_m": cond_rmse,
                "delta_125_pct": d125
            })

    df = pd.DataFrame(rows)
    print("\nSummary by Filter (Gunung Verified Terrains Average):")
    valid_df = df[df["ref_status"] == "VERIFIED"]
    summary = valid_df.groupby("filter")[[
        "ref_cov_pct", "complete_mae_m", "cond_mae_m", "cond_med_m", "cond_rmse_m", "delta_125_pct"
    ]].mean().round(3)
    print(summary)
    df.to_csv(RESULTS_DIR / "audit_gunung_fixed_protocol.csv", index=False)
    return df


if __name__ == "__main__":
    audit_middlebury()
    audit_gunung()
