"""Validate the Pareto frontier (FAST, BALANCED, QUALITY) across all Middlebury and Gunung datasets.

Evaluates:
- Middlebury Dev scenes (Adirondack, Motorcycle, Playtable)
- Middlebury Held-out scenes (Jadeplant, Piano, Pipes)
- All 4 Gunung terrains (puncakTinggi, puncakWarung, tanggaBiru, tanggaKuning)
Records:
- Disparity Bad-2.0 (%), Disparity MAE (px)
- Metric Depth MAE (m), Median Error (m), RMSE (m), Delta<1.25 (%)
- Prediction Coverage (%)
- Latency P50 and P95 (ms)
- Process CPU Time (ms)
- Peak Memory (MB)
"""

import sys
import time
import tracemalloc
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

CONFIGS = {
    "FAST": {
        "scale": 0.5,
        "num_disparities": 80,
        "block_size": 5,
        "P1": 80,
        "P2": 640,
        "use_wls": False,
        "filter": "median"
    },
    "BALANCED": {
        "scale": 0.5,
        "num_disparities": 80,
        "block_size": 5,
        "P1": 80,
        "P2": 640,
        "use_wls": True,
        "wls_lambda": 4000.0,
        "wls_sigma": 1.2
    },
    "QUALITY": {
        "scale": 1.0,
        "num_disparities": 160,
        "block_size": 7,
        "P1": 150,
        "P2": 1200,
        "use_wls": True,
        "wls_lambda": 8000.0,
        "wls_sigma": 1.5
    }
}


def run_pipeline(cfg_name: str, left_img: np.ndarray, right_img: np.ndarray):
    """Execute a single pipeline configuration and measure latency and memory."""
    cfg = CONFIGS[cfg_name]
    scale = cfg["scale"]
    w_orig, h_orig = left_img.shape[1], left_img.shape[0]

    tracemalloc.start()
    t0 = time.perf_counter()
    c0 = time.process_time()

    if scale != 1.0:
        w_s, h_s = int(w_orig * scale), int(h_orig * scale)
        l_in = cv2.resize(left_img, (w_s, h_s), interpolation=cv2.INTER_AREA)
        r_in = cv2.resize(right_img, (w_s, h_s), interpolation=cv2.INTER_AREA)
    else:
        l_in, r_in = left_img, right_img

    matcher = stereo.create_sgbm_matcher({
        "minDisparity": 0,
        "numDisparities": cfg["num_disparities"],
        "blockSize": cfg["block_size"],
        "P1": cfg["P1"],
        "P2": cfg["P2"],
        "disp12MaxDiff": 1,
        "uniquenessRatio": 10,
        "speckleWindowSize": 100,
        "speckleRange": 2,
        "mode": cv2.STEREO_SGBM_MODE_SGBM_3WAY
    })

    if cfg["use_wls"]:
        r_matcher = cv2.ximgproc.createRightMatcher(matcher)
        dl = matcher.compute(l_in, r_in)
        dr = r_matcher.compute(r_in, l_in)
        wls = cv2.ximgproc.createDisparityWLSFilter(matcher)
        wls.setLambda(cfg["wls_lambda"])
        wls.setSigmaColor(cfg["wls_sigma"])
        d_filt = wls.filter(dl, l_in, None, dr).astype(np.float32) / 16.0
    else:
        dl = matcher.compute(l_in, r_in)
        d_raw = dl.astype(np.float32) / 16.0
        d_clean = np.where(d_raw > 0, d_raw, 0.0)
        # Fast validity-aware median filter
        med = cv2.medianBlur(d_clean, 3)
        d_filt = np.where(d_clean > 0, med, 0.0)

    d_filt[d_filt <= 0] = 0.0

    if scale != 1.0:
        d_out = cv2.resize(d_filt, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR) * (1.0 / scale)
    else:
        d_out = d_filt

    wall_ms = (time.perf_counter() - t0) * 1000.0
    cpu_ms = (time.process_time() - c0) * 1000.0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return d_out, wall_ms, cpu_ms, peak_bytes / (1024 * 1024)


def validate_middlebury():
    print("=" * 80)
    print("VALIDATING PARETO FRONTIER ON MIDDLEBURY 2014")
    print("=" * 80)

    scenes = {
        "Adirondack": "dev",
        "Motorcycle": "dev",
        "Playtable": "dev",
        "Jadeplant": "heldout",
        "Piano": "heldout",
        "Pipes": "heldout"
    }

    records = []
    for sc, split in scenes.items():
        sc_dir = Path(f"data/middlebury/MiddEval3/trainingQ/{sc}")
        if not sc_dir.exists():
            continue
        im0 = cv2.imread(str(sc_dir / "im0.png"))
        im1 = cv2.imread(str(sc_dir / "im1.png"))
        gt = stereo.load_pfm(str(sc_dir / "disp0GT.pfm"))
        mask = cv2.imread(str(sc_dir / "mask0nocc.png"), cv2.IMREAD_GRAYSCALE) == 255

        for cfg_name in ["FAST", "BALANCED", "QUALITY"]:
            times = []
            for _ in range(3):
                disp, wall_ms, cpu_ms, peak_mb = run_pipeline(cfg_name, im0, im1)
                times.append(wall_ms)

            bad2 = stereo.compute_completeness_aware_bad_pixel(disp, gt, mask, 2.0)
            eval_m = mask & (gt > 0) & (gt < 1e8) & (disp > 0)
            mae = float(np.mean(np.abs(disp[eval_m] - gt[eval_m]))) if np.any(eval_m) else 999.0
            cov = stereo.compute_valid_coverage(disp)

            records.append({
                "dataset": "Middlebury",
                "scene": sc,
                "split": split,
                "config": cfg_name,
                "bad_2_0_pct": bad2,
                "mae_px": mae,
                "coverage_pct": cov,
                "latency_p50_ms": float(np.median(times)),
                "latency_p95_ms": float(np.percentile(times, 95)),
                "process_cpu_ms": cpu_ms,
                "peak_mem_mb": peak_mb
            })

    df = pd.DataFrame(records)
    print(df.groupby(["split", "config"])[["bad_2_0_pct", "mae_px", "coverage_pct", "latency_p50_ms"]].mean().round(2))
    return df


def validate_gunung():
    print("\n" + "=" * 80)
    print("VALIDATING PARETO FRONTIER ON ALL FOUR GUNUNG TERRAINS")
    print("=" * 80)

    calib = stereo.load_calibration(CALIB_PATH)
    maps_l, maps_r, _ = stereo.build_rectification_maps(calib, (1280, 720))

    terrains = ["puncakTinggi", "puncakWarung", "tanggaBiru", "tanggaKuning"]
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

        has_realsense = False
        rs_ref_depth = None
        if len(pts_3d) >= 6:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts_3d, pts_2d, K_RECT, None, reprojectionError=3.0, iterationsCount=3000
            )
            if success and inliers is not None:
                has_realsense = True
                R_mat, _ = cv2.Rodrigues(rvec)
                rs_ref_depth = stereo.project_realsense_depth_to_stereo(
                    rs_depth_raw, R_mat, tvec, K_RS, K_RECT, (720, 1280), 0.3, 6.0
                )

        for cfg_name in ["FAST", "BALANCED", "QUALITY"]:
            times = []
            for _ in range(3):
                disp, wall_ms, cpu_ms, peak_mb = run_pipeline(cfg_name, rect_l, rect_r)
                times.append(wall_ms)

            cov = stereo.compute_valid_coverage(disp)
            pred_depth = stereo.disparity_to_depth(disp, K_RECT[0, 0], 0.0975)

            if has_realsense and rs_ref_depth is not None:
                errs = stereo.compute_metric_depth_errors(pred_depth, rs_ref_depth, 0.4, 4.0)
                mae_m = errs["mae_m"]
                med_m = errs["median_m"]
                rmse_m = errs["rmse_m"]
                d125 = errs["delta_125_pct"]
                ref_cov = errs["ref_coverage_pct"]
                status = "VERIFIED"
            else:
                mae_m = med_m = rmse_m = d125 = ref_cov = np.nan
                status = "OMITTED_OUT_OF_RANGE"

            records.append({
                "dataset": "Gunung",
                "scene": t,
                "config": cfg_name,
                "reference_status": status,
                "coverage_pct": cov,
                "depth_mae_m": mae_m,
                "depth_med_m": med_m,
                "depth_rmse_m": rmse_m,
                "delta_125_pct": d125,
                "ref_coverage_pct": ref_cov,
                "latency_p50_ms": float(np.median(times)),
                "latency_p95_ms": float(np.percentile(times, 95)),
                "process_cpu_ms": cpu_ms,
                "peak_mem_mb": peak_mb
            })

    df = pd.DataFrame(records)
    print("\nGunung Results by Config:")
    valid_g = df[df["reference_status"] == "VERIFIED"]
    print(valid_g.groupby("config")[["depth_mae_m", "depth_med_m", "delta_125_pct", "ref_coverage_pct", "latency_p50_ms", "peak_mem_mb"]].mean().round(3))
    return df


def main():
    df_mid = validate_middlebury()
    df_gun = validate_gunung()

    df_all = pd.concat([df_mid, df_gun], ignore_index=True)
    df_all.to_csv(RESULTS_DIR / "pareto_frontier.csv", index=False)
    print(f"\nSaved combined Pareto frontier to {RESULTS_DIR / 'pareto_frontier.csv'}")


if __name__ == "__main__":
    main()
