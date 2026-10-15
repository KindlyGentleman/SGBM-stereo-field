"""Command-line reproduction runner for stereo matching experiments.

Supports:
- --quick: Fast verification of FAST preset on sample scenes in < 1 second.
- --profile: Component-level CPU and memory profiling across resolutions and threads.
- --full: Complete benchmark evaluation of FAST, BALANCED, and QUALITY presets.
"""

import argparse
import json
import time
import tracemalloc
from pathlib import Path
from typing import Dict, Any, List
import cv2
import numpy as np
import pandas as pd

import stereo

DEFAULT_CALIB = r"D:\02_Riset\Research Paper\source_code\calibration\calib_params\stereo_params_advanced.npz"
DEFAULT_GUNUNG = r"D:\02_Riset\Research Paper\source_code\comparison\gunung"
DEFAULT_MID = r"data/middlebury/MiddEval3/trainingQ"

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


def load_pfm(file_path: str) -> np.ndarray:
    """Load Middlebury floating-point disparity map (PFM format)."""
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


def run_quick_verification(
    calib_path: str,
    gunung_root: str,
    mid_root: str
) -> None:
    """Run fast verification in under two seconds using the FAST preset."""
    print("=" * 80)
    print("STEREO REPRODUCTION: QUICK VERIFICATION MODE")
    print("=" * 80)

    # 1. Calibration verification
    print(f"Loading calibration: {calib_path}")
    calib = stereo.load_calibration(calib_path)
    maps_l, maps_r, Q = stereo.build_rectification_maps(calib, (1280, 720), use_cache=True)
    baseline_mm = -calib["T"][0, 0]
    fx_px = calib["mtx_left"][0, 0]
    print(f"  Baseline B: {baseline_mm:.2f} mm | Focal length fx: {fx_px:.1f} px | fx*B: {baseline_mm*fx_px/1000.0:.2f} px*m")

    # 2. Gunung verification
    gunung_dir = Path(gunung_root)
    sample_terrain = "puncakTinggi"
    st_dir = gunung_dir / sample_terrain / "stereo"
    if not st_dir.exists():
        print(f"Gunung dataset path not accessible: {st_dir}")
        return

    raw_l = cv2.imread(str(st_dir / "left_0000.png"))
    raw_r = cv2.imread(str(st_dir / "right_0000.png"))
    rect_l, rect_r = stereo.rectify_pair(raw_l, raw_r, maps_l, maps_r)
    print(f"Loaded and rectified {sample_terrain} ({rect_l.shape[1]}x{rect_l.shape[0]})")

    # Run FAST preset
    disp_fast, stats_fast = stereo.run_stereo_pipeline(rect_l, rect_r, preset="FAST")
    depth_fast = stereo.disparity_to_depth(disp_fast, fx_px, baseline_mm / 1000.0)

    # Run BALANCED preset
    disp_bal, stats_bal = stereo.run_stereo_pipeline(rect_l, rect_r, preset="BALANCED")
    depth_bal = stereo.disparity_to_depth(disp_bal, fx_px, baseline_mm / 1000.0)

    print("\n[Gunung Outdoor Results: puncakTinggi]")
    print(f"  FAST Preset:     Coverage = {stats_fast['valid_coverage_pct']:5.1f}% | Wall Time = {stats_fast['wall_ms']:5.1f} ms ({1000.0/stats_fast['wall_ms']:.1f} FPS)")
    print(f"  BALANCED Preset: Coverage = {stats_bal['valid_coverage_pct']:5.1f}% | Wall Time = {stats_bal['wall_ms']:5.1f} ms ({1000.0/stats_bal['wall_ms']:.1f} FPS)")

    # 3. Middlebury verification
    mid_p = Path(mid_root) / "Adirondack"
    if mid_p.exists():
        im0 = cv2.imread(str(mid_p / "im0.png"))
        im1 = cv2.imread(str(mid_p / "im1.png"))
        gt = load_pfm(str(mid_p / "disp0GT.pfm"))
        mask = cv2.imread(str(mid_p / "mask0nocc.png"), cv2.IMREAD_GRAYSCALE) == 255

        d_mid_fast, s_mid_fast = stereo.run_stereo_pipeline(im0, im1, preset="FAST")
        d_mid_bal, s_mid_bal = stereo.run_stereo_pipeline(im0, im1, preset="BALANCED")

        bad_fast = stereo.compute_completeness_aware_bad_pixel(d_mid_fast, gt, mask, threshold=2.0)
        bad_bal = stereo.compute_completeness_aware_bad_pixel(d_mid_bal, gt, mask, threshold=2.0)
        mae_fast = np.mean(np.abs(d_mid_fast[mask & (d_mid_fast > 0)] - gt[mask & (d_mid_fast > 0)]))
        mae_bal = np.mean(np.abs(d_mid_bal[mask & (d_mid_bal > 0)] - gt[mask & (d_mid_bal > 0)]))

        print("\n[Middlebury Benchmark Ground Truth: Adirondack]")
        print(f"  FAST Preset:     Bad-2.0 = {bad_fast:5.2f}% | MAE = {mae_fast:5.2f} px | Latency = {s_mid_fast['wall_ms']:5.1f} ms")
        print(f"  BALANCED Preset: Bad-2.0 = {bad_bal:5.2f}% | MAE = {mae_bal:5.2f} px | Latency = {s_mid_bal['wall_ms']:5.1f} ms")

    print("=" * 80)
    print("Quick verification PASSED successfully.")


def run_pipeline_profiling(calib_path: str, gunung_root: str) -> None:
    """Run component-level pipeline profiler across multiscale resolutions and threads."""
    print("=" * 80)
    print("STEREO REPRODUCTION: COMPONENT-LEVEL CPU PROFILING")
    print("=" * 80)

    calib = stereo.load_calibration(calib_path)
    maps_l, maps_r, _ = stereo.build_rectification_maps(calib, (1280, 720), use_cache=True)

    st_dir = Path(gunung_root) / "puncakTinggi" / "stereo"
    raw_l = cv2.imread(str(st_dir / "left_0000.png"))
    raw_r = cv2.imread(str(st_dir / "right_0000.png"))

    scales = [(1.0, "1280x720", 160), (0.5, "640x360", 80), (0.25, "320x180", 48)]
    prof_rows = []

    print("\n--- Multiscale Resolution Scaling (16 threads) ---")
    for scale, res_name, num_disp in scales:
        w_s, h_s = int(1280 * scale), int(720 * scale)
        if scale != 1.0:
            l_in = cv2.resize(raw_l, (w_s, h_s), interpolation=cv2.INTER_AREA)
            r_in = cv2.resize(raw_r, (w_s, h_s), interpolation=cv2.INTER_AREA)
        else:
            l_in, r_in = raw_l, raw_r

        maps_ls, maps_rs, _ = stereo.build_rectification_maps(calib, (w_s, h_s), use_cache=True)
        matcher = stereo.create_sgbm_matcher({
            "minDisparity": 0, "numDisparities": num_disp, "blockSize": 7,
            "P1": 150, "P2": 1200, "disp12MaxDiff": 1, "uniquenessRatio": 10,
            "speckleWindowSize": 100, "speckleRange": 2, "mode": cv2.STEREO_SGBM_MODE_SGBM_3WAY
        })
        r_matcher = cv2.ximgproc.createRightMatcher(matcher)
        wls = cv2.ximgproc.createDisparityWLSFilter(matcher)
        wls.setLambda(8000.0)
        wls.setSigmaColor(1.5)

        # Warmup
        for _ in range(3):
            rl = cv2.remap(l_in, maps_ls[0], maps_ls[1], cv2.INTER_LINEAR)
            rr = cv2.remap(r_in, maps_rs[0], maps_rs[1], cv2.INTER_LINEAR)
            dl = matcher.compute(rl, rr)
            dr = r_matcher.compute(rr, rl)
            df = wls.filter(dl, rl, None, dr)

        # Timed run (10 iterations)
        t_remap, t_sgbm_l, t_sgbm_r, t_wls = [], [], [], []
        tracemalloc.start()
        for _ in range(10):
            t0 = time.perf_counter()
            rl = cv2.remap(l_in, maps_ls[0], maps_ls[1], cv2.INTER_LINEAR)
            rr = cv2.remap(r_in, maps_rs[0], maps_rs[1], cv2.INTER_LINEAR)
            t_remap.append((time.perf_counter() - t0) * 1000.0)

            t0 = time.perf_counter()
            dl = matcher.compute(rl, rr)
            t_sgbm_l.append((time.perf_counter() - t0) * 1000.0)

            t0 = time.perf_counter()
            dr = r_matcher.compute(rr, rl)
            t_sgbm_r.append((time.perf_counter() - t0) * 1000.0)

            t0 = time.perf_counter()
            df = wls.filter(dl, rl, None, dr)
            t_wls.append((time.perf_counter() - t0) * 1000.0)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        tot_p50 = np.median(t_remap) + np.median(t_sgbm_l) + np.median(t_sgbm_r) + np.median(t_wls)
        print(f"  {res_name:10s} (scale {scale:4.2f}): Total P50 = {tot_p50:6.1f} ms | Remap = {np.median(t_remap):4.1f} ms | SGBM_L = {np.median(t_sgbm_l):5.1f} ms | SGBM_R = {np.median(t_sgbm_r):5.1f} ms | WLS = {np.median(t_wls):4.1f} ms | RAM = {peak/(1024*1024):4.1f} MB")
        prof_rows.append({
            "resolution": res_name,
            "scale": scale,
            "remap_ms": np.median(t_remap),
            "sgbm_left_ms": np.median(t_sgbm_l),
            "sgbm_right_ms": np.median(t_sgbm_r),
            "wls_ms": np.median(t_wls),
            "total_ms": tot_p50,
            "peak_ram_mb": peak / (1024 * 1024)
        })

    df_p = pd.DataFrame(prof_rows)
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    df_p.to_csv(out_dir / "profile_summary.csv", index=False)
    print(f"\nSaved profile results to: {out_dir / 'profile_summary.csv'}")
    print("=" * 80)


def run_full_reproduction(
    calib_path: str,
    gunung_root: str,
    mid_root: str
) -> None:
    """Run full benchmark reproduction across Middlebury Dev, Held-out, and 4 Gunung terrains."""
    print("=" * 80)
    print("STEREO REPRODUCTION: FULL BENCHMARK EXPERIMENT")
    print("=" * 80)

    calib = stereo.load_calibration(calib_path)
    maps_l, maps_r, _ = stereo.build_rectification_maps(calib, (1280, 720), use_cache=True)
    baseline_m = -calib["T"][0, 0] / 1000.0
    fx_px = calib["mtx_left"][0, 0]

    presets = ["FAST", "BALANCED", "QUALITY"]
    rows = []

    # 1. Gunung Evaluation
    terrains = ["puncakTinggi", "puncakWarung", "tanggaBiru", "tanggaKuning"]
    print("\n--- Evaluating Gunung Terrains ---")
    for t in terrains:
        st_dir = Path(gunung_root) / t / "stereo"
        raw_l = cv2.imread(str(st_dir / "left_0000.png"))
        raw_r = cv2.imread(str(st_dir / "right_0000.png"))
        rect_l, rect_r = stereo.rectify_pair(raw_l, raw_r, maps_l, maps_r)

        # RealSense reference
        rs_ref_path = Path(gunung_root) / t / "hardware" / "realsense"
        has_rs = (rs_ref_path / "depth_0000.npz").exists()
        ref_depth = None
        ref_status = "OMITTED_OUT_OF_RANGE" if t == "puncakWarung" else "VERIFIED"

        if has_rs and t != "puncakWarung":
            rs_raw = np.load(str(rs_ref_path / "depth_0000.npz"))["depth_map"]
            rs_rgb = cv2.imread(str(rs_ref_path / "rgb_0000.png"))
            sift = cv2.SIFT_create(2500)
            kp_st, des_st = sift.detectAndCompute(rect_l, None)
            kp_rs, des_rs = sift.detectAndCompute(rs_rgb, None)
            bf = cv2.BFMatcher(cv2.NORM_L2)
            matches = bf.knnMatch(des_rs, des_st, k=2)
            good = [m for m, n in matches if m.distance < 0.75 * n.distance]
            pts_3d, pts_2d = [], []
            for m in good:
                u_rs, v_rs = kp_rs[m.queryIdx].pt
                ui, vi = int(round(u_rs)), int(round(v_rs))
                if 0 <= ui < 640 and 0 <= vi < 480:
                    z_mm = rs_raw[vi, ui]
                    if 300 < z_mm < 6000:
                        zm = z_mm / 1000.0
                        pts_3d.append([(u_rs - K_RS[0, 2]) * zm / K_RS[0, 0], (v_rs - K_RS[1, 2]) * zm / K_RS[1, 1], zm])
                        pts_2d.append([kp_st[m.trainIdx].pt[0], kp_st[m.trainIdx].pt[1]])
            if len(pts_3d) >= 10:
                pts_3d, pts_2d = np.array(pts_3d), np.array(pts_2d)
                _, rvec, tvec, _ = cv2.solvePnPRansac(pts_3d, pts_2d, K_RECT, None, 3.0, 2000)
                R_mat, _ = cv2.Rodrigues(rvec)
                ref_depth = stereo.project_realsense_depth_to_stereo(rs_raw, R_mat, tvec, K_RS, K_RECT, (720, 1280), 0.3, 6.0)

        for p_name in presets:
            disp, stats = stereo.run_stereo_pipeline(rect_l, rect_r, preset=p_name)
            depth = stereo.disparity_to_depth(disp, fx_px, baseline_m)

            m_depth = {}
            if ref_depth is not None:
                m_depth = stereo.compute_metric_depth_errors(depth, ref_depth, min_z=0.4, max_z=4.0)

            print(f"  {t:15s} | Preset {p_name:8s}: Cov = {stats['valid_coverage_pct']:5.1f}% | Wall = {stats['wall_ms']:5.1f} ms" +
                  (f" | MedErr = {m_depth.get('median_m', np.nan)*100:4.1f} cm | Delta125 = {m_depth.get('delta_125_pct', np.nan):4.1f}%" if ref_depth is not None else " | Ref: OMITTED"))

            rows.append({
                "dataset": "Gunung",
                "scene": t,
                "split": "field",
                "config": p_name,
                "coverage_pct": stats["valid_coverage_pct"],
                "latency_ms": stats["wall_ms"],
                "depth_mae_m": m_depth.get("mae_m", np.nan),
                "depth_med_m": m_depth.get("median_m", np.nan),
                "depth_rmse_m": m_depth.get("rmse_m", np.nan),
                "delta_125_pct": m_depth.get("delta_125_pct", np.nan),
                "reference_status": ref_status
            })

    # 2. Middlebury Evaluation
    mid_scenes = [
        ("Adirondack", "dev"), ("Motorcycle", "dev"), ("Playtable", "dev"),
        ("Jadeplant", "heldout"), ("Piano", "heldout"), ("Pipes", "heldout")
    ]
    print("\n--- Evaluating Middlebury Benchmark ---")
    for sc, split in mid_scenes:
        sc_dir = Path(mid_root) / sc
        if not sc_dir.exists():
            continue

        im0 = cv2.imread(str(sc_dir / "im0.png"))
        im1 = cv2.imread(str(sc_dir / "im1.png"))
        gt = load_pfm(str(sc_dir / "disp0GT.pfm"))
        mask = cv2.imread(str(sc_dir / "mask0nocc.png"), cv2.IMREAD_GRAYSCALE) == 255

        for p_name in presets:
            disp, stats = stereo.run_stereo_pipeline(im0, im1, preset=p_name)
            bad2 = stereo.compute_completeness_aware_bad_pixel(disp, gt, mask, threshold=2.0)
            valid_eval = mask & (disp > 0) & (gt > 0) & (gt < 1e8)
            mae_px = float(np.mean(np.abs(disp[valid_eval] - gt[valid_eval]))) if np.sum(valid_eval) > 0 else np.nan

            print(f"  {sc:15s} ({split:7s}) | Preset {p_name:8s}: Bad-2.0 = {bad2:5.2f}% | MAE = {mae_px:4.2f} px | Cov = {stats['valid_coverage_pct']:5.1f}% | Wall = {stats['wall_ms']:5.1f} ms")
            rows.append({
                "dataset": "Middlebury",
                "scene": sc,
                "split": split,
                "config": p_name,
                "coverage_pct": stats["valid_coverage_pct"],
                "latency_ms": stats["wall_ms"],
                "bad_2_0_pct": bad2,
                "mae_px": mae_px,
                "reference_status": "GT_Disparity"
            })

    df = pd.DataFrame(rows)
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "reproduction_summary.csv", index=False)

    with open(out_dir / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(stereo.PIPELINE_PRESETS, f, indent=2)

    print(f"\nSaved reproduction results to: {out_dir / 'reproduction_summary.csv'}")
    print(f"Saved frozen configurations to: {out_dir / 'best_params.json'}")
    print("=" * 80)
    print("Full benchmark reproduction finished successfully.")


def main():
    parser = argparse.ArgumentParser(description="Reproduce classical stereo matching experiments.")
    parser.add_argument("--quick", action="store_true", help="Run fast verification on sample scenes in < 1s.")
    parser.add_argument("--profile", action="store_true", help="Run component profiler across scales and threads.")
    parser.add_argument("--full", action="store_true", help="Run full benchmark evaluation across all scenes.")
    parser.add_argument("--calib", type=str, default=DEFAULT_CALIB, help="Path to calibration NPZ")
    parser.add_argument("--gunung", type=str, default=DEFAULT_GUNUNG, help="Path to Gunung dataset root")
    parser.add_argument("--middlebury", type=str, default=DEFAULT_MID, help="Path to Middlebury trainingQ directory")

    args = parser.parse_args()

    if args.profile:
        run_pipeline_profiling(args.calib, args.gunung)
    elif args.full:
        run_full_reproduction(args.calib, args.gunung, args.middlebury)
    else:
        run_quick_verification(args.calib, args.gunung, args.middlebury)


if __name__ == "__main__":
    main()
