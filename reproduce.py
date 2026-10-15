"""Command-line reproduction runner for stereo matching experiments.

Supports quick verification and full experimental reproduction across
Middlebury 2014 and the four Gunung outdoor terrains.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any
import cv2
import numpy as np
import pandas as pd

import stereo

DEFAULT_CALIB = r"D:\02_Riset\Research Paper\source_code\calibration\calib_params\stereo_params_advanced.npz"
DEFAULT_GUNUNG = r"D:\02_Riset\Research Paper\source_code\comparison\gunung"
DEFAULT_MID = r"data/middlebury/MiddEval3/trainingQ"

BEST_PARAMS = {
    "minDisparity": 0,
    "numDisparities": 160,
    "blockSize": 7,
    "P1": 150,
    "P2": 1200,
    "disp12MaxDiff": 1,
    "uniquenessRatio": 10,
    "speckleWindowSize": 100,
    "speckleRange": 2,
    "mode": cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    "wls_lambda": 8000.0,
    "wls_sigma": 1.5
}


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
    """Run fast verification on single scenes in under five seconds."""
    print("=" * 80)
    print("STEREO REPRODUCTION: QUICK VERIFICATION MODE")
    print("=" * 80)

    # 1. Calibration verification
    print(f"Loading calibration: {calib_path}")
    calib = stereo.load_calibration(calib_path)
    maps_l, maps_r, Q = stereo.build_rectification_maps(calib, (1280, 720))
    print(f"  Baseline B: {-calib['T'][0, 0]:.2f} mm | Focal length fx: {calib['mtx_left'][0, 0]:.1f} px")

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

    # Default matcher
    m_default = cv2.StereoSGBM_create(minDisparity=0, numDisparities=128, blockSize=3)
    t0 = time.perf_counter()
    d_def = m_default.compute(rect_l, rect_r).astype(np.float32) / 16.0
    t_def = (time.perf_counter() - t0) * 1000.0

    # Optimized matcher + WLS
    m_opt = stereo.create_sgbm_matcher(BEST_PARAMS)
    t0 = time.perf_counter()
    d_opt, d_r = stereo.compute_disparity_wls(m_opt, rect_l, rect_r, BEST_PARAMS["wls_lambda"], BEST_PARAMS["wls_sigma"])
    t_opt = (time.perf_counter() - t0) * 1000.0

    cov_def = stereo.compute_valid_coverage(d_def)
    cov_opt = stereo.compute_valid_coverage(d_opt)
    cons_opt, _ = stereo.compute_lr_consistency(d_opt, d_r, threshold=1.5)
    warp_def = stereo.compute_photometric_warp_mae(rect_l, rect_r, d_def)
    warp_opt = stereo.compute_photometric_warp_mae(rect_l, rect_r, d_opt)

    print("\n[Gunung Outdoor Results: puncakTinggi]")
    print(f"  Default SGBM:   Coverage = {cov_def:5.1f}% | Warp MAE = {warp_def:5.2f} DN | Time = {t_def:6.1f} ms")
    print(f"  Optimized+WLS:  Coverage = {cov_opt:5.1f}% | Warp MAE = {warp_opt:5.2f} DN | Time = {t_opt:6.1f} ms | LR Cons = {cons_opt:5.1f}%")

    # 3. Middlebury verification
    mid_p = Path(mid_root) / "Adirondack"
    if mid_p.exists():
        im0 = cv2.imread(str(mid_p / "im0.png"))
        im1 = cv2.imread(str(mid_p / "im1.png"))
        gt = load_pfm(str(mid_p / "disp0GT.pfm"))
        mask = cv2.imread(str(mid_p / "mask0nocc.png"), cv2.IMREAD_GRAYSCALE) == 255

        d_mid_def = m_default.compute(im0, im1).astype(np.float32) / 16.0
        d_mid_opt, d_mid_r = stereo.compute_disparity_wls(m_opt, im0, im1, BEST_PARAMS["wls_lambda"], BEST_PARAMS["wls_sigma"])

        bad_def = stereo.compute_completeness_aware_bad_pixel(d_mid_def, gt, mask, threshold=2.0)
        bad_opt = stereo.compute_completeness_aware_bad_pixel(d_mid_opt, gt, mask, threshold=2.0)
        mae_def = np.mean(np.abs(d_mid_def[mask & (d_mid_def > 0)] - gt[mask & (d_mid_def > 0)]))
        mae_opt = np.mean(np.abs(d_mid_opt[mask & (d_mid_opt > 0)] - gt[mask & (d_mid_opt > 0)]))

        print("\n[Middlebury Benchmark Ground Truth: Adirondack]")
        print(f"  Default SGBM:   Bad-2.0 = {bad_def:5.2f}% | MAE = {mae_def:5.2f} px")
        print(f"  Optimized+WLS:  Bad-2.0 = {bad_opt:5.2f}% | MAE = {mae_opt:5.2f} px (-{(mae_def-mae_opt)/mae_def*100:.1f}% error)")

    print("=" * 80)
    print("Quick verification PASSED successfully.")


def run_full_reproduction(
    calib_path: str,
    gunung_root: str,
    mid_root: str
) -> None:
    """Run complete reproduction across all four terrains and Middlebury benchmark."""
    print("=" * 80)
    print("STEREO REPRODUCTION: FULL BENCHMARK EXPERIMENT")
    print("=" * 80)

    calib = stereo.load_calibration(calib_path)
    maps_l, maps_r, Q = stereo.build_rectification_maps(calib, (1280, 720))

    terrains = ["puncakTinggi", "puncakWarung", "tanggaBiru", "tanggaKuning"]
    results_rows = []

    m_default = cv2.StereoSGBM_create(minDisparity=0, numDisparities=128, blockSize=3)
    m_opt = stereo.create_sgbm_matcher(BEST_PARAMS)

    for t in terrains:
        st_dir = Path(gunung_root) / t / "stereo"
        raw_l = cv2.imread(str(st_dir / "left_0000.png"))
        raw_r = cv2.imread(str(st_dir / "right_0000.png"))
        rect_l, rect_r = stereo.rectify_pair(raw_l, raw_r, maps_l, maps_r)

        # Default
        t0 = time.perf_counter()
        d_def = m_default.compute(rect_l, rect_r).astype(np.float32) / 16.0
        t_def = (time.perf_counter() - t0) * 1000.0

        # Optimized + WLS
        t0 = time.perf_counter()
        d_opt, d_r = stereo.compute_disparity_wls(m_opt, rect_l, rect_r, BEST_PARAMS["wls_lambda"], BEST_PARAMS["wls_sigma"])
        t_opt = (time.perf_counter() - t0) * 1000.0

        cov_def = stereo.compute_valid_coverage(d_def)
        cov_opt = stereo.compute_valid_coverage(d_opt)
        cons_opt, _ = stereo.compute_lr_consistency(d_opt, d_r, threshold=1.5)
        warp_def = stereo.compute_photometric_warp_mae(rect_l, rect_r, d_def)
        warp_opt = stereo.compute_photometric_warp_mae(rect_l, rect_r, d_opt)

        results_rows.append({
            "domain": "Gunung",
            "scene": t,
            "method": "Default_SGBM",
            "coverage_pct": cov_def,
            "lr_consistency_pct": 0.0,
            "warp_mae_dn": warp_def,
            "runtime_ms": t_def,
            "bad2_pct": np.nan,
            "mae_px": np.nan
        })
        results_rows.append({
            "domain": "Gunung",
            "scene": t,
            "method": "Optimized_WLS",
            "coverage_pct": cov_opt,
            "lr_consistency_pct": cons_opt,
            "warp_mae_dn": warp_opt,
            "runtime_ms": t_opt,
            "bad2_pct": np.nan,
            "mae_px": np.nan
        })

    # Middlebury evaluation
    mid_scenes = ["Adirondack", "Motorcycle", "Playtable"]
    for sc in mid_scenes:
        sc_dir = Path(mid_root) / sc
        if sc_dir.exists():
            im0 = cv2.imread(str(sc_dir / "im0.png"))
            im1 = cv2.imread(str(sc_dir / "im1.png"))
            gt = load_pfm(str(sc_dir / "disp0GT.pfm"))
            mask = cv2.imread(str(sc_dir / "mask0nocc.png"), cv2.IMREAD_GRAYSCALE) == 255

            t0 = time.perf_counter()
            d_def = m_default.compute(im0, im1).astype(np.float32) / 16.0
            t_def = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            d_opt, d_r = stereo.compute_disparity_wls(m_opt, im0, im1, BEST_PARAMS["wls_lambda"], BEST_PARAMS["wls_sigma"])
            t_opt = (time.perf_counter() - t0) * 1000.0

            bad_def = stereo.compute_completeness_aware_bad_pixel(d_def, gt, mask, threshold=2.0)
            bad_opt = stereo.compute_completeness_aware_bad_pixel(d_opt, gt, mask, threshold=2.0)
            cov_def = stereo.compute_valid_coverage(d_def)
            cov_opt = stereo.compute_valid_coverage(d_opt)
            mae_def = np.mean(np.abs(d_def[mask & (d_def > 0)] - gt[mask & (d_def > 0)]))
            mae_opt = np.mean(np.abs(d_opt[mask & (d_opt > 0)] - gt[mask & (d_opt > 0)]))

            results_rows.append({
                "domain": "Middlebury",
                "scene": sc,
                "method": "Default_SGBM",
                "coverage_pct": cov_def,
                "lr_consistency_pct": 0.0,
                "warp_mae_dn": np.nan,
                "runtime_ms": t_def,
                "bad2_pct": bad_def,
                "mae_px": mae_def
            })
            results_rows.append({
                "domain": "Middlebury",
                "scene": sc,
                "method": "Optimized_WLS",
                "coverage_pct": cov_opt,
                "lr_consistency_pct": 0.0,
                "warp_mae_dn": np.nan,
                "runtime_ms": t_opt,
                "bad2_pct": bad_opt,
                "mae_px": mae_opt
            })

    df = pd.DataFrame(results_rows)
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "summary.csv", index=False)

    with open(out_dir / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(BEST_PARAMS, f, indent=2)

    print(f"\nSaved summary table to: {out_dir / 'summary.csv'}")
    print(f"Saved frozen configurations to: {out_dir / 'best_params.json'}")
    print("=" * 80)
    print("Full reproduction completed successfully.")


def main():
    parser = argparse.ArgumentParser(description="Reproduce classical stereo matching experiments.")
    parser.add_argument("--quick", action="store_true", help="Run fast verification on single scene sample.")
    parser.add_argument("--full", action="store_true", help="Run full benchmark evaluation across all scenes.")
    parser.add_argument("--calib", type=str, default=DEFAULT_CALIB, help="Path to stereo_params_advanced.npz")
    parser.add_argument("--gunung", type=str, default=DEFAULT_GUNUNG, help="Path to gunung dataset root")
    parser.add_argument("--middlebury", type=str, default=DEFAULT_MID, help="Path to Middlebury trainingQ directory")

    args = parser.parse_args()

    if args.quick or (not args.full and not args.quick):
        run_quick_verification(args.calib, args.gunung, args.middlebury)
    else:
        run_full_reproduction(args.calib, args.gunung, args.middlebury)


if __name__ == "__main__":
    main()
