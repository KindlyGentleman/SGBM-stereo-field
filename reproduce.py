"""Experiment runner and benchmark reproduction script.

Provides command-line entry points:
- python reproduce.py --quick: Fast sanity verification (< 1 second) on synthetic scenes.
- python reproduce.py --full: Complete procedural stereo benchmark across classical matchers and perturbations.
- python reproduce.py --middlebury <path>: Optional Middlebury 2014 benchmark evaluation.
- python reproduce.py --gunung <path> --calib <path>: Optional outdoor Gunung transfer evaluation.
"""

import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import cv2
import numpy as np
import pandas as pd

import stereo


def collect_environment_metadata() -> Dict[str, Any]:
    """Gather hardware, runtime, and dependency metadata for reproducibility."""
    env = {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "processor": platform.processor() or platform.machine(),
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
    }
    try:
        import torch
        env["torch_version"] = torch.__version__
        env["torch_cpu_threads"] = torch.get_num_threads()
    except ImportError:
        env["torch_version"] = None

    return env


def run_quick_verification() -> bool:
    """Run fast unit verification on constant and layered synthetic scenes."""
    print("=" * 72)
    print("STEREO REPRODUCTION: QUICK VERIFICATION MODE (< 1 SECOND)")
    print("=" * 72)

    # 1. Constant disparity scene
    print("\n[Test 1] Planar Constant Disparity Scene (Shift = 16 px)...")
    l_const, r_const, gt_const, mask_const = stereo.generate_constant_disparity_scene(
        width=320, height=240, disparity=16, seed=42
    )

    bm_matcher = stereo.create_bm_matcher(num_disparities=48, block_size=9)
    sgbm_matcher = stereo.create_sgbm_matcher(num_disparities=48, block_size=7)

    disp_bm, t_bm = stereo.compute_disparity_bm(l_const, r_const, bm_matcher)
    disp_sgbm, t_sgbm = stereo.compute_disparity_sgbm(l_const, r_const, sgbm_matcher)

    bad_bm = stereo.compute_bad_pixel_rate(disp_bm, gt_const, mask_const, threshold=2.0)
    bad_sgbm = stereo.compute_bad_pixel_rate(disp_sgbm, gt_const, mask_const, threshold=2.0)
    mae_sgbm = stereo.compute_depth_mae(disp_sgbm, gt_const, mask_const)

    print(f"  StereoBM:   Bad@2.0 = {bad_bm:5.2f}% | Latency = {t_bm:5.2f} ms")
    print(f"  StereoSGBM: Bad@2.0 = {bad_sgbm:5.2f}% | Latency = {t_sgbm:5.2f} ms | MAE = {mae_sgbm:.3f} px")

    # 2. Layered synthetic scene
    print("\n[Test 2] Layered Synthetic Scene (3 layers, occlusions, low-texture zone)...")
    l_layer, r_layer, gt_layer, mask_layer = stereo.generate_layered_synthetic_scene(
        width=320, height=240, seed=42
    )

    disp_layer_bm, t_l_bm = stereo.compute_disparity_bm(l_layer, r_layer, bm_matcher)
    disp_layer_sgbm, t_l_sgbm = stereo.compute_disparity_sgbm(l_layer, r_layer, sgbm_matcher)

    bad_l_bm = stereo.compute_bad_pixel_rate(disp_layer_bm, gt_layer, mask_layer, threshold=2.0)
    bad_l_sgbm = stereo.compute_bad_pixel_rate(disp_layer_sgbm, gt_layer, mask_layer, threshold=2.0)
    cov_l_sgbm = stereo.compute_valid_coverage(disp_layer_sgbm, mask_layer)
    mae_l_sgbm = stereo.compute_depth_mae(disp_layer_sgbm, gt_layer, mask_layer)

    print(f"  StereoBM:   Bad@2.0 = {bad_l_bm:5.2f}% | Latency = {t_l_bm:5.2f} ms")
    print(f"  StereoSGBM: Bad@2.0 = {bad_l_sgbm:5.2f}% | Latency = {t_l_sgbm:5.2f} ms | Coverage = {cov_l_sgbm:5.1f}% | MAE = {mae_l_sgbm:.3f} px")

    # 3. Single-patch educational SAD matching
    print("\n[Test 3] Educational Single-Patch SAD Profile (Point x=110, y=100 on Foreground)...")
    disps, costs, best_d, min_cost = stereo.compute_sad_cost_profile(
        l_layer, r_layer, x=110, y=100, patch_size=9, max_disp=48
    )
    true_d = int(gt_layer[100, 110])
    print(f"  True Disparity: {true_d} px | SAD Minimum Disparity: {best_d} px | Difference: {abs(best_d - true_d)} px")

    print("\n" + "=" * 72)
    print("Quick verification finished successfully.")
    print("=" * 72)
    return True


def run_full_benchmark(results_dir: str = "results") -> None:
    """Execute complete classical stereo benchmark across scenes and perturbations."""
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("EXPERIMENT 2: CLASSICAL STEREO BENCHMARK & ROBUSTNESS STUDY")
    print("=" * 72)

    perturbations = [
        "clean",
        "brightness_up",
        "brightness_down",
        "contrast_low",
        "gaussian_noise",
        "gamma_high",
    ]

    scenes = ["constant", "layered"]
    matchers = ["StereoBM", "StereoSGBM", "StereoSGBM+CLAHE"]

    records: List[Dict[str, Any]] = []

    # Cache sample disparity arrays for publication figures
    saved_arrays: Dict[str, Any] = {}

    for scene_name in scenes:
        print(f"\n--- Scene: {scene_name.upper()} ---")

        if scene_name == "constant":
            left_base, right_base, gt, valid_mask = stereo.generate_constant_disparity_scene(
                width=320, height=240, disparity=16, seed=42
            )
        else:
            left_base, right_base, gt, valid_mask = stereo.generate_layered_synthetic_scene(
                width=320, height=240, seed=42
            )

        saved_arrays[f"{scene_name}_left"] = left_base
        saved_arrays[f"{scene_name}_right_clean"] = right_base
        saved_arrays[f"{scene_name}_gt"] = gt
        saved_arrays[f"{scene_name}_mask"] = valid_mask

        bm_matcher = stereo.create_bm_matcher(num_disparities=48, block_size=9)
        sgbm_matcher = stereo.create_sgbm_matcher(num_disparities=48, block_size=7)

        for pert in perturbations:
            pert_mode = "none" if pert == "clean" else pert
            right_pert = stereo.apply_perturbation(right_base, pert_mode, seed=42)
            saved_arrays[f"{scene_name}_right_{pert}"] = right_pert

            for matcher_name in matchers:
                if matcher_name == "StereoBM":
                    disp, runtime_ms = stereo.compute_disparity_bm(left_base, right_pert, bm_matcher)
                elif matcher_name == "StereoSGBM":
                    disp, runtime_ms = stereo.compute_disparity_sgbm(left_base, right_pert, sgbm_matcher, use_clahe=False)
                elif matcher_name == "StereoSGBM+CLAHE":
                    disp, runtime_ms = stereo.compute_disparity_sgbm(left_base, right_pert, sgbm_matcher, use_clahe=True)
                else:
                    raise ValueError(f"Unknown matcher: {matcher_name}")

                bad1 = stereo.compute_bad_pixel_rate(disp, gt, valid_mask, threshold=1.0)
                bad2 = stereo.compute_bad_pixel_rate(disp, gt, valid_mask, threshold=2.0)
                bad2_ca = stereo.compute_bad_pixel_rate(disp, gt, valid_mask, threshold=2.0, completeness_aware=True)
                mae = stereo.compute_depth_mae(disp, gt, valid_mask)
                rmse = stereo.compute_depth_rmse(disp, gt, valid_mask)
                cov = stereo.compute_valid_coverage(disp, valid_mask)

                rec = {
                    "scene": scene_name,
                    "perturbation": pert,
                    "matcher": matcher_name,
                    "bad_1.0_pct": round(bad1, 3),
                    "bad_2.0_pct": round(bad2, 3),
                    "bad_2.0_ca_pct": round(bad2_ca, 3),
                    "mae_px": round(mae, 3) if not np.isnan(mae) else None,
                    "rmse_px": round(rmse, 3) if not np.isnan(rmse) else None,
                    "coverage_pct": round(cov, 2),
                    "runtime_ms": round(runtime_ms, 2)
                }
                records.append(rec)

                # Save key sample disparity maps for visualization
                if pert in ["clean", "gaussian_noise", "contrast_low", "brightness_up"]:
                    key = f"{scene_name}_{pert}_{matcher_name.lower().replace('+', '_')}_disp"
                    saved_arrays[key] = disp

            clean_sgbm = [r for r in records if r["scene"] == scene_name and r["perturbation"] == pert and r["matcher"] == "StereoSGBM"][0]
            print(f"  {pert:<16} | SGBM Bad@2: {clean_sgbm['bad_2.0_pct']:5.2f}% | Cov: {clean_sgbm['coverage_pct']:5.1f}% | Latency: {clean_sgbm['runtime_ms']:5.2f} ms")

    # Run educational single-patch SAD scanline example
    l_layer = saved_arrays["layered_left"]
    r_layer = saved_arrays["layered_right_clean"]
    disps, sad_costs, best_d, min_cost = stereo.compute_sad_cost_profile(
        l_layer, r_layer, x=110, y=100, patch_size=9, max_disp=48
    )
    true_d = int(saved_arrays["layered_gt"][100, 110])

    sad_dict = {
        "ref_x": 110,
        "ref_y": 100,
        "patch_size": 9,
        "true_disparity": true_d,
        "best_disparity": best_d,
        "min_cost": float(min_cost),
        "disparities": disps.tolist(),
        "costs": [float(c) if np.isfinite(c) else None for c in sad_costs]
    }
    with open(out_dir / "sad_profile.json", "w", encoding="utf-8") as f:
        json.dump(sad_dict, f, indent=2)

    # Save metrics table
    metrics_df = pd.DataFrame(records)
    metrics_csv = out_dir / "metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"\nSaved metrics table to: {metrics_csv}")

    # Save disparity arrays for visualize.py
    np.savez_compressed(out_dir / "sample_disparities.npz", **saved_arrays)
    print(f"Saved sample disparity maps to: {out_dir / 'sample_disparities.npz'}")

    # Save environment metadata
    env = collect_environment_metadata()
    env_json = out_dir / "environment.json"
    with open(env_json, "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2)
    print(f"Saved environment metadata to:  {env_json}")

    print("\n" + "=" * 72)
    print("Full benchmark completed.")
    print("=" * 72)


def run_middlebury_optional(mid_path: str) -> None:
    """Optional evaluation on Middlebury 2014 dataset if path is provided."""
    p = Path(mid_path)
    if not p.exists():
        print(f"Middlebury path does not exist: {mid_path}")
        return

    print("=" * 72)
    print(f"OPTIONAL MIDDLEBURY 2014 EVALUATION: {mid_path}")
    print("=" * 72)

    scenes = [d for d in p.iterdir() if d.is_dir() and (d / "im0.png").exists()]
    if not scenes:
        print("No valid scene folders found with im0.png")
        return

    matcher = stereo.create_sgbm_matcher(num_disparities=160, block_size=7)

    for sc in scenes[:4]:
        im0 = cv2.imread(str(sc / "im0.png"))
        im1 = cv2.imread(str(sc / "im1.png"))
        gt_file = sc / "disp0GT.pfm"
        mask_file = sc / "mask0nocc.png"

        disp, t_ms = stereo.compute_disparity_sgbm(im0, im1, matcher)

        if gt_file.exists() and mask_file.exists():
            gt = stereo.load_pfm(str(gt_file))
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE) == 255
            bad2 = stereo.compute_bad_pixel_rate(disp, gt, mask, threshold=2.0)
            mae = stereo.compute_depth_mae(disp, gt, mask)
            cov = stereo.compute_valid_coverage(disp, mask)
            print(f"  {sc.name:<16} | Bad@2.0: {bad2:5.2f}% | MAE: {mae:5.2f} px | Cov: {cov:5.1f}% | Latency: {t_ms:5.1f} ms")
        else:
            print(f"  {sc.name:<16} | Disparity computed ({disp.shape[1]}x{disp.shape[0]}) in {t_ms:5.1f} ms")


def run_gunung_optional(gunung_path: str, calib_path: str) -> None:
    """Optional transfer demonstration on outdoor calibrated Gunung dataset."""
    g_p = Path(gunung_path)
    c_p = Path(calib_path)

    if not g_p.exists():
        print(f"Gunung path does not exist: {gunung_path}")
        return
    if not c_p.exists():
        print(f"Calibration file does not exist: {calib_path}")
        return

    print("=" * 72)
    print(f"OPTIONAL OUTDOOR GUNUNG TRANSFER EVALUATION")
    print(f"  Data:  {gunung_path}")
    print(f"  Calib: {calib_path}")
    print("=" * 72)

    calib = stereo.load_calibration(str(c_p))
    maps_l, maps_r, Q = stereo.build_rectification_maps(calib, (1280, 720))
    matcher = stereo.create_sgbm_matcher(num_disparities=160, block_size=7)

    terrains = ["puncakTinggi", "puncakWarung", "tanggaBiru", "tanggaKuning"]
    for t_name in terrains:
        st_dir = g_p / t_name / "stereo"
        left_path = st_dir / "left_0000.png"
        right_path = st_dir / "right_0000.png"
        if not (left_path.exists() and right_path.exists()):
            continue

        raw_l = cv2.imread(str(left_path))
        raw_r = cv2.imread(str(right_path))
        rect_l, rect_r = stereo.rectify_pair(raw_l, raw_r, maps_l, maps_r)

        disp, t_ms = stereo.compute_disparity_sgbm(rect_l, rect_r, matcher)
        cov = stereo.compute_valid_coverage(disp)
        print(f"  {t_name:<16} | Coverage: {cov:5.1f}% | Latency: {t_ms:5.1f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stereo matching reproduction and evaluation runner.")
    parser.add_argument("--quick", action="store_true", help="Run fast verification (< 1 second) on synthetic scenes.")
    parser.add_argument("--full", action="store_true", help="Run complete benchmark across scenes, matchers, and perturbations.")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to save output metrics and arrays.")
    parser.add_argument("--middlebury", type=str, default=None, help="Optional path to Middlebury 2014 dataset directory.")
    parser.add_argument("--gunung", type=str, default=None, help="Optional path to Gunung dataset directory.")
    parser.add_argument("--calib", type=str, default=None, help="Optional path to camera calibration NPZ file.")
    args = parser.parse_args()

    if args.middlebury:
        run_middlebury_optional(args.middlebury)
        return

    if args.gunung:
        calib_file = args.calib or r"D:\02_Riset\Research Paper\source_code\calibration\calib_params\stereo_params_advanced.npz"
        run_gunung_optional(args.gunung, calib_file)
        return

    if args.quick or not args.full:
        run_quick_verification()

    if args.full:
        run_full_benchmark(results_dir=args.results_dir)


if __name__ == "__main__":
    main()
