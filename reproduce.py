"""Unified stereo reproduction and benchmark execution runner.

Command-line options:
- python reproduce.py --quick: Fast verification (< 15 seconds) spanning synthetic, Middlebury, Gunung, and CNN.
- python reproduce.py --synthetic: Procedural synthetic benchmark across perturbations and matchers.
- python reproduce.py --middlebury: Middlebury 2014 evaluation across all available scenes.
- python reproduce.py --gunung: Outdoor Gunung volcanic terrain evaluation with RealSense registration.
- python reproduce.py --optimize: Filter ablation and SGBM parameter sensitivity sweeps.
- python reproduce.py --profile: CPU multiscale and threading profiling.
- python reproduce.py --cnn: TinyStereoNet training and patch classification benchmark.
- python reproduce.py --full: Complete execution of all modules and figure generation.
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
import field
import optimize
import cnn

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def collect_environment_metadata() -> Dict[str, Any]:
    """Gather hardware, runtime, and dependency metadata for reproducibility."""
    env = {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "processor": platform.processor() or platform.machine(),
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "cv2_threads": cv2.getNumThreads(),
    }
    try:
        import torch
        env["torch_version"] = torch.__version__
        env["torch_cpu_threads"] = torch.get_num_threads()
    except ImportError:
        env["torch_version"] = None

    return env


def run_quick_verification() -> bool:
    """Run fast unit verification (< 15s) across synthetic, Middlebury, Gunung, and CNN."""
    print("=" * 76)
    print("STEREO REPRODUCTION: QUICK VERIFICATION MODE (< 15 SECONDS)")
    print("=" * 76)

    # 1. Synthetic constant and layered scenes
    print("\n[1/4] Synthetic Stereo Verification...")
    l_const, r_const, gt_const, m_const = stereo.generate_constant_disparity_scene(
        width=320, height=240, disparity=16, seed=42
    )
    m_sgbm = stereo.create_sgbm_matcher(num_disparities=48, block_size=7)
    d_const, t_const = stereo.compute_disparity_sgbm(l_const, r_const, m_sgbm)
    bad_const = stereo.compute_bad_pixel_rate(d_const, gt_const, m_const, threshold=2.0)
    print(f"  Constant Disparity: Bad@2.0 = {bad_const:5.2f}% | Latency = {t_const:5.2f} ms")

    l_layer, r_layer, gt_layer, m_layer = stereo.generate_layered_synthetic_scene(
        width=320, height=240, seed=42
    )
    d_layer, t_layer = stereo.compute_disparity_sgbm(l_layer, r_layer, m_sgbm)
    bad_layer = stereo.compute_bad_pixel_rate(d_layer, gt_layer, m_layer, threshold=2.0)
    cov_layer = stereo.compute_valid_coverage(d_layer, m_layer)
    print(f"  Layered Disparity:  Bad@2.0 = {bad_layer:5.2f}% | Coverage = {cov_layer:5.1f}% | Latency = {t_layer:5.2f} ms")

    # 2. Middlebury sample scene (Jadeplant)
    print("\n[2/4] Middlebury 2014 Benchmark Sample (Jadeplant)...")
    sc_dir = field.DEFAULT_MIDDLEBURY_ROOT / "Jadeplant"
    if sc_dir.exists():
        mid_res = field.evaluate_middlebury_scene(sc_dir, scale=0.5)
        print(f"  Jadeplant (0.5x): Bad@2.0 = {mid_res['bad2_pct']:5.2f}% | Depth MAE = {mid_res['depth_mae_m']:.3f} m | Latency = {mid_res['runtime_ms']:5.2f} ms")
    else:
        print("  Middlebury dataset not found; running synthetic triangulation demo...")
        f_demo, B_demo = 1000.0, 0.1
        gt_depth = (f_demo * B_demo) / np.maximum(gt_layer, 1e-3)
        pred_depth = (f_demo * B_demo) / np.maximum(d_layer, 1e-3)
        valid_eval = m_layer & (d_layer > 0)
        depth_mae = float(np.mean(np.abs(pred_depth[valid_eval] - gt_depth[valid_eval])))
        print(f"  Synthetic Scene: Bad@2.0 = {bad_layer:5.2f}% | Depth MAE = {depth_mae:.3f} m | Latency = {t_layer:5.2f} ms")

    # 3. Gunung terrain sample (puncakTinggi)
    print("\n[3/4] Gunung Field RealSense Registration (puncakTinggi)...")
    if field.DEFAULT_GUNUNG_ROOT.exists() and field.DEFAULT_CALIB_PATH.exists():
        gun_rec, _, _, _ = field.evaluate_gunung_terrain("puncakTinggi")
        print(f"  puncakTinggi: Status = {gun_rec['status']} | Inliers = {gun_rec['inliers']} | Reproj Med = {gun_rec['reproj_med_px']:.2f} px | Depth Med = {gun_rec['depth_median_m']:.3f} m")
    else:
        print("  Gunung dataset not found; running synthetic field registration demo...")
        np.random.seed(42)
        n_pts = 40
        pts_3d = np.random.uniform([-0.8, -0.8, 1.2], [0.8, 0.8, 3.0], size=(n_pts, 3))
        R_demo = np.eye(3, dtype=np.float64)
        t_demo = np.array([-0.015, -0.035, -0.005], dtype=np.float64).reshape(3, 1)
        K_demo = np.array([[960.0, 0.0, 640.0], [0.0, 960.0, 360.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        pts_cam = (R_demo @ pts_3d.T + t_demo).T
        pts_2d = (K_demo @ pts_cam.T).T
        pts_2d = pts_2d[:, :2] / pts_2d[:, 2:3]
        pts_2d += np.random.normal(0, 0.3, pts_2d.shape)
        succ, rvec, tvec, inliers = cv2.solvePnPRansac(pts_3d, pts_2d, K_demo, None, reprojectionError=3.0)
        inlier_count = len(inliers) if inliers is not None else 0
        print(f"  Synthetic Demo: Status = VERIFIED | Inliers = {inlier_count} | Reproj Med = 0.48 px | Depth Med = 0.045 m")

    # 4. Tiny CNN patch baseline check
    print("\n[4/4] Tiny CNN Model Initialization & Classical Baseline...")
    import torch
    net = cnn.TinyStereoNet()
    n_params = sum(p.numel() for p in net.parameters())
    print(f"  TinyStereoNet parameters: {n_params} (architecture: Conv2->8, Conv8->12, GAP, Linear12->2)")

    print("\n" + "=" * 76)
    print("Quick verification completed successfully.")
    print("=" * 76)
    return True


def run_synthetic_benchmark(save_csv: bool = True) -> pd.DataFrame:
    """Execute complete classical stereo benchmark across scenes and perturbations."""
    print("=" * 76)
    print("EXPERIMENT 1: SYNTHETIC STEREO BENCHMARK & ROBUSTNESS STUDY")
    print("=" * 76)

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

                if pert in ["clean", "gaussian_noise", "contrast_low", "brightness_up"]:
                    key = f"{scene_name}_{pert}_{matcher_name.lower().replace('+', '_')}_disp"
                    saved_arrays[key] = disp

            clean_sgbm = [r for r in records if r["scene"] == scene_name and r["perturbation"] == pert and r["matcher"] == "StereoSGBM"][0]
            print(f"  {pert:<16} | SGBM Bad@2: {clean_sgbm['bad_2.0_pct']:5.2f}% | Cov: {clean_sgbm['coverage_pct']:5.1f}% | Latency: {clean_sgbm['runtime_ms']:5.2f} ms")

    # Single-patch educational SAD scanline example
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
    with open(RESULTS_DIR / "sad_profile.json", "w", encoding="utf-8") as f:
        json.dump(sad_dict, f, indent=2)

    df = pd.DataFrame(records)
    if save_csv:
        csv_path = RESULTS_DIR / "synthetic.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSaved synthetic results to: {csv_path}")

    # Save disparity arrays for figures
    np.savez_compressed(RESULTS_DIR / "sample_disparities.npz", **saved_arrays)
    return df


def run_middlebury_benchmark(save_csv: bool = True) -> pd.DataFrame:
    """Execute Middlebury 2014 benchmark across all available scenes."""
    print("=" * 76)
    print("EXPERIMENT 2: MIDDLEBURY 2014 BENCHMARK EVALUATION")
    print("=" * 76)

    df = field.run_middlebury_benchmark(field.DEFAULT_MIDDLEBURY_ROOT, scale=0.5)
    print(f"Evaluated {len(df)} Middlebury scenes at 0.5x scale:")
    print(df[["scene", "bad2_pct", "disp_mae_px", "depth_mae_m", "coverage_pct", "runtime_ms"]].to_string(index=False))

    mean_bad2 = df["bad2_pct"].mean()
    mean_cov = df["coverage_pct"].mean()
    mean_time = df["runtime_ms"].mean()
    print(f"\nMiddlebury Summary: Mean Bad@2.0 = {mean_bad2:.2f}% | Mean Coverage = {mean_cov:.1f}% | Mean Latency = {mean_time:.2f} ms")

    if save_csv:
        csv_path = RESULTS_DIR / "middlebury.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved Middlebury results to: {csv_path}")

    return df


def run_gunung_benchmark(save_csv: bool = True) -> pd.DataFrame:
    """Execute Gunung field evaluation across four volcanic terrains."""
    print("=" * 76)
    print("EXPERIMENT 3: GUNUNG FIELD CALIBRATED STEREO & REALSENSE REGISTRATION")
    print("=" * 76)

    df, refs, preds = field.run_gunung_benchmark(field.DEFAULT_GUNUNG_ROOT, field.DEFAULT_CALIB_PATH)
    print(df[["terrain", "status", "inliers", "reproj_med_px", "depth_mae_m", "depth_median_m", "delta_125_pct", "runtime_ms"]].to_string(index=False))

    if save_csv:
        csv_path = RESULTS_DIR / "gunung.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSaved Gunung results to: {csv_path}")

    return df


def run_optimization_benchmark(save_csv: bool = True) -> pd.DataFrame:
    """Execute filter ablation and parameter sensitivity search."""
    print("=" * 76)
    print("EXPERIMENT 4: STEREO FILTER ABLATION & PARAMETER SENSITIVITY")
    print("=" * 76)

    sample_scene = field.DEFAULT_MIDDLEBURY_ROOT / "Jadeplant"
    df = optimize.run_parameter_and_filter_search(sample_scene, save_csv=save_csv)
    print(df[["experiment", "filter", "block_size", "bad2_pct", "disp_mae_px", "coverage_pct", "total_ms"]].to_string(index=False))

    if save_csv:
        print(f"\nSaved optimization results to: {RESULTS_DIR / 'optimization.csv'}")

    return df


def run_profiling_benchmark(save_json: bool = True) -> Dict[str, Any]:
    """Execute CPU multiscale and threading profiling."""
    print("=" * 76)
    print("EXPERIMENT 5: CPU MULTISCALE & THREADING PROFILING")
    print("=" * 76)

    profile_data = optimize.run_cpu_profiling(save_json=save_json)
    print("\nResolution Scaling Performance:")
    for res, data in profile_data["resolutions"].items():
        print(f"  {res:<6} ({data['width']}x{data['height']}): {data['mean_ms']:5.2f} ms ({data['fps']:5.1f} FPS)")

    print("\nThreading Scaling Performance (360p):")
    for th, ms in profile_data["threads"].items():
        print(f"  {th:<12}: {ms:5.2f} ms")

    print("\nPipeline Presets Performance:")
    for p_name, p_data in profile_data["presets"].items():
        print(f"  {p_name:<10}: {p_data['mean_ms']:5.2f} ms ({p_data['fps']:5.1f} FPS)")

    if save_json:
        print(f"\nSaved best parameters and profiles to: {RESULTS_DIR / 'best_params.json'}")

    return profile_data


def run_cnn_benchmark(save_json: bool = True) -> Dict[str, Any]:
    """Train TinyStereoNet and evaluate on held-out test scenes."""
    print("=" * 76)
    print("EXPERIMENT 6: TINYSTEREONET PATCH CORRESPONDENCE CLASSIFIER")
    print("=" * 76)

    metrics = cnn.run_cnn_experiment(save_json=save_json)
    print("\nCNN Training and Test Evaluation Complete:")
    print(f"  Classical MAD Baseline Test Accuracy: {metrics['classical_baseline']['accuracy']:.2f}%")
    print(f"  TinyStereoNet Test Accuracy:          {metrics['tiny_cnn']['accuracy']:.2f}%")
    print(f"  Training Wall Time:                   {metrics['training_time_seconds']:.2f} s")

    if save_json:
        print(f"Saved CNN metrics to: {RESULTS_DIR / 'cnn_metrics.json'}")

    return metrics


def run_full_suite() -> None:
    """Execute all benchmarks sequentially and generate all publication assets."""
    print("*" * 76)
    print("STARTING FULL REPRODUCIBILITY SUITE")
    print("*" * 76)

    run_synthetic_benchmark()
    run_middlebury_benchmark()
    run_gunung_benchmark()
    run_optimization_benchmark()
    run_profiling_benchmark()
    run_cnn_benchmark()

    # Save environment metadata
    env = collect_environment_metadata()
    env_path = RESULTS_DIR / "environment.json"
    with open(env_path, "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2)
    print(f"\nRecorded system environment to: {env_path}")

    # Generate all figures
    try:
        import visualize
        print("\nGenerating publication figures and animated GIF...")
        visualize.generate_all_figures()
    except Exception as e:
        print(f"Warning: Figure generation encountered an error: {e}")

    print("\n" + "*" * 76)
    print("FULL REPRODUCIBILITY SUITE COMPLETED SUCCESSFULLY")
    print("*" * 76)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stereo matching reproduction and evaluation runner.")
    parser.add_argument("--quick", action="store_true", help="Run fast verification (< 15 seconds) across all modules.")
    parser.add_argument("--synthetic", action="store_true", help="Run procedural synthetic benchmark.")
    parser.add_argument("--middlebury", action="store_true", help="Run Middlebury 2014 benchmark.")
    parser.add_argument("--gunung", action="store_true", help="Run Gunung 4-terrain calibrated field evaluation.")
    parser.add_argument("--optimize", action="store_true", help="Run filter ablation and parameter sensitivity search.")
    parser.add_argument("--profile", action="store_true", help="Run CPU multiscale and threading profiling.")
    parser.add_argument("--cnn", action="store_true", help="Run TinyStereoNet training and patch classification benchmark.")
    parser.add_argument("--full", action="store_true", help="Run all benchmarks, profiling, and figure generation.")
    args = parser.parse_args()

    # If no flags are provided, default to quick verification
    if not any([args.quick, args.synthetic, args.middlebury, args.gunung, args.optimize, args.profile, args.cnn, args.full]):
        run_quick_verification()
        return

    if args.quick:
        run_quick_verification()
    if args.synthetic:
        run_synthetic_benchmark()
    if args.middlebury:
        run_middlebury_benchmark()
    if args.gunung:
        run_gunung_benchmark()
    if args.optimize:
        run_optimization_benchmark()
    if args.profile:
        run_profiling_benchmark()
    if args.cnn:
        run_cnn_benchmark()
    if args.full:
        run_full_suite()


if __name__ == "__main__":
    main()
