"""Component-level CPU and memory profiler for the stereo reconstruction pipeline.

Profiles:
- Image remap (rectification)
- Left StereoSGBM matching
- Right StereoSGBM matching
- Left-Right consistency check
- WLS filtering
- Disparity-to-metric-depth conversion

Evaluated at:
- Native resolution: 1280x720
- Half resolution: 640x360
- Quarter resolution: 320x180
Across thread configurations: 1, 2, 4, 8, 16 threads.
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


def profile_resolution(scale_factor: float, num_threads: int = 16, iterations: int = 30):
    """Profile individual pipeline stages at a given resolution scale."""
    cv2.setNumThreads(num_threads)
    w_base, h_base = 1280, 720
    w = int(w_base * scale_factor)
    h = int(h_base * scale_factor)

    calib = stereo.load_calibration(CALIB_PATH)
    maps_l, maps_r, Q = stereo.build_rectification_maps(calib, (w, h))

    # Load raw images
    st_dir = GUNUNG_ROOT / "puncakTinggi" / "stereo"
    raw_l = cv2.imread(str(st_dir / "left_0000.png"))
    raw_r = cv2.imread(str(st_dir / "right_0000.png"))
    if scale_factor != 1.0:
        raw_l = cv2.resize(raw_l, (w, h), interpolation=cv2.INTER_AREA)
        raw_r = cv2.resize(raw_r, (w, h), interpolation=cv2.INTER_AREA)

    # Disparity range scales with image width
    num_disp = int(np.ceil(160 * scale_factor / 16.0) * 16)
    num_disp = max(num_disp, 16)

    params = {
        "minDisparity": 0,
        "numDisparities": num_disp,
        "blockSize": 7,
        "P1": 150,
        "P2": 1200,
        "disp12MaxDiff": 1,
        "uniquenessRatio": 10,
        "speckleWindowSize": 100,
        "speckleRange": 2,
        "mode": cv2.STEREO_SGBM_MODE_SGBM_3WAY
    }
    matcher = stereo.create_sgbm_matcher(params)
    r_matcher = cv2.ximgproc.createRightMatcher(matcher)
    wls = cv2.ximgproc.createDisparityWLSFilter(matcher)
    wls.setLambda(8000.0)
    wls.setSigmaColor(1.5)

    # Warm-up (5 iterations)
    for _ in range(5):
        rect_l, rect_r = stereo.rectify_pair(raw_l, raw_r, maps_l, maps_r)
        d_l_16 = matcher.compute(rect_l, rect_r)
        d_r_16 = r_matcher.compute(rect_r, rect_l)
        f_disp = wls.filter(d_l_16, rect_l, None, d_r_16)
        disp_f = f_disp.astype(np.float32) / 16.0
        depth = stereo.disparity_to_depth(disp_f, 964.68 * scale_factor, 0.0975)

    # Profiling timed iterations
    t_remap, t_sgbm_l, t_sgbm_r, t_wls, t_depth = [], [], [], [], []
    cpu_times = []

    tracemalloc.start()
    for _ in range(iterations):
        c0 = time.process_time()

        # 1. Rectification / Remap
        t0 = time.perf_counter()
        rect_l = cv2.remap(raw_l, maps_l[0], maps_l[1], cv2.INTER_LINEAR)
        rect_r = cv2.remap(raw_r, maps_r[0], maps_r[1], cv2.INTER_LINEAR)
        t_remap.append((time.perf_counter() - t0) * 1000.0)

        # 2. Left SGBM
        t0 = time.perf_counter()
        d_l_16 = matcher.compute(rect_l, rect_r)
        t_sgbm_l.append((time.perf_counter() - t0) * 1000.0)

        # 3. Right SGBM
        t0 = time.perf_counter()
        d_r_16 = r_matcher.compute(rect_r, rect_l)
        t_sgbm_r.append((time.perf_counter() - t0) * 1000.0)

        # 4. WLS Filtering
        t0 = time.perf_counter()
        f_disp = wls.filter(d_l_16, rect_l, None, d_r_16)
        t_wls.append((time.perf_counter() - t0) * 1000.0)

        # 5. Depth conversion
        t0 = time.perf_counter()
        disp_f = f_disp.astype(np.float32) / 16.0
        depth = stereo.disparity_to_depth(disp_f, 964.68 * scale_factor, 0.0975)
        t_depth.append((time.perf_counter() - t0) * 1000.0)

        cpu_times.append((time.process_time() - c0) * 1000.0)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    total_wall = np.array(t_remap) + np.array(t_sgbm_l) + np.array(t_sgbm_r) + np.array(t_wls) + np.array(t_depth)

    result = {
        "resolution": f"{w}x{h}",
        "scale": scale_factor,
        "threads": num_threads,
        "num_disparities": num_disp,
        "remap_ms": float(np.median(t_remap)),
        "sgbm_left_ms": float(np.median(t_sgbm_l)),
        "sgbm_right_ms": float(np.median(t_sgbm_r)),
        "wls_ms": float(np.median(t_wls)),
        "depth_ms": float(np.median(t_depth)),
        "total_wall_p50_ms": float(np.median(total_wall)),
        "total_wall_p95_ms": float(np.percentile(total_wall, 95)),
        "process_cpu_p50_ms": float(np.median(cpu_times)),
        "peak_memory_mb": float(peak / (1024 * 1024))
    }
    return result


def profile_threading():
    """Benchmark OpenCV performance across thread counts on native 720p."""
    print("Profiling OpenCV Threading Scalability on Native 1280x720...")
    thread_counts = [1, 2, 4, 8, 16]
    records = []
    for tc in thread_counts:
        res = profile_resolution(scale_factor=1.0, num_threads=tc, iterations=15)
        records.append(res)
        print(f"  Threads {tc:2d} | Total Wall P50: {res['total_wall_p50_ms']:6.1f} ms | Process CPU P50: {res['process_cpu_p50_ms']:6.1f} ms | SGBM_L: {res['sgbm_left_ms']:5.1f} ms")
    return pd.DataFrame(records)


def profile_multiscale():
    """Benchmark performance across image resolutions at 16 threads."""
    print("\nProfiling Resolution Scaling (Native, Half, Quarter) at 16 threads...")
    scales = [1.0, 0.5, 0.25]
    records = []
    for s in scales:
        res = profile_resolution(scale_factor=s, num_threads=16, iterations=20)
        records.append(res)
        print(f"  Scale {s:4.2f} ({res['resolution']:9s}) | NumDisp: {res['num_disparities']:3d} | Total P50: {res['total_wall_p50_ms']:5.1f} ms | Peak RAM: {res['peak_memory_mb']:4.1f} MB")
        print(f"    Breakdown: Remap: {res['remap_ms']:.1f} ms | SGBM_L: {res['sgbm_left_ms']:.1f} ms | SGBM_R: {res['sgbm_right_ms']:.1f} ms | WLS: {res['wls_ms']:.1f} ms")
    return pd.DataFrame(records)


def main():
    print("=" * 80)
    print("EXTREME CPU STEREO PROFILER (V6 AUDIT GATE M2)")
    print("=" * 80)

    df_threads = profile_threading()
    df_threads.to_csv(RESULTS_DIR / "profile_threading.csv", index=False)

    df_scales = profile_multiscale()
    df_scales.to_csv(RESULTS_DIR / "profile_multiscale.csv", index=False)

    print("\nSaved profile results to results/profile_threading.csv and results/profile_multiscale.csv")


if __name__ == "__main__":
    main()
