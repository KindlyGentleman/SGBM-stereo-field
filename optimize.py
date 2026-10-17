"""CPU performance optimization, filter suite, and Pareto frontier profiling.

Implements classical stereo optimization:
1. Filter benchmark suite: raw, median_valid, guided, joint_bilateral, wls, confidence_edge.
2. SGBM parameter grid and sensitivity search (blockSize, numDisparities, P1, P2, mode).
3. Preset configurations: FAST (30 FPS), BALANCED (15 FPS), QUALITY (3 FPS).
4. Stage-level CPU profiling across resolutions (720p, 360p, 180p) and threads (1 to 16).
5. Generation of optimization benchmarks and best parameter configuration.
"""

import json
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np
import pandas as pd

import stereo
import field

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Filter Benchmark Suite
# ---------------------------------------------------------------------------

def apply_filter(
    filter_name: str,
    disp_raw: np.ndarray,
    left_img: np.ndarray,
    disp_r_raw: Optional[np.ndarray] = None,
    matcher: Optional[cv2.StereoSGBM] = None
) -> np.ndarray:
    """Apply one of the evaluated stereo post-processing filters.

    Filters:
    - 'raw': Raw SGBM output without modification.
    - 'median_valid': 3x3 median blur restricted to valid pixels.
    - 'guided': Guided filter using grayscale left image as structural guide.
    - 'joint_bilateral': Joint bilateral filter with color left image guidance.
    - 'wls': Weighted Least Squares disparity filter using left-right consistency.
    - 'confidence_edge': WLS filter with confidence map masking for edge preservation.
    """
    name = filter_name.lower().strip()
    valid_mask = (disp_raw >= 0.0) & np.isfinite(disp_raw)

    if name in ["raw", "none", "f0_raw"]:
        return disp_raw.copy()

    elif name in ["median_valid", "f2_median", "median"]:
        # Dilate valid values into immediate invalid neighbors to prevent zero-bleed at borders
        d_filled = np.where(valid_mask, disp_raw, 0.0).astype(np.float32)
        d_dilated = cv2.dilate(d_filled, np.ones((3, 3), np.uint8))
        d_prep = np.where(valid_mask, disp_raw, d_dilated)
        med = cv2.medianBlur(d_prep, 3)
        return np.where(valid_mask, med, -1.0)

    elif name in ["guided", "f3_guided"]:
        gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY) if left_img.ndim == 3 else left_img
        d_filled = np.where(valid_mask, disp_raw, 0.0).astype(np.float32)
        d_dilated = cv2.dilate(d_filled, np.ones((5, 5), np.uint8))
        d_prep = np.where(valid_mask, disp_raw, d_dilated)
        gf = cv2.ximgproc.guidedFilter(guide=gray, src=d_prep, radius=5, eps=1.0)
        return np.where(valid_mask, gf, -1.0)

    elif name in ["joint_bilateral", "f4_joint_bilateral"]:
        img_f = left_img.astype(np.float32) / 255.0
        d_filled = np.where(valid_mask, disp_raw, 0.0).astype(np.float32)
        d_dilated = cv2.dilate(d_filled, np.ones((5, 5), np.uint8))
        d_prep = np.where(valid_mask, disp_raw, d_dilated)
        jb = cv2.ximgproc.jointBilateralFilter(joint=img_f, src=d_prep, d=5, sigmaColor=0.1, sigmaSpace=5)
        return np.where(valid_mask, jb, -1.0)

    elif name in ["wls", "f5_wls"]:
        if matcher is None or disp_r_raw is None:
            raise ValueError("WLS filter requires both matcher and disp_r_raw")
        wls = cv2.ximgproc.createDisparityWLSFilter(matcher)
        wls.setLambda(8000.0)
        wls.setSigmaColor(1.5)

        # Convert to 16-bit signed integer for OpenCV WLS, preserving right disparity signs
        if disp_raw.dtype == np.int16:
            d_l_16 = disp_raw
        else:
            d_l_16 = np.round(disp_raw * 16.0).astype(np.int16)

        if disp_r_raw.dtype == np.int16:
            d_r_16 = disp_r_raw
        else:
            d_r_16 = np.round(disp_r_raw * 16.0).astype(np.int16)

        f_disp_raw = wls.filter(d_l_16, left_img, None, d_r_16)
        f_disp = f_disp_raw.astype(np.float32) / 16.0

        min_disp = matcher.getMinDisparity()
        invalid_sentinel_raw = (min_disp - 1) * 16
        f_disp[f_disp_raw <= invalid_sentinel_raw] = -1.0
        return f_disp

    elif name in ["confidence_edge", "f6_confidence_edge"]:
        if matcher is None or disp_r_raw is None:
            raise ValueError("Confidence-aware filter requires both matcher and disp_r_raw")
        wls = cv2.ximgproc.createDisparityWLSFilter(matcher)
        wls.setLambda(8000.0)
        wls.setSigmaColor(1.5)

        if disp_raw.dtype == np.int16:
            d_l_16 = disp_raw
        else:
            d_l_16 = np.round(disp_raw * 16.0).astype(np.int16)

        if disp_r_raw.dtype == np.int16:
            d_r_16 = disp_r_raw
        else:
            d_r_16 = np.round(disp_r_raw * 16.0).astype(np.int16)

        f_disp_raw = wls.filter(d_l_16, left_img, None, d_r_16)
        conf = wls.getConfidenceMap()
        f_disp = f_disp_raw.astype(np.float32) / 16.0

        min_disp = matcher.getMinDisparity()
        invalid_sentinel_raw = (min_disp - 1) * 16
        f_disp[f_disp_raw <= invalid_sentinel_raw] = -1.0

        max_conf = np.max(conf) if conf.size > 0 else 1.0
        if max_conf > 0:
            # Mask low confidence pixels with invalid sentinel (-1.0), not 0.0
            f_disp[conf < 0.2 * max_conf] = -1.0
        return f_disp

    else:
        raise ValueError(f"Unknown filter name: {filter_name}")


# ---------------------------------------------------------------------------
# Pipeline Presets and Runner
# ---------------------------------------------------------------------------

PIPELINE_PRESETS: Dict[str, Dict[str, Any]] = {
    "FAST": {
        "scale": 0.5,
        "numDisparities": 80,
        "blockSize": 5,
        "P1": 80,
        "P2": 640,
        "disp12MaxDiff": 1,
        "uniquenessRatio": 10,
        "speckleWindowSize": 100,
        "speckleRange": 2,
        "mode": cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        "use_wls": False,
        "filter": "median_valid",
        "description": "0.5x scale (360p), 80 disparities, single-pass SGBM + validity-aware median filter for low-latency robotics"
    },
    "BALANCED": {
        "scale": 0.75,
        "numDisparities": 128,
        "blockSize": 5,
        "P1": 80,
        "P2": 640,
        "disp12MaxDiff": 1,
        "uniquenessRatio": 10,
        "speckleWindowSize": 100,
        "speckleRange": 2,
        "mode": cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        "use_wls": False,
        "filter": "median_valid",
        "description": "0.75x scale (540p), 128 disparities, single-pass SGBM + validity-aware median filter"
    },
    "QUALITY": {
        "scale": 1.0,
        "numDisparities": 160,
        "blockSize": 5,
        "P1": 100,
        "P2": 800,
        "disp12MaxDiff": 1,
        "uniquenessRatio": 10,
        "speckleWindowSize": 100,
        "speckleRange": 2,
        "mode": cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        "use_wls": False,
        "filter": "median_valid",
        "description": "1.0x native scale (720p), 160 disparities, block size 5 (Pareto-dominant over bs=7) + median filter"
    }
}


def run_stereo_pipeline(
    left_img: np.ndarray,
    right_img: np.ndarray,
    preset: str = "BALANCED",
    custom_params: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Execute complete stereo pipeline with downscaling, matching, and filtering.

    Presets:
    - FAST: 0.5x scale, single-pass SGBM + validity-aware median.
    - BALANCED: 0.5x scale, SGBM + validity-aware median filter.
    - QUALITY: 1.0x native resolution, full-range SGBM + validity-aware median.
    """
    key = preset.upper().strip()
    if key not in PIPELINE_PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Choose from {list(PIPELINE_PRESETS.keys())}")

    cfg = dict(PIPELINE_PRESETS[key])
    if custom_params:
        cfg.update(custom_params)

    scale = cfg["scale"]
    w_orig, h_orig = left_img.shape[1], left_img.shape[0]

    t0 = time.perf_counter()
    c0 = time.process_time()

    # Stage 1: Scale inputs
    t_scale0 = time.perf_counter()
    if scale != 1.0:
        w_s, h_s = int(w_orig * scale), int(h_orig * scale)
        l_in = cv2.resize(left_img, (w_s, h_s), interpolation=cv2.INTER_AREA)
        r_in = cv2.resize(right_img, (w_s, h_s), interpolation=cv2.INTER_AREA)
    else:
        l_in, r_in = left_img, right_img
    t_scale = (time.perf_counter() - t_scale0) * 1000.0

    # Ensure grayscale inputs for OpenCV SGBM compute
    gray_l = cv2.cvtColor(l_in, cv2.COLOR_BGR2GRAY) if l_in.ndim == 3 else l_in
    gray_r = cv2.cvtColor(r_in, cv2.COLOR_BGR2GRAY) if r_in.ndim == 3 else r_in

    w_in = l_in.shape[1]
    max_possible_disp = max(16, ((w_in - 1) // 16) * 16)
    num_disp = min(cfg["numDisparities"], max_possible_disp)

    matcher = stereo.create_sgbm_matcher(
        num_disparities=num_disp,
        block_size=cfg["blockSize"],
        mode=cfg.get("mode", cv2.STEREO_SGBM_MODE_SGBM_3WAY),
        p1=cfg["P1"],
        p2=cfg["P2"]
    )

    # Stage 2: SGBM matching
    t_match0 = time.perf_counter()
    if cfg.get("use_wls", False):
        r_matcher = cv2.ximgproc.createRightMatcher(matcher)
        dl = matcher.compute(gray_l, gray_r)
        dr = r_matcher.compute(gray_r, gray_l)
        t_match = (time.perf_counter() - t_match0) * 1000.0

        # Stage 3: WLS filter
        t_filt0 = time.perf_counter()
        wls = cv2.ximgproc.createDisparityWLSFilter(matcher)
        wls.setLambda(cfg.get("wls_lambda", 8000.0))
        wls.setSigmaColor(cfg.get("wls_sigma", 1.5))
        d_filt = wls.filter(dl, gray_l, None, dr).astype(np.float32) / 16.0
        valid_filt = d_filt >= 0.0
        d_filt = np.where(valid_filt, d_filt, -1.0)
        t_filt = (time.perf_counter() - t_filt0) * 1000.0
    else:
        dl = matcher.compute(gray_l, gray_r)
        t_match = (time.perf_counter() - t_match0) * 1000.0

        t_filt0 = time.perf_counter()
        raw_valid = dl > (matcher.getMinDisparity() - 1) * 16
        d_raw = np.where(raw_valid, dl.astype(np.float32) / 16.0, -1.0)

        filt_name = cfg.get("filter", "median_valid")
        if filt_name in ["raw", "none"]:
            d_filt = d_raw
        else:
            d_filt = apply_filter(filt_name, d_raw, l_in)
        t_filt = (time.perf_counter() - t_filt0) * 1000.0

    # Stage 4: Upscale disparity if downscaled (normalized convolution)
    t_up0 = time.perf_counter()
    if scale != 1.0:
        valid_mask_in = (d_filt >= 0.0) & np.isfinite(d_filt)
        disp_filled = np.where(valid_mask_in, d_filt, 0.0).astype(np.float32)
        w_m = valid_mask_in.astype(np.float32)
        d_up = cv2.resize(disp_filled, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        w_up = cv2.resize(w_m, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        valid_up = w_up >= 0.5
        d_norm = np.zeros_like(d_up)
        d_norm[valid_up] = (d_up[valid_up] / w_up[valid_up]) * (1.0 / scale)
        d_out = np.where(valid_up, d_norm, -1.0)
    else:
        d_out = d_filt
    t_up = (time.perf_counter() - t_up0) * 1000.0

    wall_ms = (time.perf_counter() - t0) * 1000.0
    cpu_ms = (time.process_time() - c0) * 1000.0

    stats = {
        "preset": key,
        "scale": scale,
        "numDisparities": num_disp,
        "wall_ms": wall_ms,
        "cpu_ms": cpu_ms,
        "stage_scale_ms": t_scale,
        "stage_match_ms": t_match,
        "stage_filter_ms": t_filt,
        "stage_upscale_ms": t_up,
        "valid_coverage_pct": stereo.compute_valid_coverage(d_out)
    }
    return d_out, stats


# ---------------------------------------------------------------------------
# SGBM Parameter Search and Filter Benchmark
# ---------------------------------------------------------------------------

def compute_pareto_frontier(
    df: pd.DataFrame,
    latency_col: str = "total_ms",
    error_col: str = "bad2_pct",
    minimize_latency: bool = True,
    minimize_error: bool = True
) -> pd.Series:
    """Compute boolean mask of non-dominated Pareto configurations.

    A configuration A dominates B if:
    1. A is no worse than B in both latency and error.
    2. A is strictly better than B in at least one metric.

    Returns:
        Boolean Series with True for non-dominated (Pareto-optimal) points.
    """
    is_pareto = []
    n = len(df)
    latencies = df[latency_col].values
    errors = df[error_col].values

    for i in range(n):
        l_i = latencies[i]
        e_i = errors[i]
        dominated = False
        for j in range(n):
            if i == j:
                continue
            l_j = latencies[j]
            e_j = errors[j]

            better_or_equal_lat = (l_j <= l_i) if minimize_latency else (l_j >= l_i)
            better_or_equal_err = (e_j <= e_i) if minimize_error else (e_j >= e_i)

            strictly_better_lat = (l_j < l_i) if minimize_latency else (l_j > l_i)
            strictly_better_err = (e_j < e_i) if minimize_error else (e_j > e_i)

            if better_or_equal_lat and better_or_equal_err and (strictly_better_lat or strictly_better_err):
                dominated = True
                break
        is_pareto.append(not dominated)

    return pd.Series(is_pareto, index=df.index, name="is_pareto")


def run_parameter_and_filter_search(
    sample_scene_dir: Optional[Path] = None,
    save_csv: bool = True
) -> pd.DataFrame:
    """Benchmark filter candidates and parameter sensitivities on reference scene."""
    if sample_scene_dir is None:
        sample_scene_dir = field.DEFAULT_MIDDLEBURY_ROOT / "Jadeplant"

    im0_path = sample_scene_dir / "im0.png"
    im1_path = sample_scene_dir / "im1.png"
    gt_path = sample_scene_dir / "disp0GT.pfm"
    mask_path = sample_scene_dir / "mask0nocc.png"

    if not im0_path.exists():
        # Fallback to synthetic layered scene if Middlebury is absent
        im0, im1, disp_gt, mask_eval = stereo.generate_layered_synthetic_scene(320, 240, seed=42)
    else:
        im0 = cv2.imread(str(im0_path))
        im1 = cv2.imread(str(im1_path))
        disp_gt = stereo.load_pfm(str(gt_path))
        mask_eval = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) == 255

    # Enforce consistent grayscale inputs for CPU matching
    gray0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY) if im0.ndim == 3 else im0
    gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY) if im1.ndim == 3 else im1

    ndisp = 160
    b_size = 7
    matcher = stereo.create_sgbm_matcher(num_disparities=ndisp, block_size=b_size)
    r_matcher = cv2.ximgproc.createRightMatcher(matcher)

    # Time single-direction (left) matching
    t0 = time.perf_counter()
    dl_raw = matcher.compute(gray0, gray1)
    match_l_ms = (time.perf_counter() - t0) * 1000.0

    # Time reverse-direction (right) matching
    t1 = time.perf_counter()
    dr_raw = r_matcher.compute(gray1, gray0)
    match_r_ms = (time.perf_counter() - t1) * 1000.0

    # Converted floating-point disparities (preserving negative sentinels)
    disp_l = dl_raw.astype(np.float32) / 16.0
    disp_l[dl_raw <= -16] = -1.0

    filter_names = ["raw", "median_valid", "guided", "joint_bilateral", "wls", "confidence_edge"]
    rows = []

    for fname in filter_names:
        t_f0 = time.perf_counter()
        f_disp = apply_filter(fname, disp_l, im0, disp_r_raw=dr_raw, matcher=matcher)
        f_ms = (time.perf_counter() - t_f0) * 1000.0

        # WLS and confidence_edge require right matching, so their total matching includes both passes
        is_dual_pass = fname in ["wls", "confidence_edge"]
        match_ms = (match_l_ms + match_r_ms) if is_dual_pass else match_l_ms

        m_res = stereo.compute_full_disparity_metrics(f_disp, disp_gt, mask_eval)

        rows.append({
            "experiment": "filter_ablation",
            "filter": fname,
            "block_size": b_size,
            "num_disparities": ndisp,
            "match_ms": match_ms,
            "filter_ms": f_ms,
            "total_ms": match_ms + f_ms,
            "bad1_pct": m_res["bad1_pct"],
            "bad2_pct": m_res["bad2_pct"],
            "bad4_pct": m_res["bad4_pct"],
            "bad2_ca_pct": m_res["bad2_ca_pct"],
            "disp_mae_px": m_res["disp_mae_px"],
            "coverage_pct": m_res["coverage_pct"]
        })

    # Parameter sweeps: block size (tested with identical single-pass matching on gray0, gray1)
    for bs in [3, 5, 7, 9]:
        m_bs = stereo.create_sgbm_matcher(num_disparities=ndisp, block_size=bs)
        t_bs0 = time.perf_counter()
        dl_bs = m_bs.compute(gray0, gray1)
        t_bs = (time.perf_counter() - t_bs0) * 1000.0

        d_bs = dl_bs.astype(np.float32) / 16.0
        d_bs[dl_bs <= -16] = -1.0

        m_bs_res = stereo.compute_full_disparity_metrics(d_bs, disp_gt, mask_eval)

        rows.append({
            "experiment": "block_size_sweep",
            "filter": "raw",
            "block_size": bs,
            "num_disparities": ndisp,
            "match_ms": t_bs,
            "filter_ms": 0.0,
            "total_ms": t_bs,
            "bad1_pct": m_bs_res["bad1_pct"],
            "bad2_pct": m_bs_res["bad2_pct"],
            "bad4_pct": m_bs_res["bad4_pct"],
            "bad2_ca_pct": m_bs_res["bad2_ca_pct"],
            "disp_mae_px": m_bs_res["disp_mae_px"],
            "coverage_pct": m_bs_res["coverage_pct"]
        })

    df = pd.DataFrame(rows)
    df["is_pareto"] = compute_pareto_frontier(df, latency_col="total_ms", error_col="bad2_pct")
    if save_csv:
        csv_path = RESULTS_DIR / "optimization.csv"
        df.to_csv(csv_path, index=False)

    return df


# ---------------------------------------------------------------------------
# CPU Profiling Across Resolutions and Threading
# ---------------------------------------------------------------------------

def run_cpu_profiling(
    input_image: Optional[np.ndarray] = None,
    right_image: Optional[np.ndarray] = None,
    iterations: int = 10,
    save_json: bool = True
) -> Dict[str, Any]:
    """Profile CPU execution across resolutions (720p, 360p, 180p) and threads (1 to 16)."""
    if input_image is None or right_image is None:
        pt_left = field.DEFAULT_GUNUNG_ROOT / "puncakTinggi" / "stereo" / "left_0000.png"
        pt_right = field.DEFAULT_GUNUNG_ROOT / "puncakTinggi" / "stereo" / "right_0000.png"
        jp_left = field.DEFAULT_MIDDLEBURY_ROOT / "Jadeplant" / "im0.png"
        jp_right = field.DEFAULT_MIDDLEBURY_ROOT / "Jadeplant" / "im1.png"

        if pt_left.exists() and pt_right.exists():
            input_image = cv2.imread(str(pt_left))
            right_image = cv2.imread(str(pt_right))
        elif jp_left.exists() and jp_right.exists():
            # Resize Middlebury Jadeplant to 720p (1280x720) for standardized CPU profiling
            im0 = cv2.imread(str(jp_left))
            im1 = cv2.imread(str(jp_right))
            input_image = cv2.resize(im0, (1280, 720), interpolation=cv2.INTER_AREA)
            right_image = cv2.resize(im1, (1280, 720), interpolation=cv2.INTER_AREA)
        else:
            synth_l, synth_r, _, _ = stereo.generate_layered_synthetic_scene(1280, 720, seed=42)
            input_image = synth_l
            right_image = synth_r

    h_full, w_full = input_image.shape[:2]

    # Convert to 1-channel grayscale to benchmark single-channel OpenCV SGBM matching
    gray_full_l = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY) if input_image.ndim == 3 else input_image
    gray_full_r = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY) if right_image.ndim == 3 else right_image

    # Resolution scaling experiments
    scales = {"720p": 1.0, "360p": 0.5, "180p": 0.25}
    res_results: Dict[str, Dict[str, float]] = {}

    for res_name, sc in scales.items():
        w_s, h_s = int(w_full * sc), int(h_full * sc)
        img_l = cv2.resize(gray_full_l, (w_s, h_s), interpolation=cv2.INTER_AREA)
        img_r = cv2.resize(gray_full_r, (w_s, h_s), interpolation=cv2.INTER_AREA)

        num_disp = int(np.ceil((160 * sc) / 16.0) * 16)
        matcher = stereo.create_sgbm_matcher(num_disparities=num_disp, block_size=5)

        # Warm-up
        matcher.compute(img_l, img_r)

        times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            matcher.compute(img_l, img_r)
            times.append((time.perf_counter() - t0) * 1000.0)

        res_results[res_name] = {
            "scale": sc,
            "width": w_s,
            "height": h_s,
            "num_disparities": num_disp,
            "mean_ms": float(np.mean(times)),
            "std_ms": float(np.std(times)),
            "fps": float(1000.0 / np.mean(times))
        }

    # Threading experiments at 360p (balanced configuration)
    img_360_l = cv2.resize(gray_full_l, (640, 360), interpolation=cv2.INTER_AREA)
    img_360_r = cv2.resize(gray_full_r, (640, 360), interpolation=cv2.INTER_AREA)
    m_thread = stereo.create_sgbm_matcher(num_disparities=80, block_size=5)

    thread_counts = [1, 2, 4, 8, 16]
    thread_results: Dict[str, float] = {}
    orig_threads = cv2.getNumThreads()

    for n_th in thread_counts:
        cv2.setNumThreads(n_th)
        # Warm-up
        m_thread.compute(img_360_l, img_360_r)

        times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            m_thread.compute(img_360_l, img_360_r)
            times.append((time.perf_counter() - t0) * 1000.0)
        thread_results[f"threads_{n_th}"] = float(np.mean(times))

    # Restore original thread setting
    cv2.setNumThreads(orig_threads)

    # Preset benchmark
    ref_dir = field.DEFAULT_MIDDLEBURY_ROOT / "Jadeplant"
    if ref_dir.exists():
        im0_ref = cv2.imread(str(ref_dir / "im0.png"))
        im1_ref = cv2.imread(str(ref_dir / "im1.png"))
        disp_gt_ref = stereo.load_pfm(str(ref_dir / "disp0GT.pfm"))
        mask_ref = cv2.imread(str(ref_dir / "mask0nocc.png"), cv2.IMREAD_GRAYSCALE) == 255
    else:
        im0_ref, im1_ref, disp_gt_ref, mask_ref = stereo.generate_layered_synthetic_scene(320, 240, seed=42)

    preset_results: Dict[str, Dict[str, Any]] = {}
    for p_name in ["FAST", "BALANCED", "QUALITY"]:
        # Accuracy evaluation on reference scene
        d_ref, _ = run_stereo_pipeline(im0_ref, im1_ref, preset=p_name)
        m_acc = stereo.compute_full_disparity_metrics(d_ref, disp_gt_ref, mask_ref)

        # Warmup on profiling image pair
        run_stereo_pipeline(input_image, right_image, preset=p_name)
        p_times = []
        for _ in range(iterations):
            _, stats = run_stereo_pipeline(input_image, right_image, preset=p_name)
            p_times.append(stats["wall_ms"])

        preset_results[p_name] = {
            "scale": PIPELINE_PRESETS[p_name]["scale"],
            "num_disparities": PIPELINE_PRESETS[p_name]["numDisparities"],
            "block_size": PIPELINE_PRESETS[p_name]["blockSize"],
            "mean_ms": float(np.mean(p_times)),
            "std_ms": float(np.std(p_times)),
            "fps": float(1000.0 / np.mean(p_times)),
            "bad2_cond_pct": float(m_acc["bad2_pct"]),
            "bad2_ca_pct": float(m_acc["bad2_ca_pct"]),
            "coverage_pct": float(m_acc["coverage_pct"]),
            "disp_mae_px": float(m_acc["disp_mae_px"])
        }

    profile_summary = {
        "resolutions": res_results,
        "threads": thread_results,
        "presets": preset_results
    }

    if save_json:
        best_params_path = RESULTS_DIR / "best_params.json"
        with open(best_params_path, "w", encoding="utf-8") as f:
            json.dump({
                "pipeline_presets": PIPELINE_PRESETS,
                "benchmarked_performance": profile_summary
            }, f, indent=2)

    return profile_summary
