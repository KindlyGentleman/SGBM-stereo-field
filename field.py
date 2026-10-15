"""Middlebury 2014 benchmark and Gunung volcanic terrain field evaluation.

Integrates real-world stereo evaluation across:
1. Middlebury 2014 benchmark (PFM ground truth, calib.txt, non-occluded evaluation masks).
2. Four Gunung volcanic terrains (puncakTinggi, puncakWarung, tanggaBiru, tanggaKuning).
3. Intel RealSense D435i depth registration via PnP RANSAC and z-buffered forward projection.
4. Metric depth validation and honest failure reporting (e.g. out-of-range scenes).
"""

from pathlib import Path
import re
import time
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np
import pandas as pd

import stereo

# Default dataset locations
DEFAULT_MIDDLEBURY_ROOT = Path("data/middlebury/MiddEval3/trainingQ")
DEFAULT_GUNUNG_ROOT = Path(r"D:\02_Riset\Research Paper\source_code\comparison\gunung")
DEFAULT_CALIB_PATH = Path(r"D:\02_Riset\Research Paper\source_code\calibration\calib_params\stereo_params_advanced.npz")

# Rectified left camera intrinsics at 1280x720
K_RECT = np.array([
    [964.68366, 0.0, 563.89656],
    [0.0, 964.68366, 366.00543],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

# RealSense color camera intrinsics at 640x480
K_RS = np.array([
    [615.0, 0.0, 320.0],
    [0.0, 615.0, 240.0],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

# Stereocamera baseline and focal length product (fx * B)
# fx = 964.68366 px, B = 0.097497 m -> fx * B = 94.0537 px * m
FB_PRODUCT = 94.0537

GUNUNG_TERRAINS = ["puncakTinggi", "puncakWarung", "tanggaBiru", "tanggaKuning"]


# ---------------------------------------------------------------------------
# Middlebury 2014 Utilities
# ---------------------------------------------------------------------------

def parse_calib_txt(calib_path: str) -> Dict[str, Any]:
    """Parse Middlebury calib.txt configuration file.

    Extracts cam0, cam1, doffs, baseline, width, height, and ndisp.
    """
    path = Path(calib_path)
    if not path.exists():
        raise FileNotFoundError(f"calib.txt not found: {calib_path}")

    params: Dict[str, Any] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()

                if key in ["cam0", "cam1"]:
                    clean = re.sub(r"[\[\]]", "", val)
                    rows = clean.split(";")
                    mat = []
                    for row in rows:
                        mat.append([float(x) for x in row.strip().split()])
                    params[key] = np.array(mat, dtype=np.float64)
                elif key in ["doffs", "baseline", "dyavg", "dymax"]:
                    params[key] = float(val)
                elif key in ["width", "height", "ndisp", "vmin", "vmax", "isint"]:
                    params[key] = int(val)
                else:
                    params[key] = val

    return params


def discover_middlebury_scenes(data_dir: Path) -> List[Path]:
    """Find all valid Middlebury scenes with im0, im1, disp0GT, and mask0nocc."""
    scenes = []
    if not data_dir.exists():
        return scenes

    for item in sorted(data_dir.iterdir()):
        if item.is_dir():
            has_im0 = (item / "im0.png").exists()
            has_im1 = (item / "im1.png").exists()
            has_gt = (item / "disp0GT.pfm").exists()
            has_mask = (item / "mask0nocc.png").exists()
            if has_im0 and has_im1 and has_gt and has_mask:
                scenes.append(item)
    return scenes


def evaluate_middlebury_scene(
    scene_dir: Path,
    matcher: Optional[cv2.StereoSGBM] = None,
    filter_func: Optional[Any] = None,
    scale: float = 1.0
) -> Dict[str, Any]:
    """Evaluate stereo matching on a single Middlebury 2014 scene.

    Args:
        scene_dir: Path to Middlebury scene directory.
        matcher: Optional preconfigured matcher instance.
        filter_func: Optional post-processing filter callable.
        scale: Input downscale factor (e.g. 0.5 for fast evaluation).

    Returns:
        Dictionary of measured disparity and metric depth statistics.
    """
    im0_path = scene_dir / "im0.png"
    im1_path = scene_dir / "im1.png"
    gt_path = scene_dir / "disp0GT.pfm"
    mask_path = scene_dir / "mask0nocc.png"
    calib_path = scene_dir / "calib.txt"

    im0 = cv2.imread(str(im0_path))
    im1 = cv2.imread(str(im1_path))
    disp_gt = stereo.load_pfm(str(gt_path))
    mask_nocc = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) == 255
    calib = parse_calib_txt(str(calib_path)) if calib_path.exists() else {}

    h_orig, w_orig = im0.shape[:2]
    ndisp = calib.get("ndisp", 160)

    # Scale inputs if requested
    if scale != 1.0:
        w_s, h_s = int(w_orig * scale), int(h_orig * scale)
        in0 = cv2.resize(im0, (w_s, h_s), interpolation=cv2.INTER_AREA)
        in1 = cv2.resize(im1, (w_s, h_s), interpolation=cv2.INTER_AREA)
        ndisp_scaled = int(np.ceil((ndisp * scale) / 16.0) * 16)
    else:
        in0, in1 = im0, im1
        ndisp_scaled = int(np.ceil(ndisp / 16.0) * 16)

    # Configure matcher if not supplied
    if matcher is None:
        matcher = stereo.create_sgbm_matcher(num_disparities=ndisp_scaled, block_size=7)

    t0 = time.perf_counter()
    if filter_func is not None:
        disp_pred = filter_func(in0, in1, matcher)
    else:
        disp_pred, _ = stereo.compute_disparity_sgbm(in0, in1, matcher)

    # Upscale back to native resolution if scaled
    if scale != 1.0:
        disp_pred = cv2.resize(disp_pred, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR) * (1.0 / scale)

    runtime_ms = (time.perf_counter() - t0) * 1000.0

    # Calculate standard disparity metrics on non-occluded mask
    bad1 = stereo.compute_bad_pixel_rate(disp_pred, disp_gt, mask_nocc, threshold=1.0)
    bad2 = stereo.compute_bad_pixel_rate(disp_pred, disp_gt, mask_nocc, threshold=2.0)
    bad4 = stereo.compute_bad_pixel_rate(disp_pred, disp_gt, mask_nocc, threshold=4.0)
    bad2_ca = stereo.compute_bad_pixel_rate(disp_pred, disp_gt, mask_nocc, threshold=2.0, completeness_aware=True)
    mae = stereo.compute_depth_mae(disp_pred, disp_gt, mask_nocc)
    rmse = stereo.compute_depth_rmse(disp_pred, disp_gt, mask_nocc)
    coverage = stereo.compute_valid_coverage(disp_pred, mask_nocc)

    # Calculate metric depth errors if calibration is present
    fx = calib.get("cam0", np.eye(3))[0, 0]
    baseline_m = calib.get("baseline", 0.0) / 1000.0  # mm to meters
    doffs = calib.get("doffs", 0.0)

    depth_mae_m = float("nan")
    depth_rmse_m = float("nan")
    if fx > 0 and baseline_m > 0:
        depth_gt = stereo.disparity_to_depth(disp_gt, fx, baseline_m, doffs=doffs)
        depth_pred = stereo.disparity_to_depth(disp_pred, fx, baseline_m, doffs=doffs)
        joint_eval = mask_nocc & (depth_gt > 0) & (depth_pred > 0) & np.isfinite(depth_pred)
        if np.count_nonzero(joint_eval) > 0:
            diff_z = np.abs(depth_pred[joint_eval] - depth_gt[joint_eval])
            depth_mae_m = float(np.mean(diff_z))
            depth_rmse_m = float(np.sqrt(np.mean(diff_z ** 2)))

    return {
        "scene": scene_dir.name,
        "width": w_orig,
        "height": h_orig,
        "ndisp": ndisp,
        "runtime_ms": runtime_ms,
        "bad1_pct": bad1,
        "bad2_pct": bad2,
        "bad4_pct": bad4,
        "bad2_ca_pct": bad2_ca,
        "disp_mae_px": mae,
        "disp_rmse_px": rmse,
        "coverage_pct": coverage,
        "depth_mae_m": depth_mae_m,
        "depth_rmse_m": depth_rmse_m
    }


def run_middlebury_benchmark(
    data_dir: Optional[Path] = None,
    scenes: Optional[List[str]] = None,
    scale: float = 1.0,
    filter_func: Optional[Any] = None
) -> pd.DataFrame:
    """Run evaluation across Middlebury scenes and return summary DataFrame."""
    root = data_dir or DEFAULT_MIDDLEBURY_ROOT
    all_scenes = discover_middlebury_scenes(root)

    if scenes:
        all_scenes = [s for s in all_scenes if s.name in scenes]

    rows = []
    for sc in all_scenes:
        res = evaluate_middlebury_scene(sc, scale=scale, filter_func=filter_func)
        rows.append(res)

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# Gunung Field and RealSense Depth Registration Utilities
# ---------------------------------------------------------------------------

def load_realsense_depth(npz_path: str) -> np.ndarray:
    """Load RealSense depth map from NPZ archive and convert mm to meters."""
    path = Path(npz_path)
    if not path.exists():
        raise FileNotFoundError(f"RealSense NPZ not found: {npz_path}")
    with np.load(str(path), allow_pickle=False) as data:
        raw_mm = data["depth_map"]
    depth_m = np.zeros_like(raw_mm, dtype=np.float32)
    valid = (raw_mm > 0) & (raw_mm < 65535)
    depth_m[valid] = raw_mm[valid].astype(np.float32) / 1000.0
    return depth_m


def register_realsense_pnp(
    rect_left: np.ndarray,
    rs_rgb: np.ndarray,
    rs_depth_raw_mm: np.ndarray,
    K_rect: np.ndarray = K_RECT,
    K_rs: np.ndarray = K_RS,
    reproj_threshold: float = 3.0,
    n_features: int = 3000
) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Dict[str, float]]:
    """Register RealSense color sensor to rectified stereo left camera via PnP RANSAC.

    Uses 2D-3D SIFT feature matching:
    - 3D points are unprojected from RealSense RGB-D using K_rs and raw depth.
    - 2D points are detected in the rectified stereo left image.
    - PnP RANSAC finds rigid rotation R and translation t.

    Returns:
        success: Boolean flag indicating if registration succeeded.
        R_mat: 3x3 rotation matrix (RealSense to stereo left frame).
        tvec: 3x1 translation vector in meters.
        inliers: Array of inlier correspondence indices.
        stats: Dictionary of matching counts and reprojection errors.
    """
    stats = {
        "sift_matches": 0,
        "valid_3d_2d_pairs": 0,
        "inliers": 0,
        "inlier_ratio_pct": 0.0,
        "reproj_mae_px": float("nan"),
        "reproj_med_px": float("nan")
    }

    sift = cv2.SIFT_create(nfeatures=n_features)
    kp_st, des_st = sift.detectAndCompute(rect_left, None)
    kp_rs, des_rs = sift.detectAndCompute(rs_rgb, None)

    if des_st is None or des_rs is None or len(kp_st) < 10 or len(kp_rs) < 10:
        return False, None, None, None, stats

    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des_rs, des_st, k=2)
    good = [m for m, n in matches if len(matches) > 0 and m.distance < 0.75 * n.distance]
    stats["sift_matches"] = len(good)

    h_rs, w_rs = rs_depth_raw_mm.shape[:2]
    pts_3d = []
    pts_2d = []

    for m in good:
        u_rs, v_rs = kp_rs[m.queryIdx].pt
        u_i, v_i = int(round(u_rs)), int(round(v_rs))
        if 0 <= u_i < w_rs and 0 <= v_i < h_rs:
            z_mm = rs_depth_raw_mm[v_i, u_i]
            # Consider depths between 0.3m and 6.0m
            if 300 < z_mm < 6000:
                z_m = float(z_mm) / 1000.0
                x_m = (u_rs - K_rs[0, 2]) * z_m / K_rs[0, 0]
                y_m = (v_rs - K_rs[1, 2]) * z_m / K_rs[1, 1]
                pts_3d.append([x_m, y_m, z_m])
                pts_2d.append([kp_st[m.trainIdx].pt[0], kp_st[m.trainIdx].pt[1]])

    stats["valid_3d_2d_pairs"] = len(pts_3d)
    if len(pts_3d) < 8:
        return False, None, None, None, stats

    arr_3d = np.array(pts_3d, dtype=np.float64)
    arr_2d = np.array(pts_2d, dtype=np.float64)

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        arr_3d,
        arr_2d,
        K_rect,
        None,
        reprojectionError=reproj_threshold,
        iterationsCount=3000,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success or inliers is None or len(inliers) < 6:
        return False, None, None, None, stats

    R_mat, _ = cv2.Rodrigues(rvec)
    inlier_idx = inliers.ravel()
    proj_pts, _ = cv2.projectPoints(arr_3d[inlier_idx], rvec, tvec, K_rect, None)
    reproj_errs = np.linalg.norm(proj_pts.squeeze() - arr_2d[inlier_idx], axis=1)

    stats["inliers"] = len(inliers)
    stats["inlier_ratio_pct"] = float(len(inliers) / len(pts_3d) * 100.0)
    stats["reproj_mae_px"] = float(np.mean(reproj_errs))
    stats["reproj_med_px"] = float(np.median(reproj_errs))

    return True, R_mat, tvec, inliers, stats


def project_realsense_depth_to_stereo(
    rs_depth_raw_mm: np.ndarray,
    R_mat: np.ndarray,
    tvec: np.ndarray,
    K_rs: np.ndarray = K_RS,
    K_rect: np.ndarray = K_RECT,
    out_shape: Tuple[int, int] = (720, 1280),
    min_depth: float = 0.3,
    max_depth: float = 6.0
) -> np.ndarray:
    """Project RealSense depth into rectified stereo left view using z-buffering.

    Nearer 3D points overwrite farther 3D points when projecting to the same pixel.
    """
    h_out, w_out = out_shape
    ref_depth = np.zeros((h_out, w_out), dtype=np.float32)

    min_mm = int(min_depth * 1000.0)
    max_mm = int(max_depth * 1000.0)
    v_idx, u_idx = np.where((rs_depth_raw_mm > min_mm) & (rs_depth_raw_mm < max_mm))
    if len(v_idx) == 0:
        return ref_depth

    z_m = rs_depth_raw_mm[v_idx, u_idx].astype(np.float32) / 1000.0
    pts_rs = np.column_stack([
        (u_idx - K_rs[0, 2]) * z_m / K_rs[0, 0],
        (v_idx - K_rs[1, 2]) * z_m / K_rs[1, 1],
        z_m
    ])

    pts_st = (R_mat @ pts_rs.T + tvec.reshape(3, 1)).T
    z_proj = pts_st[:, 2]

    u_proj = np.round(K_rect[0, 0] * pts_st[:, 0] / z_proj + K_rect[0, 2]).astype(int)
    v_proj = np.round(K_rect[1, 1] * pts_st[:, 1] / z_proj + K_rect[1, 2]).astype(int)

    valid_mask = (u_proj >= 0) & (u_proj < w_out) & (v_proj >= 0) & (v_proj < h_out) & (z_proj > min_depth)
    u_val = u_proj[valid_mask]
    v_val = v_proj[valid_mask]
    z_val = z_proj[valid_mask]

    # Z-buffering: sort by depth descending so nearer points overwrite farther points
    sort_order = np.argsort(-z_val)
    ref_depth[v_val[sort_order], u_val[sort_order]] = z_val[sort_order]
    return ref_depth


def compute_metric_depth_errors(
    pred_depth: np.ndarray,
    ref_depth: np.ndarray,
    min_z: float = 0.4,
    max_z: float = 4.0
) -> Dict[str, float]:
    """Compute depth error statistics against registered reference depth map."""
    eval_ref = (ref_depth >= min_z) & (ref_depth <= max_z) & np.isfinite(ref_depth)
    n_ref = np.count_nonzero(eval_ref)
    if n_ref == 0:
        return {
            "eval_pixels": 0,
            "ref_coverage_pct": 0.0,
            "mae_m": float("nan"),
            "median_m": float("nan"),
            "rmse_m": float("nan"),
            "delta_125_pct": float("nan")
        }

    joint = eval_ref & (pred_depth >= min_z) & (pred_depth <= (max_z * 1.5)) & np.isfinite(pred_depth)
    n_joint = np.count_nonzero(joint)
    cov_pct = float(n_joint / n_ref * 100.0)

    if n_joint == 0:
        return {
            "eval_pixels": 0,
            "ref_coverage_pct": 0.0,
            "mae_m": float("nan"),
            "median_m": float("nan"),
            "rmse_m": float("nan"),
            "delta_125_pct": float("nan")
        }

    pred_z = pred_depth[joint]
    ref_z = ref_depth[joint]
    diff = np.abs(pred_z - ref_z)

    mae = float(np.mean(diff))
    med = float(np.median(diff))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    d125 = float(np.mean(np.maximum(pred_z / ref_z, ref_z / pred_z) < 1.25) * 100.0)

    return {
        "eval_pixels": int(n_joint),
        "ref_coverage_pct": cov_pct,
        "depth_mae_m": mae,
        "depth_median_m": med,
        "depth_rmse_m": rmse,
        "delta_125_pct": d125
    }


def evaluate_gunung_terrain(
    terrain_name: str,
    gunung_root: Path = DEFAULT_GUNUNG_ROOT,
    calib_path: Path = DEFAULT_CALIB_PATH,
    stereo_pipeline_func: Optional[Any] = None
) -> Tuple[Dict[str, Any], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Evaluate stereo matching and RealSense depth registration for one Gunung terrain.

    For terrains with far-field geometry (>10m, e.g. puncakWarung), registration is
    intentionally skipped to prevent fabricating false ground truth.

    Returns:
        record: Dictionary of registration status and error metrics.
        rect_l: Rectified left color image (1280x720).
        ref_depth: Registered RealSense depth map (or None if out of range).
        st_depth: Stereo predicted depth map in meters.
    """
    st_dir = gunung_root / terrain_name / "stereo"
    rs_dir = gunung_root / terrain_name / "hardware" / "realsense"

    raw_l_path = st_dir / "left_0000.png"
    raw_r_path = st_dir / "right_0000.png"
    rs_rgb_path = rs_dir / "rgb_0000.png"
    rs_npz_path = rs_dir / "depth_0000.npz"

    if not raw_l_path.exists() or not raw_r_path.exists():
        raise FileNotFoundError(f"Stereo images missing for terrain: {terrain_name}")

    calib = stereo.load_calibration(str(calib_path))
    maps_l, maps_r, _ = stereo.build_rectification_maps(calib, (1280, 720))

    raw_l = cv2.imread(str(raw_l_path))
    raw_r = cv2.imread(str(raw_r_path))
    rect_l, rect_r = stereo.rectify_pair(raw_l, raw_r, maps_l, maps_r)

    # Compute stereo disparity and depth
    if stereo_pipeline_func is not None:
        disp_st, pipe_stats = stereo_pipeline_func(rect_l, rect_r)
        runtime_ms = pipe_stats.get("wall_ms", 0.0)
    else:
        m = stereo.create_sgbm_matcher(num_disparities=160, block_size=7)
        disp_st, runtime_ms = stereo.compute_disparity_sgbm(rect_l, rect_r, m)

    st_depth = np.zeros_like(disp_st, dtype=np.float32)
    valid_disp = (disp_st > 1.0) & np.isfinite(disp_st)
    st_depth[valid_disp] = FB_PRODUCT / disp_st[valid_disp]

    record: Dict[str, Any] = {
        "terrain": terrain_name,
        "runtime_ms": runtime_ms,
        "valid_disp_pct": stereo.compute_valid_coverage(disp_st),
        "status": "UNREGISTERED",
        "sift_matches": 0,
        "inliers": 0,
        "reproj_med_px": float("nan"),
        "tx_m": float("nan"),
        "ty_m": float("nan"),
        "tz_m": float("nan"),
        "eval_pixels": 0,
        "ref_coverage_pct": 0.0,
        "depth_mae_m": float("nan"),
        "depth_median_m": float("nan"),
        "depth_rmse_m": float("nan"),
        "delta_125_pct": float("nan"),
        "warp_mae": stereo.compute_photometric_warp_mae(rect_l, rect_r, disp_st)
    }

    # If terrain is puncakWarung, geometry is far-field (>15m), outside RealSense range
    if terrain_name == "puncakWarung":
        record["status"] = "OUT_OF_RANGE"
        return record, rect_l, None, st_depth

    if not rs_rgb_path.exists() or not rs_npz_path.exists():
        record["status"] = "NO_REALSENSE_DATA"
        return record, rect_l, None, st_depth

    rs_rgb = cv2.imread(str(rs_rgb_path))
    rs_depth_raw = np.load(str(rs_npz_path), allow_pickle=False)["depth_map"]

    success, R_mat, tvec, inliers, reg_stats = register_realsense_pnp(
        rect_l, rs_rgb, rs_depth_raw, K_RECT, K_RS
    )

    record["sift_matches"] = reg_stats["sift_matches"]
    record["inliers"] = reg_stats["inliers"]
    record["reproj_med_px"] = reg_stats["reproj_med_px"]

    if not success or R_mat is None or tvec is None:
        record["status"] = "REGISTRATION_FAILED"
        return record, rect_l, None, st_depth

    record["status"] = "VERIFIED"
    record["tx_m"] = float(tvec[0, 0])
    record["ty_m"] = float(tvec[1, 0])
    record["tz_m"] = float(tvec[2, 0])

    ref_depth = project_realsense_depth_to_stereo(
        rs_depth_raw, R_mat, tvec, K_RS, K_RECT, (720, 1280), min_depth=0.3, max_depth=6.0
    )

    depth_stats = compute_metric_depth_errors(st_depth, ref_depth, min_z=0.4, max_z=4.0)
    record.update(depth_stats)

    return record, rect_l, ref_depth, st_depth


def run_gunung_benchmark(
    gunung_root: Path = DEFAULT_GUNUNG_ROOT,
    calib_path: Path = DEFAULT_CALIB_PATH,
    stereo_pipeline_func: Optional[Any] = None
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Run evaluation across all four Gunung volcanic terrains."""
    records = []
    refs = {}
    preds = {}

    for t in GUNUNG_TERRAINS:
        rec, rect_l, ref_d, st_d = evaluate_gunung_terrain(
            t, gunung_root=gunung_root, calib_path=calib_path, stereo_pipeline_func=stereo_pipeline_func
        )
        records.append(rec)
        if ref_d is not None:
            refs[t] = ref_d
        if st_d is not None:
            preds[t] = st_d

    df = pd.DataFrame(records)
    return df, refs, preds
