"""Stereo matching, calibration, epipolar rectification, and evaluation utilities.

Supports OpenCV StereoBM, StereoSGBM, and edge-aware WLS filtering with
completeness-aware benchmark metrics and unsupervised field diagnostics.
"""

from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import cv2
import numpy as np


def load_calibration(npz_path: str) -> Dict[str, np.ndarray]:
    """Load camera calibration matrices from NPZ archive safely."""
    path = Path(npz_path)
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found at: {npz_path}")
    
    with np.load(str(path), allow_pickle=False) as data:
        calib = {
            "mtx_left": data["mtx_left"],
            "dist_left": data["dist_left"],
            "mtx_right": data["mtx_right"],
            "dist_right": data["dist_right"],
            "R": data["R"],
            "T": data["T"],
        }
        if "Q" in data.files:
            calib["Q"] = data["Q"]
    return calib


def build_rectification_maps(
    calib: Dict[str, np.ndarray],
    img_size: Tuple[int, int] = (1280, 720),
    alpha: float = 0.0
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """Compute undistort and rectification remap coordinates using stereoRectify."""
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        calib["mtx_left"],
        calib["dist_left"],
        calib["mtx_right"],
        calib["dist_right"],
        img_size,
        calib["R"],
        calib["T"],
        alpha=alpha
    )
    map_l1, map_l2 = cv2.initUndistortRectifyMap(
        calib["mtx_left"], calib["dist_left"], R1, P1, img_size, cv2.CV_32FC1
    )
    map_r1, map_r2 = cv2.initUndistortRectifyMap(
        calib["mtx_right"], calib["dist_right"], R2, P2, img_size, cv2.CV_32FC1
    )
    return (map_l1, map_l2), (map_r1, map_r2), Q


def rectify_pair(
    left_img: np.ndarray,
    right_img: np.ndarray,
    map_left: Tuple[np.ndarray, np.ndarray],
    map_right: Tuple[np.ndarray, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """Remap stereo pair to collinear epipolar scanlines."""
    rect_l = cv2.remap(left_img, map_left[0], map_left[1], cv2.INTER_LINEAR)
    rect_r = cv2.remap(right_img, map_right[0], map_right[1], cv2.INTER_LINEAR)
    return rect_l, rect_r


def create_sgbm_matcher(params: Optional[Dict[str, Any]] = None) -> cv2.StereoSGBM:
    """Create a configured StereoSGBM instance."""
    defaults = {
        "minDisparity": 0,
        "numDisparities": 160,
        "blockSize": 7,
        "P1": 150,
        "P2": 1200,
        "disp12MaxDiff": 1,
        "uniquenessRatio": 10,
        "speckleWindowSize": 100,
        "speckleRange": 2,
        "mode": cv2.STEREO_SGBM_MODE_SGBM_3WAY
    }
    if params:
        defaults.update(params)
    
    # Ensure numDisparities is a multiple of 16
    defaults["numDisparities"] = int(np.ceil(defaults["numDisparities"] / 16.0) * 16)
    
    return cv2.StereoSGBM_create(
        minDisparity=defaults["minDisparity"],
        numDisparities=defaults["numDisparities"],
        blockSize=defaults["blockSize"],
        P1=defaults["P1"],
        P2=defaults["P2"],
        disp12MaxDiff=defaults["disp12MaxDiff"],
        uniquenessRatio=defaults["uniquenessRatio"],
        speckleWindowSize=defaults["speckleWindowSize"],
        speckleRange=defaults["speckleRange"],
        mode=defaults["mode"]
    )


def compute_disparity_wls(
    left_matcher: cv2.StereoSGBM,
    left_img: np.ndarray,
    right_img: np.ndarray,
    lmbda: float = 8000.0,
    sigma: float = 1.5
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute left disparity and apply edge-aware WLS filtering."""
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)
    
    disp_l_raw = left_matcher.compute(left_img, right_img)
    disp_r_raw = right_matcher.compute(right_img, left_img)
    
    disp_filtered = wls_filter.filter(disp_l_raw, left_img, None, disp_r_raw)
    disp_out = disp_filtered.astype(np.float32) / 16.0
    disp_r_out = disp_r_raw.astype(np.float32) / 16.0
    return disp_out, disp_r_out


def compute_valid_coverage(disp: np.ndarray) -> float:
    """Compute percentage of valid positive disparity predictions."""
    valid_count = np.count_nonzero(disp > 0)
    total_count = disp.size
    return float(valid_count / total_count * 100.0)


def compute_lr_consistency(
    disp_l: np.ndarray,
    disp_r: np.ndarray,
    threshold: float = 1.5
) -> Tuple[float, np.ndarray]:
    """Check round-trip Left-Right consistency across matched scanline points."""
    h, w = disp_l.shape
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
    
    # Map each pixel in left image to candidate x in right image
    x_matched = np.clip(np.round(x_coords - disp_l).astype(int), 0, w - 1)
    r_disp_matched = disp_r[y_coords, x_matched]
    
    # Handle OpenCV sign conventions where right matcher disparity can be negated
    if np.median(disp_r) < 0:
        r_disp_matched = -r_disp_matched
        
    lr_diff = np.abs(disp_l - r_disp_matched)
    valid_mask = disp_l > 0
    consistent_mask = (lr_diff <= threshold) & valid_mask
    
    valid_pixels = np.count_nonzero(valid_mask)
    if valid_pixels == 0:
        return 0.0, consistent_mask
    
    consistency_pct = float(np.count_nonzero(consistent_mask) / valid_pixels * 100.0)
    return consistency_pct, consistent_mask


def compute_photometric_warp_mae(
    left_img: np.ndarray,
    right_img: np.ndarray,
    disp_l: np.ndarray
) -> float:
    """Compute mean absolute difference between left image and warped right image."""
    h, w = disp_l.shape
    gray_l = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY).astype(np.float32) if left_img.ndim == 3 else left_img.astype(np.float32)
    gray_r = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY).astype(np.float32) if right_img.ndim == 3 else right_img.astype(np.float32)
    
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
    x_warp = np.clip(x_coords - disp_l, 0, w - 1).astype(np.float32)
    warped_r = cv2.remap(gray_r, x_warp, y_coords.astype(np.float32), cv2.INTER_LINEAR)
    
    valid_mask = disp_l > 0
    if np.count_nonzero(valid_mask) == 0:
        return 999.0
    
    mae = float(np.mean(np.abs(gray_l[valid_mask] - warped_r[valid_mask])))
    return mae


def compute_completeness_aware_bad_pixel(
    disp_pred: np.ndarray,
    disp_gt: np.ndarray,
    valid_mask: np.ndarray,
    threshold: float = 2.0
) -> float:
    """Completeness-aware bad-pixel rate: missing predictions count as errors."""
    gt_valid = valid_mask & (disp_gt > 0) & (disp_gt < 1e8)
    n_gt = np.sum(gt_valid)
    if n_gt == 0:
        return 0.0
    
    pred_valid = (disp_pred > 0) & gt_valid
    err = np.abs(disp_pred - disp_gt)
    
    bad_err = pred_valid & (err > threshold)
    missing = gt_valid & (~(disp_pred > 0))
    
    bad_rate = float((np.sum(bad_err) + np.sum(missing)) / n_gt * 100.0)
    return bad_rate


def reproject_to_pointcloud(
    disp: np.ndarray,
    Q: np.ndarray,
    color_img: Optional[np.ndarray] = None,
    min_disp: float = 0.5,
    max_depth: float = 50.0
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Reproject disparity map into metric 3D point cloud using matrix Q."""
    points_3d = cv2.reprojectImageTo3D(disp.astype(np.float32), Q)
    z_coords = points_3d[:, :, 2]
    
    valid_mask = (disp > min_disp) & (z_coords > 0.1) & (z_coords < max_depth) & np.isfinite(z_coords)
    valid_pts = points_3d[valid_mask]
    
    valid_colors = None
    if color_img is not None:
        rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        valid_colors = rgb[valid_mask]
        
    return valid_pts, valid_colors


def save_pointcloud_ply(
    filepath: str,
    points: np.ndarray,
    colors: Optional[np.ndarray] = None
) -> None:
    """Save 3D point cloud to Stanford PLY format."""
    p_path = Path(filepath)
    p_path.parent.mkdir(parents=True, exist_ok=True)
    
    n_pts = len(points)
    with open(p_path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n_pts}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")
        
        if colors is not None:
            for pt, col in zip(points, colors):
                f.write(f"{pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f} {int(col[0])} {int(col[1])} {int(col[2])}\n")
        else:
            for pt in points:
                f.write(f"{pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f}\n")


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


def disparity_to_depth(
    disp: np.ndarray,
    focal_length_px: float,
    baseline_m: float,
    min_disp: float = 0.5
) -> np.ndarray:
    """Convert disparity map in pixels to metric depth map in meters."""
    depth = np.zeros_like(disp, dtype=np.float32)
    valid = (disp >= min_disp) & np.isfinite(disp)
    depth[valid] = (focal_length_px * baseline_m) / disp[valid]
    return depth


def load_realsense_depth(npz_path: str) -> np.ndarray:
    """Load RealSense depth map from NPZ archive and convert mm to meters."""
    p = Path(npz_path)
    if not p.exists():
        raise FileNotFoundError(f"RealSense NPZ not found: {npz_path}")
    with np.load(str(p), allow_pickle=False) as data:
        raw_mm = data["depth_map"]
    depth_m = np.zeros_like(raw_mm, dtype=np.float32)
    valid = (raw_mm > 0) & (raw_mm < 65535)
    depth_m[valid] = raw_mm[valid].astype(np.float32) / 1000.0
    return depth_m


def project_realsense_depth_to_stereo(
    rs_depth_raw_mm: np.ndarray,
    R_mat: np.ndarray,
    tvec: np.ndarray,
    K_rs: np.ndarray,
    K_rect: np.ndarray,
    out_shape: Tuple[int, int] = (720, 1280),
    min_depth: float = 0.3,
    max_depth: float = 6.0
) -> np.ndarray:
    """Project RealSense depth map into rectified stereo left view using z-buffering."""
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


def apply_filter(
    filter_name: str,
    disp_raw: np.ndarray,
    left_img: np.ndarray,
    disp_r_raw: Optional[np.ndarray] = None,
    matcher: Optional[cv2.StereoSGBM] = None
) -> np.ndarray:
    """Apply one of the evaluated stereo filter candidates under identical inputs."""
    disp_clean = disp_raw.copy()
    disp_clean[disp_clean <= 0] = 0.0

    name = filter_name.lower().strip()
    if name in ["f0_raw", "raw"]:
        return disp_clean

    elif name in ["f1_bilateral_legacy", "bilateral_legacy"]:
        # The previous edge-preserving implementation that blurred across zero sentinels
        return cv2.bilateralFilter(disp_clean, d=9, sigmaColor=75, sigmaSpace=75)

    elif name in ["f2_median_valid", "median_valid"]:
        # Validity-aware median filter: 3x3 median on valid pixels only
        med = cv2.medianBlur(disp_clean, 3)
        return np.where(disp_clean > 0, med, 0.0)

    elif name in ["f3_guided", "guided"]:
        # Guided filter using grayscale left image as structural guide
        gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY) if left_img.ndim == 3 else left_img
        gf = cv2.ximgproc.guidedFilter(guide=gray, src=disp_clean, radius=5, eps=1.0)
        return np.where(disp_clean > 0, gf, 0.0)

    elif name in ["f4_joint_bilateral", "joint_bilateral"]:
        # Joint bilateral filter with color guidance
        img_f = left_img.astype(np.float32) / 255.0
        jb = cv2.ximgproc.jointBilateralFilter(joint=img_f, src=disp_clean, d=5, sigmaColor=0.1, sigmaSpace=5)
        return np.where(disp_clean > 0, jb, 0.0)

    elif name in ["f5_wls", "wls"]:
        # Standard WLS disparity filter
        if matcher is None or disp_r_raw is None:
            raise ValueError("WLS filter requires both matcher and disp_r_raw")
        wls = cv2.ximgproc.createDisparityWLSFilter(matcher)
        wls.setLambda(8000.0)
        wls.setSigmaColor(1.5)
        d_l_16 = (disp_raw * 16.0).astype(np.int16)
        d_r_16 = (disp_r_raw * 16.0).astype(np.int16)
        f_disp = wls.filter(d_l_16, left_img, None, d_r_16).astype(np.float32) / 16.0
        f_disp[f_disp <= 0] = 0.0
        return f_disp

    elif name in ["f6_confidence_edge", "confidence_edge"]:
        # Confidence-aware edge preservation: WLS filtered with low-confidence masking
        if matcher is None or disp_r_raw is None:
            raise ValueError("Confidence-aware filter requires both matcher and disp_r_raw")
        wls = cv2.ximgproc.createDisparityWLSFilter(matcher)
        wls.setLambda(8000.0)
        wls.setSigmaColor(1.5)
        d_l_16 = (disp_raw * 16.0).astype(np.int16)
        d_r_16 = (disp_r_raw * 16.0).astype(np.int16)
        f_disp = wls.filter(d_l_16, left_img, None, d_r_16).astype(np.float32) / 16.0
        conf = wls.getConfidenceMap()
        f_disp[f_disp <= 0] = 0.0
        f_disp[conf < 0.2 * np.max(conf)] = 0.0
        return f_disp

    else:
        raise ValueError(f"Unknown filter name: {filter_name}")


def compute_metric_depth_errors(
    pred_depth: np.ndarray,
    ref_depth: np.ndarray,
    min_z: float = 0.4,
    max_z: float = 4.0
) -> Dict[str, float]:
    """Compute depth error statistics against registered reference depth map."""
    eval_ref = (ref_depth >= min_z) & (ref_depth <= max_z) & np.isfinite(ref_depth)
    n_ref = np.sum(eval_ref)
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
    n_joint = np.sum(joint)
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
    rmse = float(np.sqrt(np.mean(diff**2)))
    d125 = float(np.mean(np.maximum(pred_z / ref_z, ref_z / pred_z) < 1.25) * 100.0)

    return {
        "eval_pixels": int(n_joint),
        "ref_coverage_pct": cov_pct,
        "mae_m": mae,
        "median_m": med,
        "rmse_m": rmse,
        "delta_125_pct": d125
    }

