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
