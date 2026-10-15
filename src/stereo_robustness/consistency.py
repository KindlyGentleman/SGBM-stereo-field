"""
Unsupervised stereo consistency and validation metrics.
Implements left-right consistency check, photometric warp residual,
disparity gradient smoothness, and edge alignment diagnostics.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, Optional

import cv2
import numpy as np


@dataclass
class UnsupervisedDiagnostics:
    total_pixels: int
    valid_disparity_pixels: int
    valid_disparity_pct: float
    lr_consistent_pixels: int
    lr_consistent_pct_of_valid: float
    lr_consistent_pct_of_total: float
    photometric_mae: float
    photometric_rmse: float
    disparity_gradient_mean: float
    disparity_gradient_median: float
    edge_alignment_score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def check_left_right_consistency(
    disp_left: np.ndarray,
    disp_right: np.ndarray,
    valid_mask_left: np.ndarray,
    valid_mask_right: np.ndarray,
    threshold: float = 1.0,
) -> Tuple[np.ndarray, int, float]:
    """
    Evaluate left-right round-trip consistency.
    For each pixel (x, y) in left image:
    x_r = x - d_L(x, y)
    d_r = d_R(x_r, y)
    Consistent if |d_L - d_r| <= threshold.
    """
    h, w = disp_left.shape
    consistent_mask = np.zeros((h, w), dtype=bool)

    y_indices, x_indices = np.indices((h, w))
    valid_left = valid_mask_left & (disp_left > 0) & np.isfinite(disp_left)

    x_r = np.round(x_indices - disp_left).astype(np.int32)
    # Check bounds in right image
    in_bounds = (x_r >= 0) & (x_r < w) & valid_left

    # Extract right disparity at mapped locations
    valid_mapped = in_bounds.copy()
    valid_mapped[in_bounds] &= valid_mask_right[y_indices[in_bounds], x_r[in_bounds]]

    d_mapped = np.zeros_like(disp_left)
    d_mapped[valid_mapped] = disp_right[y_indices[valid_mapped], x_r[valid_mapped]]

    diff = np.abs(disp_left - d_mapped)
    consistent_mask = valid_mapped & (diff <= threshold)

    n_valid = int(np.count_nonzero(valid_left))
    n_cons = int(np.count_nonzero(consistent_mask))
    pct_valid = float(n_cons / n_valid * 100.0) if n_valid > 0 else 0.0

    return consistent_mask, n_cons, pct_valid


def warp_right_to_left(
    right_img: np.ndarray,
    disp_left: np.ndarray,
    valid_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Backward-warp right image into left camera coordinate frame using left disparity.
    Returns warped image and valid warped mask.
    """
    h, w = disp_left.shape
    y_coords, x_coords = np.indices((h, w), dtype=np.float32)
    map_x = x_coords - disp_left
    map_y = y_coords

    warped = cv2.remap(right_img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    warp_valid = valid_mask & (map_x >= 0) & (map_x < w)
    return warped, warp_valid


def compute_photometric_residual(
    left_img: np.ndarray,
    warped_right_img: np.ndarray,
    eval_mask: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute photometric warp residual between left image and warped right image.
    Returns (MAE, RMSE) on valid evaluable pixels.
    """
    l_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY) if left_img.ndim == 3 else left_img
    r_gray = cv2.cvtColor(warped_right_img, cv2.COLOR_BGR2GRAY) if warped_right_img.ndim == 3 else warped_right_img

    if np.count_nonzero(eval_mask) == 0:
        return 0.0, 0.0

    diff = np.abs(l_gray.astype(np.float32) - r_gray.astype(np.float32))[eval_mask]
    mae = float(np.mean(diff))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    return mae, rmse


def compute_unsupervised_diagnostics(
    left_img: np.ndarray,
    right_img: np.ndarray,
    disp_left: np.ndarray,
    disp_right: np.ndarray,
    valid_left: np.ndarray,
    valid_right: np.ndarray,
    lr_threshold: float = 1.5,
) -> UnsupervisedDiagnostics:
    """Compute complete suite of unsupervised quality metrics for field datasets."""
    total_px = disp_left.size
    n_valid = int(np.count_nonzero(valid_left))
    valid_pct = float(n_valid / total_px * 100.0)

    cons_mask, n_cons, cons_pct_valid = check_left_right_consistency(
        disp_left, disp_right, valid_left, valid_right, threshold=lr_threshold
    )
    cons_pct_total = float(n_cons / total_px * 100.0)

    warped_r, warp_valid = warp_right_to_left(right_img, disp_left, valid_left)
    photo_mae, photo_rmse = compute_photometric_residual(left_img, warped_r, warp_valid)

    # Disparity spatial gradient
    grad_x = cv2.Sobel(disp_left, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(disp_left, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    valid_grad = grad_mag[valid_left]
    if len(valid_grad) > 0:
        grad_mean = float(np.mean(valid_grad))
        grad_med = float(np.median(valid_grad))
    else:
        grad_mean = grad_med = 0.0

    # Edge alignment score
    l_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY) if left_img.ndim == 3 else left_img
    img_edges = cv2.Canny(l_gray, 50, 150) > 0
    # Disparity discontinuity edges: high gradient
    disp_edges = (grad_mag > 3.0) & valid_left
    if np.count_nonzero(disp_edges) > 0:
        # Distance transform from image edges
        dist_to_img_edges = cv2.distanceTransform((~img_edges).astype(np.uint8), cv2.DIST_L2, 3)
        mean_edge_dist = float(np.mean(dist_to_img_edges[disp_edges]))
        # Normalized score: 1.0 / (1.0 + mean_distance_pixels)
        edge_score = float(1.0 / (1.0 + mean_edge_dist))
    else:
        edge_score = 0.0

    return UnsupervisedDiagnostics(
        total_pixels=total_px,
        valid_disparity_pixels=n_valid,
        valid_disparity_pct=valid_pct,
        lr_consistent_pixels=n_cons,
        lr_consistent_pct_of_valid=cons_pct_valid,
        lr_consistent_pct_of_total=cons_pct_total,
        photometric_mae=photo_mae,
        photometric_rmse=photo_rmse,
        disparity_gradient_mean=grad_mean,
        disparity_gradient_median=grad_med,
        edge_alignment_score=edge_score,
    )
