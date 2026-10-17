"""Classical stereo matching, synthetic scene generation, calibration, and metrics.

Provides lightweight, reproducible tools for educational stereo correspondence:
1. Synthetic scene generators (constant disparity and layered occlusion).
2. Controlled photometric perturbations.
3. Educational hand-written Sum of Absolute Differences (SAD) block matching.
4. OpenCV StereoBM and StereoSGBM wrappers.
5. Standard stereo evaluation metrics (bad@threshold, MAE, coverage).
6. Camera calibration and rectification utilities.
"""

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Dict, Tuple, Optional, Any, List
import cv2
import numpy as np


@dataclass
class DisparityResult:
    """Structured container preserving raw and processed stereo disparity.

    Attributes:
        raw: Raw signed int16 output from OpenCV matcher (scaled by 16).
        disp: Floating-point disparity in pixels (raw / 16.0).
        valid_mask: Boolean mask of valid matcher predictions (raw > (min_disp - 1) * 16).
        runtime_ms: Matching execution time in milliseconds.
        min_disp: Minimum disparity search offset.
        num_disp: Number of disparity search levels.
    """
    raw: np.ndarray
    disp: np.ndarray
    valid_mask: np.ndarray
    runtime_ms: float
    min_disp: int = 0
    num_disp: int = 48



def generate_constant_disparity_scene(
    width: int = 320,
    height: int = 240,
    disparity: int = 16,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a planar synthetic stereo pair with constant horizontal disparity.

    Returns:
        left: Grayscale image (uint8).
        right: Grayscale image (uint8) shifted by disparity.
        disp_gt: Ground-truth disparity map (float32).
        valid_mask: Boolean mask indicating unclipped overlapping pixels.
    """
    rng = np.random.default_rng(seed)
    # Generate continuous smooth texture via filtered noise
    raw_noise = rng.uniform(30, 220, (height, width + disparity)).astype(np.float32)
    tex = cv2.GaussianBlur(raw_noise, (7, 7), 2.0)
    # Add subtle structural patterns
    y_idx, x_idx = np.mgrid[0:height, 0:width + disparity]
    pattern = 20.0 * np.sin(x_idx / 12.0) * np.cos(y_idx / 12.0)
    tex = np.clip(tex + pattern, 0, 255).astype(np.uint8)

    left = tex[:, :width].copy()
    right = tex[:, disparity:width + disparity].copy()

    disp_gt = np.full((height, width), float(disparity), dtype=np.float32)
    valid_mask = np.ones((height, width), dtype=bool)
    # Under x_R = x_L - disparity, left border pixels (x_L < disparity) have no match in right view
    if disparity > 0:
        valid_mask[:, :min(disparity, width)] = False

    return left, right, disp_gt, valid_mask


def generate_layered_synthetic_scene(
    width: int = 320,
    height: int = 240,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a 3-layer synthetic scene with occlusions and texture variations.

    The scene consists of:
    - Layer 0 (background): disparity d=8 px, smooth textured gradient.
    - Layer 1 (middleground): disparity d=18 px, lower region with a low-texture patch.
    - Layer 2 (foreground): disparity d=32 px, floating textured rectangular object.

    Reference visibility mask rigorously tracks geometric occlusion:
    points occluded by nearer foreground layers in the right view are marked invalid.

    Returns:
        left: Left view image (uint8, grayscale).
        right: Right view image (uint8, grayscale).
        disp_gt: Ground-truth disparity for the left view (float32).
        valid_mask: Non-occluded, overlapping pixel mask for the left view (bool).
    """
    rng = np.random.default_rng(seed)
    d_bg, d_mg, d_fg = 8, 18, 32

    # Background texture
    bg_noise = rng.uniform(40, 180, (height, width + d_fg)).astype(np.float32)
    bg_tex = cv2.GaussianBlur(bg_noise, (11, 11), 3.0)
    y_idx, x_idx = np.mgrid[0:height, 0:width + d_fg]
    bg_tex = np.clip(bg_tex + 0.15 * x_idx + 0.15 * y_idx, 0, 255).astype(np.uint8)

    # Middleground texture
    mg_noise = rng.uniform(60, 210, (height, width + d_fg)).astype(np.float32)
    mg_tex = cv2.GaussianBlur(mg_noise, (7, 7), 1.8).astype(np.uint8)
    # Add a flat, untextured zone to create a local aperture challenge
    mg_tex[140:190, 180:250] = 135

    # Foreground texture (sharp, high-contrast details)
    fg_noise = rng.uniform(70, 240, (height, width + d_fg)).astype(np.float32)
    fg_tex = cv2.GaussianBlur(fg_noise, (5, 5), 1.2).astype(np.uint8)
    fg_tex = cv2.rectangle(fg_tex, (15, 15), (width + d_fg - 15, height - 15), 230, 2)

    # Define left-camera layer boundaries
    mask_fg_l = np.zeros((height, width), dtype=bool)
    mask_fg_l[50:150, 70:150] = True

    mask_mg_l = np.zeros((height, width), dtype=bool)
    mask_mg_l[120:220, 120:300] = True
    mask_mg_l[mask_fg_l] = False

    mask_bg_l = ~(mask_fg_l | mask_mg_l)

    # Assemble Left image and ground-truth disparity
    left = np.zeros((height, width), dtype=np.uint8)
    disp_gt = np.zeros((height, width), dtype=np.float32)

    left[mask_bg_l] = bg_tex[:, :width][mask_bg_l]
    disp_gt[mask_bg_l] = float(d_bg)

    left[mask_mg_l] = mg_tex[:, :width][mask_mg_l]
    disp_gt[mask_mg_l] = float(d_mg)

    left[mask_fg_l] = fg_tex[:, :width][mask_fg_l]
    disp_gt[mask_fg_l] = float(d_fg)

    # Assemble Right image using coordinate projection: x_r = x_l - d
    right = np.zeros((height, width), dtype=np.uint8)

    # 1. Background projected into right image
    for xr in range(width):
        xl = xr + d_bg
        if 0 <= xl < width + d_fg:
            right[:, xr] = bg_tex[:, xl]

    # 2. Middleground projected into right image
    ys_m, xs_m = np.where(mask_mg_l)
    xr_m = xs_m - d_mg
    valid_m = (xr_m >= 0) & (xr_m < width)
    right[ys_m[valid_m], xr_m[valid_m]] = mg_tex[ys_m[valid_m], xs_m[valid_m]]

    # 3. Foreground projected into right image (overwrites background and middleground)
    ys_f, xs_f = np.where(mask_fg_l)
    xr_f = xs_f - d_fg
    valid_f = (xr_f >= 0) & (xr_f < width)
    right[ys_f[valid_f], xr_f[valid_f]] = fg_tex[ys_f[valid_f], xs_f[valid_f]]

    # Compute exact non-occluded mask in left view
    # A pixel at (xl, y) projects to xr = xl - disp_gt(xl, y).
    # It is occluded if a nearer layer occupies that same xr in the right view.
    x_grid = np.tile(np.arange(width), (height, 1))
    xr_grid = x_grid - disp_gt.astype(int)
    in_bounds = (xr_grid >= 0) & (xr_grid < width)

    def _hits_mask(xr_arr: np.ndarray, d_shift: int, target_mask: np.ndarray) -> np.ndarray:
        xl_cand = xr_arr + d_shift
        valid_cand = (xl_cand >= 0) & (xl_cand < width)
        hits = np.zeros_like(target_mask, dtype=bool)
        rows = np.repeat(np.arange(height)[:, None], width, axis=1)
        hits[valid_cand] = target_mask[rows[valid_cand], xl_cand[valid_cand]]
        return hits

    fg_occludes = _hits_mask(xr_grid, d_fg, mask_fg_l)
    mg_occludes = _hits_mask(xr_grid, d_mg, mask_mg_l)

    valid_mask = in_bounds.copy()
    # Middleground is occluded where foreground lands
    valid_mask[mask_mg_l & fg_occludes] = False
    # Background is occluded where foreground or middleground lands
    valid_mask[mask_bg_l & (fg_occludes | mg_occludes)] = False

    return left, right, disp_gt, valid_mask


def apply_perturbation(
    image: np.ndarray,
    perturbation: str,
    seed: int = 42
) -> np.ndarray:
    """Apply a controlled photometric transformation to an image.

    Supported perturbations:
    - 'none' / 'clean': Unchanged image.
    - 'brightness_up': Additive brightness (+30 intensity).
    - 'brightness_down': Subtractive brightness (-30 intensity).
    - 'contrast_low': Multiplicative contrast reduction (0.5x scaling).
    - 'gaussian_noise': Additive zero-mean Gaussian noise (sigma=15).
    - 'gamma_high': Non-linear response curve with gamma=1.6.
    - 'gamma_low': Non-linear response curve with gamma=0.6.
    """
    mode = perturbation.lower().strip()
    img_f = image.astype(np.float32)

    if mode in ["none", "clean"]:
        return image.copy()

    elif mode == "brightness_up":
        return np.clip(img_f + 30.0, 0.0, 255.0).astype(np.uint8)

    elif mode == "brightness_down":
        return np.clip(img_f - 30.0, 0.0, 255.0).astype(np.uint8)

    elif mode == "contrast_low":
        # Scale toward mid-gray
        return np.clip(128.0 + (img_f - 128.0) * 0.5, 0.0, 255.0).astype(np.uint8)

    elif mode == "gaussian_noise":
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, 15.0, image.shape).astype(np.float32)
        return np.clip(img_f + noise, 0.0, 255.0).astype(np.uint8)

    elif mode == "gamma_high":
        gamma = 1.6
        lut = np.array([((i / 255.0) ** gamma) * 255.0 for i in range(256)]).astype(np.uint8)
        return cv2.LUT(image, lut)

    elif mode == "gamma_low":
        gamma = 0.6
        lut = np.array([((i / 255.0) ** gamma) * 255.0 for i in range(256)]).astype(np.uint8)
        return cv2.LUT(image, lut)

    else:
        raise ValueError(f"Unknown perturbation: {perturbation}")


def compute_sad_cost_profile(
    left_img: np.ndarray,
    right_img: np.ndarray,
    x: int,
    y: int,
    patch_size: int = 9,
    max_disp: int = 48
) -> Tuple[np.ndarray, np.ndarray, int, float]:
    """Compute 1D Sum of Absolute Differences matching cost along a scanline.

    Args:
        left_img: Grayscale left image.
        right_img: Grayscale right image.
        x, y: Reference pixel coordinates in left image.
        patch_size: Square patch size (must be odd).
        max_disp: Maximum candidate disparity to evaluate.

    Returns:
        disparities: Array of candidate disparities [0 .. max_disp - 1].
        costs: SAD cost at each disparity (np.nan where candidate patch is out of bounds).
        best_disp: Disparity that minimizes SAD cost.
        min_cost: Value of minimal SAD cost.
    """
    rad = patch_size // 2
    h, w = left_img.shape[:2]

    if not (rad <= y < h - rad and rad <= x < w - rad):
        raise ValueError(f"Reference patch at ({x}, {y}) exceeds image boundaries.")

    p_left = left_img[y - rad:y + rad + 1, x - rad:x + rad + 1].astype(np.float32)

    disparities = np.arange(max_disp, dtype=int)
    costs = np.full(max_disp, np.nan, dtype=np.float32)

    for d in disparities:
        xr = x - d
        if rad <= xr < w - rad:
            p_right = right_img[y - rad:y + rad + 1, xr - rad:xr + rad + 1].astype(np.float32)
            costs[d] = float(np.sum(np.abs(p_left - p_right)))

    valid_mask = np.isfinite(costs)
    if not np.any(valid_mask):
        raise ValueError("No valid candidate disparities within right image boundaries.")

    best_idx = np.argmin(costs[valid_mask])
    best_disp = int(disparities[valid_mask][best_idx])
    min_cost = float(costs[valid_mask][best_idx])

    return disparities, costs, best_disp, min_cost


def create_bm_matcher(
    num_disparities: int = 48,
    block_size: int = 9
) -> cv2.StereoBM:
    """Create an OpenCV StereoBM block matching instance."""
    num_disp = int(np.ceil(num_disparities / 16.0) * 16)
    b_size = block_size if block_size % 2 == 1 else block_size + 1
    return cv2.StereoBM_create(numDisparities=num_disp, blockSize=b_size)


def create_sgbm_matcher(
    num_disparities: int = 48,
    block_size: int = 7,
    mode: int = cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    p1: Optional[int] = None,
    p2: Optional[int] = None
) -> cv2.StereoSGBM:
    """Create an OpenCV StereoSGBM semi-global matching instance."""
    num_disp = int(np.ceil(num_disparities / 16.0) * 16)
    b_size = block_size if block_size % 2 == 1 else block_size + 1
    channels = 1
    default_p1 = 8 * channels * (b_size ** 2) if p1 is None else p1
    default_p2 = 32 * channels * (b_size ** 2) if p2 is None else p2

    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=b_size,
        P1=default_p1,
        P2=default_p2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        mode=mode
    )


def compute_disparity_bm(
    left_img: np.ndarray,
    right_img: np.ndarray,
    matcher: Optional[cv2.StereoBM] = None
) -> Tuple[np.ndarray, float]:
    """Execute StereoBM matching and return floating-point disparity in pixels.

    Preserves the negative invalid sentinel (raw < 0) so that valid zero disparity
    and unmatched pixels remain distinguishable.

    Returns:
        disp: Disparity map (float32). Unmatched pixels have negative values (< 0).
        runtime_ms: Execution time in milliseconds.
    """
    gray_l = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY) if left_img.ndim == 3 else left_img
    gray_r = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY) if right_img.ndim == 3 else right_img

    if matcher is None:
        matcher = create_bm_matcher()

    min_disp = matcher.getMinDisparity()
    invalid_sentinel_raw = (min_disp - 1) * 16

    t0 = time.perf_counter()
    raw = matcher.compute(gray_l, gray_r)
    runtime_ms = (time.perf_counter() - t0) * 1000.0

    disp = raw.astype(np.float32) / 16.0
    # Normalize any matcher invalid sentinel to -1.0 for consistency
    disp[raw <= invalid_sentinel_raw] = -1.0
    return disp, runtime_ms


def compute_disparity_sgbm(
    left_img: np.ndarray,
    right_img: np.ndarray,
    matcher: Optional[cv2.StereoSGBM] = None,
    use_clahe: bool = False
) -> Tuple[np.ndarray, float]:
    """Execute StereoSGBM matching and return floating-point disparity in pixels.

    Preserves the negative invalid sentinel (disp < 0) so that valid zero disparity
    and unmatched pixels remain strictly distinguishable.

    Args:
        left_img: Left image (grayscale or color).
        right_img: Right image (grayscale or color).
        matcher: Configured StereoSGBM instance.
        use_clahe: If True, applies CLAHE contrast equalization prior to matching.

    Returns:
        disp: Disparity map (float32). Unmatched pixels have negative sentinel (< 0).
        runtime_ms: Matching-only execution time in milliseconds.
    """
    gray_l = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY) if left_img.ndim == 3 else left_img.copy()
    gray_r = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY) if right_img.ndim == 3 else right_img.copy()

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_l = clahe.apply(gray_l)
        gray_r = clahe.apply(gray_r)

    if matcher is None:
        matcher = create_sgbm_matcher()

    min_disp = matcher.getMinDisparity()
    invalid_sentinel_raw = (min_disp - 1) * 16

    t0 = time.perf_counter()
    raw = matcher.compute(gray_l, gray_r)
    runtime_ms = (time.perf_counter() - t0) * 1000.0

    disp = raw.astype(np.float32) / 16.0
    # Normalize any matcher invalid sentinel to -1.0 for consistency
    disp[raw <= invalid_sentinel_raw] = -1.0
    return disp, runtime_ms


def compute_disparity_sgbm_detailed(
    left_img: np.ndarray,
    right_img: np.ndarray,
    matcher: Optional[cv2.StereoSGBM] = None,
    use_clahe: bool = False
) -> DisparityResult:
    """Execute StereoSGBM returning full DisparityResult with raw array and validity mask."""
    gray_l = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY) if left_img.ndim == 3 else left_img.copy()
    gray_r = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY) if right_img.ndim == 3 else right_img.copy()

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_l = clahe.apply(gray_l)
        gray_r = clahe.apply(gray_r)

    if matcher is None:
        matcher = create_sgbm_matcher()

    min_disp = matcher.getMinDisparity()
    num_disp = matcher.getNumDisparities()
    invalid_sentinel_raw = (min_disp - 1) * 16

    t0 = time.perf_counter()
    raw = matcher.compute(gray_l, gray_r)
    runtime_ms = (time.perf_counter() - t0) * 1000.0

    valid_mask = raw > invalid_sentinel_raw
    disp = raw.astype(np.float32) / 16.0
    disp[~valid_mask] = -1.0

    return DisparityResult(
        raw=raw,
        disp=disp,
        valid_mask=valid_mask,
        runtime_ms=runtime_ms,
        min_disp=min_disp,
        num_disp=num_disp
    )


def compute_bad_pixel_rate(
    disp_pred: np.ndarray,
    disp_gt: np.ndarray,
    valid_mask: np.ndarray,
    threshold: float = 2.0,
    completeness_aware: bool = False,
    pred_valid_mask: Optional[np.ndarray] = None,
    min_gt_disp: float = 0.0
) -> float:
    """Compute the percentage of pixels with disparity error exceeding threshold.

    Args:
        disp_pred: Predicted disparity map (float32).
        disp_gt: Reference ground-truth disparity map (float32).
        valid_mask: Boolean mask of reference pixels to evaluate.
        threshold: Error tolerance in pixels (default 2.0 px).
        completeness_aware: If True, missing predictions count as errors over total N_ref.
            Formula: Bad_CA = (N_missing + N_bad_valid) / N_ref.
            If False (conditional): Bad_valid = N_bad_valid / N_valid.
        pred_valid_mask: Optional boolean mask specifying which predictions are valid.
        min_gt_disp: Minimum valid ground-truth disparity (default 0.0 px).

    Returns:
        bad_rate: Percentage in [0.0, 100.0], or float('nan') if denominator is zero.
    """
    eval_mask = valid_mask & np.isfinite(disp_gt) & (disp_gt >= min_gt_disp)
    n_gt = np.count_nonzero(eval_mask)
    if n_gt == 0:
        return float("nan")

    if pred_valid_mask is not None:
        pred_valid = pred_valid_mask & eval_mask & np.isfinite(disp_pred)
    else:
        # Values >= 0 are valid predictions (negative sentinel indicates unmatched)
        pred_valid = eval_mask & np.isfinite(disp_pred) & (disp_pred >= 0.0)

    n_valid = np.count_nonzero(pred_valid)

    if completeness_aware:
        # Completeness-aware: missing predictions within reference region count as errors
        bad_error = pred_valid & (np.abs(disp_pred - disp_gt) > threshold)
        missing = eval_mask & (~pred_valid)
        bad_count = np.count_nonzero(bad_error) + np.count_nonzero(missing)
        return float(bad_count / n_gt * 100.0)
    else:
        # Conditional accuracy on valid predictions only
        if n_valid == 0:
            return float("nan")
        bad_count = np.count_nonzero(pred_valid & (np.abs(disp_pred - disp_gt) > threshold))
        return float(bad_count / n_valid * 100.0)


def compute_disparity_mae(
    disp_pred: np.ndarray,
    disp_gt: np.ndarray,
    valid_mask: np.ndarray,
    pred_valid_mask: Optional[np.ndarray] = None,
    min_gt_disp: float = 0.0
) -> float:
    """Mean absolute error in disparity (pixels) over jointly valid evaluation pixels."""
    eval_mask = valid_mask & np.isfinite(disp_gt) & (disp_gt >= min_gt_disp)
    if pred_valid_mask is not None:
        joint = eval_mask & pred_valid_mask & np.isfinite(disp_pred)
    else:
        joint = eval_mask & np.isfinite(disp_pred) & (disp_pred >= 0.0)

    if np.count_nonzero(joint) == 0:
        return float("nan")
    return float(np.mean(np.abs(disp_pred[joint] - disp_gt[joint])))


def compute_disparity_rmse(
    disp_pred: np.ndarray,
    disp_gt: np.ndarray,
    valid_mask: np.ndarray,
    pred_valid_mask: Optional[np.ndarray] = None,
    min_gt_disp: float = 0.0
) -> float:
    """Root mean square error in disparity (pixels) over jointly valid evaluation pixels."""
    eval_mask = valid_mask & np.isfinite(disp_gt) & (disp_gt >= min_gt_disp)
    if pred_valid_mask is not None:
        joint = eval_mask & pred_valid_mask & np.isfinite(disp_pred)
    else:
        joint = eval_mask & np.isfinite(disp_pred) & (disp_pred >= 0.0)

    if np.count_nonzero(joint) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((disp_pred[joint] - disp_gt[joint]) ** 2)))


def compute_disparity_median_ae(
    disp_pred: np.ndarray,
    disp_gt: np.ndarray,
    valid_mask: np.ndarray,
    pred_valid_mask: Optional[np.ndarray] = None,
    min_gt_disp: float = 0.0
) -> float:
    """Median absolute error in disparity (pixels) over jointly valid evaluation pixels."""
    eval_mask = valid_mask & np.isfinite(disp_gt) & (disp_gt >= min_gt_disp)
    if pred_valid_mask is not None:
        joint = eval_mask & pred_valid_mask & np.isfinite(disp_pred)
    else:
        joint = eval_mask & np.isfinite(disp_pred) & (disp_pred >= 0.0)

    if np.count_nonzero(joint) == 0:
        return float("nan")
    return float(np.median(np.abs(disp_pred[joint] - disp_gt[joint])))


# Backward compatibility aliases (note: these compute disparity error in pixels)
compute_depth_mae = compute_disparity_mae
compute_depth_rmse = compute_disparity_rmse
compute_depth_median_ae = compute_disparity_median_ae


def compute_valid_coverage(
    disp: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    pred_valid_mask: Optional[np.ndarray] = None
) -> float:
    """Percentage of valid disparity predictions (d >= 0) within the reference mask."""
    if valid_mask is not None:
        denom = np.count_nonzero(valid_mask)
        if denom == 0:
            return 0.0
        if pred_valid_mask is not None:
            num = np.count_nonzero(pred_valid_mask & valid_mask)
        else:
            num = np.count_nonzero((disp >= 0.0) & np.isfinite(disp) & valid_mask)
        return float(num / denom * 100.0)

    if pred_valid_mask is not None:
        return float(np.count_nonzero(pred_valid_mask) / disp.size * 100.0)
    return float(np.count_nonzero((disp >= 0.0) & np.isfinite(disp)) / disp.size * 100.0)


def compute_full_disparity_metrics(
    disp_pred: np.ndarray,
    disp_gt: np.ndarray,
    valid_mask: np.ndarray,
    pred_valid_mask: Optional[np.ndarray] = None,
    thresholds: Tuple[float, ...] = (1.0, 2.0, 4.0),
    min_gt_disp: float = 0.0
) -> Dict[str, Any]:
    """Calculate complete accuracy and completeness disparity evaluation.

    Satisfies Part 4.3 requirements:
    A. Conditional bad-pixel percentage over valid predictions.
    B. Valid prediction coverage over fixed reference region.
    C. Completeness-aware bad-pixel percentage.
    D. Conditional disparity MAE.
    E. RMSE.
    F. Median absolute error.
    G. Invalid prediction percentage.
    H. Total evaluated reference pixels.
    """
    eval_mask = valid_mask & np.isfinite(disp_gt) & (disp_gt >= min_gt_disp)
    n_gt = int(np.count_nonzero(eval_mask))

    if pred_valid_mask is not None:
        pred_valid = pred_valid_mask & eval_mask & np.isfinite(disp_pred)
    else:
        pred_valid = eval_mask & np.isfinite(disp_pred) & (disp_pred >= 0.0)

    n_valid = int(np.count_nonzero(pred_valid))

    if n_gt == 0:
        return {
            "eval_pixels": 0,
            "valid_predictions": 0,
            "coverage_pct": 0.0,
            "invalid_pct": 100.0,
            "bad1_pct": float("nan"),
            "bad2_pct": float("nan"),
            "bad4_pct": float("nan"),
            "bad1_ca_pct": float("nan"),
            "bad2_ca_pct": float("nan"),
            "bad4_ca_pct": float("nan"),
            "disp_mae_px": float("nan"),
            "disp_rmse_px": float("nan"),
            "disp_median_ae_px": float("nan"),
        }

    cov_pct = float(n_valid / n_gt * 100.0)
    inv_pct = float(100.0 - cov_pct)

    res: Dict[str, Any] = {
        "eval_pixels": n_gt,
        "valid_predictions": n_valid,
        "coverage_pct": cov_pct,
        "invalid_pct": inv_pct,
    }

    # Error metrics over valid predictions
    if n_valid > 0:
        errs = np.abs(disp_pred[pred_valid] - disp_gt[pred_valid])
        res["disp_mae_px"] = float(np.mean(errs))
        res["disp_rmse_px"] = float(np.sqrt(np.mean(errs ** 2)))
        res["disp_median_ae_px"] = float(np.median(errs))

        for th in thresholds:
            th_str = f"bad{int(th)}_pct" if th.is_integer() else f"bad_{th:.1f}_pct"
            th_ca_str = f"bad{int(th)}_ca_pct" if th.is_integer() else f"bad_{th:.1f}_ca_pct"

            n_bad = np.count_nonzero(errs > th)
            res[th_str] = float(n_bad / n_valid * 100.0)

            n_missing = n_gt - n_valid
            res[th_ca_str] = float((n_bad + n_missing) / n_gt * 100.0)
    else:
        res["disp_mae_px"] = float("nan")
        res["disp_rmse_px"] = float("nan")
        res["disp_median_ae_px"] = float("nan")

        for th in thresholds:
            th_str = f"bad{int(th)}_pct" if th.is_integer() else f"bad_{th:.1f}_pct"
            th_ca_str = f"bad{int(th)}_ca_pct" if th.is_integer() else f"bad_{th:.1f}_ca_pct"
            res[th_str] = float("nan")
            res[th_ca_str] = 100.0

    return res


evaluate_disparity_map = compute_full_disparity_metrics


def disparity_to_depth(
    disp: np.ndarray,
    focal_length_px: float,
    baseline_m: float,
    min_disp: float = 0.5,
    doffs: float = 0.0
) -> np.ndarray:
    """Convert disparity in pixels to metric depth in meters via Z = fB / (d + doffs)."""
    depth = np.zeros_like(disp, dtype=np.float32)
    denom = disp + doffs
    valid = (disp >= min_disp) & (denom > 0) & np.isfinite(disp)
    depth[valid] = (focal_length_px * baseline_m) / denom[valid]
    return depth


def compute_depth_sensitivity(
    depth_m: float,
    focal_length_px: float,
    baseline_m: float,
    delta_disp_px: float = 1.0
) -> Dict[str, float]:
    """Calculate first-order Taylor and exact non-linear stereo depth sensitivity.

    For ideal rectified stereo, nominal depth is Z = (f * B) / d.
    First-order sensitivity approximation:
        |Delta Z| ~ Z^2 * |Delta d| / (f * B)

    Exact non-linear sensitivity (asymmetric):
        Delta Z_closer = Z - (f * B) / (d + Delta d) = Z^2 / (fB + Z * Delta d)
        Delta Z_farther = (f * B) / (d - Delta d) - Z = Z^2 / (fB - Z * Delta d)

    Args:
        depth_m: Nominal object distance in meters.
        focal_length_px: Rectified camera focal length in pixels.
        baseline_m: Physical camera baseline in meters.
        delta_disp_px: Perturbation magnitude in pixels (default 1.0 px).

    Returns:
        Dictionary containing theoretical sensitivity values in meters.
    """
    fb = focal_length_px * baseline_m
    d_nominal = fb / depth_m

    # First-order linear approximation
    delta_z_linear = (depth_m ** 2) * abs(delta_disp_px) / fb

    # Exact non-linear triangulation
    d_plus = d_nominal + abs(delta_disp_px)
    z_closer = fb / d_plus
    delta_z_closer = depth_m - z_closer

    d_minus = d_nominal - abs(delta_disp_px)
    if d_minus > 0:
        z_farther = fb / d_minus
        delta_z_farther = z_farther - depth_m
    else:
        delta_z_farther = float("inf")

    return {
        "fb_product": fb,
        "nominal_disparity_px": d_nominal,
        "delta_z_first_order_m": delta_z_linear,
        "delta_z_exact_closer_m": delta_z_closer,
        "delta_z_exact_farther_m": delta_z_farther
    }



def compute_lr_consistency(
    disp_l: np.ndarray,
    disp_r: np.ndarray,
    threshold: float = 1.5,
    min_disp: float = 0.0
) -> Tuple[float, np.ndarray]:
    """Check round-trip Left-Right consistency across matched scanline points.

    Convention:
        Left coordinate x_L matches x_R = x_L - disp_l[y, x_L].
        Consistency requires that x_R is within image bounds, disp_r at (y, x_R) is valid,
        and |disp_l(y, x_L) - |disp_r(y, x_R)|| <= threshold.
    """
    h, w = disp_l.shape[:2]
    valid_l = np.isfinite(disp_l) & (disp_l >= min_disp)

    y_coords, x_coords = np.mgrid[0:h, 0:w]
    x_r_cand = np.round(x_coords - disp_l).astype(int)

    # Coordinate must fall strictly within right image boundaries
    in_bounds = (x_r_cand >= 0) & (x_r_cand < w)

    consistent_mask = np.zeros((h, w), dtype=bool)
    eval_mask = valid_l & in_bounds

    if np.any(eval_mask):
        ys_eval = y_coords[eval_mask]
        xs_l = x_coords[eval_mask]
        xs_r = x_r_cand[eval_mask]

        d_l_val = disp_l[ys_eval, xs_l]
        d_r_val = disp_r[ys_eval, xs_r]

        d_r_mag = np.abs(d_r_val)
        valid_r = np.isfinite(d_r_val) & (d_r_mag >= min_disp) & (d_r_val != -1.0)

        diff = np.abs(d_l_val - d_r_mag)
        consistent_pts = valid_r & (diff <= threshold)
        consistent_mask[ys_eval[consistent_pts], xs_l[consistent_pts]] = True

    valid_pixels = int(np.count_nonzero(valid_l))
    if valid_pixels == 0:
        return 0.0, consistent_mask

    consistency_pct = float(np.count_nonzero(consistent_mask) / valid_pixels * 100.0)
    return consistency_pct, consistent_mask


def compute_photometric_warp_mae(
    left_img: np.ndarray,
    right_img: np.ndarray,
    disp_l: np.ndarray,
    min_disp: float = 0.0
) -> float:
    """Compute mean absolute difference between left image and warped right image.

    Only evaluates pixels where disparity is valid and the projected coordinate
    in the right image falls strictly within [0, width - 1].
    """
    h, w = disp_l.shape[:2]
    gray_l = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY).astype(np.float32) if left_img.ndim == 3 else left_img.astype(np.float32)
    gray_r = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY).astype(np.float32) if right_img.ndim == 3 else right_img.astype(np.float32)

    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
    x_warp = x_coords - disp_l

    # Valid mask: valid disparity and warped coordinate strictly inside right image
    valid_mask = (
        np.isfinite(disp_l)
        & (disp_l >= min_disp)
        & (x_warp >= 0.0)
        & (x_warp <= float(w - 1))
    )

    if np.count_nonzero(valid_mask) == 0:
        return float("nan")

    warped_r = cv2.remap(
        gray_r,
        x_warp,
        y_coords,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    mae = float(np.mean(np.abs(gray_l[valid_mask] - warped_r[valid_mask])))
    return mae


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
        rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB) if color_img.ndim == 3 else cv2.cvtColor(color_img, cv2.COLOR_GRAY2RGB)
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



def load_calibration(npz_path: str) -> Dict[str, np.ndarray]:
    """Load and validate camera calibration matrices from NPZ archive.

    Requires exact keys: 'mtx_left', 'dist_left', 'mtx_right', 'dist_right', 'R', 'T'.
    """
    path = Path(npz_path)
    if not path.exists():
        raise FileNotFoundError(f"Calibration file does not exist: {npz_path}")

    required_keys = ["mtx_left", "dist_left", "mtx_right", "dist_right", "R", "T"]
    with np.load(str(path), allow_pickle=False) as data:
        for k in required_keys:
            if k not in data.files:
                raise KeyError(f"Missing required calibration key: {k}")

        calib = {k: data[k] for k in required_keys}
        if "Q" in data.files:
            calib["Q"] = data["Q"]

    if calib["mtx_left"].shape != (3, 3) or calib["mtx_right"].shape != (3, 3):
        raise ValueError("Camera intrinsic matrices must have shape (3, 3)")
    if calib["R"].shape != (3, 3):
        raise ValueError("Rotation matrix R must have shape (3, 3)")
    if calib["T"].size != 3:
        raise ValueError("Translation vector T must have 3 elements")

    return calib


def build_rectification_maps(
    calib: Dict[str, np.ndarray],
    img_size: Tuple[int, int] = (1280, 720),
    alpha: float = 0.0
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """Compute epipolar rectification maps via cv2.stereoRectify."""
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
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
    """Remap dual images into collinear epipolar alignment."""
    rect_l = cv2.remap(left_img, map_left[0], map_left[1], cv2.INTER_LINEAR)
    rect_r = cv2.remap(right_img, map_right[0], map_right[1], cv2.INTER_LINEAR)
    return rect_l, rect_r


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
