"""Classical stereo matching, synthetic scene generation, calibration, and metrics.

Provides lightweight, reproducible tools for educational stereo correspondence:
1. Synthetic scene generators (constant disparity and layered occlusion).
2. Controlled photometric perturbations.
3. Educational hand-written Sum of Absolute Differences (SAD) block matching.
4. OpenCV StereoBM and StereoSGBM wrappers.
5. Standard stereo evaluation metrics (bad@threshold, MAE, coverage).
6. Camera calibration and rectification utilities.
"""

from pathlib import Path
import time
from typing import Dict, Tuple, Optional, Any
import cv2
import numpy as np


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
    # Right border pixels in left image have no match if shifted beyond width
    valid_mask[:, width - disparity:] = False

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

    Returns:
        disp: Disparity map (float32). Invalid matches are set to 0.0.
        runtime_ms: Execution time in milliseconds.
    """
    gray_l = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY) if left_img.ndim == 3 else left_img
    gray_r = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY) if right_img.ndim == 3 else right_img

    if matcher is None:
        matcher = create_bm_matcher()

    t0 = time.perf_counter()
    raw = matcher.compute(gray_l, gray_r)
    runtime_ms = (time.perf_counter() - t0) * 1000.0

    disp = raw.astype(np.float32) / 16.0
    disp[disp <= 0.0] = 0.0
    return disp, runtime_ms


def compute_disparity_sgbm(
    left_img: np.ndarray,
    right_img: np.ndarray,
    matcher: Optional[cv2.StereoSGBM] = None,
    use_clahe: bool = False
) -> Tuple[np.ndarray, float]:
    """Execute StereoSGBM matching and return floating-point disparity in pixels.

    Args:
        left_img: Left image (grayscale or color).
        right_img: Right image (grayscale or color).
        matcher: Configured StereoSGBM instance.
        use_clahe: If True, applies CLAHE contrast equalization prior to matching.

    Returns:
        disp: Disparity map (float32). Invalid matches are set to 0.0.
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

    t0 = time.perf_counter()
    raw = matcher.compute(gray_l, gray_r)
    runtime_ms = (time.perf_counter() - t0) * 1000.0

    disp = raw.astype(np.float32) / 16.0
    disp[disp <= 0.0] = 0.0
    return disp, runtime_ms


def compute_bad_pixel_rate(
    disp_pred: np.ndarray,
    disp_gt: np.ndarray,
    valid_mask: np.ndarray,
    threshold: float = 2.0,
    completeness_aware: bool = False
) -> float:
    """Compute the percentage of pixels with disparity error exceeding threshold.

    Args:
        disp_pred: Predicted disparity map (float32).
        disp_gt: Reference ground-truth disparity map (float32).
        valid_mask: Boolean mask of reference pixels to evaluate.
        threshold: Error tolerance in pixels (default 2.0 px).
        completeness_aware: If True, missing predictions (pred <= 0) count as errors.

    Returns:
        bad_rate: Percentage in [0.0, 100.0].
    """
    eval_mask = valid_mask & (disp_gt > 0) & np.isfinite(disp_gt)
    n_gt = np.count_nonzero(eval_mask)
    if n_gt == 0:
        return 0.0

    pred_valid = (disp_pred > 0) & eval_mask
    error = np.abs(disp_pred - disp_gt)

    if completeness_aware:
        bad_error = pred_valid & (error > threshold)
        missing = eval_mask & (~(disp_pred > 0))
        bad_count = np.count_nonzero(bad_error) + np.count_nonzero(missing)
    else:
        # Standard Middlebury protocol on valid predictions
        n_eval = np.count_nonzero(pred_valid)
        if n_eval == 0:
            return 100.0
        bad_count = np.count_nonzero(pred_valid & (error > threshold))
        return float(bad_count / n_eval * 100.0)

    return float(bad_count / n_gt * 100.0)


def compute_depth_mae(
    disp_pred: np.ndarray,
    disp_gt: np.ndarray,
    valid_mask: np.ndarray
) -> float:
    """Mean absolute error in disparity over jointly valid evaluation pixels."""
    joint = valid_mask & (disp_gt > 0) & (disp_pred > 0) & np.isfinite(disp_pred)
    if np.count_nonzero(joint) == 0:
        return float("nan")
    return float(np.mean(np.abs(disp_pred[joint] - disp_gt[joint])))


def compute_depth_rmse(
    disp_pred: np.ndarray,
    disp_gt: np.ndarray,
    valid_mask: np.ndarray
) -> float:
    """Root mean square error in disparity over jointly valid evaluation pixels."""
    joint = valid_mask & (disp_gt > 0) & (disp_pred > 0) & np.isfinite(disp_pred)
    if np.count_nonzero(joint) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((disp_pred[joint] - disp_gt[joint]) ** 2)))


def compute_valid_coverage(
    disp: np.ndarray,
    valid_mask: Optional[np.ndarray] = None
) -> float:
    """Percentage of valid disparity predictions (d > 0)."""
    if valid_mask is not None:
        denom = np.count_nonzero(valid_mask)
        if denom == 0:
            return 0.0
        num = np.count_nonzero((disp > 0) & valid_mask)
        return float(num / denom * 100.0)
    return float(np.count_nonzero(disp > 0) / disp.size * 100.0)


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


def compute_lr_consistency(
    disp_l: np.ndarray,
    disp_r: np.ndarray,
    threshold: float = 1.5
) -> Tuple[float, np.ndarray]:
    """Check round-trip Left-Right consistency across matched scanline points."""
    h, w = disp_l.shape
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))

    x_matched = np.clip(np.round(x_coords - disp_l).astype(int), 0, w - 1)
    r_disp_matched = disp_r[y_coords, x_matched]

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
