"""
Photometric preprocessing methods for stereo matching under illumination shifts.
Implements grayscale baseline, CLAHE, histogram matching, global intensity normalization,
contrast normalization, and Census transform.
"""

from __future__ import annotations
from typing import Dict, Any, Tuple

import cv2
import numpy as np


def to_gray(img: np.ndarray) -> np.ndarray:
    """Ensure image is single-channel grayscale uint8."""
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        if img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.shape[2] == 1:
            return img[:, :, 0]
    raise ValueError(f"Unexpected image shape: {img.shape}")


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization."""
    gray = to_gray(img)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid_size)
    return clahe.apply(gray)


def apply_histogram_matching(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Match the cumulative intensity distribution of source image to reference image.
    Crucial for compensating asymmetric dual-camera exposure or gain differences.
    """
    src_gray = to_gray(source)
    ref_gray = to_gray(reference)

    src_hist, _ = np.histogram(src_gray.ravel(), 256, [0, 256])
    ref_hist, _ = np.histogram(ref_gray.ravel(), 256, [0, 256])

    src_cdf = src_hist.cumsum() / src_hist.sum()
    ref_cdf = ref_hist.cumsum() / ref_hist.sum()

    lut = np.zeros(256, dtype=np.uint8)
    g = 0
    for f in range(256):
        while g < 256 and src_cdf[f] > ref_cdf[g]:
            g += 1
        lut[f] = min(g, 255)

    return lut[src_gray]


def apply_global_normalization(img: np.ndarray) -> np.ndarray:
    """Global min-max intensity stretching across full [0, 255] dynamic range."""
    gray = to_gray(img).astype(np.float32)
    i_min, i_max = float(gray.min()), float(gray.max())
    if i_max - i_min < 1e-5:
        return gray.astype(np.uint8)
    norm = ((gray - i_min) / (i_max - i_min)) * 255.0
    return np.clip(norm, 0, 255).astype(np.uint8)


def apply_contrast_normalization(img: np.ndarray, target_mean: float = 128.0, target_std: float = 50.0) -> np.ndarray:
    """Standardize image to target mean and standard deviation."""
    gray = to_gray(img).astype(np.float32)
    current_mean = float(np.mean(gray))
    current_std = float(np.std(gray))
    if current_std < 1e-5:
        return gray.astype(np.uint8)
    standardized = ((gray - current_mean) / current_std) * target_std + target_mean
    return np.clip(standardized, 0, 255).astype(np.uint8)


def census_transform(img: np.ndarray, window_size: int = 3) -> np.ndarray:
    """
    Compute Census transform for radiometric invariance.
    For a 3x3 window, encodes local intensity relationships into an 8-bit integer.
    Non-parametric transform invariant to monotonic radiometric changes.
    """
    gray = to_gray(img)
    h, w = gray.shape
    r = window_size // 2
    census = np.zeros((h, w), dtype=np.uint8)

    # Pad image to handle borders
    padded = np.pad(gray, r, mode="edge")
    center = padded[r:h + r, r:w + r]

    bit_pos = 0
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy == 0 and dx == 0:
                continue
            neighbor = padded[r + dy:h + r + dy, r + dx:w + r + dx]
            # Set bit if neighbor < center
            bit = (neighbor < center).astype(np.uint8)
            census |= (bit << bit_pos)
            bit_pos += 1
            if bit_pos >= 8:
                break
        if bit_pos >= 8:
            break

    return census


def preprocess_pair(
    left: np.ndarray,
    right: np.ndarray,
    method: str = "gray",
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply chosen preprocessing scheme to stereo pair.
    
    Supported methods:
    - 'gray': standard BGR to grayscale
    - 'clahe': apply CLAHE independently to both images
    - 'hist_match': match right image histogram to left image
    - 'global_norm': stretch dynamic range on both images
    - 'contrast_norm': normalize mean and variance
    - 'census': compute Census transform on both images
    """
    if method == "gray":
        return to_gray(left), to_gray(right)
    elif method == "clahe":
        clip = kwargs.get("clip_limit", 2.0)
        return apply_clahe(left, clip_limit=clip), apply_clahe(right, clip_limit=clip)
    elif method == "hist_match":
        # Left remains reference, right is matched to left distribution
        l_gray = to_gray(left)
        r_matched = apply_histogram_matching(right, left)
        return l_gray, r_matched
    elif method == "global_norm":
        return apply_global_normalization(left), apply_global_normalization(right)
    elif method == "contrast_norm":
        return apply_contrast_normalization(left), apply_contrast_normalization(right)
    elif method == "census":
        return census_transform(left), census_transform(right)
    else:
        raise ValueError(f"Unknown preprocessing method: {method}")
