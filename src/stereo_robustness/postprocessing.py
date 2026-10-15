"""
Disparity postprocessing and edge-preserving filtering.
Implements WLS disparity filtering, speckle suppression, and median smoothing.
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, Optional

import cv2
import numpy as np


def apply_speckle_filter(
    disparity: np.ndarray,
    max_speckle_size: int = 100,
    max_diff: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Suppress small isolated disparity regions using cv2.filterSpeckles."""
    disp_int = (disparity * 16.0).astype(np.int16)
    # filterSpeckles requires int16 array scaled by 16
    cv2.filterSpeckles(disp_int, 0, max_speckle_size, int(max_diff * 16.0))
    filtered_disp = disp_int.astype(np.float32) / 16.0
    valid_mask = filtered_disp > 0
    return filtered_disp, valid_mask


def apply_wls_filter(
    disp_left_res,
    disp_right_res,
    left_img: np.ndarray,
    wls_lambda: float = 8000.0,
    wls_sigma: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Weighted Least Squares (WLS) fast disparity filter.
    Preserves object boundaries using left image guidance.
    """
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "createDisparityWLSFilterGeneric"):
        wls = cv2.ximgproc.createDisparityWLSFilterGeneric(False)
        wls.setLambda(float(wls_lambda))
        wls.setSigmaColor(float(wls_sigma))
        filtered = wls.filter(
            disp_left_res.raw_disparity,
            left_img,
            disparity_map_right=disp_right_res.raw_disparity,
        )
        disp_filtered = filtered.astype(np.float32) / 16.0
        valid_mask = (disp_filtered > 0) & np.isfinite(disp_filtered)
        disp_filtered[~valid_mask] = 0.0
        return disp_filtered, valid_mask
    else:
        # Fallback to bilateral-guided smoothing
        disp_smooth = cv2.bilateralFilter(disp_left_res.disparity, d=7, sigmaColor=25.0, sigmaSpace=25.0)
        valid_mask = disp_left_res.valid_mask & (disp_smooth > 0)
        return disp_smooth, valid_mask


def apply_median_filter(disparity: np.ndarray, ksize: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Apply median filter to suppress impulsive disparity noise."""
    filtered = cv2.medianBlur(disparity.astype(np.float32), ksize)
    valid_mask = filtered > 0
    return filtered, valid_mask
