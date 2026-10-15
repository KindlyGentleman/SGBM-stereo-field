"""
Scientific visualization utilities for stereo matching.
Generates consistent disparity maps, error heatmaps, and diagnostic overlays.
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np


def colorize_disparity(
    disparity: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colormap: int = cv2.COLORMAP_TURBO,
) -> np.ndarray:
    """
    Render disparity into RGB color visualization.
    Invalid pixels are mapped to pure black (0, 0, 0).
    """
    disp = disparity.astype(np.float32)
    if valid_mask is None:
        valid_mask = (disp > 0) & np.isfinite(disp)

    if np.count_nonzero(valid_mask) == 0:
        return np.zeros((*disp.shape, 3), dtype=np.uint8)

    val_min = float(vmin) if vmin is not None else float(disp[valid_mask].min())
    val_max = float(vmax) if vmax is not None else float(disp[valid_mask].max())
    if val_max - val_min < 1e-5:
        val_max = val_min + 1.0

    # Normalize to [0, 255]
    norm = np.clip((disp - val_min) / (val_max - val_min) * 255.0, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(norm, colormap)
    # OpenCV colormap is BGR; convert to RGB
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

    # Set invalid pixels to black
    color[~valid_mask] = [0, 0, 0]
    return color


def colorize_error_map(
    disp_pred: np.ndarray,
    disp_gt: np.ndarray,
    mask_eval: np.ndarray,
    pred_valid: np.ndarray,
    max_error: float = 5.0,
) -> np.ndarray:
    """
    Render spatial error map.
    Mutually valid pixels are colorized by absolute error [0, max_error].
    Missing predictions on valid GT are marked in red.
    Non-evaluated pixels are marked in dark gray.
    """
    h, w = disp_pred.shape
    error_rgb = np.full((h, w, 3), 40, dtype=np.uint8)  # Default dark gray for non-eval

    mutually_valid = mask_eval & pred_valid
    diff = np.abs(disp_pred[mutually_valid] - disp_gt[mutually_valid])
    norm_diff = np.clip(diff / max_error * 255.0, 0, 255).astype(np.uint8)

    color_err = cv2.applyColorMap(norm_diff, cv2.COLORMAP_JET)
    color_err = cv2.cvtColor(color_err, cv2.COLOR_BGR2RGB)
    if color_err.ndim == 3 and color_err.shape[1] == 1:
        color_err = color_err.squeeze(1)

    error_rgb[mutually_valid] = color_err


    # Missing prediction on valid GT: mark as bright magenta/red failure
    missing = mask_eval & (~pred_valid)
    error_rgb[missing] = [220, 20, 60]  # Crimson red

    return error_rgb


def save_image_rgb(file_path: Path | str, rgb_img: np.ndarray) -> None:
    """Save an RGB image array to disk."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(file_path), bgr)
