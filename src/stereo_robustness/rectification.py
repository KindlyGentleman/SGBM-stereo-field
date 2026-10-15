"""
Stereo rectification pathways: calibrated, uncalibrated (Hartley), and provisional alignment.
Calculates rectification mappings, homographies, and post-rectification correspondence residuals.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional

import cv2
import numpy as np

from stereo_robustness.calibration import StereoCalibrationParams, compute_rectification_models, RectificationModels
from stereo_robustness.geometry_checks import detect_and_match_features, evaluate_epipolar_geometry, GeometryDiagnostics


@dataclass
class RectificationResult:
    method: str  # 'calibrated', 'uncalibrated', or 'provisional_shift'
    rect_left: np.ndarray
    rect_right: np.ndarray
    valid_mask: np.ndarray
    before_diag: GeometryDiagnostics
    after_diag: GeometryDiagnostics
    metadata: Dict[str, Any]


def rectify_calibrated(
    left_img: np.ndarray,
    right_img: np.ndarray,
    calib_params: StereoCalibrationParams,
    alpha: float = 0.0,
) -> RectificationResult:
    """
    Pathway 1: Calibrated rectification using full camera matrices, distortion vectors,
    and stereo rotation/translation.
    """
    h, w = left_img.shape[:2]
    # Evaluate pre-rectification geometry
    pts1, pts2, _ = detect_and_match_features(left_img, right_img, method="orb")
    before_diag, _, _ = evaluate_epipolar_geometry(pts1, pts2, (h, w))

    models = compute_rectification_models(calib_params, alpha=alpha, new_image_size=(w, h))
    rect_l, rect_r = models.rectify(left_img, right_img)

    # Valid mask based on remapped pixel borders
    test_ones = np.ones((h, w), dtype=np.uint8)
    warped_l = cv2.remap(test_ones, models.map1_left, models.map2_left, cv2.INTER_NEAREST)
    warped_r = cv2.remap(test_ones, models.map1_right, models.map2_right, cv2.INTER_NEAREST)
    valid_mask = (warped_l > 0) & (warped_r > 0)

    # Evaluate post-rectification geometry
    pts1_after, pts2_after, _ = detect_and_match_features(rect_l, rect_r, method="orb")
    after_diag, _, _ = evaluate_epipolar_geometry(pts1_after, pts2_after, (h, w))

    return RectificationResult(
        method="calibrated",
        rect_left=rect_l,
        rect_right=rect_r,
        valid_mask=valid_mask,
        before_diag=before_diag,
        after_diag=after_diag,
        metadata={
            "alpha": alpha,
            "baseline_mm": calib_params.baseline_mm,
            "Q": models.Q.tolist(),
        },
    )


def rectify_uncalibrated(
    left_img: np.ndarray,
    right_img: np.ndarray,
    ransac_thresh: float = 1.0,
) -> RectificationResult:
    """
    Pathway 2: Uncalibrated rectification (Hartley's algorithm) via RANSAC Fundamental matrix
    and perspective homographies H1 and H2.
    """
    h, w = left_img.shape[:2]
    pts1, pts2, _ = detect_and_match_features(left_img, right_img, method="orb", max_features=3000)
    before_diag, inlier_mask, F = evaluate_epipolar_geometry(pts1, pts2, (h, w), ransac_thresh=ransac_thresh)

    if F is None or np.count_nonzero(inlier_mask) < 15:
        raise ValueError("Insufficient feature matches for uncalibrated fundamental matrix estimation.")

    inlier_pts1 = pts1[inlier_mask]
    inlier_pts2 = pts2[inlier_mask]

    ret, H1, H2 = cv2.stereoRectifyUncalibrated(
        inlier_pts1, inlier_pts2, F, (w, h), threshold=ransac_thresh
    )
    if not ret:
        raise RuntimeError("cv2.stereoRectifyUncalibrated failed to converge.")

    rect_l = cv2.warpPerspective(left_img, H1, (w, h))
    rect_r = cv2.warpPerspective(right_img, H2, (w, h))

    test_ones = np.ones((h, w), dtype=np.uint8)
    warped_l = cv2.warpPerspective(test_ones, H1, (w, h), flags=cv2.INTER_NEAREST)
    warped_r = cv2.warpPerspective(test_ones, H2, (w, h), flags=cv2.INTER_NEAREST)
    valid_mask = (warped_l > 0) & (warped_r > 0)

    pts1_after, pts2_after, _ = detect_and_match_features(rect_l, rect_r, method="orb")
    after_diag, _, _ = evaluate_epipolar_geometry(pts1_after, pts2_after, (h, w))

    return RectificationResult(
        method="uncalibrated",
        rect_left=rect_l,
        rect_right=rect_r,
        valid_mask=valid_mask,
        before_diag=before_diag,
        after_diag=after_diag,
        metadata={
            "H1": H1.tolist(),
            "H2": H2.tolist(),
            "F": F.tolist(),
        },
    )


def rectify_provisional_shift(
    left_img: np.ndarray,
    right_img: np.ndarray,
) -> RectificationResult:
    """
    Pathway 3: Diagnostic vertical shift compensation.
    Evaluates median vertical displacement dy and translates the right image by -dy.
    Used purely as an experimental baseline to isolate pure translation from full rectification.
    """
    h, w = left_img.shape[:2]
    pts1, pts2, _ = detect_and_match_features(left_img, right_img, method="orb")
    before_diag, inlier_mask, _ = evaluate_epipolar_geometry(pts1, pts2, (h, w))

    # Median vertical displacement
    dy = before_diag.raw_vertical_disp_median
    # Check direction: y2 - y1 = dy -> shift right by -dy
    # Form translation matrix
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, -dy]], dtype=np.float32)
    rect_l = left_img.copy()
    rect_r = cv2.warpAffine(right_img, M, (w, h))

    valid_mask = np.ones((h, w), dtype=bool)
    if dy > 0:
        valid_mask[h - int(np.ceil(dy)):, :] = False
    elif dy < 0:
        valid_mask[:int(np.ceil(abs(dy))), :] = False

    pts1_after, pts2_after, _ = detect_and_match_features(rect_l, rect_r, method="orb")
    after_diag, _, _ = evaluate_epipolar_geometry(pts1_after, pts2_after, (h, w))

    return RectificationResult(
        method="provisional_shift",
        rect_left=rect_l,
        rect_right=rect_r,
        valid_mask=valid_mask,
        before_diag=before_diag,
        after_diag=after_diag,
        metadata={"dy_shift": dy},
    )
