"""
Epipolar geometry diagnostics and feature-based correspondence validation.
Evaluates vertical correspondence displacement, fundamental matrix residuals, and spatial match coverage.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, Optional, List

import cv2
import numpy as np


@dataclass
class GeometryDiagnostics:
    total_keypoints_left: int
    total_keypoints_right: int
    total_matches: int
    inlier_matches: int
    inlier_ratio: float
    raw_vertical_disp_median: float
    raw_vertical_disp_mean: float
    raw_vertical_disp_std: float
    raw_vertical_disp_p90: float
    raw_horizontal_disp_median: float
    epipolar_error_mean: float
    epipolar_error_median: float
    spatial_coverage_pct: float  # Percentage of spatial grid cells with at least 1 inlier

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def detect_and_match_features(
    left_img: np.ndarray,
    right_img: np.ndarray,
    method: str = "orb",
    max_features: int = 2000,
    ratio_thresh: float = 0.75,
) -> Tuple[np.ndarray, np.ndarray, List[cv2.DMatch]]:
    """
    Detect local features and establish tentative matches using Lowe's ratio test.
    Returns matched keypoint coordinates (pts1, pts2) as (N, 2) float32 arrays.
    """
    gray1 = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY) if left_img.ndim == 3 else left_img
    gray2 = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY) if right_img.ndim == 3 else right_img

    if method.lower() == "sift" and hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create(nfeatures=max_features)
        norm_type = cv2.NORM_L2
    else:
        detector = cv2.ORB_create(nfeatures=max_features)
        norm_type = cv2.NORM_HAMMING

    kp1, desc1 = detector.detectAndCompute(gray1, None)
    kp2, desc2 = detector.detectAndCompute(gray2, None)

    if desc1 is None or desc2 is None or len(kp1) < 8 or len(kp2) < 8:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), []

    matcher = cv2.BFMatcher(norm_type)
    raw_matches = matcher.knnMatch(desc1, desc2, k=2)

    good_matches = []
    pts1 = []
    pts2 = []

    for m_tuple in raw_matches:
        if len(m_tuple) == 2:
            m, n = m_tuple
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
                pts1.append(kp1[m.queryIdx].pt)
                pts2.append(kp2[m.trainIdx].pt)

    pts1_arr = np.array(pts1, dtype=np.float32) if pts1 else np.empty((0, 2), dtype=np.float32)
    pts2_arr = np.array(pts2, dtype=np.float32) if pts2 else np.empty((0, 2), dtype=np.float32)
    return pts1_arr, pts2_arr, good_matches


def evaluate_epipolar_geometry(
    pts1: np.ndarray,
    pts2: np.ndarray,
    image_shape: Tuple[int, int],
    ransac_thresh: float = 1.0,
    grid_size: Tuple[int, int] = (8, 8),
) -> Tuple[GeometryDiagnostics, np.ndarray, Optional[np.ndarray]]:
    """
    Compute RANSAC fundamental matrix, inlier statistics, vertical disparity residuals,
    and spatial grid coverage.
    Returns (GeometryDiagnostics, inlier_mask, Fundamental_matrix).
    """
    n_pts = len(pts1)
    if n_pts < 8:
        diag = GeometryDiagnostics(
            total_keypoints_left=0,
            total_keypoints_right=0,
            total_matches=n_pts,
            inlier_matches=0,
            inlier_ratio=0.0,
            raw_vertical_disp_median=0.0,
            raw_vertical_disp_mean=0.0,
            raw_vertical_disp_std=0.0,
            raw_vertical_disp_p90=0.0,
            raw_horizontal_disp_median=0.0,
            epipolar_error_mean=0.0,
            epipolar_error_median=0.0,
            spatial_coverage_pct=0.0,
        )
        return diag, np.zeros((n_pts,), dtype=bool), None

    F, inlier_mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, ransac_thresh, 0.99)
    if inlier_mask is None:
        mask = np.zeros((n_pts,), dtype=bool)
    else:
        mask = inlier_mask.ravel().astype(bool)

    n_inliers = int(np.count_nonzero(mask))
    inlier_ratio = float(n_inliers / n_pts) if n_pts > 0 else 0.0

    inlier_pts1 = pts1[mask]
    inlier_pts2 = pts2[mask]

    if n_inliers > 0:
        # Vertical displacement: |y2 - y1|
        vert_disp = np.abs(inlier_pts2[:, 1] - inlier_pts1[:, 1])
        v_median = float(np.median(vert_disp))
        v_mean = float(np.mean(vert_disp))
        v_std = float(np.std(vert_disp))
        v_p90 = float(np.percentile(vert_disp, 90))

        # Horizontal displacement: x1 - x2 (disparity in rectified geometry)
        horiz_disp = inlier_pts1[:, 0] - inlier_pts2[:, 0]
        h_median = float(np.median(horiz_disp))

        # Sampson epipolar error
        if F is not None and F.shape == (3, 3):
            pts1_h = np.hstack([inlier_pts1, np.ones((n_inliers, 1))])
            pts2_h = np.hstack([inlier_pts2, np.ones((n_inliers, 1))])
            Fx1 = pts1_h @ F.T
            Ftx2 = pts2_h @ F
            denom = Fx1[:, 0]**2 + Fx1[:, 1]**2 + Ftx2[:, 0]**2 + Ftx2[:, 1]**2
            num = np.sum(pts2_h * (pts1_h @ F.T), axis=1)**2
            sampson_err = np.sqrt(num / np.maximum(denom, 1e-12))
            epi_mean = float(np.mean(sampson_err))
            epi_med = float(np.median(sampson_err))
        else:
            epi_mean = 0.0
            epi_med = 0.0

        # Spatial grid coverage
        h, w = image_shape[:2]
        gw, gh = grid_size
        cell_w, cell_h = w / gw, h / gh
        grid_occupied = set()
        for pt in inlier_pts1:
            gx = min(int(pt[0] / cell_w), gw - 1)
            gy = min(int(pt[1] / cell_h), gh - 1)
            grid_occupied.add((gx, gy))
        spatial_cov = float(len(grid_occupied) / (gw * gh) * 100.0)
    else:
        v_median = v_mean = v_std = v_p90 = h_median = epi_mean = epi_med = spatial_cov = 0.0

    diag = GeometryDiagnostics(
        total_keypoints_left=n_pts,
        total_keypoints_right=n_pts,
        total_matches=n_pts,
        inlier_matches=n_inliers,
        inlier_ratio=inlier_ratio,
        raw_vertical_disp_median=v_median,
        raw_vertical_disp_mean=v_mean,
        raw_vertical_disp_std=v_std,
        raw_vertical_disp_p90=v_p90,
        raw_horizontal_disp_median=h_median,
        epipolar_error_mean=epi_mean,
        epipolar_error_median=epi_med,
        spatial_coverage_pct=spatial_cov,
    )
    return diag, mask, F
