"""
Camera calibration and rectification map generation.
Manages intrinsics, extrinsics, reprojection matrices, and 3D point cloud generation.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import cv2
import numpy as np


@dataclass
class StereoCalibrationParams:
    mtx_left: np.ndarray
    dist_left: np.ndarray
    mtx_right: np.ndarray
    dist_right: np.ndarray
    R: np.ndarray
    T: np.ndarray
    E: Optional[np.ndarray] = None
    F: Optional[np.ndarray] = None
    image_size: Tuple[int, int] = (1280, 720)  # (width, height)
    baseline_mm: float = 0.0

    @classmethod
    def from_npz(cls, npz_path: Path | str, image_size: Tuple[int, int] = (1280, 720)) -> StereoCalibrationParams:
        npz_path = Path(npz_path)
        data = np.load(str(npz_path))
        mtx_l = data["mtx_left"]
        dist_l = data["dist_left"]
        mtx_r = data["mtx_right"]
        dist_r = data["dist_right"]
        R = data["R"]
        T = data["T"]
        E = data["E"] if "E" in data else None
        F = data["F"] if "F" in data else None
        baseline = float(np.linalg.norm(T))
        return cls(
            mtx_left=mtx_l,
            dist_left=dist_l,
            mtx_right=mtx_r,
            dist_right=dist_r,
            R=R,
            T=T,
            E=E,
            F=F,
            image_size=image_size,
            baseline_mm=baseline,
        )


@dataclass
class RectificationModels:
    R1: np.ndarray
    R2: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    Q: np.ndarray
    roi1: Tuple[int, int, int, int]
    roi2: Tuple[int, int, int, int]
    map1_left: np.ndarray
    map2_left: np.ndarray
    map1_right: np.ndarray
    map2_right: np.ndarray
    image_size: Tuple[int, int]

    def rectify(self, left: np.ndarray, right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply undistortion and rectification maps to stereo pair."""
        rect_left = cv2.remap(left, self.map1_left, self.map2_left, cv2.INTER_LINEAR)
        rect_right = cv2.remap(right, self.map1_right, self.map2_right, cv2.INTER_LINEAR)
        return rect_left, rect_right


def compute_rectification_models(
    calib: StereoCalibrationParams,
    alpha: float = 0.0,
    new_image_size: Optional[Tuple[int, int]] = None,
) -> RectificationModels:
    """
    Compute rectification rotation matrices, projection matrices, Q matrix,
    and remap lookup tables via cv2.stereoRectify and cv2.initUndistortRectifyMap.
    """
    img_size = new_image_size if new_image_size is not None else calib.image_size
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        cameraMatrix1=calib.mtx_left,
        distCoeffs1=calib.dist_left,
        cameraMatrix2=calib.mtx_right,
        distCoeffs2=calib.dist_right,
        imageSize=img_size,
        R=calib.R,
        T=calib.T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=alpha,
        newImageSize=img_size,
    )

    map1_l, map2_l = cv2.initUndistortRectifyMap(
        calib.mtx_left, calib.dist_left, R1, P1, img_size, cv2.CV_32FC1
    )
    map1_r, map2_r = cv2.initUndistortRectifyMap(
        calib.mtx_right, calib.dist_right, R2, P2, img_size, cv2.CV_32FC1
    )

    return RectificationModels(
        R1=R1,
        R2=R2,
        P1=P1,
        P2=P2,
        Q=Q,
        roi1=roi1,
        roi2=roi2,
        map1_left=map1_l,
        map2_left=map2_l,
        map1_right=map1_r,
        map2_right=map2_r,
        image_size=img_size,
    )


def reproject_to_3d(
    disparity: np.ndarray,
    Q: np.ndarray,
    min_depth: float = 0.1,
    max_depth: float = 50.0,
    valid_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reproject disparity map into 3D camera coordinates using Q matrix.
    Returns 3D points (H, W, 3) and filtering mask.
    Depth clipping bounds prevent extreme projections from invalid or near-zero disparities.
    """
    points_3d = cv2.reprojectImageTo3D(disparity.astype(np.float32), Q)
    z = points_3d[:, :, 2]

    # Convert mm to meters if Q matrix baseline was in mm (approx > 10)
    # Check baseline in Q: Q[3, 2] is -1/baseline
    if abs(Q[3, 2]) > 0.001:
        # baseline is ~ 10-1000 mm, z is in mm -> convert to meters
        z_meters = z / 1000.0
        points_3d = points_3d / 1000.0
    else:
        z_meters = z

    mask = np.isfinite(points_3d).all(axis=2) & (z_meters >= min_depth) & (z_meters <= max_depth)
    if valid_mask is not None:
        mask = mask & valid_mask

    return points_3d, mask
