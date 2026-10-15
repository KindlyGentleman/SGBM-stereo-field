"""
OpenCV StereoBM and StereoSGBM wrapper module.
Handles matcher instantiation, disparity scaling, validity mask generation,
and execution timing.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, Optional

import cv2
import numpy as np


@dataclass
class StereoBMConfig:
    numDisparities: int = 64
    blockSize: int = 15
    preFilterCap: int = 31
    textureThreshold: int = 10
    uniquenessRatio: int = 15
    speckleWindowSize: int = 100
    speckleRange: int = 32
    disp12MaxDiff: int = 1
    minDisparity: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StereoSGBMConfig:
    minDisparity: int = 0
    numDisparities: int = 64
    blockSize: int = 5
    P1: int = 600
    P2: int = 2400
    disp12MaxDiff: int = 1
    preFilterCap: int = 63
    uniquenessRatio: int = 10
    speckleWindowSize: int = 100
    speckleRange: int = 32
    mode: int = cv2.STEREO_SGBM_MODE_SGBM_3WAY

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        mode_names = {
            cv2.STEREO_SGBM_MODE_SGBM: "SGBM",
            cv2.STEREO_SGBM_MODE_HH: "HH",
            cv2.STEREO_SGBM_MODE_SGBM_3WAY: "3WAY",
            cv2.STEREO_SGBM_MODE_HH4: "HH4",
        }
        d["mode_name"] = mode_names.get(self.mode, str(self.mode))
        return d


@dataclass
class DisparityResult:
    disparity: np.ndarray  # float32 disparity in pixels
    valid_mask: np.ndarray  # boolean mask of valid disparity predictions
    raw_disparity: np.ndarray  # raw int16 disparity
    runtime_sec: float
    algorithm: str
    config: Dict[str, Any]


def build_stereo_bm(cfg: StereoBMConfig) -> cv2.StereoBM:
    """Instantiate and configure OpenCV StereoBM matcher."""
    bm = cv2.StereoBM_create(
        numDisparities=int(cfg.numDisparities),
        blockSize=int(cfg.blockSize),
    )
    bm.setPreFilterCap(int(cfg.preFilterCap))
    bm.setTextureThreshold(int(cfg.textureThreshold))
    bm.setUniquenessRatio(int(cfg.uniquenessRatio))
    bm.setSpeckleWindowSize(int(cfg.speckleWindowSize))
    bm.setSpeckleRange(int(cfg.speckleRange))
    bm.setDisp12MaxDiff(int(cfg.disp12MaxDiff))
    bm.setMinDisparity(int(cfg.minDisparity))
    return bm


def build_stereo_sgbm(cfg: StereoSGBMConfig) -> cv2.StereoSGBM:
    """Instantiate and configure OpenCV StereoSGBM matcher."""
    # Ensure P2 > P1
    p1 = int(cfg.P1)
    p2 = max(int(cfg.P2), p1 + 1)

    sgbm = cv2.StereoSGBM_create(
        minDisparity=int(cfg.minDisparity),
        numDisparities=int(cfg.numDisparities),
        blockSize=int(cfg.blockSize),
        P1=p1,
        P2=p2,
        disp12MaxDiff=int(cfg.disp12MaxDiff),
        preFilterCap=int(cfg.preFilterCap),
        uniquenessRatio=int(cfg.uniquenessRatio),
        speckleWindowSize=int(cfg.speckleWindowSize),
        speckleRange=int(cfg.speckleRange),
        mode=int(cfg.mode),
    )
    return sgbm


def compute_disparity_bm(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    cfg: StereoBMConfig,
) -> DisparityResult:
    """Compute disparity map using StereoBM on grayscale images."""
    if left_gray.ndim != 2 or right_gray.ndim != 2:
        raise ValueError("StereoBM strictly requires single-channel 8-bit grayscale images.")

    matcher = build_stereo_bm(cfg)
    t0 = time.perf_counter()
    raw = matcher.compute(left_gray, right_gray)
    t1 = time.perf_counter()

    disp = raw.astype(np.float32) / 16.0
    # OpenCV convention: invalid pixels are <= (minDisparity - 1)
    invalid_cutoff = float(cfg.minDisparity)
    valid_mask = (disp > invalid_cutoff) & np.isfinite(disp)
    disp[~valid_mask] = 0.0

    return DisparityResult(
        disparity=disp,
        valid_mask=valid_mask,
        raw_disparity=raw,
        runtime_sec=t1 - t0,
        algorithm="StereoBM",
        config=cfg.to_dict(),
    )


def compute_disparity_sgbm(
    left_img: np.ndarray,
    right_img: np.ndarray,
    cfg: StereoSGBMConfig,
) -> DisparityResult:
    """Compute disparity map using StereoSGBM on grayscale or color images."""
    matcher = build_stereo_sgbm(cfg)
    t0 = time.perf_counter()
    raw = matcher.compute(left_img, right_img)
    t1 = time.perf_counter()

    disp = raw.astype(np.float32) / 16.0
    invalid_cutoff = float(cfg.minDisparity)
    valid_mask = (disp > invalid_cutoff) & np.isfinite(disp)
    disp[~valid_mask] = 0.0

    return DisparityResult(
        disparity=disp,
        valid_mask=valid_mask,
        raw_disparity=raw,
        runtime_sec=t1 - t0,
        algorithm="StereoSGBM",
        config=cfg.to_dict(),
    )


def compute_right_disparity(
    left_img: np.ndarray,
    right_img: np.ndarray,
    cfg: StereoSGBMConfig,
) -> DisparityResult:
    """
    Compute right-view disparity map for left-right consistency checking.
    Uses cv2.ximgproc.createRightMatcher if available, otherwise swaps and flips horizontally.
    """
    matcher_left = build_stereo_sgbm(cfg)
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "createRightMatcher"):
        right_matcher = cv2.ximgproc.createRightMatcher(matcher_left)
        t0 = time.perf_counter()
        raw = right_matcher.compute(right_img, left_img)
        t1 = time.perf_counter()
        disp = raw.astype(np.float32) / 16.0
        # Right matcher disparity is negative in ximgproc convention; take abs
        disp = np.abs(disp)
        valid_mask = (disp > float(cfg.minDisparity)) & np.isfinite(disp)
        disp[~valid_mask] = 0.0
        return DisparityResult(
            disparity=disp,
            valid_mask=valid_mask,
            raw_disparity=raw,
            runtime_sec=t1 - t0,
            algorithm="StereoSGBM_Right",
            config=cfg.to_dict(),
        )
    else:
        # Fallback: flip horizontally, swap, compute left, flip back
        l_flip = cv2.flip(right_img, 1)
        r_flip = cv2.flip(left_img, 1)
        res = compute_disparity_sgbm(l_flip, r_flip, cfg)
        disp_right = cv2.flip(res.disparity, 1)
        valid_right = cv2.flip(res.valid_mask.astype(np.uint8), 1) > 0
        return DisparityResult(
            disparity=disp_right,
            valid_mask=valid_right,
            raw_disparity=cv2.flip(res.raw_disparity, 1),
            runtime_sec=res.runtime_sec,
            algorithm="StereoSGBM_Right_Flip",
            config=cfg.to_dict(),
        )
