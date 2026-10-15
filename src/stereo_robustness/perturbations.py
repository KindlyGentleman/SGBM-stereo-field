"""
Photometric perturbations for stereo robustness benchmarking.
Implements asymmetric brightness scaling, gamma shifts, and Gaussian noise.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np


@dataclass
class PerturbationMetadata:
    perturbation_type: str
    parameter_value: float
    seed: int
    saturation_low_pct: float
    saturation_high_pct: float


def apply_brightness_scaling(
    image: np.ndarray,
    alpha: float,
) -> Tuple[np.ndarray, PerturbationMetadata]:
    """Scale image intensities by factor alpha and clip to [0, 255]."""
    img_float = image.astype(np.float32) * float(alpha)
    sat_low = float(np.mean(img_float <= 0.0) * 100.0)
    sat_high = float(np.mean(img_float >= 255.0) * 100.0)
    result = np.clip(img_float, 0.0, 255.0).astype(np.uint8)
    meta = PerturbationMetadata(
        perturbation_type="brightness",
        parameter_value=float(alpha),
        seed=0,
        saturation_low_pct=sat_low,
        saturation_high_pct=sat_high,
    )
    return result, meta


def apply_gamma_correction(
    image: np.ndarray,
    gamma: float,
) -> Tuple[np.ndarray, PerturbationMetadata]:
    """Apply non-linear power-law gamma transformation."""
    if gamma <= 0:
        raise ValueError(f"Gamma must be positive, got {gamma}")
    inv_gamma = 1.0 / float(gamma)
    # Build 256-entry lookup table for speed and numerical consistency
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255.0
        for i in np.arange(256)
    ]).clip(0, 255).astype(np.uint8)
    result = table[image]
    sat_low = float(np.mean(result == 0) * 100.0)
    sat_high = float(np.mean(result == 255) * 100.0)
    meta = PerturbationMetadata(
        perturbation_type="gamma",
        parameter_value=float(gamma),
        seed=0,
        saturation_low_pct=sat_low,
        saturation_high_pct=sat_high,
    )
    return result, meta


def apply_gaussian_noise(
    image: np.ndarray,
    sigma: float,
    seed: int = 42,
) -> Tuple[np.ndarray, PerturbationMetadata]:
    """Add zero-mean Gaussian noise with standard deviation sigma."""
    if sigma <= 0:
        meta = PerturbationMetadata(
            perturbation_type="gaussian_noise",
            parameter_value=0.0,
            seed=seed,
            saturation_low_pct=float(np.mean(image == 0) * 100.0),
            saturation_high_pct=float(np.mean(image == 255) * 100.0),
        )
        return image.copy(), meta

    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, float(sigma), size=image.shape)
    noisy = image.astype(np.float32) + noise
    sat_low = float(np.mean(noisy <= 0.0) * 100.0)
    sat_high = float(np.mean(noisy >= 255.0) * 100.0)
    result = np.clip(noisy, 0.0, 255.0).astype(np.uint8)
    meta = PerturbationMetadata(
        perturbation_type="gaussian_noise",
        parameter_value=float(sigma),
        seed=seed,
        saturation_low_pct=sat_low,
        saturation_high_pct=sat_high,
    )
    return result, meta


# Standard parameter grids specified by the experimental protocol
STANDARD_BRIGHTNESS_FACTORS = [0.6, 0.8, 1.0, 1.2, 1.4]
STANDARD_GAMMA_VALUES = [0.6, 0.8, 1.0, 1.25, 1.6]
STANDARD_NOISE_SIGMAS = [0.0, 5.0, 10.0, 20.0, 30.0]
NOISE_REPETITION_SEEDS = [42, 123, 999]
