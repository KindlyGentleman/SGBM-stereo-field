"""
Dataset loading and I/O utilities for stereo evaluation.
Supports Middlebury 2014 evaluation-v3 and Gunung outdoor stereo datasets.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np


def read_pfm(file_path: Path | str) -> Tuple[np.ndarray, float]:
    """
    Read a PFM (Portable Float Map) file.
    
    Standard Middlebury format stores single-channel disparity as 'Pf',
    bottom-to-top raster order, float32 precision.
    """
    file_path = Path(file_path)
    with file_path.open("rb") as f:
        header = f.readline().decode("latin-1").strip()
        if header == "PF":
            channels = 3
        elif header == "Pf":
            channels = 1
        else:
            raise ValueError(f"Unsupported PFM header: {header}")

        dims = f.readline().decode("latin-1").strip()
        while dims.startswith("#"):
            dims = f.readline().decode("latin-1").strip()
        width, height = map(int, dims.split())

        scale_line = f.readline().decode("latin-1").strip()
        while scale_line.startswith("#"):
            scale_line = f.readline().decode("latin-1").strip()
        scale = float(scale_line)
        endian = "<" if scale < 0 else ">"

        count = width * height * channels
        data = np.fromfile(f, dtype=endian + "f")
        if data.size != count:
            raise ValueError(f"PFM data size mismatch: expected {count}, got {data.size}")

        if channels == 1:
            data = data.reshape((height, width))
        else:
            data = data.reshape((height, width, channels))

        # PFM stores raster rows bottom-to-top; flip vertically to standard image coordinates
        data = np.flipud(data)
        return data.astype(np.float32), abs(scale)


def write_pfm(file_path: Path | str, image: np.ndarray, scale: float = -1.0) -> None:
    """
    Write a 2D float32 array into standard PFM format.
    Negative scale flag denotes little-endian byte ordering.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if image.dtype != np.float32:
        image = image.astype(np.float32)

    with file_path.open("wb") as f:
        if image.ndim == 2:
            f.write(b"Pf\n")
        elif image.ndim == 3 and image.shape[2] == 3:
            f.write(b"PF\n")
        else:
            raise ValueError(f"Unsupported image shape for PFM: {image.shape}")

        height, width = image.shape[:2]
        f.write(f"{width} {height}\n".encode("latin-1"))
        f.write(f"{scale}\n".encode("latin-1"))

        flipped = np.flipud(image)
        f.write(flipped.tobytes())


@dataclass
class MiddleburyCalibration:
    cam0: np.ndarray
    cam1: np.ndarray
    doffs: float
    baseline: float
    width: int
    height: int
    ndisp: int
    vmin: int
    vmax: int
    dyavg: float
    dymax: float
    raw_text: str


def parse_middlebury_calib(file_path: Path | str) -> MiddleburyCalibration:
    """Parse Middlebury calib.txt parameter file."""
    file_path = Path(file_path)
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    data: Dict[str, Any] = {}

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip()

            if val.startswith("[") and val.endswith("]"):
                rows = val[1:-1].split(";")
                matrix = []
                for row in rows:
                    matrix.append([float(x) for x in row.strip().split()])
                data[key] = np.array(matrix, dtype=np.float64)
            else:
                try:
                    if "." in val:
                        data[key] = float(val)
                    else:
                        data[key] = int(val)
                except ValueError:
                    data[key] = val

    return MiddleburyCalibration(
        cam0=data.get("cam0", np.eye(3)),
        cam1=data.get("cam1", np.eye(3)),
        doffs=float(data.get("doffs", 0.0)),
        baseline=float(data.get("baseline", 0.0)),
        width=int(data.get("width", 0)),
        height=int(data.get("height", 0)),
        ndisp=int(data.get("ndisp", 0)),
        vmin=int(data.get("vmin", 0)),
        vmax=int(data.get("vmax", 0)),
        dyavg=float(data.get("dyavg", 0.0)),
        dymax=float(data.get("dymax", 0.0)),
        raw_text=text,
    )


@dataclass
class StereoScene:
    scene_id: str
    left_image_path: Path
    right_image_path: Path
    disp_gt_path: Optional[Path] = None
    mask_nocc_path: Optional[Path] = None
    calib_path: Optional[Path] = None
    domain: str = "middlebury"  # 'middlebury' or 'gunung'
    metadata: Optional[Dict[str, Any]] = None

    def load_images(self, to_rgb: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Load left and right images as BGR (default) or RGB uint8 arrays."""
        left = cv2.imread(str(self.left_image_path))
        right = cv2.imread(str(self.right_image_path))
        if left is None:
            raise FileNotFoundError(f"Failed to read left image: {self.left_image_path}")
        if right is None:
            raise FileNotFoundError(f"Failed to read right image: {self.right_image_path}")
        if to_rgb:
            left = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
            right = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)
        return left, right

    def load_ground_truth(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Load ground truth disparity map and non-occlusion evaluation mask.
        Disparity values equal to inf or > 1e9 are treated as invalid.
        """
        disp_gt = None
        mask_valid = None
        if self.disp_gt_path and self.disp_gt_path.exists():
            disp_gt, _ = read_pfm(self.disp_gt_path)
            mask_valid = np.isfinite(disp_gt) & (disp_gt > 0) & (disp_gt < 1e9)
            disp_gt = np.where(mask_valid, disp_gt, 0.0).astype(np.float32)

        mask_nocc = None
        if self.mask_nocc_path and self.mask_nocc_path.exists():
            # Middlebury mask-nocc is typically 255 for non-occluded valid pixels
            mask_img = cv2.imread(str(self.mask_nocc_path), cv2.IMREAD_GRAYSCALE)
            if mask_img is not None:
                mask_nocc = mask_img > 0
                if mask_valid is not None:
                    mask_nocc = mask_nocc & mask_valid

        return disp_gt, mask_nocc if mask_nocc is not None else mask_valid

    def load_calib(self) -> Optional[MiddleburyCalibration]:
        if self.calib_path and self.calib_path.exists():
            return parse_middlebury_calib(self.calib_path)
        return None


def discover_middlebury_scenes(base_dir: Path | str, split: str = "trainingQ") -> Dict[str, StereoScene]:
    """
    Discover Middlebury evaluation scenes in the given folder.
    Searches for im0.png/im1.png and disp0GT.pfm.
    """
    base_path = Path(base_dir)
    target_dir = base_path / split if (base_path / split).exists() else base_path
    scenes: Dict[str, StereoScene] = {}

    for scene_folder in sorted(target_dir.iterdir()):
        if not scene_folder.is_dir():
            continue
        scene_name = scene_folder.name
        im0 = scene_folder / "im0.png"
        im1 = scene_folder / "im1.png"
        if not (im0.exists() and im1.exists()):
            continue

        disp0 = scene_folder / "disp0GT.pfm"
        if not disp0.exists():
            disp0 = scene_folder / "disp0.pfm"
        disp0_path = disp0 if disp0.exists() else None

        mask_nocc = scene_folder / "mask0nocc.png"
        mask_path = mask_nocc if mask_nocc.exists() else None

        calib = scene_folder / "calib.txt"
        calib_path = calib if calib.exists() else None

        scenes[scene_name] = StereoScene(
            scene_id=scene_name,
            left_image_path=im0,
            right_image_path=im1,
            disp_gt_path=disp0_path,
            mask_nocc_path=mask_path,
            calib_path=calib_path,
            domain="middlebury",
            metadata={"folder": str(scene_folder)},
        )
    return scenes


def discover_gunung_scenes(base_dir: Path | str) -> Dict[str, StereoScene]:
    """
    Discover Gunung outdoor scenes under the target directory.
    Target directory contains subfolders (e.g., puncakTinggi, puncakWarung, tanggaBiru, tanggaKuning).
    """
    base_path = Path(base_dir)
    scenes: Dict[str, StereoScene] = {}

    candidate_subdirs = [
        p for p in base_path.iterdir()
        if p.is_dir() and (p / "stereo").exists()
    ]
    # If base_path itself is the stereo folder
    if not candidate_subdirs and (base_path / "left_0000.png").exists():
        candidate_subdirs = [base_path]

    for sdir in sorted(candidate_subdirs):
        scene_name = sdir.name
        stereo_folder = sdir / "stereo" if (sdir / "stereo").exists() else sdir
        left = stereo_folder / "left_0000.png"
        right = stereo_folder / "right_0000.png"

        if left.exists() and right.exists():
            rs_depth = sdir / "hardware" / "realsense" / "depth_0000.npz"
            scenes[scene_name] = StereoScene(
                scene_id=scene_name,
                left_image_path=left,
                right_image_path=right,
                disp_gt_path=None,  # No ground truth disparity
                mask_nocc_path=None,
                calib_path=None,
                domain="gunung",
                metadata={
                    "realsense_depth_path": str(rs_depth) if rs_depth.exists() else None,
                    "scene_dir": str(sdir),
                },
            )
    return scenes
