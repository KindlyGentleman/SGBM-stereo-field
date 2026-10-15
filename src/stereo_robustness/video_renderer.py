"""
Scientific video production module for stereo evaluation demonstrations.
Renders high-definition (1920x1080) title cards, animated comparisons,
disparity overlays, and compiles H.264 MP4 videos using FFmpeg.
"""

from __future__ import annotations
import os
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np


class VideoCanvas:
    """HD (1920x1080) canvas for scientific figure animations."""

    def __init__(self, width: int = 1920, height: int = 1080, bg_color: Tuple[int, int, int] = (15, 20, 28)):
        self.width = width
        self.height = height
        self.bg_color = bg_color  # Deep slate navy background

    def create_blank(self) -> np.ndarray:
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        canvas[:] = self.bg_color
        return canvas

    def draw_header(self, canvas: np.ndarray, title: str, subtitle: str = "") -> None:
        # Subtle header bar
        cv2.rectangle(canvas, (0, 0), (self.width, 90), (24, 32, 45), -1)
        cv2.line(canvas, (0, 90), (self.width, 90), (50, 70, 95), 2)
        cv2.putText(canvas, title, (50, 55), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        if subtitle:
            cv2.putText(canvas, subtitle, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (170, 195, 220), 1, cv2.LINE_AA)

    def draw_footer(self, canvas: np.ndarray, caption: str) -> None:
        cv2.rectangle(canvas, (0, self.height - 65), (self.width, self.height), (20, 26, 36), -1)
        cv2.line(canvas, (0, self.height - 65), (self.width, self.height - 65), (50, 70, 95), 1)
        cv2.putText(canvas, caption, (50, self.height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 235, 245), 1, cv2.LINE_AA)

    def place_image(
        self,
        canvas: np.ndarray,
        img: np.ndarray,
        box: Tuple[int, int, int, int],  # (x, y, w, h)
        label: str = "",
        sublabel: str = "",
        border_color: Tuple[int, int, int] = (60, 80, 110),
    ) -> None:
        x, y, w, h = box
        resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        canvas[y:y + h, x:x + w] = resized
        cv2.rectangle(canvas, (x, y), (x + w, y + h), border_color, 2)
        if label:
            # Label banner
            cv2.rectangle(canvas, (x, y - 30), (x + w, y), (30, 40, 55), -1)
            cv2.putText(canvas, label, (x + 10, y - 8), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        if sublabel:
            cv2.putText(canvas, sublabel, (x + 10, y + h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 200, 220), 1, cv2.LINE_AA)


def assemble_video_with_ffmpeg(
    frame_dir: Path | str,
    output_mp4: Path | str,
    fps: int = 30,
) -> None:
    """Compile PNG frames into H.264 yuv420p MP4 video using FFmpeg."""
    frame_dir = Path(frame_dir)
    output_mp4 = Path(output_mp4)
    output_mp4.parent.mkdir(parents=True, exist_ok=True)

    input_pattern = str(frame_dir / "frame_%05d.png")

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-i", input_pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "fast",
        str(output_mp4),
    ]

    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"Rendered video successfully to: {output_mp4}")
