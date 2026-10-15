"""
Publication-ready vector scientific illustrations.
Generates schematics for stereo geometry, cross-domain transfer protocol,
rectification principles, photometric degradations, and reconstruction failure modes.
"""

from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from stereo_robustness.ieee_style import (
    setup_ieee_style,
    create_figure,
    save_publication_figure,
    IEEE_SINGLE_COL_WIDTH,
    IEEE_DOUBLE_COL_WIDTH,
)


def generate_illustration_1_stereo_geometry(out_path: Path | str) -> None:
    """Illustration 1: Canonical rectified stereo geometry and depth triangulation."""
    setup_ieee_style()
    fig, ax = create_figure(width=IEEE_SINGLE_COL_WIDTH, aspect_ratio=0.85)

    ax.set_xlim(-2.5, 4.5)
    ax.set_ylim(-0.5, 5.5)
    ax.axis("off")

    # Camera centers O_L and O_R
    o_l = np.array([0.0, 0.0])
    o_r = np.array([2.0, 0.0])
    baseline = 2.0
    focal = 1.2
    scene_pt = np.array([1.2, 4.5])

    # Plot baseline
    ax.plot([o_l[0], o_r[0]], [o_l[1], o_r[1]], "k-", lw=1.5)
    ax.plot(o_l[0], o_l[1], "ko", ms=5)
    ax.plot(o_r[0], o_r[1], "ko", ms=5)
    ax.text(o_l[0] - 0.2, o_l[1] - 0.3, "$O_L$", fontsize=8, ha="center")
    ax.text(o_r[0] + 0.2, o_r[1] - 0.3, "$O_R$", fontsize=8, ha="center")
    ax.text(1.0, -0.35, "Baseline $B$", fontsize=8, ha="center", color="navy")

    # Image planes
    ax.plot([-0.8, 0.8], [focal, focal], "b-", lw=1.8, label="Image Plane (Left)")
    ax.plot([1.2, 2.8], [focal, focal], "r-", lw=1.8, label="Image Plane (Right)")
    ax.text(0.0, focal + 0.15, "Left Plane", fontsize=7, ha="center", color="blue")
    ax.text(2.0, focal + 0.15, "Right Plane", fontsize=7, ha="center", color="red")

    # Optical axes
    ax.plot([0.0, 0.0], [0.0, 5.0], "b--", lw=0.8, alpha=0.6)
    ax.plot([2.0, 2.0], [0.0, 5.0], "r--", lw=0.8, alpha=0.6)

    # Rays to scene point P(X, Y, Z)
    ax.plot([o_l[0], scene_pt[0]], [o_l[1], scene_pt[1]], "k-", lw=1.0)
    ax.plot([o_r[0], scene_pt[0]], [o_r[1], scene_pt[1]], "k-", lw=1.0)
    ax.plot(scene_pt[0], scene_pt[1], "g*", ms=10)
    ax.text(scene_pt[0], scene_pt[1] + 0.25, "$P(X, Y, Z)$", fontsize=8, ha="center", color="darkgreen")

    # Projected image points x_L and x_R
    # By similar triangles:
    t_l = focal / scene_pt[1]
    p_l = o_l + t_l * (scene_pt - o_l)
    t_r = focal / scene_pt[1]
    p_r = o_r + t_r * (scene_pt - o_r)

    ax.plot(p_l[0], p_l[1], "bo", ms=4)
    ax.plot(p_r[0], p_r[1], "ro", ms=4)
    ax.text(p_l[0] - 0.15, p_l[1] - 0.25, "$x_L$", fontsize=7, color="blue")
    ax.text(p_r[0] + 0.15, p_r[1] - 0.25, "$x_R$", fontsize=7, color="red")

    # Annotation box for triangulation formula
    formula_text = (
        r"$d = x_L - x_R$" + "\n" +
        r"$Z = \frac{f \cdot B}{d}$"
    )
    ax.text(3.5, 3.0, formula_text, fontsize=8, bbox=dict(boxstyle="round,pad=0.5", facecolor="aliceblue", edgecolor="gray"))
    ax.annotate("Focal length $f$", xy=(0.0, focal/2), xytext=(-1.5, focal/2),
                arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=7)

    save_publication_figure(fig, out_path)


def generate_illustration_2_transfer_protocol(out_path: Path | str) -> None:
    """Illustration 2: Strict cross-domain evaluation and anti-leakage transfer pipeline."""
    setup_ieee_style()
    fig, ax = create_figure(width=IEEE_DOUBLE_COL_WIDTH, aspect_ratio=0.35)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.5)
    ax.axis("off")

    def draw_box(x, y, w, h, text, color, header=""):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", facecolor=color, edgecolor="black", lw=0.8)
        ax.add_patch(rect)
        if header:
            ax.text(x + w / 2, y + h - 0.25, header, fontsize=7, fontweight="bold", ha="center")
            ax.text(x + w / 2, y + h / 2 - 0.15, text, fontsize=6.5, ha="center", va="center")
        else:
            ax.text(x + w / 2, y + h / 2, text, fontsize=7, ha="center", va="center")

    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.2, color="black"))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.1, label, fontsize=6, ha="center")

    # Middlebury Benchmark branch
    draw_box(0.2, 1.8, 1.8, 1.3, "15 Scene Pairs\nGround-Truth Depth\nCalib Metadata", "#E8F4F8", "Middlebury 2014")
    draw_arrow(2.0, 2.45, 2.6, 2.45)

    draw_box(2.6, 1.8, 2.0, 1.3, "Dev Split (8 Scenes)\nGrid Search\nCLAHE, HistMatch", "#E2F0D9", "Phase C: Tuning")
    draw_arrow(4.6, 2.45, 5.2, 2.45)

    draw_box(5.2, 1.8, 1.8, 1.3, "Val Split (4 Scenes)\nLock Configurations\nG0, G1, G2", "#FFF2CC", "Validation & Freeze")

    # Held-out branch (zero leakage)
    draw_arrow(6.1, 1.8, 6.1, 0.9)
    draw_box(5.2, 0.1, 1.8, 0.8, "Held-Out Test\n(Jadeplant, Shelves, ArtL)\nStrict 1-Pass", "#FCE4D6", "Benchmark Test")

    # Gunung Real-World Transfer branch
    draw_arrow(7.0, 2.45, 7.6, 2.45, "Transfer")
    draw_box(7.6, 1.8, 2.1, 1.3, "Outdoor Volcanic Terrain\nRectification Check\nLR Consistency & Warp", "#E1D5E7", "Gunung Evaluation")

    save_publication_figure(fig, out_path)


def generate_illustration_3_rectification(out_path: Path | str) -> None:
    """Illustration 3: Unrectified vs Rectified epipolar geometry."""
    setup_ieee_style()
    fig, (ax1, ax2) = create_figure(width=IEEE_SINGLE_COL_WIDTH, aspect_ratio=0.7, nrows=1, ncols=2)

    # Panel 1: Unrectified
    ax1.set_title("Unrectified (Vertical Shift)", fontsize=8)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis("off")
    # Draw image frames
    ax1.add_patch(patches.Rectangle((0.5, 1.0), 3.8, 8.0, fill=None, edgecolor="blue", lw=1.2))
    ax1.add_patch(patches.Rectangle((5.5, 1.0), 3.8, 8.0, fill=None, edgecolor="red", lw=1.2))
    ax1.text(2.4, 9.2, "Left", color="blue", ha="center", fontsize=7)
    ax1.text(7.4, 9.2, "Right", color="red", ha="center", fontsize=7)

    # Slanted epipolar lines
    ax1.plot([0.5, 4.3], [7.0, 6.0], "g-", lw=1.0)
    ax1.plot([5.5, 9.3], [4.5, 3.2], "g-", lw=1.0)
    ax1.plot(2.4, 6.5, "bo", ms=4)
    ax1.plot(7.4, 3.85, "ro", ms=4)
    # Vertical misalignment indicator
    ax1.annotate("", xy=(7.4, 6.5), xytext=(7.4, 3.85), arrowprops=dict(arrowstyle="<->", color="purple", lw=1.0))
    ax1.text(7.7, 5.1, r"$\Delta y \approx 29$px", color="purple", fontsize=6.5)

    # Panel 2: Rectified
    ax2.set_title("Rectified (Standard Stereo)", fontsize=8)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis("off")
    ax2.add_patch(patches.Rectangle((0.5, 1.0), 3.8, 8.0, fill=None, edgecolor="blue", lw=1.2))
    ax2.add_patch(patches.Rectangle((5.5, 1.0), 3.8, 8.0, fill=None, edgecolor="red", lw=1.2))
    ax2.text(2.4, 9.2, "Rect Left", color="blue", ha="center", fontsize=7)
    ax2.text(7.4, 9.2, "Rect Right", color="red", ha="center", fontsize=7)

    # Strictly horizontal collinear epipolar lines
    for y_pos in [3.0, 5.5, 7.5]:
        ax2.plot([0.5, 9.3], [y_pos, y_pos], "g--", lw=0.8, alpha=0.7)
    ax2.plot(2.4, 5.5, "bo", ms=4)
    ax2.plot(6.8, 5.5, "ro", ms=4)
    ax2.annotate("", xy=(2.4, 5.5), xytext=(6.8, 5.5), arrowprops=dict(arrowstyle="<->", color="navy", lw=1.0))
    ax2.text(4.6, 5.8, r"Disparity $d$", color="navy", fontsize=6.5, ha="center")
    ax2.text(5.0, 1.5, r"$\Delta y \approx 0$ (Collinear)", color="darkgreen", fontsize=6.5, ha="center")

    save_publication_figure(fig, out_path)


def generate_illustration_4_photometric_degradations(out_path: Path | str) -> None:
    """Illustration 4: Transfer characteristics of brightness scaling, gamma, and noise."""
    setup_ieee_style()
    fig, (ax1, ax2) = create_figure(width=IEEE_SINGLE_COL_WIDTH, aspect_ratio=0.7, nrows=1, ncols=2)

    x = np.linspace(0, 255, 256)
    # Panel 1: Brightness & Gamma curves
    ax1.set_title("Photometric Response", fontsize=8)
    ax1.plot(x, x, "k--", lw=0.8, label=r"Identity ($\alpha=1$)")
    ax1.plot(x, np.clip(1.4 * x, 0, 255), "b-", lw=1.0, label=r"Bright ($\alpha=1.4$)")
    ax1.plot(x, np.clip(0.6 * x, 0, 255), "c-", lw=1.0, label=r"Dark ($\alpha=0.6$)")
    ax1.plot(x, 255.0 * ((x / 255.0) ** (1.0 / 1.6)), "r-", lw=1.0, label=r"$\gamma=1.6$")
    ax1.plot(x, 255.0 * ((x / 255.0) ** (1.0 / 0.6)), "m-", lw=1.0, label=r"$\gamma=0.6$")
    ax1.set_xlabel("Input Intensity", fontsize=7)
    ax1.set_ylabel("Transformed", fontsize=7)
    ax1.legend(loc="upper left", fontsize=5.5)

    # Panel 2: Sensor noise distribution
    ax2.set_title("Additive Noise Models", fontsize=8)
    noise_range = np.linspace(-40, 40, 200)
    for sigma, col in [(5, "green"), (10, "orange"), (20, "red"), (30, "purple")]:
        pdf = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (noise_range / sigma) ** 2)
        ax2.plot(noise_range, pdf, color=col, lw=1.0, label=rf"$\sigma={sigma}$")
    ax2.set_xlabel("Noise Error (DN)", fontsize=7)
    ax2.set_ylabel("Probability Density", fontsize=7)
    ax2.legend(loc="upper right", fontsize=5.5)

    save_publication_figure(fig, out_path)


def generate_illustration_5_failure_modes(out_path: Path | str) -> None:
    """Illustration 5: Schematic taxonomy of stereo matching failure modes."""
    setup_ieee_style()
    fig, ax = create_figure(width=IEEE_SINGLE_COL_WIDTH, aspect_ratio=0.8)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Classical Stereo Matching Failure Modes", fontsize=8.5, fontweight="bold")

    failures = [
        ("Low Texture / Saturation", "Sky, reflective rocks: zero spatial intensity gradient leads to ambiguous matching cost minima.", "#FFEBEE", (0.5, 7.3)),
        ("Repetitive Patterns", "Gravel, uniform brickwork: periodic local correlation minima produce false spatial disparities.", "#FFF8E1", (0.5, 5.0)),
        ("Thin Occluding Foliage", "Grass blades, wire structures: foreground and background mix within the correlation window.", "#E8F5E9", (0.5, 2.7)),
        ("Geometric Epipolar Error", "Uncalibrated vertical shift dy > 2px: corresponding points do not lie on horizontal scanlines.", "#EDE7F6", (0.5, 0.4)),
    ]

    for title, desc, col, (x, y) in failures:
        rect = patches.FancyBboxPatch((x, y), 9.0, 1.9, boxstyle="round,pad=0.1", facecolor=col, edgecolor="gray", lw=0.8)
        ax.add_patch(rect)
        ax.text(x + 0.3, y + 1.35, title, fontsize=7.5, fontweight="bold", color="black")
        ax.text(x + 0.3, y + 0.6, desc, fontsize=6.5, color="dimgray", wrap=True)

    save_publication_figure(fig, out_path)

