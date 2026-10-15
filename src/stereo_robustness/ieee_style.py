"""
IEEE and SciencePlots styling configuration for publication figures.
Enforces IEEE column widths, vector typography, and accessible palettes.
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import scienceplots

# Target column dimensions in inches (IEEE standard)
IEEE_SINGLE_COL_WIDTH = 3.5
IEEE_DOUBLE_COL_WIDTH = 7.16
IEEE_GOLDEN_RATIO = (5.0 ** 0.5 - 1.0) / 2.0  # ~0.618


def setup_ieee_style() -> None:
    """Initialize matplotlib with SciencePlots IEEE styling without LaTeX dependency."""
    try:
        plt.style.use(["science", "ieee", "no-latex"])
    except Exception:
        plt.style.use("default")
        mpl.rcParams["font.family"] = "sans-serif"

    # Refined typography and formatting rules
    mpl.rcParams["font.size"] = 8
    mpl.rcParams["axes.labelsize"] = 8
    mpl.rcParams["axes.titlesize"] = 9
    mpl.rcParams["xtick.labelsize"] = 7
    mpl.rcParams["ytick.labelsize"] = 7
    mpl.rcParams["legend.fontsize"] = 7
    mpl.rcParams["figure.titlesize"] = 9
    mpl.rcParams["lines.linewidth"] = 1.0
    mpl.rcParams["lines.markersize"] = 3
    mpl.rcParams["grid.linewidth"] = 0.5
    mpl.rcParams["grid.alpha"] = 0.5
    mpl.rcParams["savefig.dpi"] = 300
    mpl.rcParams["savefig.bbox"] = "tight"


def create_figure(
    width: float = IEEE_SINGLE_COL_WIDTH,
    aspect_ratio: float = IEEE_GOLDEN_RATIO,
    nrows: int = 1,
    ncols: int = 1,
    **kwargs,
) -> Tuple[plt.Figure, Any]:
    """
    Create an IEEE publication figure with exact physical column width.
    Height is scaled according to aspect_ratio.
    """
    height = width * aspect_ratio
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), **kwargs)
    return fig, axes


def save_publication_figure(fig: plt.Figure, base_path: Path | str) -> None:
    """Save figure in PDF, SVG, and high-resolution PNG formats."""
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(base_path.with_suffix(".pdf"), dpi=300)
    fig.savefig(base_path.with_suffix(".svg"), dpi=300)
    fig.savefig(base_path.with_suffix(".png"), dpi=300)
    plt.close(fig)
