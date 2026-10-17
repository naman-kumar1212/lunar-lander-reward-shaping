"""
src/style.py
------------
Shared styling module for publication-grade figures in the Lunar Lander
Reward Shaping project.

Defines:
- Colorblind-safe palette (Wong / Okabe-Ito)
- Standardized typography and spine formatting
- Dual PNG (300 DPI, >=1600 px width) and SVG export helper
"""

from pathlib import Path
from typing import Dict, Optional
import matplotlib.pyplot as plt

# Colorblind-safe palette (Wong / Okabe-Ito convention)
VARIANT_COLORS: Dict[str, str] = {
    "baseline": "#0072B2",      # Deep Blue
    "time_penalty": "#E69F00",  # Amber / Warm Orange
    "hack": "#D55E00",          # Vermillion / Vivid Red
    "fix": "#009E73",           # Bluish Green
}

VARIANT_LABELS: Dict[str, str] = {
    "baseline": "A: Baseline (unshaped)",
    "time_penalty": "B: Time Penalty (-0.02/step)",
    "hack": "C: Hack (+1.0 hover bonus)",
    "fix": "D: Fix (Potential-based + fee)",
}

VARIANT_SHORT_NAMES: Dict[str, str] = {
    "baseline": "Baseline",
    "time_penalty": "Time Penalty",
    "hack": "Hack",
    "fix": "Fix",
}


def apply_style() -> None:
    """Set global matplotlib style parameters for clean, professional plots."""
    plt.rcParams.update({
        "figure.facecolor": "#ffffff",
        "figure.edgecolor": "#ffffff",
        "axes.facecolor": "#ffffff",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0,
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.titlepad": 12,
        "axes.labelsize": 13,
        "axes.labelweight": "medium",
        "axes.labelpad": 8,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "legend.fontsize": 10.5,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.facecolor": "#ffffff",
        "legend.edgecolor": "#dcdcdc",
        "grid.color": "#ebebeb",
        "grid.linestyle": ":",
        "grid.linewidth": 0.8,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
        "lines.antialiased": True,
        "patch.antialiased": True,
    })


def clean_spines(ax: plt.Axes) -> None:
    """Remove top and right spines, leaving a clean L-frame."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.grid(True, linestyle=":", color="#ebebeb", alpha=0.9, zorder=0)


def save_figure(
    fig: plt.Figure,
    base_name: str,
    plots_dir: Path,
    dpi: int = 300,
    min_width_px: int = 1600,
) -> None:
    """
    Save figure as both a 300 DPI PNG (>=1600px width) and an SVG.
    """
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Ensure figure width in inches is sufficient for min_width_px at given dpi
    current_width_in, current_height_in = fig.get_size_inches()
    required_width_in = min_width_px / dpi
    if current_width_in < required_width_in:
        scale = required_width_in / current_width_in
        fig.set_size_inches(required_width_in, current_height_in * scale)

    png_path = plots_dir / f"{base_name}.png"
    svg_path = plots_dir / f"{base_name}.svg"

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"[OK] Saved: {png_path.name} ({png_path.stat().st_size / 1024:.1f} KB) & {svg_path.name}")
