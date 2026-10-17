"""
src/plot.py
-----------
Phase 6 — Training curves plotting for Lunar Lander Reward Shaping (PRD §9 & §12).

Builds training curves from the 16 monitor CSVs per run using the
UNSHAPED built-in reward.
Computes per-variant mean and spread across the 3 seeds on a shared
timesteps axis (0 to 1,000,000).

Outputs:
    results/plots/training_curves.png       Combined overlay of all 4 variants
    results/plots/training_curves_grid.png  2x2 subplots with shared axes
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3.common.results_plotter import load_results, ts2xy

VARIANTS = ["baseline", "time_penalty", "hack", "fix"]
SEEDS = [0, 1, 2]

VARIANT_METADATA = {
    "baseline": {
        "label": "A: Baseline (unshaped)",
        "color": "#1f77b4",  # Blue
    },
    "time_penalty": {
        "label": "B: Time Penalty (-0.02/frame)",
        "color": "#ff7f0e",  # Orange
    },
    "hack": {
        "label": "C: Hack (+1.0 hover bonus)",
        "color": "#d62728",  # Red
    },
    "fix": {
        "label": "D: Fix (Potential-based + fee)",
        "color": "#2ca02c",  # Green
    },
}

COMMON_TIMESTEPS = np.linspace(10_000, 1_000_000, 500)


def smooth_series(values: np.ndarray, window: int = 50) -> np.ndarray:
    """Moving average smoothing for noisy episode rewards."""
    if len(values) < window:
        return values
    series = pd.Series(values)
    return series.rolling(window=window, min_periods=1).mean().values


def load_variant_curves(
    monitor_dir: Path,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load and interpolate unshaped training curves for all variants and seeds.
    """
    variant_data = {}

    for variant in VARIANTS:
        seed_curves = []
        raw_runs = []

        for seed in SEEDS:
            run_id = f"{variant}_s{seed}"
            run_dir = monitor_dir / run_id
            if not run_dir.exists():
                raise FileNotFoundError(f"Monitor directory not found: {run_dir}")

            df = load_results(str(run_dir))
            x, y = ts2xy(df, "timesteps")

            # Smooth individual episode rewards
            y_smooth = smooth_series(y, window=40)

            # Interpolate onto COMMON_TIMESTEPS grid using numpy
            y_interp = np.interp(
                COMMON_TIMESTEPS, x, y_smooth, left=float(y_smooth[0]), right=float(y_smooth[-1])
            )

            seed_curves.append(y_interp)
            raw_runs.append((x, y_smooth))

        seed_matrix = np.array(seed_curves)  # Shape: (3, 500)
        mean_curve = np.mean(seed_matrix, axis=0)
        std_curve = np.std(seed_matrix, axis=0)
        min_curve = np.min(seed_matrix, axis=0)
        max_curve = np.max(seed_matrix, axis=0)

        variant_data[variant] = {
            "mean": mean_curve,
            "std": std_curve,
            "min": min_curve,
            "max": max_curve,
            "seeds": seed_matrix,
            "raw_runs": raw_runs,
        }

    return variant_data


def plot_combined_curves(
    variant_data: Dict[str, Dict[str, np.ndarray]],
    save_path: Path,
) -> None:
    """
    Plot single-panel comparison overlay of all 4 variants with shared axes.
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Dashed line for solved threshold (+200 points, PRD §5)
    ax.axhline(
        200,
        color="#555555",
        linestyle="--",
        linewidth=1.5,
        label="Solved threshold (+200)",
        zorder=2,
    )
    ax.axhline(0, color="#cccccc", linestyle=":", linewidth=1, zorder=1)

    for variant in VARIANTS:
        data = variant_data[variant]
        meta = VARIANT_METADATA[variant]
        mean = data["mean"]
        std = data["std"]

        line = ax.plot(
            COMMON_TIMESTEPS,
            mean,
            label=meta["label"],
            color=meta["color"],
            linewidth=2.5,
            zorder=3,
        )

        ax.fill_between(
            COMMON_TIMESTEPS,
            mean - std,
            mean + std,
            color=meta["color"],
            alpha=0.18,
            zorder=2,
        )

    ax.set_title(
        "LunarLander-v3 PPO Training: Unshaped Reward vs. Timesteps\n(Mean ± 1 Std across 3 seeds per variant)",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Environment Timesteps", fontsize=11, labelpad=8)
    ax.set_ylabel("Unshaped Built-in Episode Reward", fontsize=11, labelpad=8)
    ax.set_xlim(0, 1_000_000)
    ax.set_ylim(-300, 310)

    # Format x-axis with comma separators or 'k' notation
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda val, pos: f"{int(val/1000):,}k")
    )

    ax.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=10)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved combined training curve to: {save_path}")


def plot_grid_curves(
    variant_data: Dict[str, Dict[str, np.ndarray]],
    save_path: Path,
) -> None:
    """
    Plot 2x2 grid of all 4 variants sharing identical X and Y axis limits.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300, sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for idx, variant in enumerate(VARIANTS):
        ax = axes_flat[idx]
        data = variant_data[variant]
        meta = VARIANT_METADATA[variant]

        ax.axhline(200, color="#555555", linestyle="--", linewidth=1.2, label="Solved (+200)")
        ax.axhline(0, color="#dddddd", linestyle=":", linewidth=1)

        # Plot individual seeds lightly
        for s_idx, seed_curve in enumerate(data["seeds"]):
            ax.plot(
                COMMON_TIMESTEPS,
                seed_curve,
                color=meta["color"],
                alpha=0.35,
                linewidth=1.2,
                linestyle="-.",
                label=f"Seed {s_idx}",
            )

        # Plot mean and shaded std
        ax.plot(
            COMMON_TIMESTEPS,
            data["mean"],
            color=meta["color"],
            linewidth=2.5,
            label="Mean",
        )
        ax.fill_between(
            COMMON_TIMESTEPS,
            data["mean"] - data["std"],
            data["mean"] + data["std"],
            color=meta["color"],
            alpha=0.2,
        )

        ax.set_title(meta["label"], fontsize=12, fontweight="bold", pad=8)
        ax.set_xlim(0, 1_000_000)
        ax.set_ylim(-300, 310)
        ax.grid(True, linestyle="--", alpha=0.45)
        ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

        if idx in [0, 2]:
            ax.set_ylabel("Unshaped Episode Reward", fontsize=10)
        if idx in [2, 3]:
            ax.set_xlabel("Environment Timesteps", fontsize=10)
            ax.xaxis.set_major_formatter(
                plt.FuncFormatter(lambda val, pos: f"{int(val/1000):,}k")
            )

    fig.suptitle(
        "LunarLander-v3 Training Curves by Variant (PRD §9)",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved 2x2 grid training curves to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate training curve plots")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Root directory of results (default: results)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    monitor_dir = results_dir / "monitor"
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("Loading monitor CSVs for all 12 runs ...")
    variant_data = load_variant_curves(monitor_dir)

    print("Generating training curve plots ...")
    plot_combined_curves(variant_data, plots_dir / "training_curves.png")
    plot_grid_curves(variant_data, plots_dir / "training_curves_grid.png")
    print("Done! All plots generated.")


if __name__ == "__main__":
    main()
