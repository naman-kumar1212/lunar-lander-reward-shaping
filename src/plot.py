"""
src/plot.py
-----------
Phase 6 — Professional publication-grade figure generation for Lunar Lander
Reward Shaping (PRD §8, §9 & §12).

Figures generated (both 300 DPI PNG with width >= 1600px and vector SVG):
    1. training_curves.png / .svg           Hero training reward curves
    2. episode_length_curves.png / .svg    Episode length vs timesteps (hover evidence)
    3. eval_scores_bar.png / .svg          Final evaluation score comparison with seed dots
    4. eval_success_timeout_bar.png / .svg Final success vs timeout rates with seed dots
    5. summary_panel_2x2.png / .svg        Multi-panel (a-d) overview figure for README
    6. captions.md                         Accompanying scientific figure captions
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3.common.results_plotter import load_results

from style import (
    VARIANT_COLORS,
    VARIANT_LABELS,
    VARIANT_SHORT_NAMES,
    apply_style,
    clean_spines,
    save_figure,
)

VARIANTS = ["baseline", "time_penalty", "hack", "fix"]
SEEDS = [0, 1, 2]
COMMON_TIMESTEPS = np.linspace(10_000, 1_000_000, 500)


def smooth_series(values: np.ndarray, window: int = 40) -> np.ndarray:
    """Rolling window smoothing for noisy trajectory metrics."""
    if len(values) < window:
        return values
    series = pd.Series(values)
    return series.rolling(window=window, min_periods=1).mean().values


def load_training_data(
    monitor_dir: Path,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, np.ndarray]]]:
    """
    Extract unshaped rewards and episode lengths from all 192 monitor CSVs.
    Returns interpolated matrices aligned on COMMON_TIMESTEPS.
    """
    reward_data = {}
    length_data = {}

    for variant in VARIANTS:
        rew_seed_curves = []
        rew_raw_means = []
        len_seed_curves = []
        len_raw_means = []

        for seed in SEEDS:
            run_id = f"{variant}_s{seed}"
            run_dir = monitor_dir / run_id
            if not run_dir.exists():
                raise FileNotFoundError(f"Monitor folder missing: {run_dir}")

            df = load_results(str(run_dir))
            x = df["l"].cumsum().values
            y_rew = df["r"].values
            y_len = df["l"].values

            # Raw interpolation without smoothing
            y_rew_raw = np.interp(COMMON_TIMESTEPS, x, y_rew, left=float(y_rew[0]), right=float(y_rew[-1]))
            rew_raw_means.append(y_rew_raw)

            y_len_raw = np.interp(COMMON_TIMESTEPS, x, y_len, left=float(y_len[0]), right=float(y_len[-1]))
            len_raw_means.append(y_len_raw)

            # Smoothed interpolation
            y_rew_smooth = smooth_series(y_rew, window=40)
            y_rew_interp = np.interp(COMMON_TIMESTEPS, x, y_rew_smooth, left=float(y_rew_smooth[0]), right=float(y_rew_smooth[-1]))
            rew_seed_curves.append(y_rew_interp)

            y_len_smooth = smooth_series(y_len, window=40)
            y_len_interp = np.interp(COMMON_TIMESTEPS, x, y_len_smooth, left=float(y_len_smooth[0]), right=float(y_len_smooth[-1]))
            len_seed_curves.append(y_len_interp)

        # Rewards matrix: (3, 500)
        rew_matrix = np.array(rew_seed_curves)
        raw_rew_matrix = np.array(rew_raw_means)
        reward_data[variant] = {
            "mean": np.mean(rew_matrix, axis=0),
            "std": np.std(rew_matrix, axis=0),
            "raw_mean": np.mean(raw_rew_matrix, axis=0),
            "seeds": rew_matrix,
        }

        # Lengths matrix: (3, 500)
        len_matrix = np.array(len_seed_curves)
        raw_len_matrix = np.array(len_raw_means)
        length_data[variant] = {
            "mean": np.mean(len_matrix, axis=0),
            "std": np.std(len_matrix, axis=0),
            "raw_mean": np.mean(raw_len_matrix, axis=0),
            "seeds": len_matrix,
        }

    return reward_data, length_data


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Training Reward Curves
# ─────────────────────────────────────────────────────────────────────────────
def plot_training_curves(
    reward_data: Dict[str, Dict[str, np.ndarray]],
    plots_dir: Path,
) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(9.5, 5.8), dpi=300)

    # Benchmark solved line
    ax.axhline(
        200,
        color="#555555",
        linestyle="--",
        linewidth=1.4,
        label="Solved threshold (+200)",
        zorder=2,
    )
    ax.axhline(0, color="#dddddd", linestyle="-", linewidth=0.8, zorder=1)

    for variant in VARIANTS:
        data = reward_data[variant]
        color = VARIANT_COLORS[variant]
        label = VARIANT_LABELS[variant]

        # Faint raw unsmoothed mean
        ax.plot(COMMON_TIMESTEPS, data["raw_mean"], color=color, alpha=0.15, linewidth=1.0, zorder=2)

        # Bold smoothed mean
        ax.plot(COMMON_TIMESTEPS, data["mean"], color=color, linewidth=2.4, label=label, zorder=4)

        # Shaded ±1 std band across 3 seeds
        ax.fill_between(
            COMMON_TIMESTEPS,
            data["mean"] - data["std"],
            data["mean"] + data["std"],
            color=color,
            alpha=0.18,
            zorder=3,
        )

    ax.set_title("PPO Training: Unshaped Built-in Reward vs. Timesteps", pad=12)
    ax.set_xlabel("Environment Timesteps (16 parallel envs)")
    ax.set_ylabel("Unshaped Episode Reward")
    ax.set_xlim(0, 1_000_000)
    ax.set_ylim(-320, 310)

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: f"{int(v/1000):,}k"))
    clean_spines(ax)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=10)

    save_figure(fig, "training_curves", plots_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Episode Length Curves
# ─────────────────────────────────────────────────────────────────────────────
def plot_episode_length_curves(
    length_data: Dict[str, Dict[str, np.ndarray]],
    plots_dir: Path,
) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(9.5, 5.8), dpi=300)

    # Timeout limit line
    ax.axhline(
        1000,
        color="#8c1d40",
        linestyle="--",
        linewidth=1.5,
        label="Timeout cutoff (1,000 frames)",
        zorder=2,
    )

    for variant in VARIANTS:
        data = length_data[variant]
        color = VARIANT_COLORS[variant]
        label = VARIANT_LABELS[variant]

        # Faint raw mean
        ax.plot(COMMON_TIMESTEPS, data["raw_mean"], color=color, alpha=0.15, linewidth=1.0, zorder=2)

        # Bold smoothed mean
        ax.plot(COMMON_TIMESTEPS, data["mean"], color=color, linewidth=2.4, label=label, zorder=4)

        # Shaded ±1 std band
        ax.fill_between(
            COMMON_TIMESTEPS,
            data["mean"] - data["std"],
            data["mean"] + data["std"],
            color=color,
            alpha=0.18,
            zorder=3,
        )

    ax.set_title("Training Episode Length vs. Timesteps (Reward Hacking Evidence)", pad=12)
    ax.set_xlabel("Environment Timesteps")
    ax.set_ylabel("Episode Length (Frames / Steps)")
    ax.set_xlim(0, 1_000_000)
    ax.set_ylim(0, 1060)

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: f"{int(v/1000):,}k"))
    clean_spines(ax)
    ax.legend(loc="upper left", framealpha=0.95, fontsize=10)

    save_figure(fig, "episode_length_curves", plots_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Final Evaluation Score Bars with Seed Dots
# ─────────────────────────────────────────────────────────────────────────────
def plot_eval_scores_bar(
    eval_models_df: pd.DataFrame,
    eval_summary_df: pd.DataFrame,
    plots_dir: Path,
) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)

    x_indices = np.arange(len(VARIANTS))
    bar_width = 0.55

    # Solved benchmark line
    ax.axhline(200, color="#555555", linestyle="--", linewidth=1.4, label="Solved (+200)", zorder=2)
    ax.axhline(0, color="#cccccc", linestyle="-", linewidth=0.8, zorder=1)

    for idx, variant in enumerate(VARIANTS):
        v_summary = eval_summary_df[eval_summary_df["variant"] == variant].iloc[0]
        v_models = eval_models_df[eval_models_df["variant"] == variant]
        color = VARIANT_COLORS[variant]

        mean = v_summary["mean_score"]
        spread = v_summary["score_spread"]

        # Bar with slight edge
        ax.bar(
            idx,
            mean,
            width=bar_width,
            color=color,
            alpha=0.82,
            edgecolor=color,
            linewidth=1.5,
            yerr=spread,
            capsize=6,
            error_kw={"elinewidth": 1.5, "ecolor": "#333333", "capthick": 1.5},
            zorder=3,
        )

        # Individual seed dots (with slight jitter)
        jitter = np.array([-0.12, 0.0, 0.12])
        seed_scores = v_models["mean_score"].values
        ax.scatter(
            idx + jitter,
            seed_scores,
            color="#222222",
            edgecolor="#ffffff",
            linewidth=1.2,
            s=70,
            zorder=5,
        )

        # Value label above bar
        offset = 15 if mean >= 0 else -25
        ax.text(
            idx,
            mean + offset,
            f"{mean:.1f}\n±{spread:.1f}",
            ha="center",
            va="bottom" if mean >= 0 else "top",
            fontsize=10,
            fontweight="bold",
            color="#222222",
        )

    ax.set_title("Deterministic Evaluation: Mean Score Across 3 Seeds (PRD §8)", pad=14)
    ax.set_ylabel("Evaluation Score (Raw Environment, 20 eps)")
    ax.set_xticks(x_indices)
    ax.set_xticklabels([VARIANT_SHORT_NAMES[v] for v in VARIANTS], fontweight="bold")
    ax.set_ylim(-80, 320)

    clean_spines(ax)
    # Custom legend entry for seed dots
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color="#555555", linestyle="--", linewidth=1.4),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#222222", markersize=7),
    ]
    ax.legend(custom_lines, ["Solved (+200)", "Individual Seeds"], loc="lower right", framealpha=0.95)

    save_figure(fig, "eval_scores_bar", plots_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Timeout Rate & Success Rate Grouped Bars
# ─────────────────────────────────────────────────────────────────────────────
def plot_eval_success_timeout_bar(
    eval_models_df: pd.DataFrame,
    eval_summary_df: pd.DataFrame,
    plots_dir: Path,
) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(9.2, 5.5), dpi=300)

    x_indices = np.arange(len(VARIANTS))
    bar_width = 0.35

    success_color = "#0072B2"
    timeout_color = "#D55E00"

    success_means = []
    success_stds = []
    timeout_means = []
    timeout_stds = []

    for variant in VARIANTS:
        v_summary = eval_summary_df[eval_summary_df["variant"] == variant].iloc[0]
        v_models = eval_models_df[eval_models_df["variant"] == variant]
        success_means.append(v_summary["success_rate"] * 100)
        success_stds.append(v_models["success_rate"].std() * 100)
        timeout_means.append(v_summary["timeout_rate"] * 100)
        timeout_stds.append(v_models["timeout_rate"].std() * 100)

    # Success bars
    rects1 = ax.bar(
        x_indices - bar_width / 2,
        success_means,
        width=bar_width,
        label="Landing Success Rate (Score ≥ 200)",
        color=success_color,
        alpha=0.85,
        edgecolor=success_color,
        linewidth=1.2,
        yerr=success_stds,
        capsize=4,
        error_kw={"elinewidth": 1.2, "ecolor": "#333333"},
        zorder=3,
    )

    # Timeout bars
    rects2 = ax.bar(
        x_indices + bar_width / 2,
        timeout_means,
        width=bar_width,
        label="Timeout Rate (1,000 frames cutoff)",
        color=timeout_color,
        alpha=0.85,
        edgecolor=timeout_color,
        linewidth=1.2,
        yerr=timeout_stds,
        capsize=4,
        error_kw={"elinewidth": 1.2, "ecolor": "#333333"},
        zorder=3,
    )

    # Overlay seed dots
    for idx, variant in enumerate(VARIANTS):
        v_models = eval_models_df[eval_models_df["variant"] == variant]
        jitter = np.array([-0.06, 0.0, 0.06])
        ax.scatter(
            idx - bar_width / 2 + jitter,
            v_models["success_rate"].values * 100,
            color="#222222",
            edgecolor="#ffffff",
            linewidth=1.0,
            s=45,
            zorder=5,
        )
        ax.scatter(
            idx + bar_width / 2 + jitter,
            v_models["timeout_rate"].values * 100,
            color="#222222",
            edgecolor="#ffffff",
            linewidth=1.0,
            s=45,
            zorder=5,
        )

        # Percentage text
        s_val = success_means[idx]
        t_val = timeout_means[idx]
        if s_val > 5:
            ax.text(idx - bar_width / 2, s_val + 3, f"{s_val:.1f}%", ha="center", fontsize=9.5, fontweight="bold")
        if t_val > 5:
            ax.text(idx + bar_width / 2, t_val + 3, f"{t_val:.1f}%", ha="center", fontsize=9.5, fontweight="bold")

    ax.set_title("Landing Success vs. Timeout Rate per Variant", pad=14)
    ax.set_ylabel("Percentage (%)")
    ax.set_xticks(x_indices)
    ax.set_xticklabels([VARIANT_SHORT_NAMES[v] for v in VARIANTS], fontweight="bold")
    ax.set_ylim(0, 115)

    clean_spines(ax)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=10)

    save_figure(fig, "eval_success_timeout_bar", plots_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Multi-panel 2x2 Hero Summary Figure (a-d)
# ─────────────────────────────────────────────────────────────────────────────
def plot_summary_panel_2x2(
    reward_data: Dict[str, Dict[str, np.ndarray]],
    length_data: Dict[str, Dict[str, np.ndarray]],
    eval_models_df: pd.DataFrame,
    eval_summary_df: pd.DataFrame,
    plots_dir: Path,
) -> None:
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

    # ── Panel (a): Training Rewards ───────────────────────────────────────
    ax_a = axes[0, 0]
    ax_a.axhline(200, color="#555555", linestyle="--", linewidth=1.2, label="Solved (200)")
    ax_a.axhline(0, color="#dddddd", linestyle="-", linewidth=0.8)

    for variant in VARIANTS:
        data = reward_data[variant]
        color = VARIANT_COLORS[variant]
        ax_a.plot(COMMON_TIMESTEPS, data["mean"], color=color, linewidth=2.2, label=VARIANT_SHORT_NAMES[variant])
        ax_a.fill_between(COMMON_TIMESTEPS, data["mean"] - data["std"], data["mean"] + data["std"], color=color, alpha=0.18)

    ax_a.set_title("(a) Unshaped Training Reward Curves", loc="left", fontsize=12, fontweight="bold")
    ax_a.set_xlabel("Environment Timesteps")
    ax_a.set_ylabel("Unshaped Episode Reward")
    ax_a.set_xlim(0, 1_000_000)
    ax_a.set_ylim(-320, 310)
    ax_a.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: f"{int(v/1000):,}k"))
    clean_spines(ax_a)
    ax_a.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

    # ── Panel (b): Training Episode Length ────────────────────────────────
    ax_b = axes[0, 1]
    ax_b.axhline(1000, color="#8c1d40", linestyle="--", linewidth=1.2, label="Timeout (1000)")

    for variant in VARIANTS:
        data = length_data[variant]
        color = VARIANT_COLORS[variant]
        ax_b.plot(COMMON_TIMESTEPS, data["mean"], color=color, linewidth=2.2, label=VARIANT_SHORT_NAMES[variant])
        ax_b.fill_between(COMMON_TIMESTEPS, data["mean"] - data["std"], data["mean"] + data["std"], color=color, alpha=0.18)

    ax_b.set_title("(b) Training Episode Length (Hover Evidence)", loc="left", fontsize=12, fontweight="bold")
    ax_b.set_xlabel("Environment Timesteps")
    ax_b.set_ylabel("Episode Length (Frames)")
    ax_b.set_xlim(0, 1_000_000)
    ax_b.set_ylim(0, 1060)
    ax_b.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: f"{int(v/1000):,}k"))
    clean_spines(ax_b)
    ax_b.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    # ── Panel (c): Final Evaluation Score Bars ────────────────────────────
    ax_c = axes[1, 0]
    ax_c.axhline(200, color="#555555", linestyle="--", linewidth=1.2, label="Solved (200)")
    ax_c.axhline(0, color="#cccccc", linestyle="-", linewidth=0.8)

    x_idx = np.arange(len(VARIANTS))
    for idx, variant in enumerate(VARIANTS):
        v_sum = eval_summary_df[eval_summary_df["variant"] == variant].iloc[0]
        v_mods = eval_models_df[eval_models_df["variant"] == variant]
        color = VARIANT_COLORS[variant]

        ax_c.bar(
            idx,
            v_sum["mean_score"],
            width=0.55,
            color=color,
            alpha=0.82,
            edgecolor=color,
            yerr=v_sum["score_spread"],
            capsize=5,
            error_kw={"elinewidth": 1.2, "ecolor": "#333333"},
        )
        jitter = np.array([-0.1, 0.0, 0.1])
        ax_c.scatter(idx + jitter, v_mods["mean_score"].values, color="#222222", edgecolor="#ffffff", s=50, zorder=5)

    ax_c.set_title("(c) Deterministic Evaluation Score (20 eps)", loc="left", fontsize=12, fontweight="bold")
    ax_c.set_ylabel("Evaluation Score (Raw Env)")
    ax_c.set_xticks(x_idx)
    ax_c.set_xticklabels([VARIANT_SHORT_NAMES[v] for v in VARIANTS], fontweight="bold")
    ax_c.set_ylim(-80, 320)
    clean_spines(ax_c)

    # ── Panel (d): Success vs Timeout Rate ────────────────────────────────
    ax_d = axes[1, 1]
    bar_w = 0.35
    for idx, variant in enumerate(VARIANTS):
        v_sum = eval_summary_df[eval_summary_df["variant"] == variant].iloc[0]
        v_mods = eval_models_df[eval_models_df["variant"] == variant]

        s_pct = v_sum["success_rate"] * 100
        t_pct = v_sum["timeout_rate"] * 100

        ax_d.bar(idx - bar_w/2, s_pct, width=bar_w, color="#0072B2", alpha=0.85, edgecolor="#0072B2")
        ax_d.bar(idx + bar_w/2, t_pct, width=bar_w, color="#D55E00", alpha=0.85, edgecolor="#D55E00")

        jitter = np.array([-0.05, 0.0, 0.05])
        ax_d.scatter(idx - bar_w/2 + jitter, v_mods["success_rate"].values * 100, color="#222222", edgecolor="#ffffff", s=35, zorder=5)
        ax_d.scatter(idx + bar_w/2 + jitter, v_mods["timeout_rate"].values * 100, color="#222222", edgecolor="#ffffff", s=35, zorder=5)

    ax_d.set_title("(d) Landing Success vs. Timeout Rate (%)", loc="left", fontsize=12, fontweight="bold")
    ax_d.set_ylabel("Percentage (%)")
    ax_d.set_xticks(x_idx)
    ax_d.set_xticklabels([VARIANT_SHORT_NAMES[v] for v in VARIANTS], fontweight="bold")
    ax_d.set_ylim(0, 115)
    clean_spines(ax_d)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#0072B2", label="Success Rate (Score ≥ 200)"),
        Patch(facecolor="#D55E00", label="Timeout Rate (1,000 frames)"),
    ]
    ax_d.legend(handles=legend_elements, loc="upper right", fontsize=8.5)

    fig.suptitle("LunarLander-v3 Reward Shaping Study: Results Summary", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    save_figure(fig, "summary_panel_2x2", plots_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Captions Generator
# ─────────────────────────────────────────────────────────────────────────────
def write_captions_file(plots_dir: Path) -> None:
    content = """# Figure Captions & Scientific Context

### Figure 1: `training_curves` (PNG & SVG)
**Caption:** Unshaped built-in episode reward as a function of environment timesteps across 1,000,000 training steps for all four variants in LunarLander-v3. Solid lines represent the mean across 3 independent random seeds (seeds 0, 1, 2), smoothed with a moving average window of 40 episodes; shaded bands represent ±1 standard deviation across seeds. Faint underlying lines depict the raw unsmoothed cross-seed mean. The horizontal dashed line denotes the canonical environment solved threshold (+200 points). Notice that while Variants A, B, and D converge cleanly above the solved threshold, Variant C (Hack) achieves severely depressed unshaped environment scores as it sacrifices terminal touchdown points to prolong hovering.

### Figure 2: `episode_length_curves` (PNG & SVG)
**Caption:** Episode duration (frames/timesteps) during training across 1,000,000 steps. Solid lines depict the moving-average mean (window=40) across 3 seeds; shaded bands depict ±1 standard deviation across seeds. The red dashed line denotes the environment truncation cutoff (1,000 frames). As Variant C discovers that staying upright and slow aloft earns +1.0 point per step, its episode lengths escalate rapidly towards the 1,000-frame ceiling, reaching an average of ~984 frames. Variants A, B, and D remain compact at ~300–350 frames as they actively navigate to the landing pad.

### Figure 3: `eval_scores_bar` (PNG & SVG)
**Caption:** Deterministic evaluation scores across 20 fixed evaluation episodes per seed on the raw environment (wrapper off). Colored bars represent the mean score across the 3 seeds with error bars indicating ±1 standard deviation across seeds. Black dots depict individual model evaluations (seed 0, 1, and 2). Both Baseline (259.1 ± 9.3) and Fix (253.0 ± 26.9) exceed the +200 solved threshold across all seeds, whereas Hack (−30.6 to +126.8, mean 28.9) fails to solve the raw landing objective due to reward hacking.

### Figure 4: `eval_success_timeout_bar` (PNG & SVG)
**Caption:** Comparison of landing success rate (percentage of evaluation episodes with score ≥ 200, blue bars) versus timeout rate (percentage of episodes hitting the 1,000-frame limit, orange bars). Individual seed outcomes are overlaid as dots. Variant C suffers a 73.3% timeout rate (100% in seeds 0 and 2) and 0% success rate, whereas potential-based shaping in Variant D restores a 93.3% success rate with only a 3.3% timeout rate.

### Figure 5: `summary_panel_2x2` (PNG & SVG)
**Caption:** Multi-panel overview of the reward shaping study. (a) Unshaped training reward trajectory. (b) Episode length progression demonstrating reward hacking. (c) Deterministic evaluation scores with seed distribution. (d) Final success versus timeout rate comparison across all 4 variants.
"""
    captions_path = plots_dir / "captions.md"
    with open(captions_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[OK] Saved captions to: {captions_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate publication-grade figures")
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    monitor_dir = results_dir / "monitor"
    eval_dir = results_dir / "eval"
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  Generating Professional Scientific Figures (PNG & SVG)")
    print("=" * 65)

    eval_models_df = pd.read_csv(eval_dir / "eval_per_model.csv")
    eval_summary_df = pd.read_csv(eval_dir / "summary.csv")

    print("\n1. Loading monitor CSVs across all 12 runs ...")
    reward_data, length_data = load_training_data(monitor_dir)

    print("\n2. Generating Figure 1: Training Reward Curves ...")
    plot_training_curves(reward_data, plots_dir)

    print("\n3. Generating Figure 2: Episode Length Curves ...")
    plot_episode_length_curves(length_data, plots_dir)

    print("\n4. Generating Figure 3: Final Evaluation Score Bars ...")
    plot_eval_scores_bar(eval_models_df, eval_summary_df, plots_dir)

    print("\n5. Generating Figure 4: Success vs Timeout Rate Bars ...")
    plot_eval_success_timeout_bar(eval_models_df, eval_summary_df, plots_dir)

    print("\n6. Generating Figure 5: 2x2 Multi-Panel Hero Figure ...")
    plot_summary_panel_2x2(reward_data, length_data, eval_models_df, eval_summary_df, plots_dir)

    print("\n7. Writing Figure Captions Document ...")
    write_captions_file(plots_dir)

    print("\n=======================================================")
    print("All professional figures and captions successfully created!")
    print("=======================================================")


if __name__ == "__main__":
    main()
