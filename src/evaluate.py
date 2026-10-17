"""
src/evaluate.py
---------------
Phase 6 — Evaluation protocol for Lunar Lander Reward Shaping (PRD §8).

Evaluates all 12 trained PPO models on the RAW environment (LunarLander-v3,
wrapper off) using deterministic policies across 20 fixed-seed episodes.

Outputs:
    results/eval/eval_per_model.csv   Detailed metrics per model (variant, seed)
    results/eval/summary.csv          Aggregated metrics per variant (mean & spread across 3 seeds)
    results/eval/eval_episodes.csv    Full per-episode logs for deep analysis
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

# PRD §6.2 & §6.4: 4 variants, 3 seeds
VARIANTS = ["baseline", "time_penalty", "hack", "fix"]
SEEDS = [0, 1, 2]
NUM_EPISODES = 20
BASE_EVAL_SEED = 1000  # Fixed seeds 1000..1019 for reproducibility


def evaluate_model(
    model_path: Path,
    num_episodes: int = NUM_EPISODES,
    base_seed: int = BASE_EVAL_SEED,
) -> Tuple[Dict, List[Dict]]:
    """
    Run evaluation episodes for a single model on raw LunarLander-v3.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = PPO.load(str(model_path))
    env = gym.make("LunarLander-v3")

    episode_records = []
    scores = []
    lengths = []
    successes = []
    timeouts = []

    for ep in range(num_episodes):
        eval_seed = base_seed + ep
        obs, _ = env.reset(seed=eval_seed)
        ep_reward = 0.0
        ep_len = 0
        terminated = truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            ep_reward += float(reward)
            ep_len += 1

        # LunarLander-v3 definition: score >= 200 counts as solved/success (PRD §5)
        # Timeout: truncated at 1,000 frames (PRD §5 & §8)
        is_success = bool(ep_reward >= 200.0)
        is_timeout = bool(truncated or ep_len >= 1000)

        scores.append(ep_reward)
        lengths.append(ep_len)
        successes.append(1.0 if is_success else 0.0)
        timeouts.append(1.0 if is_timeout else 0.0)

        episode_records.append({
            "episode": ep,
            "seed": eval_seed,
            "score": ep_reward,
            "length": ep_len,
            "success": is_success,
            "timeout": is_timeout,
        })

    env.close()

    metrics = {
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "success_rate": float(np.mean(successes)),
        "mean_length": float(np.mean(lengths)),
        "timeout_rate": float(np.mean(timeouts)),
    }

    return metrics, episode_records


def run_full_evaluation(
    results_dir: Path = Path("results"),
    num_episodes: int = NUM_EPISODES,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate all 12 models and write CSV outputs.
    """
    models_dir = results_dir / "models"
    eval_dir = results_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    per_model_rows = []
    all_episodes = []

    print("=" * 70)
    print("  Evaluating 12 Models on RAW LunarLander-v3 (wrapper off)")
    print(f"  {num_episodes} episodes per model, deterministic policy")
    print("=" * 70)

    for variant in VARIANTS:
        for seed in SEEDS:
            run_id = f"{variant}_s{seed}"
            model_path = models_dir / f"{run_id}.zip"
            print(f"Evaluating {run_id:<16} ... ", end="", flush=True)

            metrics, ep_records = evaluate_model(
                model_path, num_episodes=num_episodes, base_seed=BASE_EVAL_SEED
            )

            for rec in ep_records:
                rec["variant"] = variant
                rec["model_seed"] = seed
                all_episodes.append(rec)

            row = {
                "variant": variant,
                "seed": seed,
                "mean_score": metrics["mean_score"],
                "std_score": metrics["std_score"],
                "success_rate": metrics["success_rate"],
                "mean_length": metrics["mean_length"],
                "timeout_rate": metrics["timeout_rate"],
            }
            per_model_rows.append(row)
            print(
                f"Score: {metrics['mean_score']:>7.2f} ± {metrics['std_score']:>5.2f} | "
                f"Success: {metrics['success_rate']*100:>5.1f}% | "
                f"Len: {metrics['mean_length']:>5.1f} | "
                f"Timeout: {metrics['timeout_rate']*100:>5.1f}%"
            )

    df_models = pd.DataFrame(per_model_rows)
    df_episodes = pd.DataFrame(all_episodes)

    # Aggregate per variant across the 3 seeds
    summary_rows = []
    for variant in VARIANTS:
        v_df = df_models[df_models["variant"] == variant]
        summary_rows.append({
            "variant": variant,
            "mean_score": float(v_df["mean_score"].mean()),
            "score_spread": float(v_df["mean_score"].std()),
            "success_rate": float(v_df["success_rate"].mean()),
            "mean_length": float(v_df["mean_length"].mean()),
            "timeout_rate": float(v_df["timeout_rate"].mean()),
        })
    df_summary = pd.DataFrame(summary_rows)

    # Save CSV files
    df_models.to_csv(eval_dir / "eval_per_model.csv", index=False)
    df_summary.to_csv(eval_dir / "summary.csv", index=False)
    df_episodes.to_csv(eval_dir / "eval_episodes.csv", index=False)

    print("\n" + "=" * 70)
    print("  EVALUATION SUMMARY ACROSS 3 SEEDS (PRD §8 & §9)")
    print("=" * 70)
    print(df_summary.to_string(index=False))

    # Generate Markdown Table
    md_table = generate_markdown_table(df_summary)
    print("\nMarkdown Table for README:")
    print(md_table)

    return df_models, df_summary


def generate_markdown_table(df_summary: pd.DataFrame) -> str:
    """Generate the GitHub Markdown summary table."""
    headers = [
        "Variant",
        "Reward Description",
        "Mean Score (± spread)",
        "Success Rate",
        "Mean Length",
        "Timeout Rate",
    ]
    variant_desc = {
        "baseline": "Built-in reward only (unshaped)",
        "time_penalty": "Built-in - 0.02 / frame",
        "hack": "Built-in + 1.0 upright & slow bonus",
        "fix": "Potential-based + time penalty",
    }

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for _, row in df_summary.iterrows():
        v = row["variant"]
        desc = variant_desc.get(v, v)
        score_str = f"{row['mean_score']:.1f} ± {row['score_spread']:.1f}"
        succ_str = f"{row['success_rate']*100:.1f}%"
        len_str = f"{row['mean_length']:.1f}"
        timeout_str = f"{row['timeout_rate']*100:.1f}%"
        lines.append(
            f"| `{v}` | {desc} | {score_str} | {succ_str} | {len_str} | {timeout_str} |"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LunarLander-v3 models")
    parser.add_argument(
        "--episodes",
        type=int,
        default=NUM_EPISODES,
        help="Number of evaluation episodes per model (default: 20)",
    )
    args = parser.parse_args()
    run_full_evaluation(num_episodes=args.episodes)
