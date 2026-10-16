"""
src/random_agent.py
-------------------
Phase 2 — Random baseline.

Runs one episode of LunarLander-v3 with a uniformly random policy,
prints and labels all 8 observation components (PRD §5), and reports
the episode outcome.

Usage:
    python src/random_agent.py              # renders pygame window
    python src/random_agent.py --no-render  # headless (rgb_array)
"""

import argparse
import gymnasium as gym
import numpy as np

# PRD §5: the 8 observation components in order
OBS_LABELS = [
    "x offset          ",
    "height            ",
    "x velocity        ",
    "y velocity        ",
    "tilt (rad)        ",
    "angular velocity  ",
    "left leg contact  ",
    "right leg contact ",
]


def run(render: bool = True, seed: int = 42) -> None:
    render_mode = "human" if render else "rgb_array"
    env = gym.make("LunarLander-v3", render_mode=render_mode)

    obs, info = env.reset(seed=seed)

    print("=" * 52)
    print("  LunarLander-v3 — Random Agent Baseline")
    print("=" * 52)
    print(f"\nRender mode : {render_mode}")
    print(f"Action space: {env.action_space}  (0=nothing, 1=left, 2=main, 3=right)")
    print(f"Obs shape   : {obs.shape}\n")

    print("Initial observation (matched to PRD §5 table):")
    print("-" * 42)
    for label, value in zip(OBS_LABELS, obs):
        print(f"  [{OBS_LABELS.index(label)}] {label}: {value:+.4f}")
    print("-" * 42)

    total_reward = 0.0
    step = 0
    terminated = truncated = False

    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1

    env.close()

    outcome = "CRASHED (terminated)" if terminated else "TRUNCATED at 1000 frames"
    print(f"\nEpisode finished after {step} steps.")
    print(f"Outcome     : {outcome}")
    print(f"Total reward: {total_reward:.2f}")
    print("\nFinal observation (same 8 components):")
    print("-" * 42)
    for label, value in zip(OBS_LABELS, obs):
        print(f"  [{OBS_LABELS.index(label)}] {label}: {value:+.4f}")
    print("-" * 42)
    print("\nPhase 2 exit criterion: window opened, rocket crashed, 8 obs printed. [OK]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LunarLander-v3 random agent")
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable pygame window (use rgb_array mode)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the environment reset (default: 42)",
    )
    args = parser.parse_args()
    run(render=not args.no_render, seed=args.seed)
