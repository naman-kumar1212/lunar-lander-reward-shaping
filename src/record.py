"""
src/record.py
-------------
Phase 6 — Video recording and GIF generation for Lunar Lander Reward Shaping (PRD §9).

Produces:
    1. Landing clip: Recorded from the best-performing landing model (baseline_s0, seed 1010,
       score: 303.07) demonstrating safe touchdown on the pad.
    2. Hovering clip: Recorded from the reward-hacked model (hack_s0, seed 1004,
       1,000 frames timeout) demonstrating prolonged hovering to farm reward.

Outputs:
    results/clips/landing.mp4   (gitignored per PRD §4 & .gitignore)
    results/clips/hover.mp4     (gitignored per PRD §4 & .gitignore)
    results/clips/landing.gif   (optimized for README, < 10 MB)
    results/clips/hover.gif     (optimized for README, < 10 MB)
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import Optional

import PIL.Image

# Pillow 10+ compatibility patch for MoviePy 1.0.3
if not hasattr(PIL.Image, "ANTIALIAS"):
    PIL.Image.ANTIALIAS = PIL.Image.Resampling.LANCZOS

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from moviepy.editor import VideoFileClip
from stable_baselines3 import PPO


def record_episode(
    model_path: Path,
    seed: int,
    output_mp4: Path,
    max_steps: Optional[int] = None,
) -> float:
    """
    Run one deterministic episode with RecordVideo on raw LunarLander-v3.
    """
    temp_dir = output_mp4.parent / f"_temp_{output_mp4.stem}"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    model = PPO.load(str(model_path))
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=str(temp_dir),
        name_prefix=output_mp4.stem,
        episode_trigger=lambda ep: ep == 0,
    )

    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    terminated = truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(int(action))
        total_reward += float(reward)
        steps += 1
        if max_steps and steps >= max_steps:
            break

    env.close()

    # Move generated MP4 to destination
    generated_mp4s = list(temp_dir.glob("*.mp4"))
    if not generated_mp4s:
        raise RuntimeError(f"No MP4 was generated in {temp_dir}")

    if output_mp4.exists():
        output_mp4.unlink()
    shutil.move(str(generated_mp4s[0]), str(output_mp4))
    shutil.rmtree(temp_dir)

    print(
        f"[OK] Recorded {output_mp4.name}: {steps} steps, "
        f"reward: {total_reward:.2f}, file size: {output_mp4.stat().st_size:,} bytes"
    )
    return total_reward


def convert_mp4_to_gif(
    mp4_path: Path,
    gif_path: Path,
    target_fps: int = 15,
    target_width: int = 420,
    max_duration: Optional[float] = None,
) -> None:
    """
    Convert MP4 to compact, high-quality GIF (<10 MB for GitHub README).
    """
    print(f"Converting {mp4_path.name} -> {gif_path.name} (fps={target_fps}, width={target_width}) ...")
    clip = VideoFileClip(str(mp4_path))
    if max_duration and clip.duration > max_duration:
        clip = clip.subclip(0, max_duration)

    # Resize to keep dimensions clean and file small
    resized = clip.resize(width=target_width)
    resized.write_gif(
        str(gif_path),
        fps=target_fps,
        program="imageio",
        opt="nq",  # NeuQuant palette quantization for crisp visual quality
        verbose=False,
        logger=None,
    )
    clip.close()
    resized.close()

    gif_size_mb = gif_path.stat().st_size / (1024 * 1024)
    print(f"[OK] Generated {gif_path.name} ({gif_size_mb:.2f} MB)")
    if gif_size_mb > 10.0:
        print(f"[WARN] GIF size ({gif_size_mb:.2f} MB) exceeds 10 MB!")


def main():
    parser = argparse.ArgumentParser(description="Record landing and hover clips")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Root directory of results",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    models_dir = results_dir / "models"
    clips_dir = results_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────
    # 1. Landing Clip: Best landing model (baseline_s0, seed 1010, score: 303.07)
    # ─────────────────────────────────────────────────────────────────
    landing_model = models_dir / "baseline_s0.zip"
    landing_seed = 1010
    landing_mp4 = clips_dir / "landing.mp4"
    landing_gif = clips_dir / "landing.gif"

    print("\n--- Recording Landing Clip (baseline_s0, seed 1010) ---")
    record_episode(landing_model, landing_seed, landing_mp4)
    convert_mp4_to_gif(landing_mp4, landing_gif, target_fps=18, target_width=440)

    # ─────────────────────────────────────────────────────────────────
    # 2. Hovering Clip: Reward hacking model (hack_s0, seed 1004, 1,000 steps timeout)
    # ─────────────────────────────────────────────────────────────────
    hover_model = models_dir / "hack_s0.zip"
    hover_seed = 1004
    hover_mp4 = clips_dir / "hover.mp4"
    hover_gif = clips_dir / "hover.gif"

    print("\n--- Recording Hovering Clip (hack_s0, seed 1004, 1000-frame timeout) ---")
    record_episode(hover_model, hover_seed, hover_mp4)
    # 1000 frames at 50fps = 20s. We encode at 12 fps with width=400 to keep under 10MB.
    convert_mp4_to_gif(hover_mp4, hover_gif, target_fps=12, target_width=400)

    print("\n==========================================")
    print("All clips and GIFs generated successfully!")
    print(f"Landing GIF: {landing_gif} ({landing_gif.stat().st_size / (1024*1024):.2f} MB)")
    print(f"Hover GIF  : {hover_gif} ({hover_gif.stat().st_size / (1024*1024):.2f} MB)")
    print("==========================================")


if __name__ == "__main__":
    main()
