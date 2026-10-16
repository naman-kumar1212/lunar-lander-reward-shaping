"""
src/train.py
------------
PPO training script for LunarLander-v3 reward-shaping study (PRD §6.4).

Usage
-----
    # smoke test (~50k steps)
    python src/train.py --variant baseline --seed 0 --timesteps 50000

    # full run (1M steps, for Colab)
    python src/train.py --variant baseline --seed 0

CLI flags
---------
--variant       One of: baseline, time_penalty, hack, fix  [required]
--seed          Integer seed (0, 1, or 2 per PRD §6.4)     [default: 0]
--timesteps     Total environment steps                      [default: 1_000_000]
--no-subproc    Use DummyVecEnv instead of SubprocVecEnv    [flag]
--n-envs        Parallel environments                        [default: 16]

PPO hyperparameters (PRD §6.4, fixed across all variants)
----------------------------------------------------------
    n_steps=1024, batch_size=64, n_epochs=4,
    gamma=0.999, gae_lambda=0.98, ent_coef=0.01
    policy: MlpPolicy, net_arch=[64, 64]

Output
------
    results/models/<variant>_s<seed>.zip     saved model
    results/monitor/<variant>_s<seed>/       Monitor CSV (unshaped rewards)
    results/tb/<variant>_s<seed>/            TensorBoard logs (gitignored)
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure src/ is importable when run from repo root
sys.path.insert(0, str(Path(__file__).parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from wrappers import load_variant_config, make_env

# ── PRD §6.4 fixed hyperparameters ──────────────────────────────────────────
PPO_KWARGS = dict(
    policy="MlpPolicy",
    n_steps=1024,
    batch_size=64,
    n_epochs=4,
    gamma=0.999,
    gae_lambda=0.98,
    ent_coef=0.01,
    policy_kwargs=dict(net_arch=[64, 64]),
    verbose=1,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on LunarLander-v3")
    parser.add_argument(
        "--variant",
        required=True,
        choices=["baseline", "time_penalty", "hack", "fix"],
        help="Reward shaping variant (PRD §6.2)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed (PRD §6.4: 0, 1, or 2)"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1_000_000,
        help="Total training timesteps (default: 1M for Colab; use ~50k for local smoke test)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=16,
        help="Number of parallel environments (PRD §6.4: 16)",
    )
    parser.add_argument(
        "--no-subproc",
        action="store_true",
        help="Use DummyVecEnv instead of SubprocVecEnv (fallback for Windows issues)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Output directories ────────────────────────────────────────────────
    run_id = f"{args.variant}_s{args.seed}"
    results_root = Path(__file__).parent.parent / "results"
    monitor_dir = results_root / "monitor" / run_id
    model_path = results_root / "models" / f"{run_id}.zip"
    tb_log_dir = results_root / "tb"

    monitor_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Training: variant={args.variant}  seed={args.seed}")
    print(f"  Timesteps: {args.timesteps:,}  n_envs: {args.n_envs}")
    print(f"  Monitor dir : {monitor_dir}")
    print(f"  Model output: {model_path}")
    print(f"{'='*60}\n")

    # ── Load variant config ───────────────────────────────────────────────
    config = load_variant_config(args.variant)
    print(f"Variant config: {config if config else '(none — pass-through)'}\n")

    # ── Build vectorised environment ──────────────────────────────────────
    # Each parallel env gets seed = base_seed + rank so they differ.
    # Monitor sits INSIDE RewardShapingWrapper → logs unshaped reward (PRD §6.1).
    env_fns = [
        make_env(
            variant=args.variant,
            seed=args.seed + rank,
            config=config,
            monitor_dir=str(monitor_dir / f"env_{rank}"),
        )
        for rank in range(args.n_envs)
    ]

    VecEnvCls = DummyVecEnv if args.no_subproc else SubprocVecEnv
    print(f"Using {VecEnvCls.__name__} with {args.n_envs} envs …")
    try:
        vec_env = VecEnvCls(env_fns)
    except Exception as exc:
        if not args.no_subproc:
            print(f"SubprocVecEnv failed ({exc}). Falling back to DummyVecEnv.")
            vec_env = DummyVecEnv(env_fns)
        else:
            raise

    # ── Build PPO model ───────────────────────────────────────────────────
    model = PPO(
        env=vec_env,
        seed=args.seed,
        tensorboard_log=str(tb_log_dir),
        **PPO_KWARGS,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    model.learn(
        total_timesteps=args.timesteps,
        tb_log_name=run_id,
        progress_bar=False,
    )

    # ── Save model ────────────────────────────────────────────────────────
    model.save(str(model_path))
    vec_env.close()

    print(f"\n[DONE] Model saved to {model_path}")
    print(f"       Monitor CSVs in {monitor_dir}")


if __name__ == "__main__":
    # Guard required for SubprocVecEnv on Windows (spawn start method)
    import multiprocessing
    multiprocessing.freeze_support()
    main()
