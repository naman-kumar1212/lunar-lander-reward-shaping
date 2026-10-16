"""
src/wrappers.py
---------------
Reward-shaping wrapper for LunarLander-v3 (PRD §6.1 and §6.2).

Variants
--------
baseline     Built-in reward only (pass-through).
time_penalty Built-in reward - time_penalty per frame.
hack         Built-in reward + hover_bonus each frame the rocket is
             upright (|tilt| < tilt_threshold) AND slow
             (speed < speed_threshold).
fix          Potential-based shaping: r + gamma*phi(s') - phi(s)
             - time_penalty, where phi(s) = potential_coef if upright
             and slow, else 0. Bounded total bonus prevents farming.

Monitor ordering (PRD §6.1 requirement)
----------------------------------------
To log the UNSHAPED built-in reward, Monitor must sit INSIDE this
wrapper (i.e., closer to the raw environment):

    raw_env  →  Monitor(raw_env)  →  RewardShapingWrapper(...)

Use make_env() below to build the correctly ordered chain.
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor

# Default path to the variant config file
_DEFAULT_CONFIG = Path(__file__).parent.parent / "configs" / "variants.json"

# PPO gamma (used in potential-based shaping for variant D)
GAMMA = 0.999


def load_variant_config(
    variant: str,
    config_path: Path = _DEFAULT_CONFIG,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load coefficients for a variant from the JSON config, with optional overrides."""
    with open(config_path) as f:
        all_configs = json.load(f)
    if variant not in all_configs:
        raise ValueError(
            f"Unknown variant '{variant}'. "
            f"Valid options: {list(all_configs.keys())}"
        )
    cfg = dict(all_configs[variant])
    if overrides:
        cfg.update(overrides)
    return cfg


class RewardShapingWrapper(gym.Wrapper):
    """
    Gymnasium wrapper that applies variant-specific reward shaping.

    The wrapper adds its extra terms on top of the raw environment's
    built-in reward.  If a Monitor wraps the *inner* environment, the
    Monitor will record the unshaped reward (correct for training curves).

    Parameters
    ----------
    env     : gymnasium.Env already wrapped with Monitor (see make_env).
    variant : one of 'baseline', 'time_penalty', 'hack', 'fix'.
    config  : coefficient dict (from load_variant_config).
    """

    VALID_VARIANTS = {"baseline", "time_penalty", "hack", "fix"}

    def __init__(
        self,
        env: gym.Env,
        variant: str = "baseline",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(env)
        if variant not in self.VALID_VARIANTS:
            raise ValueError(
                f"variant must be one of {self.VALID_VARIANTS}, got '{variant}'"
            )
        self.variant = variant
        self.config = config or {}
        self._prev_phi: float = 0.0

    # ------------------------------------------------------------------
    # gymnasium.Wrapper interface
    # ------------------------------------------------------------------

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        obs, info = self.env.reset(**kwargs)
        self._prev_phi = self._phi(obs)
        return obs, info

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaped = self._apply_shaping(obs, reward, terminated=terminated)
        return obs, shaped, terminated, truncated, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _phi(self, obs: np.ndarray) -> float:
        """
        Potential function phi(s) used by variant D.
        Returns potential_coef if the rocket is upright and slow, else 0.
        """
        tilt = abs(float(obs[4]))
        speed = math.sqrt(float(obs[2]) ** 2 + float(obs[3]) ** 2)
        tilt_thr = float(self.config.get("tilt_threshold", 0.2))
        speed_thr = float(self.config.get("speed_threshold", 0.5))
        coef = float(self.config.get("potential_coef", 1.0))
        if tilt < tilt_thr and speed < speed_thr:
            return coef
        return 0.0

    def _apply_shaping(
        self, obs: np.ndarray, reward: float, terminated: bool = False
    ) -> float:
        if self.variant == "baseline":
            return reward

        if self.variant == "time_penalty":
            fee = float(self.config.get("time_penalty", 0.02))
            return reward - fee

        if self.variant == "hack":
            tilt = abs(float(obs[4]))
            speed = math.sqrt(float(obs[2]) ** 2 + float(obs[3]) ** 2)
            tilt_thr = float(self.config.get("tilt_threshold", 0.2))
            speed_thr = float(self.config.get("speed_threshold", 0.5))
            bonus = float(self.config.get("hover_bonus", 1.0))
            if tilt < tilt_thr and speed < speed_thr:
                return reward + bonus
            return reward

        if self.variant == "fix":
            # For potential-based shaping (Ng et al., 1999), terminal state phi = 0
            curr_phi = 0.0 if terminated else self._phi(obs)
            potential_bonus = GAMMA * curr_phi - self._prev_phi
            self._prev_phi = curr_phi
            fee = float(self.config.get("time_penalty", 0.02))
            return reward + potential_bonus - fee

        raise RuntimeError(f"Unhandled variant '{self.variant}'")  # unreachable


# ------------------------------------------------------------------
# Factory used by train.py and the pass-through test
# ------------------------------------------------------------------

def make_env(
    variant: str,
    seed: int,
    config: Optional[Dict[str, Any]] = None,
    monitor_dir: Optional[str] = None,
    render_mode: Optional[str] = None,
):
    """
    Return a callable that builds one correctly-ordered environment chain:

        LunarLander-v3  →  Monitor  →  RewardShapingWrapper

    Monitor is INSIDE the wrapper so it records the unshaped reward.
    The returned callable is compatible with DummyVecEnv / SubprocVecEnv.

    Parameters
    ----------
    variant     : reward variant name.
    seed        : environment seed (each parallel env adds its rank).
    config      : coefficient dict; loaded from configs/variants.json if None.
    monitor_dir : directory for Monitor CSV output; None = no file written.
    render_mode : passed to gym.make (None for training).
    """
    if config is None:
        config = load_variant_config(variant)

    def _init():
        env = gym.make("LunarLander-v3", render_mode=render_mode)
        env = Monitor(env, filename=monitor_dir)
        env = RewardShapingWrapper(env, variant=variant, config=config)
        env.reset(seed=seed)
        return env

    return _init
