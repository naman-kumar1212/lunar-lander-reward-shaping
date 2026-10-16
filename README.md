# Lunar Lander — Reward Shaping with PPO

> **Status:** Work in progress. Training results will be added after Colab runs complete.

## Project Summary

This project trains a PPO agent (Stable-Baselines3) to land a rocket in Gymnasium's `LunarLander-v3`, then studies how **reward shaping** changes its behaviour. It deliberately triggers **reward hacking** (the rocket learns to hover instead of landing) and then fixes it.

Built as a portfolio piece demonstrating: reinforcement learning, reward engineering, and the risks of misspecified reward functions.

---

## Variants

| ID | Name | Reward |
|----|------|--------|
| A | Baseline | Built-in reward only |
| B | Time penalty | Built-in reward − 0.02 per frame |
| C | Hack | Built-in reward + bonus for being upright and slow (causes hovering) |
| D | Fix | Potential-based shaping + time penalty (removes hovering) |

---

## Results

> _To be filled after training runs complete._

| Variant | Mean Score | Std | Success Rate | Mean Length | Timeout Rate |
|---------|-----------|-----|-------------|-------------|--------------|
| A | — | — | — | — | — |
| B | — | — | — | — | — |
| C | — | — | — | — | — |
| D | — | — | — | — | — |

---

## Reproduction Steps

### Prerequisites

- Windows 10/11, Python 3.12
- Microsoft C++ Build Tools (for Box2D)

### 1. Clone and set up

```bash
git clone <repo-url>
cd lunar-lander-reward-shaping
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run a smoke test (≈50k steps)

```bash
python src/train.py --variant baseline --seed 0 --timesteps 50000
```

### 3. Evaluate a trained model

```bash
python src/evaluate.py --model results/models/baseline_s0.zip --variant baseline
```

### 4. Record a clip

```bash
python src/record.py --model results/models/baseline_s0.zip --variant baseline
```

---

## Hack-Tuning Log

> _To be filled during Phase 4._

---

## Repository Structure

```
lunar-lander-reward-shaping/
  README.md
  requirements.txt
  configs/          # variant and PPO settings
  src/
    wrappers.py     # reward wrapper and variants
    random_agent.py # random-action baseline
    train.py        # --variant, --seed, --timesteps
    evaluate.py     # 20-episode evaluation, wrapper off
    record.py       # MP4 and GIF recording
    plot.py         # curves and results table
  notebooks/
    colab_train.ipynb
  results/          # models, CSVs, plots, clips
  docs/
    PRD.md
    DECISIONS.md
```

---

## License

MIT
