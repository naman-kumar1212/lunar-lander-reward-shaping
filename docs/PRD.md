# PRD: Lunar Lander Reward Shaping with PPO

*Status: Draft for sign-off. No code is written until this document is approved.*

## 1. Summary

Train a PPO agent (Stable-Baselines3) to land a rocket in Gymnasium's `LunarLander-v3`, then study how **reward shaping** changes its behavior. The project deliberately triggers **reward hacking** (the rocket learns to hover instead of landing) and then fixes it. The final result is a public GitHub repository with code, results, plots and embedded clips, built as a resume/portfolio piece.

## 2. Goals and non-goals

**Goals**

- Train an agent that lands reliably (mean score of at least 200 over 20 evaluation episodes, wrapper off).
- Implement custom reward rules as a Gymnasium wrapper.
- Demonstrate reward hacking with evidence (a clip, plus metrics showing episodes that run to the 1,000-frame cutoff).
- Demonstrate a fix, and compare all variants statistically across seeds.
- Produce a clean, reproducible GitHub repo that a reviewer can understand in under five minutes.

**Non-goals**

- Other algorithms (DQN, SAC, A2C) and the continuous-action environment.
- Hyperparameter search beyond the fixed settings in section 6.
- GPU training, custom neural architectures, or image-based observations.
- Real-robot or Optimus-style work (cited only as motivation).

## 3. Audience and success criteria

**Audience:** recruiters and engineers skimming a portfolio; the author as a learner.

**Project succeeds when:**

1. Variants A, B and D each reach a mean evaluation score of at least 200 (averaged over 3 seeds).
2. Variant C shows hacking: a clip of the rocket hovering, plus a measurably higher timeout rate and longer episodes than A.
3. Variant D removes the hovering behavior and lands again.
4. The README contains the results table, training curves, and embedded landing and hover clips.
5. Another person can reproduce a run from the README alone.

## 4. Environment and tooling

| Item | Decision |
| --- | --- |
| Local OS | Windows (development and smoke tests only) |
| Training | Google Colab (free tier, CPU), 12 full runs |
| Python | 3.10 or newer, in a virtual environment |
| Libraries | `gymnasium[box2d]`, `stable-baselines3`, `torch`, `moviepy`, `matplotlib`, `pandas`, `tensorboard` |
| Environment ID | `LunarLander-v3` (discrete actions) |
| Version control | Git, hosted on GitHub |

Windows note: the Box2D install may fail mentioning `swig`. Install swig first (`pip install swig`), then the `box2d` extra. If the compiler build still fails, install Microsoft C++ Build Tools. Pin every package version in `requirements.txt` once the local install works.

## 5. Environment facts the project relies on

- Observation: 8 numbers (x offset, height, x velocity, y velocity, tilt, angular velocity, left leg contact, right leg contact), at 50 frames per second.
- Actions: 4 (do nothing, left engine, main engine, right engine), one per frame.
- Built-in reward: change in a position score (distance, speed, tilt, leg contact) per frame, minus fuel cost, with +100 for a safe landing and -100 for a crash at the final frame.
- Episodes are truncated at 1,000 frames.
- An attempt of 200 points or more counts as solved.

## 6. Design

### 6.1 Reward wrapper

A `gymnasium.Wrapper` subclass sits between the environment and PPO. Each step it takes the built-in reward, adds the variant's extra terms, and returns the total. It never touches the environment's code.

**Requirements**

- The variant is selected by a single name (`baseline`, `time_penalty`, `hack`, `fix`).
- All coefficients are configurable from one config file or CLI flags.
- Training logs must record the **unshaped built-in reward** so curves are comparable across variants. Wrapper and Monitor order must be verified during implementation.
- Evaluation always runs on the raw environment with no wrapper.

### 6.2 Variants

| ID | Name | Reward |
| --- | --- | --- |
| A | Baseline | Built-in reward only |
| B | Time penalty | Built-in reward minus a small fee every frame (initial value 0.02, configurable) |
| C | Hack | Built-in reward plus a bonus of +1 each frame the rocket is upright and slow (initial thresholds: tilt under 0.2 rad, speed under 0.5; tunable) |
| D | Fix | Built-in reward plus a bonus based on the **change** in an "upright and slow" score between frames (potential-based), plus the per-frame fee from B |

Variant D works because a bonus for improvement adds up to a bounded total, so hovering can no longer farm points. Exact coefficients are finalized in the design phase and recorded in the config.

### 6.3 Hack-tuning policy

If variant C does not produce hovering on the first attempt, tune its coefficients (bonus size, thresholds) and retrain until hovering appears. **Every attempt is documented** in the README (coefficients tried, metrics, outcome), including failed ones. Only the final chosen C configuration is used for the 3-seed comparison.

### 6.4 PPO configuration

Tuned, rl-zoo-style settings for reliable landing, verified against the rl-zoo config at implementation time:

- Policy: `MlpPolicy` (2 layers of 64 units)
- Parallel environments: 16
- Timesteps: 1,000,000 per run
- `n_steps` 1024, `batch_size` 64, `n_epochs` 4, `gamma` 0.999, `gae_lambda` 0.98, `ent_coef` 0.01
- Seeds: 0, 1, 2 for every variant

These values are fixed across all variants so only the reward differs.

## 7. Experiments

4 variants x 3 seeds = **12 training runs**. Each run saves a final model, a monitor CSV with episode rewards and lengths, and TensorBoard logs.

## 8. Evaluation protocol

- 20 episodes per trained model, deterministic policy, wrapper off.
- Metrics per model: mean score, standard deviation, landing success rate, mean episode length, and timeout rate (share of episodes reaching 1,000 frames).
- Aggregate per variant as mean and spread across the 3 seeds.
- Hovering evidence: variant C should show a higher timeout rate and longer episodes than A, supported by a video clip (scores alone can hide hovering).

## 9. Outputs

- **Results table:** per-variant mean score, success rate, episode length and timeout rate.
- **Training curves:** per-variant mean with spread across seeds, using the unshaped built-in reward.
- **Clips:** one landing clip (best of A, B or D) and one hover clip (C), recorded as MP4 with `RecordVideo`, converted to GIF for the README.
- **Documentation:** README with the project story, results, hack-tuning log, and reproduction steps.

## 10. Repository structure

```
lunar-lander-reward-shaping/
  README.md
  requirements.txt
  configs/            # variant and PPO settings
  src/
    wrappers.py       # reward wrapper and variants
    random_agent.py   # random-action baseline
    train.py          # --variant, --seed, --timesteps
    evaluate.py       # 20-episode evaluation, wrapper off
    record.py         # MP4 and GIF recording
    plot.py           # curves and results table
  notebooks/
    colab_train.ipynb # runs the 12 jobs in batches
  results/            # models, CSVs, plots, clips
```

## 11. Workflow

1. **Local (Windows):** set up the environment, run the random baseline, run a short (\~50k-step) smoke test of each variant.
2. **Colab:** the notebook clones the repo, installs pinned dependencies, and runs training jobs in batches (for example one variant per session) to stay within free-tier session limits.
3. **Results transfer:** Colab pushes results to GitHub. Use a fine-grained personal access token limited to this repo, stored in **Colab Secrets**, never hardcoded or committed. Models and CSVs are small enough for normal Git; large video files are generated locally from the saved models instead of from Colab.
4. **Local:** pull results, evaluate, record clips, generate plots, write the README.

## 12. Milestones and exit criteria

| Phase | Work | Exit criterion |
| --- | --- | --- |
| 0 | PRD sign-off | You approve this document |
| 1 | Local setup | Python imports gymnasium, box2d, SB3 without errors |
| 2 | Random baseline | Window opens, rocket crashes, 8 observation numbers printed and matched to the table |
| 3 | Baseline smoke test and wrapper | Short training run works; wrapper passes a pass-through test (A equals the unwrapped env) |
| 4 | Variants and hack tuning | B, C, D implemented; hover observed in C |
| 5 | Colab training | All 12 runs finished and pushed to GitHub |
| 6 | Evaluation and recording | Metrics table, curves, landing and hover clips produced |
| 7 | README and polish | A stranger can reproduce a run from the README |

## 13. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Box2D/swig install fails on Windows | Install swig first; C++ Build Tools fallback; use Colab for any run that cannot run locally |
| Colab free-tier disconnects or session limits | Run in batches, save checkpoints, push after each batch |
| Variant C does not hover | Tuning policy in 6.3, with all attempts documented |
| High run-to-run variance | 3 seeds per variant; never judge a reward from one run |
| Old tutorials use `LunarLander-v2` | Use `v3` everywhere |
| Leaked GitHub token | Fine-grained, repo-scoped token in Colab Secrets only |
| Shaped reward distorts logged curves | Log the unshaped built-in reward for all variants |

## 14. Decisions already made

- Goal: resume/portfolio piece
- OS: Windows; Python 3.10+ installed
- Scope: standard (4 variants, 3 seeds each)
- Training: Colab, tuned PPO settings, 1M steps
- Results transfer: push from Colab to GitHub
- Hack policy: tune until hovering appears, document attempts
- Presentation: GitHub repo with README and embedded clips

## 15. Remaining items to confirm before coding

- Exact reward coefficients for B, C and D (starting values in 6.2, finalized in Phase 4).
- GitHub repository name and visibility (public recommended for a portfolio).
- Whether to include TensorBoard logs in the repo or only the CSVs.
- Approval of this PRD as written.

## 16. Definition of done

All 12 runs complete, the results table and curves are in the repo, both clips are embedded in the README, the hack-tuning log is written, and the README's reproduction steps have been followed once from a clean clone.
