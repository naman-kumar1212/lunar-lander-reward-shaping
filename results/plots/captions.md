# Figure Captions & Scientific Context

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
