# Decisions Log

All approved deviations from the PRD are recorded here.

| # | Date | Decision | Reason | Approved by |
|---|------|----------|--------|-------------|
| 1 | 2026-10-16 | Use Python 3.12 for `.venv` (not 3.13) | Box2D prebuilt wheels are more stable on 3.12; 3.13 wheel availability is unverified | User |
| 2 | 2026-10-16 | Pin `moviepy==1.0.3` (v1, not v2) | moviepy v2 has a breaking API; v1 is stable and widely documented | User |
| 3 | 2026-10-16 | Exclude TensorBoard logs from git (add to `.gitignore`) | Logs are large; only CSVs tracked; TensorBoard can be regenerated | User (default accepted) |
| 4 | 2026-10-16 | Adopt PRD initial coefficients for C and terminal $\Phi(s_{terminal})=0$ for D | Initial hack coefficients produced immediate hovering on attempt #1 (PRD §6.3); terminal $\Phi=0$ adheres to canonical Ng et al. (1999) policy invariance | Antigravity / User |
| 5 | 2026-10-17 | MoviePy Pillow 10+ patch and clip selections (landing: baseline_s0 seed 1010, hover: hack_s0 seed 1004) | Pillow >=10 deprecation patch for MoviePy 1.0.3; baseline_s0 (seed 1010) achieved peak score (303.07, 351 frames) and hack_s0 (seed 1004) demonstrated full 1,000-frame hovering timeout | Antigravity / User |
| 6 | 2026-10-17 | Visual Polish: Shared Okabe-Ito style (`src/style.py`), dual 300 DPI PNG & SVG plots, telemetry HUD overlays with motion trails, and FFmpeg two-pass palettegen GIF pipeline (<10 MB) | Elevate visual assets to publication-grade quality; enable high-resolution vector and raster rendering, live discrete thruster telemetry HUD, 0.5x touchdown slow-mo, and head-to-head synchronized Hack vs Fix comparison clip from seed 1004 | Antigravity / User |
