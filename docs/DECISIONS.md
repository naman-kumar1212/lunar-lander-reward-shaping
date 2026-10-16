# Decisions Log

All approved deviations from the PRD are recorded here.

| # | Date | Decision | Reason | Approved by |
|---|------|----------|--------|-------------|
| 1 | 2026-10-16 | Use Python 3.12 for `.venv` (not 3.13) | Box2D prebuilt wheels are more stable on 3.12; 3.13 wheel availability is unverified | User |
| 2 | 2026-10-16 | Pin `moviepy==1.0.3` (v1, not v2) | moviepy v2 has a breaking API; v1 is stable and widely documented | User |
| 3 | 2026-10-16 | Exclude TensorBoard logs from git (add to `.gitignore`) | Logs are large; only CSVs tracked; TensorBoard can be regenerated | User (default accepted) |
