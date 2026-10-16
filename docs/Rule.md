# Project Rules

1. Read docs/PRD.md fully before any action. Do not add, remove, or change scope, variants, seeds, hyperparameters, metrics, or repo structure.
2. Work on ONE PRD phase at a time (section 12). Never start the next phase without my explicit approval.
3. Before writing code, show a short implementation plan for the phase and wait for my approval.
4. At the end of each phase, prove the phase's exit criterion with real command output (not claims), then stop and summarize.
5. If something in the PRD is ambiguous, impossible, or you want to deviate, STOP and ask. Record every approved deviation in docs/DECISIONS.md.
6. Environment: Windows, Python 3.10+, virtual environment in .venv. Use LunarLander-v3 only. Pin dependency versions in requirements.txt.
7. Never hardcode secrets or tokens. Never commit .venv, large video files, or tokens.
8. Training runs are executed by me on Google Colab. You write the notebook and scripts; do not attempt full 1M-step runs locally (smoke tests of ~50k steps only).
9. Evaluation always uses the raw environment with the reward wrapper off.
10. Log the unshaped built-in reward during training so curves are comparable across variants.
