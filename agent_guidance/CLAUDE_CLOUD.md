# Cloud sessions (Claude Code on the web)
A cloud session is a fresh Ubuntu 24.04 VM, **4 vCPUs / 16 GB RAM / 30 GB disk**, with no GPU and nothing installed beyond the base image. `.claude/hooks/session-start.sh` provisions it before you get control: apt build/LaTeX/ffmpeg packages, then `uv sync --locked --all-extras --dev`. It is a no-op on a local checkout. If a build or a Tex test fails with missing headers or a missing `latex`, read that script first — the environment is probably mid-provision or the apt step warned and continued.

What is different here, and what it means for what you can conclude:
- **Render twice; baseline the second.** The first render on a fresh container populates the Manim Tex geometry cache, and its `MathTex` glyph antialiasing differs from every run after it — measured at 18 channel values over 100 frames of `text_and_media`, against a tolerance of 2. Baseline the cold run and the suite fails on the next run for no visible reason. `tests/README.md` has the measurement.
- **Watch the disk.** A full install lands around 12 GB of the 30 GB. `df` reports the allowance, not the machine, so "Avail 0" with low "Used" means the allowance is spent.

Persistence: the container is ephemeral and nothing outside git survives it. Commit anything worth keeping. Repo-level config (`CLAUDE.md`, `.claude/settings.json`, the hook) is what carries over; the environment's own **setup script** and **environment variables**, configured in the environment dialog at claude.ai/code, persist separately and are snapshotted after their first run.
