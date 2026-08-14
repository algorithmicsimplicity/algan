#!/bin/bash
# Provision a Claude Code on the web session so `pytest` and `ruff` work.
#
# Cloud sessions start from a fresh clone on a bare Ubuntu 24.04 VM: no venv, no
# system libraries beyond the base image. Two things bite in that order:
#
#   1. manimpango publishes no Linux wheels at all and pycairo may also build
#      from source, so both compile during `uv sync`. Without the Pango headers
#      the sync dies on "Package 'pangocairo' was not found". apt must come
#      first.
#   2. Tex/Text mobs shell out to LaTeX, and a chunk of the unit suite builds
#      them, so a session without LaTeX fails tests that have nothing to do with
#      the change under review.
#
# Local checkouts already have all of this, so the whole script is a no-op off
# the cloud. Everything here is idempotent: apt and uv both no-op when satisfied,
# which keeps resumed sessions fast.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel)}"

log() { echo "[session-start] $*"; }

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
# Kept to what the build and the test suite actually need. The LaTeX set is the
# one CI installs: Algan's default TeX template only pulls standalone, babel,
# amsmath and amssymb, and texlive-latex-extra brings texlive-latex-recommended
# and dvisvgm along. texlive-full would be ~10 GB and half an hour for packages
# Algan never loads.
APT_PACKAGES=(
  build-essential python3-dev pkg-config   # compile manimpango / pycairo
  libcairo2-dev libpango1.0-dev            # their headers
  texlive-latex-base texlive-latex-extra   # Tex/Text mobs
  texlive-fonts-recommended latexmk
  ffmpeg                                   # docs build only; silences a
)                                          # harmless Manim startup warning

missing=()
for pkg in "${APT_PACKAGES[@]}"; do
  dpkg -s "$pkg" >/dev/null 2>&1 || missing+=("$pkg")
done

if [ ${#missing[@]} -gt 0 ]; then
  log "installing ${#missing[@]} system package(s): ${missing[*]}"
  export DEBIAN_FRONTEND=noninteractive
  # Non-fatal: a registry hiccup should degrade the session, not refuse to start
  # it. A genuinely missing Pango header surfaces as a clear uv sync error below.
  apt-get update -qq || log "WARNING: apt-get update failed; continuing"
  apt-get install -y --no-install-recommends "${missing[@]}" \
    || log "WARNING: apt-get install failed; LaTeX or build deps may be absent"
else
  log "system packages already present"
fi

# ---------------------------------------------------------------------------
# Python environment
# ---------------------------------------------------------------------------
# --locked installs exactly uv.lock. Resolving fresh instead picks up newer
# Manim, Torch and OpenCV than Algan is tested against, and that has broken the
# package before -- a warm SVG cache made every Tex after the first raise
# KeyError on Manim 0.21 while the pinned 0.19 was fine.
log "syncing the virtual environment (uv sync --locked --all-extras --dev)"
uv sync --locked --all-extras --dev

# Torch here is the CUDA build the lockfile pins, on a VM that has no GPU. That
# is fine -- Algan detects the absence and runs on CPU -- and it is currently
# the only option, because the CPU-only wheels live on download.pytorch.org,
# which the default Trusted network policy blocks. See CLAUDE.md if you want to
# allowlist it.
log "ready: $(uv run python -c 'import algan, torch; print("torch", torch.__version__, "| cuda", torch.cuda.is_available())' 2>/dev/null | tail -1)"
