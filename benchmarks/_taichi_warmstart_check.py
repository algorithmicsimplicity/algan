"""Prove the warm-start memoization is free, and say what it buys.

`algan/utils/taichi_warmstart.py` replaces two compiler internals -- the
per-node source-position banner, and the source retrieval behind every
transform -- with memoizing copies whose entire claim is byte-identity. This is
the end-to-end audit of that claim, and the place the module's docstring points
at. `tests/unit_tests/test_taichi_warmstart.py` covers the same memos against a
duck-typed context; this runs them inside a real materialization, where a
context the tests did not think of would show up.

    uv run python benchmarks/_taichi_warmstart_check.py
    ALGAN_TAICHI_BACKEND=quadrants uv run python benchmarks/_taichi_warmstart_check.py

Three arms, each a separate process because the patch installs at import and a
`ti.static` gate is resolved when a kernel compiles:

* **off** -- `ALGAN_TAICHI_WARMSTART=0`, the compiler as shipped. The baseline
  for both the timing and the pixels.
* **on** -- the memoization, which is what a render actually uses.
* **verify** -- `ALGAN_TAICHI_WARMSTART_VERIFY=1`: every memoized value is
  recomputed the original way and compared, raising on the first byte of
  difference. Slower than **off** by construction (it does both), so read it as
  a correctness arm and never as a timing.

It renders a frame rather than timing a transform directly: the cost being
removed is paid per kernel materialization, and one trivial scene already
materializes ~22 kernels. Run it twice -- the first run pays cold compilation,
which is not what this measures.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

ARMS = {
    "off": {"ALGAN_TAICHI_WARMSTART": "0"},
    "on": {"ALGAN_TAICHI_WARMSTART": "1"},
    "verify": {
        "ALGAN_TAICHI_WARMSTART": "1",
        "ALGAN_TAICHI_WARMSTART_VERIFY": "1",
    },
}

# Rendered in the child, so the arm's environment is the process's environment.
_CHILD = """
import os, time
t0 = time.perf_counter()
import algan
from algan import PREVIEW, SETTINGS, Scene, Square
from algan.taichi_compat import describe_backend
from algan.utils.taichi_warmstart import skipped_reason
t_import = time.perf_counter()
SETTINGS.video.set(PREVIEW)
with Scene() as scene:
    Square().spawn()
    t_record = time.perf_counter()
    scene.save_frame(os.environ["FRAME_OUT"], video_settings=PREVIEW, overwrite=True)
t_done = time.perf_counter()
print(
    f"CHILD backend={describe_backend()} skipped={skipped_reason()!r} "
    f"import={t_import - t0:.2f} render={t_done - t_record:.2f}",
    flush=True,
)
"""


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def run_arm(name, out_dir, quiet):
    """Render one frame under one arm; return (seconds, render_seconds, digest)."""
    frame = out_dir / f"{name}.png"
    environment = dict(os.environ, FRAME_OUT=str(frame), **ARMS[name])
    # The daemon keeps renderer state across runs and re-executes the script,
    # either of which would make these three arms incomparable.
    environment.setdefault("ALGAN_AUTO_DAEMON", "0")
    environment.setdefault("ALGAN_USE_DAEMON", "0")
    started = time.perf_counter()
    result = subprocess.run(
        [sys.executable, "-c", _CHILD],
        env=environment,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - started
    if not quiet or result.returncode != 0:
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
    if result.returncode != 0:
        # A VERIFY arm that raises is the whole point of this script: report
        # the mismatch as a finding rather than as a broken run.
        return elapsed, None, None
    match = re.search(r"render=([0-9.]+)", result.stdout)
    render = float(match.group(1)) if match else None
    return elapsed, render, _digest(frame) if frame.exists() else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", default="off,on,verify")
    parser.add_argument("--quiet", action="store_true", help="child output only on failure")
    args = parser.parse_args(argv)

    names = [name.strip() for name in args.arms.split(",") if name.strip()]
    unknown = [name for name in names if name not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; known: {', '.join(ARMS)}")

    results = {}
    with tempfile.TemporaryDirectory(prefix="algan-warmstart-") as tmp:
        out_dir = Path(tmp)
        for name in names:
            print(f"\n=== arm {name} ===", flush=True)
            results[name] = run_arm(name, out_dir, args.quiet)

    print(f"\n{'arm':<8} {'process s':>10} {'render s':>10}  frame")
    for name in names:
        total, render, digest = results[name]
        render_text = f"{render:.2f}" if render is not None else "FAILED"
        print(f"{name:<8} {total:>10.2f} {render_text:>10}  {digest or '-'}")

    failed = [name for name in names if results[name][2] is None]
    digests = {results[name][2] for name in names if results[name][2] is not None}
    if failed:
        print(f"WARMSTART-CHECK: FAILED arms={','.join(failed)}")
        return 1
    if len(digests) != 1:
        print("WARMSTART-CHECK: FAILED the arms rendered different pixels")
        return 1
    speedup = ""
    if "off" in results and "on" in results and results["on"][1]:
        speedup = f" speedup={results['off'][1] / results['on'][1]:.2f}x"
    print(f"WARMSTART-CHECK: PASS identical frames across {len(names)} arms{speedup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
