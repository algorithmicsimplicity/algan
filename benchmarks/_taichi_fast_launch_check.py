"""Prove the fast launcher is invisible except in the clock, and that it ran.

`algan/utils/taichi_fast_launch.py` replaces ``Kernel.__call__`` with a
dispatcher that, after the first launch of each instantiation, skips the
compiler's Python re-validation and replays the C++ set-arg calls itself. Its
whole claim is byte-identity: the same compiled kernel receives the same
argument values. This is the end-to-end audit of that claim on whichever
compiler is live, and the place the module's docstring points at.
`tests/unit_tests/test_taichi_fast_launch.py` covers the dispatcher's key and
fallbacks kernel by kernel; this runs it under a real render.

    uv run python benchmarks/_taichi_fast_launch_check.py
    ALGAN_TAICHI_BACKEND=taichi uv run python benchmarks/_taichi_fast_launch_check.py

Three arms, each a separate process because the dispatcher installs at import:

* **off** -- `ALGAN_TAICHI_FAST_LAUNCH=0`, the compiler's own launch path.
  The baseline for both the timing and the pixels.
* **on** -- the dispatcher, which is what a render actually uses.
* **verify** -- `ALGAN_TAICHI_FAST_LAUNCH_VERIFY=1`: on every fast hit the
  compiler's own instantiation lookup is re-run and compared with the plan,
  raising on a disagreement. Slower than **off** by construction (it does
  both), so read it as a correctness arm and never as a timing.

Each arm renders five frames of a moving ``Square`` (``save_frame`` with a
sequence of times, so there is no video encoder between the pixels and the
digest) and reports the dispatcher's ``STATS``: the check fails unless the
frames are identical across arms **and** the on/verify arms took the fast
path for most launches -- a fast path that silently disengaged would pass a
pixel comparison vacuously. Run it twice; the first run pays cold compilation.
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
    "off": {"ALGAN_TAICHI_FAST_LAUNCH": "0"},
    "on": {"ALGAN_TAICHI_FAST_LAUNCH": "1"},
    "verify": {
        "ALGAN_TAICHI_FAST_LAUNCH": "1",
        "ALGAN_TAICHI_FAST_LAUNCH_VERIFY": "1",
    },
}

#: Below this share of fast-path launches in the on/verify arms the check
#: fails: five frames of ~30 launches each record at most one plan per
#: (kernel, instantiation) on the first frame and hit it on the other four.
MIN_FAST_SHARE = 0.6

# Rendered in the child, so the arm's environment is the process's environment.
_CHILD = """
import os, time
import algan
from algan import PREVIEW, RIGHT, SETTINGS, Scene, Square
from algan.taichi_compat import describe_backend
from algan.utils import taichi_fast_launch
SETTINGS.video.set(PREVIEW)
with Scene() as scene:
    square = Square().spawn()
    square.move(RIGHT)
    t_record = time.perf_counter()
    scene.save_frame(
        os.environ["FRAME_OUT"],
        video_settings=PREVIEW,
        at=[0.0, 0.25, 0.5, 0.75, 1.0],
        overwrite=True,
    )
t_done = time.perf_counter()
print(
    f"CHILD backend={describe_backend()} installed={taichi_fast_launch._APPLIED} "
    f"skipped={taichi_fast_launch.skipped_reason()!r} "
    f"fast={taichi_fast_launch.STATS['fast']} slow={taichi_fast_launch.STATS['slow']} "
    f"render={t_done - t_record:.2f}",
    flush=True,
)
"""


def _digest(out_dir, name):
    """One digest over every frame the arm wrote, in name order."""
    frames = sorted(out_dir.glob(f"{name}*.png"))
    if not frames:
        return None, 0
    digest = hashlib.sha256()
    for frame in frames:
        digest.update(frame.read_bytes())
    return digest.hexdigest()[:16], len(frames)


def run_arm(name, out_dir, quiet):
    """Render under one arm; return (seconds, render seconds, digest, frames, fast, slow)."""
    frame = out_dir / f"{name}.png"
    environment = dict(os.environ, FRAME_OUT=str(frame), **ARMS[name])
    # The daemon keeps renderer state across runs and re-executes the script,
    # either of which would make the arms incomparable.
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
        return elapsed, None, None, 0, 0, 0
    counts = {
        key: float(match.group(1)) if match else None
        for key in ("render", "fast", "slow")
        for match in [re.search(rf"{key}=([0-9.]+)", result.stdout)]
    }
    digest, frames = _digest(out_dir, name)
    return (
        elapsed,
        counts["render"],
        digest,
        frames,
        int(counts["fast"] or 0),
        int(counts["slow"] or 0),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", default="off,on,verify")
    parser.add_argument(
        "--quiet", action="store_true", help="child output only on failure"
    )
    args = parser.parse_args(argv)

    names = [name.strip() for name in args.arms.split(",") if name.strip()]
    unknown = [name for name in names if name not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; known: {', '.join(ARMS)}")

    results = {}
    with tempfile.TemporaryDirectory(prefix="algan-fast-launch-") as tmp:
        out_dir = Path(tmp)
        for name in names:
            print(f"\n=== arm {name} ===", flush=True)
            results[name] = run_arm(name, out_dir, args.quiet)

    print(
        f"\n{'arm':<8} {'process s':>10} {'render s':>10} {'frames':>7} {'fast':>6} {'slow':>6}  digest"
    )
    for name in names:
        total, render, digest, frames, fast, slow = results[name]
        render_text = f"{render:.2f}" if render is not None else "FAILED"
        print(
            f"{name:<8} {total:>10.2f} {render_text:>10} {frames:>7} {fast:>6} {slow:>6}  {digest or '-'}"
        )

    failed = [name for name in names if results[name][2] is None]
    if failed:
        print(f"FAST-LAUNCH-CHECK: FAILED arms={','.join(failed)}")
        return 1
    digests = {results[name][2] for name in names}
    if len(digests) != 1:
        print("FAST-LAUNCH-CHECK: FAILED the arms rendered different pixels")
        return 1
    disengaged = []
    for name in names:
        if name == "off":
            continue
        _total, _render, _digest_, _frames, fast, slow = results[name]
        if fast < MIN_FAST_SHARE * max(fast + slow, 1):
            disengaged.append(f"{name} (fast={fast}, slow={slow})")
    if disengaged:
        print(
            f"FAST-LAUNCH-CHECK: FAILED the fast path did not engage in {', '.join(disengaged)}"
        )
        return 1
    speedup = ""
    if "off" in results and "on" in results and results["on"][1]:
        speedup = f" render speedup={results['off'][1] / results['on'][1]:.2f}x (cross-process; indicative only)"
    print(f"FAST-LAUNCH-CHECK: PASS identical frames across {len(names)} arms{speedup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
