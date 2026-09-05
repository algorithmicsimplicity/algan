"""Prove the source-keyed cache index is invisible except in the clock, and say what it buys.

`algan/utils/taichi_source_key.py` lets a warm process skip the Python AST
transform for a kernel it has compiled before, by mapping a key over the
kernel's source, its arguments and every Python value its body reads to the C++
cache key of the IR that source produced. Its whole claim is soundness: the
artifact served on a hit is the one a full transform would have found. This is
the end-to-end audit of that claim, and the place the module's docstring points
at. `tests/unit_tests/test_taichi_source_key.py` covers the value rules and the
walk against fixtures; this runs the mechanism inside a real render.

    uv run python benchmarks/_taichi_source_key_check.py

Four arms, each a separate process, because the patch installs at import:

* **off** -- `ALGAN_TAICHI_SOURCE_KEY=0`, the index turned off. The baseline
  for both the timing and the pixels. This is the opt-*out* arm: the index has
  been on by default since 2026-09-05, so **on** below is what a render does
  now and **off** is the control.
* **warm** -- the index on, filling itself: every kernel misses and is stored.
  Same frontend cost as **off** plus the key computation; run so that **on**
  measures a warm index rather than a first sighting.
* **on** -- the index on and warm, which is what a render actually uses. Every
  kernel should hit.
* **verify** -- `ALGAN_TAICHI_SOURCE_KEY_VERIFY=1`: every hit is *not* taken;
  the full transform and compile run, and the C++ key they produce is compared
  with the one the index stored, raising on the first mismatch. Slower than
  **off** by construction (it does both), so read it as a correctness arm and
  never as a timing.

Each child runs under `ALGAN_LOG_TAICHI_COMPILES=1` so the per-kernel frontend
seconds can be summed from its output, which is the number the index exists to
move. It renders a frame rather than timing a transform directly: the cost being
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
    "off": {"ALGAN_TAICHI_SOURCE_KEY": "0"},
    "warm": {"ALGAN_TAICHI_SOURCE_KEY": "1"},
    "on": {"ALGAN_TAICHI_SOURCE_KEY": "1"},
    "verify": {
        "ALGAN_TAICHI_SOURCE_KEY": "1",
        "ALGAN_TAICHI_SOURCE_KEY_VERIFY": "1",
    },
}

# Rendered in the child, so the arm's environment is the process's environment.
_CHILD = """
import os, time
t0 = time.perf_counter()
import algan
from algan import PREVIEW, SETTINGS, Scene, Square
from algan.taichi_compat import describe_backend
from algan.utils import taichi_source_key as sk
t_import = time.perf_counter()
SETTINGS.video.set(PREVIEW)
with Scene() as scene:
    Square().spawn()
    t_record = time.perf_counter()
    scene.save_frame(os.environ["FRAME_OUT"], video_settings=PREVIEW, overwrite=True)
t_done = time.perf_counter()
print(
    f"CHILD backend={describe_backend()} skipped={sk.skipped_reason()!r} "
    f"import={t_import - t0:.2f} render={t_done - t_record:.2f} "
    f"hits={sk.STATS['hits']} misses={sk.STATS['misses']} poisoned={sk.STATS['poisoned']} "
    f"verified={sk.STATS['verified']} keyed={sk.STATS['keyed']} key_time={sk.STATS['key_seconds']:.3f}",
    flush=True,
)
if sk.POISONED:
    print("CHILD poisoned kernels: " + ", ".join(sorted(sk.POISONED)), flush=True)
"""

_FRONTEND = re.compile(r"frontend=([0-9.]+)s")
_STAT = re.compile(r"(hits|misses|poisoned|verified|keyed)=(\d+)")


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def run_arm(name, out_dir, quiet):
    """Render one frame under one arm; return a result dict."""
    frame = out_dir / f"{name}.png"
    environment = dict(
        os.environ, FRAME_OUT=str(frame), ALGAN_LOG_TAICHI_COMPILES="1", **ARMS[name]
    )
    # The daemon keeps renderer state across runs and re-executes the script,
    # either of which would make these arms incomparable.
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
        return {
            "process": elapsed,
            "render": None,
            "frontend": None,
            "digest": None,
            "stats": {},
        }
    match = re.search(r"render=([0-9.]+)", result.stdout)
    frontend = sum(float(value) for value in _FRONTEND.findall(result.stdout))
    stats = {
        key: int(value)
        for key, value in _STAT.findall(result.stdout.split("CHILD", 1)[-1])
    }
    return {
        "process": elapsed,
        "render": float(match.group(1)) if match else None,
        "frontend": frontend,
        "digest": _digest(frame) if frame.exists() else None,
        "stats": stats,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", default="off,warm,on,verify")
    parser.add_argument(
        "--quiet", action="store_true", help="child output only on failure"
    )
    args = parser.parse_args(argv)

    names = [name.strip() for name in args.arms.split(",") if name.strip()]
    unknown = [name for name in names if name not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; known: {', '.join(ARMS)}")

    results = {}
    with tempfile.TemporaryDirectory(prefix="algan-source-key-") as tmp:
        out_dir = Path(tmp)
        for name in names:
            print(f"\n=== arm {name} ===", flush=True)
            results[name] = run_arm(name, out_dir, args.quiet)

    print(
        f"\n{'arm':<8} {'process s':>10} {'render s':>10} {'frontend s':>11}  {'hits':>5} {'miss':>5} {'poison':>6} {'verif':>5}  frame"
    )
    for name in names:
        r = results[name]
        render_text = f"{r['render']:.2f}" if r["render"] is not None else "FAILED"
        frontend_text = f"{r['frontend']:.2f}" if r["frontend"] is not None else "-"
        stats = r["stats"]
        print(
            f"{name:<8} {r['process']:>10.2f} {render_text:>10} {frontend_text:>11}  "
            f"{stats.get('hits', 0):>5} {stats.get('misses', 0):>5} {stats.get('poisoned', 0):>6} "
            f"{stats.get('verified', 0):>5}  {r['digest'] or '-'}"
        )

    failed = [name for name in names if results[name]["digest"] is None]
    digests = {
        results[name]["digest"] for name in names if results[name]["digest"] is not None
    }
    if failed:
        print(f"SOURCE-KEY-CHECK: FAILED arms={','.join(failed)}")
        return 1
    if len(digests) != 1:
        print("SOURCE-KEY-CHECK: FAILED the arms rendered different pixels")
        return 1
    on = results.get("on")
    if on is not None and on["stats"] and on["stats"].get("misses", 0):
        print(
            f"SOURCE-KEY-CHECK: WARNING the warm arm still missed {on['stats']['misses']} kernels"
        )
    speedup = ""
    if "off" in results and on is not None and on["frontend"]:
        speedup = f" frontend {results['off']['frontend']:.2f}s -> {on['frontend']:.2f}s ({results['off']['frontend'] / on['frontend']:.1f}x)"
    print(f"SOURCE-KEY-CHECK: PASS identical frames across {len(names)} arms{speedup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
