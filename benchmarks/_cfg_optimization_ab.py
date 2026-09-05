"""What ``cfg_optimization=False`` buys in compile time, and what it costs in pixels.

`taichi_patches/PLAN.md` row 19 lists the compiler's control-flow-graph
optimization pass as the one compile-time knob that needs no patch: Quadrants
documents turning it off as roughly 6x faster compilation for 1-5 % of runtime.
Algan runs with the compiler's default (on). This measures the trade on Algan's
own kernels, and changes nothing: it is a measurement, and the number it
produces is the argument for or against flipping the default.

    uv run python benchmarks/_cfg_optimization_ab.py
    ALGAN_TAICHI_BACKEND=taichi uv run python benchmarks/_cfg_optimization_ab.py

One process per arm, because ``ti.init`` is process-global. Each arm gets a
fresh, private offline cache (``TI_OFFLINE_CACHE_FILE_PATH`` pointed at a temp
directory, so the cache the machine's real renders use is neither cleared nor
polluted), renders one frame of ``tests/fast/scene.py`` cold -- every kernel
compiled from nothing -- and then renders it again in a second process against
the cache the first one wrote, for the warm render time. The arm's
``cfg_optimization`` value reaches ``ti.init`` by wrapping
``taichi_runtime.taichi_init_kwargs`` (never by calling ``ti.init``; see
``agent_guidance/taichi.md``), and the child reads the live ``CompileConfig``
back to prove which arm it really ran.

Per-kernel compile records come from ``ALGAN_LOG_TAICHI_COMPILES=1``: the
*backend* seconds are the compile the pass belongs to (the frontend is the
Python AST transform and is the same in both arms). The two cold frames are
compared channel by channel.

Measured 2026-09-04 (Quadrants 1.3.0, x64 CPU arch, 4 shared cores -- read
the ratios, not the seconds; PREVIEW, ``at=1.0``, 23 kernels per frame):

    arm   cold backend   cold frontend   cold render   warm render   frame
    on        63.4 s          6.0 s         83.6 s        10.7 s     4c4c...
    off       30.1 s          5.8 s         49.7 s        10.8 s     4c4c...

    cfg_optimization=False: backend compile 0.47x (2.1x faster), warm render
    1.01x (inside noise), frames byte-identical (max channel delta 0).

Not the documented 6x: that figure is for kernels where the pass dominates,
and Algan's megakernels spend most of their backend time in LLVM's own O3
(`PLAN.md` §2.1).

**That 2.1x did not reproduce.** Measured again 2026-09-04 (later), same box,
both compilers, the frames byte-identical again:

    backend     arm   cold backend   cold frontend   cold render   warm render
    Quadrants   on        20.9 s         13.9 s         37.0 s        15.7 s
    Quadrants   off       21.2 s         13.5 s         36.7 s        17.9 s
    Taichi      on        18.4 s          4.7 s         25.6 s         7.3 s
    Taichi      off       19.2 s          4.9 s         26.6 s         7.8 s

    cfg_optimization=False: backend compile 1.01x (Quadrants) / 1.05x (Taichi),
    warm render inside noise, max channel delta 0.

Both arms of the second run are faster than the *off* arm of the first, so the
first run's 63.4 s "on" arm was the shared box, not the pass: on Algan's
kernels the CFG pass costs nothing measurable and buys nothing measurable, and
the default stays where the compiler put it. What neither run can see is CUDA:
the pass runs on the CHI IR before either backend, so the compile saving, if
there is one, should carry over, but the runtime cost is a GPU question
(register pressure, occupancy) and needs the T4 harness
(`agent_guidance/gpu_harnesses.md`) before the default moves.
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
SCENE_FILE = REPO_ROOT / "tests" / "fast" / "scene.py"
FONT_DIR = REPO_ROOT / "tests" / "assets" / "fonts"

ARMS = {"on": "1", "off": "0"}

# Rendered in the child, so the arm's environment is the process's environment.
_CHILD = """
import importlib.util
import os
import sys
import time
from pathlib import Path

# The fast scene pins its font to the vendored faces, as tests/conftest.py does.
import manimpango
for face in sorted(Path(os.environ["FONT_DIR"]).glob("*.ttf")):
    manimpango.register_font(str(face))

import algan
from algan import PREVIEW, Scene
from algan.rendering import taichi_runtime
from algan.taichi_compat import describe_backend, program

arm = os.environ["CFG_OPT_ARM"] == "1"
algan_kwargs = taichi_runtime.taichi_init_kwargs


def kwargs_for_this_arm():
    kwargs = algan_kwargs()
    kwargs["cfg_optimization"] = arm
    return kwargs


taichi_runtime.taichi_init_kwargs = kwargs_for_this_arm

spec = importlib.util.spec_from_file_location("_algan_fast_scene", os.environ["SCENE_FILE"])
module = importlib.util.module_from_spec(spec)
with Scene() as scene:
    spec.loader.exec_module(module)
    t0 = time.perf_counter()
    scene.save_frame(os.environ["FRAME_OUT"], video_settings=PREVIEW, at=1.0, overwrite=True)
    render_seconds = time.perf_counter() - t0
live = program().config().cfg_optimization
print(
    f"CHILD backend={describe_backend()} cfg_optimization={live} render={render_seconds:.2f}",
    flush=True,
)
"""

_COMPLETED = re.compile(
    r"\[Taichi compile\] completed (?P<name>.+?) at \S+: "
    r"frontend=(?P<frontend>[0-9.]+)s, backend=(?P<backend>[0-9.]+)s"
)


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def run_child(arm, cache_dir, frame, quiet):
    """One render of the scene under ``arm``; returns a dict of what it measured."""
    environment = dict(
        os.environ,
        CFG_OPT_ARM=ARMS[arm],
        FRAME_OUT=str(frame),
        SCENE_FILE=str(SCENE_FILE),
        FONT_DIR=str(FONT_DIR),
        TI_OFFLINE_CACHE_FILE_PATH=str(cache_dir),
        ALGAN_LOG_TAICHI_COMPILES="1",
    )
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
        raise SystemExit(f"arm {arm!r} failed (exit {result.returncode})")
    compiles = [m.groupdict() for m in _COMPLETED.finditer(result.stdout)]
    child = re.search(r"CHILD .*cfg_optimization=(\S+) render=([0-9.]+)", result.stdout)
    return {
        "wall": elapsed,
        "render": float(child.group(2)),
        "live": child.group(1),
        "kernels": len(compiles),
        "frontend": sum(float(c["frontend"]) for c in compiles),
        "backend": sum(float(c["backend"]) for c in compiles),
        "digest": _digest(frame),
    }


def compare_frames(first, second):
    """Max channel delta and the number of differing pixels between two PNGs."""
    import numpy as np
    from PIL import Image

    a = np.asarray(Image.open(first).convert("RGB"), dtype=np.int16)
    b = np.asarray(Image.open(second).convert("RGB"), dtype=np.int16)
    if a.shape != b.shape:
        return None, None
    delta = np.abs(a - b)
    return int(delta.max()), int((delta.max(axis=2) > 0).sum())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--arms", default="on,off")
    parser.add_argument(
        "--quiet", action="store_true", help="child output only on failure"
    )
    args = parser.parse_args(argv)
    names = [name.strip() for name in args.arms.split(",") if name.strip()]
    unknown = [name for name in names if name not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; known: {', '.join(ARMS)}")

    results = {}
    with tempfile.TemporaryDirectory(prefix="algan-cfg-opt-") as tmp:
        out_dir = Path(tmp)
        for name in names:
            cache_dir = out_dir / f"cache_{name}"
            print(f"\n=== arm {name}: cold (fresh cache) ===", flush=True)
            cold = run_child(name, cache_dir, out_dir / f"{name}_cold.png", args.quiet)
            print(f"=== arm {name}: warm (same cache) ===", flush=True)
            warm = run_child(name, cache_dir, out_dir / f"{name}_warm.png", args.quiet)
            results[name] = (cold, warm)
        frames = {name: out_dir / f"{name}_cold.png" for name in names}

        print(
            f"\n{'arm':<5} {'live':>5} {'kernels':>7} {'cold backend':>12} "
            f"{'cold frontend':>13} {'cold render':>11} {'cold wall':>9} "
            f"{'warm render':>11}  frame"
        )
        for name in names:
            cold, warm = results[name]
            same = "" if cold["digest"] == warm["digest"] else "  (warm frame differs!)"
            print(
                f"{name:<5} {cold['live']:>5} {cold['kernels']:>7} "
                f"{cold['backend']:>11.1f}s {cold['frontend']:>12.1f}s "
                f"{cold['render']:>10.1f}s {cold['wall']:>8.1f}s "
                f"{warm['render']:>10.1f}s  {cold['digest']}{same}"
            )
        if "on" in results and "off" in results:
            on, off = results["on"][0], results["off"][0]
            max_delta, pixels = compare_frames(frames["on"], frames["off"])
            warm_ratio = results["off"][1]["render"] / results["on"][1]["render"]
            print(
                f"\ncfg_optimization=False: backend compile "
                f"{off['backend'] / on['backend']:.2f}x, warm render {warm_ratio:.2f}x, "
                f"max channel delta {max_delta} over {pixels} differing pixel(s)"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
