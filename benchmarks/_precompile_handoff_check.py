"""Does what a precompile worker leaves on disk get used by the next render?

Two hand-offs, checked on whatever device the render runs on:

* **The source-keyed index** (any device). A worker indexes each kernel under
  the key of the arguments it materialized it with. If those are not the
  arguments a render binds, the render never looks the entry up and pays the
  whole Python frontend again. On an MPS render the zero-copy wrapper turns
  every arena tensor into an ``ExternalMetalNdarray`` before the call, so a
  worker that materialized with torch placeholders indexed every arena kernel
  under keys no render computes, and the render's recording, which saw the
  imported ndarrays, recorded none of those kernels at all.
* **The CPU object cache** (CPU arch, Quadrants patch 0012). A kernel's
  machine code is written beside the offline cache, so a fresh process links
  it instead of re-running LLVM's codegen at the kernel's first launch, and a
  worker that prepares a kernel leaves the render nothing to generate.

Each phase is a fresh process with its own ``ALGAN_CACHE_DIR`` and no pool,
and prints one ``RESULT`` line the moment it finishes, so a run that dies half
way still leaves what it measured:

1. ``record``   -- cold render in cache A; what the manifest recorded.
2. ``worker``   -- every recorded spec compiled the way a pool worker does it,
   into a fresh cache B.
3. ``render``   -- the render in cache B: index hits and misses, CPU objects
   written by the render itself (0 means the workers covered it).
4. ``render``   -- again, a fresh process on the now warm cache B.
5. ``render``   -- again with ``ALGAN_CPU_OBJECT_CACHE=0``, the off arm.

On MPS, phases 2-3 run a second time in cache C with the worker's argument
import switched off (``--old-worker``), which is the code before the fix.

    uv run python benchmarks/_precompile_handoff_check.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("ALGAN_USE_DAEMON", "0")
os.environ.setdefault("ALGAN_AUTO_DAEMON", "0")

_ENV = {
    "ALGAN_USE_DAEMON": "0",
    "ALGAN_AUTO_DAEMON": "0",
    "ALGAN_PRECOMPILE_JOBS": "0",
    "ALGAN_FIRST_RUN_POOL": "0",
    "ALGAN_STARTUP_WARMUP": "0",
    "ALGAN_LOG_TAICHI_COMPILES": "1",
}


def _objects(cache_dir):
    """``{name: (inode, mtime_ns)}`` for every CPU object file in the cache."""
    found = {}
    for path in Path(cache_dir).rglob("cpu_object_cache/*.o"):
        stat = path.stat()
        found[path.name] = (stat.st_ino, stat.st_mtime_ns)
    return found


def _written(before, after):
    return sum(1 for name, ident in after.items() if before.get(name) != ident)


def _scene_render():
    from algan import LD, OUTWARD, RIGHT, UP, WHITE, Off, PointLight, Scene, Sphere
    from algan.settings import SETTINGS
    from algan.settings._startup import _TAICHI_CACHE_DIRECTORY


    SETTINGS.raytracing.set(shadows=True)
    out = Path(tempfile.mkdtemp()) / "frame.png"
    started = time.perf_counter()
    with Scene() as scene:
        with Off():
            PointLight(
                location=RIGHT * 3.0 + OUTWARD * 3.0, color=WHITE, intensity=0.6
            ).spawn(animate=False)
            Sphere(radius=0.65).move(RIGHT * 0.4 + UP * 0.2).spawn()
        scene.save_frame(str(out), video_settings=LD)
    return time.perf_counter() - started, _TAICHI_CACHE_DIRECTORY


def _phase_render(args):
    from algan.utils import taichi_source_key as sk

    seconds, cache = _scene_render()
    return {
        "seconds": round(seconds, 2),
        "hits": sk.STATS["hits"],
        "misses": sk.STATS["misses"],
        "poisoned": sk.STATS["poisoned"],
        "cache": str(cache),
    }


def _phase_record(args):
    from algan.rendering import kernel_precompile as kp

    needed = []
    original = kp._encode_compiler_ndarray

    def counting(value):
        encoded = original(value)
        if encoded is not None:
            needed.append(encoded)
        return encoded

    kp._encode_compiler_ndarray = counting
    result = _phase_render(args)
    kp.flush_manifest()
    manifest = kp.read_manifest()
    kernels = sorted(
        {row["spec"]["kernel"].rsplit(":", 1)[1] for row in manifest["entries"].values()}
    )
    result.update(
        specs=len(manifest["entries"]),
        compiler_ndarray_args=len(needed),
        kernels=kernels,
        manifest=str(kp.manifest_path()),
    )
    return result


def _phase_worker(args):
    from algan.rendering import kernel_precompile as kp
    from algan.rendering import mps_zero_copy, taichi_runtime

    if args.old_worker:
        mps_zero_copy.launch_arguments = lambda kernel, arguments: tuple(arguments)
    with open(args.manifest, encoding="utf-8") as handle:
        manifest = json.load(handle)
    statuses = {}
    prepared = 0
    applied = None
    started = time.perf_counter()
    for row in manifest["entries"].values():
        if row["context"] != applied:
            kp.apply_context(manifest["contexts"][row["context"]])
            taichi_runtime.ensure_taichi_for_render()
            kp._touch_cache_index()
            applied = row["context"]
        try:
            outcome = kp.compile_spec(row["spec"])
        except Exception as exc:  # noqa: BLE001 -- a probe reports, it does not stop
            outcome = {"status": f"error {type(exc).__name__}: {exc}"}
        statuses[outcome["status"]] = statuses.get(outcome["status"], 0) + 1
        prepared += "ptx_seconds" in outcome
    return {
        "seconds": round(time.perf_counter() - started, 2),
        "statuses": statuses,
        "prepared": prepared,
    }


_PHASES = {"render": _phase_render, "record": _phase_record, "worker": _phase_worker}


def _child(args):
    result = _PHASES[args.phase](args)
    print("CHILD-RESULT " + json.dumps(result), flush=True)


def _run(label, phase, cache, extra_env=None, extra_args=()):
    env = dict(os.environ, **_ENV, ALGAN_CACHE_DIR=str(cache))
    if phase == "worker":
        env["ALGAN_PRECOMPILE_WORKER"] = "1"
    env.update(extra_env or {})
    before = _objects(cache)
    started = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, __file__, "--phase", phase, *extra_args],
        env=env,
        capture_output=True,
        text=True,
    )
    wall = time.perf_counter() - started
    result = None
    for line in proc.stdout.splitlines():
        if line.startswith("CHILD-RESULT "):
            result = json.loads(line[len("CHILD-RESULT ") :])
    if result is None:
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-25:]
        print(f"RESULT {label}: FAILED (exit {proc.returncode})", flush=True)
        print("\n".join("    " + t for t in tail), flush=True)
        return None
    result["process_seconds"] = round(wall, 1)
    result["cpu_objects_written"] = _written(before, _objects(cache))
    print(f"RESULT {label}: {json.dumps(result)}", flush=True)
    return result


def _driver():
    from algan.settings._startup import render_device

    device = render_device().type
    root = Path(tempfile.mkdtemp(prefix="handoff-"))
    print(f"device={device} root={root}", flush=True)
    recorded = _run("1 record (cache A, cold)", "record", root / "A")
    if recorded is None:
        return 1
    manifest = ["--manifest", recorded["manifest"]]
    arms = [("new", root / "B", [])]
    if device == "mps":
        arms.append(("old", root / "C", ["--old-worker"]))
    for name, cache, flags in arms:
        _run(f"2 worker[{name}]", "worker", cache, extra_args=manifest + flags)
        _run(f"3 render after worker[{name}]", "render", cache)
    _run("4 render again (cache B, warm)", "render", root / "B")
    _run(
        "5 render again, ALGAN_CPU_OBJECT_CACHE=0",
        "render",
        root / "B",
        extra_env={"ALGAN_CPU_OBJECT_CACHE": "0"},
    )
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=sorted(_PHASES))
    parser.add_argument("--manifest")
    parser.add_argument("--old-worker", action="store_true")
    args = parser.parse_args()
    if args.phase:
        _child(args)
        return 0
    return _driver()


if __name__ == "__main__":
    sys.exit(main())
