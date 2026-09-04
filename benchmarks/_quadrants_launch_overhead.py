r"""How much Python a warm kernel launch pays above the C++ ``launch_kernel``.

`taichi_patches/PLAN.md` §4 row 10 and §7.3 step 3 ask whether the cached fast
launcher (`algan/utils/taichi_fast_launch.py`) is worth porting to Quadrants,
whose own launch-context cache (`LaunchContextBufferCache`) marks a raw torch
tensor non-cacheable and so never engages for Algan. This is the measurement
that answers it, on whichever compiler is live:

    ALGAN_TAICHI_BACKEND=quadrants uv run python benchmarks/_quadrants_launch_overhead.py
    ALGAN_TAICHI_BACKEND=taichi    uv run python benchmarks/_quadrants_launch_overhead.py

One process, warm cache, two parts:

* **micro** -- an Algan-shaped kernel (20 ``f32`` ndarray arguments, three
  ints, one ``ti.template()`` tuple, trivial body) launched 500 times, timed
  with the fast path off and on, alternating in-process (``set_enabled``, the
  way it was designed to be A/B'd -- wall clock across processes is thermally
  polluted on the dev box). The C++ half is timed separately through a
  delegating proxy on the runtime's ``Program`` (cProfile cannot see nanobind
  calls: they are not ``PyCFunction``\\ s, so their time lands in the caller's
  self time), which gives "Python above ``prog.launch_kernel``" as the outermost
  ``Kernel.__call__`` minus the C++ launch and compile calls. On Quadrants the
  floor a fast path could reach is also measured directly: a launch context
  built by hand with the same ``set_arg_*`` calls the original path makes, then
  ``launch_kernel``. A cProfile pass attributes the Python part among the
  stages ``Kernel.__call__`` goes through -- mapper lookup / extract,
  per-argument marshalling (``_recursive_set_args``), the launch method.
* **render** -- one warm ``Scene.save_frame`` of a ``Square`` (the scene
  `benchmarks/_taichi_warmstart_check.py` uses: ~22 kernels), timed the same
  way, reporting the launch count and the Python launch overhead as a share of
  the render's wall time; then alternated off/on for wall-clock seconds.

Run it twice: the first run pays cold compilation, which is not what this
measures. The fast path's engagement is printed from ``STATS`` -- on a compiler
where the dispatcher does not install, ``fast`` stays 0 and the on/off arms
are the same arm.
"""

from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import sys
import tempfile
import time
from pathlib import Path

# The daemon keeps renderer state across runs and re-executes the script,
# either of which makes the arms incomparable.
os.environ.setdefault("ALGAN_AUTO_DAEMON", "0")
os.environ.setdefault("ALGAN_USE_DAEMON", "0")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# A module global, not a parameter of the defining function: with postponed
# annotations the compiler evaluates the kernel's annotation strings against
# the module's globals.
from algan.taichi_compat import ti  # noqa: E402

N_ARRAYS = 20
N = 64

#: (label, exact function name, file basename) -- the Python stages of a
#: launch on both compilers; a row absent on one is printed as 0. Cumulative
#: times, so the rows nest rather than sum.
STAGES = (
    ("outermost Kernel.__call__ (arch guard)", "guarded_call", "taichi_runtime.py"),
    ("  fast-launch dispatcher", "_fast_call", "taichi_fast_launch.py"),
    ("  compiler Kernel.__call__ (quadrants)", "__call__", "kernel.py"),
    ("  compiler Kernel.__call__ (taichi)", "__call__", "kernel_impl.py"),
    ("    mapper.lookup (quadrants)", "lookup", "_template_mapper.py"),
    ("    mapper.lookup (taichi)", "lookup", "kernel_impl.py"),
    ("    mapper.extract (quadrants)", "extract", "_template_mapper.py"),
    ("    mapper.extract (taichi)", "extract", "kernel_impl.py"),
    ("    _recursive_set_args (quadrants)", "_recursive_set_args", "_func_base.py"),
    ("    Kernel.launch_kernel (quadrants, python)", "launch_kernel", "kernel.py"),
    ("    Kernel.launch_kernel (taichi, python)", "launch_kernel", "kernel_impl.py"),
)


class _TimingProgram:
    """Delegates to the real ``Program``; times its launch and compile calls."""

    def __init__(self, real, timer):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_timer", timer)

    def __getattr__(self, name):
        return getattr(self._real, name)

    def launch_kernel(self, *args):
        t0 = time.perf_counter()
        try:
            return self._real.launch_kernel(*args)
        finally:
            self._timer.launch += time.perf_counter() - t0
            self._timer.launch_calls += 1

    def compile_kernel(self, *args):
        t0 = time.perf_counter()
        try:
            return self._real.compile_kernel(*args)
        finally:
            self._timer.compile += time.perf_counter() - t0


class LaunchTimer:
    """Outermost ``Kernel.__call__`` seconds against the C++ calls under it.

    ``python`` is what a fast path could remove: everything between the
    engine calling a kernel and the runtime launching it, compile excluded.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.outer = 0.0
        self.launch = 0.0
        self.compile = 0.0
        self.calls = 0
        self.launch_calls = 0

    @property
    def python(self):
        return self.outer - self.launch - self.compile

    def install(self):
        from algan.taichi_compat import PROGRAM_ATTR, submodule

        self._kernel_cls = submodule("lang.kernel_impl").Kernel
        self._previous_call = self._kernel_cls.__call__
        timer = self
        previous = self._previous_call

        def timed_call(kernel, *args, **kwargs):
            t0 = time.perf_counter()
            try:
                return previous(kernel, *args, **kwargs)
            finally:
                timer.outer += time.perf_counter() - t0
                timer.calls += 1

        self._kernel_cls.__call__ = timed_call
        self._runtime = submodule("lang.impl").get_runtime()
        self._attr = PROGRAM_ATTR
        self._real_prog = getattr(self._runtime, PROGRAM_ATTR)
        setattr(self._runtime, PROGRAM_ATTR, _TimingProgram(self._real_prog, self))
        return self

    def uninstall(self):
        self._kernel_cls.__call__ = self._previous_call
        setattr(self._runtime, self._attr, self._real_prog)

    def report(self, wall, label):
        n = max(self.calls, 1)
        print(
            f"{label}: {self.calls} launches in {wall:.3f} s wall; "
            f"Kernel.__call__ {self.outer:.3f} s, of which C++ launch_kernel {self.launch:.3f} s "
            f"({self.launch_calls} calls), compile {self.compile:.3f} s; "
            f"Python above launch_kernel {self.python:.3f} s = {self.python / n * 1e6:.0f} us/launch, "
            f"{self.python / wall * 100:.1f}% of wall"
        )


def _define_kernel():
    """Spelled out rather than generated: the compiler reads the source file."""

    @ti.kernel
    def algan_shaped(
        a0: ti.types.ndarray(dtype=ti.f32, ndim=1),
        a1: ti.types.ndarray(dtype=ti.f32, ndim=1),
        a2: ti.types.ndarray(dtype=ti.f32, ndim=1),
        a3: ti.types.ndarray(dtype=ti.f32, ndim=1),
        a4: ti.types.ndarray(dtype=ti.f32, ndim=1),
        a5: ti.types.ndarray(dtype=ti.f32, ndim=1),
        a6: ti.types.ndarray(dtype=ti.f32, ndim=1),
        a7: ti.types.ndarray(dtype=ti.f32, ndim=1),
        a8: ti.types.ndarray(dtype=ti.f32, ndim=1),
        a9: ti.types.ndarray(dtype=ti.f32, ndim=1),
        a10: ti.types.ndarray(dtype=ti.f32, ndim=1),
        a11: ti.types.ndarray(dtype=ti.f32, ndim=1),
        a12: ti.types.ndarray(dtype=ti.f32, ndim=1),
        a13: ti.types.ndarray(dtype=ti.f32, ndim=1),
        a14: ti.types.ndarray(dtype=ti.f32, ndim=1),
        a15: ti.types.ndarray(dtype=ti.f32, ndim=1),
        a16: ti.types.ndarray(dtype=ti.f32, ndim=1),
        a17: ti.types.ndarray(dtype=ti.f32, ndim=1),
        a18: ti.types.ndarray(dtype=ti.f32, ndim=1),
        a19: ti.types.ndarray(dtype=ti.f32, ndim=1),
        n: ti.i32,
        flags: ti.i32,
        seed: ti.u32,
        pipeline: ti.template(),
    ):
        for i in range(n):
            acc = 0.0
            acc += a0[i]
            acc += a1[i]
            acc += a2[i]
            acc += a3[i]
            acc += a4[i]
            acc += a5[i]
            acc += a6[i]
            acc += a7[i]
            acc += a8[i]
            acc += a9[i]
            acc += a10[i]
            acc += a11[i]
            acc += a12[i]
            acc += a13[i]
            acc += a14[i]
            acc += a15[i]
            acc += a16[i]
            acc += a17[i]
            acc += a18[i]
            acc += a19[i]
            if ti.static(bool(pipeline[0])):
                acc += 1.0
            a0[i] = acc + flags + seed

    return algan_shaped


def _profile_rows(stats):
    rows = []
    for label, name, base in STAGES:
        total = 0.0
        calls = 0
        for (file, _line, fname), (_cc, nc, _tt, ct, _callers) in stats.stats.items():
            if fname == name and os.path.basename(file) == base:
                total += ct
                calls += nc
        rows.append((label, total, calls))
    return rows


def _print_profile(stats, launches, wall, title):
    print(
        f"\n--- cProfile attribution, {title} (cumulative; C++ time lands in its caller) ---"
    )
    print(f"{'stage':<48} {'us/launch':>10} {'calls':>8} {'share':>7}")
    for label, total, calls in _profile_rows(stats):
        us = total / launches * 1e6
        print(f"{label:<48} {us:>10.1f} {calls:>8} {total / wall * 100:>6.1f}%")


def _quadrants_floor(kernel, arrays, pipeline, launches):
    """Hand-built launch context + launch_kernel: what a fast hit would cost."""
    from algan.taichi_compat import submodule

    primal = kernel._primal
    key = primal._last_launch_key
    t_kernel = primal.materialized_kernels[key]
    compiled = primal.compiled_kernel_data_by_key[key]
    prog = submodule("lang.impl").get_runtime().prog
    ints = ([N_ARRAYS, N_ARRAYS + 1], [N, 3])
    uints = ([N_ARRAYS + 2], [7])

    def build_and_launch():
        ctx = t_kernel.make_launch_context()
        for i, v in enumerate(arrays):
            ctx.set_arg_external_array_with_shape(
                i, v.data_ptr(), v.element_size() * v.nelement(), v.shape, 0
            )
        ctx.set_args_int(*ints)
        ctx.set_args_uint(*uints)
        prog.launch_kernel(compiled, ctx)
        return ctx

    ctx = build_and_launch()
    t0 = time.perf_counter()
    for _ in range(launches):
        build_and_launch()
    per_build = (time.perf_counter() - t0) / launches * 1e6
    t0 = time.perf_counter()
    for _ in range(launches):
        prog.launch_kernel(compiled, ctx)
    per_launch = (time.perf_counter() - t0) / launches * 1e6
    print(
        f"fast-hit floor (quadrants): make_launch_context + {N_ARRAYS} set_arg_external_array_with_shape "
        f"+ batched ints + launch_kernel = {per_build:.1f} us; launch_kernel alone (context reused) = "
        f"{per_launch:.1f} us"
    )


def micro(torch, fast_launch, launches, repeats):
    from algan.taichi_compat import BACKEND

    print(
        f"\n=== micro: {N_ARRAYS} ndarray(f32) + 3 ints + template tuple, trivial body, N={N} ==="
    )
    kernel = _define_kernel()
    arrays = [torch.zeros(N, dtype=torch.float32) for _ in range(N_ARRAYS)]
    pipeline = (1, None, 2, "material", None)

    def launch():
        kernel(*arrays, N, 3, 7, pipeline)

    launch()  # compile
    launch()  # first warm launch records the plan on the fast path
    ti.sync()

    results = {"off": [], "on": []}
    for _ in range(repeats):
        for arm in ("off", "on"):
            fast_launch.set_enabled(arm == "on")
            launch()
            t0 = time.perf_counter()
            for _ in range(launches):
                launch()
            results[arm].append((time.perf_counter() - t0) / launches * 1e6)
    fast_launch.set_enabled(True)
    for arm in ("off", "on"):
        print(
            f"fast-launch {arm:<3}: {min(results[arm]):8.1f} us/launch "
            f"(best of {repeats}: {[round(t, 1) for t in results[arm]]})"
        )

    for arm in ("off", "on"):
        fast_launch.set_enabled(arm == "on")
        launch()
        timer = LaunchTimer().install()
        t0 = time.perf_counter()
        for _ in range(launches):
            launch()
        wall = time.perf_counter() - t0
        timer.uninstall()
        timer.report(wall, f"fast-launch {arm}")
    fast_launch.set_enabled(True)

    if BACKEND == "quadrants":
        _quadrants_floor(kernel, arrays, pipeline, launches)

    for arm in ("off", "on"):
        fast_launch.set_enabled(arm == "on")
        launch()
        profiler = cProfile.Profile()
        t0 = time.perf_counter()
        profiler.enable()
        for _ in range(launches):
            launch()
        profiler.disable()
        wall = time.perf_counter() - t0
        _print_profile(
            pstats.Stats(profiler),
            launches,
            wall,
            f"fast-launch {arm}, {wall / launches * 1e6:.0f} us/launch under the profiler",
        )
    fast_launch.set_enabled(True)
    print(f"STATS after micro: {fast_launch.STATS}")


def render(fast_launch, repeats):
    from algan import PREVIEW, SETTINGS, Scene, Square

    print("\n=== render: warm Scene.save_frame of a Square (PREVIEW) ===")
    SETTINGS.video.set(PREVIEW)
    tmp = tempfile.mkdtemp(prefix="algan-launch-overhead-")
    frame = os.path.join(tmp, "square.png")
    with Scene() as scene:
        Square().spawn()
        t0 = time.perf_counter()
        scene.save_frame(frame, video_settings=PREVIEW, overwrite=True)
        print(f"first (possibly cold) render: {time.perf_counter() - t0:.2f} s")

        for arm in ("off", "on"):
            fast_launch.set_enabled(arm == "on")
            before = dict(fast_launch.STATS)
            timer = LaunchTimer().install()
            t0 = time.perf_counter()
            scene.save_frame(frame, video_settings=PREVIEW, overwrite=True)
            wall = time.perf_counter() - t0
            timer.uninstall()
            timer.report(wall, f"render, fast-launch {arm}")
            delta = {k: fast_launch.STATS[k] - before[k] for k in before}
            print(f"  STATS delta this render: {delta}")

        fast_launch.set_enabled(False)
        profiler = cProfile.Profile()
        t0 = time.perf_counter()
        profiler.enable()
        scene.save_frame(frame, video_settings=PREVIEW, overwrite=True)
        profiler.disable()
        wall = time.perf_counter() - t0
        stats = pstats.Stats(profiler)
        launches = sum(
            nc
            for (file, _line, name), (
                _cc,
                nc,
                _tt,
                _ct,
                _callers,
            ) in stats.stats.items()
            if name == "guarded_call" and os.path.basename(file) == "taichi_runtime.py"
        )
        _print_profile(
            stats,
            max(launches, 1),
            wall,
            f"render with fast-launch off, {launches} launches, {wall:.2f} s under the profiler",
        )

        results = {"off": [], "on": []}
        for _ in range(repeats):
            for arm in ("off", "on"):
                fast_launch.set_enabled(arm == "on")
                t0 = time.perf_counter()
                scene.save_frame(frame, video_settings=PREVIEW, overwrite=True)
                results[arm].append(time.perf_counter() - t0)
        fast_launch.set_enabled(True)
        print("\n--- alternating wall clock, warm render ---")
        for arm in ("off", "on"):
            print(
                f"fast-launch {arm:<3}: best {min(results[arm]):.2f} s, {[round(t, 2) for t in results[arm]]}"
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launches", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--no-micro", action="store_true")
    args = parser.parse_args(argv)

    import torch

    import algan  # noqa: F401  -- installs the launch wrappers
    from algan.rendering.taichi_runtime import init_taichi
    from algan.taichi_compat import describe_backend
    from algan.utils import taichi_fast_launch

    init_taichi()
    print(
        f"backend: {describe_backend()}; fast-launch installed: {taichi_fast_launch._APPLIED}"
    )
    if hasattr(taichi_fast_launch, "skipped_reason"):
        print(f"fast-launch skipped reason: {taichi_fast_launch.skipped_reason()!r}")
    if not args.no_micro:
        micro(torch, taichi_fast_launch, args.launches, args.repeats)
    if not args.no_render:
        render(taichi_fast_launch, args.repeats)
    print(f"\nSTATS at exit: {taichi_fast_launch.STATS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
