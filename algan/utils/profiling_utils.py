"""Universal scene profiler for Algan's ray-tracing renderer.

:func:`profile_scene` is the single entry point -- wrap *any* scene function
with it to get an end-to-end breakdown of video-production time. It is meant to
be the one profiler reached for whenever a scene is being optimized:

    from algan.utils.profiling_utils import profile_scene
    profile_scene(my_scene_func, render_settings, tag="my_scene")

What it reports
---------------
* **Wall time per pipeline stage** -- geometry generation, vertex shading +
  packing, STBVH temporal segmentation + build, scene merge, background
  prefill, every Taichi kernel launch, sample finalization and post-processing
  -- via device-synced timers so GPU work is attributed to the stage that
  issued it.
* **Per-Taichi-kernel GPU time** from Taichi's built-in kernel profiler
  (precise GPU-only time with launch overhead excluded), when it can be
  enabled on the current runtime. This is the authoritative signal for kernel
  optimization; the wall-time numbers include Python launch + sync overhead.
* **NVIDIA GPU specs and live telemetry** via ``nvidia-smi``: static specs
  (name, driver, compute capability, memory, max clocks) plus SM/mem clocks,
  utilization, temperature and power sampled *during* the render, so thermal
  or power throttling -- a known source of cross-run variance on this project
  -- is visible. Also detects ``nvprof`` / ``ncu`` / ``nsys`` and, on request,
  drives ``nvprof`` to report the per-kernel register usage and achieved
  occupancy that Taichi cannot surface itself.
* **Merged-scene geometry + BVH sizes and peak GPU memory** (schema-agnostic:
  it walks the merged-scene dict, so new geometry arrays / BVHs show up with
  no changes here).
* **A cProfile dump** for python-side hotspots.

Self-updating kernel hooks
--------------------------
The set of Taichi kernels is **discovered automatically**: the whole
``algan.rendering.raytracing`` package is imported, then every ``@ti.kernel``
object reachable from an imported ``algan`` module is found and *every*
reference to it is wrapped -- including the copies imported into
``primitives.py``, which is where they are actually launched. Adding a new
kernel anywhere in the engine hooks it with no edits to this file (the
rasterizer / bloom kernels are picked up too, and simply stay absent from the
report when a scene never launches them).

Usage from a benchmark script::

    from algan.utils.profiling_utils import profile_scene
    enable_ray_tracing(...)
    profile_scene(scene_func, render_settings, tag)

Env knobs (all optional):
    ALGAN_TI_KERNEL_PROFILER=0   disable the Taichi kernel profiler re-init
    ALGAN_PROFILE_RUNS=N         number of render passes (default 2: cold+warm)
    ALGAN_PROFILE_TELEMETRY=0    disable the nvidia-smi live sampler
    ALGAN_PROFILE_NVPROF=1       auto-run nvprof for registers/occupancy
    ALGAN_UNDER_NVPROF=1         (set by the nvprof child) run one lean render
"""
import os
import cProfile
import pstats
import re
import subprocess
import sys
import threading
import time

from algan import KERNEL_SETTINGS
from collections import defaultdict
from contextlib import contextmanager

import taichi as ti
import torch

# ``import algan`` initializes the Taichi runtime (via the rasterizer modules)
# and pulls in the mob / scene classes the pipeline hooks below wrap.
import algan  # noqa: F401
from algan.constants.color import GREY
from algan.utils.algan_utils import render_to_file

# Optional pipeline-hook targets. Imported defensively: a rename upstream must
# degrade the hook, not break the whole profiler.
from algan.scene_manager import SceneManager
from algan.animation.animatable import Animatable
from algan.mobs.surfaces.surface import Surface
from algan.mobs.bezier_circuit import BezierCircuitCubic

import algan.rendering.raytracing.primitives as rtp
import algan.rendering.raytracing.stbvh as stbvh_mod
from algan.scene import Scene
import algan.mobs.bezier_circuit as bzc
import algan.rendering.raytracing.tracer as rtr

OUT_DIR = os.path.join("algan_outputs", "profiling")
REPORT_PATH = "algan_profile_report.txt"

# Set True once the Taichi kernel profiler has been successfully enabled.
KERNEL_PROFILER = False
# Samples per pixel; > 1 selects the Monte Carlo kernels. Set from profile_scene.
SPP = 1


# ---------------------------------------------------------------------------
# Device sync
# ---------------------------------------------------------------------------
def _sync_devices():
    # Stage boundaries may now run on the batch-prep worker thread (scene
    # prefetch). Syncing there would serialize against the render thread's
    # GPU work (misattributing it) and ti.sync() is not safe off the main
    # thread, so only sync from the main thread.
    if threading.current_thread() is not threading.main_thread():
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    try:
        if hasattr(torch, "mps") and torch.mps.is_available():
            torch.mps.synchronize()
    except Exception:
        pass
    ti.sync()


# ---------------------------------------------------------------------------
# Stage timing
# ---------------------------------------------------------------------------
class StageTimers:
    """Accumulates wall time per named stage, with device syncs at the
    boundaries so GPU work is attributed to the stage that issued it.
    Re-entrant stages (recursion) are only timed at the outermost level.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.times = defaultdict(float)
        self.exclusive_times = defaultdict(float)
        self.counts = defaultdict(int)
        self.cuda_sync_times = defaultdict(float)
        self.launch_times = defaultdict(float)
        # Stage nesting (stack level / re-entrancy) is tracked per thread:
        # with scene batch prefetch, prep stages run on a worker thread
        # concurrently with render stages on the main thread.
        self._tls = threading.local()
        # (kernel_name, num_frames, num_rays, seconds)
        self.kernel_launches = []

    def _thread_state(self):
        tls = self._tls
        if not hasattr(tls, "stack_level"):
            tls.stack_level = 0
            tls.level_times = defaultdict(float)
            tls.active = set()
        return tls

    @contextmanager
    def stage(self, name):
        tls = self._thread_state()
        if name in tls.active:
            yield
            return
        tls.active.add(name)
        tls.stack_level += 1
        _sync_devices()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            #t1 = time.perf_counter()# - t0
            #_sync_devices()
            torch.cuda.synchronize()
            #t2 = time.perf_counter()
            ti.sync()
            t3 = time.perf_counter()
            tls.stack_level -= 1
            t = t3 - t0
            tls.level_times[tls.stack_level] += t
            self.times[name] += t
            self.exclusive_times[name] += t - tls.level_times[tls.stack_level + 1]
            tls.level_times[tls.stack_level + 1] = 0
            self.counts[name] += 1
            #self.launch_times[name] += t1 - t0
            #self.cuda_sync_times[name] += t2 - t0
            tls.active.discard(name)

    def wrap_function(self, obj, attr, name):
        """Wrap ``obj.attr`` in a stage timer (idempotent)."""
        orig = getattr(obj, attr)
        if getattr(orig, "_profiling_original", None) is not None:
            return orig  # already wrapped

        def wrapped(*args, **kwargs):
            with self.stage(name):
                return orig(*args, **kwargs)

        wrapped._profiling_original = orig
        setattr(obj, attr, wrapped)
        return orig


TIMERS = StageTimers()
SCENE_STATS = {}
# Kernel names discovered + hooked (populated by install_kernel_hooks).
DISCOVERED_KERNELS = []
# (module, attr, original) triples for uninstall.
_KERNEL_HOOKS = []


def _tensor_mb(t):
    return t.numel() * t.element_size() / 2**20


# ---------------------------------------------------------------------------
# Automatic Taichi-kernel discovery + hooking
# ---------------------------------------------------------------------------
def _is_taichi_kernel(obj):
    """True for a module-level ``@ti.kernel`` (its decorator sets this flag)."""
    return bool(getattr(obj, "_is_wrapped_kernel", False)) and callable(obj)


def _is_taichi_func(obj):
    """Best-effort detection of a ``@ti.func`` (inlined; cannot be timed)."""
    return bool(getattr(obj, "_is_taichi_function", False))


def _import_raytracing_modules():
    """Import every submodule of ``algan.rendering.raytracing`` so their kernel
    objects exist to be discovered. New kernel files are picked up here."""
    import importlib
    import pkgutil

    pkg = importlib.import_module("algan.rendering.raytracing")
    for info in pkgutil.iter_modules(pkg.__path__):
        full = f"{pkg.__name__}.{info.name}"
        try:
            importlib.import_module(full)
        except Exception as e:  # pragma: no cover - keep discovering the rest
            print(f"[profiling] skip {full}: {e}")


def _iter_algan_module_dicts():
    """Yield (module, __dict__) for every currently-imported algan module."""
    for name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        if name == "algan" or name.startswith("algan."):
            d = getattr(mod, "__dict__", None)
            if d is not None:
                yield mod, d


def discover_taichi_kernels():
    """Find every module-level Taichi kernel reachable from algan modules.

    Returns ``{id(obj): (obj, name, [(module, attr), ...])}``. The reference
    list spans *all* algan modules so a kernel imported into ``primitives.py``
    (where it is actually launched) is wrapped there too, not just in the file
    that defines it.
    """
    _import_raytracing_modules()
    kernels = {}
    for mod, d in _iter_algan_module_dicts():
        for attr, val in list(d.items()):
            if _is_taichi_kernel(val):
                kid = id(val)
                if kid not in kernels:
                    kernels[kid] = (val, getattr(val, "__name__", attr), [])
                kernels[kid][2].append((mod, attr))
    return kernels


def count_taichi_funcs():
    """Count discoverable ``@ti.func`` objects (inlined; reported for context)."""
    seen = set()
    for _mod, d in _iter_algan_module_dicts():
        for val in d.values():
            if _is_taichi_func(val):
                seen.add(id(val))
    return len(seen)


def _rays_from_last_out(args, spp=1):
    """Deterministic kernels take the [frames, pixels, channels] output buffer
    as their last positional arg. Returns (frames, rays) or None."""
    if not args:
        return None
    out = args[-1]
    if torch.is_tensor(out) and out.dim() >= 2:
        frames = out.shape[0]
        return frames, frames * out.shape[1] * max(1, spp)
    return None


# Kernels whose per-launch ray throughput we can read from their output arg.
# Everything else is still timed (wall + GPU), just without a rays/s figure.
def _det_extractor(args, kwargs):
    return _rays_from_last_out(args, spp=1)


def _mc_extractor(args, kwargs):
    return _rays_from_last_out(args, spp=max(1, int(SPP)))


KERNEL_RAY_EXTRACTORS = {
    "render_scene_stbvh": _det_extractor,
    "render_triangles_stbvh": _det_extractor,
    "render_triangles_knots_stbvh": _det_extractor,
    "render_no_pn_stbvh": _det_extractor,
    "path_trace_scene_stbvh": _mc_extractor,
    "path_trace_physical_stbvh": _mc_extractor,
}


def _make_kernel_wrapper(orig, name):
    label = f"kernel: {name}"
    extractor = KERNEL_RAY_EXTRACTORS.get(name)

    def wrapper(*args, **kwargs):
        _sync_devices()
        t0 = time.perf_counter()
        result = orig(*args, **kwargs)
        #_sync_devices()
        t1 = time.perf_counter()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        ti.sync()
        t3 = time.perf_counter()
        dt = t3 - t0
        TIMERS.times[label] += dt
        TIMERS.counts[label] += 1
        TIMERS.launch_times[label] += t1-t0
        TIMERS.cuda_sync_times[label] += t2 - t1
        if extractor is not None:
            try:
                got = extractor(args, kwargs)
                if got is not None:
                    TIMERS.kernel_launches.append((name, got[0], got[1], dt))
            except Exception:
                pass
        return result

    wrapper._profiling_kernel_wrapper = True
    wrapper._profiling_original = orig
    try:
        wrapper.__name__ = orig.__name__
    except Exception:
        pass
    return wrapper


def install_kernel_hooks():
    """Discover and wrap every Taichi kernel in the ray-tracing package.

    Idempotent: already-wrapped references are left alone. Returns the sorted
    list of hooked kernel names.
    """
    global DISCOVERED_KERNELS
    kernels = discover_taichi_kernels()
    names = []
    for _kid, (obj, name, refs) in kernels.items():
        wrapper = _make_kernel_wrapper(obj, name)
        for mod, attr in refs:
            # Skip anything already pointing at a wrapper.
            if getattr(getattr(mod, attr, None), "_profiling_kernel_wrapper", False):
                continue
            setattr(mod, attr, wrapper)
            _KERNEL_HOOKS.append((mod, attr, obj))
        names.append(name)
    DISCOVERED_KERNELS = sorted(set(names))
    return DISCOVERED_KERNELS


def uninstall_kernel_hooks():
    """Restore original kernel references (best-effort)."""
    for mod, attr, orig in _KERNEL_HOOKS:
        try:
            if getattr(getattr(mod, attr, None), "_profiling_kernel_wrapper", False):
                setattr(mod, attr, orig)
        except Exception:
            pass
    _KERNEL_HOOKS.clear()


# ---------------------------------------------------------------------------
# Merged-scene stats (schema-agnostic)
# ---------------------------------------------------------------------------
def _looks_like_bvh(v):
    return hasattr(v, "num_nodes") and hasattr(v, "get_memory_used")


def _capture_scene_stats(scene):
    """Record sizes of the merged scene once per batch, without hardcoding its
    schema: tensors are grouped by MB, ``*_bvh`` values report node/leaf/
    instance counts, and scalar counts (num_*) are copied through."""
    tensors, scalars, bvhs = {}, {}, {}
    for k, v in scene.items():
        if torch.is_tensor(v):
            tensors[k] = (tuple(v.shape), _tensor_mb(v))
        elif _looks_like_bvh(v):
            inst = None
            if hasattr(v, "leaf_prim"):
                try:
                    inst = int((v.leaf_prim >= 0).sum())
                except Exception:
                    inst = None
            bvhs[k] = dict(nodes=getattr(v, "num_nodes", None),
                           leaves=getattr(v, "num_leaves", None),
                           instances=inst,
                           mb=v.get_memory_used() / 2**20)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            scalars[k] = v
    stats = dict(tensors=tensors, scalars=scalars, bvhs=bvhs)
    stats["total_tensor_mb"] = sum(mb for _s, mb in tensors.values())
    stats["total_bvh_mb"] = sum(b["mb"] for b in bvhs.values())
    stats["cuda_allocated_after_merge_mb"] = (
        torch.cuda.memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0)
    SCENE_STATS.setdefault("batches", []).append(stats)


# ---------------------------------------------------------------------------
# Pipeline stage hooks (guarded: a missing target degrades, never breaks)
# ---------------------------------------------------------------------------
def _try_wrap(obj, attr, label):
    #if obj is not None and hasattr(obj, attr):
    TIMERS.wrap_function(obj, attr, label)


def install_pipeline_hooks():
    """Wrap the (non-kernel) pipeline entry points with stage timers."""
    # Scene-side preparation (mob state evaluation + geometry generation).
    import algan.render_loop as rl
    _try_wrap(bzc, 'build_render_primitives_batched', 'build_render_primitives_batched')
    _try_wrap(Scene, "get_batch_of_primitives", "Scene.get_batch_of_primitives")
    _try_wrap(Animatable, "get_attr_inds", "get_attr_inds")
    from algan.animation.timeline import AnimationTimeline, AttributeTimeline
    _try_wrap(AttributeTimeline, "modify",
              "AttributeTimeline.modify")
    _try_wrap(AttributeTimeline, "get",
              "AttributeTimeline.get")
    _try_wrap(AttributeTimeline, "add",
              "AttributeTimeline.add")
    _try_wrap(AttributeTimeline, "rematerialize_state_at_times",
              "AttributeTimeline.rematerialize_state_at_times")
    _try_wrap(AnimationTimeline, "set_state_to_times",
              "AnimationTimeline.set_state_to_times")
    _try_wrap(Surface, "get_render_primitives", "Surface.get_render_primitives")
    _try_wrap(BezierCircuitCubic, "get_render_primitives",
              "BezierCircuitCubic.get_render_primitives")

    # Geometry shading + packing.
    _try_wrap(rtp.RayTracedTrianglePrimitive, "project_to_screen",
              "triangles: shade + pack (project_to_screen)")
    _try_wrap(rtp.RayTracedBezierCircuitPrimitive, "project_to_screen",
              "beziers: sample + pack (project_to_screen)")
    for sub in ("_compute_samples_per_segment", "_build_circuit_geometry",
                "_build_frame_bounds"):
        _try_wrap(rtp.RayTracedBezierCircuitPrimitive, sub, f"beziers:   - {sub}")
    if hasattr(rtp, "RayTracedPNTrianglePrimitive"):
        _try_wrap(rtp.RayTracedPNTrianglePrimitive, "project_to_screen",
                  "PN triangles: shade + pack (project_to_screen)")

    # Scene merge + BVH builds. ``_merge_scene`` is timed by hand so the merged
    # scene can be captured on the batch's first (uncached) merge. NOTE:
    # by-value imports mean the reference that is actually *called* lives in
    # the importing module -- ``tracer`` calls its own ``_merge_scene`` /
    # ``_prefill_background`` / ``post_process_frames`` names and
    # ``scene_builder`` calls its own ``build_stbvh`` -- so each name is
    # wrapped in every module that holds a reference (the same strategy the
    # kernel hooks use). Wrapping only the defining module silently times
    # nothing and the cost hides in "ray traced render total excl".
    import algan.rendering.raytracing.scene_builder as scb
    orig_merge = getattr(scb._merge_scene, "_profiling_original", None) \
        or scb._merge_scene
    if getattr(scb._merge_scene, "_profiling_original", None) is None:
        def merge_wrapper(primitives):
            had_cache = getattr(primitives[0], "_rt_merged_scene", None) is not None
            with TIMERS.stage("merge collections + build BVHs"):
                scene = orig_merge(primitives)
            if not had_cache:
                try:
                    _capture_scene_stats(scene)
                except Exception as e:
                    print(f"[profiling] scene-stats capture failed: {e}")
            return scene

        merge_wrapper._profiling_original = orig_merge
        for mod in (scb, rtr, rtp):
            if getattr(mod, "_merge_scene", None) is orig_merge:
                mod._merge_scene = merge_wrapper

    _try_wrap(stbvh_mod, "build_stbvh", "  - STBVH build (in merge)")
    _try_wrap(scb, "build_stbvh", "  - STBVH build (in merge)")
    _try_wrap(stbvh_mod, "segment_primitives_in_time",
              "  - STBVH temporal segmentation")

    # Render-chunk internals (again: wrap the refs tracer actually calls).
    _try_wrap(rtr, "_prefill_background", "background prefill")
    _try_wrap(rtr, "post_process_frames",
              "post-process (downsample/FXAA/glow)")
    #_try_wrap(rtr, "_compact_active_rays", "wavefront: compact active rays")
    _try_wrap(KERNEL_SETTINGS, "render_kernel", "ray traced render total")

    # Previously-unaccounted wall time: the per-batch memory reclaim
    # (gc.collect + cuda cache release; gc dominates -- see empty_cache) and
    # the serial video-encode tail (waiting on ffmpeg after the last frame).
    # Wrap the reference render_loop actually calls (by-value import), not just
    # the defining module, so the stage is not silently empty.
    from algan.render_loop import RenderLoopMixin
    _try_wrap(rl, "empty_cache", "memory reclaim (gc + cuda cache)")
    _try_wrap(RenderLoopMixin, "_drain_video_writer",
              "video encode tail (ffmpeg drain)")


def install_instrumentation():
    """Install every hook: pipeline stages + all discovered Taichi kernels.

    Backwards-compatible name; safe to call more than once."""
    install_pipeline_hooks()
    names = install_kernel_hooks()
    print(f"[profiling] hooked {len(names)} Taichi kernels "
          f"({count_taichi_funcs()} ti.funcs are inlined and not separately timed)")
    return names


# ---------------------------------------------------------------------------
# Taichi kernel profiler (precise per-kernel GPU time)
# ---------------------------------------------------------------------------
def enable_taichi_kernel_profiler():
    """Re-init Taichi with ``kernel_profiler=True`` using the *production*
    runtime config, so kernel GPU times are measured against the real config,
    with no environment variable required.

    This re-initializes the Taichi runtime, which destroys every previously
    allocated field. That is safe here because the engine's only module-level
    Taichi field is allocated lazily (``ray_trace_taichi._ensure_globals`` /
    ``_reset_globals``) on the first render -- i.e. *after* this re-init. Must
    run before any ray-trace kernel is launched (kernels compile lazily, so
    calling it at the top of ``profile_scene`` is fine). Returns True on
    success; on any failure the profiler falls back to wall-time only."""
    global KERNEL_PROFILER
    try:
        from algan.rendering.taichi_runtime import taichi_init_kwargs
        ti.init(**taichi_init_kwargs(), kernel_profiler=True)
        # Drop the lazily-allocated global field(s) so they are rebuilt against
        # this fresh runtime on the first render rather than dangling.
        try:
            import algan.rendering.raytracing.ray_trace_taichi as _rtt
            if hasattr(_rtt, "_reset_globals"):
                _rtt._reset_globals()
        except Exception:
            pass
        KERNEL_PROFILER = True
    except Exception as e:  # pragma: no cover
        print(f"[profiling] Taichi kernel profiler unavailable ({e}); "
              f"using wall-time only.")
        KERNEL_PROFILER = False
    return KERNEL_PROFILER


# Mangled Taichi kernel names look like ``<pyname>_c<NN>_<NN>_kernel_<N>_<tag>``;
# recover the python kernel name so per-launch sub-kernels aggregate correctly
# (and prefix collisions like wf_composite vs wf_composite_aa stay distinct --
# ``query_kernel_profiler_info`` gets this wrong).
_MANGLED_KERNEL = re.compile(r"^(.+?)_c\d+_\d+_kernel_")


def _collect_taichi_kernel_gpu():
    """Build a per-(python-)kernel GPU-time table from the Taichi kernel
    profiler's raw records (summing each kernel's serial + range-for
    sub-kernels). Returns dicts sorted by total GPU time, descending."""
    if not KERNEL_PROFILER:
        return []
    try:
        from taichi.profiler.kernel_profiler import get_default_kernel_profiler
        kp = get_default_kernel_profiler()
        kp._update_records()
        records = list(kp._traced_records)
    except Exception:
        return []
    agg = {}
    for rec in records:
        m = _MANGLED_KERNEL.match(rec.name)
        name = m.group(1) if m else rec.name
        # Drop Taichi-internal kernels (runtime bootstrap, field snode
        # readers/writers, JIT evaluators) -- only engine kernels are of interest.
        if name.startswith(("runtime_", "snode_", "jit_", "ext_arr", "matrix_to_")):
            continue
        a = agg.setdefault(name, [0, 0.0, float("inf"), 0.0])
        a[0] += 1
        a[1] += rec.kernel_time
        a[2] = min(a[2], rec.kernel_time)
        a[3] = max(a[3], rec.kernel_time)
    rows = [dict(name=n, records=c, total_ms=t, avg_ms=(t / c if c else 0.0),
                 min_ms=mn, max_ms=mx)
            for n, (c, t, mn, mx) in agg.items()]
    rows.sort(key=lambda r: -r["total_ms"])
    return rows


# ---------------------------------------------------------------------------
# NVIDIA GPU specs + live telemetry (nvidia-smi)
# ---------------------------------------------------------------------------
def _which(cmd):
    from shutil import which
    return which(cmd)


def detect_profiling_tools():
    return {t: _which(t) for t in ("nvidia-smi", "nvprof", "ncu", "nsys")}


def query_gpu_static():
    """One-shot static GPU specs via nvidia-smi (None if unavailable)."""
    if not _which("nvidia-smi"):
        return None
    fields = ("name,driver_version,compute_cap,memory.total,"
              "clocks.max.sm,clocks.max.mem,pcie.link.gen.max,pcie.link.width.max")
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        line = out.stdout.strip().splitlines()[0]
        vals = [v.strip() for v in line.split(",")]
        keys = ("name", "driver", "compute_cap", "memory_total",
                "max_sm_clock", "max_mem_clock", "pcie_gen_max", "pcie_width_max")
        return dict(zip(keys, vals))
    except Exception:
        return None


_THROTTLE_BITS = {
    0x1: "GpuIdle", 0x2: "AppClocks", 0x4: "SwPowerCap", 0x8: "HwSlowdown",
    0x10: "SyncBoost", 0x20: "SwThermal", 0x40: "HwThermal",
    0x80: "HwPowerBrake", 0x100: "DisplayClock",
}


def _decode_throttle(mask):
    if not mask:
        return set()
    return {name for bit, name in _THROTTLE_BITS.items() if mask & bit}


class GpuTelemetrySampler:
    """Streams ``nvidia-smi`` telemetry in the background and summarizes it.

    Captures SM/mem clocks, utilization, temperature, power and active
    clock-throttle reasons at ~10 Hz. The summary exposes clock ranges and
    observed throttle reasons so throttling (a large source of run-to-run
    variance on this project) is not silently missed."""

    _FIELDS = ("utilization.gpu,utilization.memory,clocks.sm,clocks.mem,"
               "temperature.gpu,power.draw,clocks_throttle_reasons.active")

    def __init__(self, interval_ms=100):
        self.interval_ms = interval_ms
        self.proc = None
        self.thread = None
        self.samples = []  # (util, memutil, sm, mem, temp, power)
        self.throttles = set()
        self.available = bool(_which("nvidia-smi"))

    def _reader(self):
        for raw in self.proc.stdout:
            parts = [p.strip() for p in raw.split(",")]
            if len(parts) < 7:
                continue

            def num(x):
                try:
                    return float(x)
                except Exception:
                    return None

            util, memutil, sm, mem, temp, power = (num(parts[i]) for i in range(6))
            mask = 0
            try:
                mask = int(parts[6], 16) if parts[6].lower().startswith("0x") else int(parts[6])
            except Exception:
                mask = 0
            self.throttles |= _decode_throttle(mask)
            self.samples.append((util, memutil, sm, mem, temp, power))

    def start(self):
        if not self.available:
            return self
        try:
            self.proc = subprocess.Popen(
                ["nvidia-smi", f"--query-gpu={self._FIELDS}",
                 "--format=csv,noheader,nounits", f"-lms={self.interval_ms}"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            self.thread = threading.Thread(target=self._reader, daemon=True)
            self.thread.start()
        except Exception:
            self.available = False
        return self

    def stop(self):
        if self.proc is not None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=3)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        if self.thread is not None:
            self.thread.join(timeout=2)

    def summary(self):
        if not self.samples:
            return None

        def col(i):
            return [s[i] for s in self.samples if s[i] is not None]

        def stat(vals):
            return (min(vals), sum(vals) / len(vals), max(vals)) if vals else None

        return dict(
            n=len(self.samples),
            util=stat(col(0)), mem_util=stat(col(1)),
            sm_clock=stat(col(2)), mem_clock=stat(col(3)),
            temp=stat(col(4)), power=stat(col(5)),
            throttles=sorted(t for t in self.throttles if t != "GpuIdle"),
        )


# ---------------------------------------------------------------------------
# nvprof driver (registers / occupancy that Taichi cannot surface)
# ---------------------------------------------------------------------------
def nvprof_command_hint(script_argv):
    """A copy-pasteable command to get per-kernel registers + occupancy.

    The child render is kept lean by ``ALGAN_UNDER_NVPROF=1`` (see
    :func:`under_nvprof`)."""
    py = sys.executable
    script = " ".join(script_argv) if script_argv else "<your_benchmark_script.py>"
    return (
        "  # registers per thread (no admin needed):\n"
        f"  ALGAN_UNDER_NVPROF=1 nvprof --print-gpu-trace --csv {py} {script}\n"
        "  # achieved occupancy (may need admin on Pascal):\n"
        f"  ALGAN_UNDER_NVPROF=1 nvprof --metrics achieved_occupancy --csv {py} {script}")


def under_nvprof():
    """True when running as the child of an nvprof launch (see profile_scene)."""
    return os.environ.get("ALGAN_UNDER_NVPROF") == "1"


def _parse_nvprof_csv(text):
    """Extract per-kernel {regs, occupancy} from nvprof --csv output on stderr.

    Tolerant to the two shapes we drive: ``--print-gpu-trace`` (has a
    'Registers Per Thread' column) and ``--metrics achieved_occupancy``
    (has 'Kernel' + 'Avg' columns)."""
    import csv
    import io

    out = {}
    # nvprof prints a preamble ('==PID== ...') before the CSV header; find it.
    lines = [ln for ln in text.splitlines() if not ln.startswith("==")]
    if not lines:
        return out
    # Header row is the first line containing quoted comma-separated fields.
    hdr_idx = next((i for i, ln in enumerate(lines) if '"' in ln and "," in ln), None)
    if hdr_idx is None:
        return out
    reader = csv.reader(io.StringIO("\n".join(lines[hdr_idx:])))
    rows = list(reader)
    if len(rows) < 2:
        return out
    header = rows[0]

    def find(col):
        for i, h in enumerate(header):
            if col.lower() in h.lower():
                return i
        return None

    name_i = find("Name") if find("Name") is not None else find("Kernel")
    reg_i = find("Registers Per Thread")
    occ_i = find("achieved_occupancy") or find("Avg")
    for row in rows[1:]:
        if not row or name_i is None or name_i >= len(row):
            continue
        name = row[name_i].strip().strip('"')
        if not name or name.lower() in ("kernel", "name"):
            continue
        rec = out.setdefault(name, {})
        if reg_i is not None and reg_i < len(row):
            try:
                rec["regs"] = int(float(row[reg_i]))
            except Exception:
                pass
        if occ_i is not None and occ_i < len(row):
            try:
                rec["occupancy"] = float(row[occ_i])
            except Exception:
                pass
    return out


def run_nvprof_metrics(script_argv, timeout=1200):
    """Re-run the current benchmark under nvprof to collect registers and
    achieved occupancy per kernel. Returns a dict name->{regs, occupancy} or
    None. Opt-in (slow: it renders again) and best-effort."""
    if not _which("nvprof") or under_nvprof() or not script_argv:
        return None
    env = dict(os.environ, ALGAN_UNDER_NVPROF="1")
    py = sys.executable
    merged = {}
    for extra in (["--print-gpu-trace", "--csv"],
                  ["--metrics", "achieved_occupancy", "--csv"]):
        try:
            res = subprocess.run(["nvprof", *extra, py, *script_argv],
                                 capture_output=True, text=True, timeout=timeout,
                                 env=env)
        except Exception as e:
            print(f"[profiling] nvprof run failed: {e}")
            continue
        parsed = _parse_nvprof_csv(res.stderr)
        for name, rec in parsed.items():
            merged.setdefault(name, {}).update(rec)
    return merged or None


# ---------------------------------------------------------------------------
# A single profiled render pass
# ---------------------------------------------------------------------------
def run_once(scene_func, settings, tag="", run_index=0, telemetry=True):
    TIMERS.reset()
    SCENE_STATS.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    if KERNEL_PROFILER:
        try:
            ti.profiler.clear_kernel_profiler_info()
        except Exception:
            pass

    scene = SceneManager.reset()
    scene.set_render_settings(settings)
    scene_func()

    sampler = GpuTelemetrySampler().start() if telemetry else None
    # cProfile is opt-in (ALGAN_PROFILE_CPROFILE=1): its per-call overhead
    # inflates the python-side numbers, so the default report is wall-clean.
    use_cprofile = os.environ.get("ALGAN_PROFILE_CPROFILE", "0") == "1"
    profiler = cProfile.Profile() if use_cprofile else None
    _sync_devices()
    t0 = time.perf_counter()
    if profiler is not None:
        profiler.enable()
    render_to_file(file_name=f"profiling{tag}_run{run_index}", output_dir=OUT_DIR,
                   output_path="", render_settings=settings,
                   file_extension="mp4")
    if profiler is not None:
        profiler.disable()
    _sync_devices()
    total = time.perf_counter() - t0
    if sampler is not None:
        sampler.stop()

    dump_path = "disabled (set ALGAN_PROFILE_CPROFILE=1)"
    if profiler is not None:
        dump_path = os.path.join(OUT_DIR,
                                 f"raytracing_cprofile{tag}_run{run_index}.txt")
        try:
            with open(dump_path, "w") as f:
                pstats.Stats(profiler, stream=f).sort_stats(
                    pstats.SortKey.CUMULATIVE).print_stats()
        except Exception as e:
            print(f"[profiling] could not write cProfile dump: {e}")

    peak_alloc = (torch.cuda.max_memory_allocated() / 2**20
                  if torch.cuda.is_available() else 0.0)
    peak_reserved = (torch.cuda.max_memory_reserved() / 2**20
                     if torch.cuda.is_available() else 0.0)
    return dict(total=total, peak_alloc_mb=peak_alloc,
                peak_reserved_mb=peak_reserved,
                times=dict(TIMERS.times), counts=dict(TIMERS.counts),
                exclusive_times=dict(TIMERS.exclusive_times),
                launches=list(TIMERS.kernel_launches),
                scene_stats=[dict(b) for b in SCENE_STATS.get("batches", [])],
                kernel_gpu=_collect_taichi_kernel_gpu(),
                telemetry=sampler.summary() if sampler is not None else None,
                cprofile_path=dump_path,
                launch_times=dict(TIMERS.launch_times),
                cuda_sync_times=dict(TIMERS.cuda_sync_times),
                )


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------
def _fmt_clock_stat(s, unit="MHz"):
    return f"{s[0]:.0f}/{s[1]:.0f}/{s[2]:.0f} {unit}" if s else "n/a"


def format_report(results, static_specs=None, tools=None, nvprof=None):
    lines = []
    w = lines.append
    w("=" * 78)
    w("Algan ray-tracing scene profile")
    w("=" * 78)
    dev = (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    w(f"device: {dev}")
    if static_specs:
        w(f"  driver {static_specs.get('driver')}  "
          f"compute cap {static_specs.get('compute_cap')}  "
          f"{static_specs.get('memory_total')}  "
          f"max SM {static_specs.get('max_sm_clock')} / "
          f"mem {static_specs.get('max_mem_clock')}  "
          f"PCIe gen{static_specs.get('pcie_gen_max')} x{static_specs.get('pcie_width_max')}")
    if tools:
        avail = ", ".join(f"{k}={'yes' if v else 'no'}" for k, v in tools.items())
        w(f"  profiling tools: {avail}")
    w(f"  Taichi kernel profiler: {'ON' if KERNEL_PROFILER else 'off (wall-time only)'}")
    if DISCOVERED_KERNELS:
        w(f"  hooked kernels ({len(DISCOVERED_KERNELS)}): "
          + ", ".join(DISCOVERED_KERNELS))

    for i, res in enumerate(results, 1):
        w("")
        w("-" * 78)
        label = "cold (includes Taichi JIT compile)" if i == 1 else "warm (steady state)"
        w(f"RUN {i} ({label}): end-to-end {res['total']:.2f}s")
        w("-" * 78)
        w(f"incl is wall time, excl is time spent in function excluding sub-processes of another tracked stage")
        w(f"{'stage':<52}{'calls':>6}{'incl (s)':>10}{'incl (%)':>9}{'excl (s)':>10}{'excl (%)':>10}"
          f"{'launch':10}{'sync':10}")
        for k in res["times"]:
            if k not in res["exclusive_times"]:
                res["exclusive_times"][k] = res["times"][k]

        kp = 'kernel: '
        res["exclusive_times"]["ray traced render total"] -= sum([v for k, v in res["exclusive_times"].items()
                                                                  if k[:len(kp)] == kp])

        for name, secs in sorted(res["times"].items(), key=lambda kv: -kv[1]):
            lt = res['launch_times'][name] if name in res['launch_times'] else 0
            ct = res['cuda_sync_times'][name] if name in res['cuda_sync_times'] else 0
            excl = res["exclusive_times"].get(name, secs)
            w(f"{name:<52}{res['counts'][name]:>6}{secs:>10.3f}"
              f"{100 * secs / res['total']:>8.1f}%{excl:>10.3f}"
              f"{100 * excl / res['total']:>8.1f}%"
              f"{lt:>10.3f}{ct:>10.3f}")
        # Sum *exclusive* times so nested stages aren't double-counted (e.g.
        # Surface.get_render_primitives runs inside Scene.get_batch_of_primitives;
        # kernels run inside "ray traced render total"). Kernels bypass the stack
        # machinery, so their time is already inside the render stage's exclusive
        # time -- give them 0 here (``.get(k, 0.0)``) to avoid double-counting.

        accounted = sum(res["exclusive_times"].get(k, 0.0) for k in res["times"])# if k[:len(kp)] != kp)
        unaccounted = res["total"] - accounted
        w(f"{'(unaccounted: video encode, scene mgmt, ...)':<52}{'':>6}"
          f"{unaccounted:>10.3f}{100 * unaccounted / res['total']:>8.1f}%")

        # Precise per-kernel GPU time from the Taichi profiler. ``% run`` is the
        # kernel's GPU time as a fraction of the end-to-end run wall time (not of
        # total GPU-kernel time), so it is comparable to the stage %'s above.
        if res.get("kernel_gpu"):
            run_ms = res["total"] * 1000.0
            w("")
            w("Taichi kernel GPU time (profiler; launch overhead excluded; "
              "'recs' = serial + range-for sub-kernels):")
            w(f"  {'kernel':<40}{'recs':>6}{'total ms':>11}{'% run':>8}"
              f"{'avg ms':>10}{'max ms':>10}")
            for r in res["kernel_gpu"]:
                pct = 100 * r["total_ms"] / run_ms if run_ms else 0.0
                w(f"  {r['name']:<40}{r['records']:>6}{r['total_ms']:>11.3f}"
                  f"{pct:>7.1f}%{r['avg_ms']:>10.4f}{r['max_ms']:>10.4f}")

        # Ray throughput for the kernels we can size.
        if res["launches"]:
            w("")
            w("trace kernel launches (wall time, incl. launch + sync overhead):")
            for j, (kname, frames, rays, dt) in enumerate(res["launches"]):
                note = " (incl. JIT compile)" if (i == 1 and j == 0) else ""
                w(f"  {kname}: {frames:>4} frames, {rays / 1e6:7.2f} M rays in "
                  f"{dt:7.3f}s -> {rays / dt / 1e6:8.2f} M rays/s{note}")

        # Live GPU telemetry -> throttling visibility.
        tele = res.get("telemetry")
        if tele:
            w("")
            w(f"GPU telemetry over render ({tele['n']} samples @ ~10 Hz, min/avg/max):")
            w(f"  utilization  {_fmt_clock_stat(tele['util'], '%')}   "
              f"mem-util {_fmt_clock_stat(tele['mem_util'], '%')}")
            w(f"  SM clock     {_fmt_clock_stat(tele['sm_clock'])}   "
              f"mem clock {_fmt_clock_stat(tele['mem_clock'])}")
            if tele["temp"]:
                w(f"  temperature  {_fmt_clock_stat(tele['temp'], 'C')}   "
                  f"power {_fmt_clock_stat(tele['power'], 'W')}")
            if tele["throttles"]:
                w(f"  ** THROTTLING observed: {', '.join(tele['throttles'])} "
                  f"(run-to-run timing is unreliable) **")
            else:
                w("  no clock throttling observed")

        # Memory + scene geometry.
        w("")
        w(f"GPU memory: peak allocated {res['peak_alloc_mb']:.0f} MB, "
          f"peak reserved {res['peak_reserved_mb']:.0f} MB")
        for k, st in enumerate(res["scene_stats"]):
            w(f"  batch {k}: merged tensors {st['total_tensor_mb']:.1f} MB, "
              f"BVHs {st['total_bvh_mb']:.1f} MB "
              f"(cuda after merge {st['cuda_allocated_after_merge_mb']:.0f} MB)")
            counts = ", ".join(f"{key}={val}" for key, val in st["scalars"].items()
                               if key.startswith("num_"))
            if counts:
                w(f"    counts: {counts}")
            for bname, b in st["bvhs"].items():
                w(f"    {bname}: {b['instances']} instances, {b['nodes']} nodes, "
                  f"{b['leaves']} leaves, {b['mb']:.1f} MB")
            # Largest few tensors.
            big = sorted(st["tensors"].items(), key=lambda kv: -kv[1][1])[:6]
            for tname, (shape, mb) in big:
                if mb >= 0.05:
                    w(f"    {tname}: {shape} = {mb:.1f} MB")
        w(f"  cProfile dump: {res['cprofile_path']}")

    # Registers / occupancy from nvprof (or a hint if not run).
    w("")
    w("-" * 78)
    if nvprof:
        w("Per-kernel registers / achieved occupancy (nvprof):")
        w(f"  {'kernel':<48}{'regs':>6}{'occupancy':>12}")
        for name, rec in sorted(nvprof.items()):
            regs = rec.get("regs", "-")
            occ = rec.get("occupancy")
            occ_s = f"{occ:.3f}" if isinstance(occ, float) else "-"
            w(f"  {name[:48]:<48}{str(regs):>6}{occ_s:>12}")
    elif tools and tools.get("nvprof"):
        w("Registers / occupancy not collected in-process (Taichi CUPTI toolkit "
          "unavailable here).")
        w("Run this to get them via nvprof:")
    w("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The one universal entry point
# ---------------------------------------------------------------------------
def profile_scene(scene_func, render_settings, tag="", runs=None,
                  kernel_profiler=None, telemetry=None, nvprof=None,
                  samples_per_pixel=1):
    """Profile ``scene_func`` end-to-end and write a report.

    This is the single profiler to use when optimizing video-production time.

    Parameters
    ----------
    scene_func : callable
        Builds the scene (spawns mobs, issues animations). Called after a fresh
        ``SceneManager.reset()`` each run.
    render_settings : RenderSettings
        Passed straight to ``render_to_file`` (e.g. ``HD``).
    tag : str
        Suffix for the output mp4 / report / cProfile files.
    runs : int
        Render passes. Default 2 (env ``ALGAN_PROFILE_RUNS``): run 1 is cold
        (Taichi JIT + cold GPU clocks), run 2 is warm/steady-state -- use the
        warm numbers for optimization decisions.
    kernel_profiler : bool | None
        Enable Taichi's per-kernel GPU profiler (re-inits the runtime). Default
        auto (env ``ALGAN_TI_KERNEL_PROFILER``, on).
    telemetry : bool | None
        Sample nvidia-smi telemetry during the render. Default auto (env
        ``ALGAN_PROFILE_TELEMETRY``, on).
    nvprof : bool | None
        Re-run the script under nvprof for registers/occupancy (slow, opt-in).
        Default auto (env ``ALGAN_PROFILE_NVPROF``, off).
    samples_per_pixel : int
        Reported for ray-throughput sizing on the Monte Carlo kernels.
    """
    global SPP
    SPP = max(1, int(samples_per_pixel))

    # When launched as an nvprof child, do a single lean render (no re-init, no
    # telemetry, no nested nvprof) so nvprof profiles clean kernels.
    if under_nvprof():
        install_instrumentation()
        os.makedirs(OUT_DIR, exist_ok=True)
        run_once(scene_func, render_settings, tag, 0, telemetry=False)
        return

    if runs is None:
        runs = int(os.environ.get("ALGAN_PROFILE_RUNS", "2"))
    if kernel_profiler is None:
        kernel_profiler = os.environ.get("ALGAN_TI_KERNEL_PROFILER", "1") == "1"
    if telemetry is None:
        telemetry = os.environ.get("ALGAN_PROFILE_TELEMETRY", "1") == "1"
    if nvprof is None:
        nvprof = os.environ.get("ALGAN_PROFILE_NVPROF", "0") == "1"

    if kernel_profiler:
        enable_taichi_kernel_profiler()

    install_instrumentation()
    os.makedirs(OUT_DIR, exist_ok=True)

    tools = detect_profiling_tools()
    static_specs = query_gpu_static()

    results = []
    for i in range(1, runs + 1):
        print(f"\n===== profiling run {i}/{runs} ({'cold' if i == 1 else 'warm'}) =====")
        results.append(run_once(scene_func, render_settings, tag, i,
                                telemetry=telemetry))

    nvprof_results = None
    if nvprof:
        print("\n===== nvprof pass (registers / occupancy) =====")
        nvprof_results = run_nvprof_metrics(sys.argv)

    report = format_report(results, static_specs=static_specs, tools=tools,
                           nvprof=nvprof_results)
    print("\n" + report)
    if not nvprof_results and tools.get("nvprof"):
        print(nvprof_command_hint(sys.argv))
    if KERNEL_PROFILER:
        try:
            print("\n(Taichi's own table below; its % is of total GPU-kernel "
                  "time, not of run wall time -- see '% run' above for that.)")
            ti.profiler.print_kernel_profiler_info()  # full table incl. sub-kernels
        except Exception as e:
            print(f"(taichi kernel profiler info unavailable: {e})")

    report_path = REPORT_PATH.replace(".txt", f"_{tag}.txt")
    try:
        with open(report_path, "w") as f:
            f.write(report)
        print(f"report written to {report_path}")
    except Exception as e:
        print(f"[profiling] could not write report: {e}")
    return results
