"""Profile the deterministic (1 sample per pixel) ray traced renderer.

Renders a scene stressing both geometry types of the ray tracing backend --
many triangles (a grid of animated spheres plus stacked translucent quads)
and many bezier circuits (a paragraph of text plus a ring of orbiting,
semi-transparent circles) -- and reports where the compute time and GPU
memory go:

* wall time per pipeline stage (vertex shading + packing, STBVH builds,
  scene merge, background prefill, the trace kernel itself, post-processing)
  via monkeypatched timers around the ray tracing module's entry points;
* trace-kernel throughput in rays/second per launch (the first launch of
  run 1 includes Taichi JIT compilation; run 2 is steady state);
* GPU memory: peak torch allocation, the packed scene arrays' sizes, STBVH
  node counts/instance counts/sizes, and the per-chunk output buffer;
* a cProfile dump of the whole render for python-side hotspots.

Usage (from the repo root):

    .venv/Scripts/python.exe benchmarks/raytracing_profiling.py [--quick]

Outputs land next to this file: ``raytracing_profile_report.txt``,
``raytracing_cprofile_run<N>.txt`` and the rendered mp4 under
``algan_outputs/raytracing_profiling/``.
"""
import argparse
import cProfile
import io
import os
import pstats
import sys
import time
from collections import defaultdict
from contextlib import contextmanager

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import taichi as ti
import torch

import algan  # noqa: F401  (initializes taichi via the rasterizer modules)
from algan import SceneManager

from algan.constants.color import BLUE, GREEN, GREY, ORANGE, PURPLE, RED, WHITE, YELLOW
from algan.settings.render_settings import RenderSettings
from algan.utils.algan_utils import render_to_file

# Re-initialize Taichi with the kernel profiler before any kernel launches
# (the rasterizer modules already initialized it without profiling at import
# time; no kernels have been compiled for launch yet, so this is safe).
KERNEL_PROFILER = False
try:
    ti.init(arch=ti.cuda, kernel_profiler=True)
    KERNEL_PROFILER = True
except Exception as e:  # pragma: no cover - CPU-only fallback
    print(f"Kernel profiler unavailable ({e}); continuing without it.")
    ti.init(arch=ti.gpu)

import algan.rendering.raytracing.primitives as rtp
import algan.rendering.raytracing.stbvh as stbvh_mod
from algan.rendering.primitives.primitive import RenderPrimitive
from algan.rendering.raytracing import enable_ray_tracing
from algan.scene import Scene

OUT_DIR = os.path.join("algan_outputs",
                       "raytracing_profiling")
REPORT_PATH = "raytracing_profile_report.txt"


# ---------------------------------------------------------------------------
# Timing infrastructure
# ---------------------------------------------------------------------------
def _sync_devices():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ti.sync()


class StageTimers:
    """Accumulates wall time per named stage, with device syncs at the
    boundaries so GPU work is attributed to the stage that issued it.
    Re-entrant stages (recursion) are only timed at the outermost level.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.times = defaultdict(float)
        self.counts = defaultdict(int)
        self.active = set()
        self.kernel_launches = []  # (num_frames, num_rays, seconds)

    @contextmanager
    def stage(self, name):
        if name in self.active:
            yield
            return
        self.active.add(name)
        _sync_devices()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            _sync_devices()
            self.times[name] += time.perf_counter() - t0
            self.counts[name] += 1
            self.active.discard(name)

    def wrap_function(self, obj, attr, name):
        orig = getattr(obj, attr)

        def wrapped(*args, **kwargs):
            with self.stage(name):
                return orig(*args, **kwargs)

        wrapped._profiling_original = orig
        setattr(obj, attr, wrapped)
        return orig


TIMERS = StageTimers()
SCENE_STATS = {}
SPP = 1  # samples per pixel (set from --spp; > 1 selects the MC kernels)


def _tensor_mb(t):
    return t.numel() * t.element_size() / 2**20


def _capture_scene_stats(scene):
    """Record sizes/shapes of the merged scene a single time per batch."""
    stats = {}
    if "tri_pos" in scene:  # optimized split layout
        tv, tc = scene["tri_pos"], scene["tri_colors"]
        stats["tri_verts_mb"] = (_tensor_mb(tv)
                                 + _tensor_mb(scene["tri_norm"])
                                 + _tensor_mb(scene["tri_extra"]))
    else:
        tv, tc = scene["tri_verts"], scene["tri_colors"]
        stats["tri_verts_mb"] = _tensor_mb(tv)
    stats["num_frames"] = scene["num_frames"]
    stats["triangles"] = tv.shape[1]
    stats["tri_verts_shape"] = tuple(tv.shape)
    stats["tri_colors_shape"] = tuple(tc.shape)
    stats["tri_colors_mb"] = _tensor_mb(tc)
    stats["circuits"] = int(scene["num_circuits"])
    stats["edges_shape"] = tuple(scene["edges_2d"].shape)
    stats["edges_mb"] = _tensor_mb(scene["edges_2d"])
    stats["circuit_colors_shape"] = tuple(scene["circuit_colors"].shape)
    stats["circuit_meta_mb"] = (_tensor_mb(scene["circuit_meta"])
                                + _tensor_mb(scene["circuit_colors"])
                                + _tensor_mb(scene["circuit_border_colors"]))
    for key, bvh in (("tri_bvh", scene["tri_bvh"]),
                     ("bez_bvh", scene["bez_bvh"])):
        stats[f"{key}_instances"] = int((bvh.leaf_prim >= 0).sum())
        stats[f"{key}_leaves"] = bvh.num_leaves
        stats[f"{key}_nodes"] = bvh.num_nodes
        stats[f"{key}_mb"] = bvh.get_memory_used() / 2**20
    stats["total_scene_mb"] = (
        stats["tri_verts_mb"] + stats["tri_colors_mb"] + stats["edges_mb"]
        + stats["circuit_meta_mb"] + stats["tri_bvh_mb"] + stats["bez_bvh_mb"])
    stats["cuda_allocated_after_merge_mb"] = (
        torch.cuda.memory_allocated() / 2**20 if torch.cuda.is_available()
        else 0.0)
    SCENE_STATS.setdefault("batches", []).append(stats)


def install_instrumentation():
    """Wrap the ray tracing pipeline's entry points with stage timers."""
    # Scene-side preparation (mob state evaluation; not part of the ray
    # tracer but reported for end-to-end context).
    TIMERS.wrap_function(Scene, "get_batch_of_primitives",
                         "scene prep: mob state -> primitives")

    # Geometry packing.
    TIMERS.wrap_function(rtp.RayTracedTrianglePrimitive, "project_to_screen",
                         "triangles: shade + pack (project_to_screen)")
    TIMERS.wrap_function(rtp.RayTracedBezierCircuitPrimitive,
                         "project_to_screen",
                         "beziers: sample + pack (project_to_screen)")
    TIMERS.wrap_function(rtp.RayTracedBezierCircuitPrimitive,
                         "_compute_samples_per_segment",
                         "beziers:   - sample density")
    TIMERS.wrap_function(rtp.RayTracedBezierCircuitPrimitive,
                         "_build_circuit_geometry",
                         "beziers:   - polyline geometry")
    TIMERS.wrap_function(rtp.RayTracedBezierCircuitPrimitive,
                         "_build_frame_bounds",
                         "beziers:   - frame bounds")

    # Scene merge + BVH builds.
    orig_merge = rtp._merge_scene

    def merge_wrapper(primitives):
        had_cache = getattr(primitives[0], "_rt_merged_scene", None) is not None
        with TIMERS.stage("merge collections + build BVHs"):
            scene = orig_merge(primitives)
        if not had_cache:
            _capture_scene_stats(scene)
        return scene

    rtp._merge_scene = merge_wrapper
    TIMERS.wrap_function(rtp, "build_stbvh", "  - STBVH build (in merge)")
    TIMERS.wrap_function(stbvh_mod, "segment_primitives_in_time",
                         "  - STBVH temporal segmentation")

    # Render chunk internals.
    TIMERS.wrap_function(rtp, "_prefill_background", "background prefill")
    TIMERS.wrap_function(RenderPrimitive, "post_process_frames",
                         "post-process frames")

    def wrap_kernel(name, rays_of_args):
        orig_kernel = getattr(rtp, name)

        def kernel_wrapper(*args):
            _sync_devices()
            t0 = time.perf_counter()
            result = orig_kernel(*args)
            _sync_devices()
            dt = time.perf_counter() - t0
            TIMERS.times[f"trace kernel ({name})"] += dt
            TIMERS.counts[f"trace kernel ({name})"] += 1
            frames, rays = rays_of_args(args)
            TIMERS.kernel_launches.append((frames, rays, dt))
            return result

        setattr(rtp, name, kernel_wrapper)

    def deterministic_rays(args):
        out = args[-1]
        return out.shape[0], out.shape[0] * out.shape[1]

    def mc_rays(args):
        # Monte Carlo kernels trace frames * pixels * spp paths.
        out = args[-1]
        spp = max(1, int(SPP))
        return out.shape[0], out.shape[0] * out.shape[1] * spp

    wrap_kernel("render_scene_stbvh", deterministic_rays)
    if hasattr(rtp, "path_trace_scene_stbvh"):
        wrap_kernel("path_trace_scene_stbvh", mc_rays)
    if hasattr(rtp, "path_trace_physical_stbvh"):
        wrap_kernel("path_trace_physical_stbvh", mc_rays)
    if hasattr(rtp, "finalize_samples"):
        TIMERS.wrap_function(rtp, "finalize_samples", "finalize sample accum")
    TIMERS.wrap_function(rtp, "render_batch_ray_traced",
                         "ray traced render total")


def run_once(scene_func, settings, tag="", run_index=0):
    TIMERS.reset()
    SCENE_STATS.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    if KERNEL_PROFILER:
        ti.profiler.clear_kernel_profiler_info()

    scene = SceneManager.reset()
    scene.set_render_settings(settings)
    scene_func()

    profiler = cProfile.Profile()
    _sync_devices()
    t0 = time.perf_counter()
    profiler.enable()
    render_to_file(file_name=f"profiling{tag}_run{run_index}", output_dir=OUT_DIR,
                   output_path="", render_settings=settings,
                   file_extension="mp4", background_color=GREY.set_opacity(1.0))
    profiler.disable()
    _sync_devices()
    total = time.perf_counter() - t0

    dump_path = os.path.join(os.path.dirname(__file__),
                             f"raytracing_cprofile{tag}_run{run_index}.txt")
    with open(dump_path, "w") as f:
        pstats.Stats(profiler, stream=f).sort_stats(
            pstats.SortKey.CUMULATIVE).print_stats()

    peak_alloc = (torch.cuda.max_memory_allocated() / 2**20
                  if torch.cuda.is_available() else 0.0)
    peak_reserved = (torch.cuda.max_memory_reserved() / 2**20
                     if torch.cuda.is_available() else 0.0)
    return dict(total=total, peak_alloc_mb=peak_alloc,
                peak_reserved_mb=peak_reserved,
                times=dict(TIMERS.times), counts=dict(TIMERS.counts),
                launches=list(TIMERS.kernel_launches),
                scene_stats=[dict(b) for b in SCENE_STATS.get("batches", [])],
                cprofile_path=dump_path)


def format_report(results):
    lines = []
    w = lines.append
    w("=" * 78)
    w("Ray tracing profiling report (deterministic, 1 sample per pixel)")
    w("=" * 78)
    dev = (torch.cuda.get_device_name(0) if torch.cuda.is_available()
           else "cpu")
    w(f"device: {dev}")
    for i, res in enumerate(results, 1):
        w("")
        w("-" * 78)
        label = "cold (includes Taichi JIT)" if i == 1 else "warm"
        w(f"RUN {i} ({label}): end-to-end {res['total']:.2f}s")
        w("-" * 78)
        w(f"{'stage':<52}{'calls':>6}{'seconds':>10}{'% of total':>10}")
        for name, secs in sorted(res["times"].items(), key=lambda kv: -kv[1]):
            w(f"{name:<52}{res['counts'][name]:>6}{secs:>10.3f}"
              f"{100 * secs / res['total']:>9.1f}%")
        accounted = sum(v for k, v in res["times"].items()
                        if not k.startswith(("  -", "beziers:   -"))
                        and k != "ray traced render total")
        w(f"{'(unaccounted: video encode, scene mgmt, ...)':<52}{'':>6}"
          f"{res['total'] - accounted:>10.3f}"
          f"{100 * (res['total'] - accounted) / res['total']:>9.1f}%")

        w("")
        w("trace kernel launches:")
        for j, (frames, rays, dt) in enumerate(res["launches"]):
            note = " (incl. JIT compile)" if (i == 1 and j == 0) else ""
            w(f"  launch {j}: {frames:>4} frames, {rays / 1e6:7.2f} M rays in "
              f"{dt:7.3f}s -> {rays / dt / 1e6:8.2f} M rays/s{note}")

        w("")
        w(f"GPU memory: peak allocated {res['peak_alloc_mb']:.0f} MB, "
          f"peak reserved {res['peak_reserved_mb']:.0f} MB "
          f"(includes the render-buffer slab)")
        for k, st in enumerate(res["scene_stats"]):
            w(f"  batch {k}: merged scene arrays "
              f"{st['total_scene_mb']:.1f} MB total "
              f"(allocated after merge: "
              f"{st['cuda_allocated_after_merge_mb']:.0f} MB)")
            w(f"    triangles: {st['triangles']} prims, verts(+norm+extra) "
              f"{st['tri_verts_shape']} = {st['tri_verts_mb']:.1f} MB, colors "
              f"{st['tri_colors_shape']} = {st['tri_colors_mb']:.1f} MB")
            w(f"    tri STBVH: {st['tri_bvh_instances']} instances "
              f"({st['tri_bvh_instances'] / max(st['triangles'], 1):.2f}x prims,"
              f" {st['num_frames']} frames), {st['tri_bvh_nodes']} nodes, "
              f"{st['tri_bvh_mb']:.1f} MB")
            w(f"    beziers: {st['circuits']} circuits, edges "
              f"{st['edges_shape']} = {st['edges_mb']:.1f} MB, colors+meta "
              f"{st['circuit_meta_mb']:.1f} MB")
            w(f"    bez STBVH: {st['bez_bvh_instances']} instances "
              f"({st['bez_bvh_instances'] / max(st['circuits'], 1):.2f}x prims), "
              f"{st['bez_bvh_nodes']} nodes, {st['bez_bvh_mb']:.1f} MB")
        w(f"  cProfile dump: {res['cprofile_path']}")
    w("")
    return "\n".join(lines)


def profile_scene(scene_func, render_settings, tag=""):
    enable_ray_tracing(samples_per_pixel=SPP)
    install_instrumentation()
    os.makedirs(OUT_DIR, exist_ok=True)

    results = []
    runs = 1
    for i in range(1, runs + 1):
        print(f"\n===== profiling run {i}/{runs} =====")
        results.append(run_once(scene_func, render_settings, tag, i))

    report = format_report(results)
    print("\n" + report)
    if KERNEL_PROFILER:
        try:
            ti.profiler.print_kernel_profiler_info()
        except Exception as e:
            print(f"(taichi kernel profiler info unavailable: {e})")
    report_path = REPORT_PATH.replace(".txt", f"{tag}.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"report written to {report_path}")
