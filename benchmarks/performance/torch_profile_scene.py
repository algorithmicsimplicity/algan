"""torch.profiler capture (CPU + CUDA) of a short warm render of a workload scene.

    python benchmarks/performance/torch_profile_scene.py explainer --quality UHD --frames 12
    python benchmarks/performance/torch_profile_scene.py graphics --quality UHD --frames 6

The stage profiler (``profile_scene``) syncs at stage boundaries, so it says
how long ``compact_sheets`` took and not what it did; this says what it did.
One warm-up render compiles the kernels and fills the memory model, then the
second render is captured and reported three ways:

* **by CUDA time** -- which device ops (torch kernels and Taichi launches)
  the GPU actually spent its time on;
* **by CPU self time** -- which host-side calls the render thread spent its
  time issuing; on a launch-bound chain this is the table that matters;
* **the sync census** -- every host<->device synchronisation the capture saw
  (``cudaStreamSynchronize``, ``cudaDeviceSynchronize``, ``Memcpy DtoH``)
  with counts, which is the number of times the queue was drained per chunk.

``--scopes`` adds a fourth view: the sparse-coverage chain's host functions
(``prepare_sparse_raster_coverage``, ``compact_sheets`` and the helpers they
call, the tile loop) are wrapped in ``record_function`` scopes, and every
profiler event is attributed to its INNERMOST scope. Per scope that gives
the call count (chunks), inclusive and exclusive time, the number of torch
ops issued, the number of host<->device synchronisations, and the top ops by
self CPU time -- which is what "fuse the compaction chain" needs: the ops
per chunk and the syncs per chunk, by function. ``--memory-gb`` caps the
memory model so a small local render is cut into several chunks.

The scene functions are the workload benchmarks' (``explainer_scene.scene``,
``graphics_scene.scene``), unchanged.
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("ALGAN_USE_DAEMON", "0")
os.environ.setdefault("ALGAN_VIDEO_ENCODER", "software")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

import algan  # noqa: E402
from algan import Scene  # noqa: E402
from algan.scene_manager import SceneManager  # noqa: E402

SCENES = {"explainer": "explainer_scene", "graphics": "graphics_scene"}

SCOPE_PREFIX = "S:"

# (module path, attribute, scope label). Every one is a module-level function
# called through its own module's globals (or imported at call time, as
# ``compact_sheets`` is inside prepare_sparse_raster_coverage), so wrapping the
# module attribute is enough -- the same reasoning ``profiling_utils`` uses.
SCOPED_FUNCTIONS = (
    ("algan.rendering.raytracing.tracer", "_build_raster_tables", "raster tables"),
    ("algan.rendering.raytracing.tracer", "_run_wavefront_tiles", "tile loop"),
    ("algan.rendering.raytracing.tracer", "_read_tile_alloc", "tile alloc read"),
    (
        "algan.rendering.raytracing.raster_pipeline",
        "prepare_sparse_raster_coverage",
        "sparse discovery",
    ),
    ("algan.rendering.raytracing.raster_pipeline", "_window_pairs", "  window pairs"),
    (
        "algan.rendering.raytracing.raster_pipeline",
        "_class_pairs_flat",
        "  class pairs flat",
    ),
    (
        "algan.rendering.raytracing.raster_pipeline",
        "_exact_fragment_order",
        "  fragment sort",
    ),
    (
        "algan.rendering.raytracing.raster_pipeline",
        "_gather_fragment_arrays",
        "  fragment gather",
    ),
    (
        "algan.rendering.raytracing.raster_pipeline",
        "_opaque_prefix_keep",
        "  opaque prefix keep",
    ),
    (
        "algan.rendering.raytracing.raster_pipeline",
        "_one_mesh_pixel_caps",
        "  one-mesh caps",
    ),
    ("algan.rendering.raytracing.sheets", "compact_sheets", "  compact_sheets"),
    ("algan.rendering.raytracing.sheets", "_shade_class", "    shade class"),
    (
        "algan.rendering.raytracing.sheets",
        "_pixel_group_order",
        "    pixel group order",
    ),
    ("algan.rendering.raytracing.sheets", "_prim_split_after", "    prim split"),
    ("algan.rendering.raytracing.sheets", "_conflict_rank", "    conflict rank"),
    ("algan.rendering.raytracing.sheets", "_sheet_rank_groups", "    rank groups"),
    ("algan.rendering.raytracing.sheets", "_rank_pool_groups", "    rank pool"),
    ("algan.rendering.raytracing.sheets", "_band_composite", "    band composite"),
    ("algan.rendering.raytracing.sheets", "_sheet_class_groups", "    class groups"),
    ("algan.rendering.raytracing.sheets", "_band_reduce", "    band reduce"),
    (
        "algan.rendering.raytracing.sheets",
        "_lane_first_owners",
        "    lane first owners",
    ),
    ("algan.rendering.raytracing.sheets", "_sheet_group_counts", "    group counts"),
    ("algan.rendering.raytracing.sheets", "_sheet_walk_order", "    walk order"),
    ("algan.rendering.raytracing.sheets", "_sibling_weights", "    sibling weights"),
    ("algan.rendering.raytracing.sheets", "_sheet_offsets", "    sheet offsets"),
    ("algan.rendering.raytracing.sheets", "_lexsort", "    lexsort"),
    (
        "algan.rendering.raytracing.raster_pipeline",
        "shade_sparse_raster_coverage",
        "sparse resolve",
    ),
)

SYNC_NAMES = (
    "aten::_local_scalar_dense",
    "aten::nonzero",
    "aten::unique_consecutive",
    "aten::_unique2",
    "aten::unique_dim",
    "cudaStreamSynchronize",
    "cudaDeviceSynchronize",
    "Memcpy DtoH",
)
LAUNCH_NAMES = ("cudaLaunchKernel", "cuLaunchKernel")


def install_scopes():
    """Wrap every SCOPED_FUNCTIONS entry in a ``record_function`` scope."""
    import importlib

    from torch.profiler import record_function

    for module_path, attr, label in SCOPED_FUNCTIONS:
        module = importlib.import_module(module_path)
        fn = getattr(module, attr, None)
        if fn is None:
            print(f"[tp] scope skipped, no such function: {module_path}.{attr}")
            continue
        name = SCOPE_PREFIX + label.strip()

        def make(fn, name):
            def wrapped(*a, **k):
                with record_function(name):
                    return fn(*a, **k)

            wrapped.__name__ = fn.__name__
            wrapped.__doc__ = fn.__doc__
            return wrapped

        setattr(module, attr, make(fn, name))


def _device_self(e):
    return getattr(e, "self_device_time_total", None) or getattr(
        e, "self_cuda_time_total", 0.0
    )


def report_scopes(prof, rows):
    """Attribute every event to its innermost scope and print per-scope tables."""
    import collections

    events = prof.events()
    calls = collections.Counter()
    incl = collections.Counter()
    excl = collections.Counter()
    dev = collections.Counter()
    ops = collections.Counter()
    syncs = collections.Counter()
    launches = collections.Counter()
    per_op = collections.defaultdict(collections.Counter)
    per_op_time = collections.defaultdict(collections.Counter)
    per_op_dev = collections.defaultdict(collections.Counter)

    def scope_of(e):
        q = e.cpu_parent
        while q is not None:
            if q.name.startswith(SCOPE_PREFIX):
                return q.name[len(SCOPE_PREFIX) :]
            q = q.cpu_parent
        return None

    for e in events:
        if e.name.startswith(SCOPE_PREFIX):
            calls[e.name[len(SCOPE_PREFIX) :]] += 1
            incl[e.name[len(SCOPE_PREFIX) :]] += e.cpu_time_total
        scope = scope_of(e)
        if scope is None:
            scope = "(outside every scope)"
        excl[scope] += e.self_cpu_time_total
        dev[scope] += _device_self(e)
        parent = e.cpu_parent
        top_level = parent is None or not parent.name.startswith("aten::")
        if e.name.startswith("aten::") and top_level:
            ops[scope] += 1
        if any(e.name.startswith(s) for s in SYNC_NAMES):
            syncs[scope] += 1
        if e.name in LAUNCH_NAMES:
            launches[scope] += 1
        per_op[scope][e.name] += 1
        per_op_time[scope][e.name] += e.self_cpu_time_total
        per_op_dev[scope][e.name] += _device_self(e)

    order = [label.strip() for _m, _a, label in SCOPED_FUNCTIONS]
    order.append("(outside every scope)")
    print("================ BY SCOPE (innermost attribution) ==========", flush=True)
    print(
        f"  {'scope':<28} {'calls':>6} {'incl ms':>9} {'ms/call':>8} {'self ms':>9}"
        f" {'dev ms':>8} {'ops':>7} {'ops/call':>8} {'syncs':>6} {'launch':>7}"
    )
    labels = {label.strip(): label for _m, _a, label in SCOPED_FUNCTIONS}
    for scope in order:
        if calls[scope] == 0 and excl[scope] == 0:
            continue
        n = max(1, calls[scope])
        print(
            f"  {labels.get(scope, scope):<28} {calls[scope]:>6} {incl[scope] / 1e3:>9.1f}"
            f" {incl[scope] / n / 1e3:>8.2f} {excl[scope] / 1e3:>9.1f} {dev[scope] / 1e3:>8.1f}"
            f" {ops[scope]:>7} {ops[scope] / n:>8.1f} {syncs[scope]:>6} {launches[scope]:>7}"
        )
    print()
    for scope in order:
        if calls[scope] == 0:
            continue
        n = calls[scope]
        print(f"---- {scope.strip()}: top ops by self CPU ({n} calls) ----")
        items = sorted(per_op_time[scope].items(), key=lambda kv: -kv[1])[:rows]
        for name, t in items:
            c = per_op[scope][name]
            print(
                f"  {name[:48]:<48} {c:>7} ({c / n:>6.1f}/call) {t / 1e3:>8.1f} ms"
                f" ({t / c:>7.1f} us each)  dev {per_op_dev[scope][name] / 1e3:>7.1f} ms"
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", choices=sorted(SCENES))
    parser.add_argument("--quality", default="UHD")
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--rows", type=int, default=45)
    parser.add_argument(
        "--scopes",
        action="store_true",
        help="wrap the sparse-coverage chain in record_function scopes and "
        "report ops/syncs/time per scope (innermost attribution)",
    )
    parser.add_argument(
        "--scope-rows",
        type=int,
        default=18,
        help="top ops to list per scope under --scopes",
    )
    parser.add_argument(
        "--memory-gb",
        type=float,
        default=None,
        help="cap the memory model (SETTINGS.computing.available_memory_override) "
        "so a small render is cut into several chunks",
    )
    args = parser.parse_args(argv)

    module = __import__(SCENES[args.scene])
    settings = getattr(algan, args.quality)
    if args.memory_gb is not None:
        from algan.constants.math import GIGABYTES
        from algan.settings import SETTINGS

        SETTINGS.computing.set(
            available_memory_override=int(args.memory_gb * GIGABYTES)
        )
    if args.scopes:
        install_scopes()
    seconds = args.frames / settings.frames_per_second
    ffmpeg = ["-crf", "17", "-preset", "ultrafast"]

    def render(tag):
        SceneManager.reset()
        module.scene(seconds)
        return Scene.save_video(
            os.path.join("algan_outputs", "torch_profile", f"{args.scene}_{tag}.mp4"),
            settings,
            overwrite=True,
            reset=True,
            ffmpeg_params=ffmpeg,
        )

    print(
        f"[tp] warm-up render: {args.scene} {args.quality} {args.frames} frames",
        flush=True,
    )
    render("warm")
    print("[tp] profiled render", flush=True)
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities, record_shapes=False) as prof:
        result = render("prof")
    print(f"[tp] profiled render took {result.walltime_seconds:.2f}s wall", flush=True)

    ka = prof.key_averages()
    width = 70
    if torch.cuda.is_available():
        print("================ BY CUDA TIME ================", flush=True)
        print(
            ka.table(
                sort_by="cuda_time_total",
                row_limit=args.rows,
                max_name_column_width=width,
            )
        )
    print("================ BY CPU SELF TIME =============", flush=True)
    print(
        ka.table(
            sort_by="self_cpu_time_total",
            row_limit=args.rows,
            max_name_column_width=width,
        )
    )
    print("================ SYNC CENSUS =================", flush=True)
    total_cpu = sum(e.self_cpu_time_total for e in ka)
    for e in sorted(ka, key=lambda e: -e.count):
        name = e.key
        if any(
            s in name
            for s in (
                "Synchronize",
                "Memcpy DtoH",
                "Memcpy HtoD",
                "aten::item",
                "aten::_local_scalar_dense",
                "aten::nonzero",
                "aten::unique_consecutive",
                "aten::_unique2",
                "aten::sort",
                "aten::argsort",
            )
        ):
            print(
                f"  {name[:60]:<60} count {e.count:>7}  cpu total {e.cpu_time_total / 1e3:9.1f} ms"
                f"  ({100.0 * e.self_cpu_time_total / max(1.0, total_cpu):5.1f}% of self cpu)"
            )
    n_kernels = sum(
        e.count for e in ka if e.device_type == torch.autograd.DeviceType.CUDA
    )
    print(f"  device-side records: {n_kernels}", flush=True)
    if args.scopes:
        report_scopes(prof, args.scope_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
