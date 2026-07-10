"""Single entry point for initializing the Taichi CUDA runtime.

The rasterizer, the fragment blender and the ray tracer each need Taichi up
before they define their kernels, and they may be imported in any order.
Routing every one of them through :func:`init_taichi` guarantees the runtime
is initialized exactly once, with a consistent performance config -- calling
``ti.init`` again would discard already-compiled kernels (previously the
rasterizer modules re-initialized each other at import).

Performance config (most are Taichi's own defaults, set explicitly so they
cannot silently regress, plus two empirically-tuned register knobs):

* ``fast_math`` / ``advanced_optimization`` / ``offline_cache`` -- on by
  default in Taichi 1.7.4; pinned here.
* ``ALGAN_GPU_MAX_REG`` (env int) -> ``gpu_max_reg``: cap on registers per
  thread for CUDA codegen (ptxas ``-maxrregcount``). 0/unset leaves it to
  ptxas, which for the big deterministic ray-trace kernel settles on 128 and
  spills heavily to local memory. Raising the cap keeps more values in
  registers (fewer spills) at the cost of occupancy; the sweet spot is tuned
  empirically per kernel/GPU.
* ``ALGAN_OPT_LEVEL`` (env int) -> ``opt_level`` (Taichi default 1). Higher is
  more aggressive but can *increase* register pressure, so it is opt-in.
"""
import os

import taichi as ti
import torch

from algan.settings.defaults import COMPUTING_DEFAULTS


def sync_devices():
    """Block until all pending torch-CUDA and Taichi device work has finished."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ti.sync()


def _already_initialized():
    try:
        return ti.lang.impl.get_runtime().prog is not None
    except Exception:
        return False


def taichi_init_kwargs():
    """Algan's Taichi runtime config, as a kwargs dict.

    Exposed so the profiler (:mod:`algan.utils.profiling_utils`) can launch
    Taichi with the *exact same* performance config production uses, just
    adding ``kernel_profiler=True`` -- otherwise the benchmark silently
    measures a different (much faster) config than real renders. See
    [[algan-render-benchmarking]].
    """
    kwargs = dict(arch=ti.cpu if COMPUTING_DEFAULTS.render_on_cpu else ti.gpu, fast_math=True,
                  # advanced_optimization defaults off (it raised register
                  # pressure on the big megakernels); env ALGAN_ADV_OPT=1 to A/B.
                  advanced_optimization=os.environ.get("ALGAN_ADV_OPT", "0") == "1",
                  # debug=True inserts a bounds-check on *every* ndarray access;
                  # the ray-trace megakernels do millions of array reads per ray
                  # (BVH nodes, packed geometry), so it ran them ~11x slower with
                  # no benefit to released renders. Keep it off (env ALGAN_TI_DEBUG=1
                  # re-enables it for kernel development).
                  debug=os.environ.get("ALGAN_TI_DEBUG", "0") == "1",
                  offline_cache=True,
                  # The default 100 MB cache LRU-evicts large megakernel
                  # artifacts once several variants (general / no-PN / lean /
                  # path-trace / wavefront, plus per-config rebuilds) are
                  # compiled, forcing repeated ~minutes-long recompiles. Raise
                  # it (disk-backed) so every kernel stays cached. The field is
                  # a 32-bit int, so stay just under 2^31 bytes (~1.9 GB, still
                  # 19x the default).
                  offline_cache_max_size_of_files=1_000_000_000)
    max_reg = int(os.environ.get("ALGAN_GPU_MAX_REG", "0"))
    if max_reg > 0:
        kwargs["gpu_max_reg"] = max_reg
    opt_level = int(os.environ.get("ALGAN_OPT_LEVEL", "0"))
    if opt_level > 0:
        kwargs["opt_level"] = opt_level
    return kwargs


def init_taichi():
    """Initialize Taichi (CUDA/GPU) with Algan's perf config, once."""
    if _already_initialized():
        return
    ti.init(**taichi_init_kwargs())
