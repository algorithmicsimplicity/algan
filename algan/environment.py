"""Validation for Algan's process environment.

Environment variables are deliberately listed in one place so misspelled
``ALGAN_`` options do not fail silently. Keep this registry in sync whenever a
new environment-controlled option is added to the package.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Mapping

from algan.errors import AlganWarning

ALGAN_ENVIRONMENT_VARIABLES = frozenset(
    {
        "ALGAN_ADV_OPT",
        "ALGAN_ANALYTIC_AA",
        "ALGAN_ANALYTIC_AA_BEZ",
        "ALGAN_ANALYTIC_AA_BEZ_MIN_HALF_WIDTH",
        "ALGAN_ANALYTIC_AA_CHORD_TOLERANCE",
        "ALGAN_ANALYTIC_AA_SEAM",
        "ALGAN_ANALYTIC_AA_SECONDARY",
        "ALGAN_ANALYTIC_AA_SECONDARY_MIN_ENERGY",
        "ALGAN_ANALYTIC_AA_SLIVER",
        "ALGAN_ANALYTIC_AA_TRI",
        "ALGAN_ANIMATION_DEVICE",
        "ALGAN_BATCH_BEZIER_PREP",
        "ALGAN_BATCH_SURFACE_PREP",
        "ALGAN_BVH_ARITY",
        "ALGAN_BVH_BLOCK_F16",
        "ALGAN_BVH_BUILD",
        "ALGAN_BVH_DEFER",
        "ALGAN_BVH_REFIT",
        "ALGAN_CACHE_DIR",
        "ALGAN_DAEMON_PORT",
        "ALGAN_GLOSSY_INTERLEAVE",
        "ALGAN_GLOSSY_REFLECTION",
        "ALGAN_GPU_MAX_REG",
        "ALGAN_HDR_BUFFER_F16",
        "ALGAN_HOME",
        "ALGAN_HYBRID_RASTER",
        "ALGAN_INPLACE_AA",
        "ALGAN_KBUF",
        "ALGAN_LOG_LEVEL",
        "ALGAN_LOG_TAICHI_COMPILES",
        "ALGAN_MANIM_SVG_CACHE_MB",
        "ALGAN_MAX_SHADOW_LIGHTS",
        "ALGAN_MERGE_GPU_PEAK_FACTOR",
        "ALGAN_MERGE_ON_GPU",
        "ALGAN_MERGE_TRACK_PEAK",
        "ALGAN_OPT_DISABLE",
        "ALGAN_OPT_LEVEL",
        "ALGAN_PN_OBB",
        "ALGAN_POST_PROCESS_TONEMAP",
        "ALGAN_POST_TONEMAP_KERNEL",
        "ALGAN_PREFETCH_BATCHES",
        "ALGAN_PREFETCH_MERGE",
        "ALGAN_PROFILE_CPROFILE",
        "ALGAN_PROFILE_NVPROF",
        "ALGAN_PROFILE_RUNS",
        "ALGAN_PROFILE_TELEMETRY",
        "ALGAN_PROJECT_GPU_PEAK_FACTOR",
        "ALGAN_PROJECT_ON_GPU",
        "ALGAN_PROMOTE_CONSTANTS",
        "ALGAN_RASTER_BEZ_PRECOMPUTE",
        "ALGAN_RASTER_COVERED_SHADE",
        "ALGAN_RASTER_EMPTY_SKIP",
        "ALGAN_RASTER_PAIR_FLAGS",
        "ALGAN_RASTER_SPARSE_COVERAGE",
        "ALGAN_RASTER_SS",
        "ALGAN_RASTER_STRADDLE_CLIP",
        "ALGAN_RASTER_TRI_PRECOMPUTE",
        "ALGAN_RENDER_DEVICE",
        "ALGAN_REUSE_FETCHED_BATCH",
        "ALGAN_SLICE_ACROSS_SPAWNS",
        "ALGAN_SOFT_SHADOW_SAMPLES",
        "ALGAN_SPARSE_DISCOVERY_SAFETY",
        "ALGAN_SPLIT_TIME_WEIGHT",
        "ALGAN_STBVH_LEAF_SIZE",
        "ALGAN_STBVH_TIGHTNESS",
        "ALGAN_TAICHI_COMPILE_LOG",
        "ALGAN_TAICHI_FAST_LAUNCH",
        "ALGAN_TAICHI_FAST_LAUNCH_VERIFY",
        "ALGAN_TAICHI_WARMSTART",
        "ALGAN_TAICHI_WARMSTART_VERIFY",
        "ALGAN_TI_DEBUG",
        "ALGAN_TI_KERNEL_PROFILER",
        "ALGAN_UNDER_NVPROF",
        "ALGAN_UNSUPPORTED_FEATURE_POLICY",
        "ALGAN_WAVEFRONT_INITIAL_POOL_RATIO",
        "ALGAN_WAVEFRONT_SPLIT",
        "ALGAN_WAVEFRONT_TILE",
        "ALGAN_WAVEFRONT_TILE_AUTO",
        "ALGAN_WAVEFRONT_TILE_MAX",
        "ALGAN_WAVEFRONT_TILE_MIN",
        "ALGAN_WAVEFRONT_TILE_SAFETY",
        "ALGAN_WF_COMPACT_ACTIVE_ONLY",
        "ALGAN_WF_GEN_FUSED",
        "ALGAN_WF_GEN_FUSED_GAIN",
        "ALGAN_WF_GEN_FUSED_MIN_WIN",
        "ALGAN_WF_MEM_TRIM",
        "ALGAN_WF_NEAR_FIRST",
        "ALGAN_WF_OPAQUE_CLOSEST",
        "ALGAN_WF_OPAQUE_PREPASS",
        "ALGAN_WF_REVALIDATE_PENDING",
        "ALGAN_WF_SKIP_UNLIT_NORMAL",
        "ALGAN_WF_TEXTURED_FEATURES",
    }
)


def unknown_algan_environment_variables(
    environ: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Return sorted ``ALGAN_`` variables that this version does not use."""
    if environ is None:
        environ = os.environ
    return tuple(
        sorted(
            name
            for name in environ
            if name.startswith("ALGAN_") and name not in ALGAN_ENVIRONMENT_VARIABLES
        )
    )


def warn_for_unknown_algan_environment_variables(
    environ: Mapping[str, str] | None = None,
) -> None:
    """Warn once when the process contains unsupported Algan variables."""
    unknown = unknown_algan_environment_variables(environ)
    if not unknown:
        return
    noun = f"variable{'' if len(unknown) == 1 else 's'}"
    warnings.warn(
        f"Unknown Algan environment {noun}: {', '.join(unknown)}. "
        "These variables will be ignored; check their spelling or remove them.",
        AlganWarning,
        stacklevel=2,
    )


__all__ = [
    "ALGAN_ENVIRONMENT_VARIABLES",
    "unknown_algan_environment_variables",
    "warn_for_unknown_algan_environment_variables",
]
