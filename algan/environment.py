"""Declaration, validation and typed access for Algan's process environment.

Every environment variable Algan honors is declared in this module, and every
read of one goes through the ``env_*`` accessors below, which reject a name
that is not declared. That is what keeps the registry honest: the list is not
a copy of what the code reads, it is what the code reads *through*, so a knob
added without a declaration raises on its first read instead of quietly
becoming a variable users can misspell without warning.

Adding a variable is therefore one edit -- put its name in the right tuple
below -- and reading it is ``env_flag("ALGAN_...", default)`` or one of its
siblings at the point of use, where the default belongs next to the comment
explaining it. ``tests/unit_tests/test_environment.py`` enforces the one rule
that keeps the invariant intact: nothing in the package may reach an
``ALGAN_`` variable through :mod:`os` directly.

Values are parsed leniently and never fatally: an unusable value warns and
falls back to the caller's default, because a mistyped tuning knob should not
abort a render.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Mapping

from algan.errors import AlganConfigurationError, AlganWarning

#: Variables consumed while Torch and Taichi initialise. They must be set
#: before ``import algan`` and have no runtime Python setting; the render
#: daemon bakes their values at launch and refuses a client whose values
#: differ (see :data:`algan.daemon_client.STARTUP_ENV`, derived from this
#: tuple). Order is the order mismatches are reported in.
_STARTUP_VARIABLES = (
    "ALGAN_ANIMATION_DEVICE",
    "ALGAN_RENDER_DEVICE",
    "ALGAN_HOME",
    "ALGAN_CACHE_DIR",
    "TI_OFFLINE_CACHE_FILE_PATH",
    "ALGAN_SOFT_SHADOW_SAMPLES",
    "ALGAN_HDR_BUFFER_F16",
    "ALGAN_TI_DEBUG",
    "ALGAN_TAICHI_WARMSTART",
    "ALGAN_TAICHI_FAST_LAUNCH",
)

#: Variables read by the test and benchmark harnesses rather than by the
#: package. Declared so that a contributor who exports one -- as
#: ``CLAUDE.md``, ``tests/README.md`` and the contributing docs tell them to
#: -- does not get told it is unknown on every import.
_HARNESS_VARIABLES = (
    "ALGAN_RUN_DOC_RENDERS",
    "ALGAN_RUN_FULL_RENDERS",
    "ALGAN_UPDATE_FAST_BASELINE",
    "ALGAN_UPDATE_FULL_RENDER_BASELINES",
)

#: Everything else: read at import of the module that owns the knob, or live
#: at the point of use. Alphabetical.
_RUNTIME_VARIABLES = (
    "ALGAN_AA_DUMP",
    "ALGAN_ADV_OPT",
    "ALGAN_ANALYTIC_AA",
    "ALGAN_ANALYTIC_AA_BEZ",
    "ALGAN_ANALYTIC_AA_BEZ_MIN_HALF_WIDTH",
    "ALGAN_ANALYTIC_AA_BEZ_WEDGE",
    "ALGAN_ANALYTIC_AA_CHORD_TOLERANCE",
    "ALGAN_ANALYTIC_AA_EXACT",
    "ALGAN_ANALYTIC_AA_SEAM",
    "ALGAN_ANALYTIC_AA_SECONDARY",
    "ALGAN_ANALYTIC_AA_SECONDARY_MIN_ENERGY",
    "ALGAN_ANALYTIC_AA_RUN",
    "ALGAN_ANALYTIC_AA_ONE_MESH",
    "ALGAN_ANALYTIC_AA_RUN_FULL",
    "ALGAN_ANALYTIC_AA_RUN_RULE",
    "ALGAN_ANALYTIC_AA_SLIVER",
    "ALGAN_ANALYTIC_AA_TRI",
    "ALGAN_BATCH_BEZIER_PREP",
    "ALGAN_BATCH_SURFACE_PREP",
    "ALGAN_BEZ_BVH_SPLIT",
    "ALGAN_BLOOM_FFT_SMOOTH",
    "ALGAN_BVH_ARITY",
    "ALGAN_BVH_BLOCK_F16",
    "ALGAN_BVH_BUILD",
    "ALGAN_BVH_DEFER",
    "ALGAN_BVH_REFIT",
    "ALGAN_DAEMON_CHILD",
    "ALGAN_DAEMON_PORT",
    "ALGAN_DAEMON_TIMEOUT",
    "ALGAN_FRAG_PID_GATE",
    "ALGAN_GLOSSY_INTERLEAVE",
    "ALGAN_GLOSSY_REFLECTION",
    "ALGAN_GPU_MAX_REG",
    "ALGAN_HYBRID_RASTER",
    "ALGAN_INPLACE_AA",
    "ALGAN_KBUF",
    "ALGAN_LOG_LEVEL",
    "ALGAN_LOG_TAICHI_COMPILES",
    "ALGAN_MANIM_SVG_CACHE_MB",
    "ALGAN_MAX_SHADOW_LIGHTS",
    "ALGAN_MERGE_DEDUP_TIME",
    "ALGAN_MESH_ID",
    "ALGAN_MERGE_GPU_PEAK_FACTOR",
    "ALGAN_MERGE_ON_GPU",
    "ALGAN_MERGE_TRACK_PEAK",
    "ALGAN_OPAQUE_BVH_SKIP_DEAD",
    "ALGAN_OPT_DISABLE",
    "ALGAN_OPT_LEVEL",
    "ALGAN_PN_CRITERION_KERNEL",
    "ALGAN_POLYHEDRON_WINDING",
    "ALGAN_POST_PROCESS_TONEMAP",
    "ALGAN_POST_TONEMAP_KERNEL",
    "ALGAN_PREFETCH_BATCHES",
    "ALGAN_PREFETCH_MERGE",
    "ALGAN_PROFILE_CPROFILE",
    "ALGAN_PROFILE_NVPROF",
    "ALGAN_PROFILE_RUNS",
    "ALGAN_PROFILE_TELEMETRY",
    "ALGAN_PROGRESS",
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
    "ALGAN_REUSE_FETCHED_BATCH",
    "ALGAN_SHADOW_ANYHIT",
    "ALGAN_SLICE_ACROSS_SPAWNS",
    "ALGAN_SPARSE_DISCOVERY_SAFETY",
    "ALGAN_SPLIT_TIME_WEIGHT",
    "ALGAN_STBVH_LEAF_SIZE",
    "ALGAN_STBVH_TIGHTNESS",
    "ALGAN_TAICHI_COMPILE_LOG",
    "ALGAN_TAICHI_FAST_LAUNCH_VERIFY",
    "ALGAN_TAICHI_WARMSTART_VERIFY",
    "ALGAN_TI_KERNEL_PROFILER",
    "ALGAN_UNDER_NVPROF",
    "ALGAN_UNSUPPORTED_FEATURE_POLICY",
    "ALGAN_USE_DAEMON",
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
    "ALGAN_WATERTIGHT_TRI",
    "ALGAN_WELD_SURFACE_SEAMS",
    "ALGAN_WF_TEXTURED_FEATURES",
)

#: Every declared name, including the one variable Algan honors that is not
#: its own (Taichi's ``TI_OFFLINE_CACHE_FILE_PATH``).
_DECLARED = frozenset(_STARTUP_VARIABLES + _HARNESS_VARIABLES + _RUNTIME_VARIABLES)

#: The ``ALGAN_`` variables this version accepts. Anything else in the
#: environment with that prefix is a typo or a leftover from an older
#: version, and is reported at import.
ALGAN_ENVIRONMENT_VARIABLES = frozenset(
    name for name in _DECLARED if name.startswith("ALGAN_")
)

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


def startup_environment_variables() -> tuple[str, ...]:
    """The variables consumed while Torch and Taichi initialise, in report order."""
    return _STARTUP_VARIABLES


def _require_declared(name: str) -> None:
    if name not in _DECLARED:
        raise AlganConfigurationError(
            f"{name} is not a declared Algan environment variable. This is a "
            "bug in Algan, not in your environment: every variable the package "
            "reads must be listed in algan/environment.py, which is what lets "
            "Algan tell a real option from a misspelled one."
        )


def _raw(name: str) -> str | None:
    _require_declared(name)
    return os.environ.get(name)


def _warn_unusable(name: str, raw: str, expected: str, default) -> None:
    warnings.warn(
        f"{name}={raw!r} is not {expected}; using the default {default!r}.",
        AlganWarning,
        stacklevel=3,
    )


def env_is_set(name: str) -> bool:
    """Whether ``name`` is present in the environment, whatever its value."""
    _require_declared(name)
    return name in os.environ


def env_str(name: str, default: str | None = "") -> str | None:
    """The raw value of ``name``, or ``default`` when it is unset.

    Deliberately unstripped: callers that treat surrounding whitespace as
    significant (a path, a mode word) decide that themselves.
    """
    raw = _raw(name)
    return default if raw is None else raw


def env_flag(name: str, default: bool) -> bool:
    """``name`` parsed as a boolean, or ``default`` when unset or unusable.

    ``1``/``true``/``yes``/``on`` and ``0``/``false``/``no``/``off`` are all
    accepted, case- and whitespace-insensitively. An empty value counts as
    unset, so ``ALGAN_X=`` leaves the default in place.
    """
    raw = _raw(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if not value:
        return default
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    _warn_unusable(name, raw, "a boolean (1/0, true/false, yes/no, on/off)", default)
    return default


def env_int(name: str, default: int) -> int:
    """``name`` parsed as an integer, or ``default`` when unset or unusable."""
    raw = _raw(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        _warn_unusable(name, raw, "an integer", default)
        return default


def env_float(name: str, default: float) -> float:
    """``name`` parsed as a float, or ``default`` when unset or unusable."""
    raw = _raw(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError:
        _warn_unusable(name, raw, "a number", default)
        return default


def env_overrides(**values: str) -> dict[str, str]:
    """Declared ``name=value`` pairs, for building a child process's environment."""
    for name in values:
        _require_declared(name)
    return dict(values)


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
    "env_flag",
    "env_float",
    "env_int",
    "env_is_set",
    "env_overrides",
    "env_str",
    "startup_environment_variables",
    "unknown_algan_environment_variables",
    "warn_for_unknown_algan_environment_variables",
]
