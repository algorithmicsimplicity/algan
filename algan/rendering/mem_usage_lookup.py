"""Runtime lookup into the measured memory tables, with a probe fallback.

``mem_usage_runtime`` is a pure interpreter with no Algan dependencies; this is
the layer that knows about settings, arenas and post-process chains.

The shipped table covers common configurations. Anything it misses -- an
unusual resolution, a user's own post-process callable -- is measured on
demand by running the real pipeline once over a single frame and recording the
allocations. That is what lets a custom post-process work at all: the previous
design refused to size one unless the author attached an exact
``algan_memory_planner``.
"""

from __future__ import annotations

import json
import logging
import os

import torch

from algan.rendering.mem_usage_runtime import (
    peak_bytes,
    post_process_chain_id,
    replay,
    trace_from_events,
)

logger = logging.getLogger("algan.mem_usage")

# Probe results for this process, keyed by route key. Populated from disk on
# first miss and written back when the key is stable enough to be reusable.
_PROBE_CACHE = {}
_DISK_LOADED = set()


def _tables():
    try:
        from algan.rendering import mem_usage

        return mem_usage
    except ImportError:  # pragma: no cover - only before the first generation
        return None


def postprocess_key(*, frame_shape, frame_dtype, anti_alias_level,
                    post_processes, apply_fxaa, tonemap_enabled, tonemapping,
                    tonemap_method, tonemap_kernel):
    """Route key matching the one recorded by ``post_process_frames``.

    Must stay in step with that annotation: a mismatch shows up as a permanent
    table miss (so every render probes), not as a wrong answer.
    """
    _frames, height, width, channels = (int(x) for x in frame_shape)
    return tuple(sorted({
        "anti_alias_level": int(anti_alias_level),
        "chain": post_process_chain_id(post_processes),
        "channels": channels,
        "dtype": str(frame_dtype),
        "fxaa": int(bool(apply_fxaa)),
        "height": height,
        "tonemap": int(bool(tonemap_enabled)),
        "tonemap_kernel": int(bool(tonemap_kernel)),
        "tonemap_method": str(tonemap_method),
        "tonemapping": int(bool(tonemapping)),
        "width": width,
    }.items()))


def _cache_path(key):
    try:
        from algan.settings import SETTINGS

        root = SETTINGS.paths.cache_directory
    except Exception:
        return None
    if not root:
        return None
    import hashlib

    digest = hashlib.sha256(repr(key).encode("utf-8")).hexdigest()[:24]
    return os.path.join(str(root), "mem_probe", f"{digest}.json")


def _key_is_stable(key):
    """Whether a key may be persisted to disk.

    ``post_process_chain_id`` returns ``None`` for a closure or a lambda, whose
    identity does not survive the process. Caching one to disk would key this
    run's measurement to a different run's callable.
    """
    return dict(key).get("chain") is not None


def _load_probe_from_disk(key):
    if key in _DISK_LOADED or not _key_is_stable(key):
        return None
    _DISK_LOADED.add(key)
    path = _cache_path(key)
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return None
    tables = _tables()
    stamp = getattr(tables, "SCHEMA_FINGERPRINT", None) if tables else None
    if payload.get("fingerprint") != stamp:
        # The engine's allocation schema moved since this probe was taken.
        return None
    return tuple(tuple(event) for event in payload.get("trace", ()))


def _store_probe_to_disk(key, trace):
    if not _key_is_stable(key):
        return
    path = _cache_path(key)
    if not path:
        return
    tables = _tables()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({
                "fingerprint": getattr(tables, "SCHEMA_FINGERPRINT", None),
                "key": repr(key),
                "trace": [list(event) for event in trace],
            }, handle)
    except Exception:
        # A cache that cannot be written is a performance loss, never an error.
        pass


def _probe_frames(shape, dtype, device):
    """A probe frame that exercises the *longest* allocation path.

    Bloom short-circuits its whole blur chain on a frame with no glow, which
    allocates four buffers instead of forty. A zero-filled probe would
    therefore measure -- and reserve -- a small fraction of what a glowing
    frame actually needs.
    """
    if dtype == torch.uint8:
        frames = torch.full(shape, 128, dtype=dtype, device=device)
        frames[..., 3] = 100
        return frames
    frames = torch.full(shape, 0.5, dtype=dtype, device=device)
    frames[..., 3] = 0.7
    return frames


def probe_postprocess(memory, *, frame_shape, frame_dtype, anti_alias_level,
                      post_processes, apply_fxaa, device):
    """Measure the post-process chain over one frame; return a replay trace.

    Runs inside a temp scope on the live arena, so the probe's own allocations
    are released before the caller continues. Returns ``None`` when the probe
    itself does not fit, which the caller must read as "one frame does not fit"
    rather than as an error.
    """
    from algan.rendering.post_processing.post_process import (
        post_process_frames,
    )
    from algan.utils.memory_utils import InsufficientMemoryException

    _frames, height, width, channels = (int(x) for x in frame_shape)
    events_by_count = {}
    for count in (1, 2):
        shape = (count, height, width, channels)
        try:
            with memory.temp():
                frames = _probe_frames(shape, frame_dtype, device)
                with memory.recording() as recorder:
                    post_process_frames(
                        memory, frames, anti_alias_level,
                        post_processes=post_processes,
                        apply_fxaa=apply_fxaa)
                found = recorder.scopes("postprocess")
                if not found:
                    return None
                events_by_count[count] = list(found[0].events)
        except (InsufficientMemoryException, torch.OutOfMemoryError):
            if count == 1:
                return None
            # One frame fit but two did not. Fall back to charging the
            # single-frame peak per frame, which is always conservative:
            # peak(T) = max_k(a_k + b_k*T) <= T * max_k(a_k + b_k) for a_k >= 0.
            break
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Memory probe of the post-process chain failed (%s: %s); "
                "falling back to a conservative estimate.",
                type(exc).__name__, exc)
            return None

    if 1 not in events_by_count:
        return None
    if 2 not in events_by_count:
        single = events_by_count[1]
        # numel = 0 + numel*frames, i.e. everything scales with the frame count.
        return tuple(
            ("A", 0, event[4], event[5], event[3]) if event[0] == "alloc"
            else ("(", event[1]) if event[0] == "temp_push"
            else (")",)
            for event in single)
    try:
        return trace_from_events(
            events_by_count[1], ((1, events_by_count[1]),
                                 (2, events_by_count[2])))
    except ValueError as exc:
        logger.warning(
            "Post-process chain allocates differently at one and two frames "
            "(%s); charging the single-frame peak per frame.", exc)
        single = events_by_count[1]
        return tuple(
            ("A", 0, event[4], event[5], event[3]) if event[0] == "alloc"
            else ("(", event[1]) if event[0] == "temp_push"
            else (")",)
            for event in single)


def get_post_process_memory_required(
        frame_shape, frame_dtype, anti_alias_level, post_processes, apply_fxaa,
        *, initial_pointer=0, device=None, memory=None, tonemap_enabled=None,
        tonemapping=None, tonemap_method=None, tonemap_kernel=None):
    """Arena bytes the post-processing pipeline needs above ``initial_pointer``.

    Resolved from the measured table when the configuration is covered, and by
    a one-frame probe otherwise. Returns ``None`` when neither is possible --
    the caller should then treat the batch as not fitting, which preserves the
    existing single-frame out-of-memory diagnostic.
    """
    from algan.settings import SETTINGS

    rt = SETTINGS.raytracing
    if tonemap_enabled is None:
        tonemap_enabled = rt.is_post_process_tonemap_enabled()
    if tonemapping is None:
        tonemapping = rt.TONEMAPPING
    if tonemap_method is None:
        tonemap_method = rt.TONEMAP_METHOD
    if tonemap_kernel is None:
        tonemap_kernel = rt.POST_TONEMAP_KERNEL

    num_frames = int(frame_shape[0])
    key = postprocess_key(
        frame_shape=frame_shape, frame_dtype=frame_dtype,
        anti_alias_level=anti_alias_level, post_processes=post_processes,
        apply_fxaa=apply_fxaa, tonemap_enabled=tonemap_enabled,
        tonemapping=tonemapping, tonemap_method=tonemap_method,
        tonemap_kernel=tonemap_kernel)

    tables = _tables()
    trace = None
    if tables is not None:
        trace = tables.TRACES.get("postprocess", {}).get(key)
    if trace is None:
        trace = _PROBE_CACHE.get(key)
    if trace is None:
        trace = _load_probe_from_disk(key)
        if trace is not None:
            _PROBE_CACHE[key] = trace
    if trace is None and memory is not None:
        trace = probe_postprocess(
            memory, frame_shape=frame_shape, frame_dtype=frame_dtype,
            anti_alias_level=anti_alias_level, post_processes=post_processes,
            apply_fxaa=apply_fxaa,
            device=device or memory.data.device)
        if trace is not None:
            _PROBE_CACHE[key] = trace
            _store_probe_to_disk(key, trace)
            logger.debug("Probed post-process memory for %s.", key)
    if trace is None:
        return None
    return peak_bytes(trace, num_frames, initial_pointer)


def unit_bytes(scope, key, _include_slack=True, **drivers):
    """Bytes for a unit-coefficient scope.

    ``_include_slack`` adds the measured alignment pad. Callers that walk the
    arena pointer themselves already reproduce alignment exactly and must pass
    ``False``, or the padding is charged twice.
    """
    tables = _tables()
    if tables is None:
        return None
    entry = tables.UNITS.get(scope, {}).get(tuple(sorted(key.items())))
    if entry is None:
        return None
    total = int(entry.get("fixed", 0))
    if _include_slack:
        total += int(entry.get("align_slack", 0))
    for name, coefficient in entry.items():
        if name in ("fixed", "align_slack"):
            continue
        total += int(coefficient) * int(drivers.get(name, 0))
    return total


def unit_coefficients(scope, key):
    """Raw measured coefficients for a route, or ``None``.

    ``fixed`` and ``align_slack`` are per-occurrence; every other entry is
    bytes per unit of the driver named by its key.
    """
    tables = _tables()
    if tables is None:
        return None
    entry = tables.UNITS.get(scope, {}).get(tuple(sorted(key.items())))
    return dict(entry) if entry is not None else None


def unit_bytes_or_bound(scope, key, _include_slack=True, **drivers):
    """Like :func:`unit_bytes`, but never returns ``None`` for a known scope.

    A route the corpus never measured falls back to the largest coefficients
    across every key of that scope. That over-reserves -- possibly by a lot --
    but the alternative is either a crash or an under-estimate, and only one of
    those three is recoverable. The warning names the key so the corpus can be
    extended.
    """
    exact = unit_bytes(scope, key, _include_slack=_include_slack, **drivers)
    if exact is not None:
        return exact
    tables = _tables()
    entries = tables.UNITS.get(scope) if tables else None
    if not entries:
        return None
    logger.warning(
        "No measured memory entry for %s route %s; falling back to the "
        "largest measured route. Re-run 'python -m "
        "algan.utils.calibrate_memory' with a scene covering it.",
        scope, tuple(sorted(key.items())))
    worst = {}
    for coefficients in entries.values():
        for name, value in coefficients.items():
            worst[name] = max(worst.get(name, 0), int(value))
    slack = int(worst.pop("align_slack", 0))
    total = int(worst.pop("fixed", 0)) + (slack if _include_slack else 0)
    for name, coefficient in worst.items():
        total += coefficient * int(drivers.get(name, 0))
    return total


def density_seed(scope):
    """Corpus-seeded units-per-denominator for a value-dependent scope."""
    tables = _tables()
    if tables is None:
        return None
    entry = tables.DENSITIES.get(scope)
    if entry is None:
        return None
    return float(entry["density"]) * float(
        getattr(tables, "DENSITY_SAFETY", 1.0))


def density_structural(scope, key=()):
    """Exact bytes-per-unit coefficients of a value-dependent scope."""
    tables = _tables()
    if tables is None:
        return None
    entry = tables.DENSITIES.get(scope)
    if entry is None:
        return None
    return entry["structural"].get(tuple(key))


__all__ = [
    "density_seed",
    "density_structural",
    "get_post_process_memory_required",
    "postprocess_key",
    "probe_postprocess",
    "replay",
    "unit_bytes",
    "unit_bytes_or_bound",
    "unit_coefficients",
]
