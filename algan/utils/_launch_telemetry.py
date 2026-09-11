"""Opt-in Quadrants launch accounting, including original-path cache hits.

No arrays, kernels or launch contexts are retained by the report. A thread-local
stack attributes nested/compiler and batch-prep launches to their own calls;
only report updates take a lock. Enable/disable/reset between render jobs.
"""

from __future__ import annotations

import threading
from collections import Counter
from contextlib import contextmanager

ENABLED = False
_CACHE_CLASS = None
_LOCAL = threading.local()
_LOCK = threading.Lock()
_ROWS = {}
_OUTCOMES = ("fast", "quadrants_cache", "cold", "fallback", "error")


def configure(cache_class):
    """Register the supported compiler's cache without wrapping its hot path."""
    global _CACHE_CLASS
    _CACHE_CLASS = cache_class


def set_enabled(enabled=True):
    """Enable accounting; the cache observer is installed only on first use."""
    global ENABLED
    if enabled and _CACHE_CLASS is None:
        raise RuntimeError("launch telemetry requires the Quadrants fast dispatcher")
    if enabled and not getattr(_CACHE_CLASS, "_algan_observed", False):
        original = _CACHE_CLASS.populate_launch_ctx_from_cache

        def observed(self, *args, **kwargs):
            hit = original(self, *args, **kwargs)
            if ENABLED:
                stack = getattr(_LOCAL, "stack", ())
                if stack and stack[-1]["cache"] is self:
                    stack[-1]["cache_hit"] |= bool(hit)
            return hit

        _CACHE_CLASS.populate_launch_ctx_from_cache = observed
        _CACHE_CLASS._algan_observed = True
    ENABLED = bool(enabled)


def _identity(kernel):
    func = kernel.func
    return f"{func.__module__}.{func.__qualname__}", str(kernel.runtime._arch)


def _record(identity, outcome, reason=None):
    with _LOCK:
        row = _ROWS.setdefault(identity, {"counts": Counter(), "reasons": Counter()})
        row["counts"][outcome] += 1
        if reason is not None:
            row["reasons"][reason] += 1


def fast(kernel, *, error=False):
    _record(
        _identity(kernel),
        "error" if error else "fast",
        "fast_launch" if error else None,
    )


@contextmanager
def original(kernel, reason):
    """Observe exactly one original call; the dispatcher marks recorded cold plans."""
    stack = getattr(_LOCAL, "stack", None)
    if stack is None:
        stack = _LOCAL.stack = []
    event = {
        "cache": kernel.launch_context_buffer_cache,
        "cache_hit": False,
        "outcome": "fallback",
        "reason": reason,
    }
    identity = _identity(kernel)
    stack.append(event)
    try:
        yield event
    except BaseException:
        event["outcome"] = "error"
        raise
    finally:
        stack.pop()
        outcome = event["outcome"]
        if outcome != "error" and event["cache_hit"]:
            outcome = "quadrants_cache"
        _record(
            identity,
            outcome,
            event["reason"] if outcome in ("fallback", "error") else None,
        )


def report(*, reset=False):
    """Return a JSON-serializable snapshot, optionally clearing only the counts."""
    with _LOCK:
        rows = []
        totals = Counter()
        for (kernel, arch), data in sorted(_ROWS.items()):
            counts = {name: data["counts"][name] for name in _OUTCOMES}
            totals.update(counts)
            rows.append(
                {
                    "kernel": kernel,
                    "arch": arch,
                    **counts,
                    "reasons": dict(data["reasons"]),
                }
            )
        result = {"kernels": rows, "totals": {name: totals[name] for name in _OUTCOMES}}
        if reset:
            _ROWS.clear()
        return result
