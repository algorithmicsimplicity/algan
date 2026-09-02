"""``torch.compile`` for the render and batch-fetch pipeline, behind one switch.

Algan's per-frame arithmetic -- timeline materialization, projection, vertex
shading, the sheet compaction, post-processing -- is written as chains of small
torch operations. Eager torch pays a dispatch and a full memory round-trip for
every one of them, and on the CPU render path that overhead is a large share of
what a frame costs. ``torch.compile`` fuses such a chain into one kernel.

Every pipeline function that opts in does so through :func:`compiled`, and the
decorator resolves **at call time** against ``SETTINGS.computing.torch_compile``
(:func:`torch_compile_enabled`), so the switch works between two renders in one
process and a warm daemon adopts a client's value. Off, a decorated function is
its eager self with one attribute lookup of overhead.

The decorator is also the fallback. A compile that fails -- an unsupported
platform, a missing C++ compiler, an operator the backend cannot lower --
warns once, naming the function and the reason, and runs that function eagerly
from then on; it never fails a render. A function wrapped here must therefore
be safe to *re-run* after a partial execution: pure, or writing only outputs
computed from unmodified inputs, so that an eager retry after a mid-function
compile failure cannot apply an in-place update twice.

Two measured rules for what goes inside a compiled region, on top of the
re-runnability above. **No division by a float constant, and no
``F.normalize``, in a region whose result feeds a threshold.** Inductor
rewrites ``x / 3.0`` as ``x * (1/3.0)`` unconditionally
(``torch/_inductor/lowering.py``, ``div_prim``), which the CPU backend's
``-fno-unsafe-math`` codegen does not cover, so such a region is an ulp off
its eager arm; the PN level searches turn that ulp into a whole subdivision
level, measured as 118 channel values on a render suite that allows 2
(``rendering/logical_pn.py`` has the account). Products, sums, gathers,
``where`` and division by a *tensor* are exact. And **leave ``dynamic`` at its
default unless the trailing extents are large**: ``dynamic=True`` makes every
axis symbolic, the CPU backend cannot vectorize an innermost loop of unknown
length, and a ``[.., 3]`` chain measured up to 30x slower than with
automatic dynamic shapes, which mark only the axis that actually moved.

What ``'auto'`` (the default) resolves to is :func:`torch_compile_support`:
on wherever ``torch.compile`` runs, off on Windows and on a Python that Dynamo
does not support. Setting the field to ``True`` skips that check and tries
anyway, falling back per function as above; ``False`` is off everywhere.
``ALGAN_TORCH_COMPILE`` overrides the field, which is how an A/B script flips
arms between two renders in one process.
"""

from __future__ import annotations

import contextlib
import functools
import sys
import threading
import warnings

import torch

from algan.environment import env_flag
from algan.errors import AlganWarning

#: Backend handed to ``torch.compile``. Module-level so a test can point the
#: whole mechanism at ``"eager"`` (Dynamo tracing with no code generation) and
#: exercise the wrapper in milliseconds rather than paying an Inductor build.
_BACKEND = "inductor"

#: Dynamo recompiles a function for every distinct shape/guard set it meets
#: and gives up -- silently falling back to eager -- past this many. The
#: default of 8 is tight for a pipeline whose tensors change shape from batch
#: to batch before automatic dynamic shapes catch up; a modest raise keeps the
#: compiled arm engaged. Applied once, when the first function compiles.
_RECOMPILE_LIMIT = 64

_lock = threading.RLock()
_SUPPORT = None
_CONFIGURED = False
#: Every :class:`_CompiledFunction` created, for diagnostics
#: (:func:`compiled_functions`).
_REGISTRY: list[_CompiledFunction] = []


def torch_compile_support() -> tuple[bool, str]:
    """Whether ``torch.compile`` can run in this process, and why not.

    Returns ``(supported, reason)``; ``reason`` is empty when supported. The
    probe is cheap and memoized: none of what it checks changes within a
    process.
    """
    global _SUPPORT
    if _SUPPORT is not None:
        return _SUPPORT
    _SUPPORT = _probe_support()
    return _SUPPORT


def _probe_support() -> tuple[bool, str]:
    if sys.platform == "win32":
        # Inductor's CPU backend needs MSVC set up on PATH and a C++ toolchain
        # Algan cannot vouch for; Dynamo itself only gained Windows support
        # recently. A platform the switch defaults off on, not one it refuses.
        return False, "torch.compile is not supported on Windows"
    if not hasattr(torch, "compile"):
        return False, f"torch {torch.__version__} has no torch.compile"
    try:
        from torch._dynamo.eval_frame import check_if_dynamo_supported

        check_if_dynamo_supported()
    except Exception as exc:  # noqa: BLE001 -- any refusal means "no"
        return False, str(exc).strip() or "torch._dynamo is unavailable"
    return True, ""


def torch_compile_enabled() -> bool:
    """Whether pipeline functions run compiled right now.

    ``SETTINGS.computing.torch_compile`` decides; its default ``'auto'``
    resolves to :func:`torch_compile_support`. ``ALGAN_TORCH_COMPILE``
    overrides both. Call it; never bind the result at import time -- the
    setting is live and a script may flip it between renders.
    """
    from algan.settings import SETTINGS

    configured = SETTINGS.computing.torch_compile
    if configured == "auto":
        configured = torch_compile_support()[0]
    return env_flag("ALGAN_TORCH_COMPILE", bool(configured))


def _configure_dynamo_once():
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True
    try:
        import torch._dynamo.config as dynamo_config

        for name in ("recompile_limit", "cache_size_limit"):
            if hasattr(dynamo_config, name) and getattr(dynamo_config, name) < (
                _RECOMPILE_LIMIT
            ):
                setattr(dynamo_config, name, _RECOMPILE_LIMIT)
    except Exception:  # noqa: BLE001 -- a config knob is not worth a render
        pass


def _is_compile_failure(exc: BaseException) -> bool:
    """Whether ``exc`` came from Dynamo/Inductor rather than the function body.

    An error the function itself raises when run must propagate exactly as
    eager execution would raise it; only the compiler's own refusals are
    grounds for the eager fallback.
    """
    try:
        from torch._dynamo.exc import TorchDynamoException

        if isinstance(exc, TorchDynamoException):
            return True
    except Exception:  # noqa: BLE001
        pass
    try:
        from torch._inductor.exc import InductorError

        if isinstance(exc, InductorError):
            return True
    except Exception:  # noqa: BLE001
        pass
    module = type(exc).__module__ or ""
    return module.startswith(("torch._dynamo", "torch._inductor", "torch.fx"))


class _CompiledFunction:
    """One pipeline function, compiled lazily and demoted to eager on failure."""

    __slots__ = ("compiled", "failed", "fn", "name", "options", "reason")

    def __init__(self, fn, options):
        self.fn = fn
        self.name = f"{fn.__module__}.{fn.__qualname__}"
        self.options = options
        self.compiled = None
        self.failed = False
        self.reason = ""

    def _get_compiled(self):
        with _lock:
            if self.compiled is None:
                _configure_dynamo_once()
                self.compiled = torch.compile(self.fn, backend=_BACKEND, **self.options)
            return self.compiled

    def _demote(self, exc):
        with _lock:
            if self.failed:
                return
            self.failed = True
            # The first two non-empty lines: Dynamo's wrapper exception says
            # only "backend raised:" on its first line and names the cause on
            # the next.
            lines = [line for line in str(exc).strip().splitlines() if line.strip()]
            self.reason = f"{type(exc).__name__}: " + " ".join(lines[:2])
        warnings.warn(
            f"torch.compile failed for {self.name}; it runs eagerly for the rest "
            f"of this process ({self.reason}). Set "
            "SETTINGS.computing.torch_compile=False to silence this, or see "
            "the torch.compile documentation for the backend requirement it "
            "names.",
            AlganWarning,
            stacklevel=4,
        )

    def __call__(self, *args, **kwargs):
        if self.failed or not torch_compile_enabled():
            return self.fn(*args, **kwargs)
        try:
            compiled = self._get_compiled()
        except Exception as exc:  # noqa: BLE001 -- see _demote
            self._demote(exc)
            return self.fn(*args, **kwargs)
        try:
            return compiled(*args, **kwargs)
        except Exception as exc:
            if not _is_compile_failure(exc):
                raise
            self._demote(exc)
            return self.fn(*args, **kwargs)


def compiled(fn=None, *, dynamic=None, fullgraph=False, mode=None, options=None):
    """Compile a pipeline function with ``torch.compile`` when the switch is on.

    Usable bare (``@compiled``) or with ``torch.compile``'s own keywords
    (``@compiled(dynamic=True)``). The returned wrapper keeps the eager function
    as its ``eager`` attribute, so a parity test can call both arms.

    ``dynamic=None`` (the default) lets Dynamo specialize on the first shapes
    it sees and mark only the axes that then move as symbolic, which suits
    tensors whose leading size moves from batch to batch while their trailing
    ``[.., 3]`` extents stay put. ``dynamic=True`` makes every axis symbolic
    and costs the CPU backend its vectorized inner loop (the module docstring
    has the measurement); reserve it for regions whose trailing extents are
    large, such as a frame buffer.
    """

    def decorate(function):
        compile_options = {"dynamic": dynamic, "fullgraph": fullgraph}
        if mode is not None:
            compile_options["mode"] = mode
        if options is not None:
            compile_options["options"] = options
        state = _CompiledFunction(function, compile_options)
        with _lock:
            _REGISTRY.append(state)

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            return state(*args, **kwargs)

        wrapper.eager = function
        wrapper._algan_compiled = state
        return wrapper

    if fn is None:
        return decorate
    return decorate(fn)


def compiled_functions():
    """Diagnostics: ``(name, state)`` for every function wrapped so far.

    ``state`` is ``"compiled"`` once a compiled callable exists, ``"eager"``
    while it does not, or ``"failed: <reason>"`` after the fallback fired.
    """
    with _lock:
        records = list(_REGISTRY)
    out = []
    for record in records:
        if record.failed:
            state = f"failed: {record.reason}"
        elif record.compiled is not None:
            state = "compiled"
        else:
            state = "eager"
        out.append((record.name, state))
    return out


def reset_compiled_functions():
    """Forget every compiled callable and fallback verdict (tests only).

    The next call of each function compiles afresh. Also resets Dynamo's own
    caches so a test that changed the backend sees the change.
    """
    with _lock:
        for record in _REGISTRY:
            record.compiled = None
            record.failed = False
            record.reason = ""
    with contextlib.suppress(Exception):
        torch._dynamo.reset()


__all__ = [
    "compiled",
    "compiled_functions",
    "reset_compiled_functions",
    "torch_compile_enabled",
    "torch_compile_support",
]
