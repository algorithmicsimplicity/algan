"""The kernel compiler Algan's ``*_taichi`` modules are written against.

Algan's kernels are authored in the Taichi language. That language is served by
two interchangeable implementations: `taichi <https://github.com/taichi-dev/taichi>`_
1.7.x, and `Quadrants <https://github.com/Genesis-Embodied-AI/quadrants>`_, a
fork of it that is still maintained. Both expose the same ``ti.kernel`` /
``ti.func`` / ``ti.template`` surface Algan uses, so the engine does not care
which one compiled a kernel -- but it must not end up with *both* live in one
process, each with its own runtime, CUDA context and kernel cache.

Every module that needs the compiler therefore imports it from here::

    from algan.taichi_compat import ti

rather than ``import taichi as ti``, so the choice is made exactly once, and a
mixed process is unrepresentable rather than merely discouraged. Reach a
submodule through :func:`submodule` for the same reason -- ``import
taichi.lang.impl`` names one implementation in the import statement itself.

Selection
---------
``ALGAN_TAICHI_BACKEND`` picks the implementation: ``taichi`` (the default) or
``quadrants``. It is a startup variable, and unusually strictly so: the backend
module is bound on first use and every kernel in the process is then compiled by
it, so nothing can re-select it afterwards and the render daemon refuses a client
whose value differs rather than serving it from the wrong compiler.

Backend differences
-------------------
The two are not drop-in equal everywhere, and the differences that matter are
handled by their owners rather than papered over here:

* :mod:`algan.utils.taichi_fast_launch` and :mod:`algan.utils.taichi_warmstart`
  patch compiler internals and check the version themselves, so they no-op on a
  backend they do not recognise.
* Quadrants renamed parts of ``Kernel`` (``materialize``'s ``args`` parameter is
  ``py_args``; ``compiled_kernels`` is ``materialized_kernels``), which is why
  code touching those goes through :data:`BACKEND` rather than assuming.
"""

from __future__ import annotations

import importlib
from types import ModuleType

from algan.environment import env_str

#: The implementations this layer can bind, in the order they are tried by name.
BACKENDS = ("taichi", "quadrants")


def _select_backend():
    """The backend name ``ALGAN_TAICHI_BACKEND`` asks for.

    An unrecognised value raises rather than falling back: silently compiling
    against a different implementation than the one that was asked for is the
    single failure this module exists to prevent.
    """
    name = env_str("ALGAN_TAICHI_BACKEND", BACKENDS[0]).strip().lower()
    if name not in BACKENDS:
        raise ValueError(
            f"ALGAN_TAICHI_BACKEND={name!r} is not a known kernel compiler; "
            f"expected one of {', '.join(BACKENDS)}."
        )
    return name


#: Which implementation this process compiles its kernels with, as a string.
#: Reading this does **not** import the backend, so a module that only needs to
#: know which one was chosen (``algan.settings._startup``, picking the kernel
#: cache directory) stays off the import cost.
BACKEND = _select_backend()


def __getattr__(name):
    """Bind ``ti`` on first use (PEP 562).

    Importing the backend costs seconds, and Algan imports its kernel modules
    lazily to avoid paying it on a run that never renders (a CLI ``--help``, a
    timeline-only test). Eagerly importing it here would have moved that cost
    back to ``import algan`` for every module that reads :data:`BACKEND`.
    ``from algan.taichi_compat import ti`` still pays it at that module's import,
    exactly as ``import taichi as ti`` used to.
    """
    if name == "ti":
        module = importlib.import_module(BACKEND)
        globals()["ti"] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def submodule(name: str) -> ModuleType:
    """Import ``name`` from the bound backend, e.g. ``submodule("lang.impl")``.

    The Taichi-facing equivalent of ``from taichi.lang import impl``, written so
    the backend is not named in the import statement.
    """
    return importlib.import_module(f"{BACKEND}.{name}")


#: Attribute on a ``Kernel`` holding its compiled specializations, keyed by spec
#: key. Taichi 1.7 spells it ``compiled_kernels``; Quadrants renamed it to
#: ``materialized_kernels``. Code that reads it should go through
#: :func:`kernel_specializations` rather than hard-coding either spelling.
KERNEL_SPECIALIZATIONS_ATTR = (
    "compiled_kernels" if BACKEND == "taichi" else "materialized_kernels"
)


def kernel_specializations(kernel) -> dict:
    """The ``{spec key: compiled kernel}`` mapping ``kernel`` has materialized.

    The mapping is the backend's own live dict, not a copy: a caller may test
    membership before a ``materialize`` call and read the new entry after it.
    """
    return getattr(kernel, KERNEL_SPECIALIZATIONS_ATTR)


def backend_version() -> tuple:
    """The bound backend's version tuple, or ``()`` if it does not report one."""
    return tuple(getattr(importlib.import_module(BACKEND), "__version__", ()) or ())


def describe_backend() -> str:
    """``"taichi 1.7.4"`` -- the backend and version, for logs and ``algan info``."""
    version = ".".join(str(part) for part in backend_version())
    return f"{BACKEND} {version}" if version else BACKEND


__all__ = [
    "BACKEND",
    "BACKENDS",
    "backend_version",
    "describe_backend",
    "submodule",
    "ti",  # noqa: F822  -- bound lazily by __getattr__ above, not at module level.
]
