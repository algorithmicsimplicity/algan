"""The kernel compiler Algan's ``*_taichi`` modules are written against.

Algan's kernels are authored in the Taichi language. That language is served by
two interchangeable implementations:
`Quadrants <https://github.com/Genesis-Embodied-AI/quadrants>`_ -- **the one
Algan installs and compiles with by default** -- and
`taichi <https://github.com/taichi-dev/taichi>`_ 1.7.x, the dormant upstream
Quadrants forked from. Both expose the same ``ti.kernel`` / ``ti.func`` /
``ti.template`` surface Algan uses, so the engine does not care which one
compiled a kernel -- but it must not end up with *both* live in one process,
each with its own runtime, CUDA context and kernel cache.

Every module that needs the compiler therefore imports it from here::

    from algan.taichi_compat import ti

rather than ``import taichi as ti``, so the choice is made exactly once, and a
mixed process is unrepresentable rather than merely discouraged. Reach a
submodule through :func:`submodule` for the same reason -- ``import
taichi.lang.impl`` names one implementation in the import statement itself.
That applies to tests and benchmarks as much as to ``algan/``: a test that says
``import taichi as ti`` to declare a ``@ti.func`` is compiling that function
with a second compiler inside a process the engine is running on the first.

Selection
---------
``ALGAN_TAICHI_BACKEND`` picks the implementation: ``quadrants`` (the default)
or ``taichi``. It is a startup variable, and unusually strictly so: the backend
module is bound on first use and every kernel in the process is then compiled by
it, so nothing can re-select it afterwards and the render daemon refuses a client
whose value differs rather than serving it from the wrong compiler.

Taichi stays a fully supported arm rather than a legacy spelling, for two
reasons that have not expired: it is the A/B control every "did the compiler do
this?" question is answered against (``taichi_patches/PLAN.md`` §6.1 is that
comparison), and it is still the only compiler with a *patched* Metal wheel --
``taichi_patches/`` plus ``.github/workflows/taichi_build.yaml`` build one, and
the zero-copy MPS path in :mod:`algan.rendering.mps_zero_copy` needs it. Only
Quadrants is a declared runtime dependency; ``pip install algan[taichi]``
installs the other arm.

What binding does
-----------------
Binding is also the one moment that is guaranteed to precede the first
``@ti.func`` in the process, so two things that must happen before then happen
here rather than in ``algan/__init__.py``:

* **Taichi's version check is switched off.** taichi 1.7.x's
  ``_version_check.py`` contacts a metadata server at import (a background
  thread, once a day); Quadrants deleted the module. ``TI_SKIP_VERSION_CHECK``
  is the compiler's own opt-out, set with ``setdefault`` so a user who wants the
  check can still ask for it, and set only on the taichi arm because it means
  nothing to the other. It is not an ``ALGAN_`` variable, so it is not declared
  in :mod:`algan.environment` -- like ``ENABLE_TAICHI_HEADER_PRINT`` and
  ``TI_OFFLINE_CACHE_FILE_PATH``, it belongs to the compiler.
* **The warm-start patches are installed** (:func:`algan.utils.taichi_warmstart.apply`).
  One of them replaces the decorators' ``_inside_class`` frame walk, which runs
  when a kernel or func is *defined*, so it has to be in place before
  ``algan.scene_manager`` pulls the kernel modules in -- which is earlier than
  ``algan/__init__.py`` gets to run anything of its own after the compiler is
  up.

Backend differences
-------------------
The two are not drop-in equal everywhere, and the differences that matter are
handled by their owners rather than papered over here:

* :mod:`algan.utils.taichi_warmstart` memoizes each compiler's frontend and
  :mod:`algan.utils.taichi_fast_launch` replaces each compiler's
  ``Kernel.__call__`` with a plan-caching dispatcher; both carry a patch per
  implementation (the launch paths differ: quadrants' takes integer argument
  indices, batches its scalar setters and compiles before the launch rather
  than in it). Each checks the backend and its version itself, and both
  report a version gate they refused to fire through ``algan check`` rather
  than no-opping in silence.
* Each reads its own environment prefix -- ``TI_*`` on taichi, ``QD_*`` on
  Quadrants -- so a ``TI_`` variable Algan honours (``TI_OFFLINE_CACHE_FILE_PATH``)
  has to be carried to Quadrants as an ``init`` kwarg;
  :func:`algan.rendering.taichi_runtime.taichi_init_kwargs` does that.
* Quadrants renamed parts of ``Kernel`` (``materialize``'s ``args`` parameter is
  ``py_args``; ``compiled_kernels`` is ``materialized_kernels``), which is why
  code touching those goes through :data:`BACKEND` -- or, for that last one,
  :func:`kernel_specializations` -- rather than assuming.
* ``get_runtime().prog`` is ``None`` before ``init`` on taichi and *raises* on
  Quadrants. Ask :func:`program` instead of reading the attribute.
* A ``@ti.func`` is marked ``_is_taichi_function`` by one and
  ``_is_quadrants_function`` by the other. Ask :func:`is_compiler_func`; a
  hard-coded spelling answers ``False`` on the other backend rather than
  failing, which is a silent fall back to the Python path.
* Quadrants' template mapper caches a launch's specialization key against weak
  references to its arguments, and warns "Template mapper caching disabled"
  for any argument it cannot weakly reference. A compiler dtype is one, so a
  dtype passed as a ``ti.template()`` argument goes through
  :func:`template_dtype` first. The other is a tuple -- the injected
  fragment-pipeline tuples -- which cannot be made weakly referenceable at all;
  that one warning is filtered, from the moment this module is imported
  (:func:`_install_backend_warning_filters` says why it is harmless). Not when
  ``ti`` is bound, which usually happens inside the first render: a filter
  added inside a caller's ``warnings.catch_warnings()`` block is discarded
  when the block exits. pytest runs every test, and its own start-up, inside
  such blocks, so ``tests/conftest.py`` gives it the same filter as an ini line.
"""

from __future__ import annotations

import importlib
import os
import warnings
import weakref
from types import ModuleType

from algan.environment import env_str

#: The implementations this layer can bind. **The first is the default** --
#: ``BACKENDS[0]`` is what ``_select_backend`` falls back to -- so this tuple's
#: order is the choice, not a listing convention. Quadrants leads because it is
#: what ``pyproject.toml`` installs: it is the maintained one, it builds on a
#: current macOS runner where taichi 1.7.4 no longer does, and it renders
#: byte-identically (``taichi_patches/PLAN.md`` §6.1).
BACKENDS = ("quadrants", "taichi")


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


_QUADRANTS_BENIGN_TEMPLATE_CACHE_WARNING = (
    r"cannot create weak reference to 'tuple' object\. "
    r"Template mapper caching disabled\."
)


def _install_backend_warning_filters():
    """Hide the one Quadrants warning Algan knowingly triggers and accepts.

    The kernels that inject fragment pipelines take them as a tuple
    ``ti.template()`` argument, and a tuple cannot be weakly referenced, so
    Quadrants' template mapper warns that it will not cache those launches'
    specialization keys. That cache is not one Algan uses: on a repeat launch
    :mod:`algan.utils.taichi_fast_launch` replays a plan of its own and never
    consults the mapper, and on the compiler's own path the cache is keyed by
    the ``id()`` of every argument, which never repeats for Algan's freshly
    built tensors -- with the fast launcher off, a 3-D mesh, shadow, square and
    text render made 293 lookups and the cache answered none of them outside
    one gloss kernel, tuple or no tuple. The filter matches that exact message,
    so a different argument losing the cache (a dtype not passed through
    :func:`template_dtype`, say) still surfaces.
    """
    if BACKEND != "quadrants":
        return
    warnings.filterwarnings(
        "ignore",
        message=_QUADRANTS_BENIGN_TEMPLATE_CACHE_WARNING,
        category=UserWarning,
        module=r"quadrants\._test_tools\.warnings_helper",
    )


_install_backend_warning_filters()

#: ``{dtype: weakly referenceable handle}``, one per compiler dtype, for the
#: life of the process; see :func:`template_dtype`.
_TEMPLATE_DTYPES = {}
_TEMPLATE_DTYPE_CLASSES = {}


def template_dtype(dtype):
    """``dtype`` in the form to pass as a ``ti.template()`` argument.

    Quadrants' template mapper remembers a launch's specialization key against
    weak references to its arguments, and a compiler dtype (``DataTypeCxx``)
    cannot be weakly referenced: every kernel given one as a template argument
    warned ``cannot create weak reference to '...DataTypeCxx' object. Template
    mapper caching disabled`` and re-derived its whole key on each launch that
    went through the compiler's own path.

    The handle returned is an instance of a subclass of ``dtype``'s class that
    can be weakly referenced and is otherwise the same dtype: equal to it,
    hashing like it, with the same ``to_string()``. So everything keyed on the
    dtype keys on the handle identically -- the kernel's specialization, its
    compiled code, its source key and its precompile spec -- and a kernel given
    a handle where it used to be given the dtype compiles nothing new. There is
    one handle per dtype, kept for the process, so the mapper's ``id()``-keyed
    entry stays valid. Any other backend, and anything that is not a dtype or
    is already weakly referenceable, is returned unchanged.
    """
    if BACKEND != "quadrants":
        return dtype
    try:
        return _TEMPLATE_DTYPES[dtype]
    except (KeyError, TypeError):
        pass
    try:
        weakref.ref(dtype)
        return dtype
    except TypeError:
        pass
    base = type(dtype)
    try:
        cls = _TEMPLATE_DTYPE_CLASSES.get(base)
        if cls is None:
            cls = _TEMPLATE_DTYPE_CLASSES.setdefault(
                base,
                type(
                    f"Template{base.__name__}",
                    (base,),
                    {"__slots__": ("__weakref__",), "__module__": __name__},
                ),
            )
        handle = cls(dtype.get_ptr())
        if handle != dtype or hash(handle) != hash(dtype):
            return dtype
    except Exception:  # noqa: BLE001 -- a warning is the worst a miss costs
        return dtype
    return _TEMPLATE_DTYPES.setdefault(dtype, handle)


def __getattr__(name):
    """Bind ``ti`` on first use (PEP 562).

    Importing the backend costs seconds, and Algan imports its kernel modules
    lazily to avoid paying it on a run that never renders (a CLI ``--help``, a
    timeline-only test). Eagerly importing it here would have moved that cost
    back to ``import algan`` for every module that reads :data:`BACKEND`.
    ``from algan.taichi_compat import ti`` still pays it at that module's import,
    exactly as ``import taichi as ti`` used to.

    See "What binding does" in the module docstring for the two side effects.
    ``ti`` is bound *before* the warm-start installer runs, so a module the
    installer imports that itself asks for ``ti`` gets the bound module rather
    than re-entering here.
    """
    if name == "ti":
        if BACKEND == "taichi":
            os.environ.setdefault("TI_SKIP_VERSION_CHECK", "1")
        module = importlib.import_module(BACKEND)
        globals()["ti"] = module
        from algan.utils.taichi_warmstart import apply as _apply_warmstart

        _apply_warmstart()
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


#: Attribute on a ``Kernel`` holding its declared parameters, in order. Taichi
#: spells it ``arguments``; Quadrants renamed it to ``arg_metas``. The entries
#: are the same shape either way -- ``.annotation``, ``.name``, ``.default``.
KERNEL_ARGUMENTS_ATTR = "arguments" if BACKEND == "taichi" else "arg_metas"


def kernel_arguments(kernel) -> list:
    """``kernel``'s declared parameters, in declaration order.

    Reading the *annotations* is how the MPS zero-copy path decides which
    arguments it may adopt, so getting this wrong is not an error: an empty
    list means "import nothing", which is a silent fall back to the staging
    path (:mod:`algan.rendering.mps_zero_copy`).
    """
    return getattr(kernel, KERNEL_ARGUMENTS_ATTR)


#: Marker attribute the ``ti.func`` decorator leaves on what it wraps. Taichi
#: spells it ``_is_taichi_function``; Quadrants renamed it to
#: ``_is_quadrants_function`` (its wrapper class is ``QuadrantsCallable``).
#: ``ti.kernel``'s markers -- ``_is_wrapped_kernel``, ``_is_classkernel`` -- are
#: the same on both and need no indirection.
FUNC_MARKER_ATTR = (
    "_is_taichi_function" if BACKEND == "taichi" else "_is_quadrants_function"
)


def is_compiler_func(obj) -> bool:
    """Whether ``obj`` is a ``@ti.func`` built by the bound compiler.

    A background callable, a fragment-shader stage and a profiled attribute are
    all "a plain Python callable, or one of these". Asking here rather than for
    the attribute keeps a caller from silently answering ``False`` for every
    such object on the backend it was not written against -- which is what a
    hard-coded ``_is_taichi_function`` does under Quadrants: the deferred
    background stops being recognised as a kernel function and is evaluated in
    Python instead, once per pixel.
    """
    return bool(getattr(obj, FUNC_MARKER_ATTR, False))


#: Attribute on the compiler's runtime object holding the live ``Program``.
#: Taichi spells it ``prog`` and leaves it ``None`` until ``init``; Quadrants
#: renamed it to ``_prog`` and made ``prog`` a property that *raises* when it is
#: unset, so "is a program up?" has to be asked of the private name there.
PROGRAM_ATTR = "prog" if BACKEND == "taichi" else "_prog"


def program():
    """The compiler's live ``Program``, or ``None`` if it has not started one.

    The identity is meaningful: a re-``init`` builds a new ``Program`` and drops
    every kernel compiled against the old one, so callers compare the object
    rather than just testing for ``None``.
    """
    return getattr(submodule("lang.impl").get_runtime(), PROGRAM_ATTR, None)


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
    "FUNC_MARKER_ATTR",
    "KERNEL_ARGUMENTS_ATTR",
    "KERNEL_SPECIALIZATIONS_ATTR",
    "PROGRAM_ATTR",
    "backend_version",
    "describe_backend",
    "is_compiler_func",
    "kernel_arguments",
    "kernel_specializations",
    "program",
    "submodule",
    "template_dtype",
    "ti",  # noqa: F822  -- bound lazily by __getattr__ above, not at module level.
]
