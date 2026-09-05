r"""A source-keyed index over the offline kernel cache, so a warm hit skips the AST transform.

The problem
-----------
Even with every kernel artifact on disk, a process pays the whole Python
frontend per kernel before it can *look up* the cache: the C++ offline-cache
key is a SHA-256 over the frontend IR, and building that IR walks the kernel's
AST in Python -- twice on Quadrants, a pruning pass then an enforcing pass. For
Algan's megakernels that is ~12 s of a 12.5 s warm ``save_frame``
(``taichi_patches/PLAN.md`` §6.1), and none of it is spent on anything the
previous run did not already compute.

Quadrants ships the mechanism that fixes this, "fastcache"
(``@qd.kernel(fastcache=True)``, ``Kernel._try_load_fastcache``): a Python-side
key over the kernel's source text and argument types maps to the C++ cache key
of the IR that source last produced; on a hit the kernel data is loaded by that
key directly and the transform walks only the ``FunctionDef`` (declaring
parameters) and never the body. Its *key* cannot be used for Algan: it is
gated on a purity check that forbids captured module globals, and Algan's
kernels read a ``ti.static`` module constant in nearly every branch; and its
argument hasher fails closed on a ``ti.template()`` slot holding a tuple of
``@ti.func`` stages or a dtype, which is exactly the templated megakernels.

So this module computes **Algan's own key** and reuses everything else of
Quadrants' -- the on-disk index (``src_hasher.load``/``store``, which also
re-hashes every transitively visited ``@ti.func`` on load so an edited func
invalidates), the C++ load (``Program.load_fast_cache``) and the
``FunctionDef``-only transform.

Soundness argument
------------------
The C++ key is a pure function of the frontend IR; the frontend IR is a pure
function of (a) the compiler, (b) the compile config and device caps, (c) the
kernel's source, (d) every ``@ti.func`` it inlines, (e) the argument *types*
(dtype, ndim, grad, layout, boundary; element shape for matrix-typed arrays),
(f) the template argument *values*, and (g) every Python value the kernel body
and its callees read while being transformed -- the captured globals that
``ti.static`` gates, loop bounds and constants come from. A key that covers all
seven is sound: two processes with equal keys produce equal IR, so serving the
artifact stored under the key is what a full transform would have done.

(a) is ``quadrants.__version_str__`` plus Quadrants' own cache-value schema
version. (b) is a hash over every ``CompileConfig`` field except the print/
verbose/offline-cache ones and the process-local Metal queue address, plus
every ``DeviceCapability`` the program reports. (c) is the kernel source hash,
file and start line, exactly as Quadrants keys it. (d) is Quadrants' own: the
visited-function hashes stored beside the C++ key and re-validated on every
load. (e) is derived from the torch tensor and the annotation, at least as
finely as the template mapper's features. (f) hashes each template value by
**value with a type tag**: numbers, strings, ``None`` by ``repr``; dtypes by
their name; tuples elementwise (nested allowed); functions by source hash *and
by the closure walk below*. (g) is what makes source hashing sound rather than
heuristic: the kernel's code object is disassembled (``LOAD_GLOBAL``,
``LOAD_DEREF`` and import-bound locals, each with its ``LOAD_ATTR`` chain, in
nested code objects too), every chain is resolved to the live value it names
and hashed by the same value rules, and every function reached -- an inlined
``@ti.func``, a Python helper called under ``ti.static`` -- is walked the same
way, transitively, with a visited set. A **class** reached in kernel scope
(``ti.static(ArenaView(...))`` leaves the bare class as the chain's leaf) is
hashed by its source *and by every member by value*: the source hash cannot see
a class attribute assigned after the body ran, nor the globals a ``@ti.func``
member reads. The compiler's own namespace is exempt (its version is already
in the key) and so are builtins. Anything the rules do not know -- an instance
whose attributes are not accessed, a module used without an attribute, a name
that does not resolve -- **poisons the key**: the kernel is counted, warned
about once, and takes the full transform with nothing ever stored under an
unsound key. Two belts over (g): a fingerprint of every declared ``ALGAN_``
variable that can reach a kernel, and a serialization of ``SETTINGS.raytracing``
(with ``.experimental``) and ``SETTINGS.computing``, so a toggle read through a
path the walk cannot follow (a local ``import`` of the settings module inside
a helper is followed; ``getattr`` by string is not) still changes the key.

Known gaps -- what the key does not see:

* **Dynamic attribute access.** ``getattr(obj, name)``, ``globals()[name]`` and
  a value read through a local that was assigned from a global (``s =
  SETTINGS.raytracing; s.x``) are not chains the disassembly can follow: the
  walk stops at the root and hashes *that* by value. So the read is covered
  when the root is a container or a class (hashed by content) and **poisons**
  when it is an instance or a module -- what it never does is silently miss the
  attribute. ``globals()[name]`` and a value routed through an object the rules
  cannot render both land on the poison side.
* **Values mutated in place.** A list or dict global is hashed by content at
  key time, which is what the transform sees; a mutation *during* a transform
  is out of scope on both sides.
* **Source files edited under a running process.** The kernel and func hashes
  come from what Quadrants' source retrieval returns, which
  :mod:`algan.utils.taichi_warmstart` memoizes per function object; an edit
  after first use is neither seen here nor by the compiler
  (``agent_guidance/taichi.md`` already forbids it).
* **Compiler-internal state.** Anything the frontend reads that is not a
  value reachable from the kernel -- ``impl.current_cfg()`` fields are covered
  by (b); a future compiler switch read from the environment would not be.

``ALGAN_TAICHI_SOURCE_KEY_VERIFY=1`` is the audit of all of the above: on every
validated hit the shortcut is *not* taken, the full transform and compile run,
and the C++ key they produce is compared with the one the index stored; a
mismatch raises. It was the bar for turning this on by default, and it was met
on 2026-09-05 on a GTX 1050 (sm_61) against the patched Quadrants wheel:

===============================  =====  ========  ========  ==========
``tests/full_renders``           keyed  verified  poisoned  mismatches
===============================  =====  ========  ========  ==========
CUDA                             39     39        0         0
CPU (``x64``)                    40     40        0         0
===============================  =====  ========  ========  ==========

and on CUDA all six scenes' videos came out **byte-identical** between an
index-on run and a full-transform run -- a stronger statement than the suites'
own tolerance, and the reason the shortcut half needs no separate CPU
comparison. (The CPU arm's pixel comparisons fail on that box for an unrelated
reason: the committed CPU baselines were made on Linux. The signal read there
was the verified/poisoned counts and the absence of a raise, not the
comparison.)

So this is now **on unless ``ALGAN_TAICHI_SOURCE_KEY=0``**. Turn it off if a
render's picture is ever in question -- an unsound key is a stale *kernel*, so
it shows as a wrong picture rather than an error, and the off arm is the
control. ``algan check`` reports the state, and a render under
``ALGAN_LOG_TAICHI_COMPILES=1`` prints :data:`STATS` when it finishes.

Quadrants-only. The feature rests on ``Program.load_fast_cache`` and the
``only_parse_function_def`` transform, neither of which taichi 1.7 has, so on
that backend :func:`skipped_reason` says so and nothing is installed.
"""

from __future__ import annotations

import builtins
import contextlib
import dis
import enum
import functools
import importlib
import inspect
import os
import sys
import time
import types
import warnings

from algan.environment import env_flag, env_str

#: Bumped whenever what this module puts in the key, or how it renders a
#: component, changes -- an entry written by an older rule set must miss.
_SCHEMA_VERSION = "algan-source-key-v2"

_APPLIED = False
_SKIPPED_REASON = None

#: Counters a render reports under ``ALGAN_LOG_TAICHI_COMPILES=1``.
#:
#: ``hits`` / ``misses`` count validated index lookups; ``poisoned`` counts
#: materializations whose key could not be built (they take the full path and
#: store nothing); ``verified`` counts hits that verify mode re-derived and
#: compared; ``key_seconds`` is the whole cost of building keys in this
#: process.
STATS = {
    "hits": 0,
    "misses": 0,
    "poisoned": 0,
    "verified": 0,
    "keyed": 0,
    "key_seconds": 0.0,
}

#: Kernels that poisoned, by qualified name, with the reason. Kept so a report
#: can say *which* kernels pay the full frontend and why.
POISONED = {}


def skipped_reason():
    """``None`` if the index is live, else why it is not."""
    return _SKIPPED_REASON


def stats_summary():
    """One line for a log: the counters, and the poisoned kernels if any."""
    line = (
        f"[Taichi source key] hits={STATS['hits']} misses={STATS['misses']} "
        f"poisoned={STATS['poisoned']} verified={STATS['verified']} "
        f"keyed={STATS['keyed']} key_time={STATS['key_seconds']:.3f}s"
    )
    if _PENDING_VERIFY:
        # A verify arm proves nothing about a hit whose compile never reached
        # the store hook, so say how many are still outstanding rather than
        # letting `verified` read as full coverage.
        line += f" verify_pending={len(_PENDING_VERIFY)}"
    if POISONED:
        names = ", ".join(sorted(POISONED))
        line += f"\n[Taichi source key] poisoned kernels: {names}"
    return line


def report_if_logging():
    """Print :func:`stats_summary` when compile logging is on and the index is live."""
    if _APPLIED and env_flag("ALGAN_LOG_TAICHI_COMPILES", False):
        print(stats_summary(), flush=True)


def reset_stats():
    for name in STATS:
        STATS[name] = 0.0 if name == "key_seconds" else 0
    POISONED.clear()


# ---------------------------------------------------------------------------
# Value rules
# ---------------------------------------------------------------------------


class Poison(Exception):
    """A value the key rules do not know how to render soundly."""


class _KeyContext:
    """State for one key computation: the output strings and the visited set.

    ``out`` is the ordered list of strings that will be hashed; the walk is
    deterministic, so its order is. ``visited`` holds the id of every function
    and class already emitted in this computation -- a second reference emits
    only a back-reference, which keeps mutual recursion between funcs finite.
    """

    __slots__ = ("out", "visited", "backend", "compiler_prefix")

    def __init__(self, backend):
        self.out = []
        self.visited = set()
        self.backend = backend
        self.compiler_prefix = backend + "."


def _module_name_of(obj):
    return getattr(obj, "__module__", None) or ""


def _is_compiler_object(obj, ctx):
    """Whether ``obj`` belongs to the kernel compiler's own package.

    Its version is already in the key, so its functions, classes and modules
    are rendered by name rather than walked -- ``ti.static``, ``ti.math.vec4``,
    ``ti.types.ndarray`` are the same object for every process on one version.
    """
    name = obj.__name__ if isinstance(obj, types.ModuleType) else _module_name_of(obj)
    return name == ctx.backend or name.startswith(ctx.compiler_prefix)


def _dtype_name(value):
    """The stable name of a compiler dtype (``DataTypeCxx``/``DataType``), or ``None``."""
    type_name = type(value).__name__
    if type_name in ("DataTypeCxx", "DataType"):
        to_string = getattr(value, "to_string", None)
        return to_string() if callable(to_string) else str(value)
    return None


_SIMPLE_SCALARS = (bool, int, float, complex, str, bytes)


def _hash_value(value, ctx, depth=0):
    """Append a stable rendering of ``value`` to ``ctx.out`` or raise :class:`Poison`."""
    if depth > 32:
        raise Poison("value nests deeper than 32 levels")
    out = ctx.out
    if value is None:
        out.append("None")
        return
    value_type = type(value)
    if value_type in _SIMPLE_SCALARS:
        out.append(f"{value_type.__name__}:{value!r}")
        return
    if isinstance(value, enum.Enum):
        out.append(
            f"enum:{_module_name_of(value)}.{value_type.__qualname__}.{value.name}={value.value!r}"
        )
        return
    if isinstance(value, _SIMPLE_SCALARS):  # bool/int subclasses, numpy-free
        out.append(f"{value_type.__module__}.{value_type.__qualname__}:{value!r}")
        return
    dtype_name = _dtype_name(value)
    if dtype_name is not None:
        out.append(f"dtype:{dtype_name}")
        return
    if value_type in (tuple, list):
        out.append(f"{value_type.__name__}[{len(value)}](")
        for item in value:
            _hash_value(item, ctx, depth + 1)
        out.append(")")
        return
    if value_type in (frozenset, set):
        rendered = []
        for item in value:
            sub = _KeyContext(ctx.backend)
            sub.visited = ctx.visited
            _hash_value(item, sub, depth + 1)
            rendered.append("\x1f".join(sub.out))
        out.append(f"{value_type.__name__}[{len(value)}](")
        out.extend(sorted(rendered))
        out.append(")")
        return
    if value_type is dict:
        items = []
        for key, item in value.items():
            sub = _KeyContext(ctx.backend)
            sub.visited = ctx.visited
            _hash_value(key, sub, depth + 1)
            key_text = "\x1f".join(sub.out)
            items.append((key_text, item))
        items.sort(key=lambda pair: pair[0])
        out.append(f"dict[{len(value)}](")
        for key_text, item in items:
            out.append(key_text)
            _hash_value(item, ctx, depth + 1)
        out.append(")")
        return
    type_module = _module_name_of(value_type)
    type_qualname = f"{type_module}.{value_type.__qualname__}"
    if type_qualname == "torch.dtype":
        out.append(f"torch.dtype:{value}")
        return
    if type_module == "numpy" and value_type.__name__ == "dtype":
        out.append(f"numpy.dtype:{value.str}")
        return
    if type_module.startswith("numpy") and isinstance(
        value, (int, float, complex, bool)
    ):
        out.append(f"{type_qualname}:{value!r}")
        return
    if isinstance(value, types.ModuleType):
        if _is_compiler_object(value, ctx):
            out.append(f"compiler-module:{value.__name__}")
            return
        raise Poison(
            f"module {value.__name__!r} used as a value (no attribute accessed)"
        )
    if isinstance(value, (types.BuiltinFunctionType, types.BuiltinMethodType)):
        owner = getattr(value, "__self__", None)
        owner_name = (
            owner.__name__
            if isinstance(owner, types.ModuleType)
            else type(owner).__qualname__
        )
        out.append(f"builtin:{owner_name}.{value.__name__}")
        return
    if isinstance(
        value, (types.FunctionType, types.MethodType, functools.partial)
    ) or _is_compiler_callable(value):
        _hash_callable(value, ctx, depth)
        return
    if isinstance(value, type):
        _hash_class(value, ctx, depth)
        return
    if _is_compiler_object(value, ctx):
        # A compiler-owned *instance*: a `VectorType`/`MatrixType`, an
        # `NdarrayType`, a `Layout`. Rendered from its public state, which is
        # what the transform reads off it.
        _hash_compiler_instance(value, ctx, depth)
        return
    raise Poison(f"value of type {type_qualname} has no key rule")


def _is_compiler_callable(value):
    """A ``@ti.func``/``@ti.kernel`` wrapper, on either compiler's spelling."""
    name = type(value).__name__
    return name in (
        "QuadrantsCallable",
        "BoundQuadrantsCallable",
        "TaichiCallable",
        "BoundTaichiCallable",
    )


def _hash_compiler_instance(value, ctx, depth):
    state = getattr(value, "__dict__", None)
    if state is None:
        raise Poison(
            f"compiler object {type(value).__qualname__} carries no readable state"
        )
    ctx.out.append(
        f"compiler-instance:{_module_name_of(value)}.{type(value).__qualname__}("
    )
    for name in sorted(state):
        if name.startswith("_") and name not in ("_qd_layout",):
            continue
        ctx.out.append(name)
        _hash_value(state[name], ctx, depth + 1)
    ctx.out.append(")")


def _unwrap_callable(value):
    """The raw Python function behind a compiler wrapper, bound method or partial.

    Returns ``(function, bound_self, partial_args)``; ``bound_self`` and
    ``partial_args`` are ``None`` when there is nothing of the kind.
    """
    bound_self = None
    partial_args = None
    while True:
        name = type(value).__name__
        if isinstance(value, functools.partial):
            partial_args = (value.args, value.keywords)
            value = value.func
        elif isinstance(value, types.MethodType):
            bound_self = value.__self__
            value = value.__func__
        elif name in ("BoundQuadrantsCallable", "BoundTaichiCallable"):
            bound_self = value.instance
            value = (
                value.quadrants_callable
                if hasattr(value, "quadrants_callable")
                else value.taichi_callable
            )
        elif name in ("QuadrantsCallable", "TaichiCallable"):
            value = value.fn
        else:
            return value, bound_self, partial_args


def _hash_callable(value, ctx, depth):
    function, bound_self, partial_args = _unwrap_callable(value)
    if not isinstance(function, types.FunctionType):
        raise Poison(
            f"callable of type {type(function).__qualname__} is not a Python function"
        )
    if partial_args is not None:
        ctx.out.append("partial(")
        _hash_value(partial_args[0], ctx, depth + 1)
        _hash_value(partial_args[1], ctx, depth + 1)
        ctx.out.append(")")
    if bound_self is not None:
        ctx.out.append("bound-to(")
        _hash_value(bound_self, ctx, depth + 1)
        ctx.out.append(")")
    _hash_function(function, ctx, depth)


def _hash_function(function, ctx, depth):
    """Source hash of ``function`` plus the walk of everything it references."""
    if _is_compiler_object(function, ctx):
        ctx.out.append(
            f"compiler-function:{_module_name_of(function)}.{function.__qualname__}"
        )
        return
    marker = id(function)
    if marker in ctx.visited:
        ctx.out.append(f"function-ref:{function.__qualname__}")
        return
    ctx.visited.add(marker)
    filepath, lineno, source_hash = _function_source_hash(function)
    ctx.out.append(f"function:{filepath}:{lineno}:{source_hash}")
    if function.__defaults__:
        ctx.out.append("defaults(")
        _hash_value(function.__defaults__, ctx, depth + 1)
        ctx.out.append(")")
    if function.__kwdefaults__:
        ctx.out.append("kwdefaults(")
        _hash_value(function.__kwdefaults__, ctx, depth + 1)
        ctx.out.append(")")
    _hash_references(function, ctx, depth)


def _hash_class(cls, ctx, depth):
    """A class used in kernel scope: its source, **every member by value**, its bases.

    The class-body source hash covers what the body *said*; it does not cover
    what a class attribute was later *assigned* (``Cfg.LIMIT = from_env()``,
    the same text every run), and it does not cover the globals a callable
    member reads. Both reach the IR the moment the transform reads them off the
    class -- ``ArenaView`` is read exactly this way, as the bare leaf of
    ``ti.static(ArenaView(...))`` -- so every member is hashed by value:
    callables by source plus their own reference walk, everything else by the
    value rules, which poison on anything they do not know.

    Dunder *data* is skipped: ``__dict__`` and ``__weakref__`` are per-class
    descriptors with no rule, and ``__doc__``/``__module__``/``__qualname__``
    are already in the source hash. Dunder *functions* are still walked, so an
    ``__init__`` or ``__getitem__`` that runs in kernel scope keeps its globals
    in the key.
    """
    if cls.__module__ == "builtins" or _is_compiler_object(cls, ctx):
        ctx.out.append(f"class-ref:{cls.__module__}.{cls.__qualname__}")
        return
    marker = id(cls)
    if marker in ctx.visited:
        ctx.out.append(f"class-ref:{cls.__module__}.{cls.__qualname__}")
        return
    ctx.visited.add(marker)
    filepath, lineno, source_hash = _class_source_hash(cls)
    ctx.out.append(
        f"class:{cls.__module__}.{cls.__qualname__}:{filepath}:{lineno}:{source_hash}"
    )
    for name, member in sorted(vars(cls).items()):
        if isinstance(member, (staticmethod, classmethod)):
            member = member.__func__
        elif isinstance(member, property):
            member = member.fget
        if isinstance(member, types.FunctionType) or _is_compiler_callable(member):
            # A `@ti.func` in a class body arrives here as a compiler wrapper,
            # not as a plain function; it is the member most likely to read a
            # module global the transform then bakes in.
            ctx.out.append(f"method:{name}")
            _hash_value(member, ctx, depth + 1)
        elif not (name.startswith("__") and name.endswith("__")):
            ctx.out.append(f"attr:{name}")
            _hash_value(member, ctx, depth + 1)
    for base in cls.__bases__:
        _hash_class(base, ctx, depth + 1)


# ---------------------------------------------------------------------------
# Source hashes, memoized per object
# ---------------------------------------------------------------------------


def _source_info_and_src():
    """Quadrants' source retrieval, as currently bound -- memoized by warm-start when that is live."""
    from algan.taichi_compat import submodule

    return submodule("lang._wrap_inspect").get_source_info_and_src


def _function_source_hash(function):
    cached = getattr(function, "_algan_source_key_src", None)
    if cached is not None:
        return cached
    from algan.taichi_compat import submodule

    hash_iterable_strings = submodule(
        "lang._fast_caching.hash_utils"
    ).hash_iterable_strings
    try:
        info, src = _source_info_and_src()(function)
    except (OSError, TypeError) as exc:
        raise Poison(f"no source for {function.__qualname__}: {exc}") from None
    cached = (info.filepath, info.start_lineno, hash_iterable_strings(src))
    with contextlib.suppress(AttributeError, TypeError):
        function._algan_source_key_src = cached
    return cached


_CLASS_SOURCE = {}


def _class_source_hash(cls):
    cached = _CLASS_SOURCE.get(cls)
    if cached is not None:
        return cached
    from algan.taichi_compat import submodule

    hash_iterable_strings = submodule(
        "lang._fast_caching.hash_utils"
    ).hash_iterable_strings
    try:
        lines, lineno = inspect.getsourcelines(cls)
        filepath = inspect.getsourcefile(cls) or inspect.getfile(cls)
    except (OSError, TypeError) as exc:
        raise Poison(f"no source for class {cls.__qualname__}: {exc}") from None
    cached = (filepath, lineno, hash_iterable_strings(lines))
    _CLASS_SOURCE[cls] = cached
    return cached


# ---------------------------------------------------------------------------
# The reference walk: what a code object reads from outside its locals
# ---------------------------------------------------------------------------

#: Loads that begin a chain, by what the name means.
_GLOBAL_LOADS = frozenset({"LOAD_GLOBAL", "LOAD_NAME", "LOAD_FROM_DICT_OR_GLOBALS"})
_DEREF_LOADS = frozenset({"LOAD_DEREF", "LOAD_CLASSDEREF", "LOAD_FROM_DICT_OR_DEREF"})
_FAST_LOADS = frozenset({"LOAD_FAST", "LOAD_FAST_CHECK", "LOAD_FAST_AND_CLEAR"})
_ATTR_LOADS = frozenset({"LOAD_ATTR", "LOAD_METHOD"})

_CODE_CHAINS = {}
_CODE_LOCALS = {}


def _code_chains(code):
    """Every (kind, root, attrs) chain ``code`` and its nested code objects load.

    ``kind`` is ``"global"`` (a module global or builtin), ``"deref"`` (a
    closure cell) or ``"import"`` (a local bound by an ``import`` statement
    inside the function, whose root is then the dotted module name). ``attrs``
    is the ``LOAD_ATTR`` chain that immediately follows the load. Memoized per
    code object; the *values* are resolved fresh on every key computation.

    The import forms, as CPython 3.11-3.13 compile them: ``import a.b`` is
    ``IMPORT_NAME a.b`` then ``STORE_FAST a`` and binds the *top* package;
    ``import a.b as c`` and ``from a import b as c`` both go through
    ``IMPORT_FROM b`` first and bind that attribute; ``from a import b, c``
    issues one ``IMPORT_FROM``/``STORE_FAST`` pair per name off the one module
    left on the stack, which the closing ``POP_TOP`` drops.
    """
    cached = _CODE_CHAINS.get(code)
    if cached is not None:
        return cached
    chains = []
    imports = {}
    current = None
    pending_import = None  # (module name, attrs since the last STORE, saw IMPORT_FROM)
    seen = set()

    def flush():
        nonlocal current
        if current is not None:
            chain = (current[0], current[1], tuple(current[2]))
            if chain not in seen:
                seen.add(chain)
                chains.append(chain)
            current = None

    for instruction in dis.get_instructions(code):
        name = instruction.opname
        if name in _ATTR_LOADS and current is not None:
            current[2].append(instruction.argval)
            continue
        flush()
        if name in _GLOBAL_LOADS:
            current = ["global", instruction.argval, []]
        elif name in _DEREF_LOADS:
            current = ["deref", instruction.argval, []]
        elif name in _FAST_LOADS:
            if instruction.argval in imports:
                root, prefix = imports[instruction.argval]
                current = ["import", root, list(prefix)]
        elif name == "LOAD_FAST_LOAD_FAST":
            second = instruction.argval[1]
            if second in imports:
                root, prefix = imports[second]
                current = ["import", root, list(prefix)]
        elif name == "IMPORT_NAME":
            pending_import = (instruction.argval, [], False)
        elif name == "IMPORT_FROM" and pending_import is not None:
            module_name, attrs, _ = pending_import
            pending_import = (module_name, [*attrs, instruction.argval], True)
        elif name == "STORE_FAST" and pending_import is not None:
            module_name, attrs, saw_from = pending_import
            if saw_from:
                imports[instruction.argval] = (module_name, tuple(attrs))
                pending_import = (module_name, attrs[:-1], True)
            else:
                imports[instruction.argval] = (module_name.split(".")[0], ())
                pending_import = None
        elif name == "POP_TOP" and pending_import is not None:
            pending_import = None
    flush()
    for constant in code.co_consts:
        if isinstance(constant, types.CodeType):
            for chain in _code_chains(constant):
                if chain not in seen:
                    seen.add(chain)
                    chains.append(chain)
    cached = tuple(chains)
    _CODE_CHAINS[code] = cached
    return cached


def _code_local_names(code):
    """Every local and cell name of ``code`` and the code objects nested in it.

    A ``LOAD_DEREF`` inside a nested comprehension or lambda that names one of
    these is a runtime local captured by the nested code, not a closure of the
    function being walked, and is skipped.
    """
    cached = _CODE_LOCALS.get(code)
    if cached is not None:
        return cached
    names = set(code.co_varnames) | set(code.co_cellvars)
    for constant in code.co_consts:
        if isinstance(constant, types.CodeType):
            names |= _code_local_names(constant)
    cached = frozenset(names)
    _CODE_LOCALS[code] = cached
    return cached


def _closure_cells(function):
    names = function.__code__.co_freevars
    cells = function.__closure__ or ()
    return dict(zip(names, cells))


def _hash_references(function, ctx, depth):
    """Resolve and hash every chain ``function``'s code reads."""
    code = function.__code__
    module_globals = function.__globals__
    cells = _closure_cells(function)
    local_names = _code_local_names(code)
    for kind, root, attrs in _code_chains(code):
        if kind == "deref":
            if root in cells:
                try:
                    value = cells[root].cell_contents
                except ValueError:
                    raise Poison(
                        f"closure cell {root!r} of {function.__qualname__} is empty"
                    ) from None
            elif root in local_names:
                continue  # a local captured by a nested comprehension/lambda
            else:
                raise Poison(
                    f"{function.__qualname__} reads an unknown closure name {root!r}"
                )
        elif kind == "global":
            if root in module_globals:
                value = module_globals[root]
            elif hasattr(builtins, root):
                ctx.out.append(f"builtin:{root}")
                continue
            else:
                raise Poison(
                    f"{function.__qualname__} reads an unresolvable global {root!r}"
                )
        else:  # import
            try:
                value = importlib.import_module(root)
            except ImportError:
                raise Poison(
                    f"{function.__qualname__} imports unimportable module {root!r}"
                ) from None
        ctx.out.append(f"ref:{kind}:{'.'.join((root, *attrs))}")
        _hash_chain(function, root, attrs, value, ctx, depth)


def _hash_chain(function, root, attrs, value, ctx, depth):
    """Walk ``attrs`` from ``value`` and hash the leaf (or every step for a module chain)."""
    for index, attr in enumerate(attrs):
        if _is_compiler_object(value, ctx) and isinstance(value, types.ModuleType):
            # `ti.math.vec4`: nothing below the compiler's own module needs
            # walking, its version is in the key.
            ctx.out.append(f"compiler-attr:{value.__name__}." + ".".join(attrs[index:]))
            return
        try:
            value = getattr(value, attr)
        except AttributeError:
            raise Poison(
                f"{function.__qualname__} reads {root}.{'.'.join(attrs[: index + 1])}, which does not exist"
            ) from None
        except Exception as exc:  # a property that raises
            raise Poison(
                f"{function.__qualname__} reads {root}.{'.'.join(attrs[: index + 1])}, which raised {exc!r}"
            ) from None
    _hash_value(value, ctx, depth + 1)


# ---------------------------------------------------------------------------
# The key
# ---------------------------------------------------------------------------

#: Declared ``ALGAN_`` variables that cannot reach a kernel's IR: diagnostics,
#: daemon plumbing, the test harness, and this feature's own switches (verify
#: mode must hit the entries the plain run stored, so it cannot be in the key).
_ENV_NOT_IN_KEY = frozenset(
    {
        "ALGAN_TAICHI_SOURCE_KEY",
        "ALGAN_TAICHI_SOURCE_KEY_VERIFY",
        "ALGAN_TAICHI_WARMSTART",
        "ALGAN_TAICHI_WARMSTART_VERIFY",
        "ALGAN_TAICHI_FAST_LAUNCH",
        "ALGAN_TAICHI_FAST_LAUNCH_VERIFY",
        "ALGAN_LOG_TAICHI_COMPILES",
        "ALGAN_TAICHI_COMPILE_LOG",
        "ALGAN_LOG_LEVEL",
        "ALGAN_PROGRESS",
        "ALGAN_PROFILE_CPROFILE",
        "ALGAN_PROFILE_NVPROF",
        "ALGAN_PROFILE_RUNS",
        "ALGAN_PROFILE_TELEMETRY",
        "ALGAN_UNDER_NVPROF",
        "ALGAN_TI_KERNEL_PROFILER",
        "ALGAN_USE_DAEMON",
        "ALGAN_AUTO_DAEMON",
        "ALGAN_DAEMON_CHILD",
        "ALGAN_DAEMON_IDLE_TIMEOUT",
        "ALGAN_DAEMON_LOG_MAX_BYTES",
        "ALGAN_DAEMON_PORT",
        "ALGAN_DAEMON_RELEASE_MEMORY",
        "ALGAN_DAEMON_START_TIMEOUT",
        "ALGAN_DAEMON_TIMEOUT",
        "ALGAN_HOME",
        "ALGAN_CACHE_DIR",
        "ALGAN_VIDEO_ENCODER",
        "ALGAN_MANIM_SVG_CACHE_MB",
    }
)


def _environment_fingerprint():
    from algan.environment import _HARNESS_VARIABLES, ALGAN_ENVIRONMENT_VARIABLES

    skip = _ENV_NOT_IN_KEY | frozenset(_HARNESS_VARIABLES)
    parts = []
    for name in sorted(ALGAN_ENVIRONMENT_VARIABLES):
        if name in skip:
            continue
        value = env_str(name, None)
        if value is not None:
            # Quoted, not bare: `hash_iterable_strings` joins its inputs with a
            # single "_", so a bare `A=1_B=2` in one variable and `A=1`, `B=2`
            # in two would hash alike.
            parts.append(f"{name}={value!r}")
    # Quadrants' own switch that rewrites every kernel AST; it is in its key too.
    if os.environ.get("QD_KERNEL_COVERAGE") == "1":
        parts.append("QD_KERNEL_COVERAGE=1")
    return parts


def _settings_fingerprint(ctx):
    from algan.settings import SETTINGS

    ctx.out.append("settings.raytracing(")
    _hash_value(SETTINGS.raytracing.to_dict(), ctx)
    ctx.out.append(")settings.computing(")
    computing = SETTINGS.computing.to_dict()
    computing = {
        name: str(value) if type(value).__name__ == "device" else value
        for name, value in computing.items()
    }
    _hash_value(computing, ctx)
    ctx.out.append(")")


#: Deliberately the same list Quadrants' own ``config_hasher.EXCLUDE_PREFIXES``
#: uses, so this key is never *less* config-sensitive than the fastcache key it
#: replaces -- ``verbose_`` and not ``verbose``, which keeps the bare
#: ``verbose`` field in, exactly as the compiler keeps it.
_CONFIG_EXCLUDE_PREFIXES = ("_", "offline_cache", "print_", "verbose_")
#: Process-local, not IR-relevant: the torch MPS queue's address (Quadrants
#: #850 excluded it from the C++ key for the same reason).
_CONFIG_EXCLUDE_NAMES = frozenset(
    {"external_metal_command_queue", "external_metal_command_queue_is_torch_queue"}
)


def _program_fingerprint():
    """Compile config and device caps of the live program, memoized per program."""
    from algan.taichi_compat import program, submodule

    prog = program()
    if prog is None:
        raise Poison("no compiler program is up")
    cached = _PROGRAM_FINGERPRINT.get(id(prog))
    if cached is not None and cached[0] is prog:
        return cached[1]
    config = prog.config()
    parts = []
    for name in dir(config):
        if name.startswith(_CONFIG_EXCLUDE_PREFIXES) or name in _CONFIG_EXCLUDE_NAMES:
            continue
        parts.append(f"config.{name}={getattr(config, name)}")
    # `qd.init` keeps a few transform switches on the Python runtime rather
    # than in the C++ config; the transformer reads them per statement.
    runtime = submodule("lang.impl").get_runtime()
    for name in (
        "short_circuit_operators",
        "unrolling_limit",
        "default_fp",
        "default_ip",
        "default_up",
    ):
        value = getattr(runtime, name, None)
        parts.append(f"runtime.{name}={_dtype_name(value) or value}")
    caps = prog.get_device_caps()
    capability_enum = submodule("_lib.core.quadrants_python").DeviceCapability
    for name in sorted(dir(capability_enum)):
        member = getattr(capability_enum, name)
        if isinstance(member, capability_enum):
            parts.append(f"caps.{name}={caps.get(member)}")
    _PROGRAM_FINGERPRINT.clear()
    _PROGRAM_FINGERPRINT[id(prog)] = (prog, parts)
    return parts


_PROGRAM_FINGERPRINT = {}


def _argument_descriptor(value, annotation, ctx, depth=0):
    """Render one non-template argument by its type features, or poison."""
    from algan.taichi_compat import submodule

    ndarray_type = submodule("types.ndarray_type").NdarrayType
    matrix_type = submodule("lang.matrix").MatrixType
    primitive_types = submodule("types.primitive_types")
    out = ctx.out
    if isinstance(annotation, ndarray_type):
        dtype = annotation.dtype
        if isinstance(dtype, matrix_type):
            anno_dtype = f"matrix({dtype.n},{dtype.m},{_dtype_name(dtype.dtype)})"
            element_ndim = dtype.ndim
        else:
            anno_dtype = _dtype_name(dtype) if dtype is not None else "None"
            element_ndim = 0
        out.append(
            f"ndarray-annotation({anno_dtype},ndim={annotation.ndim},layout={annotation.layout},"
            f"grad={annotation.needs_grad},boundary={annotation.boundary})"
        )
        type_qualname = f"{_module_name_of(type(value))}.{type(value).__qualname__}"
        torch = sys.modules.get("torch")
        if torch is not None and isinstance(value, torch.Tensor):
            # A subclass (`algan.constants.color.Color`) binds exactly like a
            # bare tensor; what the transform reads is dtype, rank and grad.
            shape = tuple(value.shape)
            element_shape = shape[len(shape) - element_ndim :] if element_ndim else ()
            out.append(
                f"torch({value.dtype},ndim={value.ndim},grad={bool(value.requires_grad)},"
                f"element_shape={element_shape})"
            )
            return
        if (
            type_qualname.startswith("numpy.")
            and hasattr(value, "dtype")
            and hasattr(value, "ndim")
        ):
            shape = tuple(value.shape)
            element_shape = shape[len(shape) - element_ndim :] if element_ndim else ()
            out.append(
                f"numpy({value.dtype},ndim={value.ndim},element_shape={element_shape})"
            )
            return
        if type(value).__name__ in ("ScalarNdarray", "VectorNdarray", "MatrixNdarray"):
            out.append(
                f"ndarray({_dtype_name(value.element_type) or value.element_type},ndim={len(value.shape)},"
                f"grad={value.grad is not None},layout={getattr(value, '_qd_layout', None)})"
            )
            return
        raise Poison(f"ndarray argument of type {type_qualname} has no key rule")
    if id(annotation) in primitive_types.type_ids:
        out.append(f"scalar-annotation({_dtype_name(annotation)})")
        return
    raise Poison(f"argument annotation {annotation!r} has no key rule")


def compute_key(kernel, args):
    """Algan's source key for one materialization of ``kernel`` with ``args``.

    Returns ``(key, None)`` or ``(None, reason)`` when the key is poisoned.
    Pure: it neither touches the compiler nor stores anything.
    """
    from algan.taichi_compat import BACKEND, submodule

    started = time.perf_counter()
    try:
        fast_caching = submodule("lang._fast_caching.src_hasher")
        function_hasher = submodule("lang._fast_caching.function_hasher")
        hash_iterable_strings = submodule(
            "lang._fast_caching.hash_utils"
        ).hash_iterable_strings
        template = submodule("types").template
        compiler = importlib.import_module(BACKEND)

        ctx = _KeyContext(BACKEND)
        out = ctx.out
        out.append(_SCHEMA_VERSION)
        out.append(
            f"compiler:{BACKEND}:{getattr(compiler, '__version_str__', compiler.__version__)}"
        )
        out.append(f"schema:{fast_caching._CACHE_VALUE_SCHEMA_VERSION}")
        out.append(f"autodiff:{kernel.autodiff_mode}")
        # Decorator flags the transformer branches on (`@qd.kernel(graph=,
        # checkpoints=)`), and whether the first parameter is a class `self`.
        out.append(
            f"kernel-flags:graph={getattr(kernel, 'use_graph', False)},"
            f"checkpoints={getattr(kernel, 'use_checkpoints', False)},"
            f"classkernel={getattr(kernel, 'is_classkernel', False)}"
        )

        kernel_source_info, _src = _source_info_and_src()(kernel.func)
        out.append(
            f"kernel:{kernel_source_info.filepath}:{kernel_source_info.start_lineno}:"
            f"{function_hasher.hash_kernel(kernel_source_info)}"
        )
        out.extend(_program_fingerprint())

        arg_metas = kernel.arg_metas
        if len(args) != len(arg_metas):
            raise Poison(f"{len(arg_metas)} parameters but {len(args)} arguments")
        for index, (value, meta) in enumerate(zip(args, arg_metas)):
            annotation = meta.annotation
            out.append(f"arg[{index}]:{meta.name}")
            if annotation is template or isinstance(annotation, template):
                out.append("template(")
                _hash_value(value, ctx)
                out.append(")")
            else:
                _argument_descriptor(value, annotation, ctx)

        out.append("body(")
        ctx.visited.add(id(kernel.func))
        _hash_references(kernel.func, ctx, 0)
        out.append(")env(")
        out.extend(_environment_fingerprint())
        out.append(")")
        _settings_fingerprint(ctx)
        return hash_iterable_strings(out), None
    except Poison as poison:
        return None, str(poison)
    except Exception as unexpected:
        # A key that cannot be built is a slow kernel; a key computation that
        # *raises* is a broken render. The whole design is fail-closed, so an
        # unexpected failure poisons like any other unknown value -- named as
        # unexpected so the poisoned-kernel report still shows it as a bug.
        return (
            None,
            f"unexpected {type(unexpected).__name__} building the key: {unexpected}",
        )
    finally:
        STATS["key_seconds"] += time.perf_counter() - started


def kernel_qualname(kernel):
    function = kernel.func
    return f"{getattr(function, '__module__', '')}.{getattr(function, '__qualname__', function.__name__)}"


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------

_WARNED = set()
#: Verify mode: fast key -> the C++ key the index holds for it, awaiting the
#: compile that will re-derive it.
_PENDING_VERIFY = {}


def _warn_poisoned(kernel, reason):
    name = kernel_qualname(kernel)
    POISONED[name] = reason
    if name in _WARNED:
        return
    _WARNED.add(name)
    warnings.warn(
        f"taichi_source_key: {name} cannot be source-keyed and pays the full frontend: {reason}",
        RuntimeWarning,
        stacklevel=4,
    )


def _restore_from_cache_value(self, key, cache_value, kernel_module, src_hasher):
    """What Quadrants' ``_try_load_fastcache`` does on a validated hit, verbatim in effect.

    Loads the C++ kernel data by the stored key, records the used-parameter
    set and rebuilds the graph-do-while and checkpoint tables the skipped
    transform would have written. Returns the used-parameter set, or ``None``
    when the artifact is gone from the C++ cache (the caller then takes the
    full path, exactly as the original does).
    """
    from algan.taichi_compat import submodule

    prog = submodule("lang.impl").get_runtime().prog
    data = prog.load_fast_cache(
        cache_value.frontend_cache_key,
        self.func.__name__,
        prog.config(),
        prog.get_device_caps(),
    )
    if not data:
        self.compiled_kernel_data_by_key.pop(key, None)
        return None
    self.compiled_kernel_data_by_key[key] = data
    self.src_ll_cache_observations.cache_loaded = True
    self.used_py_dataclass_parameters_by_key_enforcing[key] = (
        cache_value.used_py_dataclass_parameters
    )
    if cache_value.graph_do_while_levels:
        level_cls = kernel_module.GraphDoWhileLevel
        self.graph_do_while_levels = [
            level_cls(cond_arg_name=name, parent_id=parent, cond_cpp_arg_id=cpp_arg_id)
            for name, parent, cpp_arg_id in cache_value.graph_do_while_levels
        ]
        self.graph_do_while_arg = self.graph_do_while_levels[0].cond_arg_name
    if cache_value.checkpoint_yield_on_args:
        self.checkpoint_yield_on_args = list(cache_value.checkpoint_yield_on_args)
        self.checkpoint_yield_on_cpp_arg_ids = list(
            cache_value.checkpoint_yield_on_cpp_arg_ids
        )
        raw_labels = list(cache_value.checkpoint_user_labels_by_cp_id)
        qualnames = list(cache_value.checkpoint_user_label_enum_qualnames) or [
            None
        ] * len(raw_labels)
        if len(qualnames) != len(raw_labels):
            qualnames = [None] * len(raw_labels)
        self.checkpoint_user_labels_by_cp_id = [
            src_hasher._resolve_intenum_member(qn, lbl)
            for qn, lbl in zip(qualnames, raw_labels)
        ]
    return cache_value.used_py_dataclass_parameters


def _quadrants_entry_points(version):
    """The Quadrants internals the patch rests on, or a string saying why not.

    Kept apart from :func:`_apply_quadrants` so the gate can be checked -- by
    ``algan check`` and by the fast suite -- without installing anything.
    """
    from algan.taichi_compat import submodule

    if version[:2] != (1, 3):
        return (
            f"the source-key patch replicates quadrants 1.3 internals; this is "
            f"{'.'.join(str(p) for p in version) or 'an unknown version'}"
        )
    try:
        kernel_module = submodule("lang.kernel")
        src_hasher = submodule("lang._fast_caching.src_hasher")
        impl = submodule("lang.impl")
        autodiff_none = submodule("types.enums").AutodiffMode.NONE
    except Exception:
        return "quadrants' fastcache internals are not where 1.3 keeps them"
    kernel_cls = getattr(kernel_module, "Kernel", None)
    if (
        kernel_cls is None
        or not hasattr(kernel_cls, "_try_load_fastcache")
        or not hasattr(kernel_cls, "materialize")
        or not hasattr(src_hasher, "load")
        or not hasattr(src_hasher, "store")
        or not hasattr(src_hasher, "_CACHE_VALUE_SCHEMA_VERSION")
        or not hasattr(impl, "on_reset")
    ):
        return "quadrants 1.3's fastcache entry points have moved"
    return kernel_module, src_hasher, impl, autodiff_none


def _build_hooks(
    kernel_module, src_hasher, autodiff_none, original_try_load, original_store
):
    """The two replacements: ``Kernel._try_load_fastcache`` and ``src_hasher.store``.

    Built by a factory rather than written at module level so a test can hold
    a pair against stub objects without installing them on the live compiler.
    """

    def _try_load_fastcache(self, args, key):
        # Quadrants' own fastcache, where a kernel opted into it: defer.
        if (
            self.runtime.src_ll_cache
            and self.quadrants_callable
            and self.quadrants_callable.is_pure
        ):
            return original_try_load(self, args, key)
        if not self.runtime.src_ll_cache or self.autodiff_mode != autodiff_none:
            return None
        fast_key, reason = compute_key(self, args)
        if fast_key is None:
            # Nothing is stored under an unsound key: `materialize` reset
            # `fast_checksum` to None before calling, and `launch_kernel`
            # only stores when it is set.
            STATS["poisoned"] += 1
            _warn_poisoned(self, reason)
            return None
        STATS["keyed"] += 1
        self.fast_checksum = fast_key
        self.src_ll_cache_observations.cache_key_generated = True
        cache_value = src_hasher.load(fast_key)
        if cache_value is None:
            STATS["misses"] += 1
            return None
        self.src_ll_cache_observations.cache_validated = True
        if env_flag("ALGAN_TAICHI_SOURCE_KEY_VERIFY", False):
            # Take the full path and let the store hook compare what the
            # compile produces against what the index promised.
            _PENDING_VERIFY[fast_key] = cache_value.frontend_cache_key
            return None
        used = _restore_from_cache_value(
            self, key, cache_value, kernel_module, src_hasher
        )
        if used is None:
            STATS["misses"] += 1
            return None
        STATS["hits"] += 1
        return used

    def _store(frontend_cache_key, fast_cache_key, *args, **kwargs):
        expected = _PENDING_VERIFY.pop(fast_cache_key, None)
        if expected is not None:
            STATS["verified"] += 1
            if expected != frontend_cache_key:
                raise RuntimeError(
                    "taichi_source_key VERIFY mismatch: the index maps source key "
                    f"{fast_cache_key} to C++ key {expected}, but a full transform "
                    f"produced {frontend_cache_key}. The key is missing an input."
                )
        return original_store(frontend_cache_key, fast_cache_key, *args, **kwargs)

    _try_load_fastcache._algan_original = original_try_load
    _store._algan_original = original_store
    return _try_load_fastcache, _store


def _apply_quadrants(version):
    entry_points = _quadrants_entry_points(version)
    if isinstance(entry_points, str):
        return _skip(entry_points)
    kernel_module, src_hasher, impl, autodiff_none = entry_points
    kernel_cls = kernel_module.Kernel
    original_store = src_hasher.store
    try_load, store = _build_hooks(
        kernel_module,
        src_hasher,
        autodiff_none,
        kernel_cls._try_load_fastcache,
        original_store,
    )
    kernel_cls._try_load_fastcache = try_load
    src_hasher.store = store
    # Rebound wherever it is *read*: kernel.py imported the module, not the
    # function, so patching the attribute is enough there; anything that did
    # `from src_hasher import store` gets the same treatment.
    for module in list(sys.modules.values()):
        if (
            getattr(module, "__name__", "").startswith("quadrants.")
            and getattr(module, "store", None) is original_store
        ):
            module.store = store
    # A re-init builds a new Program with its own config and caps.
    impl.on_reset(_PROGRAM_FINGERPRINT.clear)
    return True


def _skip(reason):
    global _SKIPPED_REASON
    _SKIPPED_REASON = reason
    return False


def apply():
    """Install the source-keyed index (idempotent; a no-op with a reason on mismatch)."""
    global _APPLIED, _SKIPPED_REASON
    if _APPLIED:
        return
    if not env_flag("ALGAN_TAICHI_SOURCE_KEY", True):
        _SKIPPED_REASON = "turned off by ALGAN_TAICHI_SOURCE_KEY=0"
        return
    try:
        from algan.taichi_compat import BACKEND, backend_version
    except Exception:
        _SKIPPED_REASON = "the kernel compiler could not be imported"
        return
    if BACKEND != "quadrants":
        _SKIPPED_REASON = (
            f"{BACKEND} has no fast-cache load path; the index is Quadrants-only"
        )
        return
    _SKIPPED_REASON = None
    _APPLIED = bool(_apply_quadrants(tuple(backend_version())))


def is_applied():
    return _APPLIED
