r"""Compile render kernels ahead of the render, several at a time.

The problem
-----------
Kernels compile lazily, at their first launch, one after another: the render
pipeline cannot launch its fifth kernel before the fourth has produced the
data the fifth reads. A cold process therefore compiles serially however many
cores the machine has -- a first Hello World on a GTX 1050 box spent ~640 s
compiling 24 kernels with 0.7-1.8 of 8 logical cores busy, and each new
feature combination (glass, shadows, metal) added up to 16 minutes more.

Compiling does not need the pipeline's data, only its *types*: what selects a
specialization is the kernel, the dtype and rank of each array argument, and
the value of each ``ti.template()`` argument. So a specialization can be
written down once (a **spec**) and compiled later, anywhere, with one-element
placeholder tensors -- without launching anything.

The mechanism
-------------
1. **Recording.** Every specialization a process materializes is encoded as a
   portable spec (:func:`encode_spec`) and merged into a manifest beside the
   kernel cache (``algan_kernel_specs.json``), with the settings context it
   was compiled under and how long it took. A spec is portable only when every
   template value is plain data or names an ``algan.*`` function, class or
   enum; anything else (a scene script's own ``@ti.func`` stage, an instance)
   is simply not recorded and keeps compiling at first launch.
2. **Precompiling.** :class:`PrecompilePool` starts worker processes
   (``python -m algan.rendering._precompile_worker``), each of which applies the
   spec's settings, builds placeholder arguments and compiles one spec at a
   time -- ``Kernel.ensure_compiled`` plus the compile-and-index half of
   ``Kernel.launch_kernel`` -- writing each artifact to disk as soon as it is
   built (``Program.dump_cache_data_to_disk``). Longest first, so the batch
   finishes as early as the slowest single kernel allows.
3. **Picking the work up.** The compiler reads the offline cache's index once,
   the first time a program uses it, and never again (measured on Quadrants
   1.3: a process that had compiled one kernel did not see a second one that
   another process dumped afterwards, through either the source-keyed index
   or the offline cache, even after dumping its own). So a worker's artifact
   helps a process only if it is on disk before that process's first
   materialization -- and that is where a running pool is waited for
   (:func:`before_first_materialization`). Authoring before that point
   overlaps the workers; the wait itself is the parallel compile instead of
   the serial one.

Nothing here can make a render wrong. Every artifact is still found by the
keys the render computes from its own arguments, settings and source -- a
worker that compiled under different settings, or a spec that no longer
matches the kernel, produces an artifact the render simply never looks up, and
the render compiles as it always did. The worst case is wasted worker time.

When it runs
------------
* At the end of ``import algan`` in a process that is plainly running a scene
  script (the daemon handoff's own test, so not the ``algan`` CLI, a test
  runner, a notebook, ``python -c`` or ``-m``), for the specializations *that
  script* rendered before, when the cache does not have them for this
  version: after an update, a kernel edit, or a cleared cache. The render
  daemon does the same at the start of a run, while its program has not
  touched the cache yet. Speculative work -- the built-in list, other
  scripts' kernels -- is left to ``algan warmup``: measured, a pool compiling
  kernels the render did not need only slowed the render's own compiles.
* From ``algan warmup`` (``rendering/kernel_warmup.py``), for every spec on
  record and the built-in common set, before any render needs them.

When the process exits with work still queued, each worker finishes the spec
it holds (and writes it to disk) and then exits; queued specs wait for the next
process. A spec is "confirmed" when a process has seen it cached under the
current **environment stamp** -- the compiler, the Python version and a
signature of the installed ``algan`` sources. In steady state every spec is
confirmed and the import-time check costs a manifest read and a directory
``stat`` walk on a background thread.

Switches
--------
``ALGAN_PRECOMPILE_JOBS`` -- worker count; ``0`` turns the pool off (recording
stays on, it is what ``algan warmup`` reads). Unset, it is one fewer than the
logical CPUs, capped by free host memory (~1.5 GB a worker), by free VRAM on
CUDA, and at 8. The pool is Quadrants-only, like the source-keyed index it
rests on (``utils/taichi_source_key.py``).
"""

from __future__ import annotations

import atexit
import contextlib
import enum
import hashlib
import importlib
import json
import os
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

from algan.environment import env_flag, env_int, env_is_set, env_overrides, env_str

_MANIFEST_SCHEMA = "algan-kernel-specs-v2"
_MANIFEST_FILENAME = "algan_kernel_specs.json"
#: Enough for every variant a user renders; oldest-used entries go first.
_MANIFEST_MAX_ENTRIES = 400
#: The common variant set ``algan warmup`` covers, recorded by
#: ``scripts/generate_kernel_specs.py``. Specs only -- each is compiled under
#: the settings of the process that reads it, never the generator's.
BUILTIN_MANIFEST = Path(__file__).with_name("kernel_specs_builtin.json")
#: Prefixes every protocol line a worker writes, so anything else that reaches
#: its stdout is ignored rather than parsed.
_PROTOCOL_PREFIX = "@@algan-precompile@@ "
#: What a worker is assumed to cost while it compiles: host RSS with torch and
#: the compiler loaded plus a megakernel's LLVM peak, and on CUDA a context and
#: the shrunken pool (``_WORKER_DEVICE_MEMORY_GB``).
_WORKER_HOST_BYTES = int(1.5 * 2**30)
_WORKER_VRAM_BYTES = int(0.6 * 2**30)
_WORKER_DEVICE_MEMORY_GB = 0.25
#: Spawning a worker costs an ``import algan`` (seconds of CPU and a GB of
#: RAM), so a batch whose expected compile time is below this is left to the
#: render. A spec with no recorded time counts as ``_UNKNOWN_SECONDS``.
_MIN_POOL_SECONDS = 15.0
_UNKNOWN_SECONDS = 5.0


class Unportable(Exception):
    """A value a spec cannot carry to another process."""


# ---------------------------------------------------------------------------
# Value encoding
# ---------------------------------------------------------------------------


def _object_path(obj):
    """``(module, qualname)`` of an importable ``algan.*`` object, or raise."""
    module = getattr(obj, "__module__", None) or ""
    qualname = getattr(obj, "__qualname__", None) or ""
    if not module.startswith("algan.") or not qualname or "<" in qualname:
        raise Unportable(f"{obj!r} is not an importable algan object")
    return module, qualname


def _resolve_path(module, qualname):
    value = importlib.import_module(module)
    for part in qualname.split("."):
        value = getattr(value, part)
    return value


def _encode(value, depth=0):
    """A JSON-safe, tagged rendering of a template value, or :class:`Unportable`.

    Tagged rather than bare so that the decoded value has exactly the type the
    template mapper and the source key saw: ``True`` and ``1`` specialize
    alike but key differently, and a tuple is not a list.
    """
    if depth > 16:
        raise Unportable("value nests deeper than 16 levels")
    if value is None:
        return ["none"]
    kind = type(value)
    if kind is bool:
        return ["bool", value]
    if kind is int:
        return ["int", value]
    if kind is float:
        return ["float", value]
    if kind is str:
        return ["str", value]
    if kind in (tuple, list):
        return [kind.__name__, [_encode(item, depth + 1) for item in value]]
    if isinstance(value, enum.Enum):
        module, qualname = _object_path(kind)
        return ["enum", module, qualname, value.name]
    from algan.utils.taichi_source_key import _dtype_name, _is_compiler_callable

    dtype = _dtype_name(value)
    if dtype is not None:
        return ["dtype", dtype]
    if f"{type(value).__module__}.{type(value).__qualname__}" == "torch.dtype":
        return ["torch_dtype", str(value).removeprefix("torch.")]
    if _is_compiler_callable(value):
        module, qualname = _object_path(value.fn)
        return ["func", module, qualname]
    if isinstance(value, types.FunctionType):
        return ["pyfunc", *_object_path(value)]
    if isinstance(value, type):
        return ["class", *_object_path(value)]
    raise Unportable(f"template value of type {kind.__qualname__} has no encoding")


def _decode(encoded):
    tag = encoded[0]
    if tag == "none":
        return None
    if tag in ("bool", "int", "float", "str"):
        return encoded[1]
    if tag == "tuple":
        return tuple(_decode(item) for item in encoded[1])
    if tag == "list":
        return [_decode(item) for item in encoded[1]]
    if tag == "enum":
        return _resolve_path(encoded[1], encoded[2])[encoded[3]]
    if tag == "dtype":
        from algan.taichi_compat import template_dtype, ti

        return template_dtype(getattr(ti, encoded[1]))
    if tag == "torch_dtype":
        import torch

        return getattr(torch, encoded[1])
    if tag == "func":
        resolved = _resolve_path(encoded[1], encoded[2])
        if getattr(getattr(resolved, "fn", None), "__qualname__", None) != encoded[2]:
            raise Unportable(f"{encoded[1]}.{encoded[2]} is no longer that @ti.func")
        return resolved
    if tag in ("pyfunc", "class"):
        resolved = _resolve_path(encoded[1], encoded[2])
        if getattr(resolved, "__qualname__", None) != encoded[2]:
            raise Unportable(f"{encoded[1]}.{encoded[2]} moved")
        return resolved
    raise Unportable(f"unknown tag {tag!r}")


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------


def _annotation_kinds():
    from algan.taichi_compat import submodule

    return (
        submodule("types").template,
        submodule("types.ndarray_type").NdarrayType,
        submodule("lang.matrix").MatrixType,
        submodule("types.primitive_types"),
    )


def encode_spec(kernel, args):
    """The portable description of one materialization, or ``None``.

    ``{"kernel": "<module>:<qualname>", "args": [...]}`` where each argument is
    ``["T", value]`` for a template, ``["A", dtype, rank, element_shape]`` for
    an array (a torch tensor; only its type reaches the IR) and ``["S", type]``
    for a scalar (its value is a runtime input, not part of the specialization). ``None`` when anything is not portable -- the kernel is not a
    module-level ``algan`` kernel, an argument is not a torch tensor, a template
    value has no encoding.
    """
    function = kernel.func
    try:
        module, qualname = _object_path(function)
        template, ndarray_type, matrix_type, primitive_types = _annotation_kinds()
        metas = kernel.arg_metas
        if len(metas) != len(args):
            return None
        import torch

        encoded = []
        for value, meta in zip(args, metas):
            annotation = meta.annotation
            if annotation is template or isinstance(annotation, template):
                encoded.append(["T", _encode(value)])
            elif isinstance(annotation, ndarray_type):
                if not isinstance(value, torch.Tensor):
                    return None
                element_ndim = (
                    annotation.dtype.ndim
                    if isinstance(annotation.dtype, matrix_type)
                    else 0
                )
                shape = tuple(value.shape)
                element_shape = (
                    list(shape[len(shape) - element_ndim :]) if element_ndim else []
                )
                encoded.append(
                    [
                        "A",
                        str(value.dtype).removeprefix("torch."),
                        value.ndim,
                        element_shape,
                    ]
                )
            elif id(annotation) in primitive_types.type_ids:
                # By type only. A scalar argument is a runtime value -- a
                # count, a frame size -- and never selects a specialization,
                # so its value must not be part of the spec's identity: two
                # batches of one scene pass different counts to one kernel.
                if type(value) not in (bool, int, float):
                    return None
                encoded.append(["S", type(value).__name__])
            else:
                return None
    except Unportable:
        return None
    except Exception:  # noqa: BLE001 -- recording must never fail a render
        return None
    return {"kernel": f"{module}:{qualname}", "args": encoded}


def _digest(value, length=32):
    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def spec_id(spec):
    return _digest(spec)


def resolve_kernel(spec):
    """The live ``Kernel`` a spec names, or :class:`Unportable`."""
    module, qualname = spec["kernel"].split(":", 1)
    try:
        wrapper = _resolve_path(module, qualname)
    except (ImportError, AttributeError) as exc:
        raise Unportable(f"{spec['kernel']} does not resolve: {exc}") from None
    kernel = getattr(wrapper, "_primal", None)
    if kernel is None or getattr(kernel.func, "__qualname__", None) != qualname:
        raise Unportable(f"{spec['kernel']} is not a kernel")
    if len(kernel.arg_metas) != len(spec["args"]):
        raise Unportable(
            f"{spec['kernel']} takes {len(kernel.arg_metas)} arguments, "
            f"the spec has {len(spec['args'])}"
        )
    return kernel


def placeholder_args(spec, device):
    """Arguments of the spec's types: one-element tensors, decoded templates."""
    import torch

    args = []
    for encoded in spec["args"]:
        kind = encoded[0]
        if kind == "A":
            _, dtype, rank, element_shape = encoded
            shape = (1,) * (rank - len(element_shape)) + tuple(element_shape)
            args.append(torch.zeros(shape, dtype=getattr(torch, dtype), device=device))
        elif kind == "S":
            args.append(_SCALAR_PLACEHOLDERS[encoded[1]])
        else:
            args.append(_decode(encoded[1]))
    return tuple(args)


_SCALAR_PLACEHOLDERS = {"bool": False, "int": 0, "float": 0.0}


def kernel_display_name(spec_or_kernel):
    """The short kernel name a progress line shows."""
    name = (
        spec_or_kernel["kernel"]
        if isinstance(spec_or_kernel, dict)
        else getattr(spec_or_kernel.func, "__qualname__", "kernel")
    )
    return name.rsplit(":", 1)[-1].rsplit(".", 1)[-1]


# ---------------------------------------------------------------------------
# The settings context a spec is compiled under
# ---------------------------------------------------------------------------


def current_context():
    """The settings and environment the source key folds in, as plain data.

    The same three inputs ``taichi_source_key`` hashes beside the kernel's own
    source (``_settings_fingerprint`` and ``_environment_fingerprint``): a
    worker that reproduces them writes index entries under exactly the keys
    this process will compute.
    """
    from algan.environment import _HARNESS_VARIABLES, ALGAN_ENVIRONMENT_VARIABLES
    from algan.settings import SETTINGS
    from algan.utils.taichi_source_key import _ENV_NOT_IN_KEY

    computing = {
        name: str(value) if type(value).__name__ == "device" else value
        for name, value in SETTINGS.computing.to_dict().items()
    }
    skip = _ENV_NOT_IN_KEY | frozenset(_HARNESS_VARIABLES) | _WORKER_ONLY_ENV
    env = {}
    for name in sorted(ALGAN_ENVIRONMENT_VARIABLES):
        if name in skip:
            continue
        value = env_str(name, None)
        if value is not None:
            env[name] = value
    if os.environ.get("QD_KERNEL_COVERAGE") == "1":
        env["QD_KERNEL_COVERAGE"] = "1"
    return {
        "raytracing": SETTINGS.raytracing.to_dict(),
        "computing": computing,
        "env": env,
    }


def context_id(context):
    return _digest(context, 16)


def apply_context(context):
    """Make this process's settings the context's (a worker, between specs)."""
    from algan.settings import SETTINGS

    SETTINGS.computing.set(**context["computing"])
    # As `SETTINGS.restore` does: a captured configuration round-trips every
    # field, the experimental and import-frozen ones included.
    SETTINGS.raytracing._restore(context["raytracing"])


#: Variables that describe the worker itself and must neither be inherited
#: into a context nor keyed.
_WORKER_ONLY_ENV = frozenset({"ALGAN_PRECOMPILE_WORKER", "ALGAN_PRECOMPILE_JOBS"})


# ---------------------------------------------------------------------------
# The environment stamp: when a confirmation stops meaning anything
# ---------------------------------------------------------------------------

_STAMP = None
_STAMP_LOCK = threading.Lock()


def environment_stamp():
    """A signature of everything that can invalidate the whole kernel cache at once.

    The compiler and its version, the Python version, Algan's version, and
    ``(path, size, mtime)`` of every source file in the installed package.
    Heuristic by design -- a stamp that fails to move only means a spec is not
    re-checked, never that a wrong kernel is used -- which is why it can afford
    to be a ``stat`` walk rather than a content hash. Memoized per process.
    """
    global _STAMP
    with _STAMP_LOCK:
        if _STAMP is not None:
            return _STAMP
        from importlib.metadata import PackageNotFoundError, version

        from algan.taichi_compat import describe_backend

        digest = hashlib.sha256()
        try:
            algan_version = version("algan")
        except PackageNotFoundError:
            algan_version = "0+unknown"
        digest.update(
            f"{describe_backend()}|{sys.version_info[:3]}|{algan_version}".encode()
        )
        root = Path(__file__).resolve().parents[1]
        for directory, subdirectories, files in os.walk(root):
            subdirectories[:] = sorted(
                d for d in subdirectories if d not in ("__pycache__", "docbuild")
            )
            for name in sorted(files):
                if not name.endswith(".py"):
                    continue
                path = os.path.join(directory, name)
                with contextlib.suppress(OSError):
                    stat = os.stat(path)
                    digest.update(
                        f"{os.path.relpath(path, root)}:{stat.st_size}:{stat.st_mtime_ns}\n".encode()
                    )
        _STAMP = digest.hexdigest()[:24]
        return _STAMP


# ---------------------------------------------------------------------------
# The manifest
# ---------------------------------------------------------------------------


def manifest_path():
    from algan.settings._startup import _TAICHI_CACHE_DIRECTORY

    return Path(_TAICHI_CACHE_DIRECTORY) / _MANIFEST_FILENAME


def _empty_manifest():
    return {"schema": _MANIFEST_SCHEMA, "contexts": {}, "entries": {}}


def read_manifest(path=None):
    """The manifest on disk, or an empty one when absent or unreadable."""
    path = Path(path) if path is not None else manifest_path()
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return _empty_manifest()
    if not isinstance(data, dict) or data.get("schema") != _MANIFEST_SCHEMA:
        return _empty_manifest()
    data.setdefault("contexts", {})
    data.setdefault("entries", {})
    return data


def read_builtin_specs(*, with_requirements=False):
    """The built-in specs: ``[(spec, seconds), ...]``, empty when there is no file.

    With ``with_requirements``, each item also carries the ``SETTINGS.raytracing``
    values its variant set (``{"shadows": True}`` for the shadow scene): those
    are compiled into the kernel, so the spec is only precompiled for a render
    whose settings match them.
    """
    try:
        with open(BUILTIN_MANIFEST, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return []
    if not isinstance(data, dict) or data.get("schema") != _MANIFEST_SCHEMA:
        return []
    items = []
    for entry in data.get("entries", {}).values():
        seconds = float(entry.get("seconds") or 0.0)
        if with_requirements:
            items.append((entry["spec"], seconds, dict(entry.get("requires") or {})))
        else:
            items.append((entry["spec"], seconds))
    return items


def entry_id(spec, context):
    """One manifest row per (spec, context): the same kernel under two settings is two artifacts."""
    return _digest({"spec": spec_id(spec), "context": context_id(context)})


_MANIFEST_LOCK = threading.Lock()
#: Rows changed in this process and not yet written, by entry id. Each carries
#: only the fields this process learned; :func:`flush_manifest` merges them.
_PENDING = {}
_PENDING_CONTEXTS = {}


#: How many scripts a row remembers as having used it, most recent first.
_SCRIPTS_PER_ROW = 8


def _note_entry(
    spec, context, *, seconds=None, confirmed=None, used=False, script=None
):
    eid = entry_id(spec, context)
    cid = context_id(context)
    with _MANIFEST_LOCK:
        row = _PENDING.setdefault(eid, {"spec": spec, "context": cid})
        if seconds is not None:
            row["seconds"] = round(float(seconds), 3)
        if confirmed is not None:
            row["confirmed"] = confirmed
        if used:
            row["last_used"] = time.time()
            row["uses"] = row.get("uses", 0) + 1
        if script:
            row["scripts"] = [script]
        _PENDING_CONTEXTS[cid] = context
    return eid


def _merge_scripts(new, old):
    merged = list(new)
    merged.extend(path for path in old if path not in merged)
    return merged[:_SCRIPTS_PER_ROW]


def flush_manifest(path=None):
    """Merge this process's rows into the manifest on disk, atomically.

    Read-merge-write, so concurrent processes lose at most each other's newest
    rows; bounded to the most recently used :data:`_MANIFEST_MAX_ENTRIES`, and
    contexts no row refers to any more are dropped.
    """
    with _MANIFEST_LOCK:
        if not _PENDING:
            return False
        pending = dict(_PENDING)
        contexts = dict(_PENDING_CONTEXTS)
        _PENDING.clear()
        _PENDING_CONTEXTS.clear()
    path = Path(path) if path is not None else manifest_path()
    manifest = read_manifest(path)
    entries = manifest["entries"]
    if any(update.get("confirmed") is True for update in pending.values()):
        stamp = environment_stamp()
        for update in pending.values():
            if update.get("confirmed") is True:
                update["confirmed"] = stamp
    for eid, update in pending.items():
        row = entries.setdefault(
            eid, {"spec": update["spec"], "context": update["context"]}
        )
        uses = update.pop("uses", 0)
        scripts = update.pop("scripts", [])
        row.update({k: v for k, v in update.items() if k != "spec"})
        row["uses"] = row.get("uses", 0) + uses
        if scripts:
            row["scripts"] = _merge_scripts(scripts, row.get("scripts", []))
    manifest["contexts"].update(contexts)
    if len(entries) > _MANIFEST_MAX_ENTRIES:
        keep = sorted(
            entries, key=lambda eid: entries[eid].get("last_used", 0.0), reverse=True
        )[:_MANIFEST_MAX_ENTRIES]
        manifest["entries"] = entries = {eid: entries[eid] for eid in keep}
    referenced = {row["context"] for row in entries.values()}
    manifest["contexts"] = {
        cid: ctx for cid, ctx in manifest["contexts"].items() if cid in referenced
    }
    import tempfile

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=str(path.parent), prefix=".kernel_specs.", suffix=".tmp"
        )
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, sort_keys=True)
        os.replace(tmp, path)
    except OSError:
        with contextlib.suppress(OSError, UnboundLocalError):
            os.unlink(tmp)
        return False
    return True


_EXPECTED = None


def expected_seconds(spec):
    """How long ``spec`` took to compile last time, under any settings; ``None`` if unknown.

    For the progress account's "about N s last time". Read from the manifest
    and the built-in list once per process.
    """
    global _EXPECTED
    if spec is None:
        return None
    if _EXPECTED is None:
        expected = {}
        with contextlib.suppress(Exception):
            for built_in, seconds in read_builtin_specs():
                if seconds:
                    expected[spec_id(built_in)] = seconds
            for row in read_manifest()["entries"].values():
                seconds = row.get("seconds")
                if seconds:
                    expected[spec_id(row["spec"])] = float(seconds)
        _EXPECTED = expected
    return _EXPECTED.get(spec_id(spec))


# ---------------------------------------------------------------------------
# Recording, from the compile boundary in `taichi_runtime`
# ---------------------------------------------------------------------------


def _recording():
    return not env_flag("ALGAN_PRECOMPILE_WORKER", False)


def current_script():
    """The scene script this process is running, as a real path, or ``None``.

    ``sys.argv[0]``, which the render daemon sets to the client's script for
    the length of each run as well; ``None`` for ``python -c``, a REPL or a
    notebook.
    """
    argv0 = sys.argv[0] if sys.argv else ""
    if argv0.endswith(".py") and os.path.isfile(argv0):
        return os.path.realpath(argv0)
    return None


def note_materializing(kernel, args):
    """A new specialization is about to be materialized: remember what it is.

    Returns a token for :func:`note_materialized`, or ``None`` when the
    specialization is not portable (or this is a worker, whose results its
    parent records).
    """
    if not _recording():
        return None
    spec = encode_spec(kernel, args)
    if spec is None:
        return None
    return spec


def note_materialized(token, *, seconds, compiled):
    """The specialization is ready. ``compiled`` when it went through a real compile.

    Recorded under the current context with the current stamp as confirmed:
    whatever produced it, the artifact now exists (or will, at this process's
    cache flush). The time is kept only for a real compile -- it is what the
    pool schedules by and what ``algan warmup`` estimates from.
    """
    if token is None:
        return
    try:
        context = current_context()
        # `True` stands for "the current stamp", resolved when the row is
        # written: the stamp walks the package sources, which a render should
        # not pay for in the middle of its first kernel.
        _note_entry(
            token,
            context,
            seconds=seconds if compiled else None,
            confirmed=True,
            used=True,
            script=current_script(),
        )
    except Exception:  # noqa: BLE001 -- recording must never fail a render
        pass


# ---------------------------------------------------------------------------
# Compiling one spec, in a worker
# ---------------------------------------------------------------------------


def compile_spec(spec, on_started=None):
    """Materialize and compile ``spec`` without launching it.

    Mirrors ``Kernel.__call__`` up to the launch: ``ensure_compiled`` (which
    goes through the source-keyed index, so a hit loads instead of compiling),
    then the compile-and-index block of ``Kernel.launch_kernel`` verbatim in
    effect -- ``Program.compile_kernel`` and ``src_hasher.store`` -- and a dump
    so the artifact is on disk the moment it exists. Returns
    ``{"status", "seconds"}``; ``status`` is ``cached`` (the index served it),
    ``offline-hit`` (the frontend ran, the backend was on disk) or ``compiled``.
    """
    from algan.settings._startup import render_device
    from algan.taichi_compat import program, submodule
    from algan.utils import taichi_source_key as sk

    impl = submodule("lang.impl")
    src_hasher = submodule("lang._fast_caching.src_hasher")
    kernel = resolve_kernel(spec)
    args = placeholder_args(spec, render_device())
    if on_started is not None:
        fast_key, _ = sk.compute_key(kernel, args)
        on_started(fast_key)
    started = time.perf_counter()
    kernel.raise_on_templated_floats = impl.current_cfg().raise_on_templated_floats
    key = kernel.ensure_compiled(*args)
    if kernel.compiled_kernel_data_by_key.get(key) is not None:
        return {"status": "cached", "seconds": time.perf_counter() - started}
    prog = impl.get_runtime().prog
    result = prog.compile_kernel(
        prog.config(), prog.get_device_caps(), kernel.materialized_kernels[key]
    )
    if kernel.fast_checksum:
        src_hasher.store(
            result.cache_key,
            kernel.fast_checksum,
            kernel.visited_functions,
            kernel.used_py_dataclass_parameters_by_key_enforcing[key],
            graph_do_while_levels=[
                (level.cond_arg_name, level.parent_id, level.cond_cpp_arg_id)
                for level in kernel.graph_do_while_levels
            ],
            checkpoint_yield_on_args=list(kernel.checkpoint_yield_on_args),
            checkpoint_yield_on_cpp_arg_ids=list(
                kernel.checkpoint_yield_on_cpp_arg_ids
            ),
            checkpoint_user_labels_by_cp_id=list(
                kernel.checkpoint_user_labels_by_cp_id
            ),
        )
    kernel.compiled_kernel_data_by_key[key] = result.compiled_kernel_data
    program().dump_cache_data_to_disk()
    return {
        "status": "offline-hit" if result.cache_hit else "compiled",
        "seconds": time.perf_counter() - started,
    }


def _lower_priority():
    """Keep workers from competing with the render or authoring process."""
    with contextlib.suppress(Exception):
        if sys.platform == "win32":
            import psutil

            psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            os.nice(5)


def worker_main():
    """The worker loop: read ops from stdin, write events to (the real) stdout.

    Anything else that prints -- the compiler, a warning, the logger -- is sent
    to stderr, which the parent points at a log file, so the protocol stream
    carries nothing but protocol.
    """
    protocol = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    _lower_priority()

    from algan.rendering import taichi_runtime
    from algan.settings._startup import render_device
    from algan.utils.taichi_source_key import flush_source_memo, is_applied

    def emit(**event):
        # The parent may have exited while this worker compiled; the artifact
        # is already dumped by then, so a closed pipe only means "stop".
        try:
            protocol.write(_PROTOCOL_PREFIX + json.dumps(event) + "\n")
            protocol.flush()
        except (OSError, ValueError):
            flush_source_memo()
            os._exit(0)

    if not is_applied():
        emit(event="fatal", reason="the source-keyed index is not live here")
        os._exit(2)
    if render_device().type == "cuda":
        taichi_runtime._INIT_OVERRIDES["device_memory_GB"] = _WORKER_DEVICE_MEMORY_GB
    contexts = {}
    applied = None
    emit(event="ready", pid=os.getpid())
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        try:
            op = json.loads(line)
        except ValueError:
            continue
        if op.get("op") == "quit":
            break
        if op.get("op") == "context":
            contexts[op["id"]] = op["context"]
            continue
        if op.get("op") != "compile":
            continue
        eid = op["id"]
        try:
            if op["context"] != applied:
                apply_context(contexts[op["context"]])
                if render_device().type == "cuda":
                    taichi_runtime._INIT_OVERRIDES["device_memory_GB"] = (
                        _WORKER_DEVICE_MEMORY_GB
                    )
                else:
                    taichi_runtime._INIT_OVERRIDES.pop("device_memory_GB", None)
                taichi_runtime.ensure_taichi_for_render()
                applied = op["context"]
            result = compile_spec(
                op["spec"],
                on_started=lambda fast_key, eid=eid: emit(
                    event="started", id=eid, fast_key=fast_key
                ),
            )
        except Unportable as exc:
            result = {"status": "skipped", "reason": str(exc), "seconds": 0.0}
        except Exception as exc:  # noqa: BLE001 -- report and carry on
            result = {
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
                "seconds": 0.0,
            }
        emit(event="done", id=eid, **result)
        flush_source_memo()
    flush_source_memo()
    protocol.flush()
    # Skips interpreter teardown and the atexit cache flush: every artifact
    # was dumped as it was built, and a worker must never be the process
    # that runs the cache's exit-time eviction.
    os._exit(0)


# ---------------------------------------------------------------------------
# The pool
# ---------------------------------------------------------------------------


class _Job:
    """One spec under one context, and where it is in the pool."""

    __slots__ = (
        "eid",
        "spec",
        "context",
        "cid",
        "expected",
        "state",
        "status",
        "seconds",
        "reason",
        "worker",
        "name",
    )

    def __init__(self, spec, context, expected):
        self.eid = entry_id(spec, context)
        self.spec = spec
        self.context = context
        self.cid = context_id(context)
        self.expected = float(expected or _UNKNOWN_SECONDS)
        #: queued -> running -> done | failed.
        self.state = "queued"
        self.status = None
        self.seconds = 0.0
        self.reason = None
        self.worker = None
        self.name = kernel_display_name(spec)


class _Worker:
    __slots__ = ("process", "contexts", "job", "alive", "index")

    def __init__(self, process, index):
        self.process = process
        self.contexts = set()
        self.job = None
        self.alive = True
        self.index = index


def _log_path():
    from algan import daemon_client

    return Path(daemon_client.algan_home()) / "precompile.log"


def _open_log():
    path = _log_path()
    with contextlib.suppress(OSError):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size > 5 * 2**20:
            path.replace(path.with_suffix(".log.1"))
        return open(path, "ab")
    return subprocess.DEVNULL


def worker_environment(context_env=None):
    """``os.environ`` for a worker: off the daemon, marked, with a context's variables."""
    env = dict(os.environ)
    env.pop("ALGAN_DAEMON_CHILD", None)
    if context_env is not None:
        current = current_context()["env"]
        for name in current:
            if name not in context_env:
                env.pop(name, None)
        env.update(context_env)
    env.update(
        env_overrides(
            ALGAN_USE_DAEMON="0",
            ALGAN_AUTO_DAEMON="0",
            ALGAN_PRECOMPILE_WORKER="1",
        )
    )
    return env


class PrecompilePool:
    """Worker processes compiling a list of jobs, longest first.

    Thread-safe: reader threads (one per worker) move jobs through their
    states under one condition variable, which :meth:`wait` blocks on.
    ``listeners`` are called (under no lock) with each event.
    """

    def __init__(self, jobs, workers, *, env=None, reason=""):
        self._condition = threading.Condition()
        self._jobs = {job.eid: job for job in jobs}
        self._queue = sorted(jobs, key=lambda job: -job.expected)
        self._workers = []
        self._wanted = max(1, min(workers, len(jobs)))
        self._env = env if env is not None else worker_environment()
        self.reason = reason
        self.started = time.perf_counter()
        self.finished = None
        #: Set by :meth:`terminate`: the pool was stopped, not finished.
        self.stopped = False
        self.listeners = []
        self._stamp = environment_stamp()
        self._log = None

    # -- lifecycle -----------------------------------------------------------

    def start(self):
        self._log = _open_log()
        for index in range(self._wanted):
            self._spawn(index)
        return self

    def _spawn(self, index):
        flags = 0
        if sys.platform == "win32":
            flags = subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS
        try:
            process = subprocess.Popen(
                [sys.executable, "-m", "algan.rendering._precompile_worker"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._log,
                env=self._env,
                creationflags=flags,
            )
        except OSError:
            return
        worker = _Worker(process, index)
        with self._condition:
            self._workers.append(worker)
        threading.Thread(
            target=self._read,
            args=(worker,),
            name="algan-precompile-reader",
            daemon=True,
        ).start()

    def terminate(self):
        """Stop every worker now; a spec in flight is lost, finished ones are on disk."""
        self.stopped = True
        with self._condition:
            workers = list(self._workers)
            for job in self._queue:
                if job.state == "queued":
                    job.state = "failed"
                    job.reason = "pool stopped"
            self._queue = []
        for worker in workers:
            with contextlib.suppress(Exception):
                worker.process.kill()
        for worker in workers:
            with contextlib.suppress(Exception):
                worker.process.wait(timeout=5)

    def release(self):
        """Let every worker finish the spec in hand, then exit; queued work is dropped.

        What process exit does: closing a worker's stdin is its ``quit``, read
        once its current compile is done and dumped, so minutes of compiling
        in flight are not thrown away by a short script -- and the script does
        not wait for it either.
        """
        # Under the lock: a reader thread dispatches (writes to a worker's
        # stdin) while holding it, and must not see the pipe close mid-write.
        with self._condition:
            for job in self._queue:
                if job.state == "queued":
                    job.state = "failed"
                    job.reason = "the process exited"
            self._queue = []
            for worker in self._workers:
                with contextlib.suppress(Exception):
                    worker.process.stdin.close()

    def wait(self, timeout=None):
        """Block until every worker has exited. Returns whether it did."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while any(worker.alive for worker in self._workers):
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._condition.wait(remaining)
        return True

    @property
    def active(self):
        with self._condition:
            return any(worker.alive for worker in self._workers)

    # -- the protocol ----------------------------------------------------------

    def _send(self, worker, op):
        try:
            worker.process.stdin.write((json.dumps(op) + "\n").encode("utf-8"))
            worker.process.stdin.flush()
            return True
        except (OSError, ValueError):
            return False

    def _dispatch(self, worker):
        """Hand ``worker`` its next job, or tell it to quit. Called under the lock."""
        while self._queue:
            job = self._queue.pop(0)
            if job.state != "queued":
                continue
            if job.cid not in worker.contexts:
                if not self._send(
                    worker, {"op": "context", "id": job.cid, "context": job.context}
                ):
                    self._queue.insert(0, job)
                    return
                worker.contexts.add(job.cid)
            job.state = "running"
            job.worker = worker
            worker.job = job
            if not self._send(
                worker,
                {"op": "compile", "id": job.eid, "spec": job.spec, "context": job.cid},
            ):
                job.state = "queued"
                job.worker = None
                worker.job = None
                self._queue.insert(0, job)
            return
        self._send(worker, {"op": "quit"})

    def _read(self, worker):
        events = []
        stream = worker.process.stdout
        for raw in iter(stream.readline, b""):
            text = raw.decode("utf-8", "replace")
            if not text.startswith(_PROTOCOL_PREFIX):
                continue
            try:
                event = json.loads(text[len(_PROTOCOL_PREFIX) :])
            except ValueError:
                continue
            with self._condition:
                kind = event.get("event")
                if kind == "ready":
                    self._dispatch(worker)
                elif kind == "started":
                    job = self._jobs.get(event.get("id"))
                    if job is not None:
                        events.append(("started", job))
                elif kind == "done":
                    job = self._jobs.get(event.get("id"))
                    if job is not None:
                        job.status = event.get("status")
                        job.seconds = float(event.get("seconds") or 0.0)
                        job.reason = event.get("reason")
                        job.state = (
                            "done"
                            if job.status in ("cached", "offline-hit", "compiled")
                            else "failed"
                        )
                        worker.job = None
                        self._record(job)
                        events.append(("done", job))
                    self._dispatch(worker)
                elif kind == "fatal":
                    events.append(("fatal", event.get("reason")))
                self._condition.notify_all()
            self._emit(events)
        with contextlib.suppress(Exception):
            worker.process.wait(timeout=10)
        with self._condition:
            worker.alive = False
            job = worker.job
            if job is not None and job.state == "running":
                job.state = "failed"
                job.reason = f"worker exited with code {worker.process.returncode}"
                events.append(("done", job))
            worker.job = None
            if not any(w.alive for w in self._workers):
                self.finished = time.perf_counter()
                # Nothing will run what is left: a worker that died before
                # taking a job leaves it queued, and a waiter must not hang.
                for queued in self._queue:
                    if queued.state == "queued":
                        queued.state = "failed"
                        queued.reason = "no worker left"
                self._queue = []
                events.append(("finished", None))
            self._condition.notify_all()
        self._emit(events)
        if self.finished is not None:
            with contextlib.suppress(Exception):
                flush_manifest()

    def _emit(self, events):
        while events:
            event = events.pop(0)
            for listener in list(self.listeners):
                with contextlib.suppress(Exception):
                    listener(self, *event)

    def _record(self, job):
        if job.state != "done":
            return
        _note_entry(
            job.spec,
            job.context,
            seconds=job.seconds if job.status == "compiled" else None,
            confirmed=self._stamp,
        )

    def counts(self):
        """``(finished, total)`` over the pool's jobs."""
        with self._condition:
            jobs = list(self._jobs.values())
            return sum(job.state in ("done", "failed") for job in jobs), len(jobs)

    def job_for(self, eid):
        return self._jobs.get(eid)

    @property
    def jobs(self):
        return list(self._jobs.values())


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------


def worker_budget(job_count):
    """How many workers to start for ``job_count`` jobs; ``0`` means none.

    ``ALGAN_PRECOMPILE_JOBS`` wins when set. Otherwise one fewer than the
    logical CPUs (the render or authoring process keeps one), capped by free
    host memory, by free VRAM when rendering on CUDA, and at 8.
    """
    if env_is_set("ALGAN_PRECOMPILE_JOBS"):
        return max(0, min(env_int("ALGAN_PRECOMPILE_JOBS", 0), job_count))
    cpus = os.cpu_count() or 1
    budget = min(cpus - 1, 8, job_count)
    with contextlib.suppress(Exception):
        import psutil

        available = psutil.virtual_memory().available
        budget = min(budget, int((available - 2 * 2**30) // _WORKER_HOST_BYTES))
    with contextlib.suppress(Exception):
        from algan.settings._startup import render_device

        if render_device().type == "cuda":
            import torch

            free, total = torch.cuda.mem_get_info()
            budget = min(budget, int((free - total // 2) // _WORKER_VRAM_BYTES))
    return max(0, budget)


def skipped_reason():
    """``None`` when the pool can run in this process, else why not."""
    from algan.taichi_compat import BACKEND
    from algan.utils import taichi_source_key as sk

    if env_flag("ALGAN_PRECOMPILE_WORKER", False):
        return "this is a precompile worker"
    if env_is_set("ALGAN_PRECOMPILE_JOBS") and env_int("ALGAN_PRECOMPILE_JOBS", 0) <= 0:
        return "turned off by ALGAN_PRECOMPILE_JOBS=0"
    if BACKEND != "quadrants":
        return f"{BACKEND} has no fast-cache load path; precompiling is Quadrants-only"
    if not sk.is_applied():
        return f"the source-keyed index is off ({sk.skipped_reason()})"
    return None


# ---------------------------------------------------------------------------
# Choosing what to precompile
# ---------------------------------------------------------------------------


def pending_jobs(context, *, include_builtin=True, manifest=None, script=None):
    """Jobs for ``context`` that this installation has not confirmed as cached.

    Every manifest row recorded under ``context`` (and, given ``script``, used
    by that script) whose confirmation is not the current stamp, then every
    built-in spec whose settings requirements ``context`` meets and that is not
    already confirmed under it. Specs recorded under *other* contexts are left
    alone: a render with other settings would not look their artifacts up.
    """
    manifest = manifest if manifest is not None else read_manifest()
    stamp = environment_stamp()
    cid = context_id(context)
    jobs = {}
    for row in manifest["entries"].values():
        if row.get("context") != cid:
            continue
        if script is not None and script not in row.get("scripts", ()):
            continue
        if row.get("confirmed") == stamp:
            jobs.setdefault(entry_id(row["spec"], context), None)
            continue
        job = _Job(row["spec"], context, row.get("seconds"))
        jobs[job.eid] = job
    if include_builtin:
        raytracing = context["raytracing"]
        for spec, seconds, requires in read_builtin_specs(with_requirements=True):
            if any(raytracing.get(name) != value for name, value in requires.items()):
                continue
            eid = entry_id(spec, context)
            if eid in jobs:
                continue
            jobs[eid] = _Job(spec, context, seconds)
    return [job for job in jobs.values() if job is not None]


def worthwhile(jobs):
    return sum(job.expected for job in jobs) >= _MIN_POOL_SECONDS and len(jobs) >= 2


# ---------------------------------------------------------------------------
# The process-wide pool, and the render's side of it
# ---------------------------------------------------------------------------

_POOL = None
_POOL_CONTEXT_ID = None
_POOL_LOCK = threading.Lock()
_STARTER = None


def active_pool():
    return _POOL


#: ``(context id, script)`` pairs already found to have nothing worth a worker
#: in this process: what this process renders is recorded as confirmed, so the
#: answer only changes when another process edits the manifest.
_SETTLED = set()


def _start_pool(context, reason, script):
    """Start the process pool for ``script``'s specs under ``context``, if worth it."""
    global _POOL, _POOL_CONTEXT_ID
    cid = context_id(context)
    if (cid, script) in _SETTLED:
        return None
    jobs = pending_jobs(context, include_builtin=False, script=script)
    workers = worker_budget(len(jobs)) if worthwhile(jobs) else 0
    if workers <= 0:
        _SETTLED.add((cid, script))
        return None
    pool = PrecompilePool(jobs, workers, reason=reason)
    from algan.rendering import kernel_progress

    pool.listeners.append(kernel_progress.pool_event)
    with _POOL_LOCK:
        if _POOL is not None and _POOL.active:
            return None
        _POOL = pool
        _POOL_CONTEXT_ID = cid
    pool.start()
    kernel_progress.pool_started(pool, workers)
    return pool


def _stop_pool():
    global _POOL
    with _POOL_LOCK:
        pool, _POOL = _POOL, None
    if pool is not None:
        pool.terminate()


def stop_implicit_pool():
    """Wait out the import-time decision and stop any pool it started.

    For a caller that runs its own pool over the same work (``algan warmup``):
    two would only split the same cores between duplicate compiles.
    """
    starter = _STARTER
    if starter is not None:
        starter.join(timeout=10)
    _stop_pool()


def start_in_background(reason, script):
    """Decide on a background thread whether to start the pool for ``script``, and start it.

    Reading the manifest and walking the sources for the stamp stays off the
    caller's critical path. Any failure is silent: this is an optimization,
    and the render compiles what it needs regardless.
    """
    global _STARTER
    if skipped_reason() is not None or not script or _program_touched():
        return

    def decide():
        with contextlib.suppress(Exception):
            _start_pool(current_context(), reason, script)

    _STARTER = threading.Thread(
        target=decide, name="algan-precompile-start", daemon=True
    )
    _STARTER.start()


def start_at_import():
    """Called at the end of ``import algan``: overlap precompiling with authoring.

    Only in a process that is plainly running a scene script -- the same test
    the daemon handoff uses (``daemon_client.is_scene_script_run``), so not
    the ``algan`` CLI, a test runner, a notebook, ``python -c`` or ``-m`` --
    and only for the specializations *this script* used before. The render
    daemon starts the same thing at the start of each run
    (:func:`start_for_script`).
    """
    from algan.daemon_client import is_scene_script_run

    try:
        if not is_scene_script_run():
            return
    except Exception:  # noqa: BLE001
        return
    start_in_background("started with the script", current_script())


def start_for_script(path):
    """The render daemon is about to run ``path``: precompile what it used before.

    Only while the daemon's program has not touched the kernel cache yet --
    the first run after it starts, or after a reset; see
    :func:`before_first_materialization` for why later is too late.
    """
    with contextlib.suppress(Exception):
        start_in_background("started with the script", os.path.realpath(path))


def on_render_start():
    """The outermost render job is starting: a last chance to start a pool.

    For a script whose ``import algan`` could not tell it was one -- run by
    ``algan render -q`` in the CLI's own process, say -- and that has not
    touched the kernel cache yet. Otherwise the import-time decision stands.
    """
    if skipped_reason() is not None or _program_touched():
        return
    starter = _STARTER
    if starter is not None:
        starter.join(timeout=10)
        return
    script = current_script()
    if script:
        with contextlib.suppress(Exception):
            _start_pool(current_context(), "started at render", script)


# ---------------------------------------------------------------------------
# The first touch
# ---------------------------------------------------------------------------

#: ``id`` of the compiler program whose kernel cache this process has touched.
#: The compiler reads the offline cache's index the first time a program uses
#: it and never again (measured on Quadrants 1.3: a process that had compiled
#: one kernel did not see a second one another process dumped afterwards --
#: not through the source-keyed index, not through the offline cache, not
#: after dumping its own). So a worker's artifact helps this process only if
#: it is on disk before this process's first materialization. An ``id``, not
#: the object, so a program that is reset is not kept alive; a new program
#: after a reset starts untouched.
_TOUCHED_PROGRAM = None
_TOUCH_LOCK = threading.Lock()


def _program_touched():
    from algan.taichi_compat import program

    try:
        prog = program()
    except Exception:  # noqa: BLE001
        return False
    return prog is not None and id(prog) == _TOUCHED_PROGRAM


def before_first_materialization():
    """Called before every new specialization; waits for the pool before the first.

    The first materialization of a program is the moment the compiler reads
    the cache's index (see :data:`_TOUCHED_PROGRAM`), so this is where a
    running pool for the same settings is waited for: whatever it has not
    written by then this process could not use anyway. Authoring before this
    point overlaps the workers; the wait itself is the parallel compile, not
    the serial one it replaces. A pool for other settings (the script changed
    ``SETTINGS`` after import) is stopped instead.
    """
    global _TOUCHED_PROGRAM
    from algan.taichi_compat import program

    try:
        prog_id = id(program())
    except Exception:  # noqa: BLE001
        return
    if prog_id == _TOUCHED_PROGRAM:
        return
    with _TOUCH_LOCK:
        if prog_id == _TOUCHED_PROGRAM:
            return
        try:
            _wait_for_pool()
        except Exception:  # noqa: BLE001 -- the render compiles regardless
            pass
        finally:
            _TOUCHED_PROGRAM = prog_id


def _wait_for_pool():
    starter = _STARTER
    if starter is not None:
        starter.join(timeout=10)
    pool = _POOL
    if pool is None or not pool.active:
        return
    if context_id(current_context()) != _POOL_CONTEXT_ID:
        _stop_pool()
        return
    from algan.rendering import kernel_progress

    kernel_progress.waiting_for_pool(pool)
    pool.wait()
    kernel_progress.waited_for_pool(pool)


def _shutdown():
    """At exit: write what was learned, and let workers finish what they hold."""
    pool = _POOL
    if pool is not None and pool.active:
        pool.release()
    with contextlib.suppress(Exception):
        flush_manifest()


atexit.register(_shutdown)
