"""The source-keyed cache index must be sound before it is fast.

``algan/utils/taichi_source_key.py`` maps a key over a kernel's source, its
arguments and every Python value its body reads to the C++ cache key of the IR
that source produced, so a warm process can skip the AST transform. The claim
is that the key is a pure function of everything that reaches the IR; the
tests here hold the value rules, the closure/global walk and the hook control
flow to that without compiling anything, and then check the whole mechanism in
subprocesses: a ``@ti.func`` edit invalidates, a template value is keyed by
value, and a ``Square`` frame rendered through a hit is byte-identical to one
rendered without the index, with verify mode clean on top.

The audit of the same claim inside a real render is
``benchmarks/_taichi_source_key_check.py``.
"""

from __future__ import annotations

import enum
import json
import math
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

from algan.taichi_compat import BACKEND, backend_version
from algan.utils import taichi_source_key as sk

quadrants_only = pytest.mark.skipif(
    BACKEND != "quadrants", reason="the source-keyed index is Quadrants-only"
)

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Off the daemon, and the index switched on, for every child below.
CHILD_ENV = {
    "ALGAN_USE_DAEMON": "0",
    "ALGAN_AUTO_DAEMON": "0",
    "ALGAN_TAICHI_SOURCE_KEY": "1",
    "ALGAN_TAICHI_BACKEND": BACKEND,
}


def _render(value):
    ctx = sk._KeyContext(BACKEND)
    sk._hash_value(value, ctx)
    return tuple(ctx.out)


def _walk(function):
    ctx = sk._KeyContext(BACKEND)
    sk._hash_function(function, ctx, 0)
    return tuple(ctx.out)


# --- fixtures the walk tests read through module globals ---------------------

_PROBE_CONSTANT = 1


class _Config:
    limit = 3


_CFG = _Config()


class _Mode(enum.Enum):
    FAST = 1
    SLOW = 2


def _reads_probe():
    return _PROBE_CONSTANT


def _reads_chain():
    return _CFG.limit


def _reads_bare_instance():
    return _CFG


def _reads_module_as_value():
    return math


def _reads_undefined_name():
    return never_defined_anywhere  # noqa: F821 -- the point of the test


def _reads_imports_locally():
    import math as m
    from os import sep

    return m.pi, sep


def _reads_in_a_comprehension():
    return [_PROBE_CONSTANT + i for i in range(3)]


def _closure_over(k):
    def add(x):
        return x + k

    return add


_FUNC_GLOBAL = 1


class _KernelScopeClass:
    """A class of the shape ``ArenaView`` has: read bare in kernel scope.

    Carries a class-level constant and a compiler callable, which are the two
    members whose *value* the class-body source hash cannot see.
    """

    LIMIT = 2

    @staticmethod
    def scale(x):
        return x * _FUNC_GLOBAL


# --- the gate -----------------------------------------------------------------


@quadrants_only
@pytest.mark.fast
def test_the_index_installs_on_this_compiler():
    """The internals the patch replicates must still be where it looks.

    Marked ``fast`` alone in this file, for the reason ``tests/README.md``
    gives for its warm-start twin: the index is version-gated to compiler
    internals (``Kernel._try_load_fastcache``, ``src_hasher``,
    ``Program.load_fast_cache``), so a compiler bump in ``pyproject.toml`` turns
    it off from elsewhere, and an index that silently stands down reads exactly
    like a slow machine. Reads the gate, installs nothing.
    """
    entry_points = sk._quadrants_entry_points(tuple(backend_version()))
    assert not isinstance(entry_points, str), (
        f"the source-keyed index cannot install on this compiler: {entry_points}"
    )


def test_it_is_off_unless_asked_for():
    reason = sk.skipped_reason()
    if os.environ.get("ALGAN_TAICHI_SOURCE_KEY", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        assert reason is None or BACKEND != "quadrants"
    else:
        assert reason is not None
        assert "ALGAN_TAICHI_SOURCE_KEY=1" in reason


# --- value rules --------------------------------------------------------------


def test_scalars_are_keyed_by_type_and_value():
    renderings = {_render(v) for v in (1, True, 1.0, "1", b"1", None, 2)}
    assert len(renderings) == 7, "two distinct scalars rendered alike"
    assert _render(0.1 + 0.2) != _render(0.3), (
        "floats must be keyed exactly, not rounded"
    )


def test_containers_are_keyed_structurally():
    assert _render(((1, 2), (3,))) != _render(((1,), (2, 3)))
    assert _render((1, 2)) != _render([1, 2]), (
        "a tuple and a list are different template values"
    )
    assert _render({"a": 1, "b": 2}) == _render({"b": 2, "a": 1})
    assert _render({"a": 1, "b": 2}) != _render({"a": 1, "b": 3})
    assert _render(frozenset({1, 2})) == _render(frozenset({2, 1}))
    assert _render(((1, (2, (3, None))),)) != _render(((1, (2, (3, 0))),))


def test_enums_and_dtypes_are_keyed_by_name():
    from algan.taichi_compat import ti

    assert _render(_Mode.FAST) != _render(_Mode.SLOW)
    assert _render(_Mode.FAST) != _render(1), "an enum member is not its value"
    assert _render(ti.f32) != _render(ti.i32)
    assert _render(ti.f32) == _render(ti.f32)
    assert any("dtype:f32" in part for part in _render(ti.f32))


def test_a_torch_dtype_is_keyed_by_name():
    import torch

    assert _render(torch.float32) != _render(torch.float16)


def test_unknown_values_poison():
    with pytest.raises(sk.Poison):
        _render(_CFG)
    with pytest.raises(sk.Poison):
        _render(math)
    with pytest.raises(sk.Poison):
        _render(object())


# --- the walk -----------------------------------------------------------------


@quadrants_only
def test_functions_are_keyed_by_source_and_by_closure():
    assert _walk(_closure_over(1)) == _walk(_closure_over(1))
    assert _walk(_closure_over(1)) != _walk(_closure_over(2)), (
        "two closures over the same source differ only in a captured value; the key must see it"
    )


@quadrants_only
def test_a_global_constant_change_changes_the_key(monkeypatch):
    before = _walk(_reads_probe)
    monkeypatch.setattr(sys.modules[__name__], "_PROBE_CONSTANT", 2)
    assert _walk(_reads_probe) != before


@quadrants_only
def test_attribute_chains_resolve_to_the_live_value(monkeypatch):
    before = _walk(_reads_chain)
    assert any(part == "int:3" for part in before), before
    monkeypatch.setattr(_CFG, "limit", 4)
    assert _walk(_reads_chain) != before


@quadrants_only
def test_locally_imported_modules_are_followed():
    out = _walk(_reads_imports_locally)
    assert f"float:{math.pi!r}" in out, (
        "`import math as m; m.pi` must resolve through the import"
    )
    assert f"str:{os.sep!r}" in out, "`from os import sep` must bind the attribute"


@quadrants_only
def test_nested_code_objects_are_walked_and_their_locals_are_not_poison():
    out = _walk(_reads_in_a_comprehension)
    assert "int:1" in out
    assert "builtin:range" in out


@quadrants_only
def test_compiler_names_are_exempt_from_the_walk():
    from algan.taichi_compat import ti  # noqa: F401 -- bound into this module's globals

    def uses_compiler():
        return ti.static(ti.f32)

    out = _walk(uses_compiler)
    assert any(part.startswith("compiler-attr:") for part in out), out


@quadrants_only
def test_a_bare_instance_or_module_poisons_the_walk():
    with pytest.raises(sk.Poison, match="no key rule"):
        _walk(_reads_bare_instance)
    with pytest.raises(sk.Poison, match="module"):
        _walk(_reads_module_as_value)


@quadrants_only
def test_an_unresolvable_global_poisons_the_walk():
    with pytest.raises(sk.Poison, match="unresolvable global"):
        _walk(_reads_undefined_name)


@quadrants_only
def test_a_class_used_in_kernel_scope_is_keyed_by_its_source():
    from algan.rendering.raytracing.arena_args_taichi import (
        ArenaBindingError,
        ArenaView,
    )

    def builds_a_view():
        return ArenaView

    def raises_an_error():
        return ArenaBindingError

    view = _walk(builds_a_view)
    assert any(part.startswith("class:") and "ArenaView" in part for part in view), view
    assert any(part == "method:__getitem__" for part in view), (
        "the class's methods are walked"
    )
    assert view != _walk(raises_an_error)


@quadrants_only
def test_a_class_attribute_is_in_the_key(monkeypatch):
    """A class read in kernel scope is keyed by its attributes, not only its source.

    ``ArenaView`` is read exactly this way -- ``ti.static(ArenaView(...))``
    leaves the bare class as the chain leaf -- so a class attribute assigned
    after the class body ran (``Cfg.LIMIT = from_env()``) reaches the IR through
    a template or a helper without changing one character of the source. The
    class-body source hash cannot see it; the key must.
    """
    before = _render(_KernelScopeClass)
    assert any(part == "int:2" for part in before), before
    monkeypatch.setattr(_KernelScopeClass, "LIMIT", 3)
    assert _render(_KernelScopeClass) != before


@quadrants_only
def test_a_callable_class_member_is_walked_for_its_own_references(monkeypatch):
    """A ``@ti.func``/``@staticmethod`` in a class body reads globals of its own.

    ``vars(cls)`` hands back a compiler wrapper (or a ``staticmethod``), not a
    plain function, and the class-body source hash covers only the text of the
    member -- never the value of the global it reads. Keying the class by its
    source alone would serve a kernel compiled against the old value.
    """
    before = _render(_KernelScopeClass)
    monkeypatch.setattr(sys.modules[__name__], "_FUNC_GLOBAL", 2)
    assert _render(_KernelScopeClass) != before


@quadrants_only
def test_every_algan_kernel_body_walks_without_poison():
    """The rules cover what Algan's own kernels actually read.

    Static half of the soundness story: every ``@ti.kernel`` under
    ``algan.rendering`` resolves each global it reads, transitively through its
    ``@ti.func`` callees, without hitting a value the rules refuse. Template
    arguments are only known at launch and are covered by the subprocess tests.
    """
    import importlib
    import pkgutil

    import algan.rendering.raytracing as package
    from algan.taichi_compat import submodule

    for info in pkgutil.iter_modules(package.__path__):
        if info.name.endswith("_taichi"):
            importlib.import_module(f"{package.__name__}.{info.name}")
    kernels = submodule("lang.impl").get_runtime().kernels
    seen = set()
    poisoned = {}
    for kernel in kernels:
        function = kernel.func
        if function in seen or not getattr(function, "__module__", "").startswith(
            "algan."
        ):
            continue
        seen.add(function)
        ctx = sk._KeyContext(BACKEND)
        ctx.visited.add(id(function))
        try:
            sk._hash_references(function, ctx, 0)
        except sk.Poison as poison:
            poisoned[sk.kernel_qualname(kernel)] = str(poison)
    assert len(seen) > 40, "the kernel modules did not register their kernels"
    assert not poisoned, poisoned


# --- the hooks, against stubs -------------------------------------------------


class _Observations:
    cache_key_generated = False
    cache_validated = False


class _Runtime:
    src_ll_cache = True


class _StubKernel:
    def __init__(self):
        self.runtime = _Runtime()
        self.quadrants_callable = None
        self.autodiff_mode = "NONE"
        self.fast_checksum = None
        self.src_ll_cache_observations = _Observations()
        self.func = _reads_probe


class _StubHasher:
    def __init__(self, value=None):
        self.value = value
        self.loaded = []
        self.stored = []

    def load(self, key):
        self.loaded.append(key)
        return self.value


class _CacheValue:
    def __init__(self, frontend_cache_key):
        self.frontend_cache_key = frontend_cache_key


@pytest.fixture
def hooks(monkeypatch):
    hasher = _StubHasher()
    original_try_load = lambda self, args, key: "ORIGINAL"  # noqa: E731

    def original_store(frontend_cache_key, fast_cache_key, *args, **kwargs):
        hasher.stored.append((frontend_cache_key, fast_cache_key))

    try_load, store = sk._build_hooks(
        None, hasher, "NONE", original_try_load, original_store
    )
    monkeypatch.setattr(sk, "STATS", dict.fromkeys(sk.STATS, 0))
    monkeypatch.setattr(sk, "POISONED", {})
    monkeypatch.setattr(sk, "_WARNED", set())
    monkeypatch.setattr(sk, "_PENDING_VERIFY", {})
    monkeypatch.delenv("ALGAN_TAICHI_SOURCE_KEY_VERIFY", raising=False)
    return try_load, store, hasher


def test_a_poisoned_key_warns_once_and_never_sets_the_checksum(hooks, monkeypatch):
    try_load, _store, hasher = hooks
    monkeypatch.setattr(sk, "compute_key", lambda kernel, args: (None, "because"))
    kernel = _StubKernel()
    with pytest.warns(RuntimeWarning, match="because"):
        assert try_load(kernel, (), ("k",)) is None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert try_load(kernel, (), ("k",)) is None
    assert not caught, "the same kernel must warn once, not per materialization"
    assert kernel.fast_checksum is None, (
        "a poisoned kernel must leave nothing for launch_kernel to store"
    )
    assert hasher.loaded == []
    assert sk.STATS["poisoned"] == 2
    assert list(sk.POISONED) == [sk.kernel_qualname(kernel)]


def test_a_miss_sets_the_checksum_so_the_compile_is_indexed(hooks, monkeypatch):
    try_load, _store, hasher = hooks
    monkeypatch.setattr(sk, "compute_key", lambda kernel, args: ("KEY", None))
    kernel = _StubKernel()
    assert try_load(kernel, (), ("k",)) is None
    assert kernel.fast_checksum == "KEY"
    assert hasher.loaded == ["KEY"]
    assert sk.STATS["misses"] == 1
    assert sk.STATS["keyed"] == 1


def test_a_hit_restores_and_returns_the_used_set(hooks, monkeypatch):
    try_load, _store, hasher = hooks
    hasher.value = _CacheValue("CPP")
    monkeypatch.setattr(sk, "compute_key", lambda kernel, args: ("KEY", None))
    restored = []
    monkeypatch.setattr(
        sk,
        "_restore_from_cache_value",
        lambda self, key, cache_value, *_: restored.append(
            (key, cache_value.frontend_cache_key)
        )
        or {"used"},
    )
    kernel = _StubKernel()
    assert try_load(kernel, (), ("k",)) == {"used"}
    assert restored == [(("k",), "CPP")]
    assert sk.STATS["hits"] == 1


def test_a_hit_whose_artifact_is_gone_counts_as_a_miss(hooks, monkeypatch):
    try_load, _store, hasher = hooks
    hasher.value = _CacheValue("CPP")
    monkeypatch.setattr(sk, "compute_key", lambda kernel, args: ("KEY", None))
    monkeypatch.setattr(sk, "_restore_from_cache_value", lambda *args: None)
    assert try_load(_StubKernel(), (), ("k",)) is None
    assert sk.STATS["misses"] == 1
    assert sk.STATS["hits"] == 0


def test_verify_mode_takes_the_full_path_and_compares_the_cpp_key(hooks, monkeypatch):
    try_load, store, hasher = hooks
    hasher.value = _CacheValue("CPP")
    monkeypatch.setattr(sk, "compute_key", lambda kernel, args: ("KEY", None))
    monkeypatch.setenv("ALGAN_TAICHI_SOURCE_KEY_VERIFY", "1")
    kernel = _StubKernel()
    assert try_load(kernel, (), ("k",)) is None, "verify mode must not shortcut"
    assert kernel.fast_checksum == "KEY"
    assert sk.STATS["hits"] == 0
    store("CPP", "KEY", set())
    assert sk.STATS["verified"] == 1
    assert hasher.stored == [("CPP", "KEY")], "the original store still runs"

    assert try_load(kernel, (), ("k",)) is None
    with pytest.raises(RuntimeError, match="VERIFY mismatch"):
        store("DIFFERENT", "KEY", set())


def test_quadrants_own_fastcache_is_deferred_to(hooks):
    try_load, _store, _hasher = hooks
    kernel = _StubKernel()
    kernel.quadrants_callable = type("Callable", (), {"is_pure": True})()
    assert try_load(kernel, (), ("k",)) == "ORIGINAL"


# --- subprocess checks --------------------------------------------------------


def _run_child(script, env, cwd):
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, **CHILD_ENV, **env},
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"child failed:\n{result.stdout}\n{result.stderr}"
    for line in result.stdout.splitlines():
        if line.startswith("REPORT "):
            return json.loads(line[len("REPORT ") :]), result
    raise AssertionError(
        f"no REPORT line in child output:\n{result.stdout}\n{result.stderr}"
    )


_PROBE_KERNELS = """
from algan.taichi_compat import ti


@ti.func
def helper(x):
    return x * {factor}


@ti.kernel
def scale(out: ti.types.ndarray(dtype=ti.f32, ndim=1)):
    for i in out:
        out[i] = helper(out[i])


@ti.kernel
def scale_by(out: ti.types.ndarray(dtype=ti.f32, ndim=1), mode: ti.template()):
    for i in out:
        if ti.static(mode == 1):
            out[i] = out[i] + 1.0
        else:
            out[i] = out[i] + 10.0
"""

_PROBE_CHILD = """
import json, sys
sys.path.insert(0, {tmp!r})
import algan  # noqa: F401
from algan.rendering.taichi_runtime import init_taichi
from algan.utils import taichi_source_key as sk
import torch
import probe_kernels

assert sk.skipped_reason() is None, sk.skipped_reason()
init_taichi()
scaled = torch.ones(4)
probe_kernels.scale(scaled)
one = torch.zeros(4)
probe_kernels.scale_by(one, 1)
ten = torch.zeros(4)
probe_kernels.scale_by(ten, 2)
print("REPORT " + json.dumps({{
    "scaled": scaled.tolist(), "one": one.tolist(), "ten": ten.tolist(),
    "hits": sk.STATS["hits"], "misses": sk.STATS["misses"], "poisoned": sk.STATS["poisoned"],
}}))
"""


@quadrants_only
def test_a_func_edit_invalidates_and_template_values_are_keyed_by_value(tmp_path):
    """Four processes over one temp module: warm, hit, edit the ``@ti.func``, hit again.

    The kernel source never changes, so Algan's key is the same before and
    after the edit; what invalidates is Quadrants' re-hash of the visited
    functions on load -- the half of the mechanism this module reuses rather
    than replaces. The templated kernel is launched with two values so both
    specializations are keyed, and by value: the second process hits on all
    three materializations.
    """
    module = tmp_path / "probe_kernels.py"
    script = _PROBE_CHILD.format(tmp=str(tmp_path))

    module.write_text(_PROBE_KERNELS.format(factor=2), encoding="utf-8")
    warm, _ = _run_child(script, {}, tmp_path)
    assert warm["scaled"] == [2.0] * 4
    assert warm["one"] == [1.0] * 4
    assert warm["ten"] == [10.0] * 4
    assert warm["poisoned"] == 0
    assert warm["misses"] == 3

    hit, _ = _run_child(script, {}, tmp_path)
    assert hit["scaled"] == [2.0] * 4
    assert (hit["hits"], hit["misses"], hit["poisoned"]) == (3, 0, 0)

    module.write_text(_PROBE_KERNELS.format(factor=3), encoding="utf-8")
    edited, _ = _run_child(script, {}, tmp_path)
    assert edited["scaled"] == [3.0] * 4, (
        "a stale artifact was served after the func changed"
    )
    assert edited["misses"] >= 1
    assert edited["poisoned"] == 0

    again, _ = _run_child(script, {}, tmp_path)
    assert again["scaled"] == [3.0] * 4
    assert (again["hits"], again["misses"]) == (3, 0)


_CLASS_TEMPLATE_KERNELS = """
import os

from algan.taichi_compat import ti

GAIN = float(os.environ["PROBE_GAIN"])


class Ops:
    LIMIT = float(os.environ["PROBE_LIMIT"])

    @ti.func
    def scale(x):
        return x * GAIN


@ti.kernel
def bake_limit(out: ti.types.ndarray(dtype=ti.f32, ndim=1), C: ti.template()):
    for i in out:
        out[i] = C.LIMIT


@ti.kernel
def bake_scale(out: ti.types.ndarray(dtype=ti.f32, ndim=1), C: ti.template()):
    for i in out:
        out[i] = C.scale(out[i])
"""

_CLASS_TEMPLATE_CHILD = """
import json, sys
sys.path.insert(0, {tmp!r})
import algan  # noqa: F401
from algan.rendering.taichi_runtime import init_taichi
from algan.utils import taichi_source_key as sk
import torch
import class_template_kernels as probe

assert sk.skipped_reason() is None, sk.skipped_reason()
init_taichi()
limit = torch.zeros(4)
probe.bake_limit(limit, probe.Ops)
scaled = torch.ones(4)
probe.bake_scale(scaled, probe.Ops)
print("REPORT " + json.dumps({{
    "limit": limit.tolist(), "scaled": scaled.tolist(),
    "hits": sk.STATS["hits"], "misses": sk.STATS["misses"], "poisoned": sk.STATS["poisoned"],
}}))
"""


@quadrants_only
def test_a_class_passed_as_a_template_is_keyed_by_what_the_transform_reads(tmp_path):
    """The end-to-end half of the two tests above: a stale artifact, or not.

    One module, one source text, three processes. The class-level constant and
    the global its ``@ti.func`` member reads both come from the environment, so
    the kernel source, the class source and the func source are byte-identical
    across all three -- exactly the case where only the key can tell the runs
    apart. A key that misses either serves the first process's compiled
    constant and the numbers come out wrong rather than the run failing.
    """
    (tmp_path / "class_template_kernels.py").write_text(
        _CLASS_TEMPLATE_KERNELS, encoding="utf-8"
    )
    script = _CLASS_TEMPLATE_CHILD.format(tmp=str(tmp_path))

    warm, _ = _run_child(script, {"PROBE_LIMIT": "1", "PROBE_GAIN": "1"}, tmp_path)
    assert warm["limit"] == [1.0] * 4
    assert warm["scaled"] == [1.0] * 4
    assert warm["poisoned"] == 0

    again, _ = _run_child(script, {"PROBE_LIMIT": "1", "PROBE_GAIN": "1"}, tmp_path)
    assert (again["hits"], again["misses"]) == (2, 0), (
        "an unchanged run must still hit; the key is over-keyed otherwise"
    )

    limit_changed, _ = _run_child(
        script, {"PROBE_LIMIT": "7", "PROBE_GAIN": "1"}, tmp_path
    )
    assert limit_changed["limit"] == [7.0] * 4, (
        "a class attribute the transform baked into the kernel is not in the key"
    )

    gain_changed, _ = _run_child(
        script, {"PROBE_LIMIT": "7", "PROBE_GAIN": "9"}, tmp_path
    )
    assert gain_changed["scaled"] == [9.0] * 4, (
        "a global read by a @ti.func in a class body is not in the key"
    )


_SQUARE_CHILD = """
import json, os, sys
import algan  # noqa: F401
from algan import PREVIEW, SETTINGS, Scene, Square
from algan.utils import taichi_source_key as sk

SETTINGS.video.set(PREVIEW)
with Scene() as scene:
    Square().spawn()
    scene.save_frame(sys.argv[1], video_settings=PREVIEW, overwrite=True)
print("REPORT " + json.dumps(dict(sk.STATS, skipped=sk.skipped_reason(), poisoned_kernels=sorted(sk.POISONED))))
"""


@quadrants_only
def test_a_square_frame_through_the_index_is_byte_identical_and_verifies(tmp_path):
    """The end-to-end claim, in four processes.

    ``off`` renders with the index disabled; ``warm`` renders with it on and
    fills the index; ``hit`` renders again and must hit on every keyed kernel
    with none poisoned; ``verify`` renders under
    ``ALGAN_TAICHI_SOURCE_KEY_VERIFY=1``, which re-derives the C++ key for
    every hit and raises on a mismatch. All four frames must be the same bytes.
    """

    def render(name, extra):
        frame = tmp_path / f"{name}.png"
        script = _SQUARE_CHILD
        result = subprocess.run(
            [sys.executable, "-c", script, str(frame)],
            env={**os.environ, **CHILD_ENV, **extra},
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"{name} failed:\n{result.stdout}\n{result.stderr}"
        )
        report = next(
            json.loads(line[len("REPORT ") :])
            for line in result.stdout.splitlines()
            if line.startswith("REPORT ")
        )
        return report, frame.read_bytes()

    off, off_bytes = render("off", {"ALGAN_TAICHI_SOURCE_KEY": "0"})
    assert off["skipped"] is not None
    assert off["keyed"] == 0

    warm, warm_bytes = render("warm", {})
    assert warm["skipped"] is None
    assert warm["keyed"] > 0, warm
    assert warm["poisoned"] == 0, warm

    hit, hit_bytes = render("hit", {})
    assert hit["poisoned"] == 0, hit["poisoned_kernels"]
    assert hit["hits"] == hit["keyed"] > 0, hit
    assert hit["misses"] == 0, hit

    verify, verify_bytes = render("verify", {"ALGAN_TAICHI_SOURCE_KEY_VERIFY": "1"})
    assert verify["verified"] == verify["keyed"] > 0, verify
    assert verify["hits"] == 0, "verify mode must never take the shortcut"

    assert warm_bytes == off_bytes, (
        "a render that filled the index differs from one without it"
    )
    assert hit_bytes == off_bytes, (
        "a render served from the index differs from one without it"
    )
    assert verify_bytes == off_bytes
