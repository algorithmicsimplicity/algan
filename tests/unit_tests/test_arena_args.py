"""The arena calling convention's two halves have to describe the same layout.

A converted kernel (`arena_args_taichi`) carries its layout twice: once as a
``_<KERNEL>_ARENA`` spec the launch wrapper packs from, and once as a binding
prologue that reads ``aoff[i]`` and ``ashp[j]`` by **literal index**. Nothing in
Python connects them. Get them out of step -- add a parameter, reorder two, drop
one -- and the kernel silently reads one array through another's offset, which
is wrong pixels, not a crash.

So this parses each converted kernel's prologue back out of its source and
checks it against the spec, index by index. It also holds the line the whole
conversion exists to hold: no kernel in the package asks for more than
`METAL_MANAGED_BUFFERS` ndarray arguments.

The launch sites are a third copy of the same layout, and they drift the same
way: a parameter added to ``call_params`` reaches the wrapper's arity check
only when that launch actually runs, so a stale site is a render-time
``ArenaBindingError`` on whichever path happens to hit it. The arity test below
reads the sites statically instead, which is why it is cheap enough to be in
the fast suite.
"""

import ast
import importlib
import inspect
from pathlib import Path

import pytest
import torch

import algan
from algan.rendering.raytracing.arena_args_taichi import (
    DTYPE_TAGS,
    ArenaBindingError,
    arena_packed,
    pack,
)

#: ndarray arguments Taichi can bind on Metal -- measured, and the reason the
#: convention exists (`algan/rendering/arena_binding.py`).
METAL_MANAGED_BUFFERS = 24

#: Every kernel converted to the convention: module, public (wrapper) name.
CONVERTED = [
    ("algan.rendering.raytracing.sheet_resolve_taichi", "sheet_resolve_shade"),
    ("algan.rendering.raytracing.wavefront_kernels_taichi", "wavefront_shade"),
    (
        "algan.rendering.raytracing.wavefront_kernels_taichi",
        "wavefront_traverse_events",
    ),
    ("algan.rendering.raytracing.raster_taichi", "raster_shadow_trace"),
    ("algan.rendering.raytracing.path_tracer_taichi", "pt_shade"),
]


def _launcher(module, name):
    """The `arena_packed` launcher behind a converted kernel's public name.

    The public name is an ordinary ``def`` that delegates to it, rather than
    the launcher bound directly: a module-level assignment to a lowercase name
    in a settings storage module reads as a shadowed settings field to
    ``raytracing_settings._shadowed_fields``, and ``raster_taichi`` is one.
    """
    return getattr(module, f"_{name}_launch")


def _kernel_ast(module, name):
    src = Path(inspect.getsourcefile(module)).read_text(encoding="utf-8")
    tree = ast.parse(src)
    return next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name
    )


def _prologue(fn):
    """The ``x = ti.static(ArenaView(arena_T, aoff[i], (ashp[j], ...)))`` lines.

    Returned as ``(name, tag, off_slot, [shape_slots])`` in source order.
    """
    out = []
    for stmt in fn.body:
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            continue
        call = stmt.value
        if not (
            isinstance(call, ast.Call) and getattr(call.func, "attr", None) == "static"
        ):
            continue
        view = call.args[0]
        if not (
            isinstance(view, ast.Call) and getattr(view.func, "id", None) == "ArenaView"
        ):
            continue
        buf, base, shape = view.args
        assert isinstance(base, ast.Subscript)
        assert base.value.id == "aoff"
        slots = [d.slice.value for d in shape.elts]
        out.append(
            (stmt.targets[0].id, buf.id.removeprefix("arena_"), base.slice.value, slots)
        )
    return out


@pytest.mark.parametrize(
    ("mod_name", "name"), CONVERTED, ids=[n for _m, n in CONVERTED]
)
def test_prologue_matches_the_spec_the_host_packs_from(mod_name, name):
    module = importlib.import_module(mod_name)
    wrapper = _launcher(module, name)
    spec = wrapper.arena_spec
    binds = _prologue(_kernel_ast(module, name + "_arena"))

    assert [b[0] for b in binds] == [s[0] for s in spec], (
        f"{name}: the binding prologue and {name}'s arena spec name different "
        "arrays, or name them in a different order"
    )

    shape_cursor = 0
    for i, ((bname, tag, off_slot, slots), (sname, stag, sndim)) in enumerate(
        zip(binds, spec)
    ):
        assert bname == sname
        assert off_slot == i, (
            f"{name}: {bname} reads aoff[{off_slot}] but the host writes its "
            f"offset at slot {i}"
        )
        assert tag == stag, (
            f"{name}: {bname} is bound from arena_{tag} but the spec packs it as {stag}"
        )
        assert len(slots) == sndim, (
            f"{name}: {bname}'s binding reads {len(slots)} shape slots for an "
            f"array the spec says has {sndim} dimensions"
        )
        assert slots == list(range(shape_cursor, shape_cursor + sndim)), (
            f"{name}: {bname} reads shape slots {slots}, but the host writes "
            f"its shape at {list(range(shape_cursor, shape_cursor + sndim))}"
        )
        shape_cursor += sndim


@pytest.mark.parametrize(
    ("mod_name", "name"), CONVERTED, ids=[n for _m, n in CONVERTED]
)
def test_the_wrapper_and_the_kernel_agree_about_the_arguments(mod_name, name):
    module = importlib.import_module(mod_name)
    wrapper = _launcher(module, name)
    kernel = getattr(module, name + "_arena")
    declared = list(inspect.signature(kernel.__wrapped__).parameters)

    order = [tag for _dtype, tag in DTYPE_TAGS]
    tags = sorted({t for _n, t, _d in wrapper.arena_spec}, key=order.index)
    tail = [f"arena_{t}" for t in tags] + ["aoff", "ashp"]
    assert declared[-len(tail) :] == tail, (
        f"{name}_arena's trailing parameters are {declared[-len(tail) :]}, but "
        f"its spec packs {tail}"
    )

    kept = declared[: -len(tail)]
    bound = [n for n, _t, _d in wrapper.arena_spec]
    assert set(kept) | set(bound) == set(wrapper.call_params), (
        f"{name}: the arguments callers pass are not the kernel's kept "
        "parameters plus its arena-bound ones"
    )
    assert not set(kept) & set(bound)
    # The kept parameters keep their original relative order, which is what
    # lets the wrapper re-split a positional call without a mapping table.
    assert kept == [p for p in wrapper.call_params if p in set(kept)]


def test_no_kernel_asks_for_more_ndarrays_than_metal_can_bind():
    """The point of the whole convention, checked over the whole package."""
    root = Path(algan.__file__).parent
    over = []
    for path in root.rglob("*.py"):
        if "external_libraries" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if not any(
                ast.unparse(d).startswith("ti.kernel") for d in node.decorator_list
            ):
                continue
            # NODE_ARG is a vector-element ndarray (raytrace_kernels_taichi):
            # still a buffer, still counts, and it cannot go through the arena
            # because a view yields a scalar element.
            n = sum(
                1
                for a in node.args.args
                if a.annotation is not None
                and (
                    "ndarray" in ast.unparse(a.annotation)
                    or ast.unparse(a.annotation) == "NODE_ARG"
                )
            )
            if n > METAL_MANAGED_BUFFERS:
                over.append(f"{node.name} ({n}) in {path.relative_to(root)}")
    assert not over, (
        "these kernels ask for more ndarray arguments than Taichi can bind on "
        "Metal; convert them to the arena convention (arena_args_taichi):\n  "
        + "\n  ".join(over)
    )


#: Converted kernels every launch site of which passes its arguments one by
#: one, so the count can be read off the source. Pinned rather than derived:
#: turning a site into a ``*args`` splat takes it out of static reach (see
#: ``sheet_resolve_shade``, which is absent for exactly that reason), and that
#: is a loss of coverage worth failing over rather than absorbing silently.
STATICALLY_COUNTABLE = {
    "wavefront_shade",
    "wavefront_traverse_events",
    "raster_shadow_trace",
    "pt_shade",
}


def _launch_sites(names):
    """Every ``name(...)`` call in the package, as ``name -> [(where, argc)]``.

    Calls that splat (``f(*args)``) or pass a keyword carry no static count and
    are reported separately. Only files that mention one of the names are
    parsed -- the whole-package walk above costs a couple of seconds, which is
    real money against the fast suite's budget.
    """
    root = Path(algan.__file__).parent
    counted = {n: [] for n in names}
    uncountable = {n: [] for n in names}
    for path in root.rglob("*.py"):
        if "external_libraries" in path.parts:
            continue
        src = path.read_text(encoding="utf-8")
        if not any(n in src for n in names):
            continue
        for node in ast.walk(ast.parse(src)):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name):
                called = func.id
            elif isinstance(func, ast.Attribute):
                called = func.attr
            else:
                continue
            if called not in names:
                continue
            where = f"{path.relative_to(root)}:{node.lineno}"
            if node.keywords or any(isinstance(a, ast.Starred) for a in node.args):
                uncountable[called].append(where)
            else:
                counted[called].append((where, len(node.args)))
    return counted, uncountable


@pytest.mark.fast
def test_every_launch_site_passes_the_arguments_the_wrapper_expects():
    """A converted kernel's call sites agree with its ``call_params``.

    In the fast suite deliberately, and the cheapest of the three layout
    checks: it is pure source reading, no Taichi and no render. The expensive
    proof that a gated variant still *compiles* stays where it has to be, in
    the render arms of ``test_weight_floor_exit.py`` -- but the drift those
    arms were pinned against (a parameter joining ``_WAVEFRONT_SHADE_PARAMS``,
    which moves every count after it) is visible from here, and was not
    otherwise caught until CI ran the unmarked suite.
    """
    expected = {}
    for mod_name, name in CONVERTED:
        module = importlib.import_module(mod_name)
        expected[name] = len(_launcher(module, name).call_params)

    counted, uncountable = _launch_sites(set(expected))

    wrong = [
        f"{name} at {where} passes {argc}, wrapper takes {expected[name]}"
        for name, sites in counted.items()
        for where, argc in sites
        if argc != expected[name]
    ]
    assert not wrong, (
        "these launch sites disagree with the argument list their wrapper "
        "packs from; the launch raises ArenaBindingError when this path "
        "renders:\n  " + "\n  ".join(wrong)
    )

    reachable = {n for n, sites in counted.items() if sites}
    assert reachable == STATICALLY_COUNTABLE, (
        "the set of kernels this test can actually check has changed, so it "
        "may now be passing vacuously. Gained: "
        f"{sorted(reachable - STATICALLY_COUNTABLE)}; lost: "
        f"{sorted(STATICALLY_COUNTABLE - reachable)} (a lost kernel is usually "
        "a site rewritten to splat its arguments -- those are listed as "
        f"{ {n: v for n, v in uncountable.items() if v} }). Update "
        "STATICALLY_COUNTABLE if the loss is intended."
    )


# --- the packer -------------------------------------------------------------


def _slices(nbytes=1 << 16):
    arena = torch.empty(nbytes, dtype=torch.uint8)
    a = arena[0:24].view(torch.float32).view(2, 3)
    b = arena[64 : 64 + 32].view(torch.int32).view(8)
    return arena, a, b


def test_pack_reports_offsets_into_the_shared_allocation():
    _arena, a, b = _slices()
    spec = (("a", "f32", 2), ("b", "i32", 1))
    arena_f32, arena_i32, aoff, ashp = pack(spec, (a, b))
    assert arena_f32.dtype is torch.float32
    assert arena_i32.dtype is torch.int32
    assert aoff.tolist() == [0, 16]  # elements, not bytes
    assert ashp.tolist() == [2, 3, 8]
    # The view really addresses the same memory the slice does.
    a.copy_(torch.arange(6, dtype=torch.float32).reshape(2, 3))
    assert torch.equal(arena_f32[0:6], a.reshape(-1))


def test_pack_rejects_an_argument_from_another_allocation():
    """Two arrays of one dtype have to share a buffer -- there is one base."""
    arena, a, b = _slices()
    b2 = arena[128 : 128 + 16].view(torch.int32).view(4)
    stray = torch.zeros(4, dtype=torch.int32)
    spec = (("a", "f32", 2), ("b", "i32", 1), ("b2", "i32", 1))
    pack(spec, (a, b, b2))  # all three in one allocation
    with pytest.raises(ArenaBindingError, match="not in the same allocation"):
        pack(spec, (a, b, stray))


def test_pack_accepts_a_dtype_whose_only_array_lives_elsewhere():
    """One array of a dtype is its own base, so where it lives cannot matter.

    The condition the kernel needs is one base pointer per dtype, not
    membership of any particular arena -- worth pinning, because it is what
    keeps a lone counter or a stub array from having to be special-cased.
    """
    _arena, a, _b = _slices()
    lone = torch.arange(4, dtype=torch.int32)
    _f32, arena_i32, aoff, _ashp = pack(
        (("a", "f32", 2), ("lone", "i32", 1)), (a, lone)
    )
    assert torch.equal(arena_i32[aoff[1] : aoff[1] + 4], lone)


def test_pack_rejects_a_mis_ordered_argument():
    _arena, a, b = _slices()
    with pytest.raises(ArenaBindingError, match="but the kernel binds it"):
        pack((("a", "f32", 2), ("b", "i32", 1)), (b, a))


def test_pack_rejects_the_wrong_rank():
    _arena, a, b = _slices()
    with pytest.raises(ArenaBindingError, match="dimensions"):
        pack((("a", "f32", 3), ("b", "i32", 1)), (a, b))


def test_pack_caches_the_tables_on_pointers_and_shapes_only():
    """The second pack of one tensor set reuses its tables; nothing else does.

    The tables are a pure function of every bound tensor's pointer, storage,
    dtype and shape, and the cache is keyed on exactly those integers -- never
    on a tensor -- so it can neither serve a stale layout nor keep an arena
    alive. A slice one element over, a re-allocated arena, and a spec of
    another rank each miss; the miss re-validates, which is why the wrong-rank
    error above still fires after a hit on the same tensors.
    """
    from algan.rendering.raytracing import arena_args_taichi as mod

    mod.clear_pack_cache()
    arena, a, b = _slices()
    spec = (("a", "f32", 2), ("b", "i32", 1))
    first = pack(spec, (a, b))
    assert len(mod._table_cache) == 1
    second = pack(spec, (a, b))
    assert len(mod._table_cache) == 1
    # The very same table tensor comes back; the arena views are rebuilt.
    assert (
        second[2].untyped_storage().data_ptr() == first[2].untyped_storage().data_ptr()
    )
    assert second[2].tolist() == first[2].tolist() == [0, 16]
    assert second[3].tolist() == [2, 3, 8]

    # One element over is another key with another offset.
    b_shifted = arena[68 : 68 + 32].view(torch.int32).view(8)
    third = pack(spec, (a, b_shifted))
    assert third[2].tolist() == [0, 17]
    assert len(mod._table_cache) == 2

    # A spec of another rank on the SAME tensors is a miss, so it is validated.
    with pytest.raises(ArenaBindingError, match="dimensions"):
        pack((("a", "f32", 3), ("b", "i32", 1)), (a, b))

    # The cache holds no tensor: the arena dies when its last user does.
    import gc
    import weakref

    probe = weakref.ref(arena)
    del arena, a, b, b_shifted, first, second, third
    gc.collect()
    assert probe() is None
    assert len(mod._table_cache) == 2
    mod.clear_pack_cache()
    assert not mod._table_cache


def test_the_wrapper_splits_a_positional_call_back_apart():
    _arena, a, b = _slices()
    seen = {}

    class _Mod:
        pass

    import sys

    mod = _Mod()
    sys.modules["_arena_args_test_mod"] = mod

    def fake_kernel(*args):
        seen["args"] = args

    mod.k = fake_kernel
    launch = arena_packed(
        "_arena_args_test_mod",
        "k",
        ("n", "a", "flag", "b"),
        (("a", "f32", 2), ("b", "i32", 1)),
    )
    launch(7, a, "flag-value", b)
    args = seen["args"]
    assert args[:2] == (7, "flag-value")  # kept, in their original order
    assert args[2].dtype is torch.float32  # then the arenas and the tables
    assert args[3].dtype is torch.int32
    assert args[4].tolist() == [0, 16]
    assert args[5].tolist() == [2, 3, 8]
    del sys.modules["_arena_args_test_mod"]


def test_the_wrapper_rejects_a_call_of_the_wrong_length():
    launch = arena_packed(
        "algan.rendering.raytracing.sheet_resolve_taichi",
        "sheet_resolve_shade_arena",
        ("n", "a"),
        (("a", "f32", 1),),
    )
    with pytest.raises(ArenaBindingError, match="takes 2 arguments, got 1"):
        launch(1)
