"""The fast launcher must be invisible except in the clock, on either compiler.

``algan/utils/taichi_fast_launch.py`` replaces ``Kernel.__call__`` with a
dispatcher that records a launch plan on the first launch of each
instantiation and replays the C++ set-arg calls on the next. Its claims, held
here against real kernels on whichever compiler is live:

* the fast key is at least as fine as the compiler's mapper key -- a
  different dtype, ndim, element shape or template value misses the plan and
  takes the original path, which is the only place validation happens;
* every argument the dispatcher does not replicate is routed to the original
  ``__call__``, observably (``STATS`` does not move, or the original raises);
* ``VERIFY`` re-derives the compiler's instantiation on every hit and raises
  when a plan disagrees with it;
* ``Kernel.reset`` drops the plans.

The end-to-end check is ``benchmarks/_taichi_fast_launch_check.py``, which
renders under ``ALGAN_TAICHI_FAST_LAUNCH_VERIFY=1`` and compares pixels across
arms.
"""

# No ``from __future__ import annotations`` here, deliberately: the kernels
# below carry ``ti.types.ndarray(...)`` annotations that taichi 1.7 reads as
# objects, not strings (it does not evaluate postponed annotations).
from collections import namedtuple

import pytest
import torch

from algan.rendering.taichi_runtime import init_taichi
from algan.taichi_compat import ti
from algan.utils import taichi_fast_launch
from algan.utils.taichi_fast_launch import STATS, skipped_reason

# Every kernel here is tiny and compiles in well under a second on the CPU
# arch, but each is a real materialization: the dispatcher's key can only be
# tested against the compiler's own instantiation choices.


@pytest.mark.fast
def test_the_dispatcher_is_live_on_this_compiler():
    """A compiler release the dispatcher does not know turns it off silently.

    Marked ``fast`` alone in this file, for the reason its warm-start twin
    gives (`tests/README.md`): it is the one test here that a change
    *elsewhere* breaks -- a compiler bump in ``pyproject.toml``, a new backend
    in ``taichi_compat``, an env-var rename. The rest are feature tests.
    """
    reason = skipped_reason()
    assert reason is None, f"fast-launch dispatcher is not installed: {reason}"
    assert taichi_fast_launch._APPLIED


@pytest.fixture(scope="module")
def kernels():
    """The kernels under test, built once: materialization is the slow part."""
    init_taichi()

    @ti.kernel
    def add_scaled(
        out: ti.types.ndarray(dtype=ti.f32, ndim=1),
        src: ti.types.ndarray(ndim=1),
        n: ti.i32,
        scale: ti.f32,
        flags: ti.template(),
    ):
        for i in range(n):
            out[i] = src[i] * scale
            if ti.static(bool(flags[0])):
                out[i] += 1.0
            if ti.static(bool(flags[1])):
                out[i] += 10.0

    @ti.kernel
    def vec_first(
        out: ti.types.ndarray(dtype=ti.f32, ndim=1),
        nodes: ti.types.ndarray(dtype=ti.types.vector(4, ti.f32), ndim=1),
        n: ti.i32,
    ):
        for i in range(n):
            out[i] = nodes[i][0] + nodes[i][3]

    @ti.kernel
    def any_rank(out: ti.types.ndarray(dtype=ti.f32), bump: ti.f32):
        for cell in ti.grouped(out):
            out[cell] += bump

    return {"add_scaled": add_scaled, "vec_first": vec_first, "any_rank": any_rank}


Flags = namedtuple("Flags", "a b")


def _primal(kernel):
    return kernel._primal


def _plans(kernel):
    return _primal(kernel).__dict__["_algan_fast_plans"]["plans"]


def _snapshot():
    return dict(STATS)


def _moved(before, key):
    return STATS[key] - before[key]


def test_a_repeat_launch_takes_the_fast_path_and_computes_the_same(kernels):
    add_scaled = kernels["add_scaled"]
    out = torch.zeros(8)
    src = torch.arange(8, dtype=torch.float32)
    add_scaled(out, src, 8, 2.0, (0, 0))  # records the plan
    expected = out.clone()
    before = _snapshot()
    out.zero_()
    add_scaled(out, src, 8, 2.0, (0, 0))
    assert _moved(before, "fast") == 1
    assert _moved(before, "slow") == 0
    assert torch.equal(out, expected)
    assert torch.equal(out, src * 2.0)


def test_a_different_template_tuple_misses_the_plan(kernels):
    add_scaled = kernels["add_scaled"]
    out = torch.zeros(4)
    src = torch.ones(4)
    add_scaled(out, src, 4, 1.0, (0, 0))
    plans_before = len(_plans(add_scaled))
    before = _snapshot()
    add_scaled(out, src, 4, 1.0, (1, 0))
    assert _moved(before, "slow") == 1
    assert _moved(before, "fast") == 0
    assert torch.equal(out, torch.full((4,), 2.0))
    add_scaled(out, src, 4, 1.0, (0, 1))
    assert torch.equal(out, torch.full((4,), 11.0))
    assert len(_plans(add_scaled)) == plans_before + 2
    # And the plans are distinct kernels: replaying each gives its own result.
    add_scaled(out, src, 4, 1.0, (1, 0))
    assert torch.equal(out, torch.full((4,), 2.0))
    add_scaled(out, src, 4, 1.0, (0, 0))
    assert torch.equal(out, torch.ones(4))


def test_a_different_dtype_misses_the_plan(kernels):
    """``src`` is declared without a dtype: the mapper specialises on it."""
    add_scaled = kernels["add_scaled"]
    out = torch.zeros(4)
    add_scaled(out, torch.ones(4, dtype=torch.float32), 4, 3.0, (0, 0))
    before = _snapshot()
    add_scaled(out, torch.ones(4, dtype=torch.int32), 4, 3.0, (0, 0))
    assert _moved(before, "slow") == 1
    assert torch.equal(out, torch.full((4,), 3.0))
    keys = list(_plans(add_scaled))
    assert any(torch.int32 in key for key in keys)
    assert any(torch.float32 in key for key in keys)


def test_a_different_ndim_misses_the_plan(kernels):
    """``out`` is declared without an ndim: the mapper specialises on it."""
    any_rank = kernels["any_rank"]
    flat = torch.zeros(6)
    any_rank(flat, 1.0)
    any_rank(flat, 1.0)
    before = _snapshot()
    grid = torch.zeros(2, 3)
    any_rank(grid, 0.5)
    assert _moved(before, "slow") == 1
    assert _moved(before, "fast") == 0
    any_rank(grid, 0.5)
    assert _moved(before, "fast") == 1
    assert torch.equal(flat, torch.full((6,), 2.0))
    assert torch.equal(grid, torch.full((2, 3), 1.0))
    assert {key[1] for key in _plans(any_rank)} == {1, 2}


def test_a_different_element_shape_misses_the_plan_and_fails_where_the_original_does(
    kernels,
):
    vec_first = kernels["vec_first"]
    out = torch.zeros(3)
    nodes = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    vec_first(out, nodes, 3)
    before = _snapshot()
    vec_first(out, nodes, 3)
    assert _moved(before, "fast") == 1
    assert torch.equal(out, nodes[:, 0] + nodes[:, 3])
    # A (3, 3) tensor has the wrong element extent. The compiler rejects it,
    # and it must be the compiler that does so: the plan for the (…, 4)
    # shape must not be handed a buffer of the wrong layout.
    before = _snapshot()
    with pytest.raises(Exception, match="element"):
        vec_first(out, torch.zeros(3, 3), 3)
    assert _moved(before, "fast") == 0
    assert all(key[-1] == (4,) for key in _plans(vec_first))


def test_the_element_dims_are_stripped_from_the_launch_shape(kernels):
    """The fast hit hands the runtime the outer shape, as the original does.

    If it passed the full (n, 4) shape, ``nodes[i]`` would index a different
    buffer layout and the sums would be wrong -- a wrong picture, not an
    error, so it is checked numerically.
    """
    vec_first = kernels["vec_first"]
    nodes = torch.rand(16, 4)
    out = torch.zeros(16)
    vec_first(out, nodes, 16)
    fast = torch.zeros(16)
    before = _snapshot()
    vec_first(fast, nodes, 16)
    assert _moved(before, "fast") == 1
    assert torch.equal(fast, out)
    assert torch.equal(fast, nodes[:, 0] + nodes[:, 3])


@pytest.mark.parametrize(
    "case",
    ["keyword", "non_contiguous", "requires_grad", "ndarray", "unsupported_template"],
)
def test_unsupported_launches_take_the_original_path(kernels, case):
    add_scaled = kernels["add_scaled"]
    out = torch.zeros(4)
    src = torch.ones(8)[::2] if case == "non_contiguous" else torch.ones(4)
    add_scaled(out, torch.ones(4), 4, 1.0, (0, 0))
    before = _snapshot()
    if case == "keyword":
        add_scaled(out, torch.ones(4), 4, 1.0, flags=(0, 0))
        assert torch.equal(out, torch.ones(4))
    elif case == "non_contiguous":
        # The original path is the one that rejects a strided view; a plan
        # replaying ``data_ptr`` over it would silently read the wrong lanes.
        with pytest.raises(Exception, match="contiguous"):
            add_scaled(out, src, 4, 1.0, (0, 0))
    elif case == "requires_grad":
        add_scaled(out, torch.ones(4, requires_grad=True), 4, 1.0, (0, 0))
        assert torch.equal(out, torch.ones(4))
    elif case == "ndarray":
        arr = ti.ndarray(ti.f32, shape=(4,))
        arr.fill(5.0)
        add_scaled(out, arr, 4, 1.0, (0, 0))
        assert torch.equal(out, torch.full((4,), 5.0))
    elif case == "unsupported_template":
        # A tuple subclass: the mapper keys it like a tuple, the dispatcher
        # only vouches for exact tuples.
        add_scaled(out, torch.ones(4), 4, 1.0, Flags(0, 0))
        assert torch.equal(out, torch.ones(4))
    assert _moved(before, "fast") == 0, (
        "the dispatcher replayed a plan for an argument it does not support"
    )


def test_disabling_routes_every_launch_to_the_original_and_keeps_the_plans(kernels):
    add_scaled = kernels["add_scaled"]
    out = torch.zeros(4)
    add_scaled(out, torch.ones(4), 4, 1.0, (0, 0))
    plans = dict(_plans(add_scaled))
    taichi_fast_launch.set_enabled(False)
    try:
        before = _snapshot()
        add_scaled(out, torch.ones(4), 4, 4.0, (0, 0))
        assert _moved(before, "fast") == 0
        assert _moved(before, "slow") == 0
        assert torch.equal(out, torch.full((4,), 4.0))
    finally:
        taichi_fast_launch.set_enabled(True)
    assert _plans(add_scaled) == plans
    before = _snapshot()
    add_scaled(out, torch.ones(4), 4, 1.0, (0, 0))
    assert _moved(before, "fast") == 1


def test_verify_mode_accepts_a_faithful_plan_and_rejects_a_corrupted_one(
    kernels, monkeypatch
):
    add_scaled = kernels["add_scaled"]
    out = torch.zeros(4)
    add_scaled(out, torch.ones(4), 4, 1.0, (0, 0))
    add_scaled(out, torch.ones(4), 4, 1.0, (1, 0))
    plans = _plans(add_scaled)
    monkeypatch.setattr(taichi_fast_launch, "VERIFY", True)
    before = _snapshot()
    add_scaled(out, torch.ones(4), 4, 1.0, (0, 0))
    assert _moved(before, "fast") == 1
    assert torch.equal(out, torch.ones(4))
    # Swap two plans: each key now points at the other instantiation.
    (key_a, plan_a), (key_b, plan_b) = list(plans.items())[:2]
    monkeypatch.setitem(plans, key_a, plan_b)
    monkeypatch.setitem(plans, key_b, plan_a)
    with pytest.raises(RuntimeError, match="instantiation mismatch"):
        add_scaled(out, torch.ones(4), 4, 1.0, (0, 0))


def test_reset_drops_the_plans(kernels):
    any_rank = kernels["any_rank"]
    primal = _primal(any_rank)
    out = torch.zeros(3)
    any_rank(out, 1.0)
    assert "_algan_fast_plans" in primal.__dict__
    primal.reset()
    assert "_algan_fast_plans" not in primal.__dict__
    # The kernel is whole after a reset: it re-materializes and re-records.
    any_rank(out, 1.0)
    before = _snapshot()
    any_rank(out, 1.0)
    assert _moved(before, "fast") == 1
    assert torch.equal(out, torch.full((3,), 3.0))
