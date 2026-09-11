"""Real ndarray plans: specialize on type, rebind all runtime state on every hit."""

# Kernel annotations must remain runtime objects, not postponed strings.
# ruff: noqa: I002
import gc
import weakref
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch

from algan.rendering.taichi_runtime import init_taichi
from algan.taichi_compat import BACKEND, ti
from algan.utils import taichi_fast_launch as fast

pytestmark = pytest.mark.skipif(
    BACKEND != "quadrants", reason="Quadrants ndarray dispatcher"
)


@pytest.fixture(autouse=True)
def verify(monkeypatch):
    init_taichi()
    monkeypatch.setattr(fast, "VERIFY", True)
    monkeypatch.setattr(fast, "ENABLED", True)
    yield
    fast.set_telemetry_enabled(False)
    fast.launch_report(reset=True)


@pytest.fixture
def kernels():
    @ti.kernel
    def scaled(
        out: ti.types.ndarray(dtype=ti.f32, ndim=1),
        src: ti.types.ndarray(ndim=1),
        n: ti.i32,
        scale: ti.f32,
        large: ti.i64,
        unsigned: ti.u32,
        flags: ti.template(),
    ):
        for i in range(n):
            out[i] = src[i] * scale + ti.cast(large + unsigned, ti.f32)
            if ti.static(flags[0]):
                out[i] += 1.0

    @ti.kernel
    def vectors(
        out: ti.types.ndarray(dtype=ti.f32, ndim=1),
        src: ti.types.ndarray(dtype=ti.types.vector(4, ti.f32), ndim=1),
    ):
        for i in out:
            out[i] = src[i][0] + src[i][3]

    @ti.kernel
    def matrices(
        out: ti.types.ndarray(dtype=ti.f32, ndim=1),
        src: ti.types.ndarray(dtype=ti.types.matrix(2, 2, ti.f32), ndim=1),
    ):
        for i in out:
            out[i] = src[i][0, 1] + src[i][1, 0]

    @ti.kernel
    def rank(out: ti.types.ndarray(dtype=ti.f32), value: ti.f32):
        for i in ti.grouped(out):
            out[i] = value

    @ti.kernel
    def grid(out: ti.types.ndarray(dtype=ti.f32, ndim=2)):
        for i, j in out:
            out[i, j] = i * 10.0 + j

    return scaled, vectors, matrices, rank, grid


def array(values, dtype=ti.f32):
    values = np.asarray(values, dtype=np.float32 if dtype == ti.f32 else np.int32)
    arr = ti.ndarray(dtype, shape=values.shape)
    arr.from_numpy(values)
    return arr


def plans(kernel):
    return kernel._primal._algan_fast_plans["plans"]


def test_rebinds_arrays_extents_scalars_and_templates(kernels):
    scaled = kernels[0]
    out, src = array([0, 0, 0]), array([1, 2, 3])
    scaled(out, src, 3, 2.0, 300, 512, (False, None))
    count = len(plans(scaled))
    for n, scale, large, unsigned in [
        (5, 0.5, 10000, 700),
        (2, -2.0, -900, 1024),
        (3, 4.0, 9999, 512),
    ]:
        out, src = array(np.zeros(n)), array(np.arange(n))
        before = fast.STATS["fast"]
        scaled(out, src, n, scale, large, unsigned, (False, None))
        assert fast.STATS["fast"] == before + 1
        assert len(plans(scaled)) == count
        np.testing.assert_array_equal(
            out.to_numpy(), np.arange(n) * scale + large + unsigned
        )
    scaled(out, src, 3, 1.0, 300, 512, (True, None))
    assert len(plans(scaled)) == count + 1
    np.testing.assert_array_equal(out.to_numpy(), np.arange(3) + 813)


def test_ndarray_key_matches_mapper_and_separates_torch(kernels):
    rank = kernels[3]
    nd = array([0, 0, 0])
    rank(nd, 1.0)
    nd_key = next(iter(plans(rank)))
    primal = rank._primal
    _, features = primal.mapper.lookup(primal.raise_on_templated_floats, (nd, 1.0))
    assert nd_key[0][0] == "ndarray"
    assert nd_key[0][1:] == features[0]
    host = torch.zeros(3)
    rank(host, 2.0)
    assert len(plans(rank)) == 2
    before = fast.STATS["fast"]
    rank(nd, 3.0)
    rank(host, 4.0)
    assert fast.STATS["fast"] == before + 2
    np.testing.assert_array_equal(nd.to_numpy(), [3, 3, 3])
    assert torch.equal(host, torch.full((3,), 4.0))


def test_dtype_rank_and_layout_specialize(kernels):
    scaled, _, _, rank, grid = kernels
    out, src = array([0, 0]), array([1, 2])
    scaled(out, src, 2, 1.0, 300, 512, (False,))
    src_i = array([3, 4], ti.i32)
    scaled(out, src_i, 2, 1.0, 300, 512, (False,))
    assert len(plans(scaled)) == 2
    scaled(out, src_i, 2, 2.0, 300, 512, (False,))
    np.testing.assert_array_equal(out.to_numpy(), [818, 820])
    flat, square = array([0, 0]), array(np.zeros((2, 2)))
    rank(flat, 1.0)
    rank(square, 2.0)
    assert len(plans(rank)) == 2
    rank(square, 3.0)
    np.testing.assert_array_equal(square.to_numpy(), np.full((2, 2), 3))
    canonical = ti.ndarray(ti.f32, shape=(2, 3))
    transposed = ti.ndarray(ti.f32, shape=(3, 2))
    # Ndarray.shape/from_numpy/to_numpy honor the canonical-axis tag. Set it
    # before first use, as Quadrants' layout-tagged tensor allocator does.
    transposed._qd_layout = (1, 0)
    grid(canonical)
    grid(transposed)
    assert len(plans(grid)) == 2
    grid(canonical)
    grid(transposed)
    expected = np.array([[0, 1, 2], [10, 11, 12]])
    np.testing.assert_array_equal(canonical.to_numpy(), expected)
    np.testing.assert_array_equal(transposed.to_numpy(), expected)


@pytest.mark.parametrize("matrix", [False, True])
def test_vector_and_matrix_elements_rebind_and_reject_wrong_shapes(kernels, matrix):
    kernel = kernels[2 if matrix else 1]
    for n in (3, 7):
        values = np.arange(n * 4, dtype=np.float32).reshape(
            (n, 2, 2) if matrix else (n, 4)
        )
        src = (
            ti.Matrix.ndarray(2, 2, ti.f32, shape=n)
            if matrix
            else ti.Vector.ndarray(4, ti.f32, shape=n)
        )
        src.from_numpy(values)
        out = ti.ndarray(ti.f32, shape=n)
        kernel(out, src)
        before = fast.STATS["fast"]
        kernel(out, src)
        assert fast.STATS["fast"] == before + 1
        expected = (
            values[:, 0, 1] + values[:, 1, 0] if matrix else values[:, 0] + values[:, 3]
        )
        np.testing.assert_array_equal(out.to_numpy(), expected)
    assert len(plans(kernel)) == 1
    bad = (
        ti.Matrix.ndarray(3, 2, ti.f32, shape=7)
        if matrix
        else ti.Vector.ndarray(3, ti.f32, shape=7)
    )
    before = fast.STATS["fast"]
    with pytest.raises(Exception, match="[Ee]lement|[Tt]ype|dtype"):
        kernel(out, bad)
    assert fast.STATS["fast"] == before
    assert len(plans(kernel)) == 1


def test_gradients_still_fall_back_after_warming_a_plan(kernels):
    rank = kernels[3]
    nd = array([0, 0])
    rank(nd, 1.0)
    nd.grad = ti.ndarray(ti.f32, shape=2)
    fast.set_telemetry_enabled(True)
    before = dict(fast.STATS)
    rank(nd, 2.0)
    assert fast.STATS["fast"] == before["fast"]
    assert fast.STATS["slow"] == before["slow"] + 1
    report = fast.launch_report()
    assert report["totals"]["fallback"] == 1
    assert report["kernels"][0]["reasons"] == {"ndarray_gradient": 1}
    np.testing.assert_array_equal(nd.to_numpy(), [2, 2])


def test_plans_do_not_keep_arrays_alive_and_reset_invalidates_handles(kernels):
    rank = kernels[3]
    nd = array([0, 0])
    rank(nd, 1.0)
    rank(nd, 2.0)
    ref = weakref.ref(nd)
    del nd
    gc.collect()
    assert ref() is None
    rank._primal.reset()
    assert "_algan_fast_plans" not in rank._primal.__dict__
    other = array([0, 0, 0, 0])
    rank(other, 3.0)
    before = fast.STATS["fast"]
    rank(other, 4.0)
    assert fast.STATS["fast"] == before + 1
    np.testing.assert_array_equal(other.to_numpy(), [4, 4, 4, 4])


def test_signed_zero_runtime_scalars_are_rebound(kernels):
    rank = kernels[3]
    nd = array([1])
    rank(nd, 0.0)
    rank(nd, -0.0)
    assert np.signbit(nd.to_numpy()[0])
    rank(nd, 0.0)
    assert not np.signbit(nd.to_numpy()[0])


def test_telemetry_observes_actual_cache_hits_and_all_fallbacks(kernels):
    scaled, _, _, rank, grid = kernels
    nd = array(np.zeros((2, 2)))
    src = array([1, 2])
    out = array([0, 0])
    fast.set_telemetry_enabled(True)
    grid(nd)  # original, records a plan AND a Quadrants context
    grid(nd)  # Algan fast
    fast.set_enabled(False)
    grid(nd)  # Quadrants context hit, not a guessed "disabled" miss
    scaled(out, src, 2, 0.5, 300, 512, (False,))  # float: original uncached
    fast.set_enabled(True)
    with pytest.raises(Exception, match="[Tt]ype|[Cc]annot|[Ii]nvalid"):
        rank(out, "bad scalar")
    report = fast.launch_report(reset=True)
    assert report["totals"] == {
        "fast": 1,
        "quadrants_cache": 1,
        "cold": 1,
        "fallback": 1,
        "error": 1,
    }
    reasons = {
        reason: count
        for row in report["kernels"]
        for reason, count in row["reasons"].items()
    }
    assert reasons == {"disabled": 1, "scalar_type": 1}
    assert not fast.launch_report()["kernels"]


def test_parallel_warm_launches_have_independent_contexts(kernels):
    rank = kernels[3]
    arrays = [array(np.zeros(16)) for _ in range(4)]
    rank(arrays[0], 0.0)
    fast.set_telemetry_enabled(True)

    def worker(pair):
        i, nd = pair
        for _ in range(6):
            rank(nd, float(i + 1))

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(worker, enumerate(arrays)))
    assert fast.launch_report()["totals"] == {
        "fast": 24,
        "quadrants_cache": 0,
        "cold": 0,
        "fallback": 0,
        "error": 0,
    }
    for i, nd in enumerate(arrays):
        np.testing.assert_array_equal(nd.to_numpy(), np.full(16, i + 1))


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires a real Apple GPU"
)
def test_mps_slices_use_current_offsets_shapes_and_scalars(kernels):
    from algan.rendering import mps_zero_copy
    from algan.settings._startup import render_device

    if render_device().type != "mps":
        pytest.skip("requires the mac-mps arm, not a CPU program on a Mac")
    assert mps_zero_copy.zero_copy_available()
    rank = kernels[3]
    storage = torch.full((80,), -99.0, device="mps")
    staged = mps_zero_copy.STATS["staged_arguments"]
    for start, n, value in [(0, 8, 1.0), (8, 13, 2.0), (40, 7, 3.0), (0, 9, 4.0)]:
        view = storage[start : start + n]
        fast.set_enabled(False)
        rank(view, value)
        reference = storage.cpu().clone()
        storage.fill_(-99)
        fast.set_enabled(True)
        rank(view, value)  # first call records a plan, others reuse it
        storage.fill_(-99)
        before = fast.STATS["fast"]
        rank(view, value)
        assert fast.STATS["fast"] == before + 1
        torch.testing.assert_close(storage.cpu(), reference, rtol=0, atol=0)
        storage.fill_(-99)
    assert len(plans(rank)) == 1
    assert mps_zero_copy.STATS["staged_arguments"] == staged


def test_mixed_torch_and_native_bindings_are_refreshed(kernels):
    scaled = kernels[0]
    native_out, native_src = array([0, 0, 0]), array([1, 2, 3])
    torch_out, torch_src = torch.zeros(3), torch.tensor([4.0, 5.0, 6.0])
    for scale in (1.0, 2.0):
        scaled(native_out, torch_src, 3, scale, 300, 512, (False,))
        scaled(torch_out, native_src, 3, scale, 300, 512, (False,))
    assert len(plans(scaled)) == 2
    np.testing.assert_array_equal(native_out.to_numpy(), [820, 822, 824])
    assert torch.equal(torch_out, torch.tensor([814.0, 816.0, 818.0]))


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires a real Apple GPU"
)
@pytest.mark.parametrize("matrix", [False, True])
def test_mps_vector_matrix_slice_offsets(kernels, matrix):
    from algan.settings._startup import render_device

    if render_device().type != "mps":
        pytest.skip("requires the mac-mps arm")
    kernel = kernels[2 if matrix else 1]
    storage = torch.arange(128, dtype=torch.float32, device="mps")
    for offset, n in ((8, 3), (32, 5), (0, 2)):
        src = storage[offset : offset + 4 * n].reshape((n, 2, 2) if matrix else (n, 4))
        out = torch.full((n + 8,), -1.0, device="mps")
        view = out[4 : 4 + n]
        fast.set_enabled(False)
        kernel(view, src)
        reference = out.cpu().clone()
        fast.set_enabled(True)
        kernel(view, src)
        out.fill_(-1)
        before = fast.STATS["fast"]
        kernel(view, src)
        assert fast.STATS["fast"] == before + 1
        torch.testing.assert_close(out.cpu(), reference, rtol=0, atol=0)
    assert len(plans(kernel)) == 1


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires a real Apple GPU"
)
def test_mps_shared_dtype_views_and_cpu_source(kernels):
    from algan.rendering import mps_zero_copy
    from algan.settings._startup import render_device

    if render_device().type != "mps":
        pytest.skip("requires the mac-mps arm")

    @ti.kernel
    def alias(
        a: ti.types.ndarray(dtype=ti.f32),
        b: ti.types.ndarray(dtype=ti.i32),
        value: ti.f32,
        count: ti.i32,
    ):
        a[0] = value
        b[1] = count

    storage = torch.zeros(16, dtype=torch.float32, device="mps")
    floated = storage[4:8]
    integer = storage.view(torch.int32)[4:8]
    staged = mps_zero_copy.STATS["staged_arguments"]
    for value, count in ((1.5, 1000), (2.5, 2000)):
        fast.set_enabled(False)
        alias(floated, integer, value, count)
        reference = storage.cpu().view(torch.int32).clone()
        fast.set_enabled(True)
        alias(floated, integer, value, count)
        storage.zero_()
        before = fast.STATS["fast"]
        alias(floated, integer, value, count)
        assert fast.STATS["fast"] == before + 1
        torch.testing.assert_close(
            storage.cpu().view(torch.int32), reference, rtol=0, atol=0
        )
    assert mps_zero_copy.STATS["staged_arguments"] == staged

    # Host inputs must still use the compiler's external-array staging path.
    # This proves value parity, not transfer-free Metal access to host memory.
    src = torch.tensor([1.0, 2.0, 3.0])
    out = storage[8:11]
    scaled = kernels[0]
    for scale in (1.0, 2.0):
        fast.set_enabled(False)
        scaled(out, src, 3, scale, 300, 512, (False,))
        reference = storage.cpu().clone()
        fast.set_enabled(True)
        scaled(out, src, 3, scale, 300, 512, (False,))
        storage[8:11].zero_()
        before = fast.STATS["fast"]
        scaled(out, src, 3, scale, 300, 512, (False,))
        assert fast.STATS["fast"] == before + 1
        torch.testing.assert_close(storage.cpu(), reference, rtol=0, atol=0)
