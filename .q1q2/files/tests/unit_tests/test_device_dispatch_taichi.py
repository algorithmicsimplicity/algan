"""Device queue and hoisted-layout kernel correctness, not performance."""

import pytest
import torch

from algan.rendering.arena_region_args import pack_regions
from algan.rendering.device_dispatch import DeviceDispatchError, DeviceDispatchOverflow
from algan.rendering.raytracing.arena_args_taichi import ArenaView
from algan.rendering.raytracing.shadow_dispatch import prepare_primary_shadow_dispatch
from algan.rendering.taichi_runtime import init_taichi, taichi_launch_is_local
from algan.settings import SETTINGS
from algan.settings._startup import render_device
from algan.taichi_compat import ti
from algan.utils.memory_utils import ManualMemory


@ti.kernel
def _guarded_consumer(capacity: ti.i32, header: ti.types.ndarray(dtype=ti.i32, ndim=1),
                      source: ti.types.ndarray(dtype=ti.i32, ndim=1),
                      out: ti.types.ndarray(dtype=ti.f32, ndim=3)):
    for i in range(capacity):
        if header[2] == 0 and header[3] == 0 and i < header[0]:
            out[i, 0, 0] = 1.0


@ti.kernel
def _hoisted_probe(n: ti.i32, buf: ti.types.ndarray(dtype=ti.f32, ndim=1),
                   off: ti.types.ndarray(dtype=ti.i32, ndim=1),
                   shape: ti.types.ndarray(dtype=ti.i32, ndim=1),
                   result: ti.types.ndarray(dtype=ti.f32, ndim=1), hoist: ti.template()):
    for i in range(n):
        view = ti.static(ArenaView(buf, off[0], (shape[0], shape[1]), hoist=hoist))
        value = 0.0
        for j in range(view.shape[1]):
            value += view[i, j]
        result[i] = value


@pytest.fixture
def arena():
    saved = SETTINGS.snapshot()
    SETTINGS.raytracing.experimental.device_dispatch = True
    init_taichi()
    device = render_device()
    if not taichi_launch_is_local(device):
        pytest.skip("Selected compiler/device pairing cannot adopt these buffers")
    # Fixed bounded fixtures; recursive scan also fits this small hard budget.
    memory = ManualMemory(0, device=device, num_bytes=32 * 1024 * 1024)
    try:
        yield memory
    finally:
        SETTINGS.restore(saved)


def _queue(memory, n, *, capacity=None, footprint=True, terminator=True):
    def make(shape, dtype, value=0):
        t = memory.get_tensor(shape, dtype)
        t.fill_(value)
        return t
    size = max(1, n)
    ids = torch.arange(size, dtype=torch.int32, device=memory.data.device)
    accepted = make((size,), torch.int32)
    accepted.copy_((ids % 3 != 1).to(torch.int32))
    source = make((size,), torch.int32)
    source.copy_(ids + (1 << 24) + 3)
    frame = make((size,), torch.int32)
    frame.copy_(ids + (1 << 24) + 19)
    mask = make((size,), torch.int32)
    mask.copy_(ids + torch.iinfo(torch.int32).min)
    payloads = {}
    for field, width in (("pos", 3), ("snrm", 3), ("fnrm", 3), ("dp", 6), ("toff", 3)):
        t = make((size, width), torch.float32)
        t.copy_((ids.to(torch.float32)[:, None] % 32) / 16)
        payloads[field] = t
    reverse = make((size,), torch.int32, -9)
    q = prepare_primary_shadow_dispatch(
        memory, n=n, accepted=accepted, source=source[:n], frame=frame, mask=mask,
        reverse=reverse, footprint=footprint, terminator=terminator,
        capacity=capacity, **payloads)
    vis = make((max(1, q.extent.capacity), 1, 3), torch.float32, -7)
    return q, vis


def _run(q, vis):
    q.run(lambda: _guarded_consumer(q.extent.capacity, q.extent.header.tensor,
                                  q.outputs["source"], vis), vis)


@pytest.mark.parametrize("n", [0, 1, 255, 256, 257, 65539])
def test_scan_pack_stability_exact_ids(arena, n):
    q, vis = _queue(arena, n)
    _run(q, vis)
    indices = q.inputs["accepted"][:n].nonzero(as_tuple=True)[0]
    count = indices.numel()
    assert q.extent.header.tensor.cpu().tolist() == [count, count, 0, 0]
    for field in q.outputs:
        assert torch.equal(q.outputs[field][:count].cpu(),
                           q.inputs[field].index_select(0, indices).cpu())
    reverse = torch.full((max(1, n),), -1, dtype=torch.int32)
    if n:
        reverse[indices.cpu()] = torch.arange(count, dtype=torch.int32)
        assert torch.equal(q.reverse.cpu(), reverse)
    assert torch.all(vis[:count, 0, 0] == 1)
    assert torch.all(vis[count:, 0, 0] == -7)


@pytest.mark.parametrize("footprint,terminator", [(False, False), (True, False), (False, True)])
def test_optional_payloads_plan_reuse(arena, footprint, terminator):
    q, vis = _queue(arena, 17, footprint=footprint, terminator=terminator)
    _run(q, vis)
    q.inputs["accepted"].zero_()
    vis.fill_(-7)
    _run(q, vis)
    assert q.extent.header.tensor.cpu().tolist() == [0, 0, 0, 0]
    assert torch.all(vis == -7) and torch.all(q.reverse == -1)


def test_overflow_does_not_write_visibility_or_mapping(arena):
    q, vis = _queue(arena, 17, capacity=1)
    with pytest.raises(DeviceDispatchOverflow):
        _run(q, vis)
    assert torch.all(vis == -7) and torch.all(q.reverse == -9)
    assert q.extent.header.tensor.cpu().tolist() == [0, 11, 1, 0]


def test_invalid_accept_flag_prevents_consumer(arena):
    q, vis = _queue(arena, 17)
    q.inputs["accepted"][3] = 2
    with pytest.raises(DeviceDispatchError, match="Invalid device queue"):
        _run(q, vis)
    assert torch.all(vis == -7) and torch.all(q.reverse == -9)


def test_descriptor_rejects_host_count_and_overwide_launch(arena):
    q, _ = _queue(arena, 3)
    with pytest.raises(TypeError):
        int(q.extent)
    with pytest.raises(ValueError, match="Flattened"):
        q.extent.launch_capacity(1 << 31)


def test_exception_completion_releases_lease(arena):
    q, vis = _queue(arena, 3)
    def fail():
        _guarded_consumer(q.extent.capacity, q.extent.header.tensor,
                          q.outputs["source"], vis)
        raise RuntimeError("host submission error")
    with pytest.raises(RuntimeError, match="host submission"):
        q.run(fail, vis)
    arena.reset()


@pytest.mark.parametrize("hoist", [False, True])
def test_hoisted_views_compile_and_rebind(arena, hoist):
    for padding, n in ((7, 5), (13, 9)):
        with arena.temp():
            arena.get_tensor((padding,), torch.uint8)
            values = arena.get_tensor((n, 3), torch.float32)
            values.copy_(torch.arange(n * 3, device=values.device).reshape(n, 3))
            result = arena.get_tensor((n,), torch.float32)
            buf, off, shape = pack_regions((("values", "f32", 2),), (values,))
            _hoisted_probe(n, buf, off, shape, result, hoist)
            ti.sync()
            assert torch.equal(result.cpu(), values.sum(1).cpu())
