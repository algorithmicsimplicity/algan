"""Sort/gather outputs outlive short-lived arena workspace, without lost bits."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from algan import SETTINGS
from algan.rendering import mps_compat, taichi_runtime
from algan.rendering.raytracing import device_sort, sheets
from algan.rendering.raytracing.array_ops import gather_rows
from algan.rendering.raytracing.sheet_order import stable_lexsort
from algan.rendering.raytracing.sheet_workspace import CompactionWorkspace
from algan.utils.memory_utils import ManualMemory


def _workspace():
    taichi_runtime.init_taichi()
    memory = ManualMemory(0, device=SETTINGS.computing.render_device, num_bytes=1 << 20)
    memory._poison = 191
    return memory, CompactionWorkspace(memory)


def _poison(memory):
    with memory.temp():
        memory.get_tensor((memory.get_num_bytes_remaining(),), torch.uint8).fill_(213)


def _reference(*keys):
    order = torch.arange(keys[0].numel(), device=keys[0].device)
    for key in reversed(keys):
        order = order[torch.argsort(key[order], stable=True)]
    return order


@pytest.mark.parametrize("size", [0, 1, 43])
@pytest.mark.parametrize("num_keys", [1, 2, 3, 4])
@pytest.mark.parametrize("owned", [False, True])
def test_sort_preserves_stable_ties_and_scratch_lifetime(size, num_keys, owned):
    memory, workspace = _workspace()
    ids = torch.arange(size, device=workspace.device)
    keys = (
        (ids * 7 % 4 - 2) * (1 << 54),
        (ids * 13 % 9).to(torch.int32),
        (ids % 5).to(torch.float32),
        (ids % 7).to(torch.float64),
    )[:num_keys]
    expected = _reference(*keys)
    out = memory.get_tensor((size,), torch.int64, persist=True) if owned else None
    before = memory.get_pointers()
    for _ in range(2):
        actual = stable_lexsort(*keys, out=out, workspace=workspace)
        if owned:
            assert actual is out
        assert memory.get_pointers() == before
        assert workspace._live_bytes == workspace._depth == 0
        _poison(memory)
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("case", ["strided", "zeros", "nonfinite"])
def test_sort_keeps_torch_float_ordering(case):
    memory, workspace = _workspace()
    depth = torch.tensor(
        [0.0, -0.0, float("nan"), 1.0, -1.0, float("inf"), 1.0], device=workspace.device
    )
    if case == "strided":
        depth = depth.repeat_interleave(2)[::2]
    elif case == "zeros":
        depth = depth[:2].repeat(7)
    key = torch.zeros(depth.numel(), dtype=torch.int32, device=workspace.device)
    actual = stable_lexsort(key, depth, workspace=workspace)
    _poison(memory)
    assert torch.equal(actual, _reference(key, depth))


@pytest.mark.parametrize("problem", ["shape", "dtype", "stride", "alias", "keys"])
def test_sort_rejects_invalid_destination_before_mutation(problem):
    memory, workspace = _workspace()
    source = torch.tensor([3, 1, 2, 1], device=workspace.device)
    out = torch.full((8,), 117, dtype=torch.int64, device=workspace.device)
    keys = [source]
    destination = out[:4]
    if problem == "shape":
        destination = out
    elif problem == "dtype":
        destination = out[:2].view(torch.int32)
    elif problem == "stride":
        destination = out[::2]
    elif problem == "alias":
        keys = [out[:4]]
    else:
        keys.append(source[:3])
    original = out.clone()
    before = memory.get_pointers()
    with pytest.raises(ValueError):
        stable_lexsort(*keys, out=destination, workspace=workspace)
    assert memory.get_pointers() == before
    assert torch.equal(out, original)


def test_sort_failure_unwinds_all_nested_stages(monkeypatch):
    memory, workspace = _workspace()
    keys = torch.tensor([3, 1, 2], device=workspace.device)
    out = memory.get_tensor((3,), torch.int64, persist=True)
    before = memory.get_pointers()
    original = torch.sort
    calls = []

    def fail_second(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise LookupError("injected sort failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "sort", fail_second)
    with pytest.raises(LookupError, match="injected"):
        stable_lexsort(keys, keys, out=out, workspace=workspace)
    assert memory.get_pointers() == before
    assert workspace._depth == workspace._live_bytes == 0
    assert torch.equal(
        stable_lexsort(keys, out=out, workspace=workspace), _reference(keys)
    )


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.float32])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("native", [False, True])
def test_gather_rows_preserves_integer_bits_and_float_payloads(
    monkeypatch, dtype, index_dtype, native
):
    memory, workspace = _workspace()
    monkeypatch.setattr(mps_compat, "mps_friendly", lambda: native)
    wide = 1 << (55 if dtype == torch.int64 else 27)
    source = torch.tensor(
        [wide + 1, -wide - 3, 17, 0, 23, -7, 42, wide + 9],
        dtype=dtype,
        device=workspace.device,
    ).reshape(4, 2)
    indices = torch.tensor([3, 1, 1, 0], dtype=index_dtype, device=workspace.device)
    out = memory.get_tensor(source.shape, dtype, persist=True)
    before = memory.get_pointers()
    actual = gather_rows(source, indices, out=out)
    assert actual is out
    assert memory.get_pointers() == before
    _poison(memory)
    assert torch.equal(actual, source[indices.to(torch.int64)])


def test_gather_fallback_is_exact_without_a_local_kernel(monkeypatch):
    memory, workspace = _workspace()
    monkeypatch.setattr(mps_compat, "mps_friendly", lambda: True)
    monkeypatch.setattr(taichi_runtime, "_live_arch", lambda: None)
    source = torch.tensor([2**60 + 1, 2**60 + 3, -(2**60) - 7], device=workspace.device)
    indices = torch.tensor([2, 0], device=workspace.device)
    with workspace.stage():
        actual = workspace.gather(source, indices)
        assert actual.tolist() == [-(2**60) - 7, 2**60 + 1]


@pytest.mark.parametrize("which", ["source", "indices"])
def test_gather_rejects_aliases_without_writing(which):
    data = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    source, indices = data.clone(), data.clone()
    out = source if which == "source" else indices
    before = out.clone()
    with pytest.raises(ValueError, match="overlap"):
        gather_rows(source, indices, out=out)
    assert torch.equal(out, before)


@pytest.mark.parametrize("native", [False, True])
def test_sheet_order_helpers_return_owned_permutations(native):
    memory, workspace = _workspace()
    pix = torch.tensor([0, 0, 0, 1, 1, 1], device=workspace.device)
    group = torch.tensor([1, 1, 0, 0, 0, 0], device=workspace.device)
    depth = torch.tensor([2.0, 1.0, 3.0, 4.0, 4.0, 2.0], device=workspace.device)
    offsets = torch.tensor([0, 3, 6], dtype=torch.int32, device=workspace.device)
    with SETTINGS.raytracing.experimental.override(sheet_pixel_sort=native):
        order = sheets._pixel_group_order(
            pix,
            group,
            depth,
            offsets,
            key_bounds=(2, 2),
            memory=memory,
            workspace=workspace,
        )
        assert torch.equal(order, _reference(pix, group, depth))
        saved = memory.get_pointers()
        with workspace.stage():
            shell = sheets._key_depth_order(group, depth, workspace=workspace)
            assert torch.equal(shell, _reference(group, depth))
        assert memory.get_pointers() == saved
        positions = torch.tensor([2, 0, 1, 5, 3, 4], device=workspace.device)
        walk = sheets._sheet_walk_order(
            pix, positions, memory=memory, workspace=workspace
        )
        _poison(memory)
        assert torch.equal(walk, _reference(positions))
        assert torch.equal(order, _reference(pix, group, depth))


def test_native_radix_destinations_and_scratch_use_the_same_arena(monkeypatch):
    """Exercise host launch ownership with a CPU oracle, not a GPU-sort claim."""
    import sys

    memory, workspace = _workspace()
    calls = []

    def kernel(
        keys,
        perm,
        work,
        tmp,
        values,
        tmp_values,
        scratch,
        count,
        n,
        dtype,
        bits,
        depth,
        mode,
    ):
        assert all(
            t.untyped_storage()._cdata == memory.data.untyped_storage()._cdata
            for t in (work, tmp, values, tmp_values, scratch, count)
        )
        selected = keys if mode in (0, 1) else keys[perm.to(torch.int64)]
        ordered = torch.argsort(selected, stable=True)
        values.copy_(ordered if mode == 0 else perm[ordered])
        calls.append(mode)

    monkeypatch.setattr(device_sort, "radix_sort_available", lambda _: True)
    monkeypatch.setitem(
        sys.modules,
        "algan.rendering.raytracing.radix_sort_taichi",
        SimpleNamespace(argsort_pairs=kernel),
    )
    keys = [
        torch.tensor(k, device=workspace.device)
        for k in ([1, 0, 0, 1], [2, 2, 1, 2], [3, 1, 1, 0])
    ]
    out = memory.get_tensor((4,), torch.int64, persist=True)
    before = memory.get_pointers()
    assert stable_lexsort(*keys, out=out, workspace=workspace) is out
    assert calls == [0, 2, 2]
    assert memory.get_pointers() == before
    _poison(memory)
    assert torch.equal(out, _reference(*keys))


@pytest.mark.parametrize(
    "perm", [torch.zeros(3), torch.zeros((1, 3), dtype=torch.int64)]
)
def test_native_sort_rejects_invalid_permutation(monkeypatch, perm):
    monkeypatch.setattr(device_sort, "radix_sort_available", lambda _: True)
    with pytest.raises(ValueError, match="integer vector"):
        device_sort.stable_argsort(torch.ones(3), perm=perm)


@pytest.mark.parametrize("columns", [1, 2])
@pytest.mark.parametrize("extreme", [False, True])
def test_validated_packed_arithmetic_has_owned_output(columns, extreme):
    """The CUDA eligibility gate is covered separately; exercise its integer math."""
    memory, workspace = _workspace()
    ids = torch.arange(37, device=workspace.device)
    keys = [(ids * 3 % 7) - 3, (ids * 5 % 11) - 5][:columns]
    if extreme:
        keys[0] += torch.iinfo(torch.int64).min + 3
        if columns > 1:
            keys[1] += torch.iinfo(torch.int64).max - 5
    bits = (0x3F800000 + ids % 13).to(torch.int32)
    bounds = [
        value for key in (*keys, bits) for value in (int(key.min()), int(key.max()))
    ]
    spans = [hi - lo + 1 for lo, hi in zip(bounds[::2], bounds[1::2])]
    out = memory.get_tensor(ids.shape, torch.int64, persist=True)
    before = memory.get_pointers()
    result = sheets._sort_packed_depth(
        keys, bits, bounds, spans, out=out, workspace=workspace
    )
    assert result is out
    assert memory.get_pointers() == before
    _poison(memory)
    assert torch.equal(result, _reference(*keys, bits.view(torch.float32)))


def test_packed_output_alias_is_rejected_before_arithmetic():
    key = torch.tensor([2, 1, 3], dtype=torch.int64)
    before = key.clone()
    bits = torch.ones(3, dtype=torch.float32).view(torch.int32)
    with pytest.raises(ValueError, match="overlap"):
        sheets._sort_packed_depth(
            (key,), bits, [1, 3, 0x3F800000, 0x3F800000], [3, 1], out=key
        )
    assert torch.equal(key, before)


def test_native_radix_failure_reclaims_scratch_and_keeps_destinations(monkeypatch):
    import sys

    memory, workspace = _workspace()
    keys = torch.tensor([3, 1, 2, 1], device=workspace.device)
    out = memory.get_tensor((4,), torch.int32, persist=True)
    sentinel = memory.get_tensor((7,), torch.int64, persist=True).fill_(2**55 + 1)
    before = memory.get_pointers()

    def fail_kernel(*args):
        assert memory.current_pointer > before[0]
        raise RuntimeError("injected radix failure")

    monkeypatch.setattr(device_sort, "radix_sort_available", lambda _: True)
    monkeypatch.setitem(
        sys.modules,
        "algan.rendering.raytracing.radix_sort_taichi",
        SimpleNamespace(argsort_pairs=fail_kernel),
    )
    with pytest.raises(RuntimeError, match="injected"):
        device_sort.stable_lexsort(keys, keys, out=out, workspace=workspace)
    assert memory.get_pointers() == before
    assert workspace._depth == workspace._live_bytes == 0
    _poison(memory)
    assert sentinel.tolist() == [2**55 + 1] * 7


@pytest.mark.parametrize(
    "invalid", ["shape", "dtype", "stride", "scalar", "index_rank", "index_dtype"]
)
def test_row_gather_validates_metadata_before_mutation(invalid):
    source = torch.arange(8, dtype=torch.int64).reshape(4, 2)
    indices = torch.tensor([2, 0], dtype=torch.int32)
    destination = torch.full((8,), 171, dtype=torch.int64)
    out = destination[:4].view(2, 2)
    if invalid == "shape":
        out = destination
    elif invalid == "dtype":
        out = destination[:2].view(torch.int32).view(2, 2)
    elif invalid == "stride":
        out = destination[::2].view(2, 2)
    elif invalid == "scalar":
        source = source[0, 0]
    elif invalid == "index_rank":
        indices = indices.view(1, 2)
    else:
        indices = indices.float()
    before = destination.clone()
    with pytest.raises(ValueError):
        gather_rows(source, indices, out=out)
    assert torch.equal(destination, before)
