"""Integer CSR scans preserve counts and honor caller-owned destinations."""

from __future__ import annotations

import pytest
import torch

from algan import SETTINGS
from algan.rendering.raytracing.array_ops import csr_offsets
from algan.utils.memory_utils import ManualMemory


@pytest.mark.parametrize("values", [[], [0], [7], [0, 2, 0, 5, 1, 0]])
@pytest.mark.parametrize("input_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("output_dtype", [torch.int32, torch.int64])
def test_csr_offsets_match_exclusive_scan(values, input_dtype, output_dtype):
    device = SETTINGS.computing.render_device
    counts = torch.tensor(values, dtype=input_dtype, device=device)
    memory = ManualMemory(0, device=device, num_bytes=1024)
    out = memory.get_tensor((len(values) + 1,), output_dtype)
    actual = csr_offsets(counts, out=out)
    expected = [0]
    for count in values:
        expected.append(expected[-1] + count)
    assert actual is out
    assert actual.tolist() == expected
    assert actual.untyped_storage()._cdata == memory.data.untyped_storage()._cdata


def test_csr_offsets_default_is_wide_and_accepts_strided_counts():
    counts = torch.tensor([2**30, 0, 2**30, 0], dtype=torch.int32)[::2]
    assert csr_offsets(counts).tolist() == [0, 2**30, 2**31]
    assert csr_offsets(counts).dtype == torch.int64


@pytest.mark.parametrize(
    "counts", [torch.ones(2), torch.ones((2, 1), dtype=torch.int32)]
)
def test_csr_offsets_rejects_noninteger_or_nonvector_counts(counts):
    with pytest.raises(ValueError, match="counts"):
        csr_offsets(counts)


@pytest.mark.parametrize(
    "out",
    [
        torch.zeros(2, dtype=torch.int64),
        torch.zeros(3),
        torch.zeros(6, dtype=torch.int64)[::2],
    ],
)
def test_csr_offsets_rejects_invalid_output_without_writing(out):
    before = out.clone()
    with pytest.raises(ValueError, match="output"):
        csr_offsets(torch.tensor([1, 2]), out=out)
    assert torch.equal(out, before)


def test_csr_offsets_rejects_overlapping_output_before_writing():
    backing = torch.tensor([3, 4, 5, 6], dtype=torch.int64)
    before = backing.clone()
    with pytest.raises(ValueError, match="overlap"):
        csr_offsets(backing[:3], out=backing)
    assert torch.equal(backing, before)
