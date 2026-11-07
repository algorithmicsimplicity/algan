"""Dense conflict-rank grouping must match the original global unique.

The kernel arm is asked for wherever a launch stages nothing, which is the
condition ``taichi_launch_is_local`` answers -- not "the device is CUDA", which
is what it was gated on until the Metal adoption made that comparison wrong.
Widening it also reaches the CPU arch, where the two arms were measured at
**9.2 ms against 52.0 ms** over 2.9M fragments and agree exactly, so this
parametrizes the expectation on the predicate rather than on a device name.
"""

from __future__ import annotations

import random

import pytest
import torch

from algan import SETTINGS
from algan.rendering.raytracing.sheets import _rank_key_base, _sheet_rank_groups
from algan.rendering.taichi_runtime import init_taichi, taichi_launch_is_local
from tests.unit_tests.mps_rounding import merge_wide_uniques


def ranks_for_masks(bands):
    parent, ranks = [], []
    for p, masks in enumerate(bands):
        counts = [0] * 8
        for mask in masks:
            claimed = [lane for lane in range(8) if mask & (1 << lane)]
            ranks.append(max((counts[lane] for lane in claimed), default=0))
            parent.append(p)
            for lane in claimed:
                counts[lane] += 1
    return parent, ranks


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("case", ["empty", "single", "drops", "deep", "random"])
def test_rank_groups_match_global_unique(device, case, monkeypatch):
    """Both arms number every (parent, rank) pair, ranks past 15 included.

    ``deep`` reaches rank 39 in one parent: the rank used to be clamped to 15
    so it fit four bits of a ``parent * 16 + rank`` key, and past that the
    torch arm's radix would let one parent's ranks collide with the next's.
    The expectation is built from the pairs themselves, not from such a key.
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    init_taichi()
    from algan.rendering.raytracing import sheet_rank_groups_taichi

    calls = []
    original = sheet_rank_groups_taichi.rank_groups

    def counted(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(sheet_rank_groups_taichi, "rank_groups", counted)
    rng = random.Random(934)
    bands = {
        "empty": [],
        "single": [[0], [255], [1 << 12]],
        "drops": [[1, 1, 2, 0, 1, 2, 4, 4], [255, 1, 0, 128, 255]],
        "deep": [[1] * 40 + [2, 0, 2] + [255] * 40, [0] * 23],
        "random": [
            [rng.randrange(65536) for _ in range(rng.randrange(1, 80))]
            for _ in range(129)
        ],
    }[case]
    parents, ranks = ranks_for_masks(bands)
    parent = torch.tensor(parents, dtype=torch.int64, device=device)
    rank = torch.tensor(ranks, dtype=torch.int32, device=device)
    pairs = list(zip(parents, ranks))
    labels = sorted(set(pairs))
    ids = {pair: i for i, pair in enumerate(labels)}
    expected_groups = torch.tensor(
        [ids[pair] for pair in pairs], dtype=torch.int64, device=device
    )
    expected_parents = torch.tensor(
        [pair[0] for pair in labels], dtype=torch.int64, device=device
    )
    expected_ranks = torch.tensor(
        [pair[1] for pair in labels], dtype=torch.int64, device=device
    )
    # compact_sheets hands in the deepest rank it already read back; the
    # torch arm otherwise reads it itself. Both must give the same groups.
    for deepest in (None, max(ranks, default=0)):
        for enabled in (False, True):
            with SETTINGS.raytracing.experimental.override(sheet_rank_groups=enabled):
                groups, cid_band, rank_of_cid = _sheet_rank_groups(
                    parent, rank, deepest=deepest
                )
            assert torch.equal(groups, expected_groups)
            assert torch.equal(cid_band, expected_parents)
            assert torch.equal(rank_of_cid, expected_ranks)
            assert groups.dtype == cid_band.dtype == rank_of_cid.dtype == torch.int64
    local = taichi_launch_is_local(torch.device(device))
    assert len(calls) == 2 * int(local and bool(parents))


def _pairs_past_the_mps_ceiling(shape):
    """``(parent, rank)`` streams whose ``parent * radix + rank`` passes 2**24.

    ``many bands`` is the renderer's own shape: 2**20 + 2**12 bands, more than
    a half-covered 1080p frame holds, with conflict ranks to 3, so the radix is
    16 and the keys of the last 4096 bands are past 2**24. ``deep band`` gets
    there in 8,399 fragments instead: 4,199 one-fragment bands and a last band
    4,200 deep, whose keys run 17,635,800 .. 17,639,999, one apart.
    """
    if shape == "deep band":
        parents = 4200
        parent = torch.cat(
            (torch.arange(parents - 1), torch.full((parents,), parents - 1))
        )
        rank = torch.cat((torch.zeros(parents - 1), torch.arange(parents)))
    else:
        g = torch.Generator().manual_seed(17)
        parents = 2**20 + 2**12
        depth = torch.randint(0, 4, (parents,), generator=g)
        depth[torch.rand(parents, generator=g) < 0.75] = 0
        parent = torch.repeat_interleave(torch.arange(parents), depth + 1)
        start = torch.cumsum(depth + 1, 0) - (depth + 1)
        rank = torch.arange(parent.numel()) - start[parent]
    return parent.to(torch.int64), rank.to(torch.int32)


def _key_groups(parent, rank):
    """The torch arm's answer on the CPU, where ``unique`` is exact."""
    base = int(rank.max()) + 1
    keys, groups = torch.unique(parent * base + rank, sorted=True, return_inverse=True)
    return groups, keys // base, keys % base


@pytest.mark.parametrize("shape", ["deep band", "many bands"])
def test_the_torch_arm_keeps_keys_past_2_24_apart_through_mps_unique(
    monkeypatch, shape
):
    """MPS-friendly mode groups the pairs without the key MPS's unique merges.

    On the Apple GPU, ``unique`` with an inverse answers for ``float32(key)``,
    so keys one apart past 2**24 merge: over 1,114,112 bands the torch arm
    lost 49,130 of 2,784,430 sub-bands (run 42351444222). The CPU cannot
    reproduce that, so ``merge_wide_uniques`` makes its ``unique`` answer the
    same way, and the arm must give the exact grouping anyway. The expectation
    is computed before the merging starts.
    """
    parent, rank = _pairs_past_the_mps_ceiling(shape)
    want = _key_groups(parent, rank)
    assert int(parent[-1]) * _rank_key_base(int(rank.max())) > 2**24

    monkeypatch.delenv("ALGAN_MPS_FRIENDLY", raising=False)
    SETTINGS.computing.set(mps_friendly=True)
    merge_wide_uniques(monkeypatch)
    with SETTINGS.raytracing.experimental.override(sheet_rank_groups=False):
        got = _sheet_rank_groups(parent, rank)

    for name, w, g in zip(("groups", "cid_band", "rank_of_cid"), want, got):
        assert g.dtype == torch.int64, name
        assert torch.equal(g, w), (
            f"{name}: {int((g != w).sum()) if g.shape == w.shape else g.shape}"
        )


@pytest.mark.parametrize("shape", ["deep band", "many bands"])
def test_the_torch_arm_keeps_keys_past_2_24_apart_on_the_device(monkeypatch, shape):
    """The same, through the real op on an Apple GPU.

    The device comes from ``torch.backends.mps.is_available()`` rather than
    from Algan's render device, so any Apple machine runs the torch arm on its
    GPU; elsewhere this checks the arm on the CPU, where ``unique`` was never
    wrong, and a green run there does not clear it.
    """
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    parent, rank = _pairs_past_the_mps_ceiling(shape)
    want = _key_groups(parent, rank)

    monkeypatch.delenv("ALGAN_MPS_FRIENDLY", raising=False)
    SETTINGS.computing.set(mps_friendly=True)
    with SETTINGS.raytracing.experimental.override(sheet_rank_groups=False):
        got = _sheet_rank_groups(parent.to(device), rank.to(device))

    for name, w, g in zip(("groups", "cid_band", "rank_of_cid"), want, got):
        g = g.cpu()
        assert g.shape == w.shape, (
            f"{name}: {tuple(g.shape)} rows on {device}, not {tuple(w.shape)}"
        )
        assert torch.equal(g, w), (
            f"{name}: {int((g != w).sum())} rows differ on {device}"
        )


@pytest.mark.parametrize("guard", ["uninitialized", "nonlocal"])
def test_rank_groups_keep_runtime_fallback(monkeypatch, guard):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from algan.rendering import taichi_runtime

    if guard == "uninitialized":
        monkeypatch.setattr(taichi_runtime, "_live_arch", lambda: None)
    else:
        monkeypatch.setattr(taichi_runtime, "_live_arch", lambda: "cuda")
        monkeypatch.setattr(
            taichi_runtime, "taichi_launch_is_local", lambda device: False
        )
    parent = torch.tensor([0, 0, 0, 1], dtype=torch.int64, device="cuda")
    rank = torch.tensor([0, 1, 0, 0], dtype=torch.int32, device="cuda")
    with SETTINGS.raytracing.experimental.override(sheet_rank_groups=True):
        actual = _sheet_rank_groups(parent, rank)
    assert torch.equal(actual[0], torch.tensor([0, 1, 0, 2], device="cuda"))
    assert torch.equal(actual[1], torch.tensor([0, 0, 1], device="cuda"))
    assert torch.equal(actual[2], torch.tensor([0, 1, 0], device="cuda"))
