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
from algan.rendering.raytracing.sheets import _sheet_rank_groups
from algan.rendering.taichi_runtime import init_taichi, taichi_launch_is_local


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
    for enabled in (False, True):
        with SETTINGS.raytracing.experimental.override(sheet_rank_groups=enabled):
            groups, cid_band, rank_of_cid = _sheet_rank_groups(parent, rank)
        assert torch.equal(groups, expected_groups)
        assert torch.equal(cid_band, expected_parents)
        assert torch.equal(rank_of_cid, expected_ranks)
        assert groups.dtype == cid_band.dtype == rank_of_cid.dtype == torch.int64
    local = taichi_launch_is_local(torch.device(device))
    assert len(calls) == int(local and bool(parents))


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
