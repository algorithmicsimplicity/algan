"""Class-uniform surface groups can retain their dense sub-band IDs exactly."""

from __future__ import annotations

import pytest
import torch

from algan import SETTINGS
from algan.rendering.raytracing import sheets


@pytest.mark.parametrize(
    ("bands", "classes", "starts", "identity"),
    [
        ([], [], [], True),
        ([0], [17], [True], True),
        ([0, 0, 1, 1], [9, 9, 4, 4], [True, False, True, False], True),
        ([1, 0, 1, 0, 2], [7, 7, 7, 7, 0], [True, False, False, False, True], True),
        ([0, 0, 0, 1], [3, 4, 3, 7], [True, False, False, True], False),
        # Mixed original group but uniform sub-bands: conservative fallback.
        ([0, 1, 0, 1], [3, 4, 3, 4], [True, False, False, False], False),
    ],
)
def test_class_group_reuse_matches_reference(bands, classes, starts, identity):
    device = SETTINGS.computing.render_device
    band = torch.tensor(bands, dtype=torch.int64, device=device)
    cls = torch.tensor(classes, dtype=torch.int64, device=device)
    new_group = torch.tensor(starts, dtype=torch.bool, device=device)
    nb = max(bands, default=-1) + 1
    with SETTINGS.raytracing.experimental.override(sheet_group_reuse=False):
        expected = sheets._sheet_class_groups(band, cls, new_group, nb)
    with SETTINGS.raytracing.experimental.override(sheet_group_reuse=True):
        actual = sheets._sheet_class_groups(band, cls, new_group, nb)
    assert actual[0] == expected[0]
    assert all(torch.equal(a, b) for a, b in zip(actual[1:], expected[1:]))
    if device.type in ("cuda", "cpu"):
        assert (actual[1] is band) == identity


def test_reuses_ordered_ids_without_enabling_pixel_sort():
    ids = torch.tensor([0, 0, 1, 7, 7, 8], device=SETTINGS.computing.render_device)
    with SETTINGS.raytracing.experimental.override(
        sheet_pixel_sort=False, sheet_group_reuse=True
    ):
        actual = sheets._unique_sorted_ids(ids)
    expected = torch.unique(ids, sorted=True, return_inverse=True)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))


def _rank_split_stream(seed, groups, max_bands, max_frags, max_rank, mixed):
    """A compaction-shaped stream: group runs in stream order, a few depth
    bands per group, conflict ranks numbered parent-major as
    ``_sheet_rank_groups`` numbers them, classes uniform per triangle.
    """
    rng = torch.Generator().manual_seed(seed)

    def draw(hi):
        return int(torch.randint(1, hi + 1, (1,), generator=rng))

    new_group, parent, rank, cls = [], [], [], []
    p = -1
    for _g in range(groups):
        classes = [
            int(torch.randint(0, 1 << 24, (1,), generator=rng)) + 1
            for _ in range(3 if mixed else 1)
        ]
        first_in_group = True
        for _b in range(draw(max_bands)):
            p += 1
            for _f in range(draw(max_frags)):
                new_group.append(first_in_group)
                first_in_group = False
                parent.append(p)
                rank.append(int(torch.randint(0, max_rank + 1, (1,), generator=rng)))
                cls.append(
                    classes[int(torch.randint(0, len(classes), (1,), generator=rng))]
                )
    parent_t = torch.tensor(parent, dtype=torch.int64)
    rank_t = torch.tensor(rank, dtype=torch.int32)
    # Parent-major dense sub-band ids, exactly as the rank split assigns them.
    _keys, band = torch.unique(parent_t * 16 + rank_t, sorted=True, return_inverse=True)
    return (
        band,
        torch.tensor(cls, dtype=torch.int64),
        torch.tensor(new_group, dtype=torch.bool),
    )


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("mixed", [True, False])
def test_class_groups_by_run_sort_match_the_global_unique(seed, mixed):
    """The within-run sort must reproduce ``torch.unique``'s numbering exactly:
    count, per-fragment group id and per-group sub-band, on streams whose
    sub-bands interleave inside a group (a rank split), with classes that
    mix within a group and classes uniform within it, and on the degenerate
    one-fragment stream.
    """
    from algan.rendering.mps_compat import band_class_groups
    from algan.rendering.taichi_runtime import init_taichi

    init_taichi()
    device = SETTINGS.computing.render_device
    if not sheets._local_sheet_sort(torch.zeros(1, dtype=torch.int64, device=device)):
        pytest.skip("the run-sort kernel cannot launch locally on this device")
    band, cls, new_group = _rank_split_stream(seed, 40, 3, 5, 2, mixed)
    band, cls, new_group = band.to(device), cls.to(device), new_group.to(device)
    want = band_class_groups(band, cls, sheets._SHADE_CLASS_BASE)
    got = sheets._class_groups_by_run_sort(band, cls, new_group)
    assert got[0] == want[0]
    assert torch.equal(got[1], want[1])
    assert torch.equal(got[2], want[2])
    assert got[1].dtype == got[2].dtype == torch.int64
    one = sheets._class_groups_by_run_sort(band[:1], cls[:1], new_group[:1])
    assert one[0] == 1
    assert torch.equal(one[1], torch.zeros(1, dtype=torch.int64, device=device))
    assert torch.equal(one[2], band[:1])


def test_class_groups_take_the_run_sort_only_when_asked():
    from algan.rendering.taichi_runtime import init_taichi

    init_taichi()
    device = SETTINGS.computing.render_device
    band, cls, new_group = _rank_split_stream(7, 12, 2, 4, 1, True)
    band, cls, new_group = band.to(device), cls.to(device), new_group.to(device)
    nb = int(band.max()) + 1
    with SETTINGS.raytracing.experimental.override(sheet_class_run_sort=False):
        reference = sheets._sheet_class_groups(band, cls, new_group, nb)
    with SETTINGS.raytracing.experimental.override(sheet_class_run_sort=True):
        actual = sheets._sheet_class_groups(band, cls, new_group, nb)
    assert actual[0] == reference[0]
    assert torch.equal(actual[1], reference[1])
    assert torch.equal(actual[2], reference[2])
