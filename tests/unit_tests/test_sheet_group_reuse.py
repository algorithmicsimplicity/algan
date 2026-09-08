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
