"""Primary shadow ordering preserves the sheet-to-visibility mapping."""

from __future__ import annotations

import pytest
import torch

from algan import SETTINGS
from algan.rendering.raytracing.raster_pipeline import _order_primary_shadow_events


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("count", [0, 1, 8191, 8192, 8193])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_primary_shadow_order_preserves_event_identity(enabled, count, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    # Gaps model rejected sheets; repeated and negative refs cover shared
    # source triangles and Bezier events with no triangle identity.
    indices = torch.arange(count, device=device, dtype=torch.int64) * 2
    refs = (torch.arange(count * 2, device=device) * 7 % 41 - 1).to(torch.int32)
    with SETTINGS.raytracing.experimental.override(shadow_primary_sort=enabled):
        ordered, sources = _order_primary_shadow_events(indices, refs)
    if enabled and device == "cuda" and count >= 8192:
        assert torch.equal(ordered.sort().values, indices)
        assert torch.equal(sources, refs.index_select(0, ordered))
        assert bool((sources[1:] >= sources[:-1]).all())
        # Model the real scatter and mode-2 visibility lookup, with a unique
        # payload per accepted sheet so duplicated refs cannot hide a mixup.
        event_ids = torch.full((count * 2,), -1, device=device, dtype=torch.int32)
        event_ids.scatter_(
            0, ordered, torch.arange(count, device=device, dtype=torch.int32)
        )
        visibility = ordered * 3 + 5
        assert torch.equal(visibility[event_ids[indices].long()], indices * 3 + 5)
        assert bool((event_ids[1::2] == -1).all())
    else:
        assert ordered is indices
        assert sources is None
