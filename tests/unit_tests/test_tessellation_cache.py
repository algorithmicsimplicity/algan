"""Render-neutrality of the on-disk tessellation cache.

Cached tile geometry is stored relative to an origin so that one entry serves
a glyph wherever it is placed. That normalisation has to be invisible: a mob
built while the cache is cold and the same mob rebuilt from the cache must be
the *same bits*, not merely close. They were not, and the gap was visible in a
render -- 336 pixels of one ``text_and_media`` frame moved by up to 18 channel
values, against the render suites' tolerance of 2 -- because analytic-AA edge
coverage turns a few ULPs of vertex position into whole channel values wherever
a coverage sample flips side of an edge.

Two things went wrong, and this module guards both: the cold path skipped the
round trip its own cache file would later impose, and the origin subtracted on
save was not the origin added back on load.
"""

from __future__ import annotations

import shutil

import pytest
import torch

from algan import TexTriangulated, TextTriangulated
from algan.mobs import triangulated_bezier_circuit as tbc
from algan.settings import SETTINGS


@pytest.fixture
def isolated_tessellation_cache(tmp_path):
    """Point the tessellation cache at an empty directory for one test."""
    saved = SETTINGS.paths.cache_directory
    SETTINGS.paths.set(cache_directory=str(tmp_path))
    try:
        yield tmp_path / "tessellations"
    finally:
        SETTINGS.paths.set(cache_directory=saved)


def _tiles_for(make_mob, monkeypatch):
    """Every tile array ``make_mob()`` feeds into the packing stage.

    ``packed_reorder`` is where the cold and cached paths converge, so it sees
    exactly the geometry each path hands downstream.
    """
    captured = []
    original = tbc.packed_reorder

    def spy(tiles, *args, **kwargs):
        captured.append(tiles.detach().cpu().clone())
        return original(tiles, *args, **kwargs)

    monkeypatch.setattr(tbc, "packed_reorder", spy)
    make_mob()
    monkeypatch.undo()
    return captured


@pytest.mark.parametrize(
    "make_mob",
    [
        pytest.param(lambda: TextTriangulated("mesh", font_size=38), id="text"),
        pytest.param(lambda: TexTriangulated(r"\alpha\beta", font_size=38), id="tex"),
    ],
)
def test_cached_tessellation_is_bit_identical_to_a_fresh_one(
    isolated_tessellation_cache, monkeypatch, make_mob
):
    cold = _tiles_for(make_mob, monkeypatch)
    assert cold, "the mob tessellated nothing, so this proves nothing"
    assert isolated_tessellation_cache.is_dir(), "the tessellation was not cached"

    warm = _tiles_for(make_mob, monkeypatch)

    assert len(warm) == len(cold)
    for index, (fresh, cached) in enumerate(zip(cold, warm)):
        assert torch.equal(cached, fresh), (
            f"piece {index} came back from the cache with different geometry: "
            f"max |delta| {(cached - fresh).abs().max().item():.3e}"
        )


def test_a_stale_entry_is_re_tessellated_rather_than_replayed(
    isolated_tessellation_cache, monkeypatch
):
    """Cache files are keyed by a version tag, so an old layout cannot be read.

    Entries written before the origin fix decode at a slightly different
    position. They have to miss, not replay.
    """
    make_mob = lambda: TextTriangulated("mesh", font_size=38)  # noqa: E731
    fresh = _tiles_for(make_mob, monkeypatch)
    entries = sorted(isolated_tessellation_cache.glob("*.txt"))
    assert entries, "nothing was cached"

    monkeypatch.setattr(tbc, "_TESSELLATION_CACHE_VERSION", "tessellation-v1-legacy")
    under_old_version = _tiles_for(make_mob, monkeypatch)

    # A different version means different keys: the old files are untouched and
    # the mob re-tessellates to the same geometry rather than reading them.
    assert sorted(isolated_tessellation_cache.glob("*.txt")) != entries
    for old, new in zip(fresh, under_old_version):
        assert torch.equal(old, new)


def test_normalisation_round_trip_uses_one_origin():
    """The save and load sides must agree on what tiles are relative to."""
    tiles = torch.tensor([[[3.5, -1.25], [4.0, 2.5]]])
    offset = torch.tensor([1.5, -0.5, 9.0])  # third component is ignored

    normalized = tiles - tbc._tile_origin(offset)
    assert torch.equal(tbc._denormalize_tiles(normalized, offset), tiles)


def test_cache_survives_a_wipe_between_builds(isolated_tessellation_cache, monkeypatch):
    """Deleting the cache directory mid-session re-tessellates identically."""
    make_mob = lambda: TextTriangulated("mesh", font_size=38)  # noqa: E731
    first = _tiles_for(make_mob, monkeypatch)
    shutil.rmtree(isolated_tessellation_cache)
    second = _tiles_for(make_mob, monkeypatch)

    for before, after in zip(first, second):
        assert torch.equal(before, after)
