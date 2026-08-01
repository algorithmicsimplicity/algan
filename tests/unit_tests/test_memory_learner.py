"""Invariants of the in-job memory learners.

The learners raise the reservation for value-dependent scopes as a render
proceeds. Two properties keep them safe, and both are easy to break by
accident:

* they must only move on a *committed* render, never from the arena preflight
  -- the preflight is called repeatedly by a binary search, and an estimate
  that moved underneath it would make the search non-monotone and trip
  ``render_loop``'s "fit was not monotone" guard; and
* they must only ever move *up*, because under-reserving costs a whole
  re-rendered chunk while over-reserving costs a slightly smaller batch.
"""

import pytest

from algan.rendering.raytracing import settings as rt_settings


@pytest.fixture(autouse=True)
def _isolate_learner():
    saved = rt_settings._SPARSE_DISCOVERY_BYTES_PER_FRAME
    yield
    rt_settings._SPARSE_DISCOVERY_BYTES_PER_FRAME = saved


def test_discovery_reservation_starts_seeded_not_at_zero():
    # Starting at zero meant the first chunk of every render job reserved
    # nothing for a pass that does allocate, over-committed, and relied on the
    # out-of-memory window-halving to recover.
    rt_settings._begin_render_job()
    assert rt_settings.sparse_discovery_bytes_for_frames(1) == 0
    rt_settings.seed_sparse_discovery_density(1920 * 1080)
    assert rt_settings.sparse_discovery_bytes_for_frames(1) > 0


def test_seed_scales_with_the_frame_size():
    rt_settings._begin_render_job()
    rt_settings.seed_sparse_discovery_density(320 * 180)
    small = rt_settings.sparse_discovery_bytes_for_frames(1)
    rt_settings._begin_render_job()
    rt_settings.seed_sparse_discovery_density(1920 * 1080)
    assert rt_settings.sparse_discovery_bytes_for_frames(1) > small


def test_seed_never_lowers_an_already_learned_reservation():
    rt_settings._begin_render_job()
    rt_settings.note_sparse_discovery_footprint(10 ** 9, 1)
    learned = rt_settings.sparse_discovery_bytes_for_frames(1)
    rt_settings.seed_sparse_discovery_density(16 * 16)
    assert rt_settings.sparse_discovery_bytes_for_frames(1) == learned


def test_observations_only_raise_the_reservation():
    rt_settings._begin_render_job()
    rt_settings.note_sparse_discovery_footprint(1_000_000, 10)
    high = rt_settings.sparse_discovery_bytes_for_frames(4)
    rt_settings.note_sparse_discovery_footprint(1_000, 10)
    assert rt_settings.sparse_discovery_bytes_for_frames(4) == high


def test_reservation_is_monotone_in_the_frame_count():
    # render_loop._max_duration_that_fits binary-searches the chunk size and
    # raises if the fit predicate is not monotone, so every additive term in
    # chunk_memory_required has to be non-decreasing in num_frames.
    rt_settings._begin_render_job()
    rt_settings.seed_sparse_discovery_density(640 * 360)
    previous = -1
    for frames in range(1, 65):
        current = rt_settings.sparse_discovery_bytes_for_frames(frames)
        assert current >= previous
        previous = current


def test_learner_is_not_updated_by_the_arena_preflight():
    # prepare_sparse_raster_coverage is the only caller of
    # note_sparse_discovery_footprint, and it runs during a committed render,
    # never from _prepared_batch_fits_render_arena. Pin that with a grep-level
    # check so a future call site added to the preflight is caught here rather
    # than as an intermittent "fit was not monotone" crash.
    import inspect

    from algan import render_loop

    source = inspect.getsource(render_loop.RenderLoopMixin
                               ._prepared_batch_fits_render_arena)
    assert "note_sparse_discovery_footprint" not in source
    assert "seed_sparse_discovery_density" not in source
