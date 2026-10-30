"""The sheet route's tile out-of-memory retry (``tracer._halve_after_oom``).

Three sites in the sparse tile loop -- the state block and resolve, the
overflow poll's compaction, and the bounce drain -- recover from running out
of memory the same way: give the attempt's arena back, halve the primaries
and size the retry's pool. Only the pool policy differs between them, and
only for a splitting batch. Host-only: a stand-in arena, no kernel.
"""

from __future__ import annotations

import pytest

from algan import SETTINGS
from algan.rendering.primitives.primitive import OutOfRenderMemory
from algan.rendering.raytracing import tracer
from algan.rendering.raytracing.truncation import PATH_TRACER_FALLBACK_SPELLING


class _Arena:
    """Records the pointers the retry gives the arena back to."""

    def __init__(self):
        self.restored = None

    def set_pointers(self, pointers):
        self.restored = pointers


@pytest.fixture(autouse=True)
def _counters(monkeypatch):
    """Fresh retry counters, so a test reads only its own retries."""
    monkeypatch.setattr(tracer, "_WAVEFRONT_POOL_RETRIES", [0])
    monkeypatch.setitem(tracer._WAVEFRONT_TILE_STATS, "oom_retries", 0)


def _retry(what="bounce scratch", cause=None, **tile):
    attempt = {
        "attempt_primary": 64,
        "pool": 64,
        "pool_ratio": 1,
        "shared_pool_capacity": 0,
        "halve_shared_pool": False,
        "demand": None,
        "tile_peak": None,
        "sparse_primary": 64,
    }
    attempt.update(tile)
    arena = _Arena()
    result = tracer._halve_after_oom(
        cause or RuntimeError("out of memory"), arena, "tile state", what, **attempt
    )
    assert arena.restored == "tile state"
    return result


@pytest.mark.parametrize("halve_shared_pool", [True, False], ids=("resolve", "drain"))
def test_a_tile_that_does_not_split_keeps_its_pool_equal_to_its_primaries(
    halve_shared_pool,
):
    """A ratio-1 tile has no spare slots, after any of the three failures.

    The poll and the drain used to retry with the failed attempt's pool, so
    the smaller tile held spare slots a fresh one never has.
    """
    assert _retry(halve_shared_pool=halve_shared_pool) == (32, 32, 0)
    assert tracer._WAVEFRONT_POOL_RETRIES[0] == 1
    assert tracer._WAVEFRONT_TILE_STATS["oom_retries"] == 1


def test_a_splitting_tile_halves_its_shared_pool_only_where_the_state_failed():
    tile = {"pool_ratio": 4, "pool": 256, "shared_pool_capacity": 256}
    # The resolve site: the state block, which the pool dominates, may be
    # what did not fit.
    assert _retry(halve_shared_pool=True, **tile) == (32, 128, 128)
    # The poll and the drain: the state fit, and their scratch follows the
    # active rays, so the shrunken tile inherits the spare slots.
    assert _retry(halve_shared_pool=False, **tile) == (32, 256, 256)


def test_a_halved_shared_pool_keeps_the_primaries_own_slots():
    tile = {"pool_ratio": 2, "pool": 40, "shared_pool_capacity": 40}
    assert _retry(halve_shared_pool=True, **tile) == (32, 32, 32)


@pytest.mark.parametrize("what", ["state", "bounce scratch"], ids=("state", "scratch"))
def test_one_covered_pixel_that_does_not_fit_is_final(what):
    SETTINGS.raytracing.set(samples_per_pixel=1)
    cause = RuntimeError("out of memory")
    with pytest.raises(OutOfRenderMemory) as excinfo:
        _retry(what=what, cause=cause, attempt_primary=1, pool=1)
    message = str(excinfo.value)
    assert message.startswith(
        f"Sparse raster {what} did not fit for one covered pixel."
    )
    assert PATH_TRACER_FALLBACK_SPELLING in message
    assert excinfo.value.__cause__ is cause
    assert tracer._WAVEFRONT_POOL_RETRIES[0] == 0
