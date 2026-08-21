"""Parity for the compact rematerialization buffer (``ALGAN_OPT_DISABLE=compactstate``).

``rematerialize_state_at_times`` used to build a ``[T, N, D]`` buffer where
``N`` is every row the scene ever allocated, scatter the window's ~30% live
rows into it, and leave the rest zero. It now materializes only the live rows
and keeps a global-row -> column map, so nothing is allocated for the dead 70%.

Every reader goes through that map, so the thing to prove is that the two paths
answer *identically* for every row -- including the rows the window did not
materialize, which must still read as zero.

This drives a real scene through both paths and compares, per attribute and per
frame batch:

  * the state of every global row, live or dead, gathered back through the
    public accessor;
  * that ``get`` still returns a view (not a copy) for a contiguous mob range
    with ``copy=False``, which is the property the whole design had to preserve.

    .venv/Scripts/python.exe benchmarks/_compact_state_parity.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch  # noqa: E402

import algan.animation_timeline.timeline as tl  # noqa: E402
from algan import (  # noqa: E402
    RIGHT,
    SMOKE_TEST,
    UP,
    Circle,
    Group,
    Off,
    Sphere,
    Square,
    Sync,
    Text,
)
from algan.scene_manager import SceneManager  # noqa: E402


def build_scene():
    """A scene with several mobs, staggered lifespans and overlapping edits.

    Staggered spawns matter: a window's working set is what makes the compact
    buffer smaller than the global one, so a scene where everything is live at
    once would not exercise the dead-row paths at all.
    """
    square = Square().spawn()
    circle = Circle().spawn()
    label = Text("compact").spawn()
    ball = Sphere(radius=0.4).spawn()
    group = Group(Square(), Circle())
    square.move(RIGHT)
    with Sync():
        square.rotate(90, [0, 0, 1])
        circle.move(UP)
        label.move(UP * 2)
    ball.move(RIGHT * 2)
    with Off():
        group.spawn()
    group.move(UP * 0.5)
    circle.despawn()
    square.scale(2)
    return square, circle, label, ball, group


def materialize(disabled, windows):
    """Full per-row state for each window, read back through the accessors."""
    tl._OPT_DISABLED = frozenset({"compactstate"} if disabled else ())
    scene = SceneManager.reset()
    scene.set_video_settings(SMOKE_TEST)
    mobs = build_scene()
    scene.initialize_frames()
    for light in scene.light_sources:
        light.is_primitive = True
    actors = [scene.camera, scene.camera.screen, *scene.light_sources, *scene.actors]

    out = []
    views = []
    widths = []
    for start, stop in windows:
        # See Scene.batch_prep_context: a direct call outside it records new
        # events on every replay and corrupts the timeline being compared.
        with scene.batch_prep_context():
            scene.get_batch_of_primitives(start, stop, actors, 10**12)
        tlm = scene.timeline_manager
        snapshot = {}
        for attr, timeline in tlm.attr_to_timeline.items():
            n = timeline.pointer + 1
            rows = torch.arange(n, dtype=torch.long)
            # Through the public accessor, so the map is what is under test.
            snapshot[attr] = timeline.get(rows, copy=True).clone()
            widths.append((int(timeline.active_state.shape[1]), n))
        out.append(snapshot)

        # The view property, on a mob whose rows are one contiguous run.
        # Tested by storage identity, NOT by ``Tensor._base``: algan runs under
        # a process-global ``torch.inference_mode()``, which switches off
        # autograd's view tracking, so ``_base`` is None even for a genuine
        # view and would report a false regression here. A gather allocates its
        # own storage, which is the difference that matters.
        square = mobs[0]
        location = tlm.attr_to_timeline["location"]
        ranges = location.mob_id_to_ranges.get(square.id)
        if ranges is not None and ranges.pairs is not None and len(ranges.pairs) == 1:
            block = location.get(ranges, copy=False)
            views.append(
                block.untyped_storage().data_ptr()
                == location.active_state.untyped_storage().data_ptr()
            )
        tlm.clear_buffers()
    return out, views, widths


def empty_working_set_reads_zero():
    """A window that materializes *no* rows must still answer reads.

    This is the case the full-render suite caught and the window sweep below
    did not: with nothing live, the compact buffer is ``[T, 0, D]``, and a read
    of any row has no column to gather from. The full-width buffer had a real
    zeroed row for it, so the failure mode is an ``IndexError`` rather than a
    wrong number -- which is why parity over live rows alone missed it.
    """
    scene = SceneManager.reset()
    scene.set_video_settings(SMOKE_TEST)
    square = Square().spawn()
    square.move(RIGHT)
    tlm = scene.timeline_manager
    timeline = tlm.attr_to_timeline["location"]
    times = torch.tensor([0.0, 0.5, 1.0])
    timeline.prepare_for_queries()
    timeline.rematerialize_state_at_times(times, active_mob_ids=[])
    assert timeline.active_state.shape[1] == 0, (
        f"expected an empty working set, got {timeline.active_state.shape[1]} columns"
    )
    rows = torch.arange(timeline.pointer + 1, dtype=torch.long)
    got = timeline.get(rows, copy=True)
    assert got.shape == (3, rows.numel(), timeline.active_state.shape[2]), (
        f"unexpected shape {tuple(got.shape)}"
    )
    assert not bool(got.any()), "unmaterialized rows must read as zero"
    # And a contiguous mob range through the RowRanges fast path.
    ranges = timeline.mob_id_to_ranges.get(square.id) or timeline.ranges_for(square.id)
    block = timeline.get(ranges, copy=True)
    assert not bool(block.any()), "unmaterialized mob range must read as zero"
    timeline.clear_buffers()
    print("empty working set: reads return zeros of the right shape")


def main():
    empty_working_set_reads_zero()
    windows = [(0, 20), (20, 45), (45, 70)]
    full, _, full_widths = materialize(True, windows)
    compact, compact_views, compact_widths = materialize(False, windows)

    # Vacuity guard. If the compact arm silently fell back to the full-width
    # buffer, everything below would pass while testing nothing -- so require
    # that it actually narrowed something, and that the other arm did not.
    saved = [(w, n) for w, n in compact_widths if w < n]
    assert saved, (
        "no attribute was materialized compactly; this compares the full path "
        "against itself and cannot detect a mapping bug"
    )
    assert all(w == n for w, n in full_widths), (
        "the full-width arm was itself compact -- the toggle is not taking effect"
    )
    narrowest = min(w / n for w, n in saved)
    print(
        f"compact arm narrowed {len(saved)} of {len(compact_widths)} materializations, "
        f"down to {narrowest:.0%} of the global row count"
    )

    assert len(full) == len(compact) == len(windows)
    total_rows = 0
    for i, (a, b) in enumerate(zip(full, compact)):
        assert set(a) == set(b), f"window {i}: attribute sets differ"
        for attr in sorted(a):
            x, y = a[attr], b[attr]
            assert x.shape == y.shape, (
                f"window {i} {attr}: shape {tuple(y.shape)} != {tuple(x.shape)}"
            )
            if not torch.equal(x, y):
                bad = (x != y).any(-1).any(0).nonzero().view(-1)
                raise AssertionError(
                    f"window {i} {attr}: {bad.numel()} of {x.shape[1]} rows differ "
                    f"(first: {bad[:8].tolist()})\n"
                    f"  full   {x[:, bad[0]].flatten()[:6].tolist()}\n"
                    f"  compact{y[:, bad[0]].flatten()[:6].tolist()}"
                )
            total_rows += int(x.shape[1])
    print(f"identical across {len(windows)} windows, {total_rows} row-states compared")
    assert compact_views, "no contiguous-range view was checked at all"
    assert all(compact_views), (
        "get(copy=False) stopped returning a view for a contiguous mob range"
    )
    print(f"get(copy=False) still returns a view ({len(compact_views)} checks)")
    print("\ncompact buffer parity holds")


if __name__ == "__main__":
    main()
