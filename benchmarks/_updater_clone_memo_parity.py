"""Parity for the updater history-clone memo (``ALGAN_OPT_DISABLE=clonememo``).

``AnimationTimeline._register_known_history_clones`` runs on *every* traced
updater Mob access -- an updater body makes thousands per frame window, and on
the reference scene it was 65 640 ``register_history_clone`` calls per pass to
re-derive a mapping that changes only when ``Mob.detach_history`` runs. It is
now memoized per (updater event, mob id), invalidated by a version counter that
``register_updater_history_split`` -- the registry's only writer -- bumps.

Registration is idempotent and ``dependency_mob_ids`` is append-only, so the
memo is safe *provided the invalidation is real*. That proviso is what this
script tests, three ways:

  * **parity** -- full per-row state of every attribute, over windows spanning
    a scene that interleaves updaters with ``detach_history``, must be
    identical with the memo on and off;
  * **non-vacuity** -- the run must actually register clones, bump the version,
    and hit the memo; a scene that never splits history would compare the slow
    path against itself and prove nothing (the trap that made
    ``_resolve_rollback_check.py``'s first version pass with the bug in place);
  * **mutation** -- with the version bump suppressed, the comparison must
    FAIL. A guard that cannot fail is not a guard.

    .venv/Scripts/python.exe benchmarks/_updater_clone_memo_parity.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

os.environ.setdefault("ALGAN_RENDER_DEVICE", "cpu")

import torch  # noqa: E402

import algan.animation_timeline.timeline as tl  # noqa: E402
from algan import (  # noqa: E402
    RIGHT,
    SMOKE_TEST,
    UP,
    Circle,
    Group,
    Square,
    Sync,
)
from algan.scene_manager import SceneManager  # noqa: E402

STATS = {"registry_entries": 0, "version": 0, "memo_hits": 0, "walks": 0}


def _instrument():
    """Count memo hits vs registry walks, so non-vacuity is measured."""
    original = tl.AnimationTimeline._register_known_history_clones

    def counted(self, event, mob_ids):
        if not tl._opt_disabled("clonememo"):
            cache = getattr(event, "_known_clone_ids", None)
            fresh = event._known_clone_version == self._updater_clone_version
            for mob_id in tuple(mob_ids):
                if fresh and cache is not None and mob_id in cache:
                    STATS["memo_hits"] += 1
                else:
                    STATS["walks"] += 1
        out = original(self, event, mob_ids)
        STATS["registry_entries"] = len(self._updater_history_clones)
        STATS["version"] = self._updater_clone_version
        return out

    tl.AnimationTimeline._register_known_history_clones = counted


def _drift(mob, time_elapsed):
    """Updater that reads and writes several mobs, so tracing sees many ids."""
    mob.location = mob.location + RIGHT * (time_elapsed * 0.05)
    return mob


def build_scene():
    """Interleaves an updater with a ``detach_history`` split.

    The split is what populates the clone registry: ``detach_history`` hands
    every descendant's recorded rows to a clone and calls
    ``register_updater_history_split`` with the whole descendant map. Doing it
    *after* the updater has already run means a live updater's memo must be
    invalidated, which is precisely the failure mode being tested.
    """
    group = Group(Square(), Circle()).spawn()
    label = Square().spawn()
    group.add_updater(_drift)
    with Sync():
        group.move(UP * 0.5)
        label.move(RIGHT * 0.5)
    # The split: the group's history moves to a clone while the updater stays.
    group.detach_history()
    group.move(RIGHT * 0.4)
    label.detach_history()
    label.move(UP * 0.3)
    return group, label


def _stale_register(self, event, mob_ids):
    """The memo with its version check removed -- the bug being guarded against.

    Note what does *not* work as a mutant: restoring the version counter after
    ``register_updater_history_split`` returns. That function re-registers every
    updater's clones before it returns, while the counter is still bumped, so
    the damage is already undone and the mutant silently passes. The bug class
    that matters is a memo entry that outlives the registry change, which is
    what this models directly.
    """
    cache = event._known_clone_ids
    registered_ids = set()
    for mob_id in tuple(mob_ids):
        ids = cache.get(mob_id)
        if ids is None:
            ids = set()
            for original, clone in self._updater_history_clones.get(mob_id, ()):
                event.register_history_clone(original, clone)
                ids.add(clone.id)
            cache[mob_id] = ids
        registered_ids.update(ids)
    return registered_ids


def materialize(disabled, windows, break_invalidation=False):
    """Per-row state of every attribute at each window, both arms."""
    tl._OPT_DISABLED = frozenset({"clonememo"} if disabled else ())
    saved = tl.AnimationTimeline._register_known_history_clones
    if break_invalidation:
        tl.AnimationTimeline._register_known_history_clones = _stale_register
    try:
        scene = SceneManager.reset()
        scene.set_video_settings(SMOKE_TEST)
        build_scene()
        scene._initialize_frames()
        for light in scene.light_sources:
            light.is_primitive = True
        actors = [
            scene.camera,
            scene.camera.screen,
            *scene.light_sources,
            *scene.actors,
        ]

        out = []
        for start, stop in windows:
            # See Scene._batch_prep_context: outside it a direct call records
            # new events on every replay and corrupts what is being compared.
            with scene._batch_prep_context():
                scene._get_batch_of_primitives(start, stop, actors, 10**12)
            tlm = scene.timeline_manager
            snapshot = {}
            for attr, timeline in tlm.attr_to_timeline.items():
                rows = torch.arange(timeline.pointer + 1, dtype=torch.long)
                snapshot[attr] = timeline.get(rows, copy=True).clone()
            out.append(snapshot)
            tlm.clear_buffers()
        return out
    finally:
        tl.AnimationTimeline._register_known_history_clones = saved


def compare(a, b, label):
    """Exact comparison; returns the first difference rather than raising."""
    if len(a) != len(b):
        return f"{label}: window count differs"
    for i, (x_all, y_all) in enumerate(zip(a, b)):
        if set(x_all) != set(y_all):
            return f"{label}: window {i} attribute sets differ"
        for attr in sorted(x_all):
            x, y = x_all[attr], y_all[attr]
            if x.shape != y.shape:
                return (
                    f"{label}: window {i} {attr}: shape {tuple(y.shape)} !="
                    f" {tuple(x.shape)}"
                )
            if not torch.equal(x, y):
                bad = (x != y).any(-1).any(0).nonzero().view(-1)
                return (
                    f"{label}: window {i} {attr}: {bad.numel()} of {x.shape[1]}"
                    f" rows differ (first: {bad[:8].tolist()})"
                )
    return None


def main():
    _instrument()
    windows = [(0, 15), (15, 35), (35, 60)]

    off = materialize(True, windows)
    walks_off = STATS["walks"]
    on = materialize(False, windows)

    # Non-vacuity, checked before the comparison is trusted.
    assert STATS["registry_entries"] > 0, (
        "no history clones were registered -- this scene never splits history,"
        " so both arms ran identical code and the comparison proves nothing"
    )
    assert STATS["version"] > 0, "the clone version never bumped"
    assert STATS["memo_hits"] > 0, "the memo was never hit; nothing was skipped"
    print(
        f"registry: {STATS['registry_entries']} originals, version"
        f" {STATS['version']}; memo {STATS['memo_hits']} hits vs"
        f" {STATS['walks'] - walks_off} walks in the memo arm"
    )

    problem = compare(off, on, "clonememo")
    assert problem is None, problem
    rows = sum(int(v.shape[1]) for w in on for v in w.values())
    print(f"identical across {len(windows)} windows, {rows} row-states compared")

    # The guard's own guard: break invalidation and require a failure.
    mutant = materialize(False, windows, break_invalidation=True)
    problem = compare(off, mutant, "stale-memo mutant")
    assert problem is not None, (
        "suppressing the version bump changed nothing -- this test cannot"
        " detect a missing invalidation and is therefore vacuous"
    )
    print(f"mutation check: stale memo is caught ({problem})")
    print("\nupdater clone-memo parity holds")


if __name__ == "__main__":
    main()
