"""The replay-window checkpoint must survive a render unchanged.

``_resolve_replay_windows`` now grows ``_resolved_row_ends`` **in place** across
a render's batches instead of cloning it each time. That is only sound because
``preserving_authoring_state`` -- which every re-renderable render enters
(``save_frame``, ``show_frame``, ``save_video(reset=False)``) -- takes a *copy*
of the checkpoint and restores it on exit, and drops the backing store with it.

If that copy ever regresses to aliasing the live dict, nothing fails loudly:
the timeline silently keeps the transient replay windows a render resolved, and
the next render animates from subtly wrong base states. This asserts the
invariant directly.

    .venv/Scripts/python.exe benchmarks/_resolve_rollback_check.py
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
    Off,
    Scene,
    Sphere,
    Square,
    Sync,
    Text,
)
from algan.scene_manager import SceneManager  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_rollback_out")

#: Set while a render is in flight; records whether the render actually grew
#: the checkpoint. Without this the whole check is vacuous -- a scene that
#: records nothing during its render restores trivially even when the snapshot
#: is aliased instead of copied, which is exactly how a broken copy would slip
#: through (it did, on the first version of this script).
MUTATED = [False]


def watch_for_mutation(tlm, baseline):
    resolve = tl.AnimationTimeline._resolve_replay_windows

    def probed(self):
        out = resolve(self)
        if self is tlm:
            for attr, rows in self._resolved_row_ends.items():
                was = baseline.get(attr)
                if was is None or rows.shape != was.shape or not torch.equal(rows, was):
                    MUTATED[0] = True
        return out

    tl.AnimationTimeline._resolve_replay_windows = probed
    return resolve


def snapshot(tlm):
    return {attr: rows.clone() for attr, rows in tlm._resolved_row_ends.items()}


def compare(before, after, label):
    assert set(before) == set(after), (
        f"{label}: attribute set changed {set(before) ^ set(after)}"
    )
    for attr, rows in before.items():
        got = after[attr]
        assert got.shape == rows.shape, (
            f"{label}: {attr} shape {tuple(got.shape)} != {tuple(rows.shape)}"
        )
        assert torch.equal(got, rows), (
            f"{label}: {attr} values changed across the render "
            f"({int((got != rows).sum())} of {rows.numel()} rows differ)"
        )


def main():
    os.makedirs(OUT, exist_ok=True)
    scene = SceneManager.reset()
    scene.set_video_settings(SMOKE_TEST)

    # Text and Sphere are here on purpose: glyph batches and surface
    # auto-resolution record edits *during batch preparation*, which is what
    # makes a render grow the checkpoint at all. A scene of plain Squares does
    # not, and cannot detect a broken snapshot.
    square = Square().spawn()
    circle = Circle().spawn()
    label = Text("rollback").spawn()
    ball = Sphere(radius=0.4).spawn()
    square.move(RIGHT)
    with Sync():
        square.rotate(90, [0, 0, 1])
        circle.move(UP)
        label.move(UP * 2)
        ball.move(RIGHT * 2)
    square.scale(2)

    tlm = scene.timeline_manager
    tlm._resolve_replay_windows()
    before = snapshot(tlm)
    prefix = (tlm._resolved_prefix_count, tlm._resolved_prefix_seq)
    resolved = tlm._replay_windows_resolved
    assert before, "expected a resolved checkpoint to compare against"
    print(
        f"checkpoint: {len(before)} attributes, "
        f"{sum(int(r.numel()) for r in before.values())} rows"
    )

    # The invariant, exercised directly. A render only grows the checkpoint if
    # it *records* while preparing batches -- the reference scene appends
    # 77-304 edits a batch (surface auto-resolution and glyph batches), but a
    # small scene appends none, which is why the render-based check below
    # cannot detect a broken snapshot on its own. Recording inside the block is
    # the same situation, reproduced deterministically.
    with tlm.preserving_authoring_state():
        tlm._resolve_replay_windows()
        with Off():
            square.move(UP)
            circle.move(RIGHT)
        tlm._resolve_replay_windows()
        for attr, rows in tlm._resolved_row_ends.items():
            was = before.get(attr)
            if was is None or rows.shape != was.shape or not torch.equal(rows, was):
                MUTATED[0] = True
    assert MUTATED[0], (
        "the checkpoint never grew, so this check cannot detect a broken "
        "snapshot -- it would pass even with the copy removed"
    )
    print("checkpoint grew in place inside the block (invariant is exercised)")
    compare(before, snapshot(tlm), "after preserving_authoring_state")

    # End-to-end through the public path as well.
    Scene.save_frame(os.path.join(OUT, "rollback"))
    compare(before, snapshot(tlm), "after save_frame")
    assert (tlm._resolved_prefix_count, tlm._resolved_prefix_seq) == prefix, (
        f"prefix moved: {(tlm._resolved_prefix_count, tlm._resolved_prefix_seq)} != {prefix}"
    )
    assert tlm._replay_windows_resolved == resolved
    print("save_frame: checkpoint, prefix and resolved flag all restored")

    # And again through a full video render with reset=False, the other path
    # that has to leave the Scene re-authorable.
    Scene.save_video(os.path.join(OUT, "rollback.mp4"))
    compare(before, snapshot(tlm), "after save_video(reset=False)")
    print("save_video(reset=False): checkpoint restored")

    # The scene must still be authorable and renderable afterwards -- the whole
    # point of the rollback.
    with Off():
        circle.move(RIGHT)
    square.move(UP)
    Scene.save_frame(os.path.join(OUT, "rollback_after"))
    print("re-authored and re-rendered after both renders")
    print("\nrollback invariant holds")


if __name__ == "__main__":
    main()
