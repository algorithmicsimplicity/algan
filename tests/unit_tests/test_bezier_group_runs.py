"""One un-batchable circuit must not revert its whole group to the per-actor build.

``build_render_primitives_batched`` reads each attribute from the timeline once
for a whole group of circuits instead of once per actor, and is roughly five
times cheaper per circuit than the per-actor build. It used to be given up
entirely for a group as soon as *one* raw primitive in the same frame batch
shared the group's batch identifier -- the merged collection carries a single
class marker, so a bucket holding both merged collections and raw primitives
could not be walked. A packed circuit (a ``Text``'s glyph pack, or anything
built by ``batch_mobs``) is exactly such a raw primitive, and it shares the
identifier of every plain filled circuit in the scene, so the revert fired on
scenes that mix text with shapes -- 51.5% of the reference scene's circuits
(``DESIGN_optimization_targets.md``, P9).

The group is now split into maximal runs of consecutive batchable actors in
actor order instead. What these tests pin is the pair of properties that makes
that legal: the run split actually reaches the batched build (otherwise the
change is inert), and the rendered frame does not move (the collection layout
is what byte-identity rests on).

``ALGAN_BEZIER_GROUP_RUNS=0`` restores the all-or-nothing revert, and is the
A/B arm here.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from algan.animation_timeline.animation_contexts import Off
from algan.constants.color import BLUE, GREEN, RED
from algan.constants.spatial import LEFT, RIGHT, UP
from algan.mobs.bezier_circuit import BezierCircuitCubic
from algan.mobs.shapes_2d import Square
from algan.scene import Scene
from algan.settings.video_settings import LD
from algan.utils.mob_utils import batch_mobs


def _clash_scene(scene):
    """Three filled circuits, the middle one packed and therefore un-batchable.

    All four squares are plain filled circuits with no texture points, so every
    one of them lands in the same batch-identifier bucket. ``batch_mobs`` gives
    the middle Mob batched control points, which ``_is_batchable_bezier``
    rejects -- so it is the raw primitive that used to poison the other two.
    Ordering matters: the pack sits *between* the two loose squares in the
    authored draw order, so the surviving run split is a genuine split (two
    runs of one) rather than one run with a raw tail.

    They are placed so that all four *overlap* at the origin. Coplanar circuits
    tie on depth and are resolved by their position in the merged arrays, so an
    overlap is the only arrangement in which a reordered merge shows up as a
    changed pixel -- four squares side by side would render the same whatever
    order they were concatenated in, and the frame comparison below would pass
    vacuously. Measured: giving the first square a ``z_index`` of 3, which is
    what moves it up the coplanar order, moves **47 350 channel values** of the
    LD frame. A reordered merge here is loudly visible.
    """
    with Off():
        first = Square(size=2.0, color=RED, stroke_width=0).move(LEFT * 0.4)
        packed = batch_mobs(
            [
                Square(size=2.0, color=GREEN, stroke_width=0),
                Square(size=2.0, color=GREEN, stroke_width=0).move(UP * 0.4),
            ]
        )
        last = Square(size=2.0, color=BLUE, stroke_width=0).move(RIGHT * 0.4)
    first.spawn(animate=False)
    packed.spawn(animate=False)
    last.spawn(animate=False)
    return first, packed, last


def _prepare_one_batch(scene, monkeypatch, runs):
    """Run ``get_batch_of_primitives`` over one frame window of the clash scene.

    Returns ``(collections, per_actor_build_calls)``. The call count is what
    tells the two arms apart without reading the collections: every circuit
    that takes the per-actor path goes through
    ``BezierCircuitCubic.get_render_primitives``.
    """
    from algan.settings import SETTINGS
    from algan.settings._startup import _ANIMATION_DEVICE
    from algan.utils.memory_utils import get_num_available_bytes

    monkeypatch.setenv("ALGAN_BEZIER_GROUP_RUNS", runs)

    calls = []
    original = BezierCircuitCubic.get_render_primitives

    def counting(self, *args, **kwargs):
        calls.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(BezierCircuitCubic, "get_render_primitives", counting)

    scene.scene_times.append(
        [
            scene.scene_times[-1][0],
            round(scene._recorded_end_time_for_render() * scene.frames_per_second),
        ]
    )
    scene.initialize_frames()
    start_ind, end_ind = scene.scene_times[-1]
    end_ind = max(end_ind, start_ind + 1)
    actors = [scene.camera, scene.camera.screen, *scene.light_sources, *scene.actors]
    max_mem = int(
        SETTINGS.computing.animation_memory_fraction
        * get_num_available_bytes(_ANIMATION_DEVICE)
    )
    with scene.batch_prep_context():
        collections, _end, _state = scene.get_batch_of_primitives(
            start_ind, end_ind, actors, max_mem
        )
    return collections, calls


def _circuit_collections(collections):
    from algan.rendering.primitives.bezier_circuit_primitive import (
        BezierCircuitPrimitive,
    )

    return [c for c in collections if isinstance(c, BezierCircuitPrimitive)]


@pytest.mark.fast
@pytest.mark.parametrize(("runs", "expected_per_actor_builds"), [("0", 3), ("1", 1)])
def test_only_the_raw_circuit_takes_the_per_actor_build(
    monkeypatch, runs, expected_per_actor_builds
):
    """With the revert, all three circuits fall back; with runs, only the pack.

    This is the non-vacuity guard for the parity test below: if the clash
    stopped happening (or stopped being repaired) the frames would match for
    the uninteresting reason that both arms ran the same code.
    """
    with Scene() as scene:
        _clash_scene(scene)
        _collections, calls = _prepare_one_batch(scene, monkeypatch, runs)
    assert len(calls) == expected_per_actor_builds


@pytest.mark.fast
def test_run_splitting_preserves_the_merged_circuit_layout(monkeypatch):
    """The concatenated geometry is identical whichever arm built it.

    Compared on the attributes whose values do not depend on how the circuits
    were divided into collections. ``next_segment_inds`` is deliberately not
    among them: the merge rewrites it into indices local to each collection, so
    a different division changes it by construction while meaning the same
    thing.
    """

    def geometry(runs):
        with Scene() as scene:
            _clash_scene(scene)
            collections, _calls = _prepare_one_batch(scene, monkeypatch, runs)
        circuits = _circuit_collections(collections)
        assert circuits, "the clash scene must produce circuit collections"
        return {
            "corners": torch.cat([c.corners for c in circuits], -3),
            "normals": torch.cat([c.normals for c in circuits], -2),
            "stroke_width": torch.cat([c.stroke_width for c in circuits], -2),
            "segments": torch.cat(
                [c.num_segments_per_object.view(-1) for c in circuits]
            ),
        }

    reverted = geometry("0")
    split = geometry("1")
    for name, expected in reverted.items():
        actual = split[name]
        assert actual.shape == expected.shape, name
        assert torch.equal(actual, expected), name


def test_run_splitting_leaves_the_rendered_frame_unchanged(tmp_path, monkeypatch):
    """The guarantee users see: the same pixels, whichever build ran.

    A render rather than a tensor comparison, because the collection *layout* is
    what byte-identity rests on and only the merge downstream of these
    collections can speak for it.
    """
    import torchvision

    def frame(runs):
        monkeypatch.setenv("ALGAN_BEZIER_GROUP_RUNS", runs)
        path = tmp_path / f"clash_{runs}.png"
        with Scene() as scene:
            _clash_scene(scene)
            scene.save_frame(str(path), video_settings=LD)
        return torchvision.io.read_image(str(path)).permute(1, 2, 0).numpy()

    reverted = frame("0")
    split = frame("1")
    assert reverted.shape == split.shape
    assert np.array_equal(reverted, split), (
        "run splitting moved "
        f"{int((reverted != split).sum())} channel values, max "
        f"{int(np.abs(reverted.astype(int) - split.astype(int)).max())}"
    )
