"""A batch's promoted 1x1 maps appended at once are the per-group maps.

Constant-property promotion turns every uniformly coloured, uniformly shaded
group of triangles into a 1x1 colour map and a 1x1 material map. A scene of
many differently coloured mobs has one group each, and appending them one by
one cost a decode and two appends per group -- a dozen launches each, on a
merge that is launch-bound on a GPU. ``scene_builder._append_promoted_maps``
decodes and concatenates them all at once on CUDA; this pins that the texel
buffer and every map's placement come out bit for bit the same.
"""

from __future__ import annotations

import pytest
import torch

from algan import PREVIEW, Cube, Off, Scene
from algan.constants.color import Color
from algan.rendering.raytracing import scene_builder

TINY = PREVIEW.set(resolution=(64, 36), frames_per_second=8)


def _merged(scene, frames):
    actors = [scene.camera, scene.camera.screen, *scene.light_sources, *scene.actors]
    with scene._batch_prep_context():
        collections, _end, state = scene._get_batch_of_primitives(
            0, frames, actors, 10**10
        )
    scene._prewarm_render_batch(collections, state)
    merged, _env = scene._prepare_merged_host_scene(collections, render_state=state)
    return merged


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the batched path is CUDA's")
def test_batched_promoted_maps_are_the_per_group_maps(monkeypatch):
    results = []
    for batched in (False, True):
        monkeypatch.setattr(scene_builder, "promoted_maps_batched", batched)
        with Scene(video_settings=TINY) as scene:
            with Off():
                cubes = [
                    Cube(color=Color((0.1 + 0.07 * i, 0.9 - 0.05 * i, 0.3)))
                    .scale(0.2)
                    .move_to(torch.tensor((-3.0 + 0.5 * i, 0.0, 0.0)))
                    .spawn()
                    for i in range(12)
                ]
            # Two of them change colour over the clip, so their maps carry
            # every frame rather than one.
            cubes[3].color = Color((0.0, 0.0, 1.0))
            cubes[7].color = Color((1.0, 0.0, 0.0))
            fps = scene.frames_per_second
            scene.scene_times.append(
                [0, max(1, round(scene._recorded_end_time_for_render() * fps))]
            )
            scene._initialize_frames()
            merged = _merged(scene, 6)
            results.append(
                (merged["textures"].cpu().clone(), merged["tri_tex_meta"].cpu().clone())
            )
    (textures_a, meta_a), (textures_b, meta_b) = results
    assert textures_a.shape == textures_b.shape
    assert torch.equal(textures_a.view(torch.int32), textures_b.view(torch.int32))
    assert torch.equal(meta_a, meta_b)
    # Keep the fixture honest: several groups, some of them per frame.
    frames = meta_a[:, 3]
    assert int((frames > 1).sum()) > 0
    assert int(meta_a[:, 0].unique().numel()) >= 12
