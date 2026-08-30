"""Wide attributes (textures) materialize their frame window on the render device.

Guards for the T4 round's texture path (DESIGN_optimization_targets.md, "The T4
round"): a colour texture's ``AttributeTimeline`` is wider than
``WIDE_ATTR_MIN_CHANNELS``, so its materialized window lives on the render
device when that is CUDA, its edit log is gathered there, the batch sizer
prices it against the render-device budget rather than the animation
device's, and the window is released once a batch's primitives are built.

The value contract is the important one: the window materialized on the
device must equal the one the animation device would have produced, bit for
bit -- the render reads the same texels either way. Measured identical on the
nn benchmark scene (md5 of the [3, 887, 1774, 5] map); this pins it on a
small texture.

Two later rounds changed what the estimators and the primitive build produce,
and because everything here is CUDA-only (``skipif`` on the render device) the
staleness was invisible on a CPU box until 2026-08-28:

* ``texture_opacity_in_kernel`` (default on) removed the host premultiply from
  both estimators' image chains -- 6 images to 5 on the render device, 2 copies
  to 1 on the animation device. The pricing test now pins **both** arms of that
  split rather than one default.
* ``texture_time_lerp`` (default on) stopped materializing an animated
  reassignment frame by frame: the window is described by K endpoint images
  stacked out of the **edit log**, which lives on the animation device. So on
  the default path the primitive's ``texture_map`` is a small host tensor, not
  a render-device image per frame. The release test is split accordingly --
  the dense arm still pins the CUDA placement, the segment arm pins the
  endpoint stack.
"""

from __future__ import annotations

import hashlib

import pytest
import torch

import algan.animation_timeline.timeline as timeline_module
from algan import ImageMob, Off, Sync
from algan.animation_timeline.timeline import WIDE_ATTR_MIN_CHANNELS
from algan.rendering.raytracing import settings as rt_settings
from algan.scene_manager import SceneManager
from algan.settings._startup import render_device

# 128x128x5 = 81920 channels: over the threshold, and small enough that the
# CPU arm of this test costs nothing.
_SIDE = 128


def _window_hash(t):
    return hashlib.md5(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _build_scene():
    """An ImageMob whose texture animates, so replay lerps every texel."""
    scene = SceneManager.reset()
    torch.manual_seed(0)
    image = torch.rand(_SIDE, _SIDE, 5)
    image[..., 3] = 0.0
    image[..., 4] = 1.0
    with Off():
        mob = ImageMob(image).spawn()
    with Sync(duration=1):
        mob.color_texture = mob.color_texture * 0.5
    return scene, mob


def _materialize(scene, mob, frames=4):
    actors = [scene.camera, scene.camera.screen, *scene.light_sources, *scene.actors]
    times = torch.arange(frames) / 10.0
    with scene.batch_prep_context():
        scene.timeline_manager.set_state_to_times(times, active_mobs=actors)
        timeline = scene.timeline_manager.attr_to_timeline[mob._color_texture_attr]
        window = timeline.active_state.clone()
        return timeline, window


def test_texture_timeline_is_wide():
    scene, mob = _build_scene()
    timeline = scene.timeline_manager.attr_to_timeline[mob._color_texture_attr]
    assert timeline.current_state.shape[-1] == _SIDE * _SIDE * 5
    assert timeline.current_state.shape[-1] >= WIDE_ATTR_MIN_CHANNELS
    # The grid child never owned a texture row; only the surface's row exists.
    assert timeline.pointer == 1


@pytest.mark.skipif(render_device().type != "cuda", reason="needs a CUDA render device")
def test_window_materializes_on_the_render_device_bit_identically(monkeypatch):
    scene, mob = _build_scene()
    timeline, device_window = _materialize(scene, mob)
    assert timeline.materialize_device is not None
    assert device_window.device.type == "cuda"
    assert (
        timeline._prepared_queries(torch.arange(4) / 10.0).sorted_values.device.type
        == "cuda"
    )

    # The same scene with the path disabled: the animation-device window.
    monkeypatch.setattr(
        timeline_module, "_wide_attr_materialize_device", lambda channels: None
    )
    scene, mob = _build_scene()
    timeline, host_window = _materialize(scene, mob)
    assert timeline.materialize_device is None
    assert host_window.device.type == "cpu"

    assert device_window.shape == host_window.shape
    assert _window_hash(device_window) == _window_hash(host_window)


@pytest.mark.skipif(render_device().type != "cuda", reason="needs a CUDA render device")
@pytest.mark.parametrize(
    ("opacity_in_kernel", "device_images", "host_copies"),
    # The image chain each estimator counts, per the two docstrings. Under
    # texture_opacity_in_kernel the host premultiply disappears from both
    # sides: the opacity rides the primitive as per-frame scalars instead.
    [(True, 5, 1), (False, 6, 2)],
)
def test_texture_is_priced_against_the_render_device_budget(
    opacity_in_kernel, device_images, host_copies
):
    # Restore what was there rather than the shipped default: these are
    # module-level globals with no reset fixture, and the suite must survive
    # being run under ALGAN_TEXTURE_OPACITY_IN_KERNEL=0.
    previous = rt_settings.texture_opacity_in_kernel
    rt_settings.set_texture_opacity_in_kernel(opacity_in_kernel)
    try:
        scene, mob = _build_scene()
        texels = _SIDE * _SIDE * 5
        assert mob._color_texture_bytes_per_timestep() == 0
        assert (
            mob._get_render_device_memory_used_per_timestep()
            == texels * 4 * device_images
        )
        # And back on the animation device when the path is off.
        timeline = scene.timeline_manager.attr_to_timeline[mob._color_texture_attr]
        timeline.materialize_device = None
        assert mob._get_render_device_memory_used_per_timestep() == 0
        assert mob._color_texture_bytes_per_timestep() == texels * 4 * host_copies
    finally:
        rt_settings.set_texture_opacity_in_kernel(previous)


def _build_one_batch(scene):
    """Prepare one 4-frame batch and hand back its textured primitives."""
    actors = [scene.camera, scene.camera.screen, *scene.light_sources, *scene.actors]
    with scene.batch_prep_context():
        primitives, end, _ = scene.get_batch_of_primitives(0, 4, actors, 10**12)
    assert end == 4
    assert primitives
    textured = [p for p in primitives if getattr(p, "texture_map", None) is not None]
    assert textured
    return textured


@pytest.mark.skipif(render_device().type != "cuda", reason="needs a CUDA render device")
def test_batch_preparation_releases_the_dense_window():
    """The dense path: one image per frame, on the render device.

    ``texture_time_lerp`` off, so the window really is materialized frame by
    frame -- which is the arrangement the render-device placement exists for.
    """
    previous = rt_settings.texture_time_lerp
    rt_settings.set_texture_time_lerp(False)
    try:
        scene, mob = _build_scene()
        timeline = scene.timeline_manager.attr_to_timeline[mob._color_texture_attr]
        textured = _build_one_batch(scene)
        # Released: the timeline is back on its authoring state, and the
        # primitive carries the texture the render needs on the render device.
        assert timeline.active_state is timeline.current_state
        assert textured[0].texture_map.device.type == "cuda"
        assert textured[0].texture_map.shape[0] == 4
    finally:
        rt_settings.set_texture_time_lerp(previous)


@pytest.mark.skipif(render_device().type != "cuda", reason="needs a CUDA render device")
@pytest.mark.skipif(
    not rt_settings.texture_time_lerp, reason="describes the texture_time_lerp path"
)
def test_batch_preparation_releases_the_segment_window():
    """The default path: K endpoints and a lerp, not one image per frame.

    ``texture_time_lerp`` describes this window as its two endpoints, and the
    gate stacks those out of the **edit log** -- which lives on the animation
    device -- so the stack is a host tensor and the render device holds no
    per-frame image at all. Asserting ``cuda`` here would be asserting that
    the optimization had not happened.
    """
    scene, mob = _build_scene()
    timeline = scene.timeline_manager.attr_to_timeline[mob._color_texture_attr]
    textured = _build_one_batch(scene)
    assert timeline.active_state is timeline.current_state
    assert mob._texture_window_lerp
    primitive = textured[0]
    assert primitive.texture_lerp is not None
    # [1, K, H, W, 5]: leading singleton declares the stack frame-static, and
    # the K endpoints must be fewer than the 4 frames they describe -- that
    # inequality is the whole point of the path.
    texture_map = primitive.texture_map
    assert texture_map.shape[0] == 1
    assert texture_map.shape[1] == 2
    assert texture_map.shape[1] < 4
    assert texture_map.device.type == "cpu"
