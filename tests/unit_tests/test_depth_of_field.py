"""Thin-lens depth of field: the camera's ``aperture`` / ``focus_distance``.

Three layers, cheapest first:

* **Camera** -- the two lens attributes are animatable, validated and kept off
  the camera's screen child; :meth:`~.Camera.focus_at` racks onto a target.
* **Kernel** -- ``pt_generate`` is driven directly with a handful of pixels, so
  the lens geometry (origins on the aperture disk, every lens ray through its
  pinhole ray's point on the focus plane, the near clip, the stochastic flag)
  is checked exactly, with no scene pipeline.
* **Render** -- the path tracer blurs what is out of focus and leaves the
  focus plane sharp, adaptive sampling runs lens pixels to the ceiling, a rack
  focus changes per frame, and the deterministic renderer refuses an open
  aperture through the unsupported-feature policy.

Outside the fast suite: a change elsewhere is not liable to break any of this
without breaking the path tracer's own tests first, and the render layer
compiles the path tracer's kernels.
"""

from __future__ import annotations

import math
import warnings

import pytest
import torch

from algan import (
    BLACK,
    LEFT,
    OUT,
    RED,
    RIGHT,
    SETTINGS,
    SMOKE_TEST,
    UP,
    WHITE,
    CameraView,
    Circle,
    Off,
    Scene,
    SceneManager,
    Seq,
    Square,
    Sync,
)
from algan.errors import (
    AlganConfigurationError,
    UnsupportedFeatureError,
    UnsupportedFeatureWarning,
)
from algan.rendering.camera import DEFAULT_FOCUS_DISTANCE, Camera

# Wide enough for the out-of-focus circle's blur disk to sit inside the frame.
VIDEO = SMOKE_TEST.set(resolution=(96, 64))


@pytest.fixture
def fresh_scene():
    snapshot = SETTINGS.snapshot()
    SceneManager.reset()
    try:
        yield
    finally:
        SceneManager.reset()
        SETTINGS.restore(snapshot)


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------


def test_default_lens_is_a_pinhole_focused_on_the_default_origin(fresh_scene):
    from algan.manim_defaults import MANIM_FOCAL_DISTANCE

    # The default focus is the default camera's distance to ORIGIN, so opening
    # the aperture of an unmoved camera keeps ORIGIN sharp.
    assert DEFAULT_FOCUS_DISTANCE == MANIM_FOCAL_DISTANCE
    with Scene() as scene:
        camera = scene.get_camera()
        assert float(camera.aperture) == 0.0
        assert float(camera.focus_distance) == DEFAULT_FOCUS_DISTANCE
        assert float(camera._depth_of(torch.zeros(3))) == pytest.approx(
            DEFAULT_FOCUS_DISTANCE
        )
        assert {"aperture", "focus_distance"} <= set(camera.animatable_attrs)
        # The lens is the camera's, not its screen proxy's.
        timeline = scene.timeline_manager.attr_to_timeline["aperture"]
        assert camera.id in timeline.mob_id_to_inds
        assert camera.screen.id not in timeline.mob_id_to_inds
        lens = camera._get_render_lens()
        assert lens.shape == (1, 1, 2)
        assert lens.flatten().tolist() == [0.0, DEFAULT_FOCUS_DISTANCE]


def test_lens_values_are_validated_on_every_route(fresh_scene):
    with Scene() as scene:
        camera = scene.get_camera()
        with Off():
            for attr, value in (
                ("aperture", -0.1),
                ("aperture", math.nan),
                ("aperture", math.inf),
                ("aperture", True),
                ("focus_distance", 0.0),
                ("focus_distance", -3.0),
                ("focus_distance", "far"),
            ):
                with pytest.raises(AlganConfigurationError):
                    setattr(camera, attr, value)
            with pytest.raises(AlganConfigurationError):
                camera.set_non_recursive(aperture=torch.tensor([[[-1.0]]]))
            with pytest.raises(AlganConfigurationError):
                camera.set(focus_distance=torch.tensor([[[0.0]]]))
            camera.aperture = 0.25
            camera.focus_distance = 7.5
        assert float(camera.aperture) == 0.25
        assert float(camera.focus_distance) == 7.5
    with pytest.raises(AlganConfigurationError):
        Camera(aperture=-1.0)
    with pytest.raises(AlganConfigurationError):
        Camera(focus_distance=0.0)


def test_lens_animates_per_frame_and_survives_the_window_slicer(fresh_scene):
    from algan.render_loop import _slice_render_state

    with Scene(video_settings=VIDEO) as scene:
        camera = scene.get_camera()
        Square().spawn(animate=False)
        with Off():
            camera.aperture = 0.4
        with Seq(runtime=1):
            camera.focus_distance = 10.0
        fps = scene.video_settings.frames_per_second
        n = int(fps) + 2
        tm = scene.timeline_manager
        tm.set_state_to_times(
            torch.arange(n) / fps, active_mobs=[camera, camera.screen]
        )
        try:
            state = scene._materialize_render_state(0, n)
        finally:
            tm.clear_buffers()
    lens = state["camera_lens"]
    assert lens.shape == (n, 1, 2)
    assert torch.all(lens[:, 0, 0] == 0.4)
    focus = lens[:, 0, 1]
    assert float(focus[0]) == pytest.approx(DEFAULT_FOCUS_DISTANCE)
    assert float(focus[-1]) == pytest.approx(10.0)
    assert torch.all(focus[1:] <= focus[:-1] + 1e-6), "the rack is not monotone"
    window = _slice_render_state(state, 2, 5, n)
    assert torch.equal(window["camera_lens"], lens[2:5])
    assert window["camera_lens"].data_ptr() == lens[2:5].data_ptr()


def test_focus_at_racks_onto_the_target_depth(fresh_scene):
    with Scene(video_settings=VIDEO) as scene:
        camera = scene.get_camera()
        near = Circle(radius=0.3).move_to(RIGHT + OUT * 8).spawn(animate=False)
        with Off():
            camera.focus_at(near)
        assert float(camera.focus_distance) == pytest.approx(12.0)
        with Off():
            camera.focus_at(UP * 2)  # a point on the ORIGIN plane
        assert float(camera.focus_distance) == pytest.approx(20.0)
        with pytest.raises(AlganConfigurationError):
            camera.focus_at(OUT * 25)  # behind the camera
        with pytest.raises(AlganConfigurationError):
            camera.focus_at((1.0, 2.0))
        # Recorded: the plane travels from 20 to the target over the pull.
        with Seq(runtime=1):
            camera.focus_at(near)
        fps = scene.video_settings.frames_per_second
        n = int(fps) + 3
        tm = scene.timeline_manager
        tm.set_state_to_times(
            torch.arange(n) / fps, active_mobs=[camera, camera.screen, near]
        )
        try:
            focus = scene._materialize_render_state(0, n)["camera_lens"][:, 0, 1]
        finally:
            tm.clear_buffers()
        assert float(focus[0]) == pytest.approx(20.0)
        assert float(focus[-1]) == pytest.approx(12.0)


def _lens_over(scene, extra_mobs=(), seconds=1.0):
    """The materialized ``camera_lens`` over ``seconds`` (+ a couple of frames)."""
    camera = scene.get_camera()
    fps = scene.video_settings.frames_per_second
    n = int(fps * seconds) + 3
    tm = scene.timeline_manager
    tm.set_state_to_times(
        torch.arange(n) / fps,
        active_mobs=[camera, camera.screen, *scene.actors, *extra_mobs],
    )
    try:
        return scene._materialize_render_state(0, n)["camera_lens"][:, 0]
    finally:
        tm.clear_buffers()


def test_focus_at_follows_a_move_recorded_before_it(fresh_scene):
    """In one ``Sync``, a move written before ``focus_at`` is replayed before
    it, so the pull re-measures the moving subject every frame and lands on
    it (the documented order; a move written after it is not seen).
    """
    with Scene(video_settings=VIDEO) as scene:
        camera = scene.get_camera()
        subject = Circle(radius=0.3).move_to(OUT * 5).spawn(animate=False)
        with Sync(runtime=1):
            subject.move(OUT * 5)  # depth 15 -> 10
            camera.focus_at(subject)
        focus = _lens_over(scene, [subject])[:, 1]
        assert float(camera.focus_distance) == pytest.approx(10.0)
    assert float(focus[0]) == pytest.approx(DEFAULT_FOCUS_DISTANCE)
    assert float(focus[-1]) == pytest.approx(10.0)


def test_overshooting_easings_never_hand_the_renderer_a_bad_lens(fresh_scene):
    """An easing like ``ease_out_back`` takes the interpolation past 1, so a
    blend between two valid distances can dip below zero. Neither a pull nor a
    plain tween may abort the render or reach the kernel negative.
    """
    from algan import easings

    with Scene(video_settings=VIDEO) as scene:
        camera = scene.get_camera()
        near = Circle(radius=0.2).move_to(OUT * 19.2).spawn(animate=False)
        with Seq(runtime=1, easing=easings.ease_out_back):
            camera.focus_at(near)  # 20 -> 0.8, overshooting below zero
        pulled = _lens_over(scene, [near])
    assert torch.all(pulled[:, 1] > 0)
    SceneManager.reset()
    with Scene(video_settings=VIDEO) as scene:
        camera = scene.get_camera()
        with Off():
            camera.aperture = 0.1
        with Sync(runtime=1, easing=easings.ease_out_back):
            camera.focus_distance = 0.8
            camera.aperture = 0.0  # overshoots below zero too
        tweened = _lens_over(scene)
    assert torch.all(tweened[:, 1] > 0)
    assert torch.all(tweened[:, 0] >= 0)


def test_near_orthographic_keeps_the_plane_in_focus(fresh_scene):
    with Scene() as scene:
        camera = scene.get_camera()
        with Off():
            camera.set_near_orthographic()
        # The eye backed off to 8,000 units (1.6x the default distance); the
        # focus followed it, so the ORIGIN plane that was in focus still is.
        depth = float(camera._depth_of(torch.zeros(3)))
        assert depth == pytest.approx(8e3, rel=1e-3)
        assert float(camera.focus_distance) == pytest.approx(depth, rel=1e-5)


def test_a_default_camera_view_copies_the_lens(fresh_scene):
    with Scene() as scene:
        main = scene.get_camera()
        with Off():
            main.aperture = 0.3
            main.focus_distance = 9.0
        view = CameraView(resolution=(32, 18))
        assert float(view.camera.aperture) == pytest.approx(0.3)
        assert float(view.camera.focus_distance) == pytest.approx(9.0)


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


def _camera_arrays():
    """A rotated, translated camera's per-frame arrays, as the tracer builds them."""
    from algan.rendering.raytracing.utils import _pixel_bases

    with Scene() as scene:
        camera = scene.get_camera()
        with Off():
            camera.rotate(30, UP)
            camera.rotate(-20, RIGHT)
            camera.move(RIGHT * 1.5 + UP * 0.5)
        co = camera.location.reshape(1, 3).float().clone()
        sp = camera.screen.location.reshape(1, 3).float().clone()
        sb = camera._get_render_screen_basis().reshape(1, 3, 3).float().clone()
    pbx, pby = _pixel_bases(sb)
    return co, sp, pbx, pby


def _generate(co, sp, pbx, pby, lens, *, width=8, height=6, samples=16, near=0.0):
    from algan.rendering.raytracing.path_tracer_taichi import (
        NEE_META_WIDTH,
        PT_ACC_WIDTH,
        pt_generate,
    )
    from algan.rendering.taichi_runtime import init_taichi

    init_taichi()
    n = width * height
    slots = n * samples
    rs_ro = torch.zeros((slots, 3))
    rs_rd = torch.zeros((slots, 3))
    rs_sca = torch.zeros((slots, 12))
    rs_pix = torch.zeros((slots,), dtype=torch.int32)
    pt_acc = torch.zeros((slots, PT_ACC_WIDTH))
    pt_generate(
        slots, n, 0, 7, 0, 0, width, height,
        float(width // 2), float(height // 2),
        co, sp, pbx, pby, float(near),
        torch.arange(n, dtype=torch.int32),
        rs_ro, rs_rd, rs_sca, rs_pix,
        torch.zeros((NEE_META_WIDTH,)),
        lens.float().reshape(-1, 2).contiguous(),
        pt_acc,
    )  # fmt: skip
    return rs_ro, rs_rd, rs_sca, pt_acc


def test_pt_generate_puts_lens_rays_through_the_pinhole_focus_point(fresh_scene):
    from algan.rendering.raytracing.path_tracer_taichi import _PT_ACC_STOCH

    co, sp, pbx, pby = _camera_arrays()
    fwd = torch.nn.functional.normalize(sp - co, dim=-1)
    radius, focus = 0.6, 11.0

    ro0, rd0, _, acc0 = _generate(co, sp, pbx, pby, torch.zeros(2))
    ro1, rd1, _, acc1 = _generate(co, sp, pbx, pby, torch.tensor([radius, focus]))

    # A zero radius is the pinhole, exactly, and takes no random decision.
    assert torch.equal(ro0, co.expand_as(ro0))
    assert torch.all(acc0[:, _PT_ACC_STOCH] == 0)
    # An open lens flags every path stochastic for adaptive sampling.
    assert torch.all(acc1[:, _PT_ACC_STOCH] == 1)

    offset = ro1 - co
    assert (offset @ fwd.T).abs().max() < 1e-4, "the aperture is not in the lens plane"
    reach = offset.norm(dim=-1)
    assert reach.max() <= radius * (1 + 1e-5)
    # A uniform disk's mean distance from its centre is 2/3 of its radius.
    assert reach.mean() == pytest.approx(2 * radius / 3, rel=0.08)
    assert torch.allclose(rd1.norm(dim=-1), torch.ones(len(rd1)), atol=1e-5)

    # Same jitter (pair 0 is untouched), so each lens ray must cross the focus
    # plane exactly where the pinhole ray of its slot does.
    def on_focus_plane(ro, rd):
        t = (focus - (ro - co) @ fwd.T) / (rd @ fwd.T)
        return ro + rd * t

    miss = (on_focus_plane(ro1, rd1) - on_focus_plane(ro0, rd0)).norm(dim=-1)
    assert miss.max() < 1e-4 * focus


def test_pt_generate_near_clip_stays_planar_for_a_lens_origin(fresh_scene):
    co, sp, pbx, pby = _camera_arrays()
    fwd = torch.nn.functional.normalize(sp - co, dim=-1)
    near = 2.5
    ro, rd, sca, _ = _generate(co, sp, pbx, pby, torch.tensor([0.8, 9.0]), near=near)
    depth = ((ro - co) @ fwd.T).squeeze(-1)
    assert torch.allclose(depth, torch.full_like(depth, near), atol=1e-4)
    t_near = sca[:, 4]
    assert torch.allclose(t_near * (rd @ fwd.T).squeeze(-1), depth, atol=1e-4)


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def _frames(build, *, samples, frames=1, **rt):
    """Raw frames of ``build(scene)``, through ``Scene.get_frames``."""
    SETTINGS.raytracing.set(samples_per_pixel=samples, denoise=False, **rt)
    with Scene(video_settings=VIDEO) as scene:
        scene.set_background(BLACK)
        with Off():
            build(scene)
        out = torch.cat([f.cpu() for f in scene.get_frames(0, frames)])
        plan = scene.last_render_plan
    return out.float(), plan


def _two_depths(aperture, focus=DEFAULT_FOCUS_DISTANCE):
    def build(scene):
        Scene.clear_lights()
        # On the ORIGIN plane (depth 20: the default focus) ...
        Square(size=1.5, color=WHITE).move_to(LEFT * 2).spawn(animate=False)
        # ... and 10 units nearer the camera.
        Circle(radius=0.8, color=RED).move_to(RIGHT * 1.5 + OUT * 10).spawn(
            animate=False
        )
        scene.camera.aperture = aperture
        scene.camera.focus_distance = focus

    return build


def _linear(encoded):
    """Decode sRGB bytes to linear light."""
    v = encoded / 255.0
    return torch.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def _edge_band(row):
    """Pixels on a scanline that are neither background nor full colour."""
    peak = float(row.max())
    return int(((row > 0.1 * peak) & (row < 0.9 * peak)).sum())


def test_out_of_focus_blurs_and_the_focus_plane_stays_sharp(fresh_scene):
    pinhole, _ = _frames(_two_depths(0.0), samples=64)
    lens, plan = _frames(_two_depths(1.0), samples=64)
    assert "depth of field" in plan.requested_features
    assert not plan.unsupported_features

    h, w = pinhole.shape[1:3]
    left, right = slice(0, w // 2), slice(w // 2, w)
    # The square sits on the focus plane: every lens ray of a pixel meets it
    # where the pinhole ray does, so its pixels barely move.
    assert (lens[0, :, left] - pinhole[0, :, left]).abs().mean() < 1.0
    # The circle at depth 10 blurs into a disk of 1 * 10 / 10 = 1 world unit
    # on the focus plane: about 8 pixels here, against a sharp edge's 1-2.
    red_pin = pinhole[0, h // 2, right, 0]
    red_lens = lens[0, h // 2, right, 0]
    assert _edge_band(red_pin) <= 4
    assert _edge_band(red_lens) >= 8
    # Defocus spreads the circle's light rather than losing it -- in linear
    # light, since the frame's bytes are sRGB-encoded after compositing.
    total_pin = _linear(pinhole[0, :, right, 0]).sum()
    total_lens = _linear(lens[0, :, right, 0]).sum()
    assert float(total_lens) == pytest.approx(float(total_pin), rel=0.05)


def test_focusing_on_the_near_object_swaps_which_is_sharp(fresh_scene):
    pinhole, _ = _frames(_two_depths(0.0), samples=64)
    near_focus, _ = _frames(_two_depths(1.0, focus=10.0), samples=64)
    h, w = pinhole.shape[1:3]
    right = slice(w // 2, w)
    assert _edge_band(near_focus[0, h // 2, right, 0]) <= 4
    # The square, 20 units out, now blurs: CoC 1 * 10 / 20 on the focus plane.
    white_row = near_focus[0, h // 2, : w // 2, 1]
    assert _edge_band(pinhole[0, h // 2, : w // 2, 1]) <= 4
    assert _edge_band(white_row) >= 8


def test_adaptive_sampling_runs_lens_pixels_to_the_ceiling(fresh_scene):
    """Unlit 2-D content converges at the floor under a pinhole; with an open
    aperture every pixel's lens sample is a random decision, so none may stop
    early (see ``_PT_ACC_STOCH``).
    """
    samples = 16
    _, pinhole = _frames(_two_depths(0.0), samples=samples)
    _, lens = _frames(_two_depths(1.0), samples=samples)
    assert pinhole.path_samples_mean < samples
    assert lens.path_samples_mean == pytest.approx(samples)


def test_a_rack_focus_changes_the_image_per_frame(fresh_scene):
    SETTINGS.raytracing.set(samples_per_pixel=32, denoise=False)
    with Scene(video_settings=VIDEO) as scene:
        scene.set_background(BLACK)
        with Off():
            _two_depths(1.0, focus=10.0)(scene)
        with Seq(runtime=1):
            scene.camera.focus_distance = DEFAULT_FOCUS_DISTANCE
        fps = int(scene.video_settings.frames_per_second)
        frames = torch.cat([f.cpu() for f in scene.get_frames(0, fps + 1)]).float()
    h, w = frames.shape[1:3]
    right = slice(w // 2, w)
    first = _edge_band(frames[0, h // 2, right, 0])
    last = _edge_band(frames[-1, h // 2, right, 0])
    assert first <= 4, "the near circle should start in focus"
    assert last >= 8, "the rack to the far plane should blur the near circle"


def test_the_deterministic_renderer_reports_an_open_aperture(fresh_scene):
    SETTINGS.raytracing.set(unsupported_feature_policy="error")
    with pytest.raises(UnsupportedFeatureError, match="depth of field"):
        _frames(_two_depths(0.5), samples=1)

    SETTINGS.raytracing.set(unsupported_feature_policy="warn")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        preview, plan = _frames(_two_depths(0.5), samples=1)
    assert any(
        issubclass(w.category, UnsupportedFeatureWarning)
        and "depth of field" in str(w.message)
        for w in caught
    )
    assert plan.unsupported_features == ("depth of field",)
    # Discarded deliberately, the preview is the pinhole frame, untouched.
    pinhole, pinhole_plan = _frames(_two_depths(0.0), samples=1)
    assert "depth of field" not in pinhole_plan.requested_features
    assert torch.equal(preview, pinhole)
