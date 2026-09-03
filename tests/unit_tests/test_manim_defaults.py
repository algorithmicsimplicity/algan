"""Tests for ``Scene.use_manim_defaults()`` and the Manim coordinate conversion."""

from __future__ import annotations

import math

import pytest
import torch

from algan import SETTINGS, Scene
from algan.manim_defaults import (
    MANIM_FOCAL_DISTANCE,
    MANIM_FRAME_HEIGHT,
    MANIM_LIGHT_SOURCE,
    manim_fov,
)


@pytest.fixture
def scene():
    """A fresh Scene, with the process-global style settings restored after."""
    style = SETTINGS.style
    saved_style = (style.default_material, style.background.clone())
    saved_placement = style.border_placement
    saved_ratio = style.manim_stroke_width_ratio
    saved_tonemapping = SETTINGS.raytracing.tonemapping
    saved_linear = SETTINGS.raytracing.linear_color_space
    created = Scene()
    try:
        yield created
    finally:
        # Restored to what was there, not to the documented default: a suite
        # that hardcodes the default silently repairs a leak from an earlier
        # test instead of exposing it.
        SETTINGS.style.set(
            default_material=saved_style[0],
            background=saved_style[1],
            border_placement=saved_placement,
            manim_stroke_width_ratio=saved_ratio,
        )
        SETTINGS.raytracing.set(
            tonemapping=saved_tonemapping, linear_color_space=saved_linear
        )


def test_manim_fov_matches_manims_projection():
    # Manim's ThreeDCamera scales a point by focal_distance / (focal_distance - z),
    # which is a pinhole eye focal_distance from the frame plane. The frame is
    # frame_height tall there, so that is the angle the camera spans.
    assert manim_fov() == pytest.approx(22.619864948, abs=1e-6)
    assert manim_fov() == pytest.approx(
        math.degrees(2 * math.atan((MANIM_FRAME_HEIGHT / 2) / MANIM_FOCAL_DISTANCE))
    )


def test_manim_fov_frames_eight_units_at_the_origin_plane():
    # The whole point of the angle: at the distance the camera sits from the
    # origin, exactly Manim's 8 units of height must be in view.
    half_height = MANIM_FOCAL_DISTANCE * math.tan(math.radians(manim_fov()) / 2)
    assert half_height == pytest.approx(MANIM_FRAME_HEIGHT / 2)


def test_there_is_no_coordinate_conversion_left_to_do():
    """The two engines' axes agree, so the helpers that converted are gone.

    They existed only because Algan's ``OUTWARD`` was ``-z``. Anything still
    importing them is carrying a mirror that would now double up.
    """
    import algan
    import algan.manim_defaults as manim_defaults

    for name in ("from_manim_coordinates", "to_manim_coordinates"):
        assert not hasattr(manim_defaults, name)
        assert name not in algan.__all__
    assert torch.equal(algan.OUTWARD, torch.tensor((0.0, 0.0, 1.0)))


def test_use_manim_defaults_positions_the_camera(scene):
    scene.use_manim_defaults()
    camera = scene.get_camera()
    # Manim's eye sits focal_distance from the frame plane, on the +z side --
    # which is Algan's OUTWARD, so (0, 0, 20).
    assert camera.location.flatten().tolist() == pytest.approx(
        [0.0, 0.0, MANIM_FOCAL_DISTANCE]
    )
    assert camera.get_fov() == pytest.approx(manim_fov(), abs=1e-4)
    assert not bool(camera.orthographic)


def test_use_manim_defaults_installs_manims_light(scene):
    scene.use_manim_defaults()
    lights = scene.get_light_sources()
    assert len(lights) == 1
    assert lights[0].location.flatten().tolist() == pytest.approx(
        list(MANIM_LIGHT_SOURCE)
    )


def test_use_manim_defaults_turns_off_tonemapping(scene):
    # Manim writes its colours out untouched. Algan's tonemap darkens every fill
    # by about 10/255, which reads as a colour error rather than a roll-off.
    SETTINGS.raytracing.set(tonemapping=True)
    scene.use_manim_defaults()
    assert SETTINGS.raytracing.tonemapping is False


def test_use_manim_defaults_uses_manims_display_referred_color_space(scene):
    # Manim does its arithmetic in sRGB: it composites alpha, antialiases and
    # gradients display-referred values directly. Algan's linear default is the
    # physically correct choice, but it puts a fill of opacity a on a**(1/2.2)
    # of the colour -- MAROON at 0.55 lands on (150,71,87) where Manim puts
    # (108,52,63).
    SETTINGS.raytracing.set(linear_color_space=True)
    scene.use_manim_defaults()
    assert SETTINGS.raytracing.linear_color_space is False


def test_the_color_space_belongs_to_the_shading_group(scene):
    # It is process-wide and recompiles kernels, so it must follow the same
    # opt-out as the rest of the colour pipeline rather than being unconditional.
    SETTINGS.raytracing.set(linear_color_space=True)
    scene.use_manim_defaults(shading=False)
    assert SETTINGS.raytracing.linear_color_space is True


def test_use_manim_defaults_installs_the_manim_material(scene):
    from algan.rendering.shaders.material_shaders import manim_shader
    from algan.rendering.shaders.materials import ManimMaterial

    scene.use_manim_defaults()
    # The default material is Manim's 3-D shading, not merely an unlit
    # passthrough: Manim shades 3-D geometry via get_shaded_rgb, and this is
    # what makes a material-less imported 3-D mob do the same.
    assert isinstance(SETTINGS.style.default_material, ManimMaterial)
    assert SETTINGS.style.default_material.shader is manim_shader


def test_use_manim_defaults_centres_a_filled_shapes_stroke(scene):
    # Manim strokes an SVG path: half the width falls outside the outline.
    # Algan lays a filled shape's stroke wholly inside by default, which puts an
    # imported shape's silhouette half a stroke width in from where Manim draws
    # it -- measured at 5.79 px on a 12 px stroke.
    SETTINGS.style.set(border_placement="inward")
    scene.use_manim_defaults()
    assert SETTINGS.style.border_placement == "centered"


def test_use_manim_defaults_can_leave_stroke_geometry_alone(scene):
    # Both settings are process-wide, so they have to be refusable together.
    SETTINGS.style.set(border_placement="inward", manim_stroke_width_ratio=2.0)
    scene.use_manim_defaults(stroke_geometry=False)
    assert SETTINGS.style.border_placement == "inward"
    assert SETTINGS.style.manim_stroke_width_ratio == 2.0


def test_use_manim_defaults_uses_manims_exact_stroke_width_ratio(scene):
    # Algan's convention is the round "Manim's number is twice Algan's". The
    # exact figure is MANIM_FRAME_HEIGHT / (PREVIEW_height * 0.01) = 2.0202:
    # they would agree if PREVIEW were 400 px tall rather than 396.
    from algan.manim_defaults import manim_stroke_width_ratio
    from algan.settings.video_settings import PREVIEW

    assert manim_stroke_width_ratio() == pytest.approx(
        MANIM_FRAME_HEIGHT / (PREVIEW.resolution[1] * 0.01)
    )
    assert manim_stroke_width_ratio() == pytest.approx(2.020202, abs=1e-6)

    SETTINGS.style.set(manim_stroke_width_ratio=2.0)
    scene.use_manim_defaults()
    assert SETTINGS.style.manim_stroke_width_ratio == pytest.approx(
        manim_stroke_width_ratio()
    )


def test_the_stroke_width_ratio_round_trips_through_the_compat_layer(scene):
    """Import and export must invert each other under either ratio.

    They are separate call sites reading one setting, which is the only reason
    a round trip survives the ratio changing underneath it.
    """
    manim = pytest.importorskip("manim")

    from algan.mobs.manim_compat import to_manim
    from algan.mobs.manim_mob import ManimMob

    for ratio in (2.0, 2.020202):
        SETTINGS.style.set(manim_stroke_width_ratio=ratio)
        source = manim.Square(side_length=2.0)
        source.set_stroke(manim.WHITE, width=8.0, opacity=1.0)
        imported = ManimMob(source, scene=scene)
        assert float(imported.stroke_width.reshape(-1)[0]) == pytest.approx(
            8.0 / ratio, abs=1e-5
        )
        assert float(to_manim(imported).stroke_width) == pytest.approx(8.0, abs=1e-4)


def test_a_flat_shade_in_3d_face_takes_the_patch_plan(scene):
    """A Cube face is FLAT, so only Manim's flag can route it to a material.

    An analytic circuit is drawn unlit; a PN patch is 3-D geometry the default
    material and the lights reach. Planarity alone would leave every Cube face
    unlit where Manim shades it.
    """
    manim = pytest.importorskip("manim")

    from algan.mobs.manim_mob import ManimMob

    # Manim builds Cube out of Square(shade_in_3d=True) faces.
    face = manim.Cube(side_length=2.0).submobjects[0]
    assert face.shade_in_3d is True
    imported = ManimMob(face, scene=scene)
    assert imported.shade_in_3d is True
    assert imported._nonplanar_plan is not None
    assert imported._nonplanar_plan.mode == "patch"

    # A flat shape WITHOUT the flag is untouched: still the analytic path.
    plain = ManimMob(
        manim.Square(side_length=2.0).set_fill(manim.BLUE, 1.0), scene=scene
    )
    assert plain.shade_in_3d is False
    assert plain._nonplanar_plan is None


def test_an_unfilled_shade_in_3d_path_stays_on_planarity(scene):
    """An open path bounds no surface, so there is no patch to make of it."""
    manim = pytest.importorskip("manim")

    from algan.mobs.manim_mob import ManimMob

    line = manim.Line(manim.LEFT, manim.RIGHT)
    line.shade_in_3d = True
    imported = ManimMob(line, scene=scene)
    assert imported.shade_in_3d is True
    assert imported._nonplanar_plan is None


def test_an_unknown_border_placement_is_rejected(scene):
    from algan.errors import AlganConfigurationError

    with pytest.raises(AlganConfigurationError):
        SETTINGS.style.set(border_placement="outward")
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(AlganConfigurationError):
            SETTINGS.style.set(manim_stroke_width_ratio=bad)


def test_use_manim_defaults_flags_are_independent(scene):
    camera_before = scene.get_camera().location.clone()
    scene.use_manim_defaults(camera=False, shading=False)
    assert torch.equal(scene.get_camera().location, camera_before)
    # Lights belong to the shading group, so they are untouched too.
    assert len(scene.get_light_sources()) == 1


def test_use_manim_defaults_leaves_video_settings_alone_by_default(scene):
    before = scene.video_settings.resolution
    scene.use_manim_defaults()
    assert scene.video_settings.resolution == before


def test_use_manim_defaults_can_set_manims_video_settings(scene):
    scene.use_manim_defaults(video_settings=True)
    assert scene.video_settings.resolution == (1920, 1080)
    assert scene.video_settings.frames_per_second == 60


def test_use_manim_defaults_returns_the_scene(scene):
    assert scene.use_manim_defaults() is scene


def test_manim_mob_imports_depth_unchanged(scene):
    """Manim's OUT is Algan's OUTWARD, so an imported point keeps its z."""
    manim = pytest.importorskip("manim")

    depth = 1.5
    source = manim.Square(side_length=2).shift(manim.OUT * depth)
    imported = ManimMobFor(source, scene)
    assert imported[..., 2].flatten()[0].item() == pytest.approx(depth)


def ManimMobFor(manim_mob, scene):
    """Import ``manim_mob`` into ``scene`` and return its control point locations."""
    from algan.mobs.manim_mob import ManimMob

    return ManimMob(manim_mob, scene=scene, add_to_scene=False).control_points.location


def test_reset_restores_algans_own_defaults(scene):
    """``reset(rebuild_timeline=False)`` re-runs the scene initializer.

    The camera is no longer what says so: Algan's own default camera now sits
    where Manim's does, so ``use_manim_defaults()`` leaves it where the
    initializer would put it anyway. The lighting still differs -- Manim lights
    from ``MANIM_LIGHT_SOURCE``, Algan's initializer from beside the camera --
    so that is what shows the initializer ran again.
    """
    scene.use_manim_defaults()
    manim_light = scene.get_light_sources()[0].location.flatten().tolist()
    assert manim_light == pytest.approx(list(MANIM_LIGHT_SOURCE))

    scene.reset(rebuild_timeline=False)

    lights = scene.get_light_sources()
    assert len(lights) == 1
    assert lights[0].location.flatten().tolist() != pytest.approx(
        list(MANIM_LIGHT_SOURCE)
    )
    # And the camera the initializer restores is Algan's own, which is Manim's.
    assert scene.get_camera().location.flatten().tolist() == pytest.approx(
        [0.0, 0.0, MANIM_FOCAL_DISTANCE]
    )
