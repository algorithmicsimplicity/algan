from __future__ import annotations

import inspect
import math

import pytest
import torch

import algan
from algan import BLUE, GREEN, RED, WHITE, Off, Scene, Seq, Square, Sync, easings
from algan.rendering.shaders.material_shaders import basic_material_shader

pytestmark = pytest.mark.usefixtures("fresh_scene")


def test_batch_mobs_is_an_authoring_export():
    namespace = {}
    exec("from algan import *", namespace)
    packed = namespace["batch_mobs"]([Square(), Square()])
    assert len(packed) == 2


def test_regular_polygon_uses_degrees_while_manim_uses_radians():
    import algan.manim as mn

    native = algan.RegularPolygon(n=5, start_angle=90)
    manim = mn.RegularPolygon(n=5, start_angle=math.pi / 2)
    torch.testing.assert_close(
        native.control_points.location,
        manim.control_points.location,
        atol=1e-5,
        rtol=1e-5,
    )
    assert native.start_angle == 90


def test_root_path_arc_and_wiggle_take_degrees():
    import algan.manim as mn

    native = algan.Line(algan.LEFT, algan.RIGHT, path_arc=60)
    manim = mn.Line(algan.LEFT, algan.RIGHT, path_arc=math.pi / 3)
    torch.testing.assert_close(
        native.control_points.location,
        manim.control_points.location,
        atol=1e-5,
        rtol=1e-5,
    )
    assert inspect.signature(algan.Wiggle).parameters[
        "rotation_angle"
    ].default == pytest.approx(math.degrees(0.02 * math.pi))


@pytest.mark.parametrize(
    ("build", "name"),
    [
        (lambda: algan.RegularPolygon(n=5, start_angle=math.pi / 2), "start_angle"),
        (lambda: algan.Line(path_arc=math.pi / 4), "path_arc"),
        (lambda: algan.Wiggle(Square().spawn(), rotation_angle=0.1), "rotation_angle"),
    ],
)
def test_radian_habits_warn_on_root_angles(build, name):
    from algan.errors import ApproximationWarning

    with pytest.warns(ApproximationWarning, match=name):
        build()


def test_root_adapters_accept_native_constructor_style_and_pose():
    rounded = algan.RoundedRectangle(
        opacity=0.3,
        fill_opacity=0.4,
        stroke_opacity=0.8,
        location=(2, 1, 0),
        unlit=True,
    )
    assert rounded.opacity.item() == pytest.approx(0.3)
    assert rounded.fill_opacity.item() == pytest.approx(0.4)
    assert rounded.stroke_opacity.item() == pytest.approx(0.8)
    torch.testing.assert_close(
        rounded.location.flatten(), torch.tensor([2.0, 1.0, 0.0])
    )
    assert all(m.shader is basic_material_shader for m in rounded.get_descendants()), [
        (type(m).__name__, getattr(m.shader, "__name__", None))
        for m in rounded.get_descendants()
    ]


@pytest.mark.fast
def test_fill_and_stroke_alpha_animate_independently_without_repainting_texture():
    scene = Scene()
    square = Square(
        color=BLUE,
        stroke_color=RED,
        fill_opacity=0.2,
        stroke_opacity=0.8,
        opacity=0.5,
        grid_width=2,
        scene=scene,
    )
    rgb = square.grid.color[..., :3].clone()
    with Off():
        square.spawn()
    with Seq(easing=easings.identity):
        square.fill_opacity = 0.6
        square.stroke_opacity = 0.2
    scene.timeline_manager.set_state_to_times(torch.tensor([0.5, 1.5]))
    torch.testing.assert_close(square.fill_opacity[:, 0, 0], torch.tensor([0.4, 0.6]))
    torch.testing.assert_close(square.stroke_opacity[:, 0, 0], torch.tensor([0.8, 0.5]))
    torch.testing.assert_close(square.grid.color[..., :3], rgb.expand(2, -1, -1))
    torch.testing.assert_close(square.opacity, torch.full_like(square.opacity, 0.5))


def test_color_assignment_keeps_explicit_component_opacity():
    square = Square(color=BLUE, fill_opacity=0.2, stroke_opacity=0.5)
    square.color = RED
    square.stroke_color = GREEN
    assert square.fill_opacity.flatten()[0].item() == pytest.approx(0.2)
    assert square.stroke_opacity.flatten()[0].item() == pytest.approx(0.5)
    torch.testing.assert_close(
        square.grid.color[..., :3].flatten(), RED[:3].to(square.grid.color)
    )
    # A color carrying its own alpha still sets it, and so does the property,
    # including raising it back to fully opaque.
    square.color = RED.set_opacity(0.7)
    assert square.fill_opacity.flatten()[0].item() == pytest.approx(0.7)
    square.fill_opacity = 1
    assert square.fill_opacity.flatten()[0].item() == pytest.approx(1.0)
    square.fill_opacity = 0.3
    algan.Group(square).color = BLUE
    assert square.fill_opacity.flatten()[0].item() == pytest.approx(0.3)


@pytest.mark.fast
def test_animated_recolor_keeps_component_opacity_on_every_frame():
    scene = Scene()
    translucent = Square(color=BLUE, fill_opacity=0.25, scene=scene)
    plain = Square(color=BLUE.set_opacity(0.25), scene=scene)
    with Off():
        translucent.spawn()
        plain.spawn()
    with Sync(easing=easings.identity):
        translucent.color = RED
        plain.color = RED
    scene.timeline_manager.set_state_to_times(torch.tensor([0.25, 0.5, 1.0]))
    torch.testing.assert_close(
        translucent.fill_opacity[:, 0, 0], torch.full((3,), 0.25)
    )
    # Without an explicit component opacity, a color write still sets alpha.
    torch.testing.assert_close(
        plain.grid.color[:, 0, -1], torch.tensor([0.4375, 0.625, 1.0])
    )
    torch.testing.assert_close(
        translucent.grid.color[:, 0, 0],
        torch.tensor([0.25, 0.5, 1.0]) * RED[0]
        + torch.tensor([0.75, 0.5, 0.0]) * BLUE[0],
    )


def test_translucent_manim_fill_survives_recolor():
    rounded = algan.RoundedRectangle(fill_opacity=0.4, stroke_opacity=0.6)
    rounded.color = RED
    rounded.stroke_color = WHITE
    assert rounded.fill_opacity.flatten()[0].item() == pytest.approx(0.4)
    assert rounded.stroke_opacity.flatten()[0].item() == pytest.approx(0.6)


def test_closed_manim_signature_still_takes_component_opacity():
    # Manim's SampleSpace spells out its keywords and has no stroke_opacity.
    space = algan.SampleSpace(stroke_opacity=0.3, fill_opacity=0.6)
    assert space.stroke_opacity.flatten()[0].item() == pytest.approx(0.3)
    assert space.fill_opacity.flatten()[0].item() == pytest.approx(0.6)


def test_unfilled_path_fill_alpha_does_not_fade_stroke():
    line = algan.Line(color=BLUE, stroke_opacity=0.7)
    line.fill_opacity = 0
    assert line.grid.color[..., -1].item() == pytest.approx(0.7)
    line.stroke_opacity = 0.2
    assert line.grid.color[..., -1].item() == pytest.approx(0.2)
    assert line.stroke_opacity.item() == pytest.approx(0.2)


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: algan.Line3D(start=(0, 0, 0), end=(1, 1, 0), unlit=True, opacity=0.4),
        lambda: algan.Cylinder(closed=True, unlit=True, opacity=0.4),
        lambda: algan.Group(algan.Sphere(), algan.Sphere(), unlit=True),
    ],
)
def test_unlit_constructor_reaches_composite_geometry(constructor):
    mob = constructor()
    assert all(m.shader is basic_material_shader for m in mob.get_descendants())


@pytest.mark.parametrize("attribute", ["fill_opacity", "stroke_opacity"])
def test_component_opacity_rejects_invalid_values(attribute):
    square = Square()
    with pytest.raises(algan.AlganConfigurationError):
        setattr(square, attribute, 1.5)


def test_unlit_constructor_matches_explicit_material_pixels():
    """Exercise both textured surfaces and Line3D through the real renderer."""
    from contextlib import closing

    def render(constructor_flag):
        scene = Scene(algan.PREVIEW.set(resolution=(48, 48), frames_per_second=4))
        with Off():
            scene.camera.fly_to((0, 0, 5), look_at=algan.ORIGIN)
            line = algan.Line3D(
                start=(-1, 0, 1),
                end=(1, 0, 1),
                radius=0.15,
                color=RED,
                opacity=0.6,
                unlit=constructor_flag,
                scene=scene,
            )
            plane = algan.Surface(
                lambda uv: torch.cat((uv * 2 - 1, torch.zeros_like(uv[..., :1])), -1),
                grid_width=2,
                grid_height=2,
                unlit=constructor_flag,
                scene=scene,
                color_texture=torch.tensor(
                    [
                        [[0.0, 0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0, 1.0]],
                        [[1.0, 1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0, 1.0]],
                    ]
                ),
            )
            if not constructor_flag:
                line.set_material(algan.UnlitMaterial(opacity=0.6))
                plane.set_material(algan.UnlitMaterial())
            line.spawn()
            plane.spawn()
        scene.wait(1)
        with closing(scene.get_frames(0, 1, post_processes=())) as frames:
            return torch.cat(list(frames)).clone()

    actual, expected = render(True), render(False)
    assert actual[..., :3].max() > 0
    torch.testing.assert_close(actual, expected, atol=2, rtol=0)
