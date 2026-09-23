"""Authoring regressions found while building a Short."""

import math

import pytest
import torch

from algan import (
    BLACK,
    BLUE,
    DOWN,
    LEFT,
    ORIGIN,
    OUT,
    RED,
    RIGHT,
    UP,
    WHITE,
    AmbientLight,
    Cube,
    DirectionalLight,
    HemisphereLight,
    Line3D,
    Off,
    PointLight,
    Prism,
    RectAreaLight,
    Scene,
    Seq,
    SpotLight,
    easings,
)
from algan.scene_manager import SceneManager
from algan.utils.mob_utils import batch_mobs


@pytest.fixture(autouse=True)
def fresh_scene():
    SceneManager.reset()
    yield
    SceneManager.reset()


@pytest.mark.parametrize(
    "light_type",
    [
        PointLight,
        DirectionalLight,
        AmbientLight,
        HemisphereLight,
        SpotLight,
        RectAreaLight,
    ],
)
def test_lights_default_to_white_and_keep_explicit_colours(light_type):
    with Scene():
        assert torch.allclose(light_type().color, WHITE)
        assert torch.allclose(light_type(color=RED).color, RED)
        assert torch.allclose(light_type(color=BLACK).color, BLACK)
        assert torch.allclose(light_type(ORIGIN, color=BLUE).color, BLUE)


@pytest.mark.parametrize("solid_type", [Prism, Cube])
@pytest.mark.parametrize(
    ("styling", "expected"),
    [
        ({"color": RED}, RED),
        ({"color": BLACK}, BLACK),
        ({"fill_color": RED}, RED),
        ({"color": RED, "fill_color": BLUE}, BLUE),
        ({"color": RED, "faces_config": {"fill_color": BLUE}}, BLUE),
    ],
)
def test_box_constructor_colours_reach_the_rendered_faces(
    solid_type, styling, expected
):
    with Scene(), Off():
        solid = solid_type(**styling).spawn()
        for face in solid._face_primitive_mobs():
            assert torch.allclose(face.color[..., :3], expected[..., :3])
        # Check the actual triangle payload too, not just the parent's colour.
        for primitive in solid.get_render_primitives():
            assert torch.allclose(primitive.colors[..., :3], expected[..., :3])


@pytest.mark.fast
@pytest.mark.parametrize("animated", [False, True])
@pytest.mark.parametrize("per_member", [False, True])
def test_packed_line3d_moves_reach_nested_cap_vertices(animated, per_member):
    with Scene() as scene:
        with Off():
            pack = batch_mobs(
                [
                    Line3D(
                        start=LEFT + UP * i * 0.5,
                        end=RIGHT + UP * i * 0.5,
                        add_to_scene=False,
                    )
                    for i in range(4)
                ]
            )
            # Keep an unpacked oracle so this checks member order as well as
            # broadcasting, including the caps' grandchildren.
            references = [
                Line3D(
                    start=LEFT + UP * i * 0.5,
                    end=RIGHT + UP * i * 0.5,
                    add_to_scene=False,
                )
                for i in range(4)
            ]
            if animated:
                pack.spawn()
        delta = torch.stack([DOWN * (i + 1) for i in range(4)]) if per_member else DOWN
        context = Seq(runtime=1, easing=easings.identity) if animated else Off()
        with context:
            pack.move(delta)
        for fraction in [0.0, 0.5, 1.0] if animated else [1.0]:
            if animated:
                scene.timeline_manager.set_state_to_times(torch.tensor([fraction]))
            for i, reference in enumerate(references):
                member = pack[i]
                offset = DOWN * (i + 1 if per_member else 1) * fraction
                actual = member.get_animated_attribute(
                    "location", include_descendants=True
                )
                expected = reference.get_animated_attribute(
                    "location", include_descendants=True
                )
                # Compare unordered geometry: packing changes buffer order.
                actual = actual.reshape(-1, 3) - offset.reshape(3)
                expected = expected.reshape(-1, 3)
                assert torch.allclose(
                    actual.sort(dim=0).values,
                    expected.sort(dim=0).values,
                    atol=2e-6,
                )
        scene.timeline_manager.clear_buffers()


def test_default_scene_camera_matches_documented_framing():
    with Scene() as scene:
        camera = scene.get_camera()
        assert torch.allclose(camera.location, OUT * 20)
        assert camera.fov == pytest.approx(math.degrees(2 * math.atan(4 / 20)))
        assert 2 * 20 * math.tan(math.radians(camera.fov) / 2) == pytest.approx(8)
