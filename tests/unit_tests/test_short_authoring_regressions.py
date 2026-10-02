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
    BezierCircuitCubic,
    Circle,
    Cube,
    DirectionalLight,
    Group,
    HemisphereLight,
    Line,
    Line3D,
    Mob,
    Off,
    PointLight,
    Prism,
    RectAreaLight,
    Scene,
    Seq,
    SpotLight,
    Square,
    Sync,
    Tex,
    Text,
    TransformMatchingTex,
    animated_function,
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


@pytest.mark.parametrize("new_color", [None, RED])
@pytest.mark.parametrize("peak", [WHITE, BLUE.set_glow(2)])
@pytest.mark.parametrize("via_color", [False, True])
def test_color_pulses_preserve_translucent_fill(new_color, peak, via_color):
    with Scene() as scene:
        with Off():
            style = (
                {"color": BLUE.set_opacity(0.2)}
                if via_color
                else {"color": BLUE, "fill_opacity": 0.2}
            )
            circle = Circle(**style, stroke_opacity=0.7).spawn()
        with Seq(runtime=1, easing=easings.identity):
            circle.pulse_color(peak, new_color=new_color)
        scene.timeline_manager.set_state_to_times(torch.tensor([0.0, 0.5, 1.0]))
        assert torch.allclose(
            circle.fill_opacity, torch.full_like(circle.fill_opacity, 0.2)
        )
        assert torch.allclose(
            circle.stroke_opacity, torch.full_like(circle.stroke_opacity, 0.7)
        )
        assert torch.allclose(circle.grid.color[1, :, :4], peak[:4])
        assert torch.allclose(
            circle.grid.color[2, :, :3], (BLUE if new_color is None else new_color)[:3]
        )


def test_color_pulse_accepts_explicit_translucent_targets():
    with Scene() as scene:
        with Off():
            circle = Circle(fill_opacity=0.2).spawn(False)
        with Seq(easing=easings.identity):
            circle.pulse_color(WHITE.set_opacity(0.5), new_color=RED.set_opacity(0.4))
        scene.timeline_manager.set_state_to_times(torch.tensor([0.0, 0.5, 1.0]))
        assert torch.allclose(
            circle.fill_opacity[:, 0, 0], torch.tensor([0.2, 0.5, 0.4])
        )


@pytest.mark.fast
def test_group_of_packed_glyphs_bounds_follow_text_move():
    with Scene(), Off():
        text = Text("hello", font="Algan Test Sans")
        group = Group(text.character_mobs[1:4])
        outer = Group(group)
        original = group.get_bounding_box().clone()
        points = group[0].control_points.location.clone()
        text.move(RIGHT * 3 + UP)
        assert torch.allclose(group[0].control_points.location, points + RIGHT * 3 + UP)
        assert torch.allclose(group.get_bounding_box(), original + RIGHT * 3 + UP)
        assert torch.allclose(outer.get_bounding_box(), group.get_bounding_box())


@pytest.mark.fast
@pytest.mark.parametrize("packed", [False, True])
def test_recursive_opacity_values_can_be_assigned_back_to_group(packed):
    with Scene() as scene:
        with Off():
            child = Text("abc", font="Algan Test Sans") if packed else Square()
            group = Group(child, Circle(opacity=0.3)).spawn(False)
        original = group.get_animated_attribute("opacity", include_descendants=True)
        own_shape = group.opacity.shape
        with Seq(runtime=1, easing=easings.identity):
            group.opacity = original * 0.5
        assert group.opacity.shape == own_shape
        scene.timeline_manager.set_state_to_times(torch.tensor([0.0, 0.5, 1.0]))
        actual = group.get_animated_attribute("opacity", include_descendants=True)
        assert torch.allclose(
            actual, original * torch.tensor([1.0, 0.75, 0.5]).view(-1, 1, 1)
        )


@pytest.mark.fast
def test_mob_initialization_preserves_subclass_name():
    class NamedMob(Mob):
        def __init__(self, **kwargs):
            self.name = "subclass name"
            super().__init__(**kwargs)

    class ClassNamedMob(Mob):
        name = "class name"

    with Scene():
        assert NamedMob().name == "subclass name"
        assert ClassNamedMob().name == "class name"
        assert NamedMob(name="explicit name").name == "explicit name"
        assert Mob().name is None


@pytest.mark.fast
@pytest.mark.parametrize("runtime", [2.0, 4.0])
def test_sync_runtime_stretches_move_alongside_tex_fade(runtime):
    with Scene() as scene:
        with Off():
            mover = Square().spawn(False)
            label = Tex("x+y").spawn(False)
        with Sync(runtime=runtime, easing=easings.identity):
            mover.move(RIGHT * 4)
            label.despawn()
        scene.timeline_manager.set_state_to_times(
            torch.tensor([0.0, runtime / 2, runtime])
        )
        assert torch.allclose(mover.location[:, 0, 0], torch.tensor([0.0, 2.0, 4.0]))


@pytest.mark.parametrize("grouped", [False, True])
def test_line_color_changes_its_visible_stroke(grouped):
    with Scene() as scene:
        with Off():
            line = Line(color=BLUE, stroke_opacity=0.4)
            subject = Group(line) if grouped else line
            subject.spawn(False)
        with Seq(runtime=1, easing=easings.identity):
            subject.color = RED
        scene.timeline_manager.set_state_to_times(torch.tensor([0.0, 0.5, 1.0]))
        expected = torch.stack([BLUE[:3], (BLUE[:3] + RED[:3]) / 2, RED[:3]])
        assert torch.allclose(line.stroke_color[:, 0, :3], expected)
        assert torch.allclose(
            line.stroke_opacity, torch.full_like(line.stroke_opacity, 0.4)
        )


@pytest.mark.parametrize("value", [None, 1.0, "not a mob", object()])
def test_animated_function_rejects_non_mob_first_argument(value):
    @animated_function(animated_args={"t": 0})
    def highlight(mob, t=1):
        mob.opacity = t

    with pytest.raises(
        TypeError, match=r"highlight.*first argument.*Mob.*" + type(value).__name__
    ):
        highlight(value)


@pytest.mark.parametrize("animate", [False, True])
@pytest.mark.parametrize("preselect", [False, True])
def test_tex_segments_spawn_independently(animate, preselect):
    with Scene() as scene:
        text = Tex("ab", "+", "c")
        first = text.get_segment(0)
        last = text.get_segment(2) if preselect else None
        scene.wait(1)
        first.spawn(animate)
        scene.wait(1)
        last_start = float(scene.animation_manager.context.timespan.current_time)
        if last is None:
            last = text.get_segment(2)
            assert text._character_batch.opacity[0, 3, 0] == 0
        last.spawn(animate)
        scene.wait(1)
        whole_start = float(scene.animation_manager.context.timespan.current_time)
        text.spawn(False)
        scene.timeline_manager.set_state_to_times(
            torch.tensor([0.5, 1.5, last_start + 0.5, whole_start + 0.5])
        )
        alpha = text._character_batch.opacity[..., 0]
        assert torch.equal(alpha[0], torch.zeros_like(alpha[0]))
        assert torch.all(alpha[1, :2] > 0)
        assert torch.equal(alpha[1, 2:], torch.zeros_like(alpha[1, 2:]))
        assert alpha[2, 2] == 0
        assert alpha[2, 3] > 0
        assert torch.allclose(alpha[3], torch.ones_like(alpha[3]))
        scene.timeline_manager.clear_buffers()
        scene.timeline_manager.set_state_to_times(
            torch.tensor([last_start + 0.5, 0.5, 1.5])
        )
        assert torch.allclose(text._character_batch.opacity[..., 0], alpha[[2, 0, 1]])


@pytest.mark.parametrize("collate", [False, True])
@pytest.mark.parametrize("first_count", [2, 3])
def test_overlapping_packed_spawns_keep_each_members_opacity(
    collate, first_count, monkeypatch
):
    if not collate:
        monkeypatch.setenv("ALGAN_OPT_DISABLE", "collate")
    with Scene() as scene:
        text = Text("abc", font="Algan Test Sans")
        text.character_mobs[0].opacity = 0.3
        text.character_mobs[1].opacity = 0.6
        with Seq(easing=easings.identity):
            text[:first_count].spawn()
            text[1:].spawn()
        scene.timeline_manager.set_state_to_times(torch.tensor([0.5, 1.0, 1.5, 2.0]))
        assert torch.allclose(
            text._character_batch.opacity[..., 0],
            torch.tensor(
                [
                    [0.15, 0.3, 0 if first_count == 2 else 0.5],
                    [0.3, 0.6, 0 if first_count == 2 else 1],
                    [0.3, 0.6, 0.5 if first_count == 2 else 1],
                    [0.3, 0.6, 1],
                ]
            ),
        )


@pytest.mark.fast
@pytest.mark.parametrize(
    ("runtime", "equalize", "expected"),
    [(None, None, 4.0), (4, False, 4.0), (4, True, 2.0)],
)
def test_sync_retains_relative_durations_when_requested(runtime, equalize, expected):
    with Scene() as scene:
        with Off():
            short = Square().spawn(False)
            long = Circle().spawn(False)
        kwargs = {} if equalize is None else {"equalize_runtimes": equalize}
        with Sync(runtime=runtime, easing=easings.identity, **kwargs):
            short.move(RIGHT * 4)
            with Seq(runtime=2):
                long.move(UP * 4)
            with Off():
                short.color = RED
        scene.timeline_manager.set_state_to_times(
            torch.tensor([1.0 if runtime is None else runtime / 2])
        )
        assert float(short.location[0, 0, 0]) == pytest.approx(expected)
        assert float(long.location[0, 0, 1]) == pytest.approx(2.0)


def test_matching_tex_fades_mismatches_without_distorting_glyphs():
    with Scene() as scene:
        with Off():
            source = Tex("W", color=RED).move(LEFT * 2).spawn(False)
            target = Tex("i", color=BLUE).move(RIGHT * 2)
        shapes = {}
        for text, color in ((source, RED), (target, BLUE)):
            points = text.character_mobs[0].control_points.location.clone()
            shapes[tuple(color[:3].tolist())] = points - points.mean(-2, keepdim=True)
        with Seq(easing=easings.identity):
            TransformMatchingTex(
                source, target, runtime=2, fade_transform_mismatches=True
            )
        scene.timeline_manager.set_state_to_times(torch.tensor([1.0]))
        visible = []
        for actor in scene.actors:
            if isinstance(actor, BezierCircuitCubic) and bool(
                (actor.opacity > 0).any()
            ):
                points = actor.control_points.location
                expected = shapes[tuple(actor.color[0, 0, :3].tolist())]
                assert torch.allclose(
                    points - points.mean(-2, keepdim=True), expected, atol=1e-5
                )
                assert float(actor.opacity[0, 0, 0]) == pytest.approx(0.5)
                visible.append(actor)
        assert len(visible) == 2
