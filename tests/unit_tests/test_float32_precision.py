"""Float32 rounding that shows on screen is reported, once per render.

Algan stores and renders positions as float32 in world space, so a point is
only as exact as its largest coordinate allows, and a camera magnifies that by
its focal length over the point's depth. Text brought a unit in front of a
camera a thousand units out, turned 35 degrees, rendered with doubled and
dashed strokes; so did text at the origin seen through the old
``set_near_orthographic`` default (1e5), turned the same way.

The render estimates, after projecting each batch, how far rounding can move
every visible point on screen (``algan.rendering.raytracing.float32_rounding``)
and warns with :class:`~algan.errors.Float32PrecisionWarning` once per render,
naming the worst Mob, from half a pixel -- the level at which stroke edges
visibly step in the calibration renders.

The estimate's pieces are checked against what they model: the ray-direction
term against a float32/float64 emulation of the kernels' ``_generate_ray``,
the chunked reduction against a brute-force one, the CPU early-out against the
full pass. Then real renders: the two cases that broke warn (and name the Mob
that broke), and scenes that are fine -- a far planet, a huge backdrop, a HUD a
sensible distance out -- do not.
"""

from __future__ import annotations

import math
import random
import warnings

import pytest
import torch

from algan import (
    BLUE,
    DARK_GRAY,
    ORIGIN,
    RED,
    RIGHT,
    SETTINGS,
    UP,
    WHITE,
    YELLOW,
    Circle,
    Cube,
    Off,
    Rectangle,
    Scene,
    Seq,
    Sphere,
    Square,
    Text,
)
from algan.errors import Float32PrecisionWarning
from algan.rendering.camera import DEFAULT_NEAR_ORTHOGRAPHIC_DISTANCE
from algan.rendering.raytracing import float32_rounding as fr
from algan.settings.video_settings import LD


class _Snapshot:
    """The camera fields a projection reads, as the render loop's shim has them."""

    def __init__(self, state, resolution):
        self.ray_origin = state["ray_origin"]
        self.screen_point = state["screen_point"]
        self.screen_basis = state["screen_basis"]
        self.screen_width = self.output_screen_width = resolution[0]
        self.screen_height = self.output_screen_height = resolution[1]


def _camera(distance=None, turn=0.0, pitch=0.0, resolution=(1280, 720)):
    """A camera snapshot, near-orthographic at ``distance`` and turned.

    ``distance`` None keeps the default perspective camera; ``turn`` and
    ``pitch`` rotate it about the origin, in degrees.
    """
    with Scene(LD.set(resolution=resolution)) as scene:
        camera = scene.get_camera()
        with Off():
            if distance is not None:
                camera.set_near_orthographic(distance)
            camera.rotate(turn, UP, about=ORIGIN)
            camera.rotate(pitch, RIGHT, about=ORIGIN)
        state = scene._materialize_render_state(0, 1)
    return _Snapshot(state, resolution)


def _render_warnings(scene_body, tmp_path, resolution=(320, 180), *, video=False):
    """Run ``scene_body(scene)``, render it, and return its float32 warnings."""
    settings = LD.set(resolution=resolution, frames_per_second=4)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with Scene(settings) as scene:
            scene_body(scene)
            if video:
                scene.save_video(str(tmp_path / "clip.mp4"), animate_fade_out=False)
            else:
                scene.save_frame(str(tmp_path / "frame.png"))
    return [w for w in caught if issubclass(w.category, Float32PrecisionWarning)]


def _bring_forward(camera, mob, depth):
    """Slide ``mob`` along the camera's rays to ``depth`` in front of the eye.

    The picture is unchanged in exact arithmetic: this is how the
    backpropagation video keeps a label panel on screen while its camera turns.
    """
    eye = camera.location.reshape(3)
    forward = camera.forward.reshape(3)
    current = float(((mob.get_center().reshape(3) - eye) * forward).sum())
    k = depth / current
    mob.scale(k)
    mob.move_to(eye + (mob.location.reshape(3) - eye) * k)


# ---------------------------------------------------------------------------
# The estimate
# ---------------------------------------------------------------------------


def _ray_direction_error_px(snapshot):
    """Worst on-screen error of the kernels' primary rays in float32.

    ``_generate_ray`` forms ``screen_point + u * dx + v * dy`` in world space,
    subtracts the eye and normalizes; this runs that in float32 and float64 on
    a grid of pixels and measures the angle between the two, in pixels.
    """
    width, height = snapshot.output_screen_width, snapshot.output_screen_height
    eye = snapshot.ray_origin.reshape(3).double()
    screen = snapshot.screen_point.reshape(3).double()
    basis = snapshot.screen_basis.reshape(3, 3).double()
    dual = torch.linalg.inv(basis)
    ys, xs = torch.meshgrid(
        torch.arange(0, height, 5.0), torch.arange(0, width, 5.0), indexing="ij"
    )
    u = ((xs + 0.5) - width / 2) / (height / 2)
    v = ((ys + 0.5) - height / 2) / (height / 2)

    def rays(dtype):
        o, s, dx, dy = (t.to(dtype) for t in (eye, screen, dual[:, 0], dual[:, 1]))
        point = s + u.to(dtype)[..., None] * dx + v.to(dtype)[..., None] * dy
        direction = point - o
        return direction / direction.norm(dim=-1, keepdim=True)

    exact = rays(torch.float64)
    error = rays(torch.float32).double() - exact
    across = error - (error * exact).sum(-1, keepdim=True) * exact
    forward = basis[2] / basis[2].norm()
    focal = (height / 2) * basis[0].norm() * ((screen - eye) @ forward)
    return float(across.norm(dim=-1).max() * focal)


@pytest.mark.parametrize(
    ("distance", "turn"), [(1e5, 35.0), (1e5, 45.0), (3e4, 35.0), (1e4, 45.0)]
)
def test_the_ray_term_matches_float32_ray_generation(distance, turn):
    snapshot = _camera(distance, turn)
    terms = fr.camera_rounding(snapshot, 1, torch.device("cpu"))
    estimated = fr.EPS32 * math.sqrt(float(terms.ray_sq[0]))
    measured = _ray_direction_error_px(snapshot)
    # The estimate is the spacing of representable directions, not an
    # expectation; the worst pixel's measured error is 0.7-1.1x of it.
    assert 0.5 < measured / estimated < 2.0, (measured, estimated)


def test_an_axis_aligned_camera_far_out_loses_nothing_across_its_view():
    snapshot = _camera(1e5)
    terms = fr.camera_rounding(snapshot, 1, torch.device("cpu"))
    assert fr.EPS32 * math.sqrt(float(terms.ray_sq[0])) < 1e-3
    assert _ray_direction_error_px(snapshot) < 1e-3
    # Something a unit in front of that eye, 160,000 units out: its large
    # coordinate lies along the view, so it rounds by nothing on screen.
    eye = snapshot.ray_origin.reshape(3)
    hud = eye + torch.tensor(
        [[0.01 * x, 0.01 * y, -1.0] for x in (-1, 1) for y in (-1, 1)]
    )
    values, _ = fr.worst_point_rounding(hud[None], terms, skip_below=0)
    assert float(values[0]) < 1e-2


def _brute_force(points, terms):
    best, at = 0.0, None
    for t in range(terms.eye.shape[0]):
        eye = terms.eye[t].double()
        forward = terms.rows[t, 0, :3].double()
        screen_rows = terms.rows[t, 1:3, :3].double()
        for m in range(points.shape[1]):
            p = points[0 if points.shape[0] == 1 else t, m].double()
            depth = float((p - eye) @ forward)
            if depth <= 0:
                continue
            u, v = terms.centre[t].double() * depth + terms.screen_distance[
                t
            ].double() * (screen_rows @ (p - eye))
            if (
                abs(float(u)) > terms.limit_u * depth
                or abs(float(v)) > terms.limit_v * depth
            ):
                continue
            across = float(((p.square() + eye.square()) * (1 - forward.square())).sum())
            score = float(terms.focal_sq[t]) * across / depth**2 + float(
                terms.ray_sq[t]
            )
            if score > best:
                best, at = score, (t, m)
    return fr.EPS32 * math.sqrt(best), at


@pytest.mark.parametrize("static", [True, False])
def test_the_chunked_reduction_finds_the_brute_force_worst(static, monkeypatch):
    frames = 5
    with Scene(LD.set(resolution=(640, 360), frames_per_second=frames)) as scene:
        camera = scene.get_camera()
        with Off():
            camera.set_near_orthographic(300)
        with Seq(runtime=1):
            camera.rotate(40, UP, about=ORIGIN)
        scene.timeline_manager.set_state_to_times(torch.arange(frames) / frames + 0.01)
        snapshot = _Snapshot(scene._materialize_render_state(0, frames), (640, 360))
        scene.timeline_manager.clear_buffers()
    terms = fr.camera_rounding(snapshot, frames, torch.device("cpu"))
    generator = torch.Generator().manual_seed(0)
    count = 1 if static else frames
    points = torch.randn(count, 120, 3, generator=generator) * 3
    # Half of them a small panel just in front of the last frame's eye.
    points[:, 60:] = points[:, 60:] * 0.01 + terms.eye[-1] * 0.995
    expected, at = _brute_force(points, terms)
    for chunk in (7, 64, 119, 1 << 20):
        monkeypatch.setattr(fr, "_CHUNK_ELEMENTS", chunk)
        values, index = fr.worst_point_rounding(points, terms, skip_below=0)
        assert tuple(index.tolist()) == at
        assert float(values[0]) == pytest.approx(expected, rel=1e-4)


def test_the_cpu_early_out_never_skips_points_that_reach_the_threshold():
    rng = random.Random(1)
    generator = torch.Generator().manual_seed(1)
    reached = skipped = 0
    for _ in range(80):
        snapshot = _camera(
            rng.choice([None, 50, 300, 1000, 5000, 3e4, 1e5]),
            rng.uniform(-60, 60),
            rng.uniform(-40, 40),
        )
        terms = fr.camera_rounding(snapshot, 1, torch.device("cpu"))
        eye = snapshot.ray_origin.reshape(3)
        depth = 10 ** rng.uniform(-3, 4)
        spread = depth * rng.uniform(0.01, 1.0)
        centre = eye + terms.rows[0, 0, :3] * depth
        points = centre + torch.randn(1, 100, 3, generator=generator) * spread * 0.3
        full, _ = fr.worst_point_rounding(points, terms, skip_below=0)
        for threshold in (0.05, 0.5, 2.0):
            may = fr._may_reach(points, terms, threshold)
            if float(full[0]) >= threshold:
                reached += 1
                assert may, (float(full[0]), threshold)
            elif not may:
                skipped += 1
    # The draw is meant to exercise both outcomes.
    assert reached > 10
    assert skipped > 20


def test_the_default_near_orthographic_distance_keeps_both_errors_below_a_pixel():
    """The trade ``DEFAULT_NEAR_ORTHOGRAPHIC_DISTANCE`` documents."""
    assert DEFAULT_NEAR_ORTHOGRAPHIC_DISTANCE == 5e3
    worst = 0.0
    for turn in (20, 35, 45):
        for resolution in ((1920, 1080), (3840, 2160)):
            snapshot = _camera(DEFAULT_NEAR_ORTHOGRAPHIC_DISTANCE, turn, 10, resolution)
            terms = fr.camera_rounding(snapshot, 1, torch.device("cpu"))
            worst = max(worst, fr.EPS32 * math.sqrt(float(terms.ray_sq[0])))
    assert worst < 0.5 * fr.VISIBLE_ERROR_PX
    # Residual perspective: a point 5 units in front of the origin plane, at
    # a corner of a 1080p frame, against where parallel projection puts it.
    snapshot = _camera(DEFAULT_NEAR_ORTHOGRAPHIC_DISTANCE, resolution=(1920, 1080))
    eye = snapshot.ray_origin.reshape(3).double()
    screen = snapshot.screen_point.reshape(3).double()
    basis = snapshot.screen_basis.reshape(3, 3).double()
    forward = basis[2] / basis[2].norm()

    def pixel(p):
        p = torch.tensor(p, dtype=torch.float64)
        projected = eye + ((screen - eye) @ forward) / ((p - eye) @ forward) * (p - eye)
        return (basis[:2] @ (projected - screen)) * 540

    corner = (4.0 * 16 / 9, 4.0)
    shift = (pixel((*corner, 5.0)) - pixel((*corner, 0.0))).norm()
    assert 0.3 < float(shift) < 1.0


def test_the_message_gives_advice_that_works_in_float32():
    hud = fr.Float32Finding(10.0, None, 999.0, 1.0, 1000.0, 0.25)
    far_camera = fr.Float32Finding(0.9, None, 3.0, 8e4, 8e4, 1.0)
    for finding in (hud, far_camera):
        message = fr.precision_message(finding)
        lowered = message.lower()
        assert "float64" not in lowered
        assert "double precision" not in lowered
        assert "child" not in lowered
        assert "parent" not in lowered
        assert "turn the scene instead of the camera" in message
    assert "look broken" in fr.precision_message(hud)
    assert "deeper in front of the camera" in fr.precision_message(hud)
    assert "look grainy" in fr.precision_message(far_camera)
    assert "set_near_orthographic() at its default distance" in fr.precision_message(
        far_camera
    )


# ---------------------------------------------------------------------------
# Renders
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_merged_collections_name_the_mob_that_built_each_element():
    """The owner tables the warning names a Mob through survive batching.

    The batch builder merges Mobs into collections in four ways -- the
    batched circuit build (one circuit per Mob), circuit collections, flat
    triangle collections (a ``Cube`` is one member per triangle) and logical
    PN collections (counted in patches) -- and each must say which Mob built
    each element the projection's record can name. Nothing fails when one
    does not: the warning just names "a Mob". Each Mob sits at its own x, so an
    element's centroid says who built it. Batch preparation only, no render.
    """
    with Scene(LD.set(resolution=(320, 180))) as scene:
        with Off():
            mobs = {
                -6.0: Square(size=1),
                -3.0: Cube(size=1),
                0.0: Sphere(radius=0.5),
                3.0: Circle(radius=0.5),
                6.0: Text("ab"),
            }
            for x, mob in mobs.items():
                mob.move_to(x * RIGHT).spawn(animate=False)
        actors = [scene.camera, scene.camera.screen, *scene.light_sources]
        collections, _, _ = scene._get_batch_of_primitives(
            0, 1, actors + list(scene.actors), 1 << 30
        )
        try:
            xs = torch.tensor(list(mobs))
            kinds = set()
            for collection in collections:
                kinds.add(type(collection).__name__)
                centroids = collection.corners.float().mean(-2)[0, :, 0]
                counts = getattr(collection, "num_segments_per_object", None)
                if counts is not None:
                    counts = counts.reshape(-1).long()
                    circuit = torch.repeat_interleave(torch.arange(len(counts)), counts)
                    centroids = (
                        torch.zeros(len(counts)).index_add_(0, circuit, centroids)
                        / counts
                    )
                for element, x in enumerate(centroids.tolist()):
                    owner = fr._owner_of(collection, element)
                    root = owner
                    while root is not None and root.parents:
                        root = root.parents[0]
                    expected = mobs[float(xs[(xs - x).abs().argmin()])]
                    assert root is expected, (type(collection).__name__, element)
        finally:
            scene.timeline_manager.clear_buffers()
    assert kinds >= {
        "RayTracedBezierCircuitPrimitive",
        "RayTracedTrianglePrimitive",
        "LogicalPNTrianglePrimitive",
    }


def test_a_hud_in_front_of_a_turned_distant_camera_warns_once_and_names_it(tmp_path):
    """The backprop video's label panel, a thousand units out.

    The record rides each projection, the batch read and the per-render scope
    live in ``render_loop``, and the Mob is named through the owner tables.
    Four frames in four batches, so a per-batch warning would show up as four.
    """

    def body(scene):
        camera = scene.get_camera()
        with Off():
            camera.set_near_orthographic(distance=625)  # the eye 1000 units out
            camera.rotate(35, UP, about=ORIGIN)
            # Squares merge into one collection: the HUD is its third member.
            squares = [Square(color=BLUE).move_to(x * RIGHT + UP) for x in (-3.0, 3.0)]
            squares.append(Square(color=WHITE, name="hud"))
            squares.append(Square(color=RED).move_to(3 * RIGHT))
            for square in squares:
                square.rotate(35, UP, about=ORIGIN)
            _bring_forward(camera, squares[2], 1.0)
            for square in squares:
                square.spawn(animate=False)
        scene.wait(1.0)

    snapshot = SETTINGS.snapshot()
    SETTINGS.computing.set(max_animation_batch_size=1)
    try:
        caught = _render_warnings(body, tmp_path, video=True)
    finally:
        SETTINGS.restore(snapshot)
    assert len(caught) == 1, [str(w.message) for w in caught]
    message = str(caught[0].message)
    assert "Square 'hud'" in message
    assert "look broken" in message
    assert "1 unit in front of a camera 1000 units from the origin" in message
    # Pointed at the user's line, not at Algan's internals.
    assert caught[0].filename == __file__


def test_a_very_distant_near_orthographic_camera_turned_warns(tmp_path):
    def body(scene):
        camera = scene.get_camera()
        with Off():
            camera.set_near_orthographic(distance=1e5)
            camera.rotate(35, UP, about=ORIGIN)
            Square(color=BLUE, name="plain").spawn(animate=False)

    caught = _render_warnings(body, tmp_path, resolution=(640, 360))
    assert len(caught) == 1
    message = str(caught[0].message)
    assert "Square 'plain'" in message
    assert "look grainy" in message
    assert "set_near_orthographic() at its default distance" in message


def test_scenes_that_round_finely_do_not_warn(tmp_path):
    """Ordinary scenes, rounding far below a pixel, stay quiet.

    A far planet, a huge backdrop, a tiny far dot, a ground plane under the
    eye, a square a tenth of a unit from it, at 1280x720 with the camera
    turning; and a HUD a unit in front of a turned camera 128 units out
    (0.16 px at 720p).
    """

    def body(scene):
        camera = scene.get_camera()
        with Off():
            Sphere(radius=400, color=BLUE).move_to(
                torch.tensor([3000.0, 1500.0, -8000.0])
            ).spawn(animate=False)
            Rectangle(width=2e4, height=2e4, color=DARK_GRAY, stroke_width=0).move_to(
                torch.tensor([0.0, 0.0, -9000.0])
            ).spawn(animate=False)
            Circle(radius=1e-3, color=RED, stroke_width=0).move_to(
                torch.tensor([0.0, 0.0, -1000.0])
            ).spawn(animate=False)
            floor = Rectangle(width=1e4, height=1e4, color=WHITE, stroke_width=0)
            floor.rotate(90, RIGHT).move_to(torch.tensor([0.0, -3.0, 0.0]))
            floor.spawn(animate=False)
            Square(size=0.02, color=YELLOW).move_to(
                torch.tensor([0.3, 0.1, 19.9])
            ).spawn(animate=False)
            Cube(color=WHITE).move_to(2 * UP).spawn(animate=False)
        with Seq(runtime=1):
            camera.rotate(20, UP, about=ORIGIN)

    assert _render_warnings(body, tmp_path, resolution=(1280, 720)) == []

    def hud_body(scene):
        camera = scene.get_camera()
        with Off():
            camera.set_near_orthographic(distance=80)
            camera.rotate(35, UP, about=ORIGIN)
            hud = Square(color=WHITE).rotate(35, UP, about=ORIGIN)
            _bring_forward(camera, hud, 1.0)
            hud.spawn(animate=False)

    assert _render_warnings(hud_body, tmp_path, resolution=(1280, 720)) == []
