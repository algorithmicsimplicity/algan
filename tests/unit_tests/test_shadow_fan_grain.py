"""The budgeted soft-shadow fan's grain holds still under a sub-pixel camera move.

The budgeted fan (``shadow_ray_budget``) fires a few rays per light and rotates
them per shading event, so a penumbra dithers rather than bands; the rotation
is hashed from the event's world position (``_shadow_fan_jitter``). Hashed
from the position's own float bits, it re-rolled at every pixel as soon as the
camera moved at all -- a pixel's hit point slides along the surface by a
fraction of a pixel per frame -- and the penumbra boiled: under a 0.002-degree
turn, far below a pixel of motion, penumbra pixels changed by up to 28 levels
where the unrotated fan's changed by 1. Keyed on the cell of a pixel-sized
world grid that holds the point (``_shadow_jitter_cell``), the rotation stays
put while the point stays in its cell. A cell is one to two pixels across, so
motion of a pixel or more per frame still re-rolls the grain: it does not
move with the surface, and nothing here claims it does.

The probe kernel pins the key; one render pins the penumbra.

Unmarked, so outside ``--fast``: only the shadow fan can break it. Note the
absent ``from __future__ import annotations``: the probe kernel's
``ti.types.ndarray()`` annotations are evaluated at run time.
"""

import torch

from algan import (
    BLACK,
    ORIGIN,
    OUT,
    RIGHT,
    SETTINGS,
    UP,
    WHITE,
    AmbientLight,
    Cube,
    DirectionalLight,
    MeshLambertMaterial,
    Off,
    Scene,
    Seq,
    Surface,
    VideoSettings,
)
from algan.rendering.raytracing.wavefront_kernels_taichi import (
    _shadow_fan_jitter,
    _shadow_jitter_cell,
)
from algan.taichi_compat import ti


@ti.kernel
def _probe(
    points: ti.types.ndarray(),
    steps: ti.types.ndarray(),
    out: ti.types.ndarray(),
):
    for i in range(points.shape[0]):
        p = ti.math.vec3(points[i, 0], points[i, 1], points[i, 2])
        dpx = ti.math.vec3(steps[i, 0], steps[i, 1], steps[i, 2])
        dpy = ti.math.vec3(steps[i, 3], steps[i, 4], steps[i, 5])
        cell = _shadow_jitter_cell(dpx, dpy)
        ju, jv = _shadow_fan_jitter(p, 0, cell)
        out[i, 0] = ju
        out[i, 1] = jv
        out[i, 2] = cell


def _rotations(points, footprint):
    """``(ju, jv, cell)`` per point, every point with a ``footprint``-wide pixel."""
    points = torch.as_tensor(points, dtype=torch.float32).reshape(-1, 3)
    steps = torch.tensor(
        [footprint, 0.0, 0.0, 0.0, footprint, 0.0], dtype=torch.float32
    ).expand(points.shape[0], 6)
    out = torch.zeros((points.shape[0], 3), dtype=torch.float32)
    _probe(points.contiguous(), steps.contiguous(), out)
    return out


def test_the_cell_is_twice_the_largest_power_of_two_within_a_pixel():
    """One to two pixels across: see _shadow_jitter_cell for why."""
    out = _rotations(torch.zeros(4, 3), 0.3)
    assert out[:, 2].tolist() == [0.5] * 4
    assert _rotations(torch.zeros(1, 3), 0.0)[0, 2] == 0.0  # no footprint
    # The shorter of the two steps: a grazing pixel's long side does not
    # merge its neighbours across the short one.
    steps = torch.tensor([[0.0, 0.0, 0.9, 0.0, 0.07, 0.0]])
    out = torch.zeros((1, 3))
    _probe(torch.zeros(1, 3), steps, out)
    assert float(out[0, 2]) == 0.125


def test_a_point_sliding_within_its_cell_keeps_its_rotation():
    """A pixel's hit point under a slow camera move: its low bits change."""
    generator = torch.Generator().manual_seed(0)
    corner = torch.tensor([0.75, -1.25, 2.5])  # a corner of a 0.25 cell
    inside = corner + 0.2499 * torch.rand((64, 3), generator=generator)
    out = _rotations(inside, 0.15)
    assert torch.unique(out[:, :2], dim=0).shape[0] == 1
    # The cells around it dither independently.
    neighbours = corner + 0.25 * torch.tensor(
        [[i, j, 0.0] for i in (-1, 0, 1) for j in (-1, 0, 1)]
    )
    assert torch.unique(_rotations(neighbours, 0.15)[:, :2], dim=0).shape[0] == 9


def test_without_a_footprint_every_point_keeps_its_own_rotation():
    """The classic route and bounce events have no footprint: the bits key."""
    base = torch.tensor([0.75, -1.25, 2.5])
    nudged = torch.stack((base, torch.nextafter(base, base + 1)))
    out = _rotations(nudged, 0.0)
    assert not torch.equal(out[0, :2], out[1, :2])


def _penumbra_frames(turn=0.0, truck=0.0, frames=2, angle=8):
    """A cube's soft shadow on a floor under a moving camera, ``frames`` frames.

    The camera turns ``turn`` degrees over the first frame step, or trucks
    sideways ``truck`` units every frame step; ``angle`` is the light's
    shadow angle.
    """
    SETTINGS.raytracing.set(shadows=True)
    video = VideoSettings(resolution=(320, 180), frames_per_second=10, supersampling=1)
    with Scene(video_settings=video) as scene:
        scene.set_background(BLACK)
        with Off():
            Scene.clear_lights()
            AmbientLight(color=WHITE, intensity=0.15).spawn(animate=False)
            DirectionalLight(
                location=RIGHT * 4 + UP * 5 + OUT * 4,
                target=ORIGIN,
                color=WHITE,
                intensity=0.6,
                shadow_angle=angle,
            ).spawn(animate=False)
            Surface(
                coord_function=lambda uv: torch.cat(
                    (
                        (uv[..., :1] - 0.5) * 8,
                        torch.zeros_like(uv[..., :1]) - 0.6,
                        (uv[..., 1:2] - 0.5) * 8,
                    ),
                    -1,
                ),
                grid_width=2,
                grid_height=2,
                color=WHITE,
            ).set_material(MeshLambertMaterial()).spawn(animate=False)
            Cube(color=WHITE).set_material(MeshLambertMaterial()).scale(0.6).spawn(
                animate=False
            )
            scene.camera.rotate(-30, RIGHT, about=ORIGIN)
            scene.camera.rotate(20, UP, about=ORIGIN)
            if truck:
                # Closer, so the penumbra is wide enough to measure.
                scene.camera.move(scene.camera.get_forward_direction() * 5)
        if truck:
            with Seq(runtime=frames / 10, easing=lambda t: t):
                scene.camera.move(RIGHT * truck * frames)
        else:
            with Seq(runtime=0.1):
                scene.camera.rotate(turn, UP, about=ORIGIN)
        out = torch.cat([f.cpu() for f in scene.get_frames(0, frames)])
    return out.float()[..., :3]


def test_a_penumbra_holds_still_under_a_sub_pixel_camera_turn():
    """Keyed on the point's bits this changed 63-72 pixels by up to 28 levels.

    The cell key cannot hold every pixel still: a sample point lying within
    the turn's displacement of a cell boundary crosses it and re-rolls, and
    its pixel changes by as much as the bits key changed it. Which points lie
    there depends on the hit points' last bits, so on the camera angle and the
    device's float rounding, which is why this counts the pixels that change
    rather than bounding the largest change. Measured on the CPU from 42
    starting angles up to 2.3 degrees past the test's: 0-3 pixels changed by
    more than 2 levels (the largest by 15); with the cell forced to 0, the
    bits key, 63-72 did. The Mac's MPS render changed a pixel by 10 at the
    test's own angle, which the old bound of 2 failed.
    """
    frames = _penumbra_frames(0.002)
    # The shadow is soft and dithered: many levels between the floor's umbra
    # (about 77) and its lit side (about 197), not two.
    floor = frames[0].mean(-1)
    levels = torch.unique(floor[(floor > 90) & (floor < 185)].round())
    assert levels.numel() > 20, levels
    changed = (frames[1] - frames[0]).abs().amax(-1) > 2
    assert int(changed.sum()) <= 8, changed.nonzero().tolist()


def test_a_slow_truck_re_rolls_little_of_the_penumbra(monkeypatch):
    """A camera trucking a tenth of a pixel per frame, against the fixed fan.

    Every penumbra pixel changes a little as the shadow slides across the
    pixel grid, the fixed fan's included; the dither adds to that whatever
    it re-rolls. With jitter cells under a pixel across the dithered
    penumbra changed 2.1 times as much from frame to frame as the fixed
    fan's (1.74 levels on average against 0.84); with cells one to two
    pixels across, 1.4 times (1.28 against 0.93).
    """
    from algan.rendering.raytracing import settings as rt_settings

    monkeypatch.setattr(rt_settings, "shadow_ray_budget", 0)
    fixed = _penumbra_frames(truck=0.005, frames=8, angle=20).mean(-1)
    monkeypatch.setattr(rt_settings, "shadow_ray_budget", 16)
    dithered = _penumbra_frames(truck=0.005, frames=8, angle=20).mean(-1)
    penumbra = ((dithered - fixed).abs() > 0.5).any(0)
    assert int(penumbra.sum()) > 200, "the shadow should have a penumbra"

    def flicker(frames):
        return float((frames[1:] - frames[:-1]).abs()[:, penumbra].mean())

    assert flicker(dithered) < 1.7 * flicker(fixed)
