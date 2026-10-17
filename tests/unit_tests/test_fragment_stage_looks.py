"""The shipped fragment-stage looks: ``fresnel_rim``, ``glass_ball`` and ``cosine_color``.

``fresnel_rim`` and ``glass_ball`` (``fragment_stage_library``) are additive
looks composed after a lit base or over a flat colour; ``cosine_color``
(``fragment_shaders``) recolours the albedo with an RGB-phase-shifted cosine of
world ``x`` before a lighting stage lights it. Each is pinned twice:

* **the formula**, by calling the stage ``@ti.func`` directly from a probe kernel
  and checking it against the expression its docstring gives -- the rim's
  ``rim_color * rim_gain * (1 - |N.V|) ** rim_power``, the glass ball's
  azimuth weighting (mean 1 over azimuth, whatever ``anisotropy`` is), its
  Gaussian ring and its two highlights placed at ``(key_x, key_y)`` and
  ``(fill_x, fill_y)``, and the cosine's three phase-shifted channels;
* **what a user sees**, on one SMOKE_TEST frame of a lit sphere: the rim adds
  its own colour at the silhouette and leaves the middle alone, the glass ball
  puts its key highlight up and to the right and its edge at top and bottom,
  the cosine bands the hue across ``x`` -- and each stage's parameters animate
  like any other attribute, so a recorded ``mob.rim_gain = 1`` moves the frame.

All three render tests draw the same three-sphere scene, with the sphere under
test at the origin and the other two parked out of frame. The renderer
specialises its shade kernels on the set of fragment pipelines a frame carries,
so one composition is one compiled kernel variant rather than three.

A feature test, outside the fast suite: only a change to these stages, or to
how a fragment pipeline reaches the kernel, breaks it.

Note the absent ``from __future__ import annotations``: the probe kernel's
``ti.types.ndarray()`` annotations are evaluated at run time, and stringifying
them stops the kernel compiling.
"""

import math

import cv2
import pytest
import torch
import torch.nn.functional as F

from algan import (
    BLACK,
    RED,
    RIGHT,
    SMOKE_TEST,
    WHITE,
    Off,
    Scene,
    Sphere,
    Sync,
    cosine_color,
    fresnel_rim,
    glass_ball,
    phong_shader,
    standard_shader,
)
from algan.rendering.taichi_runtime import init_taichi
from algan.scene_manager import SceneManager
from algan.taichi_compat import ti
from algan.utils.color_space import srgb_to_linear

# ---------------------------------------------------------------------------
# The stage, probed one fragment per row
# ---------------------------------------------------------------------------


@ti.kernel
def _stage_probe(
    stage: ti.template(),
    pos: ti.types.ndarray(),
    view: ti.types.ndarray(),
    nrm: ti.types.ndarray(),
    rgb_in: ti.types.ndarray(),
    params: ti.types.ndarray(),
    out: ti.types.ndarray(),
):
    for i in range(pos.shape[0]):
        p = ti.math.vec3(pos[i, 0], pos[i, 1], pos[i, 2])
        vd = ti.math.vec3(view[i, 0], view[i, 1], view[i, 2])
        n = ti.math.vec3(nrm[i, 0], nrm[i, 1], nrm[i, 2])
        rgb = ti.math.vec3(rgb_in[i, 0], rgb_in[i, 1], rgb_in[i, 2])
        # No light rows: these stages read none (the lit base is upstream).
        r = stage(
            p,
            vd,
            n,
            n,
            rgb,
            rgb_in[i, 3],
            params,
            0,
            i,
            0,
            params,
            params,
            0,
            0,
            0,
            ti.math.vec3(0.0, 0.0, 0.0),
        )
        for c in ti.static(range(4)):
            out[i, c] = r[c]


def run_stage(stage, normals, *, view=(0.0, 0.0, 1.0), pos=None, rgb=None, **params):
    """Evaluate ``stage`` once per normal and return ``[k, 4]`` (RGB + glow).

    ``params`` override the stage's own defaults by name; ``rgb`` is the
    incoming ``[k, 4]`` colour (a fixed dark teal with glow 0.3 by default) and
    ``view`` the direction toward the viewer, shared by every row unless a
    ``[k, 3]`` tensor is given.
    """
    init_taichi()
    nrm = torch.as_tensor(normals, dtype=torch.float32).reshape(-1, 3)
    k = nrm.shape[0]
    view = torch.as_tensor(view, dtype=torch.float32).reshape(-1, 3).expand(k, 3)
    pos = torch.zeros(k, 3) if pos is None else torch.as_tensor(pos).float()
    if rgb is None:
        rgb = torch.tensor([0.05, 0.2, 0.25, 0.3]).expand(k, 4)
    block = torch.zeros(1, k, max(stage.width, 1))
    slot = 0
    for name, width, default in stage.param_specs:
        value = params.pop(name, default)
        block[0, :, slot : slot + width] = torch.as_tensor(value, dtype=torch.float32)
        slot += width
    assert not params, f"unknown parameters {sorted(params)}"
    out = torch.zeros(k, 4)
    _stage_probe(
        stage.ti_func,
        pos.contiguous(),
        view.contiguous(),
        nrm.contiguous(),
        torch.as_tensor(rgb, dtype=torch.float32).reshape(k, 4).contiguous(),
        block,
        out,
    )
    return out


def _ring(polar_deg, count=720):
    """Unit normals at a fixed angle from +Z (the view axis), all the way round."""
    t = math.radians(polar_deg)
    phi = (torch.arange(count) + 0.5) * (2 * math.pi / count)
    return torch.stack(
        (
            math.sin(t) * torch.cos(phi),
            math.sin(t) * torch.sin(phi),
            torch.full_like(phi, math.cos(t)),
        ),
        -1,
    )


def _normal_at(x, y):
    """The view-facing unit normal whose screen-plane components are ``(x, y)``."""
    return (x, y, math.sqrt(max(0.0, 1.0 - x * x - y * y)))


# ---------------------------------------------------------------------------
# fresnel_rim: the formula
# ---------------------------------------------------------------------------


def test_fresnel_rim_adds_its_documented_term_and_nothing_else():
    """``in + rim_color * rim_gain * (1 - |N.V|) ** rim_power``, glow untouched.

    Includes normals facing away from the viewer, which the ``|N.V|`` makes
    shade like the side facing it.
    """
    gen = torch.Generator().manual_seed(0)
    normals = F.normalize(torch.randn(64, 3, generator=gen), dim=-1)
    rgb = torch.rand(64, 4, generator=gen)
    rim, power, gain = (0.2, 0.5, 0.9), 2.5, 1.7
    got = run_stage(
        fresnel_rim, normals, rgb=rgb, rim_color=rim, rim_power=power, rim_gain=gain
    )
    fres = (1.0 - normals[:, 2].abs()).clamp(0.0, 1.0)
    expected = rgb[:, :3] + torch.tensor(rim) * (gain * fres**power).unsqueeze(-1)
    torch.testing.assert_close(got[:, :3], expected, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(got[:, 3], rgb[:, 3])
    # Front and back faces at the same angle get the same rim.
    torch.testing.assert_close(
        run_stage(fresnel_rim, normals), run_stage(fresnel_rim, -normals)
    )


def test_fresnel_rim_leaves_a_head_on_fragment_alone_and_tightens_with_power():
    """Zero added head-on whatever the power; at a given angle a higher
    ``rim_power`` adds less (1 = a broad wash, 6 = a thin outline); and a
    non-positive power is clamped rather than turned into NaN.
    """
    head_on = run_stage(fresnel_rim, [(0.0, 0.0, 1.0)], rim_power=6.0)
    torch.testing.assert_close(head_on[0, :3], torch.tensor([0.05, 0.2, 0.25]))
    oblique = [_normal_at(math.sin(math.radians(60)), 0.0)]
    added = [
        float(run_stage(fresnel_rim, oblique, rim_power=p)[0, 2] - 0.25)
        for p in (1.0, 3.0, 6.0)
    ]
    assert added[0] > added[1] > added[2] > 0.0
    for power in (0.0, -2.0):
        assert torch.isfinite(run_stage(fresnel_rim, oblique, rim_power=power)).all()


# ---------------------------------------------------------------------------
# glass_ball: the formula
# ---------------------------------------------------------------------------

NO_HIGHLIGHTS = {"key_gain": 0.0, "fill_gain": 0.0}


@pytest.mark.parametrize("polar_deg", [40.0, 70.0, 85.0])
def test_glass_ball_anisotropy_moves_the_edge_without_changing_its_average(polar_deg):
    """The docstring's claim: the ``|sin(azimuth)|`` weight is normalised by
    ``2/pi`` so that turning ``anisotropy`` up redistributes the edge toward
    the top and bottom of the ball but leaves its mean over azimuth alone.
    """
    ring = _ring(polar_deg)
    flat = run_stage(glass_ball, ring, anisotropy=0.0, **NO_HIGHLIGHTS)[:, :3]
    biased = run_stage(glass_ball, ring, anisotropy=0.7, **NO_HIGHLIGHTS)[:, :3]
    base = torch.tensor([0.05, 0.2, 0.25])
    torch.testing.assert_close(
        (biased - base).mean(0), (flat - base).mean(0), rtol=2e-3, atol=1e-6
    )
    # ...and the redistribution itself: top (+y) brighter than the side (+x).
    top = int(ring[:, 1].argmax())
    side = int(ring[:, 0].argmax())
    assert float(biased[top, 2]) > float(biased[side, 2]) + 1e-3
    torch.testing.assert_close(flat[top], flat[side], rtol=1e-4, atol=1e-6)


def test_glass_ball_ring_peaks_at_ring_center():
    """``ring_*`` is a Gaussian band in ``1 - N.V``: brightest where that equals
    ``ring_center``, and ``ring_width`` across.
    """
    black = {
        "rim_color": (0.0, 0.0, 0.0),
        "edge_color": (0.0, 0.0, 0.0),
        "ring_color": (1.0, 1.0, 1.0),
        "anisotropy": 0.0,
        **NO_HIGHLIGHTS,
    }
    fres = torch.linspace(0.3, 0.95, 66)
    normals = [_normal_at(math.sqrt(1.0 - (1.0 - f) ** 2), 0.0) for f in fres.tolist()]
    got = run_stage(glass_ball, normals, rgb=torch.zeros(66, 4), **black)[:, 0]
    peak = float(fres[int(got.argmax())])
    assert peak == pytest.approx(0.66, abs=0.011)
    assert float(got.max()) == pytest.approx(1.0, abs=1e-3)
    torch.testing.assert_close(
        got, torch.exp(-(((fres - 0.66) / 0.11) ** 2)), rtol=1e-4, atol=1e-5
    )


def test_glass_ball_highlights_sit_where_key_and_fill_say():
    """The key streak peaks at ``(key_x, key_y)`` and the fill blob at
    ``(fill_x, fill_y)`` -- in units of the radius, right and up from the
    ball's centre -- at ``key_color * key_gain`` and ``key_color * fill_gain``;
    the streak is long along ``(sin, -cos)(key_angle)`` and narrow across it.
    """
    edges_off = {
        "rim_color": (0.0, 0.0, 0.0),
        "edge_color": (0.0, 0.0, 0.0),
        "ring_color": (0.0, 0.0, 0.0),
    }
    zero = torch.zeros(1, 4)
    key = run_stage(glass_ball, [_normal_at(0.6, 0.42)], rgb=zero, **edges_off)
    torch.testing.assert_close(
        key[0, :3], 0.95 * torch.tensor([1.00, 1.02, 1.04]), rtol=1e-4, atol=1e-5
    )
    fill = run_stage(
        glass_ball, [_normal_at(-0.44, 0.40)], rgb=zero, key_gain=0.0, **edges_off
    )
    torch.testing.assert_close(
        fill[0, :3], 0.22 * torch.tensor([1.00, 1.02, 1.04]), rtol=1e-4, atol=1e-5
    )
    # Along and across the streak, the same distance from its peak.
    angle, step = 0.39, 0.1
    along = (math.sin(angle) * step, -math.cos(angle) * step)
    across = (math.cos(angle) * step, math.sin(angle) * step)
    probes = [_normal_at(0.6 + d[0], 0.42 + d[1]) for d in (along, across)]
    got = run_stage(
        glass_ball, probes, rgb=torch.zeros(2, 4), fill_gain=0.0, **edges_off
    )
    long_axis = 0.95 * math.exp(-((step / 0.24) ** 2))
    short_axis = 0.95 * math.exp(-((step / 0.062) ** 2))
    assert float(got[0, 0]) == pytest.approx(long_axis, rel=2e-3)
    assert float(got[1, 0]) == pytest.approx(short_axis, rel=2e-3)
    assert long_axis > 10 * short_axis


def test_glass_ball_is_additive_and_turns_a_back_face_toward_the_viewer():
    """What it adds does not depend on the colour it is handed, glow passes
    through, and a normal facing away shades as its viewer-facing twin -- so
    the highlights keep their side of the ball.
    """
    gen = torch.Generator().manual_seed(1)
    normals = F.normalize(torch.randn(48, 3, generator=gen), dim=-1)
    normals[:, 2] = normals[:, 2].abs() + 0.05
    normals = F.normalize(normals, dim=-1)
    rgb_a = torch.rand(48, 4, generator=gen)
    rgb_b = torch.rand(48, 4, generator=gen)
    out_a = run_stage(glass_ball, normals, rgb=rgb_a)
    out_b = run_stage(glass_ball, normals, rgb=rgb_b)
    torch.testing.assert_close(
        out_a[:, :3] - rgb_a[:, :3], out_b[:, :3] - rgb_b[:, :3], rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(out_a[:, 3], rgb_a[:, 3])
    torch.testing.assert_close(run_stage(glass_ball, -normals, rgb=rgb_a), out_a)


# ---------------------------------------------------------------------------
# cosine_color: the formula
# ---------------------------------------------------------------------------


def test_cosine_color_is_three_phase_shifted_cosines_of_world_x():
    """``in_rgb * (0.5 + 0.5 cos(x f + p + k 2pi/3))`` for ``k = 0, 1, 2``.

    Only world ``x`` matters: ``y``, ``z``, the normal and the view direction
    do not enter, and glow passes through.
    """
    gen = torch.Generator().manual_seed(2)
    pos = torch.randn(64, 3, generator=gen) * 3.0
    normals = F.normalize(torch.randn(64, 3, generator=gen), dim=-1)
    rgb = torch.rand(64, 4, generator=gen)
    freq, phase = 1.7, 0.6
    got = run_stage(
        cosine_color, normals, pos=pos, rgb=rgb, frequency=freq, phase=phase
    )
    w = pos[:, 0:1] * freq + phase + torch.tensor([0.0, 1.0, 2.0]) * (2 * math.pi / 3)
    expected = rgb[:, :3] * (0.5 + 0.5 * torch.cos(w))
    # A GPU kernel's fast-math cos is good to about 1e-4 (the Mac arm runs
    # this on Metal); a wrong phase or frequency is off by tenths.
    torch.testing.assert_close(got[:, :3], expected, rtol=1e-4, atol=2e-4)
    torch.testing.assert_close(got[:, 3], rgb[:, 3])
    moved = pos.clone()
    moved[:, 1:] = torch.randn(64, 2, generator=gen)
    other = run_stage(
        cosine_color,
        torch.flip(normals, (0,)),
        pos=moved,
        rgb=rgb,
        view=F.normalize(torch.randn(64, 3, generator=gen), dim=-1),
        frequency=freq,
        phase=phase,
    )
    torch.testing.assert_close(other, got)


def test_cosine_color_defaults_are_its_documented_ones():
    """``frequency`` 4 and ``phase`` 0: white at the origin turns red, with
    green and blue at a quarter each.
    """
    got = run_stage(cosine_color, [(0.0, 0.0, 1.0)], rgb=torch.ones(1, 4))
    torch.testing.assert_close(
        got[0], torch.tensor([1.0, 0.25, 0.25, 1.0]), rtol=0, atol=2e-4
    )
    assert [(name, d) for name, _w, d in cosine_color.param_specs] == [
        ("frequency", 4.0),
        ("phase", 0.0),
    ]


# ---------------------------------------------------------------------------
# What a user sees: one scene, three looks
# ---------------------------------------------------------------------------

RADIUS = 3.0


def _render_looks(tmp_path, front, at, animate=None, setup=None):
    """Render the three-sphere scene with ``front`` at the origin.

    ``setup(spheres)`` runs before spawning (initial attribute values),
    ``animate(spheres)`` after it (recorded animations). Returns the stills at
    ``at`` as linear-light ``[H, W, 3]`` tensors.
    """
    SceneManager.reset()
    try:
        with Scene(video_settings=SMOKE_TEST) as scene:
            with Off():
                spheres = {
                    "rim": Sphere(radius=RADIUS, color=RED),
                    "glass": Sphere(radius=RADIUS, color=BLACK),
                    "cosine": Sphere(radius=RADIUS, color=WHITE),
                }
                spheres["rim"].set_fragment_shader([standard_shader, fresnel_rim])
                spheres["glass"].set_fragment_shader(glass_ball)
                spheres["cosine"].set_fragment_shader([cosine_color, phong_shader])
                if setup is not None:
                    setup(spheres)
                for k, (name, sphere) in enumerate(spheres.items()):
                    if name != front:
                        sphere.move(RIGHT * (60.0 + 20.0 * k))  # out of frame
                    sphere.spawn(animate=False)
            if animate is not None:
                animate(spheres)
            results = scene.save_frame(
                tmp_path / f"{front}.png",
                at=list(at),
                overwrite=True,
                post_processes=(),
            )
    finally:
        SceneManager.reset()
    frames = []
    for result in results:
        img = cv2.imread(str(result.output_path), cv2.IMREAD_UNCHANGED)
        rgb = torch.from_numpy(img[..., ::-1].copy()).float() / 255.0
        frames.append(srgb_to_linear(rgb))
    return frames


def _disc(frame):
    """The sphere's pixels, its centre (row, col) and radius in pixels, and
    every pixel's distance from that centre in radii.
    """
    covered = frame.amax(-1) > 0.0  # the background is black
    rows, cols = torch.nonzero(covered, as_tuple=True)
    cy, cx = rows.float().mean(), cols.float().mean()
    radius = math.sqrt(float(covered.sum()) / math.pi)
    yy, xx = torch.meshgrid(
        torch.arange(frame.shape[0]).float(),
        torch.arange(frame.shape[1]).float(),
        indexing="ij",
    )
    r = ((yy - cy) ** 2 + (xx - cx) ** 2).sqrt() / radius
    return covered, float(cy), float(cx), radius, r


def test_rendered_rim_adds_its_colour_at_the_silhouette_and_animates(tmp_path):
    """A red, standard-lit ball with a pure-blue rim whose gain animates 0 -> 1.

    At gain 0 the frame is the lit ball alone. What the rim then adds is
    blue and only blue -- it is added over the red base, not multiplied into
    it -- nothing in the middle of the ball, more toward the silhouette, and
    at the animation's midpoint half of what it adds at the end.
    """

    def setup(spheres):
        spheres["rim"].rim_color = (0.0, 0.0, 1.0)
        spheres["rim"].rim_gain = 0.0

    def animate(spheres):
        spheres["rim"].rim_gain = 1.0

    start, mid, end = _render_looks(tmp_path, "rim", (0.0, 0.5, 1.0), animate, setup)
    covered, _cy, _cx, radius, r = _disc(start)
    assert radius > 10.0, "the ball should fill most of the frame"
    assert float(start[covered][:, 0].mean()) > 3 * float(start[covered][:, 2].mean())

    added = end - start
    centre = covered & (r < 0.35)
    inner = covered & (r >= 0.55) & (r < 0.7)
    outer = covered & (r >= 0.7) & (r < 0.85)
    assert float(added[centre].abs().max()) < 0.01, "the middle must be left alone"
    assert float(added[outer][:, 2].mean()) > 0.03, "no rim at the silhouette"
    assert float(added[outer][:, 2].mean()) > 1.5 * float(added[inner][:, 2].mean())
    ring = covered & (r >= 0.55) & (r < 0.85)
    assert float(added[ring][:, :2].abs().max()) < 0.01, (
        "the rim changed red or green: it must add its own colour, not scale the base's"
    )
    # The rim is linear in its gain, and the default rate function is
    # symmetric about the animation's midpoint.
    halfway = float((mid - start)[ring][:, 2].sum() / added[ring][:, 2].sum())
    assert halfway == pytest.approx(0.5, abs=0.02)


def test_rendered_glass_ball_places_its_highlights_and_edge(tmp_path):
    """The glass ball over a black body: the key highlight sits up and to the
    right of centre, the middle stays dark, the edge is brighter at the top
    and bottom than at the sides, and moving ``key_x``/``key_y`` -- animated
    like any attribute -- moves the highlight to the lower left.
    """

    def animate(spheres):
        with Sync():
            spheres["glass"].key_x = -0.5
            spheres["glass"].key_y = -0.45

    before, after = _render_looks(tmp_path, "glass", (0.0, 1.0), animate)
    covered, cy, cx, radius, r = _disc(before)

    def brightest(frame):
        """Where the brightest red pixel is, in radii right of and above centre."""
        red = torch.where(covered, frame[..., 0], torch.zeros_like(frame[..., 0]))
        row, col = divmod(int(red.argmax()), frame.shape[1])
        return (col - cx) / radius, (cy - row) / radius

    right, up = brightest(before)
    assert right > 0.3, f"key highlight at {right:+.2f}, {up:+.2f}"
    assert up > 0.15, f"key highlight at {right:+.2f}, {up:+.2f}"
    right, up = brightest(after)
    assert right < -0.3, f"moved key at {right:+.2f}, {up:+.2f}"
    assert up < -0.2, f"moved key at {right:+.2f}, {up:+.2f}"

    blue = before[..., 2]
    centre = covered & (r < 0.3)
    assert float(blue[centre].mean()) < 0.3 * float(blue[covered & (r > 0.8)].max())

    yy, xx = torch.meshgrid(
        torch.arange(blue.shape[0]).float(),
        torch.arange(blue.shape[1]).float(),
        indexing="ij",
    )
    band = covered & (r >= 0.75) & (r < 0.92)
    vertical = band & ((xx - cx).abs() < 0.25 * radius)
    horizontal = band & ((yy - cy).abs() < 0.25 * radius)
    assert float(blue[vertical].mean()) > 1.4 * float(blue[horizontal].mean())


def test_rendered_cosine_color_bands_the_hue_across_x_and_animates(tmp_path):
    """A white, Phong-lit ball recoloured at one cosine period over its width:
    red in the middle, green to the left and blue to the right of it, and a
    ``phase`` animated to pi turns every channel into its complement -- the
    middle from red to cyan, and green and blue gone from where they peaked.

    The middle is judged by how far one channel stands above the others, not
    by its share of the total: Phong's highlight sits near the middle and adds
    the same white to every channel.
    """

    def setup(spheres):
        spheres["cosine"].frequency = 1.0

    def animate(spheres):
        spheres["cosine"].phase = math.pi

    before, after = _render_looks(tmp_path, "cosine", (0.0, 1.0), animate, setup)
    covered, cy, cx, radius, _r = _disc(before)
    row = int(round(cy))
    cols = [c for c in range(before.shape[1]) if covered[row, c]]
    assert len(cols) > 20

    def share(frame, channel, col):
        px = frame[row, col]
        return float(px[channel] / px.sum().clamp_min(1e-6))

    middle = before[row, int(round(cx))]
    assert float(middle[0] - middle[1:].max()) > 0.3, f"middle not red: {middle}"
    green_col = max(cols, key=lambda c: share(before, 1, c))
    blue_col = max(cols, key=lambda c: share(before, 2, c))
    assert green_col < cx - 0.3 * radius, "green should peak left of centre"
    assert blue_col > cx + 0.3 * radius, "blue should peak right of centre"

    middle = after[row, int(round(cx))]
    assert float(middle[1:].min() - middle[0]) > 0.3, f"middle not cyan: {middle}"
    assert share(after, 1, green_col) < 0.1, "phase pi should take the green out"
    assert share(after, 2, blue_col) < 0.1, "phase pi should take the blue out"
