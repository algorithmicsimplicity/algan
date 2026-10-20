"""Overlapping flat shapes a distant camera cannot keep in order are reported.

Every depth a camera computes is rounded at the eye's distance from the
origin, and a camera pulled back for a flat look (``distant_camera.pull_back``,
what ``set_near_orthographic`` did before Algan had an orthographic camera)
puts the eye 1.6 times its distance out.
Two flat shapes that overlap on screen therefore trade places when their
depths differ by less than a few float32 spacings of that: with the eye
16,000 units out, two squares riding the camera swapped across their whole
overlap from frame to frame at depth gaps of 0 and 0.001 and held their order
from 0.003. Near the origin the same shapes tie, their depths closer than the
renderer's coplanar tie gap, and draw order decides, so stacking layers at
one depth is ordinary there -- and the backpropagation video's network panel,
stacked that way under a camera 10,000 units out, blinked. Fixed in the world
instead, the squares keep their order however far out the camera is: their
depths come out bit-identical, a tie, and a tie is draw order's.

When the eye is far enough out for such pairs to exist, the raster pipeline
scans its sorted fragment stream for them
(``float32_rounding.depth_order_record``) and the render warns with
:class:`~algan.errors.Float32PrecisionWarning`, naming both Mobs, once. The
scan is checked against a brute-force one, then real renders: the stacked
squares warn and say when; the same squares far enough apart, fixed in the
world, under the default camera, under the orthographic camera that gives the
same flat look from nearby, or drawn by one Mob do not.

Unmarked, so outside ``--fast``: only the check itself can break it.
"""

from __future__ import annotations

import math
import re
import warnings

import pytest
import torch

from algan import (
    BLACK,
    BLUE,
    ORIGIN,
    OUT,
    RED,
    RIGHT,
    SETTINGS,
    UP,
    Group,
    Off,
    Scene,
    Square,
    VideoSettings,
)
from algan.errors import Float32PrecisionWarning
from algan.mobs.bezier_circuit import BezierCircuitCubic
from algan.rendering.raytracing import float32_rounding as fr
from algan.rendering.raytracing import raster_pipeline
from tests.unit_tests.distant_camera import pull_back

VIDEO = VideoSettings(resolution=(160, 90), frames_per_second=10, supersampling=1)
#: ``pull_back(camera, 10000)``: the eye 16,000 units out, where depths are
#: told apart from 0.0076 units.
FAR = 10000.0


# ---------------------------------------------------------------------------
# The scan
# ---------------------------------------------------------------------------


def _draws(shows, frame, circuit, share):
    """Whether a fragment draws its circuit's fill or border at ``frame``.

    ``share`` is its pixel's share in the border band, 0 to 255.
    """

    def at(table):
        return bool(table[frame % table.shape[0], circuit])

    fill_texels, filled, border_texels, bordered = shows
    fill = share < 255 and at(fill_texels) and at(filled)
    border = share > 0 and at(border_texels) and at(bordered)
    return fill or border


def _brute_force(key, ref, cov, owner_ids, shows, eye, frame_start, ppf, bits, full):
    """``depth_order_record`` one adjacent pair at a time, in plain Python."""
    depth = (key & 0xFFFFFFFF).to(torch.int32).view(torch.float32)
    last = int(owner_ids.shape[0]) - 1
    count, best = 0, None
    for i in range(int(key.shape[0]) - 1):
        if int(key[i]) >> 32 != int(key[i + 1]) >> 32:
            continue
        if int(ref[i]) >= 0 or int(ref[i + 1]) >= 0:
            continue
        if float(cov[i]) < full or float(cov[i + 1]) < full:
            continue
        a = min((-int(ref[i]) - 1) >> bits, last)
        b = min((-int(ref[i + 1]) - 1) >> bits, last)
        share_a = (-int(ref[i]) - 1) & 255
        share_b = (-int(ref[i + 1]) - 1) & 255
        frame = frame_start + (int(key[i]) >> 32) // ppf
        if not (_draws(shows, frame, a, share_a) and _draws(shows, frame, b, share_b)):
            continue
        owner_a, owner_b = int(owner_ids[a]), int(owner_ids[b])
        if owner_a < 0 or owner_b < 0 or owner_a == owner_b:
            continue
        # The same float32 arithmetic as the scan, one pair at a time.
        gap = (depth[i + 1] - depth[i]).abs()
        resolution = (fr.DEPTH_ORDER_SPACINGS * fr.EPS32) * torch.clamp_min(
            torch.maximum(depth[i + 1].abs(), depth[i].abs()), eye
        )
        if not bool(gap < resolution) or float(gap) == 0.0:
            continue
        count += 1
        ratio = float(gap / resolution)
        if best is None or ratio < best[0]:
            pixel = int(key[i + 1]) >> 32
            best = (ratio, a, b, frame_start + pixel // ppf, float(depth[i + 1]))
    return count, best


def _stream(seed, n=3000, pixels=400, ppf=100):
    """A sorted fragment stream with every kind of neighbour the scan meets.

    Circuits and triangles, whole and partial pixels, shapes that draw and
    shapes that do not, one owner and two, and gaps from zero to well past the
    resolution.
    """
    generator = torch.Generator().manual_seed(seed)
    pixel = torch.sort(torch.randint(0, pixels, (n,), generator=generator))[0]
    base = 2000.0 + 50.0 * torch.rand(pixels, generator=generator)
    # Each pixel's fragments step back from its first in fractions and
    # multiples of the resolution (at an eye 16000 units out).
    unit = fr.DEPTH_ORDER_SPACINGS * fr.EPS32 * 16000.0
    step = torch.tensor([0.0, 0.2, 0.5, 0.9, 1.5, 40.0])[
        torch.randint(0, 6, (n,), generator=generator)
    ]
    behind = torch.cumsum(step * unit, 0)
    behind -= behind[torch.searchsorted(pixel, pixel)]
    depth = (base[pixel] + behind).to(torch.float32)
    bits = depth.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    key, order = torch.sort((pixel << 32) | bits)
    circuit = torch.randint(0, 12, (n,), generator=generator)
    # Each fragment's share of its pixel in the border band: mostly fill,
    # some blended, some all border.
    border = torch.tensor([0, 0, 0, 1, 128, 254, 255])[
        torch.randint(0, 7, (n,), generator=generator)
    ]
    is_circuit = torch.rand(n, generator=generator) < 0.8
    ref = torch.where(
        is_circuit,
        -((circuit << 8) | border) - 1,
        torch.randint(0, 50, (n,), generator=generator),
    ).to(torch.int32)[order]
    cov = torch.where(
        torch.rand(n, generator=generator) < 0.85,
        torch.ones(n),
        torch.rand(n, generator=generator),
    )[order]
    # Twelve circuits by four Mobs, one of them unknown.
    owner_ids = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, -1, 3, 3, 3], dtype=torch.int32)
    # Whether each draws its fill or its border, per frame: tables of three
    # different frame counts, as the colour grids and the metadata can have.
    shows = (
        torch.rand((3, 12), generator=generator) < 0.8,  # a fill texel shows
        torch.arange(12).view(1, 12) != 4,  # filled
        torch.rand((2, 12), generator=generator) < 0.5,  # a border texel shows
        torch.rand((1, 12), generator=generator) < 0.5,  # has a border width
    )
    return key, ref, cov, owner_ids, shows, ppf


def test_the_scan_finds_the_brute_force_pairs_in_any_chunking(monkeypatch):
    full = 1.0 - 1e-6
    for seed in range(3):
        key, ref, cov, owner_ids, shows, ppf = _stream(seed)
        count, best = _brute_force(
            key, ref, cov, owner_ids, shows, 16000.0, 7, ppf, 8, full
        )
        assert count > 20, "the stream should hold pairs to find"
        for chunk in (fr._DEPTH_CHUNK, 64, 7):
            monkeypatch.setattr(fr, "_DEPTH_CHUNK", chunk)
            record = fr.depth_order_record(
                key, ref, cov, owner_ids, shows, 16000.0, 7, ppf, 8, full
            ).tolist()
            assert record[fr._D_COUNT] == count
            assert record[fr._D_RATIO] == best[0]
            assert record[fr._D_CIRCUIT_A] == best[1]
            assert record[fr._D_CIRCUIT_B] == best[2]
            assert record[fr._D_FRAME] == best[3]
            assert record[fr._D_DEPTH] == best[4]


def test_a_stream_without_pairs_reads_as_none():
    key, ref, cov, owner_ids, shows, ppf = _stream(0)
    one_owner = torch.zeros_like(owner_ids)
    record = fr.depth_order_record(
        key, ref, cov, one_owner, shows, 16000.0, 0, ppf, 8, 1.0
    )
    assert record[fr._D_COUNT] == 0
    assert math.isinf(float(record[fr._D_RATIO]))
    nothing_drawn = tuple(torch.zeros_like(table) for table in shows)
    record = fr.depth_order_record(
        key, ref, cov, owner_ids, nothing_drawn, 16000.0, 0, ppf, 8, 1.0
    )
    assert record[fr._D_COUNT] == 0
    empty = fr.depth_order_record(
        key[:1], ref[:1], cov[:1], owner_ids, shows, 16000.0, 0, ppf, 8, 1.0
    )
    assert empty[fr._D_COUNT] == 0


def test_what_a_circuit_draws_reads_alpha_glow_fill_and_border():
    from algan.rendering.raytracing.raytrace_kernels_taichi import (
        _M_BORDER_W,
        _M_FILLED,
    )

    # Two frames of four circuits' one-texel grids: r, g, b, glow, alpha.
    fill = torch.zeros((2, 4, 1, 5))
    fill[0, 0, 0, 4] = 0.5  # circuit 0: a visible fill at frame 0 only
    fill[:, 1, 0, 3] = 1.0  # circuit 1: glow alone, at zero alpha
    fill[:, 3, 0, 4] = 1.0  # circuit 3: a visible fill, but not filled
    border = torch.zeros((2, 4, 1, 5))
    border[1, 2, 0, 4] = 1.0  # circuit 2: a visible border at frame 1 only
    meta = torch.zeros((1, 4, 26))
    meta[..., _M_FILLED] = torch.tensor([1.0, 1.0, 1.0, 0.0])
    meta[..., _M_BORDER_W] = 2.0
    merged = {
        "circuit_colors": fill,
        "circuit_border_colors": border,
        "circuit_meta": meta,
    }
    shows = fr.circuit_shows(merged, torch.device("cpu"))
    assert fr.circuit_shows(merged, torch.device("cpu")) is shows
    frames = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    circuits = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    fill_only = torch.zeros(8, dtype=torch.int64)
    drawn = fr._drawn(shows, frames, circuits, fill_only, 255).tolist()
    assert drawn == [True, True, False, False, False, True, False, False]
    # Circuit 2 is an outline at frame 1: its border band draws, the inside
    # of it does not; a pixel that straddles the two draws.
    shares = torch.tensor([0, 128, 255])
    two = torch.full((3,), 2)
    drawn = fr._drawn(shows, torch.ones(3, dtype=torch.int64), two, shares, 255)
    assert drawn.tolist() == [False, True, True]


def test_circuit_owners_follow_each_way_a_primitive_records_them():
    a, b, c, d, e = (object() for _ in range(5))
    merged = {
        fr.CIRCUIT_OWNER_PARTS: (
            (3, [a, b], [1, 3], None),  # owner i drew circuits [ends[i-1], ends[i])
            (2, [c, d], None, None),  # one circuit per owner: the batched build
            (2, [e], None, None),  # one owner drew them all
            (2, None, None, b),  # a raw primitive: its lone owner
            (1, None, None, None),  # nobody known
        )
    }
    ids, mobs, host_ids = fr.circuit_owner_ids(merged, torch.device("cpu"))
    named = [mobs[i] if i >= 0 else None for i in host_ids.tolist()]
    assert named == [a, b, b, c, d, e, e, b, b, None]
    assert fr.circuit_owner_ids(merged, torch.device("cpu"))[0] is ids


def test_the_check_runs_only_where_rounding_outgrows_the_tie_step():
    tie = raster_pipeline.coplanar_tie_step()
    assert tie == 0.5e-4  # half of raytrace_kernels_taichi.depth_tie_epsilon
    assert fr.depth_order_eye(torch.tensor([[0.0, 0.0, 15.0]]), tie) is None
    assert fr.depth_order_eye(torch.tensor([[0.0, 90.0, 0.0]]), tie) is None
    assert fr.depth_order_eye(torch.tensor([[0.0, 150.0, 0.0]]), tie) == 150.0
    eyes = torch.tensor([[0.0, 0.0, 300.0], [0.0, 400.0, 0.0]])
    assert fr.depth_order_eye(eyes, tie) == 400.0
    assert fr.depth_order_eye(torch.zeros(0, 3), tie) is None
    # Tied by bin instead, a pair that close can share one out to ~210 units.
    SETTINGS.raytracing.experimental.coplanar_tie_gap = False
    tie = raster_pipeline.coplanar_tie_step()
    assert tie == 1e-4
    assert fr.depth_order_eye(torch.tensor([[0.0, 150.0, 0.0]]), tie) is None


# ---------------------------------------------------------------------------
# Renders
# ---------------------------------------------------------------------------


@pytest.fixture
def scans(monkeypatch):
    """The raster pipeline's runs of the scan, so a quiet render is not vacuous."""
    calls = []
    scan = raster_pipeline.depth_order_record

    def counted(*args):
        calls.append(1)
        return scan(*args)

    monkeypatch.setattr(raster_pipeline, "depth_order_record", counted)
    return calls


def _float32_warnings(caught):
    return [
        str(w.message)
        for w in caught
        if issubclass(w.category, Float32PrecisionWarning)
    ]


def _orbit(camera, t):
    camera.rotate(3 * t, UP, about=ORIGIN)


def _render(build, distance=FAR, seconds=0.4):
    """Render ``build(scene)``'s scene and return its float32 warnings' text.

    The camera is pulled back to ``distance`` (0 keeps the default camera,
    and ``"orthographic"`` switches it to the orthographic projection), tipped
    25 degrees and turning; ``build`` runs after that, inside ``Off``.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with Scene(video_settings=VIDEO) as scene:
            scene.set_background(BLACK)
            camera = scene.camera
            with Off():
                if distance == "orthographic":
                    camera.set_orthographic()
                elif distance:
                    pull_back(camera, distance)
                camera.rotate(-25, RIGHT, about=ORIGIN)
                build(scene)
            camera.add_updater(_orbit)
            Scene.wait(seconds)
            frames = round(seconds * VIDEO.frames_per_second)
            for _ in scene.get_frames(0, frames):
                pass
    return _float32_warnings(caught)


def _stacked_squares(gap):
    """Two overlapping squares ``gap`` apart along the view, riding the camera."""

    def build(scene):
        back = Square(size=3, color=RED, fill_opacity=1, stroke_width=0, name="back")
        front = Square(size=2, color=BLUE, fill_opacity=1, stroke_width=0, name="front")
        front.move(OUT * gap + RIGHT * 0.5)
        # Turned with the camera so they face it, and carried by it: a panel
        # on screen while the camera moves.
        panel = Group(back, front).rotate(-25, RIGHT, about=ORIGIN)
        scene.camera.add_children(panel)
        back.spawn()
        front.spawn()

    return build


def test_stacked_layers_under_a_distant_camera_warn_once_naming_both():
    caught = _render(_stacked_squares(0.001))
    assert len(caught) == 1, caught
    message = caught[0]
    assert "overlapping flat shapes" in message
    assert "Square 'back'" in message
    assert "Square 'front'" in message
    assert "16000 units from the origin" in message


def test_layers_the_camera_can_tell_apart_do_not_warn(scans):
    assert _render(_stacked_squares(0.05)) == []
    assert scans


def test_a_layer_held_invisible_does_not_warn(scans):
    """A shape at opacity 0, waiting to fade in, hides nothing either side."""

    def build(scene):
        back = Square(size=3, color=RED, fill_opacity=1, stroke_width=0)
        front = Square(size=2, color=BLUE, fill_opacity=1, stroke_width=0)
        front.move(OUT * 0.001 + RIGHT * 0.5)
        front.opacity = 0
        panel = Group(back, front).rotate(-25, RIGHT, about=ORIGIN)
        scene.camera.add_children(panel)
        back.spawn()
        front.spawn()

    assert _render(build) == []
    assert scans


def test_coplanar_layers_fixed_in_the_world_do_not_warn(scans):
    """The camera moves past them; nothing re-rounds them frame by frame."""

    def build(scene):
        back = Square(size=3, color=RED, fill_opacity=1, stroke_width=0)
        front = Square(size=2, color=BLUE, fill_opacity=1, stroke_width=0)
        front.move(RIGHT * 0.5)
        Group(back, front).rotate(-25, RIGHT, about=ORIGIN)
        back.spawn()
        front.spawn()

    assert _render(build) == []
    assert scans


def test_the_default_camera_never_runs_the_check(scans):
    """Coplanar layers near the origin: draw order's business, not rounding's."""
    assert _render(_stacked_squares(0.0), distance=0) == []
    assert scans == []


def test_the_orthographic_camera_never_runs_the_check(scans):
    """The flat look with its rays leaving from near the origin.

    The layers that flicker under the distant camera above tie here, and draw
    order decides.
    """
    assert _render(_stacked_squares(0.001), distance="orthographic") == []
    assert scans == []


def test_one_mob_crossing_itself_does_not_warn(scans):
    """A pack of strokes (a network's synapses) crosses itself everywhere, coplanar."""

    def build(scene):
        lines = []
        for k in range(8):
            a = torch.tensor([-2.0, -1.5 + k * 0.4, 0.0])
            b = torch.tensor([2.0, 1.5 - k * 0.4, 0.0])
            lines.append(
                torch.stack([a * (1 - s) + b * s for s in (0, 1 / 3, 2 / 3, 1)])
            )
        BezierCircuitCubic.from_batches(
            lines,
            filled=False,
            color=RED,
            stroke_color=RED,
            stroke_width=12,
            grid_width=1,
            grid_height=1,
        ).spawn()

    assert _render(build) == []
    assert scans


def test_the_warning_says_when_in_a_later_batch():
    """The pair first overlaps half a second in, one frame per batch."""
    SETTINGS.computing.set(max_animation_batch_size=1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with Scene(video_settings=VIDEO) as scene:
            scene.set_background(BLACK)
            with Off():
                pull_back(scene.camera, FAR)
                scene.camera.rotate(-25, RIGHT, about=ORIGIN)
                Square(size=3, color=RED, fill_opacity=1, stroke_width=0).spawn()
            Scene.wait(0.5)
            Square(size=2, color=BLUE, fill_opacity=1, stroke_width=0).move(
                OUT * 0.001
            ).spawn(animate=False)
            Scene.wait(0.3)
            for _ in scene.get_frames(0, 8):
                pass
    messages = _float32_warnings(caught)
    assert len(messages) == 1, messages
    when = float(re.search(r"worst at t=([0-9.]+)s", messages[0]).group(1))
    assert when >= 0.5
