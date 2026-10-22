"""Coplanar layers tie by depth gap, not by depth bin, in the sheet route.

The raster pipeline's fragment order used to tie two fragments of a pixel
only when ``floor(t / depth_tie_epsilon)`` agreed. Two exactly coplanar
shapes whose depths round a few ulps apart -- a panel posed from the camera's
own axes and riding it -- were then split wherever a bin edge fell between
them, the farther one won there, and as the camera moved so did those pixels:
the shape behind flickered through the one in front.
``raster_pipeline._tie_grouped_fragment_order`` sorts each pixel by exact
depth instead and cuts it into runs at gaps of half a tie step. A run
narrower than that end to end is a tie, which draw order (descending layer)
decides; a wider one is a dense stack of sub-pixel triangles and keeps the
bins.

The order is checked against a brute-force one on both of its sort arms, for
its properties -- a bin edge changes nothing, a whole tie step (what
``z_index`` adds) still separates, a dense stack keeps its binned order --
and against the switch that restores the bins; then on renders of the
squares that flickered.

Unmarked, so outside ``--fast``: only the fragment order can break it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from algan import (
    BLACK,
    BLUE,
    ORIGIN,
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
from algan.rendering.raytracing import raster_pipeline as rpl

EPS = rpl.depth_tie_epsilon
GAP = np.float32(rpl._COPLANAR_TIE_GAP)


def _layer(ref, offset):
    return (-ref - 1) >> rpl._BEZ_BORDER_BITS if ref < 0 else ref + offset


def _depth(key):
    return np.array([key & 0xFFFFFFFF], dtype=np.uint32).view(np.float32)[0]


def _reference(key, ref, offset, device="cpu"):
    """The tie-grouped order one run at a time, in plain Python.

    A dense stack's bins are the renderer's own arithmetic on ``device`` (what
    is checked here is the grouping). They differ by device: torch on CUDA
    evaluates ``t / depth_tie_epsilon`` as ``t * (1 / eps)``, which moves a
    fragment near a bin edge into the neighbouring bin -- a few hundred of
    each stream below -- so a CPU-binned reference fails a correct CUDA order.
    """
    bin_of = rpl._depth_bin(key.to(device)).tolist()
    key = key.tolist()
    ref = ref.tolist()
    n = len(key)
    stream = sorted(range(n), key=lambda i: (key[i], i))
    out = []
    i = 0
    while i < n:
        run = [stream[i]]
        j = i + 1
        while j < n:
            a, b = stream[j - 1], stream[j]
            if key[a] >> 32 != key[b] >> 32:
                break
            if not np.float32(_depth(key[b]) - _depth(key[a])) < GAP:
                break
            run.append(b)
            j += 1
        position = {g: k for k, g in enumerate(run)}
        if np.float32(_depth(key[run[-1]]) - _depth(key[run[0]])) < GAP:
            run.sort(key=lambda g: (-_layer(ref[g], offset), position[g]))
        else:
            run.sort(key=lambda g: (bin_of[g], -_layer(ref[g], offset), position[g]))
        out.extend(run)
        i = j
    return torch.tensor(out, dtype=torch.int64)


def _keys(pixels, depths):
    pixels = torch.as_tensor(pixels, dtype=torch.int64)
    depths = torch.as_tensor(depths, dtype=torch.float32)
    return (pixels << 32) | (depths.view(torch.int32).to(torch.int64) & 0xFFFFFFFF)


def _stream(seed, n, pixels, layers):
    """Fragments crowded at a few depths: exact ties, ulp-apart neighbours,
    and jitter wide enough to chain groups and to straddle bin edges.
    """
    g = torch.Generator().manual_seed(seed)
    pix = torch.randint(0, pixels, (n,), generator=g, dtype=torch.int64)
    base = torch.randint(0, 40, (n,), generator=g).float() * EPS + 5.0
    jitter = (torch.rand(n, generator=g) - 0.5) * (1.6 * EPS)
    pick = torch.rand(n, generator=g)
    t = torch.where(pick < 0.5, base, base + jitter)
    t = torch.where(pick > 0.9, torch.nextafter(base, torch.tensor(10.0)), t)
    ref = torch.randint(
        -(layers << rpl._BEZ_BORDER_BITS),
        layers,
        (n,),
        generator=g,
        dtype=torch.int32,
    )
    return _keys(pix, t), ref


_STREAMS = [(0, 2000, 30, 9), (1, 5000, 400, 50), (2, 3000, 3, 2000), (3, 60, 1, 3)]


@pytest.mark.parametrize("friendly", [False, True], ids=["plain", "mps_friendly"])
@pytest.mark.parametrize(("seed", "n", "pixels", "layers"), _STREAMS)
def test_the_order_matches_brute_force(monkeypatch, friendly, seed, n, pixels, layers):
    """Also in MPS-friendly mode, whose gathers and row copy take other ops."""
    monkeypatch.delenv("ALGAN_MPS_FRIENDLY", raising=False)
    SETTINGS.computing.set(mps_friendly=friendly)
    key, ref = _stream(seed, n, pixels, layers)
    got = rpl._tie_grouped_fragment_order(key, ref, layers)
    assert got.dtype == torch.int64
    assert torch.equal(got, _reference(key, ref, layers))


def _kernel_device():
    """The device the live Taichi arch launches on in place; skips elsewhere.

    The counting sort is gated to CUDA in the renderer and reachable on the
    CPU arch only by forcing its ``devices``.
    """
    from algan.rendering.taichi_runtime import _live_arch, init_taichi
    from algan.taichi_compat import ti

    init_taichi()
    arch = _live_arch()
    if arch == ti.cuda:
        return "cuda"
    if arch == ti.cpu:
        return "cpu"
    pytest.skip(f"the counting sort runs on the CPU or CUDA arch, not {arch}")


@pytest.mark.parametrize(("seed", "n", "pixels", "layers"), _STREAMS)
def test_the_counting_sort_arm_gives_the_same_order(
    monkeypatch, seed, n, pixels, layers
):
    device = _kernel_device()
    binned = rpl._binned_fragment_order
    taken = []

    def forced(*args, **kwargs):
        kwargs.update(min_fragments=0, devices=(device,))
        order = binned(*args, **kwargs)
        taken.append(order is not None)
        return order

    monkeypatch.setattr(rpl, "_binned_fragment_order", forced)
    key, ref = _stream(seed, n, pixels, layers)
    got = rpl._tie_grouped_fragment_order(
        key.to(device), ref.to(device), layers, num_bins=pixels
    )
    assert taken == [True]
    assert torch.equal(got.cpu(), _reference(key, ref, layers, device))


def _two_layer_order(depths, layers=(5, 9), *, tie_gap=True):
    """Which of two circuit fragments on one pixel the order puts first."""
    SETTINGS.raytracing.experimental.coplanar_tie_gap = tie_gap
    key = _keys([7, 7], depths)
    ref = torch.tensor(
        [-((layer << rpl._BEZ_BORDER_BITS) + 1) for layer in layers],
        dtype=torch.int32,
    )
    return rpl._exact_fragment_order(key, ref, 0).tolist()


@pytest.mark.parametrize("slide", [0.0, 0.25, 0.5, 0.75, 0.99])
def test_a_bin_edge_between_coplanar_layers_changes_nothing(slide):
    """Two depths a few ulps apart either side of an edge: draw order wins.

    The farther fragment (index 1) is the later-drawn layer. Slid along by any
    part of a bin, the pair keeps that order; the bins hand it to the nearer
    one exactly when the edge falls between them.
    """
    edge = np.float32(5.0 + 3 * EPS + slide * EPS)
    near = float(np.nextafter(np.nextafter(edge, np.float32(0)), np.float32(0)))
    far = float(np.nextafter(edge, np.float32(10)))
    assert _two_layer_order([near, far]) == [1, 0]
    straddles = np.floor(near / EPS) != np.floor(far / EPS)
    binned = _two_layer_order([near, far], tie_gap=False)
    assert binned == ([0, 1] if straddles else [1, 0])


def test_a_whole_tie_step_still_separates():
    """``z_index`` lifts a layer by a tie step; the later-drawn one stays behind."""
    near = 5.0 + 3.5 * EPS
    assert _two_layer_order([near, near + EPS]) == [0, 1]
    assert _two_layer_order([near, near + 0.4 * EPS]) == [1, 0]


def _one_pixel_order(depths, layers, *, tie_gap=True):
    SETTINGS.raytracing.experimental.coplanar_tie_gap = tie_gap
    key = _keys([3] * len(depths), depths)
    ref = torch.tensor(layers, dtype=torch.int32)  # triangles: layer = ref
    return rpl._exact_fragment_order(key, ref, 0).tolist()


def test_a_coplanar_stack_ties_whole():
    """Four layers within a few ulps, across a bin edge: draw order alone."""
    up, down = np.float32(10), np.float32(0)
    edge = np.float32(5.0 + 7 * EPS)
    ulps = [np.nextafter(np.nextafter(edge, down), down), np.nextafter(edge, down)]
    ulps += [edge, np.nextafter(edge, up)]
    ulps = [float(d) for d in ulps]
    keys = _keys([3] * 4, ulps)
    assert len(set(rpl._depth_bin(keys).tolist())) == 2
    depths = [ulps[2], ulps[0], ulps[3], ulps[1]]
    layers = [11, 40, 7, 23]
    assert _one_pixel_order(depths, layers) == [1, 3, 0, 2]


def test_a_dense_stack_keeps_its_binned_order():
    """Sub-pixel triangles of one tilted surface: steps finer than the gap
    across a dozen bins. Chained into one tie they would be reordered by
    triangle index over the whole span; they keep the bins instead.
    """
    g = torch.Generator().manual_seed(11)
    depths = (20.0 + torch.arange(40, dtype=torch.float64) * 0.3 * EPS).tolist()
    layers = torch.randperm(1000, generator=g)[:40].tolist()
    tied = _one_pixel_order(depths, layers)
    assert tied == _one_pixel_order(depths, layers, tie_gap=False)
    assert tied != sorted(range(40), key=lambda i: -layers[i])


def test_the_switch_restores_the_binned_order():
    key, ref = _stream(5, 4000, 60, 40)
    SETTINGS.raytracing.experimental.coplanar_tie_gap = False
    got = rpl._exact_fragment_order(key, ref, 40)
    layer = rpl._fragment_layer(ref, 40)
    layer_order = torch.argsort(layer, descending=True, stable=True)
    primary = rpl._primary_depth_key(key.index_select(0, layer_order))
    want = layer_order.index_select(0, torch.argsort(primary, stable=True))
    assert torch.equal(got, want)
    assert not torch.equal(got, _reference(key, ref, 40))


# ---------------------------------------------------------------------------
# Renders
# ---------------------------------------------------------------------------

VIDEO = VideoSettings(resolution=(160, 90), frames_per_second=10, supersampling=1)
FRAMES = 12


def _render(*, ride, lift=None, shift=0.5):
    """Frames of two coplanar squares posed from the camera's axes, as uint8.

    The blue one is spawned second, so draw order puts it in front; ``lift``
    (``"front"`` or ``"back"``) gives that square ``z_index = 1``. The camera
    turns, and the pair rides it (``ride``) or stays in the world. Riding at
    the default ``shift``, the bins swapped 802 pixels across 11 frame pairs.
    """
    with Scene(video_settings=VIDEO) as scene:
        scene.set_background(BLACK)
        camera = scene.camera
        with Off():
            back = Square(size=3, color=RED, fill_opacity=1, stroke_width=0)
            front = Square(size=2, color=BLUE, fill_opacity=1, stroke_width=0)
            front.move(RIGHT * shift)
            if lift is not None:
                (front if lift == "front" else back).z_index = 1
            camera.rotate(-25, RIGHT, about=ORIGIN)
            # A panel posed from the camera's own axes, 15 units in front of
            # the origin along the view: the two planes' depths come out a few
            # ulps apart, which is what a bin edge splits.
            panel = Group(back, front)
            axes = torch.cat((camera.right, camera.up, -camera.forward), dim=-1)
            panel.basis = axes * panel.scale_coefficient[..., :1]
            depth = ((ORIGIN - camera.location) * camera.forward).sum(-1, keepdim=True)
            panel.move_to(camera.location + camera.forward * (depth - 15))
            if ride:
                camera.add_children(panel)
            back.spawn()
            front.spawn()
        camera.add_updater(lambda c, t: c.rotate(3 * t, UP, about=ORIGIN))
        Scene.wait(FRAMES / VIDEO.frames_per_second)
        frames = torch.cat([f.cpu() for f in scene.get_frames(0, FRAMES)])
    return frames[..., :3]


def _blue(frames):
    rgb = frames.float()
    return (rgb[..., 2] > 120) & (rgb[..., 0] < 100)


@pytest.mark.parametrize("ride", [True, False], ids=["riding", "world"])
def test_coplanar_squares_draw_as_if_the_later_one_were_lifted(ride):
    """Draw order alone puts the front square where ``z_index`` would, in
    every frame: no pixel of the one behind shows through.
    """
    plain = _render(ride=ride)
    lifted = _render(ride=ride, lift="front")
    assert bool((_blue(plain).flatten(1).sum(1) > 300).all())
    assert int((plain.int() - lifted.int()).abs().max()) <= 2


def test_z_index_still_lifts_a_coplanar_layer():
    """Lifted a tie step, the square drawn first covers the one inside it.

    The bins missed this too: a step that rounds a hair short of a bin lands
    in one bin with the layer below on a few pixels (8 here), and there draw
    order put the square behind back in front.
    """
    frames = _render(ride=True, lift="back", shift=0.25)
    assert int(_blue(frames).sum()) == 0
