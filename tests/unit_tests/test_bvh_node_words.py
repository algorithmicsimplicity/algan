"""BVH sibling blocks read as packed words decode to exactly the stored lanes.

``stbvh.kernel_blocks`` hands the traversal kernels the block bytes as int64
(int32 on MPS, narrower for a misaligned storage) words instead of
``vector(bvh_arity, f16)`` lanes (``raytrace_kernels_taichi._node_row``). Every
format must decode every bit the lanes hold -- infinities, NaN payloads (refit
link words are packed integers that often are f16 NaNs), negative zero -- and
so give the group tests bit-identical masks and entry distances. The original
lane fetch is the import-time ``ALGAN_BVH_NODE_WORDS=0`` arm, and other
arities and f32 blocks are import-time settings too, so those run in a
subprocess each and are compared with this process's results.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from algan.rendering.raytracing import raytrace_kernels_taichi as rk
from algan.rendering.raytracing import stbvh
from algan.rendering.raytracing.stbvh import (
    bvh_arity,
    bvh_block_f16,
    bvh_node_words,
    kernel_blocks,
)
from algan.rendering.taichi_runtime import init_taichi
from algan.settings._startup import render_device
from algan.taichi_compat import ti

_REPO = Path(__file__).resolve().parents[2]


#: The word decodes exist only under ``bvh_node_words``; the lanes arm's own
#: checks are the subprocess comparisons below.
words_only = pytest.mark.skipif(
    not bvh_node_words, reason="ALGAN_BVH_NODE_WORDS=0 has no word formats"
)

_SUBPROCESS_ENV = {"ALGAN_USE_DAEMON": "0", "ALGAN_AUTO_DAEMON": "0"}


def _random_blocks(rows, seed):
    """Lanes full of the values a decode can get wrong."""
    gen = torch.Generator().manual_seed(seed)
    if bvh_block_f16:
        bits = torch.randint(
            -32768, 32768, (rows, 8, bvh_arity), generator=gen, dtype=torch.int32
        ).to(torch.int16)
        special = torch.tensor(
            [0x7C00, -0x0400, 0x7E01, -0x0001, -0x8000, 0x0001, 0x03FF, 0x0400],
            dtype=torch.int32,
        ).to(torch.int16)
        bits.view(-1)[: special.numel()] = special
        return bits.view(torch.float16)
    bits = torch.randint(
        -(2**31), 2**31 - 1, (rows, 8, bvh_arity), generator=gen, dtype=torch.int64
    ).to(torch.int32)
    special = torch.tensor(
        [0x7F800000, -0x00800000, 0x7FC00001, -1, -(2**31), 1, 0x007FFFFF],
        dtype=torch.int64,
    ).to(torch.int32)
    bits.view(-1)[: special.numel()] = special
    return bits.view(torch.float32)


def _formats(blocks):
    flat = blocks.reshape(blocks.shape[0], -1)
    out = {"lanes": flat}
    lane_bytes = bvh_arity * blocks.element_size()
    if lane_bytes % 8 == 0:
        out["i64"] = flat.view(torch.int64)
    if lane_bytes % 4 == 0:
        out["i32"] = flat.view(torch.int32)
    return out


@ti.kernel
def _decode_all(blocks: rk.NODE_ARG, out: ti.types.ndarray()):
    """Every value of every row through the row decode, as raw f32 bits."""
    for r in range(blocks.shape[0]):
        words = rk._node_row(blocks, r)
        for k in ti.static(range(8)):
            for c in ti.static(range(bvh_arity)):
                bits = rk._row_bits(blocks, words, k, c)
                out[r, k, c, 0] = bits
                out[r, k, c, 1] = ti.bit_cast(rk._node_bound(bits), ti.i32)
                out[r, k, c, 2] = rk._node_bits(blocks, r, k, c)


@ti.kernel
def _group_probe(
    blocks: rk.NODE_ARG,
    rays: ti.types.ndarray(),
    out: ti.types.ndarray(),
    refit: ti.template(),
):
    for i in range(rays.shape[0]):
        ro = ti.math.vec3(rays[i, 0], rays[i, 1], rays[i, 2])
        inv_rd = ti.math.vec3(rays[i, 3], rays[i, 4], rays[i, 5])
        row = i % blocks.shape[0]
        mask, near = rk._group_test(
            refit,
            0,
            row,
            ti.cast(rays[i, 6], ti.i32),
            ro,
            inv_rd,
            rays[i, 7],
            rays[i, 8],
            blocks,
        )
        out[i, 0] = mask
        for c in ti.static(range(bvh_arity)):
            out[i, 1 + c] = ti.bit_cast(near[c], ti.i32)
        out[i, 1 + bvh_arity] = rk._refit_link(row, i % bvh_arity, blocks)


def _expected_values(blocks):
    """(stored bits zero-extended, the f32 a lane converts to) per value."""
    if bvh_block_f16:
        raw = blocks.view(torch.int16).to(torch.int32) & 0xFFFF
        as_f32 = blocks.float().view(torch.int32)
    else:
        raw = blocks.view(torch.int32)
        as_f32 = raw
    return raw, as_f32


@words_only
def test_every_format_decodes_every_stored_value_exactly():
    init_taichi()
    blocks = _random_blocks(64, seed=1)
    raw, as_f32 = _expected_values(blocks)
    for name, view in _formats(blocks).items():
        out = torch.zeros(
            (64, 8, bvh_arity, 3), dtype=torch.int32, device=render_device()
        )
        _decode_all(view.to(render_device()), out)
        out = out.cpu()
        assert torch.equal(out[..., 0], raw), name
        assert torch.equal(out[..., 2], raw), name
        # NaNs compare by payload class only: a conversion may quiet them.
        got = out[..., 1].view(torch.float32)
        want = as_f32.view(torch.float32)
        nan = torch.isnan(want)
        assert torch.equal(torch.isnan(got), nan), name
        assert torch.equal(out[..., 1][~nan], as_f32[~nan]), name


def _probe_rays(n, seed):
    gen = torch.Generator().manual_seed(seed)
    ro = (torch.rand((n, 3), generator=gen) - 0.5) * 200.0
    d = torch.randn((n, 3), generator=gen)
    inv = torch.where(d.abs() > 1e-12, 1.0 / d, torch.full_like(d, 1e12))
    frame = torch.randint(0, 32768, (n, 1), generator=gen).float()
    t_lo = (torch.rand((n, 1), generator=gen) - 0.2) * 10.0
    t_hi = t_lo + torch.rand((n, 1), generator=gen) * 1e4
    return torch.cat([ro, inv, frame, t_lo, t_hi], 1).contiguous()


def _probe_blocks():
    blocks = _random_blocks(97, seed=2)
    # Plausible boxes too, so masks are not all empty: lo < hi in half the rows.
    gen = torch.Generator().manual_seed(5)
    box_lo = (torch.rand((48, 3, bvh_arity), generator=gen) - 0.5) * 100
    box_hi = box_lo + torch.rand((48, 3, bvh_arity), generator=gen) * 50
    blocks[:48, 0:3] = box_lo.to(blocks.dtype)
    blocks[:48, 3:6] = box_hi.to(blocks.dtype)
    # Frames 0..32767 (an STBVH child's interval; a refit link that is not -1).
    if bvh_block_f16:
        blocks.view(torch.int16)[:48, 6] = 0
        blocks.view(torch.int16)[:48, 7] = 0x7FFF
    else:
        blocks.view(torch.int32)[:48, 6] = 0x7FFF0000
    return blocks


def _probe(view, refit):
    rays = _probe_rays(4096, seed=3)
    out = torch.zeros(
        (rays.shape[0], 2 + bvh_arity), dtype=torch.int32, device=render_device()
    )
    _group_probe(view.to(render_device()), rays.to(render_device()), out, refit)
    return out.cpu()


def probe_digest(refit):
    """Digest of the group probe, read the way this process's kernels take
    the blocks (``kernel_blocks``): compared across processes below.
    """
    init_taichi()
    out = _probe(kernel_blocks(_probe_blocks()), refit)
    return hashlib.sha256(out.numpy().tobytes()).hexdigest()


@words_only
@pytest.mark.parametrize("refit", [0, 1])
def test_group_tests_are_bit_identical_across_formats(refit):
    init_taichi()
    blocks = _probe_blocks()
    results = {name: _probe(view, refit) for name, view in _formats(blocks).items()}
    reference = results.pop("lanes")
    assert int((reference[:, 0] != 0).sum()) > 20, "the probe must hit boxes"
    for name, out in results.items():
        assert torch.equal(out, reference), name


@words_only
def test_kernel_blocks_picks_word_views_without_copying(monkeypatch):
    blocks = _random_blocks(10, seed=4)
    words = kernel_blocks(blocks)
    lane_bytes = bvh_arity * blocks.element_size()
    expected = torch.int64 if lane_bytes % 8 == 0 else torch.int32
    assert words.dtype == expected
    assert words.shape == (10, 8 * lane_bytes // words.element_size())
    assert words.data_ptr() == blocks.data_ptr()
    # A storage one element off an 8-byte boundary cannot be read as 8-byte
    # words: 4-byte ones when it is 4-aligned, else one scalar per value.
    backing = torch.zeros(blocks.numel() + 1, dtype=blocks.dtype)
    shifted = backing[1:].view(blocks.shape)
    expect = blocks.dtype if blocks.element_size() == 2 else torch.int32
    assert kernel_blocks(shifted).dtype == expect
    # The lanes arm hands over the block array itself, lane vectors and all.
    monkeypatch.setattr(stbvh, "bvh_node_words", False)
    assert kernel_blocks(blocks) is blocks


_SUBPROCESS_PROBE = r"""
import sys
sys.path.insert(0, sys.argv[1])
import test_bvh_node_words as t
t.test_every_format_decodes_every_stored_value_exactly()
t.test_group_tests_are_bit_identical_across_formats(0)
t.test_group_tests_are_bit_identical_across_formats(1)
print("ok", t.bvh_arity, t.bvh_block_f16)
"""


@pytest.mark.parametrize(
    ("arity", "f16"),
    [(2, "1"), (8, "1"), (4, "0"), (8, "0")],
    ids=["arity2-f16", "arity8-f16", "arity4-f32", "arity8-f32"],
)
def test_other_block_layouts_decode_exactly(arity, f16):
    env = dict(
        os.environ,
        ALGAN_BVH_ARITY=str(arity),
        ALGAN_BVH_BLOCK_F16=f16,
        ALGAN_BVH_NODE_WORDS="1",
        **_SUBPROCESS_ENV,
    )
    r = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_PROBE, str(Path(__file__).parent)],
        cwd=_REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-4000:]
    assert f"ok {arity} {f16 == '1'}" in r.stdout


def _lanes_arm(code, *args):
    """Run ``code`` in a process with the original lane fetch."""
    env = dict(os.environ, ALGAN_BVH_NODE_WORDS="0", **_SUBPROCESS_ENV)
    r = subprocess.run(
        [sys.executable, "-c", code, str(Path(__file__).parent), *args],
        cwd=_REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-4000:]
    return r.stdout


_LANES_PROBE = r"""
import sys
sys.path.insert(0, sys.argv[1])
import test_bvh_node_words as t
assert not t.bvh_node_words
print("DIGEST", t.probe_digest(0), t.probe_digest(1))
"""


@words_only
def test_the_original_lane_fetch_gives_the_words_answers():
    """The lanes arm is the pre-words fetch (``vector`` annotation, the
    original ``_test_children``); its masks, entry distances and links are the
    words' to the bit.
    """
    out = _lanes_arm(_LANES_PROBE)
    line = next(ln for ln in out.splitlines() if ln.startswith("DIGEST "))
    assert line.split()[1:] == [probe_digest(0), probe_digest(1)]


def _render(path):
    from algan import (
        BLACK,
        BLUE,
        OUT,
        RED,
        RIGHT,
        SETTINGS,
        SMOKE_TEST,
        UP,
        WHITE,
        Circle,
        Cube,
        MeshLambertMaterial,
        MeshPhysicalMaterial,
        Off,
        PointLight,
        Prism,
        Scene,
        Sphere,
    )
    from algan.scene_manager import SceneManager

    settings = SMOKE_TEST.set(resolution=(96, 64))
    SceneManager.reset()
    snapshot = SETTINGS.snapshot()
    try:
        SETTINGS.raytracing.set(shadows=True, max_bounces=1)
        with Scene(video_settings=settings) as scene:
            scene.set_background(BLACK)
            with Off():
                PointLight(
                    location=RIGHT + OUT * 6 + UP * 1.5, color=WHITE, intensity=20.0
                ).spawn(animate=False)
                Prism(width=14.0, height=9.0, depth=0.1).set_material(
                    MeshLambertMaterial(color=WHITE)
                ).spawn(animate=False)
                Cube(size=1.6).set_material(MeshLambertMaterial(color=RED)).move(
                    RIGHT * -2.5 + OUT * 1.0
                ).spawn(animate=False)
                Cube(size=1.8, opacity=0.5).set_material(
                    MeshLambertMaterial(color=BLUE)
                ).move(RIGHT * 0.5 + OUT * 1.4).spawn(animate=False)
                Sphere(radius=1.0).set_material(
                    MeshPhysicalMaterial(color=WHITE, metalness=1.0, roughness=0.0)
                ).move(RIGHT * 3.0 + OUT * 1.5 + UP).spawn(animate=False)
                Circle(radius=0.7, color=RED).move(
                    RIGHT * -2.0 + OUT * 3.6 + UP * 1.8
                ).spawn(animate=False)
            result = scene.save_frame(
                str(path), video_settings=settings, overwrite=True
            )
        with Image.open(result.output_path) as image:
            return np.asarray(image.convert("RGBA")).copy()
    finally:
        SETTINGS.restore(snapshot)
        SceneManager.reset()


_LANES_FRAME = r"""
import hashlib, sys
sys.path.insert(0, sys.argv[1])
from pathlib import Path
import test_bvh_node_words as t
frame = t._render(Path(sys.argv[2]) / "lanes.png")
print("MD5", hashlib.md5(frame.tobytes()).hexdigest())
"""


@words_only
def test_frames_match_the_original_lane_fetch(tmp_path):
    """Camera, shadow and mirror-bounce traversal all read the words: the
    frame is the lanes arm's frame to the byte (a mirror, not glass, so the
    bounce does not split a pixel).
    """
    words = _render(tmp_path / "words.png")
    out = _lanes_arm(_LANES_FRAME, str(tmp_path))
    line = next(ln for ln in out.splitlines() if ln.startswith("MD5 "))
    assert line.split()[1] == hashlib.md5(words.tobytes()).hexdigest()
    assert len(np.unique(words.reshape(-1, 4), axis=0)) > 20


_ALIGNMENT = r"""
import torch
from algan.rendering.raytracing import scene_builder
blocks = torch.zeros((4, 8, 4), dtype=torch.float16)
group = {"tensors": [blocks], "node_blocks": True}
print("ALIGN", scene_builder._storage_alignment(group),
      max(2, scene_builder._upload_alignment_floor()))
"""


@pytest.mark.parametrize("words", ["1", "0"], ids=["words", "lanes"])
def test_only_the_words_arm_asks_for_aligned_block_storages(words):
    """Block storages start on 16 bytes for the 8-byte word loads; the lanes
    arm keeps the original rule -- the element size, or the upload floor
    where the Metal zero-copy patch is installed (16 on the published
    Quadrants build, which makes the two arms' layouts identical there).
    """
    r = subprocess.run(
        [sys.executable, "-c", _ALIGNMENT],
        cwd=_REPO,
        env=dict(os.environ, ALGAN_BVH_NODE_WORDS=words, **_SUBPROCESS_ENV),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert r.returncode == 0, r.stderr[-3000:]
    line = next(ln for ln in r.stdout.splitlines() if ln.startswith("ALIGN "))
    got, original = (int(v) for v in line.split()[1:])
    assert got == (max(16, original) if words == "1" else original)
