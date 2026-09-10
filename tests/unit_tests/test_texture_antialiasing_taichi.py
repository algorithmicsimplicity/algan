"""UV filtering: numerical oracles and compiled production sampler probes."""

import math

import pytest
import torch

from algan import SETTINGS
from algan.rendering.raytracing.raytrace_kernels_taichi import _sample_texture
from algan.rendering.raytracing.texture_mips import append_mip_pyramid, build_mip_levels
from algan.rendering.raytracing.texture_mips_taichi import _triangle_uv_footprint
from algan.rendering.raytracing.wavefront_kernels_taichi import (
    _sample_tex_vec5,
    _tri_material_g,
    _tri_normal_g,
)
from algan.rendering.taichi_runtime import init_taichi
from algan.taichi_compat import ti
from algan.utils.color_space import srgb_to_linear


@ti.kernel
def _sample_probe(meta: ti.types.ndarray(), bank: ti.types.ndarray(),
                  queries: ti.types.ndarray(), out: ti.types.ndarray()):
    for i in range(queries.shape[0]):
        f = ti.cast(queries[i, 0], ti.i32)
        c, a = _sample_texture(f, queries[i, 1], queries[i, 2], 0, meta, bank,
                               queries[i, 3], queries[i, 4])
        for k in ti.static(range(4)):
            out[i, k] = c[k]
        out[i, 4] = a


@ti.kernel
def _data_probe(meta: ti.types.ndarray(), bank: ti.types.ndarray(),
                queries: ti.types.ndarray(), out: ti.types.ndarray()):
    for i in range(queries.shape[0]):
        v = _sample_tex_vec5(ti.cast(queries[i, 0], ti.i32), queries[i, 1],
                             queries[i, 2], meta[0, 0], meta[0, 1], meta[0, 2],
                             meta[0, 10], bank, meta[0, 18], queries[i, 3], queries[i, 4])
        for k in ti.static(range(5)):
            out[i, k] = v[k]


@ti.kernel
def _footprint_probe(pos: ti.types.ndarray(), uv: ti.types.ndarray(),
                     meta: ti.types.ndarray(), queries: ti.types.ndarray(),
                     trim: ti.template(), out: ti.types.ndarray()):
    for i in range(queries.shape[0]):
        rd = ti.math.vec3(queries[i, 0], queries[i, 1], queries[i, 2])
        du, dv = _triangle_uv_footprint(trim, 0, 0, rd, queries[i, 3], pos, uv, meta, 0)
        out[i, 0], out[i, 1] = du, dv


def _bank(tex, *, color=True, linear=False, lerp=None, packed=False, opacity=None, wrap=(False, False)):
    tex = tex.float()
    w, h = tex.shape[-3:-1]
    meta = torch.full((1, 21), -1, dtype=torch.int32)
    meta[0, :3] = torch.tensor([0, w, h])
    meta[0, 10:13] = 1
    meta[0, 10] = tex.shape[0] if lerp is None else 1
    meta[0, 14] = meta[0, 17] = 1
    raw = tex
    if color and linear and lerp is None:
        raw = torch.cat((srgb_to_linear(tex[..., :3]), tex[..., 3:]), -1)
    if packed:
        assert lerp is not None
        bits = (tex[..., (0, 1, 2, 4)] * 255).round().to(torch.uint8).contiguous().view(torch.int32).view(torch.float32).flatten()
        bits = torch.cat((bits, bits.new_zeros((-bits.numel()) % 5)))
        parts = [bits.reshape(1, -1, 5)]
        meta[0, 15] = -2
    else:
        parts = [raw.reshape(1, -1, 5)]
    off = [parts[0].shape[1]]
    if lerp is not None:
        rows = torch.zeros((1, len(lerp), 5))
        rows[0, :, :3] = lerp
        rows[0, :, 3] = float(linear)
        meta[0, 16], meta[0, 17] = off[0], len(lerp)
        parts.append(rows)
        off[0] += len(lerp)
    if opacity is not None:
        rows = torch.zeros((1, len(opacity), 5))
        rows[0, :, 0] = torch.tensor(opacity)
        meta[0, 13], meta[0, 14] = off[0], len(opacity)
        parts.append(rows)
        off[0] += len(opacity)
    meta[0, 18] = append_mip_pyramid(parts, off, tex, color=color, linear=linear, lerp=lerp, wrap=wrap)
    return meta, torch.cat(parts, 1)


def _sample(tex, queries, *, data=False, **kwargs):
    init_taichi()
    meta, bank = _bank(tex, color=not data, **kwargs)
    q = torch.tensor(queries, dtype=torch.float32)
    out = torch.zeros((len(q), 5))
    (_data_probe if data else _sample_probe)(meta, bank, q, out)
    return out


@pytest.mark.parametrize('shape', [(8, 8), (7, 5), (1, 17), (19, 1), (3, 3), (1, 1)])
def test_pyramid_preserves_integral_without_padding_bias(shape):
    tex = torch.rand((2, *shape, 5), generator=torch.Generator().manual_seed(12))
    original = tex.clone()
    levels = build_mip_levels(tex)
    if max(shape) == 1:
        assert levels == []
    else:
        for level in levels:
            assert torch.allclose(level.mean((1, 2)), tex.mean((1, 2)), atol=2e-6)
        assert levels[-1].shape == (2, 1, 1, 5)
    assert torch.equal(tex, original)


def test_linear_light_coverage_and_glow_are_filtered_together():
    tex = torch.tensor([[[[1., 0., 0., 0.4, 1.], [0., 0., 1., 0., 0.]]]])
    out = _sample(tex, [[0, .5, .5, 8, 8]])[0]
    assert torch.allclose(out, torch.tensor([1., 0., 0., .4, .5]), atol=1e-6)
    tex = torch.zeros((1, 2, 2, 5))
    tex[..., 4] = 1
    tex[:, 0, :, :3] = 1
    out = _sample(tex, [[0, .3, .7, 8, 8]], linear=True)[0]
    assert torch.allclose(out[:3], torch.full((3,), .5), atol=1e-6)


@pytest.mark.parametrize('packed', [False, True])
def test_animated_endpoints_decode_after_time_blend_before_mip(packed):
    tex = torch.zeros((1, 2, 4, 4, 5))
    tex[..., 4] = 1
    tex[:, 1, :, :, :3] = 1
    lerp = torch.tensor([[0., 1., 0.25], [0., 1., 0.75]])
    out = _sample(tex, [[0, .2, .4, 0, 0], [1, .2, .4, 0, 0],
                        [0, .2, .4, 2, 2], [1, .2, .4, 2, 2]],
                  linear=True, lerp=lerp, packed=packed, opacity=[.2, .8])
    expect = srgb_to_linear(torch.tensor([.25, .75]))
    assert torch.allclose(out[:, 0], expect.repeat(2), atol=1e-6)
    assert torch.allclose(out[:, 4], torch.tensor([.2, .8, .2, .8]), atol=1e-6)


def test_checkerboard_minification_and_fractional_level_blend():
    tex = torch.zeros((1, 8, 8, 5))
    tex[..., 4] = 1
    tex[0, ..., :3] = ((torch.arange(8)[:, None] + torch.arange(8)[None, :]) % 2)[..., None]
    queries = [[0, .0625, .0625, rho / 8, rho / 8] for rho in [1, math.sqrt(2), 2, 80]]
    out = _sample(tex, queries)
    assert torch.allclose(out[:, 0], torch.tensor([0., .25, .5, .5]), atol=1e-6)
    assert torch.equal(out[:, 4], torch.ones(4))


def test_data_maps_keep_all_channels_and_frame_addressing():
    tex = torch.rand((3, 5, 7, 5), generator=torch.Generator().manual_seed(17))
    out = _sample(tex, [[f, .34, .76, 100, 100] for f in range(6)], data=True)
    assert torch.allclose(out, tex.mean((1, 2)).repeat((2, 1)), atol=2e-6)


def test_disabled_directory_is_exact_legacy_bilinear():
    init_taichi()
    tex = torch.rand((1, 4, 4, 5), generator=torch.Generator().manual_seed(1))
    meta, bank = _bank(tex)
    meta[0, 18] = -1
    out = torch.zeros((2, 5))
    _sample_probe(meta, bank, torch.tensor([[0., .5, .5, 0, 0], [0., .5, .5, 100, 100]]), out)
    assert torch.equal(out[0], out[1])
    assert torch.allclose(out[0], tex[0, 1:3, 1:3].mean((0, 1)), atol=1e-7)


@pytest.mark.parametrize('trim', [0, 1])
def test_cone_uses_uv_density_incidence_and_accumulated_distance(trim):
    init_taichi()
    pos = torch.tensor([[[0., 0., 0., 2., 0., 0., 0., 4., 0.]]])
    uv = torch.tensor([[[0., 0., 1., 0., 0., 1.]]])
    meta = torch.full((1, 21), -1, dtype=torch.int32)
    meta[0, 18] = 0
    q = torch.tensor([[0., 0., 1., .1], [.8, 0., .6, .1], [0., 0., 1., .4]])
    out = torch.zeros((3, 2))
    _footprint_probe(pos, uv, meta, q, trim, out)
    assert torch.allclose(out[0], torch.tensor([.05, .025]), atol=1e-6)
    assert torch.allclose(out[1], torch.tensor([.05 / .6, .025]), atol=1e-6)
    assert torch.allclose(out[2], 4 * out[0], atol=1e-6)
    pos.zero_()
    _footprint_probe(pos, uv, meta, q, trim, out)
    assert torch.equal(out, torch.zeros_like(out))


def test_antialiasing_is_a_runtime_public_setting():
    with SETTINGS.raytracing.override(texture_antialiasing=False):
        assert SETTINGS.raytracing.texture_antialiasing is False
    assert SETTINGS.raytracing.texture_antialiasing is True


@pytest.mark.parametrize('wrap', [(True, False), (False, True), (True, True)])
def test_closed_surface_mips_preserve_seams_and_unpadded_mean(wrap):
    from algan.mobs.surfaces.surface import wrap_pad_texture

    init_taichi()
    tex = torch.rand((1, 7, 5, 5), generator=torch.Generator().manual_seed(22))
    tex[..., 4] = 1
    padded = wrap_pad_texture(tex, wrap)
    pairs = []
    for rho in (0., 1., 1.5, 2., 4., 100.):
        du, dv = rho / 7, rho / 5
        if wrap[0]:
            pairs.extend([[0, 0., .31, du, dv], [0, 1., .31, du, dv]])
        if wrap[1]:
            pairs.extend([[0, .23, 0., du, dv], [0, .23, 1., du, dv]])
    out = _sample(padded, pairs, wrap=wrap)
    assert torch.allclose(out[::2], out[1::2], atol=2e-6)
    assert torch.allclose(out[-1], tex.mean((0, 1, 2)), atol=2e-6)
    # The legacy level-zero seam texel convention stays intact.
    if wrap[0]:
        out = _sample(padded, [[0, 2 / 7, 2 / 5 if wrap[1] else .5, 0., 0.]], wrap=wrap)
        assert torch.allclose(out[0], tex[0, 2, 2], atol=1e-6)


def test_old_18_column_metadata_remains_usable():
    init_taichi()
    tex = torch.rand((1, 4, 4, 5), generator=torch.Generator().manual_seed(26))
    meta, bank = _bank(tex)
    old = meta[:, :18].contiguous()
    out = torch.zeros((1, 5))
    _sample_probe(old, bank, torch.tensor([[0., .5, .5, 100., 100.]]), out)
    assert torch.allclose(out[0], tex[0, 1:3, 1:3].mean((0, 1)), atol=1e-7)


@ti.kernel
def _material_normal_probe(pos: ti.types.ndarray(), norm: ti.types.ndarray(),
                            extra: ti.types.ndarray(), col: ti.types.ndarray(),
                            uv: ti.types.ndarray(), meta: ti.types.ndarray(),
                            bank: ti.types.ndarray(), trim: ti.template(),
                            out: ti.types.ndarray()):
    for i in range(2):
        du = ti.cast(i, ti.f32)
        m, r, eta, transmission = _tri_material_g(
            trim, 0, 0, .8, .1, .1, extra, col, uv, meta, bank, 0, du, du)
        n = _tri_normal_g(trim, 0, 0, .8, .1, .1, norm, pos, uv, meta, bank, 0, du, du)
        out[i, 0], out[i, 1], out[i, 2], out[i, 3] = m, r, eta, transmission
        for c in ti.static(range(3)):
            out[i, 4 + c] = n[c]


@pytest.mark.parametrize('trim', [0, 1])
def test_material_and_normal_dispatch_use_their_own_map_sizes(trim):
    init_taichi()
    material = torch.rand((1, 8, 8, 5), generator=torch.Generator().manual_seed(32))
    normal = torch.zeros((1, 4, 4, 5))
    normal[..., 0] = .8
    normal[:, 1::2, :, 0] = -.8
    normal[..., 2] = .6
    parts, off = [], [0]
    meta = torch.full((1, 21), -1, dtype=torch.int32)
    meta[0, 10:13] = 1
    meta[0, 9] = 15
    for slot, image in ((1, material), (2, normal)):
        w, h = image.shape[1:3]
        meta[0, 3 * slot:3 * slot + 3] = torch.tensor([off[0], w, h])
        parts.append(image.reshape(1, -1, 5))
        off[0] += w * h
        meta[0, 18 + slot] = append_mip_pyramid(
            parts, off, image, color=False, linear=False, lerp=None)
    pos = torch.tensor([[[0., 0., 0., 1., 0., 0., 0., 1., 0.]]])
    norm = torch.tensor([[[0., 0., 1.] * 3]])
    uv = torch.tensor([[[0., 0., 1., 0., 0., 1.]]])
    out = torch.zeros((2, 7))
    _material_normal_probe(pos, norm, torch.zeros((1, 1, 12)),
                           torch.zeros(1, dtype=torch.int32), uv, meta,
                           torch.cat(parts, 1), trim, out)
    assert torch.allclose(out[1, :4], material.mean((0, 1, 2))[:4], atol=2e-6)
    assert torch.allclose(out[1, 4:], torch.tensor([0., 0., 1.]), atol=1e-6)
    assert abs(out[0, 4]) > .7, 'the level-zero normal-map probe was vacuous'
