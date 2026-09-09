"""End-to-end UV minification, including virtual-camera mirror footprints.

No external render baselines: a deeply minified black/white checkerboard has
an independently known linear-light mean of 0.5. A constant map with that mean
is the image oracle. These tests are deliberately outside the fast suite.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from benchmarks._texture_antialiasing import checker, render_frame


@pytest.mark.parametrize(
    ("spp", "reflection", "analytic"),
    [
        (1, False, True),
        (1, True, True),
        (4, False, True),
        (4, True, True),
        (1, False, False),
        (1, True, False),
    ],
)
def test_texture_minification_matches_independent_mean(
    tmp_path, spp, reflection, analytic
):
    def render(name, enabled, texture=None):
        image, _ = render_frame(
            tmp_path / name, spp, enabled, reflection, texture, analytic=analytic
        )
        return image[26:38, 26:38].astype(np.float32)

    bilinear = render("bilinear.png", False)
    filtered = render("filtered.png", True)
    flat = checker()
    # Authored sRGB corresponding to the physical mean of black and white.
    flat[..., :3] = 1.055 * (0.5 ** (1 / 2.4)) - 0.055
    oracle = render("mean.png", True, flat)
    assert filtered.mean() > 80, "the textured surface was not seen"
    assert filtered.std() < bilinear.std() * 0.1, (filtered.std(), bilinear.std())
    assert np.max(np.abs(filtered - oracle)) <= 2
    if reflection:
        # The off-screen walls must reach these pixels only via the mirror.
        flat[..., :3] = 0
        dark = render("black_walls.png", True, flat)
        assert dark.max() <= 2


def test_subpixel_motion_does_not_crawl(tmp_path):
    images = {}
    for enabled in (False, True):
        for step, shift in enumerate((0.0, 0.013)):
            image, _ = render_frame(
                tmp_path / f"{enabled}_{step}.png", 1, enabled, shift=shift
            )
            images[enabled, step] = image[26:38, 26:38].astype(np.float32)
    filtered_change = np.abs(images[True, 1] - images[True, 0]).mean()
    legacy_change = np.abs(images[False, 1] - images[False, 0]).mean()
    assert legacy_change > 2
    assert filtered_change < legacy_change * 0.1


def test_endpoint_pyramids_match_dense_nonlinear_animation():
    from algan.rendering.raytracing.texture_mips import build_mip_levels

    endpoints = torch.rand((1, 3, 9, 5, 5), generator=torch.Generator().manual_seed(19))
    rows = torch.tensor([[0, 1, 0.2], [1, 2, 0.7], [2, 0, 0.4]])
    frames = torch.stack(
        [
            endpoints[0, int(a)] + w * (endpoints[0, int(b)] - endpoints[0, int(a)])
            for a, b, w in rows
        ]
    )
    expected = build_mip_levels(frames, color=True, linear=True)
    actual = build_mip_levels(endpoints, color=True, linear=True, lerp=rows)
    for a, b in zip(actual, expected):
        assert torch.allclose(a, b, atol=2e-6)


def test_closed_surface_declaration_survives_primitive_collection():
    from algan import SceneManager, Sphere
    from algan.rendering.primitives.triangle_primitive import TrianglePrimitive

    SceneManager.reset()
    sphere = Sphere(color_texture=checker(8))
    primitive = sphere.get_render_primitives()
    assert primitive.texture_wrap == (True, False)
    collection = TrianglePrimitive(triangle_collection=[primitive])
    assert collection.texture_wrap == primitive.texture_wrap


def test_legacy_shared_time_bank_does_not_replicate_animation_quadratically():
    from algan.rendering.raytracing.texture_mips import append_mip_pyramid

    frames, size = 7, 8
    tex = torch.rand(
        (frames, size, size, 5), generator=torch.Generator().manual_seed(41)
    )
    parts, off = [tex.reshape(frames, -1, 5)], [size * size]
    table = append_mip_pyramid(
        parts, off, tex, color=False, linear=False, lerp=None, time_flat=False
    )
    bank = torch.cat([x.expand(frames, -1, -1) for x in parts], 1)
    last = bank[0, table, 4].view(torch.int32).item()
    descriptor = bank[0, table + last].view(torch.int32)
    offset, width, height, time_length, _ = descriptor.tolist()
    assert (width, height, time_length) == (1, 1, 1)
    assert torch.allclose(bank[:, offset], tex.mean((1, 2)), atol=2e-6)
    assert bank.shape == (frames, 64 + 16 + 4 + 1 + 4, 5)


@pytest.mark.parametrize("shape", [(3, 3), (7, 5), (1, 257), (257, 129)])
def test_npot_full_image_footprint_reaches_the_mean(shape):
    from algan.rendering.raytracing.texture_mips import build_mip_levels

    tex = torch.rand((1, *shape, 5), generator=torch.Generator().manual_seed(47))
    levels = build_mip_levels(tex)
    # The sampler uses log2(rho): rounding UP at each level would leave a
    # residual two-texel pattern even when a pixel spans the complete image.
    lod = np.log2(max(shape))
    assert lod >= len(levels)
    assert torch.allclose(levels[-1][:, 0, 0], tex.mean((1, 2)), atol=2e-6)
