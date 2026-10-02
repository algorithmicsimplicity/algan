"""Texture visibility and authoring agree with the production UV sampler."""

import pytest
import torch

from algan import SETTINGS, Scene, Surface
from algan.mobs.surfaces.surface import wrap_pad_texture
from algan.rendering.raytracing.primitives import RayTracedTrianglePrimitive
from algan.rendering.shaders.materials import _pack_material_texture
from algan.rendering.taichi_runtime import init_taichi
from tests.unit_tests.test_texture_antialiasing_taichi import _bank, _data_probe


@pytest.mark.parametrize("accelerator", ["cpu", "cuda", "mps"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("kind", ["static", "dense", "lerp"])
@pytest.mark.parametrize("opacity", [False, True])
def test_texture_visibility_across_devices_and_time_broadcasts(
    accelerator, reverse, kind, opacity
):
    if accelerator == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if accelerator == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    color_device, texture_device = (
        (accelerator, "cpu") if reverse else ("cpu", accelerator)
    )
    primitive = object.__new__(RayTracedTrianglePrimitive)
    frames = 1 if kind == "static" else 3
    texture = torch.zeros(frames, 2, 2, 5, device=texture_device)
    texture[..., 4] = torch.tensor(
        [1.0] if kind == "static" else [1.0, 1.0, 0.0], device=texture_device
    ).view(-1, 1, 1)
    if kind == "lerp":
        texture = texture[:2].unsqueeze(0)
        texture[:, 0, ..., 4] = 0
        primitive._rt_tex_lerp = torch.tensor(
            [[0.0, 1.0, 1.0], [0.0, 1.0, 1.0], [0.0, 1.0, 0.0]], device=color_device
        )
    primitive._rt_texture_map = texture
    if opacity:
        primitive._rt_tex_opacity = torch.tensor([1.0, 0.0, 1.0], device=color_device)
    colors = torch.zeros(3 if kind == "static" else 1, 2, 3, 5, device=color_device)
    lo = torch.zeros(1, 2, 3, device=color_device)
    hi = torch.ones_like(lo)
    primitive._pack_frame_visibility(lo, hi, colors, "audit visibility")
    actual = (primitive._rt_frame_hi >= primitive._rt_frame_lo).all(-1)
    expected = torch.tensor([True, not opacity, kind == "static"], device=color_device)
    assert actual.device == colors.device
    assert torch.equal(actual, expected[:, None].expand(3, 2))


def _sample_data(texture, queries, closed, antialiasing):
    init_taichi()
    meta, bank = _bank(wrap_pad_texture(texture, closed), color=False, wrap=closed)
    if not antialiasing:
        meta[0, 18] = -1
    result = torch.zeros(len(queries), 5)
    _data_probe(meta, bank, queries, result)
    return result


@pytest.mark.parametrize("antialiasing", [False, True])
@pytest.mark.parametrize(
    "closed", [(False, False), (True, False), (False, True), (True, True)]
)
def test_mixed_material_resolutions_match_independent_production_samples(
    antialiasing, closed
):
    generator = torch.Generator().manual_seed(246)
    low = torch.rand(2, 2, 3, 1, generator=generator)
    high = torch.rand(1, 6, 9, 1, generator=generator)
    with SETTINGS.raytracing.override(texture_antialiasing=antialiasing):
        packed, flags = _pack_material_texture(
            {"roughness": low, "reflectivity": high}, "cpu", closed_axes=closed
        )
        assert flags == 3
        axes = []
        for size, wraps in zip((6, 9), closed):
            axis = torch.arange(size, dtype=torch.float32)
            axes.append(
                axis / size
                if wraps
                else ((axis + 0.5) / size if antialiasing else axis / (size - 1))
            )
        queries = torch.tensor(
            [
                [frame, u, v, 0.0, 0.0]
                for frame in range(2)
                for u in axes[0]
                for v in axes[1]
            ]
        )
        actual = _sample_data(packed, queries, closed, antialiasing)
        low_map = torch.zeros(2, 2, 3, 5)
        low_map[..., 1:2] = low
        high_map = torch.zeros(1, 6, 9, 5)
        high_map[..., :1] = high
        low_sample = _sample_data(low_map, queries, closed, antialiasing)
        high_sample = _sample_data(high_map, queries, closed, antialiasing)
        torch.testing.assert_close(actual[:, 1], low_sample[:, 1], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(
            actual[:, 0], high_sample[:, 0], atol=1e-6, rtol=1e-5
        )


def test_single_open_texel_is_at_the_antialiased_surface_center():
    with Scene(), SETTINGS.raytracing.override(texture_antialiasing=True):
        surface = Surface(grid_width=2, grid_height=2)
        torch.testing.assert_close(
            surface.get_texture_locations((1, 1)), torch.zeros(1, 1, 3)
        )
