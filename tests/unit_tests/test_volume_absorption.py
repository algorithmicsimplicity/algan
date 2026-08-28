"""Volumetric absorption (Beer-Lambert) from material to kernel slot.

``MeshPhysicalMaterial.attenuation_color`` / ``attenuation_distance``
(KHR_materials_volume fields) reach the renderer as one absorption
coefficient -- ``sigma = -ln(clamp(linear(color), 1e-6, 1)) / distance`` --
packed in slots 27..29 of the built-in material parameter block, where the
wavefront bounce loop applies ``exp(-sigma * t_hit)`` to a ray leaving a
transmissive medium. What these tests pin:

* the coefficient is zero whenever the material does not attenuate (default
  fields, white colour, non-finite / non-positive distance) -- that is what
  makes the all-zeros value of a zero-padded custom-pipeline block mean
  "no absorption", the padding rule every slot obeys;
* the round trip: a ray crossing exactly ``attenuation_distance`` inside the
  medium comes out carrying exactly the *decoded* attenuation colour, and
  white comes out unchanged at any distance;
* the packed block actually carries the coefficient in its slots, and a
  non-physical material packs zeros.

Tensor assertions on the material and the primitive pack, no render and no
Taichi, so they are cheap; they are feature tests of this plumbing rather than
of anything the timeline or the Scene can break, so they stay out of the fast
suite. The rendered consequence is pinned by the renderer audit
(``benchmarks/renderer_audit/scenes/calib_absorption.json``).
"""

from __future__ import annotations

import math

import pytest
import torch

from algan import MeshPhysicalMaterial, MeshStandardMaterial, Octahedron, Off, Scene
from algan.rendering.raytracing.primitives import RayTracedTrianglePrimitive
from algan.rendering.raytracing.settings import _MAT_SLOTS
from algan.rendering.raytracing.shading_taichi import MAT_W
from algan.rendering.shaders.materials import _to_rgb
from algan.utils.color_space import srgb_to_linear

_ATTENUATION_COLOR = (0.45, 0.85, 0.55)
_ATTENUATION_DISTANCE = 1.0


def _decoded_attenuation_color():
    """Authored display-referred colour in the renderer's working space."""
    c = _to_rgb(_ATTENUATION_COLOR).detach().float().clone()
    return srgb_to_linear(c)


def _primitives(mob):
    build = getattr(mob, "get_render_primitives", None)
    got = build() if build is not None else None
    if got is None:
        return [p for child in mob.children for p in _primitives(child)]
    return got if isinstance(got, list) else [got]


def _pack(mob):
    merged = RayTracedTrianglePrimitive(triangle_collection=_primitives(mob))
    return merged._pack_material()


def _attenuation_sigma_of_material():
    """The coefficient the packer was given, recomputed from the same public
    call the pipeline itself uses.
    """
    mat = MeshPhysicalMaterial(
        attenuation_color=_ATTENUATION_COLOR,
        attenuation_distance=_ATTENUATION_DISTANCE,
    )
    return mat.get_shader_param_values()["attenuation_sigma"].detach()


def test_sigma_is_zero_when_the_material_does_not_attenuate():
    """No attenuation must pack as all-zeros: white attenuates nothing, an
    infinite or non-positive distance means no volume, and a default
    material sets neither field.
    """
    default = MeshPhysicalMaterial()
    assert (
        float(default.get_shader_param_values()["attenuation_sigma"].abs().sum()) == 0.0
    )

    white = MeshPhysicalMaterial(
        attenuation_color=(1.0, 1.0, 1.0), attenuation_distance=3.0
    )
    assert (
        float(white.get_shader_param_values()["attenuation_sigma"].abs().sum()) == 0.0
    )

    infinite = MeshPhysicalMaterial(
        attenuation_color=_ATTENUATION_COLOR, attenuation_distance=math.inf
    )
    assert (
        float(infinite.get_shader_param_values()["attenuation_sigma"].abs().sum())
        == 0.0
    )


@pytest.mark.parametrize("distance", [0.25, 1.0, 4.0])
def test_transmittance_at_one_distance_is_the_authored_colour(distance):
    """exp(-sigma * d) at d == attenuation_distance reproduces the DECODED
    attenuation colour -- the KHR_materials_volume definition -- and white
    transmits 1.0 at every distance.
    """
    mat = MeshPhysicalMaterial(
        attenuation_color=_ATTENUATION_COLOR,
        attenuation_distance=distance,
    )
    sigma = mat.get_shader_param_values()["attenuation_sigma"]
    transmittance = torch.exp(-sigma * distance)
    assert torch.allclose(transmittance, _decoded_attenuation_color(), atol=1e-6)

    white = MeshPhysicalMaterial(
        attenuation_color=(1.0, 1.0, 1.0), attenuation_distance=distance
    )
    assert torch.allclose(
        torch.exp(-white.get_shader_param_values()["attenuation_sigma"] * distance),
        torch.ones(3),
        atol=1e-7,
    )


def test_sigma_deepens_as_the_distance_shrinks():
    """A thinner medium absorbs more over the same path."""
    sigmas = [
        MeshPhysicalMaterial(
            attenuation_color=_ATTENUATION_COLOR, attenuation_distance=d
        ).get_shader_param_values()["attenuation_sigma"]
        for d in (0.5, 1.0, 2.0)
    ]
    for nearer, farther in zip(sigmas, sigmas[1:]):
        assert bool((nearer > farther).all())


def test_the_packed_block_carries_sigma_in_its_slots():
    """What the kernel reads: slots 27..29 hold the coefficient for a physical
    material and stay zero for one without the field.
    """
    start, width = _MAT_SLOTS["attenuation_sigma"]
    assert (start, width) == (27, 3)
    # Slots 30..32 (toon's num_bands, depth's near/far) were appended after
    # sigma, so it no longer ends the block -- but nothing may precede it.
    assert start + width < MAT_W

    with Scene(), Off():
        glass = Octahedron(edge_length=0.8)
        glass.set_material(
            MeshPhysicalMaterial(
                transmission=1.0,
                attenuation_color=_ATTENUATION_COLOR,
                attenuation_distance=_ATTENUATION_DISTANCE,
            )
        )
        plain = Octahedron(edge_length=0.8)
        plain.set_material(MeshStandardMaterial())
        for mob in (glass, plain):
            mob.spawn(animate=False)

        _glass_id, glass_mat = _pack(glass)
        _plain_id, plain_mat = _pack(plain)

    expected = _attenuation_sigma_of_material()
    assert glass_mat.shape[-1] == MAT_W
    assert torch.allclose(
        glass_mat[..., start : start + width].float(),
        expected.view(1, 1, 3).expand_as(glass_mat[..., start : start + width]).float(),
        atol=1e-6,
    )
    assert float(plain_mat[..., start : start + width].float().abs().sum()) == 0.0


def test_decoded_colour_matches_scene_builder_decode():
    """Sigma logs the colour in the same space the render boundary decodes to
    (``scene_builder._decode_merged_colors`` under linear_color_space), so the
    identity above holds against the decode the kernels actually see.
    """
    from algan.rendering.raytracing import settings as rt_settings

    assert rt_settings.linear_color_space
    expected = srgb_to_linear(torch.tensor(_ATTENUATION_COLOR))
    assert torch.allclose(_decoded_attenuation_color(), expected, atol=1e-6)
