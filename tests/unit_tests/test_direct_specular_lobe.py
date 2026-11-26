"""The direct lights' share of the reflected specular lobe.

A hit's outgoing energy is partitioned by the scatter sites into a reflected
share ``R`` (traced as a continuation ray), a transmitted share, and a
remainder that weights the locally shaded colour. That partition is sound as
*reflectance* -- but the continuation carrying ``R`` is a ray, and a ray can
only find light that has geometry to hit. A directional or point light is a
delta: no continuation will ever land on it, however many bounces it is given.
So the reflected lobe's response to the direct lights exists only as the
analytic GGX term the material stages evaluate, and that term rides inside the
shaded colour -- which is weighted by the share that is explicitly *not*
reflected.

Clear glass is the case that surfaced it. For ``transmission = 1`` the
transmitted share is ``1 - R`` exactly, so the remainder collapses to
``R * (1 - _mirror_share(roughness))`` -- 1.2% of the lobe at roughness 0.05.
A mirror shows the glass ball's *neighbours through it*, tinted once entering
and once leaving, and nothing of the ball's own reflection at all. Measured on
``benchmarks/renderer_audit/scenes/matlight_pbr_subset.json`` against
three-gpu-pathtracer on a black background, the ball's own disc read
(0.034, 0.063, 0.013) at g/r 1.87 -- the albedo ratio squared -- where the
reference reads g/r 1.07.

:func:`algan.rendering.raytracing.shading_taichi.direct_specular_lobe` is the
term added back at the complement of the share the shaded colour already
carries. What this file pins is the two things that make that safe:

* it really is the same GGX lobe the stage computes, not a second opinion --
  which is what makes "shaded share + traced share = 1" an identity rather
  than an approximation, and
* it prepares its normal exactly as ``_run_frag_pipeline`` does before calling
  a stage. Handing it the raw interpolated normal is not a small error: on a
  back-facing hit ``n . v`` falls to the ``1e-4`` clamp and the lobe divides by
  it, so the term explodes instead of vanishing.

Outside the fast suite: it compiles a Taichi kernel, and nothing outside the
shading stages can break it (see ``tests/README.md``).

Note the absent ``from __future__ import annotations``: the probe kernel's
``ti.types.ndarray()`` annotations are evaluated at run time, and stringifying
them stops the kernel compiling.
"""

import pytest
import torch

from algan.rendering.raytracing.settings import _MAT_DEFAULTS, _MAT_SLOTS
from algan.rendering.raytracing.shading_taichi import (
    MAT_W,
    _prep_normal,
    _sided_shading_normal,
    _stage_standard,
    direct_specular_lobe,
)
from algan.taichi_compat import ti

LIGHT_RGB = (1.0, 1.0, 1.0)


@ti.kernel
def _probe(
    nrm: ti.types.ndarray(),
    albedo: ti.types.ndarray(),
    params: ti.types.ndarray(),
    light_pos: ti.types.ndarray(),
    light_col: ti.types.ndarray(),
    metalness: ti.f32,
    roughness: ti.f32,
    ior: ti.f32,
    stage_out: ti.types.ndarray(),
    lobe_out: ti.types.ndarray(),
):
    for i in range(nrm.shape[0]):
        p = ti.math.vec3(0.0, 0.0, 0.0)
        vd = ti.math.vec3(0.0, 0.0, 1.0)
        n = ti.math.vec3(nrm[i, 0], nrm[i, 1], nrm[i, 2]).normalized()
        rgb = ti.math.vec3(albedo[i, 0], albedo[i, 1], albedo[i, 2])
        vis = ti.Vector([1.0] * 3)
        # Exactly what ``_run_frag_pipeline`` does before it calls a stage: a
        # stage never sees a raw interpolated normal. Skipping this here would
        # compare the lobe against a differently-oriented stage and call the
        # disagreement a bug in the lobe.
        sn = _sided_shading_normal(n, n, vd, params, 0, 0)
        sn = _prep_normal(sn, n, params[0, 0, 10], vd)
        s = _stage_standard(
            p, vd, sn, n, rgb, 0.0, params, 0, 0, 0, light_pos, light_col, 1, 0, vis, p
        )
        lobe = direct_specular_lobe(
            0,
            0,
            p,
            vd,
            n,
            n,
            metalness,
            roughness,
            ior,
            rgb,
            params,
            light_pos,
            light_col,
            1,
            0,
            vis,
        )
        for k in ti.static(range(3)):
            stage_out[i, k] = s[k]
            lobe_out[i, k] = lobe[k]


def _params(**overrides):
    """A one-primitive material block with ``overrides`` applied by slot name."""
    block = torch.tensor(_MAT_DEFAULTS, dtype=torch.float32).view(1, 1, MAT_W).clone()
    for name, value in overrides.items():
        start, width = _MAT_SLOTS[name]
        block[0, 0, start : start + width] = torch.tensor(
            [value] * width if not isinstance(value, (list, tuple)) else list(value),
            dtype=torch.float32,
        )
    return block


def evaluate(normals, *, metalness, roughness, albedo, light=(0.3, 0.5, 1.0)):
    """Stage output and the standalone lobe, for each surface normal.

    The point sits at the origin, viewed from +Z, with one distant white light
    along ``light``. ``metalness``/``roughness`` are written into the material
    block AND passed to the lobe, which takes them as transport arguments.
    """
    k = len(normals)
    nrm = torch.tensor(normals, dtype=torch.float32)
    alb = torch.tensor([albedo] * k, dtype=torch.float32)
    # ``one_sided`` (slot 26) is declared by the GEOMETRY, not the material, so
    # it has no ``_MAT_SLOTS`` name; the defaults leave it two-sided, which is
    # what makes the back-facing case below a normal FLIP rather than an
    # inside. Both paths have to agree, so both are exercised.
    params = _params(
        roughness=roughness,
        metalness=metalness,
        flat_shading=0.0,
        env_map_intensity=1.0,
    )
    lp = torch.tensor([[[c * 1e4 for c in light]]], dtype=torch.float32)
    lc = torch.tensor([[list(LIGHT_RGB)]], dtype=torch.float32)
    stage = torch.zeros((k, 3), dtype=torch.float32)
    lobe = torch.zeros((k, 3), dtype=torch.float32)
    _probe(nrm, alb, params, lp, lc, metalness, roughness, 1.5, stage, lobe)
    return stage, lobe


# Normals spanning head-on to grazing, plus one facing away from the viewer.
FRONT = [(0.0, 0.0, 1.0), (0.3, 0.0, 0.95), (0.6, 0.2, 0.77), (0.9, 0.1, 0.42)]
BACK = [(0.0, 0.0, -1.0), (0.3, 0.0, -0.95)]


@pytest.mark.parametrize("roughness", [0.35, 0.1, 0.05])
def test_lobe_is_the_stages_own_specular_term(roughness):
    """A BLACK albedo on a NON-metal kills the stage's diffuse and its ambient
    both -- the ambient is ``(rgb * (1 - m) + f0 * m) * amb``, which at
    ``m = 0`` is ``rgb * amb`` and so vanishes with the albedo. What is left is
    the specular lobe alone, which must be exactly what the standalone lobe
    returns.

    This is the identity the add-back's weighting depends on. The shaded share
    and the traced share are complements, so they sum to one only if both are
    weighting the *same* term; a lobe that is merely similar would deposit
    energy the surface never reflected. The sweep covers both sides of the
    ``_mirror_share`` throttle -- at roughness 0.35 one mirror ray may carry
    only 3% of the lobe, at 0.05 it carries 99%.
    """
    stage, lobe = evaluate(
        FRONT, metalness=0.0, roughness=roughness, albedo=(0.0, 0.0, 0.0)
    )
    assert torch.allclose(stage, lobe, atol=1e-6), (
        f"the standalone lobe is not the stage's specular term at "
        f"roughness {roughness}:\n"
        f"stage {stage.tolist()}\nlobe  {lobe.tolist()}"
    )


@pytest.mark.parametrize(("metalness", "roughness"), [(0.4, 0.35), (1.0, 0.1)])
def test_a_metals_stage_exceeds_its_lobe_by_a_view_independent_constant(
    metalness, roughness
):
    """A metal under a black albedo can still keep an ambient term, because its
    F0 blend leaves the dielectric floor standing: ``f0 * m * amb``. Whether
    that residue is present depends on the ambient strength the colour space
    sets, so its VALUE is not the invariant -- its view-independence is. Being
    ambient, it cannot vary with where the viewer stands, so the gap between
    the stage and the lobe must be the SAME at every normal. A gap that varied
    with geometry would mean the two disagree about the specular lobe itself,
    which is the failure this pins.

    Asserting the residue's magnitude here instead would pin a value that
    belongs to the colour space's ambient strength, not to the lobe. (An
    earlier note here blamed the offline cache for two runs across one
    ``@ti.func`` edit reporting 9.6e-5 and 0; the cache is keyed on the
    kernel's IR and does invalidate on such an edit, so whatever moved that
    number was not a stale kernel.)
    """
    stage, lobe = evaluate(
        FRONT, metalness=metalness, roughness=roughness, albedo=(0.0, 0.0, 0.0)
    )
    gap = (stage - lobe)[:, 0]
    assert (gap >= -1e-7).all(), (
        f"the lobe exceeds the stage's whole output: {gap.tolist()}"
    )
    assert torch.allclose(gap, gap[0].expand_as(gap), atol=1e-7), (
        f"the stage-minus-lobe gap varies with the view direction, so the two "
        f"do not agree about the specular lobe: {gap.tolist()}"
    )


def test_a_coloured_metals_lobe_is_bounded_by_its_stage_output():
    """With a coloured albedo the stage adds diffuse and ambient on top, so the
    lobe can no longer be equal to it -- but it must still be a *part* of it,
    never larger. A lobe that exceeds the stage's whole output is the signature
    of the two disagreeing about geometry (a normal prepared differently, a
    roughness read from a different slot), which is what makes the add-back
    deposit energy the surface never reflected.
    """
    stage, lobe = evaluate(FRONT, metalness=0.4, roughness=0.35, albedo=(1.0, 0.1, 0.1))
    assert (lobe <= stage + 1e-5).all(), (
        f"the lobe exceeds the stage's total output:\n"
        f"stage {stage.tolist()}\nlobe  {lobe.tolist()}"
    )
    assert (lobe > 0).any(), "a metal at roughness 0.35 must have some lobe"


def test_a_back_facing_hit_does_not_explode():
    """The regression this file was written for.

    ``_stage_standard`` receives a normal already oriented by
    ``_sided_shading_normal``; the scatter sites hand the lobe the RAW
    interpolated one. Without repeating that preparation the lobe divides by
    the ``n . v >= 1e-4`` clamp on every back-facing hit and returns a value
    orders of magnitude above the stage's. Two-sided geometry flips the normal
    to face the viewer, so both must agree there too.
    """
    stage, lobe = evaluate(BACK, metalness=0.0, roughness=0.35, albedo=(0.0, 0.0, 0.0))
    assert torch.isfinite(lobe).all(), f"lobe is not finite: {lobe.tolist()}"
    assert (lobe < 1e3).all(), (
        f"a back-facing hit produced a runaway lobe -- the normal was not "
        f"prepared as the stage prepares it: {lobe.tolist()}"
    )
    assert torch.allclose(stage, lobe, atol=1e-6), (
        f"back-facing hit disagrees with the stage:\n"
        f"stage {stage.tolist()}\nlobe  {lobe.tolist()}"
    )


def test_the_lobe_is_off_for_a_non_pbr_material():
    """``metalness < 0`` is the sentinel for legacy/unlit materials, which have
    no PBR specular lobe at all -- ``_material_reflectance`` returns ``R = 0``
    for them, so there is no reflected share to restore and the add-back must
    contribute exactly nothing.
    """
    _stage, lobe = evaluate(
        FRONT, metalness=-1.0, roughness=0.35, albedo=(1.0, 0.5, 0.2)
    )
    assert (lobe == 0).all(), f"a non-PBR material grew a lobe: {lobe.tolist()}"


def test_an_opaque_metal_keeps_its_highlight_when_the_render_has_bounces(tmp_path):
    """The add-back reaches every surface that hands a share to a ray, not
    only a transmissive one.

    It used to be gated on transmission, so an opaque metal -- whose reflected
    share is most of its lobe -- kept only the remainder of its own highlight:
    this sphere rendered at half the light with the default bounces that it
    did with none, and a mirror at almost nothing. Against black there is
    nothing for the reflected ray to find, so the two renders may differ only
    by the ambient fill the ray replaces, which is a few bytes along the limb.
    """
    import numpy as np
    from PIL import Image

    from algan import (
        BLACK,
        SETTINGS,
        WHITE,
        Off,
        PointLight,
        Scene,
        Sphere,
        VideoSettings,
    )
    from algan.rendering.shaders.materials import MeshStandardMaterial

    bounces = SETTINGS.raytracing.max_bounces
    assert bounces > 0, "the default render is expected to trace reflections"

    def render(name, max_bounces):
        SETTINGS.raytracing.set(shadows=False, max_bounces=max_bounces)
        with Scene() as scene:
            with Off():
                sphere = Sphere(radius=1.2)
                sphere.set_material(
                    MeshStandardMaterial(color=WHITE, metalness=1.0, roughness=0.3)
                )
                sphere.spawn()
                PointLight(
                    location=torch.tensor([2.0, 2.0, 4.0]), intensity=40.0
                ).spawn()
            result = scene.save_frame(
                tmp_path / name,
                VideoSettings(resolution=(96, 64), supersampling=1),
                overwrite=True,
                background=BLACK,
            )
        with Image.open(result.output_path) as image:
            return np.asarray(image.convert("RGB")).astype(np.float64)

    unbounced = render("unbounced.png", 0)
    bounced = render("bounced.png", bounces)

    assert unbounced.max() == 255, "the test sphere has no highlight to lose"
    kept = bounced.sum() / unbounced.sum()
    assert kept > 0.9, (
        f"with bounces the metal shows {kept:.0%} of the light it shows without"
    )
    assert (bounced <= unbounced + 2).all(), (
        "the restored highlight is brighter than the surface's own shading"
    )
