"""The path tracer's MeshPhysicalMaterial clearcoat and sheen lobes, at the
BSDF level.

The path tracer used to read none of slots 17-23 (``clearcoat``,
``clearcoat_roughness``, ``sheen``, ``sheen_roughness``, ``sheen_color``), so
both layers vanished from every path-traced render. They are now lobes of the
lit BSDF: ``_pt_lit_lobes`` reads them into a layer vector, ``_pt_lit_f_pdf``
evaluates them for next-event estimation and for the MIS pdf, and
``_pt_sample_layer`` draws them in the continuation. These probes hold the
two ends to each other (each lobe's sampler against its own evaluator,
integrated over the sphere independently of the sampler) and hold the
layered BSDF to energy conservation. The renders that show the layers on
screen, and a furnace, are in ``test_path_tracer.py``.

No ``from __future__ import annotations``: the probe kernels'
``ti.types.ndarray()`` annotations are evaluated at run time.
"""

import math

import numpy as np
import pytest
import torch

from algan.rendering.raytracing.ggx_albedo import pt_energy_tables
from algan.rendering.raytracing.path_tracer_taichi import (
    _PL_W_COAT,
    _PL_W_SHEEN,
    _pt_coat_f_pdf,
    _pt_lit_f_pdf,
    _pt_lit_lobes,
    _pt_sample_layer,
    _pt_sheen_f_pdf,
)
from algan.rendering.raytracing.shading_taichi import _MID_PHYSICAL, MAT_W
from algan.rendering.taichi_runtime import init_taichi
from algan.taichi_compat import ti


def _physical_params(
    clearcoat=0.0,
    clearcoat_roughness=0.0,
    sheen=0.0,
    sheen_roughness=1.0,
    sheen_color=(0.0, 0.0, 0.0),
):
    """One built-in physical material block, as the merge packs it."""
    params = torch.zeros((1, 1, MAT_W))
    params[0, 0, 12] = 1.5  # ior
    params[0, 0, 13] = 1.0  # specular_intensity
    params[0, 0, 14:17] = 1.0  # specular_color
    params[0, 0, 17] = clearcoat
    params[0, 0, 18] = clearcoat_roughness
    params[0, 0, 19] = sheen
    params[0, 0, 20] = sheen_roughness
    params[0, 0, 21:24] = torch.tensor(sheen_color)
    return params


@ti.kernel
def _evaluate(
    glass_energy: ti.types.ndarray(),
    params: ti.types.ndarray(),
    albedo: ti.f32,
    rough: ti.f32,
    angle: ti.f32,
    dirs: ti.types.ndarray(),
    out: ti.types.ndarray(),
):
    """Per direction: the whole lit BSDF's ``f_cos`` (red channel) and
    mixture pdf, then the coat's and the sheen's own ``f_cos`` and pdf.
    """
    for i in range(dirs.shape[0]):
        n = ti.math.vec3(0.0, 0.0, 1.0)
        rd = ti.math.vec3(ti.sin(angle), 0.0, -ti.cos(angle))
        wi = ti.math.vec3(dirs[i, 0], dirs[i, 1], dirs[i, 2])
        alb = ti.math.vec3(albedo, albedo, albedo)
        e_diff, e_spec, e_trans, f0, rough_eff, layer = _pt_lit_lobes(
            glass_energy,
            _MID_PHYSICAL,
            params,
            0,
            0,
            alb,
            0.0,
            rough,
            1.5,
            0.0,
            n,
            rd,
            0.0,
        )
        w_diff = ti.max(e_diff[0], ti.max(e_diff[1], e_diff[2]))
        w_spec = ti.max(e_spec[0], ti.max(e_spec[1], e_spec[2]))
        fc, pdf = _pt_lit_f_pdf(
            glass_energy,
            e_diff,
            e_spec,
            f0,
            rough_eff,
            n,
            rd,
            wi,
            0.0,
            w_diff,
            w_spec,
            0.0,
            0.0,
            0.0,
            alb,
            0.0,
            layer,
        )
        fc_c, pc_c = _pt_coat_f_pdf(layer, n, rd, wi)
        fc_s, pc_s = _pt_sheen_f_pdf(layer, n, rd, wi)
        out[i, 0] = fc[0]
        out[i, 1] = pdf
        out[i, 2] = fc_c
        out[i, 3] = pc_c
        out[i, 4] = fc_s[0]
        out[i, 5] = pc_s


@ti.kernel
def _lobe_weights(
    energy: ti.types.ndarray(),
    params: ti.types.ndarray(),
    albedo: ti.f32,
    rough: ti.f32,
    angle: ti.f32,
    out: ti.types.ndarray(),
):
    """``(w_diff, w_spec, w_coat, w_sheen)``: the continuation's selection
    weights at alpha 1.
    """
    n = ti.math.vec3(0.0, 0.0, 1.0)
    rd = ti.math.vec3(ti.sin(angle), 0.0, -ti.cos(angle))
    alb = ti.math.vec3(albedo, albedo, albedo)
    e_diff, e_spec, _t, _f0, _r, layer = _pt_lit_lobes(
        energy, _MID_PHYSICAL, params, 0, 0, alb, 0.0, rough, 1.5, 0.0, n, rd, 0.0
    )
    out[0] = ti.max(e_diff[0], ti.max(e_diff[1], e_diff[2]))
    out[1] = ti.max(e_spec[0], ti.max(e_spec[1], e_spec[2]))
    out[2] = layer[_PL_W_COAT]
    out[3] = layer[_PL_W_SHEEN]


@ti.kernel
def _sample(
    energy: ti.types.ndarray(),
    params: ti.types.ndarray(),
    angle: ti.f32,
    coat: ti.i32,
    u: ti.types.ndarray(),
    out: ti.types.ndarray(),
):
    """Draw the coat or the sheen lobe: ``(weight, ok)`` per sample, the
    weight being that lobe's ``f_cos / pdf`` (0 when rejected).
    """
    for i in range(u.shape[0]):
        n = ti.math.vec3(0.0, 0.0, 1.0)
        rd = ti.math.vec3(ti.sin(angle), 0.0, -ti.cos(angle))
        one = ti.math.vec3(1.0, 1.0, 1.0)
        _d, _s, _t, _f0, _r, layer = _pt_lit_lobes(
            energy, _MID_PHYSICAL, params, 0, 0, one, 0.0, 0.5, 1.5, 0.0, n, rd, 0.0
        )
        _wi, weight, ok = _pt_sample_layer(
            layer, n, rd, ti.math.vec2(u[i, 0], u[i, 1]), coat
        )
        out[i, 0] = weight[0] if ok == 1 else 0.0
        out[i, 1] = ti.cast(ok, ti.f32)


def _quadrature(nz=160, nphi=256):
    """Upper-hemisphere directions and solid-angle weights: Gauss-Legendre
    in cos(theta), midpoint in azimuth -- independent of every sampler.
    """
    x, w = np.polynomial.legendre.leggauss(nz)
    z = 0.5 * (x + 1.0)
    wz = 0.5 * w
    phi = (np.arange(nphi) + 0.5) * (2 * math.pi / nphi)
    r = np.sqrt(1 - z * z)[:, None]
    dirs = np.stack(
        np.broadcast_arrays(r * np.cos(phi), r * np.sin(phi), z[:, None]), -1
    ).reshape(-1, 3)
    weights = np.broadcast_to(wz[:, None] * (2 * math.pi / nphi), (nz, nphi))
    return torch.tensor(dirs, dtype=torch.float32), weights.reshape(-1)


def _integrals(params, albedo=1.0, rough=0.5, angle_deg=30.0):
    """Hemisphere integrals of the evaluator's six columns."""
    init_taichi()
    dirs, weights = _quadrature()
    out = torch.zeros((len(dirs), 6))
    _evaluate(
        torch.from_numpy(pt_energy_tables()),
        params,
        albedo,
        rough,
        math.radians(angle_deg),
        dirs,
        out,
    )
    return (out.double().numpy() * weights[:, None]).sum(0)


def _draws(params, angle_deg, coat, count=65536):
    init_taichi()
    u = torch.rand((count, 2), generator=torch.Generator().manual_seed(4211))
    out = torch.zeros((count, 2))
    _sample(
        torch.from_numpy(pt_energy_tables()),
        params,
        math.radians(angle_deg),
        coat,
        u,
        out,
    )
    return out.double().numpy()


@pytest.mark.parametrize(
    ("clearcoat_roughness", "angle"), [(0.3, 0.0), (0.3, 60.0), (0.7, 35.0)]
)
def test_clearcoat_sampler_matches_its_evaluator(clearcoat_roughness, angle):
    """The coat's VNDF draws and ``_pt_coat_f_pdf`` describe one lobe.

    The mean sampled weight (``f_cos / pdf`` per draw, rejected draws 0) must
    equal the evaluator's ``f_cos`` integrated over the hemisphere -- the
    continuation then carries the coat's real albedo -- and the accepted
    fraction must equal its pdf's integral, so the MIS weight an emitter hit
    reads after a coat bounce is the density that actually produced it.
    """
    params = _physical_params(clearcoat=1.0, clearcoat_roughness=clearcoat_roughness)
    sums = _integrals(params, angle_deg=angle)
    draws = _draws(params, angle, coat=1)
    assert sums[2] > 0.01, "the coat evaluated to nothing"
    assert draws[:, 0].mean() == pytest.approx(sums[2], rel=0.02)
    assert draws[:, 1].mean() == pytest.approx(sums[3], abs=0.01)


@pytest.mark.parametrize(
    ("sheen_roughness", "angle"), [(0.3, 60.0), (0.5, 30.0), (1.0, 0.0)]
)
def test_sheen_sampler_matches_its_evaluator(sheen_roughness, angle):
    """The same two identities for the sheen's cosine-weighted draws and
    ``_pt_sheen_f_pdf``; the pdf integrates to one (nothing is rejected).
    """
    params = _physical_params(
        sheen=1.0, sheen_roughness=sheen_roughness, sheen_color=(1.0, 1.0, 1.0)
    )
    sums = _integrals(params, angle_deg=angle)
    draws = _draws(params, angle, coat=0)
    assert sums[4] > 0.05, "the sheen evaluated to nothing"
    assert draws[:, 0].mean() == pytest.approx(sums[4], rel=0.03)
    assert sums[5] == pytest.approx(1.0, abs=1e-3)
    assert draws[:, 1].mean() == 1.0


@pytest.mark.parametrize("angle", [0.0, 30.0, 60.0, 80.0])
@pytest.mark.parametrize("clearcoat_roughness", [0.3, 0.6, 1.0])
def test_clearcoat_never_adds_energy_to_a_white_base(angle, clearcoat_roughness):
    """A coat on a white dielectric base reflects no more than the bare base.

    Three.js lays the coat over the base as ``base * (1 - cc F(n.v)) + cc *
    coat``: what the coat reflects it takes from what reaches the base. The
    hemisphere integral of the layered BSDF (``_pt_lit_f_pdf``, the response
    next-event estimation and MIS read) must therefore not exceed the
    uncoated one -- nor one -- while the coat must still do something.
    """
    bare = _integrals(_physical_params(), angle_deg=angle)
    coated = _integrals(
        _physical_params(clearcoat=1.0, clearcoat_roughness=clearcoat_roughness),
        angle_deg=angle,
    )
    assert coated[0] <= max(bare[0], 1.0) + 2e-3, (
        f"coated albedo {coated[0]:.4f} exceeds the bare base's {bare[0]:.4f}"
    )
    assert coated[2] > 0.01, "the coat reflected nothing"


@pytest.mark.parametrize("angle", [0.0, 45.0, 75.0, 85.0])
def test_sheen_and_clearcoat_together_stay_energy_conserving(angle):
    """Sheen on top: the base is scaled down by what the fibres reflect
    (Three.js's albedo scaling), so a white base with a white sheen at the
    default ``sheen_roughness`` (1) and a coat reflects at most what the bare
    base does.

    Not at every sheen roughness: the scaling's albedo is Three.js's
    ``IBLSheenBRDF`` fit, which falls short of the Charlie x Neubelt lobe's
    own albedo at grazing angles once ``sheen_roughness`` is low (0.79
    against 0.33 at 85 degrees and 0.3), so there the lobe adds more than
    the base gives up -- Three.js's direct lighting does the same.
    """
    params = _physical_params(
        clearcoat=1.0,
        clearcoat_roughness=0.5,
        sheen=1.0,
        sheen_roughness=1.0,
        sheen_color=(1.0, 1.0, 1.0),
    )
    sums = _integrals(params, angle_deg=angle)
    bare = _integrals(_physical_params(), angle_deg=angle)
    assert sums[4] > 0.01
    assert sums[0] <= max(bare[0], 1.0) + 5e-3, (
        f"layered albedo {sums[0]:.4f} at {angle} degrees"
    )


@pytest.mark.parametrize("angle", [0.0, 50.0])
def test_mixture_pdf_carries_every_lobe_at_its_selection_probability(angle):
    """``_pt_lit_f_pdf``'s mixture is the continuation's: each lobe's own
    pdf integral times its selection weight over the weight sum (diffuse and
    sheen integrate to one; the two GGX lobes lose what VNDF draws below the
    horizon, which the coat's own integral measures).
    """
    params = _physical_params(
        clearcoat=1.0,
        clearcoat_roughness=0.4,
        sheen=1.0,
        sheen_roughness=0.5,
        sheen_color=(1.0, 1.0, 1.0),
    )
    init_taichi()
    w = torch.zeros(4)
    _lobe_weights(
        torch.from_numpy(pt_energy_tables()), params, 1.0, 0.5, math.radians(angle), w
    )
    w_diff, w_spec, w_coat, w_sheen = (float(x) for x in w)
    assert w_coat > 0
    assert w_sheen > 0
    sums = _integrals(params, angle_deg=angle)
    spec_only = _integrals(_physical_params(), albedo=0.0, angle_deg=angle)
    total = w_diff + w_spec + w_coat + w_sheen
    expected = (
        w_diff + w_spec * spec_only[1] + w_coat * sums[3] + w_sheen * sums[5]
    ) / total
    assert sums[1] == pytest.approx(expected, abs=5e-3)
