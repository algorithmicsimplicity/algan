"""Physical and distribution-level checks of the unified glass interface."""

import math

import numpy as np
import pytest
import torch

from algan.rendering.raytracing.glass_energy import glass_energy_table
from algan.rendering.raytracing.path_tracer_taichi import (
    _NM_GLASS_LUT,
    NEE_META_WIDTH,
    _pt_glass_f_pdf,
    _pt_glass_terms,
    _pt_sample_glass,
)
from algan.rendering.taichi_runtime import init_taichi
from algan.taichi_compat import ti


def _glass_meta():
    meta = torch.cat((torch.zeros(NEE_META_WIDTH), glass_energy_table()))
    meta[_NM_GLASS_LUT] = NEE_META_WIDTH
    return meta


@ti.kernel
def _sample(
    rough: ti.f32,
    eta: ti.f32,
    angle: ti.f32,
    u: ti.types.ndarray(),
    out: ti.types.ndarray(),
    nee_meta: ti.types.ndarray(),
):
    for i in range(u.shape[0]):
        n = ti.math.vec3(0.0, 0.0, 1.0)
        rd = ti.math.vec3(ti.sin(angle), 0.0, -ti.cos(angle))
        one = ti.math.vec3(1.0, 1.0, 1.0)
        ratio = (eta - 1.0) / (eta + 1.0)
        f0 = one * ratio * ratio
        wi, wt, trans, delta, valid = _pt_sample_glass(
            rd,
            n,
            rough,
            eta,
            f0,
            0.0,
            one,
            1.0,
            ti.math.vec2(u[i, 0], u[i, 1]),
            u[i, 2],
            nee_meta,
        )
        if delta == 0:
            fc, pdf = _pt_glass_f_pdf(
                f0, rough, n, rd, wi, eta, 0.0, one, 1.0, nee_meta
            )
            wt = fc / ti.max(pdf, 1e-20)
        for k in ti.static(range(3)):
            out[i, k] = wi[k]
        out[i, 3] = trans
        out[i, 4] = valid
        out[i, 5] = wt[0] if valid else 0.0


@ti.kernel
def _evaluate(
    rough: ti.f32,
    eta: ti.f32,
    angle: ti.f32,
    directions: ti.types.ndarray(),
    out: ti.types.ndarray(),
    reverse: ti.i32,
    nee_meta: ti.types.ndarray(),
):
    for i in range(directions.shape[0]):
        n = ti.math.vec3(0.0, 0.0, 1.0)
        rd = ti.math.vec3(ti.sin(angle), 0.0, -ti.cos(angle))
        wi = ti.math.vec3(directions[i, 0], directions[i, 1], directions[i, 2])
        eta_i = eta
        if reverse != 0:
            saved = rd
            rd = -wi
            wi = -saved
            n = -n
            eta_i = 1.0 / eta
        one = ti.math.vec3(1.0, 1.0, 1.0)
        ratio = (eta_i - 1.0) / (eta_i + 1.0)
        fc, pdf = _pt_glass_f_pdf(
            one * ratio * ratio, rough, n, rd, wi, eta_i, 0.0, one, 1.0, nee_meta
        )
        out[i, 0] = fc[0]
        out[i, 1] = pdf


def _draw(rough, eta=1 / 1.5, angle=30, count=32768):
    init_taichi()
    u = torch.rand((count, 3), generator=torch.Generator().manual_seed(9187))
    out = torch.zeros((count, 6))
    _sample(rough, eta, math.radians(angle), u, out, _glass_meta())
    return out


@pytest.mark.parametrize(("eta", "angle"), [(1 / 1.5, 30), (1.5, 30), (1.5, 50)])
def test_glass_pdf_mass_matches_sampled_outcomes(eta, angle):
    """Quadrature covers each hemisphere, including rejected microfacets."""
    rough = 0.65
    sampled = _draw(rough, eta, angle)
    # Gauss-Legendre in cos(theta), midpoint quadrature in azimuth. This
    # integrates solid-angle density independently of the VNDF sampler.
    z, weights = np.polynomial.legendre.leggauss(192)
    phi = (np.arange(256) + 0.5) * (2 * math.pi / 256)
    radius = np.sqrt(1 - z * z)[:, None]
    dirs = np.stack(
        np.broadcast_arrays(radius * np.cos(phi), radius * np.sin(phi), z[:, None]),
        axis=-1,
    ).reshape(-1, 3)
    out = torch.zeros((len(dirs), 2))
    _evaluate(
        rough,
        eta,
        math.radians(angle),
        torch.tensor(dirs, dtype=torch.float32),
        out,
        0,
        _glass_meta(),
    )
    density = out[:, 1].reshape(192, 256).numpy()
    for transmitted in (False, True):
        half = z < 0 if transmitted else z > 0
        integrated = (density[half] * weights[half, None]).sum() * (2 * math.pi / 256)
        observed = (
            ((sampled[:, 3] == transmitted) & (sampled[:, 4] == 1)).float().mean()
        )
        assert integrated == pytest.approx(float(observed), abs=0.015)
        for near_horizon in (False, True):
            for positive_x in (False, True):
                z_mask = half & ((np.abs(z) < 0.5) == near_horizon)
                phi_mask = (np.cos(phi) > 0) == positive_x
                expected = (
                    density[z_mask][:, phi_mask] * weights[z_mask, None]
                ).sum() * (2 * math.pi / 256)
                in_bin = (
                    (sampled[:, 3] == transmitted)
                    & (sampled[:, 4] == 1)
                    & ((sampled[:, 2].abs() < 0.5) == near_horizon)
                    & ((sampled[:, 0] > 0) == positive_x)
                )
                assert expected == pytest.approx(
                    float(in_bin.float().mean()), abs=0.012
                )


@pytest.mark.parametrize("eta", [1 / 1.5, 1.5, 1.5 / 1.33])
def test_glass_sample_weights_obey_the_furnace_energy_bound(eta):
    sampled = _draw(0.6, eta, 35)
    # Refraction changes radiance by eta^2, not radiant power. Removing that
    # factor gives the power furnace. A mixture sample can individually weigh
    # more than one; conservation constrains the integral, not each f/pdf.
    weights = sampled[:, 5] / torch.where(sampled[:, 3] > 0, eta**2, 1.0)
    assert bool(torch.isfinite(weights).all())
    assert float(weights.mean()) == pytest.approx(1.0, abs=0.015)


def test_roughness_broadens_transmitted_directions():
    narrow = _draw(0.08)
    broad = _draw(0.65)

    def angular_variance(samples):
        directions = samples[(samples[:, 3] == 1) & (samples[:, 4] == 1), :3]
        return float(directions.var(0).sum())

    assert angular_variance(broad) > 20 * angular_variance(narrow)


def test_rough_facets_can_transmit_above_the_macro_critical_angle():
    sampled = _draw(0.65, 1.5, 50)
    valid = sampled[sampled[:, 4] == 1]
    assert float((valid[:, 3] == 1).float().mean()) > 0.05
    assert float((valid[:, 3] == 0).float().mean()) > 0.2


def test_smooth_glass_obeys_snell_and_total_internal_reflection():
    eta = 1 / 1.5
    sampled = _draw(0.001, eta, 30, count=4096)
    transmitted = sampled[(sampled[:, 3] == 1) & (sampled[:, 4] == 1)]
    expected = torch.tensor([eta * 0.5, 0, -math.sqrt(1 - (eta * 0.5) ** 2)])
    assert torch.allclose(
        transmitted[:, :3], expected.expand_as(transmitted[:, :3]), atol=1e-6
    )
    assert float(transmitted[:, 5].mean()) == pytest.approx(eta**2, abs=1e-6)
    tir = _draw(0.001, 1.5, 50, count=4096)
    assert bool((tir[:, 3] == 0).all())
    assert bool((tir[:, 4] == 1).all())
    assert torch.allclose(tir[:, 5], torch.ones(len(tir)), atol=1e-6)


def test_transmission_reciprocity_includes_the_radiance_eta_factor():
    eta, angle, rough = 1 / 1.5, 35, 0.6
    sampled = _draw(rough, eta, angle, count=1024)
    directions = sampled[(sampled[:, 3] == 1) & (sampled[:, 4] == 1), :3].contiguous()
    forward = torch.zeros((len(directions), 2))
    reverse = torch.zeros_like(forward)
    _evaluate(rough, eta, math.radians(angle), directions, forward, 0, _glass_meta())
    _evaluate(rough, eta, math.radians(angle), directions, reverse, 1, _glass_meta())
    f_forward = forward[:, 0] / directions[:, 2].abs()
    f_reverse = reverse[:, 0] / math.cos(math.radians(angle))
    assert torch.allclose(f_forward, eta**2 * f_reverse, rtol=2e-4, atol=1e-5)


@ti.kernel
def _glass_terms(
    f0_scale: ti.f32,
    eta: ti.f32,
    cosines: ti.types.ndarray(),
    out: ti.types.ndarray(),
):
    one = ti.math.vec3(1.0, 1.0, 1.0)
    for i in range(cosines.shape[0]):
        R, trans, _pr = _pt_glass_terms(one * f0_scale, cosines[i], eta, 0.0, one, 1.0)
        out[i, 0] = R[0]
        out[i, 1] = trans[0]


@pytest.mark.parametrize("eta", [1.5, 1 / 1.5])
def test_authored_specular_cannot_make_a_facet_return_more_than_it_receives(eta):
    """Reflection is the Schlick-remapped *authored* F0 while transmission is
    the exact ``1 - F``, so an authored specular above the interface's own base
    reflectance pairs a boosted reflection with an unreduced transmission.

    ``MeshPhysicalMaterial`` writes that F0 from ``specular_intensity`` and
    ``specular_color``, neither of which is clamped to the KHR [0, 1] range.
    The single-scatter cap remains necessary before the coupled compensation
    splits off its common ideal dielectric fraction. Compensation must not
    excuse an already active authored facet.
    """
    init_taichi()
    base = ((eta - 1) / (eta + 1)) ** 2
    cosines = torch.linspace(0.02, 1.0, 25)
    boosted = torch.zeros((cosines.numel(), 2))
    _glass_terms(0.5, eta, cosines, boosted)
    assert base < 0.5, "the probe must actually author above the physical base"
    assert float((boosted[:, 0] + boosted[:, 1]).max()) <= 1.0 + 1e-5

    # The default authoring is exactly energy-preserving, and stays so: at
    # ``specular_intensity = 1`` with a white specular colour the reflection is
    # exact Fresnel and the clamp must not bite.
    default = torch.zeros((cosines.numel(), 2))
    _glass_terms(base, eta, cosines, default)
    total = default[:, 0] + default[:, 1]
    assert torch.allclose(total, torch.ones_like(total), atol=1e-5)
