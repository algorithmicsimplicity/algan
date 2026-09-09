"""Energy, reciprocity, and distribution checks of the two-sided glass closure.

These are unclamped BSDF integrals, not display-white pixel comparisons. In
radiance mode the transmission integrand is divided by eta**2 for a power
furnace (equivalently, the two hemispheres have equilibrium L/n**2).
"""

import itertools
import math

import numpy as np
import pytest
import torch

from algan.rendering.raytracing.glass_energy import (
    GLASS_MU_SIZE,
    GLASS_TABLE_SHAPE,
    glass_energy_table,
)
from algan.rendering.raytracing.glass_energy_taichi import (
    _glass_compensated_fraction,
    _glass_losses,
)
from algan.rendering.raytracing.path_tracer_taichi import (
    _NM_GLASS_LUT,
    NEE_META_WIDTH,
    _pt_glass_f_pdf,
    _pt_glass_ss_f_pdf,
    _pt_glass_terms,
    _pt_sample_glass,
    _pt_sample_glass_ss,
)
from algan.rendering.taichi_runtime import init_taichi
from algan.taichi_compat import ti


def _meta():
    meta = torch.cat((torch.zeros(NEE_META_WIDTH), glass_energy_table()))
    meta[_NM_GLASS_LUT] = NEE_META_WIDTH
    return meta


@ti.kernel
def _furnace(
    cases: ti.types.ndarray(),
    samples: ti.types.ndarray(),
    meta: ti.types.ndarray(),
    out: ti.types.ndarray(),
):
    for c, i in ti.ndrange(cases.shape[0], samples.shape[0]):
        rough, eta, angle = cases[c, 0], cases[c, 1], cases[c, 2]
        n = ti.math.vec3(0.0, 0.0, 1.0)
        rd = ti.math.vec3(ti.sin(angle), 0.0, -ti.cos(angle))
        one = ti.math.vec3(1.0, 1.0, 1.0)
        f0 = one * ti.pow((eta - 1.0) / (eta + 1.0), 2.0)
        u = ti.math.vec2(samples[i, 0], samples[i, 1])
        wi, tint, trans, delta, valid = _pt_sample_glass(
            rd, n, rough, eta, f0, 0.0, one, 1.0, u, samples[i, 2], meta
        )
        if delta == 0:
            fc, pdf = _pt_glass_f_pdf(f0, rough, n, rd, wi, eta, 0.0, one, 1.0, meta)
            tint = fc / ti.max(pdf, 1e-20)
        out[c, i, 0] = tint[0] / (eta * eta if trans else 1.0) if valid else 0.0
        # Independently integrate the unmodified SS model using its sampler.
        ws, ts, xs, ds, vs = _pt_sample_glass_ss(
            rd, n, rough, eta, f0, 0.0, one, 1.0, u, samples[i, 2]
        )
        if ds == 0:
            fs, ps = _pt_glass_ss_f_pdf(f0, rough, n, rd, ws, eta, 0.0, one, 1.0)
            ts = fs / ti.max(ps, 1e-20)
        out[c, i, 1] = ts[0] / (eta * eta if xs else 1.0) if vs else 0.0
        out[c, i, 2] = out[c, i, 0] if trans == 0 else 0.0
        out[c, i, 3] = out[c, i, 0] if trans != 0 else 0.0


def _run_furnaces(rows, count=32768):
    init_taichi()
    cases = torch.tensor([[r, e, math.radians(a)] for r, e, a in rows])
    samples = torch.quasirandom.SobolEngine(3, scramble=True, seed=418).draw(count)
    out = torch.zeros((len(rows), count, 4))
    _furnace(cases, samples, _meta(), out)
    return out


def test_white_furnace_roughness_ior_and_incidence_sweep():
    rows = list(
        itertools.product(
            (0.35, 0.65, 1.0),
            (1 / 2.4, 1 / 1.5, 1 / 1.1, 1.1, 1.5 / 1.33, 1.5, 2.4),
            (0.0, 35.0, 60.0, 85.0),
        )
    )
    out = _run_furnaces(rows)
    assert torch.isfinite(out).all()
    means = out.mean(dim=1)
    errors = (means[:, 0] - 1).abs()
    worst = int(errors.argmax())
    assert float(errors.max()) < 0.015, (rows[worst], means[worst].tolist())
    assert float(means[:, 1].min()) < 0.6, (
        "sweep must expose substantial old energy loss"
    )
    assert float(means[:, 0].min()) > float(means[:, 1].min()) + 0.35
    # Both outcomes share the recovered energy; no reflection-only gain.
    assert torch.allclose(means[:, 2] + means[:, 3], means[:, 0], atol=1e-6)


@ti.kernel
def _reciprocity(
    directions: ti.types.ndarray(),
    eta: ti.f32,
    meta: ti.types.ndarray(),
    out: ti.types.ndarray(),
):
    for i in range(directions.shape[0]):
        n = ti.math.vec3(0.0, 0.0, 1.0)
        v = ti.math.vec3(0.6, 0.0, 0.8)
        wi = ti.math.vec3(directions[i, 0], directions[i, 1], directions[i, 2])
        one = ti.math.vec3(1.0, 1.0, 1.0)
        f0 = one * ti.pow((eta - 1.0) / (eta + 1.0), 2.0)
        forward, _ = _pt_glass_f_pdf(f0, 0.93, n, -v, wi, eta, 0.0, one, 1.0, meta)
        n_rev, eta_rev = n, eta
        if wi[2] < 0.0:
            n_rev, eta_rev = -n, 1.0 / eta
        reverse, _ = _pt_glass_f_pdf(
            f0, 0.93, n_rev, -wi, v, eta_rev, 0.0, one, 1.0, meta
        )
        out[i, 0] = forward[0] / ti.abs(wi[2])
        out[i, 1] = reverse[0] / v[2] * (eta * eta if wi[2] < 0 else 1.0)


@pytest.mark.parametrize("eta", [1 / 2.4, 1 / 1.5, 1.5 / 1.33, 1.5, 2.4])
def test_both_hemispheres_are_reciprocal_including_multi_scatter_only_support(eta):
    init_taichi()
    z = np.concatenate((np.linspace(-1, -0.02, 29), np.linspace(0.02, 1, 29)))
    phi = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    r = np.sqrt(1 - z * z)[:, None]
    directions = np.stack(
        np.broadcast_arrays(r * np.cos(phi), r * np.sin(phi), z[:, None]), axis=-1
    )
    directions = torch.tensor(directions.reshape(-1, 3), dtype=torch.float32)
    out = torch.zeros((len(directions), 2))
    _reciprocity(directions, eta, _meta(), out)
    assert torch.isfinite(out).all()
    assert torch.all(out[:, 0] > 0), (
        "the closure has support beyond one-facet directions"
    )
    assert torch.allclose(out[:, 0], out[:, 1], rtol=6e-4, atol=3e-5)


def test_table_averages_exactly_integrate_the_interpolated_profiles():
    table = glass_energy_table().numpy().reshape(GLASS_TABLE_SHAPE)
    mu = np.linspace(0, 1, GLASS_MU_SIZE)
    a, b = mu[:-1], mu[1:]
    left = (b - a) * (2 * a + b) / 3
    right = (b - a) * (a + 2 * b) / 3
    expected = (table[..., :-2] * left + table[..., 1:-1] * right).sum(-1)
    np.testing.assert_allclose(table[..., -1], expected, atol=4e-8)
    np.testing.assert_array_equal(table[:, 0], 0)
    np.testing.assert_array_equal(table[0, :, 0], table[0, :, 1])
    np.testing.assert_array_equal(table[-1, :, 0], table[-1, :, 1])


@ti.kernel
def _colour_budget(
    materials: ti.types.ndarray(), cosines: ti.types.ndarray(), out: ti.types.ndarray()
):
    for i, j in ti.ndrange(materials.shape[0], cosines.shape[0]):
        eta, m, T, scale = (
            materials[i, 0],
            materials[i, 1],
            materials[i, 2],
            materials[i, 3],
        )
        albedo = ti.math.vec3(materials[i, 4], materials[i, 5], materials[i, 6])
        base = ti.pow((eta - 1) / (eta + 1), 2.0)
        f0 = ti.math.vec3(base, base, base) * ((1.0 - m) * scale) + m * albedo
        colour = _glass_compensated_fraction(f0, eta, m, albedo, T)
        R, trans, _ = _pt_glass_terms(f0, cosines[j], eta, m, albedo, T)
        one = ti.math.vec3(1.0, 1.0, 1.0)
        F, Ft, _ = _pt_glass_terms(one * base, cosines[j], eta, 0.0, one, 1.0)
        for k in ti.static(range(3)):
            out[i, j, k] = R[k] - colour[k] * F[k]
            out[i, j, k + 3] = trans[k] - colour[k] * Ft[k]
            out[i, j, k + 6] = colour[k]


def test_tints_partial_transmission_and_authored_specular_keep_a_passive_remainder():
    init_taichi()
    materials = torch.tensor(
        [
            [eta, m, T, scale, *albedo]
            for eta, m, T, scale, albedo in itertools.product(
                (1 / 1.5, 1.5),
                (0.0, 0.35, 1.0),
                (0.0, 0.3, 1.0),
                (0.0, 0.5, 1.0, 8.0, 30.0),
                ((1.0, 1.0, 1.0), (0.1, 0.4, 0.8)),
            )
        ]
    )
    out = torch.zeros((len(materials), 129, 9))
    _colour_budget(materials, torch.linspace(0, 1, 129), out)
    assert float(out[:, :, :6].min()) >= -2e-6
    residual = out[:, :, :3] + out[:, :, 3:6]
    assert torch.all(residual <= 1 - out[:, :, 6:] + 2e-6)
    assert torch.all(out[materials[:, 1] == 1, :, 6:] == 0), (
        "opaque metal is not whitened"
    )
    assert torch.all(out[materials[:, 2] == 0, :, 6:] == 0), (
        "nontransmitting material is not whitened"
    )


@ti.kernel
def _smooth_parity(
    samples: ti.types.ndarray(),
    rough: ti.f32,
    eta: ti.f32,
    meta: ti.types.ndarray(),
    out: ti.types.ndarray(),
):
    for i in range(samples.shape[0]):
        n = ti.math.vec3(0.0, 0.0, 1.0)
        rd = ti.math.vec3(0.8, 0.0, -0.6)
        one = ti.math.vec3(1.0, 1.0, 1.0)
        f0 = one * ti.pow((eta - 1.0) / (eta + 1.0), 2.0)
        u = ti.math.vec2(samples[i, 0], samples[i, 1])
        w, t, x, d, v = _pt_sample_glass(
            rd, n, rough, eta, f0, 0.0, one, 1.0, u, samples[i, 2], meta
        )
        ws, ts, xs, ds, vs = _pt_sample_glass_ss(
            rd, n, rough, eta, f0, 0.0, one, 1.0, u, samples[i, 2]
        )
        out[i, 0] = (w - ws).norm() + (t - ts).norm()
        out[i, 1] = ti.abs(x - xs) + ti.abs(d - ds) + ti.abs(v - vs)
        f, pdf = _pt_glass_f_pdf(f0, rough, n, rd, w, eta, 0.0, one, 1.0, meta)
        out[i, 2] = f.norm() + pdf


@pytest.mark.parametrize("rough", [0.0, 0.001, 0.0099])
@pytest.mark.parametrize("eta", [1 / 1.5, 1.0, 1.5 / 1.33, 1.5, 2.4])
def test_smooth_glass_matches_the_old_sampler_exactly(rough, eta):
    init_taichi()
    u = torch.quasirandom.SobolEngine(3).draw(1024)
    out = torch.zeros((len(u), 3))
    _smooth_parity(u, rough, eta, _meta(), out)
    assert torch.equal(out, torch.zeros_like(out))


@ti.kernel
def _lookup_probe(
    eta: ti.f32, mu: ti.f32, meta: ti.types.ndarray(), out: ti.types.ndarray()
):
    base = ti.cast(meta[_NM_GLASS_LUT] + 0.5, ti.i32)
    a = _glass_losses(meta, base, 0.734, eta, mu)
    b = _glass_losses(meta, base, 0.734, 1.0 / eta, mu)
    for k in ti.static(range(4)):
        out[0, k] = a[k]
        out[1, k] = b[k]


def test_lookup_is_reciprocal_and_uses_the_recorded_offset():
    init_taichi()
    normal = _meta()
    padded = torch.cat(
        (normal[:NEE_META_WIDTH], torch.ones(4096), normal[NEE_META_WIDTH:])
    )
    padded[_NM_GLASS_LUT] += 4096
    a, b = torch.zeros((2, 4)), torch.zeros((2, 4))
    _lookup_probe(1.5 / 1.33, 0.271, normal, a)
    _lookup_probe(1.5 / 1.33, 0.271, padded, b)
    assert torch.equal(a, b)
    assert torch.allclose(a[0], a[1, [1, 0, 3, 2]], atol=1e-7)
