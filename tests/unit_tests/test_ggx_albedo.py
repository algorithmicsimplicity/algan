"""The path tracer's opaque GGX albedo table (``ggx_albedo``).

The Turquin compensation divides by this table's ``E1``, so it has to be the
albedo of the lobe ``_pt_lit_f_pdf`` evaluates -- ``D * G2 / (4 n.v)`` with the
exact height-correlated Smith ``G2`` and ``alpha = roughness**2`` -- and not of
some other GGX variant. Checked here against a brute-force hemisphere
quadrature that shares no code with the generator (which uses visible-normal
sampling).
"""

from __future__ import annotations

import numpy as np
import pytest

from algan.rendering.raytracing.ggx_albedo import (
    GGX_COSINE_SIZE,
    GGX_ROUGHNESS_SIZE,
    ggx_albedo_table,
    pt_energy_tables,
)
from algan.rendering.raytracing.glass_energy import (
    GLASS_ENERGY_SHAPE,
    glass_energy_table,
)


def _smith_lambda(mu, alpha):
    c2 = np.clip(mu * mu, 1e-8, 1.0)
    return 0.5 * (np.sqrt(1.0 + alpha * alpha * (1.0 - c2) / c2) - 1.0)


def _albedo_by_quadrature(roughness, mu, n=600):
    """``(A, B)`` by a midpoint rule over the reflected hemisphere."""
    alpha = max(roughness * roughness, 1e-4)
    theta = (np.arange(n) + 0.5) / n * (np.pi / 2)
    phi = (np.arange(2 * n) + 0.5) / (2 * n) * (2 * np.pi)
    t, p = np.meshgrid(theta, phi, indexing="ij")
    light = np.stack((np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)), axis=-1)
    view = np.array([np.sqrt(1.0 - mu * mu), 0.0, mu])
    half = light + view
    half /= np.linalg.norm(half, axis=-1, keepdims=True)
    n_dot_h = half[..., 2]
    d = alpha * alpha / (np.pi * (n_dot_h * n_dot_h * (alpha * alpha - 1.0) + 1.0) ** 2)
    g2 = 1.0 / (1.0 + _smith_lambda(mu, alpha) + _smith_lambda(light[..., 2], alpha))
    f = d * g2 / (4.0 * mu)
    fc = (1.0 - np.clip(half @ view, 0.0, 1.0)) ** 5
    dw = np.sin(t) * (np.pi / 2 / n) * (2 * np.pi / (2 * n))
    return float(((1.0 - fc) * f * dw).sum()), float((fc * f * dw).sum())


@pytest.mark.parametrize(
    ("r_index", "c_index"), [(8, 32), (16, 32), (32, 32), (16, 16), (32, 16), (24, 8)]
)
def test_the_table_is_the_albedo_of_the_exact_smith_lobe(r_index, c_index):
    table = ggx_albedo_table().reshape(GGX_ROUGHNESS_SIZE, GGX_COSINE_SIZE, 2)
    roughness = r_index / (GGX_ROUGHNESS_SIZE - 1)
    mu = (c_index / (GGX_COSINE_SIZE - 1)) ** 2
    a, b = _albedo_by_quadrature(roughness, mu)
    assert table[r_index, c_index, 0] == pytest.approx(a, abs=2e-3)
    assert table[r_index, c_index, 1] == pytest.approx(b, abs=2e-3)


def test_the_karis_fit_is_not_this_lobe():
    """Why the table exists: at roughness 0.5, normal incidence, the exact
    lobe keeps 0.916 of white light, where the split-sum fit says 0.725 --
    and ``1 + (1 - E1) / E1`` built on 0.725 makes a white metal 26% too bright.
    """
    table = ggx_albedo_table().reshape(GGX_ROUGHNESS_SIZE, GGX_COSINE_SIZE, 2)
    e1 = float(table[16, -1].sum())
    assert e1 == pytest.approx(0.916, abs=2e-3)
    assert e1 / (1.0 - 0.55 * 0.5) > 1.2


def test_the_ggx_table_rides_after_the_glass_table():
    combined = pt_energy_tables()
    rows = GLASS_ENERGY_SHAPE[0]
    assert combined.dtype == np.float32
    np.testing.assert_array_equal(combined[:rows], glass_energy_table())
    np.testing.assert_array_equal(combined[rows:], ggx_albedo_table())
