"""Packaging and interpolation invariants of the bounded glass lookup."""

import numpy as np
import pytest

from algan.rendering.raytracing import glass_energy


def test_shipped_glass_table_has_a_complete_bounded_layout():
    table = glass_energy.glass_energy_table()
    assert table.shape == glass_energy.GLASS_ENERGY_SHAPE
    assert table.dtype == np.float32
    assert table.flags.c_contiguous
    assert np.isfinite(table).all()
    assert ((0.0 <= table) & (table <= 1.0)).all()
    assert glass_energy.glass_energy_table() is table


def test_cosine_means_integrate_the_decoded_directional_interpolant():
    table = glass_energy.glass_energy_table().reshape(
        glass_energy.GLASS_ROUGHNESS_SIZE,
        glass_energy.GLASS_IOR_SIZE,
        glass_energy.GLASS_COSINE_SIZE + 1,
        2,
    )
    # Independent Gaussian integration per sqrt(mu) interval. The linear
    # interpolant times 4*u^3 is degree four, so three points integrate it
    # exactly (through degree five), unlike midpoint averaging.
    x, weights = np.polynomial.legendre.leggauss(3)
    edges = np.linspace(0.0, 1.0, glass_energy.GLASS_COSINE_SIZE)
    total = np.zeros(table.shape[:2] + (2,))
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        for xx, weight in zip(x, weights):
            u = (a + b) / 2 + xx * (b - a) / 2
            t = (u - a) / (b - a)
            value = (1 - t) * table[:, :, i, :] + t * table[:, :, i + 1, :]
            total += value * (4 * u**3) * weight * (b - a) / 2
    assert np.allclose(total, table[:, :, -1, :], atol=1e-7, rtol=1e-6)
    assert not table[0].any(), "the delta roughness row must have no compensation"


def test_missing_glass_asset_is_not_a_silent_single_scatter_fallback(
    monkeypatch, tmp_path
):
    glass_energy.glass_energy_table.cache_clear()
    try:
        monkeypatch.setattr(glass_energy, "__file__", str(tmp_path / "glass_energy.py"))
        with pytest.raises(FileNotFoundError):
            glass_energy.glass_energy_table()
    finally:
        glass_energy.glass_energy_table.cache_clear()
