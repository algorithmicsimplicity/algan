"""Blinn-Phong's specular lobe, against the formula three.js actually uses.

``MeshPhongMaterial`` is a declared copy of three.js's, so three.js's
``BRDF_BlinnPhong`` is the specification rather than a second opinion. It is

    F_Schlick(specularColor, 1, V.H) * G_BlinnPhong_Implicit * D_BlinnPhong
    G_BlinnPhong_Implicit = 0.25
    D_BlinnPhong          = RECIPROCAL_PI * (shininess*0.5 + 1) * (N.H)^shininess

scaled by the irradiance's ``N.L``. Algan's lobe had none of the three factors
that are not the power term: no ``0.25 * (shininess*0.5 + 1)``, no
multiplicative ``N.L`` (only a boolean gate), and no Fresnel. The visible cost
was a highlight 4x too weak relative to its own diffuse at the default
shininess of 30, and 10.25x at 80 -- and one that sharpened without
brightening, where a normalized ``D`` concentrates the same energy as it
narrows.

**The ``1/pi`` is deliberately absent from Algan's version**, and that is the
point of :func:`test_specular_to_diffuse_ratio_matches_three_js`: Algan's light
unit is pi times three.js's, so its *diffuse* lobe drops the same ``1/pi``.
What has to match three.js is the ratio between the two lobes, because that
ratio is what the highlight looks like. Dropping the factor from one lobe only
is the bug this file exists to catch.

Outside the fast suite: it compiles a Taichi kernel, and nothing outside the
shading stages can break it (see ``tests/README.md``).

Note the absent ``from __future__ import annotations``: the probe kernel's
``ti.types.ndarray()`` annotations are evaluated at run time, and stringifying
them stops the kernel compiling.
"""

import math

import pytest
import taichi as ti
import torch

from algan.rendering.raytracing.settings import _MAT_DEFAULTS, _MAT_SLOTS
from algan.rendering.raytracing.shading_taichi import MAT_W, _stage_phong

# One white light, one white surface, straight-on geometry unless stated.
LIGHT_RGB = (1.0, 1.0, 1.0)


@ti.kernel
def _probe(
    pos: ti.types.ndarray(),
    view: ti.types.ndarray(),
    nrm: ti.types.ndarray(),
    albedo: ti.types.ndarray(),
    params: ti.types.ndarray(),
    light_pos: ti.types.ndarray(),
    light_col: ti.types.ndarray(),
    out: ti.types.ndarray(),
):
    for i in range(pos.shape[0]):
        p = ti.math.vec3(pos[i, 0], pos[i, 1], pos[i, 2])
        vd = ti.math.vec3(view[i, 0], view[i, 1], view[i, 2]).normalized()
        n = ti.math.vec3(nrm[i, 0], nrm[i, 1], nrm[i, 2]).normalized()
        rgb = ti.math.vec3(albedo[i, 0], albedo[i, 1], albedo[i, 2])
        r = _stage_phong(
            p,
            vd,
            n,
            n,
            rgb,
            0.0,
            params,
            0,
            0,
            0,
            light_pos,
            light_col,
            1,
            0,
            0,
            ti.math.vec3(0.0, 0.0, 0.0),
        )
        out[i, 0] = r[0]
        out[i, 1] = r[1]
        out[i, 2] = r[2]


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


def shade(
    normals, *, shininess, specular, albedo=(0.0, 0.0, 0.0), light=(0.0, 0.0, 1.0)
):
    """Shade a point at the origin, viewed from +Z, for each surface normal.

    ``albedo`` black isolates the specular lobe (the diffuse term is
    ``albedo * ...``); a non-black albedo brings the diffuse term back so the
    two can be compared.
    """
    k = len(normals)
    pos = torch.zeros((k, 3), dtype=torch.float32)
    view = torch.tensor([(0.0, 0.0, 1.0)] * k, dtype=torch.float32)
    nrm = torch.tensor(normals, dtype=torch.float32)
    alb = torch.tensor([albedo] * k, dtype=torch.float32)
    params = _params(shininess=shininess, specular=list(specular))
    # A light far along ``light`` so its direction at the origin is ``light``.
    lp = torch.tensor([[[c * 1e4 for c in light]]], dtype=torch.float32)
    lc = torch.tensor([[list(LIGHT_RGB)]], dtype=torch.float32)
    out = torch.zeros((k, 3), dtype=torch.float32)
    _probe(pos, view, nrm, alb, params, lp, lc, out)
    return out


def three_js_specular(n_dot_h, n_dot_l, v_dot_h, shininess, specular):
    """three.js's ``BRDF_BlinnPhong * irradiance``, with the ``1/pi`` removed.

    The removal is Algan's light-unit convention, applied to both lobes (see
    the module docstring); every other factor is three.js's, verbatim.
    """
    f = specular + (1.0 - specular) * (1.0 - v_dot_h) ** 5
    d = (shininess * 0.5 + 1.0) * n_dot_h**shininess
    return f * 0.25 * d * n_dot_l


@pytest.mark.parametrize("shininess", [1.0, 5.0, 30.0, 80.0, 200.0])
def test_specular_lobe_matches_three_js_brdf(shininess):
    """Head-on, the lobe equals three.js's BRDF term for term."""
    got = shade([(0.0, 0.0, 1.0)], shininess=shininess, specular=(1.0, 1.0, 1.0))
    # Light, view and normal all along +Z, so N.H = N.L = V.H = 1.
    expected = three_js_specular(1.0, 1.0, 1.0, shininess, 1.0)
    assert got[0, 0].item() == pytest.approx(expected, rel=2e-3)


def test_the_lobe_carries_the_normalization_it_used_to_be_missing():
    """The factor that was absent is exactly ``0.25 * (shininess*0.5 + 1)``.

    Pinned as a ratio between two shininess values so it cannot be satisfied by
    a constant fudge: the old bare ``(N.H)^s`` gave the same head-on value at
    every shininess, and the normalized lobe does not.
    """
    a = shade([(0.0, 0.0, 1.0)], shininess=30.0, specular=(1.0, 1.0, 1.0))[0, 0]
    b = shade([(0.0, 0.0, 1.0)], shininess=80.0, specular=(1.0, 1.0, 1.0))[0, 0]
    # (80*0.5 + 1) / (30*0.5 + 1) = 41/16
    assert (b / a).item() == pytest.approx(41.0 / 16.0, rel=2e-3)


def test_specular_to_diffuse_ratio_matches_three_js():
    """The ratio between the lobes is three.js's, which is what the pi is about.

    three.js: specular ``0.25*(s/2+1)/pi``, diffuse ``albedo/pi`` -- the pi
    cancels, leaving ``0.25*(s/2+1)/albedo``. Algan drops the pi from both, so
    it must land on the same number.
    """
    albedo = 0.5
    shininess = 30.0
    lit = shade(
        [(0.0, 0.0, 1.0)],
        shininess=shininess,
        specular=(1.0, 1.0, 1.0),
        albedo=(albedo, albedo, albedo),
    )[0, 0].item()
    diffuse_only = shade(
        [(0.0, 0.0, 1.0)],
        shininess=shininess,
        specular=(0.0, 0.0, 0.0),
        albedo=(albedo, albedo, albedo),
    )[0, 0].item()
    spec = lit - diffuse_only
    # The diffuse lobe alone still carries the ambient fill and the energy
    # budget, so take the diffuse from the same render rather than from theory.
    assert spec / diffuse_only == pytest.approx(
        0.25 * (shininess * 0.5 + 1.0) / albedo, rel=0.05
    )


def test_a_light_behind_the_surface_contributes_no_highlight():
    """``N.L`` multiplies the lobe now, so it dies at the terminator rather
    than being switched off by a boolean at it.
    """
    behind = shade(
        [(0.0, 0.0, 1.0)],
        shininess=30.0,
        specular=(1.0, 1.0, 1.0),
        light=(0.0, 0.0, -1.0),
    )
    assert behind[0, 0].item() == pytest.approx(0.0, abs=1e-6)


def test_the_lobe_falls_off_smoothly_toward_the_terminator():
    """With ``N.L`` multiplicative the highlight fades as the light grazes;
    the boolean gate it replaced held full height right up to the edge.
    """
    angles = [0.0, 30.0, 60.0, 85.0]
    normals = [
        (math.sin(math.radians(a)), 0.0, math.cos(math.radians(a))) for a in angles
    ]
    got = shade(normals, shininess=5.0, specular=(1.0, 1.0, 1.0))
    values = [got[i, 0].item() for i in range(len(angles))]
    assert values == sorted(values, reverse=True), values
    assert values[-1] < values[0] * 0.1
