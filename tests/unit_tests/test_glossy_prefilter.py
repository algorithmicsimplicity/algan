"""The split-sum glossy route: the DFG term, the blur radius, and the
prefiltered reflection buffer's composite.

``algan/rendering/raytracing/DESIGN_glossy_prefilter.md`` is the design; the
renderer audit's REPORT.md section 4.5 is the measurement that motivated it.

Outside the fast suite for the same reason ``test_raytracing_unit.py`` is: the
end-to-end case drives a Taichi kernel variant and the compile is charged to
whichever test reaches it first.
"""

import math

import taichi as ti
import torch

# Importing algan is what initialises Taichi and Torch for this process.
import algan  # noqa: F401  (import for its side effects: ti.init lives there)
from algan.rendering.raytracing.wavefront_kernels_taichi import (
    _env_brdf_approx,
    _material_env_brdf,
    _material_reflectance,
    _mirror_share,
)


@ti.kernel
def _probe_env_brdf(f0: ti.types.ndarray(), nv: ti.types.ndarray(),
                    rough: ti.types.ndarray(), out: ti.types.ndarray()):
    for i in range(f0.shape[0]):
        e = _env_brdf_approx(
            ti.math.vec3(f0[i, 0], f0[i, 1], f0[i, 2]), nv[i], rough[i])
        for k in ti.static(range(3)):
            out[i, k] = e[k]


@ti.kernel
def _probe_material(rd: ti.types.ndarray(), nrm: ti.types.ndarray(),
                    metal: ti.types.ndarray(), ior: ti.types.ndarray(),
                    albedo: ti.types.ndarray(), rough: ti.types.ndarray(),
                    env_out: ti.types.ndarray(),
                    schlick_out: ti.types.ndarray(),
                    throttle_out: ti.types.ndarray()):
    for i in range(rd.shape[0]):
        d = ti.math.vec3(rd[i, 0], rd[i, 1], rd[i, 2])
        n = ti.math.vec3(nrm[i, 0], nrm[i, 1], nrm[i, 2])
        a = ti.math.vec3(albedo[i, 0], albedo[i, 1], albedo[i, 2])
        e = _material_env_brdf(d, n, metal[i], ior[i], a, rough[i])
        r, _dp = _material_reflectance(d, n, metal[i], ior[i], a, 0.0)
        share = _mirror_share(rough[i])
        for k in ti.static(range(3)):
            env_out[i, k] = e[k]
            schlick_out[i, k] = r[k]
            throttle_out[i, k] = r[k] * share


def _env_brdf(f0_rows, nv_rows, rough_rows):
    n = len(f0_rows)
    f0 = torch.tensor(f0_rows, dtype=torch.float32)
    nv = torch.tensor(nv_rows, dtype=torch.float32)
    rg = torch.tensor(rough_rows, dtype=torch.float32)
    out = torch.zeros((n, 3), dtype=torch.float32)
    _probe_env_brdf(f0, nv, rg, out)
    return out


def test_env_brdf_reduces_to_fresnel_at_zero_roughness():
    """A mirror's split-sum energy IS its Fresnel reflectance.

    This is what lets the route below ``_GLOSSY_MIN_ROUGHNESS`` keep Schlick
    while the route above it uses the DFG term: the two agree across the
    threshold rather than stepping.
    """
    f0 = [[0.04] * 3, [1.0] * 3, [0.95, 0.64, 0.54]]
    out = _env_brdf(f0, [1.0, 1.0, 1.0], [0.0, 0.0, 0.0])
    for row, expect in zip(out.tolist(), f0):
        for got, want in zip(row, expect):
            assert abs(got - want) < 0.02, (got, want)

    # Grazing: every material reflects (nearly) everything, whatever its f0.
    graze = _env_brdf(f0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    assert graze.min().item() > 0.95, graze


def test_env_brdf_is_bounded_and_falls_with_roughness():
    """Directional albedo: in [0, 1] everywhere, and a rougher metal reflects
    less of what arrives (the lobe spreads past the horizon and the geometry
    term takes the difference).
    """
    rows = []
    for rough in (0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0):
        for nv in (0.05, 0.25, 0.5, 0.75, 1.0):
            rows.append((rough, nv))
    out = _env_brdf([[1.0] * 3] * len(rows), [nv for _r, nv in rows],
                    [r for r, _nv in rows])
    assert out.min().item() >= 0.0, out.min().item()
    assert out.max().item() <= 1.0, out.max().item()

    for nv in (0.25, 0.5, 1.0):
        vals = [
            _env_brdf([[1.0] * 3], [nv], [r])[0, 0].item()
            for r in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
        ]
        assert all(b <= a + 1e-4 for a, b in zip(vals, vals[1:])), (nv, vals)


def test_env_brdf_beats_the_mirror_share_throttle_on_a_rough_metal():
    """The number the renderer audit measured: a metalness-1 roughness-0.35
    metal reflects ~4.7% of what it should under the throttle. The DFG term is
    the analytic answer, and it is an order of magnitude larger.
    """
    n = 1
    rd = torch.tensor([[0.0, 0.0, -1.0]] * n)
    nrm = torch.tensor([[0.0, 0.0, 1.0]] * n)
    metal = torch.tensor([1.0] * n)
    ior = torch.tensor([1.5] * n)
    albedo = torch.tensor([[1.0, 1.0, 1.0]] * n)
    rough = torch.tensor([0.35] * n)
    env = torch.zeros((n, 3))
    schlick = torch.zeros((n, 3))
    throttle = torch.zeros((n, 3))
    _probe_material(rd, nrm, metal, ior, albedo, rough, env, schlick, throttle)

    # Schlick at normal incidence on a white metal is 1: the whole lobe.
    assert abs(schlick[0, 0].item() - 1.0) < 1e-3, schlick
    # The throttle keeps a few percent of it ...
    assert throttle[0, 0].item() < 0.06, throttle
    # ... and the split-sum keeps most of it. 0.807 is the fit's exact value
    # here; what it is short of 1 is the single-scattering GGX model's own
    # energy loss (light that would have needed a second microfacet bounce),
    # which split-sum does not compensate and which is ~19% at this roughness.
    assert 0.78 < env[0, 0].item() < 0.83, env


def test_env_brdf_is_zero_for_the_unlit_sentinel():
    """``metalness < 0`` means no PBR material at all; there is no lobe to
    integrate, and a legacy/unlit surface must not gain a reflection.
    """
    rd = torch.tensor([[0.0, 0.0, -1.0]])
    nrm = torch.tensor([[0.0, 0.0, 1.0]])
    env = torch.zeros((1, 3))
    schlick = torch.zeros((1, 3))
    throttle = torch.zeros((1, 3))
    _probe_material(rd, nrm, torch.tensor([-1.0]), torch.tensor([1.5]),
                    torch.tensor([[1.0, 1.0, 1.0]]), torch.tensor([0.4]),
                    env, schlick, throttle)
    assert env.abs().max().item() == 0.0, env


def test_env_brdf_index_matched_dielectric_has_no_lobe():
    """IOR 1 is index-matched with the air around it: no interface, no
    reflection. Schlick cannot express that limit, so it is an explicit gate in
    both ``_material_reflectance`` and the split-sum term.
    """
    rd = torch.tensor([[0.0, 0.0, -1.0]])
    nrm = torch.tensor([[0.0, 0.0, 1.0]])
    env = torch.zeros((1, 3))
    schlick = torch.zeros((1, 3))
    throttle = torch.zeros((1, 3))
    _probe_material(rd, nrm, torch.tensor([0.0]), torch.tensor([1.0]),
                    torch.tensor([[1.0, 1.0, 1.0]]), torch.tensor([0.4]),
                    env, schlick, throttle)
    assert env.abs().max().item() < 1e-6, env


def test_env_brdf_metal_tint_rides_in_f0():
    """A coloured metal's reflection is tinted; a dielectric's is achromatic.
    The blend is ``mix(dielectric_f0, albedo, metalness)``, the same one
    ``_material_reflectance`` performs.
    """
    rd = torch.tensor([[0.0, 0.0, -1.0]] * 2)
    nrm = torch.tensor([[0.0, 0.0, 1.0]] * 2)
    albedo = torch.tensor([[0.95, 0.64, 0.54], [0.95, 0.64, 0.54]])
    env = torch.zeros((2, 3))
    schlick = torch.zeros((2, 3))
    throttle = torch.zeros((2, 3))
    _probe_material(rd, nrm, torch.tensor([1.0, 0.0]),
                    torch.tensor([1.5, 1.5]), albedo,
                    torch.tensor([0.3, 0.3]), env, schlick, throttle)
    metal, dielectric = env[0], env[1]
    assert metal[0].item() > metal[1].item() > metal[2].item(), metal
    assert abs(dielectric[0].item() - dielectric[1].item()) < 1e-5, dielectric


def test_blur_sigma_matches_the_design_formula():
    """The host and the kernel must agree about the lobe's screen footprint.

    ``sigma_px = k * (2 * roughness^2) / theta_px``, ``k = d_r / (d_p + d_r)``.
    Reproduced here in Python so a change to either side has to change this
    number too (DESIGN_glossy_prefilter.md section 3).
    """
    from algan.rendering.raytracing.settings import glossy_blur_sigma_px

    theta_px = 0.3948 / 480.0  # a PREVIEW frame's 22.62 degrees over 480 rows
    # Contact: the reflected surface touches the reflector, so nothing blurs.
    assert glossy_blur_sigma_px(0.35, 0.0, 5.0, theta_px) == 0.0
    # A reflection ten times further away than the reflector is nearly the
    # full lobe angle.
    far = glossy_blur_sigma_px(0.35, 50.0, 5.0, theta_px)
    full = 2.0 * 0.35 * 0.35 / theta_px
    assert 0.85 * full < far < full, (far, full)
    # An escaped ray (no hit recorded) is an infinitely distant reflection.
    assert math.isclose(
        glossy_blur_sigma_px(0.35, float("inf"), 5.0, theta_px), full,
        rel_tol=1e-6)
