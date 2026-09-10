"""The dielectric Fresnel term at a transmissive interface, both sides of it.

``_material_reflectance`` decides how a hit's energy splits between the
reflected and the transmitted branch. Getting that split right on the way *out*
of a solid is what the exact Fresnel equations are for: Schlick's approximation
is written for a ray arriving from the thin side, and evaluating it on the
inside angle understates the reflectance badly -- to the point of passing light
straight through an interface that should be a perfect mirror.

These tests call the ``@ti.func`` through a one-element kernel and compare it
against the unpolarised Fresnel equations evaluated in Python, which is the
ground truth Schlick approximates. They also pin the property the fix was
scoped by: a material that does not transmit takes exactly the path it took
before, side-blind.

Outside the fast suite: it compiles Taichi kernels, and nothing elsewhere in
the codebase can break it (see ``tests/README.md`` on what earns a ``fast``
mark).

Note the absent ``from __future__ import annotations``: the probe kernel's
``ti.types.ndarray()`` annotations are evaluated at run time, and stringifying
them stops the kernel compiling. This is the same hazard the ruff config calls
out for ``*_taichi.py`` files; ``tests/*`` already exempts ``I002``.
"""

import math

import pytest
import torch

from algan.rendering.raytracing.glass_energy import glass_energy_table
from algan.rendering.raytracing.path_tracer_taichi import (
    _pt_fresnel,
    _pt_lit_f_pdf,
    _pt_lit_lobes,
)
from algan.rendering.raytracing.shading_taichi import _MID_PHYSICAL
from algan.rendering.raytracing.wavefront_kernels_taichi import (
    _material_reflectance,
)
from algan.taichi_compat import ti

IOR = 1.5
CRITICAL_ANGLE = math.degrees(math.asin(1.0 / IOR))  # 41.81 degrees


@ti.kernel
def _probe(
    dirs: ti.types.ndarray(),
    nrm: ti.types.ndarray(),
    metalness: ti.f32,
    ior: ti.f32,
    transmission: ti.f32,
    out: ti.types.ndarray(),
):
    for i in range(dirs.shape[0]):
        rd = ti.math.vec3(dirs[i, 0], dirs[i, 1], dirs[i, 2]).normalized()
        n = ti.math.vec3(nrm[0], nrm[1], nrm[2])
        albedo = ti.math.vec3(1.0, 1.0, 1.0)
        R, diel_pass = _material_reflectance(
            rd, n, metalness, ior, albedo, transmission
        )
        out[i, 0] = R[0]
        out[i, 1] = diel_pass


def reflectance(angles_deg, *, inside, transmission=1.0, metalness=0.0):
    """``(R, diel_pass)`` for rays meeting an outward +Z normal at ``angles``.

    ``inside`` sends the ray outward through the surface (leaving the medium);
    otherwise it arrives from outside.
    """
    dirs = []
    for a in angles_deg:
        t = math.radians(a)
        # Outward normal is +Z. A ray arriving from outside travels -Z.
        z = math.cos(t) if inside else -math.cos(t)
        dirs.append((math.sin(t), 0.0, z))
    d = torch.tensor(dirs, dtype=torch.float32)
    n = torch.tensor((0.0, 0.0, 1.0), dtype=torch.float32)
    out = torch.zeros((len(dirs), 2), dtype=torch.float32)
    _probe(d, n, metalness, IOR, transmission, out)
    return out[:, 0].tolist(), out[:, 1].tolist()


def exact_fresnel(angle_deg, *, inside):
    """Unpolarised Fresnel reflectance at an ior-1.5 interface, or 1.0 past the
    critical angle. This is the physics the renderer is approximating.
    """
    eta_i, eta_t = (IOR, 1.0) if inside else (1.0, IOR)
    cos_i = math.cos(math.radians(angle_deg))
    sin_t = eta_i / eta_t * math.sin(math.radians(angle_deg))
    if sin_t >= 1.0:
        return 1.0
    cos_t = math.sqrt(1.0 - sin_t * sin_t)
    rs = ((eta_i * cos_i - eta_t * cos_t) / (eta_i * cos_i + eta_t * cos_t)) ** 2
    rp = ((eta_i * cos_t - eta_t * cos_i) / (eta_i * cos_t + eta_t * cos_i)) ** 2
    return 0.5 * (rs + rp)


# SCHLICK_TOLERANCE is Schlick's own error budget against the exact equations
# at ior 1.5, not slack for the code under test: the approximation runs about
# 0.019 low around 60 degrees and 0.022 high at 80. Everything the renderer
# does with it inherits that, on both sides of the interface.
SCHLICK_TOLERANCE = 0.03


@ti.kernel
def _pt_probe(
    glass_energy: ti.types.ndarray(),
    cos_i: ti.f32,
    eta: ti.f32,
    out: ti.types.ndarray(),
):
    one = ti.math.vec3(1.0, 1.0, 1.0)
    f0 = one * 0.04
    f = _pt_fresnel(f0, cos_i, eta, 0.0, one)
    for k in ti.static(range(3)):
        out[0, k] = f[k]
    # An interior reflection must have support in the evaluator as well as
    # the VNDF sampler. The outward normal faces away from the incident ray.
    rd = ti.math.vec3(ti.sqrt(1.0 - cos_i * cos_i), 0.0, cos_i)
    wi = ti.math.vec3(rd[0], 0.0, -rd[2])
    fc, pdf = _pt_lit_f_pdf(
        glass_energy,
        one * 0.0,
        one,
        f0,
        0.2,
        ti.math.vec3(0.0, 0.0, 1.0),
        rd,
        wi,
        0.0,
        0.0,
        1.0,
        0.0,
        eta,
        0.0,
        one,
        1.0,
    )
    out[1, 0] = fc[0]
    out[1, 1] = pdf


@pytest.mark.parametrize("angle", [0.0, 20.0, 40.0, 42.0, 60.0, 80.0])
def test_path_tracer_internal_fresnel_and_pdf(angle):
    from algan.rendering.taichi_runtime import init_taichi

    init_taichi()
    out = torch.zeros((2, 3))
    _pt_probe(
        torch.from_numpy(glass_energy_table()), math.cos(math.radians(angle)), IOR, out
    )
    assert float(out[0, 0]) == pytest.approx(
        exact_fresnel(angle, inside=True), abs=SCHLICK_TOLERANCE
    )
    assert out[1, 0] > 0, "NEE discarded the interior reflection hemisphere"
    assert out[1, 1] > 0, "MIS assigned zero density to a sampled reflection"
    if angle > CRITICAL_ANGLE:
        assert torch.equal(out[0], torch.ones(3)), "TIR must conserve all RGB energy"


def test_path_tracer_nested_interface_changes_the_critical_angle():
    from algan.rendering.taichi_runtime import init_taichi

    init_taichi()
    out = torch.zeros((2, 3))
    # At 50 degrees glass -> air is TIR, glass -> water is not.
    cos_i = math.cos(math.radians(50))
    _pt_probe(torch.from_numpy(glass_energy_table()), cos_i, IOR, out)
    assert out[0, 0] == 1
    _pt_probe(torch.from_numpy(glass_energy_table()), cos_i, IOR / 1.33, out)
    assert 0 < out[0, 0] < 0.2


@ti.kernel
def _pt_lobe_probe(params: ti.types.ndarray(), eta: ti.f32, out: ti.types.ndarray()):
    one = ti.math.vec3(1.0, 1.0, 1.0)
    n = ti.math.vec3(0.0, 0.0, 1.0)
    _diff, _spec, trans, f0, _rough = _pt_lit_lobes(
        _MID_PHYSICAL, params, 0, 0, one, 0.0, 0.001, 1.5, 1.0, n, n, eta
    )
    out[0] = f0[0]
    out[1] = trans[0]


@pytest.mark.parametrize("eta", [1.0, 1.5, 1.5 / 1.33])
def test_path_tracer_lobes_use_the_enclosing_medium_at_normal_incidence(eta):
    from algan.rendering.taichi_runtime import init_taichi

    init_taichi()
    params = torch.zeros((1, 1, 17))
    params[0, 0, 12] = 1.5
    params[0, 0, 13:17] = 1.0
    out = torch.zeros(2)
    _pt_lobe_probe(params, eta, out)
    expected = ((eta - 1) / (eta + 1)) ** 2
    assert float(out[0]) == pytest.approx(expected, abs=1e-6)
    assert float(out[1]) == pytest.approx(1 - expected, abs=1e-6)


@pytest.mark.parametrize("angle", [0.0, 20.0, 40.0, 60.0, 80.0])
def test_entering_matches_exact_fresnel(angle):
    (r,), _ = reflectance([angle], inside=False)
    assert r == pytest.approx(exact_fresnel(angle, inside=False), abs=SCHLICK_TOLERANCE)


def _inside_angle_schlick(angle_deg):
    """What the reflectance used to be: Schlick on the incident (inside)
    cosine, the side-blind form.
    """
    f0 = ((1.0 - IOR) / (1.0 + IOR)) ** 2
    return f0 + (1.0 - f0) * (1.0 - math.cos(math.radians(angle_deg))) ** 5


@pytest.mark.parametrize("angle", [0.0, 20.0, 35.0, 40.0])
def test_leaving_below_the_critical_angle_matches_exact_fresnel(angle):
    """The case Schlick cannot do on the incident angle.

    At 40 degrees inside glass the true reflectance is 0.245; the inside-angle
    Schlick says 0.041. Evaluating it on the air-side angle instead recovers
    the right answer, which is why KHR_materials_volume specifies that.
    """
    assert angle < CRITICAL_ANGLE
    (r,), _ = reflectance([angle], inside=True)
    exact = exact_fresnel(angle, inside=True)
    assert r == pytest.approx(exact, abs=SCHLICK_TOLERANCE)
    # And it is not merely inside the tolerance by luck: near the critical
    # angle the side-blind form is wrong by an order of magnitude, so pin the
    # gap rather than only the value.
    if angle >= 35.0:
        assert abs(r - exact) < 0.5 * abs(_inside_angle_schlick(angle) - exact)


@pytest.mark.parametrize("angle", [42.0, 50.0, 70.0, 89.0])
def test_total_internal_reflection_reflects_everything(angle):
    """Past the critical angle nothing crosses the interface, so the
    transmitted branch must be given zero weight -- not the mirror direction
    with the transmitted share, which is what it used to get.
    """
    assert angle > CRITICAL_ANGLE
    (r,), (diel_pass,) = reflectance([angle], inside=True)
    assert r == pytest.approx(1.0, abs=1e-5)
    assert diel_pass == pytest.approx(0.0, abs=1e-5)


def test_reflectance_rises_monotonically_towards_the_critical_angle():
    angles = [0.0, 10.0, 20.0, 30.0, 38.0, 41.0]
    r, _ = reflectance(angles, inside=True)
    assert all(b >= a for a, b in zip(r, r[1:])), r
    assert r[0] == pytest.approx(0.04, abs=0.005)


def test_an_opaque_material_is_side_blind():
    """The side test is inferred from the normal, which is only sound for a
    closed transmissive solid -- a back-facing hit on an ordinary opaque
    surface is not inside anything. So a non-transmissive material must get
    the same answer from either side, exactly as it did before the side test
    existed.
    """
    angles = [0.0, 30.0, 45.0, 60.0, 85.0]
    outside, _ = reflectance(angles, inside=False, transmission=0.0)
    inside, _ = reflectance(angles, inside=True, transmission=0.0)
    assert inside == pytest.approx(outside, abs=1e-6)


def test_a_metal_still_reflects_its_albedo_from_either_side():
    """Metalness gates transmission off entirely (``diel_pass`` folds in
    ``1 - m``), so a fully metallic surface must stay a mirror at any
    transmission and never acquire a critical angle.
    """
    angles = [0.0, 45.0, 70.0]
    r_in, pass_in = reflectance(angles, inside=True, transmission=1.0, metalness=1.0)
    r_out, _ = reflectance(angles, inside=False, transmission=1.0, metalness=1.0)
    assert r_in == pytest.approx(r_out, abs=1e-6)
    assert pass_in == pytest.approx([0.0] * len(angles), abs=1e-6)
