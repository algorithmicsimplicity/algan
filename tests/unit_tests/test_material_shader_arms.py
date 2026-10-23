"""The PyTorch material shaders' fused arms, and the BRDF helpers they are built from.

``lambert_shader``, ``phong_shader``, ``standard_shader``, ``physical_shader``
and ``toon_shader`` each hand their arithmetic to a ``@compiled(dynamic=True)``
arm (``_lambert_fused`` and so on). Those arms are what runs whenever a frame is
shaded in PyTorch rather than in the kernel: the vertex-bake path
(``SETTINGS.raytracing.experimental.fragment_shading`` off, one sample per pixel,
and no shadows, extended lights, environment map or custom fragment pipeline in
the scene) and the border colours of a curved bezier patch
(``nonplanar_circuit._shaded_border_colors``). They are compiled when
``SETTINGS.computing.torch_compile`` resolves on -- ``"auto"`` on Linux and macOS
CPU -- and run eagerly otherwise, which is every Windows render. Every other frame
is shaded in the kernel by the ``shading_taichi`` stage the same material
resolves to.

So this file pins three things:

* the compiled arm returns what the eager arm returns, and is genuinely
  compiled rather than silently demoted (against Dynamo's ``eager`` backend, as
  ``test_torch_compile.py`` does, so no Inductor build is paid);
* each torch shader agrees with the Taichi stage that normally renders the same
  material, for one point light, smooth and flat shaded -- the two paths are
  meant to be two evaluations of one formula, and a term added to one only is
  a material that looks different depending on which route a scene takes;
* the BRDF helpers against the formulas their docstrings name: GGX (Walter et
  al. 2007 with three.js's ``alpha = roughness^2``), the height-correlated
  Smith term (three.js's ``V_GGX_SmithCorrelated``), three.js's ``D_Charlie``, ``V_Neubelt`` and
  ``IBLSheenBRDF`` transcribed from its shader source, the sheen layer's energy
  compensation, and ``_light_geometry``'s clamps.

A feature test, outside the fast suite: it compiles small probe kernels, and
only a change to these shaders can break it.

Note the absent ``from __future__ import annotations``: the probe kernel's
``ti.types.ndarray()`` annotations are evaluated at run time, and stringifying
them stops the kernel compiling.
"""

import inspect
import math

import pytest
import torch
import torch.nn.functional as F

from algan import SETTINGS
from algan.rendering.raytracing import shading_taichi as st
from algan.rendering.raytracing.settings import _MAT_DEFAULTS, _MAT_SLOTS
from algan.rendering.shaders import material_shaders as ms
from algan.rendering.shaders.material_shaders import SHADER_FIXED_PARAM_COUNT
from algan.rendering.taichi_runtime import init_taichi
from algan.taichi_compat import ti
from algan.utils import torch_compile as tc

# material -> (torch shader, its fused arm, its Taichi stage, non-default params).
# Every parameter is moved off its default so a term the two paths weigh
# differently cannot hide behind a zero.
CASES = {
    "lambert": (
        ms.lambert_shader,
        ms._lambert_fused,
        st._stage_lambert,
        {
            "emissive": (0.10, 0.0, 0.05),
            "emissive_intensity": 1.5,
            "env_map_intensity": 1.2,
        },
    ),
    "phong": (
        ms.phong_shader,
        ms._phong_fused,
        st._stage_phong,
        {
            "emissive": (0.10, 0.0, 0.05),
            "emissive_intensity": 1.5,
            "specular": (0.30, 0.20, 0.10),
            "shininess": 20.0,
            "env_map_intensity": 1.2,
        },
    ),
    "standard": (
        ms.standard_shader,
        ms._standard_fused,
        st._stage_standard,
        {
            "roughness": 0.4,
            "metalness": 0.3,
            "emissive": (0.10, 0.0, 0.05),
            "emissive_intensity": 1.5,
            "env_map_intensity": 1.2,
        },
    ),
    "physical": (
        ms.physical_shader,
        ms._physical_fused,
        st._stage_physical,
        {
            "roughness": 0.4,
            "metalness": 0.3,
            "emissive": (0.10, 0.0, 0.05),
            "emissive_intensity": 1.5,
            "env_map_intensity": 1.2,
            "ior": 1.7,
            "specular_intensity": 0.8,
            "specular_color": (1.0, 0.9, 0.8),
            "clearcoat": 0.6,
            "clearcoat_roughness": 0.3,
            "sheen": 0.7,
            "sheen_roughness": 0.5,
            "sheen_color": (0.6, 0.3, 0.9),
            "transmission": 0.2,
        },
    ),
    "toon": (
        ms.toon_shader,
        ms._toon_fused,
        st._stage_toon,
        {"emissive": (0.10, 0.0, 0.05), "emissive_intensity": 1.5, "num_bands": 4.0},
    ),
}

CAMERA = (0.3, 0.5, 6.0)
LIGHT_POS = (2.0, 3.0, 4.0)
LIGHT_RGB = (0.9, 0.8, 0.7)


@pytest.fixture(autouse=True)
def _eager_unless_asked():
    """The torch shaders run eagerly unless a test turns compiling on.

    ``"auto"`` resolves on under Linux, which would put a full Inductor build of
    each arm inside the parity tests below; the compiled arm has a test of its
    own, against Dynamo's ``eager`` backend. The settings fixture in
    ``tests/conftest.py`` restores the switch afterwards.
    """
    SETTINGS.computing.set(torch_compile=False)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def _triangles(n, seed=0):
    """``n`` small triangles facing the camera, as the renderer hands them over.

    Returns ``(corners, normals, albedo)`` shaped ``[n, 3, 3]``, ``[n, 3, 3]`` and
    ``[n, 3, 4]``. The vertex normals are each corner's face normal tipped by a
    random amount, so smooth and flat shading differ, and are kept on the
    camera's side of the facet so every corner is seen from the front.
    """
    gen = torch.Generator().manual_seed(seed)
    centre = torch.randn(n, 1, 3, generator=gen) * 0.6
    corners = centre + torch.randn(n, 3, 3, generator=gen) * 0.3
    face = torch.linalg.cross(
        corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0], dim=-1
    )
    face = F.normalize(face, dim=-1)
    to_camera = F.normalize(torch.tensor(CAMERA) - corners.mean(1), dim=-1)
    face = torch.where((face * to_camera).sum(-1, keepdim=True) < 0, -face, face)
    normals = F.normalize(
        face.unsqueeze(1) + 0.35 * torch.randn(n, 3, 3, generator=gen), dim=-1
    )
    albedo = torch.cat(
        (torch.rand(n, 3, 3, generator=gen), torch.rand(n, 3, 1, generator=gen)), -1
    )
    return corners, normals, albedo


def _shader_kwargs(shader, params, leading):
    """``params`` (plus the shader's own defaults for the rest) as the tensors
    the live path hands a shader: every material parameter a tensor shaped to
    broadcast against ``leading`` corner dimensions.
    """
    kwargs = {}
    signature = inspect.signature(shader).parameters
    for name in list(signature)[SHADER_FIXED_PARAM_COUNT:]:
        value = params.get(name, signature[name].default)
        value = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
        kwargs[name] = value.view(*([1] * leading), value.numel())
    return kwargs


def _run_torch(shader, corners, normals, albedo, params):
    """``shader`` over the triangles' corners, the way the vertex path calls it."""
    lead = corners.dim() - 1
    view = (1,) * lead
    return shader(
        None,
        corners,
        normals,
        albedo,
        torch.tensor(CAMERA).view(*view, 3),
        torch.tensor(LIGHT_POS).view(*view, 3),
        torch.tensor((*LIGHT_RGB, 0.0)).view(*view, 4),
        1.0,
        1.0,
        **_shader_kwargs(shader, params, lead),
    )


# ---------------------------------------------------------------------------
# The Taichi stage, probed one fragment per corner
# ---------------------------------------------------------------------------


@ti.kernel
def _stage_probe(
    stage: ti.template(),
    pos: ti.types.ndarray(),
    view: ti.types.ndarray(),
    nrm: ti.types.ndarray(),
    face: ti.types.ndarray(),
    albedo: ti.types.ndarray(),
    params: ti.types.ndarray(),
    light_pos: ti.types.ndarray(),
    light_col: ti.types.ndarray(),
    out: ti.types.ndarray(),
):
    for i in range(pos.shape[0]):
        p = ti.math.vec3(pos[i, 0], pos[i, 1], pos[i, 2])
        vd = ti.math.vec3(view[i, 0], view[i, 1], view[i, 2]).normalized()
        n = ti.math.vec3(nrm[i, 0], nrm[i, 1], nrm[i, 2])
        fn = ti.math.vec3(face[i, 0], face[i, 1], face[i, 2])
        rgb = ti.math.vec3(albedo[i, 0], albedo[i, 1], albedo[i, 2])
        r = stage(
            p,
            vd,
            n,
            fn,
            rgb,
            albedo[i, 3],
            params,
            0,
            i,
            0,
            light_pos,
            light_col,
            light_pos.shape[1],
            0,
            0,
            ti.math.vec3(0.0, 0.0, 0.0),
        )
        for c in ti.static(range(4)):
            out[i, c] = r[c]


def _material_block(k, params):
    """A ``[1, k, MAT_W]`` material block with ``params`` written by slot name."""
    block = torch.tensor(_MAT_DEFAULTS, dtype=torch.float32).view(1, 1, -1)
    block = block.repeat(1, k, 1)
    for name, value in params.items():
        start, width = _MAT_SLOTS[name]
        block[0, :, start : start + width] = torch.as_tensor(value, dtype=torch.float32)
    return block


def _run_stage(stage, corners, normals, albedo, params):
    """The Taichi stage at every corner, with the facet's raw cross product as
    its geometric normal -- what the kernel hands a stage at a triangle hit.
    """
    init_taichi()
    n = corners.shape[0]
    face = torch.linalg.cross(
        corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0], dim=-1
    )
    pos = corners.reshape(-1, 3).contiguous()
    view = (torch.tensor(CAMERA) - pos).contiguous()
    out = torch.zeros(n * 3, 4)
    _stage_probe(
        stage,
        pos,
        view,
        normals.reshape(-1, 3).contiguous(),
        face.unsqueeze(1).expand(n, 3, 3).reshape(-1, 3).contiguous(),
        albedo.reshape(-1, 4).contiguous(),
        _material_block(n * 3, params),
        torch.tensor(LIGHT_POS).view(1, 1, 3),
        torch.tensor(LIGHT_RGB).view(1, 1, 3),
        out,
    )
    return out.view(n, 3, 4)


# ---------------------------------------------------------------------------
# Compiled arm == eager arm
# ---------------------------------------------------------------------------


@pytest.fixture
def _eager_backend(monkeypatch):
    """Dynamo tracing with no code generation, and a clean slate around it."""
    monkeypatch.setattr(tc, "_BACKEND", "eager")
    tc.reset_compiled_functions()
    yield
    tc.reset_compiled_functions()


@pytest.mark.usefixtures("_eager_backend")
@pytest.mark.parametrize("material", sorted(CASES))
def test_the_compiled_arm_returns_what_the_eager_arm_returns(material):
    """``torch_compile`` on and off give the same colours, and on really compiles.

    Shaped as a batch arrives -- ``[frames, triangles, corners, ...]`` -- and
    flat-shaded part way, so the face-normal branch of ``_shading_normal`` is
    inside the traced region too. A compile failure would not fail a render (the
    wrapper warns and runs the arm eagerly), so the arm's own state is checked:
    it must hold a compiled callable and no failure verdict.
    """
    shader, fused, _stage, params = CASES[material]
    params = {**params, "flat_shading": 0.3}
    corners, normals, albedo = _triangles(10, seed=1)
    corners = corners.view(2, 5, 3, 3)
    normals = normals.view(2, 5, 3, 3)
    albedo = albedo.view(2, 5, 3, 4)

    SETTINGS.computing.set(torch_compile=True)
    compiled = _run_torch(shader, corners, normals, albedo, params)
    state = fused._algan_compiled
    assert state.compiled is not None, f"{material}: the arm was never compiled"
    assert not state.failed, f"{material}: compile demoted to eager: {state.reason}"

    SETTINGS.computing.set(torch_compile=False)
    eager = _run_torch(shader, corners, normals, albedo, params)

    assert compiled.shape == eager.shape == (2, 5, 3, 4)
    torch.testing.assert_close(compiled, eager, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# Torch shader == Taichi stage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flat_shading", [0.0, 0.6], ids=["smooth", "flat"])
@pytest.mark.parametrize("material", sorted(CASES))
def test_the_torch_shader_agrees_with_its_taichi_stage(material, flat_shading):
    """The vertex-bake path and the fragment path shade a corner identically.

    Same corner, same normals, same light, the default linear working space: the
    torch shader and the in-kernel stage the renderer resolves the same material
    to must return the same RGB and pass the same glow. Flat shading takes the
    face normal from the corners on the torch side and from the hit's geometric
    normal in the kernel, so the ``0.6`` case also pins that both blend toward
    the same facet orientation.
    """
    shader, _fused, stage, params = CASES[material]
    params = {**params, "flat_shading": flat_shading}
    corners, normals, albedo = _triangles(24, seed=2)

    expected = _run_torch(shader, corners, normals, albedo, params)
    got = _run_stage(stage, corners, normals, albedo, params)

    # The comparison has to be over lit, non-trivial colours to mean anything.
    assert float(expected[..., :3].amax()) > 0.2
    assert float(expected[..., :3].std()) > 0.02
    # Room for a GPU kernel's fast-math pow (the Mac arm runs this on Metal);
    # a term the two paths weigh differently is off by orders more.
    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-4)


def test_flat_shading_moves_the_shade_toward_the_facet():
    """The flat-shading blend is not a no-op on either path: at 1.0 all three
    corners of a facet shade alike, which smooth shading does not.
    """
    shader, _fused, stage, params = CASES["lambert"]
    corners, normals, albedo = _triangles(8, seed=3)
    albedo = albedo[:, :1].expand_as(albedo).contiguous()  # one albedo per facet

    for run in (_run_torch, lambda _shader, *a: _run_stage(stage, *a)):
        smooth = run(shader, corners, normals, albedo, {**params, "flat_shading": 0.0})
        flat = run(shader, corners, normals, albedo, {**params, "flat_shading": 1.0})
        # Not exactly alike even when flat: each corner sits at its own
        # distance from the point light, so its light direction differs.
        assert float(flat[..., :3].std(dim=1).mean()) < 0.5 * float(
            smooth[..., :3].std(dim=1).mean()
        )


# ---------------------------------------------------------------------------
# BRDF helpers against their textbook / three.js forms
# ---------------------------------------------------------------------------


def _hemisphere_integral(fn, n=4096):
    """``2 pi * integral_0^{pi/2} fn(cos t) cos t sin t dt``, midpoint rule, f64."""
    t = (torch.arange(n, dtype=torch.float64) + 0.5) * (math.pi / 2 / n)
    c = torch.cos(t)
    return float((fn(c) * c * torch.sin(t)).sum() * (math.pi / 2 / n) * 2 * math.pi)


def _walter_ggx(n_dot_h, roughness):
    """Walter et al. 2007, eq. 33, with three.js's ``alpha = roughness^2``
    (``D_GGX``: ``RECIPROCAL_PI * a2 / pow2(dotNH^2 * (a2 - 1) + 1)``).
    """
    a2 = roughness**4
    return a2 / (math.pi * (n_dot_h**2 * (a2 - 1.0) + 1.0) ** 2)


# Kept to roughness >= 0.2, where the helper's 1e-7 floor on its denominator
# never engages. Below roughness ~0.115 the floor caps the lobe's peak; that
# range is deliberately not pinned here.
@pytest.mark.parametrize("roughness", [0.2, 0.35, 0.5, 0.8, 1.0])
def test_ggx_is_walters_distribution_and_is_normalized(roughness):
    """``ggx_distribution`` is GGX term for term, and its projected integral is 1."""
    n_dot_h = torch.linspace(0.0, 1.0, 101, dtype=torch.float64)
    r = torch.tensor(roughness, dtype=torch.float64)
    torch.testing.assert_close(
        ms.ggx_distribution(n_dot_h, r),
        _walter_ggx(n_dot_h, roughness),
        rtol=1e-9,
        atol=0.0,
    )
    total = _hemisphere_integral(lambda c: ms.ggx_distribution(c, r))
    assert total == pytest.approx(1.0, abs=2e-3)


def test_ggx_at_full_roughness_is_the_uniform_lobe():
    """``alpha = 1`` is the one roughness at which GGX is flat: ``1 / pi``."""
    n_dot_h = torch.linspace(0.0, 1.0, 11)
    got = ms.ggx_distribution(n_dot_h, torch.tensor(1.0))
    torch.testing.assert_close(got, torch.full_like(n_dot_h, 1.0 / math.pi))


def _three_v_ggx_smith_correlated(roughness, n_dot_l, n_dot_v):
    """three.js ``V_GGX_SmithCorrelated``, transcribed, on ``BRDF_GGX``'s
    ``alpha = roughness^2`` -- the visibility term, ``G2 / (4 n.l n.v)``.
    """
    a2 = roughness**4
    gv = n_dot_l * torch.sqrt(a2 + (1.0 - a2) * n_dot_v**2)
    gl = n_dot_v * torch.sqrt(a2 + (1.0 - a2) * n_dot_l**2)
    return 0.5 / (gv + gl)


def _smith_lambda(cos_theta, roughness):
    """Smith's ``Lambda`` for isotropic GGX, as the path tracer writes it."""
    t2 = (1.0 - cos_theta**2) / cos_theta**2
    return 0.5 * (torch.sqrt(1.0 + roughness**4 * t2) - 1.0)


#: (view, light) angles from the normal in degrees: head-on, mid, grazing and
#: mixed, so a term that is right only head-on (where every Smith form is 1)
#: or only symmetric cannot pass.
_SMITH_ANGLES = [
    (0.0, 0.0),
    (45.0, 45.0),
    (60.0, 60.0),
    (80.0, 80.0),
    (85.0, 30.0),
    (20.0, 75.0),
    (70.0, 5.0),
    (88.0, 88.0),
]
_SMITH_ROUGHNESS = [0.15, 0.3, 0.5, 0.8, 1.0]


def _smith_grid():
    """Every (n.v, n.l, roughness) of the angle and roughness lists above, f64."""
    rows = [
        (math.cos(math.radians(v)), math.cos(math.radians(lt)), r)
        for v, lt in _SMITH_ANGLES
        for r in _SMITH_ROUGHNESS
    ]
    nv, nl, rough = torch.tensor(rows, dtype=torch.float64).unbind(-1)
    return nv, nl, rough


def test_smith_is_three_js_height_correlated_visibility():
    """``smith_geometry`` over ``4 n.v n.l`` is three.js's ``V_GGX_SmithCorrelated``.

    It used to be UE4's Schlick-GGX with ``k = (r + 1)^2 / 8``: right head-on,
    but at roughness 0.3 with view and light both at 80 degrees it gave
    ``G = 0.249`` where the exact height-correlated term is 0.891, so back-lit
    rims rendered 3.6x dimmer than in three.js and the path tracer. The same
    term is the path tracer's ``1 / (1 + Lambda_v + Lambda_l)``.
    """
    nv, nl, rough = _smith_grid()
    got = ms.smith_geometry(nv, nl, rough)
    v = _three_v_ggx_smith_correlated(rough, nl, nv)
    torch.testing.assert_close(got, 4.0 * nv * nl * v, rtol=1e-12, atol=0.0)
    exact = 1.0 / (1.0 + _smith_lambda(nv, rough) + _smith_lambda(nl, rough))
    torch.testing.assert_close(got, exact, rtol=1e-9, atol=0.0)

    c80 = torch.tensor(math.cos(math.radians(80.0)), dtype=torch.float64)
    g = ms.smith_geometry(c80, c80, torch.tensor(0.3, dtype=torch.float64))
    assert float(g) == pytest.approx(0.891, abs=5e-4)
    # 1 head-on, falling with roughness and toward grazing.
    one = torch.tensor(1.0, dtype=torch.float64)
    assert float(ms.smith_geometry(one, one, torch.tensor(0.7))) == pytest.approx(1.0)
    rough_g = ms.smith_geometry(nv, nl, torch.ones_like(nv))
    smooth_g = ms.smith_geometry(nv, nl, torch.full_like(nv, 0.2))
    assert bool((rough_g[nv < 1.0] < smooth_g[nv < 1.0]).all())
    sweep = torch.linspace(1.0, 0.05, 12, dtype=torch.float64)
    assert bool(
        (torch.diff(ms.smith_geometry(sweep, sweep, torch.tensor(0.5))) < 0).all()
    )


@ti.kernel
def _smith_probe(
    nv: ti.types.ndarray(),
    nl: ti.types.ndarray(),
    rough: ti.types.ndarray(),
    out: ti.types.ndarray(),
):
    for i in range(nv.shape[0]):
        out[i] = st._smith_geometry(nv[i], nl[i], rough[i])


def test_the_kernels_smith_term_is_three_js_height_correlated_visibility():
    """The deterministic renderer's in-kernel ``_smith_geometry`` -- the term
    every GGX stage and the direct specular lobe divide by ``4 n.v n.l`` -- is
    ``V_GGX_SmithCorrelated`` too, at grazing and mid angles as well as head-on,
    and it is its torch twin's number.
    """
    init_taichi()
    nv, nl, rough = _smith_grid()
    out = torch.zeros(nv.shape[0], dtype=torch.float32)
    _smith_probe(nv.float(), nl.float(), rough.float(), out)
    v = _three_v_ggx_smith_correlated(rough, nl, nv)
    # Room for a GPU kernel's fast-math sqrt; the Schlick remap this replaced
    # is off by a factor of up to 3.6 on this grid.
    torch.testing.assert_close(out.double(), 4.0 * nv * nl * v, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(
        out.double(), ms.smith_geometry(nv, nl, rough), rtol=1e-4, atol=1e-6
    )


def _three_d_charlie(roughness, n_dot_h):
    """three.js ``D_Charlie``, transcribed."""
    alpha = roughness**2
    inv_alpha = 1.0 / alpha
    sin2h = torch.clamp(1.0 - n_dot_h * n_dot_h, min=0.0078125)
    return (2.0 + inv_alpha) * sin2h ** (inv_alpha * 0.5) / (2.0 * math.pi)


@pytest.mark.parametrize("roughness", [0.25, 0.5, 0.8, 1.0])
def test_charlie_sheen_is_three_js_d_charlie_and_is_normalized(roughness):
    """Same lobe as three.js, sin^2 floor included, and a normalized NDF: the
    Estevez-Kulla form integrates to 1 over the projected hemisphere (the
    floor adds less than 1e-3 at these roughnesses).
    """
    n_dot_h = torch.linspace(0.0, 1.0, 101, dtype=torch.float64)
    r = torch.tensor(roughness, dtype=torch.float64)
    torch.testing.assert_close(
        ms._d_charlie(n_dot_h, r),
        _three_d_charlie(roughness, n_dot_h),
        rtol=1e-12,
        atol=0,
    )
    # A Python float roughness takes the scalar branch and agrees.
    torch.testing.assert_close(
        ms._d_charlie(n_dot_h, roughness), ms._d_charlie(n_dot_h, r)
    )
    assert _hemisphere_integral(lambda c: ms._d_charlie(c, r)) == pytest.approx(
        1.0, abs=1e-3
    )


def test_neubelt_visibility_is_three_js_and_saturates():
    """``saturate(1 / (4 (N.L + N.V - N.L N.V)))``: a quarter head-on, clamped to
    1 toward grazing, where the unclamped form diverges.
    """
    nv = torch.tensor([1.0, 0.5, 0.3, 0.1, 0.01, 0.0], dtype=torch.float64)
    nl = torch.tensor([1.0, 0.5, 0.2, 0.1, 0.01, 0.0], dtype=torch.float64)
    expected = (1.0 / (4.0 * (nl + nv - nl * nv)).clamp_min(1e-300)).clamp(0.0, 1.0)
    torch.testing.assert_close(ms._v_neubelt(nv, nl), expected)
    assert float(ms._v_neubelt(torch.tensor(1.0), torch.tensor(1.0))) == 0.25
    assert float(ms._v_neubelt(torch.tensor(0.0), torch.tensor(0.0))) == 1.0


def _three_ibl_sheen_brdf(dot_nv, roughness):
    """three.js ``IBLSheenBRDF`` (``lights_physical_pars_fragment``), transcribed."""
    dot_nv = min(max(dot_nv, 0.0), 1.0)
    r2 = roughness * roughness
    r_inv = 1.0 / (roughness + 0.1)
    a = -1.9362 + 1.0678 * roughness + 0.4573 * r2 - 0.8469 * r_inv
    b = -0.6014 + 0.5538 * roughness - 0.4670 * r2 - 0.1255 * r_inv
    return min(max(math.exp(a * dot_nv + b), 0.0), 1.0)


def test_ibl_sheen_brdf_is_three_js_curve_fit():
    """The sheen albedo fit is three.js's, through both the tensor and the
    scalar branch, and behaves like a directional albedo: in ``[0, 1]``,
    largest at grazing view and growing with roughness head-on.
    """
    mus = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    for roughness in (0.05, 0.25, 0.5, 0.75, 1.0):
        tensor = ms._ibl_sheen_brdf(torch.tensor(mus, dtype=torch.float64), roughness)
        for i, mu in enumerate(mus):
            expected = _three_ibl_sheen_brdf(mu, roughness)
            assert float(tensor[i]) == pytest.approx(expected, rel=1e-12)
            assert ms._ibl_sheen_brdf(mu, roughness) == pytest.approx(
                expected, rel=1e-12
            )
        assert bool(((tensor >= 0.0) & (tensor <= 1.0)).all())
        assert bool((torch.diff(tensor) < 0).all()), "grazing must pick up more sheen"
    head_on = [ms._ibl_sheen_brdf(1.0, r) for r in (0.25, 0.5, 0.75, 1.0)]
    assert head_on == sorted(head_on)


def test_sheen_scales_the_base_layer_by_three_js_energy_compensation():
    """Turning sheen on scales the base lobes by ``1 - max(sheen_color * sheen)
    * max(E(N.V), E(N.L))`` and adds the Charlie lobe on top -- three.js's
    ``RE_Direct_Physical`` -- leaving the ambient and the clearcoat unscaled.

    Checked on the physical arm with the ambient and emissive switched off, so
    the direct term is the whole output.
    """
    corners, normals, albedo = _triangles(16, seed=4)
    base = {
        "roughness": 0.5,
        "metalness": 0.2,
        "sheen_roughness": 0.6,
        "sheen_color": (0.8, 0.5, 0.3),
        "clearcoat": 0.0,
    }
    pos, nrm = corners, F.normalize(normals, dim=-1)
    (_l, _v, _h, n_dot_l, n_dot_v, n_dot_h, _vh) = ms._light_geometry(
        pos, nrm, torch.tensor(CAMERA), torch.tensor(LIGHT_POS)
    )

    def shade(sheen):
        kwargs = _shader_kwargs(ms.physical_shader, {**base, "sheen": sheen}, 2)
        return ms.physical_shader(
            None,
            corners,
            normals,
            albedo,
            torch.tensor(CAMERA).view(1, 1, 3),
            torch.tensor(LIGHT_POS).view(1, 1, 3),
            torch.tensor((*LIGHT_RGB, 0.0)).view(1, 1, 4),
            1.0,
            0.0,  # no ambient
            **kwargs,
        )[..., :3]

    off = shade(0.0)
    on = shade(0.5)
    sheen_c = torch.tensor(base["sheen_color"]) * 0.5
    r = base["sheen_roughness"]
    comp = 1.0 - float(sheen_c.max()) * torch.maximum(
        ms._ibl_sheen_brdf(n_dot_v, r), ms._ibl_sheen_brdf(n_dot_l, r)
    )
    lobe = (
        sheen_c
        * ms._d_charlie(n_dot_h, r)
        * ms._v_neubelt(n_dot_v, n_dot_l)
        * torch.tensor(LIGHT_RGB)
        * n_dot_l
    )
    assert float(lobe.amax()) > 1e-3, "the sheen lobe must be visible to be pinned"
    assert float((1.0 - comp).amax()) > 1e-2, "the compensation must bite somewhere"
    torch.testing.assert_close(on, off * comp + lobe, rtol=1e-5, atol=1e-6)


def test_clearcoat_attenuates_the_layers_under_it_as_three_js_does():
    """Turning clearcoat on scales everything under the coat by
    ``1 - clearcoat * Fcc(N.V)`` (Schlick, F0 0.04) and adds the coat lobe on
    top -- three.js's ``meshphysical`` output and the path tracer's
    ``_PL_ATT``. It used to add the lobe to an unattenuated base, so a coated
    surface reflected more than arrived. Emission is left alone, as in the
    path tracer.
    """
    corners, normals, albedo = _triangles(16, seed=5)
    base = {
        "roughness": 0.5,
        "metalness": 0.2,
        "clearcoat_roughness": 0.3,
        "sheen": 0.0,
        "emissive": (0.1, 0.05, 0.0),
        "emissive_intensity": 1.0,
    }
    pos, nrm = corners, F.normalize(normals, dim=-1)
    (_l, _v, _h, n_dot_l, n_dot_v, n_dot_h, v_dot_h) = ms._light_geometry(
        pos, nrm, torch.tensor(CAMERA), torch.tensor(LIGHT_POS)
    )

    def shade(clearcoat):
        kwargs = _shader_kwargs(ms.physical_shader, {**base, "clearcoat": clearcoat}, 2)
        return ms.physical_shader(
            None,
            corners,
            normals,
            albedo,
            torch.tensor(CAMERA).view(1, 1, 3),
            torch.tensor(LIGHT_POS).view(1, 1, 3),
            torch.tensor((*LIGHT_RGB, 0.0)).view(1, 1, 4),
            1.0,
            0.5,
            **kwargs,
        )[..., :3]

    emission = torch.tensor(base["emissive"])
    off = shade(0.0) - emission
    cc = 0.8
    on = shade(cc) - emission
    r = torch.tensor(base["clearcoat_roughness"])
    lobe = (
        cc
        * ms.ggx_distribution(n_dot_h, r)
        * ms.smith_geometry(n_dot_v, n_dot_l, r)
        * ms.fresnel_schlick(v_dot_h, 0.04)
        / (4.0 * n_dot_v * n_dot_l).clamp_min(1e-4)
        * torch.tensor(LIGHT_RGB)
        * n_dot_l
    )
    att = 1.0 - cc * ms.fresnel_schlick(n_dot_v, 0.04)
    assert float(lobe.amax()) > 1e-3, "the coat lobe must be visible to be pinned"
    torch.testing.assert_close(on, off * att + lobe, rtol=1e-4, atol=1e-5)


def test_light_geometry_returns_unit_directions_and_its_documented_clamps():
    """Unit light, view and half vectors; ``N.L``, ``N.H`` and ``V.H`` floored at
    0 and ``N.V`` at ``1e-4`` -- the floor that keeps ``4 N.V N.L`` and the
    Smith terms finite when the camera sees a surface edge-on or from behind.
    """
    gen = torch.Generator().manual_seed(5)
    pos = torch.randn(64, 3, generator=gen)
    nrm = F.normalize(torch.randn(64, 3, generator=gen), dim=-1)
    cam = torch.tensor([0.0, 0.0, 8.0])
    light = torch.tensor([3.0, -2.0, 5.0])
    light_dir, view_dir, half_dir, nl, nv, nh, vh = ms._light_geometry(
        pos, nrm, cam, light
    )
    for vec in (light_dir, view_dir, half_dir):
        torch.testing.assert_close(vec.norm(dim=-1), torch.ones(64))
    torch.testing.assert_close(light_dir, F.normalize(light - pos, dim=-1))
    torch.testing.assert_close(view_dir, F.normalize(cam - pos, dim=-1))
    torch.testing.assert_close(half_dir, F.normalize(light_dir + view_dir, dim=-1))

    raw_nl = (nrm * light_dir).sum(-1, keepdim=True)
    raw_nv = (nrm * view_dir).sum(-1, keepdim=True)
    assert bool((raw_nl < 0).any()), "some points must face away from the light"
    assert bool((raw_nv < 0).any()), "some points must face away from the camera"
    torch.testing.assert_close(nl, raw_nl.clamp_min(0.0))
    torch.testing.assert_close(nv, raw_nv.clamp_min(1e-4))
    torch.testing.assert_close(nh, (nrm * half_dir).sum(-1, keepdim=True).clamp_min(0))
    torch.testing.assert_close(
        vh, (view_dir * half_dir).sum(-1, keepdim=True).clamp_min(0)
    )
    assert float(nv.min()) == pytest.approx(1e-4)
