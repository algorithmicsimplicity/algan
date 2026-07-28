"""Unit tests for the Three.js-style material system.

Fast and render-free: validates material defaults, colour parsing, the
shader-parameter contract, the numeric behaviour of each lighting shader, and
``Mob.set_material`` wiring -- without running the GPU render pipeline.

Run directly:  python tests/unit_tests/test_materials.py
Or via pytest: python -m pytest tests/unit_tests/test_materials.py
"""

import inspect
import warnings

import torch

from algan.rendering.shaders import material_shaders as ms
from algan.rendering.shaders.materials import (
    FrontSide,
    Material,
    MeshBasicMaterial,
    MeshLambertMaterial,
    MeshPhongMaterial,
    MeshStandardMaterial,
    MeshPhysicalMaterial,
    MeshToonMaterial,
    MeshNormalMaterial,
    MeshMatcapMaterial,
    MeshDepthMaterial,
    _to_rgb,
    _to_color5,
)
from algan.constants.color import WHITE

_NUM_BASE_PARAMS = len(inspect.signature(ms.basic_material_shader).parameters)  # 9

ALL_MATERIALS = [
    MeshBasicMaterial,
    MeshLambertMaterial,
    MeshPhongMaterial,
    MeshStandardMaterial,
    MeshPhysicalMaterial,
    MeshToonMaterial,
    MeshNormalMaterial,
    MeshMatcapMaterial,
    MeshDepthMaterial,
]


def _v(x):
    """Flatten an attribute (tensor or scalar) to a python float list."""
    return torch.as_tensor(x).reshape(-1).tolist()


# ---------------------------------------------------------------------------
# Colour parsing
# ---------------------------------------------------------------------------

def test_colour_parsing():
    assert _to_rgb(0xFF0000).reshape(-1).tolist() == [1.0, 0.0, 0.0]
    assert _to_rgb("#00FF00").reshape(-1).tolist() == [0.0, 1.0, 0.0]
    assert _to_rgb("#0000ff").reshape(-1).tolist() == [0.0, 0.0, 1.0]
    rgb = _to_rgb((0.25, 0.5, 0.75)).reshape(-1).tolist()
    assert max(abs(a - b) for a, b in zip(rgb, [0.25, 0.5, 0.75])) < 1e-6
    # An Algan Color passes through, keeping its rgb.
    assert _to_rgb(WHITE).reshape(-1).tolist() == [1.0, 1.0, 1.0]
    # 5-channel colour with glow/opacity preserved by _to_color5.
    assert _to_color5(0xFFFFFF).reshape(-1)[:3].tolist() == [1.0, 1.0, 1.0]
    print("ok: colour parsing")


# ---------------------------------------------------------------------------
# Three.js default settings
# ---------------------------------------------------------------------------

def test_base_defaults():
    m = Material()
    assert m.color is None
    assert m.opacity == 1.0
    assert m.transparent is False
    assert m.visible is True
    assert m.side == FrontSide
    assert m.flatShading is False
    assert m.wireframe is False
    assert m.vertexColors is False
    print("ok: base material defaults")


def test_material_defaults():
    assert MeshLambertMaterial().emissive == 0x000000
    assert MeshLambertMaterial().emissiveIntensity == 1.0

    p = MeshPhongMaterial()
    assert p.specular == 0x111111
    assert p.shininess == 30.0
    assert p.emissive == 0x000000

    s = MeshStandardMaterial()
    assert s.roughness == 1.0
    assert s.metalness == 0.0
    assert s.emissiveIntensity == 1.0
    assert s.envMapIntensity == 1.0

    ph = MeshPhysicalMaterial()
    assert ph.clearcoat == 0.0
    assert ph.clearcoatRoughness == 0.0
    assert ph.ior == 1.5
    assert ph.specularIntensity == 1.0
    assert ph.specularColor == 0xFFFFFF
    assert ph.sheen == 0.0
    assert ph.sheenRoughness == 1.0
    assert ph.sheenColor == 0x000000
    assert ph.transmission == 0.0
    assert ph.iridescence == 0.0
    # Inherits Standard defaults.
    assert ph.roughness == 1.0 and ph.metalness == 0.0

    assert MeshToonMaterial().bands == 3.0
    d = MeshDepthMaterial()
    assert d.near == 0.1 and d.far == 100.0
    print("ok: material-specific defaults match Three.js")


def test_unexpected_kwarg_raises():
    try:
        MeshStandardMaterial(not_a_real_property=1.0)
    except TypeError:
        print("ok: unexpected kwarg raises TypeError")
        return
    raise AssertionError("expected TypeError for unknown property")


# ---------------------------------------------------------------------------
# Shader-parameter contract
# ---------------------------------------------------------------------------

def test_param_contract():
    """Every key from get_shader_param_values must match the shader's extra
    parameter names exactly (so set_material wires values to the right attrs)."""
    for cls in ALL_MATERIALS:
        m = cls()
        extra = list(inspect.signature(m.shader).parameters)[_NUM_BASE_PARAMS:]
        keys = list(m.get_shader_param_values())
        assert set(keys) == set(extra), (cls.__name__, keys, extra)
    print("ok: shader-parameter contract for all materials")


# ---------------------------------------------------------------------------
# Shader numeric behaviour
# ---------------------------------------------------------------------------

def _toy_geometry():
    vloc = torch.tensor([[[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]])  # [1,3,3]
    vnrm = torch.tensor([[[0.0, 0, 1], [0, 0, 1], [0, 0, 1]]])  # +Z
    alb = torch.tensor([[[0.8, 0.1, 0.1, 0.0]]]).expand(1, 3, 4).contiguous()
    cam = torch.tensor([0.0, 0, 5]).view(1, 1, 3)
    light = torch.tensor([3.0, 0, 4]).view(1, 1, 3)
    lcol = torch.tensor([1.0, 1, 1, 0]).view(1, 1, 4)
    return vloc, vnrm, alb, cam, light, lcol


def _scalar(x):
    return torch.tensor(float(x)).view(1, 1, 1)


def _col(*x):
    return torch.tensor([float(v) for v in x]).view(1, 1, 3)


def test_basic_returns_albedo():
    vloc, vnrm, alb, cam, light, lcol = _toy_geometry()
    out = ms.basic_material_shader(None, vloc, vnrm, alb, cam, light, lcol, 1, 1)
    assert torch.allclose(out, alb)
    print("ok: basic shader returns albedo unchanged")


def test_all_shaders_output_four_channels():
    vloc, vnrm, alb, cam, light, lcol = _toy_geometry()
    flat = _scalar(0.0)
    calls = {
        "lambert": (ms.lambert_shader, (_col(0, 0, 0), _scalar(1), flat, _scalar(1))),
        "phong": (
            ms.phong_shader,
            (_col(0, 0, 0), _scalar(1), _col(0.07, 0.07, 0.07), _scalar(30), flat, _scalar(1)),
        ),
        "standard": (
            ms.standard_shader,
            (_scalar(0.3), _scalar(0.0), _col(0, 0, 0), _scalar(1), _scalar(1), flat),
        ),
        "toon": (ms.toon_shader, (_col(0, 0, 0), _scalar(1), _scalar(3), flat)),
        "matcap": (ms.matcap_shader, (flat,)),
        "normal": (ms.normal_shader, (flat,)),
        "depth": (ms.depth_shader, (_scalar(0.1), _scalar(100.0))),
    }
    for name, (fn, extra) in calls.items():
        out = fn(None, vloc, vnrm, alb, cam, light, lcol, 1, 1, *extra)
        assert out.shape[-1] == 4, (name, out.shape)
        assert torch.isfinite(out).all(), name
        assert out[..., :3].min() >= 0.0 and out[..., :3].max() <= 1.0, name
    print("ok: all lit shaders return finite 4-channel colour in [0,1]")


def test_normal_shader_encodes_normal():
    vloc, vnrm, alb, cam, light, lcol = _toy_geometry()
    out = ms.normal_shader(None, vloc, vnrm, alb, cam, light, lcol, 1, 1, _scalar(0.0))
    # +Z normal -> (0.5, 0.5, 1.0)
    assert torch.allclose(out[0, 0, :3], torch.tensor([0.5, 0.5, 1.0]), atol=1e-5)
    print("ok: normal shader maps +Z to (0.5,0.5,1.0)")


def test_emissive_brightens():
    vloc, vnrm, _alb, cam, light, lcol = _toy_geometry()
    black = torch.zeros(1, 3, 4)
    dark = ms.lambert_shader(
        None, vloc, vnrm, black, cam, light, lcol, 1, 1,
        _col(0, 0, 0), _scalar(1), _scalar(0.0), _scalar(1),
    )
    glow = ms.lambert_shader(
        None, vloc, vnrm, black, cam, light, lcol, 1, 1,
        _col(1, 0, 0), _scalar(2.0), _scalar(0.0), _scalar(1),
    )
    assert glow[0, 0, 0] > dark[0, 0, 0]
    print("ok: emissive brightens output")


def test_metalness_and_roughness_have_effect():
    vloc, vnrm, alb, cam, light, lcol = _toy_geometry()
    # No corner-axis broadcast issues: use a single-point normal/location.
    vloc1 = torch.tensor([0.0, 0, 0]).view(1, 1, 3)
    vnrm1 = torch.tensor([0.0, 0, 1]).view(1, 1, 3)
    alb1 = torch.tensor([0.8, 0.1, 0.1, 0.0]).view(1, 1, 4)
    flat = _scalar(0.0)
    args = (None, vloc1, vnrm1, alb1, cam, light, lcol, 1, 1)
    dielectric = ms.standard_shader(*args, _scalar(0.25), _scalar(0.0), _col(0, 0, 0), _scalar(1), _scalar(1), flat)
    metal = ms.standard_shader(*args, _scalar(0.25), _scalar(1.0), _col(0, 0, 0), _scalar(1), _scalar(1), flat)
    rough = ms.standard_shader(*args, _scalar(0.9), _scalar(0.0), _col(0, 0, 0), _scalar(1), _scalar(1), flat)
    assert not torch.allclose(dielectric, metal), "metalness had no effect"
    assert not torch.allclose(dielectric, rough), "roughness had no effect"
    print("ok: metalness and roughness change PBR output")


# ---------------------------------------------------------------------------
# Mob.set_material wiring
# ---------------------------------------------------------------------------

def test_set_material_wires_shader_and_attrs():
    from algan import RED, Sphere

    s = Sphere(color=RED).set_material(
        MeshStandardMaterial(metalness=1.0, roughness=0.2, emissiveIntensity=3.0)
    )
    assert s.shader is ms.standard_shader
    assert _v(s.metalness)[0] == 1.0
    assert abs(_v(s.roughness)[0] - 0.2) < 1e-6
    assert _v(s.emissive_intensity)[0] == 3.0
    # An omitted material color preserves the mob's authored color.
    assert _v(s.color)[:3] == _v(RED)[:3]
    assert isinstance(s.material, MeshStandardMaterial)
    assert set(MeshStandardMaterial().get_shader_param_values()).issubset(
        set(s.shader_specific_param_names)
    )
    explicitly_colored = Sphere(color=RED).set_material(
        MeshStandardMaterial(color=WHITE)
    )
    assert _v(explicitly_colored.color)[:3] == _v(WHITE)[:3]
    print("ok: set_material wires shader, params and colour")


def test_physical_material_registers_transport_params():
    from algan import Sphere

    s = Sphere().set_material(
        MeshPhysicalMaterial(
            metalness=0.35,
            roughness=0.2,
            transmission=0.8,
            ior=1.45,
        )
    )
    params = s.get_shader_params()
    assert abs(float(params["metalness"].reshape(-1)[0]) - 0.35) < 1e-6
    assert abs(float(params["roughness"].reshape(-1)[0]) - 0.2) < 1e-6
    assert abs(float(params["transmission"].reshape(-1)[0]) - 0.8) < 1e-6
    assert abs(float(params["ior"].reshape(-1)[0]) - 1.45) < 1e-6
    assert {"metalness", "roughness", "transmission", "ior"}.issubset(
        s.shader_specific_param_names
    )
    print("ok: physical transport parameters are registered on the mob")


def test_normal_material_keeps_mob_colour():
    from algan import Sphere

    s = Sphere()
    before = _v(s.color)[:3]
    s.set_material(MeshNormalMaterial())
    assert _v(s.color)[:3] == before, "normal material must not override colour"
    assert s.shader is ms.normal_shader
    print("ok: normal/depth materials don't override mob colour")


def test_set_material_after_spawn_raises():
    from algan import Sphere
    from algan.animatable_base.mob import ModifiedProtectedAttributeError

    s = Sphere().spawn()
    try:
        s.set_material(MeshBasicMaterial())
    except ModifiedProtectedAttributeError:
        print("ok: set_material after spawn raises")
        return
    raise AssertionError("expected ModifiedProtectedAttributeError after spawn")


# ---------------------------------------------------------------------------
# Warnings for unsupported properties
# ---------------------------------------------------------------------------

def test_texture_and_unsupported_warnings():
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        MeshStandardMaterial(roughnessMap="dummy", wireframe=True).emit_warnings()
    text = " ".join(str(w.message) for w in rec)
    assert "roughnessMap" in text
    assert "wireframe" in text
    print("ok: texture/unsupported-property warnings fire")


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

def test_legacy_pbr_shader_still_works():
    from algan.rendering.shaders.pbr_shaders import basic_pbr_shader, null_shader

    vloc, vnrm, alb, cam, light, lcol = _toy_geometry()
    # basic_pbr_shader and null_shader don't use the memory arg (default_shader
    # does, so it's exercised by the render pipeline rather than this unit test).
    out = basic_pbr_shader(
        None, vloc, vnrm, alb[..., :4], cam, light, lcol, 1, 1, 0.5, 0.5
    )
    assert out.shape[-1] == 4 and torch.isfinite(out).all()
    out2 = null_shader(None, vloc, vnrm, alb, cam, light, lcol, 1, 1)
    assert torch.allclose(out2, alb)
    print("ok: legacy basic_pbr_shader / null_shader still work")


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
    print(f"\nAll {len(fns)} material tests passed.")


if __name__ == "__main__":
    _run_all()
