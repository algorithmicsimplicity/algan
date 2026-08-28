"""Unit tests for ``manim_shader``, Algan's port of Manim's default 3-D
lighting (the vendored ``get_shaded_rgb``).

Feature test, deliberately outside the fast suite: these can only fail when
this shader's own modules change.
"""

import inspect

import numpy as np
import torch

import algan
from algan.external_libraries.manim.utils.color.core import get_shaded_rgb
from algan.rendering.raytracing.settings import _shader_material_id
from algan.rendering.shaders import material_shaders as ms
from algan.rendering.shaders.fragment_shaders import STAGE_MANIM, resolve_stage
from algan.rendering.shaders.materials import ManimMaterial
from algan.utils.color_space import linear_to_srgb, srgb_to_linear

# Mid-range albedo, so no row of the sweep below reaches either end of the
# range and the comparison against the vendored function (which does not
# clamp) is not confounded by the shader's [0, 1] clamp.
_ALBEDO = np.array([0.4, 0.35, 0.3], dtype=np.float64)
_LIGHT_SOURCE = np.array([0.0, 0.0, 4.0], dtype=np.float64)

# Unnormalized normals spanning the whole lobe: straight-on, oblique,
# edge-on, oblique from behind, straight back-facing (full negative lobe) and
# another behind -- enough signed variety that dropping the ``light < 0``
# halving branch would move several rows far outside any tolerance.
_NORMALS = [
    (0.0, 0.0, 7.0),
    (1.0, 1.0, 2.0),
    (2.0, 0.0, 0.0),
    (3.0, -1.0, -4.0),
    (0.0, 0.0, -5.0),
    (-1.0, 2.0, -1.0),
]


def _display_output(out):
    """``manim_shader``'s RGB as display-referred sRGB.

    Under the linear working space the shader returns linear light, which is
    one transfer function away from what Manim's arithmetic produces; under
    the display-referred setting it is already there.
    """
    from algan.rendering.raytracing import settings as rt_settings

    rgb = out[..., :3]
    return linear_to_srgb(rgb) if rt_settings.linear_color_space else rgb


def _run_shader(light_color):
    """Run ``manim_shader`` over the sweep, mirroring the pipeline's colour
    handling: ``_ALBEDO`` is authored display-referred (what Manim's function
    consumes), and under the linear working space the render boundary decodes
    it to linear before a stage sees it (``scene_builder._decode_merged_colors``).
    """
    from algan.rendering.raytracing import settings as rt_settings

    n = len(_NORMALS)
    vloc = torch.zeros(n, 1, 3)
    vnrm = torch.tensor(_NORMALS, dtype=torch.float32).view(n, 1, 3)
    rgb = torch.tensor(_ALBEDO, dtype=torch.float32)
    if rt_settings.linear_color_space:
        rgb = srgb_to_linear(rgb)
    rgb = rgb.expand(n, 3)
    alb = torch.cat([rgb, torch.zeros(n, 1)], dim=-1).view(n, 1, 4)
    cam = torch.zeros(1, 1, 3)
    light = torch.tensor(_LIGHT_SOURCE, dtype=torch.float32).view(1, 1, 3)
    return ms.manim_shader(None, vloc, vnrm, alb, cam, light, light_color, 1.0, 1.0)


def test_matches_vendored_get_shaded_rgb_in_display_terms():
    """The external invariant: white intensity-1 light reproduces the vendored
    Manim function, evaluated in the display-referred terms Manim works in.
    """
    out = _run_shader(torch.ones(1, 1, 4))
    disp = _display_output(out).view(len(_NORMALS), 3)

    expected = []
    for normal in _NORMALS:
        unit = np.asarray(normal, dtype=np.float64)
        unit /= np.linalg.norm(unit)
        expected.append(
            get_shaded_rgb(_ALBEDO.copy(), np.zeros(3), unit, _LIGHT_SOURCE)
        )
    expected = torch.tensor(np.stack(expected), dtype=torch.float32)

    torch.testing.assert_close(disp, expected, atol=1e-5, rtol=1e-5)


def test_zero_colour_light_row_contributes_nothing():
    """A zero-RGB light row -- a light outside its lifespan -- leaves the base
    colour untouched, whatever direction the surface faces. This is what the
    tinted design buys: every vis-weighted term carries the light colour, so
    despawned rows go inert with no explicit gate.
    """
    out = _run_shader(torch.zeros(1, 1, 4))
    disp = _display_output(out).view(len(_NORMALS), 3)
    expected = (
        torch.tensor(_ALBEDO, dtype=torch.float32).expand(len(_NORMALS), 3).clone()
    )
    torch.testing.assert_close(disp, expected, atol=1e-6, rtol=1e-5)


def test_resolves_to_stage_manim_and_material_id_zero():
    assert resolve_stage(ms.manim_shader) is STAGE_MANIM
    assert _shader_material_id(ms.manim_shader) == 0


def test_manim_material_contract():
    m = ManimMaterial()
    assert m.shader is ms.manim_shader
    params = list(inspect.signature(m.shader).parameters)
    fixed = list(inspect.signature(ms.basic_material_shader).parameters)
    # The canonical nine, in order, then exactly the material's own params.
    assert params[: len(fixed)] == fixed
    assert list(m.get_shader_param_values().keys()) == params[len(fixed) :]
    assert {"manim_shader", "STAGE_MANIM", "ManimMaterial"} <= set(algan.__all__)
