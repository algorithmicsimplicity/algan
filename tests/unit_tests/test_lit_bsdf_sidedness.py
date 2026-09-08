"""The lit BSDF and the NEE shadow gate must agree on a one-sided surface.

``_pt_lit_f_pdf`` is both ends of every MIS pair: next-event samples toward an
emitter and the BSDF paths that happen to find one are weighted against each
other with the densities it returns, which is what makes the power-heuristic
weights sum to one. The shadow gate beside it decides whether a next-event
connection is even traced.

The two normals involved differ only on a one-sided surface hit from behind:
``shade_n`` is the surface's *declared* side and keeps its outward normal
there, while ``spec_n`` is that normal turned to face the ray. The solids in
``algan/mobs/shapes_3d.py`` default to ``two_sided = False``, so an interior
face reached by an indirect bounce is the ordinary case rather than a corner.

A diffuse lobe reflects, so it must be zero when the light and the viewer are
on opposite sides of the surface -- otherwise the evaluator reports a response
for light entering the front and leaving the back of an opaque surface while
the shadow gate refuses exactly those directions, and the pair stops summing.

Outside the fast suite: it compiles Taichi kernels, and nothing elsewhere in
the codebase can break it (see ``tests/README.md`` on what earns a ``fast``
mark).

Note the absent ``from __future__ import annotations``: the probe kernel's
``ti.types.ndarray()`` annotations are evaluated at run time, and stringifying
them stops the kernel compiling.
"""

import pytest
import torch

from algan.rendering.raytracing.path_tracer_taichi import _pt_lit_f_pdf
from algan.rendering.taichi_runtime import init_taichi
from algan.taichi_compat import ti


@ti.kernel
def _diffuse_probe(
    rays: ti.types.ndarray(),
    lights: ti.types.ndarray(),
    out: ti.types.ndarray(),
):
    one = ti.math.vec3(1.0, 1.0, 1.0)
    for i in range(rays.shape[0]):
        # The surface's declared outward normal, independent of the ray.
        shade_n = ti.math.vec3(0.0, 0.0, 1.0)
        rd = ti.math.vec3(rays[i, 0], rays[i, 1], rays[i, 2]).normalized()
        wi = ti.math.vec3(lights[i, 0], lights[i, 1], lights[i, 2]).normalized()
        f_cos, pdf = _pt_lit_f_pdf(
            one * 0.8,  # e_diff
            one * 0.0,  # e_spec: isolate the diffuse lobe
            one * 0.04,  # f0
            0.5,  # roughness
            shade_n,
            rd,
            wi,
            0.0,  # w_pass
            1.0,  # w_diff
            0.0,  # w_spec
            0.0,  # w_trans
            0.0,  # eta == 0: opaque, so shade_n is NOT forced onto spec_n
            0.0,  # metalness
            one,  # albedo
            0.0,  # transmission
        )
        out[i, 0] = f_cos[0]
        out[i, 1] = pdf


# Outward normal is +Z. A ray travelling -Z arrives on the front face; one
# travelling +Z has passed through and strikes the same face from inside.
FRONT_HIT = (0.3, 0.0, -1.0)
BACK_HIT = (0.3, 0.0, 1.0)
LIGHT_IN_FRONT = (0.2, 0.0, 1.0)
LIGHT_BEHIND = (0.2, 0.0, -1.0)


def _probe(cases):
    init_taichi()
    rays = torch.tensor([c[0] for c in cases], dtype=torch.float32)
    lights = torch.tensor([c[1] for c in cases], dtype=torch.float32)
    out = torch.zeros((len(cases), 2), dtype=torch.float32)
    _diffuse_probe(rays, lights, out)
    return out


def test_a_front_lit_surface_seen_from_the_front_still_answers():
    """The ordinary case, and the control for the two below: both the light
    and the viewer sit above the declared side, so nothing is gated away.
    """
    out = _probe([(FRONT_HIT, LIGHT_IN_FRONT)])
    assert float(out[0, 0]) > 0.0, "an ordinary lit diffuse hit lost its response"
    assert float(out[0, 1]) > 0.0, "MIS assigned zero density to a lit diffuse hit"


def test_an_opaque_one_sided_surface_does_not_diffuse_light_through_itself():
    """Viewer behind, light in front. ``shade_n . wi`` is positive -- the light
    really is above the declared side -- but the outgoing direction is on the
    other side, so an opaque surface cannot carry that light to the camera.

    The shadow gate refuses this direction; before the evaluator applied the
    same test it still reported a diffuse response here, so the two MIS ends
    disagreed and the weights no longer summed to one.
    """
    out = _probe([(BACK_HIT, LIGHT_IN_FRONT)])
    assert float(out[0, 0]) == 0.0, "opaque surface diffused light through itself"
    assert float(out[0, 1]) == 0.0, "a refused direction kept a non-zero density"


def test_a_light_below_the_declared_side_stays_dark():
    """Both normals agree here, and always did: a diffuse lobe has no response
    below the horizon of the side it is declared on.
    """
    out = _probe([(BACK_HIT, LIGHT_BEHIND), (FRONT_HIT, LIGHT_BEHIND)])
    assert float(out[:, 0].abs().max()) == 0.0
    assert float(out[:, 1].abs().max()) == 0.0


@pytest.mark.parametrize("tilt", [0.05, 0.5, 0.95])
def test_the_gate_follows_the_viewer_side_at_every_grazing_angle(tilt):
    """The two arms must stay complementary as the light swings across the
    surface: whichever side the viewer is on, exactly one of them answers.
    """
    light = (tilt, 0.0, (1.0 - tilt**2) ** 0.5)
    front, back = _probe([(FRONT_HIT, light), (BACK_HIT, light)])
    assert float(front[0]) > 0.0
    assert float(back[0]) == 0.0
