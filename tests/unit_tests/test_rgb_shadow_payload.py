"""The RGB shadow-visibility payload and the colour a transmissive blocker adds.

A shadow query used to return one SCALAR visibility per light, so light that
had passed through green glass arrived grey (renderer_audit REPORT.md ss4.10).
The payload is now RGB end to end, and -- behind ``ALGAN_RGB_SHADOW_TINT`` --
a transmissive surface tints what it passes with its albedo and absorbs
Beer-Lambert over its interior chord.

Three properties would silently regress and are pinned here:

* **The reduction invariant.** Every consumer multiplies the payload straight
  into RGB except the energy-budget sums, which reduce it to a max component.
  With all three channels equal -- which is every scene with no coloured
  transmissive blocker -- those reductions must be EXACT (no arithmetic), or
  ordinary renders move. ``_vis_max_component`` of an equal-channel vector is
  asserted bit-identical, and ``_shadow_pass_through`` is asserted to carry
  today's scalar formula into every channel of its payload.
* **The direction of the tint.** A green transmissive blocker must pass more
  green than red -- and a non-transmissive or metallic one must pass nothing,
  exactly as before.
* **The coverage floor's slack.** The interior-absorption pairing reads a
  hit's coverage off its alpha, and an exact ``alpha >= 1.0`` misreads a
  fully-opaque surface whose reconstructed alpha lands an ulp short -- the ray
  then loses its whole interior absorption and its pixel comes out brighter
  than its neighbours. ``_SOLID_COVERAGE_MIN`` must keep swallowing that
  shortfall (the constant's comment carries the render-measured evidence).

The Beer-Lambert coefficient itself is pinned on the packing side (tri_extra
columns 12..14 carry ``attenuation_sigma``, width grew 12 -> 15): the march's
pairing rule needs a full scene render, which the calibration scenes in
``benchmarks/renderer_audit`` measure.

Outside the fast suite: it compiles Taichi kernels, and nothing elsewhere in
the codebase can break it (see ``tests/README.md`` on what earns a ``fast``
mark).

Note the absent ``from __future__ import annotations`` (same reason as
``test_dielectric_fresnel.py``, whose probe-kernel pattern this follows), and
that no test here calls ``ti.init``: the kernels run on whatever Taichi state
``import algan`` set up.
"""

import os
import subprocess
import sys

import pytest
import torch

from algan.rendering.raytracing.primitives import LogicalPNTrianglePrimitive
from algan.rendering.raytracing.raytrace_kernels_taichi import (
    _EXTRA_SIGMA,
    _EXTRA_W,
    _SOLID_COVERAGE_MIN,
    _flat_triangle_color,
    _shadow_pass_through,
)
from algan.rendering.raytracing.shading_taichi import (
    SHADOW_VIS_CHANNELS,
    _light_vis,
    _vis_max_component,
    max_shadow_lights,
)
from algan.rendering.shaders.materials import (
    MeshPhysicalMaterial,
    _attenuation_sigma,
)
from algan.taichi_compat import ti


# tri_extra row for one triangle: per-corner transport in 0-11, sigma in 12-14.
def _tri_extra_row(metalness, ior, transmission, sigma):
    row = torch.zeros((1, 1, _EXTRA_W), dtype=torch.float32)
    for corner in range(3):
        row[0, 0, corner * 2] = metalness
        row[0, 0, 6 + corner] = ior
        row[0, 0, 9 + corner] = transmission
    row[0, 0, _EXTRA_SIGMA : _EXTRA_SIGMA + 3] = torch.tensor(
        sigma, dtype=torch.float32
    )
    return row


@ti.kernel
def _probe_pass_through(
    extra: ti.types.ndarray(), cmeta: ti.types.ndarray(), out: ti.types.ndarray()
):
    for i in range(out.shape[0]):
        # hit_type 1 = triangle; barycentrics weight corner 0 fully.
        v = _shadow_pass_through(
            0,
            0,
            1,
            1.0,
            0.0,
            0.0,
            extra,
            cmeta,  # circuit_meta: unused for triangle hits
            ti.math.vec3(1.0, 1.0, 1.0),
        )
        out[i, 0] = v[0]
        out[i, 1] = v[1]
        out[i, 2] = v[2]


def pass_through(metalness, ior, transmission, sigma=(0.0, 0.0, 0.0)):
    extra = _tri_extra_row(metalness, ior, transmission, sigma)
    cmeta = torch.zeros((1, 1, 24), dtype=torch.float32)
    out = torch.zeros((1, 3), dtype=torch.float32)
    _probe_pass_through(extra, cmeta, out)
    return out[0].tolist()


# The f32 predecessor of 1.0: the value a fully-opaque hit's alpha takes when
# the barycentric blend reconstructs an ulp short.
_ONE_ULP_BELOW_1 = 1.0 - 2.0**-24

# Four barycentrics whose fully-opaque blend reconstructs to exactly that
# value (drawn once with the seeded sampler in
# test_a_fully_opaque_hit_is_never_below_the_solid_coverage_floor and pinned
# as literals, so the adversarial class survives any RNG change).
_ULP_UNDER_BARYS = (
    (0.2713892161846161, 0.29845374822616577),
    (0.4935935437679291, 0.187627375125885),
    (0.4786015450954437, 0.04310973361134529),
    (0.32275792956352234, 0.17284135520458221),
)


@ti.kernel
def _probe_opaque_alpha(
    tri_colors: ti.types.ndarray(),
    tri_uvs: ti.types.ndarray(),
    tri_tex_meta: ti.types.ndarray(),
    textures: ti.types.ndarray(),
    bary: ti.types.ndarray(),
    out: ti.types.ndarray(),
):
    # Alpha exactly as both shadow guards receive it: _flat_triangle_color on
    # a triangle whose corner alphas are all 1.0 (prim 0 < the coloured count,
    # so the per-vertex branch runs), then the march's clamp.
    for i in range(bary.shape[0]):
        a = bary[i, 0]
        b = bary[i, 1]
        color4 = ti.math.vec4(0.0)
        alpha = 0.0
        color4, alpha = _flat_triangle_color(
            0, 0, 1.0 - a - b, a, b, tri_colors, tri_uvs, tri_tex_meta, textures, 1
        )
        out[i] = ti.math.clamp(alpha, 0.0, 1.0)


def opaque_hit_alphas(bary):
    """Production alpha for each (a, b) row against a fully-opaque triangle."""
    tri_colors = torch.zeros((1, 1, 3, 5), dtype=torch.float32)
    tri_colors[..., :3] = 1.0
    tri_colors[..., 4] = 1.0
    # Dummies matching scene_builder's empty-scene allocations; prim 0 never
    # reaches them, but both branches compile.
    tri_uvs = torch.zeros((1, 1, 6), dtype=torch.float32)
    tri_tex_meta = torch.full((1, 10), -1, dtype=torch.int32)
    textures = torch.zeros((1, 1, 5), dtype=torch.float32)
    out = torch.zeros(bary.shape[0], dtype=torch.float32)
    _probe_opaque_alpha(
        tri_colors, tri_uvs, tri_tex_meta, textures, bary.contiguous(), out
    )
    return out


def test_equal_channel_payload_reduces_to_the_scalar_formula():
    """With a white tint the payload is today's scalar formula in every
    channel: pass = transmission * (1 - metalness) * (1 - F0).
    """
    ior = 1.5
    f0 = ((1.0 - ior) / (1.0 + ior)) ** 2
    expected = 0.75 * (1.0 - 0.25) * (1.0 - f0)
    r, g, b = pass_through(metalness=0.25, ior=ior, transmission=0.75)
    assert r == pytest.approx(expected, abs=1e-6)
    assert g == pytest.approx(expected, abs=1e-6)
    assert b == pytest.approx(expected, abs=1e-6)


def test_a_coloured_transmissive_blocker_tints_towards_its_albedo():
    """Green glass passes green and no red: each channel scales by the
    albedo channel it carries (clamped, like ``_scatter_impl``'s tint).
    """
    ior = 1.5
    f0 = ((1.0 - ior) / (1.0 + ior)) ** 2

    @ti.kernel
    def probe(
        extra: ti.types.ndarray(), cmeta: ti.types.ndarray(), out: ti.types.ndarray()
    ):
        v = _shadow_pass_through(
            0,
            0,
            1,
            1.0,
            0.0,
            0.0,
            extra,
            cmeta,
            ti.math.vec3(0.0, 1.0, 0.0),
        )
        for c in ti.static(range(3)):
            out[c] = v[c]

    extra = _tri_extra_row(0.0, ior, 1.0, (0.0, 0.0, 0.0))
    cmeta = torch.zeros((1, 1, 24), dtype=torch.float32)
    out = torch.zeros(3, dtype=torch.float32)
    probe(extra, cmeta, out)
    expected = 1.0 * (1.0 - f0)
    assert out[0].item() == pytest.approx(0.0, abs=1e-6)
    assert out[1].item() == pytest.approx(expected, abs=1e-6)
    assert out[2].item() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize(
    ("metalness", "transmission"), [(-1.0, 1.0), (1.0, 1.0), (0.0, 0.0)]
)
def test_non_transmissive_and_metallic_blockers_pass_nothing(metalness, transmission):
    """The sentinel (-1), full metalness and zero transmission all keep the
    old behaviour exactly: zero in every channel.
    """
    r, g, b = pass_through(metalness=metalness, ior=1.5, transmission=transmission)
    assert (r, g, b) == (0.0, 0.0, 0.0)


def test_max_component_reduction_is_exact_for_equal_channels():
    """The energy-budget sums stay scalar by reducing to the max component;
    for equal channels that must BE the channel value bit for bit -- any
    arithmetic there would move every ordinary lit render.
    """

    @ti.kernel
    def probe(payload: ti.types.ndarray(), out: ti.types.ndarray()):
        vec = ti.math.vec3(payload[0], payload[1], payload[2])
        out[0] = _vis_max_component(vec)

    for s in (0.0, 1.0, 0.5, 0.927, 1e-4, 0.123457):
        v = torch.full((3,), s, dtype=torch.float32)
        out = torch.zeros(1, dtype=torch.float32)
        probe(v, out)
        # Bit-exact roundtrip: the reduction IS the channel value, not a
        # recomputation of it.
        assert out[0].item() == v[0].item()


def test_light_vis_reads_the_channel_major_layout():
    """``_light_vis`` picks light li's triple from indices 3*li..3*li+2 and
    stays fully lit past the cap. The indices themselves are written through
    ``light_vis_index`` in the payload producers, so pinning the READ here
    pins the layout the writers use.
    """

    @ti.kernel
    def probe(payload: ti.types.ndarray(), lights: int, out: ti.types.ndarray()):
        # The shade kernels hold ``vis`` as a fixed-length ti.Vector, so the
        # probe does too (that is the type _light_vis sees in production).
        vec = ti.Vector([0.0] * (3 * max_shadow_lights))
        for i in ti.static(range(3 * max_shadow_lights)):
            vec[i] = payload[i]
        for li in range(lights):
            v = _light_vis(1, vec, li)
            out[li, 0] = v[0]
            out[li, 1] = v[1]
            out[li, 2] = v[2]

    payload = torch.ones(3 * max_shadow_lights, dtype=torch.float32)
    payload[3 * 1 + 0] = 0.25
    payload[3 * 1 + 1] = 0.5
    payload[3 * 1 + 2] = 0.75
    for c in range(SHADOW_VIS_CHANNELS):
        payload[c] *= 0.9
    # One light PAST the cap keeps its all-lit default.
    lights = max_shadow_lights + 1
    out = torch.zeros((lights, 3), dtype=torch.float32)
    probe(payload, lights, out)
    assert out[0].tolist() == pytest.approx([0.9, 0.9, 0.9])
    assert out[1].tolist() == pytest.approx([0.25, 0.5, 0.75])
    assert out[max_shadow_lights - 1].tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert out[max_shadow_lights].tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_index_helper_is_channel_major():
    assert SHADOW_VIS_CHANNELS == 3
    # light_vis_index is a @ti.func (kernel-only), so its formula is pinned
    # through the read side in test_light_vis_reads_the_channel_major_layout;
    # this pins the payload LENGTH convention the helper serves.
    assert 3 * max_shadow_lights == SHADOW_VIS_CHANNELS * max_shadow_lights


def test_sigma_reaches_the_packed_extra_block():
    """Beer-Lambert plumbing: the primitive's ``attenuation_sigma`` shader
    parameter lands in tri_extra columns 12..14, per primitive, and a material
    without attenuation packs zeros there. Width grows 12 -> 15.
    """
    mat = MeshPhysicalMaterial(
        transmission=1.0,
        ior=1.5,
        attenuation_color=0x00FF00,
        attenuation_distance=0.5,
    )
    params = {
        name: value
        for name, value in mat.get_shader_param_values().items()
        if isinstance(value, torch.Tensor)
    }
    corners = torch.rand(1, 2, 3, 3)
    colors = torch.rand(1, 2, 3, 5)
    prim = LogicalPNTrianglePrimitive(
        corners=corners, colors=colors, shader=None, **params
    )
    packed = prim._pack_surface_extra("test")
    assert packed.shape[-1] == _EXTRA_W
    expected = _attenuation_sigma(0x00FF00, 0.5)
    # Sigma is per-primitive: both triangles carry the same triple.
    assert packed[0, 0, _EXTRA_SIGMA : _EXTRA_SIGMA + 3].tolist() == pytest.approx(
        expected.flatten().tolist(), abs=1e-6
    )
    assert packed[0, 1, _EXTRA_SIGMA : _EXTRA_SIGMA + 3].tolist() == pytest.approx(
        expected.flatten().tolist(), abs=1e-6
    )

    bare = LogicalPNTrianglePrimitive(corners=corners, colors=colors, shader=None)
    bare_packed = bare._pack_surface_extra("test")
    assert bare_packed.shape[-1] == _EXTRA_W
    assert bare_packed[0, :, _EXTRA_SIGMA : _EXTRA_SIGMA + 3].abs().max().item() == 0.0


_GATE_OFF_PROBE = """
import os
assert os.environ.get("ALGAN_RGB_SHADOW_TINT") == "0"
import torch
from algan.taichi_compat import ti
from algan.rendering.raytracing.raytrace_kernels_taichi import (
    _EXTRA_W, _shadow_pass_through,
)

extra = torch.zeros((1, 1, _EXTRA_W), dtype=torch.float32)
for corner in range(3):
    extra[0, 0, corner * 2] = 0.0      # metalness
    extra[0, 0, 6 + corner] = 1.5      # ior
    extra[0, 0, 9 + corner] = 1.0      # transmission
cmeta = torch.zeros((1, 1, 24), dtype=torch.float32)

@ti.kernel
def probe(e: ti.types.ndarray(), cm: ti.types.ndarray(), out: ti.types.ndarray()):
    v = _shadow_pass_through(0, 0, 1, 1.0, 0.0, 0.0, e, cm,
                             ti.math.vec3(0.0, 1.0, 0.0))
    for c in ti.static(range(3)):
        out[c] = v[c]

out = torch.zeros(3, dtype=torch.float32)
probe(extra, cmeta, out)
f0 = ((1.0 - 1.5) / (1.0 + 1.5)) ** 2
expected = 1.0 * (1.0 - f0)
for c in range(3):
    assert abs(out[c].item() - expected) < 1e-6, out.tolist()
print("gate-off probe ok")
"""


def test_gate_off_ignores_the_tint():
    """With ALGAN_RGB_SHADOW_TINT=0 the producer returns today's achromatic
    value in every channel -- the green tint above must NOT appear. The gate
    is baked in at kernel compile, so this arm has to be its own process (see
    rgb_shadow_tint).
    """
    import tempfile

    env = dict(os.environ)
    env["ALGAN_RGB_SHADOW_TINT"] = "0"
    env.setdefault("ALGAN_USE_DAEMON", "0")
    env.setdefault("ALGAN_AUTO_DAEMON", "0")
    # Taichi's inspector needs real source: run from a temp file, not -c.
    with tempfile.NamedTemporaryFile(
        "w", suffix="_gate_off_probe.py", delete=False
    ) as f:
        f.write(_GATE_OFF_PROBE)
        path = f.name
    try:
        result = subprocess.run(
            [sys.executable, path],
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
    finally:
        os.unlink(path)
    assert result.returncode == 0, result.stderr


def test_a_fully_opaque_hit_is_never_below_the_solid_coverage_floor():
    """The interior-absorption pairing reads a hit's coverage off its alpha,
    so an exact ``alpha >= 1.0`` misreads a FULLY-opaque surface as partly
    transparent whenever its reconstructed alpha lands an ulp short of 1.0 --
    and a hit that misses the floor never opens (or never closes) the medium,
    so that ray loses its WHOLE interior absorption: salt-and-pepper speckle,
    every affected pixel brighter than its neighbours, growing with the chord
    being dropped. The guards therefore compare against
    ``_SOLID_COVERAGE_MIN``; this pins that the floor really swallows the
    shortfall it exists for.

    The blend is the culprit: alpha is ``w0*a0 + w1*a1 + w2*a2`` with
    ``w0 = 1 - a - b``, and in f32 that sum is not associative -- with all
    three corner alphas exactly 1.0 it reconstructs to exactly one ulp below
    1.0 for a non-negligible share of barycentrics. How large a share depends
    on operand order and on whether the backend contracts the blend into FMAs
    (which is why the constant's comment declines to quote a number), so the
    fact is pinned under ONE stated evaluation -- seeded draws evaluated
    left-to-right in torch f32, nothing reassociated -- plus literal
    barycentrics known to land an ulp short, and both must clear the shipped
    floor. The same points then go through the real ``_flat_triangle_color``,
    clamped as the march clamps: today's CPU backend folds the blend to
    exactly 1.0 there, so only the floor is asserted of the kernel -- this
    pin has to survive a backend where it does not fold.
    """
    g = torch.Generator().manual_seed(20260823)
    weights = torch.rand(3, 200_000, generator=g, dtype=torch.float32)
    weights /= weights.sum(0, keepdim=True)
    u = weights[1].contiguous()
    v = weights[2].contiguous()

    # The intersector derives the third weight and blends left-to-right;
    # evaluate the same way here -- one rounding per op, no reassociation.
    w0 = (1.0 - u) - v
    blend = (w0 + u) + v

    # The float fact itself: a non-negligible share of fully-opaque blends
    # miss 1.0, and every miss is EXACTLY the f32 predecessor of 1.0.
    under = blend < 1.0
    assert float(under.float().mean()) > 0.005  # ~4.6% at this seed
    one_ulp_below = torch.tensor(_ONE_ULP_BELOW_1, dtype=torch.float32)
    assert bool((blend[under] == one_ulp_below).all())

    pinned_u = torch.tensor([a for a, _ in _ULP_UNDER_BARYS], dtype=torch.float32)
    pinned_v = torch.tensor([b for _, b in _ULP_UNDER_BARYS], dtype=torch.float32)
    pinned_blend = ((1.0 - pinned_u) - pinned_v + pinned_u) + pinned_v
    assert bool((pinned_blend == one_ulp_below).all())

    # Corners, edge midpoints and the centroid reconstruct exactly; include
    # them so the floor is pinned over degenerate hits too.
    fixed = torch.tensor(
        [
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (0.5, 0.0),
            (0.0, 0.5),
            (0.5, 0.5),
            (1.0 / 3.0, 1.0 / 3.0),
        ],
        dtype=torch.float32,
    )
    fixed_blend = (((1.0 - fixed[:, 0]) - fixed[:, 1]) + fixed[:, 0]) + fixed[:, 1]

    # THE regression assertion: a fully-opaque surface is always treated as
    # fully covering. Against a literal 1.0 floor this fails on every value
    # above that is one ulp short.
    floor = torch.tensor(_SOLID_COVERAGE_MIN, dtype=torch.float32)
    every_blend = torch.cat([blend, pinned_blend, fixed_blend])
    assert bool((every_blend >= floor).all())

    # And the production producer agrees: through _flat_triangle_color on a
    # fully-opaque triangle, clamped as the march clamps, no barycentric
    # lands below the floor either.
    every_bary = torch.cat(
        [
            torch.stack([u, v], dim=1),
            torch.stack([pinned_u, pinned_v], dim=1),
            fixed,
        ]
    )
    assert bool((opaque_hit_alphas(every_bary) >= floor).all())
