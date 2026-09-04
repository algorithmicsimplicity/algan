"""The shadow-terminator offset: per-hit displacement arithmetic (item 20).

``_shadow_terminator_delta`` moves a shadow-ray origin from the flat facet it
was found on onto the smooth surface the triangle's three VERTEX normals
imply. A rendered frame can say that something changed near the terminator --
``benchmarks/_shadow_terminator_ab.py`` is that evidence -- but it cannot say
the offset points the right way, clamps where it must, or stays EXACTLY zero
where it must: those are properties of single hits, invisible at image scale.
These tests drive the ``@ti.func`` directly from a tiny kernel -- the same
shape as ``test_nested_ior.py`` -- over ONE reference triangle (vertices
``(0,0,0) / (1,0,0) / (0,1,0)``, face normal +z), varying only vertex normals,
barycentrics and the oriented shading normal.

Two of the pinned exactness claims are load-bearing for byte-identity rather
than mere tidiness: a FLAT facet's delta is exactly zero BY CONSTRUCTION (the
constant-normal-field equality test -- without it float evaluation leaves
ulp-scale dust on such facets, which would set ``lifted = 1`` and relax the
horizon cull on geometry that never moved), and so is a hit whose ``prim``
indexes past a trimmed ``tri_norm`` (the classic wavefront path compacts that
array to the needs-normal prefix; reading past it was an out-of-bounds read).

Unmarked, so outside ``--fast``: nothing outside this feature can break it.
"""

# No ``from __future__ import annotations`` here, deliberately: the probe
# below defines a real ``@ti.kernel``, and stringised annotations turn
# ``ti.types.ndarray()`` into text that Taichi rejects at compile time.
import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

# The reference triangle, packed the way tri_pos/tri_norm carry it:
# three vertices' xyz in sequence.
_REF_POS = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float32)

UP = np.array([0, 0, 1], dtype=np.float32)
DOWN = -UP


def _kernels():
    from algan.rendering.raytracing import shading_taichi as k

    return k


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _p(bary):
    """The point barycentrics (w0, a, b) name inside the reference triangle."""
    w0, a, b = (np.float32(x) for x in bary)
    return w0 * _REF_POS[:3] + a * _REF_POS[3:6] + b * _REF_POS[6:]


def _n9(normals):
    return np.concatenate([np.asarray(n, dtype=np.float32) for n in normals])


# A convex patch bulging toward +z: each vertex normal tilts AWAY from the
# facet's interior. Its mirror -- every transverse component flipped -- is a
# concave (dimpled) field; both are used as-is below.
CONVEX = (_unit((-0.3, -0.3, 1)), _unit((0.3, -0.3, 1)), _unit((-0.3, 0.3, 1)))
CONCAVE = (_unit((0.3, 0.3, 1)), _unit((-0.3, 0.3, 1)), _unit((0.3, -0.3, 1)))
# Agreement far inside the equality tolerance (mutual dots are 1 to f32
# rounding): the constant-field gate must treat these exactly like +z,+z,+z.
NEAR_IDENTICAL = (
    _unit((1e-8, 0, 1)),
    _unit((0, -1e-8, 1)),
    _unit((-1e-8, 1e-8, 1)),
)

# Every cell the query does not read holds values that are obviously wrong
# (as ``_rows`` does in test_nested_ior.py): a reader indexing the wrong ROW
# or wrong prim slot answers something visibly insane instead of something
# plausible. The two junk patterns differ, and neither agrees across its own
# three normals, so even the flat-facet tests would catch a misread.
_JUNK_ROW0 = np.array([9, -9, 7, 7, 9, -9, -9, 7, 9], dtype=np.float32)
_JUNK_SLOT = np.array([-8, 8, -6, -6, -8, 8, 8, -6, -8], dtype=np.float32)


def _delta(normals, bary, p, snrm, prim=0, norm_prims=1):
    """``_shadow_terminator_delta`` once, against the reference triangle.

    Data lives in ROW 1 of both arrays (queried via ``f = 1``); row 0 and any
    unused prim column hold the junk patterns above. ``tri_pos`` always gets
    two prim columns while ``tri_norm`` gets ``norm_prims`` -- fewer than
    ``tri_pos`` reproduces the trimmed needs-normal prefix of the classic
    wavefront path, which is what defect 1's guard is for.
    """
    from algan.taichi_compat import ti

    k = _kernels()
    w0, a, b = (np.float32(x) for x in bary)
    tri_pos = np.empty((2, 2, 9), dtype=np.float32)
    tri_pos[0] = _JUNK_ROW0
    tri_pos[1, 0] = _REF_POS
    tri_pos[1, 1] = _JUNK_SLOT
    tri_norm = np.empty((2, norm_prims, 9), dtype=np.float32)
    tri_norm[0] = _JUNK_ROW0
    tri_norm[1, 0] = _n9(normals)
    if norm_prims > 1:
        tri_norm[1, 1] = _JUNK_SLOT
    pos_nd = ti.ndarray(ti.f32, shape=tri_pos.shape)
    norm_nd = ti.ndarray(ti.f32, shape=tri_norm.shape)
    pos_nd.from_numpy(tri_pos)
    norm_nd.from_numpy(tri_norm)
    p_nd = ti.ndarray(ti.f32, shape=(3,))
    s_nd = ti.ndarray(ti.f32, shape=(3,))
    p_nd.from_numpy(np.asarray(p, dtype=np.float32))
    s_nd.from_numpy(np.asarray(snrm, dtype=np.float32))
    out = ti.ndarray(ti.f32, shape=(3,))

    @ti.kernel
    def run(
        tp: ti.types.ndarray(),
        tn_: ti.types.ndarray(),
        pt: ti.types.ndarray(),
        sn: ti.types.ndarray(),
        o: ti.types.ndarray(),
    ):
        d = k._shadow_terminator_delta(
            1,
            prim,
            w0,
            a,
            b,
            ti.math.vec3(pt[0], pt[1], pt[2]),
            ti.math.vec3(sn[0], sn[1], sn[2]),
            tp,
            tn_,
        )
        o[0] = d[0]
        o[1] = d[1]
        o[2] = d[2]

    run(pos_nd, norm_nd, p_nd, s_nd, out)
    return out.to_numpy()


def test_a_flat_facet_gives_exactly_the_zero_vector():
    # All three vertex normals +z, several placements. EXACT equality, not
    # approx: this is the whole byte-identity contract for flat-shaded
    # geometry (cube/circuit arms of benchmarks/_shadow_terminator_ab.py).
    # The constant-field gate makes it true by construction; trusting the
    # d_i arithmetic would not (float dust on a constant field).
    flat = (UP, UP, UP)
    for bary in (
        (1 / 3, 1 / 3, 1 / 3),
        (0.5, 0.25, 0.25),
        (0.2, 0.7, 0.1),
        (0.25, 0.25, 0.5),
    ):
        assert _delta(flat, bary, _p(bary), UP).tolist() == [0.0, 0.0, 0.0]


def test_at_a_vertex_the_delta_is_exactly_zero():
    # At a corner only that vertex's term survives, weighted by 1, and its
    # depth below its OWN tangent plane is zero there: no curved field can
    # move a hit sitting exactly on a vertex.
    for bary, pnt in (
        ((1, 0, 0), (0, 0, 0)),
        ((0, 1, 0), (1, 0, 0)),
        ((0, 0, 1), (0, 1, 0)),
    ):
        got = _delta(CONVEX, bary, pnt, UP)
        assert got.tolist() == [0.0, 0.0, 0.0]


def test_a_convex_patch_lifts_the_point_toward_the_normals():
    # Normals splayed outward at the centroid: the facet is a chord BELOW the
    # surface its normal field implies, and the offset must lift toward it.
    bary = (1 / 3, 1 / 3, 1 / 3)
    pnt = _p(bary)
    d = _delta(CONVEX, bary, pnt, UP)
    assert d[2] > 0
    verts = _REF_POS.reshape(3, 3)
    for i in range(3):
        # Toward EVERY vertex normal, and toward every vertex tangent plane
        # the hit started below: dot(p + d - v_i, n_i) strictly improves on
        # dot(p - v_i, n_i).
        assert float(np.dot(d.astype(np.float64), CONVEX[i])) > 0.0
        before = float(
            np.dot(pnt.astype(np.float64) - verts[i].astype(np.float64), CONVEX[i])
        )
        after = float(
            np.dot(
                (pnt + d).astype(np.float64) - verts[i].astype(np.float64), CONVEX[i]
            )
        )
        assert after >= before - 1e-6
        # On or above the DISPLACED vertices' tangent planes: each vertex
        # moved onto its own plane by d_i = min(0, (p-v_i).n_i), and p+d is
        # their barycentric combination -- i.e. ON the smooth surface the
        # construction defines. This is the form the fix's guarantee takes;
        # the UNdisplaced planes cannot all be satisfied (measured here as
        # -0.0681 at v1/v2: the offset averages the three clamped lifts, and
        # at whichever vertex sits deepest an average of shallower, mutually
        # angled lifts necessarily undershoots -- Hanika's offset reduces the
        # self-intersection, it is not a half-space projection).
        disp = verts[i] + min(0.0, before) * CONVEX[i]
        lhs = float(
            np.dot((pnt + d).astype(np.float64) - disp.astype(np.float64), CONVEX[i])
        )
        assert lhs >= -1e-6


def test_a_concave_patch_does_not_sink_the_point():
    # The same normals converging instead of splaying: now every
    # (p - p_i) . n_i is POSITIVE -- the chord sits on or above every vertex
    # tangent plane -- so min(0, .) clamps all three depths to zero and the
    # displacement vanishes identically. Hanika's offset only ever LIFTS: it
    # measures how far the hit lies below each plane and corrects exactly
    # that, so "already above every plane" means "nothing to do", never a
    # negative correction pulling the origin into the surface.
    bary = (1 / 3, 1 / 3, 1 / 3)
    got = _delta(CONCAVE, bary, _p(bary), UP)
    assert got.tolist() == [0.0, 0.0, 0.0]


def test_a_back_facing_hit_lifts_the_other_way():
    # The sign rule keys off the ORIENTED shading normal: snrm negated flips
    # the frame the construction runs in, so wherever the front-facing case
    # lifts up this lifts DOWN. With the convex field at the centroid nothing
    # remains at all -- mirrored, every vertex plane is already satisfied,
    # which is the concave test's "only ever lifts" seen from the other side
    # (and why the brief's literal "convex, negated => delta.z < 0" reads
    # zero here); a field that still leaves depth in the mirrored frame is
    # what shows the direction flip.
    bary = (1 / 3, 1 / 3, 1 / 3)
    pnt = _p(bary)
    d = _delta(CONCAVE, bary, pnt, DOWN)
    assert d[2] < 0
    assert _delta(CONVEX, bary, pnt, DOWN).tolist() == [0.0, 0.0, 0.0]


def test_a_degenerate_normal_field_moves_nothing():
    # One unreadable vertex normal and the whole field is unusable: return
    # the zero vector rather than displace onto garbage. Same for a field
    # degenerate by tininess rather than by being zero.
    broken = (UP, (0, 0, 0), UP)
    tiny = (UP, (1e-12, 0, 0), UP)
    bary = (0.5, 0.25, 0.25)
    for normals in (broken, tiny):
        got = _delta(normals, bary, _p(bary), UP)
        assert got.tolist() == [0.0, 0.0, 0.0]


def test_the_magnitude_is_bounded_by_the_facet():
    # The offset needs no epsilon and no clamp because it cannot run away:
    # it is a barycentric average of per-vertex lifts along unit normals,
    # bounded by the triangle itself. Convex case: |delta| under the longest
    # edge (sqrt(2) here), and genuinely nonzero so the bound is not vacuous.
    bary = (1 / 3, 1 / 3, 1 / 3)
    d = _delta(CONVEX, bary, _p(bary), UP)
    norm = float(np.linalg.norm(d))
    assert 0.0 < norm < np.sqrt(2.0)


def test_a_prim_past_a_trimmed_tri_norm_moves_nothing():
    # Defect 1's guard. The classic wavefront path compacts tri_norm to the
    # needs-normal PREFIX (_flat_triangle_normal_trim guards its own read for
    # exactly this reason: a bare prim never consumes the shading normal), so
    # the helper can be handed a normal array SHORTER than tri_pos. Querying
    # past its end must answer zero without reading out of bounds -- while a
    # prim still inside the prefix keeps working on the very same arrays.
    bary = (1 / 3, 1 / 3, 1 / 3)
    within = _delta(CONVEX, bary, _p(bary), UP, prim=0, norm_prims=1)
    assert within[2] > 0
    past = _delta(CONVEX, bary, _p(bary), UP, prim=1, norm_prims=1)
    assert past.tolist() == [0.0, 0.0, 0.0]
    # And an untrimmed array (both sides two columns wide) behaves the same
    # for the in-prefix prim: the guard is a no-op there.
    untrimmed = _delta(CONVEX, bary, _p(bary), UP, prim=0, norm_prims=2)
    assert untrimmed.tolist() == within.tolist()


def test_near_identical_vertex_normals_are_treated_as_flat():
    # Defect 2's guard. These three normals agree to ~1e-8 -- mutual dots
    # equal 1.0 at f32, far inside the 1e-6 equality tolerance -- so the
    # constant-field gate must short-circuit them exactly like +z,+z,+z.
    # Without that gate the arithmetic itself leaks: evaluating
    # d_i = min(0, (p-p_i).n_i) on this field leaves ~3e-9 of dust in delta,
    # which would set lifted = 1 and relax the horizon cull on a facet whose
    # normal field IS constant. Measured, not hypothetical.
    bary = (0.5, 0.25, 0.25)
    got = _delta(NEAR_IDENTICAL, bary, _p(bary), UP)
    assert got.tolist() == [0.0, 0.0, 0.0]


def test_the_setting_is_reachable_from_settings_and_round_trips():
    from algan import SETTINGS
    from algan.rendering.raytracing import settings as rt_settings

    before = rt_settings.shadow_terminator
    try:
        SETTINGS.raytracing.experimental.set(shadow_terminator=True)
        assert rt_settings.shadow_terminator is True
        assert rt_settings.shadow_terminator_mode() == 1
        SETTINGS.raytracing.experimental.set(shadow_terminator=False)
        assert rt_settings.shadow_terminator is False
        assert rt_settings.shadow_terminator_mode() == 0
        SETTINGS.raytracing.experimental.set(shadow_terminator="relax")
        assert rt_settings.shadow_terminator == 2
        assert rt_settings.shadow_terminator_mode() == 2
    finally:
        rt_settings.set_shadow_terminator(before)


def test_the_experimental_toggle_is_not_settable_on_the_public_section():
    from algan import SETTINGS
    from algan.errors import AlganConfigurationError

    with pytest.raises(AlganConfigurationError, match="experimental"):
        SETTINGS.raytracing.set(shadow_terminator=True)


def test_only_an_exact_two_selects_the_diagnostic_arm():
    """Mode 2's images are knowingly wrong, so nothing may land on it by
    rounding or by dtype.

    An earlier setter truncated with ``int(enabled) == 2``, which put ``2.5``
    on the diagnostic arm, and gated on ``isinstance(x, (int, float))``, which
    sent ``np.float64(2.0)`` there (a float subclass) while ``np.int32(2)`` and
    ``np.float32(2.0)`` -- the same number -- came out plain-on. Comparison by
    value fixes both, and ``True`` must stay mode 1 even though ``bool`` is an
    ``int``.
    """
    from algan.rendering.raytracing import settings as rt_settings

    before = rt_settings.shadow_terminator
    try:
        for value in (2, 2.0, np.int32(2), np.float32(2.0), np.float64(2.0)):
            rt_settings.set_shadow_terminator(value)
            assert rt_settings.shadow_terminator_mode() == 2, value
        for value in (2.5, 3, -1, True, np.bool_(True), 1.5):
            rt_settings.set_shadow_terminator(value)
            assert rt_settings.shadow_terminator_mode() == 1, value
        for value in (False, 0, None, ""):
            rt_settings.set_shadow_terminator(value)
            assert rt_settings.shadow_terminator_mode() == 0, value
    finally:
        rt_settings.set_shadow_terminator(before)
