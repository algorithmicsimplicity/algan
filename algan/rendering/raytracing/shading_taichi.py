"""Per-fragment (Taichi) shading for Algan's deterministic ray tracer.

The deterministic ray tracer normally shades *per vertex*: the PyTorch material
shader (:mod:`algan.rendering.shaders.material_shaders`) is evaluated at each
triangle corner before upload and the kernel only interpolates the baked colors
(Gouraud shading). When fragment shading is enabled
(:func:`algan.rendering.raytracing.primitives.set_fragment_shading`) the kernel
instead receives the *raw albedo* plus a compact per-primitive parameter block,
interpolates the surface normal at the hit, and evaluates the lighting model
here -- per fragment (Phong shading), so specular highlights stay crisp and
coarse meshes shade smoothly.

Shading is expressed as **stages** with a single uniform ``@ti.func`` contract
(see ``_stage_phong`` etc.). A per-primitive **pipeline** is an ordered list of
stages run left-to-right, each receiving the previous stage's output colour --
so a user recolour stage can feed a built-in lighting stage. The built-in *core
lit* materials are the first stages: the legacy diffuse
:func:`~algan.rendering.shaders.pbr_shaders.default_shader`, ``MeshBasicMaterial``
(unlit), ``MeshLambertMaterial``, ``MeshPhongMaterial``, ``MeshStandardMaterial``
and ``MeshPhysicalMaterial``.
Custom user stages (also ``@ti.func``) are composed into per-pipeline funcs by
:func:`make_pipeline_func` and injected into the shade kernel as a flat
``ti.template()`` tuple (see ``taichi-func-injection``).

Per-primitive **pipeline id** (``pid_arr``); ids 0-5 are the built-in
single-stage pipelines, ids >= ``_USER_PIPELINE_BASE`` index the injected user
pipeline tuple::

    0  default diffuse      3  phong  (Blinn-Phong diffuse + specular)
    1  basic / unlit / passthrough     4  standard (Cook-Torrance GGX PBR)
    2  lambert (diffuse)   5  physical (standard + clearcoat / sheen / ior)

Built-in material parameter block ``params[.., off:off+MAT_W]`` (per primitive),
canonical slot layout (``off`` is the stage's base offset, 0 for a built-in
single-stage pipeline)::

    0..2 emissive   3 emissive_intensity   4..6 specular   7 shininess
    8 roughness     9 metalness            10 flat_shading  11 env_map_intensity
    12 ior          13 specular_intensity  14..16 specular_color
    17 clearcoat    18 clearcoat_roughness 19 sheen         20 sheen_roughness
    21..23 sheen_color   24 transmission   25 iridescence (accepted, unused --
    matches the PyTorch shader)      26 one_sided (declared by the GEOMETRY,
    not by the material -- see ``_MAT_ONE_SIDED``)
    27..29 attenuation_sigma (Beer-Lambert absorption coefficient over the
    segment a ray spends inside a transmissive solid; applied by the wavefront
    bounce loop, not by the shading stages)

Every slot above carries a 0.0 default that means "the behaviour that existed
before" (the padding rule on ``_MAT_ONE_SIDED`` below), so the zero-padded
block of a custom pipeline keeps its historical look slot by slot.

The lighting math mirrors ``material_shaders.py`` exactly (same GGX/Smith/Schlick
terms, ``AMBIENT_STRENGTH``, ``light_intensity == ambient == 1``) and reproduces
its multi-light behaviour: each light is applied in sequence with the running
colour as the albedo (the renderer's vertex path overwrites the colour per
light), which is identical to a single light -- the common case.
"""

import taichi as ti

from algan.environment import env_int

# Width of the built-in per-primitive material parameter block (see slot map).
MAT_W = 30

# Slot 26 of that block: 1.0 when the primitive's geometry declares an outside,
# so a back-facing hit is shaded with its own normal instead of the viewer's
# side (``Mob.two_sided`` False). Not a material property -- the MOB declares
# it -- so it is not in ``_MAT_SLOTS``. Its 0.0 default ("two-sided", the
# historical behaviour) is safe because of THE PADDING RULE this block lives
# by: every slot's 0.0 must mean "the behaviour that existed before". A custom
# fragment pipeline's block is a different layout entirely and is zero-padded
# to this width when the two share a scene, so a zero read from the padding
# has to be the pre-existing behaviour of whatever reads that slot. That rule,
# not slot position, is what makes appending slots after this one safe -- new
# entries must carry a 0.0 that means "as before" (slots 27..29 do: 0.0 is no
# volumetric absorption, which is what every material did before they existed).
_MAT_ONE_SIDED = 26

# Slots 27..29 of that block: the Beer-Lambert absorption coefficient of the
# medium a transmissive solid encloses, per channel. Read by the wavefront
# bounce loop over the segment a ray spends inside, never by a shading stage.
_MAT_ATTENUATION_SIGMA = 27

# Built-in single-stage pipeline ids.
_MID_DEFAULT = 0
_MID_UNLIT = 1
_MID_LAMBERT = 2
_MID_PHONG = 3
_MID_STANDARD = 4
_MID_PHYSICAL = 5

# Pipeline ids at or above this index address the injected user pipeline tuple
# (``frag_pipelines``): user pipeline k has id ``_USER_PIPELINE_BASE + k``.
_USER_PIPELINE_BASE = 6

# Base ambient coefficient (matches material_shaders.AMBIENT_STRENGTH).
AMBIENT_STRENGTH = 0.1

# The same fill, expressed in linear light. 0.1 was chosen as a display-referred
# coefficient, and moving the working space without moving it would have made
# the ambient nearly nine times brighter: 0.1 of linear light encodes to byte
# 89, where 0.1 of an encoded value is byte 26, so every shadowed and unlit
# region would have lifted. srgb_to_linear(0.1) = 0.01003, so 0.01 is the same
# fill the old pipeline delivered -- the constant changes because the units
# changed, not because the look was retuned.
AMBIENT_STRENGTH_LINEAR = 0.01


def _ambient_strength():
    """The ambient coefficient for the active working space.

    A Python-level function rather than a constant because the two spaces need
    different numbers for the same result; call it inside ``ti.static`` so the
    value is folded in when the kernel compiles.
    """
    return AMBIENT_STRENGTH_LINEAR if _linear_color_space() else AMBIENT_STRENGTH


def _linear_color_space():
    """True when shading runs in the linear working colour space.

    Gates the two gamma-era compensations defined here -- ``_energy_scale``'s
    illumination budget and ``_run_frag_pipeline``'s peak bound. They exist to
    normalise away the overshoot that summing sRGB-encoded light creates; in
    linear light lights genuinely add (as three.js accumulates them), so both
    mechanisms are off and this is their shared gate.

    The import is local *on purpose*: ``settings.py`` imports this module (for
    ``_USER_PIPELINE_BASE``), so a module-level import back would be circular.
    Reading through the module object keeps the value live at every call --
    whatever the setting holds when a kernel containing the gate is compiled,
    never a value frozen at import time.
    """
    from algan.rendering.raytracing import settings as rt_settings

    return bool(rt_settings.LINEAR_COLOR_SPACE)


@ti.func
def _energy_scale(weight):
    """Reciprocal of the illumination budget, for energy-conserving shading.

    ``weight`` is the total illumination arriving at the surface: the ambient
    fill's coefficient plus, per light, ``(n.l) * visibility * peak(colour)``.
    Weighting by the light's own colour is what stops a rig of dim lights being
    penalised for its light *count*: three lights at 0.5 spend the same budget
    as one at 1.5, not three times as much.
    A reflective surface cannot send out more light than arrives, so once the
    incident weight passes unity the reflected terms are scaled back by it --
    the surface then reflects its albedo and no more, however many lights are
    on it.

    Below unity this is exactly 1.0, so a scene lit the way Algan lights one by
    default is bit-identical; only over-lit surfaces move. Emissive is *not*
    scaled by it: emission is not reflection, and dimming a glowing surface
    because a lamp was added would be wrong.

    Note this makes lighting normalised rather than physically additive. Two
    lamps on a white wall really do deliver twice the radiance, and the engine
    used to render that -- relying on the tonemap to compress it back. Now that
    output is display-referred (tonemapping defaults off) there is nowhere for
    the second lamp's extra radiance to go, so the budget is normalised instead
    of the result being clipped. See TONEMAP_FINDINGS.md.

    Off under the linear working colour space (:func:`_linear_color_space`):
    there lights sum plainly and this returns exactly 1.0, since normalising
    would make them stop adding. The gate is compile-time (``ti.static``), so
    the off arm is not compiled into the kernel at all.
    """
    scale = 1.0
    if ti.static(bool(not _linear_color_space())):
        scale = 1.0 / ti.max(weight, 1.0)
    return scale


# Maximum number of lights that can cast deterministic ray-traced shadows.
# Each shaded fragment collects one visibility scalar per light (1 = lit,
# 0 = occluded) into a fixed-size ``ti.Vector`` -- Taichi vector lengths are
# compile-time, so this is a compile-time cap, not a runtime one. Lights past
# the cap are still *lit*, just never shadowed. The visibility vector is
# dead-code-eliminated when shadows are off (the default), so a larger cap
# only costs registers on opt-in shadow renders. 16 covers a key/fill/rim rig
# plus a 4x4-sample area light out of the box; raise ALGAN_MAX_SHADOW_LIGHTS
# for denser area-light penumbras or larger rigs (more registers, lower
# occupancy on the shadow kernels). Each area-light emitter sample counts as
# one slot; samples past the cap light without shadowing, so an under-capped
# area light just gets a shallower umbra, never a wrong one (the default
# shader's base fade-out is power-fraction weighted -- see _stage_default). A
# truly unbounded (runtime) count would need the per-fragment visibilities in
# a global scratch buffer instead of a stack vector.
MAX_SHADOW_LIGHTS = max(1, env_int("ALGAN_MAX_SHADOW_LIGHTS", 16))


@ti.func
def _ggx_distribution(n_dot_h, roughness):
    """GGX / Trowbridge-Reitz normal distribution function."""
    a = ti.max(roughness * roughness, 1e-4)
    a2 = a * a
    d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0
    return a2 / ti.max(3.14159265 * d * d, 1e-7)


@ti.func
def _smith_geometry(n_dot_v, n_dot_l, roughness):
    """Smith geometry term with Schlick-GGX, direct-lighting k remapping."""
    r = roughness + 1.0
    k = (r * r) / 8.0
    gv = n_dot_v / ti.max(n_dot_v * (1.0 - k) + k, 1e-6)
    gl = n_dot_l / ti.max(n_dot_l * (1.0 - k) + k, 1e-6)
    return gv * gl


@ti.func
def _d_charlie(n_dot_h, sheen_roughness):
    """Charlie sheen distribution (Estevez & Kulla 2017; KHR_materials_sheen),
    as in Three.js ``D_Charlie``: exponentiated-sine micro-cylinder fibres.
    """
    alpha = ti.max(sheen_roughness, 1e-4)
    inv_alpha = 1.0 / (alpha * alpha)
    # Three.js floors sin2h at 2^-7 rather than at zero, so that sin2h^2 stays
    # representable in fp16 -- and, here, so that pow(0, large) can never turn
    # a sheenless material's zero-weighted lobe into a NaN.
    sin2h = ti.max(1.0 - n_dot_h * n_dot_h, 0.0078125)
    return ((2.0 + inv_alpha) * ti.pow(sin2h, inv_alpha * 0.5)
            / (2.0 * 3.14159265))


@ti.func
def _v_neubelt(n_dot_v, n_dot_l):
    """Ashikhmin-Preoze / Neubelt sheen visibility, as in Three.js
    ``V_Neubelt``: the cheap stand-in for the Charlie visibility term the
    KHR spec offers; clamped to [0, 1] like its ``saturate``.
    """
    return ti.min(
        1.0 / (4.0 * (n_dot_l + n_dot_v - n_dot_l * n_dot_v)), 1.0)


@ti.func
def _ibl_sheen_brdf(cos_theta, sheen_roughness):
    """Three.js ``IBLSheenBRDF``: a curve fit to the Charlie sheen BRDF
    integrated over the hemisphere, standing in for the spec's E(x) table in
    the base-layer albedo scaling. The exponent is always negative, so this
    returns a value in [0, 1].
    """
    r2 = sheen_roughness * sheen_roughness
    r_inv = 1.0 / (sheen_roughness + 0.1)
    a = -1.9362 + 1.0678 * sheen_roughness + 0.4573 * r2 - 0.8469 * r_inv
    b = -0.6014 + 0.5538 * sheen_roughness - 0.4670 * r2 - 0.1255 * r_inv
    dg = ti.exp(a * cos_theta + b)
    return ti.min(ti.max(dg, 0.0), 1.0)


@ti.func
def _shading_normal(n_interp, face_n, flat):
    """Per-fragment shading normal, optionally blended toward the (geometric)
    face normal for flat shading -- the in-kernel analogue of
    ``material_shaders._shading_normal``.
    """
    n = n_interp.normalized()
    if flat > 1e-4:
        fn = face_n.normalized()
        # Align the face normal with the interpolated normal so flat shading
        # doesn't flip lighting on a smoothed mesh.
        if fn.dot(n) < 0.0:
            fn = -fn
        n = (n * (1.0 - flat) + fn * flat).normalized()
    return n


@ti.func
def _faces_viewer(n, face_n, view_dir):
    """Whether the surface at a hit faces ``view_dir``, read off the GEOMETRIC
    normal rather than the shading normal ``n``.

    The shading normal is a smooth field interpolated INDEPENDENTLY of the
    surface it decorates -- a PN patch carries a quadratic normal field over a
    quadratic position patch, a smooth-shaded mesh carries per-vertex normals
    over flat facets -- so along a silhouette, where the surface turns away from
    the camera, ``n . view_dir`` crosses zero a little before or after the
    geometry does, and at a different sub-pixel place on every patch. Testing it
    made the scattered rim fragments that landed the wrong side of that crossing
    read as backfaces, and the callers' 180-degree flip then turned ``n . light``
    from negative to positive: the UNLIT limb of a smooth surface came out lit
    in single pixels -- bright speckle beading along the dark edge of a sphere,
    crawling between frames as adaptive dicing re-cut the patches.

    The geometric normal has no such freedom: for a ray that hits the front of a
    surface it faces the viewer by construction, and it is constant across a
    facet, so the test fires only where the surface genuinely faces away and
    never partway along a lit-to-unlit gradient.

    Its own sign carries no orientation intent (it comes from the winding or the
    parameterisation), so it is aligned to the shading normal first: the question
    is whether the surface AS THE VERTEX NORMALS ORIENT IT points away. That is
    what makes the inward-frame case below still flip.

    The alignment is also the credibility test. A fold in the parameterisation
    (a Surface pole, where the two tangents collapse onto each other) leaves a
    cross product whose direction is rounding noise, and a coin-flip side test
    would be worse than the one this replaces -- so a geometric normal that is
    nowhere near its own vertex normals, or has no length at all, is discarded
    and the old shading-normal test stands. Any real facet agrees with its
    vertex normals to far inside this margin.
    """
    side = n.dot(view_dir)
    fl = face_n.norm()
    if fl > 1e-12:
        fn = face_n * (1.0 / fl)
        d = fn.dot(n)
        if ti.abs(d) > 0.1:
            side = fn.dot(view_dir)
            if d < 0.0:
                side = -side
    return side >= 0.0


@ti.func
def _two_sided_normal(n_interp, face_n, flat, view_dir):
    """``n_interp`` flipped toward the viewer -- the historical behaviour, kept
    for geometry that has not declared an outside.

    The side is tested on the flat-BLENDED normal, which is the vector
    :func:`_prep_normal` used to hand :func:`_faces_viewer`, and the RAW normal
    is what gets negated: the stage blends again downstream, and the blend is
    odd (``_shading_normal(-n) == -_shading_normal(n)``, since it aligns the
    face normal to whichever side ``n`` is on), so negating before or after it
    gives the same vector. That makes this decision the old one by
    construction. Measured, the distinction does not reach a single pixel on
    the harness scenes -- ``_faces_viewer`` reads the geometric normal wherever
    it is credible, and the blend only moves that credibility test's own dot
    product -- so this is parity by reasoning, not a fix for anything
    observed."""
    n = n_interp
    if not _faces_viewer(_shading_normal(n_interp, face_n, flat), face_n,
                         view_dir):
        n = -n_interp
    return n


@ti.func
def _sided_shading_normal(n_interp, face_n, view_dir, params: ti.template(),
                          f, prim):
    """``n_interp``, flipped toward the viewer for TWO-SIDED geometry only.

    The shading side is a property of the surface, not of the material, so it
    is decided ONCE per hit here -- in :func:`_run_frag_pipeline`, before any
    stage runs -- rather than inside each stage's :func:`_prep_normal`. Every
    ray type therefore gets the same answer: the camera ray, the reflection
    that sees the same solid in a mirror, and the coverage pass-through behind
    a half-transparent surface.

    Two-sided (``one_sided`` 0, the default) is for geometry with no outside:
    a 2-D shape, ``Text``, a parametric ``Surface``, an imported mesh whose
    winding nobody has checked. A coarsely tessellated mesh whose perpendicular
    frame was built from an arbitrary axis can leave some patch normals
    pointing inward, and without this flip such a surface shades as an unlit
    backface (black), its apparent lighting depending on an incidental frame
    orientation rather than on its shape.

    One-sided (``one_sided`` 1) is what the built-in solids declare, having
    normals that face out (``Mob.two_sided``,
    ``tests/unit_tests/test_normal_orientation.py``). A back-facing hit on one
    of those is genuinely its inside, and lighting it as though it faced the
    camera is what turned a half-transparent solid's far shell into a second,
    brightly lit front shell -- the bright and dark planes through a fading
    Octahedron, whose far faces were lit by a key light they face away from.

    Which side the viewer is on comes from :func:`_faces_viewer` -- see there
    for why it must not be read off the shading normal.

    KNOWN LIMIT: the shadow path still orients toward the ray
    (:func:`_orient_hit_normals`), and its light-facing cull skips the shadow
    ray for a hit whose ray-facing normal points away from the light. On a
    one-sided surface shaded from BEHIND -- the inside of a solid, reachable
    only through transparency or refraction -- that is exactly the case the
    shading now lights, so such a point is lit without being shadow-tested.
    It needs a light on the far side of a translucent solid AND shadows on to
    show at all; closing it means carrying this declaration into the shadow
    event build, which is a different kernel and a wider change than the one
    this fixes.
    """
    tm = f % params.shape[0]
    n = n_interp
    if params[tm, prim, _MAT_ONE_SIDED] < 0.5:
        # Slot 10 is flat_shading, and a built-in single-stage pipeline's block
        # starts at 0 -- so this is the very value the stage is about to blend
        # with (see _two_sided_normal on why the test takes it).
        n = _two_sided_normal(n_interp, face_n, params[tm, prim, 10], view_dir)
    return n


@ti.func
def _prep_normal(n_interp, face_n, flat, view_dir):
    """Shading normal, optionally blended toward the flat (per-face) one.

    The side it is lit from is NOT decided here: the caller
    (:func:`_run_frag_pipeline`) has already oriented ``n_interp`` per the
    surface's own declaration (:func:`_sided_shading_normal`), so a stage --
    built-in or user-written -- shades whatever side it is handed. ``view_dir``
    stays in the signature for stages that were written against it and for the
    symmetry with ``material_shaders._shading_normal``.
    """
    return _shading_normal(n_interp, face_n, flat)


@ti.func
def _orient_hit_normals(snrm, fnrm, rd):
    """Shading + geometric normals of a hit, oriented for a shadow-ray origin:
    returned normalized, sharing a hemisphere, and facing back along ``rd``
    toward where the ray came from.

    The geometric normal is put in the shading normal's hemisphere so a shadow
    ray fired near the terminator does not graze the adjacent uphill facet and
    report a spurious self-shadow. Which hemisphere that is comes from
    :func:`_faces_viewer`, the same decision :func:`_prep_normal` makes, and for
    the same reason -- with the extra bite that these two must AGREE. Deciding
    the side here off the shading normal flipped exactly the silhouette
    fragments described there, which pushed the shadow-ray origin offset INTO
    the surface and inverted the facing-the-light cull: the shadow ray was
    skipped for a fragment the material stage then went on to light, so a
    silhouette pixel rendered unshadowed.
    """
    if snrm.norm() > 1e-9:
        snrm = snrm.normalized()
    if fnrm.norm() > 1e-9:
        fnrm = fnrm.normalized()
    if fnrm.dot(snrm) < 0.0:
        fnrm = -fnrm
    if not _faces_viewer(snrm, fnrm, -rd):
        snrm = -snrm
        fnrm = -fnrm
    return snrm, fnrm


@ti.func
def _shadow_terminator_delta(f, prim, w0, a, b, p, snrm,
                             tri_pos: ti.template(),
                             tri_norm: ti.template()):
    """Hanika's shadow-terminator displacement (Ray Tracing Gems II ch. 4):
    how far the smooth surface implied by the hit triangle's three VERTEX
    normals rises above the flat facet at its hit point ``p``, returned as
    the vector to displace a shadow-ray origin BY::

        d_i   = min(0, (p - p_i) . n_i)          for i in 0,1,2   (n_i unit)
        delta = -(w0 * d_0 * n_0 + a * d_1 * n_1 + b * d_2 * n_2)

    with ``(w0, a, b)`` the hit's barycentrics against ``(p_0, p_1, p_2)``
    and ``n_i`` the per-vertex normals out of ``tri_norm``. A diced PN patch
    or smooth-shaded mesh reaches the renderer as FLAT triangles under a
    quadratic normal field, so the facet is a chord below the surface it
    approximates and today's face-normal origin lift leaves neighbouring
    facets above it -- near the terminator the shadow ray then grazes away
    and strikes one of them far from the origin: acne no acceptance epsilon
    can reject (RENDERER_WORK_QUEUE.md item 20). This is what moves the
    origin onto the smooth surface instead.

    A FLAT facet returns the zero vector BY CONSTRUCTION: after normalizing,
    if the three vertex normals agree -- ``n0 . n1 > 1 - 1e-6`` and
    ``n0 . n2 > 1 - 1e-6`` -- the normal field IS constant, such a facet has
    no smooth surface to be displaced onto (that agreement is the definition
    of the flat case here, not a tolerance shortcut), and the formula is not
    evaluated at all. The equality test is what makes flat-shaded geometry
    keep the caller's origin bit for bit; trusting the arithmetic would not,
    since ``d_i`` below is evaluated in float and could leave ulp-scale dust
    on a constant field, which would set ``lifted`` in the callers and relax
    the horizon cull on geometry that never moved. Otherwise the result is
    bounded by the facet, so it needs no clamp and no epsilon.
    Each vertex normal is normalized here, and a DEGENERATE one (norm <
    1e-9) returns the zero vector: an unreadable normal field must not move
    the origin. So does a ``prim`` past a trimmed ``tri_norm``: on the
    classic wavefront path that array may be the compacted needs-normal
    prefix (:func:`_flat_triangle_normal_trim`), whose second dimension is
    shorter than ``tri_pos``'s, and a prim past it never consumes its
    shading normal.

    SIGN RULE: ``snrm`` is the hit's ORIENTED shading normal
    (:func:`_orient_hit_normals` may negate both normals so they face back
    along the incoming ray), while the ``n_i`` read here are in the raw mesh
    orientation. Negating the finished ``delta`` is NOT equivalent to
    negating the normals -- the ``min(0, .)`` clamp is not odd -- so the
    vertex normals are negated BEFORE the formula, by ONE sign shared by all
    three: ``sgn = +1 if snrm . (w0*n_0 + a*n_1 + b*n_2) >= 0 else -1``,
    tested on the raw interpolated vertex normal (never on a normal-mapped
    ``_tri_normal_g`` result).

    The caller STILL adds the face-normal lift on top of this displacement
    (``sorigin = spos + delta + fnrm * (10 * MIN_HIT_DISTANCE)``); the lift
    is what keeps flat facets working exactly as they always have.
    """
    tn = f % tri_norm.shape[0]
    tp = f % tri_pos.shape[0]
    delta = ti.math.vec3(0.0, 0.0, 0.0)
    # ``tri_norm`` may be the compacted needs-normal prefix on the classic
    # wavefront path (_flat_triangle_normal_trim), so ``prim`` can index past
    # its second dimension while indexing tri_pos fine. Such a prim never
    # consumes its shading normal -- that trim guards its own read for exactly
    # that reason -- so return zero rather than read out of bounds. On an
    # untrimmed ``tri_norm`` shape[1] is the full triangle count and this
    # never fires: one guard here covers every caller.
    if prim < tri_norm.shape[1]:
        n0 = ti.math.vec3(tri_norm[tn, prim, 0], tri_norm[tn, prim, 1],
                          tri_norm[tn, prim, 2])
        n1 = ti.math.vec3(tri_norm[tn, prim, 3], tri_norm[tn, prim, 4],
                          tri_norm[tn, prim, 5])
        n2 = ti.math.vec3(tri_norm[tn, prim, 6], tri_norm[tn, prim, 7],
                          tri_norm[tn, prim, 8])
        if (n0.norm() > 1e-9) and (n1.norm() > 1e-9) and (n2.norm() > 1e-9):
            n0 = n0.normalized()
            n1 = n1.normalized()
            n2 = n2.normalized()
            # Agreement of all three vertex normals is the DEFINITION of a
            # constant normal field -- a facet with one has no smooth surface
            # to be displaced onto. Testing it directly (instead of trusting
            # d_i below) is what keeps such a facet's delta EXACTLY zero:
            # float evaluation of d_i could otherwise leave ulp-scale dust,
            # which would set lifted = 1 in the callers and relax the horizon
            # cull on flat geometry.
            if (n0.dot(n1) <= 1.0 - 1e-6) or (n0.dot(n2) <= 1.0 - 1e-6):
                raw = w0 * n0 + a * n1 + b * n2
                sgn = 1.0
                if snrm.dot(raw) < 0.0:
                    sgn = -1.0
                n0 = n0 * sgn
                n1 = n1 * sgn
                n2 = n2 * sgn
                v0 = ti.math.vec3(tri_pos[tp, prim, 0], tri_pos[tp, prim, 1],
                                  tri_pos[tp, prim, 2])
                v1 = ti.math.vec3(tri_pos[tp, prim, 3], tri_pos[tp, prim, 4],
                                  tri_pos[tp, prim, 5])
                v2 = ti.math.vec3(tri_pos[tp, prim, 6], tri_pos[tp, prim, 7],
                                  tri_pos[tp, prim, 8])
                d0 = ti.min(0.0, (p - v0).dot(n0))
                d1 = ti.min(0.0, (p - v1).dot(n1))
                d2 = ti.min(0.0, (p - v2).dot(n2))
                delta = -(w0 * d0 * n0 + a * d1 * n1 + b * d2 * n2)
    return delta


@ti.func
def _reflect_frame(rd, snrm, fnrm):
    """Mirror direction for a ray ``rd`` reflecting off a hit, plus the outward
    normal its new origin must be offset along. Returns ``(dir, offset_n)``.

    Reflecting about the SHADING normal is what makes a smooth-shaded mesh
    mirror like the surface it approximates rather than like its facets, so
    that is the normal used. But it is interpolated independently of the
    geometry it decorates (see :func:`_faces_viewer`), and near a silhouette it
    tips past that geometry: ``snrm . rd`` turns positive on fragments the ray
    genuinely hit from the front. Reading the side off it there -- ``if
    snrm.dot(rd) > 0: snrm = -snrm``, which every one of these call sites used
    to do -- inverts the whole frame. The mirror ray is then launched INTO the
    solid, from an origin offset inside it, and immediately hits the object's
    own far side; because Fresnel at grazing incidence weights that hit at
    nearly 1, the far side is composited over the silhouette at close to full
    strength. On a coarsely tessellated sphere that is a dashed fringe of the
    opposite side's colour beading along the rim, brighter than both the
    surface and the background, and it moves as the tessellation turns.

    So the side comes from the geometric normal, via :func:`_orient_hit_normals`
    -- the same decision the material stages and the shadow rays already share,
    which is what keeps a fragment from reflecting as one side while it lights
    as the other.

    That alone does not make the reflection leave the surface: a shading normal
    tipped past the silhouette mirrors ``rd`` to a direction below the facet's
    own horizon even when the side is right. Such a direction is not a
    reflection of anything -- it points into the solid -- so the facet's own
    mirror direction is used instead, which provably leaves it. At a silhouette
    the facet is nearly edge-on, so that direction grazes away along the
    surface, which is what the smooth surface being approximated does there.
    The origin offset follows the geometric normal for the same reason.

    A hit with no usable geometric normal (a degenerate facet, a fold in a
    parameterisation) keeps the shading normal for both, i.e. the old
    behaviour: there is nothing better to appeal to.

    The origin offset only switches to the geometric normal where the direction
    did. Away from a silhouette the two normals agree on which side they are
    and the mirror direction clears the facet anyway, so offsetting along the
    shading normal is both harmless and what every reflective hit already did:
    keeping it there leaves ordinary reflective surfaces bit-for-bit unchanged
    and confines this fix to the fragments that were actually wrong.
    """
    sn, fn = _orient_hit_normals(snrm, fnrm, rd)
    out = (rd - 2.0 * rd.dot(sn) * sn).normalized()
    offset_n = sn
    if fnrm.norm() > 1e-9:
        if out.dot(fn) <= 0.0:
            out = (rd - 2.0 * rd.dot(fn) * fn).normalized()
            offset_n = fn
    return out, offset_n


@ti.func
def _light(light_pos: ti.template(), light_col: ti.template(), f, li):
    """Point-light world position and RGB colour for light ``li`` at frame ``f``."""
    tl = f % light_pos.shape[0]
    lp = ti.math.vec3(light_pos[tl, li, 0], light_pos[tl, li, 1],
                      light_pos[tl, li, 2])
    lc = ti.math.vec3(light_col[tl, li, 0], light_col[tl, li, 1],
                      light_col[tl, li, 2])
    return lp, lc


# Light type ids (column 3 of an extended packed light row; mirrors
# algan.rendering.lights.LIGHT_*).
_LT_POINT = 0
_LT_DIRECTIONAL = 1
_LT_AMBIENT = 2
_LT_HEMISPHERE = 3
_LT_SPOT = 4
_LT_AREA_SAMPLE = 5
_LT_ENV_SH = 6


@ti.func
def _light_eval(light_pos: ti.template(), light_col: ti.template(),
                f, li, pos, n):
    """Evaluate light ``li`` for a surface point ``pos`` with shading normal
    ``n``: returns ``(ld, lc, spec_w, frac)`` -- the unit direction toward the
    light, its effective RGB radiance (falloff / cone / hemisphere blending
    applied), the specular gate (0 for the direction-less ambient-like types)
    and the light's *power fraction* (packed column 15): the share of a whole
    light this row represents -- ``1/K`` for one of an area light's K emitter
    samples, 1 for every stand-alone light. Physical stages ignore it (their
    per-sample radiance already carries the 1/K); the legacy lerp-based default
    stage weights its blend total by it so an area light lerps like *one*
    light of its full colour rather than K dim ones.

    Compact rows (``light_col`` width 3, the legacy packing used whenever the
    scene has only plain point lights) take the original point-light path with
    identical arithmetic. Extended rows (width 16) carry a type id + parameters
    (packed by ``scene_builder._pack_lights``; layout documented on
    :meth:`algan.rendering.lights.Light.build_aux`).

    Ambient-like types (ambient / hemisphere / env-SH) return ``ld = n`` so the
    material stages' ``n . ld`` diffuse factor becomes 1 -- they reuse the
    stages' diffuse term unchanged, with the specular term gated off.
    """
    tl = f % light_pos.shape[0]
    lp = ti.math.vec3(light_pos[tl, li, 0], light_pos[tl, li, 1],
                      light_pos[tl, li, 2])
    lc = ti.math.vec3(light_col[tl, li, 0], light_col[tl, li, 1],
                      light_col[tl, li, 2])
    ld = (lp - pos).normalized()
    spec_w = 1.0
    frac = 1.0
    if light_col.shape[2] > 3:
        ltype = ti.cast(light_col[tl, li, 3] + 0.5, ti.i32)
        # Power fraction; <= 0 means "unset" (e.g. rows packed before this
        # column existed, or the env-SH row) and defaults to a whole light.
        frac = light_col[tl, li, 15]
        if frac <= 0.0:
            frac = 1.0
        if ltype == _LT_DIRECTIONAL:
            ld = -ti.math.vec3(light_col[tl, li, 6], light_col[tl, li, 7],
                               light_col[tl, li, 8])
        elif ltype == _LT_AMBIENT:
            ld = n
            spec_w = 0.0
        elif ltype == _LT_HEMISPHERE:
            up = ti.math.vec3(light_col[tl, li, 6], light_col[tl, li, 7],
                              light_col[tl, li, 8])
            ground = ti.math.vec3(light_col[tl, li, 12],
                                  light_col[tl, li, 13],
                                  light_col[tl, li, 14])
            h = 0.5 + 0.5 * n.dot(up)
            lc = ground * (1.0 - h) + lc * h
            ld = n
            spec_w = 0.0
        elif ltype == _LT_ENV_SH:
            # Order-1 spherical-harmonics irradiance of the environment map,
            # as a linear form A + B . n (coefficients packed host-side).
            bx = ti.math.vec3(light_col[tl, li, 6], light_col[tl, li, 7],
                              light_col[tl, li, 8])
            by = ti.math.vec3(light_col[tl, li, 9], light_col[tl, li, 10],
                              light_col[tl, li, 11])
            bz = ti.math.vec3(light_col[tl, li, 12], light_col[tl, li, 13],
                              light_col[tl, li, 14])
            lc = ti.max(lc + bx * n[0] + by * n[1] + bz * n[2],
                        ti.math.vec3(0.0, 0.0, 0.0))
            ld = n
            spec_w = 0.0
        if (ltype == _LT_POINT) or (ltype == _LT_SPOT) \
                or (ltype == _LT_AREA_SAMPLE):
            d = (lp - pos).norm()
            decay = light_col[tl, li, 4]
            if decay > 0.0:
                lc = lc / ti.pow(ti.max(d, 1e-4), decay)
            rng = light_col[tl, li, 5]
            if rng > 0.0:
                q = ti.math.clamp(d / rng, 0.0, 1.0)
                q2 = q * q
                fade = ti.math.clamp(1.0 - q2 * q2, 0.0, 1.0)
                lc = lc * (fade * fade)
        if ltype == _LT_SPOT:
            sd = ti.math.vec3(light_col[tl, li, 6], light_col[tl, li, 7],
                              light_col[tl, li, 8])
            cos_outer = light_col[tl, li, 9]
            cos_inner = light_col[tl, li, 10]
            c = (-ld).dot(sd)
            t = ti.math.clamp((c - cos_outer)
                              / ti.max(cos_inner - cos_outer, 1e-6), 0.0, 1.0)
            lc = lc * (t * t * (3.0 - 2.0 * t))
        elif ltype == _LT_AREA_SAMPLE:
            # One-sided cosine emission of the rectangle sample.
            an = ti.math.vec3(light_col[tl, li, 6], light_col[tl, li, 7],
                              light_col[tl, li, 8])
            lc = lc * ti.max((-ld).dot(an), 0.0)
    return ld, lc, spec_w, frac


@ti.func
def _light_vis(shadows: ti.template(), vis, li):
    """Per-light shadow visibility (1 lit / 0 occluded). Compiled out entirely
    when shadows are off, and falls back to fully lit beyond the shadow-ray cap.
    """
    v = 1.0
    if ti.static(shadows != 0):
        if li < MAX_SHADOW_LIGHTS:
            v = vis[li]
    return v


# ---------------------------------------------------------------------------
# Built-in core lit material stages.
#
# Stage contract (a ``@ti.func``): evaluate one shading pass for a surface hit
# and return the new RGB + glow as a ``vec4``. ``in_rgb`` is the running colour
# (the previous stage's output, or the interpolated raw albedo for the first
# stage); ``in_glow`` is the passthrough 4th channel; ``view_dir`` is the unit
# direction from the surface back toward the viewer. ``params`` is the
# per-primitive parameter ndarray and ``off`` this stage's base slot offset.
# When ``shadows`` is enabled, ``vis`` carries one visibility scalar per light;
# only the direct diffuse/specular response is gated by it (ambient/emissive
# stay lit). Stages loop the lights internally, exactly as the single-light
# vertex path overwrites the colour per light.
# ---------------------------------------------------------------------------

@ti.func
def _stage_unlit(pos, view_dir, n_interp, face_n, in_rgb, in_glow,
                 params: ti.template(), f, prim, off,
                 light_pos: ti.template(), light_col: ti.template(),
                 num_lights, shadows: ti.template(), vis):
    """MeshBasicMaterial / passthrough: returns the colour unchanged."""
    return ti.math.vec4(in_rgb[0], in_rgb[1], in_rgb[2], in_glow)


@ti.func
def _stage_default(pos, view_dir, n_interp, face_n, in_rgb, in_glow,
                   params: ti.template(), f, prim, off,
                   light_pos: ti.template(), light_col: ti.template(),
                   num_lights, shadows: ti.template(), vis):
    """default_shader: diffuse lerp of the colour toward each light colour.

    Additive over lights: gather every light's lerp weight, then blend once.
    For a single light this equals the legacy per-light lerp
    (``out*(1-w) + lc*w``); for many lights it stays stable (an area light's
    sample fan, or a key/fill/rim rig) instead of the old sequential lerp
    driving the colour toward the last light's.
    """
    flat = params[f % params.shape[0], prim, off + 10]
    n = _prep_normal(n_interp, face_n, flat, view_dir)
    out = in_rgb
    acc = ti.math.vec3(0.0, 0.0, 0.0)
    wsum = 0.0
    esum = 0.0
    tl0 = f % light_col.shape[0]
    for li in range(num_lights):
        # A zero-colour light row is a light outside its lifespan (or
        # genuinely black) and must not fade the base colour: it either was
        # filtered out of the batch's light list entirely, or belongs to a
        # batch straddling its spawn -- and the output must not depend on
        # which of those happened. Gated on the RAW row colour (not the
        # evaluated ``lc``) so live-light modifiers (spot cones, decay)
        # keep their existing fade behaviour. Hemisphere / environment-SH
        # rows radiate from their aux columns even with zero RGB (a black-sky
        # hemisphere still has a ground colour), so they are always kept --
        # their out-of-lifespan rows are inert anyway (the aux radiance
        # columns scale with opacity at materialization).
        row_live = 0
        if ((light_col[tl0, li, 0] != 0.0)
                or (light_col[tl0, li, 1] != 0.0)
                or (light_col[tl0, li, 2] != 0.0)):
            row_live = 1
        elif light_col.shape[2] > 3:
            lt0 = ti.cast(light_col[tl0, li, 3] + 0.5, ti.i32)
            if (lt0 == _LT_HEMISPHERE) or (lt0 == _LT_ENV_SH):
                row_live = 1
        if row_live == 1:
            ld, lc, _spec_w, frac = _light_eval(light_pos, light_col, f, li,
                                                pos, n)
            v = _light_vis(shadows, vis, li)
            d = ti.max(ld.dot(n), 0.0)
            w = d * d * d * d * d * 0.5 * v
            acc += lc * w
            esum += w * ti.max(lc[0], ti.max(lc[1], lc[2]))
            # The base fade-out counts each row by its *power fraction* (1/K
            # for an area light's K emitter samples, 1 otherwise), so one area
            # light displaces at most as much base colour as one point light
            # would -- while ``acc`` (whose per-sample radiance already
            # carries the 1/K) sums back to the full light colour. Without
            # this, a fully-occluded umbra under a many-sample area light
            # would revert toward the raw albedo (which can be *brighter*
            # than the dimly-lit surroundings, a "bright shadow"), because
            # the K dim samples each faded the base as a whole light while
            # only delivering 1/K of the colour.
            wsum += w * frac
    # The base fade was already bounded by ``min(wsum, 1)``; ``acc`` was not,
    # so past a total weight of 1 the two stopped being a blend and the sum ran
    # away with the extra lights. Scaling ``acc`` by the same budget makes the
    # pair a genuine convex combination of the albedo and the light colours.
    # ``max(wsum, 1)`` is exactly 1 for a single light (whose weight peaks at
    # 0.5), so the default rig is bit-identical.
    out = out * (1.0 - ti.min(wsum, 1.0)) + acc * _energy_scale(esum)
    return ti.math.vec4(out[0], out[1], out[2], in_glow)


@ti.func
def _stage_lambert(pos, view_dir, n_interp, face_n, in_rgb, in_glow,
                   params: ti.template(), f, prim, off,
                   light_pos: ti.template(), light_col: ti.template(),
                   num_lights, shadows: ti.template(), vis):
    """MeshLambertMaterial: Lambertian (diffuse-only) lighting plus emissive."""
    tm = f % params.shape[0]
    emissive = ti.math.vec3(params[tm, prim, off + 0], params[tm, prim, off + 1],
                            params[tm, prim, off + 2])
    emissive_intensity = params[tm, prim, off + 3]
    flat = params[tm, prim, off + 10]
    env = params[tm, prim, off + 11]
    n = _prep_normal(n_interp, face_n, flat, view_dir)
    # Additive multi-light accumulation over a fixed albedo: ambient + emissive
    # once, then each light's direct diffuse. For a single light this equals
    # the legacy expression; for many lights it sums correctly (the old
    # per-light overwrite collapsed the colour and re-added ambient/emissive
    # per light -- e.g. an area light's sample fan came out wrong).
    amb = ti.static(_ambient_strength())
    refl = in_rgb * (amb * env)
    wsum = amb * env
    for li in range(num_lights):
        ld, lc, _spec_w, _frac = _light_eval(light_pos, light_col, f, li,
                                             pos, n)
        v = _light_vis(shadows, vis, li)
        n_dot_l = ti.max(n.dot(ld), 0.0)
        refl += in_rgb * lc * (n_dot_l * v)
        wsum += n_dot_l * v * ti.max(lc[0], ti.max(lc[1], lc[2]))
    acc = refl * _energy_scale(wsum) + emissive * emissive_intensity
    return ti.math.vec4(acc[0], acc[1], acc[2], in_glow)


@ti.func
def _stage_phong(pos, view_dir, n_interp, face_n, in_rgb, in_glow,
                 params: ti.template(), f, prim, off,
                 light_pos: ti.template(), light_col: ti.template(),
                 num_lights, shadows: ti.template(), vis):
    """MeshPhongMaterial: Blinn-Phong diffuse + specular highlight + emissive."""
    tm = f % params.shape[0]
    emissive = ti.math.vec3(params[tm, prim, off + 0], params[tm, prim, off + 1],
                            params[tm, prim, off + 2])
    emissive_intensity = params[tm, prim, off + 3]
    specular = ti.math.vec3(params[tm, prim, off + 4], params[tm, prim, off + 5],
                            params[tm, prim, off + 6])
    shininess = params[tm, prim, off + 7]
    flat = params[tm, prim, off + 10]
    env = params[tm, prim, off + 11]
    n = _prep_normal(n_interp, face_n, flat, view_dir)
    # Additive over lights (see _stage_lambert): ambient + emissive once, then
    # each light's Blinn-Phong diffuse + specular.
    amb = ti.static(_ambient_strength())
    refl = in_rgb * (amb * env)
    wsum = amb * env
    for li in range(num_lights):
        ld, lc, spec_w, _frac = _light_eval(light_pos, light_col, f, li,
                                            pos, n)
        v = _light_vis(shadows, vis, li)
        half = (ld + view_dir).normalized()
        n_dot_l = ti.max(n.dot(ld), 0.0)
        n_dot_h = ti.max(n.dot(half), 0.0)
        spec_term = ti.pow(ti.max(n_dot_h, 1e-4), ti.max(shininess, 1e-3))
        gate = spec_w if n_dot_l > 0.0 else 0.0
        refl += (in_rgb * lc * n_dot_l
                 + specular * lc * spec_term * gate) * v
        wsum += n_dot_l * v * ti.max(lc[0], ti.max(lc[1], lc[2]))
    acc = refl * _energy_scale(wsum) + emissive * emissive_intensity
    return ti.math.vec4(acc[0], acc[1], acc[2], in_glow)


@ti.func
def _stage_standard(pos, view_dir, n_interp, face_n, in_rgb, in_glow,
                    params: ti.template(), f, prim, off,
                    light_pos: ti.template(), light_col: ti.template(),
                    num_lights, shadows: ti.template(), vis):
    """MeshStandardMaterial: metalness/roughness Cook-Torrance GGX PBR + emissive."""
    tm = f % params.shape[0]
    emissive = ti.math.vec3(params[tm, prim, off + 0], params[tm, prim, off + 1],
                            params[tm, prim, off + 2])
    emissive_intensity = params[tm, prim, off + 3]
    roughness = params[tm, prim, off + 8]
    metalness = params[tm, prim, off + 9]
    flat = params[tm, prim, off + 10]
    env = params[tm, prim, off + 11]
    n = _prep_normal(n_interp, face_n, flat, view_dir)
    # Additive over lights (see _stage_lambert): the metalness/F0 ambient +
    # emissive base once, then each light's Cook-Torrance direct term.
    one = ti.math.vec3(1.0, 1.0, 1.0)
    rgb = in_rgb
    f0 = ti.math.vec3(0.04, 0.04, 0.04) * (1.0 - metalness) + rgb * metalness
    amb = ti.static(_ambient_strength())
    refl = (rgb * (1.0 - metalness) + f0 * metalness) * (amb * env)
    wsum = amb * env
    for li in range(num_lights):
        ld, lc, spec_w, _frac = _light_eval(light_pos, light_col, f, li,
                                            pos, n)
        v = _light_vis(shadows, vis, li)
        half = (ld + view_dir).normalized()
        n_dot_l = ti.max(n.dot(ld), 0.0)
        n_dot_v = ti.max(n.dot(view_dir), 1e-4)
        n_dot_h = ti.max(n.dot(half), 0.0)
        v_dot_h = ti.max(view_dir.dot(half), 0.0)
        fresnel = f0 + (one - f0) * ti.pow(ti.max(1.0 - v_dot_h, 0.0), 5.0)
        ndf = _ggx_distribution(n_dot_h, roughness)
        geom = _smith_geometry(n_dot_v, n_dot_l, roughness)
        spec = (ndf * geom) * fresnel / ti.max(4.0 * n_dot_v * n_dot_l, 1e-4)
        k_d = (one - fresnel) * (1.0 - metalness)
        diffuse = k_d * rgb * lc * n_dot_l
        refl += (diffuse + spec * lc * (n_dot_l * spec_w)) * v
        wsum += n_dot_l * v * ti.max(lc[0], ti.max(lc[1], lc[2]))
    acc = refl * _energy_scale(wsum) + emissive * emissive_intensity
    return ti.math.vec4(acc[0], acc[1], acc[2], in_glow)


@ti.func
def _stage_physical(pos, view_dir, n_interp, face_n, in_rgb, in_glow,
                    params: ti.template(), f, prim, off,
                    light_pos: ti.template(), light_col: ti.template(),
                    num_lights, shadows: ti.template(), vis):
    """MeshPhysicalMaterial: MeshStandard plus ior-driven specular, a clearcoat
    GGX lobe, a sheen rim and (crude) transmission -- the in-kernel port of
    ``material_shaders.physical_shader`` (same terms; ``iridescence`` is
    accepted but unused, as in the PyTorch shader).
    """
    tm = f % params.shape[0]
    emissive = ti.math.vec3(params[tm, prim, off + 0], params[tm, prim, off + 1],
                            params[tm, prim, off + 2])
    emissive_intensity = params[tm, prim, off + 3]
    roughness = params[tm, prim, off + 8]
    metalness = params[tm, prim, off + 9]
    flat = params[tm, prim, off + 10]
    env = params[tm, prim, off + 11]
    ior = params[tm, prim, off + 12]
    specular_intensity = params[tm, prim, off + 13]
    specular_color = ti.math.vec3(params[tm, prim, off + 14],
                                  params[tm, prim, off + 15],
                                  params[tm, prim, off + 16])
    clearcoat = params[tm, prim, off + 17]
    clearcoat_roughness = params[tm, prim, off + 18]
    sheen = params[tm, prim, off + 19]
    sheen_roughness = params[tm, prim, off + 20]
    sheen_color = ti.math.vec3(params[tm, prim, off + 21],
                               params[tm, prim, off + 22],
                               params[tm, prim, off + 23])
    transmission = params[tm, prim, off + 24]
    n = _prep_normal(n_interp, face_n, flat, view_dir)
    one = ti.math.vec3(1.0, 1.0, 1.0)
    rgb = in_rgb
    # ior drives the dielectric base reflectivity (KHR specular workflow).
    ratio = (ior - 1.0) / ti.max(ior + 1.0, 1e-4)
    f0 = (specular_color * (ratio * ratio * specular_intensity)
          * (1.0 - metalness) + rgb * metalness)
    # Additive over lights (see _stage_lambert): the metalness/F0 ambient +
    # emissive base once, then each light's direct terms.
    amb = ti.static(_ambient_strength())
    refl = (rgb * (1.0 - metalness) + f0 * metalness) * (amb * env)
    wsum = amb * env
    for li in range(num_lights):
        ld, lc, spec_w, _frac = _light_eval(light_pos, light_col, f, li,
                                            pos, n)
        v = _light_vis(shadows, vis, li)
        half = (ld + view_dir).normalized()
        n_dot_l = ti.max(n.dot(ld), 0.0)
        n_dot_v = ti.max(n.dot(view_dir), 1e-4)
        n_dot_h = ti.max(n.dot(half), 0.0)
        v_dot_h = ti.max(view_dir.dot(half), 0.0)
        fresnel = f0 + (one - f0) * ti.pow(ti.max(1.0 - v_dot_h, 0.0), 5.0)
        ndf = _ggx_distribution(n_dot_h, roughness)
        geom = _smith_geometry(n_dot_v, n_dot_l, roughness)
        spec = (ndf * geom) * fresnel / ti.max(4.0 * n_dot_v * n_dot_l, 1e-4)
        k_d = (one - fresnel) * ((1.0 - metalness) * (1.0 - transmission))
        # SHEEN, and what it takes from the layer underneath. The fibre lobe
        # is Charlie x Neubelt (KHR_materials_sheen, and Three.js's
        # ``BRDF_Sheen`` term for term), and Three.js's ``RE_Direct_Physical``
        # then scales the base layer's irradiance by
        # ``1 - max3(sheenColor) * max(E(n.v), E(n.l))`` so the fibres cannot
        # add light the base already spent. ``sheen`` premultiplies the colour
        # exactly as ``WebGLMaterials`` does. At ``sheen == 0`` the colour is
        # zero, so the compensation is exactly 1.0 and the lobe exactly 0 --
        # every material that leaves sheen alone renders bit-for-bit as before.
        sheen_c = sheen_color * sheen
        sheen_max = ti.max(sheen_c[0], ti.max(sheen_c[1], sheen_c[2]))
        sheen_r = ti.math.clamp(sheen_roughness, 1e-4, 1.0)
        sheen_comp = 1.0 - sheen_max * ti.max(
            _ibl_sheen_brdf(n_dot_v, sheen_r),
            _ibl_sheen_brdf(n_dot_l, sheen_r))
        direct = (k_d * rgb * lc * n_dot_l
                  + spec * lc * (n_dot_l * spec_w)) * sheen_comp
        # Clearcoat: a thin dielectric GGX lobe (fixed F0 = 0.04) on top.
        # Not scaled by the sheen compensation, matching Three.js, which
        # accumulates the coat before the sheen block touches the irradiance.
        cc_ndf = _ggx_distribution(n_dot_h, clearcoat_roughness)
        cc_geom = _smith_geometry(n_dot_v, n_dot_l, clearcoat_roughness)
        cc_fresnel = 0.04 + 0.96 * ti.pow(ti.max(1.0 - v_dot_h, 0.0), 5.0)
        cc_spec = clearcoat * (cc_ndf * cc_geom * cc_fresnel) \
            / ti.max(4.0 * n_dot_v * n_dot_l, 1e-4)
        direct += lc * (cc_spec * n_dot_l * spec_w)
        # The fibre lobe itself. Gated by ``spec_w`` like every other lobe
        # here: an ambient-like light arrives along the normal by convention
        # (see _light_eval) and must not manufacture a directional rim.
        sheen_brdf = _d_charlie(n_dot_h, sheen_r) * _v_neubelt(n_dot_v, n_dot_l)
        direct += sheen_c * lc * (sheen_brdf * n_dot_l * spec_w)
        wsum += n_dot_l * v * ti.max(lc[0], ti.max(lc[1], lc[2]))
        # NO per-light transmission term. There used to be one here
        # (``rgb * lc * transmission * (1 - metalness) * 0.5``) and it double
        # counted: the transmitted share already gets carried, either as the
        # refracted ray ``_scatter_impl`` splits off or -- when nothing bends,
        # because the ior is index-matched or the split pool is absent -- as
        # part of the pass-through it folds into ``pass_w``. Meanwhile this
        # stage's own output is scaled by ``alpha * (1 - R - trans_share)``
        # precisely to make room for that. The term also had no ``n.l``, so a
        # light grazing from behind the surface still lit it, and it scaled
        # with the number of lights.
        refl += direct * v
    acc = refl * _energy_scale(wsum) + emissive * emissive_intensity
    return ti.math.vec4(acc[0], acc[1], acc[2], in_glow)


def make_pipeline_func(stages, offsets):
    """Compose an ordered list of stage ``@ti.func``s into a single ``@ti.func``.

    Taichi cannot take a nested tuple as a ``ti.template()`` argument, so each
    distinct pipeline is baked into one func here (closing over its ``stages``
    and per-stage param ``offsets``); the shade kernel then receives just a flat
    tuple of these composed funcs (see ``taichi-func-injection``). Each stage's
    ``vec4`` output threads forward as the next stage's ``in_rgb``/``in_glow``.
    """
    stages = tuple(stages)
    offsets = tuple(int(o) for o in offsets)

    @ti.func
    def pipeline_fn(pos, view_dir, n_interp, face_n, in_rgb, in_glow,
                    params: ti.template(), f, prim,
                    light_pos: ti.template(), light_col: ti.template(),
                    num_lights, shadows: ti.template(), vis):
        out = in_rgb
        g = in_glow
        for si in ti.static(range(len(stages))):
            stage = ti.static(stages[si])
            off = ti.static(offsets[si])
            r = stage(pos, view_dir, n_interp, face_n, out, g,
                      params, f, prim, off,
                      light_pos, light_col, num_lights, shadows, vis)
            out = ti.math.vec3(r[0], r[1], r[2])
            g = r[3]
        return ti.math.vec4(out[0], out[1], out[2], g)

    return pipeline_fn


# ---------------------------------------------------------------------------
# Composed pipeline funcs + the scatter contract.
#
# ``builtin_pipeline_fn`` below exists for the UNSUPPORTED legacy sorted
# wavefront (see ``wavefront_sorted_kernels_taichi``; kept for reference),
# which launches one small shade kernel per material bucket with that
# material's *pipeline func* injected as a ``ti.template()`` -- so the runtime
# pid switch of ``_run_frag_pipeline`` disappears and a warp never mixes
# materials. The six built-in single-stage materials are wrapped into composed
# pipeline funcs (lazily, cached) so built-in and user pipelines share one
# injection contract.
#
# The scatter contract below is LIVE: it is how the supported monolithic
# ``wavefront_shade`` kernel dispatches custom ray bouncing too.
#
# **Scatter contract** (user-customisable ray-bouncing): a scatter is a
# ``@ti.func`` deciding how a shaded surface event continues its ray::
#
#     scatter(rd, n_interp, face_n, hit_point, shaded, albedo, alpha,
#             reflectivity, ior, transmission, params: ti.template(), f, prim,
#             bounces_left, refraction: ti.template())
#         -> (contrib, pass_w,
#             refl_orig, refl_dir, refl_w,
#             trans_orig, trans_dir, trans_w)
#
# ``rd`` is the unit ray direction, ``shaded`` the pipeline's output colour
# (vec4: RGB + glow), ``albedo`` the raw surface colour before lighting
# (vec3), ``contrib`` the premultiplied colour committed to the ray (the
# kernel adds ``weight * contrib``). The surface properties come straight
# from the material: ``alpha`` is coverage, ``transmission`` how much light the
# covered part passes, ``reflectivity`` packed metalness (negative = non-PBR)
# and ``ior`` an unsigned magnitude.
#
# With the nested-IOR media stack on (``NESTED_IOR``, DESIGN_mesh_identity_
# open.md §H), a custom scatter's transmitted ray continues in the PARENT
# medium: the renderer copies the calling ray's media stack onto the split
# branch unchanged and passes the material's own index as ``ior``, because a
# scene carrying any custom scatter owns every fragment and the fixed
# signature above cannot say what medium the returned ``trans_dir`` leaves the
# ray in -- such scenes get no nested IOR.
#
# Transport is full-colour: ray throughput is a vec3 and the branch weights
# ``pass_w`` / ``refl_w`` / ``trans_w`` are vec3 per-channel multipliers (the
# built-in scatter tints the metal Fresnel lobe and the transmitted share by
# ``albedo``; kernels reduce a weight to its maximum component for branch
# decisions and minimum-weight culls). ``pass_w`` is the throughput multiplier
# for continuing *through* the surface to the next depth layer (used only when
# ``refl_w`` is zero). A positive ``refl_w`` bounces the ray from ``refl_orig``
# along ``refl_dir`` with throughput ``weight * refl_w``; a positive
# ``trans_w`` additionally *splits* off a second branch from ``trans_orig``
# along ``trans_dir`` -- the refracted ray for glass, or the reflection when
# the pass-through is the primary (``refl_w`` zero). The built-in scatters
# (``wavefront_kernels_taichi.default_scatter`` for solid geometry,
# ``circuit_scatter`` for thin-pane circuits) derive all of this from the
# material; attach a custom one to a
# :class:`~algan.rendering.shaders.fragment_shaders.FragmentStage` via its
# ``scatter=`` argument to override how rays bounce.
# ---------------------------------------------------------------------------

_BUILTIN_STAGE_FNS = (_stage_default, _stage_unlit, _stage_lambert,
                      _stage_phong, _stage_standard, _stage_physical)
_BUILTIN_PIPELINE_FNS = {}

# "Every pipeline id may be present" -- the ungated kernel. All bits of the
# two's-complement -1 are set, so the gating below keeps every stage without
# needing a special case (see ``_run_frag_pipeline``).
ALL_PIDS = -1


def solo_pid(pids_present, num_user_pipelines):
    """The one pipeline id a batch can hit, or -1 when more than one can.

    ``pids_present`` is the host's compile-time bitmask of the material
    pipeline ids the batch's primitives carry (bit ``p`` = id ``p``);
    :data:`ALL_PIDS` means "unknown, assume all". A batch whose triangles (or
    PN patches) all share one material lets the shade kernel call that
    material's stage unconditionally -- no per-hit id fetch and no compare.
    """
    mask = int(pids_present)
    if mask < 0:
        return -1
    top = _USER_PIPELINE_BASE + int(num_user_pipelines)
    if mask >> top:
        # An id with no injected pipeline behind it: keep the runtime switch
        # (it matches nothing and passes the colour through, as before).
        return -1
    live = [pid for pid in range(top) if (mask >> pid) & 1]
    return live[0] if len(live) == 1 else -1


def builtin_pipeline_fn(pid):
    """Composed single-stage pipeline func for built-in material id ``pid``
    (0 default, 1 unlit, 2 lambert, 3 phong, 4 standard, 5 physical), for
    injection into a per-material shade kernel of the legacy sorted path
    (unsupported). Lazily created and cached so every render reuses the same
    func objects (stable Taichi template instantiations).
    """
    pid = int(pid)
    if pid not in _BUILTIN_PIPELINE_FNS:
        _BUILTIN_PIPELINE_FNS[pid] = make_pipeline_func(
            [_BUILTIN_STAGE_FNS[pid]], [0])
    return _BUILTIN_PIPELINE_FNS[pid]


@ti.func
def _run_frag_pipeline(frag_pipelines: ti.template(), pids_present: ti.template(),
                       prim, f, pos, view_dir, n_interp, face_n, albedo, glow,
                       light_pos: ti.template(), light_col: ti.template(),
                       num_lights, pid_arr: ti.template(),
                       params: ti.template(), shadows: ti.template(), vis):
    """Evaluate a surface hit's per-primitive shading pipeline.

    ``pid_arr[f, prim]`` selects the pipeline: ids 0-5 are the built-in
    single-stage materials (dispatched directly, without the composed-func
    indirection); ids >= ``_USER_PIPELINE_BASE``
    index the injected ``frag_pipelines`` tuple. ``albedo`` is the interpolated
    raw base RGB (``glow`` the passthrough 4th channel). Returns the shaded
    RGB + glow as a ``vec4``.

    ``pids_present`` (compile-time) is the host's bitmask of the ids this
    batch's primitives actually carry, and selects one of three dispatches:

    * :data:`ALL_PIDS` (the default) -- the classic chain over every built-in
      material, byte-for-byte the pre-gating kernel.
    * exactly one reachable id -- that stage called unconditionally, with no
      id fetch and no compare at all (see :func:`solo_pid`).
    * otherwise -- a compare per *reachable* id only.

    The last two exist because this dispatch is inlined into the hottest
    kernel there is, and every stage the compiler can reach costs it, whether
    or not the scene uses that material.  Gating is byte-identical by
    construction: the mask comes from the merge-time id list of the very
    array indexed here, so it can only drop branches that never fire.
    """
    out = ti.math.vec3(albedo[0], albedo[1], albedo[2])
    g = glow
    solo = ti.static(solo_pid(pids_present, len(frag_pipelines)))
    # THE SHADING SIDE, decided once per hit rather than once per stage (see
    # _sided_shading_normal). Only a BUILT-IN pipeline's parameter block has a
    # one_sided slot -- a custom fragment pipeline lays its block out itself --
    # so a custom pipeline gets the viewer-facing normal unconditionally, which
    # is what its stages were handed before the decision moved out here.
    shade_n = n_interp
    if ti.static(solo >= 0):
        if ti.static(solo < _USER_PIPELINE_BASE):
            shade_n = _sided_shading_normal(n_interp, face_n, view_dir,
                                            params, f, prim)
        else:
            shade_n = _two_sided_normal(n_interp, face_n, 0.0, view_dir)
        if ti.static(solo == _MID_UNLIT):
            pass  # passthrough: colour returned unchanged (raw or baked).
        elif ti.static(solo < _USER_PIPELINE_BASE):
            stage = ti.static(_BUILTIN_STAGE_FNS[solo])
            r = stage(pos, view_dir, shade_n, face_n, out, g,
                      params, f, prim, 0,
                      light_pos, light_col, num_lights, shadows, vis)
            out = ti.math.vec3(r[0], r[1], r[2])
            g = r[3]
        else:
            fn = ti.static(frag_pipelines[solo - _USER_PIPELINE_BASE])
            r = fn(pos, view_dir, shade_n, face_n, out, g,
                   params, f, prim,
                   light_pos, light_col, num_lights, shadows, vis)
            out = ti.math.vec3(r[0], r[1], r[2])
            g = r[3]
    elif ti.static(pids_present == ALL_PIDS):
        # UNGATED: the classic hand-written chain, kept verbatim rather than
        # folded into the loop below. The two are semantically identical, but
        # this one is the shape the default render has always compiled, and a
        # kernel this hot is not the place to change the emitted branch
        # structure for cosmetics.
        pid = pid_arr[f % pid_arr.shape[0], prim]
        if pid < _USER_PIPELINE_BASE:
            shade_n = _sided_shading_normal(n_interp, face_n, view_dir,
                                            params, f, prim)
        else:
            shade_n = _two_sided_normal(n_interp, face_n, 0.0, view_dir)
        if pid == _MID_DEFAULT:
            r = _stage_default(pos, view_dir, shade_n, face_n, out, g,
                               params, f, prim, 0,
                               light_pos, light_col, num_lights, shadows, vis)
            out = ti.math.vec3(r[0], r[1], r[2])
            g = r[3]
        elif pid == _MID_LAMBERT:
            r = _stage_lambert(pos, view_dir, shade_n, face_n, out, g,
                               params, f, prim, 0,
                               light_pos, light_col, num_lights, shadows, vis)
            out = ti.math.vec3(r[0], r[1], r[2])
            g = r[3]
        elif pid == _MID_PHONG:
            r = _stage_phong(pos, view_dir, shade_n, face_n, out, g,
                             params, f, prim, 0,
                             light_pos, light_col, num_lights, shadows, vis)
            out = ti.math.vec3(r[0], r[1], r[2])
            g = r[3]
        elif pid == _MID_STANDARD:
            r = _stage_standard(pos, view_dir, shade_n, face_n, out, g,
                                params, f, prim, 0,
                                light_pos, light_col, num_lights, shadows, vis)
            out = ti.math.vec3(r[0], r[1], r[2])
            g = r[3]
        elif pid == _MID_PHYSICAL:
            r = _stage_physical(pos, view_dir, shade_n, face_n, out, g,
                                params, f, prim, 0,
                                light_pos, light_col, num_lights, shadows, vis)
            out = ti.math.vec3(r[0], r[1], r[2])
            g = r[3]
        elif pid == _MID_UNLIT:
            pass  # passthrough: colour returned unchanged (raw or baked).
        else:
            for pi in ti.static(range(len(frag_pipelines))):
                if pid == _USER_PIPELINE_BASE + pi:
                    fn = ti.static(frag_pipelines[pi])
                    r = fn(pos, view_dir, shade_n, face_n, out, g,
                           params, f, prim,
                           light_pos, light_col, num_lights, shadows, vis)
                    out = ti.math.vec3(r[0], r[1], r[2])
                    g = r[3]
    else:
        pid = pid_arr[f % pid_arr.shape[0], prim]
        if pid < _USER_PIPELINE_BASE:
            shade_n = _sided_shading_normal(n_interp, face_n, view_dir,
                                            params, f, prim)
        else:
            shade_n = _two_sided_normal(n_interp, face_n, 0.0, view_dir)
        # GATED: only the ids the batch carries get a branch. _MID_UNLIT never
        # needs one -- matching nothing leaves the colour unchanged, which is
        # its whole semantics.
        for mid in ti.static(range(len(_BUILTIN_STAGE_FNS))):
            if ti.static(mid != _MID_UNLIT and ((pids_present >> mid) & 1)):
                stage = ti.static(_BUILTIN_STAGE_FNS[mid])
                if pid == mid:
                    r = stage(pos, view_dir, shade_n, face_n, out, g,
                              params, f, prim, 0,
                              light_pos, light_col, num_lights, shadows, vis)
                    out = ti.math.vec3(r[0], r[1], r[2])
                    g = r[3]
        for pi in ti.static(range(len(frag_pipelines))):
            if ti.static((pids_present >> (_USER_PIPELINE_BASE + pi)) & 1):
                fn = ti.static(frag_pipelines[pi])
                if pid == _USER_PIPELINE_BASE + pi:
                    r = fn(pos, view_dir, shade_n, face_n, out, g,
                           params, f, prim,
                           light_pos, light_col, num_lights, shadows, vis)
                    out = ti.math.vec3(r[0], r[1], r[2])
                    g = r[3]
    # Bound the shaded colour to the display range. Lights accumulate here
    # without any normalisation -- each one adds its diffuse, ambient and
    # specular terms to the running colour -- so a scene with more than one
    # light drives a fully lit surface past 1.0 even though every individual
    # light is at or below unit intensity. Algan's own default rig (one white
    # PointLight) lands exactly on 1.0; tests/fast's three lights reach 2.15.
    #
    # That used to be the tonemap's problem. With tonemapping off by default
    # the encoder clamps instead, and a per-channel clamp truncates each
    # channel independently, so an over-range saturated colour loses its hue
    # and slides toward white -- a lit orange face turning flat yellow-white.
    # Scaling all three channels by the peak instead keeps the hue and only
    # gives up the brightness that had nowhere to go.
    #
    # Deliberately identity below 1.0, so everything already in range is
    # bit-identical and only pixels that were going to clip anyway change.
    # ``g`` (glow) is returned untouched, so glow remains the one thing that
    # can produce above-1.0 output for bloom to work with.
    #
    # Off under the linear working colour space (:func:`_linear_color_space`):
    # there light sums are physically additive and the sRGB OETF at the byte
    # write owns the range, so scaling by the peak would make lights stop
    # adding. The ``max(out, 0.0)`` clamp stays either way -- it is not part
    # of the bound; it stops a negative reaching the encoder's pow.
    out = ti.math.max(out, 0.0)
    if ti.static(bool(not _linear_color_space())):
        peak = ti.max(out[0], ti.max(out[1], out[2]))
        if peak > 1.0:
            out = out / peak
    return ti.math.vec4(out[0], out[1], out[2], g)
