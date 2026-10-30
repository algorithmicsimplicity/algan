"""Hybrid raster front-end kernels for deterministic primary visibility.

The frontend replaces the first classic wavefront iteration for flat triangles
and Bezier circuits.  Primitive/chunk pairs enumerate candidate pixels, exact
intersection tests reject bbox misses, and compact fragment records are
ordered by the same transitive ``(depth bin, descending layer)`` relation as
the classic tracer.  The host then compacts them into per-pixel sheets
(``sheets.compact_sheets``) and one kernel resolves, builds shadow events for,
and shades those (``sheet_resolve_taichi.sheet_resolve_shade``).

Important implementation properties:

* Triangle projection data is precomputed once per frame/primitive by
  :func:`raster_pipeline.precompute_triangle_projection`.  Straddling cases,
  and all-front triangles projected too far off-screen for the exact
  coverage lattice (see ``_AA_LATTICE_MAX_EXTENT``), use per-pixel ray
  casting instead.
* Triangle/circuit COUNT and WRITE kernels fetch sampled alpha and discard
  alpha-zero texels before sorting.  Records contain only exact-distance
  key, typed primitive reference (including the circuit border/fill blend
  weight), two intersection parameters, and an analytic coverage lane.
* Analytic anti-aliasing (``ALGAN_ANALYTIC_AA``, see DESIGN_analytic_aa.md):
  each circuit fragment carries the fraction of the pixel square its drawn
  region covers, using the configured circuit coverage estimator and the
  prepared outline geometry. The resolve folds coverage into alpha, so
  circuit silhouettes resolve continuously at
  ``anti_alias_level = 1`` instead of all-or-nothing.  The coverage lane is
  host-pre-filled to 1.0 and written by the circuit kernels and -- under
  ``analytic_aa_tri`` -- by ``raster_tri_write`` too, which carries each
  triangle fragment's exact clipped area for the sheet claims.
* :func:`raster_shadow_trace` traces the sheet resolve's sparse any-hit event
  queue and stores one visibility value per event/light. Its event storage is
  dynamically sized; the renderer's deterministic light-selection and
  truncation policy still applies. Point/spot emitter radii and
  directional angular radii use the same deterministic golden-angle fan as
  the classic wavefront path.

Emission covers the whole prepared frame window at once; each pair covers up
to ``raster_chunk`` pixels.  Future work should benchmark square block bins
and candidate-parallel block kernels. Custom scatter and the other route gates
can select the classic frontend; the camera's near clip plane cuts geometry
here instead (``_near_line``, ``_past_near``, ``_circuit_near_cover``). PN
surfaces are diced into flat triangles during geometry preparation, not
rejected merely for their PN origin.
"""
import warnings

from algan.environment import env_int
from algan.errors import AlganWarning
from algan.rendering.raytracing.arena_args_taichi import (
    ArenaView,
    RowView,
    arena_packed,
)
from algan.rendering.raytracing.raytrace_kernels_taichi import (
    _M_BASIS_U,
    _M_BASIS_V,
    _M_BORDER_CENTERED,
    _M_BORDER_W,
    _M_CENTER,
    _M_FILLED,
    _M_NORMAL,
    BARYCENTRIC_EPSILON,
    INV_DEPTH_TIE_EPSILON,
    NODE_ARG,
    CameraProjection,
    _axis_cos,
    _bezier_point_metrics_in,
    _camera_point,
    _circuit_inner_distance,
    _circuit_outer_dilation,
    _circuit_point_region,
    _circuit_query_radius,
    _generate_ray,
    _pixel_size_base,
    _projected,
    _sample_circuit_color_blend,
    _shadow_occluded,
    _tri_hit,
    min_alpha,
    min_hit_distance,
)
from algan.rendering.raytracing.shading_taichi import (
    _USER_PIPELINE_BASE,
    _orient_hit_normals_sided,
)
from algan.rendering.raytracing.transport_taichi import _offset_ray_origin, _shadow_tmax
from algan.rendering.raytracing.wavefront_kernels_taichi import (
    _ACTIVE,
    _GOLDEN_ANGLE,
    _LT_AMBIENT,
    _LT_AREA_SAMPLE,
    _LT_DIRECTIONAL,
    _LT_ENV_SH,
    _LT_HEMISPHERE,
    _R2_SEQUENCE_A1,
    _R2_SEQUENCE_A2,
    _TWO_PI,
    _first_covered_position,
    _light_zero_radiance,
    _reserve_continuation_slot,
    _reset_shell_segment,
    _shadow_fan_jitter,
    _shadow_jitter_cell,
    _tri_color_g,
    _tri_normal_g,
    _write_ior_stack,
)
from algan.settings._startup import _SOFT_SHADOW_SAMPLES as SOFT_SHADOW_SAMPLES
from algan.taichi_compat import ti

# Candidate pixels per (prim, chunk) pair: one fine-raster thread tests at
# most this many pixels, bounding load imbalance for large bboxes. A
# work-partitioning knob -- every candidate is exact-tested either way, so the
# image does not depend on it -- exposed for tuning against a card's occupancy.
# At most 32: the count pass records each pair's accepted pixels as one bit
# apiece in an i32 (``pair_accept``) that the write pass replays, so a 33rd
# pixel would have no bit and the two passes would disagree.
_RASTER_CHUNK_MAX = 32
raster_chunk = max(1, env_int("ALGAN_RASTER_CHUNK", _RASTER_CHUNK_MAX))
if raster_chunk > _RASTER_CHUNK_MAX:
    warnings.warn(
        f"ALGAN_RASTER_CHUNK={raster_chunk} exceeds the {_RASTER_CHUNK_MAX} "
        f"pixels one pair's acceptance mask can hold; using "
        f"{_RASTER_CHUNK_MAX}.",
        AlganWarning,
        stacklevel=2,
    )
    raster_chunk = _RASTER_CHUNK_MAX

# Empty typed visibility-buffer entry. Real hits pack the same strict ordering
# used by the classic deterministic tracer:
#
#   high 32 bits: floor(t / depth_tie_epsilon)
#   low  32 bits: bitwise-inverted layer index (higher layer sorts first)
#
# Triangle layers are ``layer_offset_triangles + prim``; bezier layers are the
# circuit index.  This makes the atomic minimum layer-aware and lets the host
# recover the winning geometry type/id from the layer without a second winner
# buffer. Exact t/barycentrics are recomputed only for the terminal winner.
Z_SENTINEL = 0x7FFFFFFFFFFFFFFF

# Depth bits of the record a WRITE kernel stores for a pixel the count pass
# accepted but the write pass's own geometry evaluation then missed. The two
# passes run the same code, so this should never happen; if it does, the slot
# the count reserved still has to be filled, and this fills it inertly: the
# candidate's own pixel, no samples, zero coverage, and the smallest positive
# depth, so it sorts ahead of its pixel's real fragments and the resolve skips
# it before any shading. (The write used to store the miss as-is: pixel 0, depth
# 0, full coverage -- an opaque fragment nobody emitted.) The low key bits are
# then exactly 1, which no real fragment's depth has, so the opt-in key check
# (``raster_pipeline._check_emitted_keys``) counts them.
_WRITE_MISS_T_BITS = 1

# Reach of the analytic-coverage box filter, in pixels: half the diagonal of the
# pixel square, so a nearest-edge query centred on the pixel centre is
# guaranteed to find every edge that can touch the square. (The filter itself
# uses the perpendicular half-pixel; this is the conservative query radius.)
_AA_FILTER_RADIUS = 0.7071067811865476

# Floor on the foreshortening a circuit's nearest-edge query widens for
# (``_bez_pixel_hit``). The filter's reach is measured on screen, so a plane
# tilted away from the camera spends more of itself on it along its most
# foreshortened direction, and the query grows by the inverse of how far that
# direction is squeezed. A plane within about six degrees of edge-on (squeezed
# below a tenth) keeps a query ten times the camera-facing one, which bounds the
# search and its cost; such a plane covers next to nothing on screen anyway.
_AA_MIN_SCREEN_SCALE = 0.1

# A pixel centre this close to a circuit's boundary, in camera-facing pixels, is
# taken to be on it: its closest-point vector is too short to give the boundary
# a direction, and ``_bez_pixel_hit`` orients it by the nearest segment instead.
# Plane coordinates of a few world units round at about 1e-6, a pixel is
# typically thousands of times that, so a hundredth of one leaves the vector's
# direction good to a couple of percent wherever it is used.
_AA_ON_BOUNDARY = 0.01

# How close, in pixels, a segment in the nearest-edge query's second slot must
# be to the nearest one to keep the slot against the nearest's own repeat from
# another grid cell (``_bezier_point_metrics_in``, screen mode): what the plane
# query's tie between a corner's two walls at their shared vertex comes to once
# the screen map's rounding has been through it. That rounding is of order 1e-5
# of a pixel for plane coordinates of a few world units, growing with them; a
# hundredth of a pixel is far above it and far below anything the filter shows.
_AA_RANK_TIE = 0.01

# A fragment whose analytic coverage is at least this is treated as covering the
# whole pixel: a materially opaque fragment at this coverage (with a full sample
# mask) ends its pixel's kept fragment prefix at emission (opaque truncation).
AA_FULL_COVERAGE = 1.0 - 1e-6

# Mask word layout: bits 0..N-1 the sample set, then the flags below, which sit
# at a FIXED shift above the widest supported sample set so that changing the
# sample count cannot silently collide with them. One sample set suffices for
# both claiming and occluding because the fill rule makes the partition exact.
_AA_FLAG_SHIFT = 16

# Facing bit. Adjacent triangles of a manifold mesh wind consistently, so the
# sign of the projected screen area separates a closed mesh's two sheets exactly
# and for free -- it is already computed for the coverage orientation. (A depth
# window was tried first and rejected: no threshold in pixel footprints survives
# grazing incidence, where it splits genuine shared edges.)
#
# Load-bearing downstream: the sheet compaction's band key, the closed-shell
# coverage ceiling and the one-mesh front/back sums all split a pixel's
# fragments by it. So BOTH coverage paths set it: _ss_pixel from the sign of
# the exact lattice area, _raycast_pixel (straddlers, and triangles demoted off
# the lattice) from the sign of its Moller-Trumbore determinant, calibrated to
# agree with the projected path (tests/unit_tests/test_raster_lattice_bound.py).
_AA_BACKFACE_BIT = 1 << _AA_FLAG_SHIFT

# Marks a fragment whose coverage came from the sliver rule rather than from a
# sample it actually contains: an AREAL fraction of the pixel with no position in
# it, so it attenuates every sample uniformly rather than a subset exactly (the
# same treatment a circuit's SDF coverage gets). Only reachable when
# analytic_aa_sliver is not the default ``drop``.
_AA_SLIVER_BIT = 2 << _AA_FLAG_SHIFT

# Marks every fragment of a pixel whose fragments are ALL opaque triangles of
# ONE surface (set by prepare_sparse_raster_coverage, which has the CSR to do it
# as a segment reduction). There the mesh's coverage is its near sheet's exact
# area and nothing else -- both sheets project to the same silhouette -- so the
# far sheet must not add coverage on top of it (the sheet resolve's
# per-pixel ceiling clamp reads it via the sheet records).
_AA_ONE_MESH_BIT = 4 << _AA_FLAG_SHIFT

# Marks a fragment of a MATERIAL-opaque triangle (the one-mesh rule's sense of
# opaque, before the coverage adjustment narrows it) -- set per fragment by
# prepare_sparse_raster_coverage under sheet_sample_depth. The sheet compaction
# folds it into a band's mask word so the per-sample depth gate can tell an
# enforcer sheet (may floor a strictly farther interpenetrating surface) from a
# translucent one. Every reader either masks with _AA_MASK_ALL or tests named
# flag bits, so the bit is inert where it is not read.
_AA_MAT_OPAQUE_BIT = 8 << _AA_FLAG_SHIFT  # bit 19, set per fragment

# sheet_sample_depth: the per-sheet "this sample was ceded to a strictly nearer
# other-surface enforcer" bits, computed on the host at compaction
# (sheets.compact_sheets) and applied to ``slots`` in the resolve's pre-eff
# block. Bits 20..27 of the mask word, one per sub-pixel sample; frag_msk /
# sheet_msk are int32, so bit 31 stays clear. NOTE: a future 16-sample pattern
# (_AA_PATTERN_16 above) needs sample bits 0..15 plus these flags and would
# collide with both this field and _AA_MAT_OPAQUE_BIT -- move them (and every
# reader of them) before widening the pattern.
_AA_LOSE_SHIFT = 20  # bits 20..27, per sheet

# §4.4 band continuation, set by the compaction (sheets._sibling_weights) on
# every sibling of a subdivided band but the one that closes it in walk order:
# the resolve claims such a sheet against the visibility the band met and
# leaves the band's occlusion write to the sheet that closes it. It used to
# ride in the sign of the sheet's coverage, which a sibling of zero area could
# not carry -- -0.0 is not below zero -- so its band closed early.
_AA_BAND_CONT_BIT = 1 << 28

# Marks a fragment the camera's near clip plane cut in this pixel: the cut took
# exact area from it (more than _NEAR_TRIM_AREA, decided by the write kernel's
# exact lanes), or, on the ray-cast path, a sample hit the triangle short of the
# plane. A ray that starts inside a solid crosses its shell once, not twice, so
# the closed-shell opacity ceiling must not spend this pixel's allowance as if
# the near shell were whole (sheets.compact_sheets). Set per fragment at
# emission; inert for every other reader. Bit 29, because 28 is the band
# continuation above and bit 31 stays clear.
_AA_NEAR_CUT_BIT = 1 << 29

# Marks every fragment of a pixel whose fragments are ALL triangles of ONE
# surface, whatever their material -- _AA_ONE_MESH_BIT's pixels and the
# transmissive and translucent ones besides -- set by
# prepare_sparse_raster_coverage beside the one-mesh bit, with the same
# ceiling in ``frag_cap``, and only in a batch whose primaries can refract.
# The sheet resolve applies that ceiling on such a pixel only once its walk
# has refracted at a full-alpha sheet of the solid: from there the sheet's
# refraction continuation stands for everything behind it, so a far sheet of
# the same solid must not claim the near sheet's ``corr < 1`` residue (the
# dark rim round clear glass). A translucent sheet, glass with refraction
# off, out of bounces or at alpha < 1, never refracts it that way, so the
# bit is inert there. Bit 30, because 28 and 29 are the band continuation
# and the near cut above, and bit 31 stays clear.
_AA_ONE_SOLID_BIT = 1 << 30

# Sample-less-triangle policies, matching ``settings.ANALYTIC_AA_SLIVER_MODES``.
# The live mode reaches ``_ss_pixel`` inside the ``aa`` template value (the
# geometry kernels are launched with ``1 + mode``) so that each policy gets its
# own compiled variant and its own offline-cache entry.
_AA_SLIVER_AREA = 0
_AA_SLIVER_EXACT = 1
_AA_SLIVER_DROP = 2
_AA_SLIVER_EXACT_OCC = 3

# Sub-pixel sample set. TRIANGLE coverage is the fraction of these points the
# triangle contains -- set arithmetic, not an area formula.
#
# A continuous area estimate was tried first and does not survive contact with a
# mesh, because an area alone cannot say WHERE in the pixel a fragment lies and
# the seam rule has to ADD sub-areas. Two failures follow directly. At a
# silhouette, several near-edge-on triangles each dilate half a pixel outward
# over the same sliver and their areas sum to a bright halo, dilating every mesh
# by a pixel. At a vertex, several wedges meet and their areas do not sum to the
# pixel, leaving a dark notch. Both are set-membership questions, so a set
# answers them: a fragment claims exactly the samples it contains that its group
# has not already taken. Coincident slivers claim the same sample once (no
# halo); the two halves of a shared edge claim complementary samples that sum to
# the whole pixel (no notch). Attempts to patch the area formula -- scaling by
# the fresh-sample ratio, bounding the group total by the sample count -- each
# fixed one failure and worsened the other; the set formulation has neither.
#
# The cost is real but small: the edge distances are affine in the sample
# position, so each sample is a few fused multiply-adds and no memory traffic --
# and still ONE shading per fragment, not one per sample, which is the whole
# point versus supersampling.
#
# Bezier circuits keep their continuous SDF coverage: they never group, so the
# set machinery buys them nothing, and the SDF is exact.

# Sub-pixel lattice for the exact coverage test: 1/4096 of a pixel. An edge
# function is a product of two coordinate differences, so the accumulator must
# be 64-bit -- which is also the ceiling on how fine this can go.
#
# Resolution matters because a triangle whose projected area rounds to ZERO on
# the lattice has no usable edge functions at all. Near a silhouette, where the
# surface turns edge-on, triangles get arbitrarily foreshortened; at 1/256 that
# left visible holes around every rim.
#
# What the 64 bits must hold is set by the triangle's VERTICES, not by the
# screen: a vertex just in front of the camera plane projects arbitrarily far
# off it. With M_x, M_y the triangle's largest |sx|, |sy| and W x H the screen,
# every product in _ss_pixel is at most 2^25 (M_x + W)(M_y + H) and every edge
# function, sample test and the area sum at most 2^26 (M_x + W)(M_y + H)
# lattice units. The wrapped int64 arithmetic is exact while those TRUE values
# fit, so _AA_LATTICE_MAX_EXTENT = 2^36 keeps them under 2^62; past about 2^38
# the orientation and the masks come out wrong (measured: most of a floor slab
# 0.01 in front of the camera vanished). The same bound covers _ss_pixel's
# float pre-reject: near an edge its two products are each about
# |edge| * |dx dy| / |d| <= |edge| * sqrt(2^36 / 2) px^2, d = (dx, dy) the
# pixel's offset from the edge's start, so six f32 roundings move a distance
# by at most 6 * 2^-24 * 2^17.5 = 0.066 px -- inside the 0.088 px between its
# 0.7072 threshold and the farthest of the 8 samples (the 16-sample pattern
# leaves 0.043 px, and would need 2^34). _AA_LATTICE_MAX_COORD bounds the
# coordinates themselves: at 2^21 px an f32 coordinate resolves only 1/4 px,
# and the f32 clipped areas built from it err by up to ~0.1 of a pixel -- the
# ray-cast path's own sampled-coverage error -- while span candidates (1 px
# margin) stay safe. raster_pipeline._triangle_projection_fused demotes an
# all-front triangle past either bound to flag 0.5, the per-pixel ray cast.
_AA_FIXED_SHIFT = 12
_AA_FIXED_SCALE = float(1 << _AA_FIXED_SHIFT)
_AA_FIXED_HALF = 1 << (_AA_FIXED_SHIFT - 1)
_AA_Q_INF = 1 << 62
_AA_LATTICE_MAX_EXTENT = float(1 << 36)
_AA_LATTICE_MAX_COORD = float(1 << 21)

# The standard sparse sample positions (the D3D 8x and 16x patterns), in 16ths
# of a pixel from its centre: one sample per row and per column of an 8x8 / 16x16
# grid, which resolves edges of every orientation evenly.
#
# Which one is live is a COMPILE-TIME constant, deliberately not a setting: the
# count is baked into every kernel that rasterizes but is part of no template
# argument, so the offline cache would happily serve an 8-sample kernel to a
# 16-sample build. Changing it means editing this line and clearing the cache.
# 16 matches the sampling density of the anti_alias_level=4 reference; the
# measured difference is in DESIGN_analytic_aa.md ss16.
_AA_PATTERN_8 = (
    (1, -3), (-1, 3), (5, 1), (-3, -5), (-5, 5), (-7, -1), (3, 7), (7, -7),
)
_AA_PATTERN_16 = (
    (1, 1), (-1, -3), (-3, 2), (4, -1), (-5, -2), (2, 5), (5, 3), (3, -5),
    (-2, 6), (0, -7), (-4, -6), (-6, 4), (-8, 0), (7, -4), (6, 7), (-7, -8),
)
_AA_SAMPLES = tuple(
    (x * (1 << _AA_FIXED_SHIFT) // 16, y * (1 << _AA_FIXED_SHIFT) // 16)
    for x, y in _AA_PATTERN_8
)
_AA_NUM_SAMPLES = len(_AA_SAMPLES)
_AA_MASK_ALL = (1 << _AA_NUM_SAMPLES) - 1
_AA_SAMPLE_WEIGHT = 1.0 / _AA_NUM_SAMPLES

# The near-cut line (``_near_line``) of a triangle the near clip plane does not
# cross: every pixel is this far inside it, so it keeps everything and never
# reaches the per-sample test. Its negation keeps nothing.
_NEAR_CLEAR = 1e30

# The area (in pixels) the near cut must take from a fragment before the
# fragment counts as cut (_AA_NEAR_CUT_BIT): above the clip helpers' float
# noise, and below any coverage error the shell ceiling would make visible.
_NEAR_TRIM_AREA = 1e-4


# Compare-swap network that sorts four values with five comparisons.
_AA_SORT4 = ((0, 1), (2, 3), (0, 2), (1, 3), (1, 2))


# Per-fragment debug dump (ALGAN_AA_DUMP="px,py,frame"; DESIGN_analytic_aa_v2.md
# ss7.1). Both walk kernels write one row per fragment they process at the
# requested pixel into a host-provided float buffer, so a golden host-side walk
# can recompute the pixel from the exact inputs the kernel saw and diff every
# column -- and the two walks' dumps can be diffed against each other, which is
# what catches a resolve/shadow-event desync on day one. Compiled out (a
# ``ti.static`` template flag) unless requested; the buffer's row 0 carries the
# target (px, py, frame) and the atomic row counter, so the feature costs one
# kernel argument.
#
# Row layout (_AA_DUMP_COLS wide):
#   [0] q      fragment index in the walk (the terminal row uses -1)
#   [1] kind   0 tri, 1 bez, 2 z-tri, 3 z-bez
#   [2] note   0 committed, 1 eff-skip, 2 bounce break, 3 occluded-glass
#              break, 4 far-clip break, 5 invalid, 6 seam skip
#   [3] ref    primitive / circuit id
#   [4] sid    surface id: tri_obj[ref] for a triangle, -1 - circuit for bez
#   [5] facing 1 when the fragment's backface bit is set
#   [6] msk    sample-mask low bits
#   [7] cov    frag_cov as emitted
#   [8] pop    popcount(msk)
#   [9] corr   the run correction applied (1.0 outside run mode)
#   [10] eff   effective coverage after per-sample transmittance (and corr)
#   [11] mat_alpha  material alpha before coverage
#   [12] alpha      committed alpha (mat_alpha * eff)
#   [13] trans_share
#   [14] refl_max
#   [15] t_hit
#   [16:24] svis after this fragment committed
# Terminal row: [-1, bounced, done, processed, vis_all, acc0, acc1, acc2,
#                acc3, w0, w1, w2, 0, 0, 0, 0, svis...]
_AA_DUMP_COLS = 24


@ti.func
def _aa_dump_match(dump_out: ti.template(), px, py, f):
    """Whether this thread's pixel is the one the dump targets (row 0)."""
    return ((px == ti.cast(dump_out[0, 0] + 0.5, ti.i32))
            and (py == ti.cast(dump_out[0, 1] + 0.5, ti.i32))
            and (f == ti.cast(dump_out[0, 2] + 0.5, ti.i32)))


@ti.func
def _aa_dump_reserve(dump_out: ti.template()):
    """Atomically reserve the next dump row; -1 when the buffer is full."""
    r = ti.cast(ti.atomic_add(dump_out[0, 3], 1.0) + 0.5, ti.i32) + 1
    if r >= dump_out.shape[0]:
        r = -1
    return r


@ti.func
def _aa_dump_frag(dump_out: ti.template(), q, kind, note, ref, sid, facing,
                  msk, cov, pop, corr, eff, mat_alpha, alpha, trans_share,
                  refl_max, t_hit, svis):
    r = _aa_dump_reserve(dump_out)
    if r >= 0:
        dump_out[r, 0] = ti.cast(q, ti.f32)
        dump_out[r, 1] = ti.cast(kind, ti.f32)
        dump_out[r, 2] = ti.cast(note, ti.f32)
        dump_out[r, 3] = ti.cast(ref, ti.f32)
        dump_out[r, 4] = ti.cast(sid, ti.f32)
        dump_out[r, 5] = ti.cast(facing, ti.f32)
        dump_out[r, 6] = ti.cast(msk & _AA_MASK_ALL, ti.f32)
        dump_out[r, 7] = cov
        dump_out[r, 8] = ti.cast(pop, ti.f32)
        dump_out[r, 9] = corr
        dump_out[r, 10] = eff
        dump_out[r, 11] = mat_alpha
        dump_out[r, 12] = alpha
        dump_out[r, 13] = trans_share
        dump_out[r, 14] = refl_max
        dump_out[r, 15] = t_hit
        for s in ti.static(range(_AA_NUM_SAMPLES)):
            dump_out[r, 16 + s] = svis[s]


@ti.func
def _aa_dump_terminal(dump_out: ti.template(), bounced, done, processed,
                      vis_all, acc, weight, svis):
    r = _aa_dump_reserve(dump_out)
    if r >= 0:
        dump_out[r, 0] = -1.0
        dump_out[r, 1] = 1.0 if bounced else 0.0
        dump_out[r, 2] = 1.0 if done else 0.0
        dump_out[r, 3] = ti.cast(processed, ti.f32)
        dump_out[r, 4] = vis_all
        for k in ti.static(range(4)):
            dump_out[r, 5 + k] = acc[k]
        for k in ti.static(range(3)):
            dump_out[r, 9 + k] = weight[k]
        for k in ti.static(range(4)):
            dump_out[r, 12 + k] = 0.0
        for s in ti.static(range(_AA_NUM_SAMPLES)):
            dump_out[r, 16 + s] = svis[s]


def _sliver_mode(aa):
    """Sample-less-triangle policy carried in the ``aa`` template value.

    ``aa`` is 0 (coverage off) or ``1 + mode + 4 * exact``; a plain int at
    kernel-compile time, so every use of this sits inside ``ti.static``.
    """
    return max(int(aa) - 1, 0) % 4














def _aa_run_full(aa_grp):
    """Whether the relaxed gate admits a FULL-mask fragment that covers less
    than the whole pixel (DESIGN_mesh_identity.md ss6.3.2).

    A diced mesh's silhouette produces full-mask fragments: one triangle
    owning all eight sub-pixel samples while covering a fraction of the
    pixel's area. Without the gate those are painted at 1.0 with their exact
    area unread, and on a fine Sphere they are 52% of the silhouette pixels.
    An interior fragment's ``cov`` is within dust of 1, so admitting
    ``cov < 1 - dust`` picks up exactly the silhouette and leaves the
    interior alone. Its one surviving reader is the emission truncation in
    ``prepare_sparse_raster_coverage``, which must keep a sheet's area donors
    in the stream whenever the relaxed semantics will read them.

    Carried as ``aa_grp == 2``; 3 is the one-mesh rule, which implies this
    gate.
    """
    return int(aa_grp) >= 2


#: How far below 1 a full-mask fragment's exact area must sit before the run
#: scan treats it as a silhouette rather than an interior tiling. Float dust in
#: the exact-area arithmetic is far below this; a silhouette pixel's shortfall
#: is orders of magnitude above it. Also the band inside which a full-union run
#: keeps ``corr = 1`` exactly, so a genuine interior tiling stays bit-identical.
_AA_FULL_DUST = 1e-3


def _tri_repr(aa):
    """Triangle representation carried in the geometry kernels' ``aa`` value:
    0 sampled points, 1 retired (the deleted cells emission; the value is not
    reissued), 2 run-corrected exact areas (``1 + sliver_mode + 4 * repr``,
    see :func:`_sliver_mode`).
    """
    return max(int(aa) - 1, 0) // 4


def _tri_run(aa):
    """Whether the geometry kernels emit run-corrected payloads (repr 2):
    exact clipped areas in ``frag_cov`` beside the untouched sample masks,
    slivers emitted as area donors at their clipped centroid.
    """
    return _tri_repr(aa) == 2


@ti.func
def _pixel_clip_area(vx, vy):
    """Exact area of (triangle n pixel square), the pixel centre at the origin.

    ``vx``/``vy`` are the triangle's projected vertices in pixels, already
    translated so the pixel square is [-0.5, 0.5]^2. Returns a value in [0, 1]
    that is ZERO for a triangle disjoint from the square and that SUMS EXACTLY
    over a tiling -- the two properties the continuous product-of-edge-distances
    form does not have. That form is a reconstruction filter: it deliberately
    spreads coverage half a pixel past the geometry, so a silhouette rim of
    foreshortened triangles, each tiling its neighbour, sums to far more than the
    area they actually cover and dilates the whole mesh (DESIGN_analytic_aa.md
    ss15.3).

    Method: the area is the boundary integral (1/2) o (x dy - y dx) taken around
    the triangle's outline PROJECTED ONTO THE SQUARE by a componentwise clamp.
    The clamp is the nearest-point map onto the square, so the part of the
    outline inside maps to itself while everything outside collapses onto the
    square's border, contributing nothing but the border runs that close the
    intersection -- the enclosed area is exactly (triangle n square). Sampled at
    the crossings of the four border lines, where the clamp changes linear
    branch, so each edge needs its four crossing parameters sorted (out-of-range
    ones clamp to an endpoint and drop out on their own, no branching).

    This is equivalent to Sutherland-Hodgman clipping followed by a shoelace,
    but keeps no vertex list -- hence no dynamically indexed local array, which
    Taichi handles poorly -- only a running accumulator.
    """
    acc = 0.0
    for k in ti.static(range(3)):
        ax = vx[ti.static(k)]
        ay = vy[ti.static(k)]
        bx = vx[ti.static((k + 1) % 3)]
        by = vy[ti.static((k + 1) % 3)]
        dx = bx - ax
        dy = by - ay
        # A segment parallel to a border line has no crossing with it; the zero
        # reciprocal sends both of that axis' parameters to an endpoint, which
        # the walk below skips over for free.
        invx = 0.0
        if ti.abs(dx) > 1e-20:
            invx = 1.0 / dx
        invy = 0.0
        if ti.abs(dy) > 1e-20:
            invy = 1.0 / dy
        tv = ti.math.vec4(
            ti.math.clamp((-0.5 - ax) * invx, 0.0, 1.0),
            ti.math.clamp((0.5 - ax) * invx, 0.0, 1.0),
            ti.math.clamp((-0.5 - ay) * invy, 0.0, 1.0),
            ti.math.clamp((0.5 - ay) * invy, 0.0, 1.0))
        for c in ti.static(range(len(_AA_SORT4))):
            i = ti.static(_AA_SORT4[c][0])
            j = ti.static(_AA_SORT4[c][1])
            if tv[i] > tv[j]:
                sw = tv[i]
                tv[i] = tv[j]
                tv[j] = sw
        cx = ti.math.clamp(ax, -0.5, 0.5)
        cy = ti.math.clamp(ay, -0.5, 0.5)
        for s in ti.static(range(4)):
            tt = tv[ti.static(s)]
            nx = ti.math.clamp(ax + dx * tt, -0.5, 0.5)
            ny = ti.math.clamp(ay + dy * tt, -0.5, 0.5)
            acc += cx * ny - nx * cy
            cx = nx
            cy = ny
        ex = ti.math.clamp(bx, -0.5, 0.5)
        ey = ti.math.clamp(by, -0.5, 0.5)
        acc += cx * ey - ex * cy
    # Sign follows the projected winding, which the caller knows from the exact
    # integer area; the magnitude is the same either way.
    return ti.min(ti.abs(acc) * 0.5, 1.0)


@ti.func
def _halfplane_clip_area(nx, ny, d):
    """Exact area of (half-plane n pixel square), the pixel centre at the origin.

    The covered side is ``{p : dot(n, p) + d >= 0}`` with ``n`` pointing INTO
    it, and ``d`` is the signed perpendicular distance from the pixel centre to
    the boundary line, in pixels, positive when the centre is covered.
    ``(nx, ny)`` needs no normalization -- only its direction is read.

    This is the one-crossing-edge case of :func:`_pixel_clip_area` in closed
    form.  A single straight boundary is what a pixel almost always contains, so
    it is the exact answer for a lone triangle edge and for a flattened outline
    segment, at a fraction of the cost of the full clip.

    Writing ``a >= b`` for the sorted components of the unit normal (so
    ``a^2 + b^2 = 1``, ``a >= 1/sqrt(2)``), the covered area is

        1                      d >= (a + b) / 2      the boundary clears the
        0                      d <= -(a + b) / 2     square entirely
        0.5 + d / a            |d| <= (a - b) / 2    a trapezoid: two opposite
                                                     sides of the square are cut
        dd^2 / (2 a b)         d < -(a - b) / 2      a corner triangle, dd being
        1 - dd^2 / (2 a b)     d >  (a - b) / 2      the normal distance from the
                                                     nearer corner to the line

    The branches meet continuously: at ``|d| = (a - b) / 2`` both forms give
    ``b / (2 a)``, and at ``|d| = (a + b) / 2`` the corner has shrunk to nothing.

    **This is the correction the circuit path needs.** For an AXIS-ALIGNED
    boundary ``b`` is zero, the trapezoid branch spans the whole square, and the
    result collapses to ``clamp(d + 0.5, 0, 1)`` -- which is the box filter
    :func:`_bez_pixel_hit` applies at EVERY orientation.  That filter is
    therefore exact for one orientation out of all of them: at 45 degrees it
    reports full coverage at ``d = 0.5``, where the truth is ``1 - (0.7071 -
    d)^2 = 0.957``.  The error peaks around 0.043 and is systematic in the edge's
    ANGLE, so it presents as diagonal edges carrying visibly different weight
    from horizontal ones rather than as noise.

    Verified against brute force, against :func:`_pixel_clip_area`, and for the
    complement property below by ``benchmarks/_aa_clip_area_check.py``.

    The property the seam rule needs: the two sides of one boundary sum to the
    whole pixel, ``A(n, d) + A(-n, -d) == 1``, and in the corner branches that
    holds BIT-EXACTLY (the pair computes an identical ``dd^2 / (2 a b)``, one
    returning it and the other returning one minus it).
    """
    ax = ti.abs(nx)
    ay = ti.abs(ny)
    inv_len = 1.0 / ti.max(ti.sqrt(ax * ax + ay * ay), 1e-30)
    a = ti.max(ax, ay) * inv_len
    b = ti.min(ax, ay) * inv_len
    # Half-width of the square along the normal, and the |d| below which the
    # boundary cuts two OPPOSITE sides rather than a corner.
    reach = 0.5 * (a + b)
    flat = 0.5 * (a - b)
    area = 0.0
    if d >= reach:
        area = 1.0
    elif d > -reach:
        if (d >= -flat) and (d <= flat):
            area = 0.5 + d / ti.max(a, 1e-30)
        else:
            # The cut corner's legs are dd/a and dd/b, so its area is
            # dd^2/(2ab). The division is unreachable at b == 0 -- there ``flat``
            # and ``reach`` coincide at 0.5 and the branches above take every
            # d -- but it is guarded anyway, since a predicated lane may still
            # evaluate it. Underflow is harmless for the same reason: dd <= b, so
            # a b small enough to flush dd^2 to zero has a true area of zero too.
            dd = reach - ti.abs(d)
            corner = dd * dd / ti.max(2.0 * a * b, 1e-30)
            if d < 0.0:
                area = corner
            else:
                area = 1.0 - corner
    return area


@ti.func
def _pixel_clip_centroid(vx, vy):
    """Area and CENTROID of (triangle n pixel square), the centre at the origin.

    The centroid is what replaces the centroid-of-owned-samples that the point
    representation used to evaluate a fragment's depth and barycentrics at
    (ss15). That point has to lie INSIDE the triangle: the pixel centre lies
    outside a partially covering triangle about half the time, and there the
    plane intersection is an extrapolation past the geometry, which mis-sorts
    the two sheets of a closed mesh at every silhouette. An intersection of two
    convex sets is convex, so its centroid is inside both -- unlike an
    area-weighted average of CELL CENTRES, which can easily fall outside the
    triangle and would put the bug straight back.

    Same clamped boundary integral as :func:`_pixel_clip_area`, carrying the two
    first moments alongside the area; returns ``(area, cx, cy)``, the centroid
    falling back to the pixel centre for a degenerate (zero-area) clip.
    """
    acc = 0.0
    mx = 0.0
    my = 0.0
    for k in ti.static(range(3)):
        ax = vx[ti.static(k)]
        ay = vy[ti.static(k)]
        bx = vx[ti.static((k + 1) % 3)]
        by = vy[ti.static((k + 1) % 3)]
        dx = bx - ax
        dy = by - ay
        invx = 0.0
        if ti.abs(dx) > 1e-20:
            invx = 1.0 / dx
        invy = 0.0
        if ti.abs(dy) > 1e-20:
            invy = 1.0 / dy
        tv = ti.math.vec4(
            ti.math.clamp((-0.5 - ax) * invx, 0.0, 1.0),
            ti.math.clamp((0.5 - ax) * invx, 0.0, 1.0),
            ti.math.clamp((-0.5 - ay) * invy, 0.0, 1.0),
            ti.math.clamp((0.5 - ay) * invy, 0.0, 1.0))
        for c in ti.static(range(len(_AA_SORT4))):
            i = ti.static(_AA_SORT4[c][0])
            j = ti.static(_AA_SORT4[c][1])
            if tv[i] > tv[j]:
                sw = tv[i]
                tv[i] = tv[j]
                tv[j] = sw
        cx = ti.math.clamp(ax, -0.5, 0.5)
        cy = ti.math.clamp(ay, -0.5, 0.5)
        for s in ti.static(range(4)):
            tt = tv[ti.static(s)]
            nx = ti.math.clamp(ax + dx * tt, -0.5, 0.5)
            ny = ti.math.clamp(ay + dy * tt, -0.5, 0.5)
            cr = cx * ny - nx * cy
            acc += cr
            mx += (cx + nx) * cr
            my += (cy + ny) * cr
            cx = nx
            cy = ny
        ex = ti.math.clamp(bx, -0.5, 0.5)
        ey = ti.math.clamp(by, -0.5, 0.5)
        cr = cx * ey - ex * cy
        acc += cr
        mx += (cx + ex) * cr
        my += (cy + ey) * cr
    area = ti.abs(acc) * 0.5
    ox = 0.0
    oy = 0.0
    if area > 1e-9:
        # The 1/(6A) of the polygon centroid, with acc = 2A carrying the sign,
        # so the winding cancels instead of needing to be known.
        inv6 = 1.0 / (3.0 * acc)
        ox = mx * inv6
        oy = my * inv6
    return ti.min(area, 1.0), ox, oy


@ti.func
def _two_halfplane_area(n1x, n1y, d1, n2x, n2y, d2, a1, a2):
    """Exact area of (pixel square n H1 n H2), both in (unit normal, distance) form.

    ``a1``/``a2`` are the caller's already-computed single half-plane areas.

    Two regimes, because the apex of the wedge runs off to infinity as the lines
    approach parallel:

      * NEAR-PARALLEL -- the two half-planes' complements are disjoint, so
        inclusion-exclusion collapses to ``a1 + a2 - 1`` with no clipping at all.
        Verified exact to 2.8e-16 for antiparallel lines; the error grows with
        tilt (0.022 at 10 degrees, 0.049 at 20), which is what sets the cutoff.
      * OTHERWISE -- the wedge is a convex cone, and a cone is a TRIANGLE once
        its two rays are truncated far enough out that the truncation cannot
        reach the pixel. So the exact clipped area is :func:`_pixel_clip_area`,
        which is already exact and already validated, on a big triangle.
    """
    out = 0.0
    cross = n1x * n2y - n1y * n2x
    # Trivial containment FIRST, and not merely as an optimization. A second
    # segment somewhere else in the shape is usually tens of pixels away, which
    # puts the apex of the wedge tens of pixels away too; truncating its rays at
    # any fixed length then misses the pixel entirely and reports zero coverage
    # where the answer is "the first half-plane, unchanged". Short-circuiting
    # also bounds everything below: past here both lines genuinely cross the
    # square, so |d| < 0.71 and the apex is within 0.71*2/0.2 ~ 7 pixels.
    if a2 >= 1.0 - 1e-6:
        out = a1
    elif a1 >= 1.0 - 1e-6:
        out = a2
    elif (a1 <= 1e-6) or (a2 <= 1e-6):
        out = 0.0
    elif ti.abs(cross) < 0.2:
        # Near-parallel splits in two by the normals' SIGN, and conflating them
        # is wrong by up to 0.42. Facing each other (antiparallel) the two
        # complements are disjoint and inclusion-exclusion collapses; pointing
        # the same way they are NESTED, and the intersection is simply whichever
        # half-plane is the more restrictive.
        if (n1x * n2x + n1y * n2y) < 0.0:
            out = ti.math.clamp(a1 + a2 - 1.0, 0.0, 1.0)
        else:
            out = ti.min(a1, a2)
    else:
        inv = 1.0 / cross
        # Apex: the point on both lines.
        apx = (-d1 * n2y + d2 * n1y) * inv
        apy = (-n1x * d2 + n2x * d1) * inv
        # Each boundary ray runs along its own line, in the direction that lies
        # inside the OTHER half-plane; both signs follow from the cross product.
        s = 1.0
        if cross < 0.0:
            s = -1.0
        # Long enough that truncation cannot reach the square, short enough that
        # the shoelace keeps its precision in f32.
        r = 4.0 * (ti.abs(apx) + ti.abs(apy) + 2.0)
        t1x = -n1y * (s * r)
        t1y = n1x * (s * r)
        t2x = n2y * (s * r)
        t2y = -n2x * (s * r)
        out = _pixel_clip_area(
            ti.math.vec3(apx, apx + t1x, apx + t2x),
            ti.math.vec3(apy, apy + t1y, apy + t2y))
    return out


@ti.func
def _near_cut_area(vx, vy, nx, ny, c, centroid: ti.template()):
    """Exact area of (triangle n kept side of a line n pixel square), the pixel
    centre at the origin; with ``centroid``, also that region's centroid.

    The line is the near clip plane's cut (:func:`_near_line`): ``nx * x + ny
    * y + c >= 0`` is the side it keeps, ``c`` the pixel centre's signed
    distance. Clipping a triangle by one half-plane leaves a triangle (one
    vertex kept) or a quad (two), so the region is at most two triangles,
    each measured by the exact clip of :func:`_pixel_clip_area` /
    :func:`_pixel_clip_centroid`. Returns ``(area, cx, cy)``; the centroid is
    the pixel centre when the region is empty or ``centroid`` is off.
    """
    g = ti.math.vec3(nx * vx[0] + ny * vy[0] + c,
                     nx * vx[1] + ny * vy[1] + c,
                     nx * vx[2] + ny * vy[2] + c)
    k0 = 0
    if g[0] >= 0.0:
        k0 = 1
    k1 = 0
    if g[1] >= 0.0:
        k1 = 1
    k2 = 0
    if g[2] >= 0.0:
        k2 = 1
    kept = k0 + k1 + k2
    # Rotate the vertex whose side differs from the other two's to the front.
    ax = vx[0]
    ay = vy[0]
    ga = g[0]
    bx = vx[1]
    by = vy[1]
    gb = g[1]
    cx = vx[2]
    cy = vy[2]
    gcc = g[2]
    if (k1 != k0) and (k1 != k2):
        ax = vx[1]
        ay = vy[1]
        ga = g[1]
        bx = vx[2]
        by = vy[2]
        gb = g[2]
        cx = vx[0]
        cy = vy[0]
        gcc = g[0]
    elif (k2 != k0) and (k2 != k1):
        ax = vx[2]
        ay = vy[2]
        ga = g[2]
        bx = vx[0]
        by = vy[0]
        gb = g[0]
        cx = vx[1]
        cy = vy[1]
        gcc = g[1]
    # The line's crossings of the two edges at the odd vertex (unused when
    # every vertex is on one side).
    sab = ga / ti.select(ga != gb, ga - gb, 1.0)
    sac = ga / ti.select(ga != gcc, ga - gcc, 1.0)
    pabx = ax + sab * (bx - ax)
    paby = ay + sab * (by - ay)
    pacx = ax + sac * (cx - ax)
    pacy = ay + sac * (cy - ay)
    # First triangle: the whole one, the kept corner, or half the kept quad.
    t1x = ti.math.vec3(ax, bx, cx)
    t1y = ti.math.vec3(ay, by, cy)
    if kept == 1:
        t1x = ti.math.vec3(ax, pabx, pacx)
        t1y = ti.math.vec3(ay, paby, pacy)
    elif kept == 2:
        t1x = ti.math.vec3(bx, cx, pacx)
        t1y = ti.math.vec3(by, cy, pacy)
    area = 0.0
    mx = 0.0
    my = 0.0
    if kept > 0:
        if ti.static(centroid):
            a1, ox1, oy1 = _pixel_clip_centroid(t1x, t1y)
            area = a1
            mx = a1 * ox1
            my = a1 * oy1
        else:
            area = _pixel_clip_area(t1x, t1y)
    if kept == 2:
        t2x = ti.math.vec3(bx, pacx, pabx)
        t2y = ti.math.vec3(by, pacy, paby)
        if ti.static(centroid):
            a2, ox2, oy2 = _pixel_clip_centroid(t2x, t2y)
            area += a2
            mx += a2 * ox2
            my += a2 * oy2
        else:
            area += _pixel_clip_area(t2x, t2y)
    ox = 0.0
    oy = 0.0
    if ti.static(centroid):
        if area > 1e-9:
            ox = mx / area
            oy = my / area
    return ti.min(area, 1.0), ox, oy


@ti.func
def _near_clipped_pixel_area(ax, ay, bx, by, cx, cy, clear, nl, gc):
    """:func:`_near_cut_area` for :func:`_ss_pixel`'s exact lane.

    The vertices are relative to the pixel centre, ``gc`` that centre's
    distance from the cut. ``clear`` says no triangle edge reaches the pixel,
    which leaves the cut's own half-plane in closed form -- the common case,
    the cut running through a large triangle's interior.
    """
    ca = 0.0
    if clear:
        ca = _halfplane_clip_area(nl[0], nl[1], gc)
    else:
        ca, _cx, _cy = _near_cut_area(
            ti.math.vec3(ax, bx, cx), ti.math.vec3(ay, by, cy),
            nl[0], nl[1], gc, False)
    return ca


@ti.func
def _boundary_coverage(nx, ny, d, aa: ti.template()):
    """Pixel coverage by the drawn side of one boundary, at signed distance ``d``.

    ``aa == 2`` (``analytic_aa_exact``) takes the exact angle-aware area;
    anything else keeps the box filter ``clamp(d + 0.5, 0, 1)``, which IS that
    area for an axis-aligned boundary and an approximation at every other angle
    (see :func:`_halfplane_clip_area`).  The choice is a compile-time template
    value, so the two forms never share a compiled kernel or a cache entry.

    A boundary with no direction has nothing to orient: no outline segment came
    within the query radius, so the pixel is deep inside the shape or far outside
    it, and ``|d| >= 0.5`` makes both forms agree on 1 or 0 anyway.
    """
    c = 0.0
    if ti.static(int(aa) >= 2):
        if (nx == 0.0) and (ny == 0.0):
            c = ti.math.clamp(d + 0.5, 0.0, 1.0)
        else:
            c = _halfplane_clip_area(nx, ny, d)
    else:
        c = ti.math.clamp(d + 0.5, 0.0, 1.0)
    return c


def _bez_anisotropic():
    """Whether :func:`_bez_pixel_hit` measures circuit coverage on screen.

    ``rt_settings.analytic_aa_bez_anisotropic``, read live through the module
    rather than imported by value: the use is behind ``ti.static``, so whatever
    the setting holds when a kernel compiles is what that kernel keeps, and the
    field is in ``KERNEL_COMPILED_IN_FIELDS`` so that a render which moves it
    gets kernels rebuilt around the new answer.
    """
    from algan.rendering.raytracing import settings as rt_settings

    return bool(rt_settings.analytic_aa_bez_anisotropic)


@ti.func
def _circuit_screen_map(f, ro, rd, bu, bv, screen_point: ti.template(),
                        pixel_basis_x: ti.template(),
                        pixel_basis_y: ti.template(),
                        cam_origin: ti.template(), cam_proj: ti.template()):
    """How a circuit's plane lands on the screen at one pixel's hit.

    Returns the 2x2 matrix ``M = ((m00, m01), (m10, m11))`` that takes a
    displacement ``(du, dv)`` along the plane frame ``(bu, bv)`` to a screen
    displacement of ``M (du, dv) / pixel_size`` pixels, ``pixel_size`` being the
    camera-facing footprint :func:`_bez_pixel_hit` measures widths in. Where the
    plane faces the camera ``M`` is a rotation or a reflection -- the plane
    frame is an arbitrary orthonormal pair (``primitives``' ``basis_u``), not
    the screen's axes -- and everything built on it reduces to the isotropic
    filter. A plane tilted away from the camera is squeezed along its
    foreshortened direction, which is ``M``'s smaller singular value.

    A world displacement ``w`` at the hit moves its projection across the view
    by ``w - (w . fwd) rd / (rd . fwd)`` per unit of perpendicular depth: the
    depth change it brings slides the projection back toward the axis along
    the ray. Measured along the screen's own axes, that is the exact local
    Jacobian at the pixel centre of a flat image plane, off-axis included.

    The axis runs from the eye, ``ro`` for a pinhole camera and
    ``cam_origin``'s row off it (see :func:`_axis_cos`): an orthographic ray
    starts off the axis, and its ``rd`` is the axis itself.
    """
    fwd = (ti.math.vec3(screen_point[f, 0], screen_point[f, 1],
                        screen_point[f, 2]) - ro).normalized()
    if ti.static(_projected(cam_proj)):
        fwd = (ti.math.vec3(screen_point[f, 0], screen_point[f, 1],
                            screen_point[f, 2])
               - ti.math.vec3(cam_origin[f, 0], cam_origin[f, 1],
                              cam_origin[f, 2])).normalized()
    ex = ti.math.vec3(pixel_basis_x[f, 0], pixel_basis_x[f, 1],
                      pixel_basis_x[f, 2]).normalized()
    ey = ti.math.vec3(pixel_basis_y[f, 0], pixel_basis_y[f, 1],
                      pixel_basis_y[f, 2]).normalized()
    slide = rd / ti.max(rd.dot(fwd), 1e-6)
    ax = ex - slide.dot(ex) * fwd
    ay = ey - slide.dot(ey) * fwd
    return ax.dot(bu), ax.dot(bv), ay.dot(bu), ay.dot(bv)


@ti.func
def _screen_scales(m00, m01, m10, m11):
    """The singular values ``(s_min, s_max)`` of a :func:`_circuit_screen_map` matrix.

    How far the plane's most and least foreshortened directions are scaled on
    screen: both 1 facing the camera, ``s_min`` falling toward 0 as the plane
    turns edge-on. ``s_min`` is taken as ``|det| / s_max`` so that a nearly
    singular matrix does not lose it to cancellation.
    """
    t = m00 * m00 + m01 * m01 + m10 * m10 + m11 * m11
    det = m00 * m11 - m01 * m10
    s_max = ti.sqrt(0.5 * (t + ti.sqrt(ti.max(t * t - 4.0 * det * det, 0.0))))
    return ti.abs(det) / ti.max(s_max, 1e-30), s_max


@ti.func
def _segment_width_scale(m00, m01, m10, m11, ex, ey):
    """How far a width laid across a segment in the plane spans on screen, per unit.

    ``(ex, ey)`` is the segment ON SCREEN, ``M e`` for its plane direction
    ``e`` (as :func:`_bezier_point_metrics_in` reports it in screen mode). A
    width ``w`` along the segment's plane normal moves its line ``w * k`` across
    the screen (per ``pixel_size``), ``k`` being the area ``M`` gives the unit
    square on ``e`` over the length it gives ``e``: ``|det M| |e| / |M e|``,
    which is ``|adj(M) M e| / |M e|``. 1 where the plane faces the camera and for
    a segment running along the foreshortened direction; the squeeze itself for
    one running across it. 1 for no segment at all.
    """
    k = 1.0
    le = ti.sqrt(ex * ex + ey * ey)
    if le > 1e-30:
        ax = m11 * ex - m01 * ey
        ay = m00 * ey - m10 * ex
        k = ti.sqrt(ax * ax + ay * ay) / le
    return k


@ti.func
def _popcount_samples(x):
    """Number of sub-pixel samples set, i.e. set bits in the low sample bits."""
    v = ti.cast(x, ti.u32) & ti.u32(_AA_MASK_ALL)
    v = v - ((v >> 1) & ti.u32(0x5555))
    v = (v & ti.u32(0x3333)) + ((v >> 2) & ti.u32(0x3333))
    v = (v + (v >> 4)) & ti.u32(0x0F0F)
    return ti.cast((v + (v >> 8)) & ti.u32(0x1F), ti.i32)




@ti.func
def _coverage_density(cov, msk, areal):
    """Split a fragment's coverage into WHERE it sits and HOW MUCH covers it.

    Returns ``(cmsk, nsm, dens)``: the sub-pixel samples this fragment
    attenuates, how many that is, and the fraction of each one it covers.  The
    resolve then reads a fragment as ``dens`` on ``cmsk`` and zero elsewhere::

        eff = dens * sum(svis[s] for s in cmsk) / N
        for s in cmsk: svis[s] *= 1 - alpha * dens

    The two quantities are INDEPENDENT, which is the whole point.  A mask can
    only express coverage in multiples of 1/N, so as long as it carried the
    magnitude too, an exact area had nowhere to go; splitting them lets the mask
    answer the set question (which part of the pixel, hence which fragment
    occludes which) while ``cov`` answers the measure question exactly.

    It also unifies the two fragment kinds that used to need separate branches
    at five sites each:

      * A mask whose popcount IS the coverage gives ``dens == 1`` -- what every
        triangle produces today, and what the masked branch assumed.
      * An AREAL fragment -- a circuit's SDF coverage, or a sliver's clipped
        area, both a fraction of the pixel with no position in it -- spreads over
        every sample, giving ``cmsk == all`` and ``dens == cov``, which is what
        the areal branch computed.

    Both come out BIT-IDENTICAL to the branches they replace (``cov`` is a
    multiple of 1/N in the first case and the mask is already full in the
    second), so this is a pure refactor until the geometry paths start emitting
    exact areas.
    """
    cmsk = msk
    if areal:
        cmsk = _AA_MASK_ALL
    nsm = _popcount_samples(cmsk)
    dens = 0.0
    if nsm > 0:
        # (cov * N) / nsm, and the grouping is load-bearing: when the mask IS
        # the coverage, cov is a multiple of 1/N, so cov * N is exact and the
        # quotient is exactly 1.0. Written cov * (N / nsm) it would round 8/3
        # first and land at 0.99999997, which is not the same render.
        dens = ti.min(
            cov * ti.static(float(_AA_NUM_SAMPLES)) / ti.cast(nsm, ti.f32), 1.0)
    # The clamp is inert while coverage comes from the mask, and becomes load
    # bearing with exact areas: a triangle can cover more of the pixel than its
    # share of the samples suggests, and a sample cannot be covered twice.
    return cmsk, nsm, dens






@ti.func
def _run_svis_write(svis: ti.template(), slots, a_s, trans_share, cfac,
                    rule_b: ti.template()):
    """The walk's per-sample write with the run correction folded in.

    ``svis[s] *= (1 - ak) + ak * trans_share`` with ``ak = cfac * a_s *
    slots[s]`` -- at ``cfac == 1`` this is bit-for-bit the shipped write for
    both the opaque (``trans_share == 0``) and transmitting forms. corr > 1
    can push a factor negative; rule A clamps it at zero (the claim stays
    exact, the leftover keeps a bounded residual), rule B additionally
    returns the clamped-away amount so the caller can push it onto the run's
    unowned samples at run end (ss4.4).
    """
    resid = 0.0
    for s in ti.static(range(_AA_NUM_SAMPLES)):
        ak = cfac * a_s * slots[s]
        fct = (1.0 - ak) + ak * trans_share
        if ti.static(rule_b):
            if fct < 0.0:
                resid -= fct * svis[s]
                fct = 0.0
        else:
            fct = ti.max(fct, 0.0)
        svis[s] *= fct
    return resid


@ti.func
def _run_redistribute(svis: ti.template(), run_U, resid):
    """Rule B's run-end step: remove the clamped residue from the samples the
    run did NOT own, capping at zero (residue beyond their total is dropped
    -- the owned samples are already at zero and cannot give more).
    """
    if resid > 0.0:
        tot = 0.0
        for s in ti.static(range(_AA_NUM_SAMPLES)):
            if ((run_U >> s) & 1) == 0:
                tot += svis[s]
        if tot > 1e-12:
            sc = ti.max(1.0 - resid / tot, 0.0)
            for s in ti.static(range(_AA_NUM_SAMPLES)):
                if ((run_U >> s) & 1) == 0:
                    svis[s] *= sc


@ti.func
def _order_key(t, layer):
    """Packed transitive depth-bin / descending-layer key."""
    # Keep bit 63 clear so signed ``atomic_min`` preserves unsigned depth
    # order even for pathological distances.  Layers share the 32-bit packed
    # representation used by the classic renderer.
    depth_f = ti.floor(t * INV_DEPTH_TIE_EPSILON)
    depth_bin = ti.cast(ti.math.clamp(depth_f, 0.0, 2147483647.0), ti.u64)
    layer_u = ti.cast(ti.min(ti.max(layer, 0), 2147483647), ti.u64)
    return ti.cast((depth_bin << 32) | (ti.u64(0xFFFFFFFF) - layer_u), ti.i64)


@ti.func
def _frag_t(key):
    """Recover exact positive f32 distance from a fragment's low key bits."""
    return ti.bit_cast(ti.cast(ti.cast(key, ti.u64) & ti.u64(0xFFFFFFFF), ti.u32), ti.f32)


# Quantization of a circuit fragment's border/fill blend weight, which rides in
# the low bits of its packed ref rather than in a per-fragment lane of its own:
# 8 bits resolve the blend finer than the framebuffer it lands in, and the
# circuit id has 23 bits left over, far more than a batch can hold.
_BEZ_BORDER_LEVELS = 255
_BEZ_BORDER_BITS = 8


@ti.func
def _pack_bez_ref(circuit, border_frac):
    """Negative typed fragment ref with the border weight folded into the id."""
    q = ti.cast(ti.round(ti.math.clamp(border_frac, 0.0, 1.0)
                         * _BEZ_BORDER_LEVELS), ti.i32)
    return -((circuit << _BEZ_BORDER_BITS) + q + 1)


@ti.func
def _decode_bez_ref(ref):
    """Inverse of :func:`_pack_bez_ref`: ``(circuit, border_frac)``."""
    code = -ref - 1
    return (code >> _BEZ_BORDER_BITS,
            ti.cast(code & _BEZ_BORDER_LEVELS, ti.f32)
            * (1.0 / _BEZ_BORDER_LEVELS))




@ti.func
def _near_line(sm, vm, cam_o, fwd, near_clip):
    """The near clip plane across a projected triangle, as a line on screen.

    Returns ``(nx, ny, c)``: ``nx * x + ny * y + c`` is the signed distance in
    pixels from the screen point ``(x, y)`` to where the plane cuts the
    triangle, positive on the side the plane keeps. Meaningful inside the
    triangle only, which is all :func:`_ss_pixel` asks of it.

    Depth along the forward axis is affine in world space, so across the
    triangle it is the perspective-correct blend of its vertices' depths
    ``z_i``, weighted by the screen barycentrics over the divisors ``d_i``
    (``sm[2]`` holds ``1 / d_i``). The plane therefore crosses edge ``i -> j``
    at screen parameter ``a_i / (a_i - a_j)``, with ``a_i = (z_i - near) /
    |d_i|``, and projection keeps the cut straight: the line through the two
    edge crossings. A triangle the plane only touches at a vertex keeps no
    area, and gets a line that keeps nothing.
    """
    a = ti.math.vec3(0.0, 0.0, 0.0)
    for i in ti.static(range(3)):
        v = ti.math.vec3(vm[i, 0], vm[i, 1], vm[i, 2])
        a[i] = ti.abs(sm[2, i]) * ((v - cam_o).dot(fwd) - near_clip)
    c1x = 0.0
    c1y = 0.0
    c2x = 0.0
    c2y = 0.0
    cuts = 0
    for e in ti.static(range(3)):
        i = ti.static(e)
        j = ti.static((e + 1) % 3)
        if (a[i] >= 0.0) != (a[j] >= 0.0):
            s = a[i] / (a[i] - a[j])
            x = sm[0, i] + s * (sm[0, j] - sm[0, i])
            y = sm[1, i] + s * (sm[1, j] - sm[1, i])
            if cuts == 0:
                c1x = x
                c1y = y
            else:
                c2x = x
                c2y = y
            cuts += 1
    # The vertex the plane keeps most clearly orients the line.
    kx = sm[0, 0]
    ky = sm[1, 0]
    best = a[0]
    for i in ti.static(range(1, 3)):
        if a[i] > best:
            best = a[i]
            kx = sm[0, i]
            ky = sm[1, i]
    nl = ti.math.vec3(0.0, 0.0, -_NEAR_CLEAR)
    dx = c2x - c1x
    dy = c2y - c1y
    ln = ti.sqrt(dx * dx + dy * dy)
    if (cuts == 2) and (ln > 0.0):
        nx = -dy / ln
        ny = dx / ln
        if nx * (kx - c1x) + ny * (ky - c1y) < 0.0:
            nx = -nx
            ny = -ny
        nl = ti.math.vec3(nx, ny, -(nx * c1x + ny * c1y))
    elif (cuts == 0) and (best >= 0.0):
        # No edge crosses (the host's flag and these depths disagree by a
        # rounding): keep the triangle whole.
        nl = ti.math.vec3(0.0, 0.0, _NEAR_CLEAR)
    return nl


@ti.func
def _past_near(ro, rd, t, f, cam_origin: ti.template(), fwd, near_clip):
    """Whether the point ``t`` along a primary ray lies past the near clip
    plane -- always, without one. The plane sits ``near_clip`` along the
    forward axis ``fwd`` from the eye, as ``wavefront_generate_rays`` places it.
    """
    eye = ti.math.vec3(cam_origin[f, 0], cam_origin[f, 1], cam_origin[f, 2])
    return (near_clip <= 0.0) or ((ro + t * rd - eye).dot(fwd) >= near_clip)


@ti.func
def _forward_axis(f, cam_origin: ti.template(), screen_point: ti.template()):
    """The camera's unit forward axis, the one its near plane is measured on."""
    return (ti.math.vec3(screen_point[f, 0], screen_point[f, 1],
                         screen_point[f, 2])
            - ti.math.vec3(cam_origin[f, 0], cam_origin[f, 1],
                           cam_origin[f, 2])).normalized()


@ti.func
def _ss_setup(f, prim, ss_enabled: ti.template(), tri_pos: ti.template(),
              tri_screen: ti.template(), cam_origin: ti.template(),
              screen_point: ti.template(), near_clip, aa: ti.template()):
    """Load the per-(frame, triangle) projection prepared once by the host.

    ``tri_screen[..., 0:3]`` are sx, ``3:6`` sy, ``6:9`` reciprocal
    perspective divisors, and column 9 is the path flag
    (:func:`raster_pipeline.precompute_triangle_projection` lists its
    values): above 0.5 the projected lattice path, otherwise the ray-cast
    fallback -- which includes 0.5, an all-front triangle too far off-screen
    for the lattice.  World vertices are still read from ``tri_pos`` for hit
    reconstruction and the ray-cast fallback.  This removes repeated camera
    projection setup from every count and write chunk.

    Under analytic coverage (``aa``) columns 10:13 additionally hold the
    reciprocal screen edge lengths, returned as ``il``; the host only allocates
    them when the feature is on, so ``aa`` must be derived from the table width
    the host produced rather than the live setting.

    ``nl`` is the near clip plane's cut across the triangle
    (:func:`_near_line`) when the host flagged it 2, an all-front triangle the
    plane crosses; otherwise a line that keeps everything.
    """
    tp = f % tri_pos.shape[0]
    ts = f % tri_screen.shape[0]
    v0 = ti.math.vec3(tri_pos[tp, prim, 0], tri_pos[tp, prim, 1],
                      tri_pos[tp, prim, 2])
    v1 = ti.math.vec3(tri_pos[tp, prim, 3], tri_pos[tp, prim, 4],
                      tri_pos[tp, prim, 5])
    v2 = ti.math.vec3(tri_pos[tp, prim, 6], tri_pos[tp, prim, 7],
                      tri_pos[tp, prim, 8])
    cam_o = ti.math.vec3(cam_origin[f, 0], cam_origin[f, 1], cam_origin[f, 2])
    vm = ti.Matrix([[v0[0], v0[1], v0[2]],
                    [v1[0], v1[1], v1[2]],
                    [v2[0], v2[1], v2[2]]])
    sm = ti.Matrix.zero(ti.f32, 3, 3)
    il = ti.math.vec3(0.0, 0.0, 0.0)
    nl = ti.math.vec3(0.0, 0.0, _NEAR_CLEAR)
    use_ss = 0
    if ti.static(ss_enabled):
        if tri_screen[ts, prim, 9] > 0.5:
            use_ss = 1
            for i in ti.static(range(3)):
                sm[0, i] = tri_screen[ts, prim, i]
                sm[1, i] = tri_screen[ts, prim, 3 + i]
                sm[2, i] = tri_screen[ts, prim, 6 + i]
            if ti.static(aa):
                for i in ti.static(range(3)):
                    il[i] = tri_screen[ts, prim, 10 + i]
            if tri_screen[ts, prim, 9] > 1.5:
                nl = _near_line(sm, vm, cam_o,
                                _forward_axis(f, cam_origin, screen_point),
                                near_clip)
    return use_ss, sm, vm, cam_o, il, nl


@ti.func
def _ss_pixel(px, py, sm, vm, cam_o, il, nl, aa: ti.template(),
              store_exact: ti.template(), f, cam_origin: ti.template(),
              cam_proj: ti.template()):
    """Screen-space test of one pixel against the pre-projected triangle.

    Edge functions give the 2D barycentric weights; the perspective-correct 3D
    weights are ``w_i = (E_i / d_i) / sum(E_j / d_j)`` (the shared screen-space
    area cancels). The 3D hit point ``H = sum w_i V_i`` gives the exact ray
    distance ``t = |H - cam_o|`` and barycentrics, matching Moller-Trumbore to
    float epsilon. Returns ``(ok, t, w1, w2, cov)``.

    ``d_i`` is each vertex's projection divisor, ``spread * D + persp * z``
    (``algan.rendering.projection``): its depth for a pinhole camera, one
    constant for an orthographic one, whose weights are then the affine
    screen barycentrics. Off the pinhole the ray does not start at the eye
    (``cam_o``) but at :func:`_camera_point`, and ``t`` is measured from there.

    Analytic coverage (``aa``, DESIGN_analytic_aa.md ss3). ``e_i`` is the cross
    product of edge ``i`` with the vector to the pixel centre, so ``e_i / |edge
    i|`` is the perpendicular distance in pixels, signed by the winding.  The
    winding is ``sign(e0 + e1 + e2)``, which is twice the signed screen area and
    therefore constant over the triangle -- NOT ``sign(s)``, whose perspective
    weights only agree for interior points.  Coverage is the product of the
    three clamped edge distances: exact whenever a single edge crosses the
    pixel, which is the overwhelmingly common case, and -- the property the
    seam rule depends on -- the two coverages either side of a shared edge sum
    to exactly 1.

    The hit is then accepted whenever coverage is non-zero, so the pixel CENTRE
    may lie outside the triangle.  ``t`` and the barycentrics are therefore
    evaluated at the CENTROID OF THE OWNED SAMPLES instead -- a point that is
    inside the triangle by construction, so neither the depth nor the
    barycentrics are ever extrapolations of the plane past its own edges (see
    the comment at that step for the silhouette-ordering failure this exists to
    prevent).  A fully covered fragment's centroid IS the pixel centre, so only
    partially covering ones move.  The returned barycentrics are still projected
    onto the simplex, because they go on to index colors, normals and UVs and
    must not sample outside the triangle.

    ``nl`` is the near clip plane's cut across the triangle
    (:func:`_ss_setup`): what lies on its far side is all the triangle there
    is. A sample past the line is no sample of it, and on the pixels the line
    crosses the exact areas are those of the clipped triangle. The kept part
    is convex, so the owned-sample centroid argument above still holds.
    """
    qx = ti.cast(px, ti.f32) + 0.5
    qy = ti.cast(py, ti.f32) + 0.5
    # The pixel centre's signed distance from the near cut, in pixels.
    gc = nl[0] * qx + nl[1] * qy + nl[2]
    sx0, sx1, sx2 = sm[0, 0], sm[0, 1], sm[0, 2]
    sy0, sy1, sy2 = sm[1, 0], sm[1, 1], sm[1, 2]
    e0 = (sx2 - sx1) * (qy - sy1) - (sy2 - sy1) * (qx - sx1)
    e1 = (sx0 - sx2) * (qy - sy2) - (sy0 - sy2) * (qx - sx2)
    e2 = (sx1 - sx0) * (qy - sy0) - (sy1 - sy0) * (qx - sx0)
    n0 = e0 * sm[2, 0]
    n1 = e1 * sm[2, 1]
    n2 = e2 * sm[2, 2]
    s = n0 + n1 + n2
    ok = 0
    t = 0.0
    w1 = 0.0
    w2 = 0.0
    cov = 1.0
    msk = _AA_MASK_ALL
    if ti.abs(s) > 1e-30:
        inv = 1.0 / s
        b0 = n0 * inv
        b1 = n1 * inv
        b2 = n2 * inv
        accept = ((b1 >= -BARYCENTRIC_EPSILON) and (b2 >= -BARYCENTRIC_EPSILON)
                  and (b0 >= -BARYCENTRIC_EPSILON) and (gc >= 0.0))
        c = 1.0
        m = _AA_MASK_ALL
        if ti.static(aa):
            # Cheap conservative reject in float first: no sample can be inside
            # unless every edge is within half a pixel diagonal of the centre.
            # This keeps the exact path off the overwhelming majority of
            # candidates, which are plain misses. Two-sided because the winding
            # is not known yet -- it comes from the exact integer area below,
            # the float one being untrustworthy -- so reject only when the
            # centre is far outside under BOTH readings.
            d0 = e0 * il[0]
            d1 = e1 * il[1]
            d2 = e2 * il[2]
            accept = (
                ((d0 > -0.7072) and (d1 > -0.7072) and (d2 > -0.7072))
                or ((d0 < 0.7072) and (d1 < 0.7072) and (d2 < 0.7072)))
            # A pixel wholly past the near cut has nothing of the triangle.
            accept = accept and (gc > -0.7072)
            # Whether the near cut crosses the pixel at all; when it does,
            # ``nmask`` holds the samples it keeps.
            near_live = gc < 0.7072
            nmask = _AA_MASK_ALL
            # Whether the cut took area from this fragment (the WRITE kernel's
            # exact lanes decide; see _AA_NEAR_CUT_BIT).
            trimmed = 0
            if near_live:
                nmask = 0
                for k in ti.static(range(_AA_NUM_SAMPLES)):
                    if (gc + nl[0] * ti.static(_AA_SAMPLES[k][0] / _AA_FIXED_SCALE)
                            + nl[1] * ti.static(_AA_SAMPLES[k][1]
                                                / _AA_FIXED_SCALE)) >= 0.0:
                        nmask |= 1 << k
            if accept:
                # Keep the exact path for every accepted candidate. A
                # full-interior branch adds enough edge-support arithmetic to
                # regress dense moving meshes despite skipping sample tests.
                if ti.static(aa):
                    # EXACT fixed-point coverage (DESIGN_analytic_aa.md ss15).
                    #
                    # Snap the projected vertices to a 1/4096-pixel integer lattice
                    # and evaluate the edge functions in int64. Two triangles that
                    # share an edge traverse it in opposite directions, and in
                    # exact integer arithmetic their edge functions are then exact
                    # negatives -- because E = D x (Q - V1) and the reversed edge
                    # gives -D x (Q - V2) = -(D x (Q - V1)) once D x D = 0 drops
                    # out. In FLOAT they are merely near-negatives, which is what
                    # made a sample lying on a shared edge belong to whichever
                    # triangle rounding happened to favour, and forced a choice
                    # between a silhouette halo and speckle along every shared edge.
                    #
                    # Exactness lets the classic top-left fill rule settle it: a
                    # sample exactly on an edge counts only for the triangle whose
                    # traversal of that edge runs "down", or runs "left" when
                    # horizontal. The neighbour traverses it the other way, so
                    # exactly one of the pair takes the sample -- the masks
                    # partition the pixel with no epsilon anywhere, and the same
                    # mask can serve both as what a fragment claims and as what it
                    # occludes.
                    #
                    # This relies on adjacent triangles carrying bit-identical
                    # coordinates for a shared vertex. They do: the merged soup
                    # stores each triangle's own copy, but from the same source
                    # vertex, and the projection is one elementwise torch
                    # expression. If that ever stopped holding, the failure is
                    # graceful -- back to the float-era ambiguity on those edges.
                    fx0 = ti.cast(ti.round(sx0 * _AA_FIXED_SCALE), ti.i64)
                    fx1 = ti.cast(ti.round(sx1 * _AA_FIXED_SCALE), ti.i64)
                    fx2 = ti.cast(ti.round(sx2 * _AA_FIXED_SCALE), ti.i64)
                    fy0 = ti.cast(ti.round(sy0 * _AA_FIXED_SCALE), ti.i64)
                    fy1 = ti.cast(ti.round(sy1 * _AA_FIXED_SCALE), ti.i64)
                    fy2 = ti.cast(ti.round(sy2 * _AA_FIXED_SCALE), ti.i64)
                    qxf = ((ti.cast(px, ti.i64) << _AA_FIXED_SHIFT)
                           + _AA_FIXED_HALF)
                    qyf = ((ti.cast(py, ti.i64) << _AA_FIXED_SHIFT)
                           + _AA_FIXED_HALF)
                    ex0 = fx2 - fx1
                    ey0 = fy2 - fy1
                    ex1 = fx0 - fx2
                    ey1 = fy0 - fy2
                    ex2 = fx1 - fx0
                    ey2 = fy1 - fy0
                    r0 = ex0 * (qyf - fy1) - ey0 * (qxf - fx1)
                    r1 = ex1 * (qyf - fy2) - ey1 * (qxf - fx2)
                    r2 = ex2 * (qyf - fy0) - ey2 * (qxf - fx0)
                # Orientation from the EXACT integer sum -- which is twice the
                # lattice signed area, and is the ONLY safe source for it. The
                # float sum is the same quantity but formed from three large
                # cancelling products, so its sign is unreliable for the small
                # thin triangles a dense mesh is made of. Two neighbours
                # disagreeing about their winding is exactly what stops the
                # fill rule partitioning a shared edge, and it was measured
                # doing so (scratch fill-rule harness: every double-claimed
                # sample traced back to an orientation disagreement).
                    area2 = r0 + r1 + r2
                    oi = ti.i64(1)
                    if area2 < 0:
                        oi = ti.i64(-1)
                    ec0 = oi * r0
                    ec1 = oi * r1
                    ec2 = oi * r2
                    gx0 = oi * ex0
                    gy0 = oi * ey0
                    gx1 = oi * ex1
                    gy1 = oi * ey1
                    gx2 = oi * ex2
                    gy2 = oi * ey2
                # Top-left rule as a +1 bias: a sample exactly on an owned edge
                # (q == 0) then tests strictly positive, and on a disowned one
                # it does not.
                # NOTE the names: b0/b1/b2 above are the float barycentrics and
                # must not be touched -- assigning an integer into them would
                # keep them f32, silently dragging every "exact" value below
                # into float and corrupting the barycentrics on the way out.
                    tl0 = ti.i64(0)
                    if (gy0 > 0) or ((gy0 == 0) and (gx0 < 0)):
                        tl0 = ti.i64(1)
                    tl1 = ti.i64(0)
                    if (gy1 > 0) or ((gy1 == 0) and (gx1 < 0)):
                        tl1 = ti.i64(1)
                    tl2 = ti.i64(0)
                    if (gy2 > 0) or ((gy2 == 0) and (gx2 < 0)):
                        tl2 = ti.i64(1)
                    m = 0
                    best_k = 0
                    best_q = ti.i64(-_AA_Q_INF)
                    # Running centroid of the OWNED samples, in lattice units
                    # from the pixel centre. It is what the fragment's depth and
                    # barycentrics are evaluated at (see below); zero for a fully
                    # covered pixel, because the sample pattern is centred.
                    sox = 0.0
                    soy = 0.0
                    nsm = 0
                    for k in ti.static(range(len(_AA_SAMPLES))):  # noqa: B007
                        ox = ti.static(_AA_SAMPLES[k][0])
                        oy = ti.static(_AA_SAMPLES[k][1])
                        q0 = ec0 + gx0 * oy - gy0 * ox + tl0
                        q1 = ec1 + gx1 * oy - gy1 * ox + tl1
                        q2 = ec2 + gx2 * oy - gy2 * ox + tl2
                        qq = ti.min(q0, ti.min(q1, q2))
                        if (qq > 0) and ((nmask & ti.static(1 << k)) != 0):
                            m |= 1 << k
                            sox += ti.static(float(ox))
                            soy += ti.static(float(oy))
                            nsm += 1
                    # Only the sliver policies need to know which sample the
                    # triangle came closest to.
                        if ti.static(_sliver_mode(aa) != _AA_SLIVER_DROP):
                            if (qq > best_q) and (
                                    (nmask & ti.static(1 << k)) != 0):
                                best_q = qq
                                best_k = k
                    if area2 == 0:
                    # Foreshortened to zero area on the lattice. Its edge
                    # functions are all zero, so the top-left bias alone would
                    # otherwise let it claim the entire pixel. Clearing the set
                    # hands it to the sample-less policy below, which under the
                    # default drops it -- an error bounded by one lattice unit
                    # (1/4096 px), not a hole.
                        m = 0
                        sox = 0.0
                        soy = 0.0
                        nsm = 0
                    c = (ti.cast(_popcount_samples(m), ti.f32)
                         * _AA_SAMPLE_WEIGHT)
                    if ti.static(_tri_run(aa)):
                    # RUN-CORRECTED payload (DESIGN_analytic_aa_v2.md ss4.1).
                    # MASKED fragments leave here with their SAMPLED coverage:
                    # the keep decisions key on it in count and write alike,
                    # and only the WRITE kernel -- after its keep/z/alpha
                    # culls, on actually-stored fragments -- computes the
                    # exact lane (clipping per CANDIDATE here cost count +36%
                    # / write +42% device on the meshes A/B).
                    # Sample-less DONORS are the exception: their acceptance
                    # IS "exact area > 0", so both kernels clip them, behind
                    # the one-sided oriented reject.
                    # the mask, fill rule and owned-sample centroid stay
                    # exactly as shipped -- ownership and occlusion never
                    # change -- while frag_cov carries the EXACT clipped area
                    # for the sheet compaction to sum.
                    #
                    #   * A FULL mask whose edges all clear the half-diagonal
                    #     is exactly 1.0 with no clip at all (the hot path);
                    #     otherwise the exact area, snapped to 1.0 from
                    #     0.9999 so interior fragments stay bit-clean for the
                    #     cov-keyed full-coverage gates.
                    #   * A PARTIAL mask takes the exact area, falling back to
                    #     the sampled count if the float clip degenerates to
                    #     zero against the lattice (a mask-owning fragment
                    #     must never be dropped -- its samples would show the
                    #     background through the surface).
                    #   * An EMPTY mask (a sliver, including the
                    #     lattice-degenerate area2 == 0 case) is EMITTED as an
                    #     area donor: exact area, depth and barycentrics at
                    #     the centroid of (triangle n pixel), which is inside
                    #     the triangle, so the ss15 ordering argument holds.
                        if (m == _AA_MASK_ALL) and ti.static(store_exact):
                            ffl = 1.0
                            if oi < 0:
                                ffl = -1.0
                            clear = ((ffl * d0 >= 0.7072)
                                     and (ffl * d1 >= 0.7072)
                                     and (ffl * d2 >= 0.7072))
                            if clear:
                                c = 1.0
                            else:
                                c = _pixel_clip_area(
                                    ti.math.vec3(
                                        sx0 - qx, sx1 - qx, sx2 - qx),
                                    ti.math.vec3(
                                        sy0 - qy, sy1 - qy, sy2 - qy))
                            if near_live:
                                # Every sample kept, but the cut still
                                # crosses the square somewhere.
                                kept = _near_clipped_pixel_area(
                                    sx0 - qx, sy0 - qy, sx1 - qx, sy1 - qy,
                                    sx2 - qx, sy2 - qy, clear, nl, gc)
                                if c - kept > _NEAR_TRIM_AREA:
                                    trimmed = 1
                                c = kept
                            if (c >= 0.9999) or (c <= 0.0):
                                c = 1.0
                        elif (m != 0) and ti.static(store_exact):
                            # Exact area of (triangle n pixel), stored for the
                            # sheet claims, cheapest-first: a microtriangle FULLY
                            # INSIDE the square (the sub-pixel-diced majority)
                            # is a bare shoelace; a fragment cut by ONE edge
                            # with the other two clear (the big-triangle
                            # silhouette case) is the half-plane closed form
                            # riding the edge distances already computed; only
                            # genuine corner/straddle fragments pay the clip.
                            ffl = 1.0
                            if oi < 0:
                                ffl = -1.0
                            od0 = ffl * d0
                            od1 = ffl * d1
                            od2 = ffl * d2
                            ca = -1.0
                            ax_ = sx0 - qx
                            ay_ = sy0 - qy
                            bx_ = sx1 - qx
                            by_ = sy1 - qy
                            cx2 = sx2 - qx
                            cy2 = sy2 - qy
                            if ((ti.abs(ax_) <= 0.5) and (ti.abs(ay_) <= 0.5)
                                    and (ti.abs(bx_) <= 0.5)
                                    and (ti.abs(by_) <= 0.5)
                                    and (ti.abs(cx2) <= 0.5)
                                    and (ti.abs(cy2) <= 0.5)):
                                ca = 0.5 * ti.abs(
                                    (bx_ - ax_) * (cy2 - ay_)
                                    - (by_ - ay_) * (cx2 - ax_))
                            elif ((od0 < 0.7072) and (od1 >= 0.7072)
                                    and (od2 >= 0.7072)):
                                ca = _halfplane_clip_area(
                                    -ffl * (sy2 - sy1), ffl * (sx2 - sx1),
                                    od0)
                            elif ((od1 < 0.7072) and (od0 >= 0.7072)
                                    and (od2 >= 0.7072)):
                                ca = _halfplane_clip_area(
                                    -ffl * (sy0 - sy2), ffl * (sx0 - sx2),
                                    od1)
                            elif ((od2 < 0.7072) and (od0 >= 0.7072)
                                    and (od1 >= 0.7072)):
                                ca = _halfplane_clip_area(
                                    -ffl * (sy1 - sy0), ffl * (sx1 - sx0),
                                    od2)
                            if ca < 0.0:
                                ca = _pixel_clip_area(
                                    ti.math.vec3(ax_, bx_, cx2),
                                    ti.math.vec3(ay_, by_, cy2))
                            if near_live:
                                kept = _near_clipped_pixel_area(
                                    ax_, ay_, bx_, by_, cx2, cy2,
                                    (od0 >= 0.7072) and (od1 >= 0.7072)
                                    and (od2 >= 0.7072), nl, gc)
                                if ca - kept > _NEAR_TRIM_AREA:
                                    trimmed = 1
                                ca = kept
                            if ca > 0.0:
                                c = ca
                        elif m == 0:
                            # One-sided oriented reject BEFORE any clipping:
                            # the conservative pre-test above is two-sided
                            # (winding unknown there), so on a dense mesh most
                            # sample-less candidates reaching here are plain
                            # misses on the correct winding -- and they were
                            # the bulk of the run representation's emission
                            # cost (tri_count +36% on the meshes A/B before
                            # this gate, +9% after).
                            # (The lattice-degenerate area2 == 0 case has no
                            # trustworthy winding, so it keeps the two-sided
                            # acceptance it already passed.)
                            ffl2 = 1.0
                            if oi < 0:
                                ffl2 = -1.0
                            if (area2 == 0) or ((ffl2 * d0 > -0.7072)
                                                and (ffl2 * d1 > -0.7072)
                                                and (ffl2 * d2 > -0.7072)):
                                # BOTH count and write take the centroid form:
                                # a donor's barycentrics move to the clipped
                                # centroid, and the keep decision samples the
                                # texture ALPHA at those barycentrics -- a
                                # count-only moment-free clip left the count
                                # kernel sampling at the pixel centre instead,
                                # the two kernels' keep decisions diverged on
                                # textured prims, and the write pass left
                                # UNINITIALIZED fragment rows (the
                                # text_and_media CUDA_ERROR_ILLEGAL_ADDRESS:
                                # a float bit-pattern walked as a prim id).
                                dvx = ti.math.vec3(
                                    sx0 - qx, sx1 - qx, sx2 - qx)
                                dvy = ti.math.vec3(
                                    sy0 - qy, sy1 - qy, sy2 - qy)
                                ca = 0.0
                                cx_ = 0.0
                                cy_ = 0.0
                                if near_live:
                                    ca, cx_, cy_ = _near_cut_area(
                                        dvx, dvy, nl[0], nl[1], gc, True)
                                    if ti.static(store_exact):
                                        if (_pixel_clip_area(dvx, dvy) - ca
                                                > _NEAR_TRIM_AREA):
                                            trimmed = 1
                                else:
                                    ca, cx_, cy_ = _pixel_clip_centroid(
                                        dvx, dvy)
                                if ca > 0.0:
                                    c = ca
                                    nsm = 1
                                    sox = cx_ * ti.static(_AA_FIXED_SCALE)
                                    soy = cy_ * ti.static(_AA_FIXED_SCALE)
                # A triangle thinner than the sample spacing contains no sample,
                # so the set says nothing about it. The DEFAULT policy is to let
                # it contribute nothing, exactly as supersampling does -- sound
                # because the fill rule PARTITIONS the samples, so any sample
                # this triangle misses is contained by whichever neighbour of the
                # tiling does contain it, and dropping cannot open a hole. It
                # also measures better than every alternative on every config
                # (ss16.2), including sub-pixel rods.
                #
                # The other policies exist because they were the plausible ones:
                # give it the sample it comes closest to, weighted by how much of
                # the pixel it covers, and mark it a sliver so the resolve treats
                # that claim as provisional.
                    if ti.static(_sliver_mode(aa) != _AA_SLIVER_DROP
                                 and not _tri_run(aa)):
                        if m == 0:
                        # The weight must be the EXACT clipped area: the
                        # continuous product form spreads half a pixel past the
                        # geometry, so a silhouette rim of tiling slivers sums to
                        # a halo and dilates the whole mesh (ss15.3).
                            if ti.static(
                                    _sliver_mode(aa) == _AA_SLIVER_AREA):
                                ofl = 1.0
                                if oi < 0:
                                    ofl = -1.0
                                od0 = ofl * d0
                                od1 = ofl * d1
                                od2 = ofl * d2
                                if ((od0 > -0.5) and (od1 > -0.5)
                                        and (od2 > -0.5)):
                                    m = (1 << best_k) | _AA_SLIVER_BIT
                                    c = (
                                        ti.math.clamp(
                                            od0 + 0.5, 0.0, 1.0)
                                        * ti.math.clamp(
                                            od1 + 0.5, 0.0, 1.0)
                                        * ti.math.clamp(
                                            od2 + 0.5, 0.0, 1.0)
                                    )
                            else:
                                ca = _pixel_clip_area(
                                    ti.math.vec3(
                                        sx0 - qx, sx1 - qx, sx2 - qx),
                                    ti.math.vec3(
                                        sy0 - qy, sy1 - qy, sy2 - qy))
                                if ca > 0.0:
                                # exact and exact_occ coincide under per-sample
                                # transmittance: there is no separate occlusion
                                # set to opt into, since attenuating a sample IS
                                # occluding it (ss18).
                                    m = (1 << best_k) | _AA_SLIVER_BIT
                                    c = ca
                    if ti.static(_tri_run(aa)):
                        # Acceptance widens to "any exact area": a sample-less
                        # sliver is an area donor now, not a drop.
                        accept = c > 0.0
                        if oi < 0:
                            m |= _AA_BACKFACE_BIT
                    else:
                        accept = ((m & _AA_MASK_ALL) != 0) and (c > 0.0)
                        if oi < 0:
                            m |= _AA_BACKFACE_BIT
                    if trimmed != 0:
                        m |= _AA_NEAR_CUT_BIT
                # Re-evaluate the fragment AT THE CENTROID OF THE SAMPLES IT
                # OWNS rather than at the pixel centre, which is the only point
                # the fragment is known to be visible at.
                #
                # The pixel centre lies OUTSIDE a partially covering triangle
                # roughly half the time, and there the plane intersection is an
                # EXTRAPOLATION -- past the geometry, on the wrong side of every
                # silhouette edge. Two faces meeting at a silhouette (any closed
                # mesh: their near and far sheets share that edge exactly) then
                # extrapolate in opposite directions, so beyond the edge the far
                # sheet's extrapolated distance is the SMALLER one and the rim
                # pixel sorts the two faces back to front. The far face won the
                # per-sample transmittance, occluded the near one entirely, and
                # the silhouette was drawn in the color of the geometry behind
                # it -- brighter than the surface it outlines whenever the far
                # sheet is the better lit one. Whether a given rim pixel's centre
                # falls inside or outside is pure sub-pixel phase, which is why
                # it appeared on one edge of an octahedron at one size and not at
                # another.
                #
                # The owned samples are inside the triangle by construction and a
                # triangle is convex, so their centroid is too: no extrapolation,
                # and two faces sharing a silhouette edge order by true depth.
                # The sample pattern sums to zero, so a fully covered fragment
                # re-evaluates at the pixel centre exactly and is unchanged --
                # only rim fragments move, by less than half a pixel.
                    if nsm > 0:
                        inv_n = 1.0 / ti.cast(nsm, ti.f32)
                        dqx = sox * inv_n * ti.static(1.0 / _AA_FIXED_SCALE)
                        dqy = soy * inv_n * ti.static(1.0 / _AA_FIXED_SCALE)
                        if (dqx != 0.0) or (dqy != 0.0):
                            # Edge functions are affine in the sample point, so
                            # the shift is an update rather than a re-derivation.
                            ce0 = e0 + (sx2 - sx1) * dqy - (sy2 - sy1) * dqx
                            ce1 = e1 + (sx0 - sx2) * dqy - (sy0 - sy2) * dqx
                            ce2 = e2 + (sx1 - sx0) * dqy - (sy1 - sy0) * dqx
                            cn0 = ce0 * sm[2, 0]
                            cn1 = ce1 * sm[2, 1]
                            cn2 = ce2 * sm[2, 2]
                            cs = cn0 + cn1 + cn2
                            if ti.abs(cs) > 1e-30:
                                cinv = 1.0 / cs
                                b0 = cn0 * cinv
                                b1 = cn1 * cinv
                                b2 = cn2 * cinv
        if accept:
            v0 = ti.math.vec3(vm[0, 0], vm[0, 1], vm[0, 2])
            v1 = ti.math.vec3(vm[1, 0], vm[1, 1], vm[1, 2])
            v2 = ti.math.vec3(vm[2, 0], vm[2, 1], vm[2, 2])
            hp = b0 * v0 + b1 * v1 + b2 * v2
            tt = (hp - cam_o).norm()
            if ti.static(_projected(cam_proj)):
                if cam_proj[f, 1] != 0.0:
                    tt = (hp - _camera_point(f, hp, cam_origin,
                                             cam_proj)).norm()
            if tt > min_hit_distance:
                ok = 1
                t = tt
                cov = c
                msk = m
                if ti.static(aa):
                    cb0 = ti.max(b0, 0.0)
                    cb1 = ti.max(b1, 0.0)
                    cb2 = ti.max(b2, 0.0)
                    bsum = cb0 + cb1 + cb2
                    if bsum > 1e-20:
                        inv_b = 1.0 / bsum
                        b1 = cb1 * inv_b
                        b2 = cb2 * inv_b
                w1 = b1
                w2 = b2
    return ok, t, w1, w2, cov, msk


@ti.func
def _raycast_pixel(px, py, f, vm, half_w, half_h,
                   cam_origin: ti.template(), screen_point: ti.template(),
                   pixel_basis_x: ti.template(), pixel_basis_y: ti.template(),
                   cam_proj: ti.template(), near_clip,
                   aa: ti.template()):
    """Per-pixel ray-cast fallback (Moller-Trumbore), used when a triangle
    straddles the camera plane so screen-space projection is invalid, and for
    an all-front triangle projected too far off-screen for the exact lattice
    (projection flag 0.5, see ``_AA_LATTICE_MAX_EXTENT``). Returns
    ``(ok, t, w1, w2, cov, msk)``; under analytic coverage ``msk`` carries the
    facing bit exactly as :func:`_ss_pixel` sets it.

    Each ray's hit also has to lie past the near clip plane (``near_clip``,
    0 for none), as the classic route's rays start on it. The part of the
    triangle past the plane is convex and in front of the camera, so the
    centroid re-cast below lands inside it.

    Under analytic coverage this does NOT report full coverage. It cannot use the
    screen-space edge functions -- the projection that would supply them is the
    thing that is invalid -- so it answers the same set-membership question
    directly instead: cast one ray per sub-pixel sample and test each against the
    triangle. That is the definition of the sample mask rather than an
    approximation of it, at the cost of N intersections on these pixels.

    This case is NOT rare, which is why it is worth the cost: any ground plane
    large enough to reach the horizon straddles the camera plane, and while it
    reported coverage 1 its edges were the only geometry in a scene left
    completely un-antialiased -- byte-identical to no AA at all, and the whole
    residual against anti_alias_level=2 on such a scene (ss19).

    A sample exactly on an edge shared by two straddling triangles (a quad's
    diagonal) is claimed by both, because the barycentric test has an epsilon
    rather than the exact fill rule the projected path uses. Per-sample
    transmittance makes that harmless: the nearer fragment takes the sample and
    the second one finds nothing left of it.
    """
    v0 = ti.math.vec3(vm[0, 0], vm[0, 1], vm[0, 2])
    v1 = ti.math.vec3(vm[1, 0], vm[1, 1], vm[1, 2])
    v2 = ti.math.vec3(vm[2, 0], vm[2, 1], vm[2, 2])
    e1 = v1 - v0
    e2 = v2 - v0
    ok = 0
    t = 0.0
    w1 = 0.0
    w2 = 0.0
    cov = 1.0
    msk = _AA_MASK_ALL
    fwd = _forward_axis(f, cam_origin, screen_point)
    ro, rd = _generate_ray(f, px, py, 0.5, 0.5, half_w, half_h,
                           cam_origin, screen_point,
                           pixel_basis_x, pixel_basis_y, cam_proj)
    pv = rd.cross(e2)
    det = e1.dot(pv)
    if ti.abs(det) > 1e-12:
        inv_det = 1.0 / det
        tvec = ro - v0
        b1 = tvec.dot(pv) * inv_det
        qv = tvec.cross(e1)
        b2 = rd.dot(qv) * inv_det
        inside = ((b1 >= -BARYCENTRIC_EPSILON)
                  and (b2 >= -BARYCENTRIC_EPSILON)
                  and (b1 + b2 <= 1.0 + BARYCENTRIC_EPSILON))
        th = e2.dot(qv) * inv_det
        if ti.static(aa):
            m = 0
            sox = 0.0
            soy = 0.0
            nsm = 0
            # Whether the near plane took a sample the triangle holds.
            cut = 0
            for k in ti.static(range(_AA_NUM_SAMPLES)):
                jx = ti.static(0.5 + _AA_SAMPLES[k][0] / _AA_FIXED_SCALE)
                jy = ti.static(0.5 + _AA_SAMPLES[k][1] / _AA_FIXED_SCALE)
                ros, rds = _generate_ray(f, px, py, jx, jy, half_w, half_h,
                                         cam_origin, screen_point,
                                         pixel_basis_x, pixel_basis_y, cam_proj)
                # THE SAME INTERSECTION THE RAY PATH USES (ss3.3). This is a
                # set-membership question -- is this sub-pixel sample inside
                # this triangle -- and it used to be answered by an inline
                # Moller-Trumbore dilated by BARYCENTRIC_EPSILON. The dilation
                # was not a fudge: with a float test and no exact tie-break, a
                # sample lying on an edge shared by two straddling triangles
                # must be erred one way, and double-claiming is harmless here
                # (per-sample transmittance gives the sample to the nearer
                # fragment) while dropping it is a crack.
                #
                # _tri_hit answers it without needing to choose. Under
                # watertight_tri the shared edge's function is computed from the
                # same two projected vertices in both triangles and comes out as
                # the exact negative, so exactly one neighbour accepts: no
                # dilation, no duplicate, no crack. With the gate off this is
                # bit-identical to the code it replaces -- the same dilated
                # Moller-Trumbore with the same epsilon and the same three
                # comparisons.
                #
                # The projected path (_ss_pixel) reaches the same place by a
                # different route, its exact fixed-point fill rule; this one
                # cannot use that, because the projection it would need is
                # precisely what straddling the camera plane invalidates.
                hit_s, _c1, _c2, ts = _tri_hit(ros, rds, v0, v1, v2)
                if hit_s and (ts > min_hit_distance):
                    if _past_near(ros, rds, ts, f, cam_origin, fwd, near_clip):
                        m |= 1 << k
                        sox += ti.static(float(_AA_SAMPLES[k][0]))
                        soy += ti.static(float(_AA_SAMPLES[k][1]))
                        nsm += 1
                    else:
                        cut = 1
            # Same rule as the projected path: a partially covering fragment is
            # re-cast through the CENTROID OF THE SAMPLES IT OWNS, so its depth
            # is a real intersection with the triangle rather than the centre
            # ray's extrapolation past its edges (which sorts the two sheets of
            # a closed mesh backwards along every silhouette). A fully covered
            # fragment's centroid is the pixel centre, so it keeps the centre
            # ray untouched.
            # ``det_hit`` is the determinant of whichever ray ``th`` ends up
            # from: the facing test below needs a ray that meets the plane in
            # FRONT of the camera.
            det_hit = det
            if (m != 0) and (m != _AA_MASK_ALL):
                inv_n = 1.0 / ti.cast(nsm, ti.f32)
                jxc = 0.5 + sox * inv_n * ti.static(1.0 / _AA_FIXED_SCALE)
                jyc = 0.5 + soy * inv_n * ti.static(1.0 / _AA_FIXED_SCALE)
                roc, rdc = _generate_ray(f, px, py, jxc, jyc, half_w, half_h,
                                         cam_origin, screen_point,
                                         pixel_basis_x, pixel_basis_y, cam_proj)
                pvc = rdc.cross(e2)
                detc = e1.dot(pvc)
                if ti.abs(detc) > 1e-12:
                    ivc = 1.0 / detc
                    tvc = roc - v0
                    qvc = tvc.cross(e1)
                    b1 = tvc.dot(pvc) * ivc
                    b2 = rdc.dot(qvc) * ivc
                    th = e2.dot(qvc) * ivc
                    det_hit = detc
            if (m != 0) and (th > min_hit_distance):
                ok = 1
                t = th
                cov = ti.cast(_popcount_samples(m), ti.f32) * _AA_SAMPLE_WEIGHT
                msk = m
                # Facing, as _ss_pixel reports it: there the bit is set when
                # the lattice area is negative, and for an all-front triangle
                # that area's sign is sign(big_d) * sign((V0 - eye) . N),
                # N = e1 x e2, big_d = (screen_point - eye) . n with
                # n = pixel_basis_x x pixel_basis_y. Here det = e1 . (rd x e2)
                # = -rd . N, and for a ray meeting the plane at th > 0,
                # sign(rd . N) = sign((V0 - eye) . N); every forward ray has
                # sign(rd . n) = sign(big_d). So the same bit is
                # det * (rd . n) > 0 -- and it stays defined for a straddler,
                # whose projected area is not.
                nvec = ti.math.vec3(
                    pixel_basis_x[f, 0], pixel_basis_x[f, 1],
                    pixel_basis_x[f, 2]).cross(ti.math.vec3(
                        pixel_basis_y[f, 0], pixel_basis_y[f, 1],
                        pixel_basis_y[f, 2]))
                if det_hit * rd.dot(nvec) > 0.0:
                    msk |= _AA_BACKFACE_BIT
                if cut != 0:
                    msk |= _AA_NEAR_CUT_BIT
                # The centre may lie outside the triangle when only some samples
                # are inside, so project its barycentrics onto the simplex
                # before they index colors, normals and UVs (as _ss_pixel does).
                cb1 = ti.max(b1, 0.0)
                cb2 = ti.max(b2, 0.0)
                cb0 = ti.max(1.0 - b1 - b2, 0.0)
                bsum = cb0 + cb1 + cb2
                if bsum > 1e-20:
                    inv_b = 1.0 / bsum
                    w1 = cb1 * inv_b
                    w2 = cb2 * inv_b
        else:
            if inside and (th > min_hit_distance) and _past_near(
                    ro, rd, th, f, cam_origin, fwd, near_clip):
                ok = 1
                t = th
                w1 = b1
                w2 = b2
    return ok, t, w1, w2, cov, msk


@ti.func
def _pair_pixel(prim, f, x0, y0, bw, bh, off, j,
                time_start, width, height, tile_start, tile_pixels,
                half_w, half_h, use_ss, sm, vm, cam_o, il, nl, near_clip,
                cam_origin: ti.template(), screen_point: ti.template(),
                pixel_basis_x: ti.template(), pixel_basis_y: ti.template(),
                cam_proj: ti.template(),
                aa: ti.template(), store_exact: ti.template()):
    """Test candidate ``off + j`` of a pair at its pixel center, dispatching to
    the screen-space path (``use_ss``) or the ray-cast fallback. Returns
    ``(ok, local_pixel, t, w1, w2, cov, msk)``; both paths supply analytic
    coverage under ``aa``, the fallback by casting one ray per sub-pixel sample
    (see :func:`_raycast_pixel`). ``local_pixel`` is the candidate's pixel
    whenever it lies in the tile, hit or not (the write kernels' miss record
    needs it; see ``_WRITE_MISS_T_BITS``), and 0 otherwise.
    """
    ok = 0
    lp = 0
    t = 0.0
    w1 = 0.0
    w2 = 0.0
    cov = 1.0
    msk = _AA_MASK_ALL
    o = off + j
    if o < bw * bh:
        px = x0 + o % bw
        py = y0 + o // bw
        lpi = ((f - time_start) * (width * height) + py * width + px
               - tile_start)
        if (lpi >= 0) and (lpi < tile_pixels):
            lp = lpi
            hit = 0
            th = 0.0
            b1 = 0.0
            b2 = 0.0
            cv = 1.0
            mk = _AA_MASK_ALL
            if use_ss != 0:
                hit, th, b1, b2, cv, mk = _ss_pixel(
                    px, py, sm, vm, cam_o, il, nl, aa, store_exact, f,
                    cam_origin, cam_proj)
            else:
                hit, th, b1, b2, cv, mk = _raycast_pixel(
                    px, py, f, vm, half_w, half_h, cam_origin, screen_point,
                    pixel_basis_x, pixel_basis_y, cam_proj, near_clip, aa)
            if hit != 0:
                ok = 1
                t = th
                w1 = b1
                w2 = b2
                cov = cv
                msk = mk
    return ok, lp, t, w1, w2, cov, msk


@ti.func
def _circuit_near_cover(f, px, py, n, center, half_w, half_h,
                        cam_origin: ti.template(), screen_point: ti.template(),
                        pixel_basis_x: ti.template(),
                        pixel_basis_y: ti.template(),
                        cam_proj: ti.template(), near_clip,
                        aa: ti.template()):
    """How much of a pixel's view of a circuit's plane the near clip keeps.

    A camera ray through screen offset ``L`` (``screen_point + L`` on the
    screen) meets the plane ``(p - center) . n = 0`` at the parameter
    ``N / M`` along its unnormalized direction, with ``N = (center - eye) . n
    - spread * (L . n)`` and ``M = (screen_point - eye) . n + persp * (L .
    n)`` (``_generate_ray``; a pinhole is ``persp = 1, spread = 0``), and at a
    depth of ``D * N / M`` along the forward axis, ``D`` the eye's distance
    from the screen. Both are affine on screen, so the plane is kept where
    ``D * N - near * M`` has the sign of ``M``: a half-plane, whose exact
    share of the pixel the box filter of :func:`_boundary_coverage` measures
    under ``aa`` (the circuit's own boundary filter). Without ``aa`` the
    centre alone decides.
    """
    eye = ti.math.vec3(cam_origin[f, 0], cam_origin[f, 1], cam_origin[f, 2])
    sp = ti.math.vec3(screen_point[f, 0], screen_point[f, 1],
                      screen_point[f, 2])
    bx = ti.math.vec3(pixel_basis_x[f, 0], pixel_basis_x[f, 1],
                      pixel_basis_x[f, 2])
    by = ti.math.vec3(pixel_basis_y[f, 0], pixel_basis_y[f, 1],
                      pixel_basis_y[f, 2])
    persp = 1.0
    spread = 0.0
    if ti.static(_projected(cam_proj)):
        if cam_proj[f, 1] != 0.0:
            persp = cam_proj[f, 0]
            spread = cam_proj[f, 1]
    u = (ti.cast(px, ti.f32) + 0.5 - half_w) / half_h
    v = (ti.cast(py, ti.f32) + 0.5 - half_h) / half_h
    bxn = bx.dot(n)
    byn = by.dot(n)
    ln = u * bxn + v * byn
    big_d = (sp - eye).norm()
    mm = (sp - eye).dot(n) + persp * ln
    h = big_d * ((center - eye).dot(n) - spread * ln) - near_clip * mm
    sgn = 1.0
    if mm < 0.0:
        sgn = -1.0
    # Gradient of the kept-side function per screen pixel.
    k = -sgn * (big_d * spread + near_clip * persp) / half_h
    gx = k * bxn
    gy = k * byn
    gl = ti.sqrt(gx * gx + gy * gy)
    hc = sgn * h
    c = 0.0
    if hc >= 0.0:
        c = 1.0
    if ti.static(aa):
        if gl > 0.0:
            c = _boundary_coverage(gx, gy, hc / gl, aa)
    return c


@ti.func
def _bez_pixel_hit(circuit, f, px, py, half_w, half_h,
                   cam_origin: ti.template(), screen_point: ti.template(),
                   pixel_basis_x: ti.template(), pixel_basis_y: ti.template(),
                   cam_proj: ti.template(),
                   pixel_world_scale: ti.template(),
                   circuit_meta: ti.template(), circuit_colors: ti.template(),
                   edges_2d: ti.template(), edge_accel: ti.template(),
                   aa: ti.template(), aa_min_half_width: ti.f32,
                   border_aa: ti.template(), near_clip):
    """Exact primary camera-ray/circuit hit for one known pixel.

    Returns ``(ok, t, u, v, border_frac, cov)``.  ``cov`` is 1.0 unless the
    compile-time ``aa`` template selects analytic coverage, in which case it is
    the fraction of the pixel square the circuit's drawn region covers, and the
    hit is accepted whenever that fraction is non-zero -- including pixels whose
    CENTRE is outside the circuit but whose square is not.  ``border_frac`` is
    the share of that covered area lying in the border band: 0 or 1 unless the
    ``border_aa`` template is set, where it is continuous across the border's
    inner edge.

    Analytic coverage (see DESIGN_analytic_aa.md ss4).  ``_bezier_point_metrics``
    returns the distance to the nearest outline segment plus a crossing parity,
    so ``d = +/- sqrt(min_dist_sq)`` is a signed distance (positive inside) in
    plane units.  The filter is a box in SCREEN pixels, so under the
    anisotropic coverage (ss22) the query is made on screen instead: the
    plane's own projection at the pixel (:func:`_circuit_screen_map`) carries
    the outline there, and the distance, its direction and the wedge's walls
    come back in ``pixel_size`` units of screen -- the same as the plane's where
    the plane faces the camera, squeezed along its foreshortened direction
    where it is tilted.  Widths -- the stroke, the hairline dilation -- stay in
    the plane, ``pixel_size`` multiples of it, so a stroke painted on a tilted
    plane foreshortens with it (:func:`_segment_width_scale` carries each onto
    the screen).  The drawn region is

        filled:    d > -outer_w        (outer_w = min_half_width, or half the
                                        stroke width on a centred border)
        unfilled:  |d| < border_w / 2                     (a band)

    and its coverage is the box filter ``clamp(distance_to_boundary + 0.5, 0, 1)``
    -- exact for a straight boundary crossing the pixel, which after flattening
    every boundary is.  The band form fades a sub-pixel-wide stroke by its width
    instead of dilating it, and both forms reach exactly 1 half a pixel inside,
    which is what lets the emission's opaque truncation keep culling.

    A filled circuit's border is an INNER boundary at ``d = inner_w`` cutting
    the same drawn region in two (:func:`_circuit_point_region`), so it needs its
    own box filter: the coverage of the fill-only part is subtracted from the
    total, and the remainder is the border's area share.  Without that the outer
    silhouette resolves continuously while the border/fill edge -- which can be
    the only visible edge, e.g. an outlined glyph over an invisible fill --
    stays a hard per-pixel classification.

    The near clip plane (``near_clip``, 0 for none) cuts the plane along a
    straight line on screen (:func:`_circuit_near_cover`); its share of the
    pixel multiplies the coverage, which is exact wherever only one of the
    two boundaries crosses the pixel.
    """
    ok = 0
    t = 0.0
    u = 0.0
    v = 0.0
    border_frac = 0.0
    cov = 1.0
    ro, rd = _generate_ray(f, px, py, 0.5, 0.5, half_w, half_h,
                           cam_origin, screen_point,
                           pixel_basis_x, pixel_basis_y, cam_proj)
    tm = f % circuit_meta.shape[0]
    n = ti.math.vec3(circuit_meta[tm, circuit, _M_NORMAL],
                     circuit_meta[tm, circuit, _M_NORMAL + 1],
                     circuit_meta[tm, circuit, _M_NORMAL + 2])
    denom = rd.dot(n)
    if ti.abs(denom) > 1e-9:
        center = ti.math.vec3(circuit_meta[tm, circuit, _M_CENTER],
                              circuit_meta[tm, circuit, _M_CENTER + 1],
                              circuit_meta[tm, circuit, _M_CENTER + 2])
        th = (center - ro).dot(n) / denom
        if th > min_hit_distance:
            hit = ro + th * rd - center
            bu = ti.math.vec3(circuit_meta[tm, circuit, _M_BASIS_U],
                              circuit_meta[tm, circuit, _M_BASIS_U + 1],
                              circuit_meta[tm, circuit, _M_BASIS_U + 2])
            bv = ti.math.vec3(circuit_meta[tm, circuit, _M_BASIS_V],
                              circuit_meta[tm, circuit, _M_BASIS_V + 1],
                              circuit_meta[tm, circuit, _M_BASIS_V + 2])
            uu = hit.dot(bu)
            vv = hit.dot(bv)
            # th is the slant range; pixel_world_scale wants perpendicular
            # depth (see _axis_cos). Off the pinhole cam_proj column 2 is the
            # part of the pixel's world size that does not grow with depth:
            # all of it for an orthographic camera.
            pixel_size = (pixel_world_scale[f] * th
                          * _axis_cos(f, ro, rd, screen_point, cam_origin,
                                      cam_proj))
            if ti.static(_projected(cam_proj)):
                pixel_size = cam_proj[f, 2] + pixel_size
            border_w = ti.abs(
                circuit_meta[tm, circuit, _M_BORDER_W]) * pixel_size
            outline_w = 0.6 * pixel_size
            if ti.static(aa):
                outline_w = aa_min_half_width * pixel_size
            filled = circuit_meta[tm, circuit, _M_FILLED] > 0.5
            centered = circuit_meta[tm, circuit, _M_BORDER_CENTERED] > 0.5
            # The two boundaries of the drawn region, in plane units. Which
            # distances they sit at is the whole of the inward/centred split;
            # every filter below is written against these rather than against
            # ``border_w`` directly.
            outer_w = _circuit_outer_dilation(border_w, outline_w, filled,
                                              centered)
            inner_w = _circuit_inner_distance(border_w, filled, centered)
            query_radius = _circuit_query_radius(border_w, outline_w, filled,
                                                 centered)
            # The plane's map onto the screen here, per ``pixel_size``
            # (:func:`_circuit_screen_map`), and the nearest-edge query's reach
            # measured there; both are read only by the anisotropic coverage.
            m00 = 1.0
            m01 = 0.0
            m10 = 0.0
            m11 = 1.0
            screen_reach = 0.0
            if ti.static(aa or border_aa):
                # The filter reaches half a pixel past each drawn boundary in
                # any direction, so the nearest-edge query must too -- a pixel
                # whose centre is OUTSIDE now needs a real distance, where the
                # classic path was content with "no edge within the radius".
                if ti.static(_bez_anisotropic()):
                    # Half a pixel ON SCREEN, past a drawn region whose widths
                    # are the plane's and reach at most ``s_max`` times as far
                    # there. The query ranks and accepts edges by that screen
                    # distance; the plane radius only has to hold everything it
                    # can see, ``1 / s_min`` farther along a tilted plane's
                    # foreshortened direction.
                    m00, m01, m10, m11 = _circuit_screen_map(
                        f, ro, rd, bu, bv, screen_point, pixel_basis_x,
                        pixel_basis_y, cam_origin, cam_proj)
                    s_min, s_max = _screen_scales(m00, m01, m10, m11)
                    screen_reach = (query_radius * s_max
                                    + _AA_FILTER_RADIUS * pixel_size)
                    query_radius = screen_reach / ti.max(s_min,
                                                         _AA_MIN_SCREEN_SCALE)
                else:
                    query_radius += _AA_FILTER_RADIUS * pixel_size
            te = f % edges_2d.shape[0]
            (crossings, min_dist_sq, ccu, ccv, e1x, e1y, sg1,
             sec_dist_sq, scu, scv, e2x, e2y, sg2) = _bezier_point_metrics_in(
                circuit, te, uu, vv, query_radius,
                circuit_meta.shape[1], edges_2d, edge_accel,
                ti.static(bool(aa or border_aa) and _bez_anisotropic()),
                m00, m01, m10, m11, screen_reach, _AA_RANK_TIE * pixel_size)
            inside, is_border = _circuit_point_region(
                border_w, outline_w, filled, centered, crossings, min_dist_sq)
            bf = 1.0 if is_border else 0.0
            c = 1.0
            if ti.static(aa or border_aa):
                inv_px = 1.0 / ti.max(pixel_size, 1e-30)
                # Signed distance, positive inside the outline. min_dist_sq is
                # left at 1e30 when no edge is within the query radius, which
                # correctly reads as "deep inside" / "far outside".
                d = ti.sqrt(ti.min(min_dist_sq, 1e30))
                if (crossings % 2) == 0:
                    d = -d
                # Direction in which the signed distance INCREASES, i.e. into the
                # shape. The closest-point vector runs from the query to the
                # outline, so it points out of the shape from inside and into it
                # from outside; negating it in the first case makes one gradient.
                gu = ccu
                gv = ccv
                if d > 0.0:
                    gu = -gu
                    gv = -gv
                # ``wk`` turns a width laid in the plane into the units the
                # distance is in. The anisotropic query reports the boundary ON
                # SCREEN -- its distance and direction, in ``pixel_size`` units
                # -- so the box filter and the half-plane area read the screen
                # directly, and only the widths, which are the plane's, need
                # converting: by how far the nearest segment's normal is scaled
                # there (:func:`_segment_width_scale`). Measured in the plane
                # instead (the other arm), a plane tilted away from the camera
                # got a filter narrower than a pixel along its foreshortened
                # direction, and a thin stroke on it beaded as it crossed pixel
                # rows (ss22).
                wk = 1.0
                if ti.static(_bez_anisotropic()):
                    wk = _segment_width_scale(m00, m01, m10, m11, e1x, e1y)
                    # A centre ON the boundary -- a stroke's own path, say --
                    # has a closest-point vector of rounding noise; the nearest
                    # segment's own perpendicular orients its area instead.
                    on = _AA_ON_BOUNDARY * pixel_size
                    if gu * gu + gv * gv < on * on:
                        gu = -e1y
                        gv = e1x
                signed = d + outer_w * wk
                # Which way the DRAWN region lies from the active boundary. For a
                # filled circuit that is always "deeper in", but an unfilled one
                # is a band with a wall on each side, and past its middle the
                # nearer wall is the inner one, whose covered side is outward.
                if not filled:
                    # Unfilled: the drawn band is bounded on BOTH sides, and
                    # once it is thinner than a pixel both walls cut the SAME
                    # pixel -- so reading only the nearer one (which this did)
                    # cannot express the band at all. It integrated to
                    # ``(w/2 + 1/2)^2`` instead of ``w``: a 0.30 px stroke, which
                    # is what a Manim ``Surface``'s default stroke_width comes
                    # to, was drawn with 1.40x its ink, and an axis-aligned one
                    # swung between 1.00x and 2.19x with nothing but its
                    # sub-pixel phase (``benchmarks/_aa_thin_stroke_sweep.py``).
                    # The error closes at w = 1, where the one-wall ramp happens
                    # to integrate to w exactly, which is why only sub-pixel
                    # strokes ever showed it.
                    #
                    # The two walls are ANTIPARALLEL, so their half-planes cover
                    # the whole pixel between them (their signed distances sum
                    # to the width, which is never negative) and the slab's area
                    # is exactly ``cov(near) + cov(far) - 1`` -- at any angle,
                    # and for either coverage form, since it asks nothing of
                    # ``_boundary_coverage`` beyond what it already returns.
                    half = 0.5 * border_w * wk
                    c = ti.math.clamp(
                        _boundary_coverage(-gu, -gv, (half - d) * inv_px, aa)
                        + _boundary_coverage(gu, gv, (d + half) * inv_px, aa)
                        - 1.0,
                        0.0,
                        1.0,
                    )
                else:
                    c = _boundary_coverage(gu, gv, signed * inv_px, aa)
                if ti.static(int(aa) == 3):
                    # THE ORIENTED WEDGE (DESIGN_analytic_aa_v2.md ss5). A thin
                    # stroke is a STRIP and a corner is a WEDGE; one distance
                    # describes an edge running to infinity, so a 1px glyph
                    # stem reads as solid past its near wall (ss21.2) and a
                    # corner renders as its vertex's distance CIRCLE. Both
                    # walls' inward sides come from STORAGE (edges_2d column 5,
                    # written at flatten time where the contour is known) --
                    # recovering the second wall's side from the contour
                    # handedness was the ss21.6 failure: at a corner the SDF
                    # gradient points at the vertex, so the calibration sign
                    # was arbitrary exactly where the model mattered.
                    #
                    # Validated standalone by benchmarks/_aa_wedge_check.py:
                    # worst coverage error 0.0017 (convex) / 0.0010 (reflex)
                    # over 600 random corners against brute-force polygons.
                    if filled and (sec_dist_sq < 1e29) and (sg1 != 0.0) \
                            and (sg2 != 0.0):
                        l1 = ti.sqrt(e1x * e1x + e1y * e1y)
                        l2 = ti.sqrt(e2x * e2x + e2y * e2y)
                        if (l1 > 1e-20) and (l2 > 1e-20):
                            # Wall normals: sigma times the leftward
                            # perpendicular of the stored contour direction.
                            n1x = -e1y / l1 * sg1
                            n1y = e1x / l1 * sg1
                            n2x = -e2y / l2 * sg2
                            n2y = e2x / l2 * sg2
                            # Signed distances to the wall LINES (valid whether
                            # the closest point is interior or an endpoint --
                            # the endpoint lies on the line), dilated like the
                            # single-plane path.
                            b1p = n1x * ccu + n1y * ccv
                            b2p = n2x * scu + n2y * scv
                            # Each wall's dilation in the distance's units (see
                            # ``wk``); everything else here is already on screen
                            # under the anisotropic query, the walls included.
                            w2k = 1.0
                            if ti.static(_bez_anisotropic()):
                                w2k = _segment_width_scale(
                                    m00, m01, m10, m11, e2x, e2y)
                            sd1 = (outer_w * wk - b1p) * inv_px
                            sd2 = (outer_w * w2k - b2p) * inv_px
                            nd = n1x * n2x + n1y * n2y
                            if nd < 0.9:
                                # nd >= 0.9: the second segment is the next
                                # flattening chord of the SAME wall; folding it
                                # in would halve every plain edge's coverage.
                                a1 = _halfplane_clip_area(n1x, n1y, sd1)
                                a2 = _halfplane_clip_area(n2x, n2y, sd2)
                                inter = _two_halfplane_area(
                                    n1x, n1y, sd1, n2x, n2y, sd2, a1, a2)
                                cn = n1x * n2y - n1y * n2x
                                c = inter
                                if ti.abs(cn) >= 0.2:
                                    # CONVEX vs REFLEX is which RAY of its
                                    # line each wall segment occupies: the
                                    # intersection region's boundary rays
                                    # satisfy the OTHER constraint, the
                                    # union's violate it. Read it off the
                                    # closest points against the apex; when a
                                    # closest point IS the apex, the clamped
                                    # endpoint sign (cp . d) says which way
                                    # the segment leaves it. Parity at the
                                    # pixel centre is the last-resort arbiter
                                    # (undilated distances -- the dilation
                                    # would disagree with parity in a band
                                    # around every edge).
                                    inv_cn = 1.0 / cn
                                    apx = (b1p * n2y - b2p * n1y) * inv_cn
                                    apy = (n1x * b2p - n2x * b1p) * inv_cn
                                    scl = (ti.abs(ccu) + ti.abs(ccv)
                                           + ti.abs(scu) + ti.abs(scv)
                                           + pixel_size)
                                    r1x = ccu - apx
                                    r1y = ccv - apy
                                    s1 = 0.0
                                    if (ti.abs(r1x) + ti.abs(r1y)) \
                                            > 1e-4 * scl:
                                        s1 = n2x * r1x + n2y * r1y
                                    else:
                                        t1 = ccu * e1x + ccv * e1y
                                        if ti.abs(t1) > 1e-4 * scl * l1:
                                            sgn = 1.0 if t1 > 0.0 else -1.0
                                            s1 = sgn * (n2x * e1x
                                                        + n2y * e1y)
                                    r2x = scu - apx
                                    r2y = scv - apy
                                    s2 = 0.0
                                    if (ti.abs(r2x) + ti.abs(r2y)) \
                                            > 1e-4 * scl:
                                        s2 = n1x * r2x + n1y * r2y
                                    else:
                                        t2 = scu * e2x + scv * e2y
                                        if ti.abs(t2) > 1e-4 * scl * l2:
                                            sgn = 1.0 if t2 > 0.0 else -1.0
                                            s2 = sgn * (n1x * e2x
                                                        + n1y * e2y)
                                    uni = ti.math.clamp(
                                        a1 + a2 - inter, 0.0, 1.0)
                                    if (s1 < 0.0) and (s2 < 0.0):
                                        c = uni
                                    elif not ((s1 > 0.0) and (s2 > 0.0)):
                                        ci = (crossings % 2) == 1
                                        in_i = (b1p < 0.0) and (b2p < 0.0)
                                        in_u = (b1p < 0.0) or (b2p < 0.0)
                                        if (in_u == ci) and (in_i != ci):
                                            c = uni
                bf = 0.0
                if border_w > 0.0:
                    if filled:
                        # Everything covered, minus the part deeper than the
                        # stroke width, is border. That inner edge is a boundary
                        # like any other and gets the same exact treatment, which
                        # is the only edge visible at all on an outlined glyph
                        # whose fill is invisible.
                        fill_c = _boundary_coverage(
                            gu, gv, (d - inner_w * wk) * inv_px, aa)
                        bf = ti.math.clamp(
                            (c - fill_c) / ti.max(c, 1e-6), 0.0, 1.0)
                    else:
                        bf = 1.0
            if near_clip > 0.0:
                kept = _circuit_near_cover(
                    f, px, py, n, center, half_w, half_h, cam_origin,
                    screen_point, pixel_basis_x, pixel_basis_y, cam_proj,
                    near_clip, aa)
                if ti.static(not aa):
                    inside = inside and (kept > 0.5)
                else:
                    c *= kept
            if ti.static(not aa):
                if inside:
                    ok = 1
                    t = th
                    u = uu
                    v = vv
                    border_frac = bf
            else:
                if c > 0.0:
                    ok = 1
                    t = th
                    u = uu
                    v = vv
                    cov = c
                    border_frac = bf
    return ok, t, u, v, border_frac, cov


@ti.func
def _bez_pair_pixel(circuit, f, x0, y0, bw, bh, off, j,
                    time_start, width, height, tile_start, tile_pixels,
                    half_w, half_h,
                    cam_origin: ti.template(), screen_point: ti.template(),
                    pixel_basis_x: ti.template(), pixel_basis_y: ti.template(),
                    cam_proj: ti.template(),
                    pixel_world_scale: ti.template(),
                    circuit_meta: ti.template(), circuit_colors: ti.template(),
                    edges_2d: ti.template(), edge_accel: ti.template(),
                    aa: ti.template(), aa_min_half_width: ti.f32, near_clip):
    """Pair wrapper around :func:`_bez_pixel_hit`. ``lp`` is the candidate's
    pixel whenever it lies in the tile, hit or not, as in :func:`_pair_pixel`.
    """
    ok = 0
    lp = 0
    t = 0.0
    u = 0.0
    v = 0.0
    in_border = 0.0
    cov = 1.0
    o = off + j
    if o < bw * bh:
        px = x0 + o % bw
        py = y0 + o // bw
        lpi = ((f - time_start) * (width * height) + py * width + px
               - tile_start)
        if (lpi >= 0) and (lpi < tile_pixels):
            lp = lpi
            hit, th, uu, vv, ib, cv = _bez_pixel_hit(
                circuit, f, px, py, half_w, half_h, cam_origin, screen_point,
                pixel_basis_x, pixel_basis_y, cam_proj, pixel_world_scale, circuit_meta,
                circuit_colors, edges_2d, edge_accel, aa, aa_min_half_width,
                aa, near_clip)
            if hit != 0:
                ok = 1
                t = th
                u = uu
                v = vv
                in_border = ib
                cov = cv
    return ok, lp, t, u, v, in_border, cov






@ti.kernel
def raster_tri_count(
        pairs: ti.types.ndarray(), num_pairs: int,
        tri_pos: ti.types.ndarray(), tri_screen: ti.types.ndarray(),
        tri_colors: ti.types.ndarray(), col_row: ti.types.ndarray(),
        tri_uvs: ti.types.ndarray(), tri_tex_meta: ti.types.ndarray(),
        textures: ti.types.ndarray(), num_colored_triangles: ti.i32,
        cam_origin: ti.types.ndarray(), screen_point: ti.types.ndarray(),
        pixel_basis_x: ti.types.ndarray(), pixel_basis_y: ti.types.ndarray(),
        cam_proj_table: ti.types.ndarray(),
        projected: ti.template(),
        time_start: int, width: int, height: int,
        half_w: ti.f32, half_h: ti.f32,
        tile_start: int, tile_pixels: int,
        ss_enabled: ti.template(), layer_offset_triangles: ti.f32,
        near_clip: ti.f32,
        z_cull: ti.template(), zbuf: ti.types.ndarray(),
        aa: ti.template(), partial_only: ti.template(),
        pair_count: ti.types.ndarray(),
        pair_accept: ti.types.ndarray()):
    """Count surviving nonzero-alpha transparent triangle fragments.

    ``partial_only`` (compile-time, analytic coverage only): emit just the
    partially covered fragments. No live call site instantiates it (the
    deleted dense path used it to skip pixels its opaque z-prepass already
    owned); the sparse emission always passes 0, as it does for ``z_cull``.

    ``pair_accept[p]`` records the per-pixel acceptance decisions as a bitmask
    (bit ``j`` = chunk pixel ``j`` survived; ``raster_chunk`` is at most 32,
    so one i32 holds them).
    The write pass replays these bits instead of recomputing the whole
    acceptance chain -- most importantly the texture-sampling alpha fetch --
    so this kernel IS the acceptance authority the write contract points at.
    """
    cam_proj = ti.static(CameraProjection(cam_proj_table, projected))
    for p in range(num_pairs):
        prim = pairs[p, 0]
        f = pairs[p, 1]
        x0 = pairs[p, 2]
        y0 = pairs[p, 3]
        bw = pairs[p, 4]
        bh = pairs[p, 5]
        off = pairs[p, 6]
        use_ss, sm, vm, cam_o, il, nl = _ss_setup(
            f, prim, ss_enabled, tri_pos, tri_screen, cam_origin,
            screen_point, near_clip, aa)
        layer = ti.cast(layer_offset_triangles, ti.i32) + prim
        cnt = 0
        bits = 0
        for j in range(raster_chunk):
            ok, lp, t, w1, w2, cov, msk = _pair_pixel(
                prim, f, x0, y0, bw, bh, off, j, time_start, width, height,
                tile_start, tile_pixels, half_w, half_h, use_ss, sm, vm, cam_o,
                il, nl, near_clip, cam_origin, screen_point, pixel_basis_x,
                pixel_basis_y, cam_proj, aa,
                0)
            if ok != 0:
                keep = True
                if ti.static(partial_only):
                    if ti.static(_tri_run(aa)):
                        # Under the run rule "fully covered" is keyed by the
                        # sample MASK (a full-mask fragment claims the whole
                        # pixel), so only full-mask fragments are excluded.
                        keep = (msk & _AA_MASK_ALL) != _AA_MASK_ALL
                    else:
                        keep = cov < AA_FULL_COVERAGE
                before_z = True
                if ti.static(z_cull):
                    before_z = _order_key(t, layer) < zbuf[lp]
                if keep and before_z:
                    w0 = 1.0 - w1 - w2
                    _color, alpha = _tri_color_g(
                        0, f, prim, w0, w1, w2, tri_colors, col_row, tri_uvs,
                        tri_tex_meta, textures, num_colored_triangles)
                    if ti.static(aa):
                        alpha *= cov
                    if alpha > min_alpha:
                        cnt += 1
                        bits |= 1 << j
        pair_count[p] = cnt
        pair_accept[p] = bits


@ti.kernel
def raster_tri_write(
        pairs: ti.types.ndarray(), num_pairs: int,
        pair_offset: ti.types.ndarray(),
        tri_pos: ti.types.ndarray(), tri_screen: ti.types.ndarray(),
        tri_colors: ti.types.ndarray(), col_row: ti.types.ndarray(),
        tri_uvs: ti.types.ndarray(), tri_tex_meta: ti.types.ndarray(),
        textures: ti.types.ndarray(), num_colored_triangles: ti.i32,
        cam_origin: ti.types.ndarray(), screen_point: ti.types.ndarray(),
        pixel_basis_x: ti.types.ndarray(), pixel_basis_y: ti.types.ndarray(),
        cam_proj_table: ti.types.ndarray(),
        projected: ti.template(),
        time_start: int, width: int, height: int,
        half_w: ti.f32, half_h: ti.f32,
        tile_start: int, tile_pixels: int,
        ss_enabled: ti.template(), layer_offset_triangles: ti.f32,
        near_clip: ti.f32,
        z_cull: ti.template(), zbuf: ti.types.ndarray(),
        aa: ti.template(), partial_only: ti.template(),
        frag_key: ti.types.ndarray(),
        frag_ref: ti.types.ndarray(), frag_ab: ti.types.ndarray(),
        frag_cov: ti.types.ndarray(), frag_msk: ti.types.ndarray(),
        pair_accept: ti.types.ndarray()):
    """Emit exact-distance triangle records; alpha-zero texels are discarded.

    Acceptance replays ``pair_accept`` -- the per-pixel decision bits the
    count pass recorded -- rather than recomputing the acceptance chain, so
    the two passes cannot diverge and this pass never touches the color /
    texture arrays at all. Emission order per pair is ascending chunk pixel,
    exactly the order the recompute produced. The record's geometry lanes
    (lp/t/w1/w2/cov/msk) still come from the exact ``_pair_pixel`` evaluation
    (mode 1: the cov lane carries the exact area for masked fragments; the
    count's acceptance used the sampled share, which the bits already bake
    in). Should that evaluation miss a pixel the count accepted, the slot
    gets the inert ``_WRITE_MISS_T_BITS`` record rather than the miss's
    default lanes.
    """
    cam_proj = ti.static(CameraProjection(cam_proj_table, projected))
    for p in range(num_pairs):
        prim = pairs[p, 0]
        f = pairs[p, 1]
        x0 = pairs[p, 2]
        y0 = pairs[p, 3]
        bw = pairs[p, 4]
        bh = pairs[p, 5]
        off = pairs[p, 6]
        use_ss, sm, vm, cam_o, il, nl = _ss_setup(
            f, prim, ss_enabled, tri_pos, tri_screen, cam_origin,
            screen_point, near_clip, aa)
        w = pair_offset[p]
        bits = pair_accept[p]
        for j in range(raster_chunk):
            if ((bits >> j) & 1) != 0:
                ok, lp, t, w1, w2, cov, msk = _pair_pixel(
                    prim, f, x0, y0, bw, bh, off, j, time_start, width,
                    height, tile_start, tile_pixels, half_w, half_h, use_ss,
                    sm, vm, cam_o, il, nl, near_clip, cam_origin,
                    screen_point, pixel_basis_x, pixel_basis_y, cam_proj, aa,
                    1)
                tb = ti.cast(ti.bit_cast(t, ti.u32), ti.i64)
                if ok == 0:
                    tb = ti.i64(_WRITE_MISS_T_BITS)
                    w1 = 0.0
                    w2 = 0.0
                    cov = 0.0
                    msk = 0
                frag_key[w] = (ti.cast(lp, ti.i64) << 32) | tb
                frag_ref[w] = prim
                frag_ab[w, 0] = w1
                frag_ab[w, 1] = w2
                if ti.static(aa):
                    frag_cov[w] = cov
                    frag_msk[w] = msk
                w += 1


@ti.kernel
def raster_bez_count(
        pairs: ti.types.ndarray(), num_pairs: int,
        cam_origin: ti.types.ndarray(), screen_point: ti.types.ndarray(),
        pixel_basis_x: ti.types.ndarray(), pixel_basis_y: ti.types.ndarray(),
        cam_proj_table: ti.types.ndarray(),
        projected: ti.template(),
        pixel_world_scale: ti.types.ndarray(),
        circuit_meta: ti.types.ndarray(), circuit_colors: ti.types.ndarray(),
        circuit_border_colors: ti.types.ndarray(),
        edges_2d: ti.types.ndarray(), edge_accel: ti.types.ndarray(),
        time_start: int, width: int, height: int,
        half_w: ti.f32, half_h: ti.f32,
        tile_start: int, tile_pixels: int,
        z_cull: ti.template(), zbuf: ti.types.ndarray(),
        aa: ti.template(), aa_min_half_width: ti.f32,
        near_clip: ti.f32,
        partial_only: ti.template(),
        pair_count: ti.types.ndarray(),
        pair_accept: ti.types.ndarray()):
    """Count surviving nonzero-alpha translucent circuit fragments.

    ``partial_only`` (compile-time, analytic coverage only): emit ONLY the
    partially covered fragments. No live call site instantiates it (the
    deleted dense path used it alongside its opaque z-prepass); the sparse
    emission always passes 0, as it does for ``z_cull``.

    ``pair_accept[p]`` records the per-pixel acceptance bits the write pass
    replays (see :func:`raster_tri_count`).
    """
    cam_proj = ti.static(CameraProjection(cam_proj_table, projected))
    for p in range(num_pairs):
        circuit = pairs[p, 0]
        f = pairs[p, 1]
        x0 = pairs[p, 2]
        y0 = pairs[p, 3]
        bw = pairs[p, 4]
        bh = pairs[p, 5]
        off = pairs[p, 6]
        cnt = 0
        bits = 0
        for j in range(raster_chunk):
            ok, lp, t, u, v, ib, cov = _bez_pair_pixel(
                circuit, f, x0, y0, bw, bh, off, j, time_start, width, height,
                tile_start, tile_pixels, half_w, half_h, cam_origin,
                screen_point, pixel_basis_x, pixel_basis_y, cam_proj, pixel_world_scale,
                circuit_meta, circuit_colors, edges_2d, edge_accel,
                aa, aa_min_half_width, near_clip)
            if ok != 0:
                keep = True
                if ti.static(partial_only):
                    keep = cov < AA_FULL_COVERAGE
                before_z = True
                if ti.static(z_cull):
                    before_z = _order_key(t, circuit) < zbuf[lp]
                if keep and before_z:
                    _color, alpha = _sample_circuit_color_blend(
                        circuit, f, u, v, ib, circuit_meta, circuit_colors,
                        circuit_border_colors)
                    if ti.static(aa):
                        alpha *= cov
                    if alpha > min_alpha:
                        cnt += 1
                        bits |= 1 << j
        pair_count[p] = cnt
        pair_accept[p] = bits


@ti.kernel
def raster_bez_write(
        pairs: ti.types.ndarray(), num_pairs: int,
        pair_offset: ti.types.ndarray(),
        cam_origin: ti.types.ndarray(), screen_point: ti.types.ndarray(),
        pixel_basis_x: ti.types.ndarray(), pixel_basis_y: ti.types.ndarray(),
        cam_proj_table: ti.types.ndarray(),
        projected: ti.template(),
        pixel_world_scale: ti.types.ndarray(),
        circuit_meta: ti.types.ndarray(), circuit_colors: ti.types.ndarray(),
        circuit_border_colors: ti.types.ndarray(),
        edges_2d: ti.types.ndarray(), edge_accel: ti.types.ndarray(),
        time_start: int, width: int, height: int,
        half_w: ti.f32, half_h: ti.f32,
        tile_start: int, tile_pixels: int,
        z_cull: ti.template(), zbuf: ti.types.ndarray(),
        aa: ti.template(), aa_min_half_width: ti.f32,
        near_clip: ti.f32,
        partial_only: ti.template(),
        frag_key: ti.types.ndarray(),
        frag_ref: ti.types.ndarray(), frag_ab: ti.types.ndarray(),
        frag_cov: ti.types.ndarray(),
        pair_accept: ti.types.ndarray()):
    """Emit circuit records with the border weight packed into ``frag_ref``.

    Acceptance replays the count pass's ``pair_accept`` bits (see
    :func:`raster_tri_write`) -- the passes cannot diverge, and this one
    skips the color sampling entirely; the exact ``_bez_pair_pixel``
    evaluation still supplies the record's geometry lanes (a miss there gets
    the inert ``_WRITE_MISS_T_BITS`` record).  ``frag_cov`` is
    the analytic coverage lane, pre-filled with 1.0 by the host so geometry
    without analytic coverage (flat triangles today) needs no write of its
    own.
    """
    cam_proj = ti.static(CameraProjection(cam_proj_table, projected))
    for p in range(num_pairs):
        circuit = pairs[p, 0]
        f = pairs[p, 1]
        x0 = pairs[p, 2]
        y0 = pairs[p, 3]
        bw = pairs[p, 4]
        bh = pairs[p, 5]
        off = pairs[p, 6]
        w = pair_offset[p]
        bits = pair_accept[p]
        for j in range(raster_chunk):
            if ((bits >> j) & 1) != 0:
                ok, lp, t, u, v, ib, cov = _bez_pair_pixel(
                    circuit, f, x0, y0, bw, bh, off, j, time_start, width,
                    height, tile_start, tile_pixels, half_w, half_h,
                    cam_origin, screen_point, pixel_basis_x, pixel_basis_y, cam_proj,
                    pixel_world_scale, circuit_meta, circuit_colors,
                    edges_2d, edge_accel, aa, aa_min_half_width, near_clip)
                tb = ti.cast(ti.bit_cast(t, ti.u32), ti.i64)
                if ok == 0:
                    # See _WRITE_MISS_T_BITS (raster_tri_write).
                    tb = ti.i64(_WRITE_MISS_T_BITS)
                    u = 0.0
                    v = 0.0
                    cov = 0.0
                frag_key[w] = (ti.cast(lp, ti.i64) << 32) | tb
                frag_ref[w] = _pack_bez_ref(circuit, ib)
                frag_ab[w, 0] = u
                frag_ab[w, 1] = v
                if ti.static(aa):
                    frag_cov[w] = cov
                w += 1


# Sub-pixel positions for continuation-ray supersampling, by tap count. At 4
# these are exactly where anti_alias_level=2 puts its supersamples, so a
# reflected image sampled at these positions matches that reference arm's
# geometry rather than merely resembling it. Regular grids, not random jitter:
# the renderer's output must stay deterministic and frame-independent
# (DESIGN_analytic_aa.md ss12), and a random offset would also make a mirror
# hiss between frames.
#
# The four hand-written sets are pinned BYTE-IDENTICAL -- render baselines were
# taken at them -- and every other count up to _AA_SEC_MAX is generated once at
# import from a Hammersley point set, so any requested count has real positions
# instead of silently snapping down to the nearest hand-written one.
_AA_SEC_JITTER_HANDWRITTEN = {
    1: ((0.5, 0.5),),
    2: ((0.25, 0.25), (0.75, 0.75)),
    4: ((0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)),
    8: ((0.1875, 0.3125), (0.6875, 0.1875), (0.3125, 0.8125),
        (0.8125, 0.6875), (0.0625, 0.5625), (0.5625, 0.4375),
        (0.4375, 0.9375), (0.9375, 0.0625)),
}

# Ceiling on the configured tap count (analytic_aa_secondary_samples). The
# position mask ``_sec_positions`` returns is an i32 bitfield, one bit per
# position, so bit 32 would be unreadable; and each distinct count reaches the
# resolve as its own template value whose ``ti.static(range(sec_aa))`` unrolls
# at five call sites, so every extra tap is also paid in kernel compile time.
_AA_SEC_MAX = 32


def _radical_inverse_base2(i):
    """Van der Corput radical inverse of ``i`` in base 2: its bits reversed
    past the binary point. i = 0 -> 0.0, 1 -> 0.5, 2 -> 0.25, 3 -> 0.75 ...
    """
    out = 0.0
    f = 0.5
    while i:
        if i & 1:
            out += f
        f *= 0.5
        i >>= 1
    return out


def _hammersley_secondary_positions(n):
    """A deterministic stratified set of ``n`` positions in the unit square.

    Hammersley: x runs through ``(i + 0.5) / n``, y through the radical inverse
    of ``i`` base 2. No randomness anywhere -- these tuples are baked into the
    compiled kernels, so anything stochastic here would make the render
    nondeterministic or drift between frames.
    """
    return tuple(
        ((i + 0.5) / n, _radical_inverse_base2(i)) for i in range(n)
    )


def secondary_jitter_positions(n):
    """The sub-pixel continuation-position tuple for a tap count.

    Hand-written for 1/2/4/8 (the counts the render baselines pin); generated
    from a Hammersley set for every other count up to ``_AA_SEC_MAX``. Memoised
    by the import-time dict below, so every caller shares one tuple object and
    nothing regenerates after import.
    """
    n = int(n)
    if not 1 <= n <= _AA_SEC_MAX:
        raise ValueError(
            f"analytic-AA secondary taps must be between 1 and {_AA_SEC_MAX}, "
            f"got {n}"
        )
    return _AA_SEC_JITTER[n]


# Every supported count materialises its positions once, at import: the
# kernels index this table statically, so there is no path that could hit a
# missing key.
_AA_SEC_JITTER = {
    n: (_AA_SEC_JITTER_HANDWRITTEN[n]
        if n in _AA_SEC_JITTER_HANDWRITTEN
        else _hammersley_secondary_positions(n))
    for n in range(1, _AA_SEC_MAX + 1)
}


def _nearest_jitter(k, n):
    """Index of the sub-pixel continuation position nearest to sample ``k``."""
    sx = 0.5 + _AA_SAMPLES[k][0] / _AA_FIXED_SCALE
    sy = 0.5 + _AA_SAMPLES[k][1] / _AA_FIXED_SCALE
    best, bd = 0, 1e9
    for j, (jx, jy) in enumerate(_AA_SEC_JITTER[n]):
        d = (sx - jx) ** 2 + (sy - jy) ** 2
        if d < bd:
            bd, best = d, j
    return best


def _sample_nearest_to_position(j, n):
    """Index of the coverage sample nearest to continuation position ``j``."""
    jx, jy = _AA_SEC_JITTER[n][j]
    best, bd = 0, 1e9
    for k in range(_AA_NUM_SAMPLES):
        sx = 0.5 + _AA_SAMPLES[k][0] / _AA_FIXED_SCALE
        sy = 0.5 + _AA_SAMPLES[k][1] / _AA_FIXED_SCALE
        d = (sx - jx) ** 2 + (sy - jy) ** 2
        if d < bd:
            bd, best = d, k
    return best


# Which continuation positions each coverage sample owns, per tap count; pure
# Python, evaluated at import: the kernel only ever indexes it statically.
# Entry k is a TUPLE of the position indices owned by coverage sample k.
#
# The ownership DIRECTION differs between the two regimes, deliberately:
#
# The four hand-written counts keep today's FORWARD rule -- each coverage
# sample owns the single position nearest it -- because the inverse rule does
# not reproduce it: measured over all 256 coverage masks, the two disagree at
# every hand-written count (208 masks at n=8 alone), and the render baselines
# pin those tables through this mapping. That includes today's quirks: under
# the forward rule a position can be nobody's nearest (at n = 8 position 1
# is), so a FULLY covered fragment there owns 7 positions and spawns 7 rays,
# not 8 -- pinned behaviour, not an accident to fix on the side.
#
# Every OTHER count uses the INVERSE rule: each POSITION is assigned to its
# nearest coverage sample, which may own several, and the assignment always
# covers every position. Two things force the flip. Forward ownership can
# never exceed one position per coverage sample, so with only _AA_NUM_SAMPLES
# (= 8) samples no fragment could ever own more than 8 positions however fine
# the tap table was -- that cap is what silently rendered 16 and 32 taps as 8
# rays. And a generated set cannot keep the forward rule either: its positions
# are not chosen so that each has a nearest sample (measured at n = 5, where
# one position is nobody's nearest), so some fully covered fragment would own
# fewer than n positions. Inverse ownership partitions all n positions across
# the samples, so a fully covered fragment always owns all n and spawns all n
# rays.
_AA_SEC_OWNER = {
    n: (
        # Forward rule, wrapped as 1-tuples so both regimes share one shape.
        tuple((_nearest_jitter(k, n),) for k in range(_AA_NUM_SAMPLES))
        if n in _AA_SEC_JITTER_HANDWRITTEN
        # Inverse rule: group the positions by their nearest coverage sample.
        else tuple(
            tuple(j for j in range(n)
                  if _sample_nearest_to_position(j, n) == k)
            for k in range(_AA_NUM_SAMPLES)
        )
    )
    for n in range(1, _AA_SEC_MAX + 1)
}


@ti.func
def _sec_positions(msk, n: ti.template()):
    """Which of the N continuation positions this fragment actually covers.

    A fragment spawns one secondary ray per position it owns, not N regardless
    -- which is exactly what supersampling does, one ray per sub-pixel the
    primitive covers. Gating on the fragment's total coverage instead was tried
    and is WRONG: in a dense mesh nearly every fragment partially covers its
    pixel (each triangle owns a few samples), so a "full coverage only" test
    switched secondary sampling off almost everywhere and cost a glass scene its
    entire refracted-image quality.

    A covered sample contributes EVERY position it owns (see ``_AA_SEC_OWNER``
    for why the ownership direction differs between the hand-written counts and
    the generated ones), so a fully covered fragment owns all N positions and
    spawns N rays.

    Returns ``(position_mask, count)``; count is zero only for an empty mask.
    """
    # The accumulator is i64 because at 32 taps the top bit has no positive
    # i32 literal ((1 << 31) is out of range); the final cast back to i32
    # truncates, which IS the wrap the bitfield wants -- a fully covered
    # 32-tap fragment returns -1, every bit set.
    acc = ti.cast(0, ti.i64)
    for k in ti.static(range(_AA_NUM_SAMPLES)):
        if (msk >> k) & 1:
            for j in ti.static(_AA_SEC_OWNER[n][k]):
                acc |= ti.cast(1, ti.i64) << ti.static(j)
    pm = ti.cast(acc, ti.i32)
    # Read back one bit at a time and never compare pm itself: at n = 32 the
    # top bit makes pm NEGATIVE as an i32, and the arithmetic shift still
    # reads every bit correctly -- a `pm > 0`-style test would not.
    cnt = 0
    for j in ti.static(range(n)):
        if (pm >> j) & 1:
            cnt += 1
    return pm, cnt


# ``_GLOSSY_MIN_ROUGHNESS`` (below it a reflection stays specular-perfect)
# lives in ``wavefront_kernels_taichi`` beside ``_mirror_share``: the drain
# needs it too, and this module imports that one, not the other way round.
_GLOSSY_INV_16 = 1.0 / 16.0


@ti.func
def _bayer4(px, py):
    """The canonical 4x4 ordered-dither index (0..15) of a pixel.

    ``M4[y][x] = 4*m2(y&1, x&1) + m2(y>>1, x>>1)`` with the 2x2 base
    ``[[0, 2], [3, 1]]`` written as ``m2(a, b) = 2*(a^b) + a``. A permutation of
    0..15 over every 4x4 block, arranged so neighbours are maximally far apart
    in index -- which is the property wanted here, since the index chooses which
    part of the reflection lobe the pixel samples.
    """
    yl = py & 1
    xl = px & 1
    yh = (py >> 1) & 1
    xh = (px >> 1) & 1
    return 4 * (2 * (yl ^ xl) + yl) + (2 * (yh ^ xh) + yh)


@ti.func
def _glossy_rotation(px, py, interleave: ti.template()):
    """Per-pixel offsets that rotate a fragment's lobe fan.

    Returns ``(radial_offset in (0,1), azimuth_offset in radians)``. With
    ``interleave`` off both are fixed, so every pixel samples the same few lobe
    directions and a four-tap fan reads as four ghost copies of the reflected
    image. With it on, a 4x4 Bayer index scatters the taps: the radial stratum
    is offset by a Cranley-Patterson rotation (16 sub-strata inside each of the
    fragment's own strata) and the azimuth by a golden-angle multiple of the
    same index, which decorrelates the two dimensions from each other.

    This is not decoration. The radial coordinate a tap ends up with is
    ``(j + (b + 0.5)/16) / k``, i.e. ``(16j + b + 0.5) / 16k``, and ``16j + b``
    runs over ``0 .. 16k-1`` exactly once -- so a 4x4 block's taps ARE the
    ``16k`` quantiles of the lobe, the optimal stratified set, for no extra
    rays. Verified bit-exact for k = 2, 3, 4 and 8, which matters because ``k``
    is how many sub-pixel positions the FRAGMENT covers, not the setting: a
    silhouette fragment owning two positions still samples its lobe at a perfect
    32 quantiles. Without the rotation every pixel repeats the same ``k``
    strata, so the whole image is a k-point quadrature -- measurably so
    (``benchmarks/_glossy_ggx_check.py`` Part A: KS 0.125 against the analytic
    CDF at k=4, exactly the 1/(2k) floor, against 0.008 with it).

    Fixed in SCREEN space, so it is a function of the pixel and nothing else:
    the same frame renders identically every time, and across an animation the
    pattern is stationary -- it cannot twinkle, which a time- or object-varying
    pattern would.
    """
    r_off = 0.5
    ang_off = 0.0
    if ti.static(interleave):
        b = ti.cast(_bayer4(px, py), ti.f32)
        r_off = (b + 0.5) * _GLOSSY_INV_16
        ang_off = _GOLDEN_ANGLE * b
    return r_off, ang_off


@ti.func
def _glossy_reflect(rd, n, roughness, j, k, r_off, ang_off):
    """Tap ``j`` of ``k`` on the GGX reflection lobe about normal ``n``.

    Samples the GGX / Trowbridge-Reitz normal distribution for a MICROFACET
    NORMAL and reflects ``rd`` about it, which is the same lobe
    ``shading_taichi._stage_standard`` uses for the direct highlight, at the
    same ``alpha = roughness^2``. Matching it is the point: a reflection blurred
    by a different width than the highlight beside it describes two different
    materials. (The Monte Carlo megakernel's ``rd + roughness * random_unit``
    is a NORMAL PERTURBATION, a visibly wider and differently shaped lobe --
    ~2.8x the angular width at roughness 0.18 -- and is not what this follows.)

    Deterministic: the radial coordinate is the stratum ``(j + r_off) / k`` of
    the inverted GGX CDF and the azimuth is ``GOLDEN_ANGLE * j + ang_off``, the
    same fixed low-discrepancy construction the soft-shadow fan uses. Nothing
    here reads ``ti.random``, so an animation cannot hiss between frames.

    Every tap carries an equal share of the reflected throughput and the caller
    divides by ``k``, so this redistributes the mirror ray's energy over the
    lobe without creating or destroying any: the NDF is sampled for DIRECTIONS
    only, and the per-tap Fresnel/geometry reweighting a full importance-sampled
    GGX estimator would apply is deliberately omitted (it would change the total
    at grazing, and the surrounding four-way split has already spent the
    Fresnel term).

    A tap that lands below the horizon is reflected back across ``n`` -- the
    same repair the Monte Carlo path makes -- so the returned direction always
    agrees with the ``n``-facing origin offset the caller pairs it with.
    """
    a = roughness * roughness
    u1 = (ti.cast(j, ti.f32) + r_off) / ti.cast(k, ti.f32)
    # Inverted GGX radial CDF: tan^2(theta) = a^2 * u / (1 - u).
    tan2 = (a * a) * u1 / ti.max(1.0 - u1, 1e-6)
    cos_t = 1.0 / ti.sqrt(1.0 + tan2)
    sin_t = ti.sqrt(ti.max(1.0 - cos_t * cos_t, 0.0))
    ang = _GOLDEN_ANGLE * ti.cast(j, ti.f32) + ang_off
    aref = ti.math.vec3(1.0, 0.0, 0.0)
    if ti.abs(n[0]) > 0.9:
        aref = ti.math.vec3(0.0, 1.0, 0.0)
    t1 = n.cross(aref).normalized()
    t2 = n.cross(t1)
    h = (t1 * (ti.cos(ang) * sin_t) + t2 * (ti.sin(ang) * sin_t)
         + n * cos_t).normalized()
    out = (rd - 2.0 * rd.dot(h) * h).normalized()
    if out.dot(n) <= 0.0:
        out = (out - 2.0 * out.dot(n) * n).normalized()
    return out


@ti.func
def _jittered_surface_sample(f, px, py, jx, jy, gen_meta: ti.template(),
                             is_bez, prim, hit_point, nrm,
                             tri_pos: ti.template(), tri_norm: ti.template(),
                             tri_uvs: ti.template(),
                             tri_tex_meta: ti.template(),
                             textures: ti.template(), num_colored_triangles,
                             cam_origin: ti.template(),
                             screen_point: ti.template(),
                             pixel_basis_x: ti.template(),
                             pixel_basis_y: ti.template(), cam_proj: ti.template()):
    """Re-sample one hit at a different sub-pixel position within its pixel.

    Regenerates the primary ray through ``(jx, jy)`` and asks where it would
    have met the SAME primitive. Both flat triangles and bezier circuits are
    planes, so one ray-plane solve is exact -- no re-traversal.

    For a triangle it also returns the SHADING NORMAL re-interpolated at that
    point, which on a curved mirror matters more than the shifted origin does:
    the reflected direction turns by twice the normal's change across the pixel,
    and curvature amplifies it, so reflecting several sub-samples off one shared
    normal blurs the reflection rather than resolving it. A circuit's normal is
    constant over its plane, so only its origin moves.

    Barycentrics are projected onto the simplex before they index normals or
    UVs, exactly as ``_ss_pixel`` does for a partially covering hit: a sub-sample
    of a silhouette pixel can fall outside the triangle, and extrapolated vertex
    normals are not safe on a coarse mesh.

    The returned hit point is rebuilt from those projected barycentrics as well
    (``_tri_surface_point``'s rule for the fragment itself), and ``rd`` is
    re-aimed at it. A fragment owns a tap through any covered sample in the
    pixel (``_sec_positions``), so along a mesh edge a tap's ray can miss the
    triangle, and its raw ray-plane point then lies on the triangle's plane
    PAST the shared edge. On a convex mesh that is outside the solid, above the
    neighbouring facet, so a transmitted continuation re-enters through the
    neighbour; on a concave one it is below the neighbour, so a reflection hits
    it from behind. The fixed ``10 * min_hit_distance`` spawn offset hid the
    shallower overshoots; the scale-aware ``_offset_ray_origin`` (256 ULPs,
    about 3e-5 at unit coordinates) is far smaller and exposed them as a
    bright speckle network along the mesh edges of transmissive spheres. Only
    a point on the triangle meets that offset's precondition; rebuilt from the
    vertices, it also sits on the plane to the vertices' rounding rather than
    the ray-plane solve's.

    The jitter is an offset from whatever base sub-pixel position the pass uses
    (``gen_meta[0:2]``), so it composes with in-place supersampling rather than
    overriding it. A grazing or degenerate solve keeps the un-jittered hit --
    the same ray the un-supersampled path would have spawned.

    Returns ``(rd, hit_point, normal, b1, b2)``; the barycentrics are zero for a
    circuit and let the caller re-shade a triangle at that sub-sample.
    """
    ro, rd = _generate_ray(
        f, px, py, gen_meta[0] + (jx - 0.5), gen_meta[1] + (jy - 0.5),
        gen_meta[2], gen_meta[3], cam_origin, screen_point,
        pixel_basis_x, pixel_basis_y, cam_proj)
    hp = hit_point
    nj = nrm
    bj1 = 0.0
    bj2 = 0.0
    if is_bez:
        den = rd.dot(nrm)
        if ti.abs(den) > 1e-9:
            ts = (hit_point - ro).dot(nrm) / den
            if ts > min_hit_distance:
                hp = ro + ts * rd
                # Back on the plane: ``ro + ts * rd`` is rounded at the
                # eye's magnitude, which a distant camera puts under it,
                # past the spawn offset (_plane_pt, _bezier_plane_point).
                nn = nrm.dot(nrm)
                if nn > 1e-30:
                    hp = hp - nrm * ((hp - hit_point).dot(nrm) / nn)
    else:
        tp = f % tri_pos.shape[0]
        v0 = ti.math.vec3(tri_pos[tp, prim, 0], tri_pos[tp, prim, 1],
                          tri_pos[tp, prim, 2])
        v1 = ti.math.vec3(tri_pos[tp, prim, 3], tri_pos[tp, prim, 4],
                          tri_pos[tp, prim, 5])
        v2 = ti.math.vec3(tri_pos[tp, prim, 6], tri_pos[tp, prim, 7],
                          tri_pos[tp, prim, 8])
        e1 = v1 - v0
        e2 = v2 - v0
        gn = e1.cross(e2)
        den = rd.dot(gn)
        if ti.abs(den) > 1e-9:
            ts = (v0 - ro).dot(gn) / den
            if ts > min_hit_distance:
                hp = ro + ts * rd
                nn = gn.dot(gn)
                if nn > 1e-30:
                    inv = 1.0 / nn
                    d = hp - v0
                    c1 = ti.max(gn.dot(d.cross(e2)) * inv, 0.0)
                    c2 = ti.max(gn.dot(e1.cross(d)) * inv, 0.0)
                    c0 = ti.max(1.0 - c1 - c2, 0.0)
                    s = c0 + c1 + c2
                    if s > 1e-20:
                        si = 1.0 / s
                        bj1 = c1 * si
                        bj2 = c2 * si
                        # The spawn point is rebuilt from the projected
                        # barycentrics too, not left on the plane: see the
                        # docstring for why a point past an edge speckles.
                        hp = (c0 * si) * v0 + bj1 * v1 + bj2 * v2
                        rd = (hp - ro).normalized()
                        if ti.static(_projected(cam_proj)):
                            # From where the camera ray reaching hp starts:
                            # an orthographic ray re-aimed from ``ro`` would
                            # tilt.
                            rd = (hp - _camera_point(f, hp, cam_origin,
                                                     cam_proj)).normalized()
                        nj = _tri_normal_g(
                            0, f, prim, c0 * si, bj1, bj2, tri_norm,
                            tri_pos, tri_uvs, tri_tex_meta, textures,
                            num_colored_triangles).normalized()
    return rd, hp, nj, bj1, bj2


@ti.func
def _plane_pt(f, px, py, jx, jy, gen_meta: ti.template(), p0, nrm, fallback,
              cam_origin: ti.template(), screen_point: ti.template(),
              pixel_basis_x: ti.template(), pixel_basis_y: ti.template(),
              cam_proj: ti.template()):
    """Where the sub-pixel-(jx, jy) primary ray meets the plane (p0, nrm).

    ``ro + ts * rd`` is rounded at the EYE's magnitude, in every direction
    including off the plane: a distant camera's eye can be thousands of
    units out, where float32 positions are ~1e-3 apart, far more than the
    shadow-origin offset (``_offset_ray_origin``, ~6e-5 at unit-scale
    coordinates). So the point is put back on the plane, which is formed at
    the scene's own magnitude (``out - p0``) and so is exact to its rounding.
    """
    ro, rd = _generate_ray(
        f, px, py, gen_meta[0] + (jx - 0.5), gen_meta[1] + (jy - 0.5),
        gen_meta[2], gen_meta[3], cam_origin, screen_point,
        pixel_basis_x, pixel_basis_y, cam_proj)
    den = rd.dot(nrm)
    out = fallback
    if ti.abs(den) > 1e-9:
        ts = (p0 - ro).dot(nrm) / den
        if ts > min_hit_distance:
            out = ro + ts * rd
            nn = nrm.dot(nrm)
            if nn > 1e-30:
                out = out - nrm * ((out - p0).dot(nrm) / nn)
    return out


@ti.func
def _pixel_footprint(f, px, py, gen_meta: ti.template(), hp, nrm,
                     cam_origin: ti.template(), screen_point: ti.template(),
                     pixel_basis_x: ti.template(),
                     pixel_basis_y: ti.template(), cam_proj: ti.template()):
    """World-space vectors spanning one pixel ACROSS the hit surface.

    The surface point of sub-pixel ``(jx, jy)`` is
    ``hp + (jx - 0.5) * dpx + (jy - 0.5) * dpy``. Central differences of the
    ray-plane intersection, so it is exact for the plane the hit lies in (which a
    flat triangle is) and correctly ANISOTROPIC: a surface seen at a grazing
    angle has a long footprint along the direction it recedes in, which is
    exactly where its shadow edge needs the most sub-pixel resolution.

    Six floats per shadow event, independent of how many sub-samples the trace
    then takes -- versus storing the sub-positions themselves.

    Both vectors lie IN the plane (``_plane_pt`` puts each point back on it):
    the shadow origins they spread sit only ``_offset_ray_origin``'s lift off
    the surface, and under a distant eye the raw points' rounding sank them
    below it, so their rays hit the surface they left -- per-pixel dark
    speckle on every lit slope, re-rolled each frame as the camera moved.
    """
    xp = _plane_pt(f, px, py, 1.0, 0.5, gen_meta, hp, nrm, hp, cam_origin,
                   screen_point, pixel_basis_x, pixel_basis_y, cam_proj)
    xm = _plane_pt(f, px, py, 0.0, 0.5, gen_meta, hp, nrm, hp, cam_origin,
                   screen_point, pixel_basis_x, pixel_basis_y, cam_proj)
    yp = _plane_pt(f, px, py, 0.5, 1.0, gen_meta, hp, nrm, hp, cam_origin,
                   screen_point, pixel_basis_x, pixel_basis_y, cam_proj)
    ym = _plane_pt(f, px, py, 0.5, 0.0, gen_meta, hp, nrm, hp, cam_origin,
                   screen_point, pixel_basis_x, pixel_basis_y, cam_proj)
    return xp - xm, yp - ym


@ti.func
def _footprint_on_triangle(f, prim, w0, a, b, dpx, dpy, pos_msk,
                           tri_pos: ti.template()):
    """Shrink a shadow event's footprint until every position it owns is on
    its triangle.

    ``_sub_pixel_origin`` spreads the shadow origins over a 2x2 grid at
    +-1/4 of the footprint around the event's surface point, on the
    triangle's PLANE. A fragment owns a position through any covered sample in
    the pixel (``_sec_positions``), so along a mesh edge an owned position can
    lie past the edge -- behind the neighbouring facet of a concave mesh, whose
    back its shadow ray then hits: dark acne along the edges. The fixed
    ``10 * min_hit_distance`` origin lift hid the shallow cases; the
    scale-aware ``_offset_ray_origin`` is far smaller and exposes them.

    Barycentrics are affine in position, so the largest ``k`` keeping an owned
    position ``p + k * o`` on the triangle is ``min(c_i / -dc_i)`` over the
    barycentrics ``o`` decreases. One ``k`` for all of them (the smallest)
    keeps the grid a scaled copy, which is all two stored vectors can express.
    ``(w0, a, b)`` are the event point's own barycentrics (it is
    ``_tri_surface_point``, on the triangle), so ``k`` is in [0, 1] and a
    fragment whose owned positions are all on its triangle keeps its footprint
    unchanged. ``pos_msk`` is the event's 4-bit owned-position mask.
    """
    tp = f % tri_pos.shape[0]
    v0 = ti.math.vec3(tri_pos[tp, prim, 0], tri_pos[tp, prim, 1],
                      tri_pos[tp, prim, 2])
    v1 = ti.math.vec3(tri_pos[tp, prim, 3], tri_pos[tp, prim, 4],
                      tri_pos[tp, prim, 5])
    v2 = ti.math.vec3(tri_pos[tp, prim, 6], tri_pos[tp, prim, 7],
                      tri_pos[tp, prim, 8])
    e1 = v1 - v0
    e2 = v2 - v0
    gn = e1.cross(e2)
    nn = gn.dot(gn)
    k = 1.0
    if nn > 1e-30:
        inv = 1.0 / nn
        # Rates of (c1, c2) along each footprint vector; c0 takes the rest.
        x1 = gn.dot(dpx.cross(e2)) * inv
        x2 = gn.dot(e1.cross(dpx)) * inv
        y1 = gn.dot(dpy.cross(e2)) * inv
        y2 = gn.dot(e1.cross(dpy)) * inv
        c = ti.math.vec3(w0, a, b)
        for s in ti.static(range(4)):
            if (pos_msk >> s) & 1:
                # _sub_pixel_origin's grid offsets for position s.
                su = (s & 1) * 0.5 - 0.25
                sv = ((s >> 1) & 1) * 0.5 - 0.25
                d1 = su * x1 + sv * y1
                d2 = su * x2 + sv * y2
                dc = ti.math.vec3(-d1 - d2, d1, d2)
                for i in ti.static(range(3)):
                    if dc[i] < 0.0:
                        k = ti.min(k, ti.max(c[i], 0.0) / -dc[i])
    return dpx * k, dpy * k


@ti.func
def _sub_pixel_origin(spos, dpx, dpy, s):
    """One of four 2x2-grid sub-pixel positions on the surface, indexed at
    runtime.

    Computed arithmetically rather than from a table because the shadow loop's
    sample index is a runtime value. The four offsets are exactly where
    ``anti_alias_level = 2`` puts its supersamples; a larger sample count cycles
    through them with a different soft-shadow fan angle each time.
    """
    su = ti.cast(s & 1, ti.f32) * 0.5 - 0.25
    sv = ti.cast((s >> 1) & 1, ti.f32) * 0.5 - 0.25
    return spos + su * dpx + sv * dpy


@ti.func
def _spawn_pool_ray(rs_ro: ti.template(), rs_rd: ti.template(),
                    rs_acc: ti.template(), rs_sca: ti.template(),
                    rs_int: ti.template(), rs_pix: ti.template(),
                    rs_alloc: ti.template(),
                    orig, dirn, wt, dist, bounces_left, processed, pixel, r,
                    accum_row,
                    compact: ti.template(), ior_stack: ti.template(),
                    refracting: ti.template(), medium_ior, entering):
    """Append one continuation ray to the tile's shared ray pool.

    Overflow is not an error and not a per-pixel cap: the host retries the whole
    tile with half as many primary rays, which doubles the continuation headroom
    (``refract_initial_pool_ratio``). A dropped slot silently loses that
    branch's contribution, which is why the caller must not have already
    committed throughput to it.

    The nested-IOR stack columns (module head of ``wavefront_kernels_taichi``)
    are written here too whenever the gate is on, through the same
    ``_write_ior_stack`` the bounce shader uses: ``refracting`` 1 pushes or
    pops the hit medium according to ``entering``, 0 copies the parent's stack
    (a reflection stays in the medium it was in). The parent is row ``r``,
    which for this kernel's only callers -- the sheet resolve -- is a PRIMARY
    ray, whose stack the host zeroed. ``entering`` still has to be passed
    rather than assumed: a partially covering glass fragment lets the primary
    walk on to the same solid's BACK face, and pushing there would record a
    medium the ray is leaving. Compiles out with the gate off.

    ``accum_row`` is which row of ``pix_accum`` this continuation retires into,
    and is ``r`` for every branch that belongs to the pixel itself. It is a
    parameter rather than ``r`` because the split-sum glossy route
    (``DESIGN_glossy_prefilter.md``) routes ONE continuation per pixel into a
    SECOND accumulator row, so its radiance can be prefiltered before it is
    composited; every ray that continuation goes on to spawn inherits the row
    through ``rs_int[:, 4]`` and stays part of the same reflection.
    """
    c, have_slot = _reserve_continuation_slot(rs_alloc, rs_ro.shape[0])
    if have_slot:
        for k in ti.static(range(3)):
            rs_ro[c, k] = orig[k]
            rs_rd[c, k] = dirn[k]
        for k in ti.static(range(4)):
            rs_acc[c, k] = 0.0
        rs_sca[c, 0] = wt[0]
        rs_sca[c, 1] = 0.0
        rs_sca[c, 2] = 1e30
        rs_sca[c, 3] = -1e30
        rs_sca[c, 4] = dist
        rs_sca[c, 5] = wt[1]
        rs_sca[c, 6] = wt[2]
        _write_ior_stack(rs_sca, r, c, medium_ior, entering, refracting,
                         ior_stack)
        _reset_shell_segment(rs_sca, c)
        rs_int[c, 0] = bounces_left
        rs_int[c, 1] = processed
        rs_int[c, 2] = _ACTIVE
        rs_int[c, 3] = 0
        rs_pix[c] = pixel
        if ti.static(compact):
            rs_int[c, 4] = accum_row
    return have_slot




@ti.func
def _tri_shadow_normals(f, prim, a, b, rd,
                        tri_pos: ti.template(), tri_norm: ti.template(),
                        tri_uvs: ti.template(), tri_tex_meta: ti.template(),
                        textures: ti.template(), num_colored_triangles,
                        one_sided):
    """Shading + geometric face normals of a triangle hit, oriented for a
    shadow-ray origin exactly as ``wavefront_shade``'s inline shadow block (both
    defer to :func:`_orient_hit_normals_sided`; ``one_sided`` is the hit's
    surface declaration, 0 for the viewer-oriented convention).
    """
    w0 = 1.0 - a - b
    snrm = _tri_normal_g(0, f, prim, w0, a, b, tri_norm, tri_pos, tri_uvs,
                         tri_tex_meta, textures, num_colored_triangles)
    tp = f % tri_pos.shape[0]
    v0 = ti.math.vec3(tri_pos[tp, prim, 0], tri_pos[tp, prim, 1],
                      tri_pos[tp, prim, 2])
    v1 = ti.math.vec3(tri_pos[tp, prim, 3], tri_pos[tp, prim, 4],
                      tri_pos[tp, prim, 5])
    v2 = ti.math.vec3(tri_pos[tp, prim, 6], tri_pos[tp, prim, 7],
                      tri_pos[tp, prim, 8])
    fnrm = (v1 - v0).cross(v2 - v0)
    return _orient_hit_normals_sided(snrm, fnrm, rd, one_sided)




@ti.func
def _shadow_fan_cell(
        e, li,
        event_pos: ti.template(),
        event_snrm: ti.template(),
        event_fnrm: ti.template(),
        event_frame: ti.template(),
        event_msk: ti.template(),
        t_nodes: ti.template(),
        t_node_miss: ti.template(),
        t_leaf_prim: ti.template(),
        t_leaf_tspan: ti.template(),
        t_first_leaf,
        tri_pos: ti.template(),
        tri_colors: ti.template(),
        tri_uvs: ti.template(),
        tri_tex_meta: ti.template(),
        textures: ti.template(),
        tri_extra: ti.template(),
        num_colored_triangles,
        b_nodes: ti.template(),
        b_node_miss: ti.template(),
        b_leaf_prim: ti.template(),
        b_leaf_tspan: ti.template(),
        b_first_leaf,
        circuit_meta: ti.template(),
        circuit_colors: ti.template(),
        circuit_border_colors: ti.template(),
        edges_2d: ti.template(),
        edge_accel: ti.template(),
        light_pos: ti.template(),
        light_col: ti.template(),
        num_lights,
        pixel_world_scale: ti.template(),
        cam_proj: ti.template(),
        layer_offset_triangles,
        refit: ti.template(),
        has_tri: ti.template(),
        has_bez: ti.template(),
        event_dp: ti.template(),
        event_toff: ti.template(),
        sec_aa: ti.template(),
        shadow_vis: ti.template(),
        shadow_anyhit: ti.template(),
        tri_obj: ti.template(),
        event_src_prim: ti.template(),
        eps_self,
        eps_near,
        shadow_identity: ti.template(),
        shadow_term: ti.template(),
        adaptive_taps: ti.template(),
        secondary,
        emit_queue: ti.template(), queue_data: ti.template(),
        queue_valid: ti.template(), queue_base, queue_first, queue_events):
    """Shared fan geometry for serial tracing and parallel queue emission.

    Sampling, coverage masks, jitter, zero-radiance and horizon tests are kept
    in one body. The parallel path changes traversal scheduling, not rays.
    """
    f = event_frame[e]
    ff = ti.cast(f, ti.f32)
    spos = ti.math.vec3(event_pos[e, 0], event_pos[e, 1], event_pos[e, 2])
    snrm = ti.math.vec3(event_snrm[e, 0], event_snrm[e, 1],
                        event_snrm[e, 2])
    fnrm = ti.math.vec3(event_fnrm[e, 0], event_fnrm[e, 1],
                        event_fnrm[e, 2])
    # The face-normal lift every shadow origin keeps. With the
    # shadow-terminator offset on (shadow_term == 1) the event build has
    # also stored each event's Hanika displacement onto the smooth
    # surface its vertex normals imply (_shadow_terminator_delta,
    # RENDERER_WORK_QUEUE.md item 20); ``lifted`` records whether this
    # origin GENUINELY moved (a flat facet's delta is exactly zero by
    # construction -- _shadow_terminator_delta short-circuits a constant
    # normal field), and is what licenses the horizon-cull relaxation in
    # the sample loop below. shadow_term == 2 lifts nothing -- that
    # diagnostic arm exists to show what relaxing alone does.
    sorigin = _offset_ray_origin(spos, fnrm)
    lifted = 0
    if ti.static(shadow_term != 0):
        if ti.static(shadow_term == 1):
            dx = event_toff[e, 0]
            dy = event_toff[e, 1]
            dz = event_toff[e, 2]
            if (dx != 0.0) or (dy != 0.0) or (dz != 0.0):
                sorigin = sorigin + ti.math.vec3(dx, dy, dz)
                lifted = 1
        else:
            lifted = 1
    dpx = ti.math.vec3(0.0, 0.0, 0.0)
    dpy = ti.math.vec3(0.0, 0.0, 0.0)
    if ti.static(sec_aa > 1):
        dpx = ti.math.vec3(event_dp[e, 0], event_dp[e, 1], event_dp[e, 2])
        dpy = ti.math.vec3(event_dp[e, 3], event_dp[e, 4], event_dp[e, 5])
    tl = f % light_pos.shape[0]
    # The event's material pipeline id (packed above the 4-bit sub-pixel
    # position mask at build time). In every built-in stage a zero-color
    # light row (not yet spawned, or despawned) contributes nothing
    # whatever its visibility -- every lit stage's terms carry the light
    # color as a factor -- so such rows keep their all-lit default
    # without tracing. Only user pipelines, which may read visibility
    # arbitrarily, keep the exact fan for every light.
    pid_e = event_msk[e] >> 8
    src_sid = -1
    src_prim = -1
    if ti.static(shadow_identity != 0):
        # The event's source triangle, and the mesh it belongs to. Both
        # ride their own array rather than spare mask bits: a primitive
        # index does not fit in the 16 bits that were free, and keeping
        # ``event_msk`` untouched leaves the material pipeline id above
        # bit 8 exactly as the resolve wrote it.
        src_prim = ti.cast(event_src_prim[e], ti.i32)
        if src_prim >= 0:
            src_sid = ti.cast(
                tri_obj[f % tri_obj.shape[0], src_prim], ti.i32)
    fan_exact = 1
    fan_geom = 0
    if pid_e < _USER_PIPELINE_BASE:
        fan_exact = 0
        # Geometric zero-radiance culling is valid for EVERY built-in
        # stage: each one's vis-multiplied terms carry lc, so a culled
        # fan's all-lit default multiplies zero either way (see
        # _light_zero_radiance).
        fan_geom = 1
    visibility = ti.math.vec3(1.0)
    lp = ti.math.vec3(light_pos[tl, li, 0], light_pos[tl, li, 1],
                      light_pos[tl, li, 2])
    ltype = 0
    radius = 0.0
    hu = 0.0
    hv = 0.0
    fan_col = 0
    if light_col.shape[2] > 3:
        ltype = ti.cast(light_col[tl, li, 3] + 0.5, ti.i32)
        if light_col.shape[2] > 11:
            radius = light_col[tl, li, 11]
            # The row's budgeted soft fan (scene_builder._soft_fan_sizes):
            # column 16 at a primary hit, 17 at a secondary one; 0 keeps
            # the compile-time fan below, unjittered.
            if light_col.shape[2] > 17:
                fan_col = ti.cast(
                    light_col[tl, li, 16 + secondary] + 0.5, ti.i32)
            # A rect-area row carries its CELL's half-extents along the
            # emitter plane's own axes. The ltype guard is load-bearing,
            # not defensive: columns 9/10 are a spot light's cone cosines
            # there, and reading them unguarded would turn every spot
            # light into a rect emitter.
            if ltype == _LT_AREA_SAMPLE:
                hu = light_col[tl, li, 9]
                hv = light_col[tl, li, 10]
    to_light = lp - spos
    ldist = to_light.norm()
    wi = ti.math.vec3(0.0, 0.0, 0.0)
    valid = 0
    if ltype == _LT_DIRECTIONAL:
        wi = -ti.math.vec3(light_col[tl, li, 6],
                           light_col[tl, li, 7],
                           light_col[tl, li, 8])
        ldist = 1e7
        valid = 1
    elif (ltype != _LT_AMBIENT) and (ltype != _LT_HEMISPHERE) \
            and (ltype != _LT_ENV_SH) and (ldist > 1e-5):
        wi = to_light / ldist
        valid = 1
    # A light past its range, a fragment outside a spot cone, an
    # area sample's backface: exactly zero radiance here, so the
    # fan's result multiplies zero. Skipping leaves the event's
    # all-lit default, exactly like the zero-color skip below.
    if (valid == 1) and (fan_geom == 1):
        if _light_zero_radiance(light_col, tl, li, ltype, to_light,
                                ldist) == 1:
            valid = 0
    if (valid == 1) and ((fan_exact == 1)
                         or (light_col[tl, li, 0] != 0.0)
                         or (light_col[tl, li, 1] != 0.0)
                         or (light_col[tl, li, 2] != 0.0)):
        ns = 1
        b1 = ti.math.vec3(0.0, 0.0, 0.0)
        b2 = ti.math.vec3(0.0, 0.0, 0.0)
        if radius > 0.0:
            ns = SOFT_SHADOW_SAMPLES
            if fan_col > 0:
                ns = fan_col
            if (hu > 0.0) or (hv > 0.0):
                # Rect emitter: the fan samples INSIDE this row's own
                # cell, in the light's own plane -- b1 is the packed right
                # axis and b2 the up axis recovered exactly as _rect_axes
                # builds it. The offsets do not depend on wi, so a moving
                # sub-pixel origin needs no basis rebuild.
                b1 = ti.math.vec3(light_col[tl, li, 12],
                                  light_col[tl, li, 13],
                                  light_col[tl, li, 14])
                b2 = ti.math.vec3(light_col[tl, li, 6],
                                  light_col[tl, li, 7],
                                  light_col[tl, li, 8]).cross(b1)
            else:
                aref = ti.math.vec3(1.0, 0.0, 0.0)
                if ti.abs(wi[0]) > 0.9:
                    aref = ti.math.vec3(0.0, 1.0, 0.0)
                b1 = wi.cross(aref).normalized()
                b2 = wi.cross(b1)
        if ti.static(sec_aa > 1):
            # A hard light needs the sub-pixel positions to be separate
            # rays; a legacy soft fan already has enough rays and just
            # spreads its fan over them. A BUDGETED soft fan keeps its
            # count -- raising it back to four would undo the budget --
            # and rotates through the covered positions below instead.
            if fan_col == 0:
                ns = ti.max(ns, 4)

        occ_sum = ti.math.vec3(0.0)
        n_valid = 0.0
        # Adaptive taps (adaptive_taps, hard lights under sec_aa): the
        # 2x2 sub-pixel positions are visited diagonal pair first (0, 3,
        # then 2, 1). When both diagonal taps were traced and agree
        # exactly -- the interior of a lit or of a shadowed region, which
        # is nearly every event -- the other two are taken as equal and
        # the fan is two rays instead of four. Only a pixel whose shadow
        # edge leaves the diagonal pair agreeing while an off-diagonal
        # tap differs changes: it keeps 0 or 1 where it had a quarter
        # step. A soft light keeps its full golden-angle fan.
        adaptive_fan = 0
        if ti.static(sec_aa > 1 and adaptive_taps != 0):
            if radius <= 0.0:
                adaptive_fan = 1
        occ_first = ti.math.vec3(0.0)
        # Budgeted fan (fan_col > 0): the per-event rotation that keeps a
        # few rays from banding (rect: R2 offsets; disk: azimuth and
        # radial stratum), and the first covered sub-pixel position the
        # rotation through positions starts at. Legacy fans keep literal
        # zeros here, so their arithmetic is unchanged.
        ju = 0.0
        jv = 0.0
        r_j = 0.5
        ang_j = 0.0
        first_pos = 0
        if fan_col > 0:
            # Keyed on the event's cell of a one-to-two-pixel world grid, so a
            # sub-pixel camera move does not re-roll it (see
            # _shadow_fan_jitter; a larger one still does).
            cell = 0.0
            if ti.static(sec_aa > 1):
                cell = _shadow_jitter_cell(dpx, dpy)
            ju, jv = _shadow_fan_jitter(spos, li, cell)
            r_j = jv
            ang_j = _TWO_PI * ju
            if ti.static(sec_aa > 1):
                first_pos = _first_covered_position(event_msk[e])
        for sk in range(ns):
            s = sk
            if adaptive_fan == 1:
                s = (3 * sk) & 3
            wis = wi
            ldn = ldist
            ok = 1
            sorg = sorigin
            if ti.static(sec_aa > 1):
                sp = s & 3
                if fan_col > 0:
                    # A budgeted fan visits the covered sub-pixel
                    # positions in turn rather than skipping the
                    # uncovered ones: a one-ray fan on a silhouette
                    # pixel must still trace its one ray.
                    sp = (first_pos + s) & 3
                    if ((event_msk[e] >> sp) & 1) == 0:
                        sp = first_pos
                sorg = _sub_pixel_origin(sorigin, dpx, dpy, sp)
                if ((event_msk[e] >> sp) & 1) == 0:
                    ok = 0
            off = ti.math.vec3(0.0, 0.0, 0.0)
            if radius > 0.0:
                if (hu > 0.0) or (hv > 0.0):
                    # R2 sequence across the cell: s = 0 is exactly the
                    # cell centre, so a one-sample fan degenerates to
                    # today's ray.
                    u = 0.5 + _R2_SEQUENCE_A1 * s + ju
                    v = 0.5 + _R2_SEQUENCE_A2 * s + jv
                    ru = 2.0 * (u - ti.floor(u)) - 1.0
                    rv = 2.0 * (v - ti.floor(v)) - 1.0
                    off = b1 * (hu * ru) + b2 * (hv * rv)
                else:
                    ang = _GOLDEN_ANGLE * s + ang_j
                    rr = radius * ti.sqrt(
                        (ti.cast(s, ti.f32) + r_j)
                        / ti.cast(ns, ti.f32))
                    off = (ti.cos(ang) * b1 + ti.sin(ang) * b2) * rr
                if ltype == _LT_DIRECTIONAL:
                    wis = (wi + off).normalized()
            if ltype != _LT_DIRECTIONAL:
                # Moving the origin over the pixel changes both the
                # direction and finite distance to a point/spot/area
                # emitter. Retaining the centre ray here makes the
                # samples non-convergent and can trace past the light.
                tls = lp + off - sorg
                ldn = tls.norm()
                if ldn > 1e-5:
                    wis = tls / ldn
                else:
                    ok = 0
            # Horizon cull. Today's guard is BOTH normals, and the face
            # normal's > 1e-3 term is precisely what suppresses the
            # terminator band today (a near-tangent ray strikes a
            # neighbouring facet far from the origin: RENDERER_WORK_QUEUE.md
            # item 20). Where this sample's origin GENUINELY moved onto
            # the smooth surface its vertex normals imply (lifted == 1,
            # shadow_term != 0), the face normal's horizon is not that
            # surface's, so the fnrm term drops and only the shading
            # normal's cull remains. A flat facet stores an exactly-zero
            # delta and the gate-off path never lifts, so both keep the
            # two-sided test EXACTLY as written first.
            horizon_ok = (fnrm.dot(wis) > 1e-3) \
                and (snrm.dot(wis) > 1e-4)
            if ti.static(shadow_term != 0):
                if lifted == 1:
                    horizon_ok = snrm.dot(wis) > 1e-4
            if (ok == 1) and horizon_ok:
                if ti.static(emit_queue):
                    r = queue_base + sk * queue_events + e - queue_first
                    for k in ti.static(range(3)):
                        queue_data[r, k] = sorg[k]
                        queue_data[r, 3 + k] = wis[k]
                    queue_data[r, 6] = _shadow_tmax(sorg, wis, ldn)
                    queue_valid[r] = 1 + adaptive_fan
                else:
                    n_valid += 1.0
                    occ = _shadow_occluded(
                        refit, shadow_anyhit, sorg, wis, f, ff,
                        _shadow_tmax(sorg, wis, ldn),
                        pixel_world_scale[
                            f % pixel_world_scale.shape[0]], 0.0,
                        _pixel_size_base(cam_proj, f),
                        layer_offset_triangles,
                        has_tri, has_bez,
                        t_nodes, t_node_miss, t_leaf_prim, t_leaf_tspan,
                        t_first_leaf, tri_pos, tri_colors, tri_uvs,
                        tri_tex_meta, textures, tri_extra,
                        num_colored_triangles,
                        b_nodes, b_node_miss, b_leaf_prim, b_leaf_tspan,
                        b_first_leaf, circuit_meta, circuit_colors,
                        circuit_border_colors, edges_2d, edge_accel,
                        src_sid, src_prim, eps_self, eps_near,
                        tri_obj, shadow_identity)
                    occ_sum += occ
                    if adaptive_fan == 1:
                        if sk == 0:
                            occ_first = occ
                        elif sk == 1:
                            if (n_valid == 2.0) and (occ[0] == occ_first[0]) \
                                    and (occ[1] == occ_first[1]) \
                                    and (occ[2] == occ_first[2]):
                                # Both diagonal taps traced and equal: the
                                # remaining two are taken as the same.
                                occ_sum += occ_sum
                                n_valid += n_valid
                                break
        if n_valid > 0.0:
            # Per-channel visibility; the soft-shadow fan still averages
            # over the SCALAR sample count.
            visibility = ti.math.vec3(1.0) - occ_sum / n_valid
    if ti.static(not emit_queue):
        # RGB payload: one (event, light) cell carries a triple, channel-last
        # (see raster_pipeline's allocation).
        for c in ti.static(range(3)):
            shadow_vis[e, li, c] = visibility[c]


@ti.kernel
def raster_shadow_trace_arena(
        num_events: int,
        event_pos: ti.types.ndarray(), event_snrm: ti.types.ndarray(),
        event_fnrm: ti.types.ndarray(), event_frame: ti.types.ndarray(),
        event_msk: ti.types.ndarray(),
        t_nodes: NODE_ARG, t_first_leaf: int,
        num_colored_triangles: ti.i32,
        b_nodes: NODE_ARG, b_first_leaf: int,
        num_lights: int, layer_offset_triangles: ti.f32,
        refit: ti.template(),
        has_tri: ti.template(), has_bez: ti.template(),
        event_dp: ti.types.ndarray(), event_toff: ti.types.ndarray(),
        sec_aa: ti.template(),
        shadow_vis: ti.types.ndarray(), shadow_anyhit: ti.template(),
        event_src_prim: ti.types.ndarray(),
        eps_self: ti.f32, eps_near: ti.f32,
        shadow_identity: ti.template(),
        # Shadow-terminator gate (rt_settings.shadow_terminator_mode()):
        # 0 = today's origin and today's horizon cull; 1 = add each event's
        # stored Hanika displacement (``event_toff``, written by the mode-1
        # sheet_resolve_shade build) AND relax the cull where it moved;
        # 2 = relax the cull WITHOUT moving anything (diagnostic arm).
        shadow_term: ti.template(),
        # Adaptive sub-pixel fan (rt_settings.shadow_adaptive_taps): under
        # ``sec_aa`` a hard light's four taps are visited diagonal pair first
        # and the other two are skipped when that pair agrees exactly.
        adaptive_taps: ti.template(),
        # Which of the light row's two budgeted fan sizes applies: 0 at a
        # primary (sheet-resolved) hit, 1 at a secondary (bounce) hit. See
        # rt_settings.shadow_ray_budget / shadow_bounce_rays.
        secondary: ti.i32,
        light_major: ti.template(),
        projected: ti.template(),
        arena_f32: ti.types.ndarray(),
        arena_i32: ti.types.ndarray(),
        aoff: ti.types.ndarray(),
        ashp: ti.types.ndarray()):
    """Trace the dedicated sparse any-hit shadow queue exactly.

    A zero-radius point/spot/directional light emits one hard-shadow ray.
    Non-zero emitter radii use the same fixed golden-angle fan as the classic
    wavefront shader. An area light's packed rows carry their emitter CELL's
    half-extents and the rectangle's right axis, and each row's fan places
    its samples inside that cell, in the light's own plane: the per-row
    visibility integrates the cell instead of testing its centre point, so
    summing the rows gives a continuous penumbra rather than a stack of hard
    ones (gated host-side by area_light_soft_shadows -- with it off these
    rows pack zeros and take today's single hard ray).

    ``shadow_identity`` (``shadow_identity_reject``, DESIGN_mesh_identity_open.md
    ssI) engages identity-aware rejection. ``event_src_prim`` carries each
    event's SOURCE triangle (-1 for a bezier-sourced event, which keeps the
    old absolute epsilon), ``tri_obj`` answers "which mesh is this hit on"
    during traversal, and ``eps_self`` / ``eps_near`` are the acceptance
    floors the host has already scaled to the batch's scene size. The source
    triangle keeps ``eps_self``, another triangle of the same mesh keeps
    ``eps_near``, and any other mesh is accepted down to ``t > 0`` -- so a
    contact shadow survives the epsilon that used to erase it. See
    :func:`_shadow_identity_t_min` for the tiers. With
    ``shadow_identity == 0`` every event keeps exactly the old epsilon.

    ``sec_aa`` (``analytic_aa_secondary_samples``): visibility is a BINARY query,
    so analytic coverage cannot antialias a shadow edge however exact the
    geometry is -- the stair steps just move from the silhouette to the shadow
    (ss7). With it on, the query point moves over the pixel instead: four
    sub-pixel positions on the shading surface, from the world-space footprint
    the event build stored (``event_dp``), averaged. For a HARD light that is 4
    rays instead of 1 at exactly the positions ``anti_alias_level = 2`` samples;
    for a soft one the existing fan is simply spread over those positions too,
    which costs nothing.
    """
    # Arena-bound parameters (arena_args_taichi): each name is
    # rebound to a window into its dtype's buffer, at the offset
    # the host wrote into aoff. Order is _RASTER_SHADOW_TRACE_ARENA's.
    t_node_miss = ti.static(ArenaView(arena_i32, aoff[0], (ashp[0],)))
    t_leaf_prim = ti.static(ArenaView(arena_i32, aoff[1], (ashp[1],)))
    t_leaf_tspan = ti.static(ArenaView(arena_i32, aoff[2], (ashp[2],)))
    tri_pos = ti.static(ArenaView(arena_f32, aoff[3], (ashp[3], ashp[4], ashp[5])))
    tri_colors = ti.static(ArenaView(
        arena_f32, aoff[4], (ashp[6], ashp[7], ashp[8], ashp[9])))
    tri_uvs = ti.static(ArenaView(arena_f32, aoff[5], (ashp[10], ashp[11], ashp[12])))
    tri_tex_meta = ti.static(ArenaView(arena_i32, aoff[6], (ashp[13], ashp[14])))
    textures = ti.static(ArenaView(arena_f32, aoff[7], (ashp[15], ashp[16], ashp[17])))
    tri_extra = ti.static(ArenaView(arena_f32, aoff[8], (ashp[18], ashp[19], ashp[20])))
    b_node_miss = ti.static(ArenaView(arena_i32, aoff[9], (ashp[21],)))
    b_leaf_prim = ti.static(ArenaView(arena_i32, aoff[10], (ashp[22],)))
    b_leaf_tspan = ti.static(ArenaView(arena_i32, aoff[11], (ashp[23],)))
    circuit_meta = ti.static(ArenaView(
        arena_f32, aoff[12], (ashp[24], ashp[25], ashp[26])))
    circuit_colors = ti.static(ArenaView(
        arena_f32, aoff[13], (ashp[27], ashp[28], ashp[29], ashp[30])))
    circuit_border_colors = ti.static(ArenaView(
        arena_f32, aoff[14], (ashp[31], ashp[32], ashp[33], ashp[34])))
    edges_2d = ti.static(ArenaView(arena_f32, aoff[15], (ashp[35], ashp[36], ashp[37])))
    edge_accel = ti.static(ArenaView(arena_i32, aoff[16], (ashp[38],)))
    light_pos = ti.static(ArenaView(
        arena_f32, aoff[17], (ashp[39], ashp[40], ashp[41])))
    light_col = ti.static(ArenaView(
        arena_f32, aoff[18], (ashp[42], ashp[43], ashp[44])))
    pixel_world_scale = ti.static(ArenaView(arena_f32, aoff[19], (ashp[45],)))
    tri_obj = ti.static(ArenaView(arena_i32, aoff[20], (ashp[46], ashp[47])))
    cam_proj_table = ti.static(ArenaView(arena_f32, aoff[21], (ashp[48], ashp[49])))
    cam_proj = ti.static(CameraProjection(cam_proj_table, projected))
    # One thread per (event, light) cell: every cell's fan is independent
    # (its result lands in its own ``shadow_vis[e, li]`` and all per-light
    # state initializes inside the body), so flattening the light loop into
    # the launch grid multiplies parallelism by ``num_lights`` while keeping
    # each cell's arithmetic -- including the serial golden-angle sample loop,
    # whose float accumulation order is part of the output contract --
    # bit-for-bit identical. The per-event setup is recomputed per cell
    # (pure loads, negligible next to a single occlusion march).
    for idx in range(num_events * num_lights):
        e = idx // num_lights
        li = idx - e * num_lights
        if ti.static(light_major):
            # One warp follows one emitter across nearby source events. The
            # source sort and per-cell fan remain exactly the same, but lights
            # with different types/directions no longer diverge within a warp.
            li = idx // num_events
            e = idx - li * num_events
        _shadow_fan_cell(
            e, li,
            event_pos, event_snrm, event_fnrm, event_frame,
            event_msk, t_nodes, t_node_miss, t_leaf_prim,
            t_leaf_tspan, t_first_leaf, tri_pos, tri_colors,
            tri_uvs, tri_tex_meta, textures, tri_extra,
            num_colored_triangles, b_nodes, b_node_miss, b_leaf_prim,
            b_leaf_tspan, b_first_leaf, circuit_meta, circuit_colors,
            circuit_border_colors, edges_2d, edge_accel, light_pos,
            light_col, num_lights, pixel_world_scale, cam_proj,
            layer_offset_triangles, refit, has_tri, has_bez, event_dp,
            event_toff, sec_aa, shadow_vis, shadow_anyhit,
            tri_obj, event_src_prim, eps_self, eps_near,
            shadow_identity, shadow_term, adaptive_taps, secondary,
            False, event_pos, event_msk, 0, 0, 0)



#: What ``raster_shadow_trace`` binds through the arena, in offset-table order:
#: ``aoff[i]`` is the i-th entry's element offset into its dtype's
#: buffer and ``ashp`` holds their shapes end to end. The kernel's
#: binding prologue reads those slots by literal index, so the two
#: are one edit apart -- ``tests/unit_tests/test_arena_args.py``
#: fails if they stop agreeing.
_RASTER_SHADOW_TRACE_ARENA = (
    ("t_node_miss", "i32", 1),
    ("t_leaf_prim", "i32", 1),
    ("t_leaf_tspan", "i32", 1),
    ("tri_pos", "f32", 3),
    ("tri_colors", "f32", 4),
    ("tri_uvs", "f32", 3),
    ("tri_tex_meta", "i32", 2),
    ("textures", "f32", 3),
    ("tri_extra", "f32", 3),
    ("b_node_miss", "i32", 1),
    ("b_leaf_prim", "i32", 1),
    ("b_leaf_tspan", "i32", 1),
    ("circuit_meta", "f32", 3),
    ("circuit_colors", "f32", 4),
    ("circuit_border_colors", "f32", 4),
    ("edges_2d", "f32", 3),
    ("edge_accel", "i32", 1),
    ("light_pos", "f32", 3),
    ("light_col", "f32", 3),
    ("pixel_world_scale", "f32", 1),
    ("tri_obj", "i32", 2),
    ("cam_proj_table", "f32", 2),
)

#: The argument list every launch site passes. Unchanged by the
#: conversion -- that is the point of the wrapper below.
_RASTER_SHADOW_TRACE_PARAMS = (
    "num_events", "event_pos", "event_snrm", "event_fnrm", "event_frame",
    "event_msk", "t_nodes", "t_node_miss", "t_leaf_prim", "t_leaf_tspan",
    "t_first_leaf", "tri_pos", "tri_colors", "tri_uvs", "tri_tex_meta",
    "textures", "tri_extra", "num_colored_triangles", "b_nodes",
    "b_node_miss", "b_leaf_prim", "b_leaf_tspan", "b_first_leaf",
    "circuit_meta", "circuit_colors", "circuit_border_colors", "edges_2d",
    "edge_accel", "light_pos", "light_col", "num_lights", "pixel_world_scale",
    "layer_offset_triangles", "refit", "has_tri", "has_bez", "event_dp",
    "event_toff", "sec_aa", "shadow_vis", "shadow_anyhit", "tri_obj",
    "event_src_prim", "eps_self", "eps_near", "shadow_identity",
    "shadow_term", "adaptive_taps", "secondary", "light_major", "cam_proj_table",
    "projected",
)

_raster_shadow_trace_launch = arena_packed(
    __name__, "raster_shadow_trace_arena",
    _RASTER_SHADOW_TRACE_PARAMS, _RASTER_SHADOW_TRACE_ARENA)


def raster_shadow_trace(*args):
    """Pack the arena-bound arguments, then launch ``raster_shadow_trace_arena``.

    Takes the argument list this kernel had before it was converted to
    the arena calling convention, so no launch site changed; see
    `arena_args_taichi`.
    """
    return _raster_shadow_trace_launch(*args)


@ti.kernel
def deferred_shadow_trace_arena(
        num_rows: int,
        event_count: ti.types.ndarray(), event_list: ti.types.ndarray(),
        event_pos: ti.types.ndarray(), event_snrm: ti.types.ndarray(),
        event_fnrm: ti.types.ndarray(), event_frame: ti.types.ndarray(),
        event_msk: ti.types.ndarray(),
        t_nodes: NODE_ARG, t_first_leaf: int,
        num_colored_triangles: ti.i32,
        b_nodes: NODE_ARG, b_first_leaf: int,
        num_lights: int, traced_lights: int, layer_offset_triangles: ti.f32,
        refit: ti.template(),
        has_tri: ti.template(), has_bez: ti.template(),
        event_dp: ti.types.ndarray(), event_toff: ti.types.ndarray(),
        sec_aa: ti.template(),
        shadow_vis: ti.types.ndarray(), shadow_anyhit: ti.template(),
        event_src_prim: ti.types.ndarray(),
        eps_self: ti.f32, eps_near: ti.f32,
        shadow_identity: ti.template(),
        shadow_term: ti.template(),
        adaptive_taps: ti.template(),
        secondary: ti.i32,
        light_major: ti.template(),
        projected: ti.template(),
        arena_f32: ti.types.ndarray(),
        arena_i32: ti.types.ndarray(),
        aoff: ti.types.ndarray(),
        ashp: ti.types.ndarray()):
    """``raster_shadow_trace`` for the bounce drain's events, sized on the device.

    The bounce drain (``tracer._drain_sparse_secondary``) used to read back how
    many of ``wavefront_shadow_events``' rows it accepted -- a ``nonzero`` that
    drained the stream every bounce -- and gather each payload array down to
    them. Here nothing is read back: ``event_list[:event_count[0]]`` holds the
    accepted rows (``wavefront_kernels_taichi.shadow_event_scatter``), the
    launch covers ``num_rows`` -- every row the events kernel wrote, an upper
    bound on the count -- and a cell past the count does nothing.

    Every cell is ``_shadow_fan_cell``, the body ``raster_shadow_trace`` runs,
    called with the event's ROW so the payload is read where the events kernel
    wrote it, and with ``shadow_vis`` seen at the event's compacted SLOT
    (:class:`RowView`), where the shade kernel's map points. ``event_dp`` and
    ``event_src_prim`` are one-row tables every event shares, as the zero
    footprint and the ``-1`` source the drain's events have always carried
    were. The fan, its arithmetic and its acceptance are the ones the old
    launch ran, so every cell's visibility is the same.

    ``traced_lights`` is ``min(num_lights, vis_lights)``: the shade reads no
    light past its payload's slots, so none is traced. ``num_lights`` is the
    batch's own count, which the fan body is given as before.
    """
    # Arena-bound parameters (arena_args_taichi): each name is
    # rebound to a window into its dtype's buffer, at the offset
    # the host wrote into aoff. Order is _DEFERRED_SHADOW_TRACE_ARENA's.
    t_node_miss = ti.static(ArenaView(arena_i32, aoff[0], (ashp[0],)))
    t_leaf_prim = ti.static(ArenaView(arena_i32, aoff[1], (ashp[1],)))
    t_leaf_tspan = ti.static(ArenaView(arena_i32, aoff[2], (ashp[2],)))
    tri_pos = ti.static(ArenaView(arena_f32, aoff[3], (ashp[3], ashp[4], ashp[5])))
    tri_colors = ti.static(ArenaView(
        arena_f32, aoff[4], (ashp[6], ashp[7], ashp[8], ashp[9])))
    tri_uvs = ti.static(ArenaView(arena_f32, aoff[5], (ashp[10], ashp[11], ashp[12])))
    tri_tex_meta = ti.static(ArenaView(arena_i32, aoff[6], (ashp[13], ashp[14])))
    textures = ti.static(ArenaView(arena_f32, aoff[7], (ashp[15], ashp[16], ashp[17])))
    tri_extra = ti.static(ArenaView(arena_f32, aoff[8], (ashp[18], ashp[19], ashp[20])))
    b_node_miss = ti.static(ArenaView(arena_i32, aoff[9], (ashp[21],)))
    b_leaf_prim = ti.static(ArenaView(arena_i32, aoff[10], (ashp[22],)))
    b_leaf_tspan = ti.static(ArenaView(arena_i32, aoff[11], (ashp[23],)))
    circuit_meta = ti.static(ArenaView(
        arena_f32, aoff[12], (ashp[24], ashp[25], ashp[26])))
    circuit_colors = ti.static(ArenaView(
        arena_f32, aoff[13], (ashp[27], ashp[28], ashp[29], ashp[30])))
    circuit_border_colors = ti.static(ArenaView(
        arena_f32, aoff[14], (ashp[31], ashp[32], ashp[33], ashp[34])))
    edges_2d = ti.static(ArenaView(arena_f32, aoff[15], (ashp[35], ashp[36], ashp[37])))
    edge_accel = ti.static(ArenaView(arena_i32, aoff[16], (ashp[38],)))
    light_pos = ti.static(ArenaView(
        arena_f32, aoff[17], (ashp[39], ashp[40], ashp[41])))
    light_col = ti.static(ArenaView(
        arena_f32, aoff[18], (ashp[42], ashp[43], ashp[44])))
    pixel_world_scale = ti.static(ArenaView(arena_f32, aoff[19], (ashp[45],)))
    tri_obj = ti.static(ArenaView(arena_i32, aoff[20], (ashp[46], ashp[47])))
    cam_proj_table = ti.static(ArenaView(arena_f32, aoff[21], (ashp[48], ashp[49])))
    cam_proj = ti.static(CameraProjection(cam_proj_table, projected))
    shared_dp = ti.static(RowView(event_dp, 0))
    shared_src = ti.static(RowView(event_src_prim, 0))
    # One (event, light) cell per iteration, laid out as raster_shadow_trace
    # lays them out over its exact count; a cell past the device count is
    # idle. Read per cell rather than once above the loop: the loop is the
    # kernel's parallel task, and the count is a value on the device.
    for idx in range(num_rows * traced_lights):
        e = idx // traced_lights
        li = idx - e * traced_lights
        if ti.static(light_major):
            li = idx // num_rows
            e = idx - li * num_rows
        if e < event_count[0]:
            row = event_list[e]
            vis_at_slot = ti.static(RowView(shadow_vis, e))
            _shadow_fan_cell(
                row, li,
                event_pos, event_snrm, event_fnrm, event_frame,
                event_msk, t_nodes, t_node_miss, t_leaf_prim,
                t_leaf_tspan, t_first_leaf, tri_pos, tri_colors,
                tri_uvs, tri_tex_meta, textures, tri_extra,
                num_colored_triangles, b_nodes, b_node_miss, b_leaf_prim,
                b_leaf_tspan, b_first_leaf, circuit_meta, circuit_colors,
                circuit_border_colors, edges_2d, edge_accel, light_pos,
                light_col, num_lights, pixel_world_scale, cam_proj,
                layer_offset_triangles, refit, has_tri, has_bez, shared_dp,
                event_toff, sec_aa, vis_at_slot, shadow_anyhit,
                tri_obj, shared_src, eps_self, eps_near,
                shadow_identity, shadow_term, adaptive_taps, secondary,
                False, event_pos, event_msk, 0, 0, 0)


#: ``raster_shadow_trace``'s arena layout, which this kernel binds the same
#: arrays through. Its own name, because ``test_arena_args`` holds each
#: converted kernel's prologue to the spec its wrapper packs from.
_DEFERRED_SHADOW_TRACE_ARENA = _RASTER_SHADOW_TRACE_ARENA

#: The argument list ``tracer._device_wavefront_shadows`` passes.
_DEFERRED_SHADOW_TRACE_PARAMS = (
    "num_rows", "event_count", "event_list", "event_pos", "event_snrm",
    "event_fnrm", "event_frame", "event_msk", "t_nodes", "t_node_miss",
    "t_leaf_prim", "t_leaf_tspan", "t_first_leaf", "tri_pos", "tri_colors",
    "tri_uvs", "tri_tex_meta", "textures", "tri_extra", "num_colored_triangles",
    "b_nodes", "b_node_miss", "b_leaf_prim", "b_leaf_tspan", "b_first_leaf",
    "circuit_meta", "circuit_colors", "circuit_border_colors", "edges_2d",
    "edge_accel", "light_pos", "light_col", "num_lights", "traced_lights",
    "pixel_world_scale", "layer_offset_triangles", "refit", "has_tri", "has_bez",
    "event_dp", "event_toff", "sec_aa", "shadow_vis", "shadow_anyhit", "tri_obj",
    "event_src_prim", "eps_self", "eps_near", "shadow_identity",
    "shadow_term", "adaptive_taps", "secondary", "light_major", "cam_proj_table",
    "projected",
)

_deferred_shadow_trace_launch = arena_packed(
    __name__, "deferred_shadow_trace_arena",
    _DEFERRED_SHADOW_TRACE_PARAMS, _DEFERRED_SHADOW_TRACE_ARENA)


def deferred_shadow_trace(*args):
    """Pack the arena-bound arguments, then launch ``deferred_shadow_trace_arena``.

    Same convention as :func:`raster_shadow_trace`; see `arena_args_taichi`.
    """
    return _deferred_shadow_trace_launch(*args)




@ti.kernel
def shadow_queue_prepare(
        num_events: int, event_first: int, num_lights: int,
        event_pos: ti.types.ndarray(), event_snrm: ti.types.ndarray(),
        event_fnrm: ti.types.ndarray(), event_frame: ti.types.ndarray(),
        event_msk: ti.types.ndarray(), event_dp: ti.types.ndarray(),
        event_toff: ti.types.ndarray(), light_pos: ti.types.ndarray(),
        light_col: ti.types.ndarray(), sec_aa: ti.template(),
        shadow_term: ti.template(), adaptive_taps: ti.template(), secondary: int,
        offsets: ti.types.ndarray(), sort_sources: ti.types.ndarray(),
        num_triangles: int, queue_data: ti.types.ndarray(),
        queue_valid: ti.types.ndarray(), keys: ti.types.ndarray(),
        sort_rays: ti.template()):
    """Emit bounded light-major sample planes; no traversal or readback."""
    for cell in range(num_events * num_lights):
        li = cell // num_events
        local_e = cell % num_events
        e = event_first + local_e
        base = offsets[li] * num_events
        ns = offsets[li + 1] - offsets[li]
        for sk in range(ns):
            queue_valid[base + sk * num_events + local_e] = 0
        _shadow_fan_cell(
            e, li,
            event_pos, event_snrm, event_fnrm, event_frame,
            event_msk, queue_data, queue_data, queue_data,
            queue_data, 0, queue_data, queue_data,
            queue_data, queue_data, queue_data, queue_data,
            0, queue_data, queue_data, queue_data,
            queue_data, 0, queue_data, queue_data,
            queue_data, queue_data, queue_data, light_pos,
            light_col, num_lights, queue_data, None, 0,
            0, 0, 0, event_dp,
            event_toff, sec_aa, queue_data, 0,
            queue_data, queue_data, 0, 0,
            0, shadow_term, adaptive_taps, secondary,
            True, queue_data, queue_valid, base, event_first, num_events)
        if ti.static(sort_rays):
            src = ti.max(0, sort_sources[e] + 1)
            for sk in range(ns):
                r = base + sk * num_events + local_e
                key = ti.i64(0x7fffffffffffffff)
                if queue_valid[r] != 0:
                    octant = ti.cast(queue_data[r, 3] >= 0.0, ti.i32) \
                        | (ti.cast(queue_data[r, 4] >= 0.0, ti.i32) << 1) \
                        | (ti.cast(queue_data[r, 5] >= 0.0, ti.i32) << 2)
                    key = (ti.cast(li * 8 + octant, ti.i64)
                           * (ti.cast(num_triangles, ti.i64) + 1) + src)
                keys[r] = key


@ti.kernel
def shadow_queue_trace_arena(
        num_events: int, event_first: int, num_rays: int,
        event_frame: ti.types.ndarray(), event_src_prim: ti.types.ndarray(),
        t_nodes: NODE_ARG, t_first_leaf: int, num_colored_triangles: int,
        b_nodes: NODE_ARG, b_first_leaf: int, layer_offset_triangles: ti.f32,
        refit: ti.template(), has_tri: ti.template(), has_bez: ti.template(),
        shadow_anyhit: ti.template(), shadow_identity: ti.template(),
        eps_self: ti.f32, eps_near: ti.f32,
        layout: ti.types.ndarray(), queue_data: ti.types.ndarray(),
        queue_valid: ti.types.ndarray(), order: ti.types.ndarray(),
        occ_out: ti.types.ndarray(), sort_rays: ti.template(), phase: ti.template(),
        projected: ti.template(),
        arena_f32: ti.types.ndarray(), arena_i32: ti.types.ndarray(),
        aoff: ti.types.ndarray(), ashp: ti.types.ndarray()):
    """One traversal per worker, scattered back to immutable sample slots.

    Phase zero handles soft fans and the hard fan's diagonal pair. Phase one
    visits off-diagonal hard taps only when that pair did not terminate the
    reference fan. Separate dispatches make those reads race-free.
    """
    t_node_miss = ti.static(ArenaView(arena_i32, aoff[0], (ashp[0],)))
    t_leaf_prim = ti.static(ArenaView(arena_i32, aoff[1], (ashp[1],)))
    t_leaf_tspan = ti.static(ArenaView(arena_i32, aoff[2], (ashp[2],)))
    tri_pos = ti.static(ArenaView(arena_f32, aoff[3], (ashp[3], ashp[4], ashp[5])))
    tri_colors = ti.static(ArenaView(
        arena_f32, aoff[4], (ashp[6], ashp[7], ashp[8], ashp[9])))
    tri_uvs = ti.static(ArenaView(arena_f32, aoff[5], (ashp[10], ashp[11], ashp[12])))
    tri_tex_meta = ti.static(ArenaView(arena_i32, aoff[6], (ashp[13], ashp[14])))
    textures = ti.static(ArenaView(arena_f32, aoff[7], (ashp[15], ashp[16], ashp[17])))
    tri_extra = ti.static(ArenaView(arena_f32, aoff[8], (ashp[18], ashp[19], ashp[20])))
    b_node_miss = ti.static(ArenaView(arena_i32, aoff[9], (ashp[21],)))
    b_leaf_prim = ti.static(ArenaView(arena_i32, aoff[10], (ashp[22],)))
    b_leaf_tspan = ti.static(ArenaView(arena_i32, aoff[11], (ashp[23],)))
    circuit_meta = ti.static(ArenaView(
        arena_f32, aoff[12], (ashp[24], ashp[25], ashp[26])))
    circuit_colors = ti.static(ArenaView(
        arena_f32, aoff[13], (ashp[27], ashp[28], ashp[29], ashp[30])))
    circuit_border_colors = ti.static(ArenaView(
        arena_f32, aoff[14], (ashp[31], ashp[32], ashp[33], ashp[34])))
    edges_2d = ti.static(ArenaView(arena_f32, aoff[15], (ashp[35], ashp[36], ashp[37])))
    edge_accel = ti.static(ArenaView(arena_i32, aoff[16], (ashp[38],)))
    # Keep the shared serial-shadow arena offset contract (including lights).
    light_pos = ti.static(ArenaView(  # noqa: F841
        arena_f32, aoff[17], (ashp[39], ashp[40], ashp[41])))
    light_col = ti.static(ArenaView(  # noqa: F841
        arena_f32, aoff[18], (ashp[42], ashp[43], ashp[44])))
    pixel_world_scale = ti.static(ArenaView(arena_f32, aoff[19], (ashp[45],)))
    tri_obj = ti.static(ArenaView(arena_i32, aoff[20], (ashp[46], ashp[47])))
    cam_proj_table = ti.static(ArenaView(arena_f32, aoff[21], (ashp[48], ashp[49])))
    cam_proj = ti.static(CameraProjection(cam_proj_table, projected))

    for idx in range(num_rays):
        r = idx
        if ti.static(sort_rays):
            r = ti.cast(order[idx], ti.i32)
        flag = queue_valid[r]
        slot = r // num_events
        sk = layout[slot, 0]
        first = layout[slot, 1] * num_events + r % num_events
        trace = flag != 0
        if ti.static(phase == 0):
            if flag == 2 and sk >= 2:
                trace = False
        else:
            trace = flag == 2 and sk >= 2
            # Keep this guard structurally nested: the compiler need not
            # short-circuit boolean operands. A one-slot fan has no second
            # diagonal slot, even though it never enters this phase.
            if trace:
                if queue_valid[first] == 2 and queue_valid[first + num_events] == 2:
                    same = True
                    for k in ti.static(range(3)):
                        if occ_out[first, k] != occ_out[first + num_events, k]:
                            same = False
                    if same:
                        trace = False
        if trace:
            e = event_first + r % num_events
            f = event_frame[e]
            src_prim = -1
            src_sid = -1
            if ti.static(shadow_identity):
                src_prim = event_src_prim[e]
                if src_prim >= 0:
                    src_sid = tri_obj[f % tri_obj.shape[0], src_prim]
            origin = ti.math.vec3(queue_data[r, 0], queue_data[r, 1], queue_data[r, 2])
            direction = ti.math.vec3(queue_data[r, 3], queue_data[r, 4], queue_data[r, 5])
            occ = _shadow_occluded(
                refit, shadow_anyhit, origin, direction, f, ti.cast(f, ti.f32),
                queue_data[r, 6], pixel_world_scale[f % pixel_world_scale.shape[0]],
                0.0, _pixel_size_base(cam_proj, f),
                layer_offset_triangles, has_tri, has_bez,
                t_nodes, t_node_miss, t_leaf_prim, t_leaf_tspan, t_first_leaf,
                tri_pos, tri_colors, tri_uvs, tri_tex_meta, textures, tri_extra,
                num_colored_triangles, b_nodes, b_node_miss, b_leaf_prim,
                b_leaf_tspan, b_first_leaf, circuit_meta, circuit_colors,
                circuit_border_colors, edges_2d, edge_accel, src_sid, src_prim,
                eps_self, eps_near, tri_obj, shadow_identity)
            for k in ti.static(range(3)):
                occ_out[r, k] = occ[k]


@ti.kernel
def shadow_queue_reduce(
        num_events: int, event_first: int, num_lights: int,
        offsets: ti.types.ndarray(), queue_valid: ti.types.ndarray(),
        occ_out: ti.types.ndarray(), shadow_vis: ti.types.ndarray()):
    """Reference-order RGB reduction, including its adaptive early exit."""
    for cell in range(num_events * num_lights):
        li = cell // num_events
        e = cell % num_events
        base = offsets[li] * num_events + e
        total = ti.math.vec3(0.0)
        first = ti.math.vec3(0.0)
        count = 0.0
        for sk in range(offsets[li + 1] - offsets[li]):
            r = base + sk * num_events
            if queue_valid[r] != 0:
                occ = ti.math.vec3(occ_out[r, 0], occ_out[r, 1], occ_out[r, 2])
                total += occ
                count += 1.0
                if queue_valid[r] == 2:
                    if sk == 0:
                        first = occ
                    elif sk == 1:
                        if count == 2.0 and occ[0] == first[0] \
                                and occ[1] == first[1] and occ[2] == first[2]:
                            total += total
                            count += count
                            break
        visibility = ti.math.vec3(1.0)
        if count > 0.0:
            visibility = ti.math.vec3(1.0) - total / count
        for c in ti.static(range(3)):
            shadow_vis[event_first + e, li, c] = visibility[c]


_SHADOW_QUEUE_TRACE_ARENA = _RASTER_SHADOW_TRACE_ARENA
_SHADOW_QUEUE_TRACE_PARAMS = (
    "num_events", "event_first", "num_rays", "event_frame", "event_src_prim",
    "t_nodes", "t_first_leaf", "num_colored_triangles", "b_nodes", "b_first_leaf",
    "layer_offset_triangles", "refit", "has_tri", "has_bez", "shadow_anyhit",
    "shadow_identity", "eps_self", "eps_near", "layout", "queue_data", "queue_valid",
    "order", "occ_out", "sort_rays", "phase", "t_node_miss", "t_leaf_prim",
    "t_leaf_tspan", "tri_pos", "tri_colors", "tri_uvs", "tri_tex_meta", "textures",
    "tri_extra", "b_node_miss", "b_leaf_prim", "b_leaf_tspan", "circuit_meta",
    "circuit_colors", "circuit_border_colors", "edges_2d", "edge_accel", "light_pos",
    "light_col", "pixel_world_scale", "tri_obj", "cam_proj_table", "projected",
)
_shadow_queue_trace_launch = arena_packed(
    __name__, "shadow_queue_trace_arena",
    _SHADOW_QUEUE_TRACE_PARAMS, _SHADOW_QUEUE_TRACE_ARENA)


def shadow_queue_trace(*args):
    """Launch the one-ray-per-worker traversal through the arena convention."""
    return _shadow_queue_trace_launch(*args)
