"""Sheet compaction: the fragment stream aggregated into per-pixel sheets.

``DESIGN_sheet_resolve.md`` P1 + P2. A **sheet** is a maximal same-surface
region within one pixel — keyed ``(pixel, mesh id, facing, depth band)``,
which ``sheet_shade_split`` subdivides by flat-face shading class into §4.4
siblings (see ``compact_sheets``) — with each bezier circuit fragment alone
(circuits never group; their border/fill blend is already packed per
fragment). The compaction turns the emission's depth-sorted fragment stream
into the sheet stream: exact area as a sum over the sheet's fragments, the
union of sub-pixel sample masks, a dominant (largest-area) fragment as the
shading reference, and the depth of the nearest fragment that owns a sample
(``sheet_positioned_depth``; off, the nearest fragment of any kind, which
lets a position-less area donor decide which of two interpenetrating
surfaces takes the pixel).

Everything here is a sort plus a segmented reduction — no bounded lookahead,
no per-thread walk, and no budget, which is the point: the ``_AA_MAX_RUN_SCAN``
truncation machinery and its defect tail (``DESIGN_mesh_identity.md``
§0.5/§6.7/§6.8) cannot exist in this representation.

Determinism. Integer reductions (mask OR, min-position, counts) are exact
under any summation order. The one float reduction — the area sum — follows
the §6.6.4 pattern: accumulate in float64 and round to float32, which was
measured bitwise-stable across runs where a float32 ``scatter_add_`` was not.
The eventual shipping shape (host torch vs a fixed-tree Taichi scan,
``DESIGN_sheet_resolve.md`` §10.4) is decided at Phase 2; this module is the
Phase-1 implementation and the semantic reference for whatever replaces it.

Band rules (``DESIGN_sheet_resolve.md`` §4.2, open parameter §10.1): within
``(pixel, mesh, facing)`` and sorted by depth, a gap larger than a RELATIVE
threshold starts a new band. The candidates measured in Phase 1:

``facing``
    No depth banding — at most one band per ``(mesh, facing)`` per pixel.
    The old system's behavior, and the fallback.
``prim``
    Split where the gap to the previous fragment exceeds ``band_c`` times the
    two fragments' own scales. The scale is the triangle's depth variation
    ACROSS ONE PIXEL — its camera-distance extent divided by its projected
    size in pixels (from ``tri_screen`` where the projection is valid) —
    plus one pixel's world size at the fragment's depth
    (``pixel_world_scale[f] * t``). Both terms come from the record; there
    is no absolute constant to retire later.

    The first build used the RAW camera-distance extent, and one measured
    defect retired it: a large wall's extent (several world units) swamps
    any gap in front of it, so a quad 1.0 in front of a same-id backdrop
    FUSED into one sheet and shaded with the backdrop's color — a bright
    line along the region where the two overlapped. Per-pixel slope is the
    quantity that actually bounds same-sheet neighbour gaps.

Failure directions are asymmetric (§6.2): FUSING two genuinely distinct
same-facing sheets over-claims coverage (their areas sum past the footprint),
while SPLITTING one true sheet degrades to sample-quantized compositing
between the halves — benign. So a rule should err toward splitting, and the
fusion detector below is the hard gate: within one true sheet the fill rule
partitions the sub-pixel samples, so a band in which any sample bit was
contributed twice has provably fused at least two sheets.
"""

from __future__ import annotations

import torch

from algan.environment import env_flag, env_float
from algan.rendering.mps_compat import (
    accumulate_dtype,
    band_class_groups,
    clamp_floor,
    cummax_values,
    gather_packed_key,
    kernel_index,
    reduction_index_dtype,
    taichi_accumulate_dtype,
    taichi_reduction_index_dtype,
)
from algan.rendering.raytracing import settings as rt_settings
from algan.rendering.raytracing.raster_taichi import (
    _AA_BACKFACE_BIT as AA_BACKFACE_BIT,
)
from algan.rendering.raytracing.raster_taichi import (
    _AA_LOSE_SHIFT as AA_LOSE_SHIFT,
)
from algan.rendering.raytracing.raster_taichi import (
    _AA_MASK_ALL as AA_MASK_ALL,
)
from algan.rendering.raytracing.raster_taichi import (
    _AA_MAT_OPAQUE_BIT as AA_MAT_OPAQUE_BIT,
)
from algan.rendering.raytracing.raster_taichi import (
    _AA_NUM_SAMPLES as AA_NUM_SAMPLES,
)
from algan.rendering.raytracing.raster_taichi import (
    _AA_ONE_MESH_BIT as AA_ONE_MESH_BIT,
)
from algan.rendering.raytracing.raster_taichi import (
    _AA_SLIVER_BIT as AA_SLIVER_BIT,
)
from algan.rendering.raytracing.raytrace_kernels_taichi import (
    depth_tie_epsilon,
)
from algan.rendering.raytracing.truncation import record_truncation

#: Band rules this module implements. "facing" is the no-depth-split fallback.
BAND_RULES = ("facing", "prim")

#: Largest conflict rank a sheet key can carry: the rank occupies the four low
#: bits of ``cid`` (``band_id * 16 + rank``), so one pixel resolves at most 16
#: overlapping layers of a single surface. It is a fixed ceiling that degrades
#: the image rather than raising, so ``compact_sheets`` counts what it clamps
#: (:mod:`algan.rendering.raytracing.truncation`).
SHEET_RANK_LIMIT = 15

#: Shading-class quantization (``shade_split``): a flat face's unit normal is
#: rounded to this many bins per component (~0.9 degrees). Mis-binning can only
#: ever SPLIT two near-parallel faces -- the benign direction, their shading is
#: near-identical -- never fuse a crease coarser than one bin.
SHADE_CLASS_QUANT = 64

#: Group-key stride reserving the low bits for the shading class. A packed
#: class is three (2 * SHADE_CLASS_QUANT + 1 <= 129)-valued components in 8
#: bits each, plus one to keep 0 as "smooth": < 2**25.
_SHADE_CLASS_BASE = 1 << 25

#: Interior-tiling dust band, shared with the kernels: a full-union sheet whose
#: exact area is within this of 1 composites at exactly 1, so a genuine tiling
#: stays bit-clean.
FULL_DUST = 1e-3

#: ``sheet_sample_depth``: the share of its own samples a sheet must be losing
#: before it cedes any of them. A fragment's depth is evaluated at the centroid
#: of the samples it owns, so a per-lane depth is that centroid's rather than
#: the lane's; the finer the margin the less it is entitled to decide. Ceding a
#: small minority spends that weakest reading on a pixel the sheet already
#: wins, which measured WORSE (two pixels of the reference frame regressed by
#: 110 and 55 channel values). At 0.25 the reference frame's eight artifact
#: pixels all improve and none regresses; 0.5 keeps three of the eight and
#: 0.0 (cede whatever is lost) reinstates both regressions. Tuned against that
#: reference frame rather than derived, so it is exposed for re-tuning; it
#: moves rendered output.
sheet_sample_depth_cede = min(
    1.0, max(0.0, env_float("ALGAN_SHEET_SAMPLE_DEPTH_CEDE", 0.25))
)

#: Composite a band's CONFLICT-RANK sub-bands as §4.4 siblings -- claiming
#: additively against the same incoming visibility, occluding once by their
#: summed factor -- wherever the band's own areas say it holds ONE layer.
#:
#: The rank split exists for geometry a ray genuinely crosses twice (see the
#: fill-rule block in ``compact_sheets``), and there it must stay: two
#: translucent layers attenuate per crossing. But the same key also fires on a
#: SEAM. Adjacent triangles of one surface are supposed to partition the
#: samples exactly -- ``raster_taichi``'s fixed-point top-left rule -- and they
#: do wherever they share bit-identical vertices; where they do not (a
#: T-junction between two adaptively diced patches, the camera-plane
#: straddler's epsilon barycentric test, a fold tangency) they overlap by a
#: sliver, one sample lands in both masks, and the later fragment is promoted
#: to rank 1 over a dust-sized overlap.
#:
#: Walked as independent occluders those two sheets under-claim exactly as
#: DESIGN_sheet_resolve.md §4.4 records for shading-class siblings: a band
#: whose fragments cover 1.011 of the pixel occludes only 0.92 of it, and the
#: deficit admits whatever is behind -- on ``solids_and_camera``'s Arrow3D, the
#: white Line3D running inside the opaque red arrowhead, as a bright speck on
#: the cone's shoulder. §4.4's arithmetic is what the band is owed: it commits
#: the band's own exact area whatever the split.
sheet_rank_pool = env_flag("ALGAN_SHEET_RANK_POOL", True)

#: Most exact area a FULL-union band may hold and still count, for
#: :data:`sheet_rank_pool`, as one layer the fill rule split over a seam.
#:
#: The other half of that test is the full union itself, and it is the half
#: that carries the argument: a band owning every sub-pixel sample has nothing
#: left to anti-alias, so the only question it still answers is how much it
#: occludes -- its own exact area, which is what §4.4 commits. A PARTIAL union
#: is excluded outright rather than by a looser threshold, because there area
#: and sample count disagree by up to a whole sample cell for reasons that have
#: nothing to do with layering: that disagreement IS a silhouette.
#:
#: This bound is then the overlap a seam is allowed: 1.05 lets a band's
#: fragments overrun the pixel by 5% of its area, which a T-junction sliver or
#: an epsilon-wide double claim does and a second layer does not. Measured
#: against an ``analytic_aa=False`` supersampled reference on the frame the
#: defect was found in (``solids_and_camera`` at 12.8 s), sweeping the bound
#: over the pixels it moves:
#:
#:   1.01  0 pixels move -- the speck's own band is at 1.011
#:   1.03  2 move, both strictly closer to the reference (39 -> 0, 1 -> 0)
#:   1.05  5 move, 4 closer 1 further; summed error 47 -> 6
#:   1.08  9 move, 4 closer 5 further; summed error 79 -> 51
#:   1.10  12 move, 5 closer 6 further, 1 tied; summed error 91 -> 65
#:
#: so 1.05 keeps the whole win with headroom for a seam wider than this one's,
#: and stops short of the band where the trade goes flat.
sheet_rank_pool_layers = max(1.0, env_float("ALGAN_SHEET_RANK_POOL_LAYERS", 1.05))


def resolve_pixel_reference(
    covs, msks, is_bez, alphas=None, trans=None, *, caps=None, min_alpha=0.0
):
    """The sheet resolve's per-pixel semantics, plain and sequential — the ORACLE.

    ``DESIGN_sheet_resolve.md`` §2 keeps a readable, unbounded, sequential
    implementation of §4's semantics as the verification arm for the shipping
    pipeline: the kernel must match it wherever fixed-tree and sequential
    rounding agree, and to within reassociation noise where they do not. This
    is that implementation; it is a harness dependency, never a shipping path.

    Parameters are one pixel's depth-sorted sheet list: exact areas ``covs``,
    mask words ``msks`` (low sample bits + flag bits), ``is_bez`` flags, and
    optional per-sheet material ``alphas`` (default 1, matte) and transmission
    shares ``trans`` (default 0). ``min_alpha`` mirrors the walk's
    ``eff <= min_alpha`` skip when parity with a kernel is wanted.

    Returns ``(claims, T)``: what each sheet paints, and the per-sample
    transmittance left for the background (§4.5's final sheet).

    The semantics, as settled for Phase 2:

    * AREAL sheets — circuits, and donor-only sheets (empty sample union,
      flagged ``_AA_SLIVER_BIT`` by the compaction) — claim
      ``alpha * min(area, 1)`` uniformly over every sample. This single rule
      replaces the old walk's ``run_mode 2`` sequential renormalization: with
      one record per sheet there is no chain to renormalize.
    * A FULL-union sheet composites at ``corr = 1`` inside the ``FULL_DUST``
      band (interior tilings stay bit-clean) and at ``min(area, 1)`` outside
      it (a silhouette sheet paints its exact area).
    * A PARTIAL-union sheet takes ``corr = min(area, 1) / Q`` — §4.3's rule,
      claim exact by construction.
    * ``corr > 1`` (a sheet covering more than its sample share, e.g. a
      sub-sample rod) keeps its claim exact and redistributes the clamped
      occlusion residue onto the samples the sheet does NOT own — the old
      rule B, collapsed from walk state to per-record arithmetic. No
      cross-record feedback exists.
    * The ONE-MESH ceiling survives as sheet data (``caps``, the per-pixel
      ``frag_cap`` the host reduced — max of the mesh's two sheet areas). On
      a pixel flagged single-opaque-mesh, the mesh's committed coverage is
      clamped at the ceiling, occlusion scaled with the claim (§6.6.2's
      completion). The design's first draft deleted this as "subsumed by
      per-sheet claims"; the Phase-2 ink-wobble A/B refuted that — without
      it the coarse Cylinder's far sheet re-claims the corr residue and
      wobble regresses 2-4x.
    * §4.4 BAND siblings (the shading-class split's crease faces) arrive as
      consecutive sheets, all but the last carrying a NEGATIVE area: each
      claims against the same incoming ``T`` and the band occludes once, at
      its last sheet, by the summed factor. One sheet per band -- every
      sheet with the split off -- is the unchanged arithmetic above.
    * No run scan, no seam deduplication, no engagement gate, and no
      TRUNCATED-sum machinery: fragment-walk apparatus the representation
      deletes (§7).
    """
    n = len(covs)
    N = AA_NUM_SAMPLES
    if alphas is None:
        alphas = [1.0] * n
    if trans is None:
        trans = [0.0] * n
    T = [1.0] * N
    claims = []
    mesh_ink = 0.0
    band_p = 0.0
    band_open = False
    for i in range(n):
        msk_low = msks[i] & AA_MASK_ALL
        areal = bool(is_bez[i]) or (msks[i] & AA_SLIVER_BIT) or msk_low == 0
        alpha = alphas[i]
        # A negative area marks a sheet whose §4.4 band continues at the next
        # one: it claims against the same incoming T and defers the band's
        # single occlusion write to the band's last sheet, which makes it with
        # the summed coverage factor (``sheets._sibling_weights``).
        defer = covs[i] < 0.0
        raw = abs(covs[i])
        area = min(raw, 1.0)
        # Per-sample coverage BEFORE material alpha, which is what the walk's
        # ``eff`` is and what the one-mesh ceiling bounds.
        if areal:
            p_i = area
        elif msk_low == AA_MASK_ALL:
            p_i = 1.0 if abs(1.0 - raw) <= FULL_DUST else area
        else:
            p_i = area / (bin(msk_low).count("1") / N)
        own = [1.0 if (areal or (msk_low >> s) & 1) else 0.0 for s in range(N)]
        # sheet_sample_depth: samples the host ceded to a strictly nearer
        # other-surface sheet claim nothing here -- same placement as the
        # resolve kernel's pre-``eff`` block, non-areal sheets only. The
        # coverage factor ``p_i`` stays normalized to the ORIGINAL mask
        # popcount: ceded ink is claimed by the winner.
        if not areal:
            lose = (msks[i] >> AA_LOSE_SHIFT) & AA_MASK_ALL
            if lose:
                for s in range(N):
                    if (lose >> s) & 1:
                        own[s] = 0.0
        c = [p_i * own[s] for s in range(N)]
        eff = sum(T[s] * c[s] for s in range(N)) / N
        if (
            caps is not None
            and not is_bez[i]
            and (msks[i] & AA_ONE_MESH_BIT)
            and caps[i] <= 1.0
        ):
            room = max(caps[i] - mesh_ink, 0.0)
            if eff > room:
                k = room / max(eff, 1e-9)
                p_i *= k
                eff = room
        band_p += p_i
        if eff <= min_alpha:
            claims.append(0.0)
            if not defer:
                band_p = 0.0
            band_open = defer
            continue
        claims.append(alpha * eff)
        if not is_bez[i]:
            mesh_ink += eff
        # The write factor: the band's sum at a band's last sheet, the
        # sheet's own everywhere else (identical outside a subdivided band).
        w = band_p if (defer or band_open) else p_i
        band_open = defer
        if defer:
            continue
        band_p = 0.0
        a = [alpha * w * own[s] for s in range(N)]
        ts = trans[i]
        resid = 0.0
        for s in range(N):
            fct = (1.0 - a[s]) + a[s] * ts
            if fct < 0.0:
                resid -= fct * T[s]
                fct = 0.0
            T[s] *= fct
        if resid > 0.0:
            free = [s for s in range(N) if a[s] == 0.0]
            tot = sum(T[s] for s in free)
            if tot > 1e-12:
                sc = max(1.0 - resid / tot, 0.0)
                for s in free:
                    T[s] *= sc
    return claims, T


def _lexsort(*keys):
    """Stable argsort by ``keys`` in priority order (first key most
    significant). Composes least-significant-first, the classic LSD trick the
    emission's own ``_exact_fragment_order`` uses.
    """
    order = None
    for key in reversed(keys):
        k = key if order is None else key.index_select(0, order)
        o = torch.argsort(k, stable=True)
        order = o if order is None else order.index_select(0, o)
    return order


def _rows(arr, frame_rel, time_start):
    """The row of a per-frame array for a fragment's batch-relative frame —
    the same ``(f_rel + time_start) % rows`` convention as ``_tri_obj_row``.
    """
    return (frame_rel + int(time_start)) % arr.shape[0]


def _shade_class(
    merged, frame_rel, time_start, safe_ref, is_tri, tri_present=None, num_frames=None
):
    """Per-fragment shading class for ``shade_split`` (see ``compact_sheets``).

    Returns int64 in ``[0, _SHADE_CLASS_BASE)``: 0 for smooth-shaded
    triangles (and anything the rule cannot classify), ``1 + packed quantized
    unit face normal`` for flat-shaded ones. The flat test mirrors the shade
    kernel's ``_triangle_normal`` exactly: a triangle shades FLAT when its
    three vertex normals are equal (declared flat) or all degenerate (the
    kernel then substitutes the geometric cross-product normal).

    The class is a property of the (frame, triangle), not of the fragment, so
    it is computed once per (frame, triangle) -- a ``[F, N]`` table, F the
    frames of this chunk and N the merged triangles -- and gathered per
    fragment. The arithmetic per entry is exactly what the per-fragment
    version did on the same values, so the classes are bit-identical; what
    changes is that a 4K frame's millions of fragments no longer each
    re-derive their triangle's face normal (measured 0.29 s -> a few ms per
    compaction on the nn benchmark).

    ``tri_present`` is ``bool(is_tri.any())`` when the caller already has it;
    ``num_frames`` is the chunk's frame count (``frame_rel.amax() + 1``),
    likewise passed in when the caller has already paid that sync.
    """
    n = safe_ref.numel()
    device = safe_ref.device
    tri_norm = merged.get("tri_norm")
    tri_pos = merged.get("tri_pos")
    if tri_present is None:
        tri_present = bool(is_tri.any())
    if tri_norm is None or tri_pos is None or not tri_present:
        return torch.zeros(n, dtype=torch.int64, device=device)
    if num_frames is None:
        num_frames = int(frame_rel.amax()) + 1 if n else 1
    frames = torch.arange(num_frames, device=device) + int(time_start)
    nrm = tri_norm.index_select(0, frames % tri_norm.shape[0]).reshape(
        num_frames, -1, 3, 3
    )
    mag = nrm.norm(dim=3)
    unit = nrm / clamp_floor(mag.unsqueeze(3), 1e-12)
    spread = torch.maximum(
        (unit[:, :, 1] - unit[:, :, 0]).abs().amax(dim=2),
        (unit[:, :, 2] - unit[:, :, 0]).abs().amax(dim=2),
    )
    declared_flat = (mag.amin(dim=2) > 1e-6) & (spread < 1e-6)
    # All-degenerate vertex normals: the kernel falls back to the geometric
    # normal, so the class does too (the Polyhedron family authors none).
    geometric_flat = mag.amax(dim=2) < 1e-6
    vertex_n = unit[:, :, 0]
    pos = tri_pos.index_select(0, frames % tri_pos.shape[0])
    p0 = pos[..., 0:3]
    e1 = pos[..., 3:6] - p0
    e2 = pos[..., 6:9] - p0
    gn = torch.cross(e1, e2, dim=-1)
    gn = gn / clamp_floor(gn.norm(dim=-1, keepdim=True), 1e-12)
    face_n = torch.where(geometric_flat.unsqueeze(-1), gn, vertex_n)
    q = (
        torch.round(face_n * float(SHADE_CLASS_QUANT))
        .to(torch.int64)
        .clamp_(-SHADE_CLASS_QUANT, SHADE_CLASS_QUANT)
        + SHADE_CLASS_QUANT
    )
    packed = (q[..., 0] << 16) | (q[..., 1] << 8) | q[..., 2]
    zero = torch.zeros((), dtype=torch.int64, device=device)
    table = torch.where(declared_flat | geometric_flat, packed + 1, zero)  # [F, N]
    cls = table[frame_rel, safe_ref]
    return torch.where(is_tri, cls, zero)


def _popcount_lanes(bits):
    """Number of set sample bits in each element of a mask tensor.

    The count cannot exceed ``AA_NUM_SAMPLES`` and both callers cast it to a
    float before use, so the accumulator is int32 whatever ``bits`` is: on a
    4K frame the ``zeros_like`` version held 26 MB of int64 for values below
    nine, and every one of these arrays is live at the compaction's peak.
    """
    n = int(bits.numel())
    # The kernel walks one dimension; both callers pass a flat sheet array, and
    # anything else falls through to the loop rather than silently reshaping.
    if rt_settings.sheet_mask_kernel and n and bits.dim() == 1:
        from algan.rendering.raytracing.sheet_compact_taichi import mask_popcount

        pop = torch.empty(n, dtype=torch.int32, device=bits.device)
        mask_popcount(bits.contiguous(), n, pop)
        return pop
    pop = torch.zeros(bits.shape, dtype=torch.int32, device=bits.device)
    for b in range(AA_NUM_SAMPLES):
        pop += ((bits >> b) & 1).to(torch.int32)
    return pop


def _band_reduce(band_id, msk, cov, nbands, *, want_sliver):
    """Per-band ``(area, union, fused, sliver)`` over the sorted fragments.

    ``area`` is the exact-area sum (float32, unclamped -- the caller owns the
    clamp), ``union`` the OR of the sample bits, ``fused`` marks a band some
    sample of which two fragments both claimed (the DESIGN_sheet_resolve.md
    §6.2 partition violation that proves the band holds more than one sheet),
    and ``sliver`` -- only when asked for -- whether any fragment carried the
    sliver bit.

    All four walk the same stream, so under ``sheet_mask_kernel`` they are one
    kernel pass; the torch arm below is what they were, and stays as the A/B
    arm. That arm's shape is the reason they were ever separate: the mask
    reductions are one ``scatter_add_`` per sample lane, and the area sum
    needs an f64 copy of the whole fragment array before ``scatter_add_`` will
    take it (29 MB on a 4K frame), so there was nothing to share.

    The three integer results are int32 in both arms. A union holds
    ``AA_NUM_SAMPLES`` bits and a sliver flag holds one, and every consumer
    either compares them or casts explicitly, so the width was only ever
    costing bandwidth and 13 MB an array on a 4K frame -- which matters
    because two unions (this band set's and the shading split's) are live
    across the whole second half of the compaction.

    ``area`` accumulates in float64 and rounds to float32 in BOTH arms
    (§6.6.4): a float32 atomic add is not order-reproducible on CUDA and this
    value feeds thresholds. Measured on a real frame, 81% of sheets hold one
    fragment and 17% hold two -- order-independent at any width -- but the
    remaining 1.6% run to eleven, which is enough.
    """
    device = msk.device
    n = int(msk.numel())
    acc = accumulate_dtype()
    area64 = torch.zeros(nbands, dtype=acc, device=device)
    if rt_settings.sheet_mask_kernel and n:
        from algan.rendering.raytracing.sheet_compact_taichi import (
            sheet_band_reduce,
        )

        union = torch.zeros(nbands, dtype=torch.int32, device=device)
        dup = torch.zeros(nbands, dtype=torch.int32, device=device)
        sliver = torch.zeros(
            nbands if want_sliver else 1, dtype=torch.int32, device=device
        )
        sheet_band_reduce(
            kernel_index(band_id.contiguous()),
            msk.contiguous(),
            cov.contiguous(),
            n,
            int(AA_MASK_ALL),
            int(AA_SLIVER_BIT),
            area64,
            union,
            dup,
            sliver,
            bool(want_sliver),
            taichi_accumulate_dtype(),
        )
        fused = dup != 0
        del dup
        area = area64.to(torch.float32)
        del area64
        return area, union, fused, (sliver if want_sliver else None)

    area64.scatter_add_(0, band_id, cov.to(acc))
    area = area64.to(torch.float32)
    del area64
    bits = (msk & AA_MASK_ALL).to(torch.int64)
    union = torch.zeros(nbands, dtype=torch.int32, device=device)
    fused = torch.zeros(nbands, dtype=torch.bool, device=device)
    lane = torch.zeros(nbands, dtype=torch.int64, device=device)
    for b in range(AA_NUM_SAMPLES):
        lane.zero_()
        lane.scatter_add_(0, band_id, (bits >> b) & 1)
        union |= (lane > 0).to(torch.int32) << b
        fused |= lane > 1
    del bits, lane
    sliver = None
    if want_sliver:
        sliver = torch.zeros(nbands, dtype=torch.int32, device=device)
        sliver.scatter_reduce_(
            0,
            band_id,
            ((msk & AA_SLIVER_BIT) != 0).to(torch.int32),
            reduce="amax",
            include_self=True,
        )
    return area, union, fused, sliver


def _conflict_rank(band_start, order, msk, positions):
    """Per-sorted-fragment conflict rank within its band, UNCLAMPED.

    ``rank[j]`` is the largest, over the sample lanes sorted fragment ``j``
    claims, of the number of earlier fragments of the same band claiming that
    same lane (the call site in ``compact_sheets`` explains why the sheet key
    needs it). Returns int32; the caller owns the ``max=15`` clamp and both
    arms must reach it the same way.

    Under ``sheet_rank_kernel`` one kernel walks each band forward once with
    the eight per-lane counters in registers (``sheet_compact_taichi.
    sheet_conflict_rank``); the torch arm below is what it replaced and stays
    as the A/B arm. That arm computes the same numbers lane by lane -- a
    global exclusive prefix sum minus the prefix at the band's first index,
    which is the count of earlier in-band claimants because bands are
    contiguous and disjoint. Both arms are integer and visit the stream in
    the same order, so they agree bitwise by construction rather than by an
    order-independence argument (unlike ``_band_reduce``, whose atomics need
    one). Row 0 starts a band in both arms whether or not its flag is set,
    so they agree on ANY input, not only on streams ``compact_sheets``
    produces (whose first flag is always set).

    ``positions`` is the caller's shared arange; ONLY the torch arm reads it
    (the kernel needs no positions at all, which is part of what it saves).
    int32 through the torch scan: a lane holds 0/1 and its exclusive prefix
    sum is bounded by the fragment count, so every value fits, and the loop's
    five live [n] arrays cost half what they did (70 MB of a 4K frame). The
    kernel arm keeps only the output array: the sorted+masked copy this loop
    materializes as ``bits_pre`` never exists there.
    """
    device = msk.device
    n = int(order.numel())
    if not rt_settings.sheet_rank_kernel or n == 0:
        band_first = torch.where(band_start, positions, torch.zeros_like(positions))
        band_first = cummax_values(band_first, 0)
        bits_pre = (msk.index_select(0, order) & AA_MASK_ALL).to(torch.int32)
        rank = torch.zeros(n, dtype=torch.int32, device=device)
        for b in range(AA_NUM_SAMPLES):
            lane = (bits_pre >> b) & 1
            excl = torch.cumsum(lane, 0, dtype=torch.int32) - lane
            prior = excl - excl.index_select(0, band_first)
            del excl
            rank = torch.maximum(
                rank, torch.where(lane > 0, prior, torch.zeros_like(prior))
            )
            del lane, prior
        return rank
    from algan.rendering.raytracing.sheet_compact_taichi import (
        sheet_conflict_rank,
    )

    # Uninitialized is safe: the kernel starts a band at row 0 even when its
    # flag is clear, so every row is written exactly once (see its docstring).
    rank = torch.empty(n, dtype=torch.int32, device=device)
    # Taichi has no bool ndarray, so the flags ride as the bytes they are.
    sheet_conflict_rank(
        band_start.contiguous().view(torch.uint8),
        kernel_index(order.contiguous()),
        msk.contiguous(),
        n,
        int(AA_MASK_ALL),
        rank,
    )
    return rank


def _prim_split_after(
    merged,
    cam_origin,
    pixel_world_scale,
    tri_screen,
    frame_rel,
    time_start,
    safe_ref,
    is_tri,
    t,
    t_o,
    order,
    band_c,
    num_frames=None,
):
    """The ``prim`` band rule: ``True`` where a sorted fragment's depth gap to
    its predecessor exceeds the pair's own per-pixel scale (``compact_sheets``
    documents the rule; this is only its evaluation).

    The scale's geometric part -- the triangle's depth extent over its
    projected size -- is a property of the (frame, triangle), so it is
    computed once per (frame, triangle) as an ``[F, N]`` table and gathered
    per fragment: the same arithmetic on the same values as evaluating it
    per fragment, hence bit-identical, without a per-fragment copy of every
    triangle's three world vertices and screen bounds (the compaction's
    largest transients, and 0.19 s per compaction on a 4K nn frame). Only
    the ``pixel_world_scale * t`` term is per fragment.
    """
    tri_pos = merged["tri_pos"]
    device = safe_ref.device
    if num_frames is None:
        num_frames = int(frame_rel.amax()) + 1 if safe_ref.numel() else 1
    frames = torch.arange(num_frames, device=device) + int(time_start)
    pos = tri_pos.index_select(0, frames % tri_pos.shape[0])  # [F, N, 9]
    ro = cam_origin.index_select(0, frames % cam_origin.shape[0]).view(num_frames, 1, 3)
    dmin = dmax = None
    for k in range(3):
        dk = torch.linalg.norm(pos[..., 3 * k : 3 * k + 3] - ro, dim=-1)
        dmin = dk if dmin is None else torch.minimum(dmin, dk)
        dmax = dk if dmax is None else torch.maximum(dmax, dk)
    del ro, dk, pos
    ext = dmax - dmin
    del dmin, dmax
    # Per-PIXEL depth slope: two neighbouring fragments of one sheet can
    # differ by about one pixel's worth of the surface's depth gradient,
    # not by the triangle's whole extent. Where the projection table is
    # valid, divide by the projected size in pixels; a camera-plane
    # straddler keeps the conservative raw extent.
    slope = ext
    if tri_screen is not None and tri_screen.shape[2] >= 10:
        scr = tri_screen.index_select(0, frames % tri_screen.shape[0])
        sx = scr[..., 0:3]
        span_x = sx.amax(dim=-1) - sx.amin(dim=-1)
        sy = scr[..., 3:6]
        span_y = sy.amax(dim=-1) - sy.amin(dim=-1)
        proj = torch.maximum(span_x, span_y).clamp_min_(1.0)
        valid = scr[..., 9] > 0.5
        slope = torch.where(valid, ext / proj, ext)
        del scr, sx, sy, span_x, span_y, proj, valid
    del ext
    slope_f = slope[frame_rel, safe_ref]
    del slope
    pws = pixel_world_scale[_rows(pixel_world_scale, frame_rel, time_start)]
    scale = torch.where(is_tri, slope_f + pws * t, torch.zeros_like(t))
    del pws, slope_f
    scale_o = scale.index_select(0, order)
    del scale
    thr = float(band_c) * (scale_o[1:] + scale_o[:-1])
    del scale_o
    return (t_o[1:] - t_o[:-1]) > thr


def _band_composite(band_of_frag, nbands, cov_o, msk_o):
    """Per-band aggregates and the §4.4 sibling-split gate.

    A band is what the compaction emits as ONE sheet with ``shade_split``
    off, so its aggregates are exactly that sheet's: the float64 area sum,
    the sample union, and ``corr`` -- the per-owned-sample coverage the
    resolve derives from the pair (the shipping rules, mirrored from
    ``resolve_pixel_reference``: 1 inside the full-union dust band, the
    clamped area on a full union outside it, ``area * N / pop`` on a partial
    one).

    ``split`` marks the bands a class split can partition. The one that
    cannot is the AREAL band -- empty union, or a fragment carrying the
    sliver bit -- position-less by construction: its siblings would have no
    samples to blend across and nothing to anti-alias, and the weights have
    no union to spread over. ``corr > 1`` (a band covering more area than
    its samples own) needs no exclusion: the band's write is made whole at
    its last sheet, so rule B's residue redistributes over the same unowned
    samples it always did.

    Returns ``(area, union, corr, split)``, one entry per band.
    """
    area, union, _fused, sliver = _band_reduce(
        band_of_frag, msk_o, cov_o, nbands, want_sliver=True
    )
    del _fused

    pop = _popcount_lanes(union)
    clamped = area.clamp(max=1.0)
    full = union == AA_MASK_ALL
    corr = torch.where(
        full,
        torch.where((1.0 - area).abs() <= FULL_DUST, torch.ones_like(clamped), clamped),
        clamped * float(AA_NUM_SAMPLES) / pop.clamp_min(1).to(torch.float32),
    )
    del pop, clamped, full
    split = (union != 0) & (sliver == 0)
    del sliver
    return area, union, corr, split


def _rank_pool_groups(cid_band, rank_of_cid, band_of_frag, cov_o, msk_o, nb):
    """Which conflict-rank sub-bands composite as §4.4 siblings of one band.

    Returns ``(n_group, group_of_cid)``: the compositing-group count and, per
    sub-band (per ``cid``), the group it claims into. A group is one whole band
    where that band covers the pixel once -- full sample union, exact area
    within :data:`sheet_rank_pool_layers` -- and the sub-band itself, today's
    behaviour, everywhere else. Sheets are NOT merged either way: the split
    stays exactly where the fill rule put it, and only the compositing
    arithmetic pools, so ``sheet_fused`` keeps meaning what it meant.

    ``group_of_cid`` is ``None`` when no band pooled, so a stream that gains
    nothing from this takes exactly the path it took before it existed.

    The test is per band, over the WHOLE band's fragments: exact-area sum and
    sample union, both from one ``_band_reduce`` pass. That pass is the cost,
    and it is skipped outright on a stream where no band was rank-split at all
    (``n_pool == nb``) -- 43,065 bands and 180 splits on the frame this was
    measured on, so it is the split streams that pay.
    """
    # ``cid_band`` is the pre-rank band of each sub-band, in the ORIGINAL band
    # numbering; compact it so it can index a reduction output.
    uniq_pre, pool_of_cid = torch.unique(cid_band, sorted=True, return_inverse=True)
    n_pool = int(uniq_pre.numel())
    del uniq_pre
    if n_pool == nb:
        # Every band holds exactly one sub-band: nothing to pool, and no
        # reduction pass to pay for.
        return nb, None
    pool_of_frag = pool_of_cid.index_select(0, band_of_frag)
    area, union, _fused, _sliver = _band_reduce(
        pool_of_frag, msk_o, cov_o, n_pool, want_sliver=False
    )
    del pool_of_frag, _fused, _sliver
    # A FULL union at about unit area: the band owns every sub-pixel sample and
    # its fragments' exact areas cover the pixel once. There is nothing left to
    # anti-alias inside such a band -- the only question it still answers is how
    # much it occludes, and that is its own exact area, which is exactly what
    # §4.4 commits. A partial union is excluded on purpose rather than by a
    # looser threshold: there, area and sample count disagree by up to a whole
    # sample cell for reasons that have nothing to do with layering (that IS
    # what a silhouette is), so no ratio between them can tell one layer from
    # two -- and the sweep recorded on ``sheet_rank_pool_layers`` says so.
    fuse = (union == AA_MASK_ALL) & (area <= float(sheet_rank_pool_layers))
    del union, area
    # Zeroing a sub-band's rank in the key IS the pooling: every sub-band of a
    # fused band lands on ``pool * 16``, and the key still orders by
    # ``(band, rank)``, so the groups come out in walk order.
    key = pool_of_cid * 16 + torch.where(
        fuse.index_select(0, pool_of_cid), torch.zeros_like(rank_of_cid), rank_of_cid
    )
    del fuse, pool_of_cid
    uniq_key, group_of_cid = torch.unique(key, sorted=True, return_inverse=True)
    n_group = int(uniq_key.numel())
    del uniq_key, key
    if n_group == nb:
        return nb, None
    return n_group, group_of_cid


def _sibling_weights(sheet_band, cov, msk, band_area, band_union, band_corr):
    """§4.4 compositing weights for the sheets of a subdivided band.

    The resolve walks sheets one at a time, each occluding what follows, and
    that is right for sheets of DIFFERENT surfaces. Siblings of one band are
    not different surfaces: their exact areas partition the band's, so §4.4
    has them claim additively against the SAME incoming visibility, with the
    band occluding deeper sheets ONCE by its summed claim. Walked as
    independent occluders they instead occlude each other -- the first
    sibling's write dims the samples the second reads, a donor sibling (no
    samples of its own) is treated as a uniform veil and claims almost
    nothing, and the band as a whole under-claims by a few percent. On a
    closed solid that deficit is filled by the geometry BEHIND the crease --
    its own back faces, which a specular material can leave far brighter than
    the front -- so an interior edge renders as a bright seam.

    So each sibling is handed the band's sample union and its own share of
    the band's per-sample coverage factor,

        ``p_i = corr * share_i``,   ``share_i = area_i / sum(area)``

    and every sibling but the LAST carries it negated: the sign is the flag
    that tells the resolve this band continues, so it claims ``p_i`` against
    the undimmed visibility and defers the occlusion write. The resolve sums
    the band's ``p_i`` as it walks and writes once, at the closing sibling,
    with ``corr`` -- the unsplit band's own write. Coverage is therefore
    identical to the unsplit band's whatever the material alpha, and the
    color becomes the area-weighted blend of the siblings' own shading:
    the interior-edge AA the split is for.

    The sum rides in a register, so the flag marks band CONTINUATION in walk
    order rather than membership: siblings are consecutive there in the
    ordinary case, and where another surface interleaves them (a coincident
    depth) the band closes early and its remainder composites sheet by sheet
    -- the pre-split behaviour, on a pixel where the depth order was already
    ambiguous.

    Takes the per-sheet arrays already in WALK order and returns
    ``(wgt, wmsk)``: the coverage and mask the resolve consumes, equal to the
    sheet's own where its band holds one sheet.
    """
    nb = sheet_band.numel()
    device = cov.device
    members = torch.zeros_like(band_area, dtype=torch.int64)
    members.scatter_add_(
        0, sheet_band, torch.ones(nb, dtype=torch.int64, device=device)
    )
    # ...and ONLY where the band's sheets are one unbroken run of the walk.
    # The arithmetic below is a band's, not a sheet's: it hands every sibling
    # the band's union and its share of the band's coverage factor, and that is
    # only paid back if the deferral chain below reaches the band's last sheet
    # and writes the summed occlusion there. Where something else interleaves
    # them the chain breaks, and a sibling that writes on its own would be
    # painting its own exact area over samples it does not own -- ink moved
    # off the geometry for nothing. Such a band composites sheet by sheet
    # instead, which is the pre-split behaviour this docstring already promised
    # for the interleaved case (measured: on a fold pixel of
    # ``solids_and_camera``'s saddle Surface, where the two facings alternate
    # in depth, the union substitution alone moved the pixel 36 channel values
    # AWAY from an AA-off supersampled reference).
    runs = torch.zeros_like(band_area, dtype=torch.int64)
    starts = torch.ones(nb, dtype=torch.int64, device=device)
    if nb > 1:
        starts[1:] = (sheet_band[1:] != sheet_band[:-1]).to(torch.int64)
    runs.scatter_add_(0, sheet_band, starts)
    del starts
    whole = runs == 1
    del runs
    multi = (members.index_select(0, sheet_band) > 1) & whole.index_select(
        0, sheet_band
    )
    del whole
    if not bool(multi.any()):
        return cov, msk

    acc = accumulate_dtype()
    area_g = band_area.index_select(0, sheet_band).to(acc)
    # ``clamp_floor``, not ``clamp_min``: MPS rounds a clamp's scalar bound
    # through float16 and cannot carry this floor (``mps_compat.clamp_floor``
    # has the measurement). This is the call site where that first reached a
    # frame -- a band whose siblings were all clamped to zero coverage by the
    # closed-shell ceiling has exactly zero area, the guard did not hold, and
    # the divide produced a NaN. Nothing downstream catches one: ``eff <=
    # min_alpha`` is false against a NaN like every comparison is, so the sheet
    # composited instead of dropping out and a closed shell's interior edge came
    # back attenuated twice (``DESIGN_mps_support.md`` §2.3c).
    share = cov.to(acc) / clamp_floor(area_g, 1e-12)
    del area_g
    p = band_corr.index_select(0, sheet_band).to(acc) * share

    union = band_union.index_select(0, sheet_band)
    pop = _popcount_lanes(union).clamp_min(1).to(acc)
    full = union == AA_MASK_ALL
    # The resolve reads the coverage through its own branch, so hand it the
    # value that branch turns back into p: the factor itself on a full union,
    # the sample-share fraction of it on a partial one.
    wgt = torch.where(full, p, p * pop / float(AA_NUM_SAMPLES)).to(torch.float32)

    # Negative = "this band continues at the NEXT sheet of the walk", which
    # is exactly when the register sum is safe to carry. Nothing reorders the
    # walk for it: a band whose sheets some other surface interleaves (a
    # coincident depth) simply closes early there and its remainder
    # composites sheet by sheet, as it did before the split existed.
    cont = torch.zeros(nb, dtype=torch.bool, device=device)
    if nb > 1:
        cont[:-1] = sheet_band[1:] == sheet_band[:-1]
    wgt = torch.where(multi & cont, -wgt, wgt)
    # A donor sibling carries the sliver bit because its OWN union is empty;
    # inside a band it holds the band's samples, so the areal (position-less)
    # rule no longer applies to it. Fragment slivers cannot appear here --
    # ``_band_composite`` leaves those bands whole.
    flags = msk & ~AA_MASK_ALL & ~AA_SLIVER_BIT
    wmsk = union.to(msk.dtype) | flags
    return torch.where(multi, wgt, cov), torch.where(multi, wmsk, msk)


def _lane_first_owners(band_id, msk_o, t_o, nb, n):
    """``sheet_sample_depth``'s per-sample nearest-owner table.

    Returns ``[nb, AA_NUM_SAMPLES]`` float32: for each sheet and sub-pixel
    sample lane, the exact depth of the sheet's earliest fragment owning THAT
    lane (the stream is depth-ascending within a group, so the first owner in
    sorted order is the minimum), or +inf where the sheet does not own the
    lane -- the datum the sheet record otherwise lacks
    (OX_SHEET_INTERPENETRATION_AUDIT.md ss6). The classification downstream
    never compares an unowned lane.

    Under ``sheet_sample_depth_kernel`` one kernel performs all eight lanes'
    amin scatters in a single pass over the stream
    (``sheet_compact_taichi.sheet_lane_first_owner``); the torch arm below is
    what it replaced and stays as the A/B arm -- one masked full-length
    ``where`` plus an amin ``scatter_reduce_`` per lane. Both arms reduce the
    same integers (sorted positions) per (sheet, lane) slot, so they agree
    exactly whatever order the atomics land in; everything after the table is
    identical arithmetic on identical values.
    """
    device = msk_o.device
    inf = torch.full((), float("inf"), dtype=torch.float32, device=device)
    if rt_settings.sheet_sample_depth_kernel and nb:
        from algan.rendering.raytracing.sheet_compact_taichi import (
            sheet_lane_first_owner,
        )

        # Uninitialized nowhere: the fill value IS the "no owner" sentinel.
        first_lane = torch.full(
            (nb * AA_NUM_SAMPLES,), n, dtype=torch.int32, device=device
        )
        sheet_lane_first_owner(
            kernel_index(band_id.contiguous()),
            msk_o.contiguous(),
            n,
            int(AA_MASK_ALL),
            first_lane,
        )
        has = first_lane < n
        d_lane = t_o.index_select(0, first_lane.clamp_max(max(n - 1, 0)))
        return torch.where(has, d_lane, inf).view(nb, AA_NUM_SAMPLES)

    idx_dtype = reduction_index_dtype()
    big = torch.full((), n, dtype=idx_dtype, device=device)
    positions = torch.arange(n, dtype=idx_dtype, device=device)
    sample_depths = torch.full(
        (nb, AA_NUM_SAMPLES), float("inf"), dtype=torch.float32, device=device
    )
    for lane in range(AA_NUM_SAMPLES):
        owns = ((msk_o >> lane) & 1) != 0
        masked = torch.where(owns, positions, big)
        del owns
        first_sorted = torch.full((nb,), n, dtype=idx_dtype, device=device)
        first_sorted.scatter_reduce_(
            0, band_id, masked, reduce="amin", include_self=True
        )
        first_sorted = first_sorted.to(torch.int64)
        del masked
        has = first_sorted < n
        d_lane = t_o.index_select(0, first_sorted.clamp_max(max(n - 1, 0)))
        sample_depths[:, lane] = torch.where(has, d_lane, inf)
        del first_sorted, has, d_lane
    del big, positions, inf
    return sample_depths


def compact_sheets(
    coverage,
    merged,
    cam_origin,
    pixel_world_scale,
    time_start,
    width,
    height,
    *,
    band_rule="prim",
    band_c=4.0,
    tri_screen=None,
    shade_split=False,
    positioned_depth=True,
    sample_depth=False,
):
    """Compact one emission's fragment stream into its sheet stream.

    Parameters mirror what ``prepare_sparse_raster_coverage`` was called with:
    ``coverage`` is its returned dict (the compact ``frag_*`` arrays and the
    per-pixel CSR), ``merged`` the batch's merged scene, ``cam_origin`` /
    ``pixel_world_scale`` the per-frame camera rows the band rule's relative
    scale reads.

    ``shade_split`` (``sheet_shade_split``) adds a SHADING CLASS to the
    triangle group key, so a sheet never spans a hard shading discontinuity.
    The resolve shades ONCE per sheet at its dominant fragment, which is
    licensed exactly where shading varies smoothly across the sheet; a crease
    -- two flat-shaded faces of one solid meeting inside a pixel -- violates
    that, and the fused sheet takes the dominant face's color for the whole
    pixel, un-antialiasing every interior (non-silhouette) edge. The class:

    * a FLAT-shaded triangle takes its quantized unit face normal -- either
      declared (three equal vertex normals, the replication every flat mesh
      stores) or implicit (all-zero vertex normals, where the shade kernel's
      ``_triangle_normal`` falls back to the geometric cross product -- the
      ``Polyhedron`` family);
    * a SMOOTH-shaded triangle (varying vertex normals, diced PN geometry)
      takes class 0, so curved meshes compact exactly as before.

    Faces meeting at a crease then compact into sibling sheets of one band,
    each shading with its own normal, and composite additively by exact area
    (DESIGN_sheet_resolve.md §4.4, carried by ``sheet_wgt`` / ``sheet_wmsk``
    -- see ``_sibling_weights``): the area-weighted blend across interior
    edges, over coverage identical to the unsplit band's, paid only at crease
    pixels. Bands whose claim has no exact partition are left whole
    (``_band_composite``). Off, no band subdivides and the output is
    bit-identical to before the parameter existed.

    ``positioned_depth`` (``sheet_positioned_depth``) reads a sheet's depth
    and its place in the walk off its nearest POSITIONED fragment — one that
    owns at least one sub-pixel sample — rather than off its nearest fragment
    of any kind. An area donor owns no sample and so has no position among
    the N points at which the resolve compares sheets; letting one set the
    sheet's depth hands a whole pixel to a surface that is behind at every
    one of those points. Two sheets keep their relative order under either
    rule whenever every fragment of one precedes every fragment of the other
    in the emission stream, so only interleaved (interpenetrating) sheets can
    move.

    ``sample_depth`` (``sheet_sample_depth``) computes the one per-sample datum
    the sheet record otherwise lacks (DESIGN_sheet_resolve.md §6.1.1): for each
    sub-pixel sample a sheet owns, the exact depth of its nearest fragment
    owning THAT sample. From it, a triangle sheet that is positioned,
    material-opaque (the ``_AA_MAT_OPAQUE_BIT`` the emission pipeline folds
    into the masks), of full sample union at full exact coverage, its band's
    only sheet and of non-negative weight is an ENFORCER: per pixel and
    sample it publishes that minimum depth as a floor. A SUBJECT — triangle,
    positioned, not areal, its band's only sheet, non-negative weight — then
    cedes (loses) every owned sample where the best OTHER-surface enforcer is
    strictly nearer beyond ``depth_tie_epsilon``, and the resolve zeroes those
    samples' claim/occlusion slots. Ties and near-ties keep today's walk order;
    the walk order itself never changes, only what a sheet may claim. Sheets of
    multi-sheet bands — shading-class siblings, conflict-rank splits — are
    exempt on both sides: their band's pooled arithmetic writes occlusion once,
    ignoring slots. And a sheet cedes everything it loses or nothing at all,
    and only once it is losing more than ``sheet_sample_depth_cede`` of
    what it owns: a lane's depth is its fragment's CENTROID depth rather than
    the lane's own, so a thin margin is the reading least entitled to decide. The lose bits ride bits 20..27 of BOTH
    mask words (record and weights). Off, no bit is set anywhere and the output
    is byte-identical to before.

    Returns a dict of per-sheet arrays, ordered by ``(pixel, classic order of
    the sheet's nearest fragment)`` so a walk over them front-to-back matches
    the emission's own (depth-bin, descending-layer) relation:

    ``sheet_key``
        ``(pixel << 32) | depth bits`` of the sheet's nearest fragment —
        the same packing as ``frag_key``.
    ``sheet_ref`` / ``sheet_ab``
        The DOMINANT (largest exact area, first on ties) fragment's primitive
        reference and barycentrics: the shading reference.
    ``sheet_cov``
        The sheet's exact area: float64 sum of its fragments' ``frag_cov``,
        rounded to float32. NOT clamped to 1 — ``min(area, 1)`` is the
        consumer's rule, and a raw sum above ~1 + dust on a non-fused band is
        a finding worth seeing.
    ``sheet_msk``
        Union of the sample masks, with the flag bits: facing from the band
        key, the one-mesh/sliver flags from the dominant fragment, and the
        sliver bit forced on when the union is empty (an areal, positionless
        sheet — the donors-only case). Under ``sample_depth``, bits 20..27
        additionally carry the per-sample ceded (lose) mask.
    ``sheet_wgt`` / ``sheet_wmsk``
        What the RESOLVE consumes in place of ``sheet_cov`` / ``sheet_msk``:
        equal to them for a sheet that is its band's only one, and §4.4's
        additive sibling weights (``_sibling_weights``) where a band split by
        shading class. The record above stays the sheet's own area and union;
        these carry the band's compositing arithmetic.
    ``sheet_cap``
        The dominant fragment's ``frag_cap`` (per-pixel one-mesh ceiling).
    ``sheet_nfrag``
        Fragments compacted into this sheet.
    ``sheet_fused``
        True where some sample bit was contributed by two fragments of the
        band — the fill-rule partition violation that proves the band holds
        more than one true sheet (a band-rule failure, or declared identity
        spanning genuinely overlapping geometry).
    ``sheet_offsets``
        CSR over ``coverage['covered_idx']``: sheets of covered pixel ``i``
        are ``sheet_offsets[i] : sheet_offsets[i+1]``.
    ``num_sheets``, ``num_groups``, ``num_split_groups``
        Totals; a *group* is one ``(pixel, mesh, facing)`` (triangles only),
        and a split group produced more than one band. ``num_sheets`` is an
        int; the two group counters are DIAGNOSTIC and stay 0-d device
        tensors (evaluated only when read), so the render path never pays
        their device syncs.
    """
    if band_rule not in BAND_RULES:
        raise ValueError(f"unknown band rule {band_rule!r}; one of {BAND_RULES}")
    n = int(coverage["num_fragments"])
    num_covered = int(coverage["num_covered"])
    frag_key = coverage["frag_key"][:n]
    frag_ref = coverage["frag_ref"][:n]
    frag_ab = coverage["frag_ab"][:n]
    frag_cov = coverage["frag_cov"][:n]
    frag_msk = coverage["frag_msk"][:n]
    frag_cap = coverage["frag_cap"][:n]
    device = frag_key.device

    pix = frag_key >> 32
    t = (frag_key & 0xFFFFFFFF).to(torch.int32).view(torch.float32)
    ppf = int(width) * int(height)
    frame_rel = pix // ppf

    tri_obj = merged["tri_obj"]
    is_tri = frag_ref >= 0
    safe_ref = frag_ref.clamp_min(0).to(torch.int64)
    sid = tri_obj[_rows(tri_obj, frame_rel, time_start), safe_ref].to(torch.int64)
    facing = ((frag_msk & AA_BACKFACE_BIT) != 0).to(torch.int64)
    # One arange, shared by the bezier group key here and the conflict-rank
    # scan's torch arm below (_conflict_rank) -- they were two identical
    # int64 [n] tensors.
    positions = torch.arange(n, dtype=torch.int64, device=device)
    # Triangles group by (surface, facing); every bezier fragment is its own
    # group (negative, unique — a shared sentinel would fuse adjacent
    # circuits into one "sheet" no consumer wants). The shading class is NOT
    # part of this key: bands and conflict ranks are decided class-blind, so
    # a band is the same set of fragments whatever ``shade_split`` says, and
    # the class only SUBDIVIDES that band below (see ``_sibling_weights``).
    gkey = torch.where(is_tri, sid * 2 + facing, -(positions + 2))
    del sid, facing
    # "Does this stream hold any triangle at all?" is asked by three separate
    # rules below (the shading-class split, the primitive band rule, and the
    # closed-shell alpha cap), and each ask was an [n] reduction AND a hard
    # sync on an answer that cannot change once ``frag_ref`` is fixed. Asked
    # once here instead. Eager rather than memoized behind a closure, because
    # ``is_tri`` is deleted further down to free the [n] flags early and a
    # closure would hold it past that -- and the first consumer, the shading
    # class, is on by default, so the reduction is not new work.
    tri_present = bool(is_tri.any())
    # Frames this chunk's fragments span: the per-(frame, triangle) tables
    # below are built for exactly these rows.
    num_frames = int(frame_rel.amax()) + 1 if n else 1

    cls = None
    if shade_split:
        cls = _shade_class(
            merged, frame_rel, time_start, safe_ref, is_tri, tri_present, num_frames
        )

    # ---- P1: (pixel, group, depth) order + band starts ---------------------
    order = _lexsort(pix, gkey, t)
    pix_o = pix.index_select(0, order)
    g_o = gkey.index_select(0, order)
    t_o = t.index_select(0, order)
    del pix, gkey

    new_group = torch.ones(n, dtype=torch.bool, device=device)
    if n > 1:
        new_group[1:] = (pix_o[1:] != pix_o[:-1]) | (g_o[1:] != g_o[:-1])
    del g_o

    band_start = new_group.clone()
    if band_rule == "prim" and n > 1 and tri_present:
        split_after = _prim_split_after(
            merged,
            cam_origin,
            pixel_world_scale,
            tri_screen,
            frame_rel,
            time_start,
            safe_ref,
            is_tri,
            t,
            t_o,
            order,
            band_c,
            num_frames,
        )
        band_start[1:] |= (~new_group[1:]) & split_after
        del split_after

    # ---- The solid-shell opacity ceiling (solid_shell_alpha) ----------------
    # ``Mob.opacity`` says the MOB renders at alpha a: backdrop attenuated ONCE,
    # whatever its geometry. A declared closed shell (``Mob.closed_shell`` --
    # built-ins prove it, primitives carry it merged as ``tri_closed``, folded
    # with the transmission exemption at pack time) is crossed twice by every
    # interior ray, so both of its sheets would composite and deliver the extra
    # ``a * (1 - a)`` painted with the interior's own shading -- an authored
    # 0.55 sphere rendered 0.679. The ceiling: per (pixel, SURFACE), the
    # surface's cumulative exact coverage may not exceed ``max(front, back)``,
    # the larger of its two shells' own footprint areas, spent in depth order.
    #
    # It lives HERE, on the fragments, rather than in the resolve like the
    # opaque one-mesh rule, and the difference is the point: that rule needs a
    # whole-pixel predicate (every fragment one usable opaque mesh), so a
    # translucent solid would composite correctly only where it has the pixel
    # to itself and revert to doubled over anything behind it -- a visible seam
    # along the overlap boundary, measured (see DESIGN notes in the audit).
    # Keying by (pixel, surface) has no whole-pixel requirement, so the fix is
    # uniform wherever the solid is. It costs the visibility-weighted allowance
    # spending the resolve could do -- under a partial occluder the hidden part
    # of the near shell still consumes area -- which is inert in the common
    # cases: an interior pixel holds front = back = 1 so the far sheet gets
    # zero regardless of sample visibility, and at the silhouette the cap IS
    # the shell's own area, so the rim keeps its ink (harness ``ink`` column).
    #
    # The cap is deliberately NOT clamped to 1: a ray crossing a declared shell
    # more than twice (a torus hole, a mid-morph self-overlap) attenuates per
    # crossing -- the conflict-rank machinery's measured contract -- and a
    # front sum past 1 keeps that. Plain suppression (cap = min(front, back))
    # was refuted: it flipped a rod's signed coverage error to -0.0344 and
    # notched 1676 of 3508 interior pixels.
    #
    # Applied to the FRAGMENTS, before banding and the shading-class split, so
    # every downstream aggregate -- band areas, corr, sibling shares, the
    # dominant fragment -- sees exactly the coverage that will composite. A
    # fragment clamped to zero contributes no area anywhere: its sheet falls
    # out at the resolve's ``eff <= min_alpha`` branch, claiming nothing and
    # occluding nothing. Determinism follows the §6.6.4 pattern (float64
    # accumulate, float32 round) because the cap feeds a threshold.
    closed_s = None
    shell_sid = shell_back = None
    tri_closed_arr = merged.get("tri_closed") if rt_settings.solid_shell_alpha else None
    if tri_closed_arr is not None and tri_present:
        closed_flag = (
            tri_closed_arr[
                _rows(tri_closed_arr, frame_rel, time_start), safe_ref
            ].reshape(-1)
            > 0.5
        ) & is_tri
        del tri_closed_arr
        if bool(closed_flag.any()):
            closed_s = closed_flag.index_select(0, order)
            # The compaction frees ``sid`` / ``facing`` and their sorted copies
            # long before the clamp runs, so carry the two per-fragment facts
            # it needs -- which surface, which shell -- through with it. Only
            # scenes with something declared pay for these.
            shell_sid = (
                tri_obj[_rows(tri_obj, frame_rel, time_start), safe_ref]
                .to(torch.int64)
                .index_select(0, order)
            )
            shell_back = ((frag_msk & AA_BACKFACE_BIT) != 0).index_select(0, order)
        del closed_flag
    # ``t_o`` (the sorted exact depths) stays live past this point: the
    # sheet_sample_depth block below reads it to find each sheet's nearest
    # owner per sample. It is one [n] f32 array, freed at that block.
    del frame_rel, safe_ref, t

    band_id = torch.cumsum(band_start.to(torch.int64), 0) - 1

    # ---- The fill rule is the sheet-membership oracle -----------------------
    # Within one true sheet the masks PARTITION the samples, so a band in
    # which a sample bit appears twice holds two sheets by definition --
    # whatever their depths. Depth banding cannot separate them (a mid-morph
    # self-overlap or a fold tangency has no gap), so each fragment's
    # CONFLICT RANK -- the number of prior in-band fragments sharing any of
    # its sample bits -- becomes part of the sheet key: rank k joins the
    # k-th sub-band, and each sub-band's masks partition again. Two
    # overlapping translucent layers of one mesh then attenuate TWICE, which
    # is what a ray crossing the surface twice physically does (measured:
    # without this, a morphing tetrahedron's self-overlapping faces fused
    # and rendered ~30% too light... dark; the fragment walk composited them
    # per fragment and was right). Donors (empty masks) carry rank 0 and
    # ride with their sheet's owners. Integer throughout: deterministic.
    rank = _conflict_rank(band_start, order, frag_msk, positions)
    # The rank rides in four bits of the sheet key, so a pixel resolves at most
    # SHEET_RANK_LIMIT + 1 overlapping layers of ONE surface. Past that the
    # clamp fuses the surplus into the last sub-band, where they attenuate once
    # between them instead of once each -- the region renders too light, which
    # is exactly the defect the conflict rank exists to prevent. Instrumented
    # rather than raised (RENDERER_WORK_QUEUE.md item 1): the amax is a scalar
    # reduction over a tensor the ``unique`` two statements down already
    # synchronises on, and the [n] comparison that counts the fragments is only
    # materialised in the case that is about to be reported.
    if n:
        deepest = int(rank.amax())
        if deepest > SHEET_RANK_LIMIT:
            record_truncation(
                "sheet_layers",
                int((rank > SHEET_RANK_LIMIT).sum()),
                cap=SHEET_RANK_LIMIT + 1,
            )
    rank.clamp_(max=SHEET_RANK_LIMIT)
    cid = band_id * 16 + rank
    del rank
    uniq_cid, band_id = torch.unique(cid, sorted=True, return_inverse=True)
    del cid
    nb = int(uniq_cid.numel())
    # Band identity for sheet_sample_depth's multi-sheet-band exemption: a
    # conflict-rank split makes several sheets of ONE band, and ``cid``'s low
    # four bits are the rank. Under ``shade_split`` the same rule is recovered
    # from the class key further down.
    cid_band = uniq_cid // 16
    # ...and the rank itself, which ``_rank_pool_groups`` needs to rebuild the
    # key it pools with.
    rank_of_cid = uniq_cid - cid_band * 16
    del uniq_cid
    if nb == 0:
        return None

    # ---- P2: segmented reduction over bands --------------------------------
    cov_o = frag_cov.index_select(0, order)
    msk_o = frag_msk.index_select(0, order)
    pos_o = order  # original stream position of each sorted fragment

    # ---- The ceiling, applied ----------------------------------------------
    # ``cov_o`` is this function's own gather (a copy), so it is clamped in
    # place: every consumer below -- the shading-class aggregates, the band
    # area sums, the dominant-fragment choice -- then reads exactly the
    # coverage that will composite. Within one (pixel, surface) segment the
    # stream already runs front-facing run first, each facing depth-ascending
    # (the sort key is ``(pix, sid * 2 + facing, t)``), which is depth order
    # for a shell seen from outside and the near-shell-first spend wanted
    # everywhere else. Fragments of undeclared or transmissive surfaces get
    # unique negative keys, so each is its own pass-through segment and
    # neither spends nor consumes allowance.
    if closed_s is not None:
        # Strictly greater than any surface id, so ``pix * K + sid`` cannot
        # collide across pixels (one amax sync, in the branch that needs it).
        K = int(shell_sid.amax().item()) + 2
        key = torch.where(closed_s, pix_o * K + shell_sid, -(positions + 1))
        del shell_sid
        # Stable within a key -- but the stream's own within-segment order is
        # FACING-major (``gkey = sid * 2 + facing`` sorts both facings into
        # consecutive runs), and the backface bit does not mean "far": measured
        # on an interior sphere pixel, the NEAR crossing is the one carrying
        # the bit (negative screen-space winding), so facing-run order would
        # spend the allowance on the far shell first and zero the visible one.
        # Order each segment by TRUE DEPTH instead: the near crossing spends
        # first, whichever bit it carries. (The cap itself is unaffected --
        # ``max(front, back)`` is symmetric under the swap.)
        # ``frag_key`` is the ORIGINAL stream; every other operand here is in
        # the compaction's sorted order, so the depth key must be in sorted
        # order to break ties within a segment. That is exactly ``t_o``, which
        # the sample-depth block below keeps live anyway -- so this used to
        # rebuild it: the same mask-shift-view over [n] plus the same gather,
        # for a bit-identical copy of a tensor already in hand.
        o2 = _lexsort(key, t_o)
        # Both arms need the f64 areas and their GLOBAL exclusive prefix: the
        # prefix comes out of a cub scan, and a serial register walk cannot
        # reproduce its reassociation bitwise (measured on the real nn-scene
        # 3840x2160 frame: a serial spend moved 61 of 3.13 M values and flipped
        # 10 visible f32 outputs, all sliver areas below 1e-4 -- against the
        # byte-identity contract). The kernel takes the prefix as input and
        # does everything else per segment in registers; the torch arm below
        # stays as the A/B arm.
        # ``copy=True`` because ``cov_o`` is already float32: at
        # ``accumulate_dtype() is torch.float32`` -- MPS-friendly mode --
        # ``.to`` is the identity, and the kernel arm below hands this same
        # buffer to the kernel as its reassociation-barrier ``scratch`` while
        # ``cov_o`` is its INOUT coverage. Aliased, the barrier store
        # overwrites the coverage mid-walk and the stream comes back with
        # negative areas in it.
        cov64 = cov_o.to(accumulate_dtype(), copy=True)
        c2 = cov64.index_select(0, o2)
        csum = torch.cumsum(c2, 0)
        excl_global = csum.sub_(c2)
        if rt_settings.sheet_shell_ceiling_kernel and n:
            from algan.rendering.raytracing.sheet_compact_taichi import (
                solid_shell_ceiling,
            )

            # ``cov64`` doubles as the kernel's reassociation-barrier scratch
            # (see the kernel docstring); the torch arm needed it only to
            # build ``excl`` either way.
            solid_shell_ceiling(
                key.contiguous(),
                kernel_index(o2.contiguous()),
                shell_back.contiguous().view(torch.uint8),
                excl_global,
                cov64,
                n,
                cov_o,
                taichi_accumulate_dtype(),
            )
            del key, o2, shell_back, excl_global, c2, cov64
        else:
            # Dead past the prefix in this arm; the kernel arm reuses it as
            # its reassociation-barrier scratch instead.
            del cov64
            k2 = key.index_select(0, o2)
            del key
            seg_start = torch.ones(n, dtype=torch.bool, device=device)
            if n > 1:
                seg_start[1:] = k2[1:] != k2[:-1]
            del k2
            seg = torch.cumsum(seg_start.to(torch.int64), 0) - 1
            nseg = int(seg[-1].item()) + 1
            # The running total each fragment's in-segment predecessors have
            # already spent: the global exclusive prefix minus its value at the
            # segment's first row (the same construction ``_conflict_rank``'s
            # torch arm uses, and deterministic for the same reason).
            first = torch.zeros(nseg, dtype=torch.int64, device=device)
            first.scatter_(0, seg[seg_start], torch.nonzero(seg_start).reshape(-1))
            spent = excl_global - excl_global.index_select(0, first).index_select(
                0, seg
            )
            del excl_global, first, seg_start
            # The segment's cap: its surface's two shells' own footprint areas,
            # accumulated float64 and rounded through float32 -- §6.6.4, because a
            # ceiling that wobbles in its low bits flips borderline fragments in
            # and out of being clipped.
            backf2 = shell_back.index_select(0, o2)
            del shell_back
            acc = accumulate_dtype()
            z64 = torch.zeros((), dtype=acc, device=device)
            front = torch.zeros(nseg, dtype=acc, device=device)
            back = torch.zeros(nseg, dtype=acc, device=device)
            front.scatter_add_(0, seg, torch.where(backf2, z64, c2))
            back.scatter_add_(0, seg, torch.where(backf2, c2, z64))
            del backf2, z64
            cap = torch.maximum(front, back).to(torch.float32).to(acc)
            del front, back
            scale = (
                cap.index_select(0, seg)
                .sub_(spent)
                .clamp_min_(0.0)
                .div_(c2.clamp_min_(1e-12))
                .clamp_max_(1.0)
            )
            del spent, cap, seg
            # A fragment clamped to zero carries no area into any band aggregate:
            # its sheet falls out at the resolve's ``eff <= min_alpha`` branch,
            # claiming nothing and occluding nothing.
            cov_o.index_copy_(0, o2, (c2 * scale).to(torch.float32))
            del scale, c2, o2
        closed_s = None
    # ``band_id`` is now the SUB-BAND -- the sheet this compaction would build
    # with the split off, once the conflict rank has divided it. Two things
    # subdivide it further or pool it back:
    #
    # * ``rank_pool`` decides, per BAND, whether its conflict-rank sub-bands
    #   are one layer seen twice by the fill rule (a seam) or two layers a ray
    #   really crosses. A seam's sub-bands become §4.4 siblings of one band,
    #   which is what stops an opaque surface's own sub-pixel self-overlap
    #   letting the geometry behind it through (``sheet_rank_pool``);
    # * ``shade_split`` subdivides each sub-band by shading class into §4.4
    #   siblings, except where there is nothing to anti-alias: an areal
    #   (position-less) band stays whole, exactly as it is with the split off
    #   (``_band_composite``).
    #
    # Both feed the SAME arithmetic, so ``sheet_band`` names one compositing
    # group whichever of them (or both) produced it, and the sheets of a group
    # claim additively against one incoming visibility and occlude once.
    band_area = band_union = band_corr = sheet_band = None
    n_group, group_of_cid = nb, None
    if sheet_rank_pool and nb:
        n_group, group_of_cid = _rank_pool_groups(
            cid_band, rank_of_cid, band_id, cov_o, msk_o, nb
        )
    del rank_of_cid
    if shade_split:
        band_of_frag = (
            band_id if group_of_cid is None else group_of_cid.index_select(0, band_id)
        )
        band_area, band_union, band_corr, band_split = _band_composite(
            band_of_frag, n_group, cov_o, msk_o
        )
        cls_o = cls.index_select(0, order)
        cls = None
        cls_eff = torch.where(
            band_split.index_select(0, band_of_frag), cls_o, torch.zeros_like(cls_o)
        )
        del cls_o, band_split, band_of_frag
        # Keyed by the SUB-BAND, not by the compositing group: pooling must not
        # merge two sub-bands into one sheet, only make them claim as one band.
        nb, band_id, sheet_cid = band_class_groups(band_id, cls_eff, _SHADE_CLASS_BASE)
        del cls_eff
        sheet_band = (
            sheet_cid
            if group_of_cid is None
            else group_of_cid.index_select(0, sheet_cid)
        )
        del sheet_cid
    elif group_of_cid is not None:
        # No class split: one sheet per sub-band, so the group table is already
        # per sheet. Left as ``None`` when nothing pooled, which keeps the
        # weights -- and the multi-sheet-band exemption below -- exactly as
        # they were.
        band_area, band_union, band_corr, _split = _band_composite(
            group_of_cid.index_select(0, band_id), n_group, cov_o, msk_o
        )
        del _split
        sheet_band = group_of_cid
    del group_of_cid

    # The band's aggregates in one walk of the sorted stream: exact area
    # (float64 accumulate, float32 round -- §6.6.4), the sample-mask union,
    # and the fusion detector.
    sheet_cov, union, fused, _ = _band_reduce(
        band_id, msk_o, cov_o, nb, want_sliver=False
    )
    sheet_cov.clamp_min_(0.0)

    # Nearest fragment (minimum sorted position -- the stream is depth-sorted
    # within a group, and a rank-split sheet's members need not be
    # consecutive) and the sheet's position in the classic order: the
    # MINIMUM original stream position (the emission is (pixel, depth-bin,
    # descending-layer) sorted, so min-position inherits that relation).
    #
    # Under ``sheet_band_stats_kernel`` one kernel visit per fragment fills all
    # six tables these scatters produce -- including the dominant fragment's
    # area maximum and the count, which torch computed further down -- and a
    # second resolves the dominant position against each band's completed
    # maximum. Integer mins/maxes/adds are exact under any atomics order and
    # an f32 amax has no association, so the arms agree by construction; the
    # caller-side gathers and the positioned-depth fallback ``where`` below
    # are unchanged. The torch statements stay as the A/B arm.
    if rt_settings.sheet_band_stats_kernel and nb:
        from algan.rendering.raytracing.sheet_compact_taichi import (
            band_stats_reduce,
        )

        # The five reduction outputs take ``reduction_index_dtype`` -- int64
        # here, int32 in MPS-friendly mode, where Taichi's int64 atomics abort
        # on Metal -- and widen straight back, so everything downstream sees
        # the same int64 positions either way. ``.to`` is the identity when the
        # dtype already matches, so the default path allocates and copies
        # exactly what it did.
        idx_dtype = reduction_index_dtype()
        first_sorted = torch.full((nb,), n, dtype=idx_dtype, device=device)
        min_pos = torch.full((nb,), n, dtype=idx_dtype, device=device)
        first_sorted_p = torch.full((nb,), n, dtype=idx_dtype, device=device)
        min_pos_p = torch.full((nb,), n, dtype=idx_dtype, device=device)
        cmax = torch.zeros(nb, dtype=torch.float32, device=device)
        nfrag = torch.zeros(nb, dtype=idx_dtype, device=device)
        band_stats_reduce(
            kernel_index(band_id.contiguous()),
            msk_o.contiguous(),
            pos_o.contiguous(),
            cov_o.contiguous(),
            n,
            int(AA_MASK_ALL),
            first_sorted,
            min_pos,
            first_sorted_p,
            min_pos_p,
            cmax,
            nfrag,
            bool(positioned_depth),
            taichi_reduction_index_dtype(),
        )
        first_sorted = first_sorted.to(torch.int64)
        min_pos = min_pos.to(torch.int64)
        first_sorted_p = first_sorted_p.to(torch.int64)
        min_pos_p = min_pos_p.to(torch.int64)
        nfrag = nfrag.to(torch.int64)
        nearest_orig = pos_o.index_select(0, first_sorted)
        sheet_pix = pix_o.index_select(0, first_sorted)
        if positioned_depth:
            has_pos = first_sorted_p < n
            nearest_orig = torch.where(
                has_pos,
                pos_o.index_select(0, first_sorted_p.clamp_max(max(n - 1, 0))),
                nearest_orig,
            )
            min_pos = torch.where(has_pos, min_pos_p, min_pos)
            del first_sorted_p, min_pos_p, has_pos
        else:
            del first_sorted_p, min_pos_p
    else:
        # Same narrowing as the kernel arm, and for the same reason: MPS has no
        # int64 ``scatter_reduce_(reduce='amin')`` either (§2.3). The reduced
        # values are stream positions, so int32 holds every one of them.
        idx_dtype = reduction_index_dtype()
        pos_src = pos_o.to(idx_dtype)
        positions_src = positions.to(idx_dtype)
        first_sorted = torch.full((nb,), n, dtype=idx_dtype, device=device)
        first_sorted.scatter_reduce_(
            0, band_id, positions_src, reduce="amin", include_self=True
        )
        first_sorted = first_sorted.to(torch.int64)
        nearest_orig = pos_o.index_select(0, first_sorted)
        sheet_pix = pix_o.index_select(0, first_sorted)
        min_pos = torch.full((nb,), n, dtype=idx_dtype, device=device)
        min_pos.scatter_reduce_(0, band_id, pos_src, reduce="amin", include_self=True)
        min_pos = min_pos.to(torch.int64)

        # Under ``positioned_depth`` the same two quantities, restricted to the
        # POSITIONED fragments -- the ones that own at least one sub-pixel sample.
        # An area donor owns none: it is a real piece of the surface with a real
        # area, but it has no position among the N sample points at which the
        # resolve compares one sheet against another, so it must not be what
        # decides that comparison. Falls back to the unrestricted values for a
        # sheet with no positioned fragment at all -- an areal, position-less
        # sheet, where there is nothing better to order by. See
        # ``rt_settings.sheet_positioned_depth`` for the defect this repairs.
        # ``big`` is 0-d so masking a lane costs a broadcast rather than a second
        # [n] array, and each masked copy is freed before the next is built: this
        # is the function's memory peak and a per-fragment array is 28 MB at 4K.
        if positioned_depth:
            big = torch.full((), n, dtype=idx_dtype, device=device)
            posn = (msk_o & AA_MASK_ALL) != 0
            masked = torch.where(posn, positions_src, big)
            first_sorted_p = torch.full((nb,), n, dtype=idx_dtype, device=device)
            first_sorted_p.scatter_reduce_(
                0, band_id, masked, reduce="amin", include_self=True
            )
            first_sorted_p = first_sorted_p.to(torch.int64)
            del masked
            has_pos = first_sorted_p < n
            nearest_orig = torch.where(
                has_pos,
                pos_o.index_select(0, first_sorted_p.clamp_max(max(n - 1, 0))),
                nearest_orig,
            )
            del first_sorted_p
            masked = torch.where(posn, pos_src, big)
            del posn, big
            min_pos_p = torch.full((nb,), n, dtype=idx_dtype, device=device)
            min_pos_p.scatter_reduce_(
                0, band_id, masked, reduce="amin", include_self=True
            )
            min_pos_p = min_pos_p.to(torch.int64)
            del masked
            min_pos = torch.where(has_pos, min_pos_p, min_pos)
            del min_pos_p, has_pos
        del pos_src, positions_src

    # ---- sheet_sample_depth: per-sample nearest-owner depths ---------------
    # ``d(sheet, s)``: the exact f32 depth of the sheet's nearest fragment
    # owning sample bit s. See ``_lane_first_owners``, which computes the
    # table (one masked amin scatter per lane in torch, one kernel pass under
    # ``sheet_sample_depth_kernel``).
    sample_depths = None
    if sample_depth:
        sample_depths = _lane_first_owners(band_id, msk_o, t_o, nb, n)
    del t_o
    del positions, msk_o

    # Dominant fragment: largest exact area, earliest original position on
    # ties (deterministic argmax). The fused path already built ``cmax`` and
    # ``nfrag``; only this resolution stays.
    if rt_settings.sheet_band_stats_kernel and nb:
        from algan.rendering.raytracing.sheet_compact_taichi import (
            band_stats_rep_orig,
        )

        idx_dtype = reduction_index_dtype()
        rep_orig = torch.full((nb,), n, dtype=idx_dtype, device=device)
        band_stats_rep_orig(
            kernel_index(band_id.contiguous()),
            pos_o,
            cov_o,
            cmax,
            n,
            rep_orig,
            taichi_reduction_index_dtype(),
        )
        rep_orig = rep_orig.to(torch.int64)
        del cmax, cov_o
    else:
        idx_dtype = reduction_index_dtype()
        cmax = torch.zeros(nb, dtype=torch.float32, device=device)
        cmax.scatter_reduce_(0, band_id, cov_o, reduce="amax", include_self=True)
        is_max = cov_o >= cmax.index_select(0, band_id)
        del cmax, cov_o
        big = torch.full((n,), n, dtype=idx_dtype, device=device)
        cand_pos = torch.where(is_max, pos_o.to(idx_dtype), big)
        del is_max, big
        rep_orig = torch.full((nb,), n, dtype=idx_dtype, device=device)
        rep_orig.scatter_reduce_(0, band_id, cand_pos, reduce="amin", include_self=True)
        rep_orig = rep_orig.to(torch.int64)
        del cand_pos

        nfrag = torch.zeros(nb, dtype=torch.int64, device=device)
        nfrag.scatter_add_(0, band_id, torch.ones_like(band_id))
    del band_id

    # Split-group accounting (diagnostic): groups are triangle-only. Kept
    # device-side end to end -- the group tables are over-allocated to ``nb``
    # (group ids are < the true group count <= nb) and the two counters stay
    # 0-d tensors, evaluated only when something reads them -- because this
    # block used to cost three device syncs per compaction for numbers
    # nothing on the render path consumes.
    group_id = torch.cumsum(new_group.to(torch.int64), 0) - 1
    del new_group
    bands_per_group = torch.zeros(max(nb, 1), dtype=torch.int64, device=device)
    sheet_group = group_id.index_select(0, first_sorted)
    del group_id
    bands_per_group.scatter_add_(
        0,
        sheet_group,
        torch.ones(nb, dtype=torch.int64, device=device),
    )
    tri_group = is_tri.index_select(0, order).index_select(0, first_sorted)
    tri_groups_mask = torch.zeros(max(nb, 1), dtype=torch.bool, device=device)
    tri_groups_mask.scatter_(0, sheet_group, tri_group)
    num_split_groups = ((bands_per_group > 1) & tri_groups_mask).sum()
    num_tri_groups = tri_groups_mask.sum()
    del bands_per_group, tri_groups_mask, sheet_group, tri_group
    # Last read of the sorted stream: from here the function works only in
    # per-sheet arrays, so the per-fragment ones go now rather than at the
    # return (they are 28 MB apiece on a 4K frame).
    del first_sorted, is_tri, order, pos_o

    # Flags: facing from the band key; one-mesh / sliver policy bits from the
    # dominant fragment (uniform per pixel / per emission policy); the sliver
    # bit FORCED on for an empty union, which is an areal positionless sheet
    # whatever its dominant fragment carried.
    rep_msk = frag_msk.index_select(0, rep_orig)
    flags = rep_msk & (~AA_MASK_ALL)
    del rep_msk
    empty_union = union == 0
    flags = flags | torch.where(
        empty_union,
        torch.full_like(flags, AA_SLIVER_BIT),
        torch.zeros_like(flags),
    )
    del empty_union
    sheet_msk = union.to(torch.int32) | flags
    del union, flags

    # ---- Final order: (pixel, classic order of nearest fragment) -----------
    final = torch.argsort(min_pos, stable=True)

    # §4.4's additive sibling compositing, expressed in the weights the walk
    # consumes (see ``_sibling_weights``). Where a band holds one sheet --
    # every band with ``shade_split`` off -- these ARE the sheet's own area
    # and mask, so the resolve reads exactly what it read before.
    sheet_cov_final = sheet_cov.index_select(0, final)
    sheet_msk_final = sheet_msk.index_select(0, final)
    sheet_wgt, sheet_wmsk = sheet_cov_final, sheet_msk_final
    if sheet_band is not None:
        sheet_wgt, sheet_wmsk = _sibling_weights(
            sheet_band.index_select(0, final),
            sheet_cov_final,
            sheet_msk_final,
            band_area,
            band_union,
            band_corr,
        )

    # Two gathers of the PACKED key, so both take the split form under
    # MPS-friendly mode (``gather_packed_key``): a full-width int64 gather on
    # MPS keeps only ~25 significant bits, which would leave every sheet
    # carrying the same depth.
    sheet_key = gather_packed_key(gather_packed_key(frag_key, nearest_orig), final)
    sheet_pix = sheet_pix.index_select(0, final)
    rep_final = rep_orig.index_select(0, final)

    # ---- sheet_sample_depth: classify, floor, cede --------------------------
    # Everything here works on the FINAL-ordered per-sheet arrays; the lose
    # words land in both mask outputs so the resolve (which consumes the
    # weights) and every record reader see the same thing. Off, none of this
    # runs and the outputs above are exactly what they were.
    if sample_depth:
        ppf = int(width) * int(height)
        rep_ref = frag_ref.index_select(0, rep_final)
        is_tri_sheet = rep_ref >= 0
        low = sheet_msk_final & AA_MASK_ALL
        positioned_s = low != 0
        full_s = low == AA_MASK_ALL
        mat_opaque_s = (sheet_msk_final & AA_MAT_OPAQUE_BIT) != 0
        nonareal_s = positioned_s & ((sheet_msk_final & AA_SLIVER_BIT) == 0)
        # The depth table was built in sheet order; everything below works in
        # the final (walk) order.
        sample_depths = sample_depths.index_select(0, final)
        # Band identity and the multi-sheet-band exemption: a band split into
        # siblings (shade-class split, conflict-rank split) claims against
        # band-pooled arithmetic whose single occlusion write ignores slots,
        # so gating a sibling would over-occlude. Its sheets are neither
        # subjects nor enforcers.
        if sheet_band is not None:
            band_of_sheet = sheet_band.index_select(0, final)
        else:
            band_of_sheet = cid_band.index_select(0, final)
        n_bands = int(band_of_sheet.max().item()) + 1
        members = torch.zeros(n_bands, dtype=torch.int64, device=device)
        members.scatter_add_(
            0, band_of_sheet, torch.ones(nb, dtype=torch.int64, device=device)
        )
        only_band = members.index_select(0, band_of_sheet) == 1
        del members
        positive_wgt = sheet_wgt >= 0.0
        # The surface id: one band never spans two meshes, so the dominant
        # fragment's mesh is every member's. Circuits have no sid and are
        # excluded by ``is_tri_sheet`` on both sides.
        f_rel_s = sheet_pix // ppf
        safe_rep = rep_ref.clamp_min(0).to(torch.int64)
        row_to = (f_rel_s + int(time_start)) % merged["tri_obj"].shape[0]
        sheet_sid = merged["tri_obj"][row_to, safe_rep].to(torch.int64)
        del f_rel_s, safe_rep, rep_ref, row_to

        enforcer = (
            is_tri_sheet
            & mat_opaque_s
            & full_s
            & ((sheet_cov_final - 1.0).abs() <= FULL_DUST)
            & only_band
            & positive_wgt
        )
        subject = is_tri_sheet & nonareal_s & only_band & positive_wgt

        # Per-(pixel, sample) floor over the enforcers: the minimum depth AND
        # the second minimum over DIFFERENT-surface entries, so each subject
        # compares against the best OTHER-sid enforcer at that sample.
        other_d = torch.full(
            (nb, AA_NUM_SAMPLES), float("inf"), dtype=torch.float32, device=device
        )
        enf = enforcer.nonzero(as_tuple=True)[0]
        if int(enf.numel()) > 0:
            lanes = torch.arange(AA_NUM_SAMPLES, device=device)
            epk = (
                sheet_pix.index_select(0, enf).unsqueeze(1) * AA_NUM_SAMPLES
                + lanes.view(1, -1)
            ).reshape(-1)
            edepth = sample_depths.index_select(0, enf).reshape(-1)
            esid = (
                sheet_sid.index_select(0, enf)
                .unsqueeze(1)
                .expand(-1, AA_NUM_SAMPLES)
                .reshape(-1)
            )
            ord_e = _lexsort(epk, edepth)
            epk = epk.index_select(0, ord_e)
            edepth = edepth.index_select(0, ord_e)
            esid = esid.index_select(0, ord_e)
            del ord_e
            new_group_e = torch.ones_like(epk, dtype=torch.bool)
            if epk.numel() > 1:
                new_group_e[1:] = epk[1:] != epk[:-1]
            grp = torch.cumsum(new_group_e.to(torch.int64), 0) - 1
            uniq_pk = epk[new_group_e]
            best_d = edepth[new_group_e]
            best_sid = esid[new_group_e]
            diff_sid = esid != best_sid.index_select(0, grp)
            sec_d = torch.full(
                (int(uniq_pk.numel()),),
                float("inf"),
                dtype=torch.float32,
                device=device,
            )
            if bool(diff_sid.any()):
                sec_d.scatter_reduce_(
                    0,
                    grp[diff_sid],
                    edepth[diff_sid],
                    reduce="amin",
                    include_self=True,
                )
            del diff_sid
            del new_group_e, epk, edepth, esid
            qpk = (sheet_pix.unsqueeze(1) * AA_NUM_SAMPLES + lanes.view(1, -1)).reshape(
                -1
            )
            loc = torch.searchsorted(uniq_pk, qpk).clamp_max(int(uniq_pk.numel()) - 1)
            found = uniq_pk.index_select(0, loc) == qpk
            bd = best_d.index_select(0, loc)
            bsid = best_sid.index_select(0, loc)
            sd = sec_d.index_select(0, loc)
            own_here = (
                bsid == sheet_sid.unsqueeze(1).expand(-1, AA_NUM_SAMPLES).reshape(-1)
            ).reshape(-1)
            other_d = torch.where(found & own_here, sd, bd)
            other_d = torch.where(found, other_d, other_d.new_full((), float("inf")))
            other_d = other_d.view(nb, AA_NUM_SAMPLES)
            del found, bd, bsid, sd, loc, qpk, grp, uniq_pk, best_d, best_sid
            del sec_d, lanes

        # Lose: the subject owns s AND the best other-surface enforcer there
        # is strictly nearer beyond depth_tie_epsilon -- exact ties and
        # near-ties keep today's walk order.
        lane_bits = torch.arange(AA_NUM_SAMPLES, device=device)
        owns = ((low.unsqueeze(1) >> lane_bits.view(1, -1)) & 1) == 1
        gate = owns & (other_d < sample_depths - depth_tie_epsilon)
        gate &= subject.unsqueeze(1)
        # ALL OR NOTHING, above a floor. A fragment's depth is evaluated at
        # the centroid of the samples it OWNS (raster_taichi.py:1308-1330), so
        # a lane's depth is that centroid's rather than the lane's: the finer
        # the margin, the less the comparison is entitled to decide anything.
        # A sheet losing only a thin share of its samples is reading exactly
        # that weakest margin, on a pixel it otherwise wins -- measured, ceding
        # there regressed two pixels by 110 and 55 channel values while fixing
        # nothing, because the surface behind does not always claim what was
        # ceded. So a sheet cedes everything it loses or nothing at all, and
        # only once it is losing more than sheet_sample_depth_cede of what
        # it owns.
        n_lose = gate.sum(dim=1)
        n_own = owns.sum(dim=1)
        gate &= (
            n_lose.to(torch.float32) > sheet_sample_depth_cede * n_own.to(torch.float32)
        ).unsqueeze(1)
        del n_lose, n_own
        lose_word = (
            (gate.to(torch.int64) << lane_bits.view(1, -1)).sum(dim=1).to(torch.int32)
        ) << AA_LOSE_SHIFT
        sheet_msk_final = sheet_msk_final | lose_word
        sheet_wmsk = sheet_wmsk | lose_word

    out = {
        "sheet_key": sheet_key,
        "sheet_pix": sheet_pix,
        "sheet_ref": frag_ref.index_select(0, rep_final),
        "sheet_ab": frag_ab.index_select(0, rep_final),
        "sheet_cov": sheet_cov_final,
        "sheet_msk": sheet_msk_final,
        "sheet_wgt": sheet_wgt,
        "sheet_wmsk": sheet_wmsk,
        "sheet_cap": frag_cap.index_select(0, rep_final),
        "sheet_nfrag": nfrag.index_select(0, final),
        "sheet_fused": fused.index_select(0, final),
        "num_sheets": nb,
        "num_groups": num_tri_groups,
        "num_split_groups": num_split_groups,
        "band_rule": band_rule,
        "band_c": float(band_c),
    }

    # CSR aligned with covered_idx: every covered pixel holds at least one
    # fragment, hence at least one sheet, so the two pixel sets coincide.
    counts = torch.zeros(num_covered, dtype=torch.int64, device=device)
    seg = torch.searchsorted(coverage["covered_idx"].to(torch.int64), sheet_pix)
    counts.scatter_add_(0, seg, torch.ones_like(seg))
    offsets = torch.zeros(num_covered + 1, dtype=torch.int64, device=device)
    offsets[1:] = torch.cumsum(counts, 0)
    out["sheet_offsets"] = offsets
    return out
