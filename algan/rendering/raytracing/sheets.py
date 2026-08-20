"""Sheet compaction: the fragment stream aggregated into per-pixel sheets.

``DESIGN_sheet_resolve.md`` P1 + P2. A **sheet** is a maximal same-surface
region within one pixel — keyed ``(pixel, mesh id, facing, depth band)``,
which ``SHEET_SHADE_SPLIT`` subdivides by flat-face shading class into §4.4
siblings (see ``compact_sheets``) — with each bezier circuit fragment alone
(circuits never group; their border/fill blend is already packed per
fragment). The compaction turns the emission's depth-sorted fragment stream
into the sheet stream: exact area as a sum over the sheet's fragments, the
union of sub-pixel sample masks, the nearest fragment's depth, and a dominant
(largest-area) fragment as the shading reference.

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

from algan.rendering.raytracing.raster_taichi import (
    _AA_BACKFACE_BIT as AA_BACKFACE_BIT,
)
from algan.rendering.raytracing.raster_taichi import (
    _AA_MASK_ALL as AA_MASK_ALL,
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

#: Band rules this module implements. "facing" is the no-depth-split fallback.
BAND_RULES = ("facing", "prim")

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
    ``eff <= MIN_ALPHA`` skip when parity with a kernel is wanted.

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


def _shade_class(merged, frame_rel, time_start, safe_ref, is_tri):
    """Per-fragment shading class for ``shade_split`` (see ``compact_sheets``).

    Returns int64 in ``[0, _SHADE_CLASS_BASE)``: 0 for smooth-shaded
    triangles (and anything the rule cannot classify), ``1 + packed quantized
    unit face normal`` for flat-shaded ones. The flat test mirrors the shade
    kernel's ``_triangle_normal`` exactly: a triangle shades FLAT when its
    three vertex normals are equal (declared flat) or all degenerate (the
    kernel then substitutes the geometric cross-product normal).
    """
    n = safe_ref.numel()
    device = safe_ref.device
    cls = torch.zeros(n, dtype=torch.int64, device=device)
    tri_norm = merged.get("tri_norm")
    tri_pos = merged.get("tri_pos")
    if tri_norm is None or tri_pos is None or not bool(is_tri.any()):
        return cls
    nrm = tri_norm[_rows(tri_norm, frame_rel, time_start), safe_ref].view(-1, 3, 3)
    mag = nrm.norm(dim=2)
    unit = nrm / mag.unsqueeze(2).clamp_min(1e-12)
    spread = (unit - unit[:, :1]).abs().amax(dim=(1, 2))
    declared_flat = (mag.amin(dim=1) > 1e-6) & (spread < 1e-6)
    # All-degenerate vertex normals: the kernel falls back to the geometric
    # normal, so the class does too (the Polyhedron family authors none).
    geometric_flat = mag.amax(dim=1) < 1e-6
    v = tri_pos[_rows(tri_pos, frame_rel, time_start), safe_ref]
    gn = torch.cross(v[:, 3:6] - v[:, 0:3], v[:, 6:9] - v[:, 0:3], dim=1)
    gn = gn / gn.norm(dim=1, keepdim=True).clamp_min(1e-12)
    face_n = torch.where(geometric_flat.unsqueeze(1), gn, unit[:, 0])
    q = (
        torch.round(face_n * float(SHADE_CLASS_QUANT))
        .to(torch.int64)
        .clamp_(-SHADE_CLASS_QUANT, SHADE_CLASS_QUANT)
        + SHADE_CLASS_QUANT
    )
    packed = (q[:, 0] << 16) | (q[:, 1] << 8) | q[:, 2]
    flat = is_tri & (declared_flat | geometric_flat)
    return torch.where(flat, packed + 1, cls)


def _popcount_lanes(bits):
    """Number of set sample bits in each element of an int64 mask tensor."""
    pop = torch.zeros_like(bits)
    for b in range(AA_NUM_SAMPLES):
        pop += (bits >> b) & 1
    return pop


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
    device = cov_o.device
    area64 = torch.zeros(nbands, dtype=torch.float64, device=device)
    area64.scatter_add_(0, band_of_frag, cov_o.to(torch.float64))
    area = area64.to(torch.float32)

    bits = (msk_o & AA_MASK_ALL).to(torch.int64)
    union = torch.zeros(nbands, dtype=torch.int64, device=device)
    lane = torch.zeros(nbands, dtype=torch.int64, device=device)
    for b in range(AA_NUM_SAMPLES):
        lane.zero_()
        lane.scatter_add_(0, band_of_frag, (bits >> b) & 1)
        union |= (lane > 0).to(torch.int64) << b
    sliver = torch.zeros(nbands, dtype=torch.int64, device=device)
    sliver.scatter_reduce_(
        0,
        band_of_frag,
        ((msk_o & AA_SLIVER_BIT) != 0).to(torch.int64),
        reduce="amax",
        include_self=True,
    )

    pop = _popcount_lanes(union)
    clamped = area.clamp(max=1.0)
    full = union == AA_MASK_ALL
    corr = torch.where(
        full,
        torch.where((1.0 - area).abs() <= FULL_DUST, torch.ones_like(clamped), clamped),
        clamped * float(AA_NUM_SAMPLES) / pop.clamp_min(1).to(torch.float32),
    )
    split = (union != 0) & (sliver == 0)
    return area, union, corr, split


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
    colour becomes the area-weighted blend of the siblings' own shading:
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
    multi = members.index_select(0, sheet_band) > 1
    if not bool(multi.any()):
        return cov, msk

    share = cov.to(torch.float64) / band_area.index_select(0, sheet_band).to(
        torch.float64
    ).clamp_min(1e-12)
    p = band_corr.index_select(0, sheet_band).to(torch.float64) * share

    union = band_union.index_select(0, sheet_band)
    pop = _popcount_lanes(union).clamp_min(1).to(torch.float64)
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
):
    """Compact one emission's fragment stream into its sheet stream.

    Parameters mirror what ``prepare_sparse_raster_coverage`` was called with:
    ``coverage`` is its returned dict (the compact ``frag_*`` arrays and the
    per-pixel CSR), ``merged`` the batch's merged scene, ``cam_origin`` /
    ``pixel_world_scale`` the per-frame camera rows the band rule's relative
    scale reads.

    ``shade_split`` (``SHEET_SHADE_SPLIT``) adds a SHADING CLASS to the
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
        sheet — the donors-only case).
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
        and a split group produced more than one band.
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
    positions = torch.arange(n, dtype=torch.int64, device=device)
    # Triangles group by (surface, facing); every bezier fragment is its own
    # group (negative, unique — a shared sentinel would fuse adjacent
    # circuits into one "sheet" no consumer wants). The shading class is NOT
    # part of this key: bands and conflict ranks are decided class-blind, so
    # a band is the same set of fragments whatever ``shade_split`` says, and
    # the class only SUBDIVIDES that band below (see ``_sibling_weights``).
    gkey = torch.where(is_tri, sid * 2 + facing, -(positions + 2))
    cls = None
    if shade_split:
        cls = _shade_class(merged, frame_rel, time_start, safe_ref, is_tri)

    # ---- P1: (pixel, group, depth) order + band starts ---------------------
    order = _lexsort(pix, gkey, t)
    pix_o = pix.index_select(0, order)
    g_o = gkey.index_select(0, order)
    t_o = t.index_select(0, order)

    new_group = torch.ones(n, dtype=torch.bool, device=device)
    if n > 1:
        new_group[1:] = (pix_o[1:] != pix_o[:-1]) | (g_o[1:] != g_o[:-1])

    band_start = new_group.clone()
    if band_rule == "prim" and n > 1 and bool(is_tri.any()):
        tri_pos = merged["tri_pos"]
        v = tri_pos[_rows(tri_pos, frame_rel, time_start), safe_ref]
        ro = cam_origin[_rows(cam_origin, frame_rel, time_start)]
        d = torch.stack(
            [torch.linalg.norm(v[:, 3 * k : 3 * k + 3] - ro, dim=1) for k in range(3)],
            dim=1,
        )
        ext = d.amax(dim=1) - d.amin(dim=1)
        # Per-PIXEL depth slope: two neighbouring fragments of one sheet can
        # differ by about one pixel's worth of the surface's depth gradient,
        # not by the triangle's whole extent. Where the projection table is
        # valid, divide by the projected size in pixels; a camera-plane
        # straddler keeps the conservative raw extent.
        slope = ext
        if tri_screen is not None and tri_screen.shape[2] >= 10:
            rs = _rows(tri_screen, frame_rel, time_start)
            sx = tri_screen[rs, safe_ref, 0:3]
            sy = tri_screen[rs, safe_ref, 3:6]
            proj = torch.maximum(
                sx.amax(dim=1) - sx.amin(dim=1),
                sy.amax(dim=1) - sy.amin(dim=1),
            ).clamp_min_(1.0)
            valid = tri_screen[rs, safe_ref, 9] > 0.5
            slope = torch.where(valid, ext / proj, ext)
        pws = pixel_world_scale[_rows(pixel_world_scale, frame_rel, time_start)]
        scale = torch.where(is_tri, slope + pws * t, torch.zeros_like(t))
        scale_o = scale.index_select(0, order)
        gap = t_o[1:] - t_o[:-1]
        thr = float(band_c) * (scale_o[1:] + scale_o[:-1])
        band_start[1:] |= (~new_group[1:]) & (gap > thr)

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
    # ride with their sheet's owners. Integer cumsums: deterministic.
    positions_sorted = torch.arange(n, dtype=torch.int64, device=device)
    band_first = torch.where(
        band_start, positions_sorted, torch.zeros_like(positions_sorted)
    )
    band_first = torch.cummax(band_first, 0)[0]
    bits_pre = (frag_msk.index_select(0, order) & AA_MASK_ALL).to(torch.int64)
    rank = torch.zeros(n, dtype=torch.int64, device=device)
    for b in range(AA_NUM_SAMPLES):
        lane = (bits_pre >> b) & 1
        excl = torch.cumsum(lane, 0) - lane
        prior = excl - excl.index_select(0, band_first)
        rank = torch.maximum(
            rank, torch.where(lane > 0, prior, torch.zeros_like(prior))
        )
    rank.clamp_(max=15)
    cid = band_id * 16 + rank
    uniq_cid, band_id = torch.unique(cid, sorted=True, return_inverse=True)
    nb = int(uniq_cid.numel())
    if nb == 0:
        return None

    # ---- P2: segmented reduction over bands --------------------------------
    cov_o = frag_cov.index_select(0, order)
    msk_o = frag_msk.index_select(0, order)
    pos_o = order  # original stream position of each sorted fragment

    # ---- The shading-class subdivision -------------------------------------
    # ``band_id`` is now the BAND -- the sheet this compaction would build
    # with the split off. Under ``shade_split`` each band subdivides by
    # shading class into §4.4 siblings, except where there is nothing to
    # anti-alias: an areal (position-less) band stays whole, exactly as it
    # is with the split off (``_band_composite``).
    band_area = band_union = band_corr = sheet_band = None
    if shade_split:
        band_of_frag = band_id
        band_area, band_union, band_corr, band_split = _band_composite(
            band_of_frag, nb, cov_o, msk_o
        )
        cls_o = cls.index_select(0, order)
        cls_eff = torch.where(
            band_split.index_select(0, band_of_frag), cls_o, torch.zeros_like(cls_o)
        )
        skey = band_of_frag * _SHADE_CLASS_BASE + cls_eff
        uniq_skey, band_id = torch.unique(skey, sorted=True, return_inverse=True)
        nb = int(uniq_skey.numel())
        sheet_band = uniq_skey // _SHADE_CLASS_BASE

    # Exact area: float64 accumulate, float32 round (§6.6.4 — a float32
    # scatter_add_ is order-nondeterministic on CUDA and this value feeds
    # thresholds downstream).
    area64 = torch.zeros(nb, dtype=torch.float64, device=device)
    area64.scatter_add_(0, band_id, cov_o.to(torch.float64))
    sheet_cov = area64.to(torch.float32).clamp_min_(0.0)

    # Sample-mask union + the fusion detector, one bit lane at a time
    # (integer adds are exact under any order; a lane count above 1 is the
    # fill-rule partition violation).
    bits = (msk_o & AA_MASK_ALL).to(torch.int64)
    union = torch.zeros(nb, dtype=torch.int64, device=device)
    fused = torch.zeros(nb, dtype=torch.bool, device=device)
    lane = torch.zeros(nb, dtype=torch.int64, device=device)
    for b in range(AA_NUM_SAMPLES):
        lane.zero_()
        lane.scatter_add_(0, band_id, (bits >> b) & 1)
        union |= (lane > 0).to(torch.int64) << b
        fused |= lane > 1

    # Nearest fragment (minimum sorted position -- the stream is depth-sorted
    # within a group, and a rank-split sheet's members need not be
    # consecutive) and the sheet's position in the classic order: the
    # MINIMUM original stream position (the emission is (pixel, depth-bin,
    # descending-layer) sorted, so min-position inherits that relation).
    first_sorted = torch.full((nb,), n, dtype=torch.int64, device=device)
    first_sorted.scatter_reduce_(
        0, band_id, positions_sorted, reduce="amin", include_self=True
    )
    nearest_orig = pos_o.index_select(0, first_sorted)
    sheet_pix = pix_o.index_select(0, first_sorted)
    min_pos = torch.full((nb,), n, dtype=torch.int64, device=device)
    min_pos.scatter_reduce_(0, band_id, pos_o, reduce="amin", include_self=True)

    # Dominant fragment: largest exact area, earliest original position on
    # ties (deterministic argmax).
    cmax = torch.zeros(nb, dtype=torch.float32, device=device)
    cmax.scatter_reduce_(0, band_id, cov_o, reduce="amax", include_self=True)
    is_max = cov_o >= cmax.index_select(0, band_id)
    big = torch.full((n,), n, dtype=torch.int64, device=device)
    cand_pos = torch.where(is_max, pos_o, big)
    rep_orig = torch.full((nb,), n, dtype=torch.int64, device=device)
    rep_orig.scatter_reduce_(0, band_id, cand_pos, reduce="amin", include_self=True)

    nfrag = torch.zeros(nb, dtype=torch.int64, device=device)
    nfrag.scatter_add_(0, band_id, torch.ones_like(band_id))

    # Split-group accounting (diagnostic): groups are triangle-only.
    group_id = torch.cumsum(new_group.to(torch.int64), 0) - 1
    ngroups = int(group_id[-1]) + 1 if n else 0
    bands_per_group = torch.zeros(ngroups, dtype=torch.int64, device=device)
    sheet_group = group_id.index_select(0, first_sorted)
    bands_per_group.scatter_add_(
        0,
        sheet_group,
        torch.ones(nb, dtype=torch.int64, device=device),
    )
    tri_group = is_tri.index_select(0, order).index_select(0, first_sorted)
    tri_groups_mask = torch.zeros(ngroups, dtype=torch.bool, device=device)
    tri_groups_mask.scatter_(0, sheet_group, tri_group)
    num_split_groups = int(((bands_per_group > 1) & tri_groups_mask).sum().item())
    num_tri_groups = int(tri_groups_mask.sum().item())

    # Flags: facing from the band key; one-mesh / sliver policy bits from the
    # dominant fragment (uniform per pixel / per emission policy); the sliver
    # bit FORCED on for an empty union, which is an areal positionless sheet
    # whatever its dominant fragment carried.
    rep_msk = frag_msk.index_select(0, rep_orig)
    flags = rep_msk & (~AA_MASK_ALL)
    empty_union = union == 0
    flags = flags | torch.where(
        empty_union,
        torch.full_like(flags, AA_SLIVER_BIT),
        torch.zeros_like(flags),
    )
    sheet_msk = union.to(torch.int32) | flags

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

    sheet_key = frag_key.index_select(0, nearest_orig).index_select(0, final)
    sheet_pix = sheet_pix.index_select(0, final)
    rep_final = rep_orig.index_select(0, final)
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
