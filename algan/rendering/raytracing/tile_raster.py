"""Opt-in tiled discovery and proof-based simple-interior sheet construction.

The ordinary coverage/BRDF/shadow/continuation implementations remain shared.
Only candidate generation, primary ordering and proven-simple metadata change.
All queues are count/scan/allocate/write queues: there is no local layer limit.

Scratch is allocator-owned, like the existing candidate expansion and sort
workspace. Only the existing pipeline's final raw/sheet arrays are persistent
arena allocations. No returned tensor aliases rewound arena scratch.
"""

from __future__ import annotations

import torch

from algan.rendering.mps_compat import gather_exact, gather_packed_key, kernel_index
from algan.rendering.raytracing import tile_raster_taichi as kernels
from algan.rendering.raytracing.settings import _scene_has_user_pipeline
from algan.settings import SETTINGS

rt_settings = SETTINGS.raytracing

_INT_MAX = (1 << 31) - 1


def _checked_size(value: int, label: str) -> int:
    value = int(value)
    if value < 0 or value > _INT_MAX:
        raise OverflowError(
            f"Tiled raster {label} exceeds checked int32 capacity: {value}"
        )
    return value


def _prefix(counts: torch.Tensor, label: str) -> tuple[torch.Tensor, int]:
    # Do not perform the scan in int32 and only THEN check for overflow.
    offsets = torch.empty(counts.numel() + 1, dtype=torch.int64, device=counts.device)
    offsets[0] = 0
    torch.cumsum(counts, 0, dtype=torch.int64, out=offsets[1:])
    total = _checked_size(offsets[-1].item(), label)
    return offsets.to(torch.int32), total


def _empty_stream(device: torch.device) -> dict:
    """Inert, non-null backing buffers for the all-simple merge on Metal."""
    return {
        "sheet_key": torch.empty(1, dtype=torch.int64, device=device),
        "sheet_ref": torch.empty(1, dtype=torch.int32, device=device),
        "sheet_ab": torch.empty((1, 2), dtype=torch.float32, device=device),
        "sheet_wgt": torch.empty(1, dtype=torch.float32, device=device),
        "sheet_wmsk": torch.empty(1, dtype=torch.int32, device=device),
        "sheet_cap": torch.empty(1, dtype=torch.float32, device=device),
        "sheet_offsets": torch.zeros(1, dtype=torch.int32, device=device),
        "num_sheets": 0,
    }


def tiled_specs(
    merged,
    tri_screen,
    tri_bounds,
    bez_bounds,
    camera,
    col_row,
    time_start,
    time_end,
    width,
    height,
):
    """Build two-level bins, then tile-clipped rows for the existing kernels.

    Returns ``(specs, stats)``. Fine candidates are generated
    from coarse CSR lists, not by expanding every primitive's pixel bbox.
    Boundaries/straddlers and circuits remain ordinary coverage candidates;
    only a certified triangle may supply an early opaque bound.
    """
    device = merged["tri_pos"].device
    frames = int(time_end) - int(time_start)
    _checked_size(frames * width * height, "frame-window pixels")
    nt, nb = int(merged.get("num_triangles", 0)), int(merged.get("num_circuits", 0))
    nr = _checked_size(frames * (nt + nb), "primitive records")
    stats = {"tile_candidates": 0, "tile_bbox_rejected": 0, "tile_occluded": 0}
    if nr == 0:
        return [], stats
    records = torch.empty((nr, 8), dtype=torch.int32, device=device)
    base = 0
    for kind, nprim, bounds in ((0, nb, bez_bounds), (1, nt, tri_bounds)):
        if not nprim:
            continue
        if bounds is None:
            raise ValueError("Tiled discovery requires conservative precomputed bounds")
        bf, bx, bm, _flags = bounds
        kernels.bbox_records(
            bf,
            bx,
            bm.contiguous().view(torch.uint8),
            records,
            base,
            nprim,
            time_start,
            frames,
            width,
            height,
            kind,
        )
        base += frames * nprim
    cc = torch.empty(nr, dtype=torch.int64, device=device)
    kernels.coarse_counts(records, cc, nr)
    co, nc = _prefix(cc, "coarse incidences")
    del cc
    if not nc:
        return [], stats
    cw = (width + kernels.COARSE_TILE - 1) // kernels.COARSE_TILE
    ch = (height + kernels.COARSE_TILE - 1) // kernels.COARSE_TILE
    _checked_size(frames * cw * ch, "coarse bin IDs")
    cid = torch.empty(nc, dtype=torch.int32, device=device)
    cr = torch.empty_like(cid)
    kernels.coarse_write(records, co, cid, cr, nr, time_start, cw, ch)
    del co
    order = torch.argsort(cid, stable=True)
    cid, cr = gather_exact(cid, order), gather_exact(cr, order)
    coarse_ids, counts = torch.unique_consecutive(cid, return_counts=True)
    coarse_offsets, _ = _prefix(counts, "coarse CSR")
    del cid, counts, order
    nf = _checked_size(coarse_ids.numel() * kernels.FINE_CHILDREN, "fine bins")
    fc = torch.empty(nf, dtype=torch.int64, device=device)
    kernels.fine_counts(
        records, coarse_ids, coarse_offsets, cr, fc, nf, cw, ch, width, height
    )
    fo, nfc = _prefix(fc, "fine incidences")
    candidates = torch.empty((nfc, 9), dtype=torch.int32, device=device)
    fine_ids = torch.empty(nf, dtype=torch.int32, device=device)
    kernels.fine_write(
        records,
        coarse_ids,
        coarse_offsets,
        cr,
        fo,
        candidates,
        fine_ids,
        nf,
        cw,
        ch,
        width,
        height,
    )
    # The sorted list of nonempty fine bins used to exist so the ordering pass
    # could map a pixel to its tile's bucket by binary search. It orders by
    # pixel id directly now (``tile_fragment_order``), so this -- a nonzero, a
    # gather and a sort, one more host drain among them -- is gone with it.
    del records, cr, coarse_ids, coarse_offsets, fc, fine_ids
    stats["tile_candidates"] = nfc
    if not nfc:
        return [], stats
    intervals = torch.empty((nfc, 2), dtype=torch.float32, device=device)
    flags = torch.empty(nfc, dtype=torch.int32, device=device)
    bound = torch.empty(nf, dtype=torch.float32, device=device)
    allow_cull = nt > 0 and not _scene_has_user_pipeline(merged)
    opaque = torch.zeros((frames, max(nt, 1)), dtype=torch.int32, device=device)
    if allow_cull:
        kernels.opaque_material_proofs(
            merged["tri_colors"],
            merged["tri_extra"],
            col_row,
            merged["tri_uvs"],
            merged["tri_tex_meta"],
            merged["textures"],
            opaque,
            int(merged["num_colored_triangles"]),
            nt,
            time_start,
            frames,
        )
    kernels.candidate_proofs(
        candidates,
        intervals,
        flags,
        tri_screen,
        merged["tri_pos"],
        camera,
        opaque,
        nfc,
        width,
        height,
        time_start,
        int(allow_cull),
    )
    kernels.tile_occluders(fo, intervals, flags, bound, nf)
    if allow_cull:
        # Second phase of the occlusion proof: only tiles that ended up with a
        # certified occluder need their other candidates' distance intervals.
        kernels.candidate_near(
            candidates,
            intervals,
            flags,
            bound,
            tri_screen,
            merged["tri_pos"],
            camera,
            nfc,
            width,
            height,
        )
    pc = torch.empty(nfc, dtype=torch.int64, device=device)
    counters = torch.zeros(2, dtype=torch.int32, device=device)
    # Row spans inside the tile, on the same terms as the reference frontend's
    # ``raster_span_candidates``: same kill switch, same minimum box area, and
    # the same conservative row extent. Rejecting a whole tile and following
    # the projection inside the tiles that survive are complementary, not
    # alternatives -- the box form made COUNT test 3.9x the reference's
    # candidate pixels on the nn scene (DESIGN_tiled_primary.md).
    spans = int(bool(rt_settings.raster_span_candidates))
    span_min_area = 4 * kernels.raster_chunk
    kernels.tile_pair_counts(
        candidates,
        flags,
        intervals,
        bound,
        tri_screen,
        pc,
        counters,
        nfc,
        span_min_area,
        spans,
    )
    po, npairs = _prefix(pc, "candidate chunks")
    # ``flags`` outlives the count pass now: it carries the per-candidate
    # box/span choice the write pass must reproduce exactly.
    del intervals, bound, opaque, fo, pc
    if not npairs:
        stats["tile_bbox_rejected"], stats["tile_occluded"] = counters.cpu().tolist()
        return [], stats
    pairs = torch.empty((npairs, 8), dtype=torch.int32, device=device)
    classes = torch.empty(npairs, dtype=torch.int32, device=device)
    kernels.tile_pair_write(
        candidates, po, flags, tri_screen, pairs, classes, nfc, spans
    )
    del flags
    # Group by class with ONE host round-trip, not one per class. Four
    # ``(classes == cls).nonzero()`` calls are four separate full-device
    # drains, and on this frontend the host round-trips are what is left of
    # its cost once the chunks pack properly (nn at UHD: the discovery's own
    # host time is 0.32s in the reference arm and 0.57s here). A stable sort
    # by class leaves each class's pairs in ascending candidate order --
    # exactly the permutation the per-class ``nonzero`` produced -- so the
    # fragment slots, and therefore the primary sort's original-index tie
    # key, are unchanged.
    ordered = gather_exact(pairs, torch.argsort(classes, stable=True))
    sizes = torch.bincount(classes, minlength=4)
    del classes, pairs
    # The diagnostic counters ride the same transfer rather than buying a
    # drain of their own.
    readback = torch.cat((sizes.to(torch.int64), counters.to(torch.int64))).cpu()
    sizes = readback[:4].tolist()
    stats["tile_bbox_rejected"], stats["tile_occluded"] = readback[4:].tolist()
    specs = []
    start = 0
    for count, (kind, opaque_class) in zip(
        sizes, (("bez", True), ("bez", False), ("tri", True), ("tri", False))
    ):
        if count:
            # A row slice of a contiguous [N, 8] tensor is contiguous, which
            # is what the geometry kernels' ndarray arguments require.
            specs.append((kind, ordered[start : start + count], opaque_class))
        start += count
    return specs, stats


def _primary_sort_key(keys, refs, layer_offset):
    """``(depth bin, descending layer)`` packed, the emission's own relation.

    Computed with the SAME Torch expression as the reference global sort, not a
    subtly different reciprocal-multiply inside a new kernel. ``depth_bin`` is
    already clamped into ``[0, 2**31)`` so the shift cannot reach the sign bit,
    and ``layer`` is a non-negative primitive index plus a count.
    """
    from algan.rendering.raytracing.raster_pipeline import _primary_depth_key
    from algan.rendering.raytracing.raster_taichi import _BEZ_BORDER_BITS

    depth_bin = _primary_depth_key(keys) & 0xFFFFFFFF
    layer = torch.where(
        refs < 0,
        (-refs - 1).clamp_min(0) >> _BEZ_BORDER_BITS,
        refs + int(layer_offset),
    ).to(torch.int64)
    return (depth_bin << 32) | (0x7FFFFFFF - layer)


def tile_fragment_order(keys, refs, layer_offset):
    """Order the fragment stream by ``(pixel, depth bin, -layer, emission)``.

    One 32-bit sort of the PIXEL ids, then the existing unbounded run sorter
    within each pixel's run -- the reference frontend's relation, reached with
    a narrower global key than its ``(pixel << 32) | depth_bin``.

    ``torch.sort`` hands back the permutation and the sorted keys together, so
    the run boundaries are a positional comparison on an immutable array and
    cost no gather. ``initialize=False``: ``order`` arrives as that
    permutation, so ``_sort_run`` reads each fragment's key at its ORIGINAL
    index and breaks ties on it, which is what makes the result independent of
    everything upstream.

    This replaced a per-tile pixel-bucket CSR -- a zeroed ``active_tiles x
    FINE_TILE**2`` table, a binary search over the active tiles and an atomic
    per fragment in each of two passes, a ``nonzero``, a second sort and three
    gathers -- along with the active-tile list itself, which existed only to
    be binary-searched.
    """
    from algan.rendering.raytracing.sheet_sort_taichi import key_run_order

    n = _checked_size(keys.numel(), "fragments")
    device = keys.device
    if not n:
        return torch.empty(0, dtype=torch.int64, device=device)
    # int32: a frame-window pixel ordinal is checked against that range when
    # the window is sized, and a narrower key is a cheaper radix sort.
    pixels = (keys >> 32).to(torch.int32)
    run_key, order = torch.sort(pixels, stable=True)
    del pixels
    # int32 like the CSR order it replaces: ``_sort_run`` narrows every load
    # anyway, and this halves what the sort's inner loop moves.
    order = order.to(torch.int32)
    sort_key = _primary_sort_key(keys, refs, layer_offset)
    # ``depth`` is unused at ``depth_key=False``; the kernel still binds an
    # ndarray for it, so it gets the key array rather than an allocation.
    key_run_order(run_key, sort_key, sort_key, order, n, False, False)
    return order.to(torch.int64)


def compact_interior_sheets(
    coverage, merged, camera, pixel_world_scale, time_start, width, height, **options
):
    """Bypass general sheet construction for certified WHOLE pixel runs.

    A full sample mask and area near one are not a certificate. Every retained
    layer must geometrically contain the pixel, use a distinct source surface,
    and have a separated distance interval in the existing walk order. Custom
    pipelines, uncertain opacity, closed shells and all boundaries/crossings
    keep the existing analytic compactor unchanged.
    """
    from algan.rendering.raytracing.sheets import compact_sheets

    def general(cov):
        return compact_sheets(
            cov, merged, camera, pixel_world_scale, time_start, width, height, **options
        )

    n, npixels = int(coverage["num_fragments"]), int(coverage["num_covered"])
    screen = options.get("tri_screen")
    if not n or screen is None or _scene_has_user_pipeline(merged):
        result = general(coverage)
        result["num_simple_pixels"] = 0
        return result
    _checked_size(n, "interior fragments")
    device = coverage["frag_key"].device
    eligible = torch.empty(n, dtype=torch.int32, device=device)
    distances = torch.empty((n, 2), dtype=torch.float32, device=device)
    surface = torch.empty(n, dtype=torch.int32, device=device)
    scratch = torch.empty(n, dtype=torch.int32, device=device)
    simple = torch.empty(npixels, dtype=torch.int32, device=device)
    # Missing declarations are uncertainty, not proof that a surface is open.
    uncertain = merged.get("tri_alpha_uncertain")
    closed = merged.get("tri_closed")
    if uncertain is None or closed is None:
        result = general(coverage)
        result["num_simple_pixels"] = 0
        return result
    kernels.interior_fragment_proofs(
        coverage["frag_key"],
        coverage["frag_ref"],
        coverage["frag_cov"],
        coverage["frag_msk"],
        screen,
        merged["tri_pos"],
        camera,
        merged["tri_obj"],
        uncertain.contiguous().view(torch.uint8),
        closed,
        eligible,
        distances,
        surface,
        n,
        time_start,
        width,
        height,
    )
    kernels.interior_pixels(
        coverage["run_offsets"], eligible, distances, surface, scratch, simple, npixels
    )
    n_simple = int(simple.sum().item())
    del eligible, distances, surface, scratch
    if not n_simple:
        result = general(coverage)
        result["num_simple_pixels"] = 0
        return result
    general_pixels = kernel_index((simple == 0).nonzero(as_tuple=True)[0])
    frag_counts = coverage["run_offsets"][1:] - coverage["run_offsets"][:-1]
    ng = general_pixels.numel()
    general_row = torch.empty(npixels, dtype=torch.int32, device=device)
    if ng:
        idx = general_pixels.to(torch.int64)
        general_row[idx] = torch.arange(ng, dtype=torch.int32, device=device)
        offsets, n_general = _prefix(
            gather_exact(frag_counts, idx), "general fragments"
        )
        indices = torch.empty(n_general, dtype=torch.int32, device=device)
        kernels.select_pixel_fragments(
            general_pixels, coverage["run_offsets"], offsets, indices, ng
        )
        idx_frag = indices.to(torch.int64)
        gcov = {
            name: gather_exact(coverage[name], idx_frag)
            for name in ("frag_ref", "frag_ab", "frag_cov", "frag_msk", "frag_cap")
        }
        gcov["frag_key"] = gather_packed_key(coverage["frag_key"], idx_frag)
        gcov.update(
            covered_idx=gather_exact(coverage["covered_idx"], idx),
            run_offsets=offsets,
            num_fragments=n_general,
            num_covered=ng,
        )
        stream = general(gcov)
        sheet_counts = frag_counts.clone()
        sheet_counts[idx] = (
            stream["sheet_offsets"][1:] - stream["sheet_offsets"][:-1]
        ).to(torch.int32)
        del gcov, indices, idx_frag
    else:
        stream = _empty_stream(device)
        sheet_counts = frag_counts
    out_offsets, ns = _prefix(sheet_counts, "merged sheets")
    result = {
        "sheet_key": torch.empty(ns, dtype=torch.int64, device=device),
        "sheet_ref": torch.empty(ns, dtype=torch.int32, device=device),
        "sheet_ab": torch.empty((ns, 2), dtype=torch.float32, device=device),
        "sheet_wgt": torch.empty(ns, dtype=torch.float32, device=device),
        "sheet_wmsk": torch.empty(ns, dtype=torch.int32, device=device),
        "sheet_cap": torch.empty(ns, dtype=torch.float32, device=device),
        "sheet_offsets": out_offsets,
        "num_sheets": ns,
        "num_simple_pixels": n_simple,
    }
    names = (
        "sheet_key",
        "sheet_ref",
        "sheet_ab",
        "sheet_wgt",
        "sheet_wmsk",
        "sheet_cap",
    )
    kernels.merge_interior_sheets(
        simple,
        general_row,
        coverage["run_offsets"],
        stream["sheet_offsets"],
        out_offsets,
        *(
            coverage[name]
            for name in (
                "frag_key",
                "frag_ref",
                "frag_ab",
                "frag_cov",
                "frag_msk",
                "frag_cap",
            )
        ),
        *(stream[name] for name in names),
        *(result[name] for name in names),
        npixels,
    )
    return result
