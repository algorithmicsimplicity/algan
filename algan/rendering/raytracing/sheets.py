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
The shipping implementation combines host orchestration with optional fused
kernels in ``sheet_compact_taichi.py``. The PyTorch arms remain comparison
references. On MPS, accumulation uses float32 instead; the historical float64
stability measurement does not establish bitwise reproducibility there.

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

import warnings
from contextlib import nullcontext

import torch

from algan.environment import env_flag, env_float
from algan.errors import AlganWarning
from algan.rendering.mps_compat import (
    accumulate_dtype,
    clamp_floor,
    cummax_values,
    gather_exact,
    gather_packed_key,
    kernel_index,
    mps_friendly,
    reduction_index_dtype,
    taichi_accumulate_dtype,
)
from algan.rendering.raytracing import device_sort
from algan.rendering.raytracing import settings as rt_settings
from algan.rendering.raytracing.array_ops import (
    gather_frame_table,
    gather_rows,
    group_ids_from_starts,
    require_disjoint_output,
    require_tensor_outputs,
)
from algan.rendering.raytracing.raster_taichi import (
    _AA_BACKFACE_BIT as AA_BACKFACE_BIT,  # noqa: F401 -- renderer diagnostic fixtures
)
from algan.rendering.raytracing.raster_taichi import _AA_FULL_DUST as FULL_DUST
from algan.rendering.raytracing.raster_taichi import (
    _AA_LOSE_SHIFT as AA_LOSE_SHIFT,
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
from algan.rendering.raytracing.raytrace_kernels_taichi import (
    depth_tie_epsilon,
)
from algan.rendering.raytracing.sheet_fragments import (
    SortedFragments,
    gather_sorted_fragments,
)
from algan.rendering.raytracing.sheet_geometry import (
    depth_slope_block,
    shade_class_block,
)
from algan.rendering.raytracing.sheet_grouping import (
    RankGroups,
    RankPoolGroups,
    class_groups,
    consecutive_pair_ids,
    rank_key_base,
    unique_ids,
    validate_inverse,
)
from algan.rendering.raytracing.sheet_preprocessing import (
    FragmentMetadata,
    SampleDepthMetadata,
    fragment_metadata,
    sample_depth_metadata,
)
from algan.rendering.raytracing.sheet_reduction_buffers import (
    BandComposite,
    BandReduction,
    SheetWeights,
)
from algan.rendering.raytracing.sheet_shells import (
    ShellSegments,
    apply_shell_ceiling,
    shell_segments,
)
from algan.rendering.raytracing.sheet_statistics import (
    SheetStatistics,
    sheet_statistics,
)
from algan.rendering.raytracing.sheet_workspace import CompactionWorkspace

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

# Reuse ordered pixel and band runs instead of globally sorting their keys
# again. Unsupported backends (a launch that would stage its arguments) use
# torch.
#
# **On by default since it was measured on two GPUs at once**, which is what
# the earlier note ("lower sort time and scratch memory have not yet
# translated into a repeatable whole-render speedup") was missing. The
# emission hands the compaction a stream already grouped by pixel and a CSR
# beside it, so a global lexicographic sort re-derives an order it was given:
# three stable argsorts and two ``index_select``s over the whole stream, where
# one thread per pixel run orders its own handful of fragments in place. The
# permutation is identical -- the comparator carries the original index as its
# last key -- so no output moves.
#
# Isolated, at the 2.9M fragments a warm UHD chunk carries
# (``benchmarks/_device_sort_probe.py``): **4.1 ms against torch's 215 ms on
# Metal, 0.75 ms against 15.8 ms on a T4**. Whole warm renders of
# ``nn_scene_UHD``, ABBA on one box each
# (``reports/mac_2026_09/DEVICE_SORT.md``): **57.2 s -> 43.7 s on the Mac
# runner (-23.5%)** and **8.30 s -> 7.88 s on the T4 (-5.1%)**, with every
# "on" run faster than every "off" run in both.
sheet_pixel_sort = env_flag("ALGAN_SHEET_PIXEL_SORT", True)

# Exact mixed-radix pixel/group/depth keys when their measured ranges fit i64.
# Captured UHD sorts use 37-53% less time; conservative queue-size thresholds
# avoid packing overhead on small inputs. No depth bits are discarded.
sheet_packed_sort = env_flag("ALGAN_SHEET_PACKED_SORT", True)

# Direct group diagnostics and CSR construction; independent of sheet sorting.
sheet_metadata_kernel = env_flag("ALGAN_SHEET_METADATA_KERNEL", False)

# Reuse established group IDs where another grouping cannot subdivide them.
sheet_group_reuse = env_flag("ALGAN_SHEET_GROUP_REUSE", True)

# Assign dense conflict-rank groups from per-band counts instead of sorting.
# Captured UHD input: 15.64 -> 3.29 ms, 120.15 -> 41.45 MiB temporary memory.
# Whole-render warm mean improved 1.7%.
#
# The kernel used to be reached only on CUDA, by name. It is now asked for
# wherever a launch stages nothing, which the Metal adoption made true on an
# Apple GPU as well -- and which turns out to include the arch the exclusion
# was protecting: over 2.9M fragments on the CPU arch the two arms agree
# exactly at **9.2 ms against 52.0 ms**. A launch that WOULD stage still takes
# the torch path.
sheet_rank_groups = env_flag("ALGAN_SHEET_RANK_GROUPS", True)

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


#: Largest magnitude an int32 sort key can carry.
_INT32_MAX = 2**31 - 1


def _narrow_sort_key(key, magnitude_bound):
    """``key`` as int32 when every value provably fits, else ``key`` itself.

    Only ever used for a key handed to :func:`_lexsort`. Narrowing is safe
    there in the strongest sense: a stable argsort depends on the *order* of
    the values and on the input index, and an exact int32 copy of an int64
    key has the same order and the same indices -- so the permutation is
    identical, bit for bit. What changes is the radix sort's pass count, which
    is one per key byte.

    ``magnitude_bound`` must bound ``abs(key)``, and it is the caller's job to
    know it without asking the device: the point of this is to remove work
    from the stream, not to add a reduction and a sync to it.
    """
    if key.dtype is torch.int64 and magnitude_bound <= _INT32_MAX:
        return key.to(torch.int32)
    return key


def _lexsort(*keys, out=None, workspace=None):
    """Stable argsort by ``keys`` in priority order (first key most
    significant). Composes least-significant-first, the classic LSD trick the
    emission's own ``_exact_fragment_order`` uses.

    The device arm (:func:`~algan.rendering.raytracing.device_sort.stable_lexsort`)
    is the same composition with both halves moved off torch: each pass is
    Quadrants' radix sort, and the ``index_select`` that carries the running
    permutation into the next key happens inside that sort's own seed loop. It
    declines wherever it does not apply, which is everywhere but a GPU arch in
    MPS-friendly mode today, and this falls straight through to the torch form.

    Its permutation is int32; widened here so every caller keeps the int64
    order ``torch.argsort`` hands back, which several of them pass on to a
    kernel whose element type is part of its specialization key.
    """
    if out is not None or workspace is not None:
        from algan.rendering.raytracing.sheet_order import stable_lexsort

        return stable_lexsort(*keys, out=out, workspace=workspace)
    order = device_sort.stable_lexsort(*keys)
    if order is not None:
        return order.to(torch.int64)
    order = None
    for key in reversed(keys):
        k = key if order is None else key.index_select(0, order)
        o = torch.argsort(k, stable=True)
        order = o if order is None else order.index_select(0, o)
    return order


def _local_sheet_sort(tensor):
    from algan.rendering.taichi_runtime import _live_arch, taichi_launch_is_local

    return (
        sheet_pixel_sort
        and tensor.numel() > 0
        and _live_arch() is not None
        and taichi_launch_is_local(tensor.device)
    )


def _packed_depth_order(keys, depth, *, out=None, workspace=None):
    # Nonnegative finite float32 depths have the same order as their IEEE bits.
    # Retain every bit, and subtract minima only to save unused key space.
    # Negative zero, negative/nonfinite depths and oversized key ranges keep
    # the reference stable sort, including its tie and NaN semantics.
    if (
        depth.device.type != "cuda"
        or depth.numel() < (32768 if len(keys) > 1 else 262144)
        or any(k.dtype != torch.int64 for k in keys)
        or depth.dtype != torch.float32
        or not depth.is_contiguous()
    ):
        return None
    bits = depth.view(torch.int32)
    bounds = torch.stack([v for k in (*keys, bits) for v in torch.aminmax(k)]).tolist()
    spans = [hi - lo + 1 for lo, hi in zip(bounds[::2], bounds[1::2])]
    capacity = 1
    for span in spans:
        capacity *= span
    if bounds[-2] < 0 or bounds[-1] >= 0x7F800000 or capacity > (1 << 63) - 1:
        return None
    if out is not None or workspace is not None:
        return _sort_packed_depth(
            keys, bits, bounds, spans, out=out, workspace=workspace
        )
    key = keys[0] - bounds[0]
    for i, column in enumerate(keys[1:], 1):
        key.mul_(spans[i]).add_(column - bounds[2 * i])
    key.mul_(spans[-1]).add_(bits - bounds[-2])
    return torch.argsort(key, stable=True)


def _sort_packed_depth(keys, bits, bounds, spans, *, out=None, workspace=None):
    """Pack validated bounded integer columns; keep only the output permutation.

    The caller has proved finite nonnegative depth bits and a composite range
    below signed-int64 capacity. This arithmetic does not change the packed
    path's CUDA/queue-size gate or its choice of the stable PyTorch sort.
    """
    shape, device = bits.shape, bits.device
    if out is None:
        out = torch.empty(shape, dtype=torch.int64, device=device)
    require_tensor_outputs(
        (out,), ((shape, torch.int64),), device=device, inputs=(*keys, bits)
    )
    workspace = workspace or CompactionWorkspace(device=device)
    if workspace.device != device:
        raise ValueError("sort workspace and keys must share a device")
    with workspace.stage():
        key = workspace.tensor(shape, torch.int64)
        torch.sub(keys[0], bounds[0], out=key)
        for i, column in enumerate((*keys[1:], bits), 1):
            with workspace.stage():
                delta = workspace.tensor(shape, column.dtype)
                torch.sub(column, bounds[2 * i], out=delta)
                key.mul_(spans[i]).add_(delta)
        values = workspace.tensor(shape, torch.int64)
        torch.sort(key, stable=True, out=(values, out))
    return out


def _pixel_group_order(
    pix,
    group,
    depth,
    offsets,
    *,
    key_bounds=None,
    memory=None,
    workspace=None,
    out=None,
):
    """Order pixel runs, retaining stable packed/global-sort fallbacks."""
    # Allocate the result before opening sort scratch, regardless of which
    # backend is selected. This is the same reserved permutation on every arm.
    order = out
    if order is not None:
        require_tensor_outputs(
            (order,),
            ((pix.shape, torch.int64),),
            device=pix.device,
            inputs=(pix, group, depth, *(() if offsets is None else (offsets,))),
        )
    elif memory is not None:
        order = memory.get_tensor(pix.shape, torch.int64)
    if offsets is not None and _local_sheet_sort(pix):
        from algan.rendering.raytracing.sheet_sort_taichi import pixel_group_order

        if order is None:
            order = torch.empty(pix.shape, dtype=torch.int64, device=pix.device)
        pixel_group_order(offsets, group, depth, order, offsets.numel() - 1)
        return order
    if sheet_packed_sort:
        packed = (
            _packed_depth_order((pix, group), depth)
            if order is None and workspace is None
            else _packed_depth_order(
                (pix, group), depth, out=order, workspace=workspace
            )
        )
        if packed is not None:
            return packed
    with workspace.stage() if workspace is not None else nullcontext():
        if key_bounds is not None:
            if workspace is None:
                pix = _narrow_sort_key(pix, key_bounds[0])
                group = _narrow_sort_key(group, key_bounds[1])
            else:
                if pix.dtype == torch.int64 and key_bounds[0] <= _INT32_MAX:
                    pix = workspace.copy(pix, torch.int32)
                if group.dtype == torch.int64 and key_bounds[1] <= _INT32_MAX:
                    group = workspace.copy(group, torch.int32)
        return _lexsort(pix, group, depth, out=order, workspace=workspace)


def _key_depth_order(key, depth, *, workspace=None):
    """Stable key/depth order with stage-owned permutation and gathered keys."""
    order = None if workspace is None else workspace.tensor(key.shape, torch.int64)
    if _local_sheet_sort(key):
        from algan.rendering.raytracing.sheet_sort_taichi import key_run_order

        if workspace is None:
            order = torch.argsort(key, stable=True)
            run_key = key.index_select(0, order)
            key_run_order(run_key, key, depth, order, key.numel(), False, True)
        else:
            with workspace.stage():
                _lexsort(key, out=order, workspace=workspace)
                run_key = workspace.gather(key, order)
                key_run_order(run_key, key, depth, order, key.numel(), False, True)
        return order
    if sheet_packed_sort:
        packed = (
            _packed_depth_order((key,), depth)
            if workspace is None
            else _packed_depth_order((key,), depth, out=order, workspace=workspace)
        )
        if packed is not None:
            return packed
    return _lexsort(key, depth, out=order, workspace=workspace)


def _sheet_walk_order(pix, position, *, memory=None, workspace=None):
    """Restore the stable nearest-fragment walk in a caller-owned permutation."""
    order = None if memory is None else memory.get_tensor(pix.shape, torch.int64)
    if _local_sheet_sort(pix):
        from algan.rendering.raytracing.sheet_sort_taichi import key_run_order

        # Depth is inert here: position orders each run, with stable ties.
        if order is None:
            order = torch.empty(pix.shape, dtype=torch.int64, device=pix.device)
        key_run_order(pix, position, position, order, pix.numel(), True, False)
        return order
    return _lexsort(position, out=order, workspace=workspace)


def _unique_sorted_ids(keys, *, out=None):
    """Group nondecreasing integer IDs without sorting them a second time."""
    # The validated Metal path also receives sorted IDs here. Keep its
    # consecutive grouping while retaining the CPU/CUDA optimization gates.
    if keys.device.type == "mps" or (
        (sheet_pixel_sort or sheet_group_reuse) and keys.device.type in ("cpu", "cuda")
    ):
        return unique_ids(keys, consecutive=True, out=out)
    return unique_ids(keys, out=out)


def _sheet_rank_groups(parent, rank, *, workspace=None, out=None):
    """Group ordered dense parent IDs and their full conflict ranks.

    Conflict ranks contain every value from zero to their maximum in each
    parent: each fragment increases a claimed lane's count by one, so the
    running maximum cannot jump over a rank. Ranks may decrease within a
    parent; consecutive unique would therefore be incorrect here.

    Parents and ranks are nonnegative and below the fragment count. Native
    prefix-count grouping needs no composite key. The ordinary reference arm
    uses the stream count as its collision-free radix; the MPS-friendly arm
    sorts bounded parent/rank pairs so it never narrows a wide packed key.
    Both preserve lexicographic group IDs, including decreasing ranks.
    """
    from algan.rendering.taichi_runtime import _live_arch, taichi_launch_is_local

    validate_inverse(out, parent, rank)
    if workspace is not None and workspace.device != parent.device:
        raise ValueError("grouping workspace and inputs must share a device")
    n = parent.numel()
    base = rank_key_base(n)
    if (
        sheet_rank_groups
        and 0 < n < 2**31
        and _live_arch() is not None
        and taichi_launch_is_local(parent.device)
    ):
        from algan.rendering.raytracing.array_ops import csr_offsets
        from algan.rendering.raytracing.sheet_rank_groups_taichi import rank_groups

        parents = int(parent[-1]) + 1
        workspace = workspace or CompactionWorkspace(device=parent.device)
        with workspace.stage():
            counts = workspace.tensor((parents,), torch.int32, 0)
            counts.scatter_reduce_(
                0,
                parent
                if parent.dtype == torch.int64
                else workspace.copy(parent, torch.int64),
                rank
                if rank.dtype == torch.int32
                else workspace.copy(rank, torch.int32),
                reduce="amax",
                include_self=True,
            )
            counts.add_(1)
            offsets = workspace.tensor((parents + 1,), torch.int32)
            csr_offsets(counts, out=offsets)
            nb = int(offsets[-1])
            groups = (
                torch.empty(parent.shape, dtype=torch.int64, device=parent.device)
                if out is None
                else out
            )
            cid_band = torch.empty(nb, dtype=torch.int64, device=parent.device)
            rank_of_cid = torch.empty_like(cid_band)
            rank_groups(
                parent, rank, offsets[1:], groups, cid_band, rank_of_cid, n, parents
            )
        return RankGroups(groups, cid_band, rank_of_cid)
    workspace = workspace or CompactionWorkspace(device=parent.device)
    with workspace.stage():
        if mps_friendly():
            # The shared pair sorter narrows each bounded ID, not their product.
            # Returned descriptors must remain int64 even for int32 inputs.
            parents = (
                parent
                if parent.dtype == torch.int64
                else workspace.copy(parent, torch.int64)
            )
            ranks = (
                rank if rank.dtype == torch.int64 else workspace.copy(rank, torch.int64)
            )
            count, groups, cid_band = class_groups(
                parents, ranks, base, out=out, workspace=workspace
            )
            rank_of_cid = torch.empty((count,), dtype=torch.int64, device=parent.device)
            # Repeated indices all write the same rank of that (parent, rank).
            rank_of_cid.scatter_(0, groups, ranks)
            return RankGroups(groups, cid_band, rank_of_cid)
        key = workspace.tensor(parent.shape, torch.int64)
        key.copy_(parent)
        key.mul_(base).add_(rank)
        keys, groups = unique_ids(key, out=out)
    cid_band = keys // base
    return RankGroups(groups, cid_band, keys - cid_band * base)


def _sheet_class_groups(band_id, cls_eff, new_group, nb, *, out=None, workspace=None):
    """Reuse dense sub-band IDs when each original group has a uniform class."""
    validate_inverse(out, band_id, cls_eff)
    if (
        new_group.shape != band_id.shape
        or new_group.dtype != torch.bool
        or new_group.device != band_id.device
    ):
        raise ValueError("class grouping needs matching boolean group boundaries")
    if out is not None:
        require_disjoint_output(out, new_group)
    if workspace is not None and workspace.device != band_id.device:
        raise ValueError("grouping workspace and inputs must share a device")
    if sheet_group_reuse and cls_eff.device.type in ("cpu", "cuda"):
        # Rank/depth sub-bands never cross an original (pixel, surface, facing)
        # group. Uniform classes in that larger group therefore cannot split
        # any sub-band. This sufficient check permits false negatives only:
        # mixed classes retain the full grouping algorithm below.
        mixed = (cls_eff[1:] != cls_eff[:-1]) & ~new_group[1:]
        if not bool(mixed.any()):
            if out is not None:
                out.copy_(band_id)
            return (
                nb,
                band_id if out is None else out,
                torch.arange(nb, dtype=torch.int64, device=band_id.device),
            )
    return class_groups(
        band_id, cls_eff, _SHADE_CLASS_BASE, out=out, workspace=workspace
    )


def _sheet_group_counts(new_group, band_id, order, is_tri, first_sorted, nb):
    """Count triangle groups and groups containing multiple sheets on device."""
    from algan.rendering.taichi_runtime import _live_arch, taichi_launch_is_local

    n = new_group.numel()
    if (
        sheet_metadata_kernel
        and band_id is not None
        and n
        and _live_arch() is not None
        and taichi_launch_is_local(new_group.device)
    ):
        from algan.rendering.raytracing.sheet_metadata_taichi import group_counts

        # Separate counters per 256 input positions avoid contending on two
        # global scalars. A partition holds at most 256 group starts.
        partial = torch.zeros(
            ((n + 255) // 256, 2), dtype=torch.int32, device=new_group.device
        )
        group_counts(
            new_group.contiguous().view(torch.uint8),
            band_id,
            order,
            is_tri.contiguous().view(torch.uint8),
            partial,
            n,
        )
        totals = partial.sum(dim=0, dtype=torch.int64)
        return totals[0], totals[1]

    group_id = group_ids_from_starts(new_group)
    bands_per_group = torch.zeros(
        max(nb, 1), dtype=torch.int64, device=new_group.device
    )
    sheet_group = group_id.index_select(0, first_sorted)
    del group_id
    bands_per_group.scatter_add_(0, sheet_group, torch.ones_like(sheet_group))
    tri_group = is_tri.index_select(0, order).index_select(0, first_sorted)
    tri_groups_mask = torch.zeros(max(nb, 1), dtype=torch.bool, device=new_group.device)
    tri_groups_mask.scatter_(0, sheet_group, tri_group)
    return tri_groups_mask.sum(), ((bands_per_group > 1) & tri_groups_mask).sum()


def _sheet_offsets(covered_idx, sheet_pix):
    """CSR lower bounds; independent of the optional diagnostic-counting gate."""
    covered = covered_idx.to(torch.int64)
    offsets = torch.empty(
        covered.numel() + 1, dtype=torch.int64, device=sheet_pix.device
    )
    torch.searchsorted(sheet_pix, covered, out=offsets[:-1])
    offsets[-1] = sheet_pix.numel()
    return offsets


#: Target (frame, triangle) pairs per geometry-table block. A single frame is
#: the minimum block, even if it exceeds this budget. Block-local stages bound
#: intermediate lifetimes; the retained tables, dtype and overlapping stages
#: still determine memory use. This is not a total-byte or device-memory cap.
_FRAME_TABLE_BUDGET = 1 << 20

#: Above this many (frame, triangle) pairs the table is not merely large, it is
#: evidence that ``frame_rel`` is wrong: a chunk holds tens of frames and a
#: scene holds a few hundred thousand triangles, so a plausible product is
#: single-digit millions. Blocking means the render survives it either way; the
#: warning is what stops it being silent.
_FRAME_TABLE_IMPLAUSIBLE = 64 << 20
_IMPLAUSIBLE_REPORTED = set()


def _check_frame_table(where, num_frames, num_tri, n, frame_rel=None):
    """Warn once per site when a per-(frame, triangle) table is implausible.

    The table's height is ``frame_rel.amax() + 1``, so an implausible height is
    an implausible fragment key, and the fragment key is packed inside a Taichi
    kernel. Naming the numbers here is what turns "a single 6.45 GB allocation
    failed" into a diagnosis.

    The extra reductions on ``frame_rel`` run only on the warning path, so the
    ordinary render pays one integer comparison for this.
    """
    if num_frames * num_tri < _FRAME_TABLE_IMPLAUSIBLE:
        return
    if where in _IMPLAUSIBLE_REPORTED:
        return
    _IMPLAUSIBLE_REPORTED.add(where)
    span = ""
    if frame_rel is not None and frame_rel.numel():
        span = (
            f" Frame ordinals run {int(frame_rel.amin())}..{int(frame_rel.amax())}"
            f" ({frame_rel.dtype})."
        )
    warnings.warn(
        f"{where}: the per-(frame, triangle) table is {num_frames} frames by "
        f"{num_tri} triangles for {n} fragments, which is not a frame count a "
        f"render chunk can have.{span} The fragment stream's pixel ordinals are "
        "suspect; the table is built in blocks so this does not exhaust "
        "memory, but the classes it feeds may be wrong.",
        AlganWarning,
        stacklevel=3,
    )


def _shade_class(
    merged,
    frame_rel,
    time_start,
    safe_ref,
    is_tri,
    tri_present=None,
    num_frames=None,
    *,
    out=None,
    workspace=None,
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
    workspace = workspace or CompactionWorkspace(device=device)
    if workspace.device != device:
        raise ValueError("shading-class workspace must share the input device")
    if out is None:
        out = torch.empty((n,), dtype=torch.int64, device=device)
    require_tensor_outputs(
        (out,),
        (((n,), torch.int64),),
        device=device,
        inputs=(
            frame_rel,
            safe_ref,
            is_tri,
            *(x for x in (tri_norm, tri_pos) if x is not None),
        ),
    )
    if tri_present is None:
        tri_present = bool(is_tri.any())
    if tri_norm is None or tri_pos is None or not tri_present:
        return out.zero_()
    if num_frames is None:
        num_frames = int(frame_rel.amax()) + 1 if n else 1
    num_tri = tri_norm.numel() // (tri_norm.shape[0] * 9)
    _check_frame_table("sheets._shade_class", num_frames, num_tri, n, frame_rel)
    with workspace.stage():
        table = workspace.tensor((num_frames, num_tri), torch.int64)
        block = max(1, _FRAME_TABLE_BUDGET // max(1, num_tri))
        for f0 in range(0, num_frames, block):
            f1 = min(num_frames, f0 + block)
            with workspace.stage():
                frames = workspace.tensor((f1 - f0,), torch.int64)
                torch.arange(f0, f1, out=frames)
                frames.add_(int(time_start))
                shade_class_block(
                    tri_norm,
                    tri_pos,
                    frames,
                    SHADE_CLASS_QUANT,
                    table[f0:f1],
                    workspace,
                )
        gather_frame_table(table, frame_rel, safe_ref, out=out, workspace=workspace)
        not_triangle = workspace.tensor((n,), torch.bool)
        torch.logical_not(is_tri, out=not_triangle)
        out.masked_fill_(not_triangle, 0)

    return out


def _popcount_lanes(bits, *, out=None, workspace=None):
    """Count low sample bits into checked int32 storage, reusing lane scratch."""
    workspace = workspace or CompactionWorkspace(device=bits.device)
    if workspace.device != bits.device:
        raise ValueError("popcount workspace must share the input device")
    if out is None:
        out = torch.empty(bits.shape, dtype=torch.int32, device=bits.device)
    require_tensor_outputs(
        (out,), ((bits.shape, torch.int32),), device=bits.device, inputs=(bits,)
    )
    n = int(bits.numel())
    with workspace.stage():
        if rt_settings.sheet_mask_kernel and n and bits.dim() == 1:
            from algan.rendering.raytracing.sheet_compact_taichi import mask_popcount

            source = bits if bits.is_contiguous() else workspace.copy(bits)
            mask_popcount(source, n, out)
        else:
            out.zero_()
            lane = workspace.tensor(bits.shape, bits.dtype)
            narrow = (
                lane
                if bits.dtype == torch.int32
                else workspace.tensor(bits.shape, torch.int32)
            )
            for b in range(AA_NUM_SAMPLES):
                torch.bitwise_right_shift(bits, b, out=lane)
                lane.bitwise_and_(1)
                if narrow is not lane:
                    narrow.copy_(lane)
                out.add_(narrow)
    return out


def _band_reduce(
    band_id, msk, cov, nbands, *, want_sliver, want_fused=True, workspace=None, out=None
):
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
    workspace = workspace or CompactionWorkspace(device=device)
    if out is None:
        out = BandReduction(
            torch.empty(nbands, dtype=torch.float32, device=device),
            torch.empty(nbands, dtype=torch.int32, device=device),
            torch.empty(nbands, dtype=torch.bool, device=device)
            if want_fused
            else None,
            torch.empty(nbands, dtype=torch.int32, device=device)
            if want_sliver
            else None,
        )
    require_tensor_outputs(
        out,
        (
            ((nbands,), torch.float32),
            ((nbands,), torch.int32),
            ((nbands,), torch.bool) if want_fused else None,
            ((nbands,), torch.int32) if want_sliver else None,
        ),
        device=device,
        inputs=(band_id, msk, cov),
    )
    area, union, fused, sliver = out
    for value in out:
        if value is not None:
            value.zero_()
    # Outputs belong to the caller. The f64 scratch rounds only after the
    # completed reduction; the f32 compatibility arm accumulates into area.
    with workspace.stage():
        area_acc = area if acc == torch.float32 else workspace.tensor((nbands,), acc, 0)
        if rt_settings.sheet_mask_kernel and n:
            from algan.rendering.raytracing.sheet_compact_taichi import (
                sheet_band_reduce,
            )

            dup = workspace.tensor((nbands if want_fused else 1,), torch.int32, 0)
            sliver_arg = (
                sliver if want_sliver else workspace.tensor((1,), torch.int32, 0)
            )
            sheet_band_reduce(
                kernel_index(band_id.contiguous()),
                msk.contiguous(),
                cov.contiguous(),
                n,
                int(AA_MASK_ALL),
                int(AA_SLIVER_BIT),
                area_acc,
                union,
                dup,
                sliver_arg,
                bool(want_sliver),
                taichi_accumulate_dtype(),
                bool(want_fused),
            )
            if want_fused:
                torch.ne(dup, 0, out=fused)
        else:
            with workspace.stage():
                cov_acc = cov if cov.dtype == acc else workspace.copy(cov, acc)
                area_acc.scatter_add_(0, band_id, cov_acc)
            # Counts keep their original width; only the per-lane masks and
            # shifted union bits are reused instead of allocated each iteration.
            with workspace.stage():
                bits = workspace.copy(msk, torch.int64)
                bits.bitwise_and_(AA_MASK_ALL)
                values = workspace.tensor((n,), torch.int64)
                lane = workspace.tensor((nbands,), torch.int64)
                present = workspace.tensor((nbands,), torch.bool)
                union_bit = workspace.tensor((nbands,), torch.int32)
                for b in range(AA_NUM_SAMPLES):
                    torch.bitwise_right_shift(bits, b, out=values)
                    values.bitwise_and_(1)
                    lane.zero_().scatter_add_(0, band_id, values)
                    torch.gt(lane, 0, out=present)
                    union_bit.copy_(present).bitwise_left_shift_(b)
                    union.bitwise_or_(union_bit)
                    if want_fused:
                        torch.gt(lane, 1, out=present)
                        fused.logical_or_(present)
            if want_sliver:
                with workspace.stage():
                    flag = workspace.tensor((n,), torch.int32)
                    present = workspace.tensor((n,), torch.bool)
                    torch.bitwise_and(msk, AA_SLIVER_BIT, out=flag)
                    torch.ne(flag, 0, out=present)
                    flag.copy_(present)
                    sliver.scatter_reduce_(
                        0, band_id, flag, reduce="amax", include_self=True
                    )
        if acc != torch.float32:
            area.copy_(area_acc)
    return BandReduction(area, union, fused, sliver)


def _conflict_rank(band_start, order, msk, positions, *, out=None, workspace=None):
    """Per-sorted-fragment conflict rank within its band, UNCLAMPED.

    ``rank[j]`` is the largest, over the sample lanes sorted fragment ``j``
    claims, of the number of earlier fragments of the same band claiming that
    same lane (the call site in ``compact_sheets`` explains why the sheet key
    needs it). Returns int32; both arms retain every rank within the stream
    index capacity, and the caller does not clamp the result.

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
    rank = torch.empty(n, dtype=torch.int32, device=device) if out is None else out
    if (
        rank.shape != (n,)
        or rank.dtype != torch.int32
        or rank.device != device
        or not rank.is_contiguous()
    ):
        raise ValueError(
            "conflict-rank output must be a contiguous int32 vector on the input device"
        )
    if out is not None:
        require_disjoint_output(rank, band_start, order, msk, positions)
    workspace = workspace or CompactionWorkspace(device=device)
    if not rt_settings.sheet_rank_kernel or n == 0:
        with workspace.stage():
            band_first = torch.where(band_start, positions, torch.zeros_like(positions))
            band_first = cummax_values(band_first, 0)
            bits_pre = workspace.tensor((n,), torch.int32)
            torch.index_select(msk, 0, order, out=bits_pre)
            bits_pre.bitwise_and_(AA_MASK_ALL)
            lane = workspace.tensor((n,), torch.int32)
            excl = workspace.tensor((n,), torch.int32)
            prior = workspace.tensor((n,), torch.int32)
            unowned = workspace.tensor((n,), torch.bool)
            rank.zero_()
            for b in range(AA_NUM_SAMPLES):
                torch.bitwise_right_shift(bits_pre, b, out=lane)
                lane.bitwise_and_(1)
                torch.cumsum(lane, 0, dtype=torch.int32, out=excl)
                excl.sub_(lane)
                torch.index_select(excl, 0, band_first, out=prior)
                torch.sub(excl, prior, out=prior)
                torch.eq(lane, 0, out=unowned)
                prior.masked_fill_(unowned, 0)
                torch.maximum(rank, prior, out=rank)
        return rank
    from algan.rendering.raytracing.sheet_compact_taichi import sheet_conflict_rank

    # Row zero starts a band even when its flag is clear; every row is written.
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
    *,
    out=None,
    workspace=None,
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
    workspace = workspace or CompactionWorkspace(device=device)
    if workspace.device != device:
        raise ValueError("primitive-split workspace must share the input device")
    shape = (max(0, t_o.numel() - 1),)
    if out is None:
        out = torch.empty(shape, dtype=torch.bool, device=device)
    require_tensor_outputs(
        (out,),
        ((shape, torch.bool),),
        device=device,
        inputs=(
            tri_pos,
            cam_origin,
            pixel_world_scale,
            frame_rel,
            safe_ref,
            is_tri,
            t,
            t_o,
            order,
            *(() if tri_screen is None else (tri_screen,)),
        ),
    )
    if num_frames is None:
        num_frames = int(frame_rel.amax()) + 1 if safe_ref.numel() else 1
    num_tri = tri_pos.numel() // (tri_pos.shape[0] * 9)
    _check_frame_table(
        "sheets._prim_split_after", num_frames, num_tri, t.numel(), frame_rel
    )
    # Keep the original frame blocking and table-storage rounding. Only the
    # gathered slopes survive table construction; each block reuses its scratch.
    with workspace.stage():
        scale_dtype = torch.promote_types(
            tri_pos.dtype, torch.promote_types(pixel_world_scale.dtype, t.dtype)
        )
        scale_o = workspace.tensor(t.shape, scale_dtype)
        with workspace.stage():
            slope_f = workspace.tensor(frame_rel.shape, tri_pos.dtype)
            with workspace.stage():
                slope = workspace.tensor((num_frames, num_tri), tri_pos.dtype)
                block = max(1, _FRAME_TABLE_BUDGET // max(1, num_tri))
                for f0 in range(0, num_frames, block):
                    f1 = min(num_frames, f0 + block)
                    with workspace.stage():
                        frames = workspace.tensor((f1 - f0,), torch.int64)
                        torch.arange(f0, f1, out=frames)
                        frames.add_(int(time_start))
                        depth_slope_block(
                            tri_pos,
                            cam_origin,
                            tri_screen,
                            frames,
                            slope[f0:f1],
                            workspace,
                        )
                gather_frame_table(
                    slope, frame_rel, safe_ref, out=slope_f, workspace=workspace
                )
            rows = workspace.copy(frame_rel, torch.int64)
            rows.add_(int(time_start)).remainder_(pixel_world_scale.shape[0])
            pws = workspace.gather(pixel_world_scale, rows)
            pixel_size = workspace.tensor(
                t.shape, torch.promote_types(pws.dtype, t.dtype)
            )
            torch.mul(pws, t, out=pixel_size)
            scale = workspace.tensor(t.shape, scale_dtype)
            torch.add(slope_f, pixel_size, out=scale)
            not_triangle = workspace.tensor(is_tri.shape, torch.bool)
            torch.logical_not(is_tri, out=not_triangle)
            scale.masked_fill_(not_triangle, 0)
            gather_rows(scale, order, out=scale_o)
        threshold = workspace.tensor(shape, scale_dtype)
        torch.add(scale_o[1:], scale_o[:-1], out=threshold)
        threshold.mul_(float(band_c))
        gap = workspace.tensor(shape, t_o.dtype)
        torch.sub(t_o[1:], t_o[:-1], out=gap)
        torch.gt(gap, threshold, out=out)

    return out


def _band_composite(band_of_frag, nbands, cov_o, msk_o, *, workspace=None, out=None):
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
    workspace = workspace or CompactionWorkspace(device=cov_o.device)
    if out is None:
        out = BandComposite(
            torch.empty(nbands, dtype=torch.float32, device=cov_o.device),
            torch.empty(nbands, dtype=torch.int32, device=cov_o.device),
            torch.empty(nbands, dtype=torch.float32, device=cov_o.device),
            torch.empty(nbands, dtype=torch.bool, device=cov_o.device),
        )
    require_tensor_outputs(
        out,
        (
            ((nbands,), torch.float32),
            ((nbands,), torch.int32),
            ((nbands,), torch.float32),
            ((nbands,), torch.bool),
        ),
        device=cov_o.device,
        inputs=(band_of_frag, cov_o, msk_o),
    )
    area, union, corr, split = out
    with workspace.stage():
        sliver = workspace.tensor((nbands,), torch.int32)
        _band_reduce(
            band_of_frag,
            msk_o,
            cov_o,
            nbands,
            want_sliver=True,
            want_fused=False,
            workspace=workspace,
            out=BandReduction(area, union, None, sliver),
        )
        pop = workspace.tensor((nbands,), torch.int32)
        _popcount_lanes(union, out=pop, workspace=workspace)
        clamped = workspace.tensor((nbands,), torch.float32)
        torch.clamp(area, max=1.0, out=clamped)
        full = workspace.tensor((nbands,), torch.bool)
        torch.eq(union, AA_MASK_ALL, out=full)
        # Keep subtract/abs, multiply and divide as separate f32 operations.
        delta = workspace.tensor((nbands,), torch.float32)
        torch.sub(1.0, area, out=delta)
        delta.abs_()
        predicate = workspace.tensor((nbands,), torch.bool)
        torch.le(delta, FULL_DUST, out=predicate)
        full_corr = workspace.tensor((nbands,), torch.float32)
        torch.where(
            predicate, workspace.tensor((), torch.float32, 1.0), clamped, out=full_corr
        )
        pop.clamp_min_(1)
        denominator = workspace.copy(pop, torch.float32)
        partial = workspace.tensor((nbands,), torch.float32)
        torch.mul(clamped, float(AA_NUM_SAMPLES), out=partial)
        partial.div_(denominator)
        torch.where(full, full_corr, partial, out=corr)
        torch.ne(union, 0, out=split)
        torch.eq(sliver, 0, out=predicate)
        split.logical_and_(predicate)

    return out


def _rank_pool_groups(
    cid_band, rank_of_cid, band_of_frag, cov_o, msk_o, nb, *, workspace=None, out=None
):
    """Which conflict-rank sub-bands composite as §4.4 siblings of one band.

    Returns ``(n_group, group_of_cid)``: the compositing-group count and, per
    sub-band (per ``cid``), the group it claims into. A group is one whole band
    where that band covers the pixel once -- full sample union, exact area
    within :data:`sheet_rank_pool_layers` -- and the sub-band itself, today's
    behaviour, everywhere else. Sheets are NOT merged either way: the split
    stays exactly where the fill rule put it, and only the compositing
    arithmetic pools, so ``sheet_fused`` keeps meaning what it meant.

    ``group_of_cid`` is ``None`` when no band pooled, so a stream that gains
    nothing from this takes exactly the path it took before it existed. A
    supplied int64 ``out`` owns the retained inverse; its contents are unused
    when the returned IDs are ``None``. Validate it before any output writes.

    The test is per band, over the WHOLE band's fragments: exact-area sum and
    sample union, both from one ``_band_reduce`` pass. That pass is the cost,
    and it is skipped outright on a stream where no band was rank-split at all
    (``n_pool == nb``) -- 43,065 bands and 180 splits on the frame this was
    measured on, so it is the split streams that pay.
    """
    base = rank_key_base(nb)
    validate_inverse(out, cid_band, rank_of_cid)
    if cid_band.shape != (nb,):
        raise ValueError("rank-pool inputs must have one entry per sub-band")
    device = cid_band.device
    if (
        band_of_frag.ndim != 1
        or band_of_frag.dtype not in (torch.int32, torch.int64)
        or cov_o.shape != band_of_frag.shape
        or msk_o.shape != band_of_frag.shape
        or cov_o.dtype != torch.float32
        or msk_o.dtype != torch.int32
        or any(x.device != device for x in (band_of_frag, cov_o, msk_o))
    ):
        raise ValueError("rank pooling needs matching fragment IDs, coverage and masks")
    if out is not None:
        require_tensor_outputs(
            (out,),
            (((nb,), torch.int64),),
            device=device,
            inputs=(cid_band, rank_of_cid, band_of_frag, cov_o, msk_o),
        )
    workspace = workspace or CompactionWorkspace(device=device)
    if workspace.device != device:
        raise ValueError("rank-pool workspace must share the input device")
    with workspace.stage():
        # Parents are ordered, with repeated and potentially missing IDs. Unique
        # still owns its temporary inverse; only its retained copy is staged.
        pool_of_cid = workspace.tensor((nb,), torch.int64)
        uniq_pre, _ = _unique_sorted_ids(cid_band, out=pool_of_cid)
        n_pool = int(uniq_pre.numel())
        del uniq_pre
        if n_pool == nb:
            return RankPoolGroups(nb, None)
        # Keep the key, but release reduction and membership scratch before the
        # second unique. This is maximum overlapping storage, not two sums.
        key = workspace.tensor((nb,), torch.int64)
        with workspace.stage():
            reduced = BandReduction.allocate(
                workspace, n_pool, want_fused=False, want_sliver=False
            )
            with workspace.stage():
                pool_of_frag = workspace.gather(pool_of_cid, band_of_frag)
                area, union, _fused, _sliver = _band_reduce(
                    pool_of_frag,
                    msk_o,
                    cov_o,
                    n_pool,
                    want_sliver=False,
                    want_fused=False,
                    workspace=workspace,
                    out=reduced,
                )
            # Only a full union near unit area pools. Partial unions cannot
            # distinguish a silhouette from multiple layers; keep that policy.
            fuse = workspace.tensor((n_pool,), torch.bool)
            within_area = workspace.tensor((n_pool,), torch.bool)
            torch.eq(union, AA_MASK_ALL, out=fuse)
            torch.le(area, float(sheet_rank_pool_layers), out=within_area)
            fuse.logical_and_(within_area)
            selected = workspace.gather(fuse, pool_of_cid)
            key.copy_(rank_of_cid).masked_fill_(selected, 0)
            if not mps_friendly():
                parent_key = workspace.copy(pool_of_cid)
                parent_key.mul_(base)
                key.add_(parent_key)
        # Zeroing all ranks of a selected parent cannot reorder other parents
        # or their remaining ranks. Preserve the existing unique policy.
        if mps_friendly():
            # These descriptor pairs are already ordered, even after selected
            # ranks become zero. Keep the no-sort Metal policy without packing.
            n_group, group_of_cid = consecutive_pair_ids(
                pool_of_cid, key, out=out, workspace=workspace
            )
        else:
            uniq_key, group_of_cid = _unique_sorted_ids(key, out=out)
            n_group = int(uniq_key.numel())
        if n_group == nb:
            return RankPoolGroups(nb, None)
        return RankPoolGroups(n_group, group_of_cid)


def _sibling_weights(
    sheet_band, cov, msk, band_area, band_union, band_corr, *, workspace=None, out=None
):
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
    workspace = workspace or CompactionWorkspace(device=cov.device)
    nb = sheet_band.numel()
    if out is not None:
        require_tensor_outputs(
            out,
            (((nb,), cov.dtype), ((nb,), msk.dtype)),
            device=cov.device,
            inputs=(sheet_band, cov, msk, band_area, band_union, band_corr),
        )
    with workspace.stage():
        if rt_settings.sheet_sibling_weights_kernel:
            if nb < 2:
                if out is None:
                    return SheetWeights(cov, msk)
                out[0].copy_(cov)
                out[1].copy_(msk)
                return SheetWeights(*out)
            from algan.rendering.raytracing.sheet_sibling_taichi import (
                sibling_band_counts,
                sibling_coverage_weights,
            )

            counts = workspace.tensor((band_area.numel(), 2), torch.int32, 0)
            weights, masks = (
                (torch.empty_like(cov), torch.empty_like(msk)) if out is None else out
            )
            band = sheet_band.contiguous()
            sibling_band_counts(band, nb, counts)
            sibling_coverage_weights(
                band,
                cov.contiguous(),
                msk.contiguous(),
                band_area.contiguous(),
                band_union.contiguous(),
                band_corr.contiguous(),
                counts,
                nb,
                weights,
                masks,
                taichi_accumulate_dtype(),
            )
            return SheetWeights(weights, masks)

        # Retain only the per-sheet decision across the membership stage.
        # Interleaved bands still composite sheet by sheet: the deferred sum
        # must reach the band's last sibling without another surface intervening.
        multi = workspace.tensor((nb,), torch.bool)
        with workspace.stage():
            members = workspace.tensor(band_area.shape, torch.int64, 0)
            starts = workspace.tensor((nb,), torch.int64, 1)
            members.scatter_add_(0, sheet_band, starts)
            runs = workspace.tensor(band_area.shape, torch.int64, 0)
            changed = workspace.tensor((nb,), torch.bool, True)
            if nb > 1:
                torch.ne(sheet_band[1:], sheet_band[:-1], out=changed[1:])
            starts.copy_(changed)
            runs.scatter_add_(0, sheet_band, starts)
            counts = workspace.gather(members, sheet_band)
            torch.gt(counts, 1, out=multi)
            gather_rows(runs, sheet_band, out=counts)
            torch.eq(counts, 1, out=changed)
            multi.logical_and_(changed)
        if not bool(multi.any()):
            if out is None:
                return SheetWeights(cov, msk)
            out[0].copy_(cov)
            out[1].copy_(msk)
            return SheetWeights(*out)

        if out is None:
            out = SheetWeights(
                torch.empty_like(
                    cov, dtype=torch.promote_types(cov.dtype, torch.float32)
                ),
                torch.empty_like(msk),
            )
        weights, masks = out
        acc = accumulate_dtype()
        with workspace.stage():
            area = workspace.tensor((nb,), acc)
            p = workspace.tensor((nb,), acc)
            with workspace.stage():
                gathered = workspace.gather(band_area, sheet_band)
                area.copy_(gathered)
                p.copy_(workspace.gather(band_corr, sheet_band))
            # Preserve the tiny-floor MPS workaround, the wide division and
            # multiplication, then the original f32 rounding before negation.
            clamp_floor(area, 1e-12, out=area, workspace=workspace)
            share = workspace.copy(cov, acc)
            share.div_(area)
            p.mul_(share)
            union = workspace.gather(band_union, sheet_band)
            pop = workspace.tensor((nb,), torch.int32)
            _popcount_lanes(union, out=pop, workspace=workspace)
            pop.clamp_min_(1)
            pop_acc = workspace.copy(pop, acc)
            partial = workspace.tensor((nb,), acc)
            torch.mul(p, pop_acc, out=partial)
            partial.div_(float(AA_NUM_SAMPLES))
            full = workspace.tensor((nb,), torch.bool)
            torch.eq(union, AA_MASK_ALL, out=full)
            torch.where(full, p, partial, out=partial)
            rounded = (
                weights
                if weights.dtype == torch.float32
                else workspace.tensor((nb,), torch.float32)
            )
            rounded.copy_(partial)

            # The sign (including negative zero) marks continuation at the
            # next sheet, not merely membership in the same band.
            cont = workspace.tensor((nb,), torch.bool, False)
            if nb > 1:
                torch.eq(sheet_band[1:], sheet_band[:-1], out=cont[:-1])
            cont.logical_and_(multi)
            negative = workspace.tensor((nb,), torch.float32)
            torch.neg(rounded, out=negative)
            torch.where(cont, negative, rounded, out=rounded)
            torch.where(multi, rounded, cov, out=weights)
            # Donors receive the band's union, without their own sliver flag.
            torch.bitwise_and(msk, ~AA_MASK_ALL, out=masks)
            masks.bitwise_and_(~AA_SLIVER_BIT).bitwise_or_(union)
            torch.where(multi, masks, msk, out=masks)
        return SheetWeights(*out)


def _lane_first_owners(band_id, msk_o, t_o, nb, n, *, out=None, workspace=None):
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
    out = (
        torch.empty((nb, AA_NUM_SAMPLES), dtype=torch.float32, device=device)
        if out is None
        else out
    )
    if (
        out.shape != (nb, AA_NUM_SAMPLES)
        or out.dtype != torch.float32
        or out.device != device
        or not out.is_contiguous()
    ):
        raise ValueError(
            "lane-depth output must be a contiguous float32 [sheets, samples] table on the input device"
        )
    require_disjoint_output(out, band_id, msk_o, t_o)
    if n == 0:
        out.fill_(float("inf"))
        return out
    workspace = workspace or CompactionWorkspace(device=device)
    with workspace.stage():
        if rt_settings.sheet_sample_depth_kernel and nb:
            from algan.rendering.raytracing.sheet_compact_taichi import (
                sheet_lane_first_owner,
            )

            reuse = bool(
                rt_settings.sheet_depth_reduce_kernel
                and rt_settings.sheet_depth_buffer_reuse
            )
            # The final destination itself can hold temporary integer owners.
            # It belongs to the caller, not this helper's soon-reclaimed stage.
            first_lane = (
                out.view(torch.int32).view(-1)
                if reuse
                else workspace.tensor((nb * AA_NUM_SAMPLES,), torch.int32)
            )
            first_lane.fill_(n)
            sheet_lane_first_owner(
                kernel_index(band_id.contiguous()),
                msk_o.contiguous(),
                n,
                int(AA_MASK_ALL),
                first_lane,
            )
            if rt_settings.sheet_depth_reduce_kernel:
                if reuse:
                    from algan.rendering.raytracing.sheet_depth_taichi import (
                        sheet_lane_depths_inplace,
                    )

                    sheet_lane_depths_inplace(first_lane, t_o, n)
                else:
                    from algan.rendering.raytracing.sheet_depth_taichi import (
                        sheet_lane_depths,
                    )

                    sheet_lane_depths(first_lane, t_o, n, out)
            else:
                missing = workspace.tensor(first_lane.shape, torch.bool)
                torch.ge(first_lane, n, out=missing)
                first_lane.clamp_max_(n - 1)
                gather_rows(t_o, first_lane, out=out.view(-1))
                out.view(-1).masked_fill_(missing, float("inf"))
            return out

        idx_dtype = reduction_index_dtype()
        big = workspace.tensor((), idx_dtype, n)
        positions = workspace.tensor((n,), idx_dtype)
        torch.arange(n, out=positions)
        masked = workspace.tensor((n,), idx_dtype)
        first_sorted = workspace.tensor((nb,), idx_dtype)
        inf = workspace.tensor((), torch.float32, float("inf"))
        bits = workspace.tensor(msk_o.shape, msk_o.dtype)
        owns = workspace.tensor(msk_o.shape, torch.bool)
        first_long = (
            first_sorted
            if idx_dtype == torch.int64
            else workspace.tensor((nb,), torch.int64)
        )
        has = workspace.tensor((nb,), torch.bool)
        d_lane = workspace.tensor((nb,), torch.float32)
        for lane in range(AA_NUM_SAMPLES):
            torch.bitwise_right_shift(msk_o, lane, out=bits)
            bits.bitwise_and_(1)
            torch.ne(bits, 0, out=owns)
            torch.where(owns, positions, big, out=masked)
            first_sorted.fill_(n)
            first_sorted.scatter_reduce_(
                0, band_id, masked, reduce="amin", include_self=True
            )
            if first_long is not first_sorted:
                first_long.copy_(first_sorted)
            torch.lt(first_long, n, out=has)
            # Owner indices are dead after this gather; the next lane resets
            # first_sorted. Keep validity before clamping the sentinel.
            first_long.clamp_max_(n - 1)
            gather_rows(t_o, first_long, out=d_lane)
            torch.where(has, d_lane, inf, out=out[:, lane])
    return out


def _sample_depth_lose_reference(
    sheet_pix,
    sample_depths,
    sheet_sid,
    enforcer,
    subject,
    low,
    *,
    out=None,
    workspace=None,
):
    """Expanded-lane reference with checked output and phase-local scratch.

    Preserve the two OTHER-surface minima, stable sort, strict depth epsilon,
    and all-or-nothing ceding threshold. Dynamic nonzero indices and library
    sort workspace remain external; their payloads and every fixed-size result
    are stage-owned. An omitted output retains ordinary ownership.
    """
    nb, device = sheet_pix.numel(), sheet_pix.device
    shape = (nb, AA_NUM_SAMPLES)
    inputs = (sheet_pix, sample_depths, sheet_sid, enforcer, subject, low)
    integer = (torch.int32, torch.int64)
    if (
        sheet_pix.ndim != 1
        or sheet_pix.dtype not in integer
        or sample_depths.shape != shape
        or sample_depths.dtype != torch.float32
        or sheet_sid.shape != (nb,)
        or sheet_sid.dtype not in integer
        or low.shape != (nb,)
        or low.dtype not in integer
        or enforcer.shape != (nb,)
        or enforcer.dtype != torch.bool
        or subject.shape != (nb,)
        or subject.dtype != torch.bool
        or any(value.device != device for value in inputs)
    ):
        raise ValueError("sample-depth reference needs matching sheet and lane arrays")
    workspace = workspace or CompactionWorkspace(device=device)
    if workspace.device != device:
        raise ValueError("sample-depth workspace must share the input device")
    if out is None:
        out = torch.empty((nb,), dtype=torch.int32, device=device)
    require_tensor_outputs(
        (out,), (((nb,), torch.int32),), device=device, inputs=inputs
    )
    with workspace.stage():
        other_d = workspace.tensor(shape, torch.float32, float("inf"))
        with workspace.stage():
            enf = enforcer.nonzero(as_tuple=True)[0]
            ne = int(enf.numel())
            if ne:
                size = ne * AA_NUM_SAMPLES
                # Reserve descriptor capacity before sorting so expanded input
                # and grouping scratch can be reclaimed before querying sheets.
                uniq_pk = workspace.tensor((size,), torch.int64)
                best_d = workspace.tensor((size,), torch.float32)
                best_sid = workspace.tensor((size,), sheet_sid.dtype)
                sec_d = workspace.tensor((size,), torch.float32, float("inf"))
                lanes = workspace.tensor((AA_NUM_SAMPLES,), torch.int64)
                torch.arange(AA_NUM_SAMPLES, out=lanes)
                with workspace.stage():
                    epk = workspace.tensor((size,), torch.int64)
                    edepth = workspace.tensor((size,), torch.float32)
                    esid = workspace.tensor((size,), sheet_sid.dtype)
                    with workspace.stage():
                        pixels = workspace.gather(sheet_pix, enf)
                        pixels.mul_(AA_NUM_SAMPLES)
                        raw_key = workspace.tensor((ne, AA_NUM_SAMPLES), torch.int64)
                        torch.add(pixels.unsqueeze(1), lanes.view(1, -1), out=raw_key)
                        raw_depth = workspace.gather(sample_depths, enf).view(-1)
                        surfaces = workspace.gather(sheet_sid, enf)
                        raw_sid = workspace.tensor(
                            (ne, AA_NUM_SAMPLES), sheet_sid.dtype
                        )
                        raw_sid.copy_(surfaces.unsqueeze(1).expand_as(raw_sid))
                        del enf
                        order = workspace.tensor((size,), torch.int64)
                        _lexsort(
                            raw_key.view(-1), raw_depth, out=order, workspace=workspace
                        )
                        gather_rows(raw_key.view(-1), order, out=epk)
                        gather_rows(raw_depth, order, out=edepth)
                        gather_rows(raw_sid.view(-1), order, out=esid)
                    starts = workspace.tensor((size,), torch.bool, True)
                    if size > 1:
                        torch.ne(epk[1:], epk[:-1], out=starts[1:])
                    groups = workspace.tensor((size,), torch.int64)
                    group_ids_from_starts(starts, out=groups)
                    first = starts.nonzero(as_tuple=True)[0]
                    count = int(first.numel())
                    uniq_pk, best_d = uniq_pk[:count], best_d[:count]
                    best_sid, sec_d = best_sid[:count], sec_d[:count]
                    gather_rows(epk, first, out=uniq_pk)
                    gather_rows(edepth, first, out=best_d)
                    gather_rows(esid, first, out=best_sid)
                    del first
                    diff_sid = workspace.tensor((size,), torch.bool)
                    torch.ne(esid, workspace.gather(best_sid, groups), out=diff_sid)
                    if bool(diff_sid.any()):
                        different = diff_sid.nonzero(as_tuple=True)[0]
                        sec_d.scatter_reduce_(
                            0,
                            workspace.gather(groups, different),
                            workspace.gather(edepth, different),
                            reduce="amin",
                            include_self=True,
                        )
                        del different
                # Only descriptors survive here. Query workspace reuses the
                # expanded sort/group region rather than overlapping with it.
                with workspace.stage():
                    pixels = workspace.copy(sheet_pix)
                    pixels.mul_(AA_NUM_SAMPLES)
                    qpk = workspace.tensor(shape, torch.int64)
                    torch.add(pixels.unsqueeze(1), lanes.view(1, -1), out=qpk)
                    qpk = qpk.view(-1)
                    loc = workspace.tensor(qpk.shape, torch.int64)
                    torch.searchsorted(uniq_pk, qpk, out=loc)
                    loc.clamp_max_(count - 1)
                    found = workspace.tensor(qpk.shape, torch.bool)
                    torch.eq(workspace.gather(uniq_pk, loc), qpk, out=found)
                    bd = workspace.gather(best_d, loc)
                    bsid = workspace.gather(best_sid, loc)
                    sd = workspace.gather(sec_d, loc)
                    own_here = workspace.tensor(shape, torch.bool)
                    torch.eq(bsid.view(shape), sheet_sid.unsqueeze(1), out=own_here)
                    own_here.view(-1).logical_and_(found)
                    torch.where(own_here.view(-1), sd, bd, out=other_d.view(-1))
                    found.logical_not_()
                    other_d.view(-1).masked_fill_(found, float("inf"))
            else:
                del enf

        # Preserve the original expanded-lane comparisons and f32 cede test.
        # Counts and packed words reuse integer storage only after their last
        # consumer; no floating reduction is fused or reassociated.
        with workspace.stage():
            lanes = workspace.tensor((AA_NUM_SAMPLES,), torch.int64)
            torch.arange(AA_NUM_SAMPLES, out=lanes)
            lane_values = workspace.tensor(shape, torch.int64)
            torch.bitwise_right_shift(
                low.unsqueeze(1), lanes.view(1, -1), out=lane_values
            )
            lane_values.bitwise_and_(1)
            owns = workspace.tensor(shape, torch.bool)
            torch.eq(lane_values, 1, out=owns)
            threshold = workspace.tensor(shape, torch.float32)
            torch.sub(sample_depths, depth_tie_epsilon, out=threshold)
            gate = workspace.tensor(shape, torch.bool)
            torch.lt(other_d, threshold, out=gate)
            gate.logical_and_(owns).logical_and_(subject.unsqueeze(1))
            count = workspace.tensor((nb,), torch.int64)
            lose_count = workspace.tensor((nb,), torch.float32)
            own_count = workspace.tensor((nb,), torch.float32)
            torch.sum(gate, dim=1, dtype=torch.int64, out=count)
            lose_count.copy_(count)
            torch.sum(owns, dim=1, dtype=torch.int64, out=count)
            own_count.copy_(count).mul_(sheet_sample_depth_cede)
            cede = workspace.tensor((nb,), torch.bool)
            torch.gt(lose_count, own_count, out=cede)
            gate.logical_and_(cede.unsqueeze(1))
            lane_values.copy_(gate).bitwise_left_shift_(lanes.view(1, -1))
            torch.sum(lane_values, dim=1, out=count)
            out.copy_(count).bitwise_left_shift_(AA_LOSE_SHIFT)
    return out


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
    diagnostics=True,
    resolver_memory=None,
    workspace=None,
):
    """Compact one emission's fragment stream into its sheet stream.

    Parameters mirror what ``prepare_sparse_raster_coverage`` was called with:
    ``coverage`` is its returned dict (the compact ``frag_*`` arrays and the
    per-pixel CSR), ``merged`` the batch's merged scene, ``cam_origin`` /
    ``pixel_world_scale`` the per-frame camera rows the band rule's relative
    scale reads.

    ``diagnostics`` defaults to True for inspection/parity callers. False
    omits ``sheet_nfrag``, ``sheet_fused``, ``num_groups`` and
    ``num_split_groups`` and skips their separable work. Rendering uses False;
    truncation/correctness checks are always performed.

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
    With ``resolver_memory`` supplied and ``diagnostics=False``, return a
    ``SheetBuffers`` record directly in the reverse arena instead. Its coverage
    and mask are resolver weights, not raw diagnostic areas/unions. Native
    sort permutations use forward scratch; the caller must scope that scratch
    around the call and keep it alive through the final copy. ``workspace``
    supplies nested forward-arena stages for reductions, ranks, shell prefixes,
    sibling counts and lane-depth tables. Only caller-owned results may escape
    those stages. Sorted payloads and group inverses use the surrounding
    compaction stage through the final copy; its forward storage is reclaimed
    on every exit. Decoded preprocessing metadata ends before rank grouping;
    pooling maps and sample-depth classification also use explicit stages.
    Closed-shell keys, metadata and both ceiling arms also end in preprocessing.
    Geometry blocks, coverage/weight math and reference depth competition now
    use phase-local scratch as well. Dynamic unique/nonzero temporaries,
    optional boundary conversions and library sort/scan workspace still need
    external headroom.
    """
    # Diagnostic keys are omitted when diagnostics=False; correctness and
    # truncation checks remain unconditional.
    if resolver_memory is not None and diagnostics:
        raise ValueError(
            "resolver_memory requires diagnostics=False; use the standalone record for diagnostics"
        )
    if band_rule not in BAND_RULES:
        raise ValueError(f"unknown band rule {band_rule!r}; one of {BAND_RULES}")
    device = coverage["frag_key"].device
    workspace = workspace or CompactionWorkspace(resolver_memory, device=device)
    if workspace.device != device:
        raise ValueError("compaction workspace and fragment inputs must share a device")

    with workspace.stage():
        return _compact_sheets(
            coverage,
            merged,
            cam_origin,
            pixel_world_scale,
            time_start,
            width,
            height,
            band_rule=band_rule,
            band_c=band_c,
            tri_screen=tri_screen,
            shade_split=shade_split,
            positioned_depth=positioned_depth,
            sample_depth=sample_depth,
            diagnostics=diagnostics,
            resolver_memory=resolver_memory,
            workspace=workspace,
        )


def _compact_sheets(
    coverage,
    merged,
    cam_origin,
    pixel_world_scale,
    time_start,
    width,
    height,
    *,
    band_rule,
    band_c,
    tri_screen,
    shade_split,
    positioned_depth,
    sample_depth,
    diagnostics,
    resolver_memory,
    workspace,
):
    n = int(coverage["num_fragments"])
    rank_key_base(n)  # reject impossible kernel/index capacity before allocating
    num_covered = int(coverage["num_covered"])
    frag_key = coverage["frag_key"][:n]
    frag_ref = coverage["frag_ref"][:n]
    frag_ab = coverage["frag_ab"][:n]
    frag_cov = coverage["frag_cov"][:n]
    frag_msk = coverage["frag_msk"][:n]
    frag_cap = coverage["frag_cap"][:n]
    device = frag_key.device
    owned = resolver_memory is not None

    # Sorted payloads and metadata consumers outlive decoding; reserve them
    # before its stage so lookup/table/key scratch is reusable by rank grouping.
    ppf = int(width) * int(height)
    tri_obj = merged["tri_obj"]
    positions = workspace.tensor((n,), torch.int64)
    torch.arange(n, out=positions)
    is_tri = workspace.tensor((n,), torch.bool)
    new_group = workspace.tensor((n,), torch.bool)
    band_start = workspace.tensor((n,), torch.bool)
    cls = workspace.tensor((n,), torch.int64) if shade_split else None
    sorted_out = SortedFragments.allocate(workspace, n) if owned else None
    # This permutation is budgeted separately from workspace in discovery.
    order = (
        resolver_memory.get_tensor((n,), torch.int64)
        if owned
        else torch.empty((n,), dtype=torch.int64, device=device)
    )
    with workspace.stage():
        meta = FragmentMetadata.allocate(workspace, n, triangle=is_tri)
        pix, t, frame_rel, _triangle, safe_ref, gkey = fragment_metadata(
            frag_key,
            frag_ref,
            frag_msk,
            positions,
            tri_obj,
            ppf,
            time_start,
            out=meta,
            workspace=workspace,
        )
        # One readback answers triangle presence, the chunk's frame span and
        # the sort-key bound. Reduce the small surface table rather than the
        # fragment group keys; keep the established frame-reduction width.
        if n:
            surface_max = (
                tri_obj.amax().to(torch.int64)
                if tri_obj.numel()
                else torch.zeros((), dtype=torch.int64, device=device)
            )
            probe = torch.stack(
                [
                    is_tri.any().to(torch.int64),
                    frame_rel.to(reduction_index_dtype()).amax().to(torch.int64),
                    surface_max,
                ]
            ).tolist()
            tri_present = bool(probe[0])
            # Frames this chunk's fragments span: the per-(frame, triangle) tables
            # below are built for exactly these rows.
            num_frames = int(probe[1]) + 1
            # ``gkey`` is ``sid * 2 + facing`` for a triangle and ``-(position + 2)``
            # for a bezier fragment, so this bounds both of its ends.
            gkey_bound = max(2 * int(probe[2]) + 2, n + 2)
        else:
            tri_present = False
            num_frames = 1
            gkey_bound = 1

        # ---- P1: (pixel, group, depth) order + band starts ---------------------
        order = _pixel_group_order(
            pix,
            gkey,
            t,
            coverage.get("run_offsets"),
            key_bounds=(num_frames * ppf, gkey_bound),
            workspace=workspace,
            out=order,
        )
        pix_o, t_o, cov_o, msk_o = gather_sorted_fragments(
            pix,
            t,
            frag_cov,
            frag_msk,
            order,
            out=sorted_out,
        )
        if shade_split:
            # Raw classes are needed only for this gather, not during sorting or
            # primitive-band analysis. Their frame table has a nested lifetime.
            with workspace.stage():
                raw_class = workspace.tensor((n,), torch.int64)
                _shade_class(
                    merged,
                    frame_rel,
                    time_start,
                    safe_ref,
                    is_tri,
                    tri_present,
                    num_frames,
                    out=raw_class,
                    workspace=workspace,
                )

                gather_rows(raw_class, order, out=cls)
            del raw_class
        new_group.fill_(True)
        if n > 1:
            with workspace.stage():
                g_o = workspace.gather(gkey, order)
                changed = workspace.tensor((n - 1,), torch.bool)
                torch.ne(pix_o[1:], pix_o[:-1], out=new_group[1:])
                torch.ne(g_o[1:], g_o[:-1], out=changed)
                new_group[1:].logical_or_(changed)
            del g_o, changed
        del pix, gkey

        band_start.copy_(new_group)
        if band_rule == "prim" and n > 1 and tri_present:
            with workspace.stage():
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
                    out=workspace.tensor((n - 1,), torch.bool),
                    workspace=workspace,
                )
                split_after.logical_and_(~new_group[1:])
                band_start[1:].logical_or_(split_after)
                del split_after

        # Shell coverage is private and does not participate in conflict ranks
        # (which read masks). Finish the ceiling while decoded metadata is live;
        # every shell key/lookup/prefix can then be reclaimed before rank grouping.
        tri_closed = merged.get("tri_closed")
        if rt_settings.solid_shell_alpha and tri_present and tri_closed is not None:
            with workspace.stage():
                segments = shell_segments(
                    pix_o,
                    msk_o,
                    frame_rel,
                    safe_ref,
                    is_tri,
                    order,
                    positions,
                    tri_obj,
                    tri_closed,
                    time_start,
                    out=ShellSegments.allocate(workspace, n),
                    workspace=workspace,
                )
                if segments is not None:
                    apply_shell_ceiling(
                        segments,
                        t_o,
                        cov_o,
                        order_builder=_key_depth_order,
                        use_kernel=rt_settings.sheet_shell_ceiling_kernel,
                        workspace=workspace,
                    )
            del segments
        # ``t_o`` (the sorted exact depths) stays live past this point: the
        # sheet_sample_depth block below reads it to find each sheet's nearest
        # owner per sample. Its owned storage lasts through compaction, even
        # after the final sample-depth consumer releases its Python reference.
        del frame_rel, safe_ref, t, meta, _triangle

    del sorted_out

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
    rank_ids = workspace.tensor((n,), torch.int64) if owned else None
    with workspace.stage():
        band_id = workspace.tensor((n,), torch.int64)
        group_ids_from_starts(band_start, out=band_id)
        rank = workspace.tensor((n,), torch.int32)
        _conflict_rank(
            band_start, order, frag_msk, positions, out=rank, workspace=workspace
        )
        # Preserve every rank. Grouping and rank pooling both use count-bounded
        # IDs rather than reserving four low key bits and merging excess layers.
        band_id, cid_band, rank_of_cid = _sheet_rank_groups(
            band_id, rank, workspace=workspace, out=rank_ids
        )
        del rank
    if owned:
        cid_band = workspace.copy(cid_band)
        rank_of_cid = workspace.copy(rank_of_cid)
    del rank_ids
    nb = int(cid_band.numel())
    # Band identity for sheet_sample_depth's multi-sheet-band exemption: a
    # conflict-rank split makes several sheets of ONE parent band. cid_band
    # maps each dense group back to that parent; rank_of_cid retains its rank
    # for _rank_pool_groups. Under shade_split, parent identity is recovered
    # from the class key further down.
    if nb == 0:
        return None

    # ---- P2: segmented reduction over bands --------------------------------
    pos_o = order  # original stream position of each sorted fragment

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
    with workspace.stage():
        if sheet_rank_pool and nb:
            n_group, group_of_cid = _rank_pool_groups(
                cid_band,
                rank_of_cid,
                band_id,
                cov_o,
                msk_o,
                nb,
                workspace=workspace,
                out=workspace.tensor((nb,), torch.int64) if owned else None,
            )
        del rank_of_cid
        if shade_split:
            composite = BandComposite.allocate(workspace, n_group) if owned else None
            class_ids = workspace.tensor((n,), torch.int64) if owned else None
            with workspace.stage():
                band_of_frag = (
                    band_id
                    if group_of_cid is None
                    else workspace.gather(group_of_cid, band_id)
                )
                band_area, band_union, band_corr, band_split = _band_composite(
                    band_of_frag,
                    n_group,
                    cov_o,
                    msk_o,
                    workspace=workspace,
                    out=composite,
                )
                # Classes were sorted during preprocessing. Membership flags and
                # the masked class key need not survive class grouping.
                not_split = workspace.gather(band_split, band_of_frag)
                not_split.logical_not_()
                cls_eff = workspace.copy(cls)
                cls_eff.masked_fill_(not_split, 0)
                # Key by the SUB-BAND, not the compositing group: pooling must
                # never merge rank sub-bands into the same sheet.
                nb, band_id, sheet_cid = _sheet_class_groups(
                    band_id,
                    cls_eff,
                    new_group,
                    nb,
                    out=class_ids,
                    workspace=workspace,
                )
            if owned:
                sheet_cid = workspace.copy(sheet_cid)
            sheet_band = (
                sheet_cid
                if group_of_cid is None
                else workspace.gather(group_of_cid, sheet_cid)
            )
            del sheet_cid, cls_eff, not_split, band_of_frag, band_split, cls
            del composite, class_ids
        elif group_of_cid is not None:
            # One sheet per rank sub-band; only its compositing group changes.
            composite = BandComposite.allocate(workspace, n_group) if owned else None
            with workspace.stage():
                band_of_frag = workspace.gather(group_of_cid, band_id)
                band_area, band_union, band_corr, _split = _band_composite(
                    band_of_frag,
                    n_group,
                    cov_o,
                    msk_o,
                    workspace=workspace,
                    out=composite,
                )
            del _split, composite, band_of_frag
            sheet_band = group_of_cid
        del group_of_cid

        with workspace.stage():
            # The band's aggregates in one walk of the sorted stream: exact area
            # (float64 accumulate, float32 round -- §6.6.4), the sample-mask union,
            # and the fusion detector.
            sheet_cov, union, fused, _ = _band_reduce(
                band_id,
                msk_o,
                cov_o,
                nb,
                want_sliver=False,
                want_fused=diagnostics,
                workspace=workspace,
                out=(
                    BandReduction.allocate(
                        workspace, nb, want_fused=False, want_sliver=False
                    )
                    if resolver_memory is not None
                    else None
                ),
            )
            sheet_cov.clamp_min_(0.0)

            stats = sheet_statistics(
                band_id,
                msk_o,
                positions,
                pos_o,
                pix_o,
                cov_o,
                nb,
                mask_all=AA_MASK_ALL,
                positioned=positioned_depth,
                diagnostics=diagnostics,
                workspace=workspace,
                out=(
                    SheetStatistics.allocate(workspace, nb, diagnostics=False)
                    if resolver_memory is not None
                    else None
                ),
            )
            nearest_orig = stats.nearest_fragment
            rep_orig = stats.representative_fragment
            sheet_pix = stats.pixel
            min_pos = stats.min_position
            first_sorted = stats.first_sorted
            nfrag = stats.fragment_count
            del stats, cov_o

            # Only the lane table and final-classification scratch survive to output
            # gathering. Returning the persistent records releases this whole phase.
            with workspace.stage():
                # ---- sheet_sample_depth: per-sample nearest-owner depths ---------------
                # ``d(sheet, s)``: the exact f32 depth of the sheet's nearest fragment
                # owning sample bit s. See ``_lane_first_owners``, which computes the
                # table (one masked amin scatter per lane in torch, one kernel pass under
                # ``sheet_sample_depth_kernel``).
                sample_depths = None
                if sample_depth:
                    sample_depths = workspace.tensor(
                        (nb, AA_NUM_SAMPLES), torch.float32
                    )
                    _lane_first_owners(
                        band_id,
                        msk_o,
                        t_o,
                        nb,
                        n,
                        out=sample_depths,
                        workspace=workspace,
                    )
                del t_o
                del positions, msk_o

                # Split-group accounting (diagnostic): groups are triangle-only. Kept
                # device-side end to end -- the group tables are over-allocated to ``nb``
                # (group ids are < the true group count <= nb) and the two counters stay
                # 0-d tensors, evaluated only when something reads them -- because this
                # block used to cost three device syncs per compaction for numbers
                # nothing on the render path consumes.
                if diagnostics:
                    metadata_ids = band_id if sheet_metadata_kernel else None
                    num_tri_groups, num_split_groups = _sheet_group_counts(
                        new_group, metadata_ids, order, is_tri, first_sorted, nb
                    )
                    del metadata_ids
                del band_id
                del new_group
                # Last reads of the sorted stream. Ordinary tensors release their
                # storage here; arena views remain allocated until their enclosing
                # compaction stage closes after the persistent final copy.
                del first_sorted, is_tri, order, pos_o

                # Flags: facing from the band key; one-mesh / sliver policy bits from the
                # dominant fragment (uniform per pixel / per emission policy); the sliver
                # bit FORCED on for an empty union, which is an areal positionless sheet
                # whatever its dominant fragment carried.
                sheet_msk = (
                    workspace.tensor((nb,), torch.int32)
                    if owned
                    else torch.empty((nb,), dtype=torch.int32, device=device)
                )
                with workspace.stage():
                    rep_msk = workspace.gather(frag_msk, rep_orig)
                    torch.bitwise_and(rep_msk, ~AA_MASK_ALL, out=sheet_msk)
                    sheet_msk.bitwise_or_(union)
                    empty_union = workspace.tensor((nb,), torch.bool)
                    torch.eq(union, 0, out=empty_union)
                    sliver_flag = workspace.tensor((nb,), torch.int32, 0)
                    sliver_flag.masked_fill_(empty_union, AA_SLIVER_BIT)
                    sheet_msk.bitwise_or_(sliver_flag)
                del union, rep_msk, empty_union, sliver_flag

                # ---- Final order: (pixel, classic order of nearest fragment) -----------
                # Band IDs (and their class/rank subdivisions) retain pixel order. Only
                # the sheets within a pixel need restoring to nearest-fragment order.
                final = _sheet_walk_order(
                    sheet_pix, min_pos, memory=resolver_memory, workspace=workspace
                )

                # §4.4's additive sibling compositing, expressed in the weights the walk
                # consumes (see ``_sibling_weights``). Where a band holds one sheet --
                # every band with ``shade_split`` off -- these ARE the sheet's own area
                # and mask, so the resolve reads exactly what it read before.
                sheet_cov_final = (
                    workspace.gather(sheet_cov, final)
                    if owned
                    else sheet_cov.index_select(0, final)
                )
                sheet_msk_final = (
                    workspace.gather(sheet_msk, final)
                    if owned
                    else sheet_msk.index_select(0, final)
                )
                sheet_wgt, sheet_wmsk = sheet_cov_final, sheet_msk_final
                final_band = None
                if sheet_band is not None:
                    final_band = (
                        workspace.gather(sheet_band, final)
                        if owned
                        else sheet_band.index_select(0, final)
                    )
                    sheet_wgt, sheet_wmsk = _sibling_weights(
                        final_band,
                        sheet_cov_final,
                        sheet_msk_final,
                        band_area,
                        band_union,
                        band_corr,
                        workspace=workspace,
                        out=(
                            SheetWeights.allocate(workspace, nb)
                            if resolver_memory is not None
                            else None
                        ),
                    )

                # Compose indices before gathering the packed payload, using the exact
                # MPS integer paths for both. Ordinary MPS integer gathers can round the
                # index or packed depth bits; no payload-sized key intermediate is needed.
                if resolver_memory is None:
                    sheet_key = gather_packed_key(
                        frag_key, gather_exact(nearest_orig, final)
                    )
                sheet_pix = (
                    workspace.gather(sheet_pix, final)
                    if owned
                    else sheet_pix.index_select(0, final)
                )
                # Persistent output gathers representatives itself. Only the
                # depth gate and the diagnostic record need a host-side gather.
                rep_final = None
                if not owned or sample_depth:
                    rep_final = (
                        workspace.gather(rep_orig, final)
                        if owned
                        else rep_orig.index_select(0, final)
                    )

                # ---- sheet_sample_depth: classify, floor, cede --------------------------
                # Everything here works on the FINAL-ordered per-sheet arrays; the lose
                # words land in both mask outputs so the resolve (which consumes the
                # weights) and every record reader see the same thing. Off, none of this
                # runs and the outputs above are exactly what they were.
                if sample_depth:
                    with workspace.stage():
                        # The depth table was built in sheet order; everything below works in
                        # the final (walk) order.
                        sample_depths = (
                            workspace.gather(sample_depths, final)
                            if owned
                            else sample_depths.index_select(0, final)
                        )
                        # Band identity and the multi-sheet-band exemption: a band split into
                        # siblings (shade-class split, conflict-rank split) claims against
                        # band-pooled arithmetic whose single occlusion write ignores slots,
                        # so gating a sibling would over-occlude. Its sheets are neither
                        # subjects nor enforcers.
                        if final_band is not None:
                            band_of_sheet = final_band
                        else:
                            band_of_sheet = (
                                workspace.gather(cid_band, final)
                                if owned
                                else cid_band.index_select(0, final)
                            )
                        n_bands = int(band_of_sheet.max().item()) + 1
                        only_band = workspace.tensor((nb,), torch.bool)
                        with workspace.stage():
                            members = workspace.tensor((n_bands,), torch.int64, 0)
                            members.scatter_add_(
                                0,
                                band_of_sheet,
                                workspace.tensor((nb,), torch.int64, 1),
                            )
                            torch.eq(
                                workspace.gather(members, band_of_sheet),
                                1,
                                out=only_band,
                            )
                        del members
                        rep_ref = workspace.gather(frag_ref, rep_final)
                        low, sheet_sid, enforcer, subject = sample_depth_metadata(
                            sheet_pix,
                            rep_ref,
                            sheet_msk_final,
                            sheet_cov_final,
                            sheet_wgt,
                            only_band,
                            tri_obj,
                            int(width) * int(height),
                            time_start,
                            out=SampleDepthMetadata.allocate(workspace, nb),
                            workspace=workspace,
                        )

                        if rt_settings.sheet_depth_reduce_kernel:
                            from algan.rendering.raytracing.sheet_depth_taichi import (
                                sheet_depth_lose,
                            )

                            lose_word = workspace.tensor((nb,), torch.int32)
                            sheet_depth_lose(
                                kernel_index(sheet_pix),
                                sheet_sid,
                                sample_depths,
                                low,
                                subject.contiguous().view(torch.uint8),
                                enforcer.contiguous().view(torch.uint8),
                                nb,
                                float(depth_tie_epsilon),
                                float(sheet_sample_depth_cede),
                                int(AA_LOSE_SHIFT),
                                lose_word,
                            )
                        else:
                            lose_word = _sample_depth_lose_reference(
                                sheet_pix,
                                sample_depths,
                                sheet_sid,
                                enforcer,
                                subject,
                                low,
                                out=workspace.tensor((nb,), torch.int32),
                                workspace=workspace,
                            )
                        if owned:
                            # Both masks already have caller storage. They can
                            # alias when no sibling weighting was needed; OR is
                            # idempotent, so either case needs no extra output.
                            sheet_msk_final.bitwise_or_(lose_word)
                            sheet_wmsk.bitwise_or_(lose_word)
                        else:
                            sheet_msk_final = sheet_msk_final | lose_word
                            sheet_wmsk = sheet_wmsk | lose_word

                if resolver_memory is not None:
                    from algan.rendering.raytracing.sheet_buffers import (
                        finish_sheet_buffers,
                    )

                    return finish_sheet_buffers(
                        resolver_memory,
                        coverage["covered_idx"][:num_covered],
                        final,
                        nearest_orig,
                        rep_orig,
                        frag_key,
                        frag_ref,
                        frag_ab,
                        frag_cap,
                        sheet_wgt,
                        sheet_wmsk,
                        sheet_pix,
                    )

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
                    "num_sheets": nb,
                    "band_rule": band_rule,
                    "band_c": float(band_c),
                }

                if diagnostics:
                    out.update(
                        sheet_nfrag=nfrag.index_select(0, final),
                        sheet_fused=fused.index_select(0, final),
                        num_groups=num_tri_groups,
                        num_split_groups=num_split_groups,
                    )

                # CSR aligned with covered_idx: every covered pixel holds at least one
                # fragment, hence at least one sheet, so the two pixel sets coincide.
                out["sheet_offsets"] = _sheet_offsets(
                    coverage["covered_idx"][:num_covered], sheet_pix
                )
                return out
