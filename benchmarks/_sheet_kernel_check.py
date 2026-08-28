"""A/B parity for the compaction/emission kernels
(raster_fused_gather, sheet_mask_kernel, sheet_rank_kernel,
raster_opaque_trunc_kernel, sheet_one_mesh_kernel, sheet_sample_depth_kernel,
sheet_shell_ceiling_kernel, sheet_band_stats_kernel,
raster_pair_expand_kernel).

The kernels replace multi-pass torch loops with one pass, and all are meant to
be BIT-IDENTICAL to the arm they replace: the gather copies bits, the mask
reductions are integer, the conflict-rank scan is an integer serial walk in
the stream's own order, the truncation keep-mask is an integer flag compare,
the lane-owner scan is an integer amin per slot, and both float reductions
(the exact-area sum, the one-mesh coverage sums) keep an f64 accumulator that
is only ever read through an f32 round -- that last one by measurement rather
than by construction, since an f64 atomic add reassociates and only the f32
cast makes it agree. This checks two ways, because the two catch different
mistakes:

* **unit** -- the kernels against the exact torch expressions they replaced,
  on random inputs at a 4K frame's shapes, including the ones the render never
  produces (empty bands, every-sample-shared bands, sliver flags, 4096
  addends in a single area sum, a conflict-rank stream whose FIRST band flag
  is clear -- compact_sheets never emits one, but the helper must agree with
  its torch arm on any input), plus a repeat-run check on each float sum;
* **end to end** -- four rendered frames of a scene carrying PN surfaces, flat
  polyhedra, transparency, bezier circuits and text, hashed with all nine
  toggles ON and all nine OFF.

    <venv-python> benchmarks/_sheet_kernel_check.py
"""

import hashlib
import os

os.environ.setdefault("ALGAN_USE_DAEMON", "0")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from algan import *  # noqa: E402,F403
from algan.constants.math import GIGABYTES  # noqa: E402
from algan.rendering.raytracing import sheets  # noqa: E402
from algan.rendering.raytracing.raster_pipeline import (  # noqa: E402
    _gather_fragment_arrays,
    _one_mesh_pixel_caps,
    _opaque_prefix_keep,
    _tri_obj_row,
)
from algan.rendering.raytracing.raster_taichi import (  # noqa: E402
    _AA_BACKFACE_BIT,
    _AA_MASK_ALL,
    _AA_NUM_SAMPLES,
    _AA_ONE_MESH_BIT,
    _AA_SLIVER_BIT,
)
from algan.settings import SETTINGS  # noqa: E402

EXPERIMENTAL = SETTINGS.raytracing.experimental
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
failures = []


def check(name, ok):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}")
    if not ok:
        failures.append(name)


def bits_equal(a, b):
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.dtype.is_floating_point:
        # Bitwise, so a NaN payload or a signed zero cannot pass as equal.
        width = torch.int32 if a.dtype == torch.float32 else torch.int64
        return bool((a.view(width) == b.view(width)).all())
    return bool((a == b).all())


# --------------------------------------------------------------- unit: gather
print("\ngather_fragment_arrays vs six index_selects")
g = torch.Generator(device=DEV).manual_seed(7)
for n, m in ((3_661_824, 3_290_404), (1, 1), (5000, 0), (17, 17)):
    key = torch.randint(-(1 << 45), 1 << 45, (n,), generator=g, device=DEV)
    ref = torch.randint(-9, 1 << 20, (n,), generator=g, device=DEV, dtype=torch.int32)
    ab = torch.randn(n, 2, generator=g, device=DEV)
    cov = torch.randn(n, generator=g, device=DEV)
    msk = torch.randint(0, 1 << 22, (n,), generator=g, device=DEV, dtype=torch.int32)
    opq = torch.randint(0, 2, (n,), generator=g, device=DEV) > 0
    idx = torch.randint(0, n, (m,), generator=g, device=DEV)
    args = (idx, key, ref, ab, cov, msk, opq)
    EXPERIMENTAL.set(raster_fused_gather=False)
    want = _gather_fragment_arrays(*args)
    EXPERIMENTAL.set(raster_fused_gather=True)
    got = _gather_fragment_arrays(*args)
    check(
        f"n={n} m={m}",
        all(bits_equal(a, b) for a, b in zip(want, got)) and len(want) == len(got),
    )

# ----------------------------------------------------------- unit: band reduce
print("\n_band_reduce / _popcount_lanes vs the torch passes they replace")


def mask_case(label, band, msk, cov, nb, want_sliver):
    EXPERIMENTAL.set(sheet_mask_kernel=False)
    w_area, w_union, w_fused, w_sliver = sheets._band_reduce(
        band, msk, cov, nb, want_sliver=want_sliver
    )
    w_pop = sheets._popcount_lanes(w_union)
    EXPERIMENTAL.set(sheet_mask_kernel=True)
    g_area, g_union, g_fused, g_sliver = sheets._band_reduce(
        band, msk, cov, nb, want_sliver=want_sliver
    )
    g_pop = sheets._popcount_lanes(g_union)
    ok = (
        bits_equal(w_area, g_area)
        and bits_equal(w_union, g_union)
        and bits_equal(w_fused, g_fused)
        and bits_equal(w_pop, g_pop)
        and ((w_sliver is None and g_sliver is None) or bits_equal(w_sliver, g_sliver))
    )
    check(f"{label} (fused bands: {int(w_fused.sum())})", ok)
    # The area sum is the one float reduction, and an f64 atomic add is not
    # order-independent in principle -- only far enough below an f32 ulp that
    # the cast absorbs it. That is the claim; this is the check.
    repeats = [
        sheets._band_reduce(band, msk, cov, nb, want_sliver=want_sliver)[0]
        for _ in range(4)
    ]
    check(
        f"{label}: area reproducible over 4 kernel runs",
        all(bits_equal(r, g_area) for r in repeats),
    )


n, nb = 3_661_824, 3_290_404
band = torch.randint(0, nb, (n,), generator=g, device=DEV)
msk = torch.randint(0, 1 << 22, (n,), generator=g, device=DEV, dtype=torch.int32)
# Exact areas look like the real thing: mostly whole pixels, a partial tail.
area = torch.rand(n, generator=g, device=DEV)
area = torch.where(area > 0.4, torch.ones_like(area), area)
mask_case("4K shapes, random masks, sliver", band, msk, area, nb, True)
mask_case("4K shapes, random masks, no sliver", band, msk, area, nb, False)

# Every fragment in one band, so every lane is claimed many times over: the
# fusion detector's saturated case, which random wide bands never reach, and
# the DEEPEST possible area sum -- 4096 addends into one f64 accumulator.
small = torch.zeros(4096, dtype=torch.int64, device=DEV)
dense = torch.full((4096,), _AA_MASK_ALL, dtype=torch.int32, device=DEV)
deep = torch.rand(4096, generator=g, device=DEV)
mask_case("one band, every lane shared, 4096 addends", small, dense, deep, 1, True)

# Masks that PARTITION their band -- the healthy case, which must NOT fuse.
part = torch.arange(8, dtype=torch.int64, device=DEV) % 4
lanes = (1 << (torch.arange(8, device=DEV) % _AA_NUM_SAMPLES)).to(torch.int32)
mask_case(
    "partitioning masks (must not fuse)",
    part,
    lanes,
    torch.rand(8, generator=g, device=DEV),
    4,
    True,
)

# Empty sample words plus sliver flags: the areal/donor sheets.
donor_band = torch.zeros(64, dtype=torch.int64, device=DEV)
donor_msk = torch.full((64,), _AA_SLIVER_BIT, dtype=torch.int32, device=DEV)
mask_case(
    "donors only (empty union, sliver set)",
    donor_band,
    donor_msk,
    torch.rand(64, generator=g, device=DEV),
    1,
    True,
)

# ------------------------------------------------------- unit: conflict rank
print("\n_conflict_rank vs the eight-cumsum torch scan it replaced")


def rank_case(label, band_start, order, msk, expect=None):
    """Both arms of ``sheets._conflict_rank`` must agree EXACTLY; ``expect``
    optionally pins the values themselves (the unclamped int32 ranks).
    """
    positions = torch.arange(int(msk.numel()), dtype=torch.int64, device=DEV)
    EXPERIMENTAL.set(sheet_rank_kernel=False)
    want = sheets._conflict_rank(band_start, order, msk, positions)
    EXPERIMENTAL.set(sheet_rank_kernel=True)
    got = sheets._conflict_rank(band_start, order, msk, positions)
    ok = bits_equal(want, got) and want.dtype == torch.int32
    if expect is not None:
        ok = ok and bits_equal(want, expect.to(torch.int32))
    check(f"{label} (n={msk.numel()}, max rank {int(want.max())})", ok)
    return want


rank_n = 3_661_824
rank_msk = torch.randint(
    0, 1 << (_AA_NUM_SAMPLES + 3), (rank_n,), generator=g, device=DEV, dtype=torch.int32
)
rank_order = torch.randperm(rank_n, generator=g, device=DEV)


def band_starts(p, first):
    bs = torch.rand(rank_n, generator=g, device=DEV) < p
    bs[0] = first
    return bs


rank_case("4K shapes, sparse bands", band_starts(0.01, True), rank_order, rank_msk)
rank_case("4K shapes, dense bands", band_starts(0.1, True), rank_order, rank_msk)

# One band holding the whole stream, and its saturated twin: every fragment
# claiming EVERY lane walks the unclamped ranks 0,1,2,... straight past the
# caller's clamp_(max=15).
rn = 100_000
one_band = torch.zeros(rn, dtype=torch.bool, device=DEV)
one_band[0] = True
ids = torch.arange(rn, dtype=torch.int64, device=DEV)
rand_msk = torch.randint(
    0, 1 << _AA_NUM_SAMPLES, (rn,), generator=g, device=DEV, dtype=torch.int32
)
rank_case("one band, whole stream", one_band, ids, rand_msk)
full_msk = torch.full((rn,), _AA_MASK_ALL, dtype=torch.int32, device=DEV)
want = rank_case(
    "saturated band, unclamped ranks 0..n-1",
    one_band,
    ids,
    full_msk,
    expect=torch.arange(rn, dtype=torch.int32, device=DEV),
)
check(
    "saturated band clamps to exactly 15 at the caller",
    int(want.clamp_(max=15)[-1]) == 15,
)

# Every fragment its own band: no earlier in-band fragments anywhere.
own_band = torch.ones(rn, dtype=torch.bool, device=DEV)
rank_case(
    "every fragment its own band",
    own_band,
    ids,
    rand_msk,
    expect=torch.zeros(rn, dtype=torch.int32, device=DEV),
)

# Donors: empty sample words claim nothing, so their ranks are all 0.
donor_starts = torch.rand(rn, generator=g, device=DEV) < 0.01
donor_starts[0] = True
rank_case(
    "all-zero mask words (donors)",
    donor_starts,
    ids,
    torch.zeros(rn, dtype=torch.int32, device=DEV),
    expect=torch.zeros(rn, dtype=torch.int32, device=DEV),
)

# A shuffled emission->sorted permutation: exercises the kernel's in-kernel
# gather msk[order[j]] against the torch arm's index_select.
shuffled = torch.randperm(rn, generator=g, device=DEV)
scattered = torch.rand(rn, generator=g, device=DEV) < 0.03
scattered[0] = True
rank_case("shuffled permutation", scattered, shuffled, rand_msk)

rank_case(
    "n == 1",
    torch.ones(1, dtype=torch.bool, device=DEV),
    torch.zeros(1, dtype=torch.int64, device=DEV),
    torch.randint(
        0, 1 << _AA_NUM_SAMPLES, (1,), generator=g, device=DEV, dtype=torch.int32
    ),
)

# THE LEADING-RUN REGRESSION: a stream whose FIRST flag is clear.
# compact_sheets never produces one (its new_group[0] is always set), but the
# helper must agree with the torch arm on ANY input: the cummax gives a
# leading run of clear flags band-first 0 -- one band starting at row 0 --
# and the kernel must walk that band instead of leaving those rows unwritten.
leading = torch.rand(rn, generator=g, device=DEV) < 0.03
leading[:8] = False
leading[0] = False
want = rank_case("band_start[0] clear (leading run)", leading, ids, rand_msk)
EXPERIMENTAL.set(sheet_rank_kernel=True)
again = sheets._conflict_rank(leading, ids, rand_msk, ids)
check(
    "leading run: row 0 rank 0, kernel arm reproducible",
    int(want[0]) == 0 and bits_equal(again, want),
)

# --------------------------------------------- unit: opaque-prefix truncation
print("\n_opaque_prefix_keep vs the torch first_opaque/keep chain it replaced")


def torch_keep(opaque, counts):
    """The exact statements the kernel replaces (verbatim from the pre-kernel
    truncation block).
    """
    device = opaque.device
    num_frags = int(opaque.numel())
    num_cov = int(counts.numel())
    positions = torch.arange(num_frags, dtype=torch.int64, device=device)
    segments = torch.repeat_interleave(
        torch.arange(num_cov, dtype=torch.int64, device=device), counts
    )
    starts = torch.cumsum(counts, 0) - counts
    ends = starts + counts - 1
    first_opaque = torch.full((num_cov,), num_frags, dtype=torch.int64, device=device)
    opaque_pos = opaque.nonzero(as_tuple=True)[0]
    first_opaque.scatter_reduce_(
        0,
        segments.index_select(0, opaque_pos),
        opaque_pos,
        reduce="amin",
        include_self=True,
    )
    del opaque_pos, starts
    keep_end = torch.minimum(first_opaque, ends)
    del first_opaque, ends
    keep = positions <= keep_end.index_select(0, segments)
    return keep


def trunc_case(label, opaque, counts):
    EXPERIMENTAL.set(raster_opaque_trunc_kernel=False)
    want = _opaque_prefix_keep(opaque, counts, int(opaque.numel()))
    EXPERIMENTAL.set(raster_opaque_trunc_kernel=True)
    got = _opaque_prefix_keep(opaque, counts, int(opaque.numel()))
    ref = torch_keep(opaque, counts)
    ok = (
        bits_equal(want.view(torch.uint8), ref.view(torch.uint8))
        and bits_equal(got.view(torch.uint8), ref.view(torch.uint8))
        and want.dtype == got.dtype == torch.bool
    )
    check(
        f"{label} (kept {int(ref.sum())} of {ref.numel()})",
        ok,
    )


trunc_n = 3_661_824
trunc_counts = torch.full((755_877,), 4, dtype=torch.int64, device=DEV)
trunc_counts[-1] += trunc_n - int(trunc_counts.sum())
trunc_case(
    "4K shapes, ~20% opaque",
    torch.rand(trunc_n, generator=g, device=DEV) < 0.2,
    trunc_counts,
)
trunc_case(
    "no opaque at all", torch.zeros(trunc_n, dtype=torch.bool, device=DEV), trunc_counts
)
trunc_case(
    "every fragment opaque",
    torch.ones(trunc_n, dtype=torch.bool, device=DEV),
    trunc_counts,
)
first_only = torch.zeros(trunc_n, dtype=torch.bool, device=DEV)
starts = torch.cumsum(trunc_counts, 0) - trunc_counts
first_only[starts] = True  # the first fragment of every pixel
trunc_case("first fragment of each pixel opaque", first_only, trunc_counts)
last_only = torch.zeros(trunc_n, dtype=torch.bool, device=DEV)
last_only[starts + trunc_counts - 1] = True  # the LAST fragment of each pixel
trunc_case("last fragment of each pixel opaque", last_only, trunc_counts)
one_pix_counts = torch.tensor([trunc_n], dtype=torch.int64, device=DEV)
trunc_case(
    "single pixel holding the whole stream",
    torch.rand(trunc_n, generator=g, device=DEV) < 0.5,
    one_pix_counts,
)
tiny_counts = torch.tensor([3, 1], dtype=torch.int64, device=DEV)
tiny_opaque = torch.tensor([False, False, True, False], device=DEV)
trunc_case("n == 4, two pixels", tiny_opaque, tiny_counts)

# ----------------------------------------------------- unit: one-mesh records
print("\n_one_mesh_pixel_caps vs the torch one-mesh block it replaced")


def torch_one_mesh(key, ref, cov, msk, mat_opaque, counts, tri_obj, ppf, time_start):
    """Verbatim pre-kernel block."""
    device = key.device
    num_covered = int(counts.numel())
    frame_of = _tri_obj_row(key >> 32, ppf, time_start, tri_obj.shape[0])
    safe_ref = ref.clamp_min(0).to(torch.int64)
    sid = tri_obj[frame_of, safe_ref].to(torch.int64)
    usable = (ref >= 0) & mat_opaque
    sid = torch.where(usable, sid, torch.full_like(sid, -1))
    seg = torch.repeat_interleave(
        torch.arange(num_covered, dtype=torch.int64, device=device), counts
    )
    lo = torch.full((num_covered,), 1 << 40, dtype=torch.int64, device=device)
    hi = torch.full((num_covered,), -1, dtype=torch.int64, device=device)
    lo.scatter_reduce_(0, seg, sid, reduce="amin", include_self=True)
    hi.scatter_reduce_(0, seg, sid, reduce="amax", include_self=True)
    one_mesh = (lo == hi) & (lo >= 0)
    is_back = (msk & _AA_BACKFACE_BIT) != 0
    cov_acc = cov.to(torch.float64)
    front = torch.zeros(num_covered, dtype=torch.float64, device=device)
    back = torch.zeros_like(front)
    zero = torch.zeros((), dtype=torch.float64, device=device)
    front.scatter_add_(0, seg, torch.where(is_back, zero, cov_acc))
    back.scatter_add_(0, seg, torch.where(is_back, cov_acc, zero))
    cap_pix = torch.maximum(front, back).clamp_max_(1.0).to(cov.dtype)
    msk_out = msk | torch.where(
        one_mesh.index_select(0, seg),
        torch.full_like(msk, _AA_ONE_MESH_BIT),
        torch.zeros_like(msk),
    )
    cap_s = torch.where(
        one_mesh.index_select(0, seg), cap_pix.index_select(0, seg), 2.0
    )
    return msk_out, cap_s


def mesh_case(
    label, key, ref, cov, msk, mat_opaque, counts, tri_obj, ppf=100, time_start=0
):
    args = (key, ref, cov, msk, mat_opaque, counts, tri_obj, ppf, time_start)
    EXPERIMENTAL.set(sheet_one_mesh_kernel=False)
    w_msk, w_cap = _one_mesh_pixel_caps(*args)
    EXPERIMENTAL.set(sheet_one_mesh_kernel=True)
    g_msk, g_cap = _one_mesh_pixel_caps(*args)
    ok = bits_equal(w_msk, g_msk) and bits_equal(w_cap, g_cap)
    flagged = int(((g_msk & _AA_ONE_MESH_BIT) != 0).sum())
    # The f64 sums are the one float contract: the kernel must also reproduce
    # ITSELF bitwise across runs (the torch atomics cannot promise that).
    repeats = [_one_mesh_pixel_caps(*args)[1] for _ in range(4)]
    ok = ok and all(bits_equal(r, g_cap) for r in repeats)
    check(f"{label} (flagged fragments: {flagged})", ok)


mesh_n = 3_128_845
mesh_cov_n = 700_000
mesh_counts = torch.full((mesh_cov_n,), 4, dtype=torch.int64, device=DEV)
mesh_counts[-1] += mesh_n - int(mesh_counts.sum())
mesh_ref = torch.randint(-1, 12, (mesh_n,), generator=g, device=DEV, dtype=torch.int32)
mesh_mat = torch.rand(mesh_n, generator=g, device=DEV) < 0.8
mesh_msk = torch.randint(
    0, 1 << 22, (mesh_n,), generator=g, device=DEV, dtype=torch.int32
)
mesh_cov = torch.rand(mesh_n, generator=g, device=DEV)
mesh_obj = torch.randint(0, 3, (1, 12), generator=g, device=DEV, dtype=torch.int64)
# Keys must be pixel-clustered for counts to be a valid CSR over them:
pix = torch.repeat_interleave(
    torch.arange(mesh_cov_n, dtype=torch.int64, device=DEV), mesh_counts
)
tb = torch.randint(0, 1 << 31, (mesh_n,), generator=g, device=DEV, dtype=torch.int64)
mesh_key = (pix << 32) | tb
del pix, tb
mesh_ref = torch.randint(-1, 12, (mesh_n,), generator=g, device=DEV, dtype=torch.int32)
mesh_mat = torch.rand(mesh_n, generator=g, device=DEV) < 0.8
mesh_msk = torch.randint(
    0, 1 << 22, (mesh_n,), generator=g, device=DEV, dtype=torch.int32
)
mesh_cov = torch.rand(mesh_n, generator=g, device=DEV)
mesh_obj = torch.randint(0, 3, (1, 12), generator=g, device=DEV, dtype=torch.int64)
mesh_case(
    "4K shapes, mixed surfaces/facings/opacity",
    mesh_key,
    mesh_ref,
    mesh_cov,
    mesh_msk,
    mesh_mat,
    mesh_counts,
    mesh_obj,
)

# One surface everywhere: every usable pixel flags and takes a real ceiling.
same_obj = torch.zeros(1, 12, dtype=torch.int64, device=DEV)
mesh_case(
    "one surface everywhere",
    mesh_key,
    mesh_ref.clamp_min(0),
    mesh_cov,
    mesh_msk,
    torch.ones_like(mesh_mat),
    mesh_counts,
    same_obj,
)

# Circuits only: nothing usable, nothing flagged, caps at the sentinel.
mesh_case(
    "circuits only (no flags, sentinel caps)",
    mesh_key,
    torch.full((mesh_n,), -7, dtype=torch.int32, device=DEV),
    mesh_cov,
    mesh_msk,
    mesh_mat,
    mesh_counts,
    mesh_obj,
)

# Single-fragment pixels: the walk degenerates to one row.
solo_counts = torch.ones(mesh_n, dtype=torch.int64, device=DEV)
solo_key = torch.arange(mesh_n, dtype=torch.int64, device=DEV) << 8
mesh_case(
    "every pixel a single fragment",
    solo_key,
    mesh_ref.clamp_min(0),
    mesh_cov,
    mesh_msk,
    torch.ones_like(mesh_mat),
    solo_counts,
    same_obj,
)

# All-backface single-surface pixels: front stays 0, cap = back's own area.
back_msk = mesh_msk.clone()
back_msk[mesh_ref >= 0] |= _AA_BACKFACE_BIT
mesh_case(
    "all fragments back-facing",
    mesh_key,
    mesh_ref.clamp_min(0),
    mesh_cov,
    back_msk,
    torch.ones_like(mesh_mat),
    mesh_counts,
    same_obj,
)

# ---------------------------------------------------- unit: lane-owner table
print("\n_lane_first_owners vs the eight-lane amin loop it replaced")


def lane_case(label, band, msk, t_o, nb, n):
    EXPERIMENTAL.set(sheet_sample_depth_kernel=False)
    want = sheets._lane_first_owners(band, msk, t_o, nb, n)
    EXPERIMENTAL.set(sheet_sample_depth_kernel=True)
    got = sheets._lane_first_owners(band, msk, t_o, nb, n)
    ok = bits_equal(want, got) and want.shape == (nb, _AA_NUM_SAMPLES)
    inf_want = int(torch.isinf(want).sum())
    check(f"{label} (inf entries: {inf_want})", ok)


lane_n = 3_128_845
lane_nb = 1_441_601
lane_band = torch.randint(0, lane_nb, (lane_n,), generator=g, device=DEV)
lane_msk = torch.randint(
    0, 1 << (_AA_NUM_SAMPLES + 4), (lane_n,), generator=g, device=DEV, dtype=torch.int32
)
lane_t = torch.rand(lane_n, generator=g, device=DEV) * 10 + 1.0
lane_case("4K shapes, random bands/masks", lane_band, lane_msk, lane_t, lane_nb, lane_n)

# EMPTY BANDS: ids allocated past the highest used one stay at the sentinel --
# every unowned (band, lane) slot must come back +inf in both arms.
wide_band = torch.randint(0, lane_nb // 2, (lane_n,), generator=g, device=DEV)
lane_case("half the band table unused", wide_band, lane_msk, lane_t, lane_nb, lane_n)

# Single-fragment bands: each owner is its band's only claimant.
solo_band = torch.arange(lane_n, dtype=torch.int64, device=DEV)
lane_case("every fragment its own sheet", solo_band, lane_msk, lane_t, lane_n, lane_n)

# One sheet owning everything with every lane claimed: the saturated table.
dense_band = torch.zeros(lane_n, dtype=torch.int64, device=DEV)
full_msk_l = torch.full((lane_n,), _AA_MASK_ALL, dtype=torch.int32, device=DEV)
lane_case(
    "one sheet, every lane claimed by everything",
    dense_band,
    full_msk_l,
    lane_t,
    1,
    lane_n,
)

# Donors only: no lane owned anywhere -- every entry must be +inf.
donor_msk = torch.full((lane_n,), _AA_SLIVER_BIT, dtype=torch.int32, device=DEV)
lane_case("donors only (no lane owned)", dense_band, donor_msk, lane_t, 1, lane_n)

# --------------------------------------------- unit: solid-shell ceiling
print("\nsolid_shell_ceiling vs the torch post-sort segment clamp it replaced")


def shell_case(label, key, o2, back, cov_in):
    """Both arms consume ONE cub-scan exclusive prefix, built exactly as
    ``compact_sheets`` builds it -- the prefix stays in torch under the
    toggle, so sharing it is not a shortcut, it is the production shape.
    (Two separately-built prefixes can legitimately differ: cuB's
    reassociation follows the workspace the allocator hands it, so comparing
    arms that each rebuilt the prefix would test allocator noise.)
    """
    from algan.rendering.raytracing.sheet_compact_taichi import (
        solid_shell_ceiling,
    )

    device = key.device
    nn_ = int(key.numel())
    cov64 = cov_in.to(torch.float64)
    c2 = cov64.index_select(0, o2)
    del cov64
    excl = torch.cumsum(c2, 0).sub_(c2)

    def torch_apply(cov_src):
        k2 = key.index_select(0, o2)
        seg_start = torch.ones(nn_, dtype=torch.bool, device=device)
        if nn_ > 1:
            seg_start[1:] = k2[1:] != k2[:-1]
        del k2
        seg = torch.cumsum(seg_start.to(torch.int64), 0) - 1
        nseg = int(seg[-1].item()) + 1
        first = torch.zeros(nseg, dtype=torch.int64, device=device)
        first.scatter_(0, seg[seg_start], torch.nonzero(seg_start).reshape(-1))
        spent = excl - excl.index_select(0, first).index_select(0, seg)
        del first, seg_start
        backf2 = back.index_select(0, o2)
        z64 = torch.zeros((), dtype=torch.float64, device=device)
        front = torch.zeros(nseg, dtype=torch.float64, device=device)
        back_t = torch.zeros(nseg, dtype=torch.float64, device=device)
        front.scatter_add_(0, seg, torch.where(backf2, z64, c2))
        back_t.scatter_add_(0, seg, torch.where(backf2, c2, z64))
        del backf2, z64
        cap = torch.maximum(front, back_t).to(torch.float32).to(torch.float64)
        del front, back_t
        scale = (
            cap.index_select(0, seg)
            .sub_(spent)
            .clamp_min_(0.0)
            .div_(c2.clamp_min_(1e-12))
            .clamp_max_(1.0)
        )
        del spent, cap, seg
        out_cov = cov_src.clone()
        out_cov.index_copy_(0, o2, (c2 * scale).to(torch.float32))
        return out_cov

    want = torch_apply(cov_in)
    got = cov_in.clone()
    cov64k = cov_in.to(torch.float64)  # the kernel's barrier scratch
    solid_shell_ceiling(
        key.contiguous(),
        o2.contiguous(),
        back.contiguous().view(torch.uint8),
        excl,
        cov64k,
        nn_,
        got,
    )
    ok = bits_equal(want, got)
    if not ok:
        dd = want.view(torch.int32) != got.view(torch.int32)
        print(
            f"    [diag] diffs {int(dd.sum())}/{nn_}; "
            f"first {[int(i) for i in dd.nonzero(as_tuple=True)[0][:5]]}"
        )
    # The facing-split sums are the float contract: the kernel must also
    # reproduce ITSELF bitwise across runs (serial walks, so this should be
    # exact); the shared ``excl`` makes the repeats read the same spend.
    repeats = []
    for _ in range(4):
        c = cov_in.clone()
        solid_shell_ceiling(
            key.contiguous(),
            o2.contiguous(),
            back.contiguous().view(torch.uint8),
            excl,
            cov64k,
            nn_,
            c,
        )
        repeats.append(c)
    ok = ok and all(bits_equal(r, got) for r in repeats)
    nseg = int(torch.unique(key[o2]).numel())
    del cov64k
    check(f"{label} ({nseg:,} segments)", ok)


shell_n = 3_128_845
# Keys look like the real thing: mostly closed-surface (pixel, surface)
# segments of a few fragments each over 700k pixels, plus a pass-through tail.
shell_pix = torch.randint(0, 700_000, (shell_n,), generator=g, device=DEV)
shell_sid = torch.randint(0, 4, (shell_n,), generator=g, device=DEV)
closed = torch.rand(shell_n, generator=g, device=DEV) < 0.75
neg = -(torch.arange(shell_n, dtype=torch.int64, device=DEV) + 1)
shell_key = torch.where(closed, shell_pix * 6 + shell_sid, neg)
del shell_pix, shell_sid, closed, neg
# A real o2 sorts by (key, depth): the same LSD composition _lexsort uses --
# stable argsort of the LEAST significant key first, then a stable argsort by
# key over that order. Sorting depth first and key second leaves equal keys
# contiguous runs along o2, which is what makes segments multi-fragment.
depth = torch.rand(shell_n, generator=g, device=DEV)
order1 = torch.argsort(depth, stable=True)
o2_sorted = order1[torch.argsort(shell_key.index_select(0, order1), stable=True)]
del order1, depth
shell_back = torch.rand(shell_n, generator=g, device=DEV) < 0.5
shell_cov = torch.rand(shell_n, generator=g, device=DEV)
shell_case(
    "4K shapes, mixed segments/facings", shell_key, o2_sorted, shell_back, shell_cov
)

# No closed fragment at all: every key unique-negative, every segment its own
# pass-through (cap == own area -> scale exactly 1).
all_open_key = -(torch.arange(shell_n, dtype=torch.int64, device=DEV) + 1)
open_o2 = torch.arange(shell_n, dtype=torch.int64, device=DEV)
shell_case(
    "nothing closed (pass-through segments)",
    all_open_key,
    open_o2,
    shell_back,
    shell_cov,
)

# One segment holding the whole stream: every fragment shares one key.
one_seg_key = torch.zeros(shell_n, dtype=torch.int64, device=DEV)
shell_case("one segment, whole stream", one_seg_key, open_o2, shell_back, shell_cov)

# Zero areas and sub-floor areas: exercises the 1e-12 denominator branch and
# the clamp_max(1) on a scale that would otherwise exceed 1.
tiny = torch.full((shell_n,), 1e-13, dtype=torch.float32, device=DEV)
tiny[::997] = 0.0
shell_case("zero and 1e-13 areas", one_seg_key, open_o2, shell_back, tiny)

# Single fragment: the minimal walk (n == 0 is unreachable -- the block only
# runs when some triangle is declared closed).
solo_key = torch.zeros(1, dtype=torch.int64, device=DEV)
solo_o2 = torch.zeros(1, dtype=torch.int64, device=DEV)
solo_back = torch.ones(1, dtype=torch.bool, device=DEV)
solo_cov = torch.full((1,), 0.42, dtype=torch.float32, device=DEV)
shell_case("n == 1", solo_key, solo_o2, solo_back, solo_cov)

# -------------------------------------------------- unit: band stats fusion
print("\nband_stats_reduce/rep_orig vs the five scatters they replaced")


def band_stats_case(label, band, msk, pos_o, cov, nb, positioned):
    from algan.rendering.raytracing.sheet_compact_taichi import (
        band_stats_reduce,
        band_stats_rep_orig,
    )

    n = int(band.numel())
    dev = band.device

    def torch_arm():
        fs = torch.full((nb,), n, dtype=torch.int64, device=dev)
        fs.scatter_reduce_(
            0,
            band,
            torch.arange(n, dtype=torch.int64, device=dev),
            reduce="amin",
            include_self=True,
        )
        mp = torch.full((nb,), n, dtype=torch.int64, device=dev)
        mp.scatter_reduce_(0, band, pos_o, reduce="amin", include_self=True)
        fsp = torch.full((nb,), n, dtype=torch.int64, device=dev)
        mpp = torch.full((nb,), n, dtype=torch.int64, device=dev)
        if positioned:
            big = torch.full((), n, dtype=torch.int64, device=dev)
            posn = (msk & _AA_MASK_ALL) != 0
            masked = torch.where(
                posn, torch.arange(n, dtype=torch.int64, device=dev), big
            )
            fsp.scatter_reduce_(0, band, masked, reduce="amin", include_self=True)
            masked = torch.where(posn, pos_o, big)
            mpp.scatter_reduce_(0, band, masked, reduce="amin", include_self=True)
        cm = torch.zeros(nb, dtype=torch.float32, device=dev)
        cm.scatter_reduce_(0, band, cov, reduce="amax", include_self=True)
        nf = torch.zeros(nb, dtype=torch.int64, device=dev)
        nf.scatter_add_(0, band, torch.ones_like(band))
        is_max = cov >= cm.index_select(0, band)
        cand = torch.where(
            is_max, pos_o, torch.full((n,), n, dtype=torch.int64, device=dev)
        )
        rp = torch.full((nb,), n, dtype=torch.int64, device=dev)
        rp.scatter_reduce_(0, band, cand, reduce="amin", include_self=True)
        return fs, mp, fsp, mpp, cm, nf, rp

    w_fs, w_mp, w_fsp, w_mpp, w_cm, w_nf, w_rp = torch_arm()

    fs = torch.full((nb,), n, dtype=torch.int64, device=dev)
    mp = torch.full((nb,), n, dtype=torch.int64, device=dev)
    fsp = torch.full((nb,), n, dtype=torch.int64, device=dev)
    mpp = torch.full((nb,), n, dtype=torch.int64, device=dev)
    cm = torch.zeros(nb, dtype=torch.float32, device=dev)
    nf = torch.zeros(nb, dtype=torch.int64, device=dev)
    band_stats_reduce(
        band.contiguous(),
        msk.contiguous(),
        pos_o.contiguous(),
        cov.contiguous(),
        n,
        int(_AA_MASK_ALL),
        fs,
        mp,
        fsp,
        mpp,
        cm,
        nf,
        bool(positioned),
    )
    rp = torch.full((nb,), n, dtype=torch.int64, device=dev)
    band_stats_rep_orig(
        band.contiguous(), pos_o.contiguous(), cov.contiguous(), cm, n, rp
    )
    ok = (
        bits_equal(w_fs, fs)
        and bits_equal(w_mp, mp)
        and bits_equal(w_fsp, fsp)
        and bits_equal(w_mpp, mpp)
        and bits_equal(w_cm, cm)
        and bits_equal(w_nf, nf)
        and bits_equal(w_rp, rp)
    )
    check(
        f"{label} positioned={positioned} (bands {int((w_nf > 0).sum()):,}/{nb:,})",
        ok,
    )


bs_n = 3_128_845
bs_nb = 1_441_601
bs_band = torch.randint(0, bs_nb, (bs_n,), generator=g, device=DEV)
bs_pos = torch.randperm(bs_n, generator=g, device=DEV)
bs_msk = torch.randint(
    0, 1 << (_AA_NUM_SAMPLES + 4), (bs_n,), generator=g, device=DEV, dtype=torch.int32
)
bs_cov = torch.rand(bs_n, generator=g, device=DEV)
band_stats_case("4K shapes, random bands", bs_band, bs_msk, bs_pos, bs_cov, bs_nb, True)
band_stats_case(
    "4K shapes, random bands", bs_band, bs_msk, bs_pos, bs_cov, bs_nb, False
)

# Empty band table region: ids allocated past the highest used one stay at the
# sentinel n in both arms.
band_stats_case(
    "half the band table unused",
    torch.randint(0, bs_nb // 2, (bs_n,), generator=g, device=DEV),
    bs_msk,
    bs_pos,
    bs_cov,
    bs_nb,
    True,
)

# One band holding everything; every fragment its own band; single fragment.
ones_band = torch.zeros(bs_n, dtype=torch.int64, device=DEV)
band_stats_case("one band, whole stream", ones_band, bs_msk, bs_pos, bs_cov, 1, True)
own_band = torch.arange(bs_n, dtype=torch.int64, device=DEV)
band_stats_case(
    "every fragment its own band", own_band, bs_msk, bs_pos, bs_cov, bs_n, True
)
band_stats_case(
    "n == 1",
    torch.zeros(1, dtype=torch.int64, device=DEV),
    torch.full((1,), _AA_MASK_ALL, dtype=torch.int32, device=DEV),
    torch.zeros(1, dtype=torch.int64, device=DEV),
    torch.full((1,), 0.5, dtype=torch.float32, device=DEV),
    1,
    True,
)

# ------------------------------------------------ unit: pair-row expansion
print("\npair_expand_count/write vs _class_pairs_flat's torch expression")


def torch_pair_rows(mask, x0, x1, y0, y1, f_abs):
    """The verbatim torch body of ``_class_pairs_flat``."""
    ncirc = mask.shape[1]
    idx = mask.reshape(-1).nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return None
    bx0 = x0.reshape(-1)[idx]
    by0 = y0.reshape(-1)[idx]
    bw = x1.reshape(-1)[idx] - bx0 + 1
    bh = y1.reshape(-1)[idx] - by0 + 1
    area = bw * bh
    nch = (area + (RASTER_CHUNK - 1)) // RASTER_CHUNK
    rep = torch.repeat_interleave(torch.arange(idx.numel(), device=DEV), nch)
    if rep.numel() == 0:
        return None
    base = torch.cumsum(nch, 0) - nch
    off = (torch.arange(rep.shape[0], device=DEV) - base[rep]) * RASTER_CHUNK
    rows = torch.stack(
        [
            (idx % ncirc)[rep],
            f_abs.index_select(0, idx // ncirc)[rep],
            bx0[rep],
            by0[rep],
            bw[rep],
            bh[rep],
            off,
            torch.zeros_like(rep),
        ],
        -1,
    )
    return rows.to(torch.int32).contiguous()


from algan.rendering.raytracing.raster_pipeline import (  # noqa: E402
    RASTER_CHUNK,
    _class_pairs_flat,
)


def pairs_case(label, mask, x0, x1, y0, y1, f_abs):
    import algan.rendering.raytracing.sheet_compact_taichi as sct

    ref = torch_pair_rows(mask, x0, x1, y0, y1, f_abs)
    EXPERIMENTAL.set(raster_pair_expand_kernel=False)
    want = _class_pairs_flat(mask, x0, x1, y0, y1, f_abs, DEV)
    # The comparison is only evidence if the kernel arm actually took the
    # kernels -- a gate that silently routes back to torch (a device check,
    # a shape guard) would make `got` == `want` vacuously. Count launches.
    launches = {"count": 0, "write": 0}
    real_count, real_write = sct.pair_expand_count, sct.pair_expand_write

    def counted_count(*a, **k):
        launches["count"] += 1
        return real_count(*a, **k)

    def counted_write(*a, **k):
        launches["write"] += 1
        return real_write(*a, **k)

    sct.pair_expand_count, sct.pair_expand_write = counted_count, counted_write
    try:
        EXPERIMENTAL.set(raster_pair_expand_kernel=True)
        got = _class_pairs_flat(mask, x0, x1, y0, y1, f_abs, DEV)
    finally:
        sct.pair_expand_count, sct.pair_expand_write = real_count, real_write
        EXPERIMENTAL.set(raster_pair_expand_kernel=False)
    ok = (
        launches["count"] >= 1
        and (launches["write"] >= 1) == (ref is not None)
        and (
            (ref is None and want is None and got is None)
            or (
                ref is not None
                and want is not None
                and got is not None
                and bits_equal(ref, want)
                and bits_equal(ref, got)
                and got.is_contiguous()
                and got.dtype == torch.int32
            )
        )
    )
    nrows = 0 if ref is None else int(ref.shape[0])
    check(
        f"{label} (rows out: {nrows:,}; kernel launches {launches['count']}/"
        f"{launches['write']})",
        ok,
    )


C = 49_307
Ft = 1
pmask = torch.rand(Ft, C, generator=g, device=DEV) < 0.45
# Realistic bbox sizes: the real nn-scene window's 21,877 candidates expanded
# to 6.34 M rows (mean ~290 chunks each), so keep the synthetic areas in that
# range rather than whole-screen boxes, whose expansion cannot fit in 4 GB.
px0 = torch.randint(0, 3840, (Ft, C), generator=g, device=DEV, dtype=torch.int64)
px1 = px0 + torch.randint(1, 200, (Ft, C), generator=g, device=DEV, dtype=torch.int64)
py0 = torch.randint(0, 2160, (Ft, C), generator=g, device=DEV, dtype=torch.int64)
py1 = py0 + torch.randint(1, 150, (Ft, C), generator=g, device=DEV, dtype=torch.int64)
pf = torch.arange(Ft, dtype=torch.int64, device=DEV) + 7
pairs_case(f"UHD-ish window [{Ft}x{C}], ~45% candidates", pmask, px0, px1, py0, py1, pf)

pairs_case(
    "empty window (no candidates)",
    torch.zeros(2, 500, dtype=torch.bool, device=DEV),
    torch.zeros(2, 500, dtype=torch.int64, device=DEV),
    torch.zeros(2, 500, dtype=torch.int64, device=DEV),
    torch.zeros(2, 500, dtype=torch.int64, device=DEV),
    torch.zeros(2, 500, dtype=torch.int64, device=DEV),
    torch.arange(2, dtype=torch.int64, device=DEV),
)

single_mask = torch.zeros(1, 1, dtype=torch.bool, device=DEV)
single_mask[0, 0] = True
pairs_case(
    "one candidate, one chunk (area <= RASTER_CHUNK)",
    single_mask,
    torch.tensor([[10]], dtype=torch.int64, device=DEV),
    torch.tensor([[37]], dtype=torch.int64, device=DEV),
    torch.tensor([[5]], dtype=torch.int64, device=DEV),
    torch.tensor([[30]], dtype=torch.int64, device=DEV),
    torch.tensor([3], dtype=torch.int64, device=DEV),
)

pairs_case(
    "every element a candidate",
    torch.ones(3, 2000, dtype=torch.bool, device=DEV),
    torch.zeros(3, 2000, dtype=torch.int64, device=DEV),
    torch.full((3, 2000), 31, dtype=torch.int64, device=DEV),
    torch.zeros(3, 2000, dtype=torch.int64, device=DEV),
    torch.full((3, 2000), 63, dtype=torch.int64, device=DEV),
    torch.arange(3, dtype=torch.int64, device=DEV) + 11,
)

tiny_mask = torch.zeros(1, 8, dtype=torch.bool, device=DEV)
tiny_mask[0, ::3] = True
pairs_case(
    "sparse interleaved candidates",
    tiny_mask,
    torch.zeros(1, 8, dtype=torch.int64, device=DEV),
    torch.full((1, 8), 31, dtype=torch.int64, device=DEV),
    torch.zeros(1, 8, dtype=torch.int64, device=DEV),
    torch.full((1, 8), 63, dtype=torch.int64, device=DEV),
    torch.tensor([0], dtype=torch.int64, device=DEV),
)

# ------------------------------------------------------------- end to end
print("\nrendered frames, all nine toggles ON vs all nine OFF")
SETTINGS.computing.set(available_memory_override=2 * GIGABYTES)
sphere = Sphere().scale(1.4).move(LEFT * 3).set_color(GREEN).spawn()
cube = Cube().scale(1.1).move(RIGHT * 3).set_color(BLUE).spawn()
glass = Sphere().scale(0.9).move(UP * 1.2).spawn()
glass.opacity = 0.45
circle = Circle().scale(0.8).move(DOWN * 1.8 + LEFT * 1.5).set_color(RED).spawn()
label = Text("sheets").scale(0.6).move(DOWN * 2.4 + RIGHT * 1.5).spawn()
with Sync():
    sphere.rotate(70, UP)
    cube.rotate(55, OUT + RIGHT)
    glass.move(RIGHT * 1.3)
    circle.rotate(40, OUT)
    label.move(UP * 0.3)


def render_hashes(arm):
    out = []
    for i, at in enumerate((0.0, 0.35, 0.7, 1.0)):
        path = Scene.save_frame(f"_sheet_kernel_{arm}_{i}.png", MD, at=at).output_path
        arr = np.asarray(Image.open(path).convert("RGB"))
        out.append(hashlib.sha256(arr.tobytes()).hexdigest())
    return out


_ALL_OFF = {
    "raster_fused_gather": False,
    "sheet_mask_kernel": False,
    "sheet_rank_kernel": False,
    "raster_opaque_trunc_kernel": False,
    "sheet_one_mesh_kernel": False,
    "sheet_sample_depth_kernel": False,
    "sheet_shell_ceiling_kernel": False,
    "sheet_band_stats_kernel": False,
    "raster_pair_expand_kernel": False,
}
_ALL_ON = dict.fromkeys(_ALL_OFF, True)
EXPERIMENTAL.set(**_ALL_OFF)
torch_arm = render_hashes("torch")
EXPERIMENTAL.set(**_ALL_ON)
kernel_arm = render_hashes("kernel")
for i, (a, b) in enumerate(zip(torch_arm, kernel_arm)):
    check(f"frame {i}  {a[:16]}", a == b)

print("\nFAILURES:", failures if failures else "none -- bit-identical")
raise SystemExit(1 if failures else 0)
