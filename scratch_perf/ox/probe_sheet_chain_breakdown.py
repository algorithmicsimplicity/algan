"""Per-block cost of ``sheets.compact_sheets`` on the REAL nn-scene 4K stream.

``_sheet_compact_breakdown.py`` replays representative passes standalone; this
probe runs the compaction's actual statement groups in order, on the arrays one
real frame produces, so the "which pass do we kernelise next" decision is made
on measured numbers rather than modelled ones. Inputs are captured by
monkeypatching ``compact_sheets`` during one real 4K frame of
``benchmarks/performance/nn_scene_UHD.py``'s scene; each block below is then
the source lines verbatim (torch arm, kernels left ON where they are shipped
ON), timed with CUDA syncs.

    <venv-python> scratch_perf/ox/probe_sheet_chain_breakdown.py
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("ALGAN_USE_DAEMON", "0")

import torch  # noqa: E402

from algan import *  # noqa: E402,F403
from algan.rendering.raytracing import sheets  # noqa: E402
from algan.rendering.raytracing.raster_taichi import (  # noqa: E402
    _AA_BACKFACE_BIT,
    _AA_MASK_ALL,
    _AA_NUM_SAMPLES,
    _AA_SLIVER_BIT,
)

CAP = {}
_orig = sheets.compact_sheets


def _capture(*a, **k):
    if not CAP:
        (
            coverage,
            merged,
            cam_origin,
            pixel_world_scale,
            time_start,
            width,
            height,
        ) = a[:7]
        CAP.update(
            coverage={
                kk: (vv.clone() if torch.is_tensor(vv) else vv)
                for kk, vv in coverage.items()
            },
            merged={
                kk: merged[kk].clone()
                for kk in ("tri_obj", "tri_norm", "tri_pos", "tri_closed")
                if kk in merged and merged[kk] is not None
            },
            cam_origin=cam_origin.clone(),
            pixel_world_scale=pixel_world_scale.clone(),
            time_start=time_start,
            width=width,
            height=height,
            kwargs=dict(k),
        )
    return _orig(*a, **k)


sheets.compact_sheets = _capture

sys.path.insert(0, os.path.dirname(__file__))
import probe_capture_nn  # noqa: E402

sheets.compact_sheets = _orig
CAP = probe_capture_nn.load()
torch.cuda.synchronize()

cov_in = CAP["coverage"]
merged = CAP["merged"]
cam_origin = CAP["cam_origin"]
pixel_world_scale = CAP["pixel_world_scale"]
time_start = int(CAP["time_start"])
width, height = int(CAP["width"]), int(CAP["height"])
kw = CAP["kwargs"]

results = []


def bench(label, fn):
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    results.append((label, (time.perf_counter() - t0) / 3))


# ---- verbatim P1 -----------------------------------------------------------
n = int(cov_in["num_fragments"])
frag_key = cov_in["frag_key"][:n]
frag_ref = cov_in["frag_ref"][:n]
frag_cov = cov_in["frag_cov"][:n]
frag_msk = cov_in["frag_msk"][:n]
device = frag_key.device
pix = frag_key >> 32
t = (frag_key & 0xFFFFFFFF).to(torch.int32).view(torch.float32)
ppf = width * height
frame_rel = pix // ppf
tri_obj = merged["tri_obj"]
is_tri = frag_ref >= 0
safe_ref = frag_ref.clamp_min(0).to(torch.int64)
sid = tri_obj[(frame_rel + time_start) % tri_obj.shape[0], safe_ref].to(torch.int64)
facing = ((frag_msk & _AA_BACKFACE_BIT) != 0).to(torch.int64)
positions = torch.arange(n, dtype=torch.int64, device=device)
gkey = torch.where(is_tri, sid * 2 + facing, -(positions + 2))
del sid, facing

order = sheets._lexsort(pix, gkey, t)
pix_o = pix.index_select(0, order)
g_o = gkey.index_select(0, order)
t_o = t.index_select(0, order)
del pix, gkey

new_group = torch.ones(n, dtype=torch.bool, device=device)
if n > 1:
    new_group[1:] = (pix_o[1:] != pix_o[:-1]) | (g_o[1:] != g_o[:-1])
band_c = kw.get("band_c", 2.0)
tri_screen = kw.get("tri_screen")
split_after = (
    sheets._prim_split_after(
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
    )
    if tri_screen is not None
    else None
)
shade_split = bool(kw.get("shade_split", False))

print(
    f"\ncaptured: n={n:,} covered={int(cov_in['num_covered']):,} "
    f"shade_split={shade_split} positioned_depth={kw.get('positioned_depth')} "
    f"sample_depth={kw.get('sample_depth')} band_rule={kw.get('band_rule')}"
)


def band_construct():
    ng = torch.ones(n, dtype=torch.bool, device=device)
    if n > 1:
        ng[1:] = (pix_o[1:] != pix_o[:-1]) | (g_o[1:] != g_o[:-1])
    bs = ng.clone()
    if split_after is not None:
        bs[1:] |= (~ng[1:]) & split_after
    bid = torch.cumsum(bs.to(torch.int64), 0) - 1
    return ng, bs, bid


bench("B: new_group/band_start/band_id construction", band_construct)
new_group, band_start, band_id_pre = band_construct()

# conflict rank (kernelised already; context only)
rank = sheets._conflict_rank(band_start, order, frag_msk, positions)
rank.clamp_(max=sheets.SHEET_RANK_LIMIT)
cid = band_id_pre * 16 + rank
del rank
uniq_cid, band_id = torch.unique(cid, sorted=True, return_inverse=True)
del cid
nb = int(uniq_cid.numel())
cid_band = uniq_cid // 16
del uniq_cid

cov_o = frag_cov.index_select(0, order)
msk_o = frag_msk.index_select(0, order)
pos_o = order

# solid-shell ceiling block (verbatim, only when it triggers)
tri_closed_arr = merged.get("tri_closed")
shell_triggers = False
if tri_closed_arr is not None and bool(is_tri.any()):
    closed_flag = (
        tri_closed_arr[
            (frame_rel + time_start) % tri_closed_arr.shape[0], safe_ref
        ].reshape(-1)
        > 0.5
    ) & is_tri
    shell_triggers = bool(closed_flag.any())
print(f"solid-shell block triggers: {shell_triggers}")


def solid_shell():
    closed_flag = (
        tri_closed_arr[
            (frame_rel + time_start) % tri_closed_arr.shape[0], safe_ref
        ].reshape(-1)
        > 0.5
    ) & is_tri
    closed_s = closed_flag.index_select(0, order)
    shell_sid = (
        tri_obj[(frame_rel + time_start) % tri_obj.shape[0], safe_ref]
        .to(torch.int64)
        .index_select(0, order)
    )
    shell_back = ((frag_msk & _AA_BACKFACE_BIT) != 0).index_select(0, order)
    K = int(shell_sid.amax().item()) + 2
    key = torch.where(closed_s, pix_o * K + shell_sid, -(positions + 1))
    del shell_sid
    t_all = (frag_key & 0xFFFFFFFF).to(torch.int32).view(torch.float32)
    t_all = t_all.index_select(0, order)
    o2 = sheets._lexsort(key, t_all)
    del t_all
    k2 = key.index_select(0, o2)
    del key
    c2 = cov_o.to(torch.float64).index_select(0, o2)
    seg_start = torch.ones(n, dtype=torch.bool, device=device)
    if n > 1:
        seg_start[1:] = k2[1:] != k2[:-1]
    del k2
    seg = torch.cumsum(seg_start.to(torch.int64), 0) - 1
    nseg = int(seg[-1].item()) + 1
    csum = torch.cumsum(c2, 0)
    excl_global = csum.sub_(c2)
    first = torch.zeros(nseg, dtype=torch.int64, device=device)
    first.scatter_(0, seg[seg_start], torch.nonzero(seg_start).reshape(-1))
    spent = excl_global - excl_global.index_select(0, first).index_select(0, seg)
    del excl_global, first, seg_start
    backf2 = shell_back.index_select(0, o2)
    del shell_back
    z64 = torch.zeros((), dtype=torch.float64, device=device)
    front = torch.zeros(nseg, dtype=torch.float64, device=device)
    back = torch.zeros(nseg, dtype=torch.float64, device=device)
    front.scatter_add_(0, seg, torch.where(backf2, z64, c2))
    back.scatter_add_(0, seg, torch.where(backf2, c2, z64))
    del backf2, z64
    cap = torch.maximum(front, back).to(torch.float32).to(torch.float64)
    del front, back
    scale = (
        cap.index_select(0, seg)
        .sub_(spent)
        .clamp_min_(0.0)
        .div_(c2.clamp_min_(1e-12))
        .clamp_max_(1.0)
    )
    out_cov = cov_o.clone()
    out_cov.index_copy_(0, o2, (c2 * scale).to(torch.float32))
    return out_cov


if shell_triggers:
    bench("S: solid-shell ceiling block", solid_shell)
    cov_o = solid_shell()

if shade_split:
    bench(
        "C: shade-split band composite (_band_composite)",
        lambda: sheets._band_composite(band_id, nb, cov_o, msk_o),
    )
    band_area, band_union, band_corr, band_split = sheets._band_composite(
        band_id, nb, cov_o, msk_o
    )
    cls = sheets._shade_class(merged, frame_rel, time_start, safe_ref, is_tri)
    cls_o = cls.index_select(0, order)
    cls_eff = torch.where(
        band_split.index_select(0, band_id), cls_o, torch.zeros_like(cls_o)
    )
    skey = band_id * sheets._SHADE_CLASS_BASE + cls_eff
    uniq_skey, band_id = torch.unique(skey, sorted=True, return_inverse=True)
    nb = int(uniq_skey.numel())
    sheet_band = uniq_skey // sheets._SHADE_CLASS_BASE
    del uniq_skey, skey, cls_eff, cls_o, cls

bench(
    "R: main _band_reduce (kernel, shipped)",
    lambda: sheets._band_reduce(band_id, msk_o, cov_o, nb, want_sliver=False),
)
sheet_cov, union, fused, _ = sheets._band_reduce(
    band_id, msk_o, cov_o, nb, want_sliver=False
)


def band_stats():
    first_sorted = torch.full((nb,), n, dtype=torch.int64, device=device)
    first_sorted.scatter_reduce_(
        0, band_id, positions, reduce="amin", include_self=True
    )
    nearest_orig = pos_o.index_select(0, first_sorted)
    sheet_pix_l = pix_o.index_select(0, first_sorted)
    min_pos = torch.full((nb,), n, dtype=torch.int64, device=device)
    min_pos.scatter_reduce_(0, band_id, pos_o, reduce="amin", include_self=True)
    big = torch.full((), n, dtype=torch.int64, device=device)
    posn = (msk_o & _AA_MASK_ALL) != 0
    masked = torch.where(posn, positions, big)
    first_sorted_p = torch.full((nb,), n, dtype=torch.int64, device=device)
    first_sorted_p.scatter_reduce_(0, band_id, masked, reduce="amin", include_self=True)
    del masked
    has_pos = first_sorted_p < n
    nearest_orig = torch.where(
        has_pos,
        pos_o.index_select(0, first_sorted_p.clamp_max(max(n - 1, 0))),
        nearest_orig,
    )
    del first_sorted_p
    masked = torch.where(posn, pos_o, big)
    del posn, big
    min_pos_p = torch.full((nb,), n, dtype=torch.int64, device=device)
    min_pos_p.scatter_reduce_(0, band_id, masked, reduce="amin", include_self=True)
    del masked
    min_pos = torch.where(has_pos, min_pos_p, min_pos)
    return first_sorted, nearest_orig, sheet_pix_l, min_pos


bench(f"A: per-band order stats ({nb:,} bands, positioned_depth)", band_stats)
first_sorted, nearest_orig, sheet_pix, min_pos = band_stats()


def dominant():
    cmax = torch.zeros(nb, dtype=torch.float32, device=device)
    cmax.scatter_reduce_(0, band_id, cov_o, reduce="amax", include_self=True)
    is_max = cov_o >= cmax.index_select(0, band_id)
    del cmax
    big = torch.full((n,), n, dtype=torch.int64, device=device)
    cand_pos = torch.where(is_max, pos_o, big)
    del is_max, big
    rep_orig = torch.full((nb,), n, dtype=torch.int64, device=device)
    rep_orig.scatter_reduce_(0, band_id, cand_pos, reduce="amin", include_self=True)
    del cand_pos
    nfrag = torch.zeros(nb, dtype=torch.int64, device=device)
    nfrag.scatter_add_(0, band_id, torch.ones_like(band_id))
    return rep_orig, nfrag


bench("D: dominant fragment + nfrag", dominant)
rep_orig, nfrag = dominant()

sample_depth = bool(kw.get("sample_depth", False))
if sample_depth:

    def sample_depth_block():
        big = torch.full((), n, dtype=torch.int64, device=device)
        inf = torch.full((), float("inf"), dtype=torch.float32, device=device)
        sample_depths = torch.full(
            (nb, _AA_NUM_SAMPLES), float("inf"), dtype=torch.float32, device=device
        )
        for lane in range(_AA_NUM_SAMPLES):
            owns = ((msk_o >> lane) & 1) != 0
            masked = torch.where(owns, positions, big)
            del owns
            first_lane = torch.full((nb,), n, dtype=torch.int64, device=device)
            first_lane.scatter_reduce_(
                0, band_id, masked, reduce="amin", include_self=True
            )
            del masked
            has = first_lane < n
            d_lane = t_o.index_select(0, first_lane.clamp_max(max(n - 1, 0)))
            sample_depths[:, lane] = torch.where(has, d_lane, inf)
            del first_lane, has, d_lane
        return sample_depths

    bench("G: sample-depth lane loop (8 lanes)", sample_depth_block)


# group diagnostics
def group_diag():
    group_id = torch.cumsum(new_group.to(torch.int64), 0) - 1
    ngroups = int(group_id[-1]) + 1 if n else 0
    bands_per_group = torch.zeros(ngroups, dtype=torch.int64, device=device)
    sheet_group = group_id.index_select(0, first_sorted)
    bands_per_group.scatter_add_(
        0, sheet_group, torch.ones(nb, dtype=torch.int64, device=device)
    )
    tri_group = is_tri.index_select(0, order).index_select(0, first_sorted)
    tri_groups_mask = torch.zeros(ngroups, dtype=torch.bool, device=device)
    tri_groups_mask.scatter_(0, sheet_group, tri_group)
    num_split_groups = int(((bands_per_group > 1) & tri_groups_mask).sum().item())
    num_tri_groups = int(tri_groups_mask.sum().item())
    return num_split_groups, num_tri_groups


bench("H: group diagnostics", group_diag)

rep_msk = frag_msk.index_select(0, rep_orig)
flags = rep_msk & (~_AA_MASK_ALL)
empty_union = union == 0
flags = flags | torch.where(
    empty_union, torch.full_like(flags, _AA_SLIVER_BIT), torch.zeros_like(flags)
)
sheet_msk = union.to(torch.int32) | flags
del empty_union, flags


def final_order():
    final = torch.argsort(min_pos, stable=True)
    wgt, wmsk = None, None
    sc = sheet_cov.index_select(0, final)
    sm = sheet_msk.index_select(0, final)
    if shade_split:
        wgt, wmsk = sheets._sibling_weights(
            sheet_band.index_select(0, final), sc, sm, band_area, band_union, band_corr
        )
    key = frag_key.index_select(0, nearest_orig).index_select(0, final)
    sp = sheet_pix.index_select(0, final)
    counts = torch.zeros(int(cov_in["num_covered"]), dtype=torch.int64, device=device)
    segx = torch.searchsorted(cov_in["covered_idx"].to(torch.int64), sp)
    counts.scatter_add_(0, segx, torch.ones_like(segx))
    offsets = torch.zeros(
        int(cov_in["num_covered"]) + 1, dtype=torch.int64, device=device
    )
    offsets[1:] = torch.cumsum(counts, 0)
    return key, offsets


bench("E: final argsort + sibling weights + CSR", final_order)

total = sum(d for _, d in results)
print(
    f"\n=== compact_sheets block costs on the real stream (n={n:,}, nb={nb:,}), mean of 3 ==="
)
for label, dt in sorted(results, key=lambda r: -r[1]):
    print(f"  {dt * 1000:7.1f} ms  {dt / total * 100:5.1f}%  {label}")
print(
    f"  {total * 1000:7.1f} ms   100%  (blocks benchmarked; sorts/unique/shade-class/prim-split/rank/reduce timed separately)"
)
