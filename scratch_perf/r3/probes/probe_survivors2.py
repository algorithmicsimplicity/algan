"""Survivor probe v2 for the sheet-route bounce-loop tail cohort.

Same scene as benchmarks/performance/nn_scene_PREVIEW.py, rendered at PREVIEW
on CPU. Instruments, entirely from outside production code:

* ``tracer._alloc_wavefront_state``   -> current tile's wavefront state tensors
* ``tracer._ArenaRayCompactor.select``-> per-tile compaction size sequences
* ``tracer.wavefront_traverse_events``-> compact hit-event batch per iteration
* ``tracer.wavefront_shade``          -> pre/post-shade per-ray state, the
  accumulator delta committed by the iteration, and -- for rays the iteration
  retired -- the last-hit primitive's packed material row.

All captures are tensor clones taken synchronously between kernel launches on
the CPU backend, so they are consistent snapshots.
"""
import json
import os
import sys

os.environ.setdefault("ALGAN_USE_DAEMON", "0")
os.environ.setdefault("MPLBACKEND", "Agg")

import torch

if not torch.cuda.is_available():
    torch.cuda.synchronize = lambda *a, **k: None

from algan import *  # noqa: E402,F403
from algan.mobs.neural_nets.neural_net import NeuralNetMLPV3  # noqa: E402
import algan.rendering.raytracing.tracer as TR  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MIN_WEIGHT = 1e-3  # raytrace_kernels_taichi.py:132

REC = {"selects": [], "alloc_marks": [], "shades": [], "traverses": []}

# ---- patch 1: tile state allocation ---------------------------------------
_alloc_orig = TR._alloc_wavefront_state
_ALLOC_STATE = {"current": None}


def alloc_spy(memory, tn, sca_width, **kw):
    st = _alloc_orig(memory, tn, sca_width, **kw)
    _ALLOC_STATE["current"] = st
    REC["alloc_marks"].append(len(REC["selects"]))
    return st


TR._alloc_wavefront_state = alloc_spy


# ---- patch 2: compaction ---------------------------------------------------
_sel_orig = TR._ArenaRayCompactor.select


def sel_spy(self, rs_int, desired_status, **kw):
    out = _sel_orig(self, rs_int, desired_status, **kw)
    try:
        n = int(out.numel())
        entry = {"seq": len(REC["selects"]), "n": n}
        if desired_status == 0 and n > 0:
            st = _ALLOC_STATE["current"]
            if st is not None:
                idx = out.detach().to(torch.int64)
                entry["status_hist"] = (
                    rs_int[idx, 2].detach().cpu().clone()
                )  # sanity: all zero
                if n <= 64:
                    entry["slots"] = idx.cpu().clone()
                    entry["bl"] = rs_int[idx, 0].cpu().clone()
                    entry["proc"] = rs_int[idx, 1].cpu().clone()
                    entry["w"] = torch.stack(
                        [rs_int[idx, 4].float(),
                         st[3][idx, 0], st[3][idx, 5]], dim=1).cpu().clone()
        REC["selects"].append(entry)
    except Exception as exc:
        sys.stderr.write(f"[probe] select spy error: {exc!r}\n")
    return out


TR._ArenaRayCompactor.select = sel_spy


# ---- patch 3: traverse events ----------------------------------------------
_trav_orig = TR.wavefront_traverse_events


def trav_spy(*args):
    na = args[1]
    if isinstance(na, int) and na > 0:
        try:
            REC["traverses"].append(
                {
                    "seq_hint": len(REC["selects"]),
                    "active": args[0].detach().to(torch.int64).cpu().clone(),
                    "hit_f": args[41].detach().cpu().clone(),  # [na,KBUF,4]
                    "hit_i": args[42].detach().cpu().clone(),  # [na,KBUF,2]
                }
            )
        except Exception as exc:
            sys.stderr.write(f"[probe] traverse spy error: {exc!r}\n")
    return _trav_orig(*args)


TR.wavefront_traverse_events = trav_spy


# ---- patch 4: shade ---------------------------------------------------------
_shade_orig = TR.wavefront_shade
# Positional layout verified against tracer.py:2646 and the signature at
# wavefront_kernels_taichi.py:2363.
I_LAYER_OFFSETS = 27
I_TRI_MAT_ID, I_TRI_MAT = 46, 47
I_TIME_START, I_WIDTH, I_HEIGHT = 51, 52, 53
I_RS_RO, I_RS_RD, I_RS_SCA, I_RS_INT = 55, 56, 58, 59
I_HIT_I, I_RS_PIX, I_PIX_ACCUM = 61, 62, 63


def shade_spy(*args):
    na = args[1]
    rec = None
    if isinstance(na, int) and na > 0:
        try:
            idx = args[0].detach().to(torch.int64)
            rs_int, rs_sca = args[I_RS_INT], args[I_RS_SCA]
            lo = args[I_LAYER_OFFSETS]
            accum = rs_int[idx, 4].detach().to(torch.int64).cpu()
            pa = args[I_PIX_ACCUM]
            acc_c = accum.clamp(max=pa.shape[0] - 1)
            rec = {
                "seq_hint": len(REC["selects"]),
                "na": na,
                "time_start": int(args[I_TIME_START]),
                "width": int(args[I_WIDTH]),
                "height": int(args[I_HEIGHT]),
                "far_clip": float(lo[5]),
                "gloss_base": float(lo[7]) if lo.numel() > 7 else -1.0,
                "slot": idx.cpu().clone(),
                "accum_row": accum,
                "pre_bl": rs_int[idx, 0].cpu().clone(),
                "pre_proc": rs_int[idx, 1].cpu().clone(),
                "pre_status": rs_int[idx, 2].cpu().clone(),
                "num_hits": rs_int[idx, 3].cpu().clone(),
                "pre_w": torch.stack(
                    [rs_sca[idx, 0], rs_sca[idx, 5], rs_sca[idx, 6]], 1
                ).cpu().clone(),
                "ro": args[I_RS_RO][idx].cpu().clone(),
                "rd": args[I_RS_RD][idx].cpu().clone(),
                "rs_pix": args[I_RS_PIX][idx].cpu().clone(),
                "pixacc_delta": None,
                "post_w": None,
                "post_status": None,
                "post_bl": None,
            }
            pa_pre = pa[acc_c, :].cpu().clone()
            rec["_pa_pre"] = pa_pre
            rec["_pa_args"] = None
        except Exception as exc:
            sys.stderr.write(f"[probe] shade spy(pre) error: {exc!r}\n")
            rec = None
    _shade_orig(*args)
    if rec is not None:
        try:
            idx = rec["slot"].to(torch.int64)
            rs_int, rs_sca = args[I_RS_INT], args[I_RS_SCA]
            accum = rec["accum_row"].clamp(max=args[I_PIX_ACCUM].shape[0] - 1)
            rec["post_status"] = rs_int[idx, 2].cpu().clone()
            rec["post_bl"] = rs_int[idx, 0].cpu().clone()
            rec["post_proc"] = rs_int[idx, 1].cpu().clone()
            rec["post_w"] = torch.stack(
                [rs_sca[idx, 0], rs_sca[idx, 5], rs_sca[idx, 6]], 1
            ).cpu().clone()
            rec["pixacc_delta"] = (
                args[I_PIX_ACCUM][accum, :].cpu() - rec.pop("_pa_pre")
            )
        except Exception as exc:
            sys.stderr.write(f"[probe] shade spy(post) error: {exc!r}\n")
            rec.pop("_pa_pre", None)
        if rec is not None:
            REC["shades"].append(rec)
    return None


TR.wavefront_shade = shade_spy

# ---------------------------------------------------------------------------
# Scene: verbatim body of benchmarks/performance/nn_scene_PREVIEW.py
# ---------------------------------------------------------------------------
SETTINGS.raytracing.set(shadows=True)

with Off():
    nn = NeuralNetMLPV3([5, 5, 5, 5]).move(LEFT).spawn()
    x = ImageMob("/home/user/algan/benchmarks/performance/world_map.png").move_next_to(nn, LEFT).spawn()
    label = (
        Text("Neural Net MLP v3 processing an image of the globe")
        .move_next_to(nn, DOWN)
        .spawn()
    )

with Sync(run_time=5):
    nn.move(UP)
    x.color_texture = x.color_texture * 0.5
    label.move(RIGHT * 2)

result = Scene.save_video("probe_survivors2", PREVIEW, overwrite=True)

# ---------------------------------------------------------------------------
torch.save(
    {
        k: (
            [{kk: vv for kk, vv in e.items() if kk != "snap"} for e in REC[k]]
            if k == "selects"
            else REC[k]
        )
        for k in ("selects", "shades", "traverses", "alloc_marks")
    },
    os.path.join(HERE, "survivors_v2.pt"),
)

print("\n" + "=" * 78)
print("[probe] render status:", result.status)

sel = REC["selects"]
marks = set(REC["alloc_marks"])
sequences = []
cur = []
for e in sel:
    if e["seq"] in marks and cur:
        sequences.append(cur)
        cur = []
    cur.append(e)
if cur:
    sequences.append(cur)

print(f"tiles={len(sequences)} selects={len(sel)}")
tail_rays = {}
for si, s in enumerate(sequences):
    per = [x["n"] for x in s]
    # count rays present in the plateau (last non-empty sizes before final 0)
    tail = []
    for n in reversed(per[:-1]):
        if n == 0:
            break
        tail.append(n)
    print(f"  tile{si}: {per}")

# Per-ray reconstruction across ALL bounces for slots seen at bounce >= 5.
by_slot = {}
for r in REC["shades"]:
    for j, s in enumerate(r["slot"].tolist()):
        by_slot.setdefault(s, []).append(r)

print("\n--- survivors: full per-bounce trace (weight, bounces_left) ---")
survivor_slots = set()
for si, s in enumerate(sequences):
    per = [x["n"] for x in s]
    if len(per) >= 5 and per[-2] > 0:  # something entered the last bounce
        for e in s:
            if "slots" in e and e["n"] <= 64:
                survivor_slots.update(e["slots"].tolist())

mat_rows = {}
trav_by_hint = {}
for t in REC["traverses"]:
    trav_by_hint.setdefault(t["seq_hint"], t)

first_below = []
for slot in sorted(survivor_slots):
    hist = sorted(by_slot.get(slot, []), key=lambda r: r["seq_hint"])
    if not hist:
        continue
    line = f"slot {slot}: "
    prev_post = None
    for r in hist:
        k = r["slot"].tolist().index(slot)
        wpre = r["pre_w"][k, 0].item()
        wpost = r["post_w"][k, 0].item() if r["post_w"] is not None else float("nan")
        bl_pre = r["pre_bl"][k].item()
        bl_post = r["post_bl"][k].item() if r["post_bl"] is not None else -1
        st_post = r["post_status"][k].item() if r["post_status"] is not None else -1
        nh = r["num_hits"][k].item()
        line += (
            f"|b{r['seq_hint'] - sequences[0][0]['seq']}:w{wpre:.3e}->{wpost:.3e}"
            f",bl{bl_pre:.0f}->{bl_post:.0f},nh{nh}"
            f"{',DIED' if st_post == 1 else ''} "
        )
        if prev_post is not None and prev_post >= MIN_WEIGHT > wpre:
            first_below.append((slot, r))
        prev_post = wpost
    print(line)

print(
    f"\nsurvivor slots (entered a terminal bounce): {len(survivor_slots)}; "
    f"records below MIN_WEIGHT at some bounce: {len(first_below)}"
)

# Material of the last hit for rays that died in the final iteration.
print("\n--- last-hit primitive + material row of final-iteration deaths ---")
MAT_SLOTS = {
    "roughness": 8,
    "metalness": 9,
    "ior": 12,
    "transmission": 24,
}
seen_mat = set()
for r in REC["shades"]:
    if r["post_status"] is None:
        continue
    dead = (r["post_status"] == 1).nonzero().flatten()
    if not len(dead):
        continue
    t = trav_by_hint.get(r["seq_hint"])
    if t is None:
        continue
    pos = {s.item(): i for i, s in enumerate(t["active"].tolist())}
    for k in dead.tolist():
        slot = r["slot"][k].item()
        if slot not in pos:
            continue
        i = pos[slot]
        prims = t["hit_i"][i, :, 0]
        flags = t["hit_i"][i, :, 1]
        ts = t["hit_f"][i, :, 0]
        valid = (prims >= 0) & (ts > 0)
        if not bool(valid.any()):
            continue
        q = int(valid.nonzero().flatten()[0])
        prim = int(prims[q])
        htype = int(flags[q]) & 3
        key = (prim, r["time_start"])
        if key in seen_mat or htype != 1:
            continue
        seen_mat.add(key)
        f = r["time_start"]
        mat_id_arr = None
        line = (
            f"f{f} prim={prim} pix={int(r['rs_pix'][k])} t_hit={ts[q]:.5f}"
        )
        mat_rows[key] = line
for (prim, f), line in sorted(mat_rows.items())[:20]:
    print(" ", line)
print(f"\n(unique dead-ray last-hit triangle prims sampled: {len(mat_rows)})")

with open(os.path.join(HERE, "survivors_v2_summary.json"), "w") as fh:
    json.dump(
        {
            "tiles": [[e["n"] for e in s] for s in sequences],
            "survivor_slots": sorted(int(s) for s in survivor_slots),
            "dead_last_hits": [
                {"frame": f, "prim": p, "line": l} for (p, f), l in mat_rows.items()
            ],
        },
        fh,
        indent=1,
    )
print("[probe] wrote survivors_v2.pt / survivors_v2_summary.json")
