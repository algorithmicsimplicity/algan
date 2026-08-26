"""Analyze survivors_v2.pt (vectorized): per-bounce traces of tail-cohort rays."""
import json
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MIN_WEIGHT = 1e-3   # raytrace_kernels_taichi.py:132

D = torch.load(os.path.join(HERE, "survivors_v2.pt"), weights_only=False)
sel, shades, traverses, marks = (
    D["selects"], D["shades"], D["traverses"], set(D["alloc_marks"])
)

# --- tile sequences ---------------------------------------------------------
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
plateau_slots = set()
for si, s in enumerate(sequences):
    per = [x["n"] for x in s]
    print(f"  tile{si}: {per}")
    for e in s:
        if "slots" in e:
            plateau_slots.update(int(v) for v in e["slots"].tolist())

print(f"\nslots observed in small post-shade selections: {len(plateau_slots)}")

# --- index SMALL shade records only (the tail iterations) --------------------
small_shades = [r for r in shades if r["na"] <= 512]
slot_hist = {s: [] for s in plateau_slots}
for r in small_shades:
    sl = r["slot"].numpy()
    for j, s in enumerate(sl):
        if int(s) in slot_hist:
            slot_hist[int(s)].append((r, j))

trav_by_hint = {}
for t in traverses:
    if t["active"].numel() <= 512:
        trav_by_hint.setdefault(t["seq_hint"], t)

# bounce label: within one tile, seq order == iteration order; derive from
# bounces_left arithmetic instead of guessing offsets.
rows_out = []
below = []
print("\nper-ray weight/bounces_left history (rays whose death was captured):")
shown = 0
wasted_iters = []
for slot in sorted(slot_hist):
    hist = sorted(slot_hist[slot], key=lambda rj: rj[0]["seq_hint"])
    if not hist:
        continue
    trace = []
    first_below_seq = None
    prev_post = None
    died = None
    for r, k in hist:
        wpre = float(r["pre_w"][k, 0])
        wpost = float(r["post_w"][k, 0])
        blp, blq = int(r["pre_bl"][k]), int(r["post_bl"][k])
        stq = int(r["post_status"][k])
        nh = int(r["num_hits"][k])
        trace.append(
            {"seq": r["seq_hint"], "w_pre": wpre, "w_post": wpost,
             "bl_pre": blp, "bl_post": blq, "num_hits": nh, "status_post": stq}
        )
        if stq == 1:
            died = trace[-1]
        if first_below_seq is None and prev_post is not None \
                and prev_post >= MIN_WEIGHT > wpre:
            first_below_seq = r["seq_hint"]
        prev_post = wpost
    rows_out.append(
        {"slot": slot, "trace": trace,
         "first_below_min_weight_seq": first_below_seq}
    )
    if died is not None and first_below_seq is not None:
        wasted_iters.append(len(trace) - 1 - trace.index(died))
    if died is not None and shown < 14:
        line = f" slot {slot:>6}: "
        for x in trace:
            tag = (
                "DEAD"
                if x["status_post"] == 1
                else ("bounce" if x["bl_post"] < x["bl_pre"] else "pass")
            )
            line += (
                f"[w {x['w_pre']:.2e}->{x['w_post']:.2e}"
                f" bl {x['bl_pre']}->{x['bl_post']} nh{x['num_hits']} {tag}] "
            )
        print(line)
        shown += 1

n_below = sum(1 for r in rows_out if r["first_below_min_weight_seq"] is not None)
print(
    f"\ntail rays traced: {len(rows_out)}; crossed below MIN_WEIGHT mid-flight:"
    f" {n_below}"
)
if wasted_iters:
    print(
        f"iterations spent below the floor before retirement: "
        f"min={min(wasted_iters)} max={max(wasted_iters)} "
        f"mean={sum(wasted_iters) / len(wasted_iters):.1f}"
    )

# deaths + last hits
death_kinds = {}
prim_rows = []
trav_pos = {
    h: {int(s): i for i, s in enumerate(t["active"].tolist())}
    for h, t in trav_by_hint.items()
}
for row in rows_out:
    dead = [x for x in row["trace"] if x["status_post"] == 1]
    if not dead:
        continue
    x = dead[-1]
    kind = (
        "floor after pass-through at bl=0"
        if x["bl_pre"] <= 0 and x["w_post"] < MIN_WEIGHT
        else ("floor" if x["w_post"] < MIN_WEIGHT else "other")
    )
    death_kinds[kind] = death_kinds.get(kind, 0) + 1
    # last-hit primitive from that iteration's traverse
    r = next(rr for rr in small_shades if rr["seq_hint"] == x["seq"]
             and slot_hist[row["slot"]] and any(
                 rr2[0] is rr for rr2 in slot_hist[row["slot"]]))
    k = [j for rr, j in slot_hist[row["slot"]] if rr["seq_hint"] == x["seq"]][0]
    t = trav_by_hint.get(x["seq"])
    if t is not None and row["slot"] in trav_pos.get(x["seq"], {}):
        i = trav_pos[x["seq"]][row["slot"]]
        prims = t["hit_i"][i, :, 0].numpy()
        flags = t["hit_i"][i, :, 1].numpy()
        ts = t["hit_f"][i, :, 0].numpy()
        ok = (prims >= 0) & (ts > 0)
        if ok.any():
            q = int(np.argmax(ok))
            prim_rows.append(
                {"slot": row["slot"], "seq": x["seq"],
                 "prim": int(prims[q]), "htype": int(flags[q]) & 3,
                 "t_hit": float(ts[q]), "flags": int(flags[q]),
                 "death": kind}
            )

print("death kinds:", death_kinds)
print("\nsample last-hit records:")
for p in prim_rows[:12]:
    print(" ", p)

with open(os.path.join(HERE, "survivor_traces.json"), "w") as fh:
    json.dump({"rows": rows_out, "last_hits": prim_rows}, fh, indent=1)
print("[analyze] wrote survivor_traces.json")
