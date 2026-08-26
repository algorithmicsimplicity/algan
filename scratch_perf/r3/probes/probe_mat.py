"""Paired traverse+shade capture on a single frame: full last-hit identity
(primitive id, htype, packed material row) for every tail-cohort ray."""
import json
import os

os.environ.setdefault("ALGAN_USE_DAEMON", "0")
os.environ.setdefault("MPLBACKEND", "Agg")

import torch

if not torch.cuda.is_available():
    torch.cuda.synchronize = lambda *a, **k: None

from algan import *  # noqa: E402,F403
from algan.mobs.neural_nets.neural_net import NeuralNetMLPV3  # noqa: E402
import algan.rendering.raytracing.tracer as TR  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LAST = {"trav": None}


_trav_orig = TR.wavefront_traverse_events


def trav_spy(*args):
    out = _trav_orig(*args)
    na = args[1]
    if isinstance(na, int) and 0 < na <= 512:
        LAST["trav"] = (
            args[0].detach().to(torch.int64).cpu().clone(),
            args[41].detach().cpu().clone(),   # hit_f [na,KBUF,4]
            args[42].detach().cpu().clone(),   # hit_i [na,KBUF,2]
        )
    return out


TR.wavefront_traverse_events = trav_spy

_shade_orig = TR.wavefront_shade
RECS = []


def shade_spy(*args):
    na = args[1]
    pair = LAST["trav"]
    pre = None
    if isinstance(na, int) and pair is not None and pair[0].numel() == na:
        idx = args[0].detach().to(torch.int64).cpu()
        if bool(torch.equal(idx, pair[0])):
            rs_int, rs_sca = args[59], args[58]
            pre = {
                "slot": idx.clone(),
                "bl": rs_int[idx, 0].cpu().clone(),
                "proc": rs_int[idx, 1].cpu().clone(),
                "nh": rs_int[idx, 3].cpu().clone(),
                "w": torch.stack(
                    [rs_sca[idx, 0], rs_sca[idx, 5], rs_sca[idx, 6]], 1
                ).cpu().clone(),
                "accum": rs_int[idx, 4].cpu().clone(),
                "pix": args[62][idx].cpu().clone(),
                "gloss_base": float(args[27][7]),
                "hit_f": pair[1],
                "hit_i": pair[2],
            }
    _shade_orig(*args)
    if pre is not None:
        idx = pre["slot"].to(torch.int64)
        rs_int = args[59]
        post_status = rs_int[idx, 2].cpu().clone()
        post_bl = rs_int[idx, 0].cpu().clone()
        rec = dict(pre)
        rec.pop("hit_f"), rec.pop("hit_i")
        rec["post_status"] = post_status
        rec["post_bl"] = post_bl
        rec["_hits"] = (pre["hit_f"], pre["hit_i"])
        rec["_matid_arr"] = args[46].cpu().clone() if na <= 64 else None
        rec["_mat_arr"] = args[47].cpu().clone() if na <= 64 else None
        rec["_frame"] = int(args[51])
        RECS.append(rec)
    return None


TR.wavefront_shade = shade_spy

# --- scene (verbatim) -------------------------------------------------------
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

Scene.save_frame("probe_mat", PREVIEW, at=12, overwrite=True)

# --- analysis ----------------------------------------------------------------
print(f"\n[mat] captured small shades: {len(RECS)}")
MAT_SLOTS = {"roughness": 8, "metalness": 9, "ior": 12, "transmission": 24}
seen = {}
for r in RECS:
    died = (r["post_status"] == 1)
    if not bool(died.any()):
        continue
    hits_f, hits_i = r["_hits"]
    mid, mrow = r["_matid_arr"], r["_mat_arr"]
    for k in torch.nonzero(died).flatten().tolist():
        bl_pre = int(r["bl"][k])
        wpre = float(r["w"][k, 0])
        i = k  # active ordinal == row in hit arrays
        prims = hits_i[i, :, 0].tolist()
        flags = hits_i[i, :, 1].tolist()
        ts = hits_f[i, :, 0].tolist()
        cells = [
            (p, fl, t) for p, fl, t in zip(prims, flags, ts) if p >= 0 and t > 0
        ]
        tag = "BL0" if bl_pre <= 0 else "bouncy"
        key = (r["_frame"], tuple(cells))
        line = (
            f"slot {r['slot'][k].item():>6} pix {int(r['pix'][k]):>6} "
            f"accum {int(r['accum'][k]):>6} gloss>={r['gloss_base']:.0f} "
            f"bl={bl_pre} w={wpre:.3e} hits={cells[:3]} [{tag}]"
        )
        if cells and cells[0][1] & 3 == 1 and mrow is not None:
            prim = cells[0][0]
            pid = int(mid[r["_frame"] % mid.shape[0], prim])
            row = mrow[r["_frame"] % mrow.shape[0], prim].tolist()
            mats = {n: round(row[s], 4) for n, s in MAT_SLOTS.items()}
            line += f" mat_id={pid} {mats}"
        if key not in seen:
            seen[key] = True
            print(" ", line)

with open(os.path.join(HERE, "mat_probe.json"), "w") as fh:
    import math

    def conv(o):
        if isinstance(o, torch.Tensor):
            return o.tolist()
        raise TypeError

    print("[mat] done")
