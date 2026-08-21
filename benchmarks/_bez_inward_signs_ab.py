"""A/B parity + timing for the per-circuit gathered inward-sign probe.

Captures every (edges, vert_circuit) pair the render pipeline feeds to
``_circuit_edge_inward_signs`` during a text-heavy PREVIEW render, then
checks the gathered CSR implementation returns bit-identical sigma to the
original all-page-edges masked probe (kept verbatim below as the
reference), and times both in-process (alternating, CPU tensors as in prep).

    .venv/Scripts/python.exe benchmarks/_bez_inward_signs_ab.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

import algan.rendering.raytracing.primitives as rtp  # noqa: E402
from algan import (  # noqa: E402
    BLUE,
    DOWN,
    LEFT,
    PREVIEW,
    RIGHT,
    UP,
    Circle,
    Scene,
    Sync,
    Text,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "_bez_inward_signs_out")
os.makedirs(OUT_DIR, exist_ok=True)


# --- reference: the pre-CSR implementation, verbatim -----------------------
def _circuit_parity_ref(qx, qy, ex0, ey0, ex1, ey1, same_circuit):
    y0 = ey0.unsqueeze(1)  # [T, 1, V]
    y1 = ey1.unsqueeze(1)
    v = qy.unsqueeze(-1)  # [T, Q, 1]
    straddle = (y0 > v) != (y1 > v)
    denom = y1 - y0
    denom = torch.where(denom == 0, torch.ones_like(denom), denom)
    x_cross = (
        ex0.unsqueeze(1) + (v - y0) * (ex1.unsqueeze(1) - ex0.unsqueeze(1)) / denom
    )
    hit = straddle & (x_cross > qx.unsqueeze(-1)) & same_circuit.unsqueeze(0)
    return hit.sum(-1) % 2 == 1


def _inward_signs_ref(edges, vert_circuit):
    T, V = edges.shape[0], edges.shape[1]
    device = edges.device
    if V == 0:
        return torch.zeros((T, 0), device=device)
    ex0, ey0 = edges[..., 0], edges[..., 1]
    ex1, ey1 = edges[..., 2], edges[..., 3]
    mx = 0.5 * (ex0 + ex1)
    my = 0.5 * (ey0 + ey1)
    dx = ex1 - ex0
    dy = ey1 - ey0
    length = torch.sqrt(dx * dx + dy * dy)
    degen = (length < 1e-12) | (edges[..., :4].abs() >= 1e8).any(-1)
    inv_len = 1.0 / torch.clamp(length, min=1e-12)
    lnx = -dy * inv_len
    lny = dx * inv_len
    circ = vert_circuit.to(device)

    sigma = torch.zeros((T, V), device=device)
    unresolved = ~degen
    eps = 0.05
    for _attempt in range(6):
        idx = unresolved.any(0).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            break
        budget = 4_000_000
        chunk = max(1, budget // max(T * V, 1))
        for start in range(0, idx.numel(), chunk):
            sel = idx[start : start + chunk]
            same_sel = circ[sel].view(-1, 1) == circ.view(1, -1)
            off_x = (eps * length * lnx)[:, sel]
            off_y = (eps * length * lny)[:, sel]
            qx, qy = mx[:, sel], my[:, sel]
            left = _circuit_parity_ref(
                qx + off_x, qy + off_y, ex0, ey0, ex1, ey1, same_sel
            )
            right = _circuit_parity_ref(
                qx - off_x, qy - off_y, ex0, ey0, ex1, ey1, same_sel
            )
            valid = (left != right) & unresolved[:, sel]
            s = torch.where(left, 1.0, -1.0)
            sigma[:, sel] = torch.where(valid, s, sigma[:, sel])
            unresolved[:, sel] &= ~valid
        eps *= 0.5
    return sigma


# --- capture real pipeline inputs -------------------------------------------
captured = []
_orig = rtp._circuit_edge_inward_signs


def _capturing(edges, vert_circuit):
    captured.append((edges.detach().clone(), vert_circuit.detach().clone()))
    return _orig(edges, vert_circuit)


rtp._circuit_edge_inward_signs = _capturing

with Sync():
    t1 = Text("The quick brown fox jumps over the lazy dog 0123456789").scale(0.45)
    t1.move(UP * 1.5).spawn()
    t2 = Text("pack my box with five dozen liquor jugs !@#$%&*()").scale(0.45)
    t2.spawn()
    c = Circle().scale(0.7).move(DOWN * 1.4 + LEFT * 1.5).set_color(BLUE)
    c.spawn()
with Sync():
    t1.move(RIGHT * 0.6)
    c.move(RIGHT * 0.8)

Scene.save_video(os.path.join(OUT_DIR, "capture"), PREVIEW, overwrite=True)
rtp._circuit_edge_inward_signs = _orig

if not captured:
    raise SystemExit(
        "FAIL: pipeline never called _circuit_edge_inward_signs "
        "(is the wedge mode off?)"
    )

# --- parity + timing --------------------------------------------------------
total_mismatch = 0
for i, (edges, circ) in enumerate(captured):
    ref = _inward_signs_ref(edges, circ)
    new = _orig(edges, circ)
    same = torch.equal(ref, new)
    n_circ = int(circ.max()) + 1
    print(
        f"case {i}: T={edges.shape[0]} V={edges.shape[1]} circuits={n_circ} "
        f"identical={same}"
    )
    if not same:
        diff = (ref != new).sum().item()
        total_mismatch += diff
        bad = (ref != new).any(0).nonzero(as_tuple=True)[0][:8]
        print(f"  {diff} mismatched entries, first edges: {bad.tolist()}")

edges, circ = max(captured, key=lambda p: p[0].shape[0] * p[0].shape[1] ** 2)
print(f"timing on T={edges.shape[0]} V={edges.shape[1]}:")
t_ref = t_new = 0.0
for _round in range(3):
    t0 = time.perf_counter()
    _inward_signs_ref(edges, circ)
    t_ref += time.perf_counter() - t0
    t0 = time.perf_counter()
    _orig(edges, circ)
    t_new += time.perf_counter() - t0
print(
    f"  reference {t_ref / 3:.4f} s   gathered {t_new / 3:.4f} s   "
    f"speedup {t_ref / max(t_new, 1e-9):.1f}x"
)

if total_mismatch:
    raise SystemExit(f"FAIL: {total_mismatch} sigma mismatches")
print("PASS: bit-identical sigma on all captured cases")
