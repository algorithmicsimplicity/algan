"""Count the tensor dispatches inside one _window_pairs call.

Feeds synthetically-built bounds tables (same schema/dtypes that
precompute_*_screen_bounds produce) into raster_pipeline._window_pairs under a
TorchDispatchMode, and tallies every dispatched aten op. Splits the tally into
the shared prologue and each _class_pairs_flat invocation by running them
separately. CPU tensors only -- op COUNTS are device-independent.

Run: ALGAN_USE_DAEMON=0 uv run python count_window_pairs_dispatches.py
"""

from __future__ import annotations

import os

os.environ["ALGAN_USE_DAEMON"] = "0"

from collections import Counter

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from algan.rendering.raytracing import raster_pipeline as rp
from algan.settings import SETTINGS


class OpCounter(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops = Counter()
        self.sync_ops = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func)
        self.ops[name] += 1
        if "nonzero" in name or "item" in name or "unique" in name:
            self.sync_ops[name] += 1
        return func(*args, **(kwargs or {}))


def build_bounds(frames, nprim, width, height, seed=0, frac_on=0.9):
    g = torch.Generator().manual_seed(seed)
    ymin = torch.rand(frames, nprim, generator=g) * (height - 20)
    ymax = ymin + torch.rand(frames, nprim, generator=g) * 19 + 1
    pre_f = torch.stack(((ymin - 1.0).floor(), (ymax + 1.0).ceil(), ymin, ymax), -1)
    bx = torch.randint(0, width - 52, (frames, nprim), generator=g)
    pre_x = torch.stack(
        (bx, bx + torch.randint(1, 50, (frames, nprim), generator=g)), -1
    ).to(torch.int64)
    # ~90% of primitives carry candidates in both classes so both
    # _class_pairs_flat calls do real work (nn-scene-like).
    m3 = torch.rand(frames, nprim, generator=g) < frac_on
    m4 = torch.rand(frames, nprim, generator=g) < frac_on
    pre_m = torch.stack(
        (
            torch.ones(frames, nprim, dtype=torch.bool),
            torch.ones(frames, nprim, dtype=torch.bool),
            torch.zeros(frames, nprim, dtype=torch.bool),
            m3,
            m4,
        ),
        -1,
    )
    cls_any = [[bool(m3[f].any()), bool(m4[f].any())] for f in range(frames)]
    return (pre_f, pre_x, pre_m, cls_any)


def main():
    width, height = 854, 480  # PREVIEW-scale; op counts do not depend on size
    ppf = width * height
    frames = 2
    device = torch.device("cpu")

    print(f"RASTER_PAIR_FLAGS = {SETTINGS.raytracing.RASTER_PAIR_FLAGS}")

    # Census split: both classes live / one class empty-masked / flags say
    # nothing anywhere (fast path). Same tables each time except the masks.
    nprim = 6000
    base = build_bounds(frames, nprim, width, height)

    def with_masks(m3, m4):
        pre_f, pre_x, _, _ = base
        cls_any = [[bool(m3[f].any()), bool(m4[f].any())] for f in range(frames)]
        pre_m = base[2].clone()
        pre_m[..., 3] = m3
        pre_m[..., 4] = m4
        return (pre_f, pre_x, pre_m, cls_any)

    m_on = base[2][..., 3]
    m_off = torch.zeros_like(m_on)
    for label, m3, m4 in (
        ("both classes live", m_on, m_on),
        ("one class empty   ", m_off, m_on),
        ("both empty (skip) ", m_off, m_off),
    ):
        bounds = with_masks(m3, m4)
        c = OpCounter()
        with c:
            po, pt = rp._window_pairs(bounds, 7, 0, frames * ppf, ppf, width, device)
        print(
            f"{label}: {sum(c.ops.values()):3d} dispatched ops "
            f"({sum(c.sync_ops.values())} syncs), rows={sum(p.shape[0] for p in (po, pt) if p is not None)}"
        )

    for label, nprim in (("bez-like", 6000), ("tri-like", 20000)):
        bounds = build_bounds(frames, nprim, width, height)

        c_all = OpCounter()
        with c_all:
            po, pt = rp._window_pairs(bounds, 7, 0, frames * ppf, ppf, width, device)
        n_rows = sum(None if p is None else p.shape[0] for p in (po, pt))

        c_pro = OpCounter()
        with c_pro:
            f_rel = torch.arange(0, frames, device=device)
            f_abs = f_rel + 7
            rows = f_abs % frames
            _ = bounds[0].index_select(0, rows)

        print(f"\n=== {label}: nprim={nprim}, frames={frames} ===")
        total = sum(c_all.ops.values())
        kern = sum(
            v
            for k, v in c_all.ops.items()
            if not any(s in k for s in ("view", "reshape", "slice", "detach", "select"))
        )
        print(f"dispatched aten ops TOTAL : {total}")
        print(f"  excluding pure view/meta: ~{kern}")
        print(
            f"host-syncing ops          : {sum(c_all.sync_ops.values())} -> {dict(c_all.sync_ops)}"
        )
        print(f"pair rows emitted         : {n_rows}")
        print("op breakdown:")
        for k, v in sorted(c_all.ops.items(), key=lambda kv: -kv[1]):
            print(f"  {v:3d}  {k}")

        # Per-class body alone (what one _class_pairs_flat costs)
        pre_f, pre_x, pre_m, _ = bounds
        mask = pre_m[..., 3]
        x0 = pre_x[..., 0]
        x1 = pre_x[..., 1]
        y0 = torch.randint(0, height - 2, (frames, nprim))
        y1 = y0 + 20
        f_abs_t = torch.arange(frames)
        c_cls = OpCounter()
        with c_cls:
            rp._class_pairs_flat(mask, x0, x1, y0, y1, f_abs_t, device)
        print(f"\n_class_pairs_flat alone   : {sum(c_cls.ops.values())} dispatched ops")
        for k, v in sorted(c_cls.ops.items(), key=lambda kv: -kv[1]):
            print(f"  {v:3d}  {k}")


if __name__ == "__main__":
    main()
