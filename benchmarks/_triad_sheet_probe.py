"""Dump the fragment and sheet records of one pixel of the triad frame.

The artifact under investigation is a pixel the sheet resolve awards entirely
to a surface that a supersampled reference says covers ~0% of it.  This prints
what the compaction actually saw and produced there, so the flip can be read
off the records rather than inferred.

Usage::

    <venv-python> benchmarks/_triad_sheet_probe.py --px 164 --py 214 --at 14.3
"""

from __future__ import annotations

import argparse
import os
import runpy
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser()
parser.add_argument("--px", type=int, nargs="+", default=[164])
parser.add_argument("--py", type=int, nargs="+", default=[214])
parser.add_argument("--at", type=float, default=14.3)
args = parser.parse_args()

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCENE = os.path.join(_ROOT, "tests", "full_renders", "scenes", "solids_and_camera.py")
sys.path.insert(0, os.path.join(_ROOT, "tests"))
from conftest import _register_test_fonts  # noqa: E402

_register_test_fonts()

import torch  # noqa: E402

from algan import PREVIEW, Scene  # noqa: E402
from algan.rendering.raytracing import sheets as _sheets  # noqa: E402

WIDTH, HEIGHT = 704, 396
TARGETS = [(x, y, y * WIDTH + x) for x, y in zip(args.px, args.py, strict=True)]


def _t_of(key):
    return struct.unpack("<f", struct.pack("<I", int(key) & 0xFFFFFFFF))[0]


_real_compact = _sheets.compact_sheets


def _probe(coverage, merged, *a, **kw):
    out = _real_compact(coverage, merged, *a, **kw)
    print(
        f"[probe] compaction: {int(coverage['num_covered'])} covered pixels, "
        f"{int(coverage['num_fragments'])} fragments, "
        f"{int(out['num_sheets'])} sheets"
    )
    if not getattr(_probe, "_dumped_keys", False):
        _probe._dumped_keys = True
        print("[probe] merged keys:", sorted(merged.keys()))
    covered = coverage["covered_idx"].to(torch.int64).cpu()
    runs = coverage["run_offsets"].to(torch.int64).cpu()
    key = coverage["frag_key"].cpu()
    ref = coverage["frag_ref"].cpu()
    cov = coverage["frag_cov"].cpu()
    msk = coverage["frag_msk"].cpu()
    s_key = out["sheet_key"].cpu()
    s_cov = out["sheet_cov"].cpu()
    s_wgt = out["sheet_wgt"].cpu()
    s_msk = out["sheet_msk"].cpu()
    s_wmsk = out["sheet_wmsk"].cpu()
    s_ref = out["sheet_ref"].cpu()
    s_off = out["sheet_offsets"].to(torch.int64).cpu()
    tri_obj = merged["tri_obj"]

    if not getattr(_probe, "_shapes", False):
        _probe._shapes = True
        for k in ("tri_obj", "tri_colors", "tri_mat_id", "tri_pos", "tri_mat"):
            v = merged.get(k)
            print(f"[probe] {k}: {tuple(v.shape) if v is not None else None}")

    tri_mat = merged.get("tri_mat")

    def _who(r):
        r = int(r)
        if r < 0:
            return "circuit"
        try:
            o = tri_obj
            obj = int(o[0, r]) if o.dim() == 2 else int(o[r])
        except Exception as exc:  # noqa: BLE001
            return f"?{exc}"
        mat = ""
        if tri_mat is not None:
            v = [round(float(x), 3) for x in tri_mat[0, r, :4]]
            mat = f" mat={v}"
        return f"obj={obj}{mat}"

    s_nfrag = out["sheet_nfrag"].cpu() if "sheet_nfrag" in out else None
    s_fused = out["sheet_fused"].cpu() if "sheet_fused" in out else None
    npix = WIDTH * HEIGHT
    base = int(covered.min()) // npix * npix
    print(
        f"[probe] covered_idx range {int(covered.min())}..{int(covered.max())}, "
        f"frame base {base}"
    )
    for x, y, lpi in TARGETS:
        for cand, how in (
            (base + y * WIDTH + x, "top-down"),
            (base + (HEIGHT - 1 - y) * WIDTH + x, "bottom-up"),
        ):
            hit = (covered == cand).nonzero().flatten()
            if len(hit):
                lpi = cand
                print(f"[probe] ({x},{y}) matched as {how}")
                break
        if not len(hit):
            print(f"[probe] ({x},{y}) not covered under either convention")
            continue
        i = int(hit[0])
        print(f"\n=== pixel ({x},{y}) lpi={lpi} covered slot {i} ===")
        f0, f1 = int(runs[i]), int(runs[i + 1])
        print(f"  {f1 - f0} fragments:")
        for j in range(f0, f1):
            print(
                f"    t={_t_of(key[j]):.6f} ref={int(ref[j]):6d} "
                f"cov={float(cov[j]):.4f} msk=0x{int(msk[j]) & 0xFFFFFFFF:08x} "
                f"bits={bin(int(msk[j]) & 0xFF)} {_who(ref[j])}"
            )
        a0, a1 = int(s_off[i]), int(s_off[i + 1])
        print(f"  {a1 - a0} sheets:")
        for j in range(a0, a1):
            extra = ""
            if s_nfrag is not None:
                extra += f" nfrag={int(s_nfrag[j])}"
            if s_fused is not None:
                extra += f" fused={bool(s_fused[j])}"
            print(
                f"    t={_t_of(s_key[j]):.6f} ref={int(s_ref[j]):6d} "
                f"cov={float(s_cov[j]):.4f} wgt={float(s_wgt[j]):.4f} "
                f"msk=0x{int(s_msk[j]) & 0xFFFFFFFF:08x} "
                f"wmsk=0x{int(s_wmsk[j]) & 0xFFFFFFFF:08x} "
                f"cap={float(out['sheet_cap'][j]):.3f}{extra} "
                f"{_who(s_ref[j])}"
            )
    return out


_sheets.compact_sheets = _probe

runpy.run_path(_SCENE, run_name="__algan_scene__")
Scene.save_frame("sheet_probe", PREVIEW.set(resolution=(WIDTH, HEIGHT)), at=args.at)
