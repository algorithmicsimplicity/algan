"""Acceptance harness for ``SHEET_SAMPLE_DEPTH`` (per-sample depth ceding).

The feature arbitrates a pixel where two opaque surfaces cross INSIDE it, so
the only reference that can judge it is one that resolves depth per sub-pixel
sample: the classic supersampled wavefront, which the sheet route replaces with
one depth decision per pixel. This renders the same frame three ways --

    off   sheet route, gate off (today's image)
    on    sheet route, gate on
    ref   route OFF at ``--aa``, i.e. one camera ray per sub-pixel sample

-- and scores ``on`` and ``off`` against ``ref``, reporting how many pixels the
gate moved and, of those, how many landed closer to the reference.

**One process per arm is mandatory.** ``SHEET_SAMPLE_DEPTH`` is read at import
into a module global, so flipping it in-process would leave the first arm's
value compiled into everything downstream; this script therefore re-executes
itself per arm rather than looping. That is the same discipline
``CLAUDE.md`` states for every ``ti.static``-gated toggle.

A ceded sample is claimed by the surface that won it, so the failure mode worth
watching for is a sample nobody claims -- which shows as the BACKGROUND. The
score prints that population separately; it should be empty.

Usage::

    <venv-python> benchmarks/_sample_depth_check.py [--at 14.3] [--aa 3]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _render(tag: str, at: float, env: dict, aa: int | None) -> str:
    """Run one arm in its own process and return the PNG it wrote."""
    cmd = [
        sys.executable,
        os.path.join(_ROOT, "benchmarks", "_triad_artifact_frame.py"),
        "--at",
        str(at),
        "--name",
        tag,
    ]
    if aa is not None:
        cmd += ["--aa", str(aa)]
    e = dict(os.environ)
    e["ALGAN_USE_DAEMON"] = "0"  # a warm daemon carries the other arm's globals
    e.update(env)
    out = subprocess.run(cmd, env=e, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"{tag} failed:\n{out.stdout[-2000:]}\n{out.stderr[-2000:]}")
    for line in out.stdout.splitlines():
        if line.startswith("wrote "):
            return line[len("wrote ") :].strip()
    raise SystemExit(f"{tag}: no output path in\n{out.stdout[-2000:]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--at", type=float, default=14.3)
    ap.add_argument(
        "--aa",
        type=int,
        default=3,
        help=(
            "sub-pixel samples per axis for the REFERENCE arm. The reference "
            "is far heavier than either sheet-route arm -- on a 16 GB CPU box "
            "4 does not fit this scene at 704x396 and 3 does."
        ),
    )
    ap.add_argument("--box", default="150,290,90,250", help="y0,y1,x0,x1")
    args = ap.parse_args()

    import cv2
    import numpy as np

    t = args.at
    # ``save_frame`` reads a bare name's suffix as an image format, so the
    # timestamp's decimal point cannot go in the tag: "sdc_off_14.3" asks PIL
    # to write a ".3" file and it raises.
    stamp = str(t).replace(".", "p")
    paths = {
        "off": _render(f"sdc_off_{stamp}", t, {"ALGAN_SHEET_SAMPLE_DEPTH": "0"}, None),
        "on": _render(f"sdc_on_{stamp}", t, {"ALGAN_SHEET_SAMPLE_DEPTH": "1"}, None),
        "ref": _render(f"sdc_ref_{stamp}", t, {"ALGAN_ANALYTIC_AA": "0"}, args.aa),
    }
    y0, y1, x0, x1 = (int(v) for v in args.box.split(","))
    sl = np.s_[y0:y1, x0:x1]
    off, on, ref = (
        cv2.imread(paths[k]).astype(np.int32)[sl] for k in ("off", "on", "ref")
    )

    d_off = np.abs(off - ref).max(axis=2)
    d_on = np.abs(on - ref).max(axis=2)
    changed = np.abs(on - off).max(axis=2) > 2
    n = int(changed.sum())
    print(f"\n=== SHEET_SAMPLE_DEPTH at t={t}, reference = route-off aa={args.aa} ===")
    print(f"pixels moved by the gate: {n}")
    if not n:
        print("nothing moved -- the gate found no qualifying crossing here.")
        return
    ys, xs = np.where(changed)
    e_off, e_on = d_off[ys, xs], d_on[ys, xs]
    better = int((e_on < e_off - 2).sum())
    worse = int((e_on > e_off + 2).sum())
    print(f"vs reference: {better} better, {worse} worse, {n - better - worse} ~same")
    print(f"|err| over moved px: off={int(e_off.sum())} on={int(e_on.sum())}")
    print(f"worst |err| in box:  off={int(d_off.max())} on={int(d_on.max())}")

    # A sample nobody claimed would show the background; none should.
    bg = np.median(np.concatenate([ref[0], ref[-1]]), axis=0)
    to_bg = int(
        (
            (np.abs(on[ys, xs] - bg).max(axis=1) < 20)
            & (np.abs(off[ys, xs] - bg).max(axis=1) >= 20)
        ).sum()
    )
    print(f"moved pixels that landed ON the background (unclaimed samples): {to_bg}")

    order = np.argsort(-(e_off - e_on))
    print("largest improvements (x,y, off->on):")
    for i in order[:8]:
        print(f"   ({xs[i] + x0},{ys[i] + y0}) {e_off[i]} -> {e_on[i]}")
    reg = [i for i in order[::-1] if e_on[i] > e_off[i] + 2]
    print(f"regressions ({len(reg)}):")
    for i in reg[:8]:
        print(f"   ({xs[i] + x0},{ys[i] + y0}) {e_off[i]} -> {e_on[i]}")


if __name__ == "__main__":
    main()
