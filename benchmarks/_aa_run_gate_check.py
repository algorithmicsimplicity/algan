"""Why the analytic-AA RUN correction does not fire on a diced mesh silhouette.

``benchmarks/_aa_line_check.py`` measures the symptom: a tessellated ``Cylinder``
scores 0.057 px of ink wobble against 0.014 for a flat quad, and gets *worse* the
more finely it is diced, because "its silhouette pixels are contended by several
triangles and fall back to the 8-sub-pixel-sample masks". The run rule
(``ANALYTIC_AA_RUN``, ``DESIGN_analytic_aa_v2.md`` §4) exists to give exactly
those pixels a continuous scalar instead of eighths, so the question is which of
its gates closes.

This script answers that by population statistics rather than one pixel at a
time. It intercepts the sparse-raster fragment build, which hands the resolve the
compact per-pixel fragment lists it walks (``frag_ref``/``frag_cov``/``frag_msk``
plus the ``run_offsets`` CSR), and replays the run rule's *grouping* decision on
the host for **every** covered pixel. Four mutually exclusive verdicts per pixel:

``full``
    The first fragment already has a full sample mask, so the lookahead never
    runs and no correction is wanted (v2 §4.2: the hot path must not pay).
``corrected``
    A contiguous run sharing ``(sid, facing)`` starts at a partial mask and
    ``corr = E/Q`` applies to the whole sheet. Exact.
``union-full``
    The run's masks OR to a FULL mask, so v2 §4.2 short-circuits ``corr = 1``
    and never consults ``E``. Correct when the sheet really does tile the pixel
    (``1 - E`` is float dust); real silhouette dilation of ``1 - E`` when the
    sheet contains all eight sample points but not the pixel's whole area.
``split``
    One ``(sid, facing)`` sheet's fragments are **interleaved** with another's,
    so the maximal *consecutive* run covers only part of the sheet.
``capped``
    The contiguous run is longer than ``_AA_MAX_RUN_SCAN``, so the remainder is
    left at ``corr = 1``.

MEASURED (2026-08, CPU, ``--res md``). ``split`` is ~0.02% everywhere and
``capped`` under 1%, so the *grouping* is sound -- the consecutive-run
requirement is not what costs a diced mesh its accuracy, which refutes the
obvious hypothesis. What grows with tessellation density is ``union-full``:
1.0% on the flat quad, 25.2% on a default ``Cylinder``, 72.4% at
``resolution=(256, 2)``, 87.6% on a fine ``Sphere``. Almost all of it is the
benign interior-tiling case (``1 - E <= 1e-3``: 343 / 10770 / 31096 / 23282
pixels), and the residual is a small tail of genuinely dilated silhouette
pixels -- 1 / 105 / 181 / 1004 pixels with ``1 - E`` up to 0.15 (0.30 on the
sphere). That tail sits exactly along the silhouette, which is where
``_aa_line_check`` measures ink wobble.

Grouping and magnitude are all this measures; it deliberately does not
re-derive ``svis``, which is a consequence of the walk rather than an input to
the ordering question.

Run:  <venv-python> benchmarks/_aa_run_gate_check.py [--res md|ld|hd] [--cases ...]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from algan import (  # noqa: E402
    HD,
    LD,
    MD,
    OUT,
    RIGHT,
    UP,
    WHITE,
    Cylinder,
    Off,
    Scene,
    Sphere,
    TriangleTriangulated,
)
from algan.rendering.raytracing import raster_pipeline as rp  # noqa: E402
from algan.rendering.raytracing.raster_taichi import (  # noqa: E402
    _AA_BACKFACE_BIT,
    _AA_MASK_ALL,
    _AA_MAX_RUN_SCAN,
    _AA_NUM_SAMPLES,
)

RESOLUTIONS = {"ld": LD, "md": MD, "hd": HD}


def _cases():
    """Scene builders, coarse to fine, plus a flat-triangle control."""

    def quad():
        # Two triangles, off-axis so its edges cross pixels at every sub-pixel
        # phase -- the flat-triangle control _aa_line_check uses.
        a = (RIGHT * -1.2 + UP * -0.75) * 1.0
        b = (RIGHT * 1.2 + UP * -0.55) * 1.0
        c = (RIGHT * 1.2 + UP * 0.75) * 1.0
        d = (RIGHT * -1.2 + UP * 0.55) * 1.0
        corners = torch.stack([a, b, c, a, c, d]).view(2, 3, 3)
        TriangleTriangulated(corners, color=WHITE).spawn()

    def cylinder():
        Cylinder(radius=0.9, height=1.8).rotate(24, RIGHT).spawn()

    def cylinder_fine():
        c = Cylinder(radius=0.9, height=1.8, resolution=(256, 2))
        c.rotate(24, RIGHT).spawn()

    def sphere_fine():
        Sphere(radius=0.9, resolution=(192, 96)).rotate(20, OUT).spawn()

    return {
        "quad (flat control)": quad,
        "cylinder (default)": cylinder,
        "cylinder (256x2)": cylinder_fine,
        "sphere (192x96)": sphere_fine,
    }


def _classify(sids, facings, msks, covs):
    """Verdict for one pixel's ordered fragment list, plus the magnitude error
    the run rule leaves behind.

    Replays ``_aa_run_scan`` + the ``corr`` derivation of v2 §4.2:

        E    = sum of exact clipped areas over the run
        U    = OR of the run's sample masks
        Q    = popcount(U) / N
        corr = 1                     if U == MASK_ALL
             = clamp(E / Q, 0.5, 2)  otherwise

    ``err`` is the signed coverage the pixel then claims minus the exact area
    the run actually covers: ``Q * corr - E``. It is zero whenever the
    correction is exact, and nonzero exactly where the design knowingly gives up.
    """
    n = len(sids)
    if n == 0:
        return "empty", 0, 0.0
    if (msks[0] & _AA_MASK_ALL) == _AA_MASK_ALL:
        return "full", n, 0.0

    key0 = (sids[0], facings[0])
    # The maximal CONSECUTIVE run the kernel's _aa_run_scan would take.
    run = 0
    while run < n and (sids[run], facings[run]) == key0:
        run += 1
    # Every fragment of that sheet anywhere in the pixel's list.
    sheet = sum(1 for i in range(n) if (sids[i], facings[i]) == key0)

    scanned = min(run, _AA_MAX_RUN_SCAN)
    E = sum(covs[i] for i in range(scanned))
    U = 0
    for i in range(scanned):
        U |= msks[i] & _AA_MASK_ALL
    Q = bin(U).count("1") / _AA_NUM_SAMPLES

    if U == _AA_MASK_ALL:
        # v2 §4.2: a tiling that fills the mask is taken as fully covering, so
        # the exact area is never consulted. Over-covers by (1 - E).
        return "union-full", n, 1.0 - E
    if Q <= 0.0:
        return "donor-only", n, E
    corr = min(max(E / Q, 0.5), 2.0)
    err = Q * corr - E

    if sheet > run:
        return "split", n, err
    if run > _AA_MAX_RUN_SCAN:
        return "capped", n, err
    return "corrected", n, err


def _measure(build, settings):
    """Render once, replaying the run rule's grouping for every covered pixel."""
    stats = Counter()
    modes = []
    frag_hist = Counter()
    lost_total = [0.0]
    uf_hist = Counter()

    original = rp.prepare_sparse_raster_coverage

    def spy(*args, **kwargs):
        coverage = original(*args, **kwargs)
        if coverage is None:
            return coverage
        modes.append((int(coverage["aa_tri"]), int(coverage["aa_grp"])))
        n_cov = int(coverage["num_covered"])
        if n_cov <= 0:
            return coverage
        offs = coverage["run_offsets"][: n_cov + 1].detach().cpu().tolist()
        ref = coverage["frag_ref"].detach().cpu()
        cov = coverage["frag_cov"].detach().cpu()
        msk = coverage["frag_msk"].detach().cpu()
        tri_obj = kwargs.get("merged", args[0])["tri_obj"].detach().cpu()
        # tri_obj is [T?, N]; a diced mesh's row->surface map can move per frame,
        # but a fragment's frame is not carried in the compact arrays. Every
        # case here is single-frame, so row 0 is the mapping.
        obj_row = tri_obj[0]

        refs = ref.tolist()
        covs = cov.tolist()
        msks = msk.tolist()
        for i in range(n_cov):
            lo, hi = offs[i], offs[i + 1]
            sids, faces, ms, cs = [], [], [], []
            for j in range(lo, hi):
                r = refs[j]
                if r >= 0:
                    sids.append(int(obj_row[r]))
                else:
                    sids.append(-1 - ((-r - 1) >> 8))
                faces.append(1 if (msks[j] & _AA_BACKFACE_BIT) else 0)
                ms.append(msks[j])
                cs.append(covs[j])
            verdict, nfrag, lost = _classify(sids, faces, ms, cs)
            stats[verdict] += 1
            if verdict == "union-full":
                # How far below 1 the exact area sits when the mask says full:
                # float dust (<= 1e-3) is what the short-circuit exists for,
                # anything larger is real silhouette dilation.
                d = abs(lost)
                if d <= 1e-3:
                    uf_hist["dust <=1e-3"] += 1
                elif d <= 0.05:
                    uf_hist["<=0.05"] += 1
                elif d <= 0.15:
                    uf_hist["<=0.15"] += 1
                elif d <= 0.30:
                    uf_hist["<=0.30"] += 1
                else:
                    uf_hist[">0.30"] += 1
            frag_hist[min(nfrag, 12)] += 1
            lost_total[0] += abs(lost)
        return coverage

    rp.prepare_sparse_raster_coverage = spy
    try:
        with Scene() as scene:
            with Off():
                build()
            scene.save_frame(
                str(REPO / "algan_outputs" / "_aa_run_gate.png"),
                video_settings=settings,
                overwrite=True,
            )
    finally:
        rp.prepare_sparse_raster_coverage = original
    return stats, frag_hist, lost_total[0], modes, uf_hist


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--res", choices=sorted(RESOLUTIONS), default="md")
    ap.add_argument("--cases", nargs="*", default=None)
    args = ap.parse_args()
    settings = RESOLUTIONS[args.res]

    cases = _cases()
    if args.cases:
        cases = {k: v for k, v in cases.items() if any(c in k for c in args.cases)}

    print(f"samples/pixel = {_AA_NUM_SAMPLES}, run-scan cap = {_AA_MAX_RUN_SCAN}")
    print(f"resolution = {args.res}\n")
    header = (
        f"{'case':22s} {'covered':>8s} {'full':>7s} {'corrected':>10s} "
        f"{'union-full':>11s} {'split':>6s} {'capped':>7s} {'|err| sum':>10s}"
    )
    print(header)
    print("-" * len(header))
    for name, build in cases.items():
        stats, frag_hist, lost, modes, uf_hist = _measure(build, settings)
        total = sum(stats.values())
        if not total:
            print(f"{name:22s} {'(no covered pixels)':>9s}")
            continue

        def pct(k, _stats=stats, _total=total):
            return f"{100.0 * _stats[k] / _total:7.2f}%"

        print(
            f"{name:22s} {total:8d} {pct('full'):>7s} {pct('corrected'):>10s} "
            f"{pct('union-full'):>11s} {pct('split'):>6s} {pct('capped'):>7s} "
            f"{lost:10.2f}"
        )
        print(f"{'':22s} aa_tri/aa_grp modes {sorted(set(modes))}")
        if uf_hist:
            order = ["dust <=1e-3", "<=0.05", "<=0.15", "<=0.30", ">0.30"]
            uf = "  ".join(f"{k}:{uf_hist[k]}" for k in order if uf_hist[k])
            print(f"{'':22s} union-full (1-E)  {uf}")
        hist = " ".join(
            f"{k}:{frag_hist[k]}" for k in sorted(frag_hist) if frag_hist[k]
        )
        print(f"{'':22s} fragments/pixel  {hist}")
    print(
        "\n'split'/'capped' near zero => the consecutive-run GROUPING is sound.\n"
        "What scales with tessellation density is 'union-full': the sheet's\n"
        "fragments OR to a full sample mask, so corr short-circuits to 1 and the\n"
        "exact area E is never consulted. Read its (1-E) histogram: the 'dust'\n"
        "bucket is the benign interior tiling the short-circuit exists for; the\n"
        "buckets above it are silhouette pixels dilated by (1-E), and they are\n"
        "the ones _aa_line_check's ink wobble sees."
    )


if __name__ == "__main__":
    main()
