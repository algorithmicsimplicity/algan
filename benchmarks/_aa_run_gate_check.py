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

NEGATIVE RESULT, so nobody spends the effort twice. Both obvious fixes suggested
by the table above were built and measured, and neither closes the gap:

  * Regrouping the consecutive run into an order-independent equivalence class
    cannot help, because ``split`` is already ~0.02%.
  * Consulting ``E`` instead of short-circuiting ``corr = 1`` under a full mask
    (``corr = E`` there, since ``Q == 1``, with a 1e-3 dust band so genuine
    tilings stay bit-identical) moves ``_aa_line_check`` by nothing: default
    Cylinder ink wobble 0.0568 -> 0.0566 px with coverage rms 0.0094 -> 0.0099,
    and the fine Cylinder 0.0773 -> 0.0781 / 0.0164 -> 0.0166. The dilation tail
    is real but far too small a pixel population to move a frame-wide metric --
    which the ``dust`` bucket dominating every histogram already implies.


THE svis REPLAY -- WHAT THE PIXEL ACTUALLY ENDS UP WITH
-------------------------------------------------------
The verdicts above are about GROUPING and MAGNITUDE. They cannot say what
coverage a pixel finally carries, because that is decided by ``svis``, the
per-sample transmittance the resolve threads through the fragment list. So the
second half of this harness replays ``raster_first_shade``'s walk in Python from
the same compact arrays, for every covered pixel, and compares its coverage
against the EXACT area of (footprint n pixel) -- summed from one sheet's exact
clipped areas, with the other sheet required to agree or the pixel dropped. No
supersampled reference, no fitted model. ``--verify`` proves the replay against
the kernel's own ``ALGAN_AA_DUMP`` rows rather than asserting it (measured:
worst per-fragment ``eff`` difference 5e-8 over six cases).

MEASURED (2026-08, CPU, ``--res md``, mean over silhouette pixels):

    case               silh  |actual-E|  |own-E|  |actual-own|  on-lattice
    quad (control)      827      0.0020   0.0390        0.0370        7.9%
    cube                947      0.0250   0.0405        0.0241       51.0%
    icosahedron        1000      0.0492   0.0650        0.0180       58.9%
    cylinder           2307      0.0260   0.0367        0.0116       72.5%
    cylinder (256x2)   2139      0.0211   0.0329        0.0128       70.6%
    sphere (192x96)    2628      0.0383   0.0408        0.0047       90.8%

``own`` is ``popcount(union of every fragment mask)/N``: the pixel's coverage
with all magnitude information discarded. ``on-lattice`` is the share of
silhouette pixels whose painted coverage is an exact multiple of 1/N.

THE ANSWER IS OWNERSHIP. On the flat control the machinery works as designed --
error 0.0020 against an ownership floor of 0.0390, so 95% of the quantization is
removed and only 7.9% of pixels land on the sample lattice. On a diced closed
mesh it is neutralized: the sphere's painted coverage sits 0.0047 from the pure
ownership answer, 91% of its silhouette pixels land exactly on eighths, and the
error converges on the floor from below. The signed error is positive in every
case, which is the dilation ``_aa_line_check`` reads as ink wobble.

Two mechanisms produce it, and the by-verdict line separates them:

  * ``full`` (52% of the sphere's silhouette pixels, mean error 0.042). ONE
    fragment owns all N samples while covering less than the whole pixel, so
    the run scan never starts (v2 ss4.2 gates on a partial mask) and the pixel
    is painted at 1.0. Its exact area sits unread in ``frag_cov``.
  * The FAR SHEET re-claim. A run's ``corr < 1`` scales the occlusion write as
    well as the claim, so the samples the near sheet owns keep a residual
    transmittance -- standing for the part of the pixel the sheet does not
    cover, which at a silhouette lies OUTSIDE the mesh entirely. The residue has
    no position, so the far sheet of the same solid claims it, uncorrected
    (``svis`` is no longer uniform, so its own run cannot engage). The
    ``1sheet`` column suppresses it: 0.0250 -> 0.0041 on the cube (84% of the
    error), but only 0.0383 -> 0.0346 on the sphere, where ``full`` dominates.

Both are magnitude thrown away rather than magnitude unavailable, but neither is
reachable by the run rule as scoped: the first never enters it, and the second
needs to know that two sheets belong to ONE mesh -- which is what
``DESIGN_mesh_identity.md`` ss2.2 declares and no consumer yet reads.

A SIDE FINDING, load-bearing for mesh identity. ``Polyhedron`` builds each face
from a hardcoded index list and those lists are not consistently oriented:
measured, 12 of an ``Icosahedron``'s 20 faces wind inward, 2 of 4 on a
``Tetrahedron``, 2 of 8 on an ``Octahedron``, 3 of 12 on a ``Dodecahedron``, 0
of 6 on a ``Cube``. The projected winding sign IS ``_AA_BACKFACE_BIT``, so on
those solids the facing bit does not name a sheet -- 858 of the icosahedron's
46220 covered pixels have a "front" group holding both sheets, and this harness
drops them rather than referencing them wrongly.

Run:  <venv-python> benchmarks/_aa_run_gate_check.py [--res md|ld|hd]
                                                     [--cases ...] [--verify N]
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
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
    Cube,
    Cylinder,
    Icosahedron,
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
    _AA_SLIVER_BIT,
)
from algan.rendering.raytracing.raytrace_kernels_taichi import (  # noqa: E402
    MIN_ALPHA,
    MIN_WEIGHT,
)

RESOLUTIONS = {"ld": LD, "md": MD, "hd": HD}

# A pixel counts as SILHOUETTE when the geometry covers it partially. Both ends
# are excluded: a fully covered pixel has no coverage question left, and a pixel
# the geometry barely grazes carries no ink worth attributing.
_SILH_LO = 1e-3
_SILH_HI = 1.0 - 1e-3


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

    def cube():
        # The one CONSISTENTLY WOUND Polyhedron (0 of 6 faces inward), so the
        # facing bit really does name a sheet and the exact reference holds.
        # fill_opacity is explicit: a Cube is 0.75 by default (Manim's value),
        # and the replay's coverage question is an OPAQUE one.
        Cube(side_length=1.5, fill_opacity=1.0).rotate(24, RIGHT).rotate(31, UP).spawn()

    def polyhedron():
        # A Polyhedron arrives as one collection member per TRIANGLE. With
        # MESH_ID off every triangle is its own surface and no run can span a
        # facet boundary, so 'corrected' collapses; with it on the solid is one
        # surface. Run with ALGAN_MESH_ID=0/1 to see the difference.
        #
        # 12 of its 20 faces wind inward, so its facing bit does NOT separate
        # the two sheets -- kept deliberately, as the case that exhibits it
        # (see _exact_coverage). Its 'no trustworthy reference' count is the
        # measurement; the cube above is the referenced polyhedron.
        Icosahedron(edge_length=1.4).rotate(18, RIGHT).spawn()

    def cylinder():
        Cylinder(radius=0.9, height=1.8).rotate(24, RIGHT).spawn()

    def cylinder_fine():
        c = Cylinder(radius=0.9, height=1.8, resolution=(256, 2))
        c.rotate(24, RIGHT).spawn()

    def sphere_fine():
        Sphere(radius=0.9, resolution=(192, 96)).rotate(20, OUT).spawn()

    return {
        "quad (flat control)": quad,
        "cube (flat)": cube,
        "icosahedron (flat)": polyhedron,
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


# --------------------------------------------------------------------------
# The per-sample transmittance walk, replayed on the host (ss6.3)
# --------------------------------------------------------------------------
# ``_classify`` above answers "what did the run rule decide", which is a
# question about GROUPING and MAGNITUDE. It cannot answer "what coverage did
# the pixel end up with", because that is decided by ``svis`` -- the per-sample
# transmittance the resolve carries through the fragment list, where each
# fragment attenuates exactly the samples its mask owns. The walk below is that
# resolve, replayed in Python from the same compact arrays the kernel reads.
#
# It mirrors ``raster_first_shade``'s ``aa_grp`` + run-mode path for MATTE
# OPAQUE geometry, which is what every case here builds: ``mat_alpha == 1``,
# ``trans_share == 0``, no reflection lobe (the default material's metalness is
# the ``< 0`` unlit sentinel, so ``refl_max == 0`` and the walk never takes an
# early reflection break), and ``weight`` stays at one. ``--verify`` proves the
# replay against the kernel's own per-fragment dump rather than asserting it.


def _popcount(m):
    return bin(m & _AA_MASK_ALL).count("1")


def _host_run_scan(j0, sids, faces, msks, covs, bez):
    """``_aa_run_scan`` on the host: ``(E, U, end)`` for the run at ``j0``."""
    sid0, face0 = sids[j0], faces[j0]
    n = len(sids)
    E, U, j, cnt = 0.0, 0, j0, 0
    while j < n and cnt < _AA_MAX_RUN_SCAN:
        if bez[j] or sids[j] != sid0 or faces[j] != face0:
            break
        E += covs[j]
        U |= msks[j] & _AA_MASK_ALL
        j += 1
        cnt += 1
    return E, U, j


def _host_redistribute(svis, run_U, resid):
    """Rule B's run-end step (``_run_redistribute``)."""
    if resid <= 0.0:
        return
    free = [s for s in range(_AA_NUM_SAMPLES) if not ((run_U >> s) & 1)]
    tot = sum(svis[s] for s in free)
    if tot > 1e-12:
        sc = max(1.0 - resid / tot, 0.0)
        for s in free:
            svis[s] *= sc


def _replay(sids, faces, msks, covs, bez, rule_b, consult_e=False, one_sheet=False):
    """Walk one pixel's fragment list exactly as the resolve does.

    Returns ``(ink, occ, effs)``: the coverage the geometry CLAIMS (the sum of
    the per-fragment ``eff`` that reaches the accumulator, i.e. what the pixel
    is painted with), the coverage it OCCLUDES (``1 - mean(svis)``, i.e. how
    much of the background it hides), and the per-fragment ``eff`` sequence for
    ``--verify``. The two coverages coincide except where rule B's
    redistribution moves transmittance the claim never accounted for.

    ``consult_e`` is ss6.2's counterfactual: under a full mask union take
    ``corr = E`` (``Q == 1`` there) instead of short-circuiting to 1, with the
    1e-3 dust band that keeps a genuine interior tiling bit-identical. It was
    measured and reverted on its own; it is kept here as a REPLAY variant --
    costing no kernel and no cache entry -- because ss6.4 predicts it is only
    interesting in combination with ``ALGAN_MESH_ID=1``, where a run is the
    whole sheet and ``E`` is therefore the sheet's exact area.

    ``one_sheet`` is a DIAGNOSTIC, not a proposal: once one facing group has
    committed ink, every fragment of the other commits nothing. On a closed
    convex solid that is exactly right -- both sheets project to the same
    silhouette, so the mesh's coverage is its near sheet's, never the sum --
    and it isolates the one mechanism the dumps show. A run's exact-area
    correction ``corr < 1`` scales the OCCLUSION write as well as the claim, so
    the samples the near sheet owns keep a residual transmittance standing for
    the part of the pixel it does not cover. That residue lies OUTSIDE the
    silhouette, but it has no position, so when the far sheet of the same solid
    arrives owning the same samples it claims the residue as if it were
    background showing through -- and it claims it uncorrected, because ``svis``
    is no longer uniform and its own run cannot engage. Measured on one
    ``cylinder`` pixel: near sheet claims 0.2396 (exact, corr 0.9583), far sheet
    adds 0.0104, pixel lands on 0.2500 = 2/8 against a true 0.2394.

    It is a diagnostic because the rule it applies is only sound for a CLOSED
    CONVEX mesh. Concave geometry can show several front-facing sheets with
    different footprints, so the general statement is "a mesh's coverage in a
    pixel is the UNION of its sheets, not their sum" -- which needs the mesh
    identity of ss2.2 to even ask, and a per-mesh accumulator to answer.
    Interleaving makes the suppression approximate (a near-sheet run resumed
    after a suppressed far-sheet fragment can miss its rescan), which is
    tolerable here only because ``split`` is measured at 0.02%.
    """
    n = len(sids)
    N = _AA_NUM_SAMPLES
    svis = [1.0] * N
    effs = []
    ink = 0.0
    run_end = 0
    run_mode = 0
    run_corr = 1.0
    run_pscale = 0.0
    run_vstart = 0.0
    run_claimed = 0.0
    run_U = 0
    run_resid = 0.0
    run_pending = 0
    first_face = None
    for q1 in range(n):
        raw = msks[q1]
        is_bez = bez[q1]
        sliver = (raw & _AA_SLIVER_BIT) != 0
        msk = raw & _AA_MASK_ALL
        cov = covs[q1]
        if one_sheet and first_face is not None and faces[q1] != first_face:
            effs.append(0.0)
            continue

        # -- run scan (the kernel's lookahead at a run's first fragment) ----
        if (not is_bez) and q1 >= run_end:
            if rule_b and run_pending:
                _host_redistribute(svis, run_U, run_resid)
                run_pending = 0
                run_resid = 0.0
            run_mode = 0
            run_end = q1 + 1
            if msk != _AA_MASK_ALL:
                v0 = svis[0]
                uniform = v0 > 0.0 and all(v == v0 for v in svis[1:])
                if uniform:
                    rE, rU, rj = _host_run_scan(q1, sids, faces, msks, covs, bez)
                    run_end = rj
                    if rU == _AA_MASK_ALL:
                        run_mode = 1
                        run_corr = 1.0
                        if consult_e and abs(1.0 - rE) > 1e-3:
                            run_corr = min(rE, 1.0)
                    elif rU == 0:
                        run_mode = 2
                        run_pscale = min(rE, 1.0) / max(rE, 1e-9)
                        run_vstart = v0
                        run_claimed = 0.0
                    else:
                        run_mode = 1
                        run_corr = min(rE, 1.0) / (_popcount(rU) / N)
                    if rule_b and run_mode == 1:
                        run_U = rU
                        run_resid = 0.0
                        run_pending = 1

        # -- _coverage_slots, run representation ---------------------------
        if is_bez or sliver:
            slots = [1.0] * N
            dens = cov
        else:
            slots = [1.0 if (msk >> s) & 1 else 0.0 for s in range(N)]
            dens = 1.0
        eff = sum(slots[s] * svis[s] for s in range(N)) / N * dens

        cfac = 1.0
        run_pd = 0.0
        if (not is_bez) and q1 < run_end:
            if run_mode == 1:
                cfac = run_corr
                eff *= run_corr
            elif run_mode == 2:
                run_pd = run_pscale * cov
                eff = run_pd * run_vstart
                dens = run_pd / max(1.0 - run_claimed, 1e-6)
                slots = [1.0] * N
        if eff <= MIN_ALPHA:
            effs.append(0.0)
            continue
        effs.append(eff)
        ink += eff
        if first_face is None:
            first_face = faces[q1]

        # -- _run_svis_write, trans_share == 0, mat_alpha == 1 -------------
        a_s = dens
        for s in range(N):
            ak = cfac * a_s * slots[s]
            fct = 1.0 - ak
            if rule_b:
                if fct < 0.0:
                    run_resid -= fct * svis[s]
                    fct = 0.0
            else:
                fct = max(fct, 0.0)
            svis[s] *= fct
        if run_pd > 0.0:
            run_claimed += a_s * (1.0 - run_claimed)
        if sum(svis) / N < MIN_WEIGHT:
            break
    if rule_b and run_pending:
        _host_redistribute(svis, run_U, run_resid)
    return ink, 1.0 - sum(svis) / N, effs


# How far the two sheets of a closed solid may disagree before the pixel's
# reference is thrown out. They tile the same silhouette, so agreement is exact
# up to float summation over a handful of areas.
_SHEET_TOL = 1e-3


def _exact_coverage(faces, covs):
    """The pixel's TRUE coverage by the object's footprint, from exact areas.

    Returns ``(truth, ok)``. Every case here is one closed convex opaque solid
    or a single flat sheet, so each SHEET's fragments tile the footprint exactly
    -- the near and far sheets of a closed convex surface project to the same
    silhouette. Summing one sheet's exact clipped areas is therefore the exact
    answer, with no supersampled reference and no fitted model.

    Sheets are separated by the facing bit, and the reference VALIDATES that
    separation rather than assuming it: for a closed solid both sheets must sum
    to the same area, and for an open one the single sheet must not exceed the
    pixel. A pixel that fails is dropped from the statistics and counted.

    That gate is not hypothetical. ``Polyhedron`` builds each face from a
    hardcoded index list, and those lists are not consistently oriented --
    measured on this repo, 12 of an ``Icosahedron``'s 20 faces, 2 of 4 on a
    ``Tetrahedron``, 2 of 8 on an ``Octahedron`` and 3 of 12 on a
    ``Dodecahedron`` wind inward, against 0 of 6 on a ``Cube``. The projected
    winding sign IS ``_AA_BACKFACE_BIT``, so on those solids the bit does not
    name a sheet, and a near and a far face can land in one group (measured: a
    pixel whose "front" group sums to 1.98 because it holds both sheets, while
    the true sheets tile to 1.0000 each). See the note in ``main``.
    """
    front = sum(c for f, c in zip(faces, covs) if f == 0)
    back = sum(c for f, c in zip(faces, covs) if f == 1)
    if front > 0.0 and back > 0.0:
        if abs(front - back) > _SHEET_TOL:
            return 0.0, False
        return min(0.5 * (front + back), 1.0), True
    total = max(front, back)
    if total > 1.0 + _SHEET_TOL:
        return 0.0, False
    return min(total, 1.0), True


class _Svis:
    """Accumulator for the ownership-vs-magnitude statistics (ss6.3).

    Every quantity is per SILHOUETTE pixel (partial true coverage), because a
    fully covered pixel has no coverage question and an empty one carries no
    ink. Three coverages are compared against the exact area:

    ``actual``
        What the resolve paints: the replayed walk's summed ``eff``.
    ``own``
        What SAMPLE OWNERSHIP alone says: ``popcount(union of every fragment's
        mask) / N``. This is the pixel's coverage with all magnitude
        information discarded -- the answer an 8-sample set can express and
        nothing finer.
    ``consult-E``
        ss6.2's counterfactual, replayed rather than compiled.

    ``on-lattice`` is the sharpest single number: the share of silhouette
    pixels whose painted coverage is (to 1e-4) an exact multiple of 1/N. A
    pixel on the lattice got its answer from WHICH samples were claimed; a
    pixel off it got a magnitude correction applied.
    """

    def __init__(self):
        self.n = 0
        self.err_actual = 0.0
        self.err_own = 0.0
        self.err_ce = 0.0
        self.err_1s = 0.0
        self.sig_actual = 0.0
        self.sig_own = 0.0
        self.gap_actual_own = 0.0
        self.on_lattice = 0
        self.unreferenced = 0
        self.covered = 0
        self.claim_occ_max = 0.0
        self.by_verdict = Counter()
        self.err_by_verdict = Counter()

    def add(self, verdict, truth, ok, actual, occ, own, ce, one_sheet):
        self.covered += 1
        self.claim_occ_max = max(self.claim_occ_max, abs(actual - occ))
        if not ok:
            self.unreferenced += 1
            return
        if not (_SILH_LO < truth < _SILH_HI):
            return
        self.n += 1
        self.err_actual += abs(actual - truth)
        self.err_own += abs(own - truth)
        self.err_ce += abs(ce - truth)
        self.err_1s += abs(one_sheet - truth)
        self.sig_actual += actual - truth
        self.sig_own += own - truth
        self.gap_actual_own += abs(actual - own)
        q = actual * _AA_NUM_SAMPLES
        if abs(q - round(q)) <= 1e-4:
            self.on_lattice += 1
        self.by_verdict[verdict] += 1
        self.err_by_verdict[verdict] += abs(actual - truth)


def _measure(build, settings, capture=None):
    """Render once, replaying the run rule and the resolve for every pixel.

    ``capture`` is a set of ``(px, py)`` kernel-space pixels whose fragment
    lists and replayed ``eff`` sequence are kept for ``--verify``.
    """
    stats = Counter()
    modes = []
    frag_hist = Counter()
    lost_total = [0.0]
    uf_hist = Counter()
    svis_stats = _Svis()
    silhouettes = []
    captured = {}

    original = rp.prepare_sparse_raster_coverage

    def spy(*args, **kwargs):
        coverage = original(*args, **kwargs)
        if coverage is None:
            return coverage
        aa_tri = int(coverage["aa_tri"])
        modes.append((aa_tri, int(coverage["aa_grp"])))
        # 4 is the redistribute rule, 3 the clamp rule (see _tri_run_rule_b).
        rule_b = aa_tri == 4
        n_cov = int(coverage["num_covered"])
        if n_cov <= 0:
            return coverage
        width = int(kwargs["width"] if "width" in kwargs else args[13])
        height = int(kwargs["height"] if "height" in kwargs else args[14])
        ppf = width * height
        offs = coverage["run_offsets"][: n_cov + 1].detach().cpu().tolist()
        ref = coverage["frag_ref"].detach().cpu()
        cov = coverage["frag_cov"].detach().cpu()
        msk = coverage["frag_msk"].detach().cpu()
        pix = coverage["covered_idx"].detach().cpu().tolist()
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
            sids, faces, ms, cs, bz = [], [], [], [], []
            for j in range(lo, hi):
                r = refs[j]
                if r >= 0:
                    sids.append(int(obj_row[r]))
                    bz.append(False)
                else:
                    sids.append(-1 - ((-r - 1) >> 8))
                    bz.append(True)
                faces.append(1 if (msks[j] & _AA_BACKFACE_BIT) else 0)
                ms.append(msks[j])
                cs.append(covs[j])
            verdict, nfrag, lost = _classify(sids, faces, ms, cs)
            stats[verdict] += 1

            # -- the ss6.3 measurement ---------------------------------------
            truth, ok = _exact_coverage(faces, cs)
            actual, occ, effs = _replay(sids, faces, ms, cs, bz, rule_b)
            ce, _occ_ce, _e = _replay(sids, faces, ms, cs, bz, rule_b, consult_e=True)
            one_sheet, _occ_1s, _e1 = _replay(
                sids, faces, ms, cs, bz, rule_b, one_sheet=True
            )
            union = 0
            for m in ms:
                union |= m & _AA_MASK_ALL
            own = _popcount(union) / _AA_NUM_SAMPLES
            svis_stats.add(verdict, truth, ok, actual, occ, own, ce, one_sheet)
            p = pix[i] % ppf
            py, px = p // width, p % width
            if ok and _SILH_LO < truth < _SILH_HI:
                silhouettes.append((px, py, truth, actual, own))
            if capture and (px, py) in capture:
                captured[(px, py)] = (effs, truth, actual, own)
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
    return (
        stats,
        frag_hist,
        lost_total[0],
        modes,
        uf_hist,
        svis_stats,
        silhouettes,
        captured,
    )


def _verify(build, settings, silhouettes, limit):
    """Prove the host replay against the kernel's own per-fragment dump.

    ``ALGAN_AA_DUMP`` makes both walk kernels write one row per fragment they
    process at a requested pixel (``DESIGN_analytic_aa_v2.md`` ss7.1). Re-render
    with the dump aimed at a silhouette pixel and diff column 10 (``eff``)
    against the replay's, one render per pixel. Without this the replay is a
    plausible re-reading of the kernel rather than a measured one -- and every
    number in the ss6.3 table is only worth what this check says it is.
    """
    if not silhouettes:
        return None
    # Spread the probes over the silhouette rather than taking neighbours.
    step = max(len(silhouettes) // limit, 1)
    probes = [silhouettes[i * step] for i in range(min(limit, len(silhouettes)))]
    worst = 0.0
    rows_seen = 0
    for px, py, _truth, _actual, _own in probes:
        rp.LAST_AA_DUMP.clear()
        os.environ["ALGAN_AA_DUMP"] = f"{px},{py},0"
        try:
            # The engine prints every dumped row; the harness wants the diff,
            # not a dozen 24-column rows per probe.
            with contextlib.redirect_stdout(io.StringIO()):
                _s, _f, _l, _m, _u, _sv, _sl, captured = _measure(
                    build, settings, capture={(px, py)}
                )
        finally:
            del os.environ["ALGAN_AA_DUMP"]
        rows = rp.LAST_AA_DUMP.get("resolve")
        if rows is None or not len(rows) or (px, py) not in captured:
            continue
        effs = captured[(px, py)][0]
        # Fragment rows only (q >= 0); the kernel emits one per fragment it
        # processes, an eff-skip included (note 1), which the replay records
        # as a zero. It stops at the MIN_WEIGHT break, so compare the prefix.
        frag = [r for r in rows if r[0] >= 0]
        for r in frag:
            q = int(r[0])
            if q >= len(effs):
                continue
            rows_seen += 1
            worst = max(worst, abs(float(r[10]) - effs[q]))
    return worst, rows_seen, len(probes)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--res", choices=sorted(RESOLUTIONS), default="md")
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument(
        "--verify",
        type=int,
        default=0,
        metavar="N",
        help="probe N silhouette pixels per case with ALGAN_AA_DUMP and diff "
        "the host replay against the kernel's own per-fragment eff",
    )
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
    svis_rows = []
    for name, build in cases.items():
        (
            stats,
            frag_hist,
            lost,
            modes,
            uf_hist,
            svis_stats,
            silhouettes,
            _cap,
        ) = _measure(build, settings)
        svis_rows.append((name, build, svis_stats, silhouettes))
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

    # -- ss6.3: ownership vs magnitude --------------------------------------
    head2 = (
        f"\n{'case':22s} {'silh px':>8s} {'|actual-E|':>11s} {'|own-E|':>9s} "
        f"{'|actual-own|':>13s} {'|cE-E|':>8s} {'|1sheet-E|':>11s} "
        f"{'on-lattice':>11s} {'signed':>8s}"
    )
    print(head2)
    print("-" * (len(head2) - 1))
    for name, _build, sv, _silh in svis_rows:
        if not sv.n:
            print(f"{name:22s} {'(none)':>8s}")
            continue
        n = sv.n
        print(
            f"{name:22s} {n:8d} {sv.err_actual / n:11.4f} "
            f"{sv.err_own / n:9.4f} {sv.gap_actual_own / n:13.4f} "
            f"{sv.err_ce / n:8.4f} {sv.err_1s / n:11.4f} "
            f"{100.0 * sv.on_lattice / n:10.1f}% {sv.sig_actual / n:+8.4f}"
        )
        worst = " ".join(
            f"{k}:{sv.err_by_verdict[k] / max(sv.by_verdict[k], 1):.3f}"
            f"({sv.by_verdict[k]})"
            for k in ("full", "corrected", "union-full", "split", "capped")
            if sv.by_verdict[k]
        )
        print(f"{'':22s} mean |actual-E| by verdict  {worst}")
        print(
            f"{'':22s} no trustworthy reference (sheets disagree) "
            f"{sv.unreferenced}/{sv.covered}   "
            f"claim-vs-occlusion max {sv.claim_occ_max:.2e}"
        )
    print(
        "\nE is the EXACT area of (footprint n pixel), summed from one sheet's\n"
        "clipped areas -- no supersampled reference, and the other sheet has to\n"
        "agree or the pixel is dropped. 'own' is popcount(union of every\n"
        "fragment mask)/N: the pixel's coverage with all magnitude information\n"
        "discarded. 'on-lattice' is the share of silhouette pixels whose\n"
        "painted coverage is an exact multiple of 1/N.\n\n"
        "|actual-own| near zero and on-lattice near 100% => the pixel's\n"
        "coverage is decided by WHICH of the N samples got claimed, and no\n"
        "magnitude correction survives to it. The flat quad is the control that\n"
        "makes that reading sound: one sheet, no far side, and there the\n"
        "machinery removes 95% of the ownership error.\n\n"
        "'1sheet' suppresses the far sheet of the same solid (see _replay). It\n"
        "is a diagnostic, and the size of the gap between it and 'actual' is\n"
        "what a mesh-level union rule would be worth."
    )

    if args.verify:
        print("\nreplay vs kernel dump (ALGAN_AA_DUMP):")
        for name, build, _sv, silh in svis_rows:
            res = _verify(build, settings, silh, args.verify)
            if res is None:
                print(f"  {name:22s} (no silhouette pixels)")
                continue
            worst, rows, probes = res
            tag = "PASS" if worst < 2e-5 else "FAIL"
            print(
                f"  [{tag}] {name:22s} worst |eff| diff {worst:.2e} over "
                f"{rows} fragment rows at {probes} pixels"
            )


if __name__ == "__main__":
    main()
