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
    icosahedron         898      0.0258   0.0407        0.0174       59.5%
    cylinder           2307      0.0260   0.0367        0.0116       72.5%
    cylinder (256x2)   2139      0.0211   0.0329        0.0128       70.6%
    sphere (192x96)    2628      0.0383   0.0408        0.0047       90.8%

``own`` is ``popcount(union of every fragment mask)/N``: the pixel's coverage
with all magnitude information discarded. ``on-lattice`` is the share of
silhouette pixels whose painted coverage is an exact multiple of 1/N.

WHAT THE SHIPPED WALK DOES. On the flat control the machinery works as designed
-- error 0.0020 against 0.0390 for ownership alone, so 95% of the sample
quantization is removed and only 7.9% of pixels land on the lattice. On a diced
closed mesh it is neutralized: the sphere's painted coverage sits 0.0047 from
the pure-ownership answer and 91% of its silhouette pixels land exactly on
eighths. The signed error is positive in every case -- dilation, which is what
``_aa_line_check`` reads as ink wobble.

Two mechanisms, separated by the by-verdict line:

  * ``full`` (52% of the sphere's silhouette pixels, mean error 0.042). ONE
    fragment owns all N samples while covering less than the whole pixel, so
    the run scan never starts (v2 ss4.2 gates the lookahead on a partial mask)
    and the pixel is painted at 1.0. Its sheet's exact area sits unread.
  * The FAR SHEET re-claim. A run's ``corr < 1`` scales the occlusion write as
    well as the claim, so the samples the near sheet owns keep a residual
    transmittance -- standing for the part of the pixel the sheet does not
    cover, which at a silhouette lies OUTSIDE the mesh entirely. The residue has
    no position, so the far sheet of the same solid claims it, uncorrected
    (``svis`` is no longer uniform, so its own run cannot engage). The
    ``1sheet`` column suppresses it: 0.0250 -> 0.0041 on the cube (84% of the
    error), but only 0.0383 -> 0.0346 on the sphere, where ``full`` dominates.

IT IS NOT A SAMPLING LIMIT -- IT IS A DISCARDED MAGNITUDE. Landing on the
pure-ownership answer looks like a representation ceiling, and doubling the
sample count behaves like one (see below). It is not: ``own`` is only a floor
for a scheme with no magnitude at all, and this architecture HAS one -- the run
correction produces off-lattice coverage whenever it is allowed to run. Letting
it run on full-mask pixels too (``cF``, one relaxed gate, no extra samples, no
extra kernel work on the interior hot path) recovers most of the error:

    |actual-E|         shipped   16 samples   cF (relaxed gate)
    quad (control)      0.0020       0.0028              0.0000
    cube                0.0250            -              0.0214
    icosahedron         0.0258            -              0.0120
    cylinder            0.0260       0.0126              0.0030   -88%
    cylinder (256x2)    0.0211            -              0.0030   -86%
    sphere (192x96)     0.0383       0.0236              0.0060   -84%

So the ordering of levers is: fix the gate first, and only then ask whether the
residue is worth more samples. The two flat solids are the exception -- their
``cF`` barely moves, because their error is the far-sheet re-claim, which needs
a mesh-level union rule instead (``DESIGN_mesh_identity.md`` ss6.3).

Neither is reachable by the run rule as it stands: the first is excluded by its
own gate, and the second needs to know that two sheets belong to ONE mesh --
which is what ``DESIGN_mesh_identity.md`` ss2.2 declares and no consumer reads.

WHAT IT ARBITRATES. ``DESIGN_mesh_identity.md`` ss4.5 wanted rendered coverage
against an exact reference to decide ``ALGAN_MESH_ID``, and said this harness
could not supply it. It can now, and on the scored metric the answer is NEUTRAL.
Mean |actual-E| over silhouette pixels, ``--res md``, ``ALGAN_MESH_ID=0`` ->
``1``: Cube 0.0250 -> 0.0248, Icosahedron 0.0258 -> 0.0256 (0.0264 -> 0.0262
with ``ALGAN_POLYHEDRON_WINDING=1``), quad and every single-solid ``Surface``
case unmoved -- a ``Surface`` is already one merged member, so its ``sid`` does
not move. Nothing regresses and nothing gains beyond noise.

THE SCORED COLUMN IS THE WRONG INSTRUMENT FOR A PACKED GRID, which is why
``--mesh-ab`` exists. ``_exact_coverage`` has to DROP a pixel whose facing group
holds two sheets, and on a pack of spheres those drops are concentrated exactly
where two spheres overlap -- the population MESH_ID is there to fix. So
``--mesh-ab`` differences painted coverage between the two settings per pixel
instead: no reference, so it sees every covered pixel. Measured, ``--res md``:

    case                 covered px   moved   max |d|   mean off-on
    quad (control)            33438       0    0.0000       +0.0000
    cube                      39914      17    0.0885       +0.0001
    icosahedron               46220     235    0.4968       -0.2098
    cylinder (default)        43124       0    0.0000       +0.0000
    cylinder (256x2)          43228       0    0.0000       +0.0000
    sphere (192x96)           27734       0    0.0000       +0.0000
    packed 4x4 (apart)        43560       0    0.0000       +0.0000
    packed 4x4 (overlap)      36224      18    0.2002       +0.0539

The packed prediction holds in sign and mechanism and is small in population:
18 of 36224 pixels, with ``off - on`` POSITIVE -- MESH_ID=0 paints more, the
over-claim that happens when one id for the whole pack lets a run carry across
two spheres until their masks OR to a full union. The ``apart`` control moves
ZERO pixels, which is what makes that reading sound: the effect is the packing,
not the batching. The Icosahedron's 235 moved pixels are mostly the winding
defect, not MESH_ID -- with ``ALGAN_POLYHEDRON_WINDING=1`` they fall to 11 at
mean |d| 0.024.

Getting there needed a DEFECT fixed. Both packed cases were byte-identical
across MESH_ID at first, because a packed grid is diced logical PN and
``_dice_logical_pn`` built its patch->surface map from the per-member
``_rt_obj_counts`` alone. A lone packed primitive is ONE member covering every
sphere, so the pack diced to a single id and the ``mesh_ids``
``Surface.get_render_primitives`` stamps were resolved at construction and then
thrown away. The dice now consults the declaration first, like the flat path.

That is a CORRECTION of a number this docstring carried earlier, and the reason
is worth keeping: an intermediate version of ``_exact_coverage`` accepted a
mis-wound pixel whose two sheets had landed in ONE facing group, reporting
double its true coverage. The Icosahedron's error read 0.0492 and MESH_ID
appeared to halve it. Neither survived a sound reference. The lesson is the
plain one -- an arbiter needs its own validity check before its verdicts are
worth anything, which is why the drop count is now printed beside every row.

A SIDE FINDING, load-bearing for mesh identity. ``Polyhedron`` builds each face
from a hardcoded index list and those lists are not consistently oriented:
measured, 12 of an ``Icosahedron``'s 20 faces wind inward, 2 of 4 on a
``Tetrahedron``, 2 of 8 on an ``Octahedron``, 3 of 12 on a ``Dodecahedron``, 0
of 6 on a ``Cube``. The projected winding sign IS ``_AA_BACKFACE_BIT``, so on
those solids the facing bit does not name a sheet, and this harness drops the
pixels where that shows: 960 of the icosahedron's 46220 covered pixels have one
facing group holding BOTH sheets, against 4 with ``ALGAN_POLYHEDRON_WINDING=1``
-- which is the measurement that the orientation pass works. (The obvious follow-on guess, that the winding is why
``ALGAN_MESH_ID=1`` regressed an Icosahedron under the old per-fragment metric,
is measured above and is WRONG: with the winding fixed, MESH_ID is still
neutral.)

Run:  <venv-python> benchmarks/_aa_run_gate_check.py [--res md|ld|hd]
                                                     [--cases ...] [--verify N]
                                                     [--mesh-ab]
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

# Benchmarks must never be measured inside a warm daemon: it keeps adaptive
# renderer state (the memory model's batch-size fit) across runs, so one
# benchmark would be timed against whatever ran before it.
os.environ.setdefault("ALGAN_USE_DAEMON", "0")

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
from algan.rendering.raytracing import settings as rt_settings  # noqa: E402
from algan.rendering.raytracing.raster_taichi import (  # noqa: E402
    _AA_BACKFACE_BIT,
    _AA_MASK_ALL,
    _AA_MAX_RUN_SCAN,
    _AA_NUM_SAMPLES,
    _AA_ONE_MESH_BIT,
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

    def _pack(spacing, depth):
        # ``batch_mobs`` flattens several INDEPENDENT Sphere grids into one
        # packed grid, which reaches the renderer as a SINGLE
        # LogicalPNTrianglePrimitive -- one collection member covering every
        # sphere. That is the end DESIGN_mesh_identity.md ss2.2 fixes in the
        # other direction: with MESH_ID off the whole pack is one surface, so
        # _aa_run_scan may carry a run ACROSS two spheres and sum coverage over
        # objects that merely overlap on screen; with it on, surface.py's
        # per-grid ``mesh_ids`` break the run at each sphere boundary.
        from algan.utils.mob_utils import batch_mobs

        dots = [
            Sphere(
                radius=0.28,
                resolution=(24, 12),
                color=WHITE,
                add_to_scene=False,
            ).move_to(
                RIGHT * ((i % 4) - 1.5) * spacing
                + UP * ((i // 4) - 1.5) * spacing
                + OUT * (depth if (i % 2) else -depth)
            )
            for i in range(16)
        ]
        batch_mobs(dots, add_to_scene=True).spawn()

    def line_check_cyl():
        # THE RECONCILIATION CASE. _aa_line_check's own thin frame-spanning
        # prism at 33 deg, so its ink-wobble number and this harness's coverage
        # error describe the SAME pixels. Without it the two instruments were
        # measuring different geometry -- this harness's fat Cylinder against
        # the line check's 0.045-radius rod -- and the relaxed run gate came
        # back -70% here while the line check saw no movement at all. Two
        # instruments that disagree are one instrument.
        import sys as _sys

        _sys.path.insert(0, str(REPO / "benchmarks"))
        import _aa_line_check as _alc

        from algan.scene_manager import SceneManager

        _alc.build_line("cyl", 33.0, SceneManager.instance().current_scene)

    def line_check_cyl_fine():
        """``resolution=(256, 2)`` on a 0.045-radius rod.

        Facets far below a pixel and nearly edge-on, so almost every pixel is
        silhouette BOUNDARY rather than interior. This is the case the one-mesh
        rule regresses on, and ss6.6 says why.
        """
        import sys as _sys

        _sys.path.insert(0, str(REPO / "benchmarks"))
        import _aa_line_check as _alc

        from algan.scene_manager import SceneManager

        _alc.build_line("cyl_fine", 33.0, SceneManager.instance().current_scene)

    def line_check_quad():
        """The same, for the flat control the line check uses."""
        import sys as _sys

        _sys.path.insert(0, str(REPO / "benchmarks"))
        import _aa_line_check as _alc

        from algan.scene_manager import SceneManager

        _alc.build_line("quad", 33.0, SceneManager.instance().current_scene)

    def packed_apart():
        # The CONTROL for the packed pair: same construction, spaced so no two
        # footprints can touch (centres 0.75 apart, radii summing to 0.56). Every
        # pixel is covered by at most one sphere, so a cross-sphere run is
        # geometrically impossible and MESH_ID has nothing to separate. It is
        # what makes the overlapping row below readable: whatever moves there
        # and not here is the packing, not the batching.
        _pack(0.75, 0.0)

    def packed_overlap():
        # THE case ss4.5 asks for. Centres 0.45 apart with radius 0.28 (sum
        # 0.56), so adjacent footprints DO overlap, and alternating depth so one
        # sphere of each overlapping pair is genuinely in front of the other.
        _pack(0.45, 0.30)

    return {
        "quad (flat control)": quad,
        "cube (flat)": cube,
        "icosahedron (flat)": polyhedron,
        "cylinder (default)": cylinder,
        "cylinder (256x2)": cylinder_fine,
        "sphere (192x96)": sphere_fine,
        "line-check cyl (33deg)": line_check_cyl,
        "line-check cylfine (33d)": line_check_cyl_fine,
        "line-check quad (33deg)": line_check_quad,
        "packed 4x4 (apart)": packed_apart,
        "packed 4x4 (overlap)": packed_overlap,
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


def _replay(
    sids,
    faces,
    msks,
    covs,
    bez,
    rule_b,
    consult_e=False,
    one_sheet=False,
    consult_full=False,
    one_sheet_gated=False,
    mesh_cap=False,
    mesh_cap_gated=False,
    caps=None,
):
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

    ``consult_full`` lets the ``full`` verdict reach ``consult_e`` at all. v2
    ss4.2 gates the run lookahead on a PARTIAL first mask, so a pixel whose
    first fragment owns every sample never scans, never computes ``E``, and is
    painted at 1.0 however little of it the geometry covers -- 52% of a fine
    ``Sphere``'s silhouette pixels, and the largest single contributor measured.
    The relaxed gate is "partial mask, OR a full mask whose exact area is not
    within dust of the whole pixel", which leaves the interior hot path exactly
    as it is, since there ``cov`` IS 1 to within dust.

    Deliberately scoped to the RUN and not to the fragment. A full-mask fragment
    owns every sample, so by the fill rule the rest of its sheet in that pixel
    owns none -- they are empty-mask area donors, and their area is real. Taking
    the magnitude from that one fragment's ``cov`` would drop them; taking it
    from the run's ``E`` does not, and the difference is most of the result.
    Both were measured; run scope is the column reported:

        |cF-E|        fragment scope   run scope
        quad                  0.0000      0.0000
        cube                  0.0214      0.0214    (flat: no donors)
        icosahedron           0.0369      0.0369    (flat: no donors)
        cylinder              0.0050      0.0030
        cylinder (256x2)      0.0063      0.0030
        sphere (192x96)       0.0255      0.0060

    Measured here rather than built because it is a narrower cousin of something
    v2 ss21.3 already rejected -- reconciling EVERY fragment's magnitude against
    its exact area put 5920 notches into a mesh. A full mask is exactly the case
    where that cannot happen: the fragment owning all N samples is alone in its
    sheet's sample partition, so there is no neighbour to disagree with. "Cannot
    happen by this argument" is not "cannot happen", and shipping it means
    ``_analytic_aa_fillrule_check`` and a look at the diff videos, not this
    column.
    """
    n = len(sids)
    N = _AA_NUM_SAMPLES
    svis = [1.0] * N
    effs = []
    ink = 0.0
    # ``mesh_cap`` is ss6.6's proposed successor to ``one_sheet``: instead of
    # SUPPRESSING the far sheet, cap the mesh's TOTAL claim at the larger of the
    # two sheets' exact areas. Well inside a silhouette the two are equal, so it
    # degenerates to suppression and keeps that win; at the boundary -- where
    # the near sheet's projected area shrinks toward zero while the footprint
    # does not -- the larger is the right answer and suppression under-covers.
    cap_target = 2.0
    mesh_ink = 0.0
    if mesh_cap:
        if caps is not None and len(caps):
            # Read the ceiling the kernel actually reads. Re-deriving it here
            # in float64 from the same float32 areas lands a few 1e-5 away from
            # the host's float32 sum, and since the rule CLIPS at that value the
            # difference shows up whole in the clipped fragment -- measured, it
            # failed --verify at 6e-4 on the many-fragment cases while passing
            # everywhere else. A replay models the kernel; the ceiling is one of
            # the kernel's inputs, not something to reconstruct.
            cap_target = caps[0]
        else:
            front = sum(c for f, c, z in zip(faces, covs, bez) if not z and f == 0)
            back = sum(c for f, c, z in zip(faces, covs, bez) if not z and f == 1)
            cap_target = min(max(front, back), 1.0)
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
        # ``one_sheet_gated`` is the SHIPPABLE form (aa_grp 3): the kernel
        # applies this only where the host flagged the pixel as a single opaque
        # surface, so the replay must read the same bit or --verify compares two
        # different rules. Ungated, it stays the ss6.3 diagnostic that measured
        # what the rule would be worth.
        if (
            one_sheet
            and first_face is not None
            and faces[q1] != first_face
            and ((not one_sheet_gated) or (raw & _AA_ONE_MESH_BIT))
        ):
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
            # The kernel gates the lookahead on a PARTIAL mask so the hot path
            # (an interior pixel, one full-mask fragment) never pays for it.
            # ``consult_full`` relaxes that to "partial mask, or a full mask
            # whose exact area is not within dust of the whole pixel", which
            # leaves the hot path free and is the only gate a shipped version
            # would need.
            scan = msk != _AA_MASK_ALL
            if consult_full and not scan:
                scan = cov < 1.0 - 1e-3
            if scan:
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
        # ``mesh_cap_gated`` is the SHIPPABLE form (aa_grp 3): the kernel applies
        # the ceiling only where the host flagged the pixel, so the replay must
        # read the same bit or --verify compares two rules. ``mesh_ink`` mirrors
        # the kernel's own accumulator, which counts triangle fragments only.
        if (
            mesh_cap
            and (not is_bez)
            and run_mode != 2
            and ((not mesh_cap_gated) or (raw & _AA_ONE_MESH_BIT))
        ):
            room = cap_target - mesh_ink
            if eff > room:
                eff = max(room, 0.0)
        if eff <= MIN_ALPHA:
            effs.append(0.0)
            continue
        effs.append(eff)
        ink += eff
        if not is_bez:
            mesh_ink += eff
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

#: How far below full a fully-covered pixel must be painted to count as a
#: notch. Well above the exact-area arithmetic's float dust.
_NOTCH_TOL = 1e-3


def _exact_coverage(faces, msks, covs):
    """The pixel's TRUE coverage by the object's footprint, from exact areas.

    Returns ``(truth, ok)``. Every case here is one closed convex opaque solid
    or a single flat sheet, so each SHEET's fragments tile the footprint exactly
    -- the near and far sheets of a closed convex surface project to the same
    silhouette. Summing one sheet's exact clipped areas is therefore the exact
    answer, with no supersampled reference and no fitted model.

    Sheets are separated by the facing bit, and the reference VALIDATES that
    separation rather than assuming it, with the fill rule's own property:
    WITHIN one sheet the masks PARTITION the sub-pixel samples, so no sample may
    be claimed twice. A facing group whose masks overlap is holding more than
    one sheet and the pixel is dropped.

    That test rather than the obvious ones, because the obvious ones are both
    wrong, and each was wrong in a way that produced a plausible-looking table:

      * "The two groups must agree" alone leaves a hole a mis-wound solid drives
        straight through -- with BOTH sheets in one group and the other empty, a
        pixel of true coverage 0.3 reports 0.6 and passes, since 0.6 is under
        the one-pixel bound the empty-group branch tests. This published a wrong
        Icosahedron number before it was found.
      * "A closed solid must show both groups" over-corrects and silently
        deletes the population the whole measurement is about. A pixel whose
        near sheet ends in a FULL-mask fragment is truncated by the emission
        right there (``prepare_sparse_raster_coverage``'s opaque prefix), so its
        far sheet never reaches the resolve -- and those are exactly the ``full``
        verdict pixels, 52% of a fine Sphere's silhouette. One sheet is the
        correct and complete answer for them.

    A pixel that fails is dropped from the statistics and counted.

    The gate is not hypothetical. ``Polyhedron`` builds each face from a
    hardcoded index list, and those lists are not consistently oriented --
    measured on this repo, 12 of an ``Icosahedron``'s 20 faces, 2 of 4 on a
    ``Tetrahedron``, 2 of 8 on an ``Octahedron`` and 3 of 12 on a
    ``Dodecahedron`` wind inward, against 0 of 6 on a ``Cube``. The projected
    winding sign IS ``_AA_BACKFACE_BIT``, so on those solids the bit does not
    name a sheet, and a near and a far face can land in one group (measured: a
    pixel whose "front" group sums to 1.98 because it holds both sheets, while
    the true sheets tile to 1.0000 each). ``ALGAN_POLYHEDRON_WINDING=1`` fixes
    it, and this function's drop count is the evidence that it does.
    """
    sums = [0.0, 0.0]
    claimed = [0, 0]
    for f, m, c in zip(faces, msks, covs):
        bits = m & _AA_MASK_ALL
        if bits & claimed[f]:
            return 0.0, False  # one group, two sheets
        claimed[f] |= bits
        sums[f] += c
    front, back = sums
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
        self.err_cf = 0.0
        self.err_1s = 0.0
        self.err_cap = 0.0
        self.sig_actual = 0.0
        self.sig_own = 0.0
        self.gap_actual_own = 0.0
        self.on_lattice = 0
        self.unreferenced = 0
        self.covered = 0
        self.multi_obj = 0
        self.interior = 0
        self.notched = 0
        self.notch_err = 0.0
        self.notch_max = 0.0
        self.claim_occ_max = 0.0
        self.by_verdict = Counter()
        self.err_by_verdict = Counter()
        # Painted coverage per kernel-space pixel, for the REFERENCE-FREE A/B
        # (--mesh-ab). Every covered pixel, not just the scored ones: the whole
        # point is to see the population _exact_coverage has to drop.
        self.painted = {}

    def add(
        self,
        verdict,
        truth,
        ok,
        actual,
        occ,
        own,
        ce,
        cf,
        one_sheet,
        mesh_cap=0.0,
        sids=(),
        faces=(),
    ):
        self.covered += 1
        self.claim_occ_max = max(self.claim_occ_max, abs(actual - occ))
        if not ok:
            self.unreferenced += 1
            return
        if truth >= _SILH_HI:
            # INTERIOR. Scored separately and never dropped silently, because
            # this harness reporting only silhouette pixels is how ss6.3.2's
            # relaxed gate came back an 84% win while _aa_line_check measured it
            # getting WORSE: every pixel the gate actually moved was an interior
            # one, darkened by a mean 0.027. A tiling that the resolve paints
            # below 1 is a NOTCH, and notches are what the whole run rule exists
            # to avoid -- so they belong in the same table as the win.
            self.interior += 1
            if actual < 1.0 - _NOTCH_TOL:
                self.notched += 1
                self.notch_err += 1.0 - actual
                self.notch_max = max(self.notch_max, 1.0 - actual)
            return
        if truth <= _SILH_LO:
            return
        # SOUNDNESS, not a result. _exact_coverage sums one sheet's clipped
        # areas, which is the pixel's true coverage only while that sheet
        # belongs to ONE solid. Two DISTINCT solids overlapping in a pixel sum
        # to more than the area of their union, and the partition test above
        # only catches it when their masks collide -- an overlap holding no
        # sample point slips through and inflates the reference. So count the
        # surviving pixels where a facing group holds more than one surface id:
        # while this is zero the reference is exactly as sound as its docstring
        # claims, and where it is not, the row's error is an upper bound at best.
        #
        # Reads sid, so it is only meaningful where sid names a SOLID:
        # ALGAN_MESH_ID=1 on the packed cases (with it off the whole pack is one
        # id and this cannot see the overlap), and it is noise on a Polyhedron
        # with MESH_ID off, where every triangle is already its own id.
        groups = {}
        for s, f in zip(sids, faces):
            groups.setdefault(f, set()).add(s)
        if any(len(v) > 1 for v in groups.values()):
            self.multi_obj += 1
        self.n += 1
        self.err_actual += abs(actual - truth)
        self.err_own += abs(own - truth)
        self.err_ce += abs(ce - truth)
        self.err_cf += abs(cf - truth)
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
        # aa_grp 2 is ss6.3.2's relaxed gate (ALGAN_ANALYTIC_AA_RUN_FULL). The
        # replay has to follow the KERNEL, or 'actual' silently keeps reporting
        # the shipped walk while the render does something else -- and --verify
        # would fail for a reason that is the harness's, not the renderer's.
        run_full = int(coverage["aa_grp"]) >= 2
        one_mesh = int(coverage["aa_grp"]) == 3
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
        caps_all = coverage["frag_cap"].detach().cpu().tolist()
        for i in range(n_cov):
            lo, hi = offs[i], offs[i + 1]
            sids, faces, ms, cs, bz = [], [], [], [], []
            cps = caps_all[lo:hi]
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
            truth, ok = _exact_coverage(faces, ms, cs)
            actual, occ, effs = _replay(
                sids,
                faces,
                ms,
                cs,
                bz,
                rule_b,
                consult_e=run_full,
                consult_full=run_full,
                mesh_cap=one_mesh,
                mesh_cap_gated=True,
                caps=cps,
            )
            ce, _occ_ce, _e = _replay(sids, faces, ms, cs, bz, rule_b, consult_e=True)
            cf, _occ_cf, _e2 = _replay(
                sids, faces, ms, cs, bz, rule_b, consult_e=True, consult_full=True
            )
            one_sheet, _occ_1s, _e1 = _replay(
                sids, faces, ms, cs, bz, rule_b, one_sheet=True
            )
            mcap, _occ_mc, _e3 = _replay(
                sids,
                faces,
                ms,
                cs,
                bz,
                rule_b,
                consult_e=True,
                consult_full=True,
                mesh_cap=True,
            )
            union = 0
            for m in ms:
                union |= m & _AA_MASK_ALL
            own = _popcount(union) / _AA_NUM_SAMPLES
            svis_stats.add(
                verdict,
                truth,
                ok,
                actual,
                occ,
                own,
                ce,
                cf,
                one_sheet,
                mcap,
                sids,
                faces,
            )
            p = pix[i] % ppf
            py, px = p // width, p % width
            svis_stats.painted[(px, py)] = actual
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
    where = None
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
            d = abs(float(r[10]) - effs[q])
            if d > worst:
                worst = d
                # Kept so a failure names the fragment instead of a magnitude:
                # (pixel, fragment index, kernel eff, replay eff, mask, cov).
                where = (px, py, q, float(r[10]), effs[q], int(r[7]), float(r[8]))
    return worst, rows_seen, len(probes), where


def _mesh_ab(cases, settings):
    """Reference-free A/B of ``ALGAN_MESH_ID`` on painted coverage.

    The ss6.3 table scores against ``_exact_coverage``, which has to DROP a
    pixel whose facing group holds two sheets -- and on a packed grid those
    drops are concentrated exactly where two spheres overlap, which is the
    population MESH_ID is supposed to fix. Scoring only the survivors can
    therefore report "neutral" while being blind to the effect.

    This compares the two settings against each OTHER instead: same scene, same
    frame, ``rp.prepare_sparse_raster_coverage`` replayed both ways, differenced
    per covered pixel. It needs no reference, so it sees every pixel including
    the dropped ones. It says how MUCH moves, not which side is right.
    """
    print("\nMESH_ID A/B on painted coverage (reference-free, every covered pixel):")
    head = (
        f"{'case':22s} {'pixels':>8s} {'moved':>7s} {'max |d|':>9s} "
        f"{'mean |d|':>9s} {'mean off-on':>12s}"
    )
    print(head)
    print("-" * len(head))
    for name, build in cases.items():
        painted = []
        for enabled in (False, True):
            rt_settings.set_mesh_id(enabled)
            *_rest, sv, _silh, _cap = _measure(build, settings)
            painted.append(sv.painted)
        rt_settings.set_mesh_id(False)
        off, on = painted
        shared = off.keys() & on.keys()
        signed = [off[k] - on[k] for k in shared if abs(off[k] - on[k]) > 1e-9]
        only = (len(off) - len(shared)) + (len(on) - len(shared))
        n_moved = len(signed)
        mean_abs = sum(abs(d) for d in signed) / n_moved if n_moved else 0.0
        mean_signed = sum(signed) / n_moved if n_moved else 0.0
        worst = max((abs(off[k] - on[k]) for k in shared), default=0.0)
        # A POSITIVE 'mean off-on' means MESH_ID=0 paints more than MESH_ID=1,
        # which is the signature the packed grid predicts: with one id for the
        # whole pack a run carries across two spheres and their masks OR to a
        # full union, so corr short-circuits to 1 and the pixel claims coverage
        # neither sphere has. It says the two disagree in the predicted
        # direction; it does not by itself say the smaller number is right.
        print(
            f"{name:22s} {len(shared):8d} {n_moved:7d} {worst:9.4f} "
            f"{mean_abs:9.4f} {mean_signed:+12.4f}"
            + (f"   (+{only} one-sided)" if only else "")
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--res", choices=sorted(RESOLUTIONS), default="md")
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument(
        "--mesh-ab",
        action="store_true",
        help="difference painted coverage between ALGAN_MESH_ID=0 and =1 per "
        "pixel, which needs no reference and so sees the pixels _exact_coverage "
        "has to drop",
    )
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

    if args.mesh_ab:
        _mesh_ab(cases, settings)
        return

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
        f"{'|actual-own|':>13s} {'|cE-E|':>8s} {'|cF-E|':>8s} {'|1sheet-E|':>11s} "
        f"{'|cap-E|':>9s} "
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
            f"{sv.err_ce / n:8.4f} {sv.err_cf / n:8.4f} {sv.err_1s / n:11.4f} "
            f"{sv.err_cap / n:9.4f} "
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
            f"{'':22s} scored pixels holding >1 surface id per facing group "
            f"{sv.multi_obj}/{n}"
        )
        notch = (
            f"{sv.notched}/{sv.interior}  mean {sv.notch_err / sv.notched:.4f}"
            f"  worst {sv.notch_max:.4f}"
            if sv.notched
            else f"0/{sv.interior}"
        )
        print(f"{'':22s} INTERIOR pixels painted below full (notches)  {notch}")
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
            worst, rows, probes, where = res
            tag = "PASS" if worst < 2e-5 else "FAIL"
            print(
                f"  [{tag}] {name:22s} worst |eff| diff {worst:.2e} over "
                f"{rows} fragment rows at {probes} pixels"
            )
            if tag == "FAIL" and where:
                px, py, q, k_eff, r_eff, wmsk, wcov = where
                print(
                    f"{'':9s} worst at pixel ({px},{py}) fragment {q}: "
                    f"kernel eff {k_eff:.6f} vs replay {r_eff:.6f}  "
                    f"msk={wmsk:03x} cov={wcov:.5f}"
                )


if __name__ == "__main__":
    main()
