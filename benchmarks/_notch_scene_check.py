"""Does the run-scan limit notch a REAL scene? -- ss0.5's unmeasured half.

``DESIGN_mesh_identity.md`` ss0.5 ships a known limitation deliberately. The
analytic-AA run scan stops after ``_AA_MAX_RUN_SCAN = 16`` fragments, so the
exact-area sum ``E`` it hands the relaxed gate is a PARTIAL sum -- a lower bound
on the sheet's area -- and ``run_corr = min(E, 1)`` then scales the pixel down by
precisely the area the scan never reached. On a silhouette pixel that is the
intended fix; on an interior pixel it is a notch. ss0.5 measures it on eleven
synthetic harness cases, costs it, and closes with the one thing nobody had
done:

    "Nobody has counted notched pixels in the six tests/full_renders scenes,
     which are the only realistic scenes here. ... If you are picking this up,
     measure that first."

This is that count.


WHY NOT ``_aa_run_gate_check.py --notch-probe``, WHICH ss0.5 NAMES
------------------------------------------------------------------
Because pointing it at these scenes would produce a number that means nothing,
and it would look like good news. Its "notched" verdict is ``actual < 1`` from a
host replay of the resolve, on a pixel its exact reference certifies as
interior. Both halves are built for a harness case and neither survives a real
scene:

  * ``_replay`` models MATTE OPAQUE geometry -- ``mat_alpha == 1``,
    ``trans_share == 0``, no reflection break -- which is what every harness
    case builds and is not what ``materials_and_lighting`` is.
  * ``_exact_coverage`` assumes the pixel holds ONE closed convex solid or one
    flat sheet, and DROPS the pixel when the two facing groups disagree or sum
    past 1. A sphere in front of a floor sums two footprints, so the pixel is
    dropped. In a real scene that discards exactly the population being counted,
    and the resulting "no notches" would be a property of the reference, not of
    the renderer (ss0.1 rule 1: a check must show it REACHES its case).

So this probe scores the MECHANISM rather than the symptom, which needs neither
a material model nor a reference. It replays the run scan with exactly ONE input
changed -- the scan limit -- and reports the coverage the pixel loses purely to
the truncation:

    corr(limit = 16)  vs  corr(limit = infinity),  same fragments, same rule

Both sides come from the compact fragment arrays alone (sample masks, exact
clipped areas, ``tri_obj`` surface ids). No walk, no material, no reference.
That is ss0.1 rule 2, which is what attributed the notches three times running.

The shortfall is ``corr_inf - corr_16``, in units of pixel coverage. ss0.5's
conversion applies: a coverage error shows as that fraction of the CONTRAST
between the surface and what is behind it, so x255 is its 8-bit ceiling against
a maximally contrasting background.


WHAT IT DOES NOT SEE, SAID BEFORE THE TABLE
-------------------------------------------
* **The first run only.** The kernel scans a run only where ``svis`` is still
  uniform, which is guaranteed at a pixel's first fragment and is a property of
  the walk afterwards. Restricting to the first run keeps every number exact and
  material-free; it makes the count a LOWER bound on the general effect. ss0.5's
  own diagnosis is about the first run (lifting the limit recovered 231 of 253
  rod pixels), so this is the dominant case, not a corner of it.
* **Pixels whose first fragment is a circuit.** A bezier fragment never enters
  run mode. Those pixels are counted and reported separately rather than
  silently skipped.
* **The one-mesh cap's own contribution**, which on the rod was 14 notched
  pixels of 253. That one needs the walk.
* **Painted loss is at most the shortfall.** What the pixel finally paints
  depends on what the rest of its fragment list does with the transmittance the
  short claim leaves standing. With the cap on, the far sheet cannot refill it
  (ss6.6), so the two coincide on a one-mesh pixel; elsewhere the shortfall
  bounds it from above.

Run:  <venv-python> benchmarks/_notch_scene_check.py            # the six scenes
      <venv-python> benchmarks/_notch_scene_check.py --cases    # cross-check
      <venv-python> benchmarks/_notch_scene_check.py --scenes solids materials
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks"))

import torch  # noqa: E402

from algan import HD, LD, MD, PREVIEW, SETTINGS, Off, Scene  # noqa: E402
from algan.rendering.raytracing import raster_pipeline as rp  # noqa: E402
from algan.rendering.raytracing.raster_taichi import (  # noqa: E402
    _AA_BACKFACE_BIT,
    _AA_FULL_DUST,
    _AA_MASK_ALL,
    _AA_MAX_RUN_SCAN,
    _AA_ONE_MESH_BIT,
    _aa_run_full,
    _tri_run_mode,
)
from algan.scene_manager import SceneManager  # noqa: E402

RESOLUTIONS = {"ld": LD, "md": MD, "hd": HD, "preview": PREVIEW}

FULL_RENDERS = REPO / "tests" / "full_renders"

#: How much coverage a pixel must lose to the scan limit before it is counted.
#: Well above the exact-area arithmetic's float dust, and an order of magnitude
#: below the render suites' 2/255 tolerance.
_SHORTFALL_TOL = 1e-3


class Stats:
    """One scene's counts. Every field is a population, not a sample."""

    def __init__(self, name):
        self.name = name
        self.batches = 0
        self.frames = 0
        self.covered = 0
        self.fragments = 0
        self.first_is_bez = 0
        self.scanned = 0
        self.truncated = 0
        self.arm_full = 0
        self.arm_partial = 0
        self.arm_donor = 0
        self.notched = 0
        self.notched_interior = 0
        self.notched_one_mesh = 0
        self.shortfall_sum = 0.0
        self.shortfall_worst = 0.0
        self.interior_sum = 0.0
        self.interior_worst = 0.0
        self.worst_at = None
        self.run_max = 0
        self.no_run_mode = 0
        self.frames_with_notch = set()
        # tri_obj rows vs chunk offset: the host's ONE_MESH reduction indexes
        # tri_obj by the CHUNK-relative frame while the kernel indexes it by the
        # BATCH-relative one, so a chunk that does not start at frame 0 reads a
        # different row. Recorded to say whether that is reachable here at all.
        self.multi_row_batches = 0
        self.offset_chunks = 0
        self.row_mismatch_frags = 0
        self.row_mismatch_pixels = 0
        self.baseline = "not checked"

    def add_shortfall(self, values, where):
        n = int(values.numel())
        if not n:
            return
        self.notched += n
        self.shortfall_sum += float(values.sum())
        worst = float(values.max())
        if worst > self.shortfall_worst:
            self.shortfall_worst = worst
            self.worst_at = where

    @property
    def mean_shortfall(self):
        return self.shortfall_sum / self.notched if self.notched else 0.0

    @property
    def mean_interior(self):
        n = self.notched_interior
        return self.interior_sum / n if n else 0.0


#: Covered pixels scored per pass. Only bounds host memory -- a batch of a dense
#: PREVIEW scene carries tens of millions of fragments, and the derived arrays
#: are per fragment. Runs never cross a pixel boundary, so blocking at one is
#: exact.
_BLOCK = 262144


def _probe_batch(coverage, merged, width, height, time_start, stats):
    """Score one emitted fragment stream. Pure tensor work, no walk."""
    n_cov = int(coverage["num_covered"])
    n_frag = int(coverage["num_fragments"])
    stats.batches += 1
    if n_cov <= 0 or n_frag <= 0:
        return
    aa_tri = int(coverage["aa_tri"])
    aa_grp = int(coverage["aa_grp"])
    if not _tri_run_mode(aa_tri) or not aa_grp:
        # No run rule compiled into this batch's resolve: the mechanism cannot
        # fire, and saying so is not the same as saying it did not.
        stats.no_run_mode += 1
        return

    offs_all = coverage["run_offsets"][: n_cov + 1].detach().to("cpu", torch.int64)
    pix_all = coverage["covered_idx"][:n_cov].detach().to("cpu", torch.int64)
    tri_obj = merged["tri_obj"].detach().cpu()

    ppf = int(width) * int(height)
    rows = int(tri_obj.shape[0])
    if rows > 1:
        stats.multi_row_batches += 1
        if int(time_start):
            stats.offset_chunks += 1
    stats.covered += n_cov
    stats.fragments += n_frag
    # ``bfrm``: frames in the LARGEST chunk, not the video's length -- time_start
    # is relative to the batch, which the render loop restarts per batch. The
    # video's length is on the baseline line.
    stats.frames = max(stats.frames, int(pix_all.max()) // ppf + 1 + int(time_start))

    for c0 in range(0, n_cov, _BLOCK):
        c1 = min(c0 + _BLOCK, n_cov)
        f0 = int(offs_all[c0])
        f1 = int(offs_all[c1])
        if f1 <= f0:
            continue
        _probe_block(
            coverage,
            tri_obj,
            offs_all[c0 : c1 + 1] - f0,
            pix_all[c0:c1],
            f0,
            f1,
            aa_grp,
            ppf,
            width,
            time_start,
            stats,
        )


def _all_same(sid, seg, n_cov):
    """Per pixel: do all its fragments carry one surface id? (the flag's test)"""
    lo = torch.full((n_cov,), 1 << 40, dtype=torch.int64)
    hi = torch.full((n_cov,), -1, dtype=torch.int64)
    lo.scatter_reduce_(0, seg, sid, reduce="amin", include_self=True)
    hi.scatter_reduce_(0, seg, sid, reduce="amax", include_self=True)
    return (lo == hi) & (lo >= 0)


def _probe_block(
    coverage, tri_obj, offs, pix, f0, f1, aa_grp, ppf, width, time_start, stats
):
    n_cov = int(pix.numel())
    n_frag = f1 - f0
    ref = coverage["frag_ref"][f0:f1].detach().to("cpu", torch.int64)
    cov = coverage["frag_cov"][f0:f1].detach().to("cpu", torch.float64)
    msk = coverage["frag_msk"][f0:f1].detach().to("cpu", torch.int64)

    counts = offs[1:] - offs[:-1]
    seg = torch.repeat_interleave(torch.arange(n_cov, dtype=torch.int64), counts)
    # The kernel reads tri_obj at the BATCH-relative frame (f = time_start +
    # g // ppf), so the probe must too or it groups fragments by another frame's
    # surface map.
    rows = int(tri_obj.shape[0])
    frame_rel = pix[seg] // ppf
    frame = (frame_rel + int(time_start)) % rows
    is_bez = ref < 0
    safe_ref = ref.clamp_min(0)
    sid = torch.where(
        is_bez,
        -1 - ((-ref - 1) >> 8),
        tri_obj[frame, safe_ref].to(torch.int64),
    )
    face = (msk & _AA_BACKFACE_BIT) != 0
    key = (sid << 1) | face.to(torch.int64)

    # A run breaks at a pixel boundary, a circuit fragment, or a change of
    # (surface, facing) -- exactly _aa_run_scan's three terminators.
    starts = torch.zeros(n_frag, dtype=torch.bool)
    starts[offs[:-1]] = True
    same = torch.zeros(n_frag, dtype=torch.bool)
    same[1:] = (key[1:] == key[:-1]) & (~is_bez[1:]) & (~is_bez[:-1])
    is_start = starts | (~same)
    run_id = torch.cumsum(is_start.to(torch.int64), 0) - 1
    run_len_of = torch.bincount(run_id)

    # THE HOST/KERNEL ROW SPLIT. prepare_sparse_raster_coverage's ONE_MESH
    # reduction reads tri_obj at ``pix_s // ppf`` -- the CHUNK-relative frame --
    # while every kernel reads it at ``time_start + f_rel``, the BATCH-relative
    # one, and every other frame derivation in that same file adds time_start.
    # So on a chunk that does not start at frame 0 the two ask a different
    # frame's surface map. Scored rather than asserted: count the fragments
    # whose id moves between the two rows, and the pixels whose "all one
    # surface" verdict moves with it, which is what the flag is.
    #
    # The pixel count is an UPPER bound: the host also requires every fragment
    # to be opaque (``mat_opaque_s``), which the coverage dict does not carry,
    # so a pixel disqualified by opacity in BOTH arms can still show a verdict
    # flip here. The fragment count is exact, and it is the one that says
    # whether the two rows are the same row.
    if rows > 1 and int(time_start):
        sid_host = tri_obj[frame_rel % rows, safe_ref].to(torch.int64)
        moved = (~is_bez) & (sid_host != sid)
        n_moved = int(moved.sum())
        stats.row_mismatch_frags += n_moved
        if n_moved:
            one_k = _all_same(sid, seg, n_cov)
            one_h = _all_same(sid_host, seg, n_cov)
            stats.row_mismatch_pixels += int((one_k != one_h).sum())

    j0 = offs[:-1]
    first_bez = is_bez[j0]
    stats.first_is_bez += int(first_bez.sum())
    runlen = run_len_of[run_id[j0]]
    runlen = torch.where(first_bez, torch.zeros_like(runlen), runlen)
    if int(runlen.numel()):
        stats.run_max = max(stats.run_max, int(runlen.max()))

    # The gate the kernel applies at a run's first fragment.
    msk0 = msk[j0] & _AA_MASK_ALL
    scan = msk0 != _AA_MASK_ALL
    if _aa_run_full(aa_grp):
        scan = scan | (cov[j0] < 1.0 - _AA_FULL_DUST)
    scan = scan & (~first_bez)
    stats.scanned += int(scan.sum())

    k = torch.clamp(runlen, max=_AA_MAX_RUN_SCAN)
    csum = torch.zeros(n_frag + 1, dtype=torch.float64)
    torch.cumsum(cov, 0, out=csum[1:])
    e_trunc = csum[j0 + k] - csum[j0]
    e_full = csum[j0 + runlen] - csum[j0]

    # OR of the scanned masks: 16 coherent gathers rather than a segment scan.
    u_trunc = torch.zeros(n_cov, dtype=torch.int64)
    for t in range(_AA_MAX_RUN_SCAN):
        take = t < k
        idx = torch.clamp(j0 + t, max=n_frag - 1)
        u_trunc |= torch.where(take, msk[idx] & _AA_MASK_ALL, 0)

    truncated = scan & (runlen > _AA_MAX_RUN_SCAN)
    stats.truncated += int(truncated.sum())
    full_arm = u_trunc == _AA_MASK_ALL
    donor_arm = u_trunc == 0
    stats.arm_full += int((truncated & full_arm).sum())
    stats.arm_donor += int((truncated & donor_arm).sum())
    stats.arm_partial += int((truncated & ~full_arm & ~donor_arm).sum())

    # ss0.5's arm. OR is monotone in the scan length, so a truncated union of
    # MASK_ALL stays MASK_ALL unbounded: the rule takes the same branch either
    # way and the ONLY difference is E.
    def corr(e):
        return torch.where(
            (1.0 - e).abs() > _AA_FULL_DUST, torch.clamp(e, max=1.0), 1.0
        )

    shortfall = corr(e_full) - corr(e_trunc)
    hit = truncated & full_arm & (shortfall > _SHORTFALL_TOL)
    if not bool(hit.any()):
        return
    vals = shortfall[hit]
    idx = hit.nonzero(as_tuple=True)[0]
    p = pix[idx]
    frames = p // ppf
    worst_j = int(vals.argmax())
    pw = int(p[worst_j]) % ppf
    stats.add_shortfall(
        vals,
        (int(pw % width), int(pw // width), int(frames[worst_j]) + int(time_start)),
    )
    # An interior pixel is one whose sheet really does tile it: the UNBOUNDED
    # sum reaches 1. Where it does not, corr < 1 is partly intended and only the
    # truncated part of it is this defect.
    interior = (1.0 - e_full[idx]).abs() <= _AA_FULL_DUST
    stats.notched_interior += int(interior.sum())
    if bool(interior.any()):
        # Split out on purpose: an INTERIOR pixel's whole shortfall is the
        # defect, while on a silhouette pixel part of ``corr < 1`` is the
        # relaxed gate working as designed and only the truncated part is not.
        # ss0.5's published means are over interior pixels, so a comparison
        # against them has to be too.
        iv = vals[interior]
        stats.interior_sum += float(iv.sum())
        stats.interior_worst = max(stats.interior_worst, float(iv.max()))
    stats.notched_one_mesh += int(((msk[j0[idx]] & _AA_ONE_MESH_BIT) != 0).sum())
    for f in frames.unique().tolist():
        stats.frames_with_notch.add(f + int(time_start))


class _Spy:
    """Wrap the emission so every batch's fragment stream is scored."""

    def __init__(self, stats):
        self.stats = stats
        self.original = rp.prepare_sparse_raster_coverage

    def __enter__(self):
        original = self.original
        stats = self.stats

        def spy(*args, **kwargs):
            coverage = original(*args, **kwargs)
            if coverage is not None:

                def arg(name, pos):
                    return kwargs[name] if name in kwargs else args[pos]

                _probe_batch(
                    coverage,
                    arg("merged", 0),
                    int(arg("width", 13)),
                    int(arg("height", 14)),
                    int(arg("time_start", 11)),
                    stats,
                )
            return coverage

        rp.prepare_sparse_raster_coverage = spy
        return self

    def __exit__(self, *exc):
        rp.prepare_sparse_raster_coverage = self.original
        return False


def _register_test_fonts():
    """Run ``tests/conftest.py``, which registers the vendored faces with Pango.

    Not optional, and not a detail. The full-render scenes name
    ``Algan Test Sans`` explicitly, and without the registration Pango
    substitutes whatever the host has installed -- which is a STRUCTURAL change
    to every glyph, not a sub-pixel one. Measured here before it was fixed:
    these six scenes came back 205-232 channel values from their own committed
    baselines over 8-11% of every frame, so the probe had rendered a different
    scene than the one it claimed to be measuring.

    Executed by path rather than copied so the two cannot drift; the module runs
    its registration at import.
    """
    conftest = REPO / "tests" / "conftest.py"
    spec = importlib.util.spec_from_file_location("_algan_notch_conftest", conftest)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def _against_baseline(rendered, stats):
    """Diff the probe's own render against the scene's committed baseline.

    ss0.1 rule 1: a check must show it REACHES its case. Everything this script
    reports is a statement about the six scenes as they ship, and that is only
    true if the render it scored IS the shipping one. This is the cheapest thing
    that can say so, and it is what caught the substitute-font run: 205-232
    channel values from the baseline, which is a different scene, not a drift.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    expected = FULL_RENDERS / f"expected_outputs_{device}" / rendered.name
    if not expected.exists():
        stats.baseline = f"no {device} baseline"
        return
    import cv2
    import numpy as np

    a, b = cv2.VideoCapture(str(rendered)), cv2.VideoCapture(str(expected))
    worst = moved = frames = 0
    while True:
        ok_a, fa = a.read()
        ok_b, fb = b.read()
        if not ok_a or not ok_b:
            if ok_a != ok_b:
                stats.baseline = f"FRAME COUNT MISMATCH at {frames}"
                a.release()
                b.release()
                return
            break
        delta = np.abs(fa.astype(np.int16) - fb.astype(np.int16))
        worst = max(worst, int(delta.max()))
        moved = max(moved, int((delta.max(axis=2) > 2).sum()))
        frames += 1
    a.release()
    b.release()
    stats.baseline = (
        f"baseline max|d| {worst} over <= {moved} px/frame ({frames} frames)"
        if worst
        else f"byte-identical to the {device} baseline ({frames} frames)"
    )


def _render_full_render_scene(path, stats, quality, at):
    """Render one tests/full_renders scene the way its own suite renders it."""
    snapshot = SETTINGS.snapshot()
    cwd = os.getcwd()
    out = FULL_RENDERS / "algan_outputs" / "_notch_probe"
    out.mkdir(parents=True, exist_ok=True)
    (FULL_RENDERS / "algan_cache").mkdir(parents=True, exist_ok=True)
    os.chdir(FULL_RENDERS)
    SETTINGS.paths.set(
        output_root=str(FULL_RENDERS),
        output_directory=str(out.relative_to(FULL_RENDERS)),
        cache_directory=str(FULL_RENDERS / "algan_cache"),
    )
    # The suite pins this so the frame-window split is reproducible; without it
    # the batches land differently and so do their tri_obj rows.
    SETTINGS.computing.set(available_memory_override=1536 * 1024 * 1024)
    SceneManager.reset()
    try:
        with Scene() as scene:
            name = f"_algan_notch_{path.stem}"
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
            finally:
                sys.modules.pop(name, None)
            with _Spy(stats):
                if at:
                    scene.save_frame(
                        str(out / f"{path.stem}.png"),
                        video_settings=quality,
                        at=at,
                        overwrite=True,
                    )
                else:
                    scene.save_video(
                        str(out / f"{path.stem}.mp4"),
                        video_settings=quality,
                        overwrite=True,
                        animate_fade_out=True,
                    )
        if not at and quality is PREVIEW:
            _against_baseline(out / f"{path.stem}.mp4", stats)
    finally:
        os.chdir(cwd)
        SETTINGS.restore(snapshot)
        SceneManager.reset()


def _row_split_demo(stats, quality, memory_mb):
    """Reach the host/kernel tri_obj row split, which the six scenes do not.

    Measured over all six full-render scenes and every chunk that starts past
    frame 0: **no fragment's surface id moves** between the two rows. That is
    not the bug being absent, it is the map being frame-invariant -- a diced
    primitive's row -> SOURCE SURFACE map only varies with the frame when its
    patches belong to more than one surface, and every PN primitive in those
    scenes is one mesh, so every row carries the same id whatever the dice
    level does.

    So the trigger is: ONE primitive carrying SEVERAL surfaces, diced, in a
    batch the render loop splits into more than one chunk. A packed-grid
    ``Surface`` is exactly that (ss4.5) -- ``batch_mobs`` flattens sixteen
    independent Spheres into a single logical-PN primitive whose patches carry
    sixteen declared ``mesh_ids``. Rotating it makes the adaptive levels move
    per frame, and a small memory override makes the loop chunk.
    """
    from algan import ORIGIN, OUT, RIGHT, UP, WHITE, Sphere
    from algan.utils.mob_utils import batch_mobs

    snapshot = SETTINGS.snapshot()
    SETTINGS.computing.set(available_memory_override=memory_mb * 1024 * 1024)
    SceneManager.reset()
    try:
        with Scene() as scene:
            dots = [
                Sphere(
                    radius=0.28,
                    resolution=(24, 12),
                    color=WHITE,
                    add_to_scene=False,
                ).move_to(
                    RIGHT * ((i % 4) - 1.5) * 0.75
                    + UP * ((i // 4) - 1.5) * 0.75
                    + OUT * (0.3 if (i % 2) else -0.3)
                )
                for i in range(16)
            ]
            batch_mobs(dots, add_to_scene=True).spawn()
            # Orbiting the camera is what makes the DICE levels move, and the
            # levels are what re-lay the rows. Driven from the camera rather
            # than the pack because a packed grid's location is per point.
            Scene.get_camera().orbit(25, RIGHT, about_point=ORIGIN)
            with _Spy(stats):
                scene.save_video(
                    str(REPO / "algan_outputs" / "_notch_row_split.mp4"),
                    video_settings=quality,
                    overwrite=True,
                )
    finally:
        SETTINGS.restore(snapshot)
        SceneManager.reset()


def _render_harness_case(build, stats, quality):
    """One ``_aa_run_gate_check`` case, built and rendered as that harness does."""
    with Scene() as scene:
        with Off():
            build()
        with _Spy(stats):
            scene.save_frame(
                str(REPO / "algan_outputs" / "_notch_scene_check.png"),
                video_settings=quality,
                overwrite=True,
            )


def _report(rows):
    head = (
        f"{'scene':26s} {'bfrm':>5s} {'covered px':>11s} {'scanned':>9s} "
        f"{'trunc':>7s} {'NOTCHED':>8s} {'interior':>8s} {'int mean':>8s} "
        f"{'int wst':>8s} {'x255':>5s} {'all mean':>8s}"
    )
    print(head)
    print("-" * len(head))
    for s in rows:
        print(
            f"{s.name:26s} {s.frames:5d} {s.covered:11d} {s.scanned:9d} "
            f"{s.truncated:7d} {s.notched:8d} {s.notched_interior:8d} "
            f"{s.mean_interior:8.4f} {s.interior_worst:8.4f} "
            f"{s.interior_worst * 255:5.1f} {s.mean_shortfall:8.4f}"
        )
    print()
    for s in rows:
        note = []
        if s.no_run_mode:
            note.append(f"{s.no_run_mode} batches with no run rule")
        if s.first_is_bez:
            note.append(f"{s.first_is_bez} px lead with a circuit")
        if s.truncated:
            note.append(
                f"arms full/partial/donor {s.arm_full}/{s.arm_partial}/{s.arm_donor}"
            )
        if s.notched:
            note.append(f"one-mesh {s.notched_one_mesh}/{s.notched}")
            note.append(f"worst px {s.worst_at}")
            note.append(f"{len(s.frames_with_notch)} frames carry one")
        note.append(s.baseline)
        note.append(f"longest run {s.run_max}")
        note.append(f"{s.batches} batches")
        if s.multi_row_batches:
            note.append(
                f"{s.multi_row_batches} per-frame tri_obj"
                + (f", {s.offset_chunks} at a chunk offset" if s.offset_chunks else "")
            )
        if s.row_mismatch_frags:
            note.append(
                f"HOST/KERNEL tri_obj row split: {s.row_mismatch_frags} frags, "
                f"{s.row_mismatch_pixels} px change one-mesh verdict"
            )
        print(f"{s.name:26s} {'; '.join(note)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--scenes",
        nargs="*",
        default=None,
        help="substring filter over tests/full_renders/scenes (default: all six)",
    )
    ap.add_argument(
        "--cases",
        nargs="*",
        default=None,
        help="run _aa_run_gate_check's synthetic cases instead, to cross-check "
        "this instrument against ss0.5's published table",
    )
    ap.add_argument("--res", choices=sorted(RESOLUTIONS), default=None)
    ap.add_argument(
        "--row-split-demo",
        nargs="?",
        type=int,
        const=512,
        default=None,
        metavar="MB",
        help="render a packed-grid Surface (one primitive, sixteen surfaces) "
        "under a small memory override, which is what reaches the host/kernel "
        "tri_obj row split the six scenes cannot",
    )
    ap.add_argument(
        "--at",
        nargs="*",
        type=float,
        default=None,
        help="render only these scene times (seconds) instead of the whole video",
    )
    args = ap.parse_args()

    print(f"run-scan cap = {_AA_MAX_RUN_SCAN}, dust = {_AA_FULL_DUST}")
    rows = []
    t0 = time.time()

    if args.row_split_demo is not None:
        stats = Stats(f"packed grid @{args.row_split_demo}MB")
        _row_split_demo(stats, RESOLUTIONS[args.res or "preview"], args.row_split_demo)
        _report([stats])
        return

    if args.cases is not None:
        import _aa_run_gate_check as gate

        quality = RESOLUTIONS[args.res or "md"]
        cases = gate._cases()
        if args.cases:
            cases = {k: v for k, v in cases.items() if any(c in k for c in args.cases)}
        print(f"resolution = {args.res or 'md'} (harness cases)\n")
        for name, build in cases.items():
            stats = Stats(name)
            _render_harness_case(build, stats, quality)
            rows.append(stats)
            print(f"  {name}: {stats.notched} notched ({time.time() - t0:.0f}s)")
        print()
    else:
        _register_test_fonts()
        quality = RESOLUTIONS[args.res or "preview"]
        paths = sorted(
            p
            for p in (FULL_RENDERS / "scenes").glob("*.py")
            if not p.name.startswith("_")
        )
        if args.scenes:
            paths = [p for p in paths if any(s in p.stem for s in args.scenes)]
        print(f"resolution = {args.res or 'preview'} (tests/full_renders)\n")
        for path in paths:
            stats = Stats(path.stem)
            _render_full_render_scene(path, stats, quality, args.at)
            rows.append(stats)
            print(f"  {path.stem}: {stats.notched} notched ({time.time() - t0:.0f}s)")
        print()

    _report(rows)


if __name__ == "__main__":
    main()
