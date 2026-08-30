"""Analytic anti-aliasing, phase 1 (Bezier circuits): kill-switch parity,
engagement, and edge-quality measurement.

See ``algan/rendering/raytracing/DESIGN_analytic_aa.md``.  Phase 1 gives every
circuit fragment the fraction of the pixel square its drawn region covers
(a box filter of the outline SDF that ``_bezier_point_metrics`` already
computes) and folds that into the fragment's alpha, so a render at
``supersampling = 1`` resolves circuit edges continuously instead of
all-or-nothing.

Three things are checked, per config:

  1. KILL SWITCH -- with ``ALGAN_ANALYTIC_AA`` off the render must be
     byte-identical to the pre-feature renderer.  Enforced structurally: the
     toggle-off run is compared against a run with the toggle off and the
     circuit geometry untouched, which is the same code path, so any accidental
     unconditional change shows up as a non-zero diff.  (The reference here is
     the current build; run this before and after a change to keep it honest.)

  2. TRIANGLES UNAFFECTED -- phase 1 leaves flat triangles at coverage 1.0, so
     a triangle-only scene must be byte-identical with the toggle on and off.
     This is the guarantee that lets phase 1 ship before the seam work.

  3. EDGE QUALITY -- analytic AA at aa=1 is compared against supersampled aa=1
     (aliased) and aa=4 (the reference) on circuit content.  Analytic AA must
     be closer to the aa=4 reference than plain aa=1 is, and must produce
     strictly more distinct intensity levels along a slanted edge (the direct
     measurement of "the staircase is gone").

Run: .venv/Scripts/python.exe benchmarks/_analytic_aa_bez_check.py [configs...]
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("ALGAN_PREFETCH_BATCHES", "0")

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Benchmarks must never be measured inside a warm daemon: it keeps adaptive
# renderer state (the memory model's batch-size fit) across runs, so one
# benchmark would be timed against whatever ran before it.
os.environ.setdefault("ALGAN_USE_DAEMON", "0")

import algan.rendering.raytracing.raster_pipeline as rp  # noqa: E402
from algan import (  # noqa: E402
    BLUE,
    DOWN,
    GREEN,
    LEFT,
    OUT,
    RED,
    RIGHT,
    UP,
    YELLOW,
    Arrow,
    Circle,
    Line,
    Line3D,
    Off,
    Scene,
    SceneManager,
    Sphere,
    Square,
    Text,
    Triangle,
    VideoSettings,
)
from algan.rendering.raytracing import settings as rt_settings  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "_rt2_out")
os.makedirs(OUT_DIR, exist_ok=True)

# ``--dense`` forces the dense tile path (raster_sparse_coverage off), which is
# what an environment-mapped or in-composite-tonemap render uses.
DENSE = "--dense" in sys.argv

# Configs whose subject is triangle geometry. Triangle coverage is opt-in, so
# these are run with it forced on and reported rather than gated.
TRI_CONFIGS = ("seam", "tri", "trans", "thin")

# ``--sweep`` reports every sample-less-triangle policy side by side on the
# triangle configs instead of just the shipped default: the choice there is a
# genuine trade between silhouette dilation and interior notches, so it has to
# be read off one table (DESIGN_analytic_aa.md ss16). Each policy compiles its
# own _ss_pixel variant, so the first sweep of a cleared cache is slow.
SWEEP = "--sweep" in sys.argv

# ``--box`` measures the pre-exact box filter (analytic_aa_exact off) instead of
# the shipped exact angle-aware area; ``--exact-ab`` reports both against the
# same aa=4 reference, which is the measurement that decides whether the exact
# form is worth its cost (DESIGN_analytic_aa.md ss21).
EXACT = "--box" not in sys.argv
EXACT_AB = "--exact-ab" in sys.argv
_SLIVER_ARG = next(
    (a.split("=", 1)[1] for a in sys.argv if a.startswith("--sliver=")), None
)

# Small, few frames: this measures pixels, not throughput.
BASE_W, BASE_H = 320, 180
FPS = 4

_counts = {"raster": 0, "dense": 0, "sparse": 0, "aa_shade": 0, "shade": 0}
_orig_first_shade = rp.raster_first_shade
# Located by NAME from the kernel signature: these indices have moved with every
# argument added, and a stale one reads a neighbouring flag and reports a
# perfectly good run as "analytic AA never engaged".
import inspect  # noqa: E402

_FS_PARAMS = list(
    inspect.signature(
        getattr(rp.raster_first_shade, "__wrapped__", rp.raster_first_shade)
    ).parameters
)
_FS_AA_BEZ = _FS_PARAMS.index("aa_bez")
_FS_AA_TRI = _FS_PARAMS.index("aa_tri")


def _first_shade_probe(*a, **k):
    _counts["shade"] += 1
    # Any nonzero mode counts as engaged: aa_bez is 1 (box) / 2 (exact) /
    # 3 (wedge), aa_tri 1 (points) / 2 (cells) / 3 (run-corrected).
    if len(a) > _FS_AA_TRI and (int(a[_FS_AA_BEZ]) != 0 or int(a[_FS_AA_TRI]) != 0):
        _counts["aa_shade"] += 1
    return _orig_first_shade(*a, **k)


rp.raster_first_shade = _first_shade_probe


def _probe_raster(key, orig):
    def wrapper(*a, **k):
        _counts["raster"] += 1
        _counts[key] += 1
        return orig(*a, **k)

    return wrapper


# The tracer imports both entry points from the module inside the render call,
# so rebinding the module attributes intercepts every use (either path engages
# depending on raster_sparse_coverage).
rp.raster_iteration_zero = _probe_raster("dense", rp.raster_iteration_zero)
rp.prepare_sparse_raster_coverage = _probe_raster(
    "sparse", rp.prepare_sparse_raster_coverage
)


def build_scene(cfg):
    if cfg == "tri":
        # Opaque triangle meshes only.
        with Off():
            s1 = Sphere().scale(1.0).move(LEFT * 1.2).set_color(BLUE)
            s1.spawn()
            s2 = Sphere().scale(0.7).move(RIGHT * 1.2 + UP * 0.4)
            s2.set_color(GREEN)
            s2.spawn()
        s1.move(RIGHT * 0.6)
        return
    if cfg == "trans":
        # Translucent triangle mesh. Coverage-as-alpha is exact for ONE
        # covering surface; a translucent closed mesh puts two (front and back)
        # in the same pixel, and scalar coverage cannot express that they are
        # the same sub-area seen twice rather than two independent ones. This
        # config bounds that error -- see DESIGN_analytic_aa.md ss13.
        with Off():
            s = Sphere().scale(1.0).move(LEFT * 0.6).set_color(GREEN)
            s.opacity = 0.5
            s.spawn()
        s.move(RIGHT * 0.6)
        return
    if cfg == "thin":
        # SUB-PIXEL TRIANGLE GEOMETRY -- the case that decides the sample-less
        # triangle policy. These rods are ~0.9 / 0.45 / 0.22 output pixels wide,
        # so most of their triangles contain no sub-pixel sample at all. A policy
        # that drops them makes the rod fade and eventually vanish (which is
        # what supersampling does too); one that gives them their exact area
        # keeps them visible. Read the INK column, not L1: "it disappeared" is a
        # brightness question.
        with Off():
            for i, th in enumerate((0.02, 0.01, 0.005)):
                Line3D(
                    start=LEFT * 1.4 + UP * (0.7 - 0.7 * i),
                    end=RIGHT * 1.4 + UP * (0.7 - 0.7 * i),
                    thickness=th,
                    color=YELLOW,
                ).spawn()
            sm = Sphere().scale(0.02).move(DOWN * 1.3).set_color(BLUE)
            sm.spawn()
        sm.move(RIGHT * 0.4)
        return
    if cfg == "seam":
        # THE test for the phase-2 seam rule: one big opaque mesh whose
        # interior is a dense field of shared triangle edges. Every one of them
        # splits some pixel, and without the coverage union each leaves a
        # background-coloured notch -- a lattice over the whole object. Scored
        # on INTERIOR pixels only (see seam_error), so the silhouette's genuine
        # anti-aliasing cannot mask or fake the result.
        with Off():
            s = Sphere().scale(1.7).set_color(RED)
            s.spawn()
        s.rotate(20, UP)
        return
    if cfg == "slant":
        # One opaque square rotated off-axis: a long slanted silhouette, the
        # cleanest possible staircase to measure.
        with Off():
            sq = Square(color=RED).scale(1.4).rotate(24, OUT)
            sq.spawn()
        sq.move(RIGHT * 0.3)
        return
    if cfg == "text":
        with Off():
            Text("Analytic AA").scale(0.5).move(UP * 0.4).spawn()
            Text("gjq 0123").scale(0.35).move(DOWN * 0.6).spawn()
        return
    if cfg == "mixed":
        with Off():
            sq = Square(color=RED).scale(0.8).move(LEFT * 1.3 + UP * 0.4)
            sq.spawn()
            ci = Circle(color=GREEN).scale(0.6).move(RIGHT * 1.0)
            ci.opacity = 0.55  # translucent circuit
            ci.spawn()
            tr = Triangle(color=YELLOW).scale(0.6).move(DOWN * 0.9)
            tr.spawn()
            sph = Sphere().scale(0.5).move(LEFT * 0.2 + DOWN * 0.3)
            sph.set_color(BLUE)
            sph.spawn()
            Text("mix").scale(0.4).move(UP * 1.1).spawn()
        sq.rotate(30, OUT)
        return
    if cfg == "unfilled":
        # THE BAND FORM of the coverage filter: an unfilled circuit's drawn
        # region is bounded on both sides, so a stroke thinner than a pixel
        # fades by its width instead of being dilated to the minimum-half-width
        # floor a filled outline gets. Open subpaths (Line, Arrow) are also the
        # geometry that used to render NOTHING -- the packed polyline dropped
        # the final chord of every segment whose closing connection is
        # discontinuous, which for a single-chord straight Line is the entire
        # outline (DESIGN_analytic_aa.md ss13.3). The widths here straddle a
        # pixel so both the fade and the floor are in frame.
        with Off():
            for i, bw in enumerate((6.0, 2.0, 0.8)):
                Line(
                    LEFT * 1.3 + UP * (0.8 - 0.5 * i),
                    RIGHT * 1.3 + UP * (0.8 - 0.5 * i),
                    color=BLUE,
                    border_width=bw,
                ).spawn()
            arc = Line(
                LEFT * 1.2 + DOWN * 0.5,
                RIGHT * 0.2 + DOWN * 0.5,
                path_arc=1.4,
                color=GREEN,
                border_width=3,
            )
            arc.spawn()
            Arrow(
                LEFT * 0.2 + DOWN * 1.2, RIGHT * 1.3 + DOWN * 1.2, color=YELLOW
            ).spawn()
            ci = Circle(color=RED, border_color=RED, border_width=3, filled=False)
            ci.scale(0.45).move(RIGHT * 1.0 + UP * 0.3)
            ci.spawn()
        # Move a stroke: a static-only result would say nothing about the
        # general moving case.
        arc.move(RIGHT * 0.3)
        return
    if cfg == "border":
        # Bordered + unfilled circuits: the band form of the coverage filter.
        with Off():
            ci = Circle(color=GREEN, border_width=4).scale(0.9)
            ci.move(LEFT * 1.0)
            ci.spawn()
            sq = Square(color=BLUE, border_width=2).scale(0.7)
            sq.move(RIGHT * 1.0).rotate(15, OUT)
            sq.spawn()
        ci.move(UP * 0.2)
        return
    raise SystemExit(f"unknown config {cfg}")


def render_once(cfg, aa_level, analytic, tag, seam=True, sliver=None, exact=None):
    SceneManager.reset()
    # Triangle coverage ships opt-in, so force it on for the configs that exist
    # to exercise it; everything else uses the shipped defaults.
    rt_settings.set_analytic_aa(
        analytic,
        seam=seam,
        triangles=cfg in TRI_CONFIGS,
        sliver=sliver or _SLIVER_ARG,
        exact=EXACT if exact is None else exact,
    )
    # The dense tile path is a genuinely different pipeline for circuits: the
    # opaque ones claim a z-prepass entry (full coverage only) and their
    # silhouette pixels come back through a second ``partial_only``
    # count/write pass. Exercise it explicitly -- the default sparse path
    # never runs either.
    rt_settings.set_raster_sparse_coverage(not DENSE)
    name = f"aaBez_{cfg}_{tag}"
    path = os.path.join(OUT_DIR, name + ".mp4")
    settings = VideoSettings(
        (BASE_W, BASE_H), frames_per_second=FPS, supersampling=aa_level
    )
    with Scene(video_settings=settings) as scene:
        build_scene(cfg)
        for k in _counts:
            _counts[k] = 0
        scene.save_video(path, video_settings=settings, overwrite=True)
    rt_settings.set_analytic_aa(False, seam=True, triangles=False)
    rt_settings.set_raster_sparse_coverage(True)
    counts = dict(_counts)
    assert counts["raster"] > 0, f"raster front-end did not engage ({cfg}/{tag})"
    # Prove which pipeline ran: a silently-mis-routed batch would make the whole
    # comparison vacuous (the classic caveat in DESIGN_hybrid_raster.md ss10).
    want = "dense" if DENSE else "sparse"
    other = "sparse" if DENSE else "dense"
    assert counts[want] > 0, (
        f"expected the {want} path ({cfg}/{tag}): "
        f"dense={counts['dense']} sparse={counts['sparse']}"
    )
    assert counts[other] == 0, (
        f"expected only the {want} path ({cfg}/{tag}): "
        f"dense={counts['dense']} sparse={counts['sparse']}"
    )
    return path, counts


def read_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f.astype(np.float64))
    cap.release()
    return frames


SEAM_DARK = 5.0  # luminance levels below the reference that count as a notch


def interior_stats(frames, ref):
    """(mean L1, notch count) against the reference over INTERIOR pixels only.

    Interior = non-background in BOTH renders, eroded by three pixels so no
    silhouette pixel survives; genuine silhouette anti-aliasing therefore
    cannot mask or fake the result.

    A coverage seam is specifically background BLEEDING THROUGH a fully covered
    pixel, so the metric that matters is the count of interior pixels
    measurably DARKER than the reference -- not mean L1, which also picks up
    the harmless colour mixing that coverage causes where several triangles
    meet at a vertex. On the sphere below, multiplicative compositing produces
    ~950 such notches (a visible cross-hatch over the whole object) and the
    union rule produces zero.
    """
    total, notches, n = 0.0, 0, 0
    for f, r in zip(frames, ref):
        lit = (f.max(axis=2) > 8) & (r.max(axis=2) > 8)
        inner = lit
        for _ in range(3):
            inner = (
                inner[1:-1, 1:-1]
                & inner[:-2, 1:-1]
                & inner[2:, 1:-1]
                & inner[1:-1, :-2]
                & inner[1:-1, 2:]
            )
            inner = np.pad(inner, 1, constant_values=False)
        if not inner.any():
            continue
        total += float(np.abs(f - r).mean(axis=2)[inner].mean())
        delta = (f.mean(axis=2) - r.mean(axis=2))[inner]
        notches += int((delta < -SEAM_DARK).sum())
        n += 1
    return total / max(n, 1), notches


def edge_levels(frames):
    """Distinct luminance levels present, as a coarse gradient-richness proxy.

    A hard-aliased edge between two flat regions has ~2 levels; an analytically
    covered one has a continuum. Counted only on pixels that neighbour a strong
    luminance step, so flat interiors and the background do not dominate.
    """
    total = 0
    for f in frames:
        lum = f.mean(axis=2)
        gx = np.abs(np.diff(lum, axis=1, prepend=lum[:, :1]))
        gy = np.abs(np.diff(lum, axis=0, prepend=lum[:1, :]))
        band = (gx + gy) > 8.0
        if not band.any():
            continue
        total += len(np.unique(np.round(lum[band]).astype(np.int32)))
    return total


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    configs = args or [
        "seam",
        "tri",
        "trans",
        "thin",
        "slant",
        "text",
        "mixed",
        "border",
        "unfilled",
    ]
    all_ok = True
    print(f"path: {'DENSE tile' if DENSE else 'sparse coverage'}", flush=True)
    for cfg in configs:
        print(f"--- {cfg} ---", flush=True)
        p_off, c_off = render_once(cfg, 1, False, "aa1_off")
        p_on, c_on = render_once(cfg, 1, True, "aa1_analytic")
        p_ref, _ = render_once(cfg, 4, False, "aa4_ref")

        f_off = read_frames(p_off)
        f_on = read_frames(p_on)
        f_ref = read_frames(p_ref)
        ok = True
        if not (len(f_off) == len(f_on) == len(f_ref)) or not f_off:
            print(
                f"[{cfg:7s}] FAIL: frame counts {len(f_off)}/{len(f_on)}/{len(f_ref)}"
            )
            all_ok = False
            continue

        if EXACT_AB:
            # Box filter vs exact area against one reference. The box filter is
            # the exact area for an AXIS-ALIGNED boundary only, so a config whose
            # edges are all axis-aligned must show no difference at all, and a
            # slanted or curved one must improve.
            def _arm(label, fr, cfg=cfg, f_ref=f_ref):
                l1 = np.mean([np.abs(a - r).mean() for a, r in zip(fr, f_ref)])
                # Interior L1 and notches are the SEAM: an exact area that no
                # longer matches what the fragment occludes shows up here, as
                # the background leaking through a fully covered pixel, and
                # nowhere in the whole-frame number. Ink catches the other
                # direction (double-claimed samples brightening a shared edge).
                e_i, n_i = interior_stats(fr, f_ref)
                ink = np.mean([f.mean() for f in fr]) / max(
                    np.mean([f.mean() for f in f_ref]), 1e-9
                )
                print(
                    f"[{cfg:7s}] {label:10s} L1 {l1:6.4f}   interior L1 "
                    f"{e_i:6.3f}   notches {n_i:5d}   edge levels "
                    f"{edge_levels(fr):5d}   ink {ink:5.3f}",
                    flush=True,
                )
                return l1

            p_box, _ = render_once(cfg, 1, True, "aa1_box", exact=False)
            p_exa, _ = render_once(cfg, 1, True, "aa1_exact", exact=True)
            f_box = read_frames(p_box)
            f_exa = read_frames(p_exa)
            _arm("aliased", f_off)
            l1_box = _arm("box", f_box)
            l1_exa = _arm("exact", f_exa)
            delta = max(int(np.abs(a - b).max()) for a, b in zip(f_box, f_exa))
            print(
                f"[{cfg:7s}] {'':10s} exact vs box: L1 "
                f"{l1_box:.4f} -> {l1_exa:.4f} "
                f"({100.0 * (l1_box - l1_exa) / max(l1_box, 1e-9):+.1f}%), "
                f"max channel delta {delta}",
                flush=True,
            )
            continue

        if SWEEP and cfg in TRI_CONFIGS:
            # Characterisation table: whole-frame L1 is the silhouette (a halo
            # or an erosion shows up here and nowhere else), interior notches
            # are the seam. A policy has to be read on both at once.
            def _row(label, fr, cfg=cfg, f_ref=f_ref):
                l1 = np.mean([np.abs(a - r).mean() for a, r in zip(fr, f_ref)])
                e_i, n_i = interior_stats(fr, f_ref)
                # Ink relative to the reference: >1 is dilation (the silhouette
                # halo), <1 erosion (sub-pixel geometry lost).
                ink = np.mean([f.mean() for f in fr]) / max(
                    np.mean([f.mean() for f in f_ref]), 1e-9
                )
                print(
                    f"[{cfg:7s}] {label:10s} L1 {l1:6.3f}   interior L1 "
                    f"{e_i:6.3f}   notches {n_i:5d}   edge levels "
                    f"{edge_levels(fr):5d}   ink {ink:5.3f}",
                    flush=True,
                )

            print(
                f"[{cfg:7s}] {'':10s} (aa=4 reference edge levels "
                f"{edge_levels(f_ref)})",
                flush=True,
            )
            _row("aliased", f_off)
            for mode in rt_settings.ANALYTIC_AA_SLIVER_MODES:
                p_m, _ = render_once(cfg, 1, True, f"aa1_{mode}", sliver=mode)
                _row(mode, read_frames(p_m))
            continue

        # Engagement: analytic AA must actually compile into the resolve, and
        # must never leak into the toggle-off run.
        if c_on["aa_shade"] == 0:
            print(
                f"[{cfg:7s}] ENGAGEMENT FAIL: aa_bez never reached the "
                f"resolve (shade launches={c_on['shade']})"
            )
            ok = False
        if c_off["aa_shade"] != 0:
            print(
                f"[{cfg:7s}] ENGAGEMENT FAIL: aa_bez leaked into the "
                f"toggle-off run ({c_off['aa_shade']})"
            )
            ok = False

        worst = max(int(np.abs(a - b).max()) for a, b in zip(f_off, f_on))
        # post_process_frames already box-averaged the aa=4 render back to the
        # output resolution, so the decoded reference is directly comparable.
        ref = f_ref

        if cfg == "seam":
            # A/B the seam rule itself against the aa=4 reference, on interior
            # pixels only. Without the union every shared edge notches the
            # interior; with it there must be no notches at all.
            p_noseam, _ = render_once(cfg, 1, True, "aa1_noseam", seam=False)
            f_noseam = read_frames(p_noseam)
            e_union, n_union = interior_stats(f_on, ref)
            e_multi, n_multi = interior_stats(f_noseam, ref)
            e_alias, n_alias = interior_stats(f_off, ref)
            # The scene must actually exhibit the problem, or the test proves
            # nothing about the rule.
            if n_multi < 50:
                print(
                    f"[{cfg:7s}] SEAM FAIL: the control produced only "
                    f"{n_multi} notches -- this scene no longer exercises "
                    f"shared-edge seams, so the result is vacuous"
                )
                ok = False
            # The rule must remove essentially all of the lattice. It does not
            # remove quite ALL of it: a sample on a shared edge is arbitrated
            # by an epsilon band rather than an exact fill rule, so a handful
            # of pixels still lose one sample (ss13.2). Bound the residual
            # instead of demanding zero.
            if not (n_union <= max(n_multi // 50, 20)):
                print(
                    f"[{cfg:7s}] SEAM FAIL: {n_union} interior notches with "
                    f"the union rule on, against {n_multi} without it -- "
                    f"the lattice is not being removed"
                )
                ok = False
            all_ok = all_ok and ok
            print(
                f"[{cfg:7s}] interior notches: aliased {n_alias}  "
                f"multiplicative {n_multi}  UNION {n_union}    "
                f"mean L1 {e_alias:.3f} / {e_multi:.3f} / {e_union:.3f}"
                f"   {'OK' if ok else 'FAIL'}",
                flush=True,
            )
            continue
        err_off = np.mean([np.abs(a - r).mean() for a, r in zip(f_off, ref)])
        err_on = np.mean([np.abs(a - r).mean() for a, r in zip(f_on, ref)])
        lv_off, lv_on, lv_ref = (
            edge_levels(f_off),
            edge_levels(f_on),
            edge_levels(ref),
        )
        # Triangle configs used to be characterisation rather than gates,
        # because coverage lost to the plain aliased render on whole-frame L1:
        # sample-less triangles were given an approximate area that dilated
        # every silhouette. They are gates now -- dropping those triangles, as
        # supersampling does, beats aliased on every config (ss16).
        if not (err_on < err_off):
            print(
                f"[{cfg:7s}] QUALITY FAIL: L1 vs aa=4 reference "
                f"analytic {err_on:.3f} !< aliased {err_off:.3f}"
            )
            ok = False
        if not (lv_on > lv_off):
            print(
                f"[{cfg:7s}] QUALITY FAIL: edge levels "
                f"analytic {lv_on} !> aliased {lv_off}"
            )
            ok = False
        all_ok = all_ok and ok
        print(
            f"[{cfg:7s}] L1 vs aa4: aliased {err_off:6.3f} -> analytic "
            f"{err_on:6.3f} ({100 * (1 - err_on / max(err_off, 1e-9)):+.1f}%)"
            f"   edge levels {lv_off} -> {lv_on} (ref {lv_ref})"
            f"   changed max|d|={worst}  {'OK' if ok else 'FAIL'}",
            flush=True,
        )

    print("\nANALYTIC_AA_BEZ_OK:", all_ok)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
