"""A/B the sheet resolve against the shipped fragment walk, lossless.

``DESIGN_sheet_resolve.md`` Phase 2's parity harness. One scene carrying every
feature the Phase-2 route supports -- flat and diced triangle meshes, a torus
fold, text circuits, a translucent stack, a mirror and a glass sphere (both
continuation classes), moving -- rendered under ``ALGAN_SHEET_RESOLVE`` off
and on, both lossless, plus an A/A of each arm for run-to-run determinism.

Engagement is asserted, not assumed (ss0.1 rule 1): the sheet kernel's launch
count is printed per arm, and an ON arm that never launched it is reported as
a FAILURE, not as byte-identity.

Output is EXPECTED to move: the sheet resolve deletes the run-scan budget,
the one-mesh cap and the engagement gate, and shades once per sheet at its
dominant fragment. What this harness bounds is HOW MUCH and WHERE -- read the
worst-frame panel (benchmarks/_diff_frame.py) before concluding anything.

Run:  <venv-python> benchmarks/_sheet_resolve_parity.py [--res ld|md]
      [--scene basic|all]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

os.environ.setdefault("ALGAN_USE_DAEMON", "0")

OUT = REPO / "algan_outputs" / "sheet_parity"


def build_scene(variant="basic"):
    import torch

    from algan import (
        BLUE,
        DARKER_GRAY,
        GREEN,
        IN,
        LEFT,
        PURE_RED,
        RIGHT,
        UP,
        WHITE,
        YELLOW,
        MeshPhysicalMaterial,
        MeshStandardMaterial,
        Off,
        Scene,
        Sphere,
        Square,
        Sync,
        Text,
        Torus,
        TriangleTriangulated,
    )
    from algan import (
        OUT as OUTV,
    )

    Scene.set_background_color(DARKER_GRAY)
    movers = []
    with Off():
        a = (RIGHT * -1.1 + UP * -0.65) * 1.0
        b = (RIGHT * 1.1 + UP * -0.45) * 1.0
        c = (RIGHT * 1.1 + UP * 0.65) * 1.0
        d = (RIGHT * -1.1 + UP * 0.45) * 1.0
        corners = torch.stack([a, b, c, a, c, d]).view(2, 3, 3)
        quad = TriangleTriangulated(corners, color=GREEN).move(
            LEFT * 2.2 + UP * 1.4 + IN * 1.2
        )
        quad.spawn(animate=False)
        movers.append(quad)
        ball = Sphere(radius=0.7, resolution=(96, 48), color=PURE_RED).move(
            RIGHT * 2.0 + UP * 1.3
        )
        ball.spawn(animate=False)
        movers.append(ball)
        ring = (
            Torus(major_radius=0.9, minor_radius=0.3, color=YELLOW)
            .rotate(78, RIGHT)
            .move(UP * 1.6 + IN * 0.5)
        )
        ring.spawn(animate=False)
        movers.append(ring)
        label = Text("Sheets!").scale(0.8).move(IN * 0.2 + UP * 0.1)
        label.spawn(animate=False)
        movers.append(label)
        veil = (
            Square(color=WHITE, opacity=0.35).scale(1.2).move(LEFT * 1.6 + OUTV * 0.9)
        )
        veil.spawn(animate=False)
        movers.append(veil)
        mirror = Sphere(radius=0.55, color=BLUE).move(
            RIGHT * 1.9 + UP * -1.3 + IN * 0.4
        )
        mirror.set_material(MeshStandardMaterial(metalness=1.0, roughness=0.0))
        mirror.spawn(animate=False)
        movers.append(mirror)
        glass = Sphere(radius=0.55, color=WHITE).move(LEFT * 1.9 + UP * -1.3 + IN * 0.4)
        glass.set_material(
            MeshPhysicalMaterial(transmission=0.9, ior=1.5, roughness=0.0)
        )
        glass.spawn(animate=False)
        movers.append(glass)
    if variant == "env":
        # Deterministic gradient env map (float tensor: taken as authored).
        h, w = 8, 16
        xs = torch.linspace(0.0, 1.0, w).view(1, w, 1).expand(h, w, 1)
        ys = torch.linspace(0.15, 0.9, h).view(h, 1, 1).expand(h, w, 1)
        env = torch.cat([xs * ys, ys, (1.0 - xs) * ys], dim=2).contiguous()
        Scene.set_environment_map(env, intensity=1.0, ambient=True)
    with Sync(run_time=1.0):
        for i, mob in enumerate(movers):
            mob.rotate(10 + 3 * i, OUTV if i % 2 else UP)


ARGS_SCENE_PREFIX = ""


def render_arm(tag, on, res, variant="basic"):
    from algan import LD, MD, SETTINGS, Scene
    from algan.rendering.raytracing import sheet_resolve_taichi as srt
    from algan.scene_manager import SceneManager

    quality = {"ld": LD, "md": MD}[res]
    SceneManager.reset()
    SETTINGS.computing.set(available_memory_override=1_400_000_000)
    SETTINGS.raytracing.experimental.set(sheet_resolve=bool(on))
    if variant == "tm":
        # The in-kernel tonemap configuration: the toggle the route gate
        # reads. Applied to BOTH arms so the A/B compares resolves, not
        # tonemap pipelines.
        SETTINGS.raytracing.experimental.set(post_process_tonemap=False)
    launches = {"n": 0}
    orig = srt.sheet_resolve_shade

    def counting(*a, **k):
        launches["n"] += 1
        return orig(*a, **k)

    srt.sheet_resolve_shade = counting
    try:
        scene = SceneManager.instance().current_scene
        scene.set_video_settings(quality)
        build_scene(variant)
        Scene.save_video(
            str(OUT / f"{ARGS_SCENE_PREFIX}{tag}.mp4"),
            quality,
            overwrite=True,
            codec="libx264rgb",
            ffmpeg_params=["-crf", "0"],
        )
    finally:
        srt.sheet_resolve_shade = orig
        SETTINGS.raytracing.experimental.set(sheet_resolve=False)
        if variant == "tm":
            SETTINGS.raytracing.experimental.set(post_process_tonemap=True)
    return launches["n"]


def diff(a, b):
    import cv2
    import numpy as np

    ca, cb = cv2.VideoCapture(str(a)), cv2.VideoCapture(str(b))
    worst = moved = frames = 0
    worst_frame = (0, 0)
    while True:
        ok_a, fa = ca.read()
        ok_b, fb = cb.read()
        if not ok_a or not ok_b:
            break
        d = np.abs(fa.astype(np.int16) - fb.astype(np.int16))
        m = int((d.max(axis=2) > 2).sum())
        if int(d.max()) > worst_frame[0]:
            worst_frame = (int(d.max()), frames)
        worst = max(worst, int(d.max()))
        moved += m
        frames += 1
    ca.release()
    cb.release()
    return worst, moved, frames, worst_frame[1]


def build_matte_scene():
    """The verify scene: matte only, so the sequential oracle's radiometric
    model (alpha + transmission, no reflection lobes) covers everything the
    kernel does at the probed pixels.
    """
    import torch

    from algan import (
        DARKER_GRAY,
        GREEN,
        IN,
        LEFT,
        PURE_RED,
        RIGHT,
        UP,
        WHITE,
        YELLOW,
        Off,
        Scene,
        Sphere,
        Square,
        Text,
        Torus,
        TriangleTriangulated,
    )

    Scene.set_background_color(DARKER_GRAY)
    with Off():
        a = (RIGHT * -1.1 + UP * -0.65) * 1.0
        b = (RIGHT * 1.1 + UP * -0.45) * 1.0
        c = (RIGHT * 1.1 + UP * 0.65) * 1.0
        d = (RIGHT * -1.1 + UP * 0.45) * 1.0
        corners = torch.stack([a, b, c, a, c, d]).view(2, 3, 3)
        TriangleTriangulated(corners, color=GREEN).move(
            LEFT * 2.2 + UP * 1.4 + IN * 1.2
        ).spawn(animate=False)
        Sphere(radius=0.7, resolution=(96, 48), color=PURE_RED).move(
            RIGHT * 2.0 + UP * 1.3
        ).spawn(animate=False)
        Torus(major_radius=0.9, minor_radius=0.3, color=YELLOW).rotate(78, RIGHT).move(
            UP * 1.6 + IN * 0.5
        ).spawn(animate=False)
        Text("Sheets!").scale(0.8).move(IN * 0.2 + UP * 0.1).spawn(animate=False)
        Square(color=WHITE, opacity=0.35).scale(1.2).move(LEFT * 1.6 + UP * -1.2).spawn(
            animate=False
        )


def _verify_frame(quality, probe=None):
    """Render the matte scene's single frame; return the captured streams."""
    from algan import SETTINGS
    from algan.rendering.raytracing import raster_pipeline as rp
    from algan.scene_manager import SceneManager

    captured = {}
    original = rp.prepare_sparse_raster_coverage

    def spy(*args, **kwargs):
        cov = original(*args, **kwargs)
        if cov is None or not cov.get("sheets"):
            return cov
        width = int(kwargs.get("width", args[13]))
        height = int(kwargs.get("height", args[14]))
        ppf = width * height
        t_start = int(kwargs.get("time_start", args[11]))
        offs = cov["sheet_offsets"].cpu().tolist()
        pix = cov["covered_idx"].cpu().tolist()
        refs = cov["sheet_ref"].cpu().tolist()
        covs = cov["sheet_cov"].cpu().tolist()
        msks = cov["sheet_msk"].cpu().tolist()
        caps = cov["sheet_cap"].cpu().tolist()
        for i, p in enumerate(pix):
            lo, hi = offs[i], offs[i + 1]
            f = t_start + p // ppf
            pp = p % ppf
            captured[(pp % width, pp // width, f)] = [
                (refs[j], covs[j], msks[j], caps[j]) for j in range(lo, hi)
            ]
        return cov

    SceneManager.reset()
    SETTINGS.raytracing.experimental.set(sheet_resolve=True)
    rp.prepare_sparse_raster_coverage = spy
    try:
        scene = SceneManager.instance().current_scene
        scene.set_video_settings(quality)
        build_matte_scene()
        scene.save_frame(
            str(OUT / "verify.png"), video_settings=quality, overwrite=True
        )
    finally:
        rp.prepare_sparse_raster_coverage = original
        SETTINGS.raytracing.experimental.set(sheet_resolve=False)
    return captured


def verify_oracle(res, n_probe):
    """Diff the kernel's per-sheet commits against the sequential oracle.

    ``ALGAN_AA_DUMP`` makes the sheet kernel write one row per sheet at a
    probed pixel; the oracle recomputes the pixel from the captured sheet
    stream, taking each sheet's material alpha and transmission share from the
    dump (they are shading results the oracle does not model). Committed rows
    must agree on the claim to f32-vs-f64 noise.
    """
    from algan import LD, MD
    from algan.rendering.raytracing import raster_pipeline as rp
    from algan.rendering.raytracing.raytrace_kernels_taichi import MIN_ALPHA
    from algan.rendering.raytracing.sheets import resolve_pixel_reference

    quality = {"ld": LD, "md": MD}[res]
    captured = _verify_frame(quality)
    # Prefer pixels holding several sheets with partial coverage: the ones
    # where the arithmetic can actually diverge.
    probes = sorted(
        captured.items(),
        key=lambda kv: (
            -len(kv[1]),
            -sum(1 for _r, c, _m, _cp in kv[1] if c < 0.999),
        ),
    )
    probes = [k for k, v in probes if len(v) >= 2][:n_probe]
    worst = 0.0
    rows_checked = 0
    where = None
    for px, py, f in probes:
        os.environ["ALGAN_AA_DUMP"] = f"{px},{py},{f}"
        try:
            import contextlib
            import io

            with contextlib.redirect_stdout(io.StringIO()):
                _verify_frame(quality)
        finally:
            os.environ.pop("ALGAN_AA_DUMP", None)
        rows = rp.LAST_AA_DUMP.get("sheet-resolve")
        if rows is None or not len(rows):
            print(f"  probe ({px},{py},{f}): no dump rows -- SKIPPED")
            continue
        stream = captured[(px, py, f)]
        covs = [c for _r, c, _m, _cp in stream]
        msks = [m for _r, _c, m, _cp in stream]
        bez = [r < 0 for r, _c, _m, _cp in stream]
        caps = [cp for _r, _c, _m, cp in stream]
        alphas = [1.0] * len(stream)
        trans = [0.0] * len(stream)
        frag_rows = [r for r in rows if r[0] >= 0]
        for r in frag_rows:
            q = int(r[0])
            if q < len(stream) and int(r[2]) == 0:
                alphas[q] = float(r[11])
                trans[q] = float(r[13])
        claims, _T = resolve_pixel_reference(
            covs, msks, bez, alphas, trans, caps=caps, min_alpha=MIN_ALPHA
        )
        for r in frag_rows:
            q = int(r[0])
            if int(r[2]) != 0 or q >= len(claims):
                continue
            d = abs(claims[q] - float(r[12]))
            rows_checked += 1
            if d > worst:
                worst = d
                where = (px, py, f, q, claims[q], float(r[12]))
    tag = "PASS" if worst < 2e-5 else "FAIL"
    print(
        f"[{tag}] oracle vs kernel: worst |claim| diff {worst:.2e} over "
        f"{rows_checked} committed sheet rows at {len(probes)} pixels"
    )
    if where and tag == "FAIL":
        px, py, f, q, oc, kc = where
        print(
            f"      worst at ({px},{py},{f}) sheet {q}: oracle {oc:.6f} "
            f"vs kernel {kc:.6f}"
        )
    return 0 if tag == "PASS" else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--res", choices=("ld", "md"), default="ld")
    ap.add_argument(
        "--scene",
        choices=("basic", "env", "tm"),
        default="basic",
        help="scene variant: basic, env-mapped (dense walk vs sparse sheets), "
        "or in-kernel tonemap (post_process_tonemap off in both arms)",
    )
    ap.add_argument(
        "--verify",
        type=int,
        default=0,
        metavar="N",
        help="probe N pixels with ALGAN_AA_DUMP and diff the sheet kernel's "
        "per-sheet commits against the sequential oracle",
    )
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    global ARGS_SCENE_PREFIX
    ARGS_SCENE_PREFIX = "" if args.scene == "basic" else f"{args.scene}_"

    if args.verify:
        return verify_oracle(args.res, args.verify)

    runs = [
        ("off_a", False),
        ("off_b", False),
        ("on_a", True),
        ("on_b", True),
    ]
    launches = {}
    for tag, on in runs:
        print(
            f"-- rendering {tag} (sheet_resolve={'ON' if on else 'off'}, "
            f"scene={args.scene})"
        )
        launches[tag] = render_arm(tag, on, args.res, args.scene)
        print(f"   sheet kernel launches: {launches[tag]}")

    ok = True
    if launches["on_a"] == 0 or launches["on_b"] == 0:
        print("FAILURE: the ON arm never launched the sheet kernel -- the")
        print("route did not engage, so nothing below compares anything.")
        ok = False
    if launches["off_a"] != 0 or launches["off_b"] != 0:
        print("FAILURE: the OFF arm launched the sheet kernel.")
        ok = False

    for name, x, y in (
        ("A/A off", "off_a", "off_b"),
        ("A/A on ", "on_a", "on_b"),
        ("off vs on", "off_a", "on_a"),
    ):
        worst, moved, frames, wf = diff(
            OUT / f"{ARGS_SCENE_PREFIX}{x}.mp4", OUT / f"{ARGS_SCENE_PREFIX}{y}.mp4"
        )
        print(
            f"{name}: max|d| {worst}, moved px (>2) {moved} over {frames} "
            f"frames, worst at frame {wf}"
        )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
