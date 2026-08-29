"""How many deterministic taps does a glossy reflection need to stop speckling?

``SETTINGS.raytracing.set(glossy_reflection=True)`` spreads a rough reflector's
continuation rays over its GGX lobe instead of sending them all down the mirror
direction. The lobe is sampled by the continuations the fragment already spawns
-- ``ALGAN_ANALYTIC_AA_SECONDARY``, four of them by default -- and four taps
cannot integrate a wide lobe, so the reflected image arrives as four ghost
copies (plain) or an ordered dither (interleaved). That artefact, not the
energy, is why the setting ships off.

The tap count is a live env int, so the question "does raising it fix the
speckle, and what does it cost" is answerable by measurement. The answer is no:
prefiltering does (``DESIGN_glossy_prefilter.md``), and it is what
``glossy_reflection`` selects by default now -- the fan is reachable with
``prefilter=False`` and is what the tap rows below measure. This renders one
scene at a series of tap counts, plus one prefiltered row, and reports per
render:

* **reflection efficiency** -- the mean linear radiance of the reflected floor
  in the sphere over the same floor seen directly. Unit-free, so it can be
  compared against the path tracer's number for the same scene.
* **speckle** -- the RMS of the reflected disc's high-pass residual (the disc
  minus a 5x5 box blur of itself) as a fraction of its mean. A smooth blurred
  reflection has a small residual; ghosts and dither have a large one. Measured
  inside the disc only, away from the silhouette.
* **seconds** -- wall clock for the whole frame.

    <venv-python> benchmarks/renderer_audit/glossy_probe.py
    <venv-python> benchmarks/renderer_audit/glossy_probe.py --taps 1 4 16 64
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
OUT = _HERE / "out" / "glossy"

DEFAULT_TAPS = (4, 8, 16, 32)


def _srgb_to_linear(u8):
    import numpy as np

    c = np.asarray(u8, dtype=np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _measure(path):
    """Reflection efficiency and speckle of ``calib_mirror``'s rough gold ball.

    The sample windows are the ones ``metrics.mirror_reflection`` uses, so the
    efficiency here is comparable with the numbers in ``REPORT.md``: the floor
    strip below the balls, and the lower half of the ball at image x 0.66 (the
    roughness-0.35 metal -- the one glossy sampling changes).
    """
    import cv2
    import numpy as np

    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise SystemExit(f"missing render {path}")
    lin = _srgb_to_linear(im[:, :, :3][:, :, ::-1])
    h, w = lin.shape[:2]
    grey = lin.mean(axis=2)

    direct = float(
        grey[int(0.88 * h) : int(0.96 * h), int(0.40 * w) : int(0.60 * w)].mean()
    )
    cx, cy = 0.66 * w, 0.60 * h
    patch = grey[
        int(cy + 0.03 * h) : int(cy + 0.09 * h),
        int(cx - 0.04 * w) : int(cx + 0.04 * w),
    ]
    refl = float(patch.mean())

    # High-pass residual over the whole ball: the ball window minus a 5x5 box
    # blur of itself, RMS, as a fraction of the window mean. A blurred
    # reflection has a small residual; ghost copies and ordered dither have a
    # large one. Read one kernel radius inside the window so the silhouette
    # does not contribute its own (legitimate) edge.
    k = 5
    ball = grey[int(0.44 * h) : int(0.76 * h), int(cx - 0.09 * w) : int(cx + 0.09 * w)]
    smooth = cv2.blur(ball, (k, k))
    resid = (ball - smooth)[k:-k, k:-k]
    inner = ball[k:-k, k:-k]
    speckle = float(np.sqrt((resid**2).mean()) / max(inner.mean(), 1e-9))
    contrast = float(inner.std() / max(inner.mean(), 1e-9))
    return {
        "reflected": refl,
        "direct": direct,
        "efficiency": refl / max(direct, 1e-9),
        "speckle": speckle,
        "ball_contrast": contrast,
    }


def _crawl(render_batch, scene_path, nudge):
    """How much the image moves when the camera moves half a pixel.

    The dither that makes glossy reflections crawl is fixed in SCREEN space, so
    a sub-pixel camera move slides every surface point into a different cell of
    the pattern while barely moving the geometry. Rendering the same scene twice,
    a half pixel apart, therefore separates the two: whatever changes beyond what
    the geometry's own half-pixel move explains is the pattern, not the picture.

    Reported as the mean absolute difference over the rough ball, in 8-bit
    channel values, for each arm -- so the arms can be compared against each
    other and against the same measurement with glossy off, which is the
    control (a mirror ray's direction is a smooth function of position, so its
    half-pixel difference is the geometry's alone).
    """
    spec = json.loads(Path(scene_path).read_text())
    shifted = OUT / f"{spec['name']}_nudged.json"
    spec = json.loads(json.dumps(spec))
    spec["camera"]["position"][0] += nudge
    spec["camera"]["target"][0] += nudge
    spec["name"] = spec["name"] + "_nudged"
    shifted.write_text(json.dumps(spec))

    out = {}
    for label, taps, glossy, interleave, prefilter in (
        ("glossy off", None, False, None, False),
        ("glossy, prefiltered", None, True, None, True),
        ("glossy, interleaved fan", 8, True, "1", False),
        ("glossy, plain fan", 8, True, "0", False),
    ):
        # Both camera positions in ONE process. The nudge is a number in the
        # scene spec -- it changes no `ti.static` gate, so the pair compiles
        # the same kernels and the arm is still measured against itself. The
        # gates that DO have to differ (glossy, interleave, prefilter) still
        # get a process each, which is the rule this loop exists to respect.
        stem = f"crawl_{label.replace(' ', '_').replace(',', '')}"
        (a, _), (b, _) = render_batch(
            [f"{stem}_a", f"{stem}_b"],
            taps,
            glossy,
            interleave,
            [scene_path, shifted],
            prefilter,
        )
        out[label] = _pair_difference(a, b)
    return out


def _pair_difference(a, b, floor=6.0):
    """Mean absolute difference between two renders, and the same over the
    region's own mean.

    Measured over the pixels either image calls content (any channel above
    ``floor``/255), not over a fixed window: the reflecting region is in a
    different place in every scene, and averaging the background in would divide
    the answer by however much black the frame happens to contain. The ratio is
    the number to compare across arms -- an arm that reflects ten times as much
    light will move ten times as many 8-bit values for the same relative change.
    """
    import cv2
    import numpy as np

    ia = cv2.imread(str(a), cv2.IMREAD_UNCHANGED).astype(np.float64)[..., :3]
    ib = cv2.imread(str(b), cv2.IMREAD_UNCHANGED).astype(np.float64)[..., :3]
    mask = (ia.max(axis=2) > floor) | (ib.max(axis=2) > floor)
    if not mask.any():
        return {"mad": 0.0, "mean": 0.0, "relative": 0.0}
    mad = float(np.abs(ia[mask] - ib[mask]).mean())
    mean = float(ia[mask].mean())
    return {"mad": mad, "mean": mean, "relative": mad / max(mean, 1e-9)}


def _figure(render, scene_path, out_path):
    """A contact sheet of every Algan glossy arm against the path tracer.

    The numbers in the crawl table say the arms differ; this says *how*. Each
    panel is labelled and left at its own exposure -- the point is the shape of
    the reflection, and the arms whose energy is right are already comparable.
    """
    import cv2
    import numpy as np

    spec = json.loads(Path(scene_path).read_text())
    panels = []
    for label, taps, glossy, interleave, prefilter in (
        ("algan: default (mirror share)", None, False, None, False),
        ("algan: glossy, prefiltered", None, True, None, True),
        ("algan: glossy, plain fan", 8, True, "0", False),
        ("algan: glossy, interleaved fan", 8, True, "1", False),
    ):
        path, _ = render(
            f"fig_{len(panels)}", taps, glossy, interleave, scene_path, prefilter
        )
        panels.append((label, cv2.imread(str(path), cv2.IMREAD_UNCHANGED)[..., :3]))
    ref = _HERE / "out" / f"{spec['name']}.three_pathtrace.png"
    if ref.exists():
        panels.insert(
            0,
            (
                "three.js path tracer",
                cv2.imread(str(ref), cv2.IMREAD_UNCHANGED)[..., :3],
            ),
        )
    h, w = panels[0][1].shape[:2]
    cols = 2
    rows = -(-len(panels) // cols)
    sheet = np.zeros((h * rows, w * cols, 3), np.uint8)
    for i, (label, im) in enumerate(panels):
        y, x = (i // cols) * h, (i % cols) * w
        sheet[y : y + h, x : x + w] = im[:h, :w]
        cv2.putText(
            sheet,
            label,
            (x + 8, y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            sheet,
            label,
            (x + 8, y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(out_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--scene", type=Path, default=_HERE / "scenes" / "calib_mirror.json"
    )
    ap.add_argument("--taps", type=int, nargs="+", default=list(DEFAULT_TAPS))
    ap.add_argument("--aa", type=int, default=3)
    ap.add_argument(
        "--crawl",
        type=float,
        default=None,
        metavar="WORLD_UNITS",
        help="also run the half-pixel camera-shift comparison, nudging the "
        "camera by this many world units (0.008 is half a pixel on "
        "calib_mirror). Skips the tap sweep.",
    )
    ap.add_argument(
        "--figure",
        action="store_true",
        help="write a 2x2 contact sheet of the three Algan arms against the "
        "path tracer's render of the same scene (which must already exist in "
        "out/). Skips the tap sweep.",
    )
    args = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []

    def _render_batch(
        suffixes, taps, glossy, interleave=None, scenes=None, prefilter=None
    ):
        """Render several scenes in ONE subprocess; return a (path, seconds) each.

        Every argument that gates kernel compilation -- ``glossy``,
        ``interleave``, ``prefilter``, ``taps`` -- is per-invocation, so one
        call is one setting combination. That is the invariant this batching
        must not break: those reach the kernels as ``ti.static`` gates resolved
        at compile time, and a second *setting* in the same process would
        silently reuse the first one's compiled code. Batching over ``scenes``
        is safe because a scene is data the kernels read at runtime.

        Worth doing because the per-process cost is not the render. A fresh
        interpreter pays ``import algan`` plus a full kernel preparation pass;
        on a 480x360 audit scene that is the large majority of the wall time,
        and it is why the crawl comparison below renders each arm's two camera
        positions together rather than one process each.
        """
        env = dict(os.environ)
        if taps is not None:
            env["ALGAN_ANALYTIC_AA_SECONDARY"] = str(taps)
        if interleave is not None:
            env["ALGAN_GLOSSY_INTERLEAVE"] = interleave
        # ALWAYS explicit. The prefiltered split-sum route is the DEFAULT half
        # of glossy_reflection now, so a tap-fan arm that leaves this unset
        # silently measures the prefilter instead and reports it as the fan.
        env["ALGAN_GLOSSY_PREFILTER"] = "1" if prefilter else "0"
        scenes = [args.scene] if scenes is None else list(scenes)
        cmd = [
            sys.executable,
            str(_HERE / "algan_render.py"),
            *[str(scene) for scene in scenes],
            "--out",
            str(OUT),
            "--suffix",
            *suffixes,
            "--aa",
            str(args.aa),
            "--no-tonemap",
        ]
        if glossy:
            cmd.append("--glossy")
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            raise SystemExit(proc.stderr[-2000:])
        # One JSON line per render, in the order the scenes were given. Other
        # lines on stdout (Taichi's banner, the kernel-preparation notice) are
        # not JSON, so they are skipped rather than counted.
        infos = []
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            infos.append(json.loads(line))
        if len(infos) != len(scenes):
            raise SystemExit(
                f"expected {len(scenes)} renders, parsed {len(infos)} "
                f"from: {proc.stdout[-2000:]}"
            )
        return [(Path(i["output"]), i["seconds"]) for i in infos]

    def _render(suffix, taps, glossy, interleave=None, scene=None, prefilter=None):
        scenes = None if scene is None else [scene]
        return _render_batch([suffix], taps, glossy, interleave, scenes, prefilter)[0]

    if args.figure:
        name = json.loads(Path(args.scene).read_text()).get("name", args.scene.stem)
        out = _HERE / "out" / f"{name}.compare.jpg"
        print(_figure(_render, args.scene, out))
        return

    if args.crawl is not None:
        result = _crawl(_render_batch, args.scene, args.crawl)
        print(
            f"a {args.crawl} world-unit (half pixel) camera nudge moves the "
            "reflecting region by:"
        )
        print(f"{'':24s} {'mad':>7} {'mean':>7} {'mad/mean':>9}")
        for label, v in result.items():
            print(f"{label:24s} {v['mad']:7.3f} {v['mean']:7.2f} {v['relative']:9.4f}")
        (OUT / "glossy_crawl.json").write_text(json.dumps(result, indent=2))
        return

    path, seconds = _render("mirror_off", None, False)
    rows.append(dict(taps=0, glossy=False, seconds=seconds, **_measure(path)))
    # The tap sweep is about the FAN, so every row above pins the route to it.
    # The prefiltered route takes one ray whatever the tap count, so it is one
    # row rather than a column of identical ones -- and it is the row the sweep
    # exists to be compared against.
    path, seconds = _render("mirror_prefilter", None, True, prefilter=True)
    rows.append(
        dict(taps=1, glossy=True, prefilter=True, seconds=seconds, **_measure(path))
    )
    for taps in args.taps:
        path, seconds = _render(f"mirror_glossy{taps}", taps, True)
        rows.append(dict(taps=taps, glossy=True, seconds=seconds, **_measure(path)))

    print(
        f"{'taps':>6} {'efficiency':>11} {'speckle':>9} {'contrast':>9} {'seconds':>8}"
    )
    for r in rows:
        label = "off" if not r["glossy"] else str(r["taps"])
        if r.get("prefilter"):
            label = "pre"
        print(
            f"{label:>6} {r['efficiency']:11.4f} {r['speckle']:9.4f} "
            f"{r['ball_contrast']:9.4f} {r['seconds']:8.1f}"
        )
    (OUT / "glossy_probe.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
