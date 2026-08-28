"""Acceptance check for a ``RectAreaLight``'s shadow: penumbra or staircase?

An area light that is *integrated* over its emitting surface casts one
**penumbra**. An area light approximated by ``K`` point emitters, each carrying
``1/K`` of the power and each casting its own hard shadow, casts a
**staircase**: the receiver's brightness can only take the ``K + 1`` values
``0/K, 1/K, ... K/K``, one per emitter it can still see. At Algan's shipped
``samples = 4`` that is five levels, and it reads in the frame as a fan of
separate shadows rather than a gradient.

The two are trivially separable numerically, which is what this harness
exploits. It is the automation of the reproduction written out in
``benchmarks/renderer_audit/TASK_area_light_shadow_banding.md``:

* split ``scenes/calib_lights.json`` into one scene per light, because the
  other three lights fill the rect-area shadow and a four-light frame measures
  a contaminated scanline;
* render ``cl_rect_area`` with the fix ON and OFF;
* read a scanline two probe-radii below ``probe_rect`` and report the plateau
  levels, the flatness, and the smallest ``k/K`` grid those levels sit on
  (``shadow_band_probe.py`` does the reading).

**The arms are separate processes on purpose.** ``ALGAN_AREA_LIGHT_SOFT_SHADOWS``
is read into a module global at import, so flipping it inside one process would
leave the first arm's value in place and report its numbers as the second
arm's -- the same trap ``CLAUDE.md`` records for ``ti.static`` gates.

Four things are checked, and the last three are the ones a "fix" that merely
blurs the staircase would fail:

``grid``
    ON must fit **no** small-integer ``k/K`` grid. This is the primary
    criterion: a staircase names its own emitter count, a penumbra names none.

``umbra``
    The darkest point must **converge**, not keep rising. This is deliberately
    not "the umbra must stay as dark as it was", because it does not, and it
    should not: the old ``k x k`` grid of point emitters spans only
    ``(1 - 1/k)`` of the rectangle, so at the shipped ``samples = 4`` it
    shadowed from an emitter **half** the authored width and height (measured:
    a 1.8 x 1.0 rectangle's emitter centres span 0.9 x 0.5). A too-small
    emitter casts a too-large, too-dark umbra. Sampling the authored rectangle
    necessarily lets some light back in, and on ``calib_lights`` that shows as
    the scanline minimum moving 0.009 -> 0.039.

    What separates that from a fix that merely floods the umbra is that it
    **settles**: 0.039 at ``samples = 4``, 0.035 at 9, 0.032 at 16, while
    flatness falls 0.73 -> 0.59 -> 0.54 toward the path tracer's 0.49. So the
    check renders a second, finer ON arm and requires the umbra not to be
    climbing between them.

``radiance``
    With **shadows off**, the two arms must be byte-identical on the area-light
    scene itself. The fix is a visibility change and nothing else: it packs no
    radiance, moves no emitter, and rescales no power fraction. A single
    differing channel here means it leaked into the lighting term -- which is
    the half of this light that ``REPORT.md`` §6.7 leaves open, and must not be
    changed by accident.

``controls``
    ``cl_point`` and ``cl_spot`` must render **byte-identically** between the
    arms. Neither light is an area light, so nothing about them may move; a
    single differing channel means the change reached rows it had no business
    touching.

One ceiling worth knowing before reading any ``--samples`` sweep:
``max_shadow_lights`` is 16 and **every emitter sample spends one slot**, so an
area light past ``samples = 16`` has its surplus rows lit but unshadowed and its
shadow washes out (measured: the scanline minimum jumps to 0.73 at
``samples = 64``). The render warns about it -- see ``truncation.py``'s
``shadow_lights`` counter. That ceiling is what makes "just raise ``samples``"
a poor substitute for this fix rather than an alternative to it.

Usage::

    <venv-python> benchmarks/_area_light_shadow_check.py
    <venv-python> benchmarks/_area_light_shadow_check.py --samples 4 9 16

``--samples`` re-runs the ON arm at other emitter counts, which is how you see
the shape of the trade: cost is linear in ``samples`` for both lighting and
shadow rays, and the point of the fix is that the banding is gone at the
*shipped default* rather than only asymptotically.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_AUDIT = _ROOT / "benchmarks" / "renderer_audit"
_CALIB = _AUDIT / "scenes" / "calib_lights.json"

#: The one scene under test, and the two that must not move.
_SUBJECT = "rect_area"
_CONTROLS = ("point", "spot")

#: Emitter count for the convergence arm. 16 is ``max_shadow_lights``, so it
#: is the finest grid whose rows all still cast shadow -- past it the surplus
#: rows are lit but unshadowed and the measurement means nothing.
_FINE = 16


def _split_scenes(dest: Path, samples: int | None) -> dict[str, Path]:
    """Write one single-light scene per light in ``calib_lights.json``.

    The full four-light frame cannot show the defect: the other three lights
    fill the rect-area shadow and the probe reads a washed-out ramp that fits
    no grid -- an honest reading of a contaminated measurement.
    """
    base = json.loads(_CALIB.read_text())
    out = {}
    for light in base["lights"]:
        kind = light["type"]
        light = dict(light)
        if kind == _SUBJECT and samples is not None:
            light["samples"] = samples
        spec = dict(base, name=f"cl_{kind}", lights=[light])
        path = dest / f"cl_{kind}.json"
        path.write_text(json.dumps(spec, indent=1))
        out[kind] = path
    return out


def _render(scene: Path, out_dir: Path, arm_on: bool, shadows: bool = True) -> Path:
    """Render one scene in its own process with the flag set for this arm."""
    env = dict(os.environ)
    env["ALGAN_AREA_LIGHT_SOFT_SHADOWS"] = "1" if arm_on else "0"
    # A warm daemon does not adopt import-time environment variables and would
    # serve the other arm's module globals; algan_render.py sets this too.
    env["ALGAN_USE_DAEMON"] = "0"
    cmd = [
        sys.executable,
        str(_AUDIT / "algan_render.py"),
        str(scene),
        "--out",
        str(out_dir),
        "--no-tonemap",
    ]
    if not shadows:
        cmd.append("--no-shadows")
    proc = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        # Captured rather than inherited so a passing run reads as a table
        # instead of six copies of Taichi's banner -- but a failing one has to
        # show what the renderer said.
        sys.stderr.write(proc.stdout + proc.stderr)
        raise SystemExit(f"render of {scene.name} failed ({proc.returncode})")
    return out_dir / f"{scene.stem}.algan.png"


def _probe(scene: Path, image: Path) -> tuple[str, dict]:
    """Run ``shadow_band_probe`` on one image and parse what it reports."""
    proc = subprocess.run(
        [
            sys.executable,
            str(_AUDIT / "shadow_band_probe.py"),
            str(scene),
            "--object",
            "probe_rect",
            "--images",
            str(image),
            "--labels",
            "algan",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    text = proc.stdout
    found = {}
    m = re.search(r"min ([\d.]+)\s+flatness ([\d.]+)", text)
    if m:
        found["min"] = float(m.group(1))
        found["flatness"] = float(m.group(2))
    m = re.search(r"plateau levels below [\d.]+: \[(.*?)\]", text)
    if m:
        found["levels"] = [float(v) for v in m.group(1).split(",") if v.strip()]
    m = re.search(r"sit on a k/(\d+) grid", text)
    found["grid_k"] = int(m.group(1)) if m else None
    return text, found


def _max_channel_diff(a: Path, b: Path) -> int:
    """Largest per-channel difference between two rendered PNGs."""
    import numpy as np

    sys.path.insert(0, str(_AUDIT))
    from material_probe import load_rgb  # noqa: PLC0415  (path set above)

    ia = (load_rgb(a) * 255.0).round().astype(np.int32)
    ib = (load_rgb(b) * 255.0).round().astype(np.int32)
    if ia.shape != ib.shape:
        return 255
    return int(np.abs(ia - ib).max())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--samples",
        type=int,
        nargs="*",
        default=None,
        help="extra emitter counts to re-run the ON arm at (default: the "
        "scene's own, which is the shipped default of 4)",
    )
    ap.add_argument(
        "--keep",
        type=Path,
        default=None,
        help="directory to keep the renders in (default: a temp dir)",
    )
    args = ap.parse_args(argv)

    work = (
        Path(args.keep) if args.keep else Path(tempfile.mkdtemp(prefix="area_shadow_"))
    )
    work.mkdir(parents=True, exist_ok=True)
    failures = []
    try:
        scenes = _split_scenes(work, None)

        print("=" * 72)
        print("SUBJECT: cl_rect_area, probe_rect -- is the shadow a penumbra?")
        print("=" * 72)
        readings = {}
        for arm in ("off", "on"):
            out_dir = work / arm
            out_dir.mkdir(exist_ok=True)
            image = _render(scenes[_SUBJECT], out_dir, arm_on=(arm == "on"))
            text, found = _probe(scenes[_SUBJECT], image)
            readings[arm] = found
            print(f"\n--- ALGAN_AREA_LIGHT_SOFT_SHADOWS={int(arm == 'on')}")
            print("\n".join(text.splitlines()[3:]))

        off, on = readings["off"], readings["on"]
        if on.get("grid_k") is not None:
            failures.append(
                f"grid: ON still sits on a k/{on['grid_k']} grid -- the "
                "staircase is finer but it is still a staircase"
            )

        print()
        print("=" * 72)
        print(f"UMBRA: does it converge? (ON at the scene's samples vs {_FINE})")
        print("=" * 72)
        fine_dir = work / "fine"
        fine_dir.mkdir(exist_ok=True)
        fine = _split_scenes(fine_dir, _FINE)[_SUBJECT]
        _, fine_read = _probe(fine, _render(fine, fine_dir, arm_on=True))
        print(
            f"  ON, scene's samples: min {on.get('min'):.3f}  "
            f"flatness {on.get('flatness'):.2f}"
        )
        print(
            f"  ON, samples={_FINE}:{'':9}min {fine_read.get('min'):.3f}  "
            f"flatness {fine_read.get('flatness'):.2f}"
        )
        print(
            f"  (for reference, the arm this replaces: OFF min {off.get('min'):.3f}, "
            f"flatness {off.get('flatness'):.2f})"
        )
        if fine_read.get("min", 0.0) > on.get("min", 0.0) + 0.02:
            failures.append(
                f"umbra: still climbing -- {on['min']:.3f} at the scene's "
                f"samples, {fine_read['min']:.3f} at {_FINE}. A converging fix "
                "settles; one that floods the umbra keeps getting brighter"
            )

        print()
        print("=" * 72)
        print("RADIANCE: shadows OFF, the arms must be byte-identical")
        print("=" * 72)
        no_shadow = work / "noshadow"
        no_shadow.mkdir(exist_ok=True)
        a = _render(scenes[_SUBJECT], no_shadow / "off", arm_on=False, shadows=False)
        b = _render(scenes[_SUBJECT], no_shadow / "on", arm_on=True, shadows=False)
        diff = _max_channel_diff(a, b)
        print(f"  cl_rect_area, --no-shadows: max channel difference = {diff}")
        if diff != 0:
            failures.append(
                f"radiance: the arms differ by {diff} channel values with "
                "shadows off -- the fix reached the lighting term, which it "
                "must not (that is REPORT.md 6.7's separate defect)"
            )

        print()
        print("=" * 72)
        print("CONTROLS: cl_point and cl_spot must be byte-identical")
        print("=" * 72)
        for kind in _CONTROLS:
            a = _render(scenes[kind], work / "off", arm_on=False)
            b = _render(scenes[kind], work / "on", arm_on=True)
            diff = _max_channel_diff(a, b)
            print(f"  cl_{kind}: max channel difference between arms = {diff}")
            if diff != 0:
                failures.append(
                    f"controls: cl_{kind} moved by {diff} channel values -- a "
                    "non-area light must not see this change at all"
                )

        for extra in args.samples or []:
            print()
            print("=" * 72)
            print(f"ON arm at samples={extra}")
            print("=" * 72)
            more = _split_scenes(work / f"s{extra}", extra)
            (work / f"s{extra}").mkdir(parents=True, exist_ok=True)
            image = _render(more[_SUBJECT], work / f"s{extra}", arm_on=True)
            text, _ = _probe(more[_SUBJECT], image)
            print("\n".join(text.splitlines()[3:]))

        print()
        if failures:
            print("FAIL")
            for line in failures:
                print(f"  - {line}")
        else:
            print(
                "PASS: no k/K grid on the ON arm, the umbra converges, the "
                "unshadowed render is byte-identical, and both non-area "
                "controls are byte-identical."
            )
        return 1 if failures else 0
    finally:
        if args.keep is None:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
