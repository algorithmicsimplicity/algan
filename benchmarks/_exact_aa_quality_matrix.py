"""Rendered quality and fallback-engagement gate for exact analytic AA.

Each configuration is rendered three ways: whole-frame classic-ray AA=4
(reference), whole-frame classic-ray AA=2 (fallback-cost baseline), and exact
analytic AA with the requested AA=2 grid used only by sparse fallback pixels.
The script reports
linear image error, warm-ish wall time, fallback pixels/primary paths, and the
classifier's reason counters.  It also leaves every frame in
``benchmarks/_exact_aa_quality_out`` for manual inspection.

Run one or more named cases, or omit names for the full matrix::

    .venv/Scripts/python.exe benchmarks/_exact_aa_quality_matrix.py angle dense
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("ALGAN_PREFETCH_BATCHES", "0")
os.environ.setdefault("ALGAN_ANALYTIC_AA_EXACT_COVERAGE", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algan import (  # noqa: E402
    BLUE,
    DOWN,
    GREEN,
    LEFT,
    OUT,
    RED,
    RIGHT,
    SETTINGS,
    UP,
    WHITE,
    YELLOW,
    Cube,
    MeshPhysicalMaterial,
    MeshStandardMaterial,
    Off,
    Scene,
    Square,
    Text,
    TriangleMesh,
    VideoSettings,
)
from algan.rendering.raytracing import settings as rt_settings  # noqa: E402
from algan.rendering.raytracing import tracer  # noqa: E402
from algan.rendering.raytracing.raster_pipeline import (  # noqa: E402
    get_exact_aa_fallback_counts,
    reset_exact_aa_fallback_counts,
)
from algan.scene_manager import SceneManager  # noqa: E402

OUTPUT = Path(__file__).with_name("_exact_aa_quality_out")
OUTPUT.mkdir(parents=True, exist_ok=True)
PERFORMANCE_MODE = "--performance" in sys.argv
_PERFORMANCE_RESOLUTION = os.environ.get("EXACT_AA_PERF_RESOLUTION", "1280x720")
try:
    _PERFORMANCE_RESOLUTION = tuple(
        int(value) for value in _PERFORMANCE_RESOLUTION.lower().split("x")
    )
except ValueError as exc:
    raise SystemExit("EXACT_AA_PERF_RESOLUTION must look like WIDTHxHEIGHT") from exc
if len(_PERFORMANCE_RESOLUTION) != 2 or min(_PERFORMANCE_RESOLUTION) <= 0:
    raise SystemExit("EXACT_AA_PERF_RESOLUTION must look like WIDTHxHEIGHT")
RESOLUTION = _PERFORMANCE_RESOLUTION if PERFORMANCE_MODE else (256, 144)
CASES = (
    "angle",
    "dense",
    "text",
    "transparency",
    "interpenetration",
    "materials",
)


def _spawn(mob):
    mob.spawn(animate=False)
    return mob


def _dense_grid():
    n = 18
    xs = torch.linspace(-1.7, 1.7, n + 1)
    ys = torch.linspace(-0.95, 0.95, n + 1)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    vertices = torch.stack((xx, yy, torch.zeros_like(xx)), -1).reshape(-1, 3)
    faces = []
    stride = n + 1
    for y in range(n):
        for x in range(n):
            a = y * stride + x
            b = a + 1
            c = a + stride
            d = c + 1
            faces.extend(((a, b, d), (a, d, c)))
    mesh = TriangleMesh(vertices, torch.tensor(faces, dtype=torch.int64))
    mesh.set_color(BLUE).rotate(17, OUT).rotate(24, UP)
    return mesh


def _build(case):
    if case == "angle":
        colors = (RED, GREEN, BLUE, YELLOW)
        for i, angle in enumerate((-37, -11, 19, 43)):
            square = Square(side_length=0.85, color=colors[i], border_width=2)
            square.move(LEFT * 2.1 + RIGHT * (1.4 * i))
            square.rotate(angle, OUT).rotate(18 + 5 * i, UP)
            _spawn(square)
        return

    if case == "dense":
        _spawn(_dense_grid())
        return

    if case == "text":
        _spawn(Text("Exact AA").scale(0.55).move(UP * 0.42))
        _spawn(Text("gjpq 08").scale(0.42).move(DOWN * 0.55).rotate(7, OUT))
        return

    if case == "transparency":
        first = Square(side_length=2.15, color=RED, opacity=0.58)
        first.move(LEFT * 0.35).rotate(24, OUT)
        second = Square(side_length=1.95, color=BLUE, opacity=0.52)
        second.move(RIGHT * 0.35 + UP * 0.15).rotate(-19, OUT)
        _spawn(first)
        _spawn(second)
        return

    if case == "interpenetration":
        first = Cube(side_length=1.75, fill_opacity=1.0).set_color(RED)
        first.move(LEFT * 0.35).rotate(27, UP).rotate(18, OUT)
        second = Cube(side_length=1.65, fill_opacity=1.0).set_color(BLUE)
        second.move(RIGHT * 0.35 + UP * 0.15).rotate(-31, UP).rotate(-13, OUT)
        _spawn(first)
        _spawn(second)
        return

    if case == "materials":
        _spawn(
            Square(side_length=0.9, color=YELLOW)
            .move(LEFT * 0.75 + UP * 0.4 - OUT * 1.7)
            .rotate(18, OUT)
        )
        glass = Cube(side_length=1.35, fill_opacity=1.0).set_color(BLUE)
        glass.move(LEFT * 0.55).rotate(22, UP).set_material(
            MeshPhysicalMaterial(transmission=0.9, roughness=0.02, ior=1.45)
        )
        _spawn(glass)
        mirror = Square(side_length=1.45, color=RED, border_color=WHITE)
        mirror.move(RIGHT * 0.85).rotate(-48, UP).set_material(
            MeshStandardMaterial(metalness=1.0, roughness=0.0)
        )
        _spawn(mirror)
        overlay = Square(side_length=1.2, color=GREEN, opacity=0.42)
        overlay.move(RIGHT * 0.15 + UP * 0.65 + OUT * 0.3).rotate(-17, OUT)
        _spawn(overlay)
        return

    raise ValueError(f"unknown exact-AA case: {case}")


def _render(case, tag, aa, analytic):
    settings = VideoSettings(RESOLUTION, frames_per_second=1, anti_alias_level=aa)
    SceneManager.reset()
    rt_settings.ANALYTIC_AA = analytic
    rt_settings.ANALYTIC_AA_EXACT_COVERAGE = True
    rt_settings.ANALYTIC_AA_FORCE_FALLBACK = False
    # The acceptance reference is the complete ray path that sparse fallback
    # invokes, not the old hybrid first-hit rasterizer whose tilted-circuit
    # coverage this benchmark is specifically replacing.
    rt_settings.HYBRID_RASTER = analytic
    reset_exact_aa_fallback_counts()
    tracer._EXACT_AA_FALLBACK_PIXELS[0] = 0
    tracer._EXACT_AA_FALLBACK_PRIMARY_PATHS[0] = 0
    path = OUTPUT / f"{case}_{tag}_{RESOLUTION[0]}x{RESOLUTION[1]}.png"
    with Scene(video_settings=settings) as scene:
        with Off():
            _build(case)
        start = time.perf_counter()
        scene.save_frame(path, video_settings=settings, overwrite=True)
        elapsed = time.perf_counter() - start
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"failed to read rendered frame {path}")
    return (
        image.astype(np.float64),
        elapsed,
        tracer._EXACT_AA_FALLBACK_PIXELS[0],
        tracer._EXACT_AA_FALLBACK_PRIMARY_PATHS[0],
        get_exact_aa_fallback_counts(),
    )


def main():
    selected = tuple(arg for arg in sys.argv[1:] if not arg.startswith("--"))
    selected = selected or CASES
    unknown = sorted(set(selected) - set(CASES))
    if unknown:
        raise SystemExit(f"unknown case(s): {', '.join(unknown)}")

    snapshot = SETTINGS.snapshot()
    globals_before = (
        rt_settings.ANALYTIC_AA,
        rt_settings.ANALYTIC_AA_EXACT_COVERAGE,
        rt_settings.ANALYTIC_AA_FORCE_FALLBACK,
        rt_settings.HYBRID_RASTER,
    )
    rows = []
    try:
        SETTINGS.raytracing.set(shadows=True, tonemapping=False)
        print(
            f"{'case':18s} {'exact':>9s} {'aa2':>9s} {'ratio':>7s} "
            f"{'t(ex)':>8s} {'t(aa2)':>8s} {'fallback':>10s} reasons",
            flush=True,
        )
        for case in selected:
            if PERFORMANCE_MODE:
                reference = None
            else:
                reference, _t4, *_ = _render(case, "aa4_reference", 4, False)
            _render(case, "aa2_warmup", 2, False)
            baseline, t2, *_ = _render(case, "aa2_baseline", 2, False)
            _render(case, "exact_warmup", 2, True)
            exact, te, fallback, paths, reasons = _render(case, "exact", 2, True)
            if reference is None:
                exact_error = aa2_error = 0.0
                ratio = 1.0
                quality_ok = True
            else:
                exact_error = float(np.abs(exact - reference).mean())
                aa2_error = float(np.abs(baseline - reference).mean())
                ratio = exact_error / max(aa2_error, 1e-12)
                # The rollout contract is deliberately strict: the exact path
                # must match or improve on whole-frame AA=2 against the same
                # AA=4 proxy.  The two-value tolerance belongs to direct
                # fallback parity tests, not to this aggregate quality gate.
                quality_ok = exact_error <= aa2_error
            path_ok = paths == fallback * 4
            rows.append((case, quality_ok, path_ok, te, t2, fallback))
            engaged = ",".join(f"{k}={v}" for k, v in reasons.items() if v)
            print(
                f"{case:18s} {exact_error:9.3f} {aa2_error:9.3f} "
                f"{ratio:7.2f} {te:8.2f} {t2:8.2f} {fallback:10d} "
                f"{engaged or '-'}",
                flush=True,
            )
    finally:
        (
            rt_settings.ANALYTIC_AA,
            rt_settings.ANALYTIC_AA_EXACT_COVERAGE,
            rt_settings.ANALYTIC_AA_FORCE_FALLBACK,
            rt_settings.HYBRID_RASTER,
        ) = globals_before
        SETTINGS.restore(snapshot)
        SceneManager.reset()

    ordinary = [row for row in rows if row[0] in {"angle", "dense", "text"}]
    faster = not ordinary or (
        sum(row[3] for row in ordinary) < sum(row[4] for row in ordinary)
    )
    bad_quality = [case for case, quality, *_ in rows if not quality]
    bad_paths = [case for case, _quality, paths, *_ in rows if not paths]
    if ordinary:
        print(f"ordinary exact faster than full classic-ray AA=2: {faster}")
    if bad_quality:
        print("quality failures:", ", ".join(bad_quality))
    if bad_paths:
        print("path-count failures:", ", ".join(bad_paths))
    speed_failed = PERFORMANCE_MODE and bool(ordinary) and not faster
    return 1 if bad_quality or bad_paths or speed_failed else 0


if __name__ == "__main__":
    with torch.inference_mode():
        raise SystemExit(main())
