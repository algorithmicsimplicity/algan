"""Exact parity and engagement check for sparse hybrid-raster coverage.

Runs the same hybrid rasterizer with ``raster_sparse_coverage`` off/on.  The
sparse run must discover exact coverage, allocate/resolve only covered rows,
and produce pixel-identical decoded video.

Run:
    .venv/Scripts/python.exe benchmarks/_raster_sparse_coverage_parity.py
    .venv/Scripts/python.exe benchmarks/_raster_sparse_coverage_parity.py size0
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("ALGAN_PREFETCH_BATCHES", "0")

import cv2
import numpy as np
import torch

import algan.rendering.raytracing.raster_pipeline as rp
import algan.rendering.raytracing.tracer as tracer
from algan import (
    BLUE,
    DOWN,
    GRAY,
    GREEN,
    LEFT,
    RED,
    RIGHT,
    UP,
    WHITE,
    AmbientLight,
    Circle,
    Off,
    PointLight,
    RenderSettings,
    Scene,
    SceneManager,
    Sphere,
    Square,
    Sync,
    Triangle,
    render_to_file,
)
from algan.mobs.neural_nets.neural_net import NeuralNetMLPV3
from algan.mobs.shapes_2d import QuadTriangulated
from algan.rendering.raytracing import settings as rt_settings
from algan.rendering.raytracing.primitives import set_fragment_shading
from algan.rendering.raytracing.settings import set_shadows
from algan.rendering.shaders.materials import (
    MeshLambertMaterial,
    MeshPhysicalMaterial,
    MeshStandardMaterial,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "_rt2_out")
SETTINGS = RenderSettings((320, 180), 2)
CONFIGS = ("size0", "tri0", "bez", "split", "shadow", "drain")
_stats = {"prepare": 0, "shade": 0, "composite": 0, "covered": 0, "secondary": 0}


_prepare = rp.prepare_sparse_raster_coverage
_shade = rp.shade_sparse_raster_coverage
_composite = tracer.wf_composite_accum_sparse
_secondary = tracer.wavefront_shade


def _prepare_probe(*args, **kwargs):
    _stats["prepare"] += 1
    result = _prepare(*args, **kwargs)
    if result is not None:
        _stats["covered"] += int(result["num_covered"])
    return result


def _shade_probe(*args, **kwargs):
    _stats["shade"] += 1
    return _shade(*args, **kwargs)


def _composite_probe(*args, **kwargs):
    _stats["composite"] += 1
    return _composite(*args, **kwargs)


def _secondary_probe(*args, **kwargs):
    # Under sparse coverage every wavefront_shade launch comes from the
    # secondary drain (the primary is raster_first_shade), so this counts the
    # bounce iterations the ``drain`` config exists to cover.
    _stats["secondary"] += 1
    return _secondary(*args, **kwargs)


rp.prepare_sparse_raster_coverage = _prepare_probe
rp.shade_sparse_raster_coverage = _shade_probe
tracer.wf_composite_accum_sparse = _composite_probe
tracer.wavefront_shade = _secondary_probe


def _ground(y=-1.4, half=7.0):
    corners = torch.tensor(
        (
            (-half, y, -half),
            (half, y, -half),
            (half, y, half),
            (-half, y, half),
        )
    ).float()
    return QuadTriangulated(corners, color=GRAY)


def build_scene(config):
    if config == "size0":
        with Off():
            Triangle().scale(0).spawn()
        Scene.wait(1)
        return

    if config == "tri0":
        with Off():
            Sphere().scale(0).spawn()
        Scene.wait(1)
        return

    if config == "bez":
        with Off():
            Square(color=RED).scale(0.9).move(LEFT * 1.3).spawn()
            circle = Circle(color=GREEN).scale(0.7).move(RIGHT * 1.2)
            circle.opacity = 0.55
            circle.spawn()
        circle.move(LEFT * 0.5)
        return

    if config == "split":
        with Off():
            Sphere().scale(6).set_color(GREEN).move(UP * 4).spawn()
            glass = Sphere().scale(0.7).move(LEFT * 1.3)
            glass.set_material(
                MeshPhysicalMaterial(transmission=0.9, roughness=0.05, ior=1.5)
            )
            glass.spawn()
            metal = Sphere().scale(0.6).move(RIGHT * 1.3).set_color(RED)
            metal.set_material(MeshStandardMaterial(metalness=0.9, roughness=0.1))
            metal.opacity = 0.5
            metal.spawn()
        with Sync():
            glass.move(RIGHT * 0.4)
            metal.move(LEFT * 0.4 + UP * 0.2)
        return

    if config == "drain":
        # Spheres whose scatter reserves continuation slots, so covered pixels
        # stay active past the primary shade and run the sparse secondary
        # drain (iterations >= 1 of wavefront_shade on compact rows).
        with Off():
            net = NeuralNetMLPV3([3, 3, 3]).spawn()
        net.move(LEFT * 0.4)
        return

    manager = SceneManager.instance()
    manager.light_sources = [
        PointLight(
            location=UP * 6 + RIGHT * 3,
            color=WHITE,
            intensity=1.0,
        ).spawn(animate=False),
        AmbientLight(color=WHITE, intensity=0.3).spawn(animate=False),
    ]
    with Off():
        _ground().spawn()
        sphere = Sphere().scale(0.9).move(LEFT * 1.2 + DOWN * 0.4)
        sphere.set_material(MeshLambertMaterial(color=BLUE))
        sphere.spawn()
    sphere.move(RIGHT * 2.4)


def render(config, sparse, tag):
    SceneManager.reset()
    set_fragment_shading(config in ("split", "shadow"))
    set_shadows(config == "shadow")
    rt_settings.set_hybrid_raster(True)
    rt_settings.set_raster_empty_skip(True)
    rt_settings.set_raster_covered_shade(True)
    rt_settings.set_post_process_tonemap(True)
    rt_settings.set_raster_sparse_coverage(sparse)
    build_scene(config)
    for key in _stats:
        _stats[key] = 0
    name = f"sparse_coverage_{config}_{tag}"
    render_to_file(
        file_name=name,
        output_dir=OUT_DIR,
        output_path="",
        render_settings=SETTINGS,
        file_extension="mp4",
    )
    return os.path.join(OUT_DIR, name + ".mp4"), dict(_stats)


def compare(path_a, path_b):
    cap_a = cv2.VideoCapture(path_a)
    cap_b = cv2.VideoCapture(path_b)
    worst = 0
    changed = 0
    frames = 0
    while True:
        read_a, frame_a = cap_a.read()
        read_b, frame_b = cap_b.read()
        assert read_a == read_b
        if not read_a:
            break
        diff = np.abs(frame_a.astype(np.int16) - frame_b.astype(np.int16))
        worst = max(worst, int(diff.max()))
        changed += int(np.count_nonzero(diff))
        frames += 1
    cap_a.release()
    cap_b.release()
    return frames, worst, changed


def main():
    requested = tuple(sys.argv[1:]) or CONFIGS
    unknown = set(requested) - set(CONFIGS)
    if unknown:
        raise SystemExit(f"Unknown configs: {sorted(unknown)}")
    os.makedirs(OUT_DIR, exist_ok=True)
    for config in requested:
        dense_path, dense_stats = render(config, False, "dense")
        sparse_path, sparse_stats = render(config, True, "sparse")
        for key in ("prepare", "shade", "composite", "covered"):
            assert dense_stats[key] == 0
        if config == "drain":
            # The path that used to crash: a compile-time-invalid ti.static on
            # an ndarray shape made every secondary wavefront_shade launch fail.
            assert sparse_stats["secondary"] > 0, (
                "drain config no longer reaches the sparse secondary shade"
            )
        if config in ("size0", "tri0"):
            # Its invalid/empty materialized bounds prove batch-wide zero
            # coverage before the exact COUNT discovery kernel is needed.
            assert sparse_stats["prepare"] == 0
            assert sparse_stats["covered"] == 0
            assert sparse_stats["shade"] == 0
            assert sparse_stats["composite"] == 0
        else:
            assert sparse_stats["prepare"] > 0
            assert sparse_stats["covered"] > 0
            assert sparse_stats["shade"] > 0
            assert sparse_stats["composite"] > 0
        frames, worst, changed = compare(dense_path, sparse_path)
        print(
            f"{config}: {frames} frames, max diff {worst}, "
            f"changed channels {changed}, sparse={sparse_stats}"
        )
        assert worst == 0
    rt_settings.set_raster_sparse_coverage(True)
    set_fragment_shading(False)
    set_shadows(False)
    print("PASS: sparse coverage engaged and decoded videos are pixel-exact")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
