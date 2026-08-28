"""Continuation-pool demand of a glossy reflector.

The glossy lobe reuses the continuations a fragment ALREADY spawns -- the same
``_sec_positions`` taps, redirected -- so pool demand should be bit-for-bit the
same as before it existed. This proves that rather than asserting it: it hooks
``raster_pipeline.raster_first_shade`` to read ``rs_alloc[0]`` (which keeps
counting past capacity, so it reports true demand even on an overflow) against
the covered-pixel count, and reads ``tracer._WAVEFRONT_POOL_RETRIES``.

A retry is the failure that presents as nothing but a slow render: an overflow
DISCARDS the finished tile and re-renders it (DESIGN_analytic_aa.md ss19.2a).

Run: .venv/Scripts/python.exe benchmarks/_glossy_pool_check.py
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("ALGAN_PREFETCH_BATCHES", "0")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402

from algan import (  # noqa: E402
    DOWN,
    GRAY_A,
    OUT,
    RED,
    RIGHT,
    UP,
    WHITE,
    Off,
    RenderSettings,
    SceneManager,
    Sphere,
    Text,
    render_to_file,
)
from algan.rendering.lights import AmbientLight, DirectionalLight  # noqa: E402
from algan.rendering.raytracing import (
    raster_pipeline,  # noqa: E402
    set_fragment_shading,  # noqa: E402
    tracer,  # noqa: E402
)
from algan.rendering.raytracing import settings as rt_settings  # noqa: E402
from algan.rendering.shaders.materials import MeshStandardMaterial  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "_rt2_out")
os.makedirs(OUT_DIR, exist_ok=True)

STATS = []


def _hook():
    real = raster_pipeline.raster_first_shade

    def wrapped(*args, **kwargs):
        out = real(*args, **kwargs)
        # Tail of the signature is
        #   ..., pix_accum, rs_alloc, frag_shadow_id, z_shadow_id, shadow_vis
        # so rs_alloc is args[-4] and pix_accum args[-5]. Shape-checked rather
        # than trusted: getting this off by one reports -0.5 slots per pixel,
        # which reads as "no demand" instead of as a broken probe.
        accum, alloc = args[-5], args[-4]
        assert alloc.ndim == 1, (
            f"rs_alloc probe hit the wrong rank: {tuple(alloc.shape)}"
        )
        assert alloc.numel() == 2, (
            f"rs_alloc probe hit the wrong size: {tuple(alloc.shape)}"
        )
        assert accum.ndim == 2, (
            f"pix_accum probe hit the wrong rank: {tuple(accum.shape)}"
        )
        assert accum.shape[1] == 7, (
            f"pix_accum probe hit the wrong width: {tuple(accum.shape)}"
        )
        # rs_alloc[0] starts at num_primary (the pool sits above the primaries),
        # so the continuations this launch actually wanted is the excess.
        STATS.append(
            (
                int(alloc[0].item()) - int(accum.shape[0]),
                int(accum.shape[0]),
                int(alloc[1].item()),
            )
        )
        return out

    raster_pipeline.raster_first_shade = wrapped


def build(roughness):
    with Off():
        AmbientLight(color=WHITE, intensity=0.3).spawn(animate=False)
        DirectionalLight(
            location=RIGHT * 4 + UP * 4 + OUT * 3, color=WHITE, intensity=1.0
        ).spawn(animate=False)
        Text("MATERIAL STUDY", font_size=52, weight="BOLD", color=WHITE).move(
            UP * 1.55
        ).spawn()
        Text("Standard", font_size=24, color=GRAY_A).move(DOWN * 1.4).spawn()
        ball = Sphere(radius=0.48).scale(2.2)
        ball.set_material(
            MeshStandardMaterial(color=RED, roughness=roughness, metalness=0.75)
        )
        ball.spawn()


def run(roughness, glossy):
    STATS.clear()
    tracer._WAVEFRONT_POOL_RETRIES[0] = 0
    SceneManager.reset()
    set_fragment_shading(True)
    rt_settings.set_analytic_aa(True, bezier=True, triangles=True)
    rt_settings.set_glossy_reflection(glossy)
    build(roughness)
    render_to_file(
        file_path=os.path.join(
            OUT_DIR, f"glossyPool_r{roughness:g}_{'on' if glossy else 'off'}.mp4"
        ),
        video_settings=RenderSettings((864, 486), 1, super_sampling_anti_aliasing=1),
    )
    slots = sum(s[0] for s in STATS)
    covered = sum(s[1] for s in STATS)
    overflow = max((s[2] for s in STATS), default=0)
    return slots, covered, overflow, tracer._WAVEFRONT_POOL_RETRIES[0]


def main():
    _hook()
    print(
        f"{'roughness':>9s} {'glossy':>7s} {'slots/covered px':>17s} "
        f"{'overflow flag':>14s} {'pool retries':>13s}"
    )
    for roughness in (0.0, 0.18, 0.6, 1.0):
        for glossy in (False, True):
            slots, covered, overflow, retries = run(roughness, glossy)
            print(
                f"{roughness:9.2f} {str(glossy):>7s} "
                f"{slots / max(covered, 1):17.3f} {overflow:14d} "
                f"{retries:13d}",
                flush=True,
            )
    rt_settings.set_glossy_reflection(True)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
