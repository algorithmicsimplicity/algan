"""Parity check: sorted-material wavefront vs the monolithic shade kernel.

Renders mixed-material scenes (Lambert / Phong / Standard / Basic spheres, a
reflective cylinder, bezier squares, optionally a custom fragment pipeline and
a refractive glass sphere) twice in-process -- material sorting OFF (the
classic monolithic ``wavefront_shade``) and ON (the Cycles-style
peel / sort / per-material ``wf_shade_event`` pipeline) -- and reports the max
/ mean per-pixel abs difference (0-255) per config. The sorted path evaluates
the exact same math per hit, so at AA=1 (save_frame) the two should match to
within a couple of LSBs.

    .venv/Scripts/python.exe benchmarks/_wf_sorted_parity_check.py

ALGAN_PARITY_PN=1 renders the mesh mobs as curved PN patches instead of flat
triangles (exercises the PN event path). A final smoke render checks that a
custom *scatter* (user-controlled ray bouncing, sorted path only) runs.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from algan import (  # noqa: E402
    BLUE,
    DOWN,
    GREEN,
    LEFT,
    OUT,
    RED,
    RIGHT,
    UP,
    WHITE,
    YELLOW,
    Cylinder,
    MeshBasicMaterial,
    MeshLambertMaterial,
    MeshPhongMaterial,
    MeshPhysicalMaterial,
    MeshStandardMaterial,
    SceneManager,
    Sphere,
    Square,
    Sync,
)
from algan.rendering.raytracing import (  # noqa: E402
    set_fragment_shading,
    set_shadows,
)
from algan.rendering.raytracing.settings import (
    set_wavefront_sort_materials,  # noqa: E402
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "_tc_out")
os.makedirs(OUT_DIR, exist_ok=True)

PN_TRIANGLES = os.environ.get("ALGAN_PARITY_PN", "0") == "1"
if PN_TRIANGLES:
    from algan.rendering.raytracing.primitives import RayTracedPNTrianglePrimitive
    from algan.settings.renderer_settings import RENDERER_SETTINGS

    RENDERER_SETTINGS.triangle_primitive = RayTracedPNTrianglePrimitive


def build_mixed(with_custom=False):
    with Sync():
        Sphere().scale(0.9).move(LEFT * 3).set_material(
            MeshLambertMaterial(color=BLUE)
        ).spawn()
        Sphere().scale(0.9).move(LEFT * 1).set_material(
            MeshPhongMaterial(color=RED, specular=0xFFFFFF, shininess=50)
        ).spawn()
        Sphere().scale(0.9).move(RIGHT * 1).set_material(
            MeshStandardMaterial(color=GREEN, metalness=0.8, roughness=0.3)
        ).spawn()
        Sphere().scale(0.9).move(RIGHT * 3).set_material(
            MeshBasicMaterial(color=YELLOW)
        ).spawn()
        # Reflection comes from the material (metalness) since the renderer-
        # side set_reflectivity control was removed by the material-transport
        # rework.
        (
            Cylinder(radius=0.4, height=1.6)
            .move(DOWN * 1.6)
            .set_material(
                MeshStandardMaterial(color=WHITE, metalness=0.6, roughness=0.4)
            )
            .spawn()
        )
        # Two bezier circuits (peeled inline, never material events).
        Square(color=GREEN).scale(0.6).move(UP * 1.8 + OUT * 0.5).spawn()
        Square(color=GREEN).scale(0.6).move(UP * 1.8 + LEFT * 1.4 + OUT * 0.5).spawn()
        if with_custom:
            from algan.rendering.shaders.fragment_shaders import cosine_color
            from algan.rendering.shaders.material_shaders import phong_shader

            Sphere().scale(0.8).move(UP * 1.6 + RIGHT * 3).set_fragment_shader(
                [cosine_color, phong_shader]
            ).spawn()


def build_refract():
    with Sync():
        # Backdrop the glass bends: colored squares + a lit sphere behind.
        Square(color=RED).scale(0.9).move(LEFT * 1.2 - OUT * 1.5).spawn()
        Square(color=BLUE).scale(0.9).move(RIGHT * 1.2 - OUT * 1.5).spawn()
        Sphere().scale(0.7).move(UP * 1.6 - OUT * 1.0).set_material(
            MeshPhongMaterial(color=GREEN, specular=0xFFFFFF, shininess=40)
        ).spawn()
        # Glass comes from the material (ior + transmission) since the
        # renderer-side set_refractive_index control was removed by the
        # material-transport rework.
        (
            Sphere()
            .scale(1.2)
            .set_material(
                MeshPhysicalMaterial(
                    color=WHITE, opacity=0.12, ior=1.5, transmission=1.0
                )
            )
            .spawn()
        )


def render_once(sort_on, frag, shadows, build_fn, tag, **build_kwargs):
    SceneManager.reset()
    set_fragment_shading(frag)
    set_shadows(shadows)
    set_wavefront_sort_materials(sort_on)
    build_fn(**build_kwargs)
    scene = SceneManager.instance()
    out = os.path.join(OUT_DIR, f"wfsort_{tag}.png")
    frames = scene.save_frame(out)
    # save_frame returns CHW float tensors in [0, 1]; take the last.
    return frames[-1].permute(1, 2, 0).float().cpu().numpy() * 255.0


def compare(a, b):
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return d.max(), d.mean(), float((d > 1.0).mean() * 100.0)


def main():
    configs = [
        ("frag", {"frag": True, "shadows": False, "build_fn": build_mixed}),
        ("shadow", {"frag": True, "shadows": True, "build_fn": build_mixed}),
        (
            "custom",
            {
                "frag": True,
                "shadows": False,
                "build_fn": build_mixed,
                "with_custom": True,
            },
        ),
        ("refract", {"frag": True, "shadows": False, "build_fn": build_refract}),
    ]
    all_ok = True
    for name, cfg in configs:
        cfg = dict(cfg)
        build_fn = cfg.pop("build_fn")
        frag, shadows = cfg.pop("frag"), cfg.pop("shadows")
        mono = render_once(False, frag, shadows, build_fn, f"{name}_mono", **cfg)
        srt = render_once(True, frag, shadows, build_fn, f"{name}_sorted", **cfg)
        mx, mn, pct = compare(mono, srt)
        ok = mx <= 2.0
        all_ok = all_ok and ok
        print(
            f"[{name:8s}] max|d|={mx:5.1f}  mean|d|={mn:6.4f}  "
            f">1LSB={pct:5.2f}%  {'OK' if ok else 'MISMATCH'}",
            flush=True,
        )

    # Custom-scatter test: the monolith now supports custom ray bouncing, so
    # the monolith (default) and the sorted pipeline (forced) must AGREE on a
    # custom-scatter scene, and "auto" must route it to the *monolith* (which
    # is faster on built-in materials). A forced-mirror sphere exercises the
    # scatter's reflected branch.
    def build_scatter():
        from algan.rendering.shaders.fragment_shaders import forced_mirror_scatter

        with Sync():
            Sphere().scale(0.9).move(LEFT * 1.5).set_material(
                MeshLambertMaterial(color=BLUE)
            ).spawn()
            Sphere().scale(0.9).move(RIGHT * 1.5).set_fragment_shader(
                forced_mirror_scatter
            ).spawn()

    import algan.rendering.raytracing.tracer as tracer_mod

    sorted_calls = [0]
    _orig_sorted = tracer_mod._raytrace_render_wavefront_sorted

    def _counting_sorted(*a, **k):
        sorted_calls[0] += 1
        return _orig_sorted(*a, **k)

    tracer_mod._raytrace_render_wavefront_sorted = _counting_sorted

    sorted_calls[0] = 0
    mono = render_once(False, True, False, build_scatter, "scatter_mono")
    mono_used_sorted = sorted_calls[0] > 0
    srt = render_once(True, True, False, build_scatter, "scatter_sorted")
    sorted_calls[0] = 0
    render_once("auto", True, False, build_scatter, "scatter_autos")
    auto_used_monolith = sorted_calls[0] == 0
    tracer_mod._raytrace_render_wavefront_sorted = _orig_sorted

    mx, mn, _pct = compare(mono, srt)
    scatter_ok = mx <= 2.0 and (not mono_used_sorted) and auto_used_monolith
    all_ok = all_ok and scatter_ok
    print(
        f"[scatter ] mono==sorted max|d|={mx:5.1f} mean|d|={mn:.4f}  "
        f"mono!=sorted-kernel={not mono_used_sorted}  "
        f"auto->monolith={auto_used_monolith}  "
        f"{'OK' if scatter_ok else 'MISMATCH'}",
        flush=True,
    )
    print("\nPARITY_OK:", all_ok)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
