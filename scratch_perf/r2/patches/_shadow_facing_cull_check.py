"""Stress scene for the receiver-facing whole-fan shadow cull (A/B arm render).

usage: _shadow_facing_cull_check.py <out.mp4>

The arm comes from ALGAN_SHADOW_RECEIVER_CULL in the environment. Renders one
second of a scene that packs every case most likely to break the cull:

* an open, TWO-SIDED parametric surface with a light BEHIND it (the flip
  decision must agree between stage and shadow path or pixels move);
* a SPOT light (cone factor on top of the facing test);
* a point light with a non-zero ``shadow_radius`` (soft fan: samples can face
  the emitter while its centre faces away -- exactly the case where culling
  would be wrong if the stage terms did not carry max(N.L, 0));
* a RectAreaLight (per-cell emitter rows, same soft-fan reasoning);
* mobs carrying a CUSTOM fragment pipeline and ManimMaterial (both must keep
  the exact fan: a user pipeline reads visibility arbitrarily, and manim's
  offset is nonzero past the horizon);
* lambert / phong / standard / physical / toon solids (the stages the cull is
  sound for).

Output is lossless H.264 RGB; compare arms with benchmarks/_video_diff.py.
"""

import os
import sys

os.environ["ALGAN_USE_DAEMON"] = "0"

import taichi as ti  # noqa: E402

from algan import *  # noqa: E402,F403
from algan.rendering.raytracing.shading_taichi import (  # noqa: E402
    light_vis_index,
)
from algan.rendering.shaders.fragment_shaders import FragmentStage  # noqa: E402

SETTINGS.computing.available_memory_override = 3 * 1024**3
SETTINGS.raytracing.set(shadows=True)


@ti.func
def _user_stage(
    pos,
    view_dir,
    n_interp,
    face_n,
    in_rgb,
    in_glow,
    params: ti.template(),
    f,
    prim,
    off,
    light_pos: ti.template(),
    light_col: ti.template(),
    num_lights,
    shadows: ti.template(),
    vis,
    cam_pos,
):
    """Deliberately visibility-hungry: re-lightens by how LITTLE each light
    is occluded, so any forced all-lit default on this pipeline shows up
    immediately.
    """
    total = 0.0
    for li in range(num_lights):
        for c in ti.static(range(3)):
            total += vis[light_vis_index(li, c)]
    out = in_rgb * (1.0 + 0.05 * total)
    return ti.math.vec4(out[0], out[1], out[2], in_glow)


USER_STAGE = FragmentStage(_user_stage, [])


def build_scene():
    Scene.set_background_color(BLACK)

    with Off():
        # Lights: spot from behind-left, soft-radius point front-right,
        # rect area above-behind.
        SpotLight(
            location=LEFT * 4 + UP * 2 + IN * 4,
            target=ORIGIN,
            cone_angle=50.0,
            penumbra=0.5,
            intensity=1.2,
        ).spawn(animate=False)
        PointLight(
            location=RIGHT * 3 + UP * 1.5 + OUT * 4,
            shadow_radius=0.35,
            intensity=1.0,
        ).spawn(animate=False)
        RectAreaLight(
            location=UP * 4 + IN * 3,
            target=ORIGIN,
            width=2.5,
            height=2.5,
            samples=4,
            intensity=1.0,
        ).spawn(animate=False)

        # Open two-sided sheet lit from behind (light sits at -z = IN; camera
        # looks from +z = OUT, so we see the sheet's unlit side).
        def sheet(u, v):
            return RIGHT * u + UP * v + OUT * 0.6 * ti.sin(2.5 * u) * ti.cos(2.5 * v)

        sheet_mob = Surface(
            coord_function=sheet,
            u_range=(-1.5, 1.5),
            v_range=(-1.0, 1.0),
            resolution=(48, 32),
        ).move(LEFT * 1.6 + DOWN * 0.8)
        sheet_mob.set_material(MeshStandardMaterial(roughness=0.6, metalness=0.1))
        sheet_mob.spawn(animate=False)

        # One cube per built-in lit stage, plus manim and a custom pipeline.
        mats = [
            (MeshLambertMaterial(), LEFT * 0.0 + UP * 1.2),
            (MeshPhongMaterial(shininess=40), RIGHT * 1.6 + UP * 1.2),
            (
                MeshStandardMaterial(roughness=0.35, metalness=0.7),
                LEFT * 1.6 + DOWN * 1.4,
            ),
            (
                MeshPhysicalMaterial(roughness=0.3, clearcoat=0.6, sheen=0.4),
                ORIGIN + DOWN * 1.4,
            ),
            (MeshToonMaterial(bands=4), RIGHT * 1.6 + DOWN * 1.4),
            (ManimMaterial(), UP * 2.6),
        ]
        for mat, loc in mats:
            cube = Cube(side_length=0.9).move(loc)
            cube.set_material(mat)
            cube.spawn(animate=False)

        custom = Cube(side_length=0.9).move(LEFT * 3.2 + UP * 0.4)
        custom.set_fragment_shader(USER_STAGE)
        custom.spawn(animate=False)


if __name__ == "__main__":
    build_scene()

    with Sync(run_time=1.0):
        pass

    r = Scene.save_video(
        sys.argv[1],
        PREVIEW,
        overwrite=True,
        ffmpeg_params=["-c:v", "libx264rgb", "-qp", "0"],
    )
    print("rendered", sys.argv[1], f"wall={r.duration_seconds:.2f}s", flush=True)
