"""Custom compiled fragment stage composed before built-in lighting.

Run as a real source file. Do NOT add `from __future__ import annotations`:
the compiler needs live ti.template() annotations. Do not call ti.init().
The world-space modulation demonstrates the API, not a prescribed appearance.
"""

from algan import Off, PI, PREVIEW, Scene, Seq, Sphere, easings
from algan.taichi_compat import ti
from algan.rendering.shaders.fragment_shaders import FragmentStage, STAGE_STANDARD


@ti.func
def fragment_modulation(
    pos, view_dir, n_interp, face_n, in_rgb, in_glow,
    params: ti.template(), f, prim, off,
    light_pos: ti.template(), light_col: ti.template(), num_lights,
    shadows: ti.template(), vis, cam_pos,
):
    tm = f % params.shape[0]
    frequency = params[tm, prim, off + 0]
    phase = params[tm, prim, off + 1]
    weight = 0.5 + 0.5 * ti.cos(pos[0] * frequency + phase)
    return ti.math.vec4(
        in_rgb[0] * weight,
        in_rgb[1] * weight,
        in_rgb[2] * weight,
        in_glow,  # This channel is glow, not opacity.
    )


MODULATION = FragmentStage(
    fragment_modulation,
    [
        ("modulation_frequency", 1, 4.0),
        ("modulation_phase", 1, 0.0),
    ],
)


def build_scene():
    with Off():
        subject = Sphere(radius=0.8)
        subject.set_fragment_shader([MODULATION, STAGE_STANDARD])
        subject.roughness = 0.4
        subject.metalness = 0.0
        subject.spawn()
    with Seq(runtime=2.0, easing=easings.identity):
        subject.modulation_phase = 2 * PI
    Scene.wait(0.5)


if __name__ == "__main__":
    build_scene()
    result = Scene.save_video("renders/fragment_shader.mp4", PREVIEW)
    print(result.status, result.output_path)
