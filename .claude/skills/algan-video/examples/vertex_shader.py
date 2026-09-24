"""Custom PyTorch vertex-shader contract with an animatable parameter.

The arithmetic ignores lighting intentionally; it is not a PBR shader or a
recommended appearance. The output has RGB + glow, NOT RGBA.
"""

from algan import Off, PI, PREVIEW, Scene, Seq, Sphere, easings
import torch


def vertex_modulation(
    memory, vertex_location, vertex_normal, albedo_color,
    camera_location, light_origin, light_color, light_intensity,
    ambient_light_intensity, modulation_frequency=4.0, modulation_phase=0.0,
):
    angle = (vertex_location[..., 0:1] * modulation_frequency
             + modulation_phase)
    weight = 0.5 + 0.5 * torch.cos(angle)
    rgb = albedo_color[..., :3] * weight
    glow = albedo_color[..., 3:4].expand_as(rgb[..., :1])
    return torch.cat((rgb, glow), dim=-1)


def build_scene():
    with Off():
        subject = Sphere(radius=0.8)
        subject.set_shader(vertex_modulation)
        subject.spawn()
    with Seq(runtime=2.0, easing=easings.identity):
        subject.modulation_phase = 2 * PI
    Scene.wait(0.5)


if __name__ == "__main__":
    build_scene()
    result = Scene.save_video("renders/vertex_shader.mp4", PREVIEW)
    print(result.status, result.output_path)
