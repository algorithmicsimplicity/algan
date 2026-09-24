"""Text, semantic formula parts, and an animated native surface texture.

Requires Algan, torch, and a working TeX installation for Tex. Pango and the
requested font are needed for system-font Text; otherwise Algan may use TeX.
All wording, layout, and texture values are placeholders demonstrating calls.
No external image or font file is bundled.
"""

from algan import DOWN, Off, PREVIEW, Scene, Sphere, Sync, Tex, Text, UP
import torch


def make_texture(reverse=False):
    # Native UV texture order: [width, height, R/G/B/glow/opacity].
    values = torch.linspace(0.2, 0.8, 16).view(16, 1)
    if reverse:
        values = 1.0 - values
    texture = torch.zeros(16, 8, 5)
    texture[..., 0] = values
    texture[..., 1] = 0.5
    texture[..., 2] = 1.0 - values
    texture[..., 4] = 1.0
    return texture


def build_scene():
    with Off():
        Text("Sample text", font_size=36).move_to(UP * 2).spawn()
        formula = Tex(r"a", r"+b", r"=c", font_size=36)
        formula.move_to(DOWN * 2).spawn()
        subject = Sphere(radius=0.7, color_texture=make_texture()).spawn()
    with Sync(runtime=1.0):
        subject.color_texture = make_texture(reverse=True)
        formula.get_segment(0).opacity = 0.4
    Scene.wait(0.5)


if __name__ == "__main__":
    build_scene()
    result = Scene.save_video("renders/text_and_texture.mp4", PREVIEW)
    print(result.status, result.output_path)
