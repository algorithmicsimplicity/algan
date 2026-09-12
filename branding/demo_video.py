"""The Algan demo video: three waves pass through "Algorithmic Animation" and leave the logo.

Usage:  python branding/demo_video.py STILLS [outdir]     # key-moment stills at preview size
        python branding/demo_video.py HD [outdir]         # the video (PREVIEW / LD / MD / HD / PRODUCTION)

Beats: write-on; colour wave; glow wave; mirror wave (a chrome copy of the text is revealed
under the plain one, reflecting three solids hidden behind the camera); every glyph except
A-l-g-a-n dissolves and the survivors slide together; the A loses its crossbar to the wave;
the solids fly in and settle into the banner layout, which is the last frame and the logo.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

from algan import *  # noqa: F403

# The vendored Manim is registered by importing algan first, so this import stays below it.
# isort: off
import manim as mn  # noqa: E402

# isort: on

MODE = sys.argv[1] if len(sys.argv) > 1 else "STILLS"
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else "algan_outputs/branding"

# Franklin Gothic Heavy: the thickest face installed here. Thick strokes are what make the
# reflected gallery readable through the letters. Override with argv[3] / argv[4] to compare.
FONT = sys.argv[3] if len(sys.argv) > 3 else "Franklin Gothic"
WEIGHT = sys.argv[4] if len(sys.argv) > 4 else "HEAVY"
CHROME = {"color": Color("#eef1f5"), "metalness": 1.0, "roughness": 0.08}
CUBE_BLUE = Color("#2b52e6")
CONE_GOLD = Color("#f2c85a")
WAVE_BLUE = "#4da3ff"
FLOOR_TOP = -1.5
TITLE = "Algorithmic" + chr(10) + "Animation"
TITLE_WIDTH = 12.8
KEEP = [0, 1, 2, 15, 19]  # A l g . . . a . . . n  ->  "Algan"
LAG, WAVE = 2.0, 1.2  # seconds for a wave to cross the text; band width

# Final (banner) layout, identical to branding/logo_scene.py
WORD_LEFT, WORD_HEIGHT = -7.2, 3.1
# The video is 16:9 and the camera's field of view is vertical, so it sits further back than
# the 16:6 banner render to fit the same composition.
CAM_END, LOOK_END = torch.tensor([-0.3, 2.3, 24.5]), torch.tensor([-0.3, -0.45, 0.0])
# The camera starts at the letters' own height: rays reflecting off letters that stand on a mirror
# floor otherwise dip into the floor before they reach the gallery behind the camera.
CAM_START, LOOK_START = torch.tensor([0.0, -0.2, 21.0]), torch.tensor([0.0, -0.5, 0.0])


def studio_env(back_lon=270.0, H=512, W=1024):
    """Black studio with a bright band just below the horizon behind the camera (chrome text)
    and a softbox up-left (highlights on the solids).
    """
    lat = torch.linspace(90, -90, H).view(H, 1).expand(H, W)
    lon = torch.linspace(0, 360, W).view(1, W).expand(H, W)
    d = (lon - back_lon + 180) % 360 - 180
    env = torch.zeros(H, W, 3)
    env += ((lat.clamp(min=0) / 90) ** 2.2 * 0.10).unsqueeze(-1)
    band = (
        torch.sigmoid((lat + 5) / 1.2)
        * torch.sigmoid((5.5 - lat) / 1.2)
        * torch.exp(-((d / 80) ** 2))
        * 1.3
    )
    env += band.unsqueeze(-1) * torch.tensor([0.92, 0.96, 1.0])
    box = ((d + 25).abs() < 22).float() * ((lat > 30) & (lat < 55)).float() * 3.0
    env += box.unsqueeze(-1)
    env *= (0.15 + 0.85 * torch.sigmoid((lat + 5) / 1.5)).unsqueeze(-1)
    return env


def place_word(text, left, height):
    """Scale a Text to ``height`` and stand it on the floor with its left edge at ``left``."""
    text.scale_to_height(height)
    lo = text.get_bounding_box_min()[0, 0]
    text.move(UP * (FLOOR_TOP + 0.05 - lo[1]) + RIGHT * (left - lo[0]))
    return text


def wave_crossbar(lam):
    """The wave that replaces the A's crossbar, sized from the lambda glyph."""
    lo, hi = lam.get_bounding_box_min()[0, 0], lam.get_bounding_box_max()[0, 0]
    h, w = (hi[1] - lo[1]).item(), (hi[0] - lo[0]).item()
    half = 0.5 * w * 0.66
    ts = np.linspace(-half, half, 64)
    pts = np.stack([ts, 0.13 * np.sin(np.pi * 1.5 * ts / half), np.zeros_like(ts)], -1)
    wave = ManimMob(
        mn.VMobject(stroke_width=24, stroke_color=WAVE_BLUE).set_points_smoothly(pts)
    )
    wave.color = Color(WAVE_BLUE, glow=0.5)
    cx = 0.5 * (lo[0] + hi[0]).item() + 0.07 * w
    wave.move_to(torch.tensor([cx, lo[1].item() + 0.38 * h, 0.05]))
    return wave


Scene.set_environment_map(studio_env(), intensity=1.0, ambient=True)

with Off():
    # The floor stops short of the camera: what lies behind the camera must stay visible to
    # the mirror letters, which see the region just below the horizon there.
    # Paper-thin: the slab's front face is a little vertical mirror facing the camera otherwise.
    floor = Prism(width=80, height=0.02, depth=45.5, color=Color("#07080b"))
    floor.set_material(
        MeshStandardMaterial(color=Color("#07080b"), metalness=0.55, roughness=0.0)
    )
    floor.move(UP * (FLOOR_TOP - 0.01) + IN * 7.25).spawn()

    cam = Scene.get_camera()
    cam.move_to(CAM_START)
    cam.look_at(LOOK_START)

    # The title, twice: a plain layer in front and a chrome layer behind it. Two lines,
    # filling most of the frame width, standing on the floor.
    def place_title(text):
        text.scale_to_width(TITLE_WIDTH)
        lo, hi = text.get_bounding_box_min()[0, 0], text.get_bounding_box_max()[0, 0]
        text.move(UP * (FLOOR_TOP + 0.05 - lo[1]) + LEFT * 0.5 * (lo[0] + hi[0]))
        return text

    top = place_title(Text(TITLE, font=FONT, weight=WEIGHT, font_size=110, color=WHITE))
    top.move(OUT * 0.03)
    mirror = Text(TITLE, font=FONT, weight=WEIGHT, font_size=110)
    mirror.set_material(MeshStandardMaterial(**CHROME))
    place_title(mirror)
    # The camera sits at the block's own height: a reflected ray that dips hits the mirror floor
    # and bounces up, so the gallery behind the camera must straddle the horizon either way.
    lo, hi = mirror.get_bounding_box_min()[0, 0], mirror.get_bounding_box_max()[0, 0]
    cy = 0.5 * (lo[1] + hi[1]).item()
    cam.move_to(torch.tensor([0.0, cy, CAM_START[2].item()]))
    cam.look_at(torch.tensor([0.0, cy, 0.0]))

    # The gallery hidden behind the camera: what the chrome letters reflect. A flat mirror
    # facing the camera shows the region behind it doubled in x, so it spreads wide. The three
    # logo solids live here too, oversized, and fly in at the end.
    sphere = Sphere(radius=4.0, color=WHITE).set_material(
        MeshStandardMaterial(color=WHITE, metalness=1.0, roughness=0.04)
    )
    sphere.move(LEFT * 10 + UP * 0.5 + OUT * 26).spawn()
    cube = Cube(size=6.5, fill_opacity=1.0).set_material(
        MeshStandardMaterial(color=CUBE_BLUE, metalness=0.1, roughness=0.4)
    )
    cube.rotate(38, UP).move(RIGHT * 4 + UP * 0.5 + OUT * 27).spawn()
    cone = Cone(radius=3.6, height=8.0, closed=True).set_material(
        MeshStandardMaterial(color=CONE_GOLD, metalness=0.15, roughness=0.35)
    )
    cone.move(RIGHT * 13 + UP * 1.0 + OUT * 26).spawn()

    grid = NumberPlane(
        x_range=(-24, 24, 2),
        y_range=(-8, 16, 2),
        background_line_style={
            "stroke_color": "#3a4a7a",
            "stroke_width": 3,
            "stroke_opacity": 0.8,
        },
        axis_config={"stroke_color": "#5a6aa0", "stroke_width": 4},
    )
    grid.move(UP * 4 + OUT * 38).spawn()
    gallery = [
        Torus(ring_radius=3.2, tube_radius=1.0, color=TEAL)
        .rotate(65, RIGHT)
        .move(LEFT * 4 + UP * 6.5 + OUT * 30),
        Icosahedron(edge_length=3.2, color=PURPLE)
        .rotate(20, RIGHT)
        .move(RIGHT * 9 + UP * 7 + OUT * 29),
        Sphere(radius=2.4)
        .set_material(
            MeshPhysicalMaterial(color=WHITE, transmission=1.0, ior=1.5, roughness=0.0)
        )
        .move(RIGHT * 1 + UP * 6 + OUT * 27),
        Dodecahedron(edge_length=2.0, color=MAROON)
        .rotate(30, UP)
        .move(LEFT * 14 + UP * 5 + OUT * 31),
        Cylinder(radius=1.6, height=6, color=GREEN)
        .rotate(25, OUT)
        .move(RIGHT * 16 + UP * 4 + OUT * 32),
        Octahedron(edge_length=3.4, color=RED_B)
        .rotate(35, UP)
        .move(LEFT * 3 + UP * 1.0 + OUT * 34),
        Sphere(radius=1.7).set_material(COPPER).move(RIGHT * 13 + UP * 9 + OUT * 31),
        Sphere(radius=1.3, color=YELLOW).move(LEFT * 16 + UP * 10 + OUT * 29),
        Arrow3D(
            start=LEFT * 12 + DOWN * 2 + OUT * 33,
            end=LEFT * 6 + UP * 9 + OUT * 33,
            shaft_radius=0.12,
            tip_length=1.0,
            color=YELLOW,
        ),
        Arrow3D(
            start=RIGHT * 20 + UP * 12 + OUT * 33,
            end=RIGHT * 8 + UP * 2 + OUT * 33,
            shaft_radius=0.12,
            tip_length=1.0,
            color=ORANGE,
        ),
        Prism(width=5, height=1.2, depth=1.2, color=BLUE_D)
        .rotate(-30, OUT)
        .move(RIGHT * 6 + UP * 11 + OUT * 31),
        Cube(size=2.6, fill_opacity=1.0, color=GREEN_D)
        .rotate(45, UP)
        .rotate(35, RIGHT)
        .move(LEFT * 9 + UP * 9.5 + OUT * 33),
    ]
    gallery = Group(gallery).spawn()

    RectAreaLight(
        location=UP * 12 + OUT * 4 + LEFT * 4,
        target=ORIGIN,
        width=6,
        height=6,
        samples=9,
        color=WHITE,
        intensity=30,
    ).spawn()
    # Short-range lights on the hidden gallery (the letters see its lower half). Inverse-square
    # falloff keeps them off the final composition near the origin.
    for x, y in ((-12.0, 2.0), (-2.0, 8.0), (6.0, 1.0), (14.0, 6.0), (2.0, 13.0)):
        PointLight(
            location=RIGHT * x + UP * y + OUT * 21,
            color=WHITE,
            intensity=120,
            decay=2,
            distance=12,
        ).spawn()
    PointLight(
        location=RIGHT * 10 + UP * 4 + OUT * 10, color=Color("#cfe0ff"), intensity=0.5
    ).spawn()

# ---- beat 1: the title writes itself ----------------------------------------------------
top.spawn(False).write(runtime=1.6)
with Off():
    mirror.spawn(False)  # hidden under the now-opaque plain layer
Scene.wait(0.5)
# ---- beats 2-4: three continuous spatial waves, left to right ----------------------------
top.wave_color(
    TEAL,
    new_color=Color(WAVE_BLUE),
    direction=RIGHT,
    lag_duration=LAG,
    wave_length=WAVE,
)
top.wave_color(
    GOLD.set_glow(1.2),
    new_color=Color(WAVE_BLUE).set_glow(0.15),
    direction=RIGHT,
    lag_duration=LAG,
    wave_length=WAVE,
)
top.wave_color(
    Color(WAVE_BLUE).set_glow(0.15).set_opacity(0.0),
    new_color=Color(WAVE_BLUE).set_glow(0.15).set_opacity(0.0),
    direction=RIGHT,
    lag_duration=LAG,
    wave_length=WAVE,
)
Scene.wait(0.8)

# ---- beat 5: everything but A-l-g-a-n dissolves; the survivors slide into the wordmark ----
with Off():
    algan = Text("Algan", font=FONT, weight=WEIGHT, font_size=200).set_material(
        MeshStandardMaterial(**CHROME)
    )
    place_word(algan, WORD_LEFT, WORD_HEIGHT)
    # Centre wordmark + solids on the end camera: cone's right edge sits 3.15 + 0.9 past x0.
    a_lo, a_hi = algan.get_bounding_box_min()[0, 0], algan.get_bounding_box_max()[0, 0]
    total = (a_hi[0] - a_lo[0]).item() + 1.05 + 4.05
    algan.move(RIGHT * ((CAM_END[0].item() - 0.5 * total) - a_lo[0].item()))
    targets = [algan.character_mobs[k].get_center() for k in range(5)]
    grow = (
        algan.character_mobs[1].get_height() / mirror.character_mobs[1].get_height()
    ).item()
with Sync(runtime=1.5):
    with Lag(0.08):
        for i in range(len(mirror.character_mobs)):
            if i not in KEEP:
                mirror.character_mobs[i].opacity = 0.0
    for k, i in enumerate(KEEP):
        v = mirror.character_mobs[i]
        v.scale(grow)
        v.move_to(targets[k])
with Off():
    mirror.despawn()
    algan.spawn(False)

# ---- beat 6: the wave takes the crossbar; the solids fly in; the camera settles ----------
with Off():
    wordmark = Text("Λlgan", font=FONT, weight=WEIGHT, font_size=200).set_material(
        MeshStandardMaterial(**CHROME)
    )
    wordmark.scale_to_height(WORD_HEIGHT)
    wordmark.move(
        algan.character_mobs[1].get_center() - wordmark.character_mobs[1].get_center()
    )
    lam = wordmark.character_mobs[0]
    wave = wave_crossbar(lam)
    word_right = wordmark.get_bounding_box_max()[0, 0][0].item()
    x0 = word_right + 1.05
    finals = {
        "sphere": (RIGHT * x0 + UP * (FLOOR_TOP + 1.0) + OUT * 1.6, 1.0 / 4.0),
        "cube": (RIGHT * (x0 + 1.9) + UP * (FLOOR_TOP + 0.85) + IN * 0.5, 1.7 / 6.5),
        "cone": (RIGHT * (x0 + 3.15) + UP * (FLOOR_TOP + 1.0) + OUT * 1.1, 2.0 / 8.0),
    }
with Sync(runtime=1.8):
    algan = algan.become(wordmark, minimize_movement=True)
    with Seq():
        Scene.wait(0.5)
        wave.spawn()
    with Lag(0.25):
        for mob, key in ((sphere, "sphere"), (cube, "cube"), (cone, "cone")):
            loc, s = finals[key]
            with Sync():
                mob.move_to(loc)
                mob.scale(s)
    cam.move_to(CAM_END)
    cam.look_at(LOOK_END)
    # The rest of the gallery leaves as the solids arrive, so the final logo reflects only
    # the studio environment.
    gallery.despawn()
    grid.despawn()
Scene.wait(2.4)

os.makedirs(OUTDIR, exist_ok=True)
if MODE == "STILLS":
    times = [1.0, 6.3, 9.9, 11.3, 15.5]
    for t in times:
        Scene.save_frame(
            os.path.join(OUTDIR, f"demo_still_{t:04.1f}.png"), PREVIEW, at=t
        )
else:
    quality = {
        "PREVIEW": PREVIEW,
        "LD": LD,
        "MD": MD,
        "HD": HD,
        "PRODUCTION": PRODUCTION,
    }[MODE]
    Scene.save_video(
        os.path.join(OUTDIR, f"algan_demo_{MODE}.mp4"), quality, overwrite=True
    )
print("DONE")
