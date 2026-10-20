"""The Algan demo video: "Algorithmic animation" turns to chrome, keeps its acronym, and becomes
the logo.

Usage:  python branding/demo_video.py STILLS [outdir]     # key-moment stills at preview size
        python branding/demo_video.py HD [outdir]         # the video (PREVIEW / LD / MD / HD / PRODUCTION)

Beats, with the camera moving throughout:
  0.0  "Algorithmic animation" writes itself on two lines while the camera drifts sideways.
  2.5  One blue wave sweeps through and leaves the letters a mirror; they reflect three large
       solids and a grid standing behind the camera, which slide across them as it orbits.
  5.5  The letters outside A-L-G and A-N fade out in a wave, left to right, while the five
       survivors slide into "Algan"; the gallery fades out of their reflections as they go,
       so the mirror settles into the logo's chrome.
  7.7  The A's crossbar lifts off as a straight bar, bends into y = sin x on faint axes, and
       settles back into the A, now a lambda, as the wave.
 10.5  The sphere rolls in and the cube and cone drop onto the mirror while the camera cranes
       up to the banner framing (branding/logo_scene.py), which is the logo.
 12.9  A tagline fades in above the logo, then everything fades to black.

The logo itself -- typeface, materials, studio, layout -- comes from branding/brand.py.
The y = sin x label is typeset with LaTeX, so a TeX installation is needed.
"""

from __future__ import annotations

import math
import os
import sys

import brand
import torch

from algan import *  # noqa: F403

MODE = sys.argv[1] if len(sys.argv) > 1 else "STILLS"
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else "algan_outputs/branding"

TITLE = "Algorithmic" + chr(10) + "animation"
TITLE_WIDTH = 12.8
KEEP = [0, 1, 2, 11, 12]  # A l g . . . . . . . . a n  ->  "Algan"
WORD_HEIGHT = 3.1
ASPECT = 16 / 9
TAGLINE = "Programmatic animation for explanatory mathematics"
FADE = 0.7


def at(x, y, z):
    return torch.tensor([float(x), float(y), float(z)])


Scene.set_environment_map(brand.studio_env(), intensity=1.0, ambient=True)

with Off():
    # Long enough that its front edge never comes into shot (it runs well behind the camera),
    # with the gallery standing on it back there.
    brand.make_floor(depth=150.0, front=70.0).spawn()

    # ---- the final logo's layout, which every later beat aims at ---------------------------
    wordmark = brand.wordmark("Λlgan", WORD_HEIGHT)
    brand.float_word(wordmark, 0.0)
    left, right = brand.logo_extent(wordmark, WORD_HEIGHT)
    wordmark.move(LEFT * 0.5 * (left + right))
    word_right = wordmark.get_bounding_box_max()[0, 0][0].item()
    crossbar = brand.Crossbar(wordmark.character_mobs[0])
    targets = [wordmark.character_mobs[k].get_center() for k in range(5)]
    cam = Scene.get_camera()
    look_end = at(0.0, brand.FLOOR_TOP + 1.45, 0.0)
    cam_end = brand.frame_camera(cam, look_end, right - left, ASPECT, 8.0, margin=1.22)
    # While the crossbar plays on its axes above the word, the frame centres on the word and
    # looks a little higher; the solids, arriving later, take the space on the right.
    lo, hi = (
        wordmark.get_bounding_box_min()[0, 0],
        wordmark.get_bounding_box_max()[0, 0],
    )
    word_centre = 0.5 * (lo[0] + hi[0]).item()
    look_mid = at(word_centre, brand.FLOOR_TOP + 2.9, 0.0)
    cam_mid = brand.frame_camera(cam, look_mid, right - left, ASPECT, 4.0, margin=1.1)

    # ---- the title, twice: a plain layer in front and a chrome layer behind it -------------
    def place_title(text):
        text.scale_to_width(TITLE_WIDTH)
        lo, hi = text.get_bounding_box_min()[0, 0], text.get_bounding_box_max()[0, 0]
        text.move(UP * (brand.FLOOR_TOP + 0.05 - lo[1]) + LEFT * 0.5 * (lo[0] + hi[0]))
        return text

    top = place_title(
        Text(TITLE, font=brand.FONT, weight=brand.WEIGHT, font_size=110, color=WHITE)
    )
    top.move(OUT * 0.03)
    mirror = place_title(
        Text(TITLE, font=brand.FONT, weight=brand.WEIGHT, font_size=110)
    )
    mirror.set_material(brand.chrome())
    lo, hi = mirror.get_bounding_box_min()[0, 0], mirror.get_bounding_box_max()[0, 0]
    cy = 0.5 * (lo[1] + hi[1]).item()
    look_title = at(0.0, cy, 0.0)
    # The camera starts at the letters' height: flat chrome facing it reflects what stands
    # behind it, and a camera above the letters would see mostly floor in them.
    cam.move_to(at(-1.8, cy + 0.35, 21.0))
    cam.look_at(look_title)

    # ---- the gallery behind the camera: what the chrome title reflects ---------------------
    # A flat mirror facing the camera shows the region behind it, magnified with distance, so a
    # few large, recognisable things read better than many small ones: the logo's own solids,
    # oversized, and a bright grid behind them.
    # The sphere is checkered: a plain white one, this close to its lights, fills the letters
    # that reflect it with flat white, and they read as painted rather than mirrored.
    gallery = [
        Sphere(
            radius=3.0,
            color_texture=get_checkerboard(
                (Color("#e9edf4"), Color(brand.DEEP_BLUE)), resolution=(8, 6)
            ),
        )
        .set_material(MeshStandardMaterial(color=WHITE, roughness=0.3))
        .move(at(-10.0, brand.FLOOR_TOP + 3.0, 38.0)),
        Cube(size=5.0, fill_opacity=1.0)
        .set_material(brand.blue_lacquer())
        .rotate(30, UP)
        .move(at(0.5, brand.FLOOR_TOP + 2.5, 42.0)),
        Cone(radius=3.0, height=7.0, closed=True)
        .set_material(brand.satin_gold())
        .move(at(10.5, brand.FLOOR_TOP + 3.5, 38.0)),
        NumberPlane(
            x_range=(-30, 30, 2),
            y_range=(-2, 18, 2),
            background_line_style={
                "stroke_color": "#4a68b8",
                "stroke_width": 4,
                "stroke_opacity": 0.9,
            },
            axis_config={"stroke_color": "#7d9cf0", "stroke_width": 5},
        ).move(at(0.0, brand.FLOOR_TOP + 10.0, 52.0)),
    ]
    for mob in gallery:
        mob.spawn()
    # Short-range lights on the gallery's camera-facing side. Their falloff keeps them off
    # everything near the origin.
    for x in (-10.0, 0.5, 10.5):
        PointLight(
            location=at(x, 6.0, 31.0), color=WHITE, intensity=140, decay=2, distance=14
        ).spawn()
    brand.make_lights()

    # ---- the pieces that arrive later -----------------------------------------------------
    sphere, cube, cone = brand.make_solids(WORD_HEIGHT)
    layout = brand.solid_layout(word_right, WORD_HEIGHT)
    tagline = Text(
        TAGLINE,
        font=brand.FONT,
        weight=brand.WEIGHT,
        font_size=60,
        color=Color("#9aaccc"),
    )
    tagline.scale_to_width(0.55 * (right - left))
    # Above the logo, high enough that its reflection in the floor falls below the frame.
    tagline.move_to(at(0.0, brand.FLOOR_TOP + WORD_HEIGHT + 2.4, 0.0))
    tagline.set_material(MeshBasicMaterial(color=Color("#9aaccc")))

# ---- 0.0: the title writes itself while the camera drifts --------------------------------
with Sync():
    top.spawn(False).write(runtime=1.8)
    with Sync(runtime=2.5, easing=easings.ease_in_out_sine):
        cam.move_to(at(-0.6, cy + 0.4, 21.0))
        cam.look_at(look_title)
with Off():
    mirror.spawn(False)  # hidden under the now-opaque plain layer
Scene.wait(0.2)

# ---- 2.5: one wave leaves the letters chrome; the camera keeps orbiting -------------------
with Sync():
    top.wave_color(
        Color(brand.BRAND_BLUE).set_glow(0.35),
        new_color=Color(brand.BRAND_BLUE).set_opacity(0.0),
        direction=RIGHT,
        lag_duration=1.6,
        wave_length=1.2,
    )
    with Sync(runtime=2.8, easing=easings.identity):
        cam.move_to(at(1.9, cy + 0.5, 20.0))
        cam.look_at(look_title)
with Off():
    top.despawn()

# ---- 5.5: the rest fade out in a wave, the survivors slide into "Algan" --------------------
survivors = [mirror.character_mobs[i] for i in KEEP]
leavers = [mob for i, mob in enumerate(mirror.character_mobs) if i not in KEEP]
xs = [mob.get_center().flatten()[0].item() for mob in leavers]
x_lo, x_hi = min(xs), max(xs)
SWEEP = 1.3  # seconds for the fade to cross the title, left to right
grow = (
    wordmark.character_mobs[1].get_height() / mirror.character_mobs[1].get_height()
).item()
with Sync():
    for mob, x in zip(leavers, xs):
        with Seq():
            Scene.wait(SWEEP * (x - x_lo) / (x_hi - x_lo) + 0.01)
            with Sync(runtime=0.5):
                mob.opacity = 0.0
    with Seq():
        Scene.wait(0.3)
        with Sync(runtime=1.9, easing=easings.ease_in_out_sine):
            for k, mob in enumerate(survivors):
                mob.scale(grow)
                mob.move_to(targets[k])
    # The gallery fades out of the reflections as the word forms, so the mirror settles into
    # the logo's chrome, which reflects only the studio.
    with Seq():
        Scene.wait(0.5)
        with Sync(runtime=1.7):
            for mob in gallery:
                mob.opacity = 0.0
    with Sync(runtime=2.2, easing=easings.ease_in_out_sine):
        cam.move_to(cam_mid)
        cam.look_at(look_mid)
with Off():
    for mob in gallery:
        mob.despawn()

# ---- 7.7: the crossbar lifts off, becomes y = sin x, and comes back as the wave ------------
axes_centre = at(word_centre, brand.FLOOR_TOP + WORD_HEIGHT + 2.3, 0.0)
half = 3.3
radius = 0.07
axes = Axes(
    x_range=[-1.5 * math.pi, 1.5 * math.pi, 0.5 * math.pi],
    y_range=[-1.4, 1.4, 1],
    x_length=2 * half,
    y_length=2.8 * half / (1.5 * math.pi),
    tips=False,
    axis_config={"stroke_color": "#6c80b0", "stroke_width": 3},
)
axes.move_to(axes_centre)
label = Tex(r"y = \sin x", font_size=40, color=Color("#c8d6f0"))
label.move_to(axes_centre + at(half + 0.9, 0.55, 0.0))

with Off():
    mirror.despawn()
    wordmark.spawn(False)
    bar = crossbar.build(amplitude=0.0).spawn()
    start = crossbar.center
    axes.spawn(False)
    axes.opacity = 0.0
    label.spawn(False)
    label.opacity = 0.0


def bar_shape(centre, **shape):
    """The crossbar's tube at ``centre``, as offsets from where it was built."""
    tube = crossbar.shape(**shape)
    return lambda uv: tube(uv) + (centre - start)


with Sync(runtime=0.7):
    bar.set_location_by_function(
        bar_shape(axes_centre, amplitude=0.0, half=half, radius=radius)
    )
    axes.opacity = 0.45
with Sync(runtime=0.9):
    bar.set_location_by_function(
        bar_shape(
            axes_centre, amplitude=half / (1.5 * math.pi), half=half, radius=radius
        )
    )
    label.opacity = 1.0
Scene.wait(0.4)
with Sync(runtime=0.8):
    bar.set_location_by_function(bar_shape(crossbar.center))
    axes.opacity = 0.0
    label.opacity = 0.0
with Off():
    axes.despawn()
    label.despawn()

# ---- 10.5: the solids arrive and the camera cranes to the logo ---------------------------
sphere_end, _ = layout["sphere"]
cube_end, _ = layout["cube"]
cone_end, _ = layout["cone"]
roll = 9.0
with Off():
    sphere.move_to(sphere_end + RIGHT * roll).spawn(False)
    cube.move_to(cube_end + UP * 9.0).spawn(False)
    cone.move_to(cone_end + UP * 9.0).spawn(False)
with Sync():
    with Sync(runtime=1.3, easing=easings.ease_out_cubic):
        sphere.move_to(sphere_end)
        sphere.rotate(math.degrees(roll / sphere.get_width().item() * 2), OUT)
    with Seq():
        Scene.wait(0.4)
        with Sync(runtime=0.9, easing=easings.ease_out_bounce):
            cube.move_to(cube_end)
    with Seq():
        Scene.wait(0.8)
        with Sync(runtime=0.9, easing=easings.ease_out_bounce):
            cone.move_to(cone_end)
    with Sync(runtime=2.4, easing=easings.ease_in_out_sine):
        cam.move_to(cam_end)
        cam.look_at(look_end)

# ---- 12.9: a tagline, then fade to black --------------------------------------------------
with Off():
    tagline.spawn(False)
    tagline.opacity = 0.0
with Sync(runtime=0.7):
    tagline.opacity = 1.0
Scene.wait(1.0)
# A black card in front of the lens: the studio's backdrop is the environment map, which no
# Mob's opacity reaches.
with Off():
    forward = (look_end - cam_end) / torch.linalg.norm(look_end - cam_end)
    card = Square(color=BLACK).scale(6.0)
    card.set_material(MeshBasicMaterial(color=BLACK))
    card.move_to(cam_end + forward * 1.0)
    card.spawn(False)
    card.opacity = 0.0
with Sync(runtime=FADE):
    card.opacity = 1.0
Scene.wait(0.1)

os.makedirs(OUTDIR, exist_ok=True)
if MODE == "STILLS":
    times = [1.2, 4.0, 5.7, 6.1, 6.5, 6.9, 7.3, 7.7, 9.3, 10.9, 11.8, 13.2, 14.5, 15.3]
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
