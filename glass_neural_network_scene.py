# ruff: noqa: C408
#   The per-layer ``dict(...)`` calls are keyword tables, kept for readability
#   against the fitted coefficients.
"""Recreation of ``glass_neural_network.png`` as a 3D Algan scene.

Nineteen glass spheres in five fully-connected layers over a dark navy backdrop.

The geometry is a real 3D scene: ``Sphere`` mobs at measured world positions,
viewed by a long-lens camera.  Node centres and radii were read off the reference
image and converted through the camera frustum (7 world units of height at the
z = 0 plane).

Sphere appearance is the shipped ``glass_ball`` fragment stage
(``mob.set_fragment_shader``), with its coefficients fitted to the reference.

Usage::

    .venv/Scripts/python.exe glass_neural_network_scene.py draft   # 719x213
    .venv/Scripts/python.exe glass_neural_network_scene.py full    # 1438x426
    .venv/Scripts/python.exe glass_neural_network_scene.py big     # 2876x852
"""

from __future__ import annotations

import math
import sys

import torch

from algan import *

# ---------------------------------------------------------------------------
# Reference-image geometry (pixels in the 719x213 source) -> world units.
# ---------------------------------------------------------------------------
REF_W, REF_H = 719, 213
SCALE = 7.0 / REF_H  # the camera shows 7 world units of height at the z = 0 plane

# Long lens: keep 3.5 world units of half-height at z = 0 from 10x the distance.
CAMERA_DISTANCE = 70.0
CAMERA_FOV = 2 * math.degrees(math.atan(3.5 / CAMERA_DISTANCE))


def wx(px):
    return (px - REF_W / 2) * SCALE


def wy(py):
    return (REF_H / 2 - py) * SCALE


def wr(pr):
    return pr * SCALE


def rgb(*v):
    """0-255 reference units -> the renderer's 0-1 linear units."""
    return tuple(c / 255.0 for c in v)


# Each layer: x, radius, node y's, and the four-term appearance fit (body, wide
# Fresnel lobe, narrow Fresnel lobe, Gaussian silhouette ring) least-squares
# fitted to the median radial profile of that layer's spheres in the reference.
LAYERS = [
    dict(  # input layer
        x=91.0, r=15.0, ys=[58.0, 97.0, 136.0, 175.0],
        body=rgb(3.6, 17.8, 35.1),
        rim1=rgb(-3.0, 42.6, 97.8),
        rim2=rgb(96.3, 76.3, 30.5),
        ring=rgb(47.9, 37.7, 32.3), ring_f0=0.640, ring_fw=0.110,
    ),
    dict(  # first hidden layer
        x=267.5, r=18.0, ys=[19.0, 59.0, 99.0, 140.0, 181.0],
        body=rgb(-0.2, 22.7, 53.4),
        rim1=rgb(27.1, 108.9, 171.6),
        rim2=rgb(75.1, -6.1, -77.8),
        ring=rgb(19.1, 18.6, 18.0), ring_f0=0.660, ring_fw=0.060,
    ),
    dict(  # second hidden layer
        x=449.5, r=18.0, ys=[19.0, 59.0, 99.0, 140.0, 181.0],
        body=rgb(-0.2, 22.7, 53.4),
        rim1=rgb(27.1, 108.9, 171.6),
        rim2=rgb(75.1, -6.1, -77.8),
        ring=rgb(19.1, 18.6, 18.0), ring_f0=0.660, ring_fw=0.060,
    ),
    dict(  # teal layer
        x=594.0, r=14.5, ys=[66.0, 111.0, 156.0],
        body=rgb(2.0, 49.2, 71.2),
        rim1=rgb(16.9, 90.1, 104.5),
        rim2=rgb(222.0, 212.0, 158.0),
        ring=rgb(-24.4, 22.5, 24.7), ring_f0=0.450, ring_fw=0.210,
    ),
    # The output balls are only 21 px across and their measured profile is
    # non-monotone near the edge, which drives an unstable negative ring. They
    # borrow the input layer's (well-conditioned) edge shape instead.
    dict(  # output layer
        x=675.0, r=10.4, ys=[94.5, 129.0],
        body=rgb(4.0, 19.0, 40.0),
        rim1=rgb(0.0, 58.0, 118.0),
        rim2=rgb(96.3, 76.3, 30.5),
        ring=rgb(55.0, 50.0, 45.0), ring_f0=0.550, ring_fw=0.180,
    ),
]

RIM_POWER = 0.75  # broad Fresnel lobe
EDGE_POWER = 2.6  # sharp Fresnel lobe
ANISOTROPY = 0.70  # top/bottom-bright edge, measured off the reference
KEY_ANGLE = 0.39  # streak tilt, radians

# Glowing "signal" accents strung along the wires, located by thresholding the
# reference above luminance 150 and taking each blob's centroid and core size.
# (x_px, y_px, core_radius_px, brightness)
ACCENTS = [
    (356.2, 101.4, 3.4, 1.00),
    (219.5, 132.2, 2.2, 0.92),
    (402.7, 123.2, 2.2, 0.97),
    (295.9, 95.3, 1.8, 0.96),
    (171.8, 112.7, 2.2, 0.98),
    (532.4, 109.7, 2.0, 1.00),
    (199.7, 122.8, 2.2, 0.95),
    (561.8, 109.6, 1.5, 0.94),
    (336.3, 94.7, 2.0, 0.93),
    (323.5, 95.6, 1.9, 0.97),
    (510.0, 106.9, 1.9, 0.98),
    (110.6, 102.3, 1.6, 0.94),
    (236.2, 133.1, 1.5, 0.93),
    (137.7, 108.0, 1.4, 0.98),
    (618.7, 108.6, 1.2, 0.88),
    (494.7, 129.0, 1.5, 0.94),
    (377.7, 109.7, 1.4, 0.96),
    (485.0, 132.9, 1.2, 0.94),
    (300.8, 133.0, 1.1, 0.94),
    (290.1, 136.6, 1.1, 0.87),
    (157.1, 110.8, 1.1, 0.98),
    (189.1, 118.9, 1.0, 0.96),
    (472.0, 137.8, 1.1, 0.92),
    (481.2, 99.8, 1.0, 0.89),
    # Loose sparkles in the empty right-hand background.
    (657.0, 70.0, 0.9, 0.70),
    (660.0, 29.0, 0.7, 0.45),
    (500.0, 62.0, 0.7, 0.45),
]
ACCENT_CORE_SCALE = 0.52  # the detected blob size is mostly halo, not core
ACCENT_RGB = (0.52, 0.92, 1.00)
ACCENT_GLOW = 1.15

LINE_COLOR = Color(rgb(120, 158, 196))
LINE_WIDTH = 1.35
LINE_OPACITY = 0.62

RESOLUTIONS = {
    "draft": VideoSettings(resolution=(719, 213), anti_alias_level=2),
    "full": VideoSettings(resolution=(1438, 426), anti_alias_level=3),
    "big": VideoSettings(resolution=(2876, 852), anti_alias_level=3),
}


# ---------------------------------------------------------------------------
# Why the ``glass_ball`` stage rather than a material
#
# A physical glass ball -- MeshPhysicalMaterial(transmission=...) plus an
# environment map bright in the directions the camera cannot see -- does produce
# this look; the rim comes from refraction, not from grazing reflection. It is
# not used here for two reasons: the environment map replaces the backdrop (so
# the navy gradient would have to be baked into it at the resolution the frustum
# leaves), and refraction through curved PN geometry currently speckles.
#
# The stage instead reproduces the measured falloff directly: its coefficients
# below are least-squares fits to the median radial profile of the reference's
# own spheres, per layer.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Background: the reference's navy gradient, measured off the source image.
# ---------------------------------------------------------------------------
BG_BASE = rgb(0.0, 4.0, 16.5)
BG_HALO = rgb(1.0, 16.5, 30.5)


def background(x, y, t):
    """Dark navy with a broad, slightly left-of-centre core.

    ``x``/``y`` arrive in [0, 1) with y = 0 at the bottom of the frame, and the
    result must carry the per-frame axis -- hence the ``ones_like(t)`` broadcast.
    Constants have to be built with ``x.new_tensor``: the callback runs on the
    render device.
    """
    sx = (x - 0.50) * 2.0
    sy = (y - 0.50) * 2.0
    wx_ = torch.where(sx < 0, torch.full_like(sx, 0.95), torch.full_like(sx, 0.62))
    halo = torch.exp(-((sx / wx_) ** 2) - (sy / 0.78) ** 2) * torch.ones_like(t)
    return x.new_tensor(BG_BASE) + x.new_tensor(BG_HALO) * halo


# ---------------------------------------------------------------------------
# Scene.
# ---------------------------------------------------------------------------
def build():
    # Author colours directly in output units: the reference is an ordinary LDR
    # image, and every value here is already the final pixel value. This alone is
    # enough for linear output -- the composite keeps its HDR buffer and the post
    # stage clamps instead of applying a curve, so bloom still has headroom.
    SETTINGS.raytracing.set(tonemapping=False)

    Scene.clear_light_sources()
    Scene.add_light_source(
        AmbientLight(color=WHITE, intensity=1.0).spawn(animate=False)
    )

    with Off():
        # The reference is a long-lens (near-orthographic) shot: every ball is a
        # circle. The default camera at this 3.4:1 aspect has a 119 deg
        # *horizontal* field of view, which correctly projects an off-axis sphere
        # as an ellipse stretched by 1/cos(59 deg) ~ 1.9 at the frame edge. Pull
        # back to 10x the distance and narrow the fov to keep the same framing.
        camera = Scene.get_camera()
        camera.set_fov(CAMERA_FOV)
        camera.move_to(OUT * CAMERA_DISTANCE)

    with Off():
        nodes = []
        for layer in LAYERS:
            column = []
            for y_px in layer["ys"]:
                sphere = Sphere(
                    center=RIGHT * wx(layer["x"]) + UP * wy(y_px),
                    radius=wr(layer["r"]),
                    color=Color(layer["body"]),
                )
                sphere.set_fragment_shader(glass_ball)
                sphere.rim_color = layer["rim1"]
                sphere.rim_power = RIM_POWER
                sphere.edge_color = layer["rim2"]
                sphere.edge_power = EDGE_POWER
                sphere.ring_color = layer["ring"]
                sphere.ring_center = layer["ring_f0"]
                sphere.ring_width = layer["ring_fw"]
                sphere.anisotropy = ANISOTROPY
                sphere.key_angle = KEY_ANGLE
                column.append(
                    (sphere, wx(layer["x"]), wy(y_px), wr(layer["r"]))
                )
            nodes.append(column)

        lines = []
        # Short vertical stems between neighbours inside the two hidden layers
        # (the reference has them there and nowhere else).
        for li in (1, 2):
            column = nodes[li]
            for (_, cx, cy_a, r_a), (_, _, cy_b, r_b) in zip(column, column[1:]):
                lines.append(
                    Line(
                        RIGHT * cx + UP * (cy_a - r_a),
                        RIGHT * cx + UP * (cy_b + r_b),
                        color=LINE_COLOR,
                        border_width=LINE_WIDTH,
                        opacity=LINE_OPACITY * 0.8,
                    )
                )

        for left, right in zip(nodes, nodes[1:]):
            for _, lx, ly, lr in left:
                start = RIGHT * (lx + lr) + UP * ly
                for _, rx, ry, rr in right:
                    end = RIGHT * (rx - rr) + UP * ry
                    lines.append(
                        Line(
                            start,
                            end,
                            color=LINE_COLOR,
                            border_width=LINE_WIDTH,
                            opacity=LINE_OPACITY,
                        )
                    )

        # Signal pulses. These are luminous mobs, not painted-on decoration:
        # `glow` feeds the renderer's glow channel and the bloom post-process
        # spreads it, which is what gives the soft halo around each core.
        accents = []
        for x_px, y_px, r_px, amp in ACCENTS:
            accents.append(
                Dot(
                    point=RIGHT * wx(x_px) + UP * wy(y_px) + OUT * 0.02,
                    radius=wr(r_px * ACCENT_CORE_SCALE),
                    color=Color(tuple(c * amp for c in ACCENT_RGB)),
                    glow=ACCENT_GLOW * amp,
                )
            )

        for line in lines:
            line.spawn(animate=False)
        for accent in accents:
            accent.spawn(animate=False)
        for column in nodes:
            for sphere, *_ in column:
                sphere.spawn(animate=False)
    return nodes, lines


def main():
    quality = sys.argv[1] if len(sys.argv) > 1 else "draft"
    build()
    Scene.save_frame(
        f"glass_nn_{quality}", RESOLUTIONS[quality], background_color=background
    )


if __name__ == "__main__":
    main()
