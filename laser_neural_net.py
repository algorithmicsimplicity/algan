"""Recreation of ``laser_neural_net.jpg`` as an Algan Mob.

The reference is a 2752x1536 render of a fully-connected 5-8-5-3 network drawn as
glowing laser beams: cyan rings for the neurons, blue beams into the first hidden
layer, white-to-red beams across the middle, red-to-gold beams into the output,
solid arrows feeding the input layer and pointing back into the output layer.

Everything below is authored in **reference-image pixel coordinates** (measured off
the jpg) and mapped into world space by :func:`p`, so the geometry can be checked
against the source image directly.
"""

from __future__ import annotations

import math
from functools import partial

import torch

from algan import (
    HD,
    OUT,
    RIGHT,
    SETTINGS,
    UP,
    Circle,
    Color,
    Group,
    Line,
    Off,
    Polygon,
    Scene,
    VideoSettings,
)
from algan.rendering.post_processing.bloom import bloom_filter

# --------------------------------------------------------------------------- #
# Reference-image frame
# --------------------------------------------------------------------------- #

IMG_W, IMG_H = 2752, 1536
FRAME_H = 7.0  # Algan's visible height at the origin plane
PX = FRAME_H / IMG_H  # world units per reference pixel


def _scene(scene) -> dict:
    """``scene=`` kwargs, omitted entirely when no explicit Scene was given."""
    return {} if scene is None else {"scene": scene}


def p(x: float, y: float):
    """Reference-image pixel (x right, y down) -> world point on the origin plane."""
    return RIGHT * ((x - IMG_W / 2) * PX) + UP * ((IMG_H / 2 - y) * PX)


#: A ``border_width`` of 1 draws a stroke 1/454 of the frame height wide --
#: measured, not documented; it is resolution- and ``anti_alias_level``-
#: independent, so reference pixels convert with a single constant.
STROKE_UNIT = 454.0


def stroke(ref_px: float) -> float:
    """Reference-image pixel width -> ``border_width``."""
    return ref_px * STROKE_UNIT / IMG_H


# --------------------------------------------------------------------------- #
# Measured geometry
# --------------------------------------------------------------------------- #

LAYERS: list[tuple[float, list[float]]] = [
    (761.0, [522, 660, 802, 942, 1092]),
    (1423.0, [248, 388, 526, 666, 806, 946, 1090, 1236]),
    (2028.0, [530, 668, 808, 948, 1090]),
    (2440.0, [672, 806, 946]),
]

# A neuron is two thin concentric rings with a fine dashed ring inside them.
NODE_R = 39.0  # outer ring radius, reference px -- also where beams attach
RING_W = 2.0
RING2_R = 34.5
RING2_W = 1.5
INNER_R = 29.5  # the fine dashed ring
INNER_W = 1.5
INNER_DASHES = 40

BEAM_W = 3.9  # laser beam width, reference px
CORE_W = 1.8  # the white-hot filament inside it
CORE_MIX = 0.62  # how far the core is mixed towards white

# ``glow`` feeds the bloom pass as ``glow**3 * strength``, so it is very
# non-linear -- these are tuned against the halo profile in the reference.
GLOW_BEAM = 0.32
GLOW_CORE = 0.24
GLOW_RING = 0.31

# The reference's beams are not uniformly bright: sampling the halo 9px off-axis
# gives a U along the beam, ~2.4x stronger at the ends than mid-span.  The
# brightening is *local* -- the reference keeps the space between beams dark and
# spends its light in the last stretch, so this is a short span and a big boost
# rather than a raised floor.
# The two ends differ (~1.8x at the source, ~2.5x at the target), and the source
# end needs the smaller share anyway: a whole fan diverges from one point there,
# so equal boosts blow it out into a wedge.
END_SPAN = 0.18
START_BOOST = 0.40
END_BOOST = 1.10
END_WHITEN = 0.22

# Where the fans meet the rings the reference puts an outright hot spot.
HOTSPOT_R = 5.0  # reference px
HOTSPOT_GLOW = 0.52
HOTSPOT_MIX = 0.80  # how far the spot is mixed towards white

# Everything is coplanar, so pieces that must occlude each other are nudged a
# hair towards the camera (``OUT``) rather than relying on draw order.
Z_CORE = 0.004
Z_DISC = 0.010
Z_RING = 0.016

# Colours sampled off the reference.
RING_C = "#3EC6FA"
RING_INNER_C = "#7FD8FF"
DISC_C = Color("#071827")  # the dark interior of a neuron

# Per-boundary beam colours: blue into the first hidden layer, white->red across
# the middle, red->gold into the output.  ``lo``/``hi`` are where along the beam
# the hand-over starts and finishes -- measured off the reference, the white/red
# swap happens abruptly around 55%, while red bleeds into gold only near the end.
BEAMS = [
    {"start": "#1478FF", "end": "#63CEFF", "steps": 3, "lo": 0.0, "hi": 1.0},
    {"start": "#EDF1FF", "end": "#FF3418", "steps": 9, "lo": 0.30, "hi": 0.62,
     "glow_scale": 0.78},  # white blooms hardest at equal glow
    {"start": "#FF3A18", "end": "#FFB733", "steps": 7, "lo": 0.35, "hi": 1.05},
]

# Arrow geometry and colours, both measured off the reference.
ARROW_IN = {
    "tip": 715, "tail": 544, "head_len": 41, "head_half": 19, "shaft": 7.5,
    "head": "#57EFFE", "neck_color": "#4EE8FD", "tail_color": "#1C527D",
    "glow": 0.34,
}
ARROW_OUT = {
    "tip": 2500, "tail": 2634, "head_len": 32, "head_half": 16, "shaft": 6.0,
    "head": "#FEB483", "neck_color": "#F79E74", "tail_color": "#0A1220",
    "glow": 0.34,
}


# --------------------------------------------------------------------------- #
# Pieces
# --------------------------------------------------------------------------- #


def _lerp_hex(a: str, b: str, t: float) -> tuple[float, float, float]:
    ca, cb = Color(a), Color(b)
    return tuple(float(v) for v in (ca[:3] * (1 - t) + cb[:3] * t))


def _towards_white(rgb, t: float) -> tuple[float, float, float]:
    return tuple(v * (1 - t) + t for v in rgb)


def _neuron(x: float, y: float, scene=None) -> list:
    """A neuron: an opaque dark disc, the bright ring, and a fine dashed ring.

    The disc is what makes the beam fans converge the way they do in the
    reference -- beams are aimed at the node *centre* and simply vanish under
    the disc, so their apparent meeting point is the ring's leading edge.
    """
    parts = [
        Circle(
            radius=NODE_R * PX,
            location=p(x, y) + OUT * Z_DISC,
            color=DISC_C,
            border_color=DISC_C,
            border_width=stroke(1.0),
            **_scene(scene),
        ),
        Circle(
            radius=NODE_R * PX,
            location=p(x, y) + OUT * Z_RING,
            filled=False,
            border_color=Color(RING_C, glow=GLOW_RING),
            border_width=stroke(RING_W),
            **_scene(scene),
        ),
        Circle(
            radius=RING2_R * PX,
            location=p(x, y) + OUT * Z_RING,
            filled=False,
            border_color=Color(RING_C, glow=GLOW_RING * 0.85),
            border_width=stroke(RING2_W),
            **_scene(scene),
        ),
    ]

    dash_color = Color(RING_INNER_C, glow=0.08, opacity=0.35)
    step = 2 * math.pi / INNER_DASHES
    for i in range(INNER_DASHES):
        a0 = i * step
        a1 = a0 + step * 0.5  # 50% duty cycle
        parts.append(
            Line(
                p(x + INNER_R * math.cos(a0), y + INNER_R * math.sin(a0)) + OUT * Z_RING,
                p(x + INNER_R * math.cos(a1), y + INNER_R * math.sin(a1)) + OUT * Z_RING,
                color=dash_color,
                border_width=stroke(INNER_W),
                **_scene(scene),
            )
        )
    return parts


def _beam(start_px, end_px, spec, scene=None) -> list:
    """One laser beam: a coloured stroke with a white-hot core inside it.

    The gradient along the beam is faked with constant-colour segments; the
    core is what gives the reference its saturated-clipping-to-white look.
    """
    (x0, y0), (x1, y1) = start_px, end_px
    steps = spec["steps"]
    segs = []
    for i in range(steps):
        a, b = i / steps, (i + 1) / steps
        # Overlap neighbours slightly so the joins disappear under the glow.
        a = max(0.0, a - 0.004)
        b = min(1.0, b + 0.004)
        s = (i + 0.5) / steps
        u = min(1.0, max(0.0, (s - spec["lo"]) / (spec["hi"] - spec["lo"])))
        rgb = _lerp_hex(spec["start"], spec["end"], u * u * (3 - 2 * u))

        # Ramps to 1 at each end of the beam, 0 through the middle.
        def ramp(v):
            v = min(1.0, max(0.0, 1.0 - v / END_SPAN))
            return v * v * (3 - 2 * v)

        e = ramp(s)  # nearness to the source end
        f = ramp(1.0 - s)  # nearness to the target end
        boost = 1 + START_BOOST * e + END_BOOST * f
        scale = spec.get("glow_scale", 1.0)

        pa = p(x0 + (x1 - x0) * a, y0 + (y1 - y0) * a)
        pb = p(x0 + (x1 - x0) * b, y0 + (y1 - y0) * b)
        segs.append(
            Line(
                pa,
                pb,
                color=Color(rgb, glow=GLOW_BEAM * scale * boost),
                border_width=stroke(BEAM_W),
                **_scene(scene),
            )
        )
        segs.append(
            Line(
                pa + OUT * Z_CORE,
                pb + OUT * Z_CORE,
                color=Color(
                    _towards_white(rgb, min(0.9, CORE_MIX + END_WHITEN * max(e, f))),
                    glow=GLOW_CORE * scale * boost,
                ),
                border_width=stroke(CORE_W),
                **_scene(scene),
            )
        )
    return segs


def _hotspot(x: float, y: float, hex_color: str, glow=HOTSPOT_GLOW, scene=None):
    """The bright bead where a fan of beams meets a ring."""
    return Circle(
        radius=HOTSPOT_R * PX,
        location=p(x, y) + OUT * (Z_RING + Z_CORE),
        color=Color(_towards_white(Color(hex_color)[:3].tolist(), HOTSPOT_MIX),
                    glow=glow),
        border_color=Color(_towards_white(Color(hex_color)[:3].tolist(), HOTSPOT_MIX),
                           glow=glow),
        border_width=stroke(1.0),
        **_scene(scene),
    )


def _quad(pts, **kwargs):
    """A filled Polygon with the winding Algan's circuit fill expects.

    Bezier-circuit borders run *inward*, so a slice whose border is a different
    colour loses that band of fill; every filled piece here therefore paints its
    own border in its own colour.
    """
    x = torch.stack([q.reshape(-1)[:2] for q in pts])
    area = float(
        (x[:, 0] * x.roll(-1, 0)[:, 1] - x.roll(-1, 0)[:, 0] * x[:, 1]).sum()
    )
    if area > 0:  # normalize to the clockwise-on-screen winding that fills
        pts = pts[::-1]
    return Polygon(*pts, **kwargs)


def _arrow(spec, y, scene=None):
    """A solid arrow whose shaft ramps from a dark tail to a bright neck."""
    x_tip, x_tail = spec["tip"], spec["tail"]
    head_len, head_half, shaft_half = spec["head_len"], spec["head_half"], spec["shaft"]
    sign = 1.0 if x_tip > x_tail else -1.0
    x_neck = x_tip - sign * head_len
    glow = spec["glow"]

    parts = [
        _quad(
            [q + OUT * Z_RING for q in (p(x_tip, y), p(x_neck, y - head_half), p(x_neck, y + head_half))],
            color=Color(spec["head"], glow=glow),
            border_color=Color(spec["head"], glow=glow),
            border_width=stroke(1.0),
            **_scene(scene),
        )
    ]

    # Opaque colour ramp rather than an opacity ramp: overlapping translucent
    # slices would composite twice and band the shaft.
    steps = 32
    for i in range(steps):
        # Exactly abutting, never overlapping: two overlapping edges each
        # anti-alias to full coverage and sum to a bright tick.
        xa = x_tail + (x_neck - x_tail) * (i / steps)
        xb = x_tail + (x_neck - x_tail) * ((i + 1) / steps)
        t = (i + 0.5) / steps  # 0 at the tail, 1 at the neck
        slice_color = Color(
            _lerp_hex(spec["tail_color"], spec["neck_color"], t), glow=glow * t
        )
        # Each slice sits a hair in front of the previous one: they overlap so
        # no seam shows, and coplanar overlaps would z-fight into tick marks.
        z = Z_RING + i * 1e-4
        parts.append(
            _quad(
                [
                    q + OUT * z
                    for q in (
                        p(xa, y - shaft_half),
                        p(xb, y - shaft_half),
                        p(xb, y + shaft_half),
                        p(xa, y + shaft_half),
                    )
                ],
                color=slice_color,
                border_color=slice_color,
                border_width=stroke(1.0),
                **_scene(scene),
            )
        )

    # A faint rim, drawn as an OPEN polyline: the reference's arrows have no
    # cap across the tail, they just fade out.
    rim = [
        p(x_tail, y - shaft_half),
        p(x_neck, y - shaft_half),
        p(x_neck, y - head_half),
        p(x_tip, y),
        p(x_neck, y + head_half),
        p(x_neck, y + shaft_half),
        p(x_tail, y + shaft_half),
    ]
    for a, b in zip(rim, rim[1:]):
        parts.append(
            Line(
                a + OUT * (Z_RING + Z_CORE),
                b + OUT * (Z_RING + Z_CORE),
                color=Color("#DDF4FF", glow=0.12, opacity=0.45),
                border_width=stroke(1.3),
                **_scene(scene),
            )
        )
    return parts


# --------------------------------------------------------------------------- #
# The Mob
# --------------------------------------------------------------------------- #


def laser_neural_net(scene=None) -> Group:
    """Build the whole diagram as one :class:`Group`."""
    beams, nodes, arrows, spots = [], [], [], []

    # Beams leave from one point on the right edge of their source ring and run
    # all the way to the centre of their target, where the target's disc hides
    # them -- that is the geometry the reference actually uses.
    for spec, (left, right) in zip(BEAMS, zip(LAYERS, LAYERS[1:])):
        (xa, ys_a), (xb, ys_b) = left, right
        for ya in ys_a:
            for yb in ys_b:
                beams += _beam((xa + NODE_R, ya), (xb, yb), spec, scene=scene)
        # A bead where each fan leaves its source ring and where it lands.
        for ya in ys_a:
            spots.append(
                _hotspot(xa + NODE_R, ya, spec["start"],
                         glow=HOTSPOT_GLOW * 0.6, scene=scene)
            )
        for yb in ys_b:
            spots.append(_hotspot(xb - NODE_R, yb, spec["end"], scene=scene))

    for x, ys in LAYERS:
        for y in ys:
            nodes += _neuron(x, y, scene=scene)

    for y in LAYERS[0][1]:
        arrows += _arrow(ARROW_IN, y, scene=scene)
    for y in LAYERS[-1][1]:
        arrows += _arrow(ARROW_OUT, y, scene=scene)

    return Group(beams + arrows + nodes + spots, **_scene(scene))


BACKGROUND = Color("#03111F")

#: Renders at exactly the reference image's size, so the output can be diffed
#: against ``laser_neural_net.jpg`` pixel for pixel.
REFERENCE = VideoSettings(resolution=(IMG_W, IMG_H), frames_per_second=1)

#: Algan's default bloom pairs a tight rim with a *wide* tail (``glow_spread``
#: is 10% of the frame height at ``tail_weight`` 0.6).  That is a lovely soft
#: glow, but with 500-odd emissive beams the tails sum into a haze that fills
#: the gaps between them, and the reference keeps those gaps black.  Narrowing
#: the tail and paying for it with ``strength`` gives the reference's look: a
#: hot, local halo over a dark field.
TIGHT_BLOOM = partial(
    bloom_filter, glow_spread=0.015, tail_weight=0.15, strength=60
)


if __name__ == "__main__":
    import sys

    hd = "--hd" in sys.argv
    SETTINGS.video.set(HD if hd else REFERENCE)

    with Off():
        net = laser_neural_net()
        net.spawn()

    Scene.save_frame(
        "laser_neural_net_hd" if hd else "laser_neural_net",
        background_color=BACKGROUND,
        post_processes=None if "--default-bloom" in sys.argv else (TIGHT_BLOOM,),
    )
