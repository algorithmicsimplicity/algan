"""What the logo is made of, shared by the banner and icon (``logo_scene.py``) and by the
demo video that ends on it (``demo_video.py``): the typeface, the colours and materials, the
studio, the crossbar and the layout of the wordmark and the three solids.

Both scripts run as ``python branding/<script>.py``, which puts this directory on the path.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

from algan import *  # noqa: F403

HERE = Path(__file__).resolve().parent

# ---- typeface ------------------------------------------------------------------------------
# Libre Franklin Black with a lambda added, tightened and renamed (fonts/build_font.py). It is
# registered with Pango for this process only, so nothing has to be installed.
FONT_FILE = HERE / "fonts" / "AlganFranklin-Black.ttf"
FONT, WEIGHT = "Algan Franklin", "HEAVY"


def register_font():
    import manimpango

    if not manimpango.register_font(str(FONT_FILE)):
        raise RuntimeError(f"Pango could not load {FONT_FILE}")


register_font()

# The font's own proportions, in font units of its 1000-unit em: what the layout is sized by.
CAP, X_HEIGHT, DESCENDER = 742.0, 530.0, 175.0
# The A's crossbar runs from 144 to 314 units up; the lambda keeps the A's legs, so the wave sits
# where the crossbar was, centred between the legs (the right foot is the wider one).
BAR_Y, BAR_X, BAR_HALF = 0.31 * CAP, 388.0, 178.0

# ---- colour --------------------------------------------------------------------------------
# One blue: the wave's is the brand colour, and the glass cube is a deeper shade of its hue.
BRAND_BLUE = "#4da3ff"
DEEP_BLUE = "#17509c"
GOLD = "#f2c85a"
CHROME_TINT = "#eef1f5"
# Near black, a touch towards the brand blue: the studio's backdrop.
BACKDROP = "#0b0f17"
FLOOR_TOP = -1.5


def chrome():
    return MeshStandardMaterial(color=Color(CHROME_TINT), metalness=1.0, roughness=0.08)


def mirror_ball():
    return MeshStandardMaterial(color=Color(CHROME_TINT), metalness=1.0, roughness=0.03)


def blue_lacquer():
    """Deep blue under a clear coat. Glass was the first idea, but in a black studio there is
    nothing behind it to refract, so it rendered as a dark box.
    """
    return MeshPhysicalMaterial(
        color=Color(DEEP_BLUE),
        roughness=0.35,
        clearcoat=1.0,
        clearcoat_roughness=0.04,
    )


def satin_gold():
    return MeshStandardMaterial(color=Color(GOLD), metalness=1.0, roughness=0.32)


def wave_material():
    """The crossbar glows a little of its own light, so it reads in front of dark chrome."""
    return MeshStandardMaterial(
        color=Color(BRAND_BLUE),
        metalness=0.0,
        roughness=0.25,
        emissive=Color(BRAND_BLUE),
        emissive_intensity=0.55,
    )


# ---- the studio ----------------------------------------------------------------------------
def _linear(hex_color):
    rgb = torch.tensor([int(hex_color[i : i + 2], 16) / 255.0 for i in (1, 3, 5)])
    return torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def studio_env(back_lon=270.0, H=512, W=1024, horizon=0.35, soft=0.9):
    """A dark studio the chrome has something to reflect in.

    The camera looks towards longitude ``back_lon - 180``. Flat letters facing it reflect the
    region just below the horizon *behind* it, the top of a letter seeing the lowest elevation,
    so a bright band there, cut off above, gives them their chrome horizon. A broad, dimmer band
    runs round the sides and back for the curved solids, a hard softbox up and to the left puts
    a sharp highlight on them and a soft one to the right a broad one. In front, behind the
    logo, there is only the tinted backdrop.
    """
    lat = torch.linspace(90, -90, H).view(H, 1).expand(H, W)
    lon = torch.linspace(0, 360, W).view(1, W).expand(H, W)
    d = (lon - back_lon + 180) % 360 - 180
    # 0 across the camera's view, 1 behind and to the sides. The view reaches the horizon (the
    # camera looks down past it), so without this the bands' tails tint the backdrop grey.
    away_from_front = torch.sigmoid((105 - d.abs()) / 8)
    env = _linear(BACKDROP).expand(H, W, 3).clone()
    env += ((lat.clamp(min=0) / 90) ** 2.2 * 0.08).unsqueeze(-1)
    band = (
        torch.sigmoid((lat + 5) / 1.2)
        * torch.sigmoid((5.5 - lat) / 1.2)
        * torch.exp(-((d / 80) ** 2))
        * 1.3
    ) * away_from_front
    env += band.unsqueeze(-1) * torch.tensor([0.92, 0.96, 1.0])
    horizon = (
        torch.sigmoid((lat + 2) / 2.0) * torch.sigmoid((16 - lat) / 4.0) * horizon
    ) * away_from_front
    env += horizon.unsqueeze(-1) * torch.tensor([0.9, 0.95, 1.0])
    box = ((d + 25).abs() < 22).float() * ((lat > 30) & (lat < 55)).float() * 3.0
    env += box.unsqueeze(-1)
    soft = (
        torch.sigmoid((18 - (d - 45).abs()) / 4)
        * torch.sigmoid((14 - (lat - 32).abs()) / 4)
        * soft
    )
    env += soft.unsqueeze(-1) * torch.tensor([1.0, 0.97, 0.92])
    env *= (0.15 + 0.85 * torch.sigmoid((lat + 5) / 1.5)).unsqueeze(-1)
    return env


def make_floor(depth=60.0, front=None):
    """A dark mirror, quiet enough that the reflected word does not compete with the real one.

    ``front``: where the floor's near edge is, in z; by default it runs well past any camera.
    """
    floor = Prism(width=90, height=0.02, depth=depth, color=Color("#07080b"))
    floor.set_material(
        MeshStandardMaterial(color=Color("#07080b"), metalness=0.72, roughness=0.0)
    )
    z = 0.0 if front is None else front - 0.5 * depth
    floor.move(UP * (FLOOR_TOP - 0.01) + OUT * z)
    return floor


def make_lights():
    RectAreaLight(
        location=UP * 12 + OUT * 4 + LEFT * 4,
        target=ORIGIN,
        width=6,
        height=6,
        samples=9,
        color=WHITE,
        intensity=30,
    ).spawn()
    PointLight(
        location=RIGHT * 10 + UP * 4 + OUT * 10, color=Color("#cfe0ff"), intensity=0.5
    ).spawn()


# ---- the crossbar --------------------------------------------------------------------------
def tube(half, amplitude, radius, periods=1.5, along=96, around=16):
    """A round-ended tube along ``y = amplitude * sin(...)``, ``-half <= x <= half``.

    Returns a function from the surface's ``(u, v)`` grid to points around the origin, which is
    both what builds the Surface and what ``set_location_by_function`` morphs it to: a straight
    bar is ``amplitude=0``, and the grid matches whatever the amplitude, so they morph cleanly.
    """
    k = math.pi * periods / half

    def shape(uv):
        u, v = uv[..., :1], uv[..., 1:2]
        s = (u * 2 - 1) * (
            half + radius
        )  # along the bar, past each end by the cap's radius
        x = s.clamp(-half, half)
        over = (s.abs() - half).clamp(min=0)
        r = torch.sqrt((radius**2 - over**2).clamp(min=0))
        y = amplitude * torch.sin(k * x)
        slope = amplitude * k * torch.cos(k * x)
        norm = torch.sqrt(1 + slope**2)
        nx, ny = -slope / norm, 1 / norm
        tx, ty = 1 / norm, slope / norm
        x = x + torch.sign(s) * over * tx
        y = y + torch.sign(s) * over * ty
        theta = v * 2 * math.pi
        c, si = torch.cos(theta), torch.sin(theta)
        return torch.cat([x + r * c * nx, y + r * c * ny, r * si], -1)

    return shape


TUBE_GRID = {"grid_width": 96, "grid_height": 16}


class Crossbar:
    """The wave across the lambda, sized from the lambda glyph."""

    def __init__(self, lam):
        lo, hi = lam.get_bounding_box_min()[0, 0], lam.get_bounding_box_max()[0, 0]
        unit = (hi[1] - lo[1]).item() / CAP  # world units per font unit
        self.unit = unit
        self.center = torch.tensor(
            [lo[0].item() + (BAR_X - 22.0) * unit, lo[1].item() + BAR_Y * unit, 0.0]
        )
        self.half = BAR_HALF * unit
        self.amplitude = 0.075 * CAP * unit
        self.radius = 0.062 * CAP * unit

    def shape(self, amplitude=None, half=None, radius=None):
        return tube(
            self.half if half is None else half,
            self.amplitude if amplitude is None else amplitude,
            self.radius if radius is None else radius,
        )

    def build(self, **kwargs):
        bar = Surface(self.shape(**kwargs), **TUBE_GRID)
        bar.set_material(wave_material())
        bar.move_to(self.center)
        return bar


# ---- wordmark and solids -------------------------------------------------------------------
def wordmark(text="Λlgan", height=3.1, font_size=200):
    """The chrome wordmark, ``height`` tall from the g's descender to the cap line."""
    word = Text(text, font=FONT, weight=WEIGHT, font_size=font_size)
    word.set_material(chrome())
    word.scale_to_height(height)
    return word


def float_word(word, left):
    """Stand the word just clear of the floor (the g's descender hovers over it), left edge at
    ``left``. The wordmark floats: a baseline on the floor would sink the g into it.
    """
    lo = word.get_bounding_box_min()[0, 0]
    word.move(UP * (FLOOR_TOP + 0.05 - lo[1]) + RIGHT * (left - lo[0]))
    return word


def solid_sizes(word_height):
    """The sphere as tall as a lowercase letter, the cone as tall as a capital, the cube
    between: they rise left to right, like Manim's circle, square and triangle.
    """
    unit = word_height / (CAP + DESCENDER)
    return {
        "sphere": 0.5 * X_HEIGHT * unit,  # radius
        "cube": (X_HEIGHT + 0.25 * (CAP - X_HEIGHT)) * unit,  # edge
        "cone": (0.40 * CAP * unit, CAP * unit),  # radius, height
        "em": 1000 * unit,
    }


CUBE_TURN = 38.0


def solid_layout(word_right, word_height):
    """Where the three solids stand, left to right from the word's right edge.

    Half a letter between the n and the sphere, and a clear gap between each pair, so nothing
    overlaps from the front. Returns ``{name: (centre, size)}``.
    """
    s = solid_sizes(word_height)
    gap = 0.2 * s["em"]
    cube_half = (
        0.5
        * s["cube"]
        * (math.cos(math.radians(CUBE_TURN)) + math.sin(math.radians(CUBE_TURN)))
    )
    x = word_right + 0.30 * s["em"] + s["sphere"]
    sphere = (RIGHT * x + UP * (FLOOR_TOP + s["sphere"]) + OUT * 0.5, s["sphere"])
    x += s["sphere"] + gap + cube_half
    cube = (RIGHT * x + UP * (FLOOR_TOP + 0.5 * s["cube"]) + IN * 0.2, s["cube"])
    x += cube_half + gap + s["cone"][0]
    cone = (RIGHT * x + UP * (FLOOR_TOP + 0.5 * s["cone"][1]) + OUT * 0.3, s["cone"])
    return {"sphere": sphere, "cube": cube, "cone": cone, "right": x + s["cone"][0]}


def make_solids(word_height):
    """The sphere, cube and cone at their logo size, unplaced."""
    s = solid_sizes(word_height)
    sphere = Sphere(radius=s["sphere"], color=WHITE).set_material(mirror_ball())
    cube = Cube(size=s["cube"], fill_opacity=1.0).set_material(blue_lacquer())
    cube.rotate(CUBE_TURN, UP)
    cone = Cone(radius=s["cone"][0], height=s["cone"][1], closed=True)
    cone.set_material(satin_gold())
    return sphere, cube, cone


def place_solids(solids, word_right, word_height):
    """Move the three solids to their logo positions (from ``solid_layout``)."""
    layout = solid_layout(word_right, word_height)
    for mob, name in zip(solids, ("sphere", "cube", "cone")):
        mob.move_to(layout[name][0])
    return solids


def logo_extent(word, word_height):
    """The left and right edges of wordmark plus solids, for centring a camera on them."""
    left = word.get_bounding_box_min()[0, 0][0].item()
    right = word.get_bounding_box_max()[0, 0][0].item()
    return left, solid_layout(right, word_height)["right"]


def frame_camera(cam, look, width, aspect, elevation, margin=1.12):
    """Place ``cam`` to fit ``width`` world units across at ``look``, from ``elevation`` degrees
    above, with ``margin`` to spare. Returns the camera location.
    """
    half_height = 0.5 * width * margin / aspect
    distance = half_height / math.tan(math.radians(cam.get_fov()) * 0.5)
    a = math.radians(elevation)
    location = look + torch.tensor(
        [0.0, distance * math.sin(a), distance * math.cos(a)]
    )
    return location
