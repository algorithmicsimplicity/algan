"""Sunlit water rippling over the tiles of a swimming pool, path traced.

Usage:  python examples/water_pool.py STILL [quality] [spp] [time]
        python examples/water_pool.py VIDEO [quality] [spp]
"""

import math
import sys

import torch

from algan import *

MODE = sys.argv[1] if len(sys.argv) > 1 else "STILL"
QUALITY = globals()[sys.argv[2]] if len(sys.argv) > 2 else PREVIEW
SPP = int(sys.argv[3]) if len(sys.argv) > 3 else 64
AT = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0

SETTINGS.raytracing.set(samples_per_pixel=SPP, max_bounces=8)

POOL_W, POOL_D, DEPTH = 10.0, 7.0, 1.6  # the basin, in x, z and y
WATER_LEVEL = -0.18
LOOP = 6.0  # seconds; every wave completes whole periods, so the video loops

SUN = torch.tensor([0.30, 0.75, -1.0])
SUN = SUN / SUN.norm()


def sky(height=1024, width=2048):
    """An HDR sky: deep blue overhead, hazy at the horizon, and a sun far brighter than white."""
    lat = (0.5 - (torch.arange(height) + 0.5) / height) * math.pi
    lon = ((torch.arange(width) + 0.5) / width - 0.5) * 2 * math.pi
    lat, lon = torch.meshgrid(lat, lon, indexing="ij")
    d = torch.stack((lat.cos() * lon.cos(), lat.sin(), lat.cos() * lon.sin()), -1)
    up = d[..., 1:2].clamp(min=0)
    zenith = torch.tensor([0.12, 0.30, 0.75])
    horizon = torch.tensor([0.80, 0.88, 0.98])
    env = horizon + (zenith - horizon) * up**0.45
    ground = torch.tensor([0.28, 0.30, 0.24])
    env = torch.where(d[..., 1:2] < 0, ground, env)
    cos_sun = (d * SUN).sum(-1, keepdim=True)
    env = env + torch.tensor([1.0, 0.92, 0.78]) * 0.5 * torch.exp((cos_sun - 1) * 30)
    disc = (cos_sun > math.cos(math.radians(1.2))).float()
    return env + torch.tensor([1.0, 0.95, 0.85]) * 600.0 * disc


def rect(width, height, center, right, up, texture):
    """A flat textured rectangle spanning ``right`` (u) and ``up`` (v) about ``center``."""
    right = torch.tensor(right, dtype=torch.float32)
    up = torch.tensor(up, dtype=torch.float32)
    center = torch.tensor(center, dtype=torch.float32)

    def f(uv):
        return (uv[..., :1] - 0.5) * width * right + (uv[..., 1:] - 0.5) * height * up

    return Surface(f, grid_width=2, grid_height=2, color_texture=texture).move(center)


def tiles(cells_u, cells_v, tile="#e4f2f6", grout="#a9c4cc"):
    return get_grid_lines(
        Color(grout),
        Color(tile),
        resolution=(cells_u, cells_v),
        line_width=0.08,
        texture_resolution=(min(2048, cells_u * 32), min(2048, cells_v * 32)),
    )


def wall_tiles(length, height, cells_per_unit=4):
    """Pale tiles with a band of deep blue mosaic along the waterline."""
    cu, cv = round(length * cells_per_unit), round(height * cells_per_unit)
    body = tiles(cu, cv)
    band = tiles(cu * 2, cv * 2, tile="#1d5f8a", grout="#16455f")
    band = torch.nn.functional.interpolate(
        band.permute(2, 0, 1)[None].float(), size=body.shape[:2], mode="nearest"
    )[0].permute(1, 2, 0)
    v = torch.linspace(0, 1, body.shape[1]).view(1, -1, 1)
    in_band = (v > 1 - 0.32 / height).float()
    return (body * (1 - in_band) + band * in_band).as_subclass(Color)


def ladder_rail(z, inner_x, outer_x, bottom, top, radius=0.035):
    """A chrome grab rail: up out of the water, over the coping, and down onto the deck."""
    bend = (outer_x - inner_x) / 2
    legs = (top - bottom, math.pi * bend, top)
    total = sum(legs)

    def centre_and_normal(s):
        d = s * total
        leg_in = d < legs[0]
        leg_out = d > legs[0] + legs[1]
        angle = ((d - legs[0]) / bend).clamp(0, math.pi)
        arc = torch.stack(
            (inner_x + bend - bend * angle.cos(), top + bend * angle.sin()), -1
        )
        arc_n = torch.stack((-angle.cos(), angle.sin()), -1)
        up = torch.stack((torch.full_like(d, inner_x), bottom + d), -1)
        down = torch.stack(
            (torch.full_like(d, outer_x), top - (d - legs[0] - legs[1])), -1
        )
        c = torch.where(
            leg_in[..., None], up, torch.where(leg_out[..., None], down, arc)
        )
        n = torch.where(
            leg_in[..., None],
            torch.tensor([-1.0, 0.0]),
            torch.where(leg_out[..., None], torch.tensor([1.0, 0.0]), arc_n),
        )
        return c, n

    def f(uv):
        c, n = centre_and_normal(uv[..., 0])
        a = uv[..., 1] * 2 * math.pi
        x = c[..., 0] + radius * a.cos() * n[..., 0]
        y = c[..., 1] + radius * a.cos() * n[..., 1]
        zz = z + radius * a.sin()
        return torch.stack((x, y, zz), -1)

    return Surface(f, resolution=(96, 16))


WAVES = [  # direction (degrees), wavelength, amplitude, steepness
    (25, 3.4, 0.032, 0.7),
    (-30, 2.1, 0.020, 0.7),
    (80, 1.4, 0.012, 0.6),
    (155, 0.9, 0.007, 0.5),
    (-115, 0.6, 0.0045, 0.5),
    (40, 0.37, 0.0025, 0.4),
]


def gerstner(points, t):
    """Sum of Gerstner waves: points move in small circles, giving sharp crests and broad troughs."""
    seconds = t[..., 0] * LOOP
    x, y, z = points.unbind(-1)
    ox, oy, oz = x, y, z
    for angle, length, amplitude, steepness in WAVES:
        k = 2 * math.pi / length
        # Deep-water dispersion, rounded to a whole number of cycles per loop.
        cycles = max(1, round(math.sqrt(9.8 * k) * 0.35 * LOOP / (2 * math.pi)))
        omega = 2 * math.pi * cycles / LOOP
        dx, dz = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        phase = k * (dx * x + dz * z) - omega * seconds
        ox = ox + steepness * amplitude * dx * phase.cos()
        oz = oz + steepness * amplitude * dz * phase.cos()
        oy = oy + amplitude * phase.sin()
    return torch.stack((ox, oy, oz), -1)


Scene.set_environment_map(sky(), intensity=1.0, ambient=True)
Scene.clear_lights()

with Off():
    DirectionalLight(
        location=SUN * 30, target=ORIGIN, intensity=2.2, color=Color("#fff1dc")
    ).spawn()

    hw, hd = POOL_W / 2, POOL_D / 2
    rect(POOL_W, POOL_D, (0, -DEPTH, 0), (1, 0, 0), (0, 0, -1), tiles(40, 28)).spawn()
    for center, right, length in (
        ((0, -DEPTH / 2, -hd), (1, 0, 0), POOL_W),
        ((0, -DEPTH / 2, hd), (-1, 0, 0), POOL_W),
        ((-hw, -DEPTH / 2, 0), (0, 0, -1), POOL_D),
        ((hw, -DEPTH / 2, 0), (0, 0, 1), POOL_D),
    ):
        rect(length, DEPTH, center, right, (0, 1, 0), wall_tiles(length, DEPTH)).spawn()

    # Lane stripes on the floor.
    for z in (-hd / 2, hd / 2):
        rect(
            POOL_W - 2.0,
            0.22,
            (0, -DEPTH + 0.004, z),
            (1, 0, 0),
            (0, 0, -1),
            tiles(16, 1, tile="#123f63", grout="#0e3350"),
        ).spawn()

    # The deck: sandstone pavers stretching away on every side.
    pavers = dict(tile="#c9ad84", grout="#8f7a5c")
    far = 40.0
    for center, size in (
        ((0, 0, -hd - far / 2), (POOL_W + 2 * far, far)),
        ((0, 0, hd + far / 2), (POOL_W + 2 * far, far)),
        ((-hw - far / 2, 0, 0), (far, POOL_D)),
        ((hw + far / 2, 0, 0), (far, POOL_D)),
    ):
        w, d = size
        rect(
            w, d, center, (1, 0, 0), (0, 0, -1), tiles(round(w), round(d), **pavers)
        ).spawn()

    chrome = MeshStandardMaterial(color=Color("#e6e8ea"), metalness=1.0, roughness=0.08)
    for z in (-1.9, -1.1):
        ladder_rail(z, hw - 0.18, hw + 0.42, -1.1, 0.75).set_material(chrome).spawn()
    for i in range(3):
        step = Prism(width=0.14, height=0.04, depth=0.8, color=Color("#e6e8ea"))
        step.set_material(chrome).move(
            torch.tensor([hw - 0.12, -0.35 - 0.33 * i, -1.5])
        ).spawn()

    water = Surface(
        lambda uv: torch.cat(
            (
                (uv[..., :1] - 0.5) * POOL_W,
                torch.zeros_like(uv[..., :1]),
                (0.5 - uv[..., 1:]) * POOL_D,
            ),
            -1,
        ),
        resolution=(200, 140),
    ).move(UP * WATER_LEVEL)
    water.set_material(
        MeshPhysicalMaterial(
            color=Color("#9fe0ea"),
            transmission=1.0,
            ior=1.33,
            roughness=0.02,
        )
    )
    water.spawn()

    camera = Scene.get_camera()
    camera.move_to(torch.tensor([-2.6, 7.6, 11.2])).look_at(
        torch.tensor([0.7, -0.8, -0.9])
    )

Homotopy(water, gerstner, runtime=LOOP, easing=easings.linear)

if MODE == "STILL":
    Scene.save_frame(f"water_pool_{SPP}spp_{AT:.1f}.png", QUALITY, at=AT)
else:
    Scene.save_video(f"water_pool_{SPP}spp.mp4", QUALITY, overwrite=True)
