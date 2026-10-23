"""Sunlit water rippling over the tiles of a swimming pool, path traced.

The water is a Surface deformed by a sum of Gerstner waves (a Homotopy), with a
refractive physical material. Algan does not trace caustics, so the script
computes them itself: every frame it refracts a dense grid of sun rays through
the same wave surface, gathers where they land on the pool floor, and paints
the result into the floor's texture.

Usage::

    python examples/water_pool.py          # the 6 s looping video, 1080p at 80 spp
    python examples/water_pool.py still    # one 1080p frame at 256 spp
    python examples/water_pool.py draft    # a quick low-resolution, low-sample video

The full video is path traced at 80 samples per pixel: about 45 minutes on a
Tesla T4 GPU, and far longer on a CPU, so start with ``draft``.
"""

from __future__ import annotations

import math
import sys

import torch

from algan import *

MODE = sys.argv[1] if len(sys.argv) > 1 else "video"
QUALITY, SPP = {"video": (HD, 80), "still": (HD, 256), "draft": (PREVIEW, 8)}[MODE]
STILL_AT = 2.0  # seconds into the loop

SETTINGS.raytracing.set(samples_per_pixel=SPP, max_bounces=8)

POOL_W, POOL_D, DEPTH = 10.0, 7.0, 1.6  # the basin, in x, z and y
WATER_LEVEL = -0.12
LOOP = 6.0  # seconds; every wave completes whole periods, so the video loops
FPS = QUALITY.frames_per_second
WATER_IOR = 1.33

SUN = torch.tensor([0.30, 0.82, -1.0])
SUN = SUN / SUN.norm()

FLOOR_TEXELS = (1024, 716)  # 40 x 28 tiles, ~25 texels each
FLOOR_TILES = (40, 28)

RNG = torch.Generator().manual_seed(7)


# -- Sky ----------------------------------------------------------------------


def sky(height=1024, width=2048):
    """An HDR sky: deep blue overhead, hazy at the horizon, and a sun far brighter than white."""
    lat = (0.5 - (torch.arange(height) + 0.5) / height) * math.pi
    lon = ((torch.arange(width) + 0.5) / width - 0.5) * 2 * math.pi
    lat, lon = torch.meshgrid(lat, lon, indexing="ij")
    d = torch.stack((lat.cos() * lon.cos(), lat.sin(), lat.cos() * lon.sin()), -1)
    up = d[..., 1:2].clamp(min=0)
    zenith = torch.tensor([0.10, 0.27, 0.72])
    horizon = torch.tensor([0.78, 0.86, 0.97])
    env = horizon + (zenith - horizon) * up**0.45
    # A few soft, wispy clouds, so the reflections on the water have some structure.
    clouds = get_noise(
        (BLACK, WHITE),
        resolution=(24, 6),
        octaves=5,
        seed=3,
        texture_resolution=(width, height),
    )[..., 0].T
    clouds = ((clouds - 0.55) * 4).clamp(0, 1) * (up[..., 0] > 0.04).float()
    env = env * (1 - 0.8 * clouds[..., None]) + 1.05 * clouds[..., None]
    ground = torch.tensor([0.30, 0.31, 0.25])
    env = torch.where(d[..., 1:2] < 0, ground, env)
    cos_sun = (d * SUN).sum(-1, keepdim=True)
    env = env + torch.tensor([1.0, 0.92, 0.78]) * 0.6 * torch.exp((cos_sun - 1) * 30)
    disc = (cos_sun > math.cos(math.radians(1.2))).float()
    return env + torch.tensor([1.0, 0.95, 0.85]) * 600.0 * disc


# -- Textures -----------------------------------------------------------------


def tiles(cells, texels, tile, grout, line_width=0.08, variation=0.05):
    """Grouted tiles, each a slightly different shade, as real glazed tiles are."""
    texture = get_grid_lines(
        Color(grout),
        Color(tile),
        resolution=cells,
        line_width=line_width,
        texture_resolution=texels,
    )
    u = torch.arange(texels[0]) * cells[0] // texels[0]
    v = torch.arange(texels[1]) * cells[1] // texels[1]
    shade = 1 + variation * torch.randn(cells, generator=RNG)
    grain = 1 + 0.025 * torch.randn(texels, generator=RNG)
    texture[..., :3] *= (shade[u][:, v] * grain)[..., None]
    return texture.clamp(0, 1)


def wall_tiles(length, height, per_unit=4):
    """Pale tiles with a band of deep blue mosaic along the waterline."""
    cells = (round(length * per_unit), round(height * per_unit))
    texels = (cells[0] * 24, cells[1] * 24)
    body = tiles(cells, texels, "#dcecf1", "#a2bcc5")
    band = tiles(
        (cells[0] * 3, cells[1] * 3), texels, "#1d5f8a", "#123b55", variation=0.12
    )
    v = torch.linspace(0, 1, texels[1]).view(1, -1, 1)
    in_band = (v > 1 - 0.3 / height).float()
    return body * (1 - in_band) + band * in_band


FLOOR = tiles(FLOOR_TILES, FLOOR_TEXELS, "#cde9f0", "#98b8c2")
# Lane stripes, painted into the floor tiles.
_v = (torch.arange(FLOOR_TEXELS[1]) + 0.5) / FLOOR_TEXELS[1]
_u = (torch.arange(FLOOR_TEXELS[0]) + 0.5) / FLOOR_TEXELS[0]
_stripe = ((_v.view(1, -1) - 0.25).abs() < 0.016) | (
    (_v.view(1, -1) - 0.75).abs() < 0.016
)
_stripe = _stripe & ((_u.view(-1, 1) - 0.5).abs() < 0.4)
FLOOR[..., :3] = torch.where(
    _stripe[..., None],
    torch.tensor([0.05, 0.20, 0.36]) * FLOOR[..., :3] / 0.85,
    FLOOR[..., :3],
)


# -- Waves --------------------------------------------------------------------

WAVES = [  # direction (degrees), wavelength, amplitude, steepness
    (25, 3.4, 0.022, 0.8),
    (-30, 2.1, 0.012, 0.8),
    (80, 1.4, 0.0065, 0.7),
    (155, 0.9, 0.0042, 0.6),
    (-115, 0.6, 0.0032, 0.6),
    (-10, 0.47, 0.0024, 0.5),
    (40, 0.37, 0.0020, 0.5),
    (115, 0.29, 0.0015, 0.5),
    (-70, 0.23, 0.0011, 0.4),
]


def gerstner(points, t):
    """Sum of Gerstner waves: points move in small circles, giving sharp crests and broad troughs."""
    seconds = t[..., 0] * LOOP
    x, y, z = points.unbind(-1)
    ox, oy, oz = x, y, z
    for angle, length, amplitude, steepness in WAVES:
        k = 2 * math.pi / length
        # Deep-water dispersion, slowed for a pool and rounded to whole cycles per loop.
        cycles = max(1, round(math.sqrt(9.8 * k) * 0.35 * LOOP / (2 * math.pi)))
        omega = 2 * math.pi * cycles / LOOP
        dx, dz = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        phase = k * (dx * x + dz * z) - omega * seconds
        ox = ox + steepness * amplitude * dx * phase.cos()
        oz = oz + steepness * amplitude * dz * phase.cos()
        oy = oy + amplitude * phase.sin()
    return torch.stack((ox, oy, oz), -1)


# -- Caustics -----------------------------------------------------------------


def sun_on_floor(seconds, rays=(2400, 1680), waves=True):
    """How much sunlight reaches each floor texel through the water, by refracting sun rays.

    Rays start on a grid over the water's rest plane; each carries the sunlight
    falling on its patch of the displaced surface, is refracted by Snell's law
    at the local normal, and is splatted where it meets the floor.
    """
    u = (torch.arange(rays[0]) + 0.5) / rays[0]
    v = (torch.arange(rays[1]) + 0.5) / rays[1]
    u, v = torch.meshgrid(u, v, indexing="ij")
    rest = torch.stack(
        ((u - 0.5) * POOL_W, torch.full_like(u, WATER_LEVEL), (0.5 - v) * POOL_D), -1
    )
    p = gerstner(rest, torch.tensor([[[seconds / LOOP]]])) if waves else rest
    du = torch.gradient(p, dim=0)[0]
    dv = torch.gradient(p, dim=1)[0]
    area = torch.cross(du, dv, dim=-1)  # points up; its length is the patch's area
    n = area / area.norm(dim=-1, keepdim=True)
    d = -SUN
    cos_i = -(n * d).sum(-1, keepdim=True)
    flux = (area * SUN).sum(-1).clamp(min=0)  # sunlight intercepted by the patch
    eta = 1 / WATER_IOR
    k = 1 - eta**2 * (1 - cos_i**2)
    r = eta * d + (eta * cos_i - k.clamp(min=0).sqrt()) * n
    r0 = ((1 - WATER_IOR) / (1 + WATER_IOR)) ** 2
    flux = flux * (1 - (r0 + (1 - r0) * (1 - cos_i[..., 0]) ** 5))
    s = (p[..., 1] + DEPTH) / (-r[..., 1]).clamp(min=1e-3)
    hit = p + s[..., None] * r
    # Bilinear splat into the floor texture's (u, v) grid.
    fu = (hit[..., 0] / POOL_W + 0.5) * FLOOR_TEXELS[0] - 0.5
    fv = (0.5 - hit[..., 2] / POOL_D) * FLOOR_TEXELS[1] - 0.5
    iu, iv = fu.floor(), fv.floor()
    au, av = fu - iu, fv - iv
    out = torch.zeros(FLOOR_TEXELS[0] * FLOOR_TEXELS[1])
    for ou, wu in ((0, 1 - au), (1, au)):
        for ov, wv in ((0, 1 - av), (1, av)):
            cu, cv = iu + ou, iv + ov
            ok = (cu >= 0) & (cu < FLOOR_TEXELS[0]) & (cv >= 0) & (cv < FLOOR_TEXELS[1])
            idx = (cu * FLOOR_TEXELS[1] + cv)[ok].long()
            out.index_add_(0, idx, (flux * wu * wv)[ok])
    out = out.view(1, 1, *FLOOR_TEXELS)
    # The sun is half a degree wide, which softens the pattern by about a texel.
    blur = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])
    blur = (blur[:, None] * blur[None]) / 256
    out = torch.nn.functional.conv2d(out, blur.view(1, 1, 5, 5), padding=2)
    return out[0, 0]


_FLAT = sun_on_floor(0.0, waves=False)
_LIT = _FLAT.max() * 0.02 < _FLAT


def floor_texture(seconds):
    """The floor tiles, brightened along the caustic network and dimmed between its lines."""
    ratio = torch.where(
        _LIT, sun_on_floor(seconds) / _FLAT.clamp(min=1e-9), torch.ones(())
    )
    gain = (0.5 + 0.5 * ratio) / 1.5
    texture = FLOOR.clone()
    texture[..., :3] = (texture[..., :3] * gain[..., None]).clamp(0, 1)
    return texture


# -- Scene --------------------------------------------------------------------


def rect(width, height, center, right, up, texture):
    """A flat textured rectangle spanning ``right`` (u) and ``up`` (v) about ``center``."""
    right = torch.tensor(right, dtype=torch.float32)
    up = torch.tensor(up, dtype=torch.float32)

    def f(uv):
        return (uv[..., :1] - 0.5) * width * right + (uv[..., 1:] - 0.5) * height * up

    return Surface(f, grid_width=2, grid_height=2, color_texture=texture).move(
        torch.tensor(center, dtype=torch.float32)
    )


def coping(length, edge, along, outward, lip, thick, over, texture):
    """One side of the pool's coping, swept along ``along`` from a bullnose profile.

    The profile runs over the flat top from the deck side, round a half-round
    nose that overhangs the water by ``over``, and back along the underside.
    """
    along = torch.tensor(along, dtype=torch.float32)
    outward = torch.tensor(outward, dtype=torch.float32)
    r = thick / 2
    legs = (lip - r + over, math.pi * r, lip - r + over)
    total = sum(legs)

    def f(uv):
        d = uv[..., 1:] * total
        nose = -over + r  # the nose's centre, across the edge
        angle = ((d - legs[0]) / r).clamp(0, math.pi)
        across = torch.where(
            d < legs[0],
            lip - d,
            torch.where(
                d > legs[0] + legs[1],
                nose + (d - legs[0] - legs[1]),
                nose - r * angle.sin(),
            ),
        )
        height = torch.where(
            d < legs[0],
            torch.full_like(d, thick),
            torch.where(
                d > legs[0] + legs[1], torch.zeros_like(d), r + r * angle.cos()
            ),
        )
        return (uv[..., :1] - 0.5) * length * along + across * outward + height * UP

    return Surface(f, grid_width=2, grid_height=40, color_texture=texture).move(
        torch.tensor(edge, dtype=torch.float32)
    )


def ladder_rail(z, inner_x, outer_x, bottom, top, radius=0.035):
    """A chrome grab rail: up out of the water, over the coping, and down onto the deck."""
    bend = (outer_x - inner_x) / 2
    legs = (top - bottom, math.pi * bend, top)
    total = sum(legs)

    def f(uv):
        d = uv[..., 0] * total
        leg_in = (d < legs[0])[..., None]
        leg_out = (d > legs[0] + legs[1])[..., None]
        angle = ((d - legs[0]) / bend).clamp(0, math.pi)
        arc = torch.stack(
            (inner_x + bend - bend * angle.cos(), top + bend * angle.sin()), -1
        )
        arc_n = torch.stack((-angle.cos(), angle.sin()), -1)
        up = torch.stack((torch.full_like(d, inner_x), bottom + d), -1)
        down = torch.stack(
            (torch.full_like(d, outer_x), top - (d - legs[0] - legs[1])), -1
        )
        c = torch.where(leg_in, up, torch.where(leg_out, down, arc))
        n = torch.where(
            leg_in,
            torch.tensor([-1.0, 0.0]),
            torch.where(leg_out, torch.tensor([1.0, 0.0]), arc_n),
        )
        a = uv[..., 1] * 2 * math.pi
        return torch.stack(
            (
                c[..., 0] + radius * a.cos() * n[..., 0],
                c[..., 1] + radius * a.cos() * n[..., 1],
                z + radius * a.sin(),
            ),
            -1,
        )

    return Surface(f, resolution=(96, 16))


Scene.set_environment_map(sky(), intensity=1.0, ambient=True)
Scene.clear_lights()

hw, hd = POOL_W / 2, POOL_D / 2
with Off():
    DirectionalLight(
        location=SUN * 30, target=ORIGIN, intensity=4.5, color=Color("#fff1dc")
    ).spawn()

    start = STILL_AT if MODE == "still" else 0.0
    floor = rect(
        POOL_W, POOL_D, (0, -DEPTH, 0), (1, 0, 0), (0, 0, -1), floor_texture(start)
    )
    floor.spawn()
    for center, right, length in (
        ((0, -DEPTH / 2, -hd), (1, 0, 0), POOL_W),
        ((0, -DEPTH / 2, hd), (-1, 0, 0), POOL_W),
        ((-hw, -DEPTH / 2, 0), (0, 0, -1), POOL_D),
        ((hw, -DEPTH / 2, 0), (0, 0, 1), POOL_D),
    ):
        rect(length, DEPTH, center, right, (0, 1, 0), wall_tiles(length, DEPTH)).spawn()

    # The deck: sandstone pavers stretching away on every side.
    far = 40.0
    for center, (w, d) in (
        ((0, 0, -hd - far / 2), (POOL_W + 2 * far, far)),
        ((0, 0, hd + far / 2), (POOL_W + 2 * far, far)),
        ((-hw - far / 2, 0, 0), (far, POOL_D)),
        ((hw + far / 2, 0, 0), (far, POOL_D)),
    ):
        cells = (round(w / 0.6), round(d / 0.6))
        rect(
            w,
            d,
            center,
            (1, 0, 0),
            (0, 0, -1),
            tiles(
                cells,
                (min(4096, cells[0] * 16), min(4096, cells[1] * 16)),
                "#b9a080",
                "#7f6f58",
                line_width=0.05,
                variation=0.04,
            ),
        ).spawn()

    # Bullnose coping: pale stone that overhangs the water all the way round.
    lip, thick, over = 0.4, 0.08, 0.06
    stone = get_noise(
        (Color("#c9c0ad"), Color("#e2dbcb"), Color("#d3cbb9")),
        resolution=(48, 3),
        octaves=5,
        seed=11,
        texture_resolution=(1024, 64),
    )
    for length, center, along, outward in (
        (POOL_W + 2 * lip, (0, 0, -hd), (1, 0, 0), (0, 0, -1)),
        (POOL_W + 2 * lip, (0, 0, hd), (-1, 0, 0), (0, 0, 1)),
        (POOL_D, (-hw, 0, 0), (0, 0, 1), (-1, 0, 0)),
        (POOL_D, (hw, 0, 0), (0, 0, -1), (1, 0, 0)),
    ):
        coping(length, center, along, outward, lip, thick, over, stone).set_material(
            MeshStandardMaterial(roughness=0.75)
        ).spawn()

    chrome = MeshStandardMaterial(color=Color("#e6e8ea"), metalness=1.0, roughness=0.08)
    for z in (-1.9, -1.1):
        ladder_rail(z, hw - 0.18, hw + 0.55, -1.1, 0.8).set_material(chrome).spawn()
    for i in range(3):
        Prism(width=0.14, height=0.04, depth=0.8, color=Color("#e6e8ea")).set_material(
            chrome
        ).move(torch.tensor([hw - 0.12, -0.35 - 0.33 * i, -1.5])).spawn()

    water = Surface(
        lambda uv: torch.cat(
            (
                (uv[..., :1] - 0.5) * POOL_W,
                torch.zeros_like(uv[..., :1]),
                (0.5 - uv[..., 1:]) * POOL_D,
            ),
            -1,
        ),
        resolution=(250, 175),
    ).move(UP * WATER_LEVEL)
    water.set_material(
        MeshPhysicalMaterial(
            color=Color("#b3ebf0"), transmission=1.0, ior=WATER_IOR, roughness=0.015
        )
    )
    water.spawn()

    camera = Scene.get_camera()
    camera.move_to(torch.tensor([-2.8, 5.4, 11.4])).look_at(
        torch.tensor([0.6, -0.7, -1.2])
    )

with Sync():
    Homotopy(water, gerstner, runtime=LOOP, easing=easings.linear)
    if MODE != "still":
        # One caustic texture per frame; Algan blends between them.
        keys = round(LOOP * FPS)
        with Seq(runtime_per_part=LOOP / keys, easing=easings.linear):
            for i in range(1, keys + 1):
                floor.color_texture = floor_texture(i * LOOP / keys)

if MODE == "still":
    Scene.save_frame("water_pool.png", QUALITY, at=STILL_AT)
else:
    Scene.save_video(f"water_pool_{MODE}.mp4", QUALITY, overwrite=True)
