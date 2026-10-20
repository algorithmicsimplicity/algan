"""The flat vector Algan logo, in three tiers, from the logo's own typeface.

Usage:  python branding/make_logo_svg.py [outdir]

Writes, to ``docs/source/_static`` by default:

- ``algan-logo.svg`` / ``-dark.svg``: the full lockup, the wordmark and the three solids, for
  wide placements (a README or a docs hero);
- ``algan-logo-sidebar.svg`` / ``-dark.svg``: the wordmark alone, which is what reads in the
  docs' 240 px sidebar;
- ``algan-icon-light.svg`` / ``-dark.svg``: the symbol, the lambda and its wave, centred in a
  square, for avatars;
- ``algan-favicon.svg`` and ``favicon.ico``: the symbol again with a thicker wave, which is all
  that survives at 16 to 64 px.

Glyphs are outlined paths from ``fonts/AlganFranklin-Black.ttf``, so no font is needed to
display them. The solids are drawn as the 3-D logo shows them -- seen from slightly above,
straight on -- so the sphere, cube and cone share one viewpoint: Manim's circle, square and
triangle, given depth.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont

HERE = Path(__file__).resolve().parent
OUT = Path(
    sys.argv[1] if len(sys.argv) > 1 else HERE.parent / "docs" / "source" / "_static"
)
FONT = HERE / "fonts" / "AlganFranklin-Black.ttf"

BLUE = "#4da3ff"
INK = {"light": "#1c1f26", "dark": "#e8ebf0"}
BACKGROUND = "#0b0f17"
CUBE = ("#3b78d0", "#17509c", "#0f3a75")  # top, front, edge
CONE = ("#f6d474", "#e0b23f", "#a87c1c")  # lit, shade, base
SPHERE = ("#ffffff", "#c6ccd6", "#5d6472", "#1e222b")

font = TTFont(FONT)
glyphs = font.getGlyphSet()
cmap = font.getBestCmap()
CAP = font["OS/2"].sCapHeight
X_HEIGHT = font["OS/2"].sxHeight
# Where the A's crossbar was, in font units (see brand.py): the wave replaces it.
BAR_Y, BAR_X, BAR_HALF = 0.31 * CAP, 388.0, 178.0


def glyph(ch):
    name = cmap[ord(ch)]
    pen = SVGPathPen(glyphs)
    glyphs[name].draw(pen)
    bounds = BoundsPen(glyphs)
    glyphs[name].draw(bounds)
    return pen.getCommands(), glyphs[name].width, bounds.bounds


def word(text, x, baseline, cap_px):
    """Outlined ``text`` with its baseline at ``baseline``; returns (svg, right edge, scale)."""
    s = cap_px / CAP
    paths, pen_x = [], x
    for ch in text:
        d, advance, _ = glyph(ch)
        paths.append(
            f'<path transform="translate({pen_x:.2f},{baseline:.2f}) '
            f'scale({s:.6f},{-s:.6f})" d="{d}"/>'
        )
        pen_x += advance * s
    return "\n".join(paths), pen_x, s


def wave(lambda_x, baseline, s, width_factor):
    """The crossbar wave for a lambda drawn at ``lambda_x``, as a stroked path."""
    cx, cy = lambda_x + BAR_X * s, baseline - BAR_Y * s
    half, amp = BAR_HALF * s, 0.075 * CAP * s
    points = []
    for i in range(49):
        t = -half + 2 * half * i / 48
        points.append((cx + t, cy - amp * math.sin(1.5 * math.pi * t / half)))
    d = "M" + " L".join(f"{px:.2f},{py:.2f}" for px, py in points)
    return (
        f'<path d="{d}" fill="none" stroke="{BLUE}" stroke-width="{width_factor * CAP * s:.1f}" '
        'stroke-linecap="round" stroke-linejoin="round"/>'
    )


def defs():
    return (
        '<defs><radialGradient id="sph" cx="0.36" cy="0.30" r="0.72">'
        f'<stop offset="0" stop-color="{SPHERE[0]}"/><stop offset="0.32" stop-color="{SPHERE[1]}"/>'
        f'<stop offset="0.72" stop-color="{SPHERE[2]}"/><stop offset="1" stop-color="{SPHERE[3]}"/>'
        "</radialGradient></defs>"
    )


def solids(x, floor, unit):
    """Sphere, cube and cone left to right from ``x``, standing on ``floor``; ``unit`` is one
    font unit in px. Sized as in the 3-D logo: the sphere as tall as a lowercase letter, the
    cone as a capital, the cube between, half a letter after the word and a gap between each.
    """
    em = 1000 * unit
    gap = 0.2 * em
    r = 0.5 * X_HEIGHT * unit
    x += 0.30 * em
    sphere = (
        f'<circle cx="{x + r:.1f}" cy="{floor - r:.1f}" r="{r:.1f}" fill="url(#sph)"/>'
    )
    x += 2 * r + gap
    edge = (X_HEIGHT + 0.25 * (CAP - X_HEIGHT)) * unit
    top = 0.16 * edge  # the top face, seen from a little above
    cube = (
        f'<polygon points="{x:.1f},{floor - edge:.1f} {x + edge:.1f},{floor - edge:.1f} '
        f'{x + edge:.1f},{floor:.1f} {x:.1f},{floor:.1f}" fill="{CUBE[1]}"/>'
        f'<polygon points="{x:.1f},{floor - edge:.1f} {x + 0.12 * edge:.1f},{floor - edge - top:.1f} '
        f'{x + 1.12 * edge:.1f},{floor - edge - top:.1f} {x + edge:.1f},{floor - edge:.1f}" '
        f'fill="{CUBE[0]}"/>'
        f'<polygon points="{x + edge:.1f},{floor - edge:.1f} {x + 1.12 * edge:.1f},{floor - edge - top:.1f} '
        f'{x + 1.12 * edge:.1f},{floor - top:.1f} {x + edge:.1f},{floor:.1f}" fill="{CUBE[2]}"/>'
    )
    x += 1.12 * edge + gap
    cr, ch = 0.40 * CAP * unit, CAP * unit
    ry = 0.16 * cr
    cx = x + cr
    cone = (
        f'<ellipse cx="{cx:.1f}" cy="{floor - ry:.1f}" rx="{cr:.1f}" ry="{ry:.1f}" fill="{CONE[2]}"/>'
        f'<polygon points="{cx:.1f},{floor - ch:.1f} {cx - cr:.1f},{floor - ry:.1f} '
        f'{cx + cr:.1f},{floor - ry:.1f}" fill="{CONE[1]}"/>'
        f'<polygon points="{cx:.1f},{floor - ch:.1f} {cx - cr:.1f},{floor - ry:.1f} '
        f'{cx - 0.2 * cr:.1f},{floor:.1f}" fill="{CONE[0]}"/>'
    )
    return sphere + cube + cone, x + 2 * cr


def svg(width, height, body, background=None, radius=0):
    rect = (
        f'<rect width="{width}" height="{height}" rx="{radius}" fill="{background}"/>'
        if background
        else ""
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height:.0f}" '
        f'width="{width:.0f}" height="{height:.0f}" role="img" aria-label="Algan">\n'
        f"<title>Algan</title>{defs()}{rect}\n{body}\n</svg>\n"
    )


def lockup(ink, with_solids):
    cap, pad = 190.0, 30.0
    baseline = pad + cap
    paths, right, s = word("Λlgan", pad, baseline, cap)
    body = f'<g fill="{ink}">\n{paths}\n</g>\n' + wave(pad, baseline, s, 0.11)
    if with_solids:
        drawn, right = solids(right, baseline, s)
        body += drawn
    descender = -glyph("g")[2][1] * s
    return svg(right + pad, baseline + descender + pad, body)


def symbol(ink, background, size, wave_width, radius):
    """The lambda and its wave centred in a ``size`` square."""
    _, advance, (x0, _, x1, _) = glyph("Λ")
    cap = 0.62 * size
    s = cap / CAP
    x = 0.5 * size - 0.5 * (x0 + x1) * s
    baseline = 0.5 * size + 0.5 * cap
    paths, _, _ = word("Λ", x, baseline, cap)
    body = f'<g fill="{ink}">\n{paths}\n</g>\n' + wave(x, baseline, s, wave_width)
    return svg(size, size, body, background, radius)


def favicon_ico(path):
    """Rasterize the favicon at the sizes browsers ask for, straight from the font."""
    from PIL import Image, ImageDraw, ImageFont

    big = 256
    image = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((0, 0, big - 1, big - 1), radius=50, fill=BACKGROUND)
    cap = 0.66 * big
    face = ImageFont.truetype(str(FONT), int(round(cap * 1000 / CAP)))
    _, _, (x0, _, x1, _) = glyph("Λ")
    s = cap / CAP
    x = 0.5 * big - 0.5 * (x0 + x1) * s
    baseline = 0.5 * big + 0.5 * cap
    draw.text((x, baseline), "Λ", font=face, fill=INK["dark"], anchor="ls")
    cx, cy = x + BAR_X * s, baseline - BAR_Y * s
    half, amp = BAR_HALF * s, 0.075 * CAP * s
    points = [
        (cx + t, cy - amp * math.sin(1.5 * math.pi * t / half))
        for t in (-half + 2 * half * i / 48 for i in range(49))
    ]
    width = int(round(0.16 * CAP * s))
    draw.line(points, fill=BLUE, width=width, joint="curve")
    for end in (points[0], points[-1]):
        draw.ellipse(
            (
                end[0] - width / 2,
                end[1] - width / 2,
                end[0] + width / 2,
                end[1] + width / 2,
            ),
            fill=BLUE,
        )
    image.save(path, sizes=[(16, 16), (32, 32), (48, 48), (64, 64)])


OUT.mkdir(parents=True, exist_ok=True)
for theme, ink in INK.items():
    suffix = "" if theme == "light" else "-dark"
    (OUT / f"algan-logo{suffix}.svg").write_text(lockup(ink, True), encoding="utf-8")
    (OUT / f"algan-logo-sidebar{suffix}.svg").write_text(
        lockup(ink, False), encoding="utf-8"
    )
(OUT / "algan-icon-light.svg").write_text(
    symbol(INK["light"], None, 512, 0.11, 0), encoding="utf-8"
)
(OUT / "algan-icon-dark.svg").write_text(
    symbol(INK["dark"], BACKGROUND, 512, 0.11, 96), encoding="utf-8"
)
(OUT / "algan-favicon.svg").write_text(
    symbol(INK["dark"], BACKGROUND, 512, 0.16, 100), encoding="utf-8"
)
favicon_ico(OUT / "favicon.ico")
print("wrote SVGs and favicon.ico to", OUT)
