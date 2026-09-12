"""Flat vector Algan logo (wordmark, icon, favicon) from Franklin Gothic Heavy outlines.

Usage:  python branding/make_logo_svg.py [outdir] [path/to/FRAHV.TTF]

Writes algan-logo-sidebar.svg / -dark.svg (docs sidebar), algan-icon-light.svg / -dark.svg and
algan-favicon.svg. Glyphs are outlined paths, so no font is needed to display them.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont

OUT = sys.argv[1] if len(sys.argv) > 1 else "docs/source/_static"
FONT = (
    sys.argv[2]
    if len(sys.argv) > 2
    else next(
        p
        for p in [
            r"C:/Windows/Fonts/FRAHV.TTF",
            os.path.expanduser("~/.fonts/FRAHV.TTF"),
            "/usr/share/fonts/truetype/msttcorefonts/FRAHV.TTF",
        ]
        if os.path.exists(p)
    )
)
WAVE = "#3d8ef5"
CUBE = ("#5b82ff", "#2b52e6", "#1b3aa8")  # top, front-left, front-right
CONE = ("#f6d474", "#e0b23f", "#b98a1f")  # lit, shade, base
SPHERE = ("#ffffff", "#c6ccd6", "#5d6472", "#1e222b")

font = TTFont(FONT)
gs = font.getGlyphSet()
cmap = font.getBestCmap()
upm = font["head"].unitsPerEm
# Older OS/2 tables (Franklin Gothic Heavy's is version 1) carry no sCapHeight: measure "H".
CAP = getattr(font["OS/2"], "sCapHeight", 0) or 0
if not CAP:
    from fontTools.pens.boundsPen import BoundsPen

    _pen = BoundsPen(gs)
    gs[cmap[ord("H")]].draw(_pen)
    CAP = _pen.bounds[3]


def glyph(ch):
    name = cmap[ord(ch)]
    pen = SVGPathPen(gs)
    gs[name].draw(pen)
    return pen.getCommands(), gs[name].width


def word_paths(text, x, baseline, cap_px):
    """Return (svg, right_edge, per-glyph [(x_left, adv)]) in px. y flipped by transform."""
    s = cap_px / CAP
    out, pos, pen_x = [], [], x
    for ch in text:
        d, adv = glyph(ch)
        out.append(
            f'<path transform="translate({pen_x:.2f},{baseline:.2f}) scale({s:.6f},{-s:.6f})" d="{d}"/>'
        )
        pos.append((pen_x, adv * s))
        pen_x += adv * s
    return "\n".join(out), pen_x, pos


def wave_path(cx, cy, half_w, amp, periods=1.5, n=48):
    pts = []
    for i in range(n + 1):
        t = -half_w + 2 * half_w * i / n
        pts.append((cx + t, cy - amp * math.sin(math.pi * periods * t / half_w)))
    return "M" + " L".join(f"{px:.2f},{py:.2f}" for px, py in pts)


def solids(x, floor_y, u):
    """Three flat solids; u = one world unit in px. Returns svg (cube, cone, sphere order)."""
    # cube: edge-on, size 1.7u, standing on the floor
    cs = 1.7 * u
    cx = x + 2.0 * u
    hw = cs * 0.60
    top_dy = cs * 0.22
    ftop = floor_y - cs
    cube = (
        f'<polygon points="{cx:.1f},{ftop - top_dy:.1f} {cx + hw:.1f},{ftop - top_dy * 0.45:.1f} {cx:.1f},{ftop + top_dy * 0.1:.1f} {cx - hw:.1f},{ftop - top_dy * 0.45:.1f}" fill="{CUBE[0]}"/>'
        f'<polygon points="{cx - hw:.1f},{ftop - top_dy * 0.45:.1f} {cx:.1f},{ftop + top_dy * 0.1:.1f} {cx:.1f},{floor_y:.1f} {cx - hw:.1f},{floor_y - top_dy * 0.55:.1f}" fill="{CUBE[1]}"/>'
        f'<polygon points="{cx:.1f},{ftop + top_dy * 0.1:.1f} {cx + hw:.1f},{ftop - top_dy * 0.45:.1f} {cx + hw:.1f},{floor_y - top_dy * 0.55:.1f} {cx:.1f},{floor_y:.1f}" fill="{CUBE[2]}"/>'
    )
    # cone: radius .9u height 2u
    r = 0.9 * u
    h = 2.0 * u
    ccx = x + 3.35 * u
    base = floor_y - 0.02 * u
    ry = r * 0.28
    cone = (
        f'<ellipse cx="{ccx:.1f}" cy="{base:.1f}" rx="{r:.1f}" ry="{ry:.1f}" fill="{CONE[2]}"/>'
        f'<polygon points="{ccx:.1f},{base - h:.1f} {ccx - r:.1f},{base:.1f} {ccx + r:.1f},{base:.1f}" fill="{CONE[1]}"/>'
        f'<polygon points="{ccx:.1f},{base - h:.1f} {ccx - r * 0.15:.1f},{base + ry * 0.2:.1f} {ccx + r:.1f},{base:.1f}" fill="{CONE[0]}"/>'
    )
    # sphere: radius 1u, in front
    sr = 1.0 * u
    scx = x + 0.7 * u
    scy = floor_y - sr
    sphere = f'<circle cx="{scx:.1f}" cy="{scy:.1f}" r="{sr:.1f}" fill="url(#sph)"/>'
    return cube + cone + sphere, ccx + r


def defs():
    return (
        '<defs><radialGradient id="sph" cx="0.36" cy="0.30" r="0.72">'
        f'<stop offset="0" stop-color="{SPHERE[0]}"/><stop offset="0.32" stop-color="{SPHERE[1]}"/>'
        f'<stop offset="0.72" stop-color="{SPHERE[2]}"/><stop offset="1" stop-color="{SPHERE[3]}"/>'
        "</radialGradient></defs>"
    )


def banner(ink, bg=None):
    W, H = 1200, 340
    cap = 190.0
    base_y = 250.0
    u = (
        cap / 2.25
    )  # one world unit of the render, in px: the wordmark's cap height is ~2.25 units
    paths, right, pos = word_paths("\u039blgan", 40, base_y, cap)
    lam_x, lam_adv = pos[0]
    lam_w = lam_adv * 0.92
    cx = lam_x + lam_adv / 2 + 0.05 * lam_w
    cy = base_y - 0.40 * cap
    wave = wave_path(cx, cy, 0.34 * lam_adv, 0.058 * cap)
    sol, edge = solids(right + 0.3 * u, base_y + 0.03 * cap, u)
    bgrect = f'<rect width="{W}" height="{H}" fill="{bg}"/>' if bg else ""
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="Algan">
<title>Algan</title>{defs()}{bgrect}
<g fill="{ink}" stroke="{ink}" stroke-width="{0.03 * cap:.1f}" stroke-linejoin="round">
{paths}
</g>
<path d="{wave}" fill="none" stroke="{WAVE}" stroke-width="{0.075 * cap:.1f}" stroke-linecap="round" stroke-linejoin="round"/>
{sol}
</svg>'''


def icon(ink, bg=None):
    W = 512
    cap = 236.0
    base_y = 440.0
    paths, right, pos = word_paths("Λ", 30, base_y, cap)
    lam_x, lam_adv = pos[0]
    cx = lam_x + lam_adv / 2 + 0.05 * lam_adv
    cy = base_y - 0.40 * cap
    wave = wave_path(cx, cy, 0.34 * lam_adv, 0.058 * cap)
    u = (
        W - 22 - right
    ) / 3.75  # cluster spans x .. x + 4.25u, starting 0.5u left of `right`
    sol, edge = solids(right - 0.5 * u, base_y + 0.03 * cap, u)
    bgrect = f'<rect width="{W}" height="{W}" rx="96" fill="{bg}"/>' if bg else ""
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {W}" width="{W}" height="{W}" role="img" aria-label="Algan">
<title>Algan</title>{defs()}{bgrect}
<g fill="{ink}" stroke="{ink}" stroke-width="{0.03 * cap:.1f}" stroke-linejoin="round">
{paths}
</g>
<path d="{wave}" fill="none" stroke="{WAVE}" stroke-width="{0.075 * cap:.1f}" stroke-linecap="round" stroke-linejoin="round"/>
{sol}
</svg>'''


def favicon(ink, bg):
    """Lambda + wave only: the three solids are noise at 16-32 px."""
    W = 512
    cap = 330.0
    base_y = 430.0
    paths, right, pos = word_paths("Λ", 0, base_y, cap)
    lam_x, lam_adv = pos[0]
    dx = (W - lam_adv) / 2
    paths, right, pos = word_paths("Λ", dx, base_y, cap)
    cx = dx + lam_adv / 2 + 0.05 * lam_adv
    cy = base_y - 0.40 * cap
    wave = wave_path(cx, cy, 0.36 * lam_adv, 0.07 * cap)
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {W}" width="{W}" height="{W}" role="img" aria-label="Algan">
<title>Algan</title><rect width="{W}" height="{W}" rx="100" fill="{bg}"/>
<g fill="{ink}" stroke="{ink}" stroke-width="{0.05 * cap:.1f}" stroke-linejoin="round">
{paths}
</g>
<path d="{wave}" fill="none" stroke="{WAVE}" stroke-width="{0.11 * cap:.1f}" stroke-linecap="round" stroke-linejoin="round"/>
</svg>'''


Path(OUT, "algan-logo-sidebar.svg").write_text(banner("#1c1f26"), encoding="utf-8")
Path(OUT, "algan-logo-sidebar-dark.svg").write_text(banner("#e8ebf0"), encoding="utf-8")
Path(OUT, "algan-icon-light.svg").write_text(icon("#1c1f26"), encoding="utf-8")
Path(OUT, "algan-icon-dark.svg").write_text(
    icon("#e8ebf0", bg="#0b0c10"), encoding="utf-8"
)
Path(OUT, "algan-favicon.svg").write_text(
    favicon("#f2f4f8", "#0b0c10"), encoding="utf-8"
)
print("wrote SVGs to", OUT)
