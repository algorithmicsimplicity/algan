"""Build ``AlganFranklin-Black.ttf``, the logo's typeface, from Libre Franklin Black.

Usage:  python branding/fonts/build_font.py path/to/LibreFranklin-Black.ttf

The logo was drawn in Franklin Gothic Heavy, which ships only with Windows, so the branding
scripts rendered a different face (or Pango's silent fallback) everywhere else. Libre Franklin
(SIL OFL 1.1, https://github.com/googlefonts/Libre-Franklin) is its open revival. This script
makes three changes to its static Black instance and writes the result beside itself:

- adds a Greek capital lambda (U+039B), which Libre Franklin lacks: the A's outer contour with
  the crossbar and counter cut away, so the two glyphs stay interchangeable in the video;
- tightens every advance by 2% of the em, and the lambda's by a further 4%, whose diagonal
  otherwise leaves a wider gap before the l than the other letters have;
- renames the family "Algan Franklin", so an installed Libre Franklin, which has no lambda, can
  never be picked in its place.

The OFL carries no Reserved Font Name for Libre Franklin, so the modified font may keep any
name; the new one only prevents that collision. ``OFL.txt`` beside this file is its licence.
"""

from __future__ import annotations

import sys
from pathlib import Path

from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont

OUT = Path(__file__).with_name("AlganFranklin-Black.ttf")
FAMILY = "Algan Franklin"
TRACKING = 20  # font units (of 1000) taken off every advance
LAMBDA_EXTRA = 40  # and off the lambda's


def add_lambda(font):
    """The A's outer contour up to its right foot, then straight up the inner leg edges."""
    pen = TTGlyphPen(font.getGlyphSet())
    pen.moveTo((43, 0))
    pen.qCurveTo((32, 0), (22, 21), (28, 37))
    pen.lineTo((270, 719))
    pen.qCurveTo((275, 731), (287, 742), (300, 742))
    pen.lineTo((497, 742))
    pen.qCurveTo((523, 742), (533, 715))
    pen.lineTo((773, 34))
    pen.qCurveTo((777, 24), (770, 0), (759, 0))
    pen.lineTo((563, 0))
    pen.qCurveTo((546, 0), (541, 17))
    pen.lineTo((386, 505))
    pen.qCurveTo((384, 510), (376, 510), (374, 505))
    pen.lineTo((219, 15))
    pen.qCurveTo((217, 9), (206, 0), (195, 0))
    pen.closePath()
    name = "uni039B"
    font["glyf"][name] = pen.glyph()
    font["hmtx"][name] = font["hmtx"][font.getBestCmap()[ord("A")]]
    if name not in font.getGlyphOrder():
        font.setGlyphOrder(font.getGlyphOrder() + [name])
    for table in font["cmap"].tables:
        if table.isUnicode():
            table.cmap[0x39B] = name
    font["maxp"].numGlyphs = len(font.getGlyphOrder())
    return name


def track(font, lam):
    hmtx = font["hmtx"]
    for name in font.getGlyphOrder():
        advance, lsb = hmtx[name]
        if advance:
            hmtx[name] = (
                advance - TRACKING - (LAMBDA_EXTRA if name == lam else 0),
                lsb,
            )


def rename(font):
    names = {
        1: f"{FAMILY} Black",
        3: f"{FAMILY.replace(' ', '')}-Black;algan",
        4: f"{FAMILY} Black",
        6: f"{FAMILY.replace(' ', '')}-Black",
        16: FAMILY,
        17: "Black",
    }
    table = font["name"]
    for record in list(table.names):
        if record.nameID in names:
            table.setName(
                names[record.nameID],
                record.nameID,
                record.platformID,
                record.platEncID,
                record.langID,
            )
    # Variable-font remnants name the instance after the old family.
    if "STAT" in font:
        del font["STAT"]


def main(source):
    font = TTFont(source)
    lam = add_lambda(font)
    track(font, lam)
    rename(font)
    font.save(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main(sys.argv[1])
