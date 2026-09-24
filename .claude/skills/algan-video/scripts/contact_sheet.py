#!/usr/bin/env python3
"""Build a labelled contact sheet from rendered stills or from a rendered video.

Stills (for example the save-frame checkpoints of a Project):
    python contact_sheet.py stills "renders/stills/s3_*.png" -o review/scene3.png

Video (samples frames at a fixed rate with FFmpeg, labelled with their time):
    python contact_sheet.py video renders/scenes/3_intro.mp4 --fps 1 -o review/scene3_motion.png

A contact sheet is a review aid: it shows composition, framing, text legibility and
how motion develops between samples. It does not replace watching the clip for
timing, audio, or motion between samples. Requires Pillow; video mode also
requires an FFmpeg binary (PATH or imageio-ffmpeg).
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile


def _natural_key(path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", os.path.basename(path))]


def _ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def build_sheet(images, labels, out, per_row, thumb_width, background=(40, 40, 40)):
    from PIL import Image, ImageDraw

    if not images:
        raise SystemExit("No images to place on the sheet.")
    thumbs = []
    for path in images:
        im = Image.open(path).convert("RGB")
        if thumb_width and im.width != thumb_width:
            im = im.resize((thumb_width, round(im.height * thumb_width / im.width)))
        thumbs.append(im)
    w = max(t.width for t in thumbs)
    h = max(t.height for t in thumbs)
    label_h, gap = 16, 6
    rows = (len(thumbs) + per_row - 1) // per_row
    cols = min(per_row, len(thumbs))
    sheet = Image.new("RGB", (cols * (w + gap), rows * (h + label_h + gap)), background)
    draw = ImageDraw.Draw(sheet)
    for k, (im, text) in enumerate(zip(thumbs, labels)):
        x, y = (k % per_row) * (w + gap), (k // per_row) * (h + label_h + gap)
        sheet.paste(im, (x, y + label_h))
        draw.text((x + 3, y + 2), text[: max(8, w // 6)], fill=(230, 230, 230))
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    sheet.save(out)
    return out, len(thumbs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("stills", help="Sheet from image files matching one or more glob patterns.")
    s.add_argument("patterns", nargs="+")
    v = sub.add_parser("video", help="Sheet from frames sampled out of a video.")
    v.add_argument("video")
    v.add_argument("--fps", type=float, default=1.0, help="Samples per second of video (default 1).")
    for p in (s, v):
        p.add_argument("-o", "--out", required=True, help="Output PNG path.")
        p.add_argument("--per-row", type=int, default=8, help="Thumbnails per row (default 8).")
        p.add_argument("--thumb-width", type=int, default=240,
                       help="Thumbnail width in pixels; 0 keeps the source size (default 240).")
    args = parser.parse_args(argv)

    try:
        import PIL  # noqa: F401
    except ImportError:
        print("Pillow is required: python -m pip install pillow", file=sys.stderr)
        return 2

    if args.mode == "stills":
        files = sorted({f for p in args.patterns for f in glob.glob(p)}, key=_natural_key)
        if not files:
            print("No files matched.", file=sys.stderr)
            return 1
        labels = [os.path.splitext(os.path.basename(f))[0] for f in files]
        out, n = build_sheet(files, labels, args.out, args.per_row, args.thumb_width)
    else:
        ffmpeg = _ffmpeg()
        if ffmpeg is None:
            print("No FFmpeg binary found (PATH or imageio-ffmpeg).", file=sys.stderr)
            return 2
        if not os.path.isfile(args.video):
            print(f"Video not found: {args.video}", file=sys.stderr)
            return 1
        with tempfile.TemporaryDirectory() as tmp:
            scale = f",scale={args.thumb_width}:-2" if args.thumb_width else ""
            cmd = [ffmpeg, "-loglevel", "error", "-i", args.video, "-vf", f"fps={args.fps}{scale}",
                   os.path.join(tmp, "f_%05d.png")]
            subprocess.run(cmd, check=True)
            files = sorted(glob.glob(os.path.join(tmp, "f_*.png")), key=_natural_key)
            # With the fps filter, sample k is centred near t = (k + 0.5) / fps.
            labels = [f"t={(k + 0.5) / args.fps:.1f}s" for k in range(len(files))]
            out, n = build_sheet(files, labels, args.out, args.per_row, 0)
    print(f"{out}: {n} thumbnails")
    return 0


if __name__ == "__main__":
    sys.exit(main())
