"""Compare renderer-audit images side by side and report where they differ.

Two images of the same scene from two renderers will never be byte-equal, and a
raw pixel diff is dominated by whichever renderer is globally brighter -- which
is a units question, not a transport question. So this reports both:

* the **raw** difference, in 8-bit channel values and after decoding to linear;
* an **exposure-matched** difference, where the second image is scaled by the
  single linear factor that best matches the first over the pixels both call
  non-background. What survives that scaling is structural: a reflection that
  is in the wrong place, a refraction that does not invert, a shadow that is
  missing, a falloff with the wrong shape.

It also writes a contact sheet (``<scene>.compare.png``): the two inputs, their
signed difference and the exposure-matched difference, so the eye can do the
part it is better at than any metric.

    <venv-python> benchmarks/renderer_audit/compare.py out/showcase.algan.png \\
        out/showcase.three_pathtrace.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def srgb_to_linear(u8):
    c = np.asarray(u8, dtype=np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(lin):
    lin = np.clip(np.asarray(lin, dtype=np.float64), 0.0, None)
    return np.where(lin <= 0.0031308, lin * 12.92,
                    1.055 * lin ** (1 / 2.4) - 0.055)


def _load(path):
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise SystemExit(f"cannot read {path}")
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    return im[:, :, :3][:, :, ::-1].astype(np.uint8)  # -> RGB


def compare(path_a, path_b, out_dir=None, label_a="A", label_b="B"):
    a = _load(path_a)
    b = _load(path_b)
    if a.shape != b.shape:
        raise SystemExit(f"shape mismatch: {a.shape} vs {b.shape}")

    la, lb = srgb_to_linear(a), srgb_to_linear(b)

    # "Content" = anything either image lit above a floor. Comparing over the
    # background would let a large empty frame dominate every statistic.
    content = (la.max(axis=2) > 0.004) | (lb.max(axis=2) > 0.004)
    n = int(content.sum())

    # One global linear exposure factor: the least-squares scale taking B to A
    # over content pixels. Whatever is left after this is not a brightness
    # difference.
    fa, fb = la[content], lb[content]
    k = float((fa * fb).sum() / max((fb * fb).sum(), 1e-12)) if n else 1.0
    lb_matched = lb * k

    raw8 = np.abs(a.astype(np.int32) - b.astype(np.int32))
    matched8 = np.abs(a.astype(np.int32)
                      - np.round(linear_to_srgb(lb_matched) * 255.0).astype(np.int32))

    stats = {
        "a": str(path_a), "b": str(path_b),
        "resolution": [int(a.shape[1]), int(a.shape[0])],
        "content_pixels": n,
        "exposure_factor_b_to_a": round(k, 4),
        "raw": {
            "mean_abs_8bit": round(float(raw8[content].mean()) if n else 0.0, 2),
            "p95_abs_8bit": round(float(np.percentile(raw8[content], 95)) if n else 0.0, 1),
            "max_abs_8bit": int(raw8.max()),
            "mean_linear_ratio_a_over_b": round(
                float(fa.mean() / max(fb.mean(), 1e-12)) if n else 0.0, 3),
        },
        "exposure_matched": {
            "mean_abs_8bit": round(float(matched8[content].mean()) if n else 0.0, 2),
            "p95_abs_8bit": round(float(np.percentile(matched8[content], 95)) if n else 0.0, 1),
            "frac_pixels_over_16": round(
                float((matched8[content].max(axis=-1) > 16).mean()) if n else 0.0, 4),
        },
    }

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        name = Path(path_a).name.split(".")[0]
        bm = np.round(np.clip(linear_to_srgb(lb_matched) * 255.0, 0, 255)).astype(np.uint8)
        # Difference panels are amplified 3x: at 1x an interesting structural
        # difference is often invisible next to the images it came from.
        d_raw = np.clip(raw8 * 3, 0, 255).astype(np.uint8)
        d_mat = np.clip(matched8 * 3, 0, 255).astype(np.uint8)
        top = np.concatenate([a, b], axis=1)
        bottom = np.concatenate([d_mat, bm], axis=1)
        sheet = np.concatenate([top, bottom], axis=0)
        sheet = _annotate(sheet, a.shape[1], a.shape[0],
                          [label_a, label_b,
                           f"|{label_a} - {label_b}x{k:.2f}| x3",
                           f"{label_b} x{k:.2f}"])
        # JPEG for the contact sheet: it is a figure to look at, not a baseline
        # to compare against, and path-traced noise makes it a 2 MB PNG.
        cv2.imwrite(str(out_dir / f"{name}.compare.jpg"), sheet[:, :, ::-1],
                    [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        cv2.imwrite(str(out_dir / f"{name}.diff_raw.png"), d_raw[:, :, ::-1])
        stats["contact_sheet"] = str(out_dir / f"{name}.compare.jpg")

    return stats


def _annotate(sheet, w, h, labels):
    sheet = sheet.copy()
    for i, text in enumerate(labels):
        x = (i % 2) * w + 6
        y = (i // 2) * h + 18
        cv2.putText(sheet, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(sheet, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return sheet


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--label-a", default="algan")
    ap.add_argument("--label-b", default="three")
    args = ap.parse_args(argv)
    print(json.dumps(compare(args.a, args.b, args.out, args.label_a, args.label_b),
                     indent=2))


if __name__ == "__main__":
    main()
