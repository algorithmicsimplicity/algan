"""Extract the worst-differing frame of two videos as a side-by-side PNG.

``_video_diff.py`` says how far output moved; this says *what* moved, which is the
question a re-baseline actually turns on ("look at the result before committing"
-- CLAUDE.md, and every re-baseline note in DESIGN_mesh_identity.md).

Writes one PNG: baseline | actual | amplified difference, stacked left to right
with the difference scaled so a 5-value change is visible rather than black. The
frame chosen is the one with the most pixels over tolerance, not the one with the
largest single channel value -- a one-pixel spike is not what you need to look at.

Usage:
    .venv/Scripts/python.exe benchmarks/_diff_frame.py baseline.mp4 actual.mp4 out.png
    .venv/Scripts/python.exe benchmarks/_diff_frame.py a.mp4 b.mp4 out.png --frame 36
    .venv/Scripts/python.exe benchmarks/_diff_frame.py a.mp4 b.mp4 out.png --crop
"""

import argparse

import cv2
import numpy as np

AMPLIFY = 12
TOL = 2


def read_all(path):
    frames = []
    cap = cv2.VideoCapture(path)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame.copy())
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return np.stack(frames)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("baseline")
    ap.add_argument("actual")
    ap.add_argument("out")
    ap.add_argument("--frame", type=int, default=None)
    ap.add_argument(
        "--crop",
        action="store_true",
        help="crop to the bounding box of the difference, so a thin silhouette "
        "change is not two pixels of a 1280-wide frame",
    )
    args = ap.parse_args()

    a, b = read_all(args.baseline), read_all(args.actual)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    delta = np.abs(a.astype(np.int16) - b.astype(np.int16))
    over = (delta.max(axis=-1) > TOL).reshape(n, -1).sum(axis=1)
    idx = args.frame if args.frame is not None else int(over.argmax())

    fa, fb = a[idx], b[idx]
    d = delta[idx]
    print(
        f"frame {idx} of {n}: {int(over[idx])} px over tol {TOL}, "
        f"max channel {int(d.max())}"
    )

    if args.crop and over[idx]:
        ys, xs = np.where(d.max(axis=-1) > TOL)
        pad = 24
        y0, y1 = max(0, ys.min() - pad), min(d.shape[0], ys.max() + pad + 1)
        x0, x1 = max(0, xs.min() - pad), min(d.shape[1], xs.max() + pad + 1)
        fa, fb, d = fa[y0:y1, x0:x1], fb[y0:y1, x0:x1], d[y0:y1, x0:x1]
        print(f"cropped to y[{y0}:{y1}] x[{x0}:{x1}]")

    amp = np.clip(d.astype(np.int32) * AMPLIFY, 0, 255).astype(np.uint8)
    gap = np.full((fa.shape[0], 8, 3), 255, dtype=np.uint8)
    panel = np.concatenate([fa, gap, fb, gap, amp], axis=1)
    cv2.imwrite(args.out, panel)
    print(f"wrote {args.out}  (baseline | actual | difference x{AMPLIFY})")


if __name__ == "__main__":
    main()
