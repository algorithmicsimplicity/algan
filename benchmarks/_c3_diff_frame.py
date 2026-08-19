"""Dump one frame of the C.3 A/B pair side by side, plus a moved-pixel map."""

import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "tests" / "full_renders" / "algan_outputs" / "_c3_ab"


def frame_at(path, idx):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    assert ok, f"no frame {idx} in {path}"
    return frame


def main():
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 290
    fa = frame_at(OUT / "shapes_off.mp4", idx)
    fb = frame_at(OUT / "shapes_on.mp4", idx)
    delta = np.abs(fa.astype(np.int16) - fb.astype(np.int16)).max(axis=2)
    moved = delta > 2
    print(f"frame {idx}: moved px {int(moved.sum())}, max |d| {int(delta.max())}")
    ys, xs = np.nonzero(moved)
    print(f"  moved bbox x [{xs.min()},{xs.max()}] y [{ys.min()},{ys.max()}] "
          f"of {fa.shape[1]}x{fa.shape[0]}")
    # Distribution of |d| over moved pixels.
    vals, counts = np.unique(delta[moved], return_counts=True)
    print("  |d| histogram:", dict(zip(vals.tolist(), counts.tolist())))
    # Worst pixels, for an ALGAN_AA_DUMP probe.
    flat = np.argsort(delta, axis=None)[::-1][:10]
    for k in flat:
        y, x = divmod(int(k), delta.shape[1])
        print(f"  worst: (x={x}, y={y}) |d|={int(delta[y, x])} "
              f"off={fa[y, x].tolist()} on={fb[y, x].tolist()}")
    amp = np.zeros_like(fa)
    amp[..., 2] = np.clip(delta * 8, 0, 255)  # red = moved, amplified
    side = np.concatenate([fa, fb, amp], axis=1)
    dest = OUT / f"diff_{idx}.png"
    cv2.imwrite(str(dest), side)
    cv2.imwrite(str(OUT / f"off_{idx}.png"), fa)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
