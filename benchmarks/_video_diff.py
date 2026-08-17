"""Frame-wise worst-channel diff between two videos.

Usage:
    .venv/Scripts/python.exe benchmarks/_video_diff.py a.mp4 b.mp4

Prints the maximum absolute channel difference over all frames (0 = byte
identical decode), the frame it occurs on, and a small histogram summary --
the same measure the render suites gate on (<= 2 passes).

Also reports **how many pixels** moved, which is what decides whether a
re-baseline is reviewable: a peak of 53 channel values means one thing over 435
pixels of a silhouette outline and quite another over half the frame. The suites
themselves gate only on the peak, so this is the second number to quote whenever
output moves (DESIGN_mesh_identity.md ss3.5 quotes both).
"""

import sys

import cv2
import numpy as np


def main():
    path_a, path_b = sys.argv[1], sys.argv[2]
    cap_a = cv2.VideoCapture(path_a)
    cap_b = cv2.VideoCapture(path_b)
    worst = 0
    worst_frame = -1
    counts = {}
    # A pixel counts as moved when ANY of its channels moved by more than the
    # suites' tolerance, so these numbers describe what the gate would see.
    tol = 2
    worst_moved = 0
    worst_moved_frame = -1
    total_moved = 0
    frames_over_tol = 0
    pixels_per_frame = 0
    i = 0
    while True:
        ok_a, frame_a = cap_a.read()
        ok_b, frame_b = cap_b.read()
        if not ok_a or not ok_b:
            if ok_a != ok_b:
                print(f"FRAME COUNT MISMATCH at {i}: a={ok_a} b={ok_b}")
            break
        delta = np.abs(frame_a.astype(np.int16) - frame_b.astype(np.int16))
        d = int(delta.max())
        moved = int((delta.max(axis=2) > tol).sum())
        pixels_per_frame = delta.shape[0] * delta.shape[1]
        counts[d] = counts.get(d, 0) + 1
        if d > worst:
            worst = d
            worst_frame = i
        if moved > worst_moved:
            worst_moved = moved
            worst_moved_frame = i
        total_moved += moved
        if moved:
            frames_over_tol += 1
        i += 1
    cap_a.release()
    cap_b.release()
    print(f"frames compared: {i}")
    print(f"worst channel diff: {worst} (frame {worst_frame})")
    if i:
        print(
            f"pixels over tol {tol}: worst frame {worst_moved} of "
            f"{pixels_per_frame} ({worst_moved / max(pixels_per_frame, 1):.3%}, "
            f"frame {worst_moved_frame}); mean {total_moved / i:.1f}/frame; "
            f"{frames_over_tol} of {i} frames affected"
        )
    top = sorted(counts.items(), key=lambda kv: -kv[0])[:8]
    print("per-frame max-diff histogram (worst 8 buckets):")
    for d, n in top:
        print(f"  diff {d:3d}: {n} frames")


if __name__ == "__main__":
    main()
