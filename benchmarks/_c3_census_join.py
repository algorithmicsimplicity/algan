"""Join the run census against the A/B moved-pixel masks, frame by frame.

For each moved frame of the arm-ON vs arm-OFF diff: how many moved pixels hold
a truncated (>= 17 fragment) triangle run, what the max-run-length distribution
looks like on moved vs unmoved covered pixels, and how many truncated-run
pixels did NOT move. Decides between "the arm's direct population paints the
move" and "something couples pixels the arm never touched".

Pixel mapping note: covered_idx encodes pix = f_rel * ppf + p with p in KERNEL
rows (bottom-up); video rows are top-down, so y_video = height - 1 - py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "tests" / "full_renders" / "algan_outputs" / "_c3_ab"

MOVED_FRAMES = list(range(286, 298))


def load_batches():
    z = np.load(OUT / "run_census.npz")
    n_batches = len([k for k in z.files if k.endswith("_meta")])
    batches = []
    for i in range(n_batches):
        meta = z[f"b{i:03d}_meta"]
        key = f"b{i:03d}_run_pix"
        data = None
        if key in z.files:
            data = {
                "run_pix": z[f"b{i:03d}_run_pix"],
                "run_len": z[f"b{i:03d}_run_len"],
                "run_tri": z[f"b{i:03d}_run_tri"],
                "cov_pix": z[f"b{i:03d}_cov_pix"],
                "cov_nfrag": z[f"b{i:03d}_cov_nfrag"],
            }
        batches.append((meta, data))
    return batches


def moved_mask(idx):
    caps = []
    for name in ("shapes_off.mp4", "shapes_on.mp4"):
        cap = cv2.VideoCapture(str(OUT / name))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        cap.release()
        assert ok
        caps.append(frame.astype(np.int16))
    return np.abs(caps[0] - caps[1]).max(axis=2) > 2


def main():
    batches = load_batches()
    # Segments restart time_start at 0; find where the second segment begins
    # (first batch whose time_start drops back).
    seg_break = None
    last = -1
    for i, (meta, _) in enumerate(batches):
        if meta[0] < last:
            seg_break = i
            break
        last = meta[0]
    seg1 = batches[:seg_break]
    seg2 = batches[seg_break:]
    seg1_frames = max(int(m[1]) for m, _ in seg1)
    print(f"segment 1: {len(seg1)} batches, {seg1_frames} frames; "
          f"segment 2: {len(seg2)} batches")

    for idx in MOVED_FRAMES if len(sys.argv) < 2 else [int(sys.argv[1])]:
        rel = idx - seg1_frames  # frame index inside segment 2
        batch = None
        for meta, data in seg2:
            if meta[0] <= rel < meta[1]:
                batch = (meta, data)
                break
        if batch is None or batch[1] is None:
            print(f"frame {idx}: no census batch")
            continue
        meta, data = batch
        t0, _t1, width, height = (int(v) for v in meta)
        ppf = width * height
        f_rel = rel - t0  # frame inside this batch's window
        moved = moved_mask(idx)
        ys, xs = np.nonzero(moved)
        py = (height - 1) - ys
        moved_p = set((py * width + xs).tolist())

        in_frame = (data["run_pix"] // ppf) == f_rel
        run_p = data["run_pix"][in_frame] % ppf
        run_len = data["run_len"][in_frame]
        run_tri = data["run_tri"][in_frame]

        cov_in = (data["cov_pix"] // ppf) == f_rel
        cov_p = data["cov_pix"][cov_in] % ppf
        cov_n = data["cov_nfrag"][cov_in]

        # Per-pixel max triangle-run length.
        max_len = {}
        for p, ln, tr in zip(run_p.tolist(), run_len.tolist(), run_tri.tolist()):
            if tr and ln > max_len.get(p, 0):
                max_len[p] = ln
        trunc_p = {p for p, ln in max_len.items() if ln >= 17}
        moved_trunc = len(moved_p & trunc_p)
        unmoved_trunc = len(trunc_p - moved_p)

        moved_lens = [max_len.get(p, 0) for p in moved_p]
        moved_cov = [n for p, n in zip(cov_p.tolist(), cov_n.tolist())
                     if p in moved_p]
        hist = np.histogram(moved_lens, bins=[0, 1, 5, 9, 13, 17, 33, 1000])[0]
        print(f"frame {idx} (seg2 rel {rel}, batch t=[{t0},{int(meta[1])}) "
              f"f_rel {f_rel}):")
        print(f"  moved px {len(moved_p)}; with trunc run {moved_trunc}; "
              f"trunc px not moved {unmoved_trunc}; "
              f"covered px this frame {len(cov_p)}")
        print(f"  moved max-tri-run-len hist [0,1-4,5-8,9-12,13-16,17-32,33+]: "
              f"{hist.tolist()}")
        if moved_cov:
            q = np.percentile(moved_cov, [50, 90, 99]).astype(int).tolist()
            print(f"  moved per-px frag count median/p90/p99: {q}, "
                  f"max {max(moved_cov)}")


if __name__ == "__main__":
    main()
