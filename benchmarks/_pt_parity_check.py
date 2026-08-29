"""Path-traced vs deterministic 2-D compositing parity, measured.

The path tracer's contract on unlit 2-D content is *equality*, not
resemblance: transparency composites deterministically (throughput-weighted,
never stochastic alpha), the layer tie-break is shared with the deterministic
route, and the only legitimate difference is at geometry edges, where
jittered-sample anti-aliasing replaces analytic coverage. So this check
renders one dense all-unlit 2-D scene both ways -- deterministic at
``samples_per_pixel = 1``, path-traced at 256 (the plan of record's figure;
override with argv[1]) -- and requires flat interiors to agree to at most one
channel count, while reporting how wide the edge band is and how far it
moves.

The scene stacks the author-order edge cases: translucent circuits
overlapping at the SAME depth (spawn order decides), a shape spawned last
but placed behind (depth decides), borders, and an opaque occluder.

Run: .venv/Scripts/python.exe benchmarks/_pt_parity_check.py [spp]
Exit status 0 iff the flat-interior parity holds.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("ALGAN_USE_DAEMON", "0")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from algan import (  # noqa: E402
    BLACK,
    BLUE,
    DOWN,
    GREEN,
    LEFT,
    ORANGE,
    OUT,
    PREVIEW,
    RED,
    RIGHT,
    SETTINGS,
    UP,
    WHITE,
    YELLOW,
    Circle,
    Off,
    Scene,
    SceneManager,
    Square,
    Triangle,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "_pt_parity_out")
os.makedirs(OUT_DIR, exist_ok=True)

VS = PREVIEW.set(resolution=(128, 128), frames_per_second=5)


def _build():
    Scene.set_background(BLACK)
    with Off():
        # Same depth, overlapping: author order is the only separator.
        a = Square(side_length=3.2, color=RED).set_opacity(0.5)
        a.move(LEFT * 0.9 + UP * 0.8)
        a.spawn(animate=False)
        b = Circle(radius=1.6, color=GREEN).set_opacity(0.5)
        b.move(RIGHT * 0.4 + UP * 0.2)
        b.spawn(animate=False)
        c = Triangle(color=YELLOW).set_opacity(0.4)
        c.scale(2.2)
        c.move(DOWN * 1.2)
        c.spawn(animate=False)
        # Spawned after the stack but placed behind it: depth beats author
        # order.
        backdrop = Square(side_length=6.5, color=BLUE).set_opacity(0.6)
        backdrop.move(-OUT * 2.0)
        backdrop.spawn(animate=False)
        # An opaque occluder on top of everything.
        cap = Circle(radius=0.7, color=WHITE)
        cap.move(RIGHT * 1.8 + DOWN * 1.6)
        cap.spawn(animate=False)
        # A bordered translucent shape (border compositing rides the same
        # peel).
        ring = Circle(radius=1.0, color=ORANGE, border_color=WHITE)
        ring.set_opacity(0.5)
        ring.move(LEFT * 2.2 + DOWN * 0.6)
        ring.spawn(animate=False)


def _render(name, spp):
    snapshot = SETTINGS.snapshot()
    SceneManager.reset()
    try:
        SETTINGS.raytracing.set(samples_per_pixel=spp)
        with Scene(video_settings=VS) as scene:
            _build()
            result = scene.save_frame(
                os.path.join(OUT_DIR, name), video_settings=VS, overwrite=True
            )
        if spp > 1:
            assert result.render_plan.backend == "path_tracer", (
                result.render_plan.backend
            )
        frame = cv2.imread(str(result.output_path), cv2.IMREAD_UNCHANGED)
        return torch.from_numpy(frame[..., :3].astype(np.int32))
    finally:
        SceneManager.reset()
        SETTINGS.restore(snapshot)


def main():
    spp = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    det = _render("parity_det.png", 1)
    pt = _render("parity_pt.png", spp)

    det_f = det.float().permute(2, 0, 1).unsqueeze(0)
    pooled_max = torch.nn.functional.max_pool2d(det_f, 3, stride=1, padding=1)
    pooled_min = -torch.nn.functional.max_pool2d(-det_f, 3, stride=1, padding=1)
    flat = (pooled_max - pooled_min).squeeze(0).amax(0) < 2

    err = (det - pt).abs().amax(-1)
    flat_err = err[flat]
    edge_err = err[~flat]
    n_flat = int(flat.sum())
    n_all = err.numel()

    diff = (det - pt).abs().amax(-1).clamp(0, 255).to(torch.uint8).numpy()
    cv2.imwrite(os.path.join(OUT_DIR, "parity_diff.png"), diff)

    print(f"spp                 : {spp}")
    print(f"flat interior pixels: {n_flat}/{n_all} ({100.0 * n_flat / n_all:.1f}%)")
    print(f"flat max error      : {int(flat_err.max())}")
    print(f"flat mean error     : {float(flat_err.float().mean()):.4f}")
    print(f"flat pixels > 1     : {int((flat_err > 1).sum())}")
    print(f"edge max error      : {int(edge_err.max()) if edge_err.numel() else 0}")
    print(f"edge mean error     : {float(edge_err.float().mean()):.2f}")
    print(f"frames + diff in    : {OUT_DIR}")

    ok = n_flat > n_all // 4 and int(flat_err.max()) <= 1
    print("PARITY: " + ("OK" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
