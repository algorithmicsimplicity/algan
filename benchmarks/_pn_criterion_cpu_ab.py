"""A/B: the fused subdivision-level criterion kernels on a CPU arch.

The level searches that decide how finely each logical PN patch and each bezier
segment is diced are reductions written as ~30 elementwise torch passes over
large scratch. ``algan/rendering/raytracing/logical_pn_taichi.py`` holds fused
Taichi kernels for all three, and
``rendering/raytracing/settings.py::PN_CRITERION_KERNEL`` documents them at
**67.9 s (8.5%)** and **18.4 s (2.3%)** of a reference ``save_video``.

Until 2026-08-26 they ran only on a CUDA render device. The stated reason was
staging: "elsewhere the criterion's tensors live on the CPU, where launching
Taichi against them stages every argument through VRAM". That reason cannot
apply when the render device *is* the CPU -- the arch is then x64, the tensors
are already host tensors, and nothing stages. This measures what that was
costing.

    ALGAN_USE_DAEMON=0 uv run python benchmarks/_pn_criterion_cpu_ab.py

One process per arm, because the arm is selected by an env var read at import
and because Taichi's compiled state is process-global. The kernel arm's first
run pays a cold kernel compile; the harness discards a warm-up render per arm
so that lands outside the timed region, and says so in its output.

Reports, per arm: whole-``save_video`` wall time, time inside the three
criterion entry points, and -- because these kernels are **not** bit-identical
to torch (Taichi runs under ``fast_math``, so borderline patches round to a
neighbouring subdivision level) -- the per-channel deviation between the two
arms' frames. A level that moved is the intended behaviour, not a defect; the
number is here so its size is on the record rather than assumed.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent

#: Frames per arm. Enough that the level searches run on several batches, few
#: enough that three rounds of two arms finish on a 4-vCPU container.
FRAMES = 24
ROUNDS = int(os.environ.get("AB_ROUNDS", "3"))


SCENE = '''
import json, time, sys, os
import torch
from algan import *

_TIMES = {"edge": 0.0, "flatness": 0.0, "bezier": 0.0, "calls": 0}


def _instrument():
    """Time the three criterion entry points, whichever arm they take.

    Wrapping the entry points rather than the kernels means the torch arm is
    measured at exactly the same boundary as the kernel arm -- otherwise the
    comparison would be a kernel against a whole function.
    """
    from algan.rendering.raytracing import primitives as P

    def wrap(cls, name, key):
        original = getattr(cls, name)

        def timed(*args, **kwargs):
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                _TIMES[key] += time.perf_counter() - started
                _TIMES["calls"] += 1

        setattr(cls, name, timed)

    wrap(P.LogicalPNTrianglePrimitive, "_edge_chord_error", "edge")
    wrap(P.LogicalPNTrianglePrimitive, "_patch_flatness_error", "flatness")


def build():
    """A PN-heavy scene: every mob here dices through the level searches."""
    sphere = Sphere(radius=1).move(LEFT * 2).spawn()
    Torus().spawn()
    Cylinder().move(RIGHT * 2).spawn()
    return sphere


def main():
    out = sys.argv[1]
    frames = int(sys.argv[2])
    warmup = sys.argv[3] == "warmup"

    _instrument()
    settings = PREVIEW

    sphere = build()
    sphere.rotate(90, UP)

    if warmup:
        # Pays the cold kernel compile (and the Tex/geometry caches) outside
        # the timed region.
        Scene.save_video(out + "_warm", settings)
        for key in _TIMES:
            _TIMES[key] = 0

    started = time.perf_counter()
    Scene.save_video(out, settings)
    wall = time.perf_counter() - started

    print("RESULT " + json.dumps({
        "wall": wall,
        "edge": _TIMES["edge"],
        "flatness": _TIMES["flatness"],
        "calls": _TIMES["calls"],
    }))


main()
'''


def run_arm(kernel_on: bool, out_stem: str, scene_path: Path) -> dict:
    """Render one arm in its own process and return its timing record."""
    environment = dict(os.environ)
    environment["ALGAN_RENDER_DEVICE"] = "cpu"
    environment["ALGAN_USE_DAEMON"] = "0"
    environment["ALGAN_PN_CRITERION_KERNEL"] = "1" if kernel_on else "0"
    completed = subprocess.run(
        [sys.executable, str(scene_path), out_stem, str(FRAMES), "warmup"],
        cwd=str(_REPO),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        print(completed.stdout[-4000:])
        print(completed.stderr[-4000:], file=sys.stderr)
        raise SystemExit(f"arm kernel_on={kernel_on} failed")
    for line in completed.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT ") :])
    print(completed.stdout[-4000:])
    raise SystemExit("arm produced no RESULT line")


def frame_deviation(a: Path, b: Path):
    """Max and mean per-channel deviation between two rendered videos."""
    import cv2
    import numpy as np

    ca, cb = cv2.VideoCapture(str(a)), cv2.VideoCapture(str(b))
    worst, total, count, moved = 0, 0.0, 0, 0
    try:
        while True:
            oka, fa = ca.read()
            okb, fb = cb.read()
            if not (oka and okb):
                break
            diff = np.abs(fa.astype(np.int16) - fb.astype(np.int16))
            worst = max(worst, int(diff.max()))
            total += float(diff.mean())
            moved += int((diff > 2).sum())
            count += 1
    finally:
        ca.release()
        cb.release()
    return worst, (total / count if count else 0.0), count, moved


def main() -> int:
    scene_path = _HERE / "_pn_criterion_cpu_ab_scene.py"
    scene_path.write_text(SCENE)
    # save_video resolves its output root relative to the *scene* script, which
    # this harness writes into benchmarks/ -- so the videos land there, not in
    # the repo-root algan_outputs. Look in both rather than assuming.
    outputs = next(
        (d for d in (_HERE / "algan_outputs", _REPO / "algan_outputs") if d.is_dir()),
        _REPO / "algan_outputs",
    )
    try:
        print("=" * 76)
        print("PN/bezier level-search criterion: torch vs fused Taichi, CPU arch")
        print("=" * 76)
        print(f"{FRAMES} frames, {ROUNDS} rounds per arm, one process per arm.")
        print("Each arm renders once to warm caches and the kernel compile, then")
        print("the timed render follows in the same process.\n")

        arms = {False: [], True: []}
        for round_index in range(ROUNDS):
            for kernel_on in (False, True):
                stem = f"_pn_ab_{'kernel' if kernel_on else 'torch'}_{round_index}"
                record = run_arm(kernel_on, stem, scene_path)
                arms[kernel_on].append(record)
                label = "kernel" if kernel_on else "torch "
                print(
                    f"  round {round_index}  {label}  wall {record['wall']:6.2f}s  "
                    f"criterion {record['edge'] + record['flatness']:6.2f}s  "
                    f"({record['calls']} calls)"
                )

        def median(records, key):
            return statistics.median(r[key] for r in records)

        print()
        print(f"{'':10s} {'wall':>10s} {'criterion':>12s} {'share':>8s}")
        summary = {}
        for kernel_on, label in ((False, "torch"), (True, "kernel")):
            records = arms[kernel_on]
            wall = median(records, "wall")
            criterion = median(records, "edge") + median(records, "flatness")
            summary[label] = (wall, criterion)
            print(
                f"{label:10s} {wall:9.2f}s {criterion:11.2f}s {criterion / wall:7.1%}"
            )

        torch_wall, torch_criterion = summary["torch"]
        kernel_wall, kernel_criterion = summary["kernel"]
        print()
        print(
            f"criterion speedup : {torch_criterion / max(kernel_criterion, 1e-9):.2f}x"
        )
        print(f"whole-render      : {torch_wall / max(kernel_wall, 1e-9):.2f}x")
        saved = torch_wall - kernel_wall
        print(
            f"render time saved : {saved:.2f}s of {torch_wall:.2f}s "
            f"({saved / torch_wall:+.1%})"
        )

        # Output moves by design (fast_math level rounding); record how much.
        torch_video = outputs / f"_pn_ab_torch_{ROUNDS - 1}.mp4"
        kernel_video = outputs / f"_pn_ab_kernel_{ROUNDS - 1}.mp4"
        if torch_video.exists() and kernel_video.exists():
            worst, mean, frames, moved = frame_deviation(torch_video, kernel_video)
            print()
            print(f"frames compared   : {frames}")
            print(f"max deviation     : {worst} channel values")
            print(f"mean deviation    : {mean:.3f}")
            print(f"channels beyond 2 : {moved}")
            print(
                "\nA nonzero deviation is expected: Taichi runs under fast_math, so a\n"
                "borderline patch rounds to a neighbouring subdivision level. The\n"
                "number is here so its size is recorded rather than assumed."
            )
        else:
            print("\n(no videos to compare -- check the output directory)")
        return 0
    finally:
        scene_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
