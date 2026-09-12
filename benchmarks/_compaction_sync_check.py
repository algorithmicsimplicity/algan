"""Byte-identity harness for host-side changes to the sheet compaction.

Renders a fixed set of frames of a scene that exercises every compaction
branch the sync consolidation touched -- closed shells at partial opacity
(the shell ceiling's key), flat-shaded polyhedra beside smooth ones (mixed
shading classes, the class-group reuse test), translucent overlap (conflict
ranks and rank pooling), circuits and text (bezier fragments, opaque
truncation) -- and writes each frame as ``.npy`` plus its SHA-256, so two
checkouts can be compared exactly::

    <python> benchmarks/_compaction_sync_check.py out/before      # on the old commit
    <python> benchmarks/_compaction_sync_check.py out/after       # on the new one
    <python> benchmarks/_compaction_sync_check.py --compare out/before out/after

``--compare`` prints, per frame, the number of differing pixels and the
largest channel deviation, and exits 1 if any frame differs. Under
``--quality`` a preset name selects the resolution (default ``MD``);
``--scene explainer`` renders the explainer workload scene instead, whose
stream reaches the class split's grouping sort (see ``workload_scene``).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

os.environ.setdefault("ALGAN_USE_DAEMON", "0")

import numpy as np  # noqa: E402

FRAMES = (0.0, 0.3, 0.6, 1.0)

HERE = os.path.dirname(os.path.abspath(__file__))


def _repo_root():
    """The checkout this runs against: the script's own tree, or -- when a
    copy of the script runs from elsewhere against another checkout, as the
    Kaggle A/B steps do -- the working directory, which the runner sets to
    the repository root.
    """
    own = os.path.dirname(HERE)
    if os.path.isdir(os.path.join(own, "benchmarks", "performance")):
        return own
    return os.getcwd()


#: A flat-shaded glTF mesh: one surface whose triangles carry many distinct
#: face normals -- the kind of geometry that makes a (pixel, surface) group
#: MIX shading classes. (The built-in solids class as smooth.)
MODEL = os.path.join(
    _repo_root(), "tests", "full_renders", "assets", "textured_icosphere.glb"
)


def scene():
    from algan import (
        BLUE,
        DOWN,
        GREEN,
        LEFT,
        OUT,
        RED,
        RIGHT,
        UP,
        Circle,
        Cube,
        Dodecahedron,
        MeshStandardMaterial,
        Model3D,
        Sphere,
        Sync,
        Text,
    )

    sphere = Sphere().scale(1.4).move(LEFT * 3).set_color(GREEN).spawn()
    cube = Cube().scale(1.1).move(RIGHT * 3).set_color(BLUE).spawn()
    glass = Sphere().scale(0.9).move(UP * 1.2).spawn()
    glass.opacity = 0.45
    veil = Cube().scale(0.8).move(UP * 1.2 + RIGHT * 0.6).set_color(RED).spawn()
    veil.opacity = 0.5
    dodeca = Dodecahedron().scale(0.7).move(DOWN * 1.6 + RIGHT * 0.2)
    dodeca.set_material(MeshStandardMaterial(roughness=0.6))
    dodeca.spawn()
    model = Model3D(MODEL, fit_to_size=1.3).move(UP * 1.4 + LEFT * 1.6).spawn()
    circle = Circle().scale(0.8).move(DOWN * 1.8 + LEFT * 1.5).set_color(RED).spawn()
    label = Text("sheets").scale(0.6).move(DOWN * 2.4 + RIGHT * 1.5).spawn()
    with Sync():
        sphere.rotate(70, UP)
        cube.rotate(55, OUT + RIGHT)
        glass.move(RIGHT * 1.3)
        veil.rotate(40, UP + OUT)
        dodeca.rotate(90, UP)
        model.rotate(120, UP + RIGHT)
        circle.rotate(40, OUT)
        label.move(UP * 0.3)


def workload_scene(name):
    """The performance workload scenes, for a parity check on a real stream.

    The explainer scene renders deterministically on the T4 (identical video
    digests run to run) and its glyph fills, axes and heat map send some
    chunks through the class split's grouping sort, which the scene above
    does not reach -- so it is the arm that checks that path end to end on a
    GPU. The graphics scene is NOT byte-stable across runs (split pixels), so
    it is not offered here.
    """
    sys.path.insert(0, os.path.join(_repo_root(), "benchmarks", "performance"))
    module = __import__(f"{name}_scene")
    module.scene(1.0)


def render(out_dir, quality, which):
    import algan
    from algan import Scene

    os.makedirs(out_dir, exist_ok=True)
    if which == "default":
        scene()
    else:
        workload_scene(which)
    preset = getattr(algan, quality)
    results = Scene.save_frame(
        os.path.join(out_dir, "frame.png"), preset, at=list(FRAMES), overwrite=True
    )
    from PIL import Image

    for i, result in enumerate(results):
        arr = np.asarray(Image.open(result.output_path).convert("RGB"))
        np.save(os.path.join(out_dir, f"frame_{i}.npy"), arr)
        digest = hashlib.sha256(arr.tobytes()).hexdigest()
        print(f"frame {i} t={FRAMES[i]:.2f}  sha256 {digest[:16]}  {arr.shape}")


def compare(a, b):
    bad = 0
    for i in range(len(FRAMES)):
        fa = np.load(os.path.join(a, f"frame_{i}.npy")).astype(np.int32)
        fb = np.load(os.path.join(b, f"frame_{i}.npy")).astype(np.int32)
        if fa.shape != fb.shape:
            print(f"frame {i}: shape {fa.shape} vs {fb.shape}")
            bad += 1
            continue
        diff = np.abs(fa - fb)
        pixels = int((diff.max(axis=-1) > 0).sum())
        print(
            f"frame {i}: differing pixels {pixels:8d} / {fa.shape[0] * fa.shape[1]}"
            f"   max channel deviation {int(diff.max())}"
        )
        bad += pixels > 0
    print("IDENTICAL" if bad == 0 else f"DIFFERENT in {bad} frame(s)")
    return 1 if bad else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", nargs="?")
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    parser.add_argument("--quality", default="MD")
    parser.add_argument(
        "--scene",
        default="default",
        choices=("default", "explainer"),
        help="the scene above (default) or the explainer workload scene",
    )
    args = parser.parse_args(argv)
    if args.compare:
        return compare(*args.compare)
    if not args.out_dir:
        parser.error("an output directory or --compare is required")
    render(args.out_dir, args.quality, args.scene)
    return 0


if __name__ == "__main__":
    sys.exit(main())
