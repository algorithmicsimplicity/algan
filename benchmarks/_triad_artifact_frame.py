"""Render ONE frame of the real ``solids_and_camera`` scene.

The axis-triad artifacts (white specks where the red arrow's shaft meets its
head and where it meets the Dot3D, red/green pixels inside the sphere) live at
the end of Act 3, frame 168 of the committed baselines -- t = 16.8 s at
``PREVIEW``'s 10 fps.  Rendering the scene's own source and asking for that one
time reproduces the baseline pixel-exactly, which a hand-cut repro does not:
the triad's pose depends on ``Group.rotate``'s default pivot and on the camera
having orbited out and back.

Usage::

    <venv-python> benchmarks/_triad_artifact_frame.py [--at 16.8] [--name NAME]
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser()
parser.add_argument("--at", type=float, nargs="+", default=[16.8])
parser.add_argument("--name", default="frame168")
parser.add_argument("--scale", type=int, default=1)
parser.add_argument("--aa", type=int, default=None)
parser.add_argument(
    "--video",
    action="store_true",
    help="render the whole scene losslessly instead of single frames",
)
parser.add_argument(
    "--tol",
    type=float,
    default=None,
    help=(
        "force every Surface's geometry/render tolerances to this value "
        "(world units and pixels alike). Tighter dicing; use it to tell a "
        "tessellation artifact from a compositing one."
    ),
)
parser.add_argument(
    "--recolor",
    action="store_true",
    help=(
        "tint the triad's Line3D magenta and its Dot3D yellow. Geometry, and "
        "so the pose, is untouched -- only which object a leaking pixel came "
        "from changes, which is what identifies it."
    ),
)
args = parser.parse_args()

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCENE = os.path.join(_ROOT, "tests", "full_renders", "scenes", "solids_and_camera.py")

# The scene pins a vendored font; tests/conftest.py is what registers it.
sys.path.insert(0, os.path.join(_ROOT, "tests"))
from conftest import _register_test_fonts  # noqa: E402

_register_test_fonts()

from algan import PREVIEW, Scene  # noqa: E402

if args.tol is not None:
    from algan.mobs.surfaces.surface import Surface

    _surface_init = Surface.__init__

    def _tight_init(self, *a, **kw):
        # geometry_tolerance is deliberately left alone: it sets the
        # construction grid, which moves a mob's centre and so breaks Act 2's
        # move_to_point_along_arc. Only the per-frame dice is tightened.
        kw["render_tolerance"] = args.tol
        kw["render_tolerance_pixels"] = args.tol
        return _surface_init(self, *a, **kw)

    Surface.__init__ = _tight_init

_scene_path = _SCENE
if args.recolor:
    import tempfile

    with open(_SCENE) as fh:
        src = fh.read()
    for old, new in (
        # A bare ``color=`` on either would be overwritten: the enclosing
        # Group propagates its own default WHITE to its children, which is why
        # both render white today whatever they were built with. A material
        # colour is not propagated, so this is the tint that survives.
        (
            "thickness=0.03, color=GRAY_A)",
            "thickness=0.03, color=GRAY_A)"
            ".set_material(MeshBasicMaterial(color=MAGENTA))",
        ),
        (
            "radius=0.14, color=WHITE)",
            "radius=0.14, color=WHITE).set_material(MeshBasicMaterial(color=YELLOW))",
        ),
    ):
        assert old in src, old
        src = src.replace(old, new)
    fd, _scene_path = tempfile.mkstemp(suffix=".py", text=True)
    with os.fdopen(fd, "w") as fh:
        fh.write(src)
    print("RECOLORED SCENE ->", _scene_path)

runpy.run_path(_scene_path, run_name="__algan_scene__")

kw = {"resolution": (704 * args.scale, 396 * args.scale)}
if args.aa is not None:
    kw["super_sampling_anti_aliasing"] = args.aa
if args.video:
    r = Scene.save_video(
        args.name,
        PREVIEW.set(**kw),
        codec="libx264rgb",
        ffmpeg_params=["-crf", "0"],
    )
    print("wrote", r.output_path)
    print("truncations", r.render_plan.truncations.as_dict())
    raise SystemExit(0)

times = args.at if len(args.at) > 1 else args.at[0]
results = Scene.save_frame(args.name, PREVIEW.set(**kw), at=times)
for r in results if isinstance(results, list) else [results]:
    print("wrote", r.output_path)
    print("truncations", r.render_plan.truncations.as_dict())
