"""Render coarse surfaces close up and look for cracks between diced patches.

Adjacent logical PN patches choose their own subdivision levels, so any mistake
in the shared-boundary handling shows up as background bleeding through the
seams. This renders meshes coarse enough that neighbouring patches genuinely
disagree, on a garish background, and reports every interior pixel that is not
covered by the mob.

    .venv/Scripts/python.exe benchmarks/_logical_pn_crack_check.py
"""

from __future__ import annotations

import numpy as np

from algan import *  # noqa: F403
from algan.mobs.shapes_3d import Sphere, Torus


def report(name, make_mob):
    from algan.scene_manager import SceneManager

    SceneManager.reset()
    make_mob().spawn()
    result = Scene.save_frame(  # noqa: F405
        f"_crack_{name}",
        HD,
        overwrite=True,  # noqa: F405
        background=PURE_RED,  # noqa: F405
    )

    import cv2

    image = cv2.imread(str(result.output_path))
    red = (image[..., 2].astype(int) > 200) & (image[..., 1].astype(int) < 60)
    covered = ~red
    ys, xs = np.nonzero(covered)
    if not len(ys):
        print(f"{name}: mob not visible?!")
        return
    # Background pixels strictly inside the mob's bounding silhouette, scanned
    # row by row between the first and last covered pixel: a crack is a red
    # pixel with mob on both sides of it.
    holes = 0
    for y in np.unique(ys):
        row = covered[y]
        first, last = np.argmax(row), len(row) - 1 - np.argmax(row[::-1])
        holes += int((~row[first : last + 1]).sum())
    print(
        f"{name}: {covered.sum()} covered px, {holes} interior background px "
        f"({result.output_path})"
    )


if __name__ == "__main__":
    coarse = {"geometry_tolerance": 0.2, "max_grid_resolution": 12}
    report("sphere_coarse", lambda: Sphere(**coarse).scale(2.2))
    report("torus_coarse", lambda: Torus(**coarse).scale(2.2).rotate(60, RIGHT))  # noqa: F405
    report(
        "sphere_coarse_tight",
        lambda: Sphere(render_tolerance=0.0002, **coarse).scale(2.2),
    )
