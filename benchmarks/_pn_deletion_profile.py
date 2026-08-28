"""Kernel device times for the PN-patch deletion A/B (ss4.3, ss4.4).

``DESIGN_mesh_identity.md`` ss2.1 deleted the unreachable curved PN-patch
renderer: 12 merged keys, two STBVH builds per batch, ~10 parameters off every
traverse/shade signature, and the ``has_pn`` template dimension off four kernels.
Output is byte-identical (ss4.2, confirmed on CUDA: the pre-deletion commit's own
baselines reproduce bit-for-bit with the branch's gates off). What was never
measured is whether the kernels got FASTER, slower, or neither -- removing a
template dimension changes register allocation, and that can cut either way.

Wall clock cannot answer it: thermal throttling swings cross-process throughput
~2x on this machine. So this reports Taichi's per-kernel **device** times, which
are measured on the GPU and are insensitive to host-side drift, plus the
per-batch BVH build time and the offline-cache entry count (ss4.4's compile
surface).

Run it in BOTH trees, each with its own Taichi cache -- the offline cache does
not invalidate on ``@ti.func`` edits and the two trees share kernel *names*, so
one cache would serve the wrong compiled code::

    git worktree add ../algan-pre efb3a95
    cp benchmarks/_pn_deletion_profile.py ../algan-pre/benchmarks/

    ALGAN_CACHE_DIR=<scratch>/pn_pre  ALGAN_MESH_ID=0 ALGAN_POLYHEDRON_WINDING=0 \
        .venv/Scripts/python.exe ../algan-pre/benchmarks/_pn_deletion_profile.py pre
    ALGAN_CACHE_DIR=<scratch>/pn_post ALGAN_MESH_ID=0 ALGAN_POLYHEDRON_WINDING=0 \
        .venv/Scripts/python.exe benchmarks/_pn_deletion_profile.py post

The gates are pinned OFF in both arms so the comparison isolates the deletion
from ss3.5/ss3.7's flips (which do change how much work the resolve does).

The scene is deliberately PN-heavy -- ``Sphere``/``Cylinder``/``Cone``/``Torus``
all reach the renderer as diced logical PN -- with shadows on and everything
moving, because a static fast path is off-limits and because the deleted code
sat in the traverse/shade signatures these exercise. ``ss4.3`` names the kernels
to read: ``wavefront_shade``, ``wavefront_traverse``, both MC megakernels,
``raster_first_shade``, and the per-batch BVH build.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algan import *  # noqa: F403,E402
from algan.utils.profiling_utils import profile_scene  # noqa: E402


def build_scene():
    Scene.set_background(DARKER_GRAY)
    solids = []
    with Off():
        AmbientLight(color=WHITE, intensity=0.25).spawn(animate=False)
        PointLight(location=LEFT * 4 + UP * 4 + OUT * 4).spawn(animate=False)
        DirectionalLight(
            location=RIGHT * 3 + UP * 6 + OUT * 5, target=ORIGIN, color=WHITE
        ).spawn(animate=False)

        ground = Square(color=GRAY).scale(10)
        ground.rotate(90, RIGHT).move(DOWN * 2.2)
        ground.spawn(animate=False)

        # Every curved family, so the diced-logical-PN path is fully exercised.
        solids.append(Sphere(radius=0.9, color=YELLOW).move(LEFT * 2.4))
        solids.append(Cylinder(radius=0.55, height=1.8, color=RED).move(LEFT * 0.8))
        solids.append(Cone(radius=0.6, height=1.6, color=GREEN).move(RIGHT * 0.8))
        solids.append(
            Torus(major_radius=0.7, minor_radius=0.22, color=BLUE).move(RIGHT * 2.4)
        )
        # One flat-triangle solid, so the triangle tree is non-trivial too.
        solids.append(Cube(color=PURPLE).move(UP * 1.8))
        for solid in solids:
            solid.spawn(animate=False)

    with Sync(run_time=2):
        for i, solid in enumerate(solids):
            solid.rotate(90 * (i + 1), UP)
        Scene.get_camera().move(RIGHT * 0.5 + UP * 0.3)


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "arm"
    quality = globals()[sys.argv[2]] if len(sys.argv) > 2 else MD
    SETTINGS.computing.set(available_memory_override=1_400_000_000)
    SETTINGS.raytracing.set(shadows=True)
    print(f"arm={tag}  algan from: {os.path.dirname(os.path.dirname(__file__))}")
    import algan

    print(f"algan package: {algan.__file__}")
    print(f"taichi cache : {SETTINGS.paths.cache_directory}")
    profile_scene(
        build_scene,
        quality,
        tag=f"_pn_deletion_{tag}",
        telemetry=False,
        nvprof=False,
    )


if __name__ == "__main__":
    main()
