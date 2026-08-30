"""The item-1 population of DESIGN_renderer_structural_candidates.md: static
textured mobs beside parked PN geometry, one small mover, camera parked.

Four ImageMobs share ONE image file (texture content dedup), nothing about
them animates (per-map time collapse + the static window collapse), most of
the geometry is parked (merge-time geometry collapse + the BVH builders'
static branches) while one cube keeps moving so this stays a *general* moving
scene rather than a static-only fast path. Shadows are on so the shadow path
and its per-batch scene diagonal run.

Structured like ``nn_scene_PREVIEW.py``: two profiled runs, read RUN 2.
"""

import os

os.environ["ALGAN_USE_DAEMON"] = "0"

from algan import *
from algan.utils.profiling_utils import profile_scene


def scene():
    duration = 10
    SETTINGS.raytracing.set(shadows=True)

    with Off():
        for off in (LEFT * 4.2, LEFT * 1.4, RIGHT * 1.4, RIGHT * 4.2):
            ImageMob("world_map.png").scale(0.55).move(off + UP * 1.4).spawn()
        Sphere().scale(0.8).move(DOWN * 1.2 + LEFT * 2.5).spawn()
        Cylinder().scale(0.6).move(DOWN * 1.2 + RIGHT * 2.5).spawn()
        mover = Cube().scale(0.5).move(DOWN * 1.2).spawn()
        Text("static gallery").move(DOWN * 2.6).spawn()

    with Sync(duration=duration):
        mover.rotate(360, UP)


profile_scene(
    scene,
    PREVIEW,
    "static_gallery_PREVIEW",
    runs=2,
    kernel_profiler=False,
    save_video_kwargs={"ffmpeg_params": ["-crf", "17", "-preset", "ultrafast"]},
)
