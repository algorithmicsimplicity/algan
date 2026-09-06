r"""``pt_baseline`` with an area-light scene: the quads arm against the rows arm.

Roadmap section 6a-ter turns every ``RectAreaLight`` into two emissive
triangles for the path tracer instead of ``K = k*k`` packed cell rows. This
adds the scene that measures what that costs and buys end to end -- the
``lit`` solids under four ``samples = 16`` area lights, i.e. 64 packed rows
against 8 triangles -- and hands over to ``pt_baseline`` for everything else
(the stage table, the kernel profiler, the ``RESULTS`` line). Everything
``pt_baseline`` accepts is accepted here; ``--scene area_lights`` is the
addition.

The two arms::

    uv run python benchmarks/performance/pt_area_lights.py \
        --scene area_lights --resolution 1280x720 --tag quads
    ALGAN_PT_AREA_LIGHT_QUADS=0 uv run python benchmarks/performance/pt_area_lights.py \
        --scene area_lights --resolution 1280x720 --tag rows

``pt_area_light_quads`` is read host-side at render time (no compiled
variant), so one process could run both; two processes keep the harness's
cold/warm accounting per arm. Equal-spp variance of the two arms is
``benchmarks/_pt_area_light_quad_variance.py``; this script is their speed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from algan import (  # noqa: E402
    BLACK,
    LEFT,
    ORIGIN,
    OUT,
    RIGHT,
    SETTINGS,
    UP,
    WHITE,
    Off,
    RectAreaLight,
    Scene,
    Sync,
)
from benchmarks.performance import pt_baseline as pb  # noqa: E402


def scene_area_lights():
    """The ``lit`` solids under four 16-sample area lights, shadows on."""
    SETTINGS.raytracing.set(shadows=True)
    Scene.set_background(BLACK)
    Scene.clear_lights()

    with Off():
        for where in (
            UP * 4.0 + LEFT * 2.5 + OUT * 2.0,
            UP * 4.0 + RIGHT * 2.5 + OUT * 2.0,
            UP * 2.5 + OUT * 4.5 + LEFT * 1.0,
            UP * 2.5 + OUT * 4.5 + RIGHT * 1.0,
        ):
            RectAreaLight(
                location=where,
                target=ORIGIN,
                width=1.5,
                height=1.5,
                samples=16,
                color=WHITE,
                # Four lights at the lit scene's total strength.
                intensity=1.4,
            ).spawn(animate=False)
        ball, box, coated = pb._solids()

    with Sync(runtime=pb.DURATION):
        ball.move(UP * 0.9)
        box.rotate(60, UP)
        coated.rotate(45, RIGHT)


pb.SCENES["area_lights"] = scene_area_lights


if __name__ == "__main__":
    raise SystemExit(pb.main())
