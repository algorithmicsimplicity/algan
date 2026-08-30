"""The bounced-ray weight-floor exit: gated, compiled, and reaching the drain.

``wavefront_shade``'s post-loop block retires a ray whose throughput fell
under ``min_weight`` even when its last processed hit took an in-place
reflection branch -- previously every such ray rode to the bounce cap, because
all three reflect branches ``break`` past the in-loop floor test and the
peel-complete tests exclude bounced rays (scratch_perf/r3/ox/
REPORT_immortal_rays.md).

The host-side tests pin the settings plumbing. The render arms exist because
a host-side assertion cannot see whether a Taichi kernel COMPILES or whether
the gate actually reaches the call site: the gate rides ``wavefront_shade``
as a ``ti.template()`` argument read live off ``rt_settings``, so each arm
compiles its own variant through a real render of a reflective scene -- the
same blind spot that once shipped a broken soft-shadow fan
(``test_area_light_soft_shadow.py`` keeps render arms for exactly this
reason). The scene is deliberately reflective (an ior=5 shell, the nn scene's
own material family) so the sheet resolve spawns pooled reflections and the
drain loop -- the new predicate's only home -- actually launches.

The wiring spy patches ``tracer.wavefront_shade`` and inspects the argument
list rather than wrapping ``rt_settings``: the drain loops resolve
``rt_settings`` through ``raytrace_render_wavefront``'s function-local
rebinding (a closure), so a patched module global would silently observe
nothing while everything worked.
"""

from __future__ import annotations

import pytest

from algan import (
    BLACK,
    OUT,
    RIGHT,
    SMOKE_TEST,
    UP,
    WHITE,
    MeshLambertMaterial,
    MeshPhysicalMaterial,
    Off,
    PointLight,
    Prism,
    Scene,
    Sphere,
)
from algan.rendering.raytracing import settings as rt_settings
from algan.rendering.raytracing import tracer
from algan.scene_manager import SceneManager
from algan.settings import SETTINGS

# Position of the ``weight_floor_exit`` template in wavefront_shade's
# parameter list as the tracer passes it (positionally), and the total the
# call sites must pass. Pinned so a future signature reorder breaks HERE,
# loudly, instead of silently un-gating the kernel.
_WEIGHT_FLOOR_EXIT_ARG_INDEX = 46
_EXPECTED_SHADE_ARGS = 68


def test_experimental_setting_surfaces_and_drives_the_legacy_global():
    """``SETTINGS.raytracing.experimental.weight_floor_exit`` is the supported
    way to flip the gate, and it writes the global the tracer call sites read.
    """
    previous = SETTINGS.raytracing.experimental.weight_floor_exit
    try:
        SETTINGS.raytracing.experimental.weight_floor_exit = False
        assert rt_settings.weight_floor_exit is False
        SETTINGS.raytracing.experimental.weight_floor_exit = True
        assert rt_settings.weight_floor_exit is True
    finally:
        SETTINGS.raytracing.experimental.weight_floor_exit = previous


def _render_one_reflective_frame(tmp_path, name):
    """Render one 32x32 frame of a strongly reflective sphere over a ground
    plane. Kept deliberately minimal: the assertions downstream are "the gated
    kernel compiled, the drain loop launched, and the frame rendered".
    """
    output_path = tmp_path / name
    SceneManager.reset()
    try:
        with Scene(video_settings=SMOKE_TEST) as scene:
            scene.set_background(BLACK)
            with Off():
                Scene.clear_lights()
                PointLight(
                    location=UP * 3.0 + RIGHT * 2.0,
                    color=WHITE,
                    intensity=1.0,
                ).spawn(animate=False)
                (
                    Sphere(radius=0.6)
                    .set_material(
                        MeshPhysicalMaterial(color=WHITE, roughness=0.12, ior=5)
                    )
                    .move(OUT * 2.0)
                    .spawn(animate=False)
                )
                (
                    Prism(dimensions=(6.0, 6.0, 0.1))
                    .set_material(MeshLambertMaterial(color=WHITE))
                    .spawn(animate=False)
                )
            result = scene.save_frame(
                output_path,
                video_settings=SMOKE_TEST,
                overwrite=True,
            )
    finally:
        SceneManager.reset()
    return result


@pytest.mark.parametrize("weight_floor", [False, True], ids=("gate_off", "gate_on"))
def test_gated_kernel_compiles_and_gate_reaches_the_drain(
    tmp_path,
    monkeypatch,
    weight_floor,
):
    """Both gate variants of ``wavefront_shade`` COMPILE and every drain launch
    carries the arm's own gate value.

    This is not a redundant pixel test and must stay unmarked (never
    ``fast``): its whole purpose is compile + wiring coverage, which no
    host-side unit test can provide. Taichi rejects an out-of-scope local at
    kernel-compile time and nothing else exercises these variants, so deleting
    this test re-opens that blind spot.

    The spy pins three things at once: the render really did launch the drain
    loop (a non-reflective scene would pass vacuously with zero launches),
    every launch carries the new template argument, and its value is the one
    this arm set.
    """
    previous = SETTINGS.raytracing.experimental.weight_floor_exit
    SETTINGS.raytracing.experimental.weight_floor_exit = weight_floor
    seen = []
    real_shade = tracer.wavefront_shade

    def _spy(*args, **kwargs):
        seen.append(args)
        return real_shade(*args, **kwargs)

    monkeypatch.setattr(tracer, "wavefront_shade", _spy)
    try:
        result = _render_one_reflective_frame(tmp_path, f"weight_floor_{weight_floor}")
    finally:
        SETTINGS.raytracing.experimental.weight_floor_exit = previous

    assert result.output_path.exists()
    assert seen, (
        "the drain loop never launched, so the new predicate never executed; "
        "the scene must stay reflective enough to spawn continuation rays"
    )
    for args in seen:
        assert len(args) == _EXPECTED_SHADE_ARGS
        assert args[_WEIGHT_FLOOR_EXIT_ARG_INDEX] == weight_floor
