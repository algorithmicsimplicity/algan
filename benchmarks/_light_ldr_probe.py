"""Where does a lit surface's linear value actually land?

With tonemapping off (the default since 2026-08-22) anything above linear 1.0
clips, so the invariant we want is: a surface with ``glow == 0`` never exceeds
1.0 however many lights are on it, and only ``glow > 0`` produces HDR.

This measures the peak linear value the post stage receives, for a scene built
one light at a time, so the contribution of each light -- and whether they
accumulate past 1.0 -- is visible rather than inferred. It also checks whether
``SETTINGS.raytracing.light_intensity`` reaches the default deterministic path
at all, which it may not.

    <venv-python> benchmarks/_light_ldr_probe.py
"""

from __future__ import annotations

import torch

from algan import (
    SETTINGS,
    AmbientLight,
    Color,
    Cube,
    DirectionalLight,
    MeshLambertMaterial,
    MeshPhysicalMaterial,
    MeshStandardMaterial,
    PointLight,
    Scene,
    Sphere,
    Square,
)
from algan.constants.spatial import DOWN, LEFT, ORIGIN, OUT, RIGHT, UP
from algan.rendering.post_processing import post_process

WHITE_C = Color((1.0, 1.0, 1.0))


class Peak:
    """Highest linear channel the post stage was handed, and how much is over."""

    def __init__(self):
        self.peak = 0.0
        self.over = 0
        self.total = 0

    def observe(self, frame):
        rgb = frame[..., :3].float()
        self.peak = max(self.peak, float(rgb.max()))
        self.over += int((rgb > 1.0).sum())
        self.total += rgb.numel()

    @property
    def over_pct(self):
        return 100.0 * self.over / self.total if self.total else 0.0


def measure(build, *, glow=0.0, light_intensity=None, ambient=None):
    """Render one still of ``build``'s scene and return its :class:`Peak`."""
    peak = Peak()
    original = post_process._finalize_on_device

    def wrapper(frame, *args, **kwargs):
        if frame.dtype != torch.uint8:
            peak.observe(frame)
        return original(frame, *args, **kwargs)

    post_process._finalize_on_device = wrapper
    snapshot = SETTINGS.snapshot()
    try:
        if light_intensity is not None:
            SETTINGS.raytracing.set(light_intensity=light_intensity)
        if ambient is not None:
            SETTINGS.raytracing.set(ambient_light=ambient)
        scene = Scene()
        build(scene, glow)
        scene.save_frame(
            "algan_outputs/tonemap_check/light_probe.png",
            SETTINGS.video.set(resolution=(160, 120)),
            overwrite=True,
            background_color=Color((0.0, 0.0, 0.0)),
        )
    finally:
        post_process._finalize_on_device = original
        SETTINGS.restore(snapshot)
    return peak


def _lit_cube(scene, glow):
    """A white Lambert cube plus a white flat fill, under the scene's lights."""
    cube = Cube(side_length=1.6)
    cube.set_material(MeshLambertMaterial(color=WHITE_C))
    cube.move(LEFT * 1.6 + DOWN * 0.2)
    cube.spawn()
    sq = Square(side_length=1.2, color=WHITE_C)
    sq.move(RIGHT * 2.4 + UP * 0.2)
    if glow:
        sq.glow = glow
    sq.spawn()


def scene_default(scene, glow):
    """Whatever lights Algan creates on its own -- one white PointLight."""
    _lit_cube(scene, glow)


def _extra_lights(*lights):
    def build(scene, glow):
        _lit_cube(scene, glow)
        for light in lights:
            light().spawn(animate=False)

    return build


SCENARIOS = [
    ("default scene lights only", scene_default),
    (
        "+ AmbientLight(0.45)",
        _extra_lights(lambda: AmbientLight(color=WHITE_C, intensity=0.45)),
    ),
    (
        "+ Ambient + Directional(0.85)",
        _extra_lights(
            lambda: AmbientLight(color=WHITE_C, intensity=0.45),
            lambda: DirectionalLight(
                location=RIGHT * 5 + UP * 5 + OUT * 4,
                target=ORIGIN,
                color=WHITE_C,
                intensity=0.85,
            ),
        ),
    ),
    (
        "+ Ambient + Directional + Point(0.6)   <- tests/fast's rig",
        _extra_lights(
            lambda: AmbientLight(color=WHITE_C, intensity=0.45),
            lambda: DirectionalLight(
                location=RIGHT * 5 + UP * 5 + OUT * 4,
                target=ORIGIN,
                color=WHITE_C,
                intensity=0.85,
            ),
            lambda: PointLight(
                location=LEFT * 3 + UP * 2 + OUT * 3,
                color=WHITE_C,
                intensity=0.6,
            ),
        ),
    ),
]


def _three_lights():
    return (
        lambda: AmbientLight(color=WHITE_C, intensity=0.45),
        lambda: DirectionalLight(
            location=RIGHT * 5 + UP * 5 + OUT * 4,
            target=ORIGIN,
            color=WHITE_C,
            intensity=0.85,
        ),
        lambda: PointLight(
            location=LEFT * 3 + UP * 2 + OUT * 3, color=WHITE_C, intensity=0.6
        ),
    )


def scene_secondary_rays(scene, glow):
    """Mirror-like and transmissive surfaces under the same three-light rig.

    These spawn continuation rays, so a pixel composites several shading
    events. The pipeline tail bounds each event; this checks whether their
    sum stays in range too.
    """
    mirror = Sphere(radius=1.0)
    mirror.set_material(
        MeshStandardMaterial(color=WHITE_C, roughness=0.05, metalness=1.0)
    )
    mirror.move(LEFT * 1.6)
    mirror.spawn()

    glassy = Sphere(radius=1.0)
    glassy.set_material(
        MeshPhysicalMaterial(color=WHITE_C, roughness=0.05, transmission=0.9, ior=1.5)
    )
    glassy.move(RIGHT * 1.6)
    glassy.spawn()

    backdrop = Square(side_length=12, color=WHITE_C)
    backdrop.move(OUT * -4)
    backdrop.spawn()

    for light in _three_lights():
        light().spawn(animate=False)


def main():
    import os

    os.makedirs("algan_outputs/tonemap_check", exist_ok=True)

    print()
    print("Peak linear value reaching the post stage (glow = 0 everywhere).")
    print("Anything above 1.000 clips now that tonemapping defaults off.")
    print(f"{'scenario':>52} {'peak':>8} {'% > 1':>8}")
    print("-" * 72)
    for name, build in SCENARIOS:
        p = measure(build)
        print(f"{name:>52} {p.peak:8.3f} {p.over_pct:7.3f}%")

    print()
    print("Secondary rays: does the per-event bound survive compositing?")
    p = measure(scene_secondary_rays)
    print(
        f"{'metal + transmissive spheres, 3 lights, glow=0':>52} "
        f"{p.peak:8.3f} {p.over_pct:7.3f}%"
    )

    print()
    print("Is there a 'default light intensity' to turn down?")
    try:
        SETTINGS.raytracing.set(light_intensity=0.5)
        print("  light_intensity accepted -- it does reach a live renderer")
    except Exception as exc:  # noqa: BLE001 -- the message is the result
        print(f"  light_intensity REFUSED by the settings layer:\n    {exc}")

    print()
    print("Glow is what SHOULD produce HDR (default lights, one flat fill):")
    for g in (0.0, 0.5, 1.5):
        p = measure(scene_default, glow=g)
        print(f"{'glow=' + str(g):>52} {p.peak:8.3f} {p.over_pct:7.3f}%")


if __name__ == "__main__":
    main()
