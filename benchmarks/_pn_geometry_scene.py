"""A PN-surface-heavy fixture for the geometry-build A/B (records only).

``tests/fast/scene.py`` deliberately contains no ``Surface`` subclass, so it
never reaches the logical PN dice, the control-net builders or the boundary
snap.  This scene is all of them: spheres, a torus and a cylinder, moving and
rotating so the dice cannot collapse the batch's frames onto one another, plus
one light so the shading path is real.  Like the fast scene it only *records*;
``benchmarks/_torch_compile_ab.py`` owns the Scene and the settings.
"""

from __future__ import annotations

from algan import *

Scene.set_background(DARKER_GRAY)

with Off():
    AmbientLight(color=WHITE, intensity=0.45).spawn(animate=False)
    DirectionalLight(
        location=RIGHT * 5 + UP * 5 + OUT * 4,
        target=ORIGIN,
        color=WHITE,
        intensity=0.9,
    ).spawn(animate=False)

    balls = Group(
        Sphere(radius=0.55).set_material(MeshLambertMaterial(color=BLUE)),
        Sphere(radius=0.55).set_material(
            MeshStandardMaterial(color=RED, roughness=0.3, metalness=0.6)
        ),
        Sphere(radius=0.55).set_material(MeshBasicMaterial(color=TEAL)),
    ).arrange_in_line(RIGHT, buffer=0.8)
    balls.move(UP * 1.2 - balls.get_center())

    torus = Torus(ring_radius=0.9, tube_radius=0.28).set_material(
        MeshLambertMaterial(color=ORANGE)
    )
    torus.move(LEFT * 1.6 + DOWN * 1.1)

    tube = Cylinder(radius=0.5, height=1.4).set_material(
        MeshStandardMaterial(color=GREEN_A, roughness=0.5)
    )
    tube.move(RIGHT * 1.6 + DOWN * 1.1)

with Seq():
    with Sync(runtime=0.6):
        balls.spawn()
        torus.spawn()
        tube.spawn()
    # Motion, so the batch's frames carry genuinely different geometry and the
    # dice cannot reuse one frame's answer for the rest.
    with Sync(runtime=1.2):
        torus.rotate(120, RIGHT)
        tube.rotate(90, OUT)
        balls[0].move(UP * 0.6)
        balls[2].move(DOWN * 0.6)
