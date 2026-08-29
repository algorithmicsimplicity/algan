"""Unlit 2-D compositing and the closed-shell ring under the path tracer.

Overlapping translucent circuits at the same depth composite in author
order (the layer tie-break); a translucent backdrop sits behind them (depth
beats author order -- it spawns last); and a rotating translucent
closed-shell prism exercises the camera-segment opacity ring on every frame
of the sweep. All of this is the path tracer's deterministic transport --
the frame should be essentially noise-free at 8 samples.
"""

from algan import *

SETTINGS.raytracing.set(samples_per_pixel=8)

Scene.set_background(BLACK)

with Off():
    front_a = Square(side_length=2.6, color=RED).set_opacity(0.5)
    front_a.move(LEFT * 0.8 + UP * 0.3)
    front_a.spawn(animate=False)
    front_b = Square(side_length=2.6, color=GREEN).set_opacity(0.5)
    front_b.move(RIGHT * 0.4 + DOWN * 0.5)
    front_b.spawn(animate=False)
    # Spawned last but placed behind: author order must lose to depth.
    backdrop = Square(side_length=7.0, color=BLUE).set_opacity(0.6)
    backdrop.move(-OUT * 2.0)
    backdrop.spawn(animate=False)

    # A translucent closed shell: its interior must read the authored
    # opacity once, not once per crossed face (the opacity ring).
    shell = Prism(dimensions=(1.5, 1.5, 1.5))
    shell.set_material(
        MeshLambertMaterial(color=BLACK, emissive=WHITE, emissive_intensity=1.0)
    )
    shell.set_opacity(0.55)
    shell.move(RIGHT * 2.6 + UP * 1.4)
    shell.spawn(animate=False)

shell.rotate(70, UP + RIGHT * 0.4)
