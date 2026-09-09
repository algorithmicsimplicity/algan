"""Short Algan explanatory-video template.

Keep the output preset unpinned so `algan render ... -q preview|hd` can select it.
"""

from algan import *


# Opening hierarchy: one title, one primary visual.
title = Text("Core idea", font_size=72).move(UP * 2.5).spawn()
diagram = Circle(color=BLUE).spawn()

Scene.wait(0.35)

# One conceptual change, animated together.
with Sync(runtime=1.4):
    diagram.move(RIGHT * 1.8)
    diagram.color = YELLOW
    title.scale(0.92)

Scene.wait(0.45)

# Reuse the same object for continuity.
diagram = diagram.become(Square(color=YELLOW, add_to_scene=False))

with Sync(runtime=1.2):
    diagram.move_to(ORIGIN)
    diagram.rotate(45, OUT)
    title.color = BLUE

Scene.wait(0.65)

# Leave quality overridable from the CLI.
Scene.save_video("algan_scene")
