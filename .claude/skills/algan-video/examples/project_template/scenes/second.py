"""Scene 2: mechanics only (placeholder shape, placeholder narration)."""

from algan import *

from common import say


def second():
    with Off():
        shape = Triangle().scale(0.7).spawn()

    with say("Placeholder narration for the second scene.", hold=0.5):
        shape.rotate(120, OUT)
    Scene.save_frame('second_rotated', at=[-0.2])
