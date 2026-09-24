"""Scene 1: mechanics only (placeholder shapes, placeholder narration).

Shows: instant setup in Off(), a camera move that spans a whole Speech block while
other actions run in sequence, and named checkpoints for review stills.
"""

from algan import *

from common import say


def first():
    camera = Scene.get_camera()
    with Off():
        camera.fly_to(OUT * 12 + UP * 2, look_at=ORIGIN)
        a = Square().scale(0.6).move_to(LEFT * 1.5)
        b = Circle().scale(0.6).move_to(RIGHT * 1.5)

    Scene.save_frame('first_open', at=[0.1])

    # Speech rescales the whole block to the clip's length. equalize_runtimes
    # stretches every direct child of the Sync to the longest one, so the camera
    # move (no explicit runtime) spans the sequence beside it. Keep everything
    # else inside that one sequence.
    with say("This is placeholder narration for the first beat.", hold=0.3):
        with Sync(equalize_runtimes=True):
            camera.fly_to(OUT * 9 + UP * 1, look_at=ORIGIN)
            with Seq():
                with Seq(runtime=1.0):
                    a.spawn()
                with Seq(runtime=1.0):
                    b.spawn()
    Scene.save_frame('first_both_shapes', at=[-0.2])

    with say("And a second placeholder line for the next beat.", hold=0.3):
        with Sync():
            a.move(RIGHT * 1.5)
            b.move(LEFT * 1.5)
    Scene.save_frame('first_merged', at=[-0.2])
