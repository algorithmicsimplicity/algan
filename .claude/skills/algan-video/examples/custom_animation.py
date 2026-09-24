"""A finite motion described by an explicitly interpolated parameter.

Requires Algan and torch. Replace the path equation with the user's motion.
"""

from algan import (
    LEFT, Off, PI, PREVIEW, RIGHT, Scene, Seq, Square, UP, animated_function,
    easings,
)
import torch


@animated_function(animated_args={"u": 0.0})
def move_on_curve(mob, u):
    # u is batched during materialization; this describes a state, not a step.
    mob.location = RIGHT * (4.0 * u - 2.0) + UP * torch.sin(PI * u)


def build_scene():
    with Off():
        subject = Square().scale(0.25).move_to(LEFT * 2).spawn()
    with Seq(runtime=2.0, easing=easings.identity):
        move_on_curve(subject, 1.0)
    Scene.wait(0.5)


if __name__ == "__main__":
    build_scene()
    result = Scene.save_video("renders/custom_animation.mp4", PREVIEW)
    print(result.status, result.output_path)
