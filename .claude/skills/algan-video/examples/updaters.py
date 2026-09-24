"""Batched, elapsed-time updaters and a live dependency between Mobs.

Requires Algan and its torch dependency. Values only demonstrate the API.
"""

from algan import (
    Circle, DOWN, LEFT, Off, OUT, PI, PREVIEW, RIGHT, Scene, Seq, Triangle,
    UP, easings,
)
import torch


def spin(mob, t, degrees_per_second):
    # t is elapsed time [frames, 1, 1], not a per-frame delta.
    mob.rotate(degrees_per_second * t, OUT)


def vertical_offset(mob, t, amplitude, frequency):
    # This offset is recomputed on top of the materialized timeline state.
    mob.move(UP * amplitude * torch.sin(2 * PI * frequency * t))


def follow(mob, t, target):
    # Read the target during replay, not once during authoring.
    mob.move_next_to(target, DOWN, buffer=0.2)


def build_scene():
    with Off():
        subject = Triangle().scale(0.5).move_to(LEFT).spawn()
        follower = Circle().scale(0.15).spawn()

    spin_id = subject.add_updater(spin, 90.0)
    offset_id = subject.add_updater(vertical_offset, 0.25, 1.0)
    follow_id = follower.add_updater(follow, subject)

    with Seq(runtime=2.0, easing=easings.identity):
        subject.move(RIGHT * 2)
    Scene.wait(0.25)

    # Retain the Mob and the integer updater IDs separately.
    subject.remove_updater(spin_id)
    subject.remove_updater(offset_id)
    follower.remove_updater(follow_id)
    Scene.wait(0.5)


if __name__ == "__main__":
    build_scene()
    result = Scene.save_video("renders/updaters.mp4", PREVIEW)
    print(result.status, result.output_path)
