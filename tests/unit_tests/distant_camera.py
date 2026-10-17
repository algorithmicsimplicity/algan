"""A framing-preserving distant perspective camera, for the float32 tests.

Before Algan had an orthographic camera, ``Camera.set_near_orthographic(distance)``
stood in for one by backing the eye and its screen ``distance`` apart along the
view axis while keeping the frame on the plane through ``ORIGIN``. The camera is
truly orthographic now, but a distant perspective camera is still something a
script can build, and the tests that hold float32 rounding at a far eye -- the
precision warnings, the depth-order check, shadows and reflections seen from
thousands of units out -- still need one. :func:`pull_back` is that move.
"""

from __future__ import annotations

import torch

from algan import Sync


def pull_back(camera, distance):
    """Back ``camera`` off until its eye is ``distance`` from its screen.

    The eye and the screen slide along the forward axis together, keeping the
    frame the camera sees on the plane through ``ORIGIN`` (or on its screen
    plane, when ``ORIGIN`` is not in front of it), and ``focus_distance`` grows
    by the eye's move so the plane in focus holds still. Recorded like any
    camera move: call it under ``Off()`` for an immediate change. Returns the
    camera.
    """
    forward = camera.get_forward_direction()
    screen_distance = (camera.screen.location - camera.location).norm(
        dim=-1, keepdim=True
    )
    depth = (-camera.location * forward).sum(-1, keepdim=True)
    depth = torch.where(depth > 0, depth, screen_distance)
    scale_change = 1 - distance / screen_distance
    location = camera.location + forward * depth * scale_change
    screen_location = (
        camera.screen.location + forward * (depth - screen_distance) * scale_change
    )
    with Sync(animation_manager=camera.animation_manager):
        camera.set_non_recursive(location=location)
        camera.screen.set_non_recursive(location=screen_location)
        camera.set_non_recursive(
            focus_distance=(camera.focus_distance - depth * scale_change).clamp_min(
                1e-4
            )
        )
    return camera
