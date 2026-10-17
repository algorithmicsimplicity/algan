"""The camera's projection, as arithmetic on a render snapshot.

One model covers the pinhole camera, the true orthographic camera and every
blend between them, so that an animated switch is one continuous family rather
than two renderers. Per frame the camera is

* ``eye`` -- the camera's location, the centre of the plane every camera ray
  starts on (the *camera plane*, perpendicular to the forward axis);
* ``screen`` -- the centre of the screen plane, ``screen_distance`` (``d``)
  along the forward axis;
* the pixel bases ``pbx`` / ``pby`` -- the world steps of one unit of
  normalized screen coordinate on the screen plane (``screen_half_height``
  times the camera's right and up, see ``raytracing.utils._pixel_bases``);
* ``(persp, spread)`` -- the projection. A pinhole camera is ``(1, 0)``; a
  true orthographic camera whose frame is ``2 * H`` world units tall is
  ``(0, H / screen_half_height)``. ``Camera`` stores the pair as
  ``persp`` and ``spread * screen_half_height`` and tweens both linearly, so a
  switch at blend weight ``w`` is ``(1 - w, w * H / screen_half_height)``.

The ray through normalized screen coordinate ``(u, v)``, with
``L = u * pbx + v * pby``, starts at ``eye + spread * L`` and runs along
``(screen - eye) + persp * L``. At ``(1, 0)`` that is the pinhole's
``eye -> screen + L``; at ``persp = 0`` every ray is parallel to the forward
axis and the origins tile the camera plane. In between it is exactly a pinhole
camera whose eye sits ``d * spread / persp`` behind the camera plane, with the
camera plane as its near limit -- which is why a blend keeps the frame of the
plane at depth ``H * d / screen_half_height`` fixed while perspective drains
away. Nothing is ever visible behind the camera plane: it is where every ray
starts, for a pinhole (whose rays all start at the eye) as for the rest.

Everything a renderer derives from the camera follows from that ray:

* a point at depth ``z`` (along forward, from the camera plane) lands at
  ``L = lateral * d / z_eff`` with ``z_eff = spread * d + persp * z`` -- the
  perspective divide with ``z`` replaced by ``z_eff`` (:func:`effective_depth`);
* the world size of one pixel there is ``pixel_world_scale * z_eff``, i.e.
  ``pixel_world_scale * persp`` per unit depth plus a constant
  ``pixel_world_scale * spread * d`` (:func:`footprint_terms`);
* the direction back towards the camera from a point ``p`` is
  ``persp * (eye - p) - spread * (screen - eye)`` (:func:`view_directions`).

The pinhole case is arranged to reproduce the historical arithmetic bit for
bit: every blend term is either multiplied by ``spread == 0`` or by
``persp == 1``.
"""

from __future__ import annotations

import torch

#: The snapshot's ``camera_projection`` for a pinhole camera.
PINHOLE = (1.0, 0.0)


def projection_rows(projection, frames, device, dtype=torch.float32):
    """``[frames, 2]`` ``(persp, spread)`` rows from a snapshot entry.

    ``projection`` is the render snapshot's ``camera_projection`` (any shape
    ending in 2 whose leading dimension is the frame count or 1), or ``None``
    for a pinhole camera -- camera doubles and older snapshots carry none.
    """
    if projection is None:
        return torch.tensor(PINHOLE, device=device, dtype=dtype).expand(frames, 2)
    rows = projection.reshape(projection.shape[0], 2).to(device=device, dtype=dtype)
    if rows.shape[0] != frames:
        rows = rows.expand(frames, 2)
    return rows


def is_pinhole(projection) -> bool:
    """Whether every frame of a snapshot entry is a pinhole camera.

    The renderer keeps its historical code paths for such batches, so a
    perspective render does not change by a bit.
    """
    if projection is None:
        return True
    rows = projection.reshape(-1, 2)
    return bool(((rows[:, 0] == 1.0) & (rows[:, 1] == 0.0)).all())


def kernel_projection(rows, stand_in):
    """``(table, projected)``: what a kernel takes for a batch's projection.

    ``rows`` is the batch's per-frame table -- the render kernels' ``[T, 6]``
    ``cam_proj`` (``(persp, spread, footprint base, screen - eye)``) or the
    tessellation criteria's ``[T, 2]`` ``(persp, spread)`` -- or ``None`` for a
    pinhole batch. A pinhole batch's kernels are compiled without the code
    that reads the table (``raytrace_kernels_taichi.CameraProjection``,
    ``logical_pn_taichi._frame_projection``), so a perspective render does
    not change by a bit; ``stand_in`` then fills its slot unread. It must be a
    2-D float32 array, so the slot keeps one compiled variant, and for a
    kernel that binds the table through the arena it must live in that arena:
    the launch's ``cam_origin`` is both.
    """
    if rows is None:
        return stand_in, False
    return rows, True


def effective_depth(depth, screen_distance, persp, spread):
    """The divisor that replaces ``depth`` in the perspective divide.

    ``spread * screen_distance + persp * depth``, in the same units as
    ``depth`` -- which callers may have scaled (by a normal's length, say),
    provided ``screen_distance`` is scaled alike. Equal to ``depth`` for a
    pinhole camera, bit for bit.
    """
    return spread * screen_distance + persp * depth


def footprint_terms(pixel_world_scale, screen_distance, persp, spread):
    """``(slope, base)`` of the world size of one pixel against depth.

    A pixel at planar depth ``z`` from the camera plane is
    ``slope * z + base`` world units across: ``pixel_world_scale * persp``
    and ``pixel_world_scale * spread * screen_distance``. A pinhole camera's
    slope is ``pixel_world_scale`` itself and its base zero; an orthographic
    camera's slope is zero.
    """
    return pixel_world_scale * persp, pixel_world_scale * (spread * screen_distance)


def view_directions(points, eye, screen, persp, spread):
    """Unnormalized directions from ``points`` back along their camera rays.

    ``persp * (eye - points) - spread * (screen - eye)``. For a pinhole camera
    that is ``eye - points`` exactly; for an orthographic one it is minus the
    forward axis for every point.
    """
    toward = eye - points
    if not torch.is_tensor(spread) and spread == 0.0 and persp == 1.0:
        return toward
    return persp * toward - spread * (screen - eye)


def ray_origins_for_points(points, eye, screen, forward, persp, spread):
    """Where the camera ray through each point starts, on the camera plane.

    ``eye`` for a pinhole camera; the point slid back along the forward axis
    onto the camera plane for an orthographic one. ``forward`` is the unit
    forward axis. This is the camera position a shader sees for the point:
    the direction from the point to it is the view direction, and the
    distance to it is the length of the camera ray. A point behind the camera
    plane, which no ray reaches, is answered as if it lay on that plane.
    """
    rel = points - eye
    depth = (rel * forward).sum(-1, keepdim=True)
    distance = ((screen - eye) * forward).sum(-1, keepdim=True)
    lateral = rel - depth * forward
    z_eff = effective_depth(depth.clamp_min(0.0), distance, persp, spread)
    safe = torch.where(z_eff > 1e-30, z_eff, torch.ones_like(z_eff))
    return eye + lateral * (spread * distance / safe)
