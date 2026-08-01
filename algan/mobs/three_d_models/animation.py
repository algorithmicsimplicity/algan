"""Keyframe-animation evaluation for imported 3-D models (rigid / node
animation, Phase 3).

The importer stores each animation clip as a set of per-node
:class:`~algan.mobs.three_d_models.scene_data.NodeAnimation` tracks (keyframed local
translation / rotation / scale). This module turns those tracks into concrete
per-node local ``4x4`` transforms at any time ``t``; :class:`ThreeDModelMob`
then composes them down the node hierarchy and bakes the result into per-frame
world-space vertex positions -- the representation the spatio-temporal BVH
already renders, and the same substrate the later skeletal / morph phases feed.

Everything here is pure tensor math (no rendering, no Algan mob state), so the
evaluation is unit-testable on its own and reused unchanged by the skeletal
phase (which drives the very same per-bone TRS tracks).
"""
from __future__ import annotations

import torch


def quaternion_to_matrix(q):
    """Rotation matrix ``[..., 3, 3]`` from a quaternion ``[..., 4]`` in glTF
    ``(x, y, z, w)`` order. The quaternion is normalized first.
    """
    q = torch.as_tensor(q, dtype=torch.float32)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    m = torch.stack([
        1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
        2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
        2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy),
    ], dim=-1)
    return m.reshape(*q.shape[:-1], 3, 3)


def matrix_to_quaternion(m):
    """Quaternion ``(x, y, z, w)`` from a pure-rotation ``3x3`` matrix (columns
    already unit-length). Uses the numerically stable branch selection.
    """
    m = torch.as_tensor(m, dtype=torch.float32)
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        s = torch.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] >= m[1, 1] and m[0, 0] >= m[2, 2]:
        s = torch.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] >= m[2, 2]:
        s = torch.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = torch.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return torch.stack([x, y, z, w])


def decompose_trs(matrix):
    """Decompose an affine ``4x4`` transform into translation ``[3]``, a unit
    rotation quaternion ``[4]`` (x, y, z, w) and scale ``[3]``.

    Assumes a TRS matrix (no shear), which node transforms are; scale is the
    per-column length of the linear block and the rotation is that block with
    the scale divided out (a negative determinant folds one axis' sign into the
    scale so the rotation stays proper).
    """
    matrix = torch.as_tensor(matrix, dtype=torch.float32)
    translation = matrix[:3, 3].clone()
    linear = matrix[:3, :3]
    scale = linear.norm(dim=0).clamp_min(1e-12)
    if torch.det(linear) < 0:
        scale = scale.clone()
        scale[0] = -scale[0]
    rot = linear / scale
    return translation, matrix_to_quaternion(rot), scale


def compose_trs(translation, rotation_quat, scale, device=None):
    """Recompose translation ``[3]``, rotation quaternion ``[4]`` and scale
    ``[3]`` into an affine ``4x4`` transform ``M = T @ R @ S``.
    """
    rot = quaternion_to_matrix(rotation_quat)
    scale = torch.as_tensor(scale, dtype=torch.float32)
    linear = rot * scale.unsqueeze(0)  # scale columns
    out = torch.eye(4, dtype=torch.float32)
    out[:3, :3] = linear
    out[:3, 3] = torch.as_tensor(translation, dtype=torch.float32)
    if device is not None:
        out = out.to(device)
    return out


def _segment(times, t):
    """Index ``i`` and fraction ``f`` so ``t`` lies in ``[times[i], times[i+1]]``
    (clamped to the ends). ``times`` is a sorted 1-D tensor.
    """
    n = times.shape[0]
    if n == 1 or t <= float(times[0]):
        return 0, 0.0
    if t >= float(times[-1]):
        return n - 1, 0.0
    i = int(torch.searchsorted(times, torch.tensor(float(t)), right=True)) - 1
    i = max(0, min(i, n - 2))
    t0, t1 = float(times[i]), float(times[i + 1])
    f = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
    return i, f


def sample_vector_track(times, values, t):
    """Linearly-interpolated ``[D]`` sample of a keyframed vector track
    (``times`` ``[K]``, ``values`` ``[K, D]``) at time ``t``.
    """
    i, f = _segment(times, t)
    if f == 0.0 or i + 1 >= values.shape[0]:
        return values[i].clone()
    return values[i] * (1 - f) + values[i + 1] * f


def sample_quaternion_track(times, quats, t):
    """Spherically-interpolated (slerp) ``[4]`` sample of a keyframed rotation
    track (``times`` ``[K]``, ``quats`` ``[K, 4]`` in x, y, z, w) at ``t``.
    """
    i, f = _segment(times, t)
    q0 = quats[i]
    if f == 0.0 or i + 1 >= quats.shape[0]:
        return q0 / q0.norm().clamp_min(1e-12)
    q1 = quats[i + 1]
    q0 = q0 / q0.norm().clamp_min(1e-12)
    q1 = q1 / q1.norm().clamp_min(1e-12)
    dot = float((q0 * q1).sum())
    if dot < 0:  # take the shorter arc
        q1 = -q1
        dot = -dot
    if dot > 0.9995:  # near-parallel: nlerp to avoid division blow-up
        q = q0 * (1 - f) + q1 * f
        return q / q.norm().clamp_min(1e-12)
    theta0 = torch.acos(torch.tensor(min(1.0, dot)))
    theta = theta0 * f
    sin0 = torch.sin(theta0).clamp_min(1e-12)
    s0 = torch.sin(theta0 - theta) / sin0
    s1 = torch.sin(theta) / sin0
    return q0 * s0 + q1 * s1


def evaluate_node_local_transform(node, channel, t, device=None):
    """Local ``4x4`` transform of ``node`` at time ``t``. Channels present on
    ``channel`` (a :class:`NodeAnimation` or ``None``) override the matching
    component of the node's rest transform; absent components keep the rest
    value. When the node has no animation channel the rest transform is returned
    unchanged.
    """
    rest = node.transform
    if channel is None:
        if rest is None:
            out = torch.eye(4, dtype=torch.float32)
            return out.to(device) if device is not None else out
        rest = torch.as_tensor(rest, dtype=torch.float32)
        return rest.to(device) if device is not None else rest

    if rest is None:
        translation = torch.zeros(3)
        rotation = torch.tensor([0.0, 0.0, 0.0, 1.0])
        scale = torch.ones(3)
    else:
        translation, rotation, scale = decompose_trs(rest)

    if channel.positions is not None and channel.position_times is not None:
        translation = sample_vector_track(
            channel.position_times, channel.positions, t)
    if channel.rotations is not None and channel.rotation_times is not None:
        rotation = sample_quaternion_track(
            channel.rotation_times, channel.rotations, t)
    if channel.scalings is not None and channel.scaling_times is not None:
        scale = sample_vector_track(channel.scaling_times, channel.scalings, t)

    return compose_trs(translation, rotation, scale, device=device)


def evaluate_animated_locals(nodes, clip, t, device=None):
    """Per-node local ``4x4`` transforms at time ``t`` for every node, applying
    ``clip``'s channels (matched to nodes by name) over the rest pose.
    """
    channels = {}
    if clip is not None:
        for ch in clip.channels:
            channels[ch.node_name] = ch
    return [evaluate_node_local_transform(node, channels.get(node.name), t,
                                          device=device)
            for node in nodes]


def compose_world_from_locals(nodes, locals_):
    """World ``4x4`` per node from per-node local transforms, composed down the
    hierarchy (nodes are depth-first with ``parent < child``): ``world[i] =
    world[parent] @ local[i]``.
    """
    world = [None] * len(nodes)
    for i, node in enumerate(nodes):
        if node.parent < 0:
            world[i] = locals_[i]
        else:
            world[i] = world[node.parent] @ locals_[i]
    return world


def clip_key_times(clip):
    """Sorted unique keyframe times across all of a clip's channels (seconds)."""
    times = []
    for ch in clip.channels:
        for tt in (ch.position_times, ch.rotation_times, ch.scaling_times):
            if tt is not None:
                times.extend(float(x) for x in tt.reshape(-1))
    if not times:
        return [0.0]
    return sorted({round(x, 6) for x in times})


def sample_times(duration, fps, key_times=None):
    """Times (seconds) at which to bake the animation: a uniform ``fps`` grid
    over ``[0, duration]`` unioned with any authored ``key_times`` so keyed
    poses are hit exactly. Always includes ``0`` and ``duration``.
    """
    duration = max(float(duration), 0.0)
    ts = {0.0, duration}
    if fps and fps > 0 and duration > 0:
        n = int(round(duration * fps))
        for k in range(n + 1):
            ts.add(min(duration, k / float(fps)))
    if key_times:
        for kt in key_times:
            if 0.0 <= kt <= duration:
                ts.add(float(kt))
    return sorted(ts)
