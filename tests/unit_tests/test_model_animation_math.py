"""The keyframe math an imported model's animation is evaluated with.

``algan.mobs.three_d_models.animation`` turns a clip's per-node keyframe tracks
into local ``4x4`` transforms and composes them down the node hierarchy; every
pose ``Model3D.play_animation`` bakes goes through it. It is pure tensor math,
so these tests need no Scene and no render: quaternion <-> matrix conversion
(in every branch of the stable matrix-to-quaternion selection, including the
half turns where the trace is negative), TRS decomposition, linear and
spherical track sampling, rest-pose overrides, hierarchy composition and the
bake's sample times.

A quaternion and its negation are the same rotation, so rotations are compared
sign-agnostically. Feature tests: they fail only when this module changes, so
they stay outside the fast suite.
"""

from __future__ import annotations

import math

import pytest
import torch

from algan.mobs.three_d_models import animation as anim
from algan.mobs.three_d_models.scene_data import (
    AnimationData,
    NodeAnimation,
    NodeData,
)


def _quat(axis, degrees):
    """Unit quaternion ``(x, y, z, w)`` turning ``degrees`` about ``axis``."""
    axis = torch.tensor(axis, dtype=torch.float64)
    axis = axis / axis.norm()
    half = math.radians(degrees) / 2
    return torch.cat([axis * math.sin(half), torch.tensor([math.cos(half)])]).float()


def _random_quats(n, seed):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(n, 4, generator=g)
    return q / q.norm(dim=-1, keepdim=True)


def _assert_same_rotation(actual, expected, atol=1e-5):
    """``actual`` and ``expected`` are one rotation: equal up to overall sign."""
    error = min((actual - expected).abs().max(), (actual + expected).abs().max())
    assert error <= atol, f"{actual.tolist()} is not +-{expected.tolist()}"


def _angle_degrees(q):
    """The rotation angle of a unit quaternion, in ``[0, 180]`` degrees."""
    return math.degrees(2 * math.acos(min(1.0, abs(float(q[3])))))


def _branch(m):
    """Which branch of the stable selection ``matrix_to_quaternion`` takes."""
    if m[0, 0] + m[1, 1] + m[2, 2] > 0:
        return "trace"
    if m[0, 0] >= m[1, 1] and m[0, 0] >= m[2, 2]:
        return "x"
    return "y" if m[1, 1] >= m[2, 2] else "z"


# --- quaternions and matrices ------------------------------------------------


def test_quaternion_to_matrix_is_a_proper_rotation_of_the_normalized_input():
    quats = _random_quats(64, seed=0)
    matrices = anim.quaternion_to_matrix(quats)

    assert matrices.shape == (64, 3, 3), "a batch of quaternions -> a batch of R"
    eye = torch.eye(3).expand(64, 3, 3)
    torch.testing.assert_close(
        matrices @ matrices.transpose(-1, -2), eye, atol=1e-5, rtol=0
    )
    torch.testing.assert_close(torch.linalg.det(matrices), torch.ones(64))
    # The input is normalized first, so a scaled quaternion is the same turn.
    torch.testing.assert_close(anim.quaternion_to_matrix(quats * 3.5), matrices)


@pytest.mark.parametrize(
    ("axis", "before", "after"),
    [
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
        ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
    ],
)
def test_a_quarter_turn_follows_the_right_hand_rule(axis, before, after):
    """glTF rotations act on column vectors in a right-handed frame."""
    rotation = anim.quaternion_to_matrix(_quat(axis, 90))
    torch.testing.assert_close(
        rotation @ torch.tensor(before, dtype=torch.float32),
        torch.tensor(after, dtype=torch.float32),
        atol=1e-6,
        rtol=0,
    )


def test_matrix_to_quaternion_round_trips_random_rotations_in_every_branch():
    quats = _random_quats(300, seed=1)
    branches = set()
    for q in quats:
        matrix = anim.quaternion_to_matrix(q)
        branches.add(_branch(matrix))
        recovered = anim.matrix_to_quaternion(matrix)
        assert recovered.shape == (4,)
        assert abs(float(recovered.norm()) - 1) < 1e-5
        _assert_same_rotation(recovered, q)
    assert branches == {"trace", "x", "y", "z"}, "a branch was never exercised"


@pytest.mark.parametrize(
    "axis", [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, -2, 0.5), (-1, 1, 1)]
)
@pytest.mark.parametrize("degrees", [180.0, 179.9, 150.0])
def test_matrix_to_quaternion_is_stable_near_a_half_turn(axis, degrees):
    """Near 180 degrees the trace is close to -1 and ``w`` close to 0, which is
    what the largest-diagonal branches exist for: dividing by ``4w`` there would
    amplify rounding into a wrong axis.
    """
    q = _quat(axis, degrees)
    matrix = anim.quaternion_to_matrix(q)
    assert _branch(matrix) != "trace"
    recovered = anim.matrix_to_quaternion(matrix)
    assert torch.isfinite(recovered).all()
    _assert_same_rotation(recovered, q, atol=2e-6)


def test_matrix_to_quaternion_of_the_identity_is_the_identity_quaternion():
    torch.testing.assert_close(
        anim.matrix_to_quaternion(torch.eye(3)), torch.tensor([0.0, 0.0, 0.0, 1.0])
    )


# --- TRS composition ---------------------------------------------------------


def test_compose_trs_scales_then_rotates_then_translates():
    translation = torch.tensor([1.0, -2.0, 3.0])
    scale = torch.tensor([2.0, 3.0, 4.0])
    matrix = anim.compose_trs(translation, _quat((0, 0, 1), 90), scale)

    point = torch.tensor([1.0, 1.0, 1.0, 1.0])
    # Scaled to (2, 3, 4), turned a quarter about +z to (-3, 2, 4), then moved.
    torch.testing.assert_close(
        matrix @ point, torch.tensor([-2.0, 0.0, 7.0, 1.0]), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(matrix[3], torch.tensor([0.0, 0.0, 0.0, 1.0]))


def test_decompose_trs_recovers_the_components_it_was_composed_from():
    g = torch.Generator().manual_seed(2)
    for q in _random_quats(200, seed=3):
        translation = torch.randn(3, generator=g) * 5
        scale = torch.rand(3, generator=g) * 3 + 0.1
        matrix = anim.compose_trs(translation, q, scale)

        t, r, s = anim.decompose_trs(matrix)

        torch.testing.assert_close(t, translation)
        torch.testing.assert_close(s, scale, atol=1e-5, rtol=1e-5)
        _assert_same_rotation(r, q, atol=2e-5)


def test_decompose_trs_returns_a_copy_of_the_translation():
    matrix = anim.compose_trs([1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0])
    translation, _, _ = anim.decompose_trs(matrix)
    translation[0] = 99.0
    assert float(matrix[0, 3]) == 1.0


@pytest.mark.parametrize(
    "scale",
    [(-2.0, 3.0, 4.0), (2.0, -3.0, 4.0), (2.0, 3.0, -4.0), (-1.0, -1.0, -1.0)],
)
def test_a_mirror_is_folded_into_the_first_scale_axis(scale):
    """A negative determinant cannot be a rotation, so its sign goes on the
    first scale axis; whichever axis was mirrored, recomposing gives the same
    matrix back and the rotation stays proper.
    """
    q = _quat((1, 2, 3), 40)
    matrix = anim.compose_trs([0.5, 0.0, -1.0], q, scale)

    t, r, s = anim.decompose_trs(matrix)

    assert float(s[0]) < 0
    assert (s[1:] > 0).all()
    torch.testing.assert_close(s.abs(), torch.tensor(scale).abs(), rtol=1e-5, atol=0)
    assert float(torch.linalg.det(anim.quaternion_to_matrix(r))) > 0
    torch.testing.assert_close(anim.compose_trs(t, r, s), matrix, atol=1e-5, rtol=0)


def test_two_mirrored_axes_are_a_half_turn_with_positive_scale():
    matrix = anim.compose_trs([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [-2.0, -3.0, 4.0])

    _, r, s = anim.decompose_trs(matrix)

    torch.testing.assert_close(s, torch.tensor([2.0, 3.0, 4.0]))
    _assert_same_rotation(r, _quat((0, 0, 1), 180))


# --- track sampling ----------------------------------------------------------


def test_vector_tracks_interpolate_linearly_and_hold_their_end_values():
    times = torch.tensor([1.0, 2.0, 4.0])
    values = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, -2.0], [2.0, 6.0, -2.0]])
    expected = {
        0.0: values[0],  # before the first key: held
        1.0: values[0],
        1.5: torch.tensor([1.0, 0.0, -1.0]),
        2.0: values[1],
        3.0: torch.tensor([2.0, 3.0, -2.0]),
        4.0: values[2],
        9.0: values[2],  # after the last key: held
    }
    for t, value in expected.items():
        torch.testing.assert_close(anim.sample_vector_track(times, values, t), value)


def test_a_vector_sample_never_aliases_the_keyframes():
    values = torch.tensor([[1.0, 2.0, 3.0]])
    sample = anim.sample_vector_track(torch.tensor([0.0]), values, 0.0)
    sample[0] = 99.0
    assert float(values[0, 0]) == 1.0


def test_quaternion_tracks_slerp_at_constant_angular_speed():
    """A 120 degree key pair is far outside the nlerp shortcut, so this is the
    true slerp: the angle grows linearly with time about the keys' own axis.
    """
    axis = (1.0, 2.0, 2.0)
    times = torch.tensor([0.0, 3.0])
    quats = torch.stack([_quat(axis, 0), _quat(axis, 120)])
    for t in [0.3, 0.75, 1.5, 2.25, 2.9]:
        sample = anim.sample_quaternion_track(times, quats, t)
        assert abs(float(sample.norm()) - 1) < 1e-6
        _assert_same_rotation(sample, _quat(axis, 120 * t / 3), atol=1e-5)


def test_quaternion_tracks_take_the_shorter_arc():
    """``q`` and ``-q`` are one rotation, and a file may store either. A key
    pair on opposite hemispheres must still turn the short way: from rest to a
    120 degree key, half way is 60 degrees -- not the 120 degrees the other way
    round that slerping straight at ``-q`` passes through.
    """
    times = torch.tensor([0.0, 1.0])
    end = _quat((0, 0, 1), 120)
    stored_flipped = torch.stack([_quat((0, 0, 1), 0), -end])

    half = anim.sample_quaternion_track(times, stored_flipped, 0.5)

    _assert_same_rotation(half, _quat((0, 0, 1), 60), atol=1e-5)
    assert _angle_degrees(half) == pytest.approx(60, abs=1e-3)


def test_quaternion_tracks_clamp_outside_their_keys_and_return_unit_rotations():
    times = torch.tensor([1.0, 2.0])
    # Stored un-normalized, as accessors of normalized integers decode.
    first, last = _quat((0, 1, 0), 10), _quat((0, 1, 0), 70)
    quats = torch.stack([first * 3, last * 0.5])

    for t, expected in [(0.0, first), (1.0, first), (2.0, last), (5.0, last)]:
        sample = anim.sample_quaternion_track(times, quats, t)
        torch.testing.assert_close(sample, expected)
    middle = anim.sample_quaternion_track(times, quats, 1.5)
    assert abs(float(middle.norm()) - 1) < 1e-6
    _assert_same_rotation(middle, _quat((0, 1, 0), 40), atol=1e-5)


def test_nearly_identical_quaternion_keys_interpolate_without_blowing_up():
    """Below about 3.6 degrees apart slerp divides by a vanishing sine, so the
    sampler normalizes a linear blend there instead.
    """
    times = torch.tensor([0.0, 1.0])
    quats = torch.stack([_quat((1, 0, 0), 30.0), _quat((1, 0, 0), 30.01)])
    sample = anim.sample_quaternion_track(times, quats, 0.5)
    assert torch.isfinite(sample).all()
    assert abs(float(sample.norm()) - 1) < 1e-6
    _assert_same_rotation(sample, _quat((1, 0, 0), 30.005), atol=1e-6)


def test_quaternion_tracks_pick_the_segment_containing_t():
    times = torch.tensor([0.0, 1.0, 2.0])
    quats = torch.stack([_quat((0, 0, 1), d) for d in (0, 90, 180)])
    _assert_same_rotation(
        anim.sample_quaternion_track(times, quats, 1.5), _quat((0, 0, 1), 135)
    )
    _assert_same_rotation(
        anim.sample_quaternion_track(times, quats, 1.0), _quat((0, 0, 1), 90)
    )


# --- node transforms and the hierarchy ----------------------------------------


def _trs(translation=(0, 0, 0), rotation=(0, 0, 0, 1), scale=(1, 1, 1)):
    return anim.compose_trs(translation, rotation, scale)


def test_a_node_without_a_channel_keeps_its_rest_transform():
    rest = _trs((1, 2, 3), _quat((0, 1, 0), 30), (2, 2, 2))
    torch.testing.assert_close(
        anim.evaluate_node_local_transform(NodeData(transform=rest), None, 0.5), rest
    )
    torch.testing.assert_close(
        anim.evaluate_node_local_transform(NodeData(transform=None), None, 0.5),
        torch.eye(4),
    )


@pytest.mark.parametrize("keyed", ["translation", "rotation", "scale"])
def test_a_channel_overrides_only_the_components_it_keys(keyed):
    rest_t, rest_r, rest_s = (1.0, 2.0, 3.0), _quat((1, 0, 0), 30), (2.0, 0.5, 1.5)
    rest = _trs(rest_t, rest_r, rest_s)
    times = torch.tensor([0.0, 1.0])
    key_t = torch.tensor([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    key_r = torch.stack([_quat((0, 0, 1), 0), _quat((0, 0, 1), 90)])
    key_s = torch.tensor([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]])
    channel = NodeAnimation(node_name="n")
    if keyed == "translation":
        channel.position_times, channel.positions = times, key_t
    elif keyed == "rotation":
        channel.rotation_times, channel.rotations = times, key_r
    else:
        channel.scaling_times, channel.scalings = times, key_s

    local = anim.evaluate_node_local_transform(NodeData(transform=rest), channel, 1.0)

    t, r, s = anim.decompose_trs(local)
    torch.testing.assert_close(
        t, key_t[1] if keyed == "translation" else torch.tensor(rest_t)
    )
    _assert_same_rotation(r, key_r[1] if keyed == "rotation" else rest_r)
    torch.testing.assert_close(
        s, key_s[1] if keyed == "scale" else torch.tensor(rest_s), atol=1e-5, rtol=0
    )


def test_a_channel_on_a_node_without_a_rest_transform_starts_from_identity():
    channel = NodeAnimation(
        node_name="n",
        position_times=torch.tensor([0.0, 2.0]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 4.0, 0.0]]),
    )
    local = anim.evaluate_node_local_transform(NodeData(transform=None), channel, 1.0)
    expected = torch.eye(4)
    expected[1, 3] = 2.0
    torch.testing.assert_close(local, expected)


def test_clip_channels_are_matched_to_nodes_by_name():
    nodes = [
        NodeData(name="a", transform=_trs((1, 0, 0))),
        NodeData(name="b", transform=_trs((0, 1, 0))),
    ]
    channel = NodeAnimation(
        node_name="b",
        position_times=torch.tensor([0.0, 1.0]),
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]]),
    )
    clip = AnimationData(name="c", runtime=1.0, channels=[channel])

    a, b = anim.evaluate_animated_locals(nodes, clip, 1.0)
    torch.testing.assert_close(a, nodes[0].transform)
    torch.testing.assert_close(b[:3, 3], torch.tensor([0.0, 0.0, 5.0]))

    rest = anim.evaluate_animated_locals(nodes, None, 1.0)
    for local, node in zip(rest, nodes):
        torch.testing.assert_close(local, node.transform)


def test_world_transforms_apply_the_parent_after_the_child():
    """``world[child] = world[parent] @ local[child]``: a parent's quarter turn
    swings its child's offset round with it.
    """
    nodes = [
        NodeData(name="root", parent=-1),
        NodeData(name="arm", parent=0),
        NodeData(name="hand", parent=1),
    ]
    locals_ = [
        _trs((2, 0, 0)),
        _trs((0, 0, 0), _quat((0, 0, 1), 90)),
        _trs((1, 0, 0)),
    ]
    world = anim.compose_world_from_locals(nodes, locals_)
    torch.testing.assert_close(world[0], locals_[0])
    # Composed the other way round the hand would land at (1, 2, 0).
    torch.testing.assert_close(
        world[2][:3, 3], torch.tensor([2.0, 1.0, 0.0]), atol=1e-6, rtol=0
    )


# --- what the bake samples ----------------------------------------------------


def test_clip_key_times_is_the_sorted_union_of_every_channels_keys():
    clip = AnimationData(
        name="c",
        channels=[
            NodeAnimation(
                node_name="a",
                position_times=torch.tensor([0.5, 0.0]),
                rotation_times=torch.tensor([0.25, 0.5, 1.0]),
            ),
            NodeAnimation(
                node_name="b",
                # float32 noise on a shared key merges with it.
                scaling_times=torch.tensor([2.0, 1.0 + 1e-7]),
            ),
        ],
    )
    assert anim.clip_key_times(clip) == [0.0, 0.25, 0.5, 1.0, 2.0]
    assert anim.clip_key_times(AnimationData(name="empty")) == [0.0]


def test_sample_times_cover_the_runtime_on_the_fps_grid_and_every_key_inside_it():
    assert anim.sample_times(1.0, 4) == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert anim.sample_times(1.0, 4, key_times=[0.1, 0.5, 2.0, -1.0]) == [
        0.0,
        0.1,
        0.25,
        0.5,
        0.75,
        1.0,
    ]
    # A runtime off the grid still ends exactly on the runtime.
    assert anim.sample_times(0.9, 2) == [0.0, 0.5, 0.9]
    assert anim.sample_times(1.1, 2) == [0.0, 0.5, 1.0, 1.1]
    # No grid: just the ends and the keys.
    assert anim.sample_times(2.0, 0, key_times=[0.7]) == [0.0, 0.7, 2.0]
    assert anim.sample_times(0.0, 30, key_times=[0.0]) == [0.0]
    assert anim.sample_times(-1.0, 30) == [0.0]
