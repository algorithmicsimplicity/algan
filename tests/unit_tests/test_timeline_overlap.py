"""Tests that the global timeline rematerializes overlapping edits correctly.

Run directly: .venv/Scripts/python.exe -m pytest tests/unit_tests/test_timeline_overlap.py -q

Multiple animations may edit the same attribute of the same mob during
overlapping time intervals (e.g. [0, 1], [0, 0.5] and [0.5, 1]). The timeline
must rebuild the recorded chain of states at any query time: the base state is
the pre-modification snapshot of the earliest-executed edit still unfinished at
that time, and every function application whose replay window covers the time
is re-applied on top, in execution order (held at its final parameters past its
own end while an earlier-executed overlapping animation is still running). See
``AnimationTimeline._resolve_replay_windows``.

These are logic-level checks (no rendering): scenes are recorded, the timeline
is materialized at hand-picked times exactly as the render loop does it, and
the resulting attribute values are compared against the analytic composition
of the recorded animations (linear rate functions).
"""

import math

import pytest
import torch

from algan import Group, Mob
from algan.animation_timeline.animation_contexts import Off, Seq, Sync
from algan.constants import easings
from algan.scene_manager import SceneManager

# In the fast suite: replay of overlapping edits is the hardest part of the
# recording engine and the part every authored animation depends on.
pytestmark = pytest.mark.fast

R = torch.tensor([1.0, 0.0, 0.0])
U = torch.tensor([0.0, 1.0, 0.0])
OUT = torch.tensor([0.0, 0.0, 1.0])


@pytest.fixture(autouse=True)
def fresh_scene():
    SceneManager.reset()


def _lin(t, s, e):
    """Linear-rate progress of an animation over [s, e] at time t."""
    return max(0.0, min(1.0, (t - s) / (e - s)))


def _now():
    return SceneManager.instance().current_scene.animation_manager.context.timespan.current_time


def _materialize(times, mobs, attr="location"):
    """Materialize the global timeline at ``times`` the way the render loop
    does, and return each mob's values there ([T, rows, channels]).
    """
    tm = mobs[0].scene.timeline_manager
    with Off(
        record_attr_modifications=False, record_funcs=False, priority_level=math.inf
    ):
        tm.set_state_to_times(torch.tensor(times, dtype=torch.float32))
        out = [m.get_animated_attribute(attr).clone() for m in mobs]
    tm.clear_buffers()
    return out


def _assert_matches(offsets, actual, expected, atol=1e-4):
    for i, dt in enumerate(offsets):
        assert torch.allclose(actual[i, 0], expected[i], atol=atol), (
            f"at t=+{dt}: got {actual[i, 0].tolist()}, expected {expected[i].tolist()}"
        )


def test_sequential_edits():
    """Non-overlapping edits (the common case) must be unaffected."""
    m = Mob().spawn(animate=False)
    t0 = _now()
    with Seq(easing=easings.identity):
        m.move(R * 2)  # [t0, t0+1]
        m.move(U * 2)  # [t0+1, t0+2]
    offs = [0.25, 0.75, 1.0, 1.5, 2.0, 2.5]
    expected = [R * 2 * _lin(dt, 0, 1) + U * 2 * _lin(dt, 1, 2) for dt in offs]
    (actual,) = _materialize([t0 + dt for dt in offs], [m])
    _assert_matches(offs, actual, expected)


def test_replay_window_resolution_extends_a_cached_prefix():
    """Appending edits after a completed resolution must match a cold replay.

    Still-heavy authoring workflows resolve the timeline repeatedly.  The
    incremental path starts from the prior per-row end checkpoint; resetting
    that checkpoint here forces the original full-history algorithm and gives
    a direct result-by-result reference.
    """
    m = Mob().spawn(animate=False)
    with Sync(easing=easings.identity):
        with Seq(duration=2):
            m.move(R * 2)
        with Seq(duration=1):
            m.move(U * 2)

    timeline = m.scene.timeline_manager
    timeline._resolve_replay_windows()
    prefix_count = timeline._resolved_prefix_count
    assert prefix_count > 0

    with Sync(easing=easings.identity):
        with Seq(duration=3):
            m.move(OUT * 2)
        with Seq(duration=0.5):
            m.move(R)
    timeline._resolve_replay_windows()
    assert timeline._resolved_prefix_count > prefix_count
    incremental = [
        edit.replay_end
        for attr_timeline in timeline.attr_to_timeline.values()
        for edit in attr_timeline.edits
    ]

    timeline._resolved_prefix_count = 0
    timeline._resolved_prefix_seq = -1
    timeline._resolved_row_ends = {}
    timeline._replay_windows_resolved = False
    for attr_timeline in timeline.attr_to_timeline.values():
        for edit in attr_timeline.edits:
            edit.replay_end = None
    for event in timeline.function_timeline.function_applications:
        event.replay_end = None
    timeline._resolve_replay_windows()
    cold = [
        edit.replay_end
        for attr_timeline in timeline.attr_to_timeline.values()
        for edit in attr_timeline.edits
    ]

    assert cold == pytest.approx(incremental)


def test_clear_buffers_retains_query_indexes_until_recording_changes_history():
    """Finishing a render must not discard indexes over an unchanged edit log."""
    m = Mob().spawn(animate=False)
    t0 = _now()
    m.move(R)

    _materialize([t0 + 0.5], [m])
    attr_timeline = m.scene.timeline_manager.attr_to_timeline["location"]
    assert attr_timeline._is_ready_for_queries
    assert attr_timeline._query_cache
    key, prepared = next(iter(attr_timeline._query_cache.items()))
    first_edit_descriptor = attr_timeline._edits_sorted[0]

    _materialize([t0 + 0.75], [m])
    assert attr_timeline._query_cache[key] is prepared

    m.move(U)
    assert not attr_timeline._is_ready_for_queries
    assert not attr_timeline._query_cache

    _materialize([t0 + 1.5], [m])
    assert attr_timeline._edits_sorted[0] is first_edit_descriptor
    assert attr_timeline._prepared_edit_count == len(attr_timeline.edits)


def test_edits_ending_at_same_time():
    """Two edits of the same rows ending at the same time: the base must be
    the pre-value of the first-executed one, and both must replay in order.
    """
    m = Mob().spawn(animate=False)
    t0 = _now()
    with Sync(easing=easings.identity):
        m.move(R * 2)  # [t0, t0+1]
        m.move(U * 2)  # [t0, t0+1]
    offs = [0.25, 0.5, 0.9, 1.0, 1.5]
    expected = [(R * 2 + U * 2) * _lin(dt, 0, 1) for dt in offs]
    (actual,) = _materialize([t0 + dt for dt in offs], [m])
    _assert_matches(offs, actual, expected)


def test_nested_overlap():
    """[0,1] and [0,0.5]: the later-executed edit ends first. Its pre-value
    must not be used as base while the first animation is mid-flight, and its
    finished contribution must persist through [0.5, 1).
    """
    m = Mob().spawn(animate=False)
    t0 = _now()
    with Sync(easing=easings.identity):
        with Seq(duration=1):
            m.move(R * 2)  # [t0, t0+1]
        with Seq(duration=0.5):
            m.move(U * 2)  # [t0, t0+0.5]
    offs = [0.25, 0.4999, 0.5, 0.75, 1.0, 1.5]
    expected = [R * 2 * _lin(dt, 0, 1) + U * 2 * _lin(dt, 0, 0.5) for dt in offs]
    (actual,) = _materialize([t0 + dt for dt in offs], [m])
    _assert_matches(offs, actual, expected)


def test_three_overlapping_intervals():
    """Edits on [0,1], [0,0.5] and [0.5,1] applied together."""
    m = Mob().spawn(animate=False)
    t0 = _now()
    with Sync(easing=easings.identity):
        with Seq(duration=1):
            m.move(R * 2)  # [t0, t0+1]
        with Seq(duration=1):
            m.move(U * 2)  # [t0, t0+0.5]
            m.move(OUT * 2)  # [t0+0.5, t0+1]
    offs = [0.25, 0.4999, 0.5, 0.75, 0.9999, 1.0, 1.5]
    expected = [
        R * 2 * _lin(dt, 0, 1) + U * 2 * _lin(dt, 0, 0.5) + OUT * 2 * _lin(dt, 0.5, 1)
        for dt in offs
    ]
    (actual,) = _materialize([t0 + dt for dt in offs], [m])
    _assert_matches(offs, actual, expected)


def test_partial_row_overlap():
    """mob1 animated on [0,2]; a group edit covering mob1+mob2 rows on [0,1].
    The group edit overlaps an earlier edit only on mob1's rows; both mobs'
    rows must nevertheless stay consistent (the group animation's replay
    window is extended on all of its rows together).
    """
    m1 = Mob().spawn(animate=False)
    m2 = Mob().spawn(animate=False)
    g = Group([m1, m2])
    t0 = _now()
    with Sync(easing=easings.identity):
        with Seq(duration=2):
            m1.move(R * 2)  # [t0, t0+2], m1 rows only
        with Seq(duration=1):
            g.move(U * 2)  # [t0, t0+1], m1+m2 rows
    offs = [0.5, 0.9999, 1.0, 1.5, 2.0, 2.5]
    exp1 = [R * 2 * _lin(dt, 0, 2) + U * 2 * _lin(dt, 0, 1) for dt in offs]
    exp2 = [U * 2 * _lin(dt, 0, 1) for dt in offs]
    a1, a2 = _materialize([t0 + dt for dt in offs], [m1, m2])
    _assert_matches(offs, a1, exp1)
    _assert_matches(offs, a2, exp2)


def test_overlapping_rotations_continuity():
    """Two overlapping rotations ([0,1] and [0,0.5]) of one mob's basis.
    Rotations compose by reading the current basis, so this exercises the
    non-additive replay chain: the state must be continuous across the inner
    edit's end and settle exactly on the recorded final basis.
    """
    m = Mob().spawn(animate=False)
    t0 = _now()
    with Sync(easing=easings.identity):
        with Seq(duration=1):
            m.rotate(90, OUT)  # [t0, t0+1]
        with Seq(duration=0.5):
            m.rotate(90, U)  # [t0, t0+0.5]
    offs = [0.4999, 0.5, 0.9999, 1.0, 1.5]
    (basis,) = _materialize([t0 + dt for dt in offs], [m], attr="basis")
    # Continuity across the inner edit's end and the outer edit's end (the
    # 1e-4 time step bounds the motion between the compared frames).
    assert torch.allclose(basis[0], basis[1], atol=2e-3), "jump at t=0.5"
    assert torch.allclose(basis[2], basis[3], atol=2e-3), "jump at t=1.0"
    # After both animations end, the state is the recorded final basis.
    assert torch.allclose(basis[3], basis[4], atol=1e-5), "settled state drifts"


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
