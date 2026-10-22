"""An unrecorded attribute write stores what the decorated path stores.

``Mob.set_animated_attribute`` hands a write that nothing records -- the Mob's
subtree has not spawned, nothing is replaying -- to
``Mob._apply_change_immediately`` instead of the ``animated_function``-wrapped
``_apply_change``. That is most of what building and configuring a scene does
(``set_material`` writes fourteen attributes per part), so it skips the
decorator's per-call work and the body's second read of the rows. These tests
hold it to the decorated path: the same rows, the same bits, the same recorded
edits and timing, and the same materialized frames, for the writes it takes,
and the decorated path for the writes it must not take.
"""

from __future__ import annotations

import pytest
import torch

import algan.animation_timeline.timeline as timeline_module
from algan import (
    BLUE,
    GREEN,
    RED,
    RIGHT,
    UP,
    YELLOW,
    Circle,
    Group,
    Lag,
    MeshPhysicalMaterial,
    MeshStandardMaterial,
    Off,
    Prism,
    Scene,
    Seq,
    Square,
    Sync,
    Text,
    Triangle,
)
from algan.animatable_base.mob import Mob


def _author_mix(scene):
    """Pre-spawn writes of every flavour the fast path takes, then a spawn."""
    Prism(width=0.5, height=0.7, depth=0.3, color=BLUE).set_material(
        MeshStandardMaterial(color=RED, roughness=0.18, metalness=0.55)
    )
    Circle(radius=0.4).set_material(
        MeshPhysicalMaterial(transmission=1.0, ior=1.5, roughness=0.0, opacity=0.4)
    )
    group = Group([Square(), Circle()])
    group.color = BLUE * 0.5  # recursive, Color arithmetic
    group.opacity = 0.75
    group.set_non_recursive(glow=0.25)
    text = Text("ab")
    text.color = RED  # a packed subtree
    with Off():
        square = Square().spawn()
    square.color = RED  # spawned: recorded, the decorated path
    return [scene]


def _author_in_contexts(scene):
    """Unrecorded writes inside Sync, Lag, Seq and Off, between recorded ones.

    An unspawned Mob's write records nothing, but it is made inside the
    context, so the timing of the recorded animations around it -- and the
    contexts' own ends -- must come out exactly as before.
    """
    actor = Square().spawn()
    waiting = Circle()
    with Sync():
        waiting.color = RED
        actor.move(RIGHT)
        waiting.opacity = 0.5
    with Lag(0.5):
        late = Square()
        late.color = BLUE
        actor.rotate(30)
        late.set(glow=0.2, opacity=0.8)
        actor.color = GREEN
    with Off():
        hidden = Triangle()
        hidden.opacity = 0.3
        actor.move(UP)
    with Seq(runtime=2):
        entering = Square()
        entering.color = YELLOW
        actor.move(-RIGHT)
        entering.spawn()
        entering.move(UP)
    waiting.spawn()
    return [scene]


def _author_explicit_alpha(scene):
    """A colour written over a Manim fill opacity keeps that opacity."""
    filled = Square()
    filled.fill_opacity = 0.4  # marks the parts whose alpha a colour keeps
    filled.color = RED  # unspawned: the fast path, through _keep_explicit_alpha
    group = Group([filled, Circle()])
    group.color = BLUE  # recursive, reaching the marked rows and the others
    group.spawn()
    filled.color = GREEN  # spawned: recorded
    return [scene]


def _author_other_scene(scene):
    """Mobs of a Scene that is not the active one write to their own Scene."""
    with Scene():  # active from here on; ``scene`` is covered by it
        square = Square(scene=scene)
        square.color = RED
        square.opacity = 0.6
        group = Group([square, Circle(scene=scene)], scene=scene)
        group.color = BLUE * 0.25
        group.spawn()
        group.color = GREEN
    return [scene]


def _author_prespawn_updater(scene):
    """An updater added before spawning, writing to Mobs that never spawn.

    Its first call (dependency tracing, at authoring) and its replays (at
    materialization) are both off the fast path; the writes around them are
    on it.
    """
    follower = Circle()
    helper = Square()
    helper.color = RED

    def follow(mob, t):
        helper.opacity = (t * 0.25).clamp(0.0, 1.0)
        mob.color = BLUE * 0.5 + RED * (t * 0.1).clamp(0.0, 0.5)

    follower.add_updater(follow)
    follower.opacity = 0.9
    follower.spawn()
    follower.move(RIGHT)
    return [scene]


AUTHORS = [
    _author_mix,
    _author_in_contexts,
    _author_explicit_alpha,
    _author_other_scene,
    _author_prespawn_updater,
]


def _bits(tensor):
    tensor = tensor.detach().contiguous()
    return (
        tuple(tensor.shape),
        tensor.dtype,
        tensor.view(torch.uint8).flatten().tolist() if tensor.numel() else [],
    )


def _edit_time(edit):
    time = edit.time
    time = getattr(time, "time", time)
    return _bits(time) if torch.is_tensor(time) else repr(time)


def _state(scene):
    """Everything a write leaves behind, then every frame materialized."""
    timeline = scene.timeline_manager
    names = sorted(timeline.attr_to_timeline)
    state = {"end": repr(getattr(scene, "max_time", None))}
    for name in names:
        attr = timeline.attr_to_timeline[name]
        state[name] = (
            _bits(attr.current_state[:, : attr.pointer]),
            [inds.tolist() for inds in attr.mob_id_to_inds.values()],
            [
                (edit.indexes.tolist(), _edit_time(edit), edit.seq)
                for edit in attr.edits
            ],
        )
    times = torch.linspace(0.0, max(float(getattr(scene, "max_time", 0)), 1.0), 9)
    timeline.set_state_to_times(times)
    for name in names:
        state[f"{name}@frames"] = _bits(timeline.attr_to_timeline[name].active_state)
    return state


def _author_state(monkeypatch, author, disabled):
    monkeypatch.setattr(timeline_module, "_OPT_DISABLED", frozenset(disabled))
    with Scene() as scene:
        return [_state(each) for each in author(scene)]


@pytest.mark.parametrize("author", AUTHORS, ids=lambda f: f.__name__[len("_author_") :])
def test_unrecorded_writes_store_the_decorated_paths_bits(monkeypatch, author):
    # Whatever the first scene of the process initializes -- Quadrants among
    # it, which turns denormal flushing on for the rest of the process --
    # happens in this warm-up arm, so both compared arms run in the same
    # floating-point state.
    _author_state(monkeypatch, author, ())
    fast = _author_state(monkeypatch, author, ())
    slow = _author_state(monkeypatch, author, ("unrecordedset",))
    assert len(fast) == len(slow)
    for fast_scene, slow_scene in zip(fast, slow):
        assert fast_scene.keys() == slow_scene.keys()
        for name in fast_scene:
            # Bitwise, not allclose: -0.0 and every rounding must match.
            assert fast_scene[name] == slow_scene[name], name


def test_the_fast_path_is_taken_before_spawn_and_not_after(monkeypatch):
    calls = []
    original = Mob._apply_change_immediately

    def counting(self, attr, *args, **kwargs):
        calls.append(attr)
        return original(self, attr, *args, **kwargs)

    monkeypatch.setattr(Mob, "_apply_change_immediately", counting)
    with Scene():
        square = Square()
        calls.clear()  # construction takes it too
        square.color = BLUE
        square.opacity = 0.5
        assert calls == ["color", "opacity"]
        calls.clear()
        square.spawn()
        square.color = RED
        assert calls == []


def test_an_updaters_writes_keep_the_decorated_path(monkeypatch):
    """Dependency tracing and replay both see the updater's writes."""
    calls = []
    original = Mob._apply_change_immediately

    def counting(self, attr, *args, **kwargs):
        calls.append(attr)
        return original(self, attr, *args, **kwargs)

    monkeypatch.setattr(Mob, "_apply_change_immediately", counting)
    with Scene() as scene:
        helper = Square()
        follower = Circle()
        follower.add_updater(lambda mob, t: setattr(helper, "glow", t * 0.0))
        follower.spawn()
        calls.clear()
        scene.timeline_manager.set_state_to_times(torch.tensor([0.0, 0.5]))
        assert "glow" not in calls


@pytest.mark.parametrize("switch", ["unrecordedset", "fastpath"])
def test_each_kill_switch_turns_the_fast_path_off(monkeypatch, switch):
    monkeypatch.setattr(timeline_module, "_OPT_DISABLED", frozenset({switch}))
    with Scene():
        assert not Square()._writes_immediately("color")


def test_a_group_location_write_by_name_keeps_the_decorated_path():
    """Group re-reads ``location`` through its pivot follow, which can write."""
    with Scene():
        group = Group([Square(), Circle()])
        assert not group._writes_immediately("location")
        assert group._writes_immediately("color")
