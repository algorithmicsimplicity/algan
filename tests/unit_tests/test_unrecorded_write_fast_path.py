"""An unrecorded attribute write stores what the decorated path stores.

``Mob.set_animated_attribute`` hands a write that nothing records -- the Mob's
subtree has not spawned, nothing is replaying -- to
``Mob._apply_change_immediately`` instead of the ``animated_function``-wrapped
``_apply_change``. That is most of what building and configuring a scene does
(``set_material`` writes fourteen attributes per part), so it skips the
decorator's per-call work and the body's second read of the rows. These tests
hold it to the decorated path: the same rows, the same bits, for the writes
it takes, and the decorated path for the writes it must not take.
"""

from __future__ import annotations

import pytest
import torch

import algan.animation_timeline.timeline as timeline_module
from algan import (
    BLUE,
    RED,
    Circle,
    Group,
    MeshPhysicalMaterial,
    MeshStandardMaterial,
    Off,
    Prism,
    Scene,
    Square,
    Text,
)
from algan.animatable_base.mob import Mob


def _author():
    """Pre-spawn writes of every flavour the fast path takes, then a spawn."""
    prism = Prism(width=0.5, height=0.7, depth=0.3, color=BLUE).set_material(
        MeshStandardMaterial(color=RED, roughness=0.18, metalness=0.55)
    )
    sphere_ish = Circle(radius=0.4).set_material(
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
    return [prism, sphere_ish, group, text, square]


def _state(scene):
    timelines = scene.timeline_manager.attr_to_timeline
    return {
        name: (
            tl.current_state[:, : tl.pointer].clone(),
            [inds.tolist() for inds in tl.mob_id_to_inds.values()],
            len(tl.edits),
        )
        for name, tl in timelines.items()
    }


def _author_state(monkeypatch, disabled):
    monkeypatch.setattr(timeline_module, "_OPT_DISABLED", frozenset(disabled))
    with Scene() as scene:
        _author()
        return _state(scene)


def test_unrecorded_writes_store_the_decorated_paths_bits(monkeypatch):
    fast = _author_state(monkeypatch, ())
    slow = _author_state(monkeypatch, ("unrecordedset",))
    assert fast.keys() == slow.keys()
    for name in fast:
        fast_rows, fast_owners, fast_edits = fast[name]
        slow_rows, slow_owners, slow_edits = slow[name]
        assert fast_owners == slow_owners, name
        assert fast_edits == slow_edits, name
        assert fast_rows.dtype == slow_rows.dtype, name
        # Bitwise, not allclose: NaN-free by construction, and -0.0 must stay.
        assert torch.equal(fast_rows.view(torch.uint8), slow_rows.view(torch.uint8)), (
            name
        )


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
