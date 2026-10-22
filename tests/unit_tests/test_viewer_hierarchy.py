"""Render-free tests for the pure helpers in ``algan.viewer.hierarchy``.

Stand-in objects replace Mobs and Scenes; nothing is rendered.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from algan.animation_timeline.timeline import WIDE_ATTR_MIN_CHANNELS
from algan.viewer import hierarchy as h

pytestmark = pytest.mark.fast


class FakeMob:
    def __init__(
        self,
        id=0,
        name=None,
        children=(),
        parents=(),
        spawned=False,
        lifespan=None,
        components=None,
    ):
        self.id = id
        self.name = name
        self.children = list(children)
        self.parents = list(parents)
        self._spawned = spawned
        if lifespan is not None:
            self.lifespan = lifespan
        if components is not None:
            self._non_component = components

    def is_spawned_in_subtree(self):
        return self._spawned


class Camera(FakeMob):
    pass


class Light(FakeMob):
    pass


def _lifespan(start, end):
    return SimpleNamespace(start=lambda: start, end=lambda: end)


# --- node_id -----------------------------------------------------------------


def test_node_id_is_python_identity_not_mob_id():
    a, b = FakeMob(id=7), FakeMob(id=7)
    assert h.node_id(a) == id(a)
    assert h.node_id(a) != h.node_id(b)


# --- _is_spawned ---------------------------------------------------------------


def test_is_spawned_prefers_subtree_probe():
    assert h._is_spawned(FakeMob(spawned=True)) is True
    assert h._is_spawned(FakeMob(spawned=False)) is False


def test_is_spawned_falls_back_to_is_spawned():
    mob = SimpleNamespace(is_spawned=lambda: 1)
    assert h._is_spawned(mob) is True


def test_is_spawned_skips_raising_probe():
    def boom():
        raise RuntimeError

    mob = SimpleNamespace(is_spawned_in_subtree=boom, is_spawned=lambda: True)
    assert h._is_spawned(mob) is True


def test_is_spawned_without_probes_is_false():
    assert h._is_spawned(object()) is False


# --- _children_of ----------------------------------------------------------------


def test_children_of_hides_components_by_default():
    user, comp = FakeMob(1), FakeMob(2)
    mob = FakeMob(children=[user, comp])
    mob.get_non_component_children = lambda: [user]
    assert h._children_of(mob) == [user]
    assert h._children_of(mob, include_components=True) == [user, comp]


def test_children_of_without_getter_uses_children():
    c = FakeMob(1)
    assert h._children_of(FakeMob(children=[c])) == [c]


def test_children_of_falls_back_when_getter_raises():
    c = FakeMob(1)
    mob = FakeMob(children=[c])

    def boom():
        raise ValueError

    mob.get_non_component_children = boom
    assert h._children_of(mob) == [c]


def test_children_of_handles_missing_or_none_children():
    assert h._children_of(object()) == []
    assert h._children_of(SimpleNamespace(children=None)) == []
    assert h._children_of(SimpleNamespace(children=None), include_components=True) == []


# --- _describe / children ----------------------------------------------------------


def test_describe_fields():
    child = FakeMob(2)
    mob = FakeMob(
        5, name="ball", children=[child], spawned=True, lifespan=_lifespan(1, 3)
    )
    d = h._describe(mob)
    assert d == {
        "node": id(mob),
        "label": "ball (FakeMob #5)",
        "type": "FakeMob",
        "kind": "mob",
        "spawned": True,
        "lifespan": [1.0, 3.0],
        "has_children": True,
        "num_children": 1,
    }
    assert isinstance(d["lifespan"][0], float)


def test_describe_without_lifespan_and_unnamed():
    d = h._describe(FakeMob(3, name="_"), kind="camera")
    assert d["lifespan"] == [None, None]
    assert d["label"] == "FakeMob #3"
    assert d["kind"] == "camera"
    assert d["has_children"] is False
    assert d["num_children"] == 0


def test_describe_unresolvable_lifespan_is_none():
    d = h._describe(FakeMob(lifespan=_lifespan(None, 2)))
    assert d["lifespan"] == [None, None]


def test_children_describes_each_child():
    a, b = FakeMob(1), FakeMob(2)
    rows = h.children(FakeMob(children=[a, b]))
    assert [r["node"] for r in rows] == [id(a), id(b)]


# --- roots / index / mob_by_timeline_id ---------------------------------------------


def _scene():
    grandchild = FakeMob(3)
    child = FakeMob(2, children=[grandchild])
    root = FakeMob(1, children=[child])
    child.parents = [root]
    grandchild.parents = [child]
    cam = Camera(10, children=[FakeMob(11)])
    light = Light(20)
    # Construction order: child before parent, as ``actors`` allows.
    scene = SimpleNamespace(
        actors=[child, grandchild, root], camera=cam, light_sources=[light]
    )
    return scene, root, child, grandchild, cam, light


def test_roots_lists_root_mobs_then_camera_then_lights():
    scene, root, _, _, cam, light = _scene()
    rows = h.roots(scene)
    assert [r["node"] for r in rows] == [id(root), id(cam), id(light)]
    assert [r["kind"] for r in rows] == ["mob", "camera", "light"]


def test_roots_without_camera_or_lights():
    mob = FakeMob(1)
    rows = h.roots(SimpleNamespace(actors=[mob]))
    assert [r["node"] for r in rows] == [id(mob)]
    rows = h.roots(SimpleNamespace(actors=[mob], camera=None, light_sources=None))
    assert len(rows) == 1


def test_index_reaches_every_node_including_camera_children():
    scene, root, child, grandchild, cam, light = _scene()
    found = h.index(scene)
    expected = {root, child, grandchild, cam, light, cam.children[0]}
    assert set(found) == {id(m) for m in expected}
    assert all(found[id(m)] is m for m in expected)


def test_index_tolerates_cycles_and_shared_children():
    a, b = FakeMob(1), FakeMob(2)
    shared = FakeMob(3)
    a.children = [b, shared]
    b.children = [a, shared]
    found = h.index(SimpleNamespace(actors=[a]))
    assert set(found) == {id(a), id(b), id(shared)}


def test_mob_by_timeline_id_first_actor_wins():
    first, clone, other = FakeMob(4), FakeMob(4), FakeMob(5)
    out = h.mob_by_timeline_id(SimpleNamespace(actors=[first, clone, other]))
    assert out == {4: first, 5: other}


# --- materialized / _read / attributes_of --------------------------------------------


def test_materialized_none_is_a_noop():
    scene = SimpleNamespace()  # no timeline_manager: touching it would raise
    with h.materialized(scene, None):
        pass


class FakeTimeline:
    def __init__(self, ids, width):
        self.mob_id_to_inds = {i: [0] for i in ids}
        self.current_state = torch.zeros(1, width)


class AttrMob(FakeMob):
    def __init__(self, id, values, attrs=()):
        super().__init__(id)
        self._values = values
        self.animatable_attrs = list(attrs)

    def get_animated_attribute(self, name, include_descendants=False, copy=True):
        v = self._values[name]
        if isinstance(v, Exception):
            raise v
        return v


def _manager(**timelines):
    return SimpleNamespace(attr_to_timeline=timelines)


def test_read_derived_when_no_timeline():
    row = h._read(_manager(), AttrMob(1, {}), "scale_coefficient")
    assert row == {
        "name": "scale_coefficient",
        "value": None,
        "channels": None,
        "note": "derived",
    }


def test_read_derived_when_mob_has_no_rows():
    row = h._read(_manager(location=FakeTimeline([2], 3)), AttrMob(1, {}), "location")
    assert row["note"] == "derived"


def test_read_named_channels():
    mob = AttrMob(1, {"location": torch.tensor([[1.0, 2.0, 3.0]])})
    row = h._read(_manager(location=FakeTimeline([1], 3)), mob, "location")
    assert row["value"] == [1.0, 2.0, 3.0]
    assert row["channels"] == ["x", "y", "z"]
    assert row["shape"] == [1, 3]
    assert row["note"] is None


def test_read_channel_names_omitted_on_length_mismatch():
    mob = AttrMob(1, {"color": torch.zeros(2, 5)})
    row = h._read(_manager(color=FakeTimeline([1], 5)), mob, "color")
    assert len(row["value"]) == 10
    assert row["channels"] is None


def test_read_truncates_long_values():
    n = h.MAX_VALUES + 10
    mob = AttrMob(1, {"basis": torch.arange(n, dtype=torch.float32)})
    row = h._read(_manager(basis=FakeTimeline([1], 9)), mob, "basis")
    assert len(row["value"]) == h.MAX_VALUES
    assert row["value"][-1] == h.MAX_VALUES - 1
    assert row["note"] == f"first {h.MAX_VALUES} of {n} values"


def test_read_wide_attribute_is_not_read():
    mob = AttrMob(1, {"tex": AssertionError("must not be read")})
    row = h._read(_manager(tex=FakeTimeline([1], WIDE_ATTR_MIN_CHANNELS)), mob, "tex")
    assert row["value"] is None
    assert row["note"] == f"{WIDE_ATTR_MIN_CHANNELS} channels (too wide to display)"


def test_read_unset_and_unreadable():
    mob = AttrMob(1, {"a": AttributeError("x"), "b": ValueError("y")})
    mgr = _manager(a=FakeTimeline([1], 3), b=FakeTimeline([1], 3))
    assert h._read(mgr, mob, "a")["note"] == "unset"
    assert h._read(mgr, mob, "b")["note"] == "unreadable (ValueError)"


def test_attributes_of_dedupes_names_in_order():
    mob = AttrMob(1, {}, attrs=["location", "color", "location"])
    scene = SimpleNamespace(timeline_manager=_manager())
    rows = h.attributes_of(scene, mob)
    assert [r["name"] for r in rows] == ["location", "color"]


def test_attributes_at_none_reads_current_state():
    mob = AttrMob(1, {"location": torch.tensor([4.0, 5.0, 6.0])}, attrs=["location"])
    scene = SimpleNamespace(timeline_manager=_manager(location=FakeTimeline([1], 3)))
    rows = h.attributes_at(scene, mob, None)
    assert rows[0]["value"] == [4.0, 5.0, 6.0]


class RecordingManager:
    def __init__(self):
        self.calls = []

    def set_state_to_times(self, times):
        self.calls.append(("set", times.tolist(), times.dtype))

    def clear_buffers(self):
        self.calls.append(("clear",))


def test_materialized_sets_then_clears_state():
    mgr = RecordingManager()
    with h.materialized(SimpleNamespace(timeline_manager=mgr), 2):
        assert mgr.calls == [("set", [2.0], torch.float32)]
    assert mgr.calls[-1] == ("clear",)


def test_materialized_clears_state_on_error():
    mgr = RecordingManager()
    with (
        pytest.raises(KeyError),
        h.materialized(SimpleNamespace(timeline_manager=mgr), 0.5),
    ):
        raise KeyError
    assert [c[0] for c in mgr.calls] == ["set", "clear"]
