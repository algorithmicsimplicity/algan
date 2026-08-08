import numpy as np
import torch

from algan import Axes, BLUE, Group, Off, RIGHT, Scene, Star, UP


def _tensor(value):
    return value.detach().clone().as_subclass(torch.Tensor)


def _non_geometry_state(root):
    """Snapshot concrete animatable rows that a geometry edit should not touch."""
    tm = root.scene.timeline_manager
    state = []
    for mob in root.get_descendants():
        attrs = {}
        for attr in dict.fromkeys(mob.animatable_attrs):
            if attr in {"location", "basis"}:
                continue
            timeline = tm.attr_to_timeline.get(attr)
            if timeline is None or mob.id not in timeline.mob_id_to_inds:
                continue
            attrs[attr] = _tensor(
                mob.get_animated_attribute(attr, include_descendants=False)
            )
        state.append((mob, attrs))
    return state


def _assert_same_state(before, after):
    assert len(before) == len(after)
    for (before_mob, before_attrs), (after_mob, after_attrs) in zip(before, after):
        assert before_mob is after_mob
        assert before_attrs.keys() == after_attrs.keys()
        for attr, before_value in before_attrs.items():
            assert torch.equal(before_value, after_attrs[attr]), (
                type(before_mob).__name__,
                attr,
            )


def test_parent_move_is_synced_before_direct_manim_compat_transform():
    with Scene() as scene:
        star = Star(scene=scene)
        with Off(animation_manager=scene.animation_manager):
            Group(star).move(UP * 1.35)

        # The retained Manim object is synchronized lazily.  Work out exactly
        # what Manim's next rotation should produce from that synchronized state.
        expected = star.get_manim_mobject().copy()
        expected.rotate(0.2)

        with Off(animation_manager=scene.animation_manager):
            star.rotate(0.2)

        np.testing.assert_allclose(
            star.get_manim_mobject().get_center(),
            expected.get_center(),
            atol=1e-6,
        )
        # Regression guard for the original failure: the star must remain near
        # its translated position instead of jumping back to the origin.
        assert float(star.location[..., 1].mean()) > 1.0


def test_parent_animatable_state_survives_delegated_geometry_edit():
    with Scene() as scene:
        star = Star(scene=scene)
        group = Group(star)
        with Off(animation_manager=scene.animation_manager):
            group.move(UP * 1.1 + RIGHT * 0.3)
            group.color = BLUE
            group.opacity = 0.37
            group.glow = 0.61

        before = _non_geometry_state(star)
        with Off(animation_manager=scene.animation_manager):
            # ``width`` is a Manim property, so this goes through the backing
            # Mobject and rebuilds the Algan geometry.
            star.set(width=2.3)
        after = _non_geometry_state(star)

        _assert_same_state(before, after)


def test_composite_query_and_saved_delegate_see_later_parent_move():
    with Scene() as scene:
        axes = Axes(
            x_range=(-2, 2, 1),
            y_range=(-2, 2, 1),
            scene=scene,
        )
        c2p = axes.c2p
        origin_before = c2p(0, 0)
        shift = UP * 0.8 + RIGHT * 0.45

        with Off(animation_manager=scene.animation_manager):
            Group(axes).move(shift)

        # ``c2p`` was looked up before the parent edit.  The delegated wrapper
        # must synchronize/rebind at call time, not only in __getattr__.
        origin_after = c2p(0, 0)
        torch.testing.assert_close(
            _tensor(origin_after - origin_before).reshape(-1, 3)[0],
            _tensor(shift).reshape(-1, 3)[0],
        )


def test_direct_backing_edit_preserves_algan_only_state():
    with Scene() as scene:
        star = Star(scene=scene)
        with Off(animation_manager=scene.animation_manager):
            Group(star).set(opacity=0.35, glow=0.27, color=BLUE)

        before = _non_geometry_state(star)
        location_before = star.location.clone()
        backing = star.get_manim_mobject()
        backing.shift(np.array([0.6, 0.2, 0.0]))
        star.sync_from_manim()

        after = _non_geometry_state(star)
        _assert_same_state(before, after)
        torch.testing.assert_close(
            (star.location - location_before).reshape(-1, 3).mean(0),
            torch.tensor([0.6, 0.2, 0.0], dtype=star.location.dtype),
        )
