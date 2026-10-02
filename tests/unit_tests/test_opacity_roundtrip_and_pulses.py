"""Opacity survives color pulses and subtree-value assignment."""

import pytest
import torch

from algan import (
    BLUE,
    GREEN,
    RED,
    Circle,
    Group,
    Mob,
    Scene,
    Seq,
    Sync,
    batch_mobs,
    easings,
)

pytestmark = [pytest.mark.fast, pytest.mark.usefixtures("fresh_scene")]


@pytest.mark.parametrize("glow", [False, True])
def test_color_pulse_keeps_explicit_fill_alpha_at_peak_and_destination(glow):
    scene = Scene()
    circle = Circle(color=BLUE, fill_opacity=0.25).spawn(animate=False)
    peak = RED.set_glow(2) if glow else RED
    with Sync(easing=easings.identity):
        circle.pulse_color(peak, new_color=GREEN)
    scene.timeline_manager.set_state_to_times(torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]))
    torch.testing.assert_close(
        circle.fill_opacity, torch.full_like(circle.fill_opacity, 0.25)
    )
    expected_rgb = torch.stack(
        [
            BLUE[:3],
            (BLUE[:3] + RED[:3]) / 2,
            RED[:3],
            (RED[:3] + GREEN[:3]) / 2,
            GREEN[:3],
        ]
    )
    torch.testing.assert_close(circle.grid.color[:, 0, :3], expected_rgb)
    if glow:
        torch.testing.assert_close(
            circle.grid.color[:, 0, 3], torch.tensor([0.0, 1.0, 2.0, 1.0, 0.0])
        )


@pytest.mark.parametrize("shapes", [False, True])
def test_recursive_opacity_roundtrip_keeps_row_layout_and_history(shapes):
    scene = Scene()
    factory = Circle if shapes else Mob
    group = Group(factory(opacity=0.3), factory(opacity=0.7)).spawn(animate=False)
    original = group.get_animated_attribute("opacity", include_descendants=True)
    rows = group._get_attr_ranges("opacity", include_descendants=True).tensor().clone()
    with Sync(easing=easings.identity):
        group.opacity = original * 0.5
    torch.testing.assert_close(
        group._get_attr_ranges("opacity", include_descendants=True).tensor(), rows
    )
    scene.timeline_manager.set_state_to_times(torch.tensor([0.0, 0.5, 1.0]))
    actual = group.get_animated_attribute("opacity", include_descendants=True)
    torch.testing.assert_close(
        actual, original * torch.tensor([1.0, 0.75, 0.5]).view(-1, 1, 1)
    )


@pytest.mark.parametrize("explicit_alpha", [False, True])
def test_pulse_still_accepts_colors_with_their_own_alpha(explicit_alpha):
    scene = Scene()
    kwargs = {"fill_opacity": 0.25} if explicit_alpha else {}
    circle = Circle(color=BLUE.set_opacity(0.25), **kwargs).spawn(animate=False)
    with Sync(easing=easings.identity):
        circle.pulse_color(RED.set_opacity(0.6), new_color=GREEN.set_opacity(0.8))
    scene.timeline_manager.set_state_to_times(torch.tensor([0.0, 0.5, 1.0]))
    torch.testing.assert_close(
        circle.grid.color[:, 0, -1], torch.tensor([0.25, 0.6, 0.8])
    )


def test_group_pulse_restores_each_parts_alpha_and_color():
    scene = Scene()
    circles = [
        Circle(color=BLUE, fill_opacity=0.2),
        Circle(color=GREEN, fill_opacity=0.7),
    ]
    group = Group(*circles).spawn(animate=False)
    with Sync(easing=easings.identity):
        group.pulse_color(RED.set_glow(2))
    scene.timeline_manager.set_state_to_times(torch.tensor([0.0, 0.5, 1.0]))
    for circle, color, alpha in zip(circles, [BLUE, GREEN], [0.2, 0.7]):
        torch.testing.assert_close(
            circle.fill_opacity, torch.full_like(circle.fill_opacity, alpha)
        )
        torch.testing.assert_close(
            circle.grid.color[:, 0, :3], torch.stack([color[:3], RED[:3], color[:3]])
        )


@pytest.mark.parametrize("packed", [False, True])
def test_roundtrip_preserves_prior_edits_and_scalar_assignment(packed):
    scene = Scene()
    mobs = [Circle(opacity=0.3), Circle(opacity=0.7)]
    group = batch_mobs(mobs) if packed else Group(*mobs)
    group.spawn(animate=False)
    start = group.get_animated_attribute("opacity", include_descendants=True)
    with Seq(easing=easings.identity):
        group.opacity = 0.8
        target = group.get_animated_attribute("opacity", include_descendants=True) * 0.5
        group.opacity = target
        group.opacity = 0.2
    scene.timeline_manager.set_state_to_times(
        torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    )
    actual = group.get_animated_attribute("opacity", include_descendants=True)
    expected = torch.cat(
        [
            start,
            (start + 0.8) / 2,
            torch.full_like(start, 0.8),
            torch.full_like(start, 0.6),
            torch.full_like(start, 0.4),
            torch.full_like(start, 0.2),
        ]
    )
    torch.testing.assert_close(actual, expected)


def test_nonrecursive_assignment_can_still_expand_own_rows():
    mob = Mob()
    values = torch.tensor([0.2, 0.8]).view(1, 2, 1)
    mob.set_animated_attribute("opacity", values, recursive=False)
    torch.testing.assert_close(mob.opacity, values)
