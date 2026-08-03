"""Tests that ``wave_color`` re-samples coarse Mobs for the duration of the wave.

Run directly: .venv/Scripts/python.exe -m pytest tests/unit_tests/test_wave_color_resolution.py -q

A colour wave is carried by vertex colours, so a Mob sampled more coarsely than
the wave's band is wide draws it as a few flat facets. ``wave_color`` therefore
refines any part too coarse to show the wave -- a Surface by re-running its
``coord_function`` over a denser ``(u, v)`` grid, a bezier circuit by laying down
finer fill and border texture grids -- and drops the resolution again once the
block containing the wave is over.

These are logic-level checks (no rendering): scenes are recorded, resolutions are
read at each stage, and the timeline is materialized exactly as the render loop
does it to confirm the refined and restored topologies both replay.
"""

import pytest
import torch

from algan.animation_timeline.animation_contexts import Off, Seq, Sync
from algan.constants.color import PURE_BLUE, YELLOW
from algan.constants.rate_funcs import identity
from algan.constants.spatial import RIGHT, UP
from algan.mobs.group import Group
from algan.mobs.shapes_2d import Rectangle, Square
from algan.mobs.shapes_3d import Cylinder
from algan.mobs.surfaces.surface import Surface
from algan.scene_manager import SceneManager


@pytest.fixture(autouse=True)
def fresh_scene():
    SceneManager.reset()
    yield
    SceneManager.reset()


def flat_sheet(**kwargs):
    """A sheet whose shape needs only its four corners, spanning [-1, 1]^2."""
    return Surface(
        coord_function=lambda uv: torch.cat(
            ((uv - 0.5) * 2, torch.zeros_like(uv[..., :1])), -1
        ),
        **kwargs,
    ).spawn(animate=False)


def refined_resolutions(mob, attr_names, wave):
    """Run ``wave`` and report the resolution during it and after the block."""
    recorded = {}
    with Sync(run_time=4):
        wave()
        recorded["during"] = tuple(getattr(mob, name) for name in attr_names)
    recorded["after"] = tuple(getattr(mob, name) for name in attr_names)
    return recorded


def test_flat_surface_is_refined_for_the_wave_and_restored_after():
    sheet = flat_sheet()
    before = (sheet.grid_width, sheet.grid_height)

    resolutions = refined_resolutions(
        sheet,
        ("grid_width", "grid_height"),
        lambda: sheet.wave_color(PURE_BLUE, wave_length=0.5),
    )

    # The wave travels along UP, which is the sheet's v axis, so only that axis
    # needs more samples.
    during = resolutions["during"]
    assert during[1] > before[1]
    assert during[0] == before[0]
    assert resolutions["after"] == before
    assert sheet.grid.location.shape[-2] == before[0] * before[1]


def test_refined_surface_keeps_its_shape_and_replays_at_both_resolutions():
    sheet = flat_sheet()
    coarse = sheet.grid.location.clone()

    with Sync(run_time=4):
        sheet.wave_color(PURE_BLUE, wave_length=0.5)
        refined = sheet.grid.location.clone()
        # Re-sampling runs coord_function over a denser grid, so every point of
        # the refined sheet still lies in the original's plane and extent.
        assert refined.shape[-2] > coarse.shape[-2]
        assert torch.allclose(refined.amin(-2), coarse.amin(-2), atol=1e-5)
        assert torch.allclose(refined.amax(-2), coarse.amax(-2), atol=1e-5)
        assert refined[..., 2].abs().max() < 1e-5

    timeline = sheet.scene.timeline_manager
    timeline.set_state_to_times(torch.linspace(0.0, 4.0, 9))
    assert sheet.grid.location.shape == (9, coarse.shape[-2], 3)
    timeline.clear_buffers()


def test_surface_already_fine_enough_is_left_alone():
    sheet = flat_sheet(grid_width=40, grid_height=40)

    resolutions = refined_resolutions(
        sheet,
        ("grid_width", "grid_height"),
        lambda: sheet.wave_color(PURE_BLUE),
    )

    assert resolutions["during"] == (40, 40)
    assert resolutions["after"] == (40, 40)


def test_samples_per_wave_none_disables_refinement():
    sheet = flat_sheet()
    before = (sheet.grid_width, sheet.grid_height)

    resolutions = refined_resolutions(
        sheet,
        ("grid_width", "grid_height"),
        lambda: sheet.wave_color(PURE_BLUE, wave_length=0.5, samples_per_wave=None),
    )

    assert resolutions["during"] == before


def test_unanimated_block_does_not_refine():
    sheet = flat_sheet()
    before = (sheet.grid_width, sheet.grid_height)

    with Off():
        sheet.wave_color(PURE_BLUE, wave_length=0.5)
        assert (sheet.grid_width, sheet.grid_height) == before

    assert (sheet.grid_width, sheet.grid_height) == before


def test_filled_circuit_texture_grid_is_refined_for_a_colour_wave():
    square = Square(color=YELLOW).spawn(animate=False)
    assert square.num_texture_points == 1

    resolutions = refined_resolutions(
        square,
        ("grid_width", "grid_height", "num_texture_points"),
        lambda: square.wave_color(PURE_BLUE, direction=RIGHT + UP),
    )

    during_width, during_height, during_points = resolutions["during"]
    assert during_width > 1
    assert during_width == during_height
    assert during_points == during_width * during_height
    assert resolutions["after"] == (1, 1, 1)
    assert square.texture_points.location.shape[-2] == 1
    assert square.texture_points.color.shape[-2] == 1
    assert square.border_texture_points.location.shape[-2] == 1
    assert square.border_texture_points.color.shape[-2] == 1


def test_refined_circuit_texture_points_stay_inside_the_shape_and_replay():
    square = Square(color=YELLOW).spawn(animate=False)
    corner = square.texture_points.location.clone()

    with Sync(run_time=4):
        square.wave_color(PURE_BLUE, direction=RIGHT + UP)
        points = square.texture_points.location
        colors = square.texture_points.color
        border_points = square.border_texture_points.location
        border_colors = square.border_texture_points.color
        assert points.shape[-2] == square.num_texture_points
        assert colors.shape[-2] == square.num_texture_points
        assert torch.equal(border_points, points)
        assert border_colors.shape[-2] == square.num_texture_points
        # The refined grid spans the circuit's own frame, so it still contains
        # the corner the single original sample sat at.
        distances = (points.reshape(-1, 3) - corner.reshape(1, 3)).norm(dim=-1)
        assert distances.min().item() < 1e-4

    timeline = square.scene.timeline_manager
    timeline.set_state_to_times(torch.linspace(0.0, 4.0, 9))
    assert square.texture_points.color.shape == (9, 1, 5)
    assert square.border_texture_points.color.shape == (9, 1, 5)
    timeline.clear_buffers()


def test_circuit_wave_color_animates_fill_and_border_texture_grids():
    square = Square(
        color=YELLOW,
        border_color=YELLOW,
        texture_grid_size=2,
    ).spawn(animate=False)

    square.wave_color(
        PURE_BLUE,
        direction=RIGHT + UP,
        samples_per_wave=None,
    )

    timeline = square.scene.timeline_manager
    timeline.set_state_to_times(torch.linspace(0.0, 3.0, 13))
    fill_colors = square.texture_points.color
    border_colors = square.border_texture_points.color
    static = YELLOW.view(1, 1, -1).expand_as(fill_colors)

    assert not torch.allclose(fill_colors, static)
    assert not torch.allclose(border_colors, static)
    timeline.clear_buffers()


def test_composite_wave_has_one_speed_across_wide_and_split_parts():
    """A panel and separately represented glyphs share one spatial wave."""
    panel = Rectangle(
        width=6,
        height=1,
        color=YELLOW,
        texture_grid_size=33,
    )
    glyphs = [
        Square(
            side_length=0.25,
            color=YELLOW,
            texture_grid_size=3,
        ).move(RIGHT * x)
        for x in (-2.0, -1.0, 0.0, 1.0, 2.0)
    ]
    composite = Group([panel, *glyphs]).spawn(animate=False)

    with Sync(run_time=1.5, rate_func=identity):
        composite.wave_color(
            PURE_BLUE,
            wave_length=0.5,
            direction=RIGHT,
            samples_per_wave=None,
        )

    times = torch.linspace(0.0, 1.5, 301)
    timeline = composite.scene.timeline_manager
    timeline.set_state_to_times(times)

    def positions_and_peak_times(part):
        positions = part.location[0, :, 0]
        color_distance = (
            part.color[..., :3] - PURE_BLUE.rgb.view(1, 1, 3)
        ).square().sum(-1)
        return positions, times[color_distance.argmin(0)]

    panel_positions, panel_peak_times = positions_and_peak_times(
        panel.texture_points
    )
    glyph_positions, glyph_peak_times = zip(
        *(positions_and_peak_times(glyph.texture_points) for glyph in glyphs)
    )
    glyph_positions = torch.cat(glyph_positions)
    glyph_peak_times = torch.cat(glyph_peak_times)

    # Fit the panel's timing line, then require the glyph samples at the same
    # world positions to land on it. Previously each glyph's event start also
    # carried its spatial offset, approximately halving the text-wave speed.
    centered_positions = panel_positions - panel_positions.mean()
    panel_speed = (
        centered_positions
        * (panel_peak_times - panel_peak_times.mean())
    ).sum() / centered_positions.square().sum()
    predicted_glyph_times = panel_peak_times.mean() + panel_speed * (
        glyph_positions - panel_positions.mean()
    )
    torch.testing.assert_close(
        glyph_peak_times,
        predicted_glyph_times,
        atol=times.diff().max().item() * 2,
        rtol=0,
    )
    timeline.clear_buffers()


def test_opacity_only_wave_leaves_a_circuit_alone():
    # A circuit's opacity is one shader parameter for the whole fill, so extra
    # texels cannot make an opacity wave (the Text spawn fade) any smoother.
    square = Square(color=YELLOW).spawn(animate=False)

    resolutions = refined_resolutions(
        square,
        ("grid_width", "num_texture_points"),
        lambda: square.wave_color(None, opacity=0.0, direction=RIGHT + UP),
    )

    assert resolutions["during"] == (1, 1)


def test_unfilled_circuit_refines_its_border_texture_for_a_color_wave():
    square = Square(color=YELLOW, filled=False).spawn(animate=False)

    resolutions = refined_resolutions(
        square,
        ("grid_width", "num_texture_points"),
        lambda: square.wave_color(PURE_BLUE, direction=RIGHT + UP),
    )

    assert resolutions["during"][0] > 1
    assert resolutions["during"][1] == resolutions["during"][0] ** 2


def test_restore_waits_for_the_outermost_block_so_siblings_stay_visible():
    # A sibling animation recorded after the wave still starts back at the
    # block's own cursor, so the restore -- which hands the refined rows to a
    # clone that despawns at that instant -- has to wait for the whole block.
    sheet = flat_sheet()

    with Sync(run_time=6):
        with Seq(run_time=3):
            sheet.wave_color(PURE_BLUE, wave_length=0.5)
        assert sheet.grid_height > 4
        with Seq(run_time=6):
            sheet.move(RIGHT * 2)
        assert sheet.grid_height > 4

    assert sheet.grid_height == 4
    # The restored surface picks up where the refined one left off, so it starts
    # where the move ended rather than back at the origin.
    assert sheet.grid.location.mean(-2)[..., 0].item() == pytest.approx(2.0, abs=1e-4)

    timeline = sheet.scene.timeline_manager
    timeline.set_state_to_times(torch.linspace(0.0, 6.0, 13))
    surfaces = [actor for actor in sheet.scene.actors if isinstance(actor, Surface)]
    visible = torch.stack(
        [surface.grid.opacity.mean(-2)[..., 0] for surface in surfaces]
    )
    centers = torch.stack(
        [surface.grid.location.mean(-2)[..., 0] for surface in surfaces]
    )
    # Exactly one incarnation is on screen at a time, and the one on screen
    # crosses the block continuously rather than jumping at the restore.
    assert torch.equal(visible.sum(0), torch.ones(13))
    shown = centers.gather(0, visible.argmax(0, keepdim=True))[0]
    steps = shown[1:] - shown[:-1]
    assert shown[0].item() == pytest.approx(0.0, abs=1e-4)
    assert shown[-1].item() == pytest.approx(2.0, abs=1e-4)
    assert (steps > 0).all()
    assert steps.max().item() < 0.5
    timeline.clear_buffers()


def test_updater_keeps_moving_the_visible_surface_across_history_splits():
    # Resolution refinement hands each completed topology to a historical
    # clone. A persistent updater must follow that ownership transfer; writing
    # only the final live Surface would leave the visible historical clone
    # stationary for the whole wave.
    sheet = flat_sheet()
    sheet.add_updater(lambda mob, time_elapsed: mob.move_to(RIGHT * time_elapsed))

    with Sync(run_time=4):
        sheet.wave_color(PURE_BLUE, wave_length=0.5)

    times = torch.linspace(0.0, 4.0, 9)
    timeline = sheet.scene.timeline_manager
    timeline.set_state_to_times(times)
    surfaces = [actor for actor in sheet.scene.actors if isinstance(actor, Surface)]
    visible = torch.stack(
        [surface.grid.opacity.mean(-2)[..., 0] for surface in surfaces]
    )
    centers = torch.stack(
        [surface.grid.location.mean(-2)[..., 0] for surface in surfaces]
    )
    shown = centers.gather(0, visible.argmax(0, keepdim=True))[0]

    assert torch.equal(visible.sum(0), torch.ones_like(times))
    assert torch.allclose(shown, times, atol=1e-4)
    timeline.clear_buffers()


def thin_cylinder():
    """A synapse-shaped Cylinder: long, thin, and obliquely oriented."""
    cylinder = Cylinder(color=YELLOW).scale(0.02)
    cylinder.move_between_points(
        torch.tensor([-0.5, -0.25, 0.0]), torch.tensor([0.5, 0.25, 0.0])
    )
    return cylinder.spawn(animate=False)


def test_detached_history_preserves_the_surface_function():
    # Surface._change_resolution builds its surface function before calling
    # detach_history and evaluates it afterwards, and that evaluator re-reads
    # live Mob state (a Cylinder's coord_function is built from its basis). So
    # detach_history has to leave the transform alone: when a bug in the basis
    # setter let it inflate the basis instead, the refined grid came out scaled
    # by the inflation, while the fit's own residual stayed zero and reported
    # nothing wrong. Repeat the detach, because that failure amplified.
    cylinder = thin_cylinder()
    base_grid = cylinder.get_base_grid().clone()
    # coord_function normalizes its argument in place, so hand it a fresh copy.
    original_points = cylinder.coord_function_active(base_grid.clone())
    original_basis = cylinder.basis.clone()
    original_location = cylinder.location.clone()

    for _ in range(10):
        cylinder.detach_history()

    torch.testing.assert_close(cylinder.basis, original_basis, atol=1e-6, rtol=0)
    torch.testing.assert_close(cylinder.location, original_location, atol=1e-6, rtol=0)
    torch.testing.assert_close(
        cylinder.coord_function_active(base_grid.clone()),
        original_points,
        atol=1e-6,
        rtol=0,
    )


def test_repeated_waves_leave_a_thin_cylinder_the_size_it_was():
    # Refining and restoring each hand the surface to a detach_history clone,
    # which re-assigns its basis. That round trip used to amplify the basis's
    # float-noise shear, so a neural net's synapses -- thin Cylinders waved
    # once per activation -- grew until they stretched across the screen.
    synapse = thin_cylinder()
    points = synapse.grid.location.reshape(-1, 3)
    original_extent = points.amax(0) - points.amin(0)

    for _ in range(20):
        with Sync(run_time=1):
            synapse.wave_color(PURE_BLUE, wave_length=0.7)

    points = synapse.grid.location.reshape(-1, 3)
    torch.testing.assert_close(
        points.amax(0) - points.amin(0), original_extent, atol=1e-4, rtol=0
    )


def test_top_level_wave_restores_immediately_after_itself():
    sheet = flat_sheet()
    sheet.wave_color(PURE_BLUE, wave_length=0.5)
    assert (sheet.grid_width, sheet.grid_height) == (4, 4)

    timeline = sheet.scene.timeline_manager
    timeline.set_state_to_times(torch.linspace(0.0, 2.0, 9))
    assert sheet.grid.location.shape == (9, 16, 3)
    timeline.clear_buffers()
