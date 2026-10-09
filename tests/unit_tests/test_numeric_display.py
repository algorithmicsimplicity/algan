from __future__ import annotations

import pytest
import torch

from algan import (
    RED,
    DecimalNumber,
    Off,
    Scene,
    Square,
    Sync,
    animated_function,
    easings,
)
from algan.mobs.bezier_circuit import BezierCircuitCubic


def _drawn(part):
    """Per-frame alpha the renderer draws ``part``'s glyphs with, shape ``(T,)``.

    Opacity times fill alpha: a DecimalNumber picks each slot's glyph through
    its color alpha and leaves ``opacity`` to fades, so neither factor alone
    says what is on screen.
    """
    values = [
        (circuit.opacity * circuit.fill_opacity).flatten(1).amax(1)
        for circuit in part.get_descendants(include_self=True)
        if isinstance(circuit, BezierCircuitCubic)
    ]
    return torch.stack(values).amax(0)


def _displayed_value(display, frame=None, threshold=0.5):
    def opacity(glyph):
        value = _drawn(glyph)
        return float((value if frame is None else value[frame]).max())

    if getattr(display, "significant_figures", None) is not None:
        return "".join(
            character
            for slot in display._significant_slots
            for character, glyph in zip(
                display._significant_characters, slot.character_mobs
            )
            if opacity(glyph) > threshold
        )

    digits = []
    for digit_mob in display.digit_mobs:
        visible = [
            digit
            for digit, glyph in enumerate(digit_mob.character_mobs)
            if opacity(glyph) > threshold
        ]
        digits.append("" if not visible else str(visible[0]))

    integer_digits = "".join(digits[: display.integer_places]) or "0"
    fractional_digits = "".join(digits[display.integer_places :])
    sign = "-" if opacity(display.negative_sign) > threshold else ""
    decimal = "." if display.decimal_places else ""
    return sign + integer_digits + decimal + fractional_digits


@pytest.mark.parametrize(
    ("initial", "target", "decimal_places", "expected"),
    [
        (0.0, 10000, 2, "10000.00"),
        (0.0, -12345.678, 2, "-12345.68"),
        (9.994, 9.999, 2, "10.00"),
        (1, 123456, 0, "123456"),
    ],
)
def test_numeric_display_grows_integer_slots(initial, target, decimal_places, expected):
    with Scene() as scene:
        display = DecimalNumber(initial, decimal_places=decimal_places).spawn(
            animate=False
        )

        display.value = target

        assert _displayed_value(display) == expected
        assert display.integer_places == len(expected.lstrip("-").split(".")[0])


def test_num_integer_places_is_a_minimum_not_a_limit():
    with Scene() as scene:
        display = DecimalNumber(7, decimal_places=1, integer_places=3).spawn(
            animate=False
        )

        assert display.integer_places == 3
        assert _displayed_value(display) == "7.0"

        display.value = 12345.6

        assert display.integer_places == 5
        assert _displayed_value(display) == "12345.6"


def test_grown_slots_replay_the_interpolated_value():
    with Scene() as scene:
        display = DecimalNumber(0.0, decimal_places=2).spawn(animate=False)
        with Sync(runtime=1, easing=easings.identity):
            display.value = 10000

        scene.timeline_manager.set_state_to_times(torch.tensor([0.0, 0.5, 1.0]))

        assert _displayed_value(display, 0) == "0.00"
        assert _displayed_value(display, 1) == "5000.00"
        assert _displayed_value(display, 2) == "10000.00"


def test_numeric_display_can_grow_more_than_once_and_then_shrink():
    with Scene() as scene:
        display = DecimalNumber(0, decimal_places=0).spawn(animate=False)

        display.value = 100
        display.value = 100000
        display.value = 7

        assert display.integer_places == 6
        assert _displayed_value(display) == "7"


def test_negative_sign_tracks_the_first_visible_integer_digit():
    with Scene() as scene:
        display = DecimalNumber(10000, decimal_places=2).spawn(animate=False)
        with Off():
            display.value = 0
        with Sync(runtime=1, easing=easings.identity):
            display.value = -100

        scene.timeline_manager.set_state_to_times(torch.tensor([0.5, 1.0]))

        for frame, visible_integer_places in enumerate((2, 3)):
            leading_digit = display.digit_mobs[
                display.integer_places - visible_integer_places
            ].character_mobs[0]
            torch.testing.assert_close(
                display.negative_sign.location[frame] - leading_digit.location[frame],
                display._negative_sign_offset.squeeze(0),
                atol=1e-6,
                rtol=0,
            )
            assert float(_drawn(display.negative_sign)[frame]) > 0.5


def _assert_hidden(display, frames):
    # Inspect geometry descendants too: the root's masked opacity alone cannot
    # catch a glyph write that re-enables the renderable rows during replay.
    for part in display.get_descendants():
        assert torch.count_nonzero(part.opacity[frames]) == 0


@pytest.mark.parametrize("animated", [False, True])
@pytest.mark.parametrize("spawn_later", [False, True])
@pytest.mark.parametrize("precision", [None, 3])
def test_value_setup_preserves_unspawned_state(animated, spawn_later, precision):
    with Scene() as scene:
        kwargs = {} if precision is None else {"significant_figures": precision}
        display = DecimalNumber(1, **kwargs)
        with Sync() if animated else Off():
            display.value = -123
        assert not display.is_spawned()
        assert all(not part.is_spawned() for part in display.get_descendants())
        scene.wait(1)
        if spawn_later:
            display.spawn(animate=False)
        scene.wait(1)

        scene.timeline_manager.set_state_to_times(torch.tensor([0.0, 0.5, 1.5]))
        _assert_hidden(display, [0, 1] if spawn_later else [0, 1, 2])
        if spawn_later:
            assert _displayed_value(display, 2) == (
                "-123.00" if precision is None else "-123"
            )


@pytest.mark.fast
@pytest.mark.parametrize("spawn_later", [False, True])
@pytest.mark.parametrize("driver", ["animation", "updater"])
@pytest.mark.parametrize("precision", [None, 3])
def test_replayed_value_changes_respect_later_spawn(driver, spawn_later, precision):
    with Scene() as scene:
        kwargs = {} if precision is None else {"significant_figures": precision}
        display = DecimalNumber(1, integer_places=3, **kwargs)
        clock = Square().spawn(animate=False)

        def update(_mob, t):
            display.value = -100 * t

        if driver == "updater":
            clock.add_updater(update)
        else:
            with Sync(runtime=2, easing=easings.identity):
                clock.animate_function(update, t=2)
            scene.animation_manager.context.rewind(2)
        scene.wait(1)
        if spawn_later:
            display.spawn(animate=False)
        scene.wait(1)

        # Repeat in smaller, nonchronological windows to catch seek/batch leaks.
        for times in ([0.0, 0.5, 1.5], [1.5], [0.5]):
            scene.timeline_manager.set_state_to_times(torch.tensor(times))
            hidden = [i for i, t in enumerate(times) if not spawn_later or t < 1]
            _assert_hidden(display, hidden)
            if spawn_later and 1.5 in times:
                assert _displayed_value(display, times.index(1.5)) == (
                    "-150.00" if precision is None else "-150"
                )
            scene.timeline_manager.clear_buffers()


@pytest.mark.parametrize("animated", [False, True])
@pytest.mark.parametrize("precision", [None, 3])
def test_value_changes_preserve_despawned_state(animated, precision):
    with Scene() as scene:
        kwargs = {} if precision is None else {"significant_figures": precision}
        display = DecimalNumber(1, **kwargs).spawn(animate=False)
        scene.wait(1)
        display.despawn(animate=False)
        with Sync(runtime=1) if animated else Off():
            display.value = -123
        scene.wait(1)
        assert display.is_despawned()

        scene.timeline_manager.set_state_to_times(torch.tensor([0.5, 1.0, 1.5, 2.0]))
        assert _displayed_value(display, 0) == "1.00"
        _assert_hidden(display, [1, 2, 3])


@pytest.mark.parametrize("animated", [False, True])
def test_value_changes_remain_visible(animated):
    with Scene() as scene:
        display = DecimalNumber(1).spawn(animate=False)
        scene.wait(1)
        with Sync(runtime=1, easing=easings.identity) if animated else Off():
            display.value = -123
        scene.wait(1)
        scene.timeline_manager.set_state_to_times(torch.tensor([0.5, 1.5, 2.5]))
        assert [_displayed_value(display, i) for i in range(3)] == [
            "1.00",
            "-61.00" if animated else "-123.00",
            "-123.00",
        ]


@animated_function(animated_args={"u": 0.0})
def _count_inside(number, u):
    number.value = 0.123 + 0.333 * u


@animated_function(animated_args={"u": 0.0})
def _fade_inside(number, u):
    number.opacity = 1 - u


@animated_function(animated_args={"u": 0.0})
def _count_and_fade_inside(number, u):
    number.value = 0.123 + 0.333 * u
    number.opacity = 1 - u


def _count_then_fade(count_inside_function):
    def author(number):
        if count_inside_function:
            _count_inside(number, 1.0)
        else:
            number.value = 0.456
        number.opacity = 0

    return author


# Variant -> (authoring, whether it counts 0.123 -> 0.456, opacity at time u).
# Letters are the issue report's; A (a recorded fade) worked before the fix.
_FADES = {
    "A_recorded": (lambda n: setattr(n, "opacity", 0), False, lambda u: 1 - u),
    "A_recorded_to_half": (
        lambda n: setattr(n, "opacity", 0.5),
        False,
        lambda u: 1 - u / 2,
    ),
    "D_in_function": (lambda n: _fade_inside(n, 1.0), False, lambda u: 1 - u),
    "E_counted": (_count_then_fade(False), True, lambda u: 1 - u),
    "F_counted_in_function": (_count_then_fade(True), True, lambda u: 1 - u),
    "G_one_function": (
        lambda n: _count_and_fade_inside(n, 1.0),
        True,
        lambda u: 1 - u,
    ),
    "via_color": (lambda n: n.set_opacity_via_color(0), False, lambda u: 1 - u),
}


@pytest.mark.parametrize("variant", sorted(_FADES))
@pytest.mark.parametrize("precision", [None, 3])
def test_fades_draw_only_the_selected_glyphs(variant, precision):
    # Glyph selection used to live in opacity: a fade written inside an
    # animated function showed every glyph of every slot ("-8.888"), and a
    # count replayed during a recorded fade re-showed changed digits at full
    # opacity. Selection is now color alpha, multiplied by opacity to draw.
    author, counts, opacity_at = _FADES[variant]
    if precision is None:
        kwargs = {"decimal_places": 3}
    else:
        kwargs = {"significant_figures": precision}
    with Scene() as scene:
        display = DecimalNumber(0.123, **kwargs).spawn(animate=False)
        with Sync(runtime=1, easing=easings.identity):
            author(display)
        scene.wait(1)
        # Clear of rounding ties while counting: 0.223, 0.323 and 0.423.
        times = [0.3, 0.6, 0.9, 1.5]
        scene.timeline_manager.set_state_to_times(torch.tensor(times))
        for frame, t in enumerate(times):
            u = min(t, 1.0)
            opacity = opacity_at(u)
            if opacity > 0:
                value = 0.123 + 0.333 * u if counts else 0.123
                shown = _displayed_value(display, frame, threshold=1e-4)
                assert shown == f"{value:.3f}"
            # Each glyph is drawn at the fade's opacity or not at all.
            for part in display.get_descendants():
                if not isinstance(part, BezierCircuitCubic):
                    continue
                drawn = (part.opacity * part.fill_opacity)[frame].flatten()
                hidden = drawn.abs() < 1e-5
                faded = (drawn - opacity).abs() < 1e-5
                assert bool((hidden | faded).all()), (t, drawn)


def test_recoloring_keeps_hidden_glyphs_hidden():
    with Scene() as scene:
        display = DecimalNumber(0.123, decimal_places=3).spawn(animate=False)
        with Sync(runtime=1, easing=easings.identity):
            display.value = 0.456
            display.color = RED
        scene.wait(1)
        scene.timeline_manager.set_state_to_times(torch.tensor([0.3, 1.5]))
        assert _displayed_value(display, 0, threshold=1e-4) == "0.223"
        assert _displayed_value(display, 1, threshold=1e-4) == "0.456"


def test_slots_grown_during_a_fade_fade_with_it():
    with Scene() as scene:
        display = DecimalNumber(1, decimal_places=0).spawn(animate=False)
        with Sync(runtime=1, easing=easings.identity):
            display.value = 1000
            display.opacity = 0
        scene.timeline_manager.set_state_to_times(torch.tensor([0.75]))
        drawn = [
            float((part.opacity * part.fill_opacity).max())
            for part in display.get_descendants()
            if isinstance(part, BezierCircuitCubic)
        ]
        assert _displayed_value(display, 0, threshold=1e-4) == "750"
        assert max(drawn) == pytest.approx(0.25, abs=1e-5)


@pytest.mark.parametrize(
    ("value", "precision", "expected"),
    [
        (0, 1, "0"),
        (0, 3, "0.00"),
        (-0.0, 3, "0.00"),
        (1.25, 2, "1.2"),  # Exact ties round to even, in both directions.
        (1.75, 2, "1.8"),
        (-1.25, 2, "-1.2"),
        (-1.75, 2, "-1.8"),
        (12, 4, "12.00"),
        (123, 3, "123"),
        (999.75, 3, "1.00e+03"),
        (0.00125, 3, "0.00125"),
        (0.0001, 3, "0.000100"),
        (0.00001, 3, "1.00e-05"),
        (-0.00001, 3, "-1.00e-05"),
        (1e30, 3, "1.00e+30"),
        (1e-30, 3, "1.00e-30"),
    ],
)
def test_significant_figures_formatting(value, precision, expected):
    with Scene():
        display = DecimalNumber(value, significant_figures=precision).spawn(
            animate=False
        )
        assert _displayed_value(display) == expected


@pytest.mark.parametrize("precision", [0, -1])
def test_nonpositive_significant_figures_are_rejected(precision):
    with pytest.raises(ValueError, match="significant_figures"):
        DecimalNumber(1, significant_figures=precision)


@pytest.mark.parametrize(
    "precision", [True, False, 2.0, 2.5, "3", float("inf"), float("nan")]
)
def test_noninteger_significant_figures_are_rejected(precision):
    with pytest.raises(TypeError, match="significant_figures"):
        DecimalNumber(1, significant_figures=precision)


def test_significant_figures_override_decimal_place_layout():
    with Scene():
        display = DecimalNumber(
            12, decimal_places=0, integer_places=8, significant_figures=4
        )
        assert _displayed_value(display) == "12.00"


@pytest.mark.parametrize("value", [float("inf"), -float("inf"), float("nan")])
def test_significant_figures_require_finite_values(value):
    from algan.errors import AlganConfigurationError

    with Scene():
        with pytest.raises(AlganConfigurationError, match="value must be finite"):
            DecimalNumber(value, significant_figures=3)
        display = DecimalNumber(1, significant_figures=3)
        with pytest.raises(AlganConfigurationError, match="value must be finite"):
            display.value = value
        assert _displayed_value(display) == "1.00"


@pytest.mark.parametrize(
    ("initial", "target", "expected"),
    [
        (0, 2000, ["0.00", "500", "1.50e+03", "2.00e+03"]),
        (-0.0002, 0.0002, ["-0.000200", "-0.000100", "0.000100", "0.000200"]),
    ],
)
def test_significant_figures_follow_interpolated_values(initial, target, expected):
    with Scene() as scene:
        display = DecimalNumber(initial, significant_figures=3).spawn(animate=False)
        with Sync(runtime=1, easing=easings.identity):
            display.value = target
        parts = tuple(display.get_descendants())
        times = [0.0, 0.25, 0.75, 1.0]
        scene.timeline_manager.set_state_to_times(torch.tensor(times))
        assert [_displayed_value(display, i) for i in range(4)] == expected
        scene.timeline_manager.clear_buffers()
        for i in (2, 1, 3, 0):
            scene.timeline_manager.set_state_to_times(torch.tensor([times[i]]))
            assert _displayed_value(display, 0) == expected[i]
            assert tuple(display.get_descendants()) == parts
            scene.timeline_manager.clear_buffers()


def test_significant_figures_immediate_updates_and_zero():
    with Scene() as scene:
        display = DecimalNumber(1, significant_figures=3).spawn(animate=False)
        scene.wait(1)
        with Off():
            display.value = -0.00001
        scene.wait(1)
        with Off():
            display.value = 0
        scene.wait(1)
        scene.timeline_manager.set_state_to_times(torch.tensor([0.5, 1.5, 2.5]))
        assert [_displayed_value(display, i) for i in range(3)] == [
            "1.00",
            "-1.00e-05",
            "0.00",
        ]


def test_significant_figures_round_the_stored_value_for_tensor_inputs():
    with Scene() as scene:
        # Both inputs become exact halfway values in the scene's float32
        # storage. Formatting the original float64 would round the other way.
        display = DecimalNumber(
            torch.tensor(1.25000002, dtype=torch.float64), significant_figures=2
        ).spawn(animate=False)
        assert _displayed_value(display) == "1.2"
        scene.wait(1)
        display.value = torch.tensor(1.74999998, dtype=torch.float64)
        assert _displayed_value(display) == "1.8"
        scene.timeline_manager.set_state_to_times(torch.tensor([0.5, 1.5, 2.0]))
        assert [_displayed_value(display, i) for i in range(3)] == ["1.2", "1.5", "1.8"]
