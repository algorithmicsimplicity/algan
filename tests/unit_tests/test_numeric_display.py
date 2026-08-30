from __future__ import annotations

import pytest
import torch

from algan import DecimalNumber, Off, Scene, Sync, easings


def _displayed_value(display, frame=None):
    def opacity(glyph):
        value = glyph.opacity if frame is None else glyph.opacity[frame]
        return float(value.max())

    digits = []
    for digit_mob in display.digit_mobs:
        visible = [
            digit
            for digit, glyph in enumerate(digit_mob.character_mobs)
            if opacity(glyph) > 0.5
        ]
        digits.append("" if not visible else str(visible[0]))

    integer_digits = "".join(digits[: display.integer_places]) or "0"
    fractional_digits = "".join(digits[display.integer_places :])
    sign = "-" if opacity(display.negative_sign) > 0.5 else ""
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
        with Sync(duration=1, easing=easings.identity):
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
        with Sync(duration=1, easing=easings.identity):
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
            assert any(
                float(part.opacity[frame].max()) > 0.5
                for part in display.negative_sign.get_descendants()
            )
