"""``map_mob_over_inputs`` visits both ends of a sequence of inputs.

Run directly: .venv/bin/python -m pytest tests/unit_tests/test_map_mob_over_inputs.py -q

The un-animated jump from the first few inputs to the last few assigned the
input Mob itself to ``mob.location`` rather than its location (plus the mob's
offset), so every call died in ``cast_to_direction`` before reaching the end of
the sequence.

A feature test, so it sits outside the fast suite: it only breaks when this
helper itself changes.
"""

from algan import RIGHT, UP, Off, Scene, Square
from algan.utils.animation_utils import map_mob_over_inputs


def test_mob_visits_the_first_and_last_inputs_keeping_its_offset():
    with Scene():
        inputs = [Square().move_to(RIGHT * i) for i in range(10)]
        mob = Square().move_to(UP)
        with Off():
            for m in [*inputs, mob]:
                m.spawn()

        visited = []

        def record(m, inp):
            visited.append(
                (
                    inputs.index(inp),
                    [round(float(v), 4) for v in m.location.reshape(-1)[:2]],
                )
            )

        map_mob_over_inputs(mob, record, inputs, percent_shown=0.3)

        # First three inputs, a silent jump to inputs[-3], then the last two.
        assert visited == [
            (0, [0.0, 1.0]),
            (1, [1.0, 1.0]),
            (2, [2.0, 1.0]),
            (8, [8.0, 1.0]),
            (9, [9.0, 1.0]),
        ]
