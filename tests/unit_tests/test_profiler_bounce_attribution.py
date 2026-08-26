"""The per-bounce-iteration attribution added to the stage profiler.

The wavefront bounce loop labels its two kernels per iteration
(``wavefront:   - bounce <i> <phase>``) and records the rays entering each
iteration through ``stage(..., items=...)``; :func:`format_report` assembles
those rows into the bounce table. These tests pin the aggregation: item
totals accumulate per label, iterations past the cap land in one bucket, and
the table derives each iteration's continuation count from the next row's
rays-in rather than inventing one for the final iteration.
"""

import algan.utils.profiling_utils as pu
from algan.utils.profiling_utils import (
    TIMERS,
    _bounce_rows,
    _format_bounce_table,
    stage,
)


def _fresh_timers():
    timers = type(TIMERS)()
    timers.reset()
    return timers


def test_stage_items_accumulate_per_label():
    timers = _fresh_timers()
    for _ in range(3):
        with timers.stage("wavefront:   - bounce 0 shade", items=100):
            pass
    with timers.stage("wavefront:   - bounce 0 shade", items=50):
        pass
    assert timers.item_totals["wavefront:   - bounce 0 shade"] == 350
    assert timers.counts["wavefront:   - bounce 0 shade"] == 4


def test_stage_items_optional():
    timers = _fresh_timers()
    with timers.stage("plain"):
        pass
    assert "plain" not in timers.item_totals


def test_module_stage_is_null_without_hooks(monkeypatch):
    monkeypatch.setattr(pu, "_HOOKS_INSTALLED", False)
    TIMERS.reset()
    with stage("wavefront:   - bounce 0 shade", items=5):
        pass
    assert TIMERS.counts["wavefront:   - bounce 0 shade"] == 0
    assert "wavefront:   - bounce 0 shade" not in TIMERS.item_totals


def _res_with_bounces(per_iteration):
    """A fake run result whose stage rows describe ``per_iteration`` bounces.

    Each entry is ``(rays_in, shade_s, traverse_s)``; rays are recorded on
    both of an iteration's phases, as the drain does.
    """
    times, counts, items = {}, {}, {}
    for i, (rays, shade, trav) in enumerate(per_iteration):
        label = str(i) if i < 8 else "8+"
        for phase, secs in (("shade", shade), ("traverse", trav)):
            name = f"wavefront:   - bounce {label} {phase}"
            times[name] = times.get(name, 0.0) + secs
            counts[name] = counts.get(name, 0) + 1
            items[name] = items.get(name, 0) + rays
    return {"times": times, "counts": counts, "item_totals": items}


def test_bounce_rows_collect_phases_and_rays():
    res = _res_with_bounces([(1000, 2.0, 1.0), (400, 1.0, 0.5)])
    rows = _bounce_rows(res)
    # Both phases of an iteration carry the same rays-in; the row must report
    # it once.
    assert rows[0] == {
        "shade": 2.0,
        "traverse": 1.0,
        "shade_calls": 1,
        "traverse_calls": 1,
        "shade_rays": 1000,
        "traverse_rays": 1000,
        "rays": 1000,
    }
    assert list(rows) == [0, 1]


def test_capped_iterations_share_one_row_sorted_last():
    per = [(100, 0.1, 0.05)] * 10
    res = _res_with_bounces(per)
    rows = _bounce_rows(res)
    assert list(rows)[-1] == float("inf")
    assert rows[float("inf")]["shade_calls"] == 2  # iterations 8 and 9
    assert rows[7]["shade_calls"] == 1


def test_continuations_are_the_next_rows_rays():
    res = _res_with_bounces([(1000, 2.0, 1.0), (400, 1.0, 0.5), (50, 0.2, 0.1)])
    table = _format_bounce_table(res)
    lines = table.splitlines()
    data = [ln for ln in lines if ln.startswith("   ") and "bounce" not in ln]
    # Rows: bounce 0, 1, 2 then the total line -- continuations column of row
    # i must equal rays-in of row i+1; the last row shows "-" (unobserved).
    assert "1000" in data[0]
    assert "400" in data[0]
    assert "400" in data[1]
    assert "50" in data[1]
    assert "50" in data[2]
    assert data[2].rstrip().endswith("-")
    assert "total" in lines[-2]


def test_no_bounce_rows_no_table():
    assert _format_bounce_table({"times": {}, "counts": {}, "item_totals": {}}) == ""
