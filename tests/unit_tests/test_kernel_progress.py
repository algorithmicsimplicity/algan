"""What a render says while it waits for kernels (``algan/rendering/kernel_progress.py``).

The account replaced a single "several minutes" line that then said nothing
for as long as the wait lasted. Held here without compiling anything: one
header per render, a numbered line per real compile and none for a kernel the
offline cache served, an announced long compile resolved either way, worker
waits counted, a summary only when something was compiled, and a header seen
while authoring not repeated by the render that follows.
"""

from __future__ import annotations

import logging

import pytest

from algan.rendering import kernel_progress as kp


@pytest.fixture
def lines(monkeypatch):
    """Capture what the account logs, fresh for each test."""
    captured = []
    monkeypatch.setattr(kp, "_STATE", kp._State())
    monkeypatch.setattr(kp, "_device_phrase", lambda: "the CPU")
    monkeypatch.setattr(kp, "_pool_suffix", lambda: "")

    class Handler(logging.Handler):
        def emit(self, record):
            captured.append(record.getMessage())

    handler = Handler()
    logger = logging.getLogger("algan")
    logger.addHandler(handler)
    previous = logger.level
    logger.setLevel(logging.INFO)
    yield captured
    logger.removeHandler(handler)
    logger.setLevel(previous)


class _Kernel:
    pass


NAME = "algan.rendering.raytracing.sheet_resolve_taichi.sheet_resolve_shade_arena[specialization=0, autodiff=NONE]"


def test_one_header_per_render_and_a_line_per_real_compile(lines):
    kp.render_started()
    first, second, served = _Kernel(), _Kernel(), _Kernel()
    kp.index_missed()
    kp.index_missed()
    kp.compiled(first, NAME, 32.0, cold=True)
    kp.compiled(served, NAME, 0.4, cold=False)
    kp.compiled(second, "m.tonemap_to_u8[specialization=0]", 0.1, cold=True)
    kp.render_finished()
    headers = [line for line in lines if line.startswith("Preparing render kernels")]
    assert len(headers) == 1
    assert "for the CPU" in headers[0]
    assert "algan warmup" in headers[0]
    assert any(
        line.startswith("  kernel 1: sheet_resolve_shade_arena compiled in 32.0 s")
        for line in lines
    )
    assert any(line.startswith("  kernel 2: tonemap_to_u8") for line in lines)
    assert not any("0.4 s" in line for line in lines)
    assert lines[-1].startswith("Render kernels ready: 2 compiled here")


def test_a_render_that_compiles_nothing_says_nothing(lines):
    kp.render_started()
    kernel = _Kernel()
    kp.materializing(kernel, NAME)
    kp.materialized(kernel)
    kp.compiled(_Kernel(), NAME, 0.2, cold=False)
    kp.render_finished()
    assert lines == []


def test_a_long_compile_is_announced_and_resolved_either_way(lines):
    kp.render_started()
    cold, served, quick = _Kernel(), _Kernel(), _Kernel()
    kp.backend_starting(cold, NAME, frontend_seconds=12.0, expected=None)
    kp.compiled(cold, NAME, 30.0, cold=True)
    kp.backend_starting(served, NAME, frontend_seconds=0.5, expected=35.0)
    kp.compiled(served, NAME, 0.3, cold=False)
    kp.backend_starting(quick, NAME, frontend_seconds=0.1, expected=1.0)
    assert sum("compiling sheet_resolve_shade_arena" in line for line in lines) == 2
    assert any("about 35.0 s last time" in line for line in lines)
    assert any("loaded from the kernel cache" in line for line in lines)


def test_worker_waits_are_numbered_with_the_compiles(lines):
    kp.render_started()
    kp.compiled(_Kernel(), NAME, 3.0, cold=True)

    class Job:
        name = "wavefront_shade_arena"

    kp.waited_for_worker(Job(), 4.5, ready=True)
    kp.waited_for_worker(Job(), 1.0, ready=False)
    kp.render_finished()
    assert any(
        line.startswith(
            "  kernel 2: wavefront_shade_arena compiled by a background worker"
        )
        for line in lines
    )
    assert "1 from background workers" in lines[-1]


def test_a_header_seen_while_authoring_is_not_repeated_by_the_render(lines):
    kp.index_missed()
    kp.render_started()
    kp.index_missed()
    kp.render_finished()
    assert sum(line.startswith("Preparing render kernels") for line in lines) == 1
    # ... but the next render that compiles says it again.
    kp.render_started()
    kp.index_missed()
    assert sum(line.startswith("Preparing render kernels") for line in lines) == 2
