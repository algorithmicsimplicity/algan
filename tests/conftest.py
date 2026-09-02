"""Shared fixtures and options for the Algan test suites.

Five things live here because they are cross-cutting:

* the ``fast`` marker and ``--fast``, which together define the fast suite:
  the tests marked ``fast`` and nothing else. See ``tests/README.md``;
* live ``pytest.log`` piping and per-test wall times, for every suite that a
  run actually collects from;
* a wall-clock report against the fast suite's budget, so the suite cannot
  creep past it unnoticed;
* the frame-by-frame video comparison, shared by ``tests/fast/`` and
  ``tests/full_renders/`` so the two cannot drift apart on tolerance;
* per-test isolation of the two pieces of process-global state Algan owns --
  ``SETTINGS`` and the active-Scene stack. Leaking either between tests makes
  failures depend on test order, which is the hardest kind of flake to chase.
"""

from __future__ import annotations

import atexit
import contextlib
import re
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from algan import SETTINGS
from algan.scene_manager import SceneManager

# The render scenes name this family explicitly instead of taking Pango's
# default, because ``Text`` resolves ``font=""`` through fontconfig and the
# glyph advances then change with whatever the machine happens to have
# installed. That is not a hypothetical: the CPU and CUDA baselines were
# rendered on different machines and their Text differed by up to 230 channel
# values -- structurally, not by a sub-pixel shift -- while the geometry and the
# dvisvgm-backed Tex agreed to a mean of 0.36.
#
# Registering the vendored faces makes text depend on bytes in the repository
# rather than on the host, so a container image that changes its fonts can no
# longer look like a renderer regression. See tests/assets/fonts/LICENSE.txt.
FONT_DIR = Path(__file__).resolve().parent / "assets" / "fonts"
TEST_FONT = "Algan Test Sans"


def _register_test_fonts():
    """Make the vendored faces visible to Pango for this process."""
    import manimpango

    for face in sorted(FONT_DIR.glob("*.ttf")):
        if not manimpango.register_font(str(face)):
            raise RuntimeError(f"could not register the vendored font {face}")
    if TEST_FONT not in manimpango.list_fonts():
        raise RuntimeError(
            f"registered {FONT_DIR} but Pango still does not offer {TEST_FONT!r}; "
            "the render scenes would silently fall back to a substitute font"
        )


_register_test_fonts()

# The fast suite is meant to stay inside a development loop. This is reported,
# not enforced: the number moves with machine load and thermal state, and a
# timing-based failure would be a flake. A run that exceeds it means the
# curation has drifted -- take a marker off something, rather than raising the
# number reflexively.
#
# The history is worth knowing. The budget was 120 s, then 150 s, while the fast
# suite was *everything not marked slow* and grew with every test anyone added.
# Curating it by hand (only ``fast``-marked tests run) took it from 910 of the
# 1038 collected tests to 191, of which 190 are behavioural and one is the
# render.
# The render is now most of the cost -- it was measured at ~50 s of the old
# 112-147 s suite on CUDA -- so the budget is set to leave it room to pay a
# kernel compile. Raising it again is a deliberate trade of loop time for
# coverage; the point of the number is that growth has to be a decision.
FAST_SUITE_BUDGET_SECONDS = 75.0

# Small per-pixel drift is expected and tolerated: torch CPU rate-function
# evaluation rounds differently depending on the materialization window, so
# byte-identity across re-windowed state is unattainable.
MAX_CHANNEL_DIFFERENCE = 2


# ---------------------------------------------------------------------------
# Live log piping, one copy for all three suites
# ---------------------------------------------------------------------------
# Each suite directory gets a ``pytest.log`` holding everything the terminal
# showed, plus a wall time per test. Written and flushed as the run happens,
# not at the end: a render suite takes minutes and the whole point is watching
# it move. This used to be two byte-identical copies inside the render test
# modules, opened from a session fixture -- which was too late to catch the
# progress line, and left ``tests/unit_tests`` with no log at all.
TESTS_ROOT = Path(__file__).resolve().parent
SUITE_DIRS = ("fast", "full_renders", "unit_tests")

#: Open log handles for this run; the timing hooks write straight to these so
#: the per-test lines land in the file without also cluttering the terminal.
_LOG_FILES: list[Any] = []

#: Whether the log is currently at the start of a line. The terminal writes
#: progress a character at a time ("....") with no newline, so a timing line
#: appended blind lands mid-dots as ``.[   0.75s] PASSED ...``. Tracked across
#: both writers so the timing lines can break to a fresh line first.
_AT_LINE_START = True

#: Every tee wrapper installed this run, so a log handle can be detached from
#: all of them before it is closed.
_TEE_STREAMS: list[Any] = []


class _TeeStream:
    """A stream wrapper that writes to an underlying stream and registered log files."""

    def __init__(self, original_stream: Any) -> None:
        self._orig = original_stream
        self._log_files: list[Any] = []

    def add_file(self, file_obj: Any) -> None:
        if file_obj not in self._log_files:
            self._log_files.append(file_obj)

    def write(self, s: str) -> int:
        global _AT_LINE_START
        res = self._orig.write(s)
        if self._log_files:
            clean = re.sub("\\x1b\\[[0-9;]*[a-zA-Z]", "", s)
            for f in list(self._log_files):
                with contextlib.suppress(Exception):
                    f.write(clean)
                    f.flush()
            if clean:
                _AT_LINE_START = clean.endswith("\n")
        return res

    def flush(self) -> None:
        self._orig.flush()
        for f in list(self._log_files):
            with contextlib.suppress(Exception):
                f.flush()

    def isatty(self) -> bool:
        return getattr(self._orig, "isatty", lambda: False)()

    def fileno(self) -> int:
        return self._orig.fileno()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._orig, name)


def _setup_log_piping(log_path: Path, config: Any = None) -> Any:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w", encoding="utf-8")  # noqa: SIM115

    if not hasattr(sys.stdout, "add_file"):
        sys.stdout = _TeeStream(sys.stdout)
    if not hasattr(sys.stderr, "add_file"):
        sys.stderr = _TeeStream(sys.stderr)

    tees = [sys.stdout, sys.stderr]

    if config is not None:
        tr = config.pluginmanager.get_plugin("terminalreporter")
        if tr and hasattr(tr, "_tw"):
            if not hasattr(tr._tw._file, "add_file"):
                tr._tw._file = _TeeStream(tr._tw._file)
            tees.append(tr._tw._file)

    for tee in tees:
        tee.add_file(log_file)
        if tee not in _TEE_STREAMS:
            _TEE_STREAMS.append(tee)

    # Closed for real in ``pytest_unconfigure``; this is only the backstop for
    # a run that dies before pytest gets to unconfigure.
    atexit.register(_close_log_file, log_file)
    return log_file


def _close_log_file(log_file: Any) -> None:
    """Detach a log handle from every tee, then close it once."""
    if log_file.closed:
        return
    for tee in _TEE_STREAMS:
        if log_file in tee._log_files:
            tee._log_files.remove(log_file)
    with contextlib.suppress(Exception):
        log_file.flush()
    log_file.close()


def pytest_unconfigure(config):
    """Close the logs deterministically, at the end of the run.

    Leaving this to ``atexit`` alone raced the interpreter's own teardown and
    the handles were reported as ``ResourceWarning: unclosed file`` on the way
    out. The terminal writer's tee keeps its reference either way, hence the
    detach in ``_close_log_file``.
    """
    while _LOG_FILES:
        _close_log_file(_LOG_FILES.pop())


def _write_to_logs(text: str) -> None:
    """Write straight to the log files, bypassing the terminal.

    Breaks to a fresh line first when the terminal has left a partial one, so
    a timing line never lands in the middle of the progress dots.
    """
    global _AT_LINE_START
    if not _AT_LINE_START:
        text = "\n" + text
    for f in list(_LOG_FILES):
        with contextlib.suppress(Exception):
            f.write(text)
            f.flush()
    if text:
        _AT_LINE_START = text.endswith("\n")


def _collected_suite_dirs(items) -> list[Path]:
    """The suite directories this run actually has tests in.

    Derived from the selected items rather than from ``config.args`` so that a
    ``--fast`` run, which collects all three directories but selects from two,
    does not leave a stale empty log in the third.
    """
    dirs = set()
    for item in items:
        with contextlib.suppress(ValueError, TypeError):
            rel = Path(str(item.fspath)).resolve().relative_to(TESTS_ROOT)
            if rel.parts and rel.parts[0] in SUITE_DIRS:
                dirs.add(TESTS_ROOT / rel.parts[0])
    return sorted(dirs)


def pytest_collection_finish(session):
    """Open a live log per collected suite, before the first test runs."""
    if _LOG_FILES:
        return
    for suite_dir in _collected_suite_dirs(session.items):
        _LOG_FILES.append(_setup_log_piping(suite_dir / "pytest.log", session.config))
    if not _LOG_FILES:
        return
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    _write_to_logs(
        f"# algan test run started {stamp} -- {len(session.items)} tests collected\n"
        f"# columns: [wall time] OUTCOME test id\n"
    )


# nodeid -> seconds accumulated across setup/call/teardown, and the outcome to
# report. A test's wall time is all three phases: an expensive fixture is part
# of what the run costs, and attributing only ``call`` hides it.
_PHASE_SECONDS: dict[str, float] = {}
_OUTCOMES: dict[str, str] = {}
#: (seconds, nodeid) for the slowest-tests table appended at the end.
_DURATIONS: list[tuple[float, str]] = []


def pytest_runtest_logreport(report):
    """Accumulate each test's wall time and log one line when it finishes."""
    if not _LOG_FILES:
        return
    nodeid = report.nodeid
    _PHASE_SECONDS[nodeid] = _PHASE_SECONDS.get(nodeid, 0.0) + report.duration
    if report.when == "call":
        _OUTCOMES[nodeid] = report.outcome.upper()
    elif report.failed:
        # A failure outside ``call`` is a broken fixture, not a failed test.
        _OUTCOMES[nodeid] = f"ERROR({report.when})"
    elif report.when == "setup" and report.skipped:
        _OUTCOMES[nodeid] = "SKIPPED"
    if report.when != "teardown":
        return
    seconds = _PHASE_SECONDS.pop(nodeid, 0.0)
    outcome = _OUTCOMES.pop(nodeid, "UNKNOWN")
    _DURATIONS.append((seconds, nodeid))
    _write_to_logs(f"[{seconds:8.2f}s] {outcome:<14} {nodeid}\n")


def _write_slowest_table(limit: int = 25) -> None:
    if not _LOG_FILES or not _DURATIONS:
        return
    total = sum(seconds for seconds, _ in _DURATIONS)
    lines = [
        "",
        f"=== slowest {min(limit, len(_DURATIONS))} of {len(_DURATIONS)} tests "
        f"({total:.1f}s in tests, excluding collection and teardown of the run) ===",
    ]
    for seconds, nodeid in sorted(_DURATIONS, reverse=True)[:limit]:
        share = seconds / total if total else 0.0
        lines.append(f"[{seconds:8.2f}s] {share:5.1%}  {nodeid}")
    _write_to_logs("\n".join(lines) + "\n")


def pytest_addoption(parser):
    parser.addoption(
        "--fast",
        action="store_true",
        default=False,
        dest="fast",
        help=(
            "Run only the fast suite: the tests marked 'fast', which are a "
            "hand-picked set covering the mechanisms every change routes "
            "through. Everything else is deselected. See tests/README.md."
        ),
    )


def pytest_collection_modifyitems(config, items):
    """Reduce the run to the ``fast`` marker when ``--fast`` is passed.

    Deselected rather than skipped, deliberately: the fast suite excludes most
    of the suite by design, and hundreds of ``s`` characters would bury the
    handful of skips that actually mean something (a missing baseline, an
    absent optional dependency).
    """
    if not config.getoption("fast"):
        return
    selected, deselected = [], []
    for item in items:
        target = selected if item.get_closest_marker("fast") else deselected
        target.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected


@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    config._algan_fast_started = time.perf_counter()


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Append the slowest-tests table to the logs, and report the fast budget.

    The table goes to the log only. The terminal already has ``--durations``
    for anyone who wants it there, and the fast suite's budget line is what CI
    reads off the terminal.
    """
    _write_slowest_table()
    if not config.getoption("fast"):
        return
    started = getattr(config, "_algan_fast_started", None)
    if started is None:
        return
    elapsed = time.perf_counter() - started
    share = elapsed / FAST_SUITE_BUDGET_SECONDS
    line = (
        f"fast suite: {elapsed:.0f}s of its {FAST_SUITE_BUDGET_SECONDS:.0f}s budget "
        f"({share:.0%})"
    )
    if elapsed > FAST_SUITE_BUDGET_SECONDS:
        terminalreporter.write_line(
            line + " -- over budget; take 'fast' off the newest test in it",
            yellow=True,
        )
    else:
        terminalreporter.write_line(line)


@pytest.fixture(autouse=True)
def _isolate_global_settings():
    """Undo any settings a test wrote, however it exited.

    ``SETTINGS`` sections keep stable identity, so ``restore`` writes values
    back into the same objects the engine already holds references to.
    """
    snapshot = SETTINGS.snapshot()
    try:
        yield
    finally:
        SETTINGS.restore(snapshot)


@pytest.fixture
def fresh_scene():
    """A pristine active-Scene stack, torn down again afterwards.

    Prefer ``with Scene() as scene:`` inside a test when you need a handle on
    the Scene; use this when the code under test resolves the *current* Scene
    itself.
    """
    SceneManager.reset()
    try:
        yield SceneManager.instance().current_scene
    finally:
        SceneManager.reset()


@pytest.fixture
def assert_video_matches_baseline():
    """Compare a rendered video to its baseline frame by frame.

    Shared by both render suites so they cannot drift apart on tolerance. A
    frame that exceeds ``MAX_CHANNEL_DIFFERENCE`` on any channel is written to
    ``diff_path`` as a difference video before the assertion fires.
    """
    import cv2
    import numpy as np

    from algan import get_file_writer

    def compare(actual_path, expected_path, diff_path, fallback_fps=10):
        actual = cv2.VideoCapture(str(actual_path))
        expected = cv2.VideoCapture(str(expected_path))
        expected_fps = expected.get(cv2.CAP_PROP_FPS) or fallback_fps
        writer = None
        frame_count = 0
        max_difference = 0
        worst_frame = -1

        try:
            while True:
                actual_ok, actual_frame = actual.read()
                expected_ok, expected_frame = expected.read()
                if not actual_ok or not expected_ok:
                    assert actual_ok == expected_ok, (
                        f"{actual_path.name} has a different frame count from its "
                        f"baseline (diverged at frame {frame_count})"
                    )
                    break

                assert actual_frame.shape == expected_frame.shape, (
                    f"{actual_path.name} rendered at {actual_frame.shape}, expected "
                    f"{expected_frame.shape}"
                )
                difference = np.abs(
                    actual_frame.astype(np.int16) - expected_frame.astype(np.int16)
                ).astype(np.uint8)
                frame_difference = int(difference.max())
                if frame_difference > max_difference:
                    max_difference = frame_difference
                    worst_frame = frame_count
                frame_count += 1

                if frame_difference > MAX_CHANNEL_DIFFERENCE:
                    if writer is None:
                        diff_path.parent.mkdir(parents=True, exist_ok=True)
                        height, width = difference.shape[:2]
                        writer = get_file_writer(
                            str(diff_path),
                            (width, height),
                            codec="libx264rgb",
                            fps=expected_fps,
                            with_mask=False,
                            ffmpeg_params=["-crf", "0", "-preset", "fast"],
                            audiofile=None,
                            audio_codec=None,
                        )
                    writer.write_frame(difference[..., ::-1])
        finally:
            actual.release()
            expected.release()
            if writer is not None:
                writer.close()

        assert frame_count > 0, f"{actual_path.name} did not contain any frames"
        assert max_difference <= MAX_CHANNEL_DIFFERENCE, (
            f"{actual_path.name} differs from its baseline by up to "
            f"{max_difference} channel values (worst at frame {worst_frame}); "
            f"see {diff_path}"
        )

    return compare


@pytest.fixture
def wide_kernel_arms():
    """Skip a narrow-vs-wide kernel comparison where the wide arm cannot exist.

    Several tests are A/B: they launch the float64 / int64 kernel arm and
    assert the narrowed one answers it, or check a kernel against a torch
    reference computed at the wide dtype. That needs a device that HAS the wide
    arm. Metal has no f64 at all and no int64 atomic
    (``DESIGN_mps_support.md`` §1.2), so on an MPS render device the wide
    launch does not answer worse -- it fails to compile
    (``Type f64 not supported``) or aborts inside Taichi, which is precisely
    what MPS-friendly mode exists to avoid and not a regression to report.

    Skipping loses no coverage that MPS depends on. The narrow arm is what an
    Apple GPU runs; the comparison that validates it against the wide one runs
    on every CPU and CUDA machine, which is where the wide arm is available to
    compare against.
    """
    if str(SETTINGS.computing.render_device) == "mps":
        pytest.skip(
            "the float64 / int64 kernel arm cannot compile on Metal, so there "
            "is nothing here to compare against; the comparison runs on CPU "
            "and CUDA"
        )
