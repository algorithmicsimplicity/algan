"""Algan's logger: the level threshold, the PERF level's contract, and the
render-progress style.

``PERF`` exists so the renderer's self-healing events -- the batch splits and
pool retries it performs when a chunk does not fit -- have somewhere to go that
is off by default. They were ``WARNING`` before, which made a healthy render
look broken; worse, ``WARNING`` is what a user sets to quiet the progress
output, so those messages were the *only* thing such a user saw.
"""

from __future__ import annotations

import ast
import io
import logging
import sys
from pathlib import Path

import pytest

from algan.logging.logger import (
    PERF,
    PROGRESS_STYLES,
    get_logger,
    get_progress_style,
    logger,
    resolve_progress_style,
    set_log_level,
    set_progress_style,
)

ALGAN_ROOT = Path(__file__).resolve().parents[2] / "algan"

#: Modules whose retry paths must stay off the WARNING channel.
_SELF_HEALING_MODULES = (
    ALGAN_ROOT / "render_loop.py",
    ALGAN_ROOT / "rendering" / "raytracing" / "tracer.py",
)

#: Words that mark a message as describing recovery rather than a fault.
_RECOVERY_WORDS = ("retry", "retrying", "splitting", "did not fit", "overflowed")


@pytest.fixture(autouse=True)
def _restore_log_level():
    """A test that changes the level must not leak it into the next one."""
    before = logger.level
    try:
        yield
    finally:
        logger.setLevel(before)


def test_perf_sits_between_debug_and_info():
    """Below INFO so it is off by default; above DEBUG so it is not all-or-nothing."""
    assert logging.DEBUG < PERF < logging.INFO


def test_perf_level_name_round_trips():
    """``addLevelName`` must register both directions.

    ``set_log_level("PERF")`` and ``ALGAN_LOG_LEVEL=PERF`` both go through
    ``Logger.setLevel`` with a string, which resolves via the name table. If only
    the number-to-name direction were registered, both would raise.
    """
    assert logging.getLevelName(PERF) == "PERF"
    assert logging.getLevelName("PERF") == PERF


@pytest.mark.parametrize(
    ("setting", "perf_visible", "info_visible", "debug_visible"),
    [
        ("INFO", False, True, False),  # the default: PERF is hidden
        ("PERF", True, True, False),  # opt in without the rest of DEBUG
        ("DEBUG", True, True, True),  # DEBUG is below PERF, so it includes it
        ("WARNING", False, False, False),  # quieting the console quiets PERF too
    ],
)
def test_level_thresholds(setting, perf_visible, info_visible, debug_visible):
    """Levels are thresholds: each shows itself and everything more severe."""
    set_log_level(setting)
    child = get_logger("raytracing")  # engine modules log through children
    assert child.isEnabledFor(PERF) is perf_visible
    assert child.isEnabledFor(logging.INFO) is info_visible
    assert child.isEnabledFor(logging.DEBUG) is debug_visible


def test_child_loggers_defer_to_the_root_level():
    """One ``set_log_level`` call has to govern every module.

    Children are created with NOTSET so they inherit; if one ever set its own
    level, the single knob would silently stop working for that module.
    """
    for name in ("raytracing", "scene", "memory_model"):
        assert get_logger(name).level == logging.NOTSET


@pytest.mark.parametrize(
    "module_path", _SELF_HEALING_MODULES, ids=lambda path: path.name
)
def test_recovery_messages_are_not_warnings(module_path):
    """Self-healing events must not be logged at WARNING.

    WARNING should mean "you may need to act". Spending it on an event the
    engine recovers from by itself trains people to ignore the level that
    matters, and it is what made a working render read as a broken one.
    """
    tree = ast.parse(module_path.read_text(encoding="utf-8"))

    offenders = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "warning"
        ):
            continue
        # Collect the literal text of the message, across f-strings and implicit
        # concatenation, so the check sees the whole sentence.
        text = " ".join(
            part.value.lower()
            for argument in node.args
            for part in ast.walk(argument)
            if isinstance(part, ast.Constant) and isinstance(part.value, str)
        )
        if any(word in text for word in _RECOVERY_WORDS):
            offenders.append((node.lineno, text[:70]))

    assert not offenders, (
        f"{module_path.name} logs recovery at WARNING: {offenders}; "
        "use logger.log(PERF, ...) so a healthy render stays quiet"
    )


class _FakeTerminal(io.StringIO):
    def isatty(self):
        return True


@pytest.fixture
def stderr_console(monkeypatch):
    """Replace ``sys.stderr`` with a terminal or a pipe, and the sniffed env."""

    def install(*, terminal, **environment):
        for name in ("CI", "PYTEST_CURRENT_TEST", "PYCHARM_HOSTED"):
            monkeypatch.delenv(name, raising=False)
        for name, value in environment.items():
            monkeypatch.setenv(name, value)
        monkeypatch.setattr(
            sys, "stderr", _FakeTerminal() if terminal else io.StringIO()
        )

    return install


@pytest.fixture(autouse=True)
def _restore_progress_style():
    before = get_progress_style()
    try:
        yield
    finally:
        set_progress_style(before)


def test_progress_style_defaults_to_auto():
    assert get_progress_style() == "auto"


@pytest.mark.parametrize(
    ("terminal", "environment", "expected"),
    [
        # A terminal acts on carriage returns, so the bar draws.
        (True, {}, "bar"),
        # A bare pipe keeps them verbatim; ten log lines instead.
        (False, {}, "log"),
        # PyCharm's run console is a pipe that renders \r anyway. This is the
        # case isatty() gets wrong, and the reason the sniff list exists.
        (False, {"PYCHARM_HOSTED": "1"}, "bar"),
        # Capture wins over a pty: CI attaches one and still stores the bytes.
        (True, {"CI": "true"}, "log"),
        (True, {"PYTEST_CURRENT_TEST": "test_x"}, "log"),
        # ...including when both signals disagree.
        (False, {"PYTEST_CURRENT_TEST": "t", "PYCHARM_HOSTED": "1"}, "log"),
        # An explicit negative is not a capture signal.
        (True, {"CI": "false"}, "bar"),
        (True, {"CI": "0"}, "bar"),
    ],
)
def test_auto_resolves_by_whether_carriage_returns_are_acted_on(
    stderr_console, terminal, environment, expected
):
    stderr_console(terminal=terminal, **environment)
    assert resolve_progress_style() == expected


@pytest.mark.parametrize("style", [s for s in PROGRESS_STYLES if s != "auto"])
def test_an_explicit_style_overrides_every_sniff(stderr_console, style):
    """The escape hatch has to be absolute -- the sniff list is never complete."""
    set_progress_style(style)
    stderr_console(terminal=False, CI="true", PYTEST_CURRENT_TEST="t")
    assert resolve_progress_style() == style


def test_auto_never_survives_resolution(stderr_console):
    """Callers switch on the result, so "auto" must not reach them."""
    stderr_console(terminal=True)
    assert resolve_progress_style() in ("bar", "log", "none")


def test_style_is_normalized():
    set_progress_style("  BAR ")
    assert get_progress_style() == "bar"


def test_unknown_style_is_rejected():
    with pytest.raises(ValueError, match="Unknown progress style"):
        set_progress_style("barr")


def test_rejecting_a_style_leaves_the_previous_one(stderr_console):
    set_progress_style("log")
    with pytest.raises(ValueError):
        set_progress_style("nope")
    assert get_progress_style() == "log"


def test_stderr_is_resolved_at_emit_time_not_at_import():
    """The daemon swaps sys.stderr per job, after this module was imported.

    A handler bound to stderr at construction writes every record to the
    daemon's own console and the requesting client sees nothing at all.
    """
    captured = io.StringIO()
    real, sys.stderr = sys.stderr, captured
    try:
        get_logger("scene").info("swapped-stderr marker")
        for handler in logger.handlers:
            handler.flush()
    finally:
        sys.stderr = real
    assert "swapped-stderr marker" in captured.getvalue()


def test_console_handler_is_still_discoverable_by_tqdm():
    """``logging_redirect_tqdm`` routes log lines around a live bar.

    It finds the console handler by testing ``handler.stream in {stdout,
    stderr}``, so the deferred lookup has to satisfy an ordinary read.
    """
    from tqdm.contrib.logging import (
        _get_first_found_console_logging_handler as find_console_handler,
    )

    assert find_console_handler(logger.handlers) is not None
