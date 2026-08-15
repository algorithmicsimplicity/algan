"""Algan's logger: the level threshold, and the PERF level's contract.

``PERF`` exists so the renderer's self-healing events -- the batch splits and
pool retries it performs when a chunk does not fit -- have somewhere to go that
is off by default. They were ``WARNING`` before, which made a healthy render
look broken; worse, ``WARNING`` is what a user sets to quiet the progress
output, so those messages were the *only* thing such a user saw.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

import pytest

from algan.logging.logger import PERF, get_logger, logger, set_log_level

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
