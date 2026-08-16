"""Algan's library logger, backed by Python's standard ``logging`` module.

All of Algan's diagnostic and progress output goes through the ``"algan"``
logger. By default it prints bare messages to stderr at ``INFO`` level (so
render-progress messages stay visible, as they always were). To quiet or
raise verbosity, either:

* set the ``ALGAN_LOG_LEVEL`` environment variable before importing algan
  (e.g. ``ALGAN_LOG_LEVEL=WARNING`` silences progress output), or
* call :func:`set_log_level` at any time, or
* attach your own handlers to ``logging.getLogger("algan")`` after calling
  ``logger.handlers.clear()``.

Levels are thresholds, not selectors: a level shows itself and everything more
severe, so ``INFO`` also shows warnings and errors.

Algan adds one level of its own, :data:`PERF`, between ``DEBUG`` and ``INFO``::

    ALGAN_LOG_LEVEL = PERF  # or set_log_level("PERF")

It carries the renderer's self-healing events -- the batch splits and pool
retries it performs when a chunk does not fit. Those are the memory model
working as designed, not faults, so they are below ``INFO`` and invisible by
default; they used to be ``WARNING``, which made a healthy render look broken
and was the only thing left on screen for anyone who set ``WARNING`` to quiet
the progress output. Turn ``PERF`` on when a render is slower than expected and
you want to see how it is being budgeted. Being below ``INFO``, it also means
``DEBUG`` includes these messages.

*How* a render reports its progress is a separate choice from *whether* it is
logged at all, and is controlled by :func:`set_progress_style` or the
``ALGAN_PROGRESS`` environment variable -- see :data:`PROGRESS_STYLES`.
"""

from __future__ import annotations

import logging
import os
import sys

from algan.environment import env_str

#: Renderer budget/recovery diagnostics: below ``INFO`` so they stay off by
#: default, above ``DEBUG`` so turning them on does not also enable every other
#: debug message in the package.
PERF = 15
logging.addLevelName(PERF, "PERF")


class _LiveStderrHandler(logging.StreamHandler):
    """A ``StreamHandler`` that looks up ``sys.stderr`` when it emits.

    ``logging.StreamHandler()`` binds whichever object is ``sys.stderr`` at
    construction -- for this module, at import. The render daemon replaces
    ``sys.stderr`` for the duration of each job with a stream that tees to the
    requesting client, so a handler bound at import writes every record to the
    daemon's own console and the client sees nothing at all. Deferring the
    lookup is the same trick the standard library plays with its own
    handler-of-last-resort (``logging._StderrHandler``), for the same reason.

    ``stream`` is deliberately a read-only property, which is what makes
    bypassing ``StreamHandler.__init__`` necessary: that initializer assigns
    to it. Readers are unaffected -- ``tqdm.contrib.logging`` checks
    ``handler.stream in {sys.stdout, sys.stderr}`` to find the console handler
    it should route around a live bar, and this satisfies it by construction.
    """

    def __init__(self, level=logging.NOTSET):
        logging.Handler.__init__(self, level)

    @property
    def stream(self):
        return sys.stderr


logger = logging.getLogger("algan")
if not logger.handlers:
    _handler = _LiveStderrHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(env_str("ALGAN_LOG_LEVEL", "INFO").upper())
    # Don't double-print through the root logger if the application configured it.
    logger.propagate = False


# Third-party loggers that would otherwise talk over Algan's console. Manim's
# installed package configures its own rich-formatted logger at INFO, so
# creating a Tex used to print through it even after set_log_level("WARNING").
_THIRD_PARTY_LOGGERS = ("manim",)


class _AlganLevelFilter(logging.Filter):
    """Hold a third-party logger to Algan's level.

    A filter rather than ``setLevel``: manim's ``make_logger`` calls
    ``setLevel`` on its own logger at import time and would clobber a level set
    beforehand, but it never touches filters, so this holds whichever order the
    two modules happen to load in. Reading the level live also means
    :func:`set_log_level` needs no per-logger bookkeeping.

    Scope: this applies to records logged on the named logger itself, not to
    records propagating up from its children.
    """

    def filter(self, record):
        return record.levelno >= logger.getEffectiveLevel()


def _quiet_third_party():
    for name in _THIRD_PARTY_LOGGERS:
        third_party = logging.getLogger(name)
        if not any(isinstance(f, _AlganLevelFilter) for f in third_party.filters):
            third_party.addFilter(_AlganLevelFilter())


_quiet_third_party()


def get_logger(name=None):
    """Return Algan's logger, or a child of it when ``name`` is given."""
    return logger if name is None else logger.getChild(name)


#: Accepted render-progress styles.
#:
#: ``"bar"``
#:     A tqdm progress bar on stderr. Its in-place updates need a consumer that
#:     acts on carriage returns.
#: ``"log"``
#:     At most ten ordinary log lines over the render, and nothing at all for a
#:     render too short for a progress report to tell anyone anything.
#: ``"none"``
#:     No progress output. Other log messages are unaffected.
#: ``"auto"``
#:     Decide per render; see :func:`resolve_progress_style`.
PROGRESS_STYLES = ("auto", "bar", "log", "none")

#: Environments that capture stderr into a stored log, where a bar's carriage
#: returns are kept verbatim rather than acted on and the bar becomes hundreds
#: of lines. Checked *before* the terminal test: a captured run can still have
#: a pty attached.
_CAPTURED_ENV_VARS = ("CI", "PYTEST_CURRENT_TEST")

#: Consoles that are not terminals but do act on carriage returns, so a bar
#: renders correctly in them. ``isatty()`` says no and is simply wrong; there
#: is no portable way to ask, so this is a list and will always be incomplete
#: -- ``ALGAN_PROGRESS=bar`` is the escape hatch for anything not on it.
#: PyCharm's run console sets ``PYCHARM_HOSTED``, the same signal colorama
#: keys off; its "Emulate terminal in output console" option allocates a real
#: pty instead and is caught by the terminal test.
_BAR_CAPABLE_ENV_VARS = ("PYCHARM_HOSTED",)

_progress_style = "auto"


def _env_flag(name):
    """Whether ``name`` is set to something other than an explicit negative."""
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() not in ("", "0", "false", "no", "off")


def _stderr_is_terminal():
    """Whether ``sys.stderr`` is a terminal *right now*.

    Deliberately not cached. The render daemon replaces ``sys.stderr`` for the
    duration of each job with a stand-in reporting the requesting *client's*
    terminal -- a stream that does not exist yet when this module is imported
    into a warm daemon.
    """
    try:
        return bool(sys.stderr.isatty())
    except Exception:
        # A stand-in stream may not implement isatty, or may be closed.
        return False


def _in_ipython():
    """Whether this is an IPython/Jupyter session, whose output area redraws.

    Only inspects an already-imported IPython: asking the question must not be
    what drags it into the process.
    """
    ipython = sys.modules.get("IPython")
    if ipython is None:
        return False
    try:
        return ipython.get_ipython() is not None
    except Exception:
        return False


def set_progress_style(style):
    """Set how renders report progress.

    Parameters
    ----------
    style
        One of :data:`PROGRESS_STYLES`: ``"bar"``, ``"log"``, ``"none"`` or
        ``"auto"`` (the default).

    Raises
    ------
    ValueError
        If ``style`` is not one of :data:`PROGRESS_STYLES`.

    Notes
    -----
    Progress output is emitted at ``INFO``, so ``set_log_level("WARNING")``
    silences it whatever the style.
    """
    global _progress_style
    normalized = str(style).strip().lower()
    if normalized not in PROGRESS_STYLES:
        raise ValueError(
            f"Unknown progress style {style!r}; expected one of "
            f"{', '.join(repr(s) for s in PROGRESS_STYLES)}."
        )
    _progress_style = normalized


def get_progress_style():
    """Return the configured progress style, one of :data:`PROGRESS_STYLES`."""
    return _progress_style


def resolve_progress_style():
    """Resolve the configured style to a concrete one for this render.

    Returns ``"bar"``, ``"log"`` or ``"none"``; ``"auto"`` never survives.

    Under ``"auto"`` the question being answered is not "is this a terminal"
    but "will whatever reads stderr act on a carriage return". Those differ,
    and ``isatty()`` is only a proxy: a captured CI or pytest log has one kept
    verbatim, while several consoles that are not terminals (PyCharm's run
    console, a notebook) render one correctly. So capture is ruled out first,
    then a terminal is taken at its word, then the known bar-capable consoles
    get their exception.
    """
    if _progress_style != "auto":
        return _progress_style
    if any(_env_flag(name) for name in _CAPTURED_ENV_VARS):
        return "log"
    if _stderr_is_terminal():
        return "bar"
    if any(_env_flag(name) for name in _BAR_CAPABLE_ENV_VARS):
        return "bar"
    if _in_ipython():
        return "bar"
    return "log"


def set_log_level(level):
    """Set the verbosity of Algan's console output.

    Parameters
    ----------
    level
        A standard ``logging`` level name or value, e.g. ``"WARNING"`` to
        silence progress messages or ``"DEBUG"`` for extra detail. Algan's own
        ``"PERF"`` (see :data:`PERF`) sits between the two and adds the
        renderer's budget and recovery diagnostics to the default output.

    Notes
    -----
    This also holds the noisier third-party loggers Algan pulls in (currently
    Manim's) to the same level, so one call quiets the whole console.
    """
    logger.setLevel(level.upper() if isinstance(level, str) else level)
    _quiet_third_party()


# Applied after the logger exists so a bad value can be reported through it.
# Unlike ALGAN_LOG_LEVEL this is not read at import for any technical reason --
# set_progress_style works at any time -- it is read here only so the variable
# behaves like every other ALGAN_ one.
_ENV_PROGRESS = env_str("ALGAN_PROGRESS", None)
if _ENV_PROGRESS:
    try:
        set_progress_style(_ENV_PROGRESS)
    except ValueError as _exc:
        logger.warning("ALGAN_PROGRESS ignored: %s", _exc)
