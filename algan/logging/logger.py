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
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("algan")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(os.environ.get("ALGAN_LOG_LEVEL", "INFO").upper())
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


def set_log_level(level):
    """Set the verbosity of Algan's console output.

    Parameters
    ----------
    level
        A standard ``logging`` level name or value, e.g. ``"WARNING"`` to
        silence progress messages or ``"DEBUG"`` for extra detail.

    Notes
    -----
    This also holds the noisier third-party loggers Algan pulls in (currently
    Manim's) to the same level, so one call quiets the whole console.
    """
    logger.setLevel(level.upper() if isinstance(level, str) else level)
    _quiet_third_party()
