"""Algan's library logger, backed by Python's standard :mod:`logging`.

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


def get_logger(name=None):
    """Return Algan's logger, or a child of it when ``name`` is given."""
    return logger if name is None else logger.getChild(name)


def set_log_level(level):
    """Set the verbosity of Algan's console output.

    Parameters
    ----------
    level
        A standard :mod:`logging` level name or value, e.g. ``"WARNING"`` to
        silence progress messages or ``"DEBUG"`` for extra detail.
    """
    logger.setLevel(level.upper() if isinstance(level, str) else level)
