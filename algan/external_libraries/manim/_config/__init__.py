"""The global Manim config object.

Upstream also builds a ``rich``-backed logger here and installs it on the
*root* logger at import time. Algan owns its own console output and must not
have a library reconfigure logging out from under the importing application,
so this copy hands out a plain :mod:`logging` logger with a ``NullHandler``
and a ``console`` that is a thin ``print`` wrapper. That is everything the
vendored geometry subset asks for, and it keeps ``rich`` out of Algan's
dependency set.
"""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from .utils import ManimConfig, ManimFrame, make_config_parser

__all__ = [
    "config",
    "console",
    "error_console",
    "frame",
    "logger",
    "tempconfig",
]

_RICH_MARKUP = re.compile(r"\[/?[a-z_ ]+\]")


class _Console:
    """The sliver of ``rich.console.Console`` the vendored subset calls."""

    def __init__(self, stream: Any) -> None:
        self._stream = stream

    def print(self, *args: Any, **kwargs: Any) -> None:
        kwargs.pop("style", None)
        text = " ".join(_RICH_MARKUP.sub("", str(a)) for a in args)
        print(text, file=self._stream, **kwargs)


#: Reachable as ``manim.logger`` or ``logging.getLogger("manim")``.
logger = logging.getLogger("manim")
logger.addHandler(logging.NullHandler())

console = _Console(sys.stdout)
error_console = _Console(sys.stderr)

parser = make_config_parser()
config = ManimConfig().digest_parser(parser)
frame = ManimFrame(config)


@contextmanager
def tempconfig(temp: ManimConfig | dict[str, Any]) -> Generator[None, None, None]:
    """Temporarily modify the global ``config`` object.

    Inside the ``with`` statement the modified config is in force; on exit the
    original values are restored.

    Examples
    --------
    .. code-block:: pycon

       >>> config["frame_height"]
       8.0
       >>> with tempconfig({"frame_height": 100.0}):
       ...     print(config["frame_height"])
       100.0
       >>> config["frame_height"]
       8.0
    """
    global config
    original = config.copy()

    temp = {k: v for k, v in temp.items() if k in original}

    # update(), never assignment: every module holds a reference to this one
    # object, and rebinding the name here would not reach any of them.
    config.update(temp)
    try:
        yield
    finally:
        config.update(original)
