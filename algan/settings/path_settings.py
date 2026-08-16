"""Where Algan reads assets from and writes output to.

Output resolves as ``output_root / output_directory / name``. A bare filename
lands in the output directory; anything containing a directory separator is used
as given.

The defaults are chosen so that a script just works from wherever you launch it:
``output_root`` is the directory holding the main script (the working directory
when there is no script, as in a REPL or notebook), and ``output_filename`` is
that script's stem -- so ``Scene.save_video()`` with no arguments writes
``<script>.mp4`` beside it.

``cache_directory`` is where content-addressed caches live: Tex and SVG geometry,
triangulated outlines, Taichi's offline kernel cache.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

from algan.settings._startup import _CACHE_DIRECTORY
from algan.settings.abstract_settings import Settings


def _main_script_path():
    """Path of the script Python was started with, if there is one.

    Absent for ``-c``, ``-m``, REPL and most embedding hosts.
    """
    main = sys.modules.get("__main__")
    path = getattr(main, "__file__", None)
    if not path:
        return None
    try:
        return os.path.abspath(path)
    except OSError:
        return None


def _default_output_root():
    """Directory the script lives in, falling back to the working directory.

    ``sys.path[0]`` used to fill this role, but it is the script's directory
    only under ``python script.py``; under ``-m`` or ``-c`` it is the working
    directory or empty, which silently moved everyone's output.
    """
    script = _main_script_path()
    if script is None:
        return os.getcwd()
    return os.path.dirname(script)


def _default_output_filename():
    """Name renders get when no path is passed: the script's own name."""
    script = _main_script_path()
    if script is None:
        return "algan_render_output"
    stem = os.path.splitext(os.path.basename(script))[0]
    return stem or "algan_render_output"


@dataclass
class PathSettings(Settings):
    """Runtime-adjustable content-cache and output paths.

    Output resolution is ``output_root / output_directory / name`` for a bare
    filename; a path with a directory in it is used exactly as supplied.

    ``ALGAN_HOME`` and the Taichi offline-cache path are initialization-only
    environment configuration. ``cache_directory`` remains public because
    Algan's content caches are consulted lazily and can safely move at runtime.
    """

    cache_directory: str = str(_CACHE_DIRECTORY)
    output_root: str = field(default_factory=_default_output_root)
    output_directory: str = "algan_outputs"
    output_filename: str = field(default_factory=_default_output_filename)

    def __post_init__(self):
        for name in (
            "cache_directory",
            "output_root",
            "output_directory",
            "output_filename",
        ):
            object.__setattr__(self, name, os.fspath(getattr(self, name)))
