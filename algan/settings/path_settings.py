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


def output_root_for(script):
    """Directory the script lives in, falling back to the working directory.

    ``sys.path[0]`` used to fill this role, but it is the script's directory
    only under ``python script.py``; under ``-m`` or ``-c`` it is the working
    directory or empty, which silently moved everyone's output.

    Takes the script explicitly so that the render daemon can compute what a
    fresh process running *that* script would have used: the daemon resolves
    these defaults once at its own startup, where there is no user script at
    all, and would otherwise write every client's video into its own directory.
    """
    if script is None:
        return os.getcwd()
    return os.path.dirname(script)


def output_filename_for(script):
    """Name renders get when no path is passed: the script's own name."""
    if script is None:
        return "algan_render_output"
    stem = os.path.splitext(os.path.basename(script))[0]
    return stem or "algan_render_output"


def _default_output_root():
    return output_root_for(_main_script_path())


def _default_output_filename():
    return output_filename_for(_main_script_path())


@dataclass
class PathSettings(Settings):
    """Runtime-adjustable content-cache and output paths.

    Output resolution is ``output_root / output_directory / name`` for a bare
    filename; a path with a directory in it is used exactly as supplied.

    ``ALGAN_HOME`` and the Taichi offline-cache path are initialization-only
    environment configuration. ``cache_directory`` remains public because
    Algan's content caches are consulted lazily and can safely move at runtime.

    Attributes
    ----------
    ffmpeg_binary
        Path to the FFmpeg executable video encoding runs. Defaults to
        ``None``, meaning Algan picks one: the ``FFMPEG_BINARY`` environment
        variable if set, else moviepy's configured binary (often imageio-ffmpeg's
        static build, which carries no NVENC encoders), else ``ffmpeg`` on the
        PATH. Setting it pins that choice -- useful when the build moviepy found
        lacks a codec the system FFmpeg has.
    """

    cache_directory: str = str(_CACHE_DIRECTORY)
    output_root: str = field(default_factory=_default_output_root)
    output_directory: str = "algan_outputs"
    output_filename: str = field(default_factory=_default_output_filename)
    ffmpeg_binary: str | None = None

    def __post_init__(self):
        for name in (
            "cache_directory",
            "output_root",
            "output_directory",
            "output_filename",
        ):
            object.__setattr__(self, name, os.fspath(getattr(self, name)))
        # Optional, so it is normalised only when set -- os.fspath(None) raises.
        if self.ffmpeg_binary is not None:
            object.__setattr__(self, "ffmpeg_binary", os.fspath(self.ffmpeg_binary))
