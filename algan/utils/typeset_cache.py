"""Concurrency-safe builds into the shared LaTeX and Typst SVG caches.

Every Algan process typesets into one directory per toolchain under
``SETTINGS.paths.cache_directory`` (``manim/Tex`` and ``manim/Typst``), and the
vendored Manim was written for a directory only one process uses. Its
``tex_to_svg_file`` compiled each formula *in* ``tex_dir`` and then deleted every
file there that was not an ``.svg`` or ``.tex`` -- including another process's
``.dvi``, ``.aux`` and ``.log`` mid-build, which that process's ``dvisvgm`` then
reported as an installation that "does not support converting .dvi files to SVG".
Its writes were not atomic either: a reader that found ``<hash>.svg`` could be
reading a file ``dvisvgm`` was still writing.

:func:`build_tex_svg` replaces that function -- ``scripts/vendor_manim.py`` routes
the vendored ``tex_to_svg_file`` here. Each build runs in a private directory
beside ``tex_dir``, and only finished files enter the cache, by
:func:`os.replace`:

- a build deletes nothing but its own directory, so no process can break another;
- a cached ``.svg`` is always whole, so ``exists()`` is a safe cache-hit test;
- two processes building the same formula both succeed, and the second rename
  replaces identical bytes.

The layout is unchanged -- ``tex_hash(source) + ".svg"`` beside its ``.tex`` -- so
SVGs cached by earlier versions are reused, and the build directory is a
*sibling* of ``tex_dir`` rather than inside it, so an older Algan sharing the
cache, still sweeping ``tex_dir`` after every formula, cannot reach it either.

The other writers into the shared cache publish through :func:`building` for
the same reason: Pango text, which used to rewrite its cached SVG in place on
every construction, and the vendored Typst writer. (The SVG parser no longer
writes a scratch copy beside the file it parses at all.)
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

#: Build directories older than this are leftovers of a process killed
#: mid-build (no LaTeX run takes this long), and are swept away.
_STALE_BUILD_SECONDS = 24 * 3600

#: How much of a failed LaTeX log goes into the exception message.
_MAX_REPORTED_ERRORS = 3
_MAX_LINES_PER_ERROR = 8

_swept_build_dirs: set[str] = set()
_sweep_lock = threading.Lock()


@contextlib.contextmanager
def building(destination: Path):
    """Build a cache file privately, then publish it in one atomic rename.

    Yields an absolute path, unique to this call, in ``destination``'s
    directory; whatever is written there replaces ``destination`` when the
    block exits normally, and is deleted if it raises. Meant for
    content-addressed cache entries, where every writer of ``destination``
    writes the same bytes: if the rename fails because another process holds
    ``destination`` open (Windows refuses to replace an open file), the entry
    already there is the one this build would have written, and it is kept.
    """
    destination = Path(destination)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent
    )
    os.close(descriptor)
    partial = Path(os.path.abspath(temporary))
    try:
        yield partial
        _publish(partial, destination)
    finally:
        with contextlib.suppress(OSError):
            partial.unlink()


def write_atomically(path: Path, data: bytes) -> Path:
    """Write ``data`` to ``path`` so no reader ever sees a partial file.

    See :func:`building`, which this is the one-shot form of.
    """
    with building(path) as partial:
        partial.write_bytes(data)
    return Path(path)


def _publish(source: Path, destination: Path) -> None:
    """Move a finished file into the cache in one atomic rename."""
    try:
        os.replace(source, destination)
    except OSError:
        # Content-addressed: whoever got there first wrote these same bytes.
        if not destination.exists():
            raise


def build_directory(tex_dir: Path) -> Path:
    """Where private LaTeX builds for ``tex_dir`` run: a sibling directory.

    Not inside ``tex_dir``: an older Algan sharing the cache deletes every
    non-``.svg``/``.tex`` entry there after each formula, and would trip over
    (or delete) a build in progress.
    """
    return tex_dir.with_name(f"{tex_dir.name}-build")


def build_tex_svg(expression, environment=None, tex_template=None) -> Path:
    r"""Typeset ``expression`` to an SVG in Manim's ``tex_dir``, safely shared.

    A drop-in for vendored Manim's ``tex_to_svg_file``: same arguments, same
    content-addressed cache file (``tex_hash(source) + ".svg"``), but compiled
    in a private directory and moved into the cache only when finished.

    Parameters
    ----------
    expression
        The LaTeX to typeset, e.g. ``\sqrt{2}``.
    environment
        The environment to typeset it in, e.g. ``"align*"``. Defaults to
        ``None``, inserting ``expression`` into the template verbatim.
    tex_template
        The ``TexTemplate`` to typeset with. Defaults to ``None``, meaning
        ``config["tex_template"]``.

    Returns
    -------
    pathlib.Path
        The cached SVG.
    """
    from manim import config
    from manim.utils.tex_file_writing import tex_hash

    if tex_template is None:
        tex_template = config["tex_template"]
    if environment is not None:
        source = tex_template.get_texcode_for_expression_in_env(expression, environment)
    else:
        source = tex_template.get_texcode_for_expression(expression)

    tex_dir = Path(config.get_dir("tex_dir"))
    stem = tex_hash(source)
    svg_file = tex_dir / f"{stem}.svg"
    if svg_file.exists():
        return svg_file

    tex_dir.mkdir(parents=True, exist_ok=True)
    builds = build_directory(tex_dir)
    builds.mkdir(parents=True, exist_ok=True)
    _sweep_stale_builds(builds)
    work = Path(tempfile.mkdtemp(prefix=f"{stem}.", dir=builds))
    keep_work = bool(config["no_latex_cleanup"])
    try:
        tex_file = work / f"{stem}.tex"
        tex_file.write_text(source, encoding="utf-8")
        output_format = tex_template.output_format
        built = _compile(tex_file, tex_template.tex_compiler, output_format, builds)
        built_svg = _convert_to_svg(built, output_format)
        # The source goes first, so a cached SVG always has its .tex beside
        # it -- the same pair Manim leaves, for anyone debugging a formula.
        _publish(tex_file, tex_dir / tex_file.name)
        _publish(built_svg, svg_file)
    finally:
        if not keep_work:
            shutil.rmtree(work, ignore_errors=True)
    return svg_file


def _compile(tex_file: Path, tex_compiler, output_format: str, builds: Path) -> Path:
    """Run the template's TeX compiler(s) on ``tex_file``, in its own directory."""
    from manim.utils.tex_file_writing import make_tex_compilation_command

    compilers = [tex_compiler] if isinstance(tex_compiler, str) else list(tex_compiler)
    work = tex_file.parent
    for compiler in compilers:
        command = make_tex_compilation_command(compiler, output_format, tex_file, work)
        try:
            completed = subprocess.run(command, stdout=subprocess.DEVNULL)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"LaTeX typesetting needs `{compiler}`, and it was not found on PATH."
            ) from None
        if completed.returncode != 0:
            raise ValueError(
                _latex_failure_message(compiler, completed.returncode, tex_file, builds)
            )
    output = tex_file.with_suffix(output_format)
    if not output.exists():
        raise ValueError(
            f"{compilers[-1]} reported success but wrote no {output_format} file "
            f"for {tex_file.name}. "
            + _log_excerpt_or_note(tex_file.with_suffix(".log"), builds)
        )
    return output


def _latex_failure_message(compiler, returncode, tex_file: Path, builds: Path) -> str:
    return (
        f"{compiler} could not typeset this formula (exit status {returncode}). "
        + _log_excerpt_or_note(tex_file.with_suffix(".log"), builds)
    )


def _log_excerpt_or_note(log_file: Path, builds: Path) -> str:
    """The errors a LaTeX log reports, and where the whole log was kept.

    The build directory is deleted on the way out, so a failed log is moved
    next to it (``<hash>.log``, replaced by the formula's next failure) rather
    than pointed at where it will no longer be.
    """
    if not log_file.exists():
        return (
            f"It wrote no log file either, so it did not get as far as reading "
            f"{log_file.with_suffix('.tex').name}; check that the TeX "
            "installation runs on its own."
        )
    lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
    kept = builds / log_file.name
    try:
        _publish(log_file, kept)
    except OSError:
        kept = None
    excerpt = _log_errors(lines)
    where = f"\nFull log: {kept}" if kept is not None else ""
    if not excerpt:
        return f"The log names no error.{where}"
    return "LaTeX reported:\n" + "\n".join(excerpt) + where


def _log_errors(lines: list[str]) -> list[str]:
    """Each ``! error`` of a LaTeX log, through the ``l.<n>`` line locating it.

    Manim's own report went through its logger, which Algan's vendored copy
    leaves silent, so the exception message is the only place it can appear.
    """
    from manim.utils.tex_file_writing import LATEX_ERROR_INSIGHTS

    excerpt = []
    starts = [i for i, line in enumerate(lines) if line.startswith("!")]
    for start in starts[:_MAX_REPORTED_ERRORS]:
        block = lines[start : start + _MAX_LINES_PER_ERROR]
        for offset, line in enumerate(block):
            if re.match(r"l\.\d+", line):
                # The line after ``l.<n>`` continues the offending source line.
                block = lines[start : start + offset + 2]
                break
        excerpt.extend("  " + line.rstrip() for line in block if line.strip())
        for pattern, insight in LATEX_ERROR_INSIGHTS:
            matching = re.search(pattern, lines[start][2:])
            if matching is not None:
                excerpt.extend("  " + hint for hint in insight(matching))
    return excerpt


def _convert_to_svg(built: Path, output_format: str) -> Path:
    """Convert the compiler's output to an SVG beside it with ``dvisvgm``."""
    svg = built.with_suffix(".svg")
    command = [
        "dvisvgm",
        *(["--pdf"] if output_format == ".pdf" else []),
        "--page=1",
        "--no-fonts",
        "--verbosity=0",
        f"--output={svg.as_posix()}",
        built.as_posix(),
    ]
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            "LaTeX typesetting needs `dvisvgm` to convert its output to SVG, "
            "and it was not found on PATH."
        ) from None
    if svg.exists():
        return svg
    reported = completed.stderr.strip()
    message = (
        f"dvisvgm could not convert {built.name} to SVG "
        f"(exit status {completed.returncode})"
        + (f":\n  {reported}" if reported else ".")
    )
    if output_format != ".dvi":
        needs = " built with Ghostscript support" if output_format == ".pdf" else ""
        message += (
            f"\nConverting {output_format} files needs dvisvgm 2.4 or newer{needs}."
        )
    raise ValueError(message)


def _sweep_stale_builds(builds: Path) -> None:
    """Remove builds abandoned by killed processes, once per process.

    Only entries older than a day go: a live build is seconds old, so another
    process's work in progress is never touched.
    """
    key = os.fspath(builds)
    with _sweep_lock:
        if key in _swept_build_dirs:
            return
        _swept_build_dirs.add(key)
    cutoff = time.time() - _STALE_BUILD_SECONDS
    try:
        entries = list(builds.iterdir())
    except OSError:
        return
    for entry in entries:
        try:
            if entry.stat().st_mtime >= cutoff:
                continue
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink()
        except OSError:
            continue
