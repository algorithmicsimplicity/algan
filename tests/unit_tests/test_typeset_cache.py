"""Processes sharing one typesetting cache must not break each other's builds.

Every Algan process typesets into the same ``cache/manim/Tex`` (and ``texts``,
``Typst``). Vendored Manim compiled each formula in that directory and then
deleted every non-``.svg``/``.tex`` file there -- another process's ``.dvi`` and
``.log`` mid-build included -- so two renders run at once failed formulas that
compile fine alone, blaming the dvisvgm version. Its SVG parser also wrote a
fixed-name scratch copy beside the SVG it parsed, and Pango text rewrote its
cached SVG in place on every construction.

These run the builds in threads, which share the directory exactly as two
processes do; ``algan.utils.typeset_cache`` is the fix.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import threading
import time
import xml.etree.ElementTree as ET

import pytest

from algan.mobs import text as text_module
from algan.utils import typeset_cache

mn = text_module.mn

needs_latex = pytest.mark.skipif(
    shutil.which("latex") is None or shutil.which("dvisvgm") is None,
    reason="needs a TeX distribution with dvisvgm",
)


@pytest.fixture
def tex_dir(tmp_path):
    """Point Manim's ``tex_dir`` at an empty directory for one test."""
    config = mn.config
    saved = config.tex_dir
    directory = tmp_path / "Tex"
    config.tex_dir = os.fspath(directory)
    try:
        yield directory
    finally:
        config.tex_dir = saved


def _tex_to_svg(expression):
    from manim.utils.tex_file_writing import tex_to_svg_file

    return tex_to_svg_file(expression, environment="align*")


@needs_latex
def test_concurrent_builds_into_one_tex_dir_all_succeed(tex_dir):
    stamp = time.time_ns()
    failures, built = [], []

    def typeset(tag):
        for k in range(6):
            try:
                built.append(_tex_to_svg(rf"\frac{{{tag}_{{{k}}}}}{{{stamp}}}"))
            except Exception as error:  # noqa: BLE001 - collected, asserted below
                failures.append(f"{tag}{k}: {type(error).__name__}: {error}")

    threads = [threading.Thread(target=typeset, args=(tag,)) for tag in "ABC"]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert failures == []
    assert len(set(built)) == 18
    for svg in built:
        ET.parse(svg)  # whole files, never one still being written


@needs_latex
def test_a_build_deletes_nothing_but_its_own_files(tex_dir):
    tex_dir.mkdir(parents=True)
    foreign = [tex_dir / name for name in ("other.dvi", "other.aux", "other.log")]
    for path in foreign:
        path.write_text("another process's build in flight")
    in_flight = typeset_cache.build_directory(tex_dir) / "other.build"
    in_flight.mkdir(parents=True)

    svg = _tex_to_svg(rf"x^{{{time.time_ns()}}}")

    assert all(path.exists() for path in foreign)
    assert in_flight.is_dir()
    assert svg.parent == tex_dir
    assert svg.with_suffix(".tex").exists()
    # This build's own private directory is gone.
    assert sorted(p.name for p in in_flight.parent.iterdir()) == ["other.build"]


@needs_latex
def test_an_existing_svg_is_returned_without_building(tex_dir, monkeypatch):
    first = _tex_to_svg(r"\alpha + \beta")
    monkeypatch.setattr(
        typeset_cache, "_compile", lambda *a, **k: pytest.fail("rebuilt a cache hit")
    )
    assert _tex_to_svg(r"\alpha + \beta") == first


@needs_latex
def test_latex_errors_are_reported_from_the_log(tex_dir):
    with pytest.raises(ValueError) as caught:
        _tex_to_svg(r"\definitelynotacommand{x}")

    message = str(caught.value)
    assert "Undefined control sequence" in message
    assert "dvisvgm" not in message
    kept = pathlib.Path(message.rsplit("Full log: ", 1)[1].strip())
    assert kept.is_file()
    assert kept.parent == typeset_cache.build_directory(tex_dir)


@needs_latex
def test_a_missing_dvi_is_not_blamed_on_the_dvisvgm_version(tex_dir, monkeypatch):
    def compile_without_output(tex_file, *args):
        return tex_file.with_suffix(".dvi")  # as if deleted from under us

    monkeypatch.setattr(typeset_cache, "_compile", compile_without_output)
    with pytest.raises(ValueError) as caught:
        _tex_to_svg(rf"y^{{{time.time_ns()}}}")

    message = str(caught.value)
    assert "dvisvgm could not convert" in message
    assert "updating dvisvgm" not in message
    assert not list(tex_dir.glob("*.svg"))


def test_stale_builds_are_swept_and_live_ones_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(typeset_cache, "_swept_build_dirs", set())
    stale, live = tmp_path / "stale.x", tmp_path / "live.x"
    stale.mkdir()
    live.mkdir()
    two_days_ago = time.time() - 2 * 24 * 3600
    os.utime(stale, (two_days_ago, two_days_ago))

    typeset_cache._sweep_stale_builds(tmp_path)

    assert not stale.exists()
    assert live.is_dir()


def test_write_atomically_keeps_an_entry_another_writer_holds(tmp_path, monkeypatch):
    target = tmp_path / "entry.svg"
    target.write_bytes(b"<svg/>")

    def refuse(*args):
        raise PermissionError("held open by another process")

    monkeypatch.setattr(typeset_cache.os, "replace", refuse)
    typeset_cache.write_atomically(target, b"<svg/>")
    assert target.read_bytes() == b"<svg/>"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["entry.svg"]

    with pytest.raises(PermissionError):
        typeset_cache.write_atomically(tmp_path / "missing.svg", b"<svg/>")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["entry.svg"]


def test_svg_parsing_writes_nothing_beside_the_source(tmp_path):
    source = tmp_path / "shape.svg"
    source.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
        '<path d="M1 1 L9 1 L9 9 Z"/></svg>'
    )
    # The fixed scratch name upstream wrote, parsed and deleted: another
    # process parsing the same cached SVG owned it at the same time.
    scratch = tmp_path / "shape_.svg"
    scratch.write_text("another process's scratch copy")

    mob = mn.SVGMobject(os.fspath(source), use_svg_cache=False)

    assert len(mob.submobjects) == 1
    assert scratch.read_text() == "another process's scratch copy"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["shape.svg", "shape_.svg"]


def test_pango_cache_hit_does_not_rewrite_the_cached_svg(tmp_path):
    pytest.importorskip("manimpango")
    if not hasattr(mn, "Text"):
        pytest.skip("manimpango is installed but did not import")
    config = mn.config
    saved = config.text_dir
    config.text_dir = os.fspath(tmp_path)
    try:
        mn.Text("written once", use_svg_cache=False)
        (svg,) = tmp_path.glob("*.svg")
        before = svg.stat()
        mn.Text("written once", use_svg_cache=False)
        after = svg.stat()
    finally:
        config.text_dir = saved

    assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
    assert sorted(p.name for p in tmp_path.iterdir()) == [svg.name]
