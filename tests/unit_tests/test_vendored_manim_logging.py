"""What the vendored Manim logs reaches Algan's console.

Its logger was upstream's ``"manim"`` with only a ``NullHandler``, so every
warning it logged -- an SVG element it could not import, a deprecated
argument, a font that is not installed -- was dropped unless the application
had configured the root logger. It is now ``"algan.manim"``, a child of
Algan's logger.
"""

from __future__ import annotations

import importlib
import logging

import pytest

from algan.logging.logger import logger as algan_logger
from algan.logging.logger import set_log_level
from algan.settings import SETTINGS
from algan.utils import manim_svg_cache as svg_cache

manim = importlib.import_module("manim")

#: A 2x2 PNG, the raster Cairo embeds for a color emoji.
_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAAFUlEQVR4nGP8f4LhPwMD"
    "AwMTAxQAACuXAsoUVo9TAAAAAElFTkSuQmCC"
)


@pytest.fixture(autouse=True)
def _restore_log_level():
    # Manim's own level too: leaving a tempconfig that set config.verbosity
    # does not put it back.
    before, manims_before = algan_logger.level, manim.logger.level
    try:
        yield
    finally:
        algan_logger.setLevel(before)
        manim.logger.setLevel(manims_before)


class _Recorder(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


def test_every_vendored_module_logs_on_one_logger_under_algans():
    for module in ("manim._config.utils", "manim.utils.deprecation"):
        assert importlib.import_module(module).logger is manim.logger
    assert manim.logger.name == "algan.manim"
    assert manim.logger.parent is algan_logger


def test_a_vendored_manim_warning_prints_on_algans_console(capsys):
    from manim.utils.deprecation import deprecated

    @deprecated(since="v0.1", replacement="new_shape")
    def old_shape():
        return 1

    assert old_shape() == 1
    assert "old_shape has been deprecated" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("algan_level", "record_level", "printed"),
    [
        ("INFO", logging.WARNING, True),
        ("INFO", logging.ERROR, True),
        # Manim's INFO is a running account (config files read, each Typst
        # file compiled), and Algan's INFO is its render progress.
        ("INFO", logging.INFO, False),
        ("DEBUG", logging.INFO, True),
        ("ERROR", logging.WARNING, False),
        ("ERROR", logging.ERROR, True),
    ],
)
def test_algans_level_governs_vendored_manim(
    capsys, algan_level, record_level, printed
):
    set_log_level(algan_level)
    manim.logger.log(record_level, "a message from the vendored Manim")
    assert ("a message from the vendored Manim" in capsys.readouterr().err) is printed


def test_manims_own_verbosity_does_not_override_algans_level(capsys):
    """``config.verbosity`` sets the Manim logger's own level, as Manim's
    config does from ``default.cfg`` when it is built.
    """
    set_log_level("ERROR")
    with manim.tempconfig({"verbosity": "DEBUG"}):
        assert manim.logger.level == logging.DEBUG
        manim.logger.warning("a vendored warning")
        manim.logger.info("a vendored info message")
    assert capsys.readouterr().err == ""


@pytest.fixture
def parse_svg(tmp_path):
    """Import an SVG through Manim's parser; return it and what it logged."""
    # An SVG another run already parsed is replayed from Algan's cache, and
    # never reaches the parser these tests are about.
    SETTINGS.paths.set(cache_directory=str(tmp_path / "cache"))
    saved = dict(svg_cache._MEM_CACHE)
    svg_cache._MEM_CACHE.clear()

    def parse(name, source):
        path = tmp_path / name
        path.write_text(source)
        recorder = _Recorder()
        manim.logger.addHandler(recorder)
        try:
            svg = manim.SVGMobject(str(path))
        finally:
            manim.logger.removeHandler(recorder)
        return svg, [record.getMessage() for record in recorder.records]

    try:
        yield parse
    finally:
        svg_cache._MEM_CACHE.clear()
        svg_cache._MEM_CACHE.update(saved)


def test_an_image_in_an_svg_is_imported_without_an_error(parse_svg):
    """A VGroup takes only vector shapes, and adding the image to one raised.

    The image was still drawn, but the error was logged for every emoji in a
    Text, which now prints.
    """
    svg, said = parse_svg(
        "emoji.svg",
        '<svg xmlns="http://www.w3.org/2000/svg" width="48" height="16" '
        'viewBox="0 0 48 16">'
        '<path d="M0 12 L5 0 L10 12 Z" fill="#fff"/>'
        '<g id="glyph"><image x="18" y="0" width="12" height="12" '
        f'href="data:image/png;base64,{_PNG}"/></g>'
        '<path d="M36 0 L46 0 L46 12 L36 12 Z" fill="#fff"/>'
        "</svg>",
    )

    assert said == []
    assert [type(mob).__name__ for mob in svg.submobjects] == [
        "VMobjectFromSVGPath",
        "ImageMobject",
        "VMobjectFromSVGPath",
    ]


def test_svg_text_left_out_is_named_with_how_to_keep_it(parse_svg):
    """Manim does not import an SVG's ``<text>``, which it reported only as
    ``Unsupported element type: <class 'svgelements.svgelements.Text'>``.
    """
    _, said = parse_svg(
        "labels.svg",
        '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">'
        '<rect x="0" y="0" width="100" height="10" fill="#fff"/>'
        '<text x="10" y="20">Label</text>'
        # Each <tspan> is an element of its own, and their <text> has no text.
        '<text x="10" y="40"><tspan>Alpha</tspan> <tspan dy="10">Beta</tspan></text>'
        '<text x="10" y="60">   </text>'
        '<text x="10" y="70"/>'
        '<text x="10" y="80">\n    A caption that runs on\n    for a good few lines,'
        "\n    longer than a warning should quote.\n</text>"
        f'<text x="10" y="90">{"https://example.com/" + "a" * 60}</text>'
        "</svg>",
    )

    how = (
        " is left out. To keep it, convert the text to paths in the SVG "
        "(Inkscape: Path > Object to Path; Illustrator: Type > Create Outlines)."
    )
    assert said == [
        f"SVG text is not imported, so {text!r}{how}"
        for text in (
            "Label",
            "Alpha",
            "Beta",
            "A caption that runs on for a good few lines, longer [...]",
            # One word, which shortening by words would leave as "[...]".
            "https://example.com/" + "a" * 34 + " [...]",
        )
    ]
