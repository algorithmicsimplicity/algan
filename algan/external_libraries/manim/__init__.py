"""Manim Community's geometry subset, vendored for Algan.

This is not Manim. It is the part of Manim that *builds Mobjects* -- the
Mobject graph, the Bezier and SVG/LaTeX machinery, and the shape, graphing,
text and 3-D classes on top of them. Manim's animations, scenes, cameras,
renderers, CLI and plugin system are absent, because Algan supplies all of
those itself: :class:`~algan.mobs.manim_mob.ManimMob` takes the cubic Bezier
circuits a Manim Mobject produces and turns them into Algan render primitives,
and everything after that is Algan's.

Reachable as ``manim`` -- Algan registers it under that name before importing
any Mob module -- and as ``algan.external_libraries.manim``. See
``VENDORING.md`` in this directory for the provenance and the exact set of
edits made to upstream.
"""

from __future__ import annotations

#: The upstream Manim Community release this subset was taken from.
__version__ = "0.21.0"

# isort: off

# Config first: every module below reads the global config as it is imported.
from ._config import config, console, error_console, frame, logger, tempconfig

# isort: on

from .constants import *
from .mobject.frame import *
from .mobject.geometry.arc import *
from .mobject.geometry.boolean_ops import *
from .mobject.geometry.labeled import *
from .mobject.geometry.line import *
from .mobject.geometry.polygram import *
from .mobject.geometry.shape_matchers import *
from .mobject.geometry.tips import *
from .mobject.graph import *
from .mobject.graphing.coordinate_systems import *
from .mobject.graphing.functions import *
from .mobject.graphing.number_line import *
from .mobject.graphing.probability import *
from .mobject.graphing.scale import *
from .mobject.logo import *
from .mobject.matrix import *
from .mobject.mobject import *
from .mobject.svg.brace import *
from .mobject.svg.svg_mobject import *
from .mobject.table import *
from .mobject.text.code_mobject import *
from .mobject.text.numbers import *
from .mobject.text.tex_mobject import *
from .mobject.three_d.polyhedra import *
from .mobject.three_d.three_d_utils import *
from .mobject.three_d.three_dimensions import *
from .mobject.types.image_mobject import *
from .mobject.types.point_cloud_mobject import *
from .mobject.types.vectorized_mobject import *
from .mobject.value_tracker import *
from .mobject.vector_field import *
from .utils import color, rate_functions, unit
from .utils.color import *
from .utils.config_ops import *
from .utils.file_ops import *
from .utils.images import *
from .utils.iterables import *
from .utils.paths import *
from .utils.rate_functions import *
from .utils.simple_functions import *
from .utils.space_ops import *
from .utils.tex import *
from .utils.tex_templates import *

SVG_GLOBALS.image_class = ImageMobject

#: Whether Pango-rendered text is available.
#:
#: ``Text``, ``MarkupText`` and ``Paragraph`` are the only classes that need
#: ``manimpango``, which publishes no Linux wheel -- requiring it would put a
#: source build of Pango in front of every Linux ``pip install algan``, which
#: is the cost this whole directory exists to avoid. Install the optional
#: extra (``pip install "algan[pango]"``) to get them.
#:
#: The three classes are withheld rather than left importable-and-broken,
#: because that is what Algan tests: :class:`algan.Text` renders through
#: LaTeX's text mode when ``hasattr(manim, "Text")`` is false, and
#: :mod:`algan.mobs.manim_compat` leaves them out of the compatibility
#: registry. (The *module* still imports either way -- ``Text`` is
#: ``Brace``'s default label class and ``Paragraph`` is ``Table``'s default
#: entry class, so ``mobject/text/text_mobject.py`` reaches manimpango through
#: the lazy proxies in ``manim/_pango.py``.)
from ._pango import available as _pango_available

PANGO_AVAILABLE = _pango_available()

if PANGO_AVAILABLE:
    from .mobject.text.text_mobject import *  # noqa: F401
