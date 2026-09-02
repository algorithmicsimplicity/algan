"""The opt-in Manim shape-style profile behind ``SETTINGS.style.shape_style_profile``.

Algan's built-in shapes carry their own default styling (a ``Square`` is a red
filled circuit with a wide white border; Manim's is an unfilled white outline of
stroke width 4). Setting ``SETTINGS.style.set(shape_style_profile="manim")``
asks Algan's *native* shape classes to adopt Manim Community's constructor
defaults instead; setting it back to ``"algan"`` restores Algan's own.

The numbers are never written down here. Enabling the profile constructs each
mapped Manim class once, reads ``fill_color`` / ``fill_opacity`` /
``stroke_color`` / ``stroke_opacity`` / ``stroke_width`` off the instance, and
caches the snapshot -- so the values cannot drift from whatever ``manim``
package is installed (or vendored). A class missing from that package is
skipped rather than raised, leaving that shape on Algan's defaults.

The snapshot feeds three default-resolution sites -- the places Algan already
chooses a shape's defaults:

- :func:`algan.mobs.shapes_2d._translate_vector_style_kwargs` for every
  bezier-circuit shape (``Square``, ``Circle``, ``Triangle``, ``Polygon``,
  ``Line``, ...),
- the fill-color fallback in
  :meth:`algan.mobs.surfaces.surface.Surface.__init__` for the curved solids
  (``Sphere``, ``Cylinder``, ``Cone``, ``Torus``),
- the face styling in
  :meth:`algan.mobs.shapes_3d.Polyhedron.__init__` for the flat-sided solids
  (``Prism``, ``Cube``).

An explicit keyword from the caller always wins over the profile at each site.

Everything in this module is internal; the user-facing switch is the setting
itself.
"""

from __future__ import annotations

import torch

from algan.utils.lazy_import import LazyModule

# Deferred exactly like algan.mobs.manim_mob: enabling the profile is the only
# thing that pays manim's import, and ``import algan`` must not.
_manim = LazyModule("manim", extras=("algan.utils.manim_svg_cache",))

#: Algan shape class name -> Manim class name whose constructor defaults the
#: profile adopts. Most names are identical; the mapping stays explicit so a
#: future divergence has somewhere to live. Matched against the most-derived
#: class of the Mob being constructed.
_MANIM_SHAPE_STYLE_CLASSES = {
    # Bezier-circuit shapes.
    "Square": "Square",
    "Rectangle": "Rectangle",
    "Circle": "Circle",
    "Triangle": "Triangle",
    "RegularPolygon": "RegularPolygon",
    "Polygon": "Polygon",
    "Line": "Line",
    "Dot": "Dot",
    # Algan's current Star and Arrow are Manim-compatibility wrappers, which
    # already carry these defaults by construction; the entries are latent and
    # light up only if a native Algan class of that name ever exists.
    "Star": "Star",
    "Arrow": "Arrow",
    # Curved solids (Surface-backed).
    "Sphere": "Sphere",
    "Cylinder": "Cylinder",
    "Cone": "Cone",
    "Torus": "Torus",
    # Flat-sided solids (Polyhedron-backed).
    "Cube": "Cube",
    "Prism": "Prism",
}

#: Positional arguments a mapped Manim constructor requires. Anything absent
#: from this table is constructed bare.
_MANIM_SHAPE_CONSTRUCTION_ARGS = {
    "Line": lambda mn: (mn.LEFT, mn.RIGHT),
    "Arrow": lambda mn: (mn.LEFT, mn.RIGHT),
    "Polygon": lambda mn: (mn.LEFT, mn.RIGHT, mn.UP),
}

#: Manim fills below this opacity count as absent -- the same rule ManimMob
#: applies when deciding a converted VMobject's ``filled``.
_VISIBLE_OPACITY_THRESHOLD = 1e-5

# One snapshot per shape name, resolved on first need and kept for the process:
# manim's version cannot change mid-session, and constructing its classes is
# not free. Values are the dicts built by ``_resolve_shape_style``, or None for
# a shape that was skipped.
_SHAPE_STYLES: dict[str, dict | None] | None = None


def _to_algan_color(manim_color, opacity=None):
    """Convert a Manim color to an Algan Color, folding in an opacity."""
    from algan.constants.color import Color

    rgba = manim_color.to_rgba()
    alpha = float(rgba[3])
    if opacity is not None:
        alpha *= float(opacity)
    return Color([float(c) for c in rgba[:3]], glow=0, opacity=alpha)


def _read_opacity(value):
    """Read a (possibly tensor) Manim opacity as one float."""
    return float(torch.as_tensor(value).max().item())


def _resolve_shape_style(name, manim_name):
    """Snapshot one shape's Manim constructor defaults.

    Returns None -- rather than raising -- when the class is missing from the
    installed manim, fails to construct, or does not expose style attributes;
    the shape simply keeps Algan's defaults under the profile.
    """
    manim_cls = getattr(_manim, manim_name, None)
    if manim_cls is None:
        return None
    try:
        args_factory = _MANIM_SHAPE_CONSTRUCTION_ARGS.get(name)
        instance = (
            manim_cls(*args_factory(_manim))
            if args_factory is not None
            else manim_cls()
        )
        fill_color = getattr(instance, "fill_color", None)
        fill_opacity = getattr(instance, "fill_opacity", None)
        stroke_color = getattr(instance, "stroke_color", None)
        stroke_opacity = getattr(instance, "stroke_opacity", None)
        stroke_width = getattr(instance, "stroke_width", 0)

        fill_opacity_value = (
            _read_opacity(fill_opacity) if fill_opacity is not None else None
        )
        stroke_width_value = float(stroke_width) if stroke_width is not None else 0.0

        checker_color = None
        checkerboard = getattr(instance, "checkerboard_colors", None)
        if checkerboard:
            try:
                if len(checkerboard) > 1:
                    checker_color = _to_algan_color(checkerboard[1], 1.0)
            except TypeError:
                checker_color = None

        return {
            "color": (
                _to_algan_color(fill_color, fill_opacity_value)
                if fill_color is not None
                else None
            ),
            "stroke_color": (
                _to_algan_color(stroke_color, stroke_opacity)
                if stroke_color is not None
                else None
            ),
            # Manim's OWN unit, deliberately unconverted: the snapshot is
            # cached at profile-enable time, and
            # ``manim_stroke_width_ratio`` can change afterwards (that is what
            # ``use_manim_defaults`` does), so a converted value would go
            # stale. ``_manim_shape_style_for`` converts on the way out.
            "stroke_width_manim": stroke_width_value,
            "filled": (
                fill_opacity_value is not None
                and fill_opacity_value > _VISIBLE_OPACITY_THRESHOLD
            ),
            "fill_opacity": fill_opacity_value,
            "checker_color": checker_color,
        }
    except Exception:
        return None


def _ensure_manim_shape_styles() -> dict[str, dict | None]:
    """Resolve every mapped shape once and cache the snapshots."""
    global _SHAPE_STYLES
    if _SHAPE_STYLES is None:
        _SHAPE_STYLES = {
            name: _resolve_shape_style(name, manim_name)
            for name, manim_name in _MANIM_SHAPE_STYLE_CLASSES.items()
        }
        skipped = sorted(name for name, style in _SHAPE_STYLES.items() if style is None)
        if skipped:
            from algan.logging.logger import get_logger

            get_logger().debug(
                "Manim shape-style profile: no %s default available for %s; "
                "those shapes keep Algan's defaults.",
                getattr(_manim, "__name__", "manim"),
                ", ".join(skipped),
            )
    return _SHAPE_STYLES


def _warm_manim_shape_style_cache():
    """Resolve the snapshots now, so enabling pays the manim import itself."""
    _ensure_manim_shape_styles()


def _manim_shape_style_for(shape_cls):
    """The cached Manim style snapshot for ``shape_cls``, or None.

    None unless the profile is switched on
    (``SETTINGS.style.shape_style_profile == "manim"``), or when the class is
    not one of the mapped shapes. Called at the three default-resolution sites,
    always with an explicit keyword still winning over whatever it returns.
    """
    from algan.settings import SETTINGS

    if SETTINGS.style.shape_style_profile != "manim":
        return None
    style = _ensure_manim_shape_styles().get(shape_cls.__name__)
    if style is None:
        return None
    # The cached snapshot holds Manim's own stroke width; convert it here, so
    # the live ratio is read on every construction rather than frozen into the
    # cache. Copied rather than mutated -- the snapshot is shared.
    return {
        **style,
        "stroke_width": (
            style["stroke_width_manim"] / SETTINGS.style.manim_stroke_width_ratio
        ),
    }
