"""Manim Community Mobject parity manifest.

The inventory is generated from all source-defined subclasses of ``Mobject``
in Manim Community v0.21.0 -- the release vendored under
``algan/external_libraries/manim`` -- including abstract helpers and
renderer-specific OpenGL classes.  It is kept explicit so future Manim releases
can be diffed without importing a separately installed Manim package at
runtime.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

MANIM_COMMUNITY_VERSION = "0.21.0"

#: Mobjects in the manifest's Manim release that Algan deliberately does not
#: carry, and why. Named rather than silently omitted, so the inventory still
#: adds up against upstream.
#:
#: ``Typst`` and ``MathTypst`` are the only two, and they are what v0.21.0
#: added over v0.20.1. They typeset through Typst rather than LaTeX, which
#: means a second document toolchain and the optional ``typst`` package for a
#: second spelling of what ``Tex``/``MathTex`` already produce -- and Algan
#: converts the Bezier output, not the markup. See ``VENDORING.md``.
MANIM_UNVENDORED_MOBJECT_NAMES = ("MathTypst", "Typst")

MANIM_MOBJECT_NAMES = (
    "AbstractImageMobject",
    "Angle",
    "AnnotationDot",
    "AnnularSector",
    "Annulus",
    "Arc",
    "ArcBetweenPoints",
    "ArcBrace",
    "ArcPolygon",
    "ArcPolygonFromArcs",
    "Arrow",
    "Arrow3D",
    "ArrowCircleFilledTip",
    "ArrowCircleTip",
    "ArrowSquareFilledTip",
    "ArrowSquareTip",
    "ArrowTip",
    "ArrowTriangleFilledTip",
    "ArrowTriangleTip",
    "ArrowVectorField",
    "Axes",
    "BackgroundRectangle",
    "BarChart",
    "Brace",
    "BraceBetweenPoints",
    "BraceLabel",
    "BraceText",
    "BulletedList",
    "Circle",
    "Code",
    "ComplexPlane",
    "ComplexValueTracker",
    "Cone",
    "ConvexHull",
    "ConvexHull3D",
    "Cross",
    "Cube",
    "CubicBezier",
    "CurvedArrow",
    "CurvedDoubleArrow",
    "CurvesAsSubmobjects",
    "Cutout",
    "Cylinder",
    "DashedLine",
    "DashedVMobject",
    "DecimalMatrix",
    "DecimalNumber",
    "DecimalTable",
    "DiGraph",
    "Difference",
    "Dodecahedron",
    "Dot",
    "Dot3D",
    "DotCloud",
    "DoubleArrow",
    "Elbow",
    "Ellipse",
    "Exclusion",
    "FullScreenRectangle",
    "FunctionGraph",
    "GenericGraph",
    "Graph",
    "Group",
    "Icosahedron",
    "ImageMobject",
    "ImageMobjectFromCamera",
    "ImplicitFunction",
    "Integer",
    "IntegerMatrix",
    "IntegerTable",
    "Intersection",
    "Label",
    "LabeledArrow",
    "LabeledDot",
    "LabeledLine",
    "LabeledPolygram",
    "Line",
    "Line3D",
    "ManimBanner",
    "MarkupText",
    "MathTable",
    "MathTex",
    "MathTexPart",
    "Matrix",
    "Mobject",
    "Mobject1D",
    "Mobject2D",
    "MobjectMatrix",
    "MobjectTable",
    "NumberLine",
    "NumberPlane",
    "Octahedron",
    "OpenGLAnnularSector",
    "OpenGLAnnulus",
    "OpenGLArc",
    "OpenGLArcBetweenPoints",
    "OpenGLArrow",
    "OpenGLArrowTip",
    "OpenGLCircle",
    "OpenGLCubicBezier",
    "OpenGLCurvedArrow",
    "OpenGLCurvedDoubleArrow",
    "OpenGLCurvesAsSubmobjects",
    "OpenGLDashedLine",
    "OpenGLDashedVMobject",
    "OpenGLDot",
    "OpenGLDoubleArrow",
    "OpenGLElbow",
    "OpenGLEllipse",
    "OpenGLGroup",
    "OpenGLImageMobject",
    "OpenGLLine",
    "OpenGLMobject",
    "OpenGLPGroup",
    "OpenGLPMPoint",
    "OpenGLPMobject",
    "OpenGLPoint",
    "OpenGLPolygon",
    "OpenGLRectangle",
    "OpenGLRegularPolygon",
    "OpenGLRoundedRectangle",
    "OpenGLSector",
    "OpenGLSquare",
    "OpenGLSurface",
    "OpenGLSurfaceGroup",
    "OpenGLSurfaceMesh",
    "OpenGLTangentLine",
    "OpenGLTexturedSurface",
    "OpenGLTipableVMobject",
    "OpenGLTriangle",
    "OpenGLVGroup",
    "OpenGLVMobject",
    "OpenGLVector",
    "OpenGLVectorizedPoint",
    "PGroup",
    "PMobject",
    "Paragraph",
    "ParametricFunction",
    "Point",
    "PointCloudDot",
    "PolarPlane",
    "Polygon",
    "Polygram",
    "Polyhedron",
    "Prism",
    "Rectangle",
    "RegularPolygon",
    "RegularPolygram",
    "RightAngle",
    "RoundedRectangle",
    "SVGMobject",
    "SampleSpace",
    "ScreenRectangle",
    "Sector",
    "SingleStringMathTex",
    "Sphere",
    "Square",
    "Star",
    "StealthTip",
    "StreamLines",
    "Surface",
    "SurroundingRectangle",
    "Table",
    "TangentLine",
    "TangentialArc",
    "Tetrahedron",
    "Tex",
    "Text",
    "ThreeDAxes",
    "ThreeDVMobject",
    "TipableVMobject",
    "Title",
    "Torus",
    "Triangle",
    "TrueDot",
    "Underline",
    "Union",
    "UnitInterval",
    "VDict",
    "VGroup",
    "VMobject",
    "VMobjectFromSVGPath",
    "ValueTracker",
    "Variable",
    "Vector",
    "VectorField",
    "VectorizedPoint",
    "_BooleanOps",
)

MANIM_OPENGL_MOBJECT_NAMES = (
    "OpenGLAnnularSector",
    "OpenGLAnnulus",
    "OpenGLArc",
    "OpenGLArcBetweenPoints",
    "OpenGLArrow",
    "OpenGLArrowTip",
    "OpenGLCircle",
    "OpenGLCubicBezier",
    "OpenGLCurvedArrow",
    "OpenGLCurvedDoubleArrow",
    "OpenGLCurvesAsSubmobjects",
    "OpenGLDashedLine",
    "OpenGLDashedVMobject",
    "OpenGLDot",
    "OpenGLDoubleArrow",
    "OpenGLElbow",
    "OpenGLEllipse",
    "OpenGLGroup",
    "OpenGLImageMobject",
    "OpenGLLine",
    "OpenGLMobject",
    "OpenGLPGroup",
    "OpenGLPMPoint",
    "OpenGLPMobject",
    "OpenGLPoint",
    "OpenGLPolygon",
    "OpenGLRectangle",
    "OpenGLRegularPolygon",
    "OpenGLRoundedRectangle",
    "OpenGLSector",
    "OpenGLSquare",
    "OpenGLSurface",
    "OpenGLSurfaceGroup",
    "OpenGLSurfaceMesh",
    "OpenGLTangentLine",
    "OpenGLTexturedSurface",
    "OpenGLTipableVMobject",
    "OpenGLTriangle",
    "OpenGLVGroup",
    "OpenGLVMobject",
    "OpenGLVector",
    "OpenGLVectorizedPoint",
)

MANIM_PRIVATE_MOBJECT_NAMES = ("_BooleanOps",)

MANIM_EXTERNAL_TOOL_MOBJECT_NAMES = (
    "BraceLabel",
    "BraceText",
    "BulletedList",
    "Code",
    "DecimalMatrix",
    "DecimalNumber",
    "DecimalTable",
    "Integer",
    "IntegerMatrix",
    "IntegerTable",
    "Label",
    "LabeledArrow",
    "LabeledDot",
    "LabeledLine",
    "LabeledPolygram",
    "MarkupText",
    "MathTable",
    "MathTex",
    "MathTexPart",
    "Matrix",
    "MobjectMatrix",
    "MobjectTable",
    "Paragraph",
    "SingleStringMathTex",
    "Table",
    "Tex",
    "Text",
    "Title",
    "Variable",
)


#: Mobjects the vendored Manim exports only when the optional ``manimpango``
#: is installed. All three are Pango-rendered text, all three have a native
#: Algan spelling, and ``algan.Text`` falls back to LaTeX's text mode without
#: them -- so a Pango-less install is a supported configuration rather than a
#: parity gap, and :func:`missing_manim_mobjects` does not report them.
MANIM_PANGO_MOBJECT_NAMES = ("MarkupText", "Paragraph", "Text")


def missing_manim_mobjects(namespace: Mapping[str, Any]) -> tuple[str, ...]:
    """Return v0.21.0 Mobject names absent from an Algan namespace.

    The Pango-only names (:data:`MANIM_PANGO_MOBJECT_NAMES`) are exempt when
    they are absent as a set, which is what a build without the optional
    ``manimpango`` looks like. One of them missing on its own is still a gap.
    """
    missing = tuple(name for name in MANIM_MOBJECT_NAMES if name not in namespace)
    if all(name in missing for name in MANIM_PANGO_MOBJECT_NAMES):
        missing = tuple(n for n in missing if n not in MANIM_PANGO_MOBJECT_NAMES)
    return missing


def validate_manim_mobject_parity(namespace: Mapping[str, Any]) -> None:
    """Raise a useful error if the supplied namespace is missing parity names."""
    missing = missing_manim_mobjects(namespace)
    if missing:
        raise RuntimeError(
            f"Missing {len(missing)} Manim {MANIM_COMMUNITY_VERSION} Mobjects: "
            + ", ".join(missing)
        )


__all__ = [
    "MANIM_COMMUNITY_VERSION",
    "MANIM_MOBJECT_NAMES",
    "MANIM_OPENGL_MOBJECT_NAMES",
    "MANIM_PANGO_MOBJECT_NAMES",
    "MANIM_PRIVATE_MOBJECT_NAMES",
    "MANIM_EXTERNAL_TOOL_MOBJECT_NAMES",
    "MANIM_UNVENDORED_MOBJECT_NAMES",
    "missing_manim_mobjects",
    "validate_manim_mobject_parity",
]
