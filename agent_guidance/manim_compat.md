# The `algan.manim` boundary

## There is exactly one Manim, and Algan ships it

`manim` is **not** a dependency. `algan/external_libraries/manim/` is Manim Community's
geometry subset — the Mobject graph, the Bezier and SVG/LaTeX machinery, the shape,
graphing, text and 3-D classes — with the animations, scenes, cameras, renderers, CLI and
plugin system left out, because Algan supplies all of those. It is generated, not
hand-maintained: `scripts/vendor_manim.py` rebuilds it from an upstream sdist and
`algan/external_libraries/manim/VENDORING.md` records the version, the cut, and every edit
made to upstream source. **Do not hand-edit anything under that directory** — change the
script and re-run it, or the next bump silently drops your fix.

`algan/external_libraries/manim_alias.py` registers it as `manim` before any Mob module is
imported, through a `sys.meta_path` finder rather than a bare `sys.modules["manim"]` entry.
That matters: with only the package aliased, `from manim.mobject.svg.brace import BraceText`
re-executes `brace.py` under the second name and defines a *second* `BraceText`, and every
`isinstance` across that seam answers `False`. If you are adding a first-party `import
manim...`, it is already correct; if you are debugging one, `manim.X is
algan.external_libraries.manim.X` is the invariant, and
`tests/unit_tests/test_manim_mobject_parity.py` pins it.

Two groups of Mobjects are absent, and both are declared in `manim_parity.py` rather than
silently missing. `Typst`/`MathTypst` (`MANIM_UNVENDORED_MOBJECT_NAMES`) are not vendored at
all. `Text`/`MarkupText`/`Paragraph` (`MANIM_PANGO_MOBJECT_NAMES`) need `manimpango`, which
is the optional `algan[pango]` extra because it publishes no Linux wheel — without it the
vendored package withholds those three names, `algan.Text` falls back to LaTeX's text mode,
and `manim_compat` wraps what is there. **`hasattr(mn, "Text")` is the test, never `import
manimpango`**: the module `text_mobject` imports fine either way (`Text` is `Brace`'s default
label class), it is the export that is conditional.


`algan.manim` wraps **every** Manim class, natives included, so `mn.Sphere` exists beside
Algan's `Sphere`. The rule is uniform: **a name in `mn.` follows Manim's conventions; the same
name at the root follows Algan's.** Two conventions actually differ, and both are
converted at that boundary and nowhere else:

| | root (Algan) | `mn.` (Manim) |
| :--- | :--- | :--- |
| Angles | degrees — `Arc(angle=90)` | radians — `mn.Arc(angle=PI/2)` |
| Stroke width | `Arrow(stroke_width=4)` | twice that — `mn.Arrow(stroke_width=8)` |

The z axis used to be a third row. It is not any more: Algan's `OUTWARD` is `+z`, the same
as Manim's `OUT`, so an imported point keeps the numbers it was written with. There is no
`Scene.manim_coordinates` and no `from_manim_coordinates`/`to_manim_coordinates` — they
existed only to mirror what no longer needs mirroring. See `mobs_geometry.md`.

**Names are converted at the same boundary.** Manim's `mobject=` is `mob=` at the root and
`element_to_mobject=` is `element_to_mob=`; the five classes Manim spells after `Mobject`
are `SVGMob`, `MobMatrix`, `MobTable`, `DashedMob` and `CurvesAsChildren`. Every Manim
spelling raises `AlganConfigurationError` at the root naming the Algan one, and every one of
them still works verbatim under `mn.`. `algan/utils/api_renames.py` holds the table and both
mechanisms (`_reject_renamed_keywords`, the `@_renamed_keywords` decorator), which is also
where the "that looks like radians" warning lives.

## An adapter carries its own signature and docstring

Delegating handed the root spellings Manim's `__signature__` and Manim's docstring, which
then said the wrong thing in the one place a user looks: `help(Arc)` reported
`angle: float = 1.5707963267948966` for an argument this layer reads as degrees, and
`Brace(mobject: 'Mobject', direction: 'Vector3D')` named a keyword and two types Algan does
not have. `_root_signature` and `_root_docstring` in `manim_adapters.py` build both:

- angle defaults are restated in degrees, and `stroke_width`'s in Algan's unit;
- annotations survive only if they name a plain builtin, so no Manim type alias is displayed;
- Manim's prose is **replaced**, not appended to — a `.. manim::` block is Manim scene code
  that Algan's docs build would execute and render into Algan's own reference pages.

Those docstring bodies are *generated* — Manim's summary line plus the converted parameter
list, with each entry stating its unit and default rather than its meaning, and a `Notes`
section saying so. A hand-written Algan docstring goes in `_WRAPPER_DOCSTRINGS` in
`manim_compat.py` and wins over the generated one; `MathTex` and `Title` have them.

## Everything is adapted unless it says why not

`algan/mobs/manim_adapters.py` gives every wrapped Manim class a root spelling that converts
and delegates. The set is **computed, not curated** — `_ADAPTED` is the registry minus two
exclusion lists, so a class added to the compatibility layer is adapted automatically:

- `_NATIVE` — Algan implements it itself (`Sphere`, `Circle`, `Text`, ...). The native class
  keeps the root name; Manim's stays reachable as `mn.<name>`. Two root spellings of one thing
  is what the boundary exists to prevent.
- `_NOT_ADAPTED` — a Manim base class (`VMobject`, `TipableVMobject`), a container Algan spells
  differently (`VGroup`, `VDict`) or a non-Mob construct (`ValueTracker`). Each entry carries
  its reason.

`tests/unit_tests/test_manim_adapter_conventions.py` checks both lists against what the root
namespace actually resolves each name to, so a class that gains or loses a native
implementation fails there rather than silently shadowing — root `from algan import *` pulls
the adapters in *before* the native modules, so a stale `_NATIVE` entry would otherwise just
be overwritten without a word.

## Angle parameters are derived, not listed

`_ANGLE_PARAMS` is built by walking each Manim class's **whole MRO**, not its own signature.
That matters: `Sector` takes `angle` through `**kwargs` from `AnnularSector` and
`ArrowTriangleFilledTip` takes `start_angle` from `ArrowTriangleTip`, so a signature-driven
table misses them and leaves the root spelling reading degrees as radians.

Every parameter whose name looks like an angle, or whose default is a multiple of `pi/4`, must
appear in `_ANGLE_PARAM_NAMES` (converted) or `_NOT_ANGLE_PARAM_NAMES` (waived, with what it
actually is). One in neither **raises at import**, so a Manim upgrade that adds an angle cannot
land as a silent wrong conversion. Waivers are worth reading before adding one: `arc_center` is
a point, `azimuth_units` is a string, and `other_angle` is a *bool*.

`_NESTED_ANGLE_PARAMS` handles the one shape the detector cannot see into —
`ArcPolygon(arc_config={"angle": 90})`, where the angles are keys of a forwarded kwargs dict.

Stroke width is doubled for every adapter unconditionally rather than per class, because a
Manim class accepts `stroke_width` whether or not its signature names it (`Star` and
`DashedLine` take it through `**kwargs` to `VMobject`).

## Known upstream quirk

Vendored Manim's `Arrow` is not reproducible: two `manim.Arrow` instances built with identical
arguments in one process disagree from the second onwards, with no Algan code involved. Don't
write a parity test against `Arrow`; use `DashedLine` for `Line`-inherited parameters.

## The `/2` stroke conversion

It exists in exactly four places, all of which genuinely straddle the boundary: `manim_compat`
(export), `manim_mob` (import), `manim_adapters` (the root spellings), and
`shape_style_profiles` (reading Manim's own constructor defaults). Native classes take Algan's
unit end to end.
