# The `algan.manim` boundary

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
