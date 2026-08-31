# The `algan.manim` boundary

`algan.manim` wraps **every** Manim class, natives included, so `mn.Sphere` exists beside
Algan's `Sphere`. The rule is uniform: **a name in `mn.` follows Manim's conventions; the same
name at the root follows Algan's.** Three conventions actually differ, and all three are
converted at that boundary and nowhere else:

| | root (Algan) | `mn.` (Manim) |
| :--- | :--- | :--- |
| Angles | degrees — `Arc(angle=90)` | radians — `mn.Arc(angle=PI/2)` |
| Stroke width | `Arrow(stroke_width=4)` | twice that — `mn.Arrow(stroke_width=8)` |
| z axis | `OUTWARD` is `-z` | `OUT` is `+z` (`Scene.manim_coordinates`) |

99 compat-only classes (`Axes`, `Brace`, `Table`, `Arrow`, ...) have no native implementation,
so `algan/mobs/manim_adapters.py` gives a curated subset a root spelling that converts and
delegates. Adding one is a table entry, not a constructor: `_ADAPTED` lists the classes and
`_ANGLE_PARAMS` the angle arguments; stroke width is doubled for every adapter unconditionally,
because a Manim class accepts `stroke_width` whether or not its signature names it. A class
with a native implementation must stay out of `_ADAPTED` — the module asserts this, since two
root spellings of one thing is what the boundary exists to prevent.

The `/2` stroke conversion exists in exactly four places, all of which genuinely straddle the
boundary: `manim_compat` (export), `manim_mob` (import), `manim_adapters` (the root spellings),
and `shape_style_profiles` (reading Manim's own constructor defaults). Native classes take
Algan's unit end to end.