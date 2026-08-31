# API audit: `Animatable.__init__`, `Mob.__init__`, `Surface`, `BezierCircuitCubic`

Audit of the four core building blocks, against `DOCSTRINGS.md` and against what the code
actually does. Every claim below was checked by reading the source and, where marked
**verified**, by running it.

All four names are in `algan.__all__`, so all of it is Tier 1 by `DOCSTRINGS.md` §1.

Severity key: **BUG** — the code does something other than what is documented, or nothing at
all. **UX** — it works, but the user gets it wrong or cannot find it. **DOC** — the docstring
does not meet §4/§5/§6/§9.

---

## 0. The question you asked: can `Surface`'s Manim argument names be dropped?

**Yes — every one of `fill_color`, `fill_opacity`, `checkerboard_colors`, `stroke_color`,
`stroke_width`, `should_make_jagged`, `surface_piece_config`,
`pre_function_handle_to_anchor_scale_factor` and `func` can go.** There is no live consumer
that needs them on `Surface` itself. Details, since two of them have a wrinkle:

**They are a second, redundant compatibility layer.** `algan.manim.Surface` is not this class —
`_WRAPPED_MANIM_CLASS_NAMES` in `algan/mobs/manim_compat.py` wraps *vendored Manim's* `Surface`
and imports the result through `ManimMob`. The native curved shapes have their own translator,
`_surface_resolution_kwargs` in `algan/mobs/shapes_3d.py:149`, which already maps
`fill_color`→`color`, `fill_opacity`→`opacity`, `checkerboard_colors`→`color`/`checkered_color`,
and **strips** `stroke_*`, `should_make_jagged`, `surface_piece_config` and
`pre_function_handle_to_anchor_scale_factor` before they ever reach `Surface.__init__`. So for
`Sphere`, `Cone` and `Cylinder`, the copies inside `Surface.__init__` are dead code that never
runs. This is exactly the arrangement `shapes_2d._translate_vector_style_kwargs` uses for the
Bezier family, and it is the right one: the compatibility spelling lives in the shape
constructors, the base class stays native.

Three of them are inert wherever they are reached from:

| Argument | What it does today |
| --- | --- |
| `should_make_jagged` | stored on `self`, read by nothing in the package |
| `pre_function_handle_to_anchor_scale_factor` | stored on `self`, read by nothing |
| `surface_piece_config` | stored on `self`, read by nothing |
| `stroke_color`, `stroke_width` | stored on `self`, read by nothing — a `Surface` has no stroke. Silent no-op |

The two with a wrinkle:

- **`func`** is real plumbing, but only for `Surface`'s own Manim-style two-argument path, and it
  carries the bug in §3.1 below. If you want to keep a `func(u, v)` spelling, it belongs in the
  Manim layer, not here.
- **`u_range` / `v_range` / `resolution` must stay.** They are not compatibility aliases any
  more: `Sphere`, `Cone`, `Cylinder` and `Torus` pass `u_range`/`v_range` straight into
  `Surface.__init__`, which owns them, and they are documented native API on those classes (in
  radians, per `DOCSTRINGS.md` §4.3). `resolution` is likewise reached natively. Keep all three;
  they just need documenting on `Surface` (§3.2).

Dropping the rest removes 8 of `Surface.__init__`'s 26 parameters and deletes the
`if manim_function is not None:` styling branch that causes §3.1.

**One more to retire while you are there:** `resolution_shrink_margin` is documented as
"Deprecated compatibility argument", is still validated with a `ValueError`, and is read at
`surface.py:1859` inside `_select_auto_resolution` — which is gated on `_auto_resolution_enabled`,
hard-coded `False` at line 1101. It is unreachable. Per `DOCSTRINGS.md` §12 (no compatibility
aliases in private beta) it should be deleted outright rather than kept as a documented no-op.

---

## 1. `Animatable.__init__`

```python
def __init__(self, scene=None, add_to_scene=True, name="_", init=True,
             animation_manager=None, data_sub_inds=None,
             parent_batch_sizes=None, is_primitive=False)
```

**1.1 UX/DOC — three internal parameters are documented as user-facing.** `data_sub_inds`,
`parent_batch_sizes` and `is_primitive` are batching and render-registration internals; their
docstring entries talk about "shared attribute rows" and "expanded for this animatable's
attributes". `DOCSTRINGS.md` §13 names this exact case as a current anti-pattern ("Internal
mechanics in a user docstring — `Animatable` class docstring's `data_sub_inds` /
`parent_batch_sizes` entries"). Recommend: make them keyword-only and underscore them
(`_data_sub_inds`, `_parent_batch_sizes`, `_is_primitive`), or keep the names and mark the
entries `Internal:` per §2 Tier 2. `is_primitive` is also written as a plain attribute by half a
dozen mob constructors, so it may be simplest to drop it from the signature entirely.

**1.2 DOC — no default is stated for any of the eight parameters.** §4.2 requires each one in
prose. `scene=None` in particular means "the active Scene", which is the single most important
thing this constructor does and is not written down anywhere in the entry.

**1.3 UX — `name` is vestigial.** It is assigned and then read by nothing: there is no lookup by
name, no use in `__repr__`, and no use in any error message. Its docstring is "The name of this
animatable", which tells a reader it is for something. Either give it a job (it would improve the
`AttributeError` messages that `_MANIM_METHOD_HINTS` already words carefully) or delete it. The
default `"_"` is a placeholder that should not be user-visible either way.

**1.4 UX/DOC — `init` is circular and dangerous.** "Whether this animatable should be
initialized" says nothing; what it controls is whether `on_init()` and the context's `on_init`
hook run. Passing `init=False` leaves a half-built object with no documented way to finish it
(the answer is `mob.init()`, which is documented on the *method* but not linked from here).
Recommend renaming to `run_init_hooks` and cross-referencing `Animatable.init`, or making it
private — no user script in the repo, docs or tests passes it.

**1.5 UX/DOC — `add_to_scene` is the most useful flag here and the least explained.** It is used
throughout the test suite for building Mobs that must not render, and the class comment at
`animatable.py:292` explains the real contract (a composite must make the same decision for its
parts) far better than the docstring entry does. The user-facing meaning — "construct this Mob
but keep it out of the render; use it for morph targets, layout templates, and geometry you will
attach to something else" — should be the entry, with `Defaults to True`. Its interaction with
an explicit `scene=` (the Scene is still bound; only registration is skipped) is undocumented.

**1.6 DOC — the class summary is one line of nothing followed by four sentences of timeline
internals.** "Base class for anything that needs animation" does not restate the name (good) but
does not say what the object *is* either, and the paragraph after it is `AnimationTimeline`,
`TimelineManager`, per-attribute buffers keyed by `id`, and `Lifespan` — §1's "internal mechanics
do not belong in a user-facing docstring". There is no `Examples` section (§9). The `Attributes`
section documents only `animatable_attrs`.

---

## 2. `Mob.__init__`

```python
def __init__(self, location=ORIGIN, basis=squish(torch.eye(3)), color=None,
             opacity=1, glow=0, *args, **kwargs)
```

**2.1 BUG (verified) — `mob.scale = 2` silently does nothing.** The class docstring's first
paragraph says "Mobs posses the animatable attributes location, basis (orientation), scale, and
color". There is no `scale` attribute: the animatable one is `scale_coefficient` and `scale` is a
*method*. Assigning `mob.scale = 2` shadows the bound method with an `int` on the instance, no
error, no visible change — and it also breaks every later `mob.scale(...)` call on that object.
Verified: `Square().scale = 2` leaves `type(m.scale) is int`.

This is the single worst UX defect in the audit, because the docstring actively teaches the
wrong name. Fix at minimum by correcting the sentence to name `scale_coefficient` and pointing
at `Mob.scale()`; better, add a `scale` property that forwards to `scale_coefficient`, or a
`__setattr__` guard that raises with the same style as `_MANIM_METHOD_HINTS`.

**2.2 BUG (verified) — a sixth positional argument lands in `Animatable.scene`.** `*args` sits
between `glow` and `**kwargs` and is forwarded to `Animatable.__init__`, whose first parameter is
`scene`. `Mob(loc, basis, color, 1, 0, x)` therefore binds `x` as the Scene and fails deep inside
with `AttributeError: 'str' object has no attribute 'get_new_id'`. Nothing passes positional args
through `Mob` — make everything after `location` keyword-only (`def __init__(self, location=ORIGIN,
*, basis=..., color=None, opacity=1, glow=0, **kwargs)`).

**2.3 UX — `basis` is a flattened 9-vector.** Constructing one by hand requires
`squish(torch.eye(3))`, and `squish` is not a name a script author should need. The docstring
explains the row convention well, but the shape is a serialization detail of the timeline
leaking into the constructor. Recommend accepting a `(*, 3, 3)` tensor as well and flattening
internally — a one-line `if value.shape[-2:] == (3, 3)` in the cast — and saying so in the entry.
Consider whether the user-facing spelling should be `orientation`.

**2.4 DOC — defaults missing on three of five parameters** (§4.2): `color` says "If None, it uses
the default color" without naming the literal or what a subclass's default actually is;
`opacity` and `glow` state no default at all. `glow` additionally has **no unit and no range** —
"The glow intensity of the Mob" is the whole entry (§4.3 makes units mandatory).

**2.5 DOC — `**kwargs` does not name the kwargs users actually reach for** (§4.5). "Passed to
`Animatable` base class" should be "Passed to `Animatable` — notably `scene` and
`add_to_scene`", since those two are the reason anyone touches the passthrough.

**2.6 DOC — no `Attributes` section** (§10), although `location`, `basis`, `color`, `opacity`,
`glow` and `scale_coefficient` are exactly the assignment targets the animation system exists
for. The excellent `#:` comments on `two_sided`, `closed_shell`, `casts_shadows`,
`receives_shadows` cover the render flags, but the class docstring never mentions that this
family exists or that all of them must be set before `spawn()`.

**2.7 DOC — no annotations on the return, and "Moveable" is a typo** for "Movable" in the summary.

---

## 3. `Surface`

26 constructor parameters; **12 of them are undocumented**, two documented defaults are wrong,
and one taught native feature is missing from the docstring entirely.

**3.1 BUG (verified) — an explicit `color=` is silently overwritten on the Manim-function path.**
At `surface.py:1071`, inside the `if manim_function is not None:` branch:

```python
kwargs["color"] = checkerboard_colors[0]   # not setdefault
```

`checkerboard_colors` has just been defaulted to `[BLUE_D, BLUE_E]` three lines earlier, so
`Surface(func=f, color=RED)` renders `BLUE_D`. Verified: the constructed Mob's colour comes back
`(0.161, 0.671, 0.792)` — `BLUE_D` — not `RED`. The only escape is the undiscoverable
`checkerboard_colors=False`. Deleting the Manim styling branch (§0) removes this.

**3.2 BUG (verified) — a native `coord_function` with an optional second parameter is silently
reinterpreted as Manim's `func(u, v)`.** `_looks_like_manim_surface_function` (line 165) sniffs
*any* callable with two or more positional parameters, or `*args`, and reroutes it. So

```python
def ripple(uv, amplitude=0.3): ...
Surface(coord_function=ripple)
```

is called as `ripple(u, v)` and dies with `TypeError: 'float' object is not subscriptable` from
inside the user's own function body — no mention of `Surface`, `func`, or what happened.
Verified. The heuristic is guessing at the caller's intent from a signature; with `func` gone it
disappears with it. If a `func(u, v)` spelling is kept anywhere, it must be opt-in by keyword
only.

**3.3 BUG — two documented defaults are wrong.** The class docstring says `render_tolerance`
"Defaults to ``0.001``" (signature: `0.0005`) and `render_tolerance_pixels` "Defaults to
``1.0``" (signature: `0.5`). Verified by `inspect.signature`. Both are user-visible quality/cost
knobs, so a factor-of-two error in the prose is a real support problem.

**3.4 DOC — 12 undocumented parameters**, against `DOCSTRINGS.md` §4 ("every parameter in the
signature gets an entry, in signature order"): `checkered_color`, `ignore_normals`, `func`,
`u_range`, `v_range`, `resolution`, `surface_piece_config`, `fill_color`, `fill_opacity`,
`checkerboard_colors`, `stroke_color`, `stroke_width`, `should_make_jagged`,
`pre_function_handle_to_anchor_scale_factor`. After §0's removals, five of these remain and all
five need entries.

**`checkered_color` is the sharp one**: it is a genuine native feature, it is taught in
`docs/source/new_user_tutorials/three_d_basics.rst:147` and used in the mob gallery, and it
appears nowhere in the class docstring. A user who sees it in the tutorial has no reference page
to look it up on. `ignore_normals` is likewise real (read by `render_loop.py:1683`) and
undocumented.

**3.5 DOC — no annotations at all on `__init__`** (§4.1), so with `autodoc_typehints =
"description"` the rendered reference shows no types for any Surface parameter. Same defect
`DOCSTRINGS.md` §13 records for `Scene.save_video`.

**3.6 DOC — no `Examples` and no `Animation` section** on a Tier-1 class (§9, §6), despite
`Surface` being the class users subclass to make their own geometry.

**3.7 UX — `set_shape_to` is the only `set_*` on the class that does not return `self`.**
`set_location_by_function`, `set_color_by_function`, `set_color_by_image`,
`set_fill_by_checkerboard` and `set_fill_by_value` all chain; `set_shape_to` returns `None`, so
`surf.set_shape_to(other).move(UP)` raises. Its docstring also has no `Returns` section, no
`Animation` section (it *is* recorded — it wraps a `Sync`), and third-person phrasing
("Changes this surface's...") against §3's imperative rule.

**3.8 UX — three Manim-named public methods sit on the native class**, the same question as §0
one level up:

- `func(u, v)` — raises `AttributeError` unless the surface was built the Manim way. A public
  method that is usually an error is worse than no method.
- `set_fill_by_checkerboard(*colors, opacity=None)` — "fill" is not an Algan word for a surface
  (a `Surface` has no fill/stroke distinction; that is the Bezier family's vocabulary). Native
  spelling would be `set_checkerboard_colors`. One-line docstring, no `Parameters`, no
  `Animation`, no `Returns` — though it does return `self`.
- `set_fill_by_value(axes, colorscale=None, axis=2, **kwargs)` — the first parameter is a Manim
  `Axes` object, undocumented; `axes` (an object) and `axis` (an int) are a near-homonym pair one
  character apart in the same signature; and it swallows a `colors=` alias out of `**kwargs`.
  Native spelling would be `set_color_by_axis(axis=2, colorscale=..., axes=None)`, or it could
  fold into `set_color_by_function`, which already does the general case.

**3.9 UX — public-named internals.** `get_render_primitives` and
`clear_geometry_resolution_cache` are renderer/cache plumbing on a class whose every other public
method is authoring API. §2 Tier 2's advice applies: prefer renaming with a leading `_`.

**3.10 DOC — the three tolerance properties speak in internals.** "Construction-time absolute
world-space PN fitting tolerance" and "Per-frame flat-triangle tessellation tolerance, as a
screen fraction" assume the reader knows what a PN triangle is and that dicing happens per
frame. The *constructor* entries for the same three values are excellent — the properties should
be one plain sentence each plus a `See Also` back to the class docstring, not a compressed
restatement in jargon.

**3.11 DOC — `vertices` setter is undocumented.** The property documents reading; assigning is
mentioned in one clause ("Writing it moves the surface's vertices") with no statement of the
required shape, what happens on a mismatch, or that the grid resolution is fixed at construction
so a differently sized write cannot work.

---

## 4. `BezierCircuitCubic`

The class docstring is the strongest of the four — the colour/texture-grid explanation and the
`z_index` entry are model Tier-1 writing. The defects are concentrated in the parameters and the
undocumented methods.

**4.1 BUG (verified) — `portion_of_curve_drawn` does not exist.** It is:

- a constructor parameter, `portion_of_curve_drawn=1.0` (line 403);
- registered as animatable, `register_attrs_as_animatable(["stroke_width",
  "portion_of_curve_drawn"], ...)` (line 450);
- documented in the class docstring as "How much of the path is drawn, from 0 (nothing) to 1 (all
  of it). **Animating it is what draws a shape on.** Defaults to ``1.0``."

The constructor never assigns it, and nothing in the package reads it — the renderer has no such
input. Verified: `Square(portion_of_curve_drawn=0.3).portion_of_curve_drawn` raises
`AttributeError: 'Square' object has no attribute 'portion_of_curve_drawn'`. The one write in
the tree, `algan/animations/indication.py:770`, therefore also does nothing.

Drawing-on is really done by `set_control_points_to_partial` / `draw` (§4.2). So the docstring
points users at a parameter that is silently discarded, and away from the two methods that
actually work. Either wire the attribute up to the primitive, or delete it from the signature
and the registration and rewrite the entry to name `draw()`.

**4.2 DOC — the two methods that do the drawing have no docstrings at all.**
`draw(t=1.0)` (line 1234) is an `@animated_function` — a recorded, user-shaped API — with zero
documentation. `set_control_points_to_partial(full_control_points, start_t, end_t)` (line 1302)
likewise, and it raises three distinct `ValueError`s a caller can trigger. `draw` should get the
full §2 treatment (it is the honest answer to §4.1); `set_control_points_to_partial` is only ever
called from `algan/animations/`, so it should be `_set_control_points_to_partial`.

**4.3 UX/DOC — `stroke_color`'s property docstring contradicts the class docstring.** The class
says "Color of the border stroke. Defaults to ``WHITE``"; the property says "Per-vertex colors
sampled across the circuit's border texture grid." Both are true, but the property is written
from the implementation's side and is what a user sees in `help()` and in the reference. Reading
it back gives a `(N, 5)` tensor, not the `Color` that was assigned — §5's rule that a getter must
state its convention. No `Animation` section on a recorded, animatable attribute (§6).

**4.4 UX (verified) — `control_points` must be a tensor, and says so nowhere.** The constructor
calls `control_points.view(...)` on line 414 before anything casts, so a list of points raises
`AttributeError: 'list' object has no attribute 'view'`. Every other geometry entry point in
Algan runs `cast_to_tensor` first, and the docstring's "shape ``(*, 3)`` in world units" reads
like any of them. One `cast_to_tensor` call fixes it.

**4.5 UX — `BezierCurveCubic` is exported with no docstring.** It is in `algan.__all__`, it is a
two-line subclass that sets `filled=False`, and `__doc__` is `None` — so it renders in the
reference as a bare name. It also inherits `BezierCircuitCubic`'s class docstring semantics
(closed loop) which is precisely what it is not.

**4.6 DOC — `get_default_color` and `get_animatable_attrs` have no docstrings**, while
`Surface.get_default_color` has one. `get_animatable_attrs` is internal and should be `_`-prefixed
(same as §3.9); `get_default_color` is a documented subclass hook elsewhere and should match.

**4.7 DOC — no annotations on `__init__`** (§4.1), same rendering consequence as §3.5.

---

## Suggested order of work

1. **§4.1 and §2.1** — two documented attributes that do not exist. Both actively teach a wrong
   API; both are cheap to fix.
2. **§3.1, §3.2, §2.2, §4.4** — four silent failures (wrong colour, opaque `TypeError`, argument
   landing in `scene`, `AttributeError` on a list). All four are removed or fixed by small,
   local changes.
3. **§0** — drop the eight Manim aliases and `resolution_shrink_margin` from `Surface.__init__`.
   This is the largest single readability win: 26 parameters down to 17, and it deletes the
   branch behind §3.1 and the heuristic behind §3.2. Check `tests/unit_tests/test_ux_regressions.py`
   and the Manim-parity suites first; `_surface_resolution_kwargs` already covers the shapes.
4. **§3.3, §3.4, §2.4, §1.2** — the documentation defects that mislead about values: wrong
   defaults, missing defaults, missing parameters, missing units.
5. **§3.8, §3.9, §4.2, §4.6** — naming: rename the Manim-named methods to native spellings,
   underscore the internals, document or privatize `draw` / `set_control_points_to_partial`.
6. **§1.1, §1.3, §1.4, §1.5** — the `Animatable` signature: internals out of the public
   docstring, `name` decided one way or the other, `init` and `add_to_scene` written for users.
7. **§3.5, §3.6, §4.7, §2.7, §1.6** — annotations, `Examples`, `Animation` sections.

## Cross-cutting inconsistency found on the way

`Torus.__init__` sets `grid_width = resolution[0]` while `_surface_resolution_kwargs` (used by
`Sphere`, `Cone`, `Cylinder`) sets `grid_width = resolution[0] + 1`, because Manim's `resolution`
counts patches and Algan's grid counts vertices. Verified: `Sphere(resolution=(8, 8))` gets a
9x9 grid and `Torus(resolution=(8, 8))` gets 8x8. Same keyword, same family of shapes, different
meaning. `Torus` should route through `_surface_resolution_kwargs` like its siblings — which
would also give it the `fill_color` / `checkerboard_colors` handling it currently lacks (today it
falls through to `Surface`'s copies, the ones §0 proposes deleting, so this is a prerequisite
for that removal).
