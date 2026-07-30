# Docstring standard for Algan's user-facing API

This document defines how docstrings on Algan's **user-facing API** must be written. It is
prescriptive, not descriptive: much of the current codebase does not meet it yet (see
[Current state](#current-state-and-migration-order)). When you touch a public method, bring its
docstring up to this standard.

Audience for a user-facing docstring: **someone writing an animation script who has never read
Algan's source and never will.** They read it as a tooltip in an IDE, as `help(Mob.move)` in a REPL,
and as a rendered page in the Sphinx reference. It has to answer "what will this do to my scene, and
what do I type" without a single look at the implementation.

Internal mechanics (timeline rows, replay windows, arena pointers, kernel arg widths) do **not**
belong in user-facing docstrings. They belong in `#` comments, in `AGENTS_DETAILED.md`, or in the
docstrings of the internal objects themselves.

---

## 1. What counts as user-facing

Tier 1 — **full standard applies** (everything in this document):

- Any name exported by `from algan import *` (i.e. present in `algan.__all__`).
- Any public method or property on `Animatable`, `Mob` and its mixins, `Group`, `Scene`, mob
  classes (`Square`, `Text`, `Surface`, `ThreeDModelMob`, ...), `AnimationContext` subclasses
  (`Seq`, `Sync`, `Lag`, `Off`, `Audio`, `Speech`), `Camera`, lights, `Material` classes, and the
  `SETTINGS` sections and presets.
- Any function that appears in a tutorial, a docs example, or `README.md`.

Tier 2 — **summary line + Parameters/Returns where non-obvious**: public-named helpers that users
can reach but are not taught (`get_forward_basis`, `setattr_without_record`, `refresh_history`).
Prefer fixing the naming: if a method is not meant for users, rename it with a leading `_` rather
than writing a user-facing docstring for it. If it cannot be renamed (subclass hook, historical
name), say so in one line: `"""Internal: ..."""`.

Tier 3 — **no docstring required**: `_`-prefixed functions, dunders, kernel bodies, one-line
accessors whose name is a complete description (`is_spawned`, `get_children`) *provided* the return
type is annotated.

If you are unsure which tier something is in, check whether it has a stub in
`docs/source/reference/`. If it is in the rendered reference, it is Tier 1.

---

## 2. The canonical shape

NumPy style, parsed by `sphinx.ext.napoleon`. This is the full skeleton; omit sections that have
nothing to say, but keep this order:

```
Summary line: one imperative sentence, ends with a period.

Extended description: what it does in scene terms, what the geometry/semantics are,
and when to reach for this instead of the neighbouring method.

Animation
---------
Whether the call is recorded, its default duration, how to change it. (See §6.)

Parameters
----------
name
    What it means, what forms it accepts, its units, and its default.

Returns
-------
Type
    What comes back.

Raises
------
:class:`.SomeError`
    Under what condition.

See Also
--------
:meth:`~.Mob.related_method` : One-line reason a reader would go there instead.

Examples
--------
.. algan:: UniqueExampleName

    from algan import *

    ...
    Scene.save_video()
```

A worked Tier-1 exemplar — this is what `Mob.rotate` should look like (it currently has no
`Parameters` section at all, and does not state that its angle is in degrees):

```python
@animated_function(
    animated_args={"num_degrees": 0},
    unique_args=["axis", "about_point"],
)
def rotate(
    self,
    num_degrees: float | torch.Tensor,
    axis: torch.Tensor = OUT,
    about_point: torch.Tensor | None = None,
) -> Mob:
    """Rotate the Mob about an axis, optionally around a point in space.

    With the default ``about_point=None`` only the Mob's orientation changes and it
    stays where it is. Given an ``about_point``, the Mob also travels around the axis
    through that point, like a planet orbiting while spinning. To move around a point
    *without* re-orienting the Mob, use :meth:`~.Mob.orbit`.

    Animation
    ---------
    Recorded as an animation: the rotation sweeps from 0 to ``num_degrees`` over the
    current context's duration (1 second by default). Wrap the call in a context to
    change that -- ``with Seq(run_time=3): mob.rotate(90)`` -- or in ``Off()`` to apply
    it instantly without animating. Applies to this Mob and all of its descendants.

    Parameters
    ----------
    num_degrees
        How far to rotate, **in degrees**, counter-clockwise around ``axis``.
        Accepts a per-Mob tensor of shape ``(*, 1)`` for batched Mobs.
    axis
        Axis to rotate around; does not need to be normalized. Defaults to ``OUT``
        (the +z axis, pointing out of the screen), which spins a flat 2-D shape in
        the screen plane.
    about_point
        3-D point, shape ``(*, 3)``, that the Mob rotates around. Defaults to
        ``None``, meaning rotate in place about the Mob's own center.

    Returns
    -------
    :class:`~.Mob`
        This Mob, so calls can be chained.

    Examples
    --------
    .. algan:: Example1MobRotate

        from algan import *

        square = Square().spawn()
        square.rotate(90)
        square.rotate(180, axis=UP)
        square.rotate(90, about_point=RIGHT * 2)

        Scene.save_video()
    """
```

---

## 3. Summary line

- **Imperative mood**: "Rotate the Mob...", not "Rotates the Mob..." and never "This method
  rotates...". Existing docstrings are split roughly 50/50 between imperative and third-person;
  imperative is the PEP 257 rule and the tiebreak.
- One sentence, one line where possible, on the same line as the opening `"""`, ending in a period.
- Say what it does **to the scene**, not what it does to the object model. ✅ "Move the Mob so it
  sits just outside the screen edge." ❌ "Set the location attribute using the boundary helper."
- Do not restate the method name. ❌ `set_opacity`: "Sets the opacity." ✅ "Fade the Mob to a given
  opacity, 0 for invisible and 1 for fully opaque."
- For classes, name the thing and its defining property: ✅ "A rectangle with rounded corners,
  drawn as a cubic bezier circuit." ❌ `SurroundingRectangle`: "A rectangle." (real current text —
  it is a copy-paste of `Rectangle`'s and never mentions that it surrounds another Mob).

## 4. Parameters

Every parameter in the signature gets an entry, **in signature order**, including `color`,
`**kwargs`, and anything inherited-but-overridden. `Rectangle` currently documents `height` before
`width` (the signature is the other way round), documents an `*args` it does not accept, and omits
its `color` parameter — all three are defects.

### 4.1 Types come from annotations, not from the docstring

`docs/source/conf.py` sets `autodoc_typehints = "description"`, so Sphinx injects each parameter's
type from its annotation. A type written in the docstring **overrides** the annotation and then
silently drifts from it.

- **Annotate every parameter and the return** of a Tier-1 function. `Scene.save_video` has a
  thoroughly written docstring but zero annotations, so the rendered reference shows no types at
  all for it.
- Write the docstring entry as a bare name, with no ` : type` suffix:

```
✅  location
        The target 3-D location, shape ``(*, 3)``.

❌  location : torch.Tensor
        The target 3-D location.
```

The `Returns` section is the exception — see §5.

### 4.2 Defaults must be stated, in prose, in the description

This is the rule that is most often broken. `Square.side_length` ("Length of each side of the
square.") never says it is 2; `Circle.radius` never says it is 1.

State the default in the last sentence of the entry, as ``Defaults to X.``:

| Kind of default | What the description must say |
|---|---|
| Literal | ``Defaults to ``2``.`` |
| `None` meaning "compute something" | Say **what happens**, then the literal: "Defaults to ``None``, meaning the Mob's own center." |
| `None` meaning "read a setting" | Name the setting: "Defaults to ``SETTINGS.style.buffer`` (``0.6``)." |
| A constant | Name it and explain it: "Defaults to ``OUT`` (the +z axis, out of the screen)." |
| Mutable/derived | Explain the derivation: "Defaults to an identity matrix (no rotation, unit scale)." |
| Keyword-only flag | Say which behaviour is the default one, not just the value: `overwrite` → "Defaults to True: an existing file at the destination is replaced." |

Never write "optional" as the entire statement of a default, and never rely on the signature
default being visible — with `autodoc_typehints = "description"` and autosummary's `:nosignatures:`
it frequently is not.

### 4.3 Units, frames of reference, and shapes are mandatory

A number without a unit is a support ticket.

- **Angles are in degrees** everywhere in Algan. Say "in degrees" in every angle parameter.
- **Distances** are in world units unless the method name says screen (`move_to_screen_position`,
  `move_to_edge`); for screen-space parameters say so and give the range ("``x`` and ``y`` in
  screen units, where ``(0, 0)`` is the center").
- **Times** are in seconds.
- **Colors** accept an Algan `Color`, a named constant (`BLUE`), or anything `Color()` accepts.
- **Tensor parameters** state the shape with the batch convention already used in `Mob`'s class
  docstring: ``Shape: `(*, 3)` where `*` denotes zero or more batch dimensions.``
- **Percentages vs pixels**: if a value changed meaning (e.g. `render_tolerance` is now a fraction
  of screen height, not a pixel count), the docstring must say which, explicitly.

### 4.4 Accepted alternative forms

Many Algan parameters are polymorphic. Enumerate what is accepted, in the description, not just in
the annotation. ✅ `target_mob`: "The Mob to move next to, or a 3-D point (tensor) to treat as the
target." Also state coercion behaviour a user can observe: that Python floats/lists are cast to
tensors, that `direction` need not be normalized, that a `Mob` argument contributes its boundary
rather than its center.

### 4.5 `*args` / `**kwargs`

Never leave them undocumented, and never document them as "extra arguments". Point at the
destination so the reader can follow the chain:

```
*args, **kwargs
    Passed to :class:`~.BezierCircuitCubic`.
```

If a specific kwarg is the reason users reach for the passthrough, name it:

```
**kwargs
    Passed to :meth:`~.Mob.move_to` -- notably ``path_arc_angle`` to curve the path.
```

If the function *consumes* kwargs for compatibility and ignores them (`_translate_vector_style_kwargs`
drops `z_index`, `sheen_factor`, ...), say that they are accepted and have no effect, so users of
ported Manim scripts are not left guessing.

## 5. Returns

- **If the method returns `self` for chaining, the docstring must say so.** Use exactly this
  wording, so it is greppable and consistent:

```
Returns
-------
:class:`~.Mob`
    This Mob, so calls can be chained.
```

  Current text varies between "The Mob instance itself.", "The Mob instance itself, allowing for
  method chaining.", and nothing at all. `Animatable.spawn` returns `self` — the `Square().spawn()`
  idiom in `README.md` and `CLAUDE.md` depends on it — but documents no return at all.
- NumPy style requires a type on the first line of the section, and it wins over the annotation.
  Keep the two in sync, and use a role for Algan types (`:class:`~.Mob``) so the rendered page links.
- If the return is something the caller must keep, say what it is *for*: `add_updater` gets this
  right ("An ID identifying the updater... can be used to remove the updater... using
  :meth:`~.Animatable.remove_updater`").
- If the return is a structure, name its fields: `save_video` → "``status`` (``"rendered"`` or
  ``"skipped"``), ``output_path``, ``duration_seconds``, ``render_plan``".
- **Omit the section entirely** for methods that return `None`. Do not write "Returns nothing."
- Getters must state what convention the value is in: normalized or not, world or screen space,
  a view or a copy.

## 6. The `Animation` section — Algan-specific and mandatory

Algan is lazy: a method call may mutate state now, record a timed animation, or be forbidden after
spawn. **A user cannot infer which from the name**, so every Tier-1 method that touches scene state
carries an `Animation` section. State, in one short paragraph:

1. **Is it recorded?** "Recorded as an animation" vs "Takes effect immediately and is not animated".
2. **Default duration**, if recorded: "over the current context's duration (1 second by default)".
3. **What interpolates**, for `@animated_function` methods: name the animated argument and its start
   value, e.g. "sweeps from 0 to ``num_degrees``" (`animated_args={"num_degrees": 0}`).
4. **How to change the timing**: one inline example, `with Seq(run_time=3): ...` or `with Off(): ...`.
5. **Propagation**: whether the change applies to descendants (most `Mob` attribute writes do) or
   only to this Mob (`set_non_recursive`).
6. **Spawn-order constraints**: if the method must be called *before* `spawn()`, say so here and
   raise in code. `set_shader` / `set_fragment_shader` / `set_material` all have this constraint;
   they document it and also document the `Raises` — keep that pattern.

Do **not** explain *how* recording works (edit records, replay windows, `mob_id_to_inds`) in a
user-facing docstring. "Recorded on the Scene's timeline and replayed at render time" is the whole
of what a script author needs.

**How the heading is wired up.** In NumPy style, napoleon only treats a heading as a section if it is
registered; an unregistered one passes through verbatim and breaks the docs build with an
unexpected-section-title error. `Animation` is registered in `docs/source/conf.py`:

```python
napoleon_custom_sections = ["Animation", "Tests", ("Test", "Tests")]
```

so it renders as a titled block. Any *other* new heading needs the same registration first.

## 7. Raises

Document an exception when the user can trigger it by writing ordinary scene code:

```
Raises
------
:class:`.ModifiedProtectedAttributeError`
    If called on a Mob that has already been spawned.
```

Include the guard exceptions (post-spawn shader/material changes), argument-validation errors, and
settings errors (`AlganConfigurationError`). Do not document internal assertion failures or
`OutOfRenderMemory` retries, which the engine handles.

## 8. See Also

Add one when the reader is likely in the wrong place. Algan has dense method families
(`move_next_to`, `move_inline_with_edge`, `move_inline_with_center`, `move_inline_with_mob`,
`move_inline_with_boundary`) where the differences are subtle — that is exactly the case for
`See Also`, with a reason attached to each entry:

```
See Also
--------
:meth:`~.Mob.move_next_to` : Place this Mob beside another with a buffer.
:meth:`~.Mob.move_inline_with_center` : Align centers without changing the other axis.
```

## 9. Examples

- **Every Tier-1 class and every Tier-1 method a user calls directly gets at least one example.**
- Use the `.. algan::` directive for anything visual. It executes the body during the docs build and
  embeds the rendered video, so:
  - the body must be a complete script, starting with `from algan import *`;
  - it must call `Scene.save_video()` **exactly once** — zero videos and two videos are both build
    errors from `_find_video`;
  - the directive argument is a name used for the output file. Follow the existing convention
    `Example{N}{Owner}` (`Example1Mob`, `Example1MobSet`, `Example1Group`) and keep it unique
    across the docs;
  - options are available when useful: `:quality:` (`low`/`medium`/`high`/`fourk`),
    `:save_last_frame:` for a still, `:save_as_gif:`, `:hide_source:`, `:no_autoplay:`.
    Prefer `:save_last_frame:` for anything static — every rendered example costs docs-build time.
- Use a plain ```` .. code-block:: python ```` for non-visual API (settings, `save_video` paths,
  return-value handling), as `Scene.save_video` does.
- Examples must be *runnable as written* and show the parameter being documented. Prefer three short
  lines that each exercise one argument over one line that exercises none.
- Precede each example with a half-sentence of intent ("Create a square and move it to the left:").

## 10. Class docstrings

`conf.py` sets `autoclass_content = "both"`, so the class docstring and the `__init__` docstring are
concatenated. **Put constructor parameters in the class docstring and leave `__init__`
undocumented** — users instantiate `Square(...)`, not `Square.__init__(...)`. `Mob`'s class
docstring is the model to copy: summary, what the object *is*, `Parameters` for the constructor with
shapes and defaults, then `Examples`.

Also, for classes:

- Document public animatable attributes in an `Attributes` section, with units and ranges — these
  are the assignment targets in `mob.color = BLUE`, so they are API surface.
- Say what the class is made of when it affects behaviour ("a cubic bezier circuit", "a triangle
  mesh via `Surface`"), because that determines which renderer features apply.
- Subclasses must not inherit a stale parent summary. Every shape class needs its own first line.

## 11. Formatting mechanics

- reStructuredText, not Markdown. ``` ``literal`` ``` for code, values, and settings names;
  `*emphasis*`; no backtick-single-quote pairs.
- Roles for cross-references: `:meth:`~.Mob.move_to``, `:class:`~.Mob``, `:attr:`~.Mob.location``,
  `:func:`~.draw_border_then_fill``. The `~.` prefix renders the short name and keeps lines short.
- **Code in a docstring must be in a directive.** `set_shader`'s recovery recipe is currently written
  as bare indented lines inside a paragraph, which Sphinx renders as one run-on line. Use:

```
    .. code-block:: python

        with Off():
            new_mob = mob.clone(spawn=False)
            ...
```

- Wrap at the file's prevailing width (~88 columns). Blank line before every section heading and
  before the closing `"""` only when the last section is a multi-line block.
- One blank line between the summary and the extended description. No blank line between the opening
  `"""` and the summary.
- Sections we use: `Parameters`, `Animation` (see §6), `Returns`, `Raises`, `See Also`, `Examples`,
  `Attributes`, `Notes`, and `.. versionadded::` / `.. deprecated::` directives. Nothing else. Any
  other heading must be registered in `napoleon_custom_sections` first or it breaks the build.
  (`napoleon_custom_sections` currently registers `Tests`; that is a vendored-Manim artifact — do not
  use it in Algan code.)

## 12. Deprecation and API-change notes

Algan is in private beta with **no compatibility aliases**: there is exactly one name for each
thing. Consequently:

- Do not document removed or aliased names, even to be helpful. If you find a docstring mentioning
  `render_to_file`, `render_settings`, or `RenderSettings`, delete the mention.
- A parameter that still exists but is deprecated (e.g. `clone(reset_history=...)`, which warns)
  needs a `.. deprecated::` note in its entry saying what to use instead.
- When a default changes, the docstring is part of the change, not follow-up work. `reset=False` on
  `save_video` and the render-tolerance unit change are the kind of thing users only learn from a
  docstring.

## 13. Anti-patterns, with the current-code instance

| Anti-pattern | Where it exists today |
|---|---|
| Default value not stated | `Square.side_length`, `Circle.radius`, `Rectangle.width/height` |
| Angle without "degrees" | `Mob.rotate`, `Mob.orbit` (no `Parameters` section at all) |
| Returns `self`, says nothing | `Animatable.spawn` |
| No docstring on a taught API | `Animatable.clone` (6 parameters, one deprecated), `Animatable.despawn` |
| Copy-pasted parent summary | `SurroundingRectangle` ("A rectangle.") |
| Parameters out of signature order / phantom `*args` | `Rectangle`, `Square`, `Circle` |
| Types duplicated in the docstring | `move_to`, `move_inline_with_edge` (vs `move_next_to`, which is correct) |
| No annotations, so no types render | `Scene.save_video`, `Scene.save_frame` |
| Bare indented code, not a `code-block` | `Mob.set_shader` |
| Internal mechanics in a user docstring | `Animatable` class docstring's `data_sub_inds` / `parent_batch_sizes` entries |

## 14. Checklist

Before you commit a change to a Tier-1 function:

- [ ] Summary line is one imperative sentence and does not merely restate the name.
- [ ] Every signature parameter is documented, in order, with no phantom entries.
- [ ] Every parameter and the return are annotated in the signature; no types repeated in
      `Parameters`.
- [ ] Every default is stated in prose, including what `None` means.
- [ ] Units (degrees / seconds / world vs screen), tensor shapes, and accepted alternative forms are
      stated.
- [ ] `Animation` section says recorded-or-immediate, default duration, propagation, and any
      before-spawn constraint.
- [ ] `Returns` present, with the exact chaining sentence if it returns `self`; absent if it returns
      `None`.
- [ ] User-triggerable exceptions are in `Raises`.
- [ ] At least one runnable example; `.. algan::` bodies call `Scene.save_video()` exactly once and
      use a unique directive name.
- [ ] Cross-references use roles, code uses `code-block`, literals use double backticks.
- [ ] If you changed a default, a unit, or a name, the docstring changed with it.
- [ ] `docs/source/reference/` stubs updated if members were added or removed, and
      `.venv/Scripts/python.exe docs/make_and_open_docs.py --skip-examples --no-open` builds clean.

## 15. Current state and migration order

Measured over `algan/animatable_base/*.py` and `algan/scene.py`: **84 of 179 public functions have
no docstring at all.** Of those that do, the most common defect is an unstated default, followed by
missing units and missing `Animation` semantics.

Fix in this order, since it tracks what users hit first:

1. **Mob transforms** — `move*`, `rotate`, `orbit`, `scale`, `set_*`: add `Animation` sections,
   degrees, defaults.
2. **Lifecycle** — `spawn`, `despawn`, `clone`, `become`, `add_updater`: `clone` and `despawn` need
   docstrings from scratch.
3. **Shape and text constructors** — one accurate summary per class, correct parameter lists with
   defaults.
4. **Output and settings** — `Scene.save_video` / `save_frame` annotations; `SETTINGS` sections and
   presets.
5. **Materials, lights, camera** — already the strongest area; mainly needs examples.

Do not batch-rewrite by script. These docstrings are the product; each one needs a human decision
about what the reader needs to know.
