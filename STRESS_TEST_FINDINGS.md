# Algan stress test — findings

A deliberate attempt to break Algan through its public API: not "does the happy path
work", but "what does a user hit when they reach for something reasonable and it isn't
quite what Algan expects". The brief was that anything a user might plausibly try
should either work or say what to do instead.

Every finding below was found on `main` (98b40b5) and **has since been fixed on
this branch**. Each one carries a **Fixed** line saying what changed and where
its regression test lives. `uv run python stress_test/reproductions.py` runs one
check per finding and prints `FIXED` or `BUG`; `pytest -q --fast` is what
actually guards them.

## What was exercised

About 400 cases across six sweeps, each in its own `Scene`:

| Sweep | What it covered |
| --- | --- |
| Authoring surface | 143 cases: contexts (`Sync`/`Seq`/`Lag`/`Off`/`OnInit`/`ComposeRateFunc`), lifecycle, hierarchy, `become`, degenerate geometry, text, camera, lights, materials, updaters, multi-Scene |
| Renderer robustness | 53 of those degenerate scenes rendered at `SMOKE_TEST`, with the output frame checked for blank/NaN |
| Class sweep | Every one of the 170 `Mob` subclasses in `algan.__all__` constructed, spawned and rendered |
| Output & settings | 67 cases: textures/images, `SETTINGS` mutation, `save_video`/`save_frame` arguments, post-processing, audio, encoders |
| Timing semantics | 30 scenes rendered to video and their durations measured against what the context API promises |
| Scale | 2000 mobs, hierarchy depth 300, 2000 sequential animations, 200 updaters, plots, groups, mixed 2D/3D, glass stacks |

CPU-only cloud container, no GPU. Renderer findings are CPU-path only; nothing here
depends on CUDA.

---

## 1. Re-entering a context object silently breaks every later `Sync` in the process

**Severity: high — silent wrong output, and it escapes the script that caused it.**

Entering the same `AnimationContext` object while it is already entered corrupts a
process-global `ContextVar`. From that point on, every `Sync()` and `Lag()` in the
process — *in any Scene* — has no effect, and animations that should overlap play one
after another instead. Nothing is raised and nothing is warned.

```python
with Scene():
    square = Square().spawn()
    context = Sync()
    with context:
        with context:            # same object, entered twice
            square.move(RIGHT)

# from here on, in this process:
with Scene(video_settings=SMOKE_TEST):
    s = Square().spawn(animate=False)
    with Sync():                 # three moves that should take 1s together
        s.move(RIGHT * 0.1); s.move(RIGHT * 0.1); s.move(RIGHT * 0.1)
    Scene.save_video("out", SMOKE_TEST)   # -> 3.00s, not 1.00s
```

**Root cause.** `AnimationContext.__enter__` stores its `ContextVar` reset token on
`self._manager_override_token` (`animation_contexts.py:376`). Entering the same object
a second time overwrites that attribute, so the outer entry's token is lost; the inner
`__exit__` resets the inner token and clears the attribute, and the outer `__exit__`
finds `None` and resets nothing (`animation_contexts.py:538-542`, `648-649`). The
`.set()` from the outer entry is never undone, so `_ANIMATION_MANAGER_OVERRIDE` keeps
pointing at the first Scene's `AnimationManager` forever. Every later `Sync()` then
records against that dead manager's stuck, `finished=True` context instead of the
active Scene's, and the active Scene's animations fall back to its root `Seq`
(`lag_ratio=1`).

The same entry also self-references `prev_context`: on the inner entry
`self.prev_context = am.context` is `self`, so the context stack can never unwind.

**Why it is reachable.** Storing a context in a variable and using it from a helper is
ordinary code, and it is enough:

```python
context = Sync()
def draw():
    with context:
        square.move(RIGHT)
with context:
    draw()
```

**Why it escapes the script.** The render daemon is on by default for
`python my_scene.py`, and it is one warm process serving many scripts. A single script
that trips this poisons the daemon, and *every later render served by it* silently gets
sequential timing — including scripts that never reuse a context. This was found
exactly that way: a stress script tripped it, and unrelated renders in the same session
came out a second longer each. `algan daemon --stop` clears it; nothing in the output
says anything is wrong.

Measured with the same script, same Algan, same settings:

| | fresh daemon | poisoned daemon |
| --- | --- | --- |
| `Sync` of 3 | 1.00 s | 3.00 s |
| `Lag(0)` of 3 | 1.00 s | 3.00 s |
| `Lag(0.5)` of 3 | 2.00 s | 3.00 s |
| `Sync(run_time=2)` of 3 | 2.00 s | 3.00 s |
| `Seq` of 3 | 3.00 s | 3.00 s |

**Fix taken.** Make `__enter__` refuse a context that is already entered, with a
message pointing at `Lag`/`Sync` being cheap to construct fresh; or hold the tokens in
a list so re-entry unwinds correctly. A guard is the safer of the two, since the
self-referential `prev_context` is broken regardless of the token.

**Fixed.** `AnimationContext.__enter__` now raises `ContextReuseError` on a context
object that is already entered *or* has already been exited. A context describes one
block -- its timespan, its child list and its reset token all belong to that block --
so reuse has no meaning to preserve, and the message says to construct a new one.
Tests: `test_an_animation_context_object_refuses_a_second_with_block` and
`test_sync_still_overlaps_after_a_rejected_context_reuse` in
`tests/unit_tests/test_ux_regressions.py` (both `fast`). Check: `F1`.

---

## 2. Creating a Mob inside an updater crashes the render

**Severity: high — breaks the standard counter idiom, with an error that names nothing
the user wrote.**

Any Mob constructed while an updater runs makes the render fail:

```python
square = Square(color=BLUE).spawn()
square.add_updater(lambda mob, t: Circle(radius=0.1, color=RED))   # not even spawned
Scene.wait(1)
Scene.save_frame("out.png")
# IndexError: index 11 is out of bounds for dimension 0 with size 11
```

The new Mob claims rows on the attribute timeline during replay, and
`materialize_additional_rows` then indexes `self._end_points` — sized before those rows
existed — out of bounds (`animation_timeline/timeline.py:1700`).

It affects bare construction, `spawn()`, and `become()` inside an updater. `despawn()`
inside an updater is fine.

**The reason it matters more than it looks.** It takes out the canonical
number-that-counts-up, because the Manim compatibility layer rebuilds its Mob tree on
every sync (`manim_compat.py:617` → `ManimMob(...)`):

```python
tracker = ValueTracker(0).spawn()
number = DecimalNumber(0).spawn()
number.add_updater(lambda mob, t: mob.set_value(tracker.get_value()))
tracker.set_value(5)
Scene.save_frame("out.png")
# IndexError: index 248 is out of bounds for dimension 0 with size 248
```

`DecimalNumber` and `Integer` both fail this way. Algan's own `DecimalNumber` works,
and so does a `ValueTracker` driving a `Square`'s location — it is Mob *creation* that
breaks, not updaters or trackers.

`docs/source/new_user_tutorials/updaters.rst` documents the `(mob, t)` signature and
the "write updaters as functions of `t`" rule but says nothing about this, and the
`IndexError` mentions neither the updater nor the Mob.

**Fix taken.** Either grow the materialization buffers when replay allocates rows,
or detect Mob creation during an updater and raise something that names the updater and
points at building the Mob outside it (`with Off():`) and mutating it instead.

**Fixed.** A Mob built while frames materialize is now *ephemeral*: skipped by updater
dependency tracing and by materialization (it has no history in a window that predates
it), read and written through `current_state` so it answers with its constructed values
rather than zeros, kept out of `Scene.actors`, and rolled back when the replay ends. A
render therefore leaves the Scene exactly as authored, which is what lets `save_video`
be called twice and give the same video.

What an updater still cannot do is *reshape* a Mob that existed before the render: row
layout is fixed at authoring time and the batch's window was materialized against the
rows it had. That now raises `UnsupportedFeatureError` naming the updater and pointing
at `DecimalNumber`, which does count. Tests:
`tests/unit_tests/test_updater_mob_creation.py`. Checks: `F2`, `F2b`, `F2c`.

---

## 3. `save_video()` ignores the Scene's own video settings; `save_frame()` honours them

**Severity: high — a public setter that does nothing, and two output paths that
disagree.**

```python
with Scene(video_settings=SMOKE_TEST) as scene:      # 32x32 @ 2fps
    Square(color=BLUE).spawn()
    Scene.save_frame("still.png")   # 32x32      <- Scene's settings
    Scene.save_video("clip.mp4")    # 864x486 @ 15fps  <- SETTINGS.video
```

`scene.video_settings` reports `(32, 32)` throughout. `Scene.set_video_settings(...)`,
whose docstring says "Set this Scene's resolution, frame rate and anti-aliasing", makes
no difference either — the video is still rendered at `SETTINGS.video`.

`save_video`'s own docstring does say `video_settings` "Defaults to `None`, meaning
`SETTINGS.video`", so this is deliberate at that call site. But it leaves
`Scene(video_settings=...)` and `set_video_settings()` with no effect on the thing they
name, and it makes the two output calls in the same Scene disagree about resolution and
frame rate.

**Fix taken.** Default `save_video`'s `video_settings` to the Scene's own (which
itself defaults to `SETTINGS.video` at construction), matching `save_frame`. If the
current precedence is intended, `set_video_settings` and the `Scene(video_settings=)`
argument should say in their docstrings that they do not affect `save_video`.

**Fixed.** Both paths now go through `Scene._resolve_video_settings`: an argument wins,
then the Scene's own if it was *given* them, then `SETTINGS.video`. That last step is
what keeps `SETTINGS.video.set(HD)` working after the Scene exists, which matters
because the default Scene is built by the first Mob -- before most scripts' first line.
Tests: `test_the_scenes_own_video_settings_reach_both_render_calls` and
`test_a_later_settings_change_still_reaches_a_scene_that_chose_nothing` (both `fast`).
Check: `F3`.

---

## 4. The README quickstart does not run

**Severity: high — it is the first code anyone runs, and it is on the PyPI page.**

`README.md`'s Quickstart uses three names that do not exist:

| README | Actual API |
| --- | --- |
| `sphere.spawn(duration=1.0)` | `spawn()` takes no `duration`; the enclosing context sets timing |
| `with Sync(duration=2.0):` | the parameter is `run_time` |
| `sphere.shift([2, 0, 0])` | `Mob.move(...)` — there is no `shift` |

`spawn(duration=` and `Sync(duration=` appear nowhere else in the repository, and
`docs/source/manim_migration_guide.rst` already gives the right translations. Nothing
tests the root README: `test_doc_examples.py` collects `.. algan::` directives under
`docs/source/` only.

Corrected:

```python
with Seq():
    with Seq(run_time=1.0):
        sphere.spawn()
    with Sync(run_time=2.0):
        sphere.rotate(180)
        sphere.move([2, 0, 0])
        sphere.color = RED
```

**Fix taken.** Fix the README, and extend the doc-example collector to the root
README so it cannot drift again.

**Fixed.** The README now reads `sphere.spawn()`, `with Sync(run_time=2.0):` and
`sphere.move([2, 0, 0])`, and `tests/unit_tests/test_doc_examples.py` collects
`README.md`'s fenced Python alongside the docs' blocks, so it goes through the same
tiers as every other example and cannot drift again. Check: `F4`.

---

## 5. Colour arguments accept only `Color`, and fail with an internal error

**Severity: medium-high — everyone tries a hex string first.**

```python
Square(color="#ff0000")   # AttributeError: 'str' object has no attribute 'reshape'
Square(color="red")       # AttributeError: 'str' object has no attribute 'reshape'
Square(color=0xFF0000)    # AttributeError: 'int' object has no attribute 'reshape'
Square(color=(1, 0, 0))   # AttributeError: 'tuple' object has no attribute 'reshape'
mob.color = "#ff0000"     # AttributeError: 'str' object has no attribute 'shape'
mob.color = [1, 0, 0]     # RuntimeError: The size of tensor a (3) must match ... b (5)
```

`Color("#ff0000")` and `Color("red")` both work, so the value is parseable — it is the
Mob-side plumbing that does not parse it. And the parser already exists and is already
public-facing: `materials._to_color5` accepts "a hex int (`0xff0000`), a hex string
(`"#ff0000"`), an RGB tuple in `[0, 1]`, or an existing `Color` / tensor", which is why
`MeshStandardMaterial(color=0x8B5A2B)` is how the shipped presets are written
(`constants/material_presets.py:39`).

So the same literal is valid as a material colour and an `AttributeError` as a Mob
colour. The error surfaces from `mob._prepare_buffers` (`mob.py:1109`) with no mention
of colour at all.

**Fix taken.** Route the Mob `color` constructor argument and the `color` setter
through `_to_color5`.

**Fixed.** `constants.color.to_color` is now the one parser, applied to any attribute
whose name contains "color" and at the three places that handle a colour before
`Mob.__init__` sees it: 2-D circuits, surface vertex grids, and the Manim constructor
bridge (whose own parser reads a tuple of floats as a *list of colours*). Material
colour parameters are RGB rather than Algan's five channels, so a parsed colour is
trimmed for those instead of widened. An unparseable value raises `InvalidColorError`
naming it. Test: `test_every_colour_spelling_reaches_every_mob` (`fast`, parametrized
over five spellings and five Mob classes). Check: `F5`.

---

## 6. `spawn()` after `despawn()` is a silent no-op

**Severity: medium — the outcome is a blank video with no diagnostic.**

```python
square = Square(color=BLUE).spawn()
Scene.wait(1)
square.despawn()
Scene.wait(1)
square.spawn()          # returns self, is_spawned() is True, nothing appears
Scene.wait(1)
```

Frames after the re-spawn are empty. `spawn()` returns early because `is_spawned()`
stays True after a despawn (`animatable.py:1447`, and `is_spawned`'s own docstring says
so), and no warning is emitted.

`despawn()`'s docstring does carry the answer — "A despawned Mob cannot be brought
back; clone it before despawning if you need it again later" — but only there. A user
who reaches for `spawn()` gets no signal at the call that failed, and Manim's
`FadeOut` → `FadeIn` of the same mobject makes this a natural thing to try.

**Fix taken.** Warn from `spawn()` when `is_despawned()`, with the clone advice
from `despawn()`'s docstring.

**Fixed.** `spawn()` warns with `DespawnedMobWarning` when the Mob is despawned,
carrying the clone-before-despawn advice that was only in `despawn`'s docstring.
Spawning an already-spawned (not despawned) Mob stays a quiet no-op -- that one is
documented and harmless. Test:
`test_spawning_a_despawned_mob_warns_instead_of_doing_nothing` (`fast`). Check: `F6`.

---

## 7. A surface that tessellates to zero triangles crashes the renderer

**Severity: medium.**

```python
Sphere(radius=0).spawn();     Scene.save_frame("out.png")
# RuntimeError: The expanded size of the tensor (1) must match the existing size (0)
#   at non-singleton dimension 1. Target sizes: [1, 1, -1]. Tensor sizes: [1, 0, 5]
```

Also `Sphere(radius=1e-9)`, `Cylinder(radius=0)` and a `Surface` with a degenerate
`u_range`/`v_range`. The empty primitive reaches `broadcast_all`
(`utils/tensor_utils.py:257`) via `triangle_primitive.py:267`.

The `1e-9` case is the one that will actually be hit — a radius computed from data, or
a value that shrinks toward zero. Neighbouring degenerate shapes are fine:
`Sphere(radius=-1)`, `Cone(base_radius=0)`, `Cylinder(height=0)`, `Torus(tube_radius=0)`
and `Cube(size=0)` all render (blank or mirrored), so the behaviour is also
inconsistent between primitives.

**Fix taken.** Drop empty primitives before the merge, or reject a degenerate
extent at construction with a message naming the parameter.

**Fixed.** `Surface._build_render_primitive` returns `None` when the tessellation has no
triangles; the callers already treat that as "this actor contributes no geometry".
Test: `test_a_surface_with_no_extent_renders_nothing_rather_than_failing`. Check: `F7`.

---

## 8. `set_material()` does not check what it was given

**Severity: medium.**

```python
Sphere().set_material(GOLD)     # AttributeError: 'Color' object has no attribute 'shader'
Sphere().set_material(None)     # AttributeError: 'NoneType' object has no attribute 'shader'
Sphere().set_material("gold")   # AttributeError: 'str' object has no attribute 'shader'
```

`set_material(GOLD)` is a natural mistake precisely because `CHROME` and `COPPER` *are*
material presets while `GOLD` is a colour. `set_shader` next door validates properly
(`TypeError: 42 is not a callable object`), and `set_material` already raises a good
`ModifiedProtectedAttributeError` for the post-spawn case — it just does not type-check
`material` before touching `material.shader` (`mob_materials.py:262`).

**Fix taken.** `isinstance(material, Material)` check with a message listing the
presets and the `Mesh*Material` classes.

**Fixed.** `set_material` checks `isinstance(material, Material)` and names the material
classes and the presets. Test: `test_set_material_rejects_a_non_material` (`fast`).
Check: `F8`.

---

## 9. Smaller errors that name the wrong thing

Each of these is a poor diagnostic rather than a functional break, ordered by how
likely a user is to meet it.

| Trigger | What was reported | What it says now |
| --- | --- | --- |
| `Text("")` or `Text("   ")`, on `spawn()` | `RuntimeError: torch.cat(): expected a non-empty list of Tensors` | **Fixed.** It spawns. A string with no glyphs has nothing for the entrance wave to stagger, and `animate_lagged_by_location` now returns rather than reducing over an empty list (`F9`) |
| `Seq(lag_ratio=0.5)`, `Sync(lag_ratio=0.5)` | `TypeError: algan...Lag.__init__() got multiple values for keyword argument 'lag_ratio'` | **Fixed.** *"Seq is Lag with ratio=1, so it takes no lag_ratio of its own. Use Lag(0.5) for that overlap"* (`F10`) |
| `save_video(codec="notacodec")` | after a full render, `FileNotFoundError: '..._temp.mp4' -> '....mp4'` | **Fixed.** The codec is checked against FFmpeg's encoder list before rendering: 0.2 s instead of 27 s, and the message names the codec (`F11`) |
| `ImageMob(numpy_array)` | `TypeError: zeros_like(): argument 'input' must be Tensor, not numpy.ndarray` | **Fixed.** numpy arrays and nested sequences are accepted, uint8 scaled by 255 (`F12`) |
| `ImageMob(torch.zeros(8, 8, 2))` | `ValueError: color_texture must have shape [W, H, 5], got (8, 8, 4).` | **Fixed.** `Color.add_defaults` widens only 3 and 4 channels, so the complaint reports `(8, 8, 2)` — the shape that was passed (`F13`) |
| `Scene.set_background_color("not a color")` | `RuntimeError: [Errno 2] No such file or directory: 'not a color'` | **Fixed.** A background string is read as a colour first; if it is neither a colour nor a file that exists, the message says so (`F14`) |
| a Mob used after `save_video(reset=True)` | `AttributeError: Square owns no rows of the 'location' attribute timeline` | **Fixed.** The message now names `save_video(reset=True)` as one of the two ways a Mob loses its rows (`F15`) |
| `mob.shift(...)`, `mob.animate...`, `next_to`, `to_edge`, `arrange`, `set_fill` | bare `AttributeError` | **Fixed.** `Mob.__getattr__` names the Algan call for ~20 Manim methods. An ordinary typo keeps the ordinary message (`F16`) |
| `add_updater(lambda mob: ...)` | `TypeError: <lambda>() takes 1 positional argument but 2 were given` | **Fixed.** The signature is checked before the updater is recorded, and the message shows the `(mob, t)` form (`F18`) |
| `save_video(post_processes=(42,))` | `TypeError: 'int' object is not callable`, after the whole render | **Fixed.** Each pass is checked for callability before rendering (`F19`) |
| `Code("print(1)")` | `FileNotFoundError: 'print(1)'` | Unfixed. The first positional is `code_file`; `code_string=` is the one for a literal. This is Manim's own signature, inherited by the compatibility wrapper |
| `Quad()` | `RuntimeError: stack expects a non-empty TensorList` | **Fixed.** *"Quad needs its vertices: pass them as points (Quad(LEFT, RIGHT, UP)) or as one [N, 3] tensor."* |

## 10. Accepted in silence

These neither work nor say anything. Listed because the brief was that a user should
always get one or the other; none of them crash, and several are arguably fine as
clamping — but nothing distinguishes them from a scene the user got right.

* **Fixed.** `mob.set_parent_to(mob)` and parent cycles were accepted, while `Group.add`
  rejected the same shapes with `HierarchyError: A Mob cannot be its own child` (`F17`).
  `set_parent_to` now walks up from the proposed parent and raises `HierarchyError` if
  it arrives back at the caller. A chain that is not a cycle is unaffected.
  *(Since renamed to `add_parent`, which also links the downward half, so the
  cycle check walks children rather than parents. `F17` covers the new name.)*
* `move_to(nan)` / `move_to(inf)` render a blank frame. A NaN that came out of the
  user's own arithmetic yields a black video and no clue where it came from.
* `opacity = 5.0` and `opacity = -1.0` clamp; `scale(-1)` mirrors; `scale(0)`,
  `scale(1e9)` and `rotate(90, ORIGIN)` (a zero axis) all render blank.
* `Square(size=-1)`, `Circle(radius=-1)` construct and render mirrored.
* `Scene.get_camera().despawn()` produces a black video.
* `Text("hi 🎉")` renders byte-identically to `Text("hi")` — the emoji is dropped with
  no warning. `Text("a\x00b")` fails with
  `ValueError: Pango cannot recognize your color '#FFFFFF' for text 'a b'`, which
  blames the colour for a problem with the string.
* `set_animated_attribute("bogus", 1.0)` succeeds.
* `FullScreenRectangle()` renders nothing at any colour — its outline sits exactly on
  the frame edge. `ScreenRectangle` and `Rectangle` are visible.

The rest are unfixed, and deliberately: each is a value the renderer treats sensibly
(clamping, mirroring, drawing nothing), and warning about every one of them would cost
more in noise than it returns. They are recorded here because "silently reasonable" and
"silently wrong" are indistinguishable to a reader, and a NaN in particular is worth
revisiting.

---

## Two bugs found while fixing the above

Neither was in the sweep -- both surfaced while chasing finding 2 -- and both are fixed.

**`DecimalNumber.set_value()` made the number vanish.** Any `set_value` on a
`DecimalNumber` or `Integer` rendered a blank frame, with or without an updater, and
retroactively: frames from *before* the call were blank too.
`_sync_manim_node_from_algan` pushed the Algan side's style onto every Manim node,
including the point-less root whose colour and opacity rows are placeholders. Manim
treats a point-less node's style as a template, and `set_value` rebuilds its glyphs and
then calls `init_colors()` -- which broadcast that placeholder over the whole family at
opacity 0. Style is now synchronized only onto nodes that draw something. Test:
`test_set_value_leaves_a_manim_number_visible`. Check: `F2d`.

**Growing an attribute buffer mid-render discarded the batch's frame window.**
`AttributeTimeline.add` pointed `active_state` back at `current_state` whenever it grew
the buffer. Claiming rows mid-render then dropped the materialized window, after which
every read answered with the single timeless frame: the primitives came out one frame
deep while the lights stayed the batch's depth, and the light packer raised a
shape error naming neither. Reachable only from an updater that builds a Mob, which is
why finding 2 was hiding it.

---

## What held up

Worth recording, because it is most of the surface and it is the reason the list above
is as short as it is.

* **The renderer is hard to crash.** 53 deliberately degenerate scenes — NaN and
  infinite positions, zero and negative extents, 2000-vertex polygons, parent cycles,
  40 lights, a light inside the geometry, out-of-range material parameters, transmissive
  2-D shapes, coplanar z-fighting — produced exactly one crash (finding 7). Everything
  else rendered or came out blank.
* **`SETTINGS` validation is exemplary.** Every malformed write is caught with a
  message that names the fix: `SETTINGS.video = HD` →
  *"SETTINGS.video has stable identity; call SETTINGS.video.set(...)"*; a bad field name
  lists the valid ones; presets refuse mutation and point at `preset.set(...)`;
  `frames_per_second=0` and a zero resolution are both rejected.
* **Scene-level warnings do the right thing.** `EmptySceneWarning` and
  `NeverSpawnedMobWarning` both fire with actionable text.
* **Post-spawn material and shader changes** raise `ModifiedProtectedAttributeError`
  with the full clone-and-respawn recipe in the message.
* **Vector arguments** are checked and explained:
  *"displacement must be a 3-D vector of shape (\*, 3), such as RIGHT, UP \* 2 or [1, 0, 0]; got [1, 2]"*.
  Python lists, tuples and numpy arrays all work where a vector is expected.
* **Timing semantics are correct** — once finding 1 is not in play. `Seq`, `Sync`,
  `Lag(r)`, `run_time`, `run_time_unit`, nested contexts, `Off` inside a timed context,
  `wait` including negative and zero, and rate functions were all measured against
  rendered frame counts and all matched.
* **Context cleanup is otherwise sound.** An exception raised inside a `with Sync()`,
  a context object reused sequentially, an abandoned generator holding a context open,
  and `Scene.terminate()` called inside a context all leave global state clean. Only
  the self-nesting case of finding 1 leaks.
* **Scale is a non-issue at these sizes.** 2000 mobs, hierarchy depth 300, 2000
  sequential animations and 200 simultaneous updaters all recorded and rendered without
  complaint.
* **Multi-Scene isolation works.** Nested `Scene` contexts, and repeated identical
  scenes rendered in one process, produced identical output every time.

---

## Reproducing

```bash
uv run python stress_test/reproductions.py          # every check
uv run python stress_test/reproductions.py --list   # ids and titles
uv run python stress_test/reproductions.py F1 F3    # selected
```

Every check should print `FIXED` except `F11` on a machine whose FFmpeg cannot be asked
for its encoder list, where it reports that it skipped.

The tests are the authority, and they run in CI:

```bash
uv run -m pytest -q --fast                                 # the marked ones
uv run -m pytest -q tests/unit_tests/test_updater_mob_creation.py
uv run -m pytest -q tests/unit_tests/test_ux_regressions.py
```
