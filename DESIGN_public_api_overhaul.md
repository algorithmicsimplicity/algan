# Public API Overhaul — Plan of Record

**Status**: planned, not started
**Scope**: every public class, method, function, parameter, setting and constant in `algan`.
**Compatibility**: none required. Algan is in closed beta; there are no user scripts to protect.
Rename and delete outright — no deprecation shims, no aliases (see `CLAUDE.md`, "Public API").

This document supersedes the review document it was derived from. Where the two disagree, this
one wins: several claims in the review were written against an API that does not exist, and are
recorded as **not doing** in the appendix so nobody re-proposes them.

---

## Sequencing

The phases are ordered so that structural decisions land before the mechanical renames that
depend on them. Two dependencies are load-bearing:

- **Phase 1 before Phase 6.** `border_width` cannot become `stroke_width` while `stroke_width`
  is still a live Manim-compat kwarg meaning *twice* `border_width` (see Phase 6).
- **Phase 0 before everything.** Without a snapshot test, a 471-name export surface silently
  drifts under this much churn.

Phases 2–5 are independent of each other and can be done in any order or in parallel.
Phase 7 is mechanical and should go last, when no more names are moving.

Each phase is one PR. Run `uv run -m pytest -q --fast` after every change and the full suite
before pushing each phase.

---

## Phase 0 — Pin the surface

No behaviour change. Makes every later phase reviewable as a diff of the public API.

1. Add `tests/unit_tests/test_public_api_surface.py` holding a checked-in sorted snapshot of
   `algan.__all__`, failing with a diff when the two disagree, refreshed by
   `ALGAN_UPDATE_API_SNAPSHOT=1` (declare the variable in `algan/environment.py`; every
   `ALGAN_` read must go through an accessor there).
2. Mark it `fast`. It qualifies under the `tests/README.md` rule: a change anywhere in the
   package can move the export set.
3. Extend `tests/unit_tests/test_ux_regressions.py` with a call to each method this plan
   renames, so the renames are caught at call sites and not just in the name list.

**Verification**: `--fast` green; snapshot file contains 471 names.

---

## Phase 1 — Extract the Manim compat layer into `algan.manim`

The largest structural change and the one that unblocks the most downstream cleanup.

### Why

`Arc`, `Sector`, `Annulus`, `Angle`, `Arrow`, `Axes`, `Graph`, `Matrix`, `Table`, `Brace`,
`Vector`, `DashedLine`, `DecimalNumber`, `Integer`, `ValueTracker`, `TrueDot`, `PointCloudDot`
and ~130 more currently resolve to `algan.mobs.manim_compat` / `point_cloud` / `opengl_compat`
and land in `from algan import *` beside native classes. A user cannot tell `Circle` (native,
`shapes_2d`) from `Arc` (Manim wrapper) from the name, and the two follow different conventions —
most visibly radians vs degrees for angles, and the stroke-width factor of two in Phase 6.

Putting the compat layer behind a module makes the convention boundary explicit and legible:

```python
from algan import *
import algan.manim as mn

square = Square().spawn()   # native Algan, degrees
arc = mn.Arc(angle=PI / 2)  # Manim, radians — and it says so
```

### Work

1. Create `algan/manim/__init__.py` (a real package; `algan.mobs.manim_compat` and friends stay
   where they are as the implementation, and the new package is the public face).
2. Re-export from it: everything in `algan.mobs.manim_compat.__all__`, `opengl_compat.__all__`,
   `point_cloud.__all__`, `image_compat.__all__`, `manim_parity.__all__`, plus `manim_mob`.
3. **Remove the native-omission carve-out: wrap every Manim class, including ones with a native
   equivalent.** `_WRAPPED_MANIM_CLASS_NAMES` currently skips "existing native Algan classes"
   (~20 of them), which made sense while both surfaces shared one namespace and a name could
   only mean one thing. Once they are separate namespaces that reason is gone, and the clean
   rule is: **`Sphere` is Algan's, `mn.Sphere` is Manim's.** Every name in `algan.manim` then
   behaves by Manim's conventions with no exceptions to memorise, and no native class is
   reachable by two spellings.

   Add to the wrap list: `Square`, `Circle`, `Line`, `Rectangle`, `Polygon`, `Dot`, `Triangle`,
   `RegularPolygon`, `Sphere`, `Cylinder`, `Cone`, `Cube`, `Torus`, `Prism`, `Surface`,
   `Arrow3D`, `Dot3D`, `Text`, `Tex`, `Code`.

   **Verified feasible**: all twenty wrap through `_make_manim_wrapper` and construct
   successfully, 3-D geometry included — `ManimMob` converts Manim `Surface` subclasses
   (`Sphere`, `Torus`, `Cone`) into curved quad patches rather than rejecting them. Nothing in
   the omitted set needs special handling.
4. Move `Mobject = Mob` and `GenericGraph = Graph` out of `algan/__init__.py` and into
   `algan/manim/__init__.py`.
5. `install_opengl_aliases(globals())` moves with them, called against the `algan.manim`
   namespace, and now resolves `OpenGLSquare → mn.Square` — the **Manim wrapper**, not the
   native class. This is a behaviour change: the OpenGL aliases currently point at native Algan
   classes. It is the correct one (an `OpenGL*` name is a Manim-renderer name and should follow
   Manim's conventions), but it must be called out in the PR body, and it means the aliases now
   depend on step 3 having run.
6. Delete the compat names from root `algan.__all__`. The mechanism is already there: add the
   compat modules to `_INTERNAL_EXPORT_MODULES` in `algan/__init__.py`.
7. Keep `algan.manim` out of `algan.__all__` — it is reached by `import`, not by star-import.
8. Update `validate_manim_mobject_parity` / `MANIM_MOBJECT_NAMES` in `manim_parity.py` to check
   the new namespace.

### Consequences to resolve during the work

- **Compat-only classes leave the root namespace with nothing behind them.** Step 3 gives every
  *native* class a Manim twin, but the reverse gap stays: `Axes`, `NumberLine`, `Graph`,
  `Table`, `Matrix`, `Brace`, `Vector`, `Arrow`, `Angle`, `Sector`, `Annulus` and the rest have
  no native equivalent, so after this phase `from algan import *` has no axes, no graph, no
  table and no arrow. For an explanatory-maths engine that is a real hole. Either users write
  `mn.Axes()` for these, or Algan grows native versions — decide before this phase ships, and
  record it here either way. Writing native replacements is out of scope for this plan.
- **The text family stops being split.** `Text`, `Tex` and `Code` are native and now get Manim
  twins; `MathTexPart`, `SingleStringMathTex`, `Paragraph`, `MarkupText`, `Title` and
  `BulletedList` are compat-only and fall under the gap above.
- `docs/source/` examples using compat classes need the `import algan.manim as mn` line.

### Explicitly NOT doing

Per the API policy, nothing in the compat layer is deleted or renamed. `ValueTracker`,
`ComplexValueTracker`, `TrueDot`, `PointCloudDot`, `VGroup`, `VDict`, `PGroup`, the `OpenGL*`
family and the Manim `DecimalNumber`/`Integer` all survive unchanged inside `algan.manim`.
The review's proposals to consolidate or deprecate them are withdrawn.

**Verification**: full suite. `from algan import *` no longer binds `Arc`; `import algan.manim as
mn; mn.Arc` works. Snapshot test shows only compat names leaving.

---

## Phase 2 — Remove leaked internals from `algan.__all__`

Pure subtraction. Nothing is renamed and nothing moves; the names stay importable at their real
paths, they just stop landing in the user's namespace.

Add to `_INTERNAL_EXPORT_NAMES` in `algan/__init__.py`:

| Name(s) | Home |
| :--- | :--- |
| `check_codec_is_available`, `resolve_encode_binary`, `select_video_encoder` | `utils.video_encoding` |
| `wrap_pad_texture`, `surface_closed_axes`, `surface_weld_flags` | `mobs.surfaces.surface` |
| `orient_faces_outward` | `mobs.shapes_3d` |
| `attr_ranges_for_mob` | `animatable_base.animatable` |
| `release_torch_memory` | `utils.memory_utils` |
| `fragment_light`, `fragment_light_vis`, `prep_normal`, `shading_normal`, `smith_geometry`, `ggx_distribution` | `rendering.raytracing.shading_taichi` |
| `basic_material_shader`, `basic_pbr_shader`, `depth_shader`, `lambert_shader`, `manim_shader`, `matcap_shader`, `normal_shader`, `phong_shader`, `physical_shader`, `standard_shader`, `toon_shader` | `rendering.shaders.*` |
| `RenderPlan`, `TruncationCounts` | `rendering.raytracing.*` |
| `FragmentStage`, `STAGE_LAMBERT`, `STAGE_MANIM`, `STAGE_PHONG`, `STAGE_PHYSICAL`, `STAGE_STANDARD`, `STAGE_UNLIT` | `rendering.shaders.fragment_shaders` |
| `ANIMATABLE_PROPERTY_VERSION` | `animatable_base` |

Additionally:

- **`override_moviepy_ffmpeg_binary` becomes a setting**, not an exported function: add
  `SETTINGS.paths.ffmpeg_binary` and have the encoder read it. Monkey-patching a global from a
  star-imported function is the wrong shape for what is configuration.
- **Give `algan/constants/rate_funcs.py` an `__all__`.** It currently re-exports `math`, `torch`
  and `annotations` to anyone who touches `algan.rate_funcs`.

### NOT doing

**The `*Triangulated` family stays public**: `QuadTriangulated`, `TexTriangulated`,
`TextTriangulated`, `TriangleTriangulated`, `TriangulatedBezierCircuit`, `TriangleVertices`,
`BezierCircuitCubic`. These are not tessellation plumbing — they are the surface-backed
counterparts of the bezier-circuit classes, and users need them because circuits cannot cast or
receive shadows while surfaces can. The review was wrong to list them as leakage. Document *why*
they exist rather than hiding them.

**Verification**: `--fast`. Snapshot diff is subtraction only.

---

## Phase 3 — `Mob` surface: privatize, consolidate, rename

### 3a. Make private (prefix `_`, delete from the public surface)

All eleven verified present on `Mob` today:

`apply_absolute_change_two`, `retroactive`, `set_to_retroactive`, `set_to_current`,
`generate_animatable_attr_set_get_methods`, `check_properties_are_valid`, `morph_soup_parts`,
`morph_kind`, `reorder_batch_to_minimize_movement`, `resolved_shadow_flags`, `mesh_key`.

`morph_soup_parts` is overridden in `mobs/shapes_3d.py` and `mobs/point_cloud.py` — rename all
three together or the override silently stops overriding.

### 3b. Consolidate duplicate geometry queries

| Delete | Keep |
| :--- | :--- |
| `get_boundary_edge_point`, `get_boundary_edge_point_recursive`, `get_boundary_in_direction`, `Group.get_boundary_edge_point2` | `get_edge_point(direction, recursive=False)` |
| `Group.get_mob_midpoint` | `get_center()` — already the identical bbox min/max midpoint |
| `Group.mobs` | `Group.children` — `mobs` already just returns `self.children`, so this is an alias and the API policy forbids it |

`Group.get_boundary_edge_point2` takes the extremum across members rather than over the whole
bounding box, which is a real behavioural difference from `Mob.get_boundary_edge_point` — fold it
in as the `Group` override of `get_edge_point`, do not just delete it.

### 3c. Rename

| From | To |
| :--- | :--- |
| `get_axis_aligned_lower_corner` / `_upper_corner` / `_size` | `get_bounding_box_min()` / `_max()` / `_size()` |
| `get_points_evenly_along_direction(direction, num_points=3)` | `sample_points_in_direction(direction, count=3)` |
| `get_upwards_direction` / `get_upwards_basis` | `get_up_direction` / `get_up_basis` |
| `move_to_edge(edge, buffer)` | `move_to_screen_edge(direction, buffer)` |
| `move_to_corner(edge1, edge2, buffer)` | `move_to_screen_corner(directions, buffer)` |
| `move_out_of_screen(edge, ...)` | `move_off_screen(direction, ...)` |
| `move_to_point_along_square(destination, displacement)` | `move_to_point_with_displacement(destination, displacement)` |
| `move_between(loc1, loc2)` | `move_between(start, end)` |
| `set_x_y_coord(xy_coords)` | `set_xy(coords)` |
| `rotate(num_degrees, axis, about_point)` | `rotate(angle, axis, about=None, *, degrees=True)` |
| `orbit(num_degrees, axis, about_point)` | `orbit(angle, axis, about=None, *, degrees=True)` |
| `move_to(location, path_arc_angle)` and `move_to_point_along_arc(point, arc_angle_degrees, ...)` | `move_to(location, arc_angle=None)` |

`about_point → about`, **not** `center`. `center` is already taken three ways in this API
(`get_center()`, `Camera.center_on`, and the `Text(center=)` bool in Phase 5), and
`rotate(90, OUT, center=...)` reads as "center the mob".

### 3d. Direction accessors

For `d` in `right`, `up`, `forward`: `get_{d}_direction()` with a `.{d}` property alias, and
`get_{d}_basis()` with no alias. This is the one place the plan adds an alias deliberately;
record it in `CLAUDE.md`'s alias roster alongside `IN`/`OUT` and the settings spellings.

### 3e. Coordinate access

Add `.x` / `.y` / `.z` / `.xy` properties. Then **delete** the six methods they replace —
`get_x_coord`, `get_y_coord`, `get_z_coord`, `set_x_coord`, `set_y_coord`, `set_z_coord` — plus
`get_individual_coords` / `set_individual_coords`, replaced by `get_coord(indices)` /
`set_coord(indices, value)` (note `coord_indexes` → `indices`).

Leaving both spellings is exactly the duplication this overhaul exists to remove.

### 3f. Unify the four `move_inline_with_*` methods

`move_inline_with_boundary`, `move_inline_with_center`, `move_inline_with_edge`,
`move_inline_with_mob` collapse into:

```python
align_with(mob, direction, anchor='center' | 'edge' | 'boundary', buffer=None)
```

**`anchor=`, not `align_to=`** — `Group.arrange_in_line` gets an `align_to` parameter in Phase 4
that takes a *direction vector*. Same name with two types in one API is what we are here to fix.

### 3g. `look` / `look_at` axis naming

`look(direction, with_axis='forward')` and `look_at(point, with_axis='forward')`. Accept
`'right'`, `'up'`, `'forward'` case-insensitively, mapping to basis indices 0/1/2; raise on an
unrecognised name.

**Behaviour is unchanged.** The current `axis=2` already aims the mob's local forward axis along
the look direction — the review claimed the opposite. This is a naming change only; do not
"fix" the semantics.

**Verification**: full suite — `move_to_edge` alone has 16 test call sites and 6 in docs.

---

## Phase 4 — Scene, Camera, Lights, Group

### Scene

| From | To |
| :--- | :--- |
| `add_light_source` / `remove_light_source` / `clear_light_sources` | `add_light` / `remove_light` / `clear_lights` |
| `show_frame(time_stamp=None)` | `show_frame(at=None)` |
| `render_audio_to_file(file_path, frames_per_second=44100, ...)` | `save_audio(file_path, sample_rate=44100, ...)` |
| `length_to_num_pixels` / `num_pixels_to_length` | `length_to_pixels` / `pixels_to_length` |

Lifecycle consolidation — five methods to two:

- `clear()` and `clear_scene()` are aliases of each other. Keep one.
- `despawn_scene()` → **`despawn_mobs()`**: despawns every spawned Mob, animated.
- `clear_scene()` folds into `despawn_mobs()` as a parameter (it is `despawn_mobs` over 0.5s
  followed by dropping actors with no renderable history) — or stays as a distinct method if the
  actor-retention behaviour does not fit a flag. Decide during implementation; do not ship both.
- `reset_scene()` → merge into **`reset()`**. `reset()` already calls `reset_scene()` as its last
  step; the difference is only whether the timeline is rebuilt. One method, one flag.

**`Scene.terminate()` is not part of this.** It pops the scene off the `SceneManager` active
stack — a different concept that the review wrongly grouped with the reset family. Leave it.

### Camera

| From | To |
| :--- | :--- |
| `move_to_make_mob_center_of_view(mob, buffer_portion=0.7)` | `center_on(mob, buffer_portion=0.7)` |
| `set_euler_angles(angle_1, angle_2, angle_3)` | `set_euler_angles(yaw, pitch, roll, *, degrees=True)` |
| `retroactive_center(mob)` | `_retroactive_center` |
| `get_render_screen_basis()` | `_get_render_screen_basis` |
| `screen_scale` | `frame_height` (it is the half-height of the virtual screen) |

`set_to_orthographic()` and `set_near_orthographic(distance=1e5)` are two spellings of one
concept — the same duplication being removed from `Scene`. Consolidate to
`set_orthographic(near=False, distance=...)` or equivalent.

**Keep `screen_distance` and `fov` both.** `fov` already exists as a constructor parameter and
as `get_fov`/`set_fov`; it is a derived spelling of `screen_distance` given `screen_scale`.
`screen_scale` is not focal distance, so collapsing the pair onto `fov` alone loses a degree of
freedom. The review's `focal_distance`/`fov` proposal is withdrawn.

### Lights

- `Light.build_aux`, `is_extended`, `num_samples`, `get_sample_positions` → private
  (`_build_aux`, `_is_extended`, `_num_samples`, `_get_sample_positions`). All four are
  overridden across `PointLight`, `DirectionalLight`, `HemisphereLight`, `SpotLight` and
  `RectAreaLight` — rename every override in the same commit.
- `SpotLight(angle=30.0)` → `SpotLight(cone_angle=30.0, ..., *, degrees=True)`.

**`DirectionalLight` keeps `target`.** It shares `_TargetedLight` with `SpotLight` and
`RectAreaLight`, which legitimately have both a position and a target; giving it a `direction`
instead forks that base for one subclass. The review's proposal is withdrawn.

### Group

| From | To |
| :--- | :--- |
| `arrange_in_grid(..., buffer=None, column_buffer=None, ...)` | `row_buffer=None, column_buffer=None` |
| `arrange_in_line(..., equal_displacement=False, alignment_direction=None)` | `equal_widths=False, align_to=None` |

**Verification**: full suite.

---

## Phase 5 — Class and parameter naming

### Timeline contexts

| From | To |
| :--- | :--- |
| `Lag(lag_ratio, run_time=None)` | `Lag(ratio=0.5, run_time=None)` |
| `AnimationContext(same_run_time=...)` | `match_durations` |
| `Speech(script, ...)` | `Speech(transcript, ...)` |
| `Audio(file_path_or_clip, wait_at_end=0)` | `Audio(source, *, wait_at_end=0.0)` |

`same_run_time` lives on `AnimationContext`, not on `Sync` — it is inherited, and
`Sync(same_run_time=True)` is how it is reached (`test_ux_regressions.py:74`). Rename it on the
base class; the `_CONTEXT_ONLY_PARAMS` map at `animation_contexts.py:57` needs updating with it.

**`OnInit` stays.** It is not an alias of `Off(spawn_at_end=True)` — `Off` has no such parameter.
`OnInit(func)` runs a callable over every Mob *constructed* inside the block, which is a real
feature with no replacement. The review's deprecation is withdrawn.

`Lag`'s `ratio` rename must keep `_reject_fixed_lag_ratio` working — `Sync` and `Seq` raise a
helpful `TypeError` when handed a `lag_ratio`, and that message names the parameter.

### Shapes

| From | To |
| :--- | :--- |
| `Square(side_length=2)` / `Cube(side_length=2)` | accept `size=2` as the primary name |
| `Prism(dimensions=(3, 2, 1))` | `Prism(width=3, height=2, depth=1)` |
| `Torus(major_radius=1.5, minor_radius=0.5)` | `Torus(ring_radius=1.5, tube_radius=0.5)` |
| `Line3D(thickness=0.02)` | `Line3D(radius=0.02)` |
| `Arrow3D(thickness=0.02, height=0.3, base_radius=0.08)` | `Arrow3D(shaft_radius=0.02, tip_length=0.3, tip_radius=0.08)` |
| `ThreeDModelMob(normalize=True, normalize_size=2.0)` | `Model3D(fit_to_size=2.0)` |
| `ThreeDModelMob.bake_animation(...)` | `Model3D.precompute_animation(...)` |
| `ImageMob` | keep — see below |

`Square`/`Cube` take `size` as *the* name, not alongside `side_length`; two spellings is what
this overhaul removes. `Prism` currently defaults to `(3, 2, 1)`, so the three-parameter form
defaults to `width=3, height=2, depth=1`.

**`ImageMob` is not renamed to `Image`.** `Image` is far too generic for a namespace users share
with PIL and numpy, and it would sit beside compat `ImageMobject`. If the `Mob` suffix is the
objection, `ImageSurface` fits — it subclasses `Surface`.

`DashedLine(dashed_ratio=)` is a Manim compat class and is untouched by Phase 1's policy.

### Text

| From | To |
| :--- | :--- |
| `t2c`, `t2f`, `t2g`, `t2s`, `t2w` | `color_map`, `font_map`, `gradient_map`, `slant_map`, `weight_map` |
| `should_center=True` | `center=True` |
| `line_spacing=-1` | `line_spacing=None` |
| `slant='NORMAL'` / `weight='NORMAL'` | accept lowercase, case-insensitively |
| `Tex(arg_separator=' ')` | `delimiter=' '` |
| `Tex.get_segment(i)` | `get_segment(index)` |

`slant` and `weight` values are forwarded to Pango, which wants `"NORMAL"`, `"BOLD"`,
`"ITALIC"`, `"OBLIQUE"`. **Normalize at the boundary** — accept any case from the user, upper-case
before forwarding. Do not change what is sent to Pango.

### Other

| From | To |
| :--- | :--- |
| `NumericDisplay(value, num_decimal_places=2)` | `DecimalNumber(value, decimal_places=2)` |
| `Surface.grid` | `Surface.mesh`, plus `Surface.vertices` property → `mesh.location` |
| `scene_function` | `@algan_scene` |
| `clear_cache(taichi_kernels=False)` | `clear_cache(include_kernels=False)` + `clear_cached_kernels()` |
| `draw_border_then_fill` | `DrawBorderThenFill` (class, matching `Indicate`/`Wiggle`) |
| `TranscriptAudioMismatchError` | `AudioTranscriptMismatchError` |
| `SETTINGS.video.audio_frames_per_second` | `audio_sample_rate` |
| `SETTINGS.video.super_sampling_anti_aliasing` | `supersampling`, keeping `ssaa`/`SSAA` aliases |

The `NumericDisplay → DecimalNumber` rename **requires Phase 1 to have landed** — the compat
`DecimalNumber` must be out of the root namespace first, or the two collide.

`NumericDisplay`'s first parameter is already `value`; only `num_decimal_places` changes.
`num_integer_places` should become `integer_places` in the same pass.

**`to_color` stays internal**, not folded into `Color()`. It deliberately passes tensors through
untouched so a per-row colour buffer is not collapsed to one colour, and `Color(tensor)`
returning the tensor unchanged is surprising. Add it to `_INTERNAL_EXPORT_NAMES` in Phase 2.

**Verification**: full suite plus `pytest -q tests/full_renders` — `Torus`, `Prism` and `Line3D`
are PN geometry, invisible to `--fast`.

---

## Phase 6 — `border_*` → `stroke_*`

**Depends on Phase 1.** Do not start before it lands.

`stroke_width` and `stroke_color` are the standard terms for bezier-path outlines and
`border_width`/`border_color` should adopt them. But both names already exist in the codebase
meaning different things, and the units differ:

```python
# settings/shape_style_profiles.py:157-159
# Algan's border_width convention is half Manim's stroke_width
"border_width": stroke_width_value / 2,
```

So today `stroke_width` is a Manim-facing constructor kwarg that is halved on its way to the
native `border_width` attribute. A blind rename produces one name meaning two things depending
on which door the value came through — strictly worse than the status quo.

### Decision — settled

**Keep Algan's unit and move the `/2` conversion to the `algan.manim` boundary.**
`border_width` → `stroke_width` everywhere native, and native `stroke_width=1` means exactly
what `border_width=1` meant. The halving survives only inside `algan.manim`, where it is
explicitly a Manim-convention translation and sits beside every other Manim convention.

The rejected alternative was adopting Manim's unit (dropping the `/2`, doubling every native
stroke width): a simpler concept, but it moves rendered output everywhere and needs a full
re-baseline for no user-visible gain.

This is why the phase is blocked on Phase 1 — the conversion has nowhere clean to live until
`algan.manim` exists.

### Work

1. Rename the `Mob` attributes `border_width` / `border_color` → `stroke_width` / `stroke_color`.
   `_FIVE_CHANNEL_COLOR_ATTRS` in `mob.py:81` names `border_color` and must move with it.
2. Update the compat translation in `shape_style_profiles.py` and `shapes_2d.py` to write the new
   attribute name, keeping the `/2`.
3. Update the removed-method hint at `mob.py:116` (`set_stroke` → "set the `stroke_color` and
   `stroke_width` attributes") — it now reads oddly, and may no longer be worth having.
4. Sweep `bezier_circuit.py` (29), `text.py` (22), `bezier_circuit_primitive.py` (20),
   `raytrace_kernels_taichi.py` (20), `shapes_2d.py` (18), plus tests (112) and docs (14).

Kernel files carry 20+ occurrences: `*_taichi.py` are linted but never formatted, and a rename
there is a kernel edit, so expect a full cold recompile and clear the offline cache before any
A/B timing.

**Verification**: full suite plus `tests/full_renders`. Output must be **byte-identical** — this
is a rename, not a rendering change. If baselines move, something was converted twice.

---

## Phase 7 — Global mechanical renames

Last, when no more names are moving. Two sweeps, each its own commit:

1. **`run_time` → `duration`** (175 in `algan/`, 227 in tests, 64 in docs). Also
   `run_time_part` → `duration_per_part` and `run_time_unit` → `duration_unit` (19 combined).
   `DEFAULT_RUN_TIME` → `DEFAULT_DURATION`. The `_reject_fixed_lag_ratio` and
   `_CONTEXT_ONLY_PARAMS` strings mention these names in user-visible error text.
2. **`rate_func` → `easing`** (83 / 110 / 30) and `rate_func_compose` → `composed_easing`;
   `ComposeRateFunc` → `ComposedEasing`. The `rate_funcs` module is already spelled `ease_*`
   throughout, so this makes it self-consistent. `DEFAULT_RATE_FUNC` → `DEFAULT_EASING`.

Do these with review, not blind `sed`: `run_time` appears inside docstring prose and inside
`animation_contexts.py`'s parameter-owner table, and `rate_func` appears as a substring of
`rate_funcs` (the module) which is **not** being renamed to `easings` — decide that explicitly.

**Verification**: full suite. Grep for the old names across `algan/`, `tests/`, `docs/`,
`benchmarks/` and `README.md` to confirm zero survivors.

---

## Phase 8 — Documentation and baselines

1. Rewrite `CLAUDE.md`'s "Public API" section: the `algan.manim` split, the new alias roster
   (`IN`/`OUT`, `fps`/`ssaa`, and the new `.up`/`.right`/`.forward` direction properties), and
   the lifecycle methods that survived.
2. Update `agent_guidance/api_settings.md` for the new `SETTINGS.paths.ffmpeg_binary` and the
   renamed video fields.
3. Sweep `docs/source/` for every renamed symbol; the `.. algan::` examples execute at build
   time, so a missed rename is a build failure rather than a silent stale doc.
4. `uv run python docs/make_and_open_docs.py --skip-examples --no-open` for the structural pass,
   then a full build.
5. **No re-baselining should be needed.** Every phase in this plan is a rename or a namespace
   move; none of them may move a pixel. A baseline that shifts is a bug, not a new expectation —
   find it before regenerating.

---

## Appendix — Withdrawn proposals

Recorded so they are not re-proposed. Each was checked against the source.

| Proposal | Why withdrawn |
| :--- | :--- |
| Material `camelCase` → `snake_case` (whole section) | Already done. `MeshStandardMaterial` and `MeshPhysicalMaterial` are `roughness_map`, `metalness_map`, `clearcoat_roughness`, `env_map_intensity`, `specular_color`, `sheen_roughness`, `attenuation_color`. No camelCase exists in `materials.py`. |
| Unify `Scene.add_light` with `add_light_source` | `add_light` does not exist. Kept as a rename in Phase 4, without the duplication rationale. |
| `Sync(same_run_time=)` → `match_durations` | The parameter is on `AnimationContext`, not `Sync`. Kept in Phase 5, relocated. |
| Deprecate `OnInit` for `Off(spawn_at_end=True)` | `Off` has no `spawn_at_end`. `OnInit(func)` is a construction hook with no replacement. |
| `NumericDisplay(number=)` → `value` | Already `value`. |
| Camera `screen_distance`/`screen_scale` → `focal_distance`/`fov` | `fov` already exists. `screen_scale` is frame half-height, not focal distance; collapsing loses a degree of freedom. |
| `Scene.terminate` in the reset family | Pops the `SceneManager` active-scene stack. Unrelated concept. |
| `Arc(angle=90, degrees=True)` | `Arc` is a Manim compat class. Radians is Manim's contract; Phase 1 makes that explicit instead. |
| Consolidate `TrueDot`/`PointCloudDot`/`DotCloud`; deprecate `ValueTracker`; drop `VGroup`/`VDict`/`OpenGL*` | All compat-layer classes. Policy forbids deleting a Manim name for duplicating an Algan one. |
| `DirectionalLight(direction=)` | Shares `_TargetedLight` with `SpotLight`/`RectAreaLight`; forking it for one subclass costs more than it buys. |
| `to_color(value)` → `Color(value)` | `to_color` passes tensors through untouched by design; `Color(tensor)` returning the tensor is surprising. Stays internal. |
| `*Triangulated` family is leaked internals | They are public API: circuits cannot cast or receive shadows, surfaces can, so users need the surface-backed variants. |
| `ImageMob` → `Image` | Too generic for a star-imported namespace shared with PIL/numpy. |
| `about_point` → `center` | `center` is already overloaded three ways. Using `about` instead. |
| `align_with(align_to=)` | Collides in type with `arrange_in_line(align_to=)`, a direction vector. Using `anchor=`. |
| `look`'s `axis` semantics are wrong | They are correct — `axis=2` aims local forward along the direction. Renamed only. |
