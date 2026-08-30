# Public API Overhaul — Progress

Companion to `DESIGN_public_api_overhaul.md`, which is the specification. This file is the
running state: what has landed, what is left, and what changed about the plan along the way.

Update it in the same commit as the work it describes.

---

## Status

| Phase | Scope | State |
| :--- | :--- | :--- |
| 0 | Pin the surface with a snapshot test | **Done** |
| 1 | Extract the Manim compat layer into `algan.manim` | **Done** |
| 1b | Native adapters for the curated root subset | **Done** |
| 2 | Remove leaked internals from `algan.__all__` | **Done** |
| 3 | `Mob` surface: privatize, consolidate, rename | **Done** |
| 4 | Scene, Camera, Lights, Group | **Done** |
| 5 | Class and parameter naming | Not started |
| 6 | `border_*` → `stroke_*` | Not started |
| 7 | `run_time` → `duration`, `rate_func` → `easing` | Not started |
| 8 | Documentation and baselines | Not started |

**Export count**: 471 at the start → 379 after Phase 1b → 361 after Phase 2. Phase 3 moves no
exports: everything it touches is a `Mob` method or property, not a module-level name.

---

## Landed

### Phase 0 — snapshot test

- `tests/unit_tests/test_public_api_surface.py`, marked `fast`. Holds the export roster in
  `tests/unit_tests/public_api_snapshot.txt` and fails with the added/removed names spelled out.
  `ALGAN_UPDATE_API_SNAPSHOT=1` refreshes it.
- Three further guards ride along: every exported name resolves; none is private or duplicated;
  no adapter shadows a native class (added in 1b).
- `ALGAN_UPDATE_API_SNAPSHOT` declared in `algan/environment.py`'s `_HARNESS_VARIABLES`.

### Phase 1 — `algan.manim`

- `_WRAPPED_MANIM_CLASS_NAMES` grew 99 → 119: the native-omission carve-out is gone, so
  `mn.Sphere`, `mn.Square`, `mn.Text` and 17 more now exist beside Algan's.
- New `algan/manim/__init__.py` re-exports `manim_compat`, `opengl_compat`, `point_cloud`,
  `image_compat` and `manim_parity`, plus `Mobject = Mob` and `GenericGraph`.
- `install_opengl_aliases` runs against that namespace, so `mn.OpenGLSquare` is now Manim's
  `Square` rather than Algan's native one — the intended behaviour change.
- The five compat modules are out of root's star-imports; 92 names left `algan.__all__`.
- Four test modules moved their compat imports to `algan.manim`:
  `test_manim_compat_movement`, `test_manim_compat_sync`, `test_morph_become_audit`,
  `test_point_cloud_rendering`.

### Phase 1b — native adapters

- New `algan/mobs/manim_adapters.py`: 65 curated classes get a native root spelling that
  converts Algan's conventions and delegates to the compat wrapper.
- Angle conversion is declared per class in `_ANGLE_PARAMS` (8 classes), applied only to
  arguments the caller actually supplied.
- `Arc(angle=90)` and `mn.Arc(angle=PI/2)` verified to build identical geometry, as does a
  no-conversion class (`Ellipse`).

### Phase 2 — leaked internals

Eighteen names left `algan.__all__`, all still importable at their real paths:

- Video encoding: `check_codec_is_available`, `resolve_encode_binary`,
  `select_video_encoder`, `override_moviepy_ffmpeg_binary`
- Surface plumbing: `wrap_pad_texture`, `surface_closed_axes`, `surface_weld_flags`,
  `orient_faces_outward`
- Engine internals: `attr_ranges_for_mob`, `release_torch_memory`,
  `ANIMATABLE_PROPERTY_VERSION`, `to_color`
- Raw Taichi shading maths: `fragment_light`, `fragment_light_vis`, `prep_normal`,
  `shading_normal`, `smith_geometry`, `ggx_distribution`

Also landed: `SETTINGS.paths.ffmpeg_binary` (outranks every other candidate, for every
codec — the reason to pin a binary is that moviepy's build lacks a codec yours has, so it
must beat the probe rather than join it; default behaviour byte-for-byte unchanged), and an
explicit `__all__` on `algan/constants/rate_funcs.py`.

### Phase 3 — the `Mob` surface

**3a — privatized.** `_apply_absolute_change_two`, `_set_to_retroactive`, `_set_to_current`,
`_generate_animatable_attr_set_get_methods`, `_check_properties_are_valid`, `_morph_soup_parts`
(with its two overrides), `_morph_kind`, `_reorder_batch_to_minimize_movement`,
`_resolved_shadow_flags` (with its seven callers) and `Mob._mesh_key`. `retroactive()` stayed
public — see plan change 5.

**3b — consolidated.** `get_boundary_edge_point` + `get_boundary_edge_point_recursive` →
`get_edge_point(direction, recursive=True)`; `get_boundary_in_direction` →
`get_boundary_point`; `Group.get_boundary_edge_point2` deleted; `Group.get_mob_midpoint`
privatized; `Group.mobs` deleted in favour of `children`.

**3c — renamed.** `get_bounding_box_min` / `_max` / `_size`, `sample_points_in_direction`,
`get_up_direction` / `get_up_basis`, `move_to_screen_edge`, `move_to_screen_corner`,
`move_off_screen`, `move_to_point_with_displacement`, `move_between(start, end)`,
`rotate(angle, axis, about, *, degrees=True)`, `orbit(...)` likewise, and
`move_to(location, arc_angle=None)` absorbing `move_to_point_along_arc` (now `_move_along_arc`).

**3d — direction properties.** `.right` / `.up` / `.forward`, recorded in `CLAUDE.md`'s alias
roster, which now names four exceptions rather than three.

**3e — coordinate properties.** `.x` / `.y` / `.z` / `.xy`, replacing the six single-axis
methods; `get_coord(indices, centered=False)` / `set_coord(indices, value)` replace
`get_individual_coords` / `set_individual_coords`.

**3f — one alignment method.** `align_with(mob, direction, anchor, buffer, from_mob)` replaces
all four `move_inline_with_*`.

**3g — named look axes.** `look(direction, with_axis='forward')` and `look_at` likewise, with
`'right'`/`'up'`/`'forward'` matched case-insensitively and anything else raising.

**Verified**: full `tests/unit_tests` green on CPU (2408 passed, 139 skipped), `--fast` green
(404 passed), `ruff check` and `ruff format --check` both back to their pre-existing failures.
The pixel-compared suites are noted under follow-ups.

Also landed: `test_ux_regressions.py` gained
`test_the_mob_positioning_surface_answers_to_its_public_names`, which calls every new name and
asserts every removed one is gone, per Phase 0 step 3.

### Phase 4 — Scene, Camera, Lights, Group

**Scene.** `add_light` / `remove_light` / `clear_lights` (the first two existed already as
aliases, so this deletes a duplication as well as renaming); `show_frame(at=None)`;
`save_audio(file_path, sample_rate=44100, ...)`; `length_to_pixels` / `pixels_to_length`. Five
lifecycle methods become two: `despawn_mobs(retain_history=False, duration=None, **kwargs)` and
`reset(rebuild_timeline=True)`, with a private `_rebuild_contents()` carrying the half of
`reset` that `Scene.__init__` needs.

**Camera.** `center_on`; `set_euler_angles(yaw, pitch, roll, *, degrees=True)`;
`_retroactive_center` and `_get_render_screen_basis` privatized;
`screen_scale`/`screen_scale_factor` collapse into `screen_half_height`; `set_to_orthographic`
deleted in favour of `set_near_orthographic`.

**Lights.** `_build_aux`, `_is_extended`, `_num_samples`, `_get_sample_positions` privatized
across `Light` and all five subclasses; `SpotLight(cone_angle=..., degrees=True)`.

**Group.** `arrange_in_grid(row_buffer=, column_buffer=)` and
`arrange_in_line(equal_widths=, align_to=)`.

`test_ux_regressions.py` gained
`test_the_scene_camera_light_and_group_surface_answers_to_its_public_names`, the Phase 4
counterpart of the Phase 3 one.

Two guards had to move with the work, and both were caught by the full suite rather than by
`--fast`. `test_render_coverage_audit`'s "never import the world" rule allowed only
`from algan import *` and `torch`; since Phase 1 made `algan.manim` the public spelling of the
compat layer, a scene covering compat geometry has to reach it the way a user would, so
`algan.manim` joins that allowlist. And two `.. algan::` examples in
`lighting_and_shadows.rst` build a `SpotLight(angle=...)`.

---

## Plan changes made during implementation

Recorded here and in the design doc, so the two do not drift.

1. **`stroke_width` conversion moved out of Phase 1b into Phase 6.** Converting it in 1b would
   change what `Arrow(stroke_width=6)` renders while the native attribute is still spelled
   `border_width` and its unit is unsettled — and no phase in this plan may move a pixel.
   Phase 6 renames the attribute and decides the unit, so the conversion belongs there. Costs
   nothing: no test or doc passes `stroke_width` to any of the six affected classes.

2. **Only explicitly-passed angle arguments are converted, never defaults.** Manim's defaults are
   already radians and already correct (`Arc`'s `angle=TAU/4` is a quarter turn either way);
   binding defaults before converting would read `1.57` as degrees.

3. **Eleven names the plan called leaks are public API and stay exported.** The design doc's
   Phase 2 table was assembled from names that *look* internal, and two groups in it were
   wrong:

   - `FragmentStage` and `STAGE_LAMBERT`/`MANIM`/`PHONG`/`PHYSICAL`/`STANDARD`/`UNLIT`,
     plus `RenderPlan` and `TruncationCounts`, are documented user-facing API —
     `shaders_and_materials.rst` gives the stage constants their own reference table, and
     `renderer_limitations.rst` documents the render plan as script-readable.
   - The fragment-shader callables (`phong_shader`, `standard_shader`, ...) were removed and
     had to be put back. One tutorial passage imports them by module path, which looked like
     the contract, but the executable `.. algan::` examples in the same file open with
     `from algan import *` and then pass `standard_shader` to `set_fragment_shader`, and
     `test_builtin_fragment_pipeline_is_available_to_star_imports` asserts exactly that.

   **Lesson for the remaining phases: check tests and executable doc examples before
   removing a name, not prose alone.** Both errors were caught by the full suite, neither by
   `--fast`.

4. **`algan.manim` is imported at the bottom of `algan/__init__.py`, behind an assignment.**
   Ruff's isort hoists a bare import into the block above, and from there `algan.manim` →
   `Mob` → `algan.animated_function` hits a partially-initialised module. The
   `_MANIM_NAMESPACE_ANCHOR = None` line is what keeps the import where it has to be; it is not
   otherwise meaningful.

5. **`retroactive()` stays public; only the cursor pair under it is privatized.** The design
   doc's 3a list had all three. But `retroactive()` is the `with` block a user writes to record
   earlier in the video — a documented feature with a worked example — while
   `set_to_retroactive` / `set_to_current` are the raw authoring-cursor moves it wraps and the
   only thing a caller of them gains is the chance to forget the second one. Privatizing those
   two leaves the concept exactly one public spelling, which is what the overhaul is for.

6. **Three of Phase 3b's "duplicates" were not duplicates.** Same failure mode as the Phase 2
   table, and caught the same way — by reading the implementations rather than the names:
   `get_boundary_in_direction` computes a different point from `get_boundary_edge_point` (the
   extreme projected back onto the center line, which is what every placement method measures
   with), `Group.get_mob_midpoint` is the midpoint of member *locations* and not the bounding
   box, so replacing it with `get_center()` would move every Group's anchor, and
   `Group.get_boundary_edge_point2` was uncalled dead code rather than a behavioural variant.
   The first two are kept (renamed and privatized respectively); only the third is deleted.

7. **`recursive` defaults to True on `get_edge_point`**, not False as the design table said.
   Every existing caller reached the recursive spelling, so False would have silently changed
   what a Group measures — a pixel move, which no phase in this plan may make.

8. **`set_x_y_coord` is deleted, not renamed to `set_xy`.** 3c asked for the rename and 3e adds
   an `.xy` property; keeping both would be the duplication 3e's own text calls out. The
   property wins, consistently with the six single-axis methods it also replaces.

9. **`move_next_to(align_edge=...)` changes behaviour, and this is the one place Phase 3 moves
   a pixel.** `align_edge` was implemented on `move_inline_with_boundary`, which moved the
   *whole* boundary-to-boundary displacement rather than its component along the alignment
   axis. For `caption.move_next_to(chart, RIGHT, align_edge=DOWN)` that displacement is
   `(-2.4, -0.7, 0)` on a 3x2 chart: the secondary alignment slid the caption back on top of
   the chart in x, undoing the placement it was supposed to refine. `align_with` projects onto
   the axis, as its whole family always documented, so the call now does what the tutorial
   prose says. Nothing pixel-compared uses `align_edge` — one doc example does, and its output
   changes to the correct picture.

10. **`ManimCompatMob.rotate` takes the rename too, despite the no-touching-compat-names rule.**
    That override exists precisely *because* it follows Algan's `rotate` rather than Manim's --
    degrees, Algan's axis constants, a real basis rotation -- and says so in its docstring.
    Its parameters were Algan's spelling, not Manim's, so they move with Algan's: `angle`,
    `about`, and the new `degrees`. Manim's own `rotate(angle, axis, about_point)` is untouched
    where the vendored library defines it. Caught by the full suite, not by `--fast`.

11. **`HemisphereLight.up` had to move.** A `Light` is a `Mob`, and 3d's new read-only `Mob.up`
   property shadowed the attribute the constructor writes. Its parameter is still `up`; the
   attribute is now `sky_direction`. (`NeuralNetMLP.forward` also shadows the new `.forward`
   property, but it is an unexported class and a subclass attribute wins, so it still works;
   noted rather than changed.)


12. **Camera's `screen_scale` becomes `screen_half_height`, not `frame_height`.** The design
    table named `frame_height` while its own parenthetical said the value is a *half*-height —
    and it is a half-height of the virtual screen at `screen_distance`, not of the frame: the
    default 2.5 at distance 5 shows a half-height of 3.5 at the origin plane. `frame_height` is
    also Manim's name for its full 8-unit frame, so that spelling would mislead exactly the
    readers most likely to reach for it. The constructor parameter and the attribute it wrote
    (`screen_scale` / `screen_scale_factor`) were themselves two spellings of one number, and
    collapse into this one.

13. **`set_orthographic(near=False, ...)` is not buildable; `set_near_orthographic` survives
    alone.** The proposed flag's default value would name true parallel-ray projection, which
    this renderer does not implement, so `near` would have had one usable setting.
    `set_to_orthographic` was already a pure deprecation shim — it warned and called the other
    method — so deleting it is the ordinary treatment of a second spelling, and the survivor is
    the name that describes what actually happens.

14. **`Scene.add_light` and `remove_light` already existed** as class-level aliases of
    `add_light_source` / `remove_light_source`, contradicting the design appendix's "`add_light`
    does not exist". Phase 4 therefore removes a duplication here rather than only renaming.

---

## Known follow-ups

- **`Phase 8` doc sweep will be large.** `docs/source/galleries/mob_gallery.rst` already has one
  example using `Arc(radius=1, start_angle=0, angle=3.14)` in radians, which now means 3.14
  *degrees*. Every compat class in the docs needs auditing for the same, not just this one.
- **`UnitInterval` and `TangentialArc` left the root namespace** and were not put in the curated
  subset. Reachable as `mn.UnitInterval` / `mn.TangentialArc`. Revisit if they turn out to be
  reached for often.
- **Phase 3 renamed names inside `docs/source/` as it went**, because the `.. algan::` examples
  execute at build time and would otherwise fail. That is a spot fix on the renamed symbols
  only; the Phase 8 sweep still owes the prose (the positioning tutorial's method tables now
  describe `align_with` three times over, for instance) and a real doc build.
- **`tests/full_renders`: three of its four failures are older than this branch, and the CPU
  baselines are stale on `master`.** Run three ways on CPU:

  | Scene | This branch | Before Phase 3 (`faf4dae`) | Pre-overhaul `master` (`0b482bf`) |
  | :--- | :--- | :--- | :--- |
  | `materials_and_lighting` | 14 @ frame 21 | 14 @ 21 | **14 @ 21** |
  | `shapes_and_timeline` | 26 @ frame 293 | (never rendered) | **26 @ 293** |
  | `solids_and_camera` | 200 @ frame 179 | 200 @ 179 | **200 @ 179** |
  | `manim_compat_and_plots` | 192 @ frame 119 | 192 @ 119 | **passes** |

  (Deviations are channel values; the tolerance is 2.) The first three fail identically on
  pre-overhaul `master`, so they are nothing to do with this work — the committed
  `expected_outputs_cpu` set does not match what this renderer produces on this machine, and
  re-baselining them is a separate job for someone who can look at the frames. **This means
  `tests/full_renders` cannot be used as a pass/fail gate on CPU as things stand; compare
  scene-by-scene against `master` instead.**

  `manim_compat_and_plots` was the one real regression, and it is Phase 1b's, exactly as the
  `Arc(angle=3.14)` follow-up above predicted: the scene builds
  `ArcBetweenPoints(start=..., end=..., angle=1.6)`, which meant 1.6 *radians* until 1b gave the
  name a native, degrees-taking adapter, after which the same line draws a 1.6-degree arc — an
  almost straight line. Fixed by reaching that one class through `mn.`, which is the same object
  the scene used before, so the frames go back to the baseline.

  Two lessons, both the same shape as the Phase 2 one already recorded here: **a phase is not
  verified until the pixel suites have run**, and **`--fast` cannot see any of this** — it has no
  compat geometry and no PN geometry.

- **`tests/path_traced` has still not been run** on this branch. Note also that this is a
  CPU-only session, so the CUDA baselines cannot be spoken for either way.

---

## Working notes

- **Do not edit source files while a test run is in flight.** A `ruff --fix` landing mid-run
  produced a spurious `test_environment` failure that did not reproduce. Same hazard class as
  the Taichi one in `CLAUDE.md`, for the same reason.
- The fast suite's self-reported time is junk until the third consecutive run. Cold: 84s.
  Warm: 32-33s of a 75s budget.
