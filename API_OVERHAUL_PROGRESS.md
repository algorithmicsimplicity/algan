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
| 3 | `Mob` surface: privatize, consolidate, rename | Not started |
| 4 | Scene, Camera, Lights, Group | Not started |
| 5 | Class and parameter naming | Not started |
| 6 | `border_*` → `stroke_*` | Not started |
| 7 | `run_time` → `duration`, `rate_func` → `easing` | Not started |
| 8 | Documentation and baselines | Not started |

**Export count**: 471 at the start → 379 after Phase 1b → 361 after Phase 2.

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

---

## Known follow-ups

- **`Phase 8` doc sweep will be large.** `docs/source/galleries/mob_gallery.rst` already has one
  example using `Arc(radius=1, start_angle=0, angle=3.14)` in radians, which now means 3.14
  *degrees*. Every compat class in the docs needs auditing for the same, not just this one.
- **`UnitInterval` and `TangentialArc` left the root namespace** and were not put in the curated
  subset. Reachable as `mn.UnitInterval` / `mn.TangentialArc`. Revisit if they turn out to be
  reached for often.
- **`tests/full_renders` and `tests/path_traced` have not been run yet** on this branch. They are
  the pixel-comparing suites; Phase 1/1b should not have moved output, but that is unverified.
  Run before Phase 8, and note that this is a CPU-only session so the CUDA baselines cannot be
  spoken for.

---

## Working notes

- **Do not edit source files while a test run is in flight.** A `ruff --fix` landing mid-run
  produced a spurious `test_environment` failure that did not reproduce. Same hazard class as
  the Taichi one in `CLAUDE.md`, for the same reason.
- The fast suite's self-reported time is junk until the third consecutive run. Cold: 84s.
  Warm: 32-33s of a 75s budget.
