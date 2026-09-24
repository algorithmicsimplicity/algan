---
name: algan-video
description: >-
  Create, edit, render, and troubleshoot videos using the Algan Python animation
  library. Use when an agent must create 2D/3D animated or still-shot visual
  content using computer graphics. Algan is a successor to Manim, it can do everything Manim can,
  as well as 3D ray-traced rendering.
metadata:
  compatibility: >-
    Requires a Python execution environment with Algan and its matching dependencies,
    file access, and FFmpeg for video export. Text, LaTeX, speech alignment, and some
    model formats have optional dependencies. Rendering requires a supported CPU/GPU
    backend. Custom compiled shaders must be defined in real Python source files.
  version: "1.3.0"
  source-checked: "2026-09-24"
  algan-source-commit: "45c1d6a3fd47f83b09b7a82bb77f5d080f457556 + local changes (Algan 0.0.2)"
---

# Algan video production

Turn the user's requested visual content into Algan scene code, then render and
verify the requested deliverables. Algan is the production tool, not the subject
of the task.

## Scope and creative direction

This skill covers only how to realize a vision with Algan: API mechanics,
workflow, rendering, and verification. It deliberately contains no visual-style or
design guidance, so that it can be combined with any user's style instructions.
Take all creative direction (content, style, palette, typography, composition,
camera language, pacing, transitions, sound) from the user and from their own
style instructions, such as a project `AGENTS.md` or a separate style skill,
including how much creative latitude you have and when to consult them. Where
those instructions ask you to design, design within them. Nothing in this package
is a style source: its colors, sizes, motions, and example values only
demonstrate the API. Do not add unrequested content such as music, narration, or
captions.

If the user provides a narration script for the video, then use it as given.
You may suggest script improvements to the user but you MUST NOT change the script
without the user's explicit permission. Use explicit `with Speech(...):` blocks,
each containing an exact, meaningful transcript segment and the animations that
belong to those words. Do not mechanically divide narration into a requested
number of chunks and dispatch animations by chunk index. Reuse visual helpers,
while keeping the narration-to-action mapping readable at the call site.

A transcript is sufficient to author this workflow. Use Speech's default local
source for scratch timing when no recording is supplied; state that it is scratch
speech. Do not ask for WPM estimates or require a recording to proceed. A later
recording can replace the speech source without rewriting the animation blocks.

Reuse decisions and assets already supplied. Resolve mechanical details with
minimal, stated assumptions. Consult the user about creative decisions as their
instructions direct; without such instructions, ask about a missing creative
decision only when it really blocks the requested result. Do not silently replace
a required effect with an approximation that changes the vision. Explain any
actual limitation.

## Production workflow

1. **Identify the deliverable.** Extract the specified content, assets, durations,
   aspect ratio, resolution, frame rate, audio requirements, output format, and
   transparency/compositing requirements. Keep user text and narration exact.
   Distinguish a runnable source project from a fully exported production video.
   For a long multi-scene request without an explicit full-export requirement,
   deliver the complete source, review frames and representative clips. Do not
   automatically launch a full production export. An explicit export request
   authorizes it; an explicit request to skip rendering takes precedence.
2. **Check the environment.** Read [setup and rendering](references/setup-and-rendering.md).
   Reuse a working interpreter and Algan installation if one exists. Run
   `scripts/check_environment.py` with that interpreter and use `algan check`
   from the same environment. Install only needed dependencies. Record the actual
   Algan version/source location. These references describe the commit above;
   prefer the installed version's verified API when it differs.
3. **Select the relevant references.** Use the table below. Load advanced sections
   only when needed. Prefer public Algan methods and existing materials; write
   custom callbacks or shaders when the requested behavior calls for them.
4. **Author the scene.** Keep initial placement and static configuration separate
   from animation. Keep assets relative to the project, construct reusable
   functions for repeated mechanics, and separate scene authoring from export.
   Use `Project` for independently rendered scenes when useful; for a multi-scene
   narrated video, start from the layout in `examples/project_template/`.
   Place named `Scene.save_frame` checkpoints throughout the script. Authoring
   checkpoints does not mean rendering every checkpoint on every iteration.
   ```python
   Scene.save_frame("important_moment", at=[0.0, 2.5, 5.0])   # check_0.0.png, check_2.5.png, ...
   ```
   `at` is seconds on the timeline; a negative value counts back from the current
   time. Use `at` to get stills before, during, and after key animations.
   Give the stills informative names describing which part of the script they are showing.
5. **Validate, then inspect selected renders.** Use `project.validate()` first to
   author every scene and resolve Speech timing without rendering. This can
   generate/cache speech and update transcripts; it does not execute render-time
   updaters or prove visual correctness. When the user supplied a script, pass it
   (`project.validate(script=Path("script.txt"))`, or `--validate --script
   script.txt`) so the Speech strings are checked against it word for word.
   Build a compact storyboard from selected checkpoints with
   `project.render_screenshots(contact_sheet=True)` (or `--render-screenshots
   --contact-sheet`), then render representative draft clips
   and review their motion with `scripts/contact_sheet.py video ... --fps 1`
   (a sampled sheet catches framing and motion problems between checkpoints; it
   does not replace watching for timing and audio). Inspect starts, transitions,
   updater-dependent motion, final holds, text, texture orientation, shadows,
   shader output, and sound timing against the user's specification. A draft
   tests implementation, not a license to redesign. Use the intended renderer for
   a representative final-quality test. If a scene renders much slower than the
   others or runs out of memory, profile it before changing anything; see
   [performance](references/performance.md).
6. **Deliver the requested scope.** Export the full video only when it is part of
   the requested deliverable. For source projects, provide the complete runnable
   project, reviewed samples, and full-export command. For requested exports,
   use the requested format and settings. Check the
   returned output path, file existence, metadata, decoding, audio, and alpha
   when relevant. Provide the video, reproducible source, required asset manifest,
   exact invocation, and any known limitations. Never label unexecuted code or an
   uninspected render as verified.

## Task-to-reference map

| Need | Read | Starting example |
|---|---|---|
| Installation, settings, rendering, output paths | [Setup and rendering](references/setup-and-rendering.md) | `examples/basic_scene.py` |
| Transforms, timing, easing, lifecycle, morphs | [Animation and updaters](references/animation-and-updaters.md) | `examples/basic_scene.py` |
| Followers, continuous motion, custom animations | [Animation and updaters](references/animation-and-updaters.md) | `examples/updaters.py`, `examples/custom_animation.py` |
| Shapes, grouping, layout, text, equations, image/model assets | [Geometry, text, and assets](references/geometry-text-and-assets.md) | `examples/text_and_texture.py` |
| Materials, glass, lights, cameras, renderer choice | [Materials, lights, and camera](references/materials-lights-and-camera.md) | `examples/fragment_shader_taichi.py` |
| A custom PyTorch vertex shader or compiled fragment pipeline | [Custom shaders](references/custom-shaders.md) | `examples/vertex_shader.py`, `examples/fragment_shader_taichi.py` |
| Audio, recorded narration, multiple scenes | [Audio and projects](references/audio-and-projects.md) | `examples/multi_scene_project.py`, `examples/project_template/` |
| Slow renders, out-of-memory, many objects, heavy text | [Performance](references/performance.md) | `profile_scene`, `batch_mobs` |
| Reviewing stills and motion | Production workflow step 5 | `render_screenshots(contact_sheet=True)`, `scripts/contact_sheet.py` |
| Backgrounds, bloom, post-processing, transparent overlays | [Compositing and post-processing](references/compositing-and-postprocessing.md) | See reference snippets |
| Failed or incorrect output | [Troubleshooting](references/troubleshooting.md) | `scripts/verify_video.py` |
| API drift or uncertain support | [API sources](references/api-sources.md) | Inspect only the relevant installed API |

## Essential Algan rules

**Do not call `Scene.view()`** in an unattended run. It starts a web viewer and
blocks until Ctrl-C.

**Keep all code below `from algan import *`.** A warm render daemon may re-run
the script in another process; lines above the import execute twice.

**The first render is slow** (tens of seconds to minutes) while kernels
compile and cache. Later renders start warm. This is not a hang.

**Set `ALGAN_USE_DAEMON=0` in the environment if the daemon misbehaves** or you
need a fully isolated run. Between runs the daemon reloads every edited helper
module the script imported, wherever it lives. It keeps Algan, installed packages
and any user package containing a compiled extension loaded (it says so). When
Algan's own source changes it shuts down and runs the script in a fresh process.

**Cancel through `algan daemon cancel`.** It interrupts the active script and
keeps queued scripts. Wait for cleanup before retrying. For older versions or a
failed recovery, see the [Windows recovery recipe](references/troubleshooting.md#cancellation-and-recovery).

**Always prefer screen-relative helpers** when laying out Mobs on the screen.
Methods such as `Mob.fit_to_screen` position and scale mobs to take up
a specified rectangular portion of the screen, without you needing to
figure out the exact world coordinates.

**Authoring records a timeline.** Rendering materializes that recording later,
possibly in parallel frame batches. Do not implement frame-by-frame animation
with a Python sleep loop or assume callbacks run once in chronological order.

**Configure, then spawn, then animate.** Unspawned objects can be positioned
without consuming animation time. Visible objects must be spawned. Set materials
(or `unlit=True` at construction), shaders, `two_sided`, `casts_shadows`, and
`receives_shadows` before spawning.
Use `with Off():` for instantaneous setup, including camera and light changes.
`spawn()` normally animates appearance; use `Off()` or `spawn(False)` for an
instant appearance. Do not accidentally spend the opening seconds constructing
objects or lights one after another.

**Timing belongs in contexts.** `Seq` is sequential, `Sync` simultaneous,
`Lag(ratio)` staggered, and `Off` instantaneous. Use `runtime`, not `run_time` or
`duration`. `runtime` is the whole block's duration; `runtime_per_part` is per
child animation. Put ordinary Mob calls inside a context instead of passing
timing keywords to them. Specialized convenience APIs can have their own timing
arguments; check their signatures rather than generalizing.

**Use Algan units and names.** Every angle in the root API is in degrees:
transforms (`rotate`, `look`, camera `fov`) and constructor angles such as
`RegularPolygon(start_angle=...)`, `Line(path_arc=...)`, `Arc(angle=...)` and
`Wiggle(rotation_angle=...)`. Only `algan.manim` classes take Manim's radians. A
root angle that looks like radians (for example `PI / 4`) warns; fix the value
rather than ignoring the warning.
`move(v)` is a displacement; `move_to(p)` is an absolute destination. Use
`easings.identity` for a specified linear interpolation and `easing=` to select
a requested easing. Do not write Manim's `Scene.play`, `.animate`, or
`rate_func=`. Geometry imported through `algan.manim` has a separate convention;
see the geometry reference.

**Updaters are functions of elapsed time.** The callback is `(mob, t, ...)`;
`t` is seconds since attachment, as a torch tensor of shape `[frames, 1, 1]`.
It is not `dt` or a Python scalar. Use torch operations and broadcasting. Do not
accumulate state, call `.item()` on frame batches, use `math.sin(t)`, create Mobs
per callback, or mutate external Python state. Keep the ID returned by
`add_updater` to remove that updater later. An updater does not create video
duration: author animations or `Scene.wait(...)` too.

**Animatable values and static configuration are different.** Assign properties
on the Mob after installing a material/shader, for example `ball.roughness = 0.4`.
Do not replace the material object after spawn. Background settings, many light
shape parameters, and clip planes are not timeline animations. Use separate
scenes or supported time-dependent callbacks when necessary.

**Filled 2D shapes have a default outline.** Native filled shapes draw a white,
screen-space outline (`stroke_width=5`) unless told otherwise; set the stroke the
specification calls for, or `stroke_width=0`. For a see-through fill with a solid
outline, use `fill_opacity=` and `stroke_opacity=` on one Mob rather than two
Mobs. See the geometry reference.

**Move the camera with `fly_to`.** `camera.fly_to(position, look_at=target,
via=..., look_at_via=...)` moves position and aim together with a level horizon;
`move` plus `look_at` in one `Sync` does not track the target.
`camera.visible_size_at(point)` gives the visible width and height at a point's
depth for framing, including portrait frames. See the camera reference.

**Colors are not uniformly RGBA.** Mob colors and native surface color textures
use `[R, G, B, glow, opacity]`. A vertex shader receives/returns RGB plus glow;
a fragment stage returns `vec4(R, G, B, glow)`, not alpha. Stage RGB parameters
need three components. Preserve the correct trailing channels.

**Preserve settings objects.** Use `SETTINGS.video.set(...)`, not replacement
assignment to `SETTINGS.video`. Presets are immutable: `HD.set(...)` returns a
new preset. `samples_per_pixel=1` selects the deterministic hybrid renderer;
values greater than one select the path tracer, not merely a higher-quality
version of the same algorithm. Test required effects in the final mode.

## Minimal complete authoring pattern

This demonstrates the mechanics only. Replace the object, changes, values, and
export settings with the user's requirements.

```python
from algan import *


def build_scene():
    with Off():
        shape = Square().scale(0.5).move_to(LEFT).spawn()

    with Sync(runtime=1.5):
        shape.move(RIGHT * 2)
        shape.rotate(90, OUT)

    Scene.wait(0.5)


if __name__ == "__main__":
    build_scene()
    result = Scene.save_video("renders/example.mp4", PREVIEW)
    print(result.output_path)
```

Ordinary methods on spawned objects and assignments to animatable attributes
record changes. A Python function containing several such calls does not
automatically make them simultaneous; give it an appropriate context.

For a custom motion with finite duration, use `@animated_function` with explicitly
interpolated arguments. For a continuing rule, use an updater. For changing
surface appearance per pixel, use a fragment stage. These are different tools,
not interchangeable forms of a generic callback.

## Delivery and verification contract

Run examples as real `.py` files using the chosen environment. Adapt their
preview settings before a final delivery; they are not production specifications.
Do not upload assets to external services, use paid synthesis, or overwrite
unrelated files without authorization. Do not include font files in deliverables.

`scripts/verify_video.py` checks a local video with FFprobe and can decode it with
FFmpeg. It does not inspect visual correctness or prove meaningful alpha data;
inspect images and a composited test for those. Check `RenderResult.status` so
an existing skipped file is not mistaken for a new render.

When rendering is unavailable, deliver the scene source and precise execution
instructions, state the concrete blocker, and distinguish syntax checks,
mathematical callback tests, and actual renders. Do not fabricate output paths,
frame inspections, performance figures, or execution success.
