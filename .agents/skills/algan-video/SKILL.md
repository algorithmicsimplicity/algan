---
name: algan-video
description: Create, edit, render, inspect, and refine animated video content with the Algan Python animation library, including 2D/3D scenes, mathematical text, camera work, lighting, materials, narration, and multi-scene projects.
---

# Algan Video

Use Algan to deliver finished animated video content, not just a code sketch. Work from the current Algan checkout or installed package, render at a cheap preview quality, inspect the result, iterate on visible problems, and only then make the requested final render.

## Source of truth

Algan changes quickly. Never rely on an old remembered API when the current checkout is available.

1. If you are inside an Algan source checkout, read `AGENTS.md` first.
2. For authoring questions, prefer the current files under:
   - `docs/source/new_user_tutorials/`
   - `docs/source/advanced_user_tutorials/`
   - `docs/source/reference/`
3. When documentation and implementation disagree, the source code wins.
4. Do not modify Algan itself unless the user asked for a library change. A video-authoring task should normally create or edit scene/project files only.

The compact current API reference bundled with this skill is in [references/API_QUICK_REFERENCE.md](references/API_QUICK_REFERENCE.md). Read only the sections needed for the task.

## Operating workflow

### 1. Resolve the deliverable

Infer the requested aspect ratio, duration, visual style, resolution, frame rate, narration/audio needs, and output path from the prompt and supplied assets. If something is unspecified, choose a sensible production default rather than blocking:

- explanatory/technical video: 16:9;
- draft render: `PREVIEW`;
- final quality when none is specified: `HD`;
- use a single scene for one short shot, and `Project` for a longer piece with independently renderable scenes.

Do not invent copyrighted assets, logos, narration recordings, or fonts the user did not supply or authorize. Prefer Algan-native geometry, text, LaTeX, gradients, materials, and generated procedural visuals.

### 2. Verify the environment before authoring heavily

Prefer the repository virtual environment when it exists:

- Windows: `.venv/Scripts/python.exe`
- macOS/Linux: `.venv/bin/python`

Otherwise use the active Python environment.

Check that Algan imports and, when available, run `algan check`. In a source checkout, use the checkout itself; do not silently fall back to an unrelated globally installed Algan.

If the checkout needs installation, prefer an editable install into the existing environment. Do not rebuild or replace a working environment unnecessarily.

For automated/remote agent runs, prefer a fresh process over the warm daemon unless the environment is intentionally using the daemon. `algan render ... --no-daemon` avoids stale process state and makes failures easier to attribute.

### 3. Design the shot before coding

For every scene, decide:

- what the viewer should notice first;
- what changes over time;
- which objects persist to maintain visual continuity;
- where labels/equations belong;
- whether camera motion adds information or only distraction;
- whether 3D lighting/material effects are actually needed.

For explanatory content, favor a small number of meaningful simultaneous motions over many unrelated motions. Keep important text and equations comfortably inside the frame. Prefer transforming or recoloring an existing object over replacing it when continuity helps the explanation.

See [references/PRODUCTION_WORKFLOW.md](references/PRODUCTION_WORKFLOW.md) for composition, pacing, QA, and performance guidance.

### 4. Author with Algan's recording model

A normal single-scene file starts with:

```python
from algan import *
```

Algan records changes lazily. Mobs must be spawned before they appear or animate.

Core rules:

- Create a Mob, position/style it, then call `.spawn()`.
- A normal animatable change takes one second unless a surrounding animation context changes the timing.
- Use `Off()` for setup that must be instantaneous.
- Use `Sync()` for simultaneous animation.
- Use `Seq()` for explicit sequences.
- Use `Lag(ratio)` for staggered animation.
- Use `runtime=` for total block duration and `runtime_per_part=` when each child should have the same duration.
- Use `Scene.wait(seconds)` or `mob.wait(seconds)` for holds.
- Apply materials, shaders, and geometry declarations before spawning.
- Assign the result of `become()` back to the variable.
- For a morph target that should never be drawn separately, construct it with `add_to_scene=False`.
- Treat the camera and lights as Mobs: they can be animated with the same movement/orientation methods.
- Use Three.js-style material classes such as `MeshStandardMaterial` and `MeshPhysicalMaterial`; do not reach for obsolete ad-hoc reflectivity APIs.

A minimal pattern:

```python
from algan import *

title = Text("A useful idea", font_size=72).move(UP * 2.5).spawn()
diagram = Circle(color=BLUE).spawn()

with Sync(runtime=1.5):
    diagram.move(RIGHT * 2)
    diagram.color = YELLOW
    title.scale(0.9)

Scene.wait(0.5)
Scene.save_video("scene")
```

For a longer video, use `Project` with zero-argument scene functions. Scene functions managed by a `Project` do **not** call `Scene.save_video()` themselves.

### 5. Keep output quality overridable

During ordinary authoring, prefer:

```python
Scene.save_video("scene_name")
```

without hard-coding a quality preset. That lets an agent or user choose quality from the CLI:

```bash
algan render scene.py --no-daemon -q preview
algan render scene.py --no-daemon -q hd
```

Pass a preset directly only when the file itself must pin the quality.

For still-image QA, `Scene.save_frame(...)` can render one or several timestamps without destroying the authored scene.

### 6. Preview-render before the final render

Do not jump directly to `HD`, `PRODUCTION`, or `UHD` unless the requested output is tiny or the user explicitly wants only a final render.

First render at `PREVIEW` or `LD`. If the scene is expensive, use `SMOKE_TEST` to prove the pipeline works before a visual preview.

Check:

- the render completed and the expected output file exists;
- duration and frame rate are plausible;
- all intended Mobs were spawned;
- no unintended Mob was left registered but unspawned;
- text/equations fit the frame;
- objects do not overlap unintentionally;
- camera motion frames the subject throughout;
- colors remain legible against the background;
- 3D forms have enough lighting contrast;
- transparency, glow, shadows, and reflections are intentional rather than accidental.

### 7. Visually inspect key moments

If your environment can inspect images or video, use it. Do not declare a visual result correct from code alone.

Prefer one of these:

- inspect the rendered preview video directly;
- render key timestamps with `Scene.save_frame(..., at=[...])`;
- extract representative frames from the encoded preview with FFmpeg.

Inspect the opening frame, the densest/compositionally hardest moment, the end state, and at least one mid-transition frame.

When you find a visual problem, edit the scene and re-render the cheapest output that can verify the fix.

### 8. Final-render only after preview QA

Render the user-requested final quality and path. If no final quality was specified, `HD` is a reasonable default for a finished 16:9 explanatory video.

For expensive 3D work, enable costly effects only when they visibly matter:

- shadows are off by default;
- more samples, supersampling, reflections/refractions, depth of field, and high resolutions can dominate render time;
- use preview settings while composing and reserve production settings for the final pass.

### 9. Return the actual deliverables

At completion, report:

- the scene/project source file(s);
- the final video path;
- any preview or QA frames worth keeping;
- the quality preset used;
- any missing optional dependency that prevented a requested feature.

Do not claim a video was rendered or visually validated unless you actually rendered or inspected it.

## Content-specific guidance

### Text and mathematics

Use `Text` for ordinary text and `Tex` for LaTeX. LaTeX strings should normally be raw strings. Keep labels short and use hierarchy through size, position, and color instead of filling the frame with prose.

For multipart formulas, split `Tex` into segments and animate segments with `get_segment(...)` when the explanation needs to emphasize one term.

### 3D scenes

The default camera is perspective and the third axis is `OUT`/`IN`. Use `Scene.get_camera()` to animate the camera. Use `Scene.get_light_sources()` or create explicit light Mobs when lighting must be controlled.

Set a material before spawning a 3D Mob. Turn on ray-traced shadows with `SETTINGS.raytracing.set(shadows=True)` only when the shot benefits from them.

### Narration and audio

Use `Audio(...)` to make a block inherit the duration of an audio clip. Use `Speech(...)` only when the speech source or system TTS dependency is available.

Narration is Scene-owned. For recorded narration, configure `Scene.current().audio_manager` rather than inventing process-global audio state.

### Transparent/compositing deliverables

If the user needs a foreground layer for external compositing, consult the current transparent-background and premultiplied-over documentation before choosing the codec/container. Do not assume ordinary alpha compositing is sufficient for additive glow.

## Common failure modes

- **Nothing appears:** the Mob was never spawned.
- **A setup action animates unexpectedly:** put it before spawn or inside `Off()`.
- **A material/shader call fails or does not take effect:** it was set after spawn.
- **A morph leaves an extra target in the scene:** construct the target with `add_to_scene=False`.
- **The rendered output ignores CLI quality:** the script/project pinned its own `video_settings`.
- **A `Project` produces duplicate unmanaged renders:** scene functions called `Scene.save_video()` themselves.
- **A camera orbit eases strangely:** use `easings.identity` for constant-speed mechanical/orbital motion.
- **Text works on one machine but not another:** check optional Pango/LaTeX dependencies with `algan check`.
- **Speech fails on Linux:** the system TTS engine may be missing; use recorded audio or install/configure the requested speech backend.
- **An agent rerun uses surprising old state:** bypass/restart the warm daemon and render in a fresh process.

## Bundled templates

- [templates/single_scene.py](templates/single_scene.py): a clean short explanatory scene.
- [templates/multi_scene_project.py](templates/multi_scene_project.py): a longer video split into independently renderable scenes.

Copy and adapt these rather than starting from a blank file when they match the task.
