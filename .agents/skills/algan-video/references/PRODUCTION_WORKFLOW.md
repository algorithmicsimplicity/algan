# Production workflow for Algan video agents

This guide is about making the rendered result effective, not merely syntactically valid.

## 1. Translate the prompt into shots

Before writing code, turn the request into a short shot plan. For each shot, write down:

- the concept or claim the viewer should understand;
- the main visual object;
- supporting labels/equations;
- the transition in;
- the action/change;
- the hold needed for comprehension;
- the transition out;
- approximate duration.

A short technical explanation is usually clearer as several simple shots than one scene containing every element at once.

Use `Project` when multiple shots should be independently rendered and concatenated.

## 2. Establish visual hierarchy

One element should dominate each moment. Use these levers in roughly this order:

1. position;
2. size;
3. motion;
4. brightness/color contrast;
5. glow or other effects.

Do not make every object bright, moving, and glowing simultaneously.

For equations and labels:

- keep margins around the frame;
- avoid long prose when a short label will do;
- keep font sizes consistent by role;
- use one accent color for the currently discussed item;
- prefer segment/glyph emphasis over replacing the entire formula.

## 3. Stage initial conditions explicitly

Prepare the opening composition before the first intentional animation. Positioning a Mob before it is spawned is naturally instantaneous; use `Off()` when several already-spawned objects need a coordinated instantaneous setup.

Avoid accidental one-second setup moves at the start of the video.

## 4. Pace for comprehension

Default one-second animations are a useful starting point, not a rule.

Typical heuristics:

- simple color/opacity emphasis: ~0.3–0.8 s;
- small object move: ~0.5–1.0 s;
- major transformation: ~0.8–1.8 s;
- camera reframe/orbit: ~1.5–4 s;
- post-explanation hold: ~0.3–1.0 s;
- title card: enough time to read comfortably.

Use `Sync` when actions express one conceptual change. Use `Seq` when the viewer must notice one step before the next. Use `Lag` for repeated/staggered structures.

Use `easings.identity` for motions that should be mechanically uniform. The normal smooth easing is usually better for entrances, exits, and explanatory moves.

## 5. Prefer continuity over cuts inside a shot

When two states represent the same conceptual object, animate the object between them. Good continuity devices include:

- moving to a new location;
- scaling;
- recoloring;
- rotating;
- changing material properties;
- `become(...)`;
- moving the camera while the object remains fixed.

Hard replacement is appropriate when the semantic object truly changes or when a scene boundary is clearer.

## 6. Use 3D only when it communicates something

3D is valuable for geometry, topology, physical structure, optics, spatial algorithms, or depth relationships. It is unnecessary overhead for a flat equation or simple chart.

For 3D:

- choose a camera angle that reveals shape;
- maintain subject framing through camera animation;
- ensure at least one useful key/fill light relationship;
- use materials to communicate substance, not as decoration;
- enable shadows only when depth cues matter;
- keep reflective/transmissive objects from obscuring the actual teaching point.

## 7. Preview cheaply and often

A reliable iteration ladder:

1. `SMOKE_TEST` if the pipeline or dependencies are uncertain;
2. `PREVIEW` while composing;
3. optionally `MD` for a sharper QA pass;
4. requested final preset only after the shot is visually settled.

Do not spend final-render time diagnosing layout.

For a CLI-driven scene that leaves quality unpinned:

```bash
algan render scene.py --no-daemon -q preview
```

Then:

```bash
algan render scene.py --no-daemon -q hd
```

Use a warm daemon only when you intentionally want rapid local rerenders and understand its process-state rules.

## 8. Inspect images, not just logs

A successful process exit proves only that rendering completed.

At minimum inspect:

- first visible frame;
- a frame during each major transition;
- visually densest frame;
- final frame.

Look for:

- cropping;
- unwanted overlap;
- illegible text;
- weak foreground/background contrast;
- z-order/occlusion surprises;
- bad camera framing;
- unintended transparency;
- overexposed glow;
- shadow/reflection noise;
- sudden jumps caused by wrong context timing.

If the environment cannot play video, render or extract still frames.

## 9. Validate the encoded deliverable

For a final video, verify:

- output path;
- container/codec is appropriate;
- resolution;
- frame rate;
- duration;
- audio presence when requested;
- alpha channel behavior when requested.

When a transparent compositing deliverable contains additive glow, follow Algan's current premultiplied-over export contract rather than assuming a conventional straight-alpha workflow.

## 10. Performance choices

Make expensive effects earn their cost.

High-impact cost drivers can include:

- output resolution and frame rate;
- supersampling/AA;
- path/ray-tracing samples;
- reflections/refractions and ray bounces;
- soft shadows and many lights;
- depth of field;
- dense 3D geometry;
- high-cost post processing.

Use a cheap approximation during composition if the final effect is expensive. Do not reduce quality in a way that changes the visual design you are trying to validate.

## 11. Handle assets deliberately

Use stable paths relative to the project when possible. Verify every supplied image, audio file, model, or font can be opened before depending on it deeply.

If a requested optional subsystem is unavailable:

- plain `Text` may need Pango/ManimPango for its preferred backend;
- `Tex` needs a TeX installation;
- `Speech` needs a working speech source/system TTS backend;
- imported 3D formats may need optional dependencies.

Run `algan check` and report the exact missing dependency. Prefer a graceful substitute only when it preserves the user's requested result.

## 12. Completion criteria

A video task is complete when:

- source is saved;
- preview was rendered;
- representative frames or the video were inspected;
- visible problems found during QA were fixed;
- final output was rendered at the intended quality;
- final file properties are plausible;
- the user receives the source and video paths/artifacts.

If the environment blocks rendering, provide the finished source plus the exact blocker and the command the user can run, but do not label the unrendered result as validated.
