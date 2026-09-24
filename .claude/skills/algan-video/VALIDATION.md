# Validation record

## Update 1.3.0 (2026-09-24)

Checks ran on Windows 10 with Python 3.11.6 and Algan 0.0.2 from a source
checkout (`45c1d6a3` plus uncommitted local changes: component-opacity keeping,
`fly_to` roll easing and `look_at_via`, degree-only root angles, Bezier outline
reuse under a moving camera, script-mismatch context, daemon handling of
compiled packages), rendering on CUDA (GeForce GTX 1050).

| Check | Confirmed result |
|---|---|
| python syntax | 15 Python files compiled. |
| skill frontmatter | Valid YAML; version 1.3.0; 262 core lines; description 285 characters. |
| markdown links | 34 local links resolved, including section anchors. |
| markdown python syntax | 58 snippets parsed. |
| component opacity | The geometry reference's `RoundedRectangle(fill_opacity=0.2, stroke_opacity=1, ...)` snippet: after `fill_opacity = 0.6` then `color = RED`, `fill_opacity` read 0.6; rendered translucent red fill with a solid white outline. Frames inspected. |
| unlit constructor | `Line3D(..., unlit=True)` rendered its flat color beside the same line lit. Frames inspected. |
| batch_mobs star import | Pack built from `from algan import *` alone; members faded in with `Lag` and one recolored. Frames inspected. |
| root angles | `RegularPolygon(start_angle=PI / 2)` and `Line(path_arc=PI / 4)` each raised the "looks like radians" warning; `Line(path_arc=60)` rendered a 60-degree arc. |
| camera | The materials reference's `fly_to` snippet (Off establishing shot, then a 3 s move with `via`) rendered with a level horizon; `visible_size_at(ORIGIN)` returned width/height 0.5625 for a 540x960 frame. Frames inspected. |
| project template | Thin `run_cli` driver: `--validate --script script.txt` matched; a one-word change reported `word 13 ... in scene '0_first'` with context. `--render-screenshots --contact-sheet` wrote four stills and a labelled sheet (inspected); `--render-video` rendered both scenes at DRAFT; `--concatenate-videos` joined them. |
| daemon module reload | A script in a subfolder importing `common.py` from its parent printed the edited value on each of three runs (1, 2, 3; the last two in the warm daemon, 0.3-0.4 s renders). |

The performance figures in `references/performance.md` come from a benchmark of
30 lines of monospace text under a moving camera (three passes per variant,
warm pass quoted); they are one measurement, not a general benchmark.

## Update 1.2.0 (2026-09-23)

Checks ran on Windows 10 with Python 3.11.6, PyTorch 2.7.1+cu128 and Algan 0.0.2
from a source checkout (`91685dbf` plus uncommitted local changes) rendering on
CUDA (GeForce GTX 1050), after using the skill for a two-minute, eight-scene
narrated video.

| Check | Confirmed result |
|---|---|
| python syntax | 15 Python files compiled. |
| skill frontmatter | Valid YAML; version 1.2.0; 265 core lines; description 285 characters. |
| markdown links | 34 local links resolved, including section anchors. |
| markdown python syntax | 55 snippets parsed (one list-indented snippet after dedenting). |
| project template | `validate --script` authored both scenes with scratch speech and matched the script; a deliberately extended script produced a word-level mismatch report. `stills`, `draft` (PREVIEW) and `concat` rendered with Algan; stills inspected. |
| contact sheets | `contact_sheet.py` built sheets from checkpoint stills and from frames sampled out of a rendered Algan clip; a non-matching pattern exited 1; `--help` works. |
| text as image | An `ImageMob` built from a PIL-rasterized `[H, W, 5]` array rendered upright with its colors; `set_color_by_image` inside `Off()` swapped the image. Frames inspected. |
| packed Mobs | `batch_mobs` pack members animated individually (opacity stagger, color, move, rotate, glow); frames inspected. |
| camera rig | The roll-free position+target camera move rendered as intended, including a curved `via` path; frames inspected. |
| block-spanning change | `Sync(equalize_runtimes=True)` stretched a bare 1 s move and a wrapped move to a 2 s sibling sequence; with a longer explicit filler the content was stretched instead. In the template, the camera move spans the whole first Speech clip (timeline queried). |
| daemon module reload | A script at the project root picked up edits to `common.py` between daemon runs; a script in a subfolder importing `common.py` from its parent kept the stale module across three runs. |

Not executed for this update: evaluation cases 11 and 12 are definitions and have
not been run by an agent. The timings in `references/performance.md` are one
measured data point, not a benchmark.

## Original record (1.1.0)

Created: 2026-09-10.

Algan source baseline: `f9e6d73c12de35f7e14315c2d49df34de570c644`.

## Executed checks

Checks ran with Python 3.13.5 and PyTorch 2.10.0+cpu on CPU. The shader and updater mathematical tests extracted self-contained functions; they did not import Algan.

| Check | Confirmed result |
|---|---|
| python syntax | 9 Python files compiled to code objects without executing imports. |
| skill frontmatter | Valid YAML; matching name; 195 core lines; description 535 characters. |
| markdown links | 27 local file links resolved. |
| markdown python syntax | 47 Python snippets syntax checked; not executed in Algan. |
| vertex tensor contract | 4 CPU tensor cases passed, including static color broadcast across animated frames; glow preserved; no input mutation. |
| fragment static contract | 16 arguments, live template annotation syntax, no future annotations, and preserved fourth glow component. Compiler execution NOT tested. |
| isolated updater arithmetic | Batched/single-time offsets agree; elapsed-time rotation and explicit target argument checked with a stub, not Algan. |
| evaluation definitions | 10 distinct behavioral case definitions parsed; these cases have NOT been run by an Algan agent. |
| utility cli help | Both utility --help commands exit successfully. |
| environment failure detection | Current environment correctly reported missing Algan/compiler rather than reporting render readiness. |
| video positive verification | Synthetic FFmpeg fixture (NOT an Algan render) passed dimension/rate/duration/audio checks and full decode; filename contains a space. |
| video negative verification | 5 failure cases correctly rejected: dimensions, rate, duration, audio policy, and missing file. |

The static-color/animated-parameter test initially exposed a shape mismatch in the vertex example. The delivered function explicitly broadcasts its unchanged glow channel before concatenating it with RGB; all four tensor cases passed after that correction.

## Not executed

Algan and its matching Quadrants/Taichi compiler are not installed in the validation environment. No example was imported into Algan, no shader was compiled by its backend, and no Algan video was rendered. The successful FFmpeg fixture was generated independently to exercise the video-checking utility; it is not an Algan demonstration render.

Visual correctness, GPU/backend behavior, text/font/LaTeX rendering, audio alignment or synthesis, and alpha compositing have not been tested end to end here. The ten behavioral evaluation cases are definitions for later agent evaluation, not claimed successful runs.

The fragment example was checked for its 16-argument contract, annotation syntax, and glow return component against the inspected source. Those static checks do not establish successful compilation.

## Before using the examples in production

Use the intended Algan environment, run its diagnostics, and render the relevant small examples there. Check batched callbacks and shader animation in the intended renderer, then inspect actual frames, audio, and any compositing intermediate against the user’s requirements. Record the installed Algan version and any deviations from the pinned API. Only report tests that actually ran.
