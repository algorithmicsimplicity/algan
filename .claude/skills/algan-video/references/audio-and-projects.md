# Audio synchronization and multi-scene projects

Use only the sound, narration, voice, and timing the user requested or approved.
Do not introduce narration, music, or an external speech service as part of
using this skill. Do not upload private recordings or use paid synthesis without
authorization.

## Audio determines context timing

`Audio` is an animation context whose duration comes from a clip:

```python
with Audio('assets/audio.wav', wait_at_end=0):
    with Sync():
        first.move(RIGHT)
        second.move(LEFT)
```

The context rescales its contents to the clip duration. `Audio` defaults to no
extra hold; `Speech` defaults to a one-second hold after its clip. Set
`wait_at_end` deliberately when exact timing matters. Context nesting still
applies: a sequence inside a clip differs from simultaneous child actions.
Do not confuse a clip-length context with starting a permanent background track.

To use a trimmed existing clip with the source-checked MoviePy interface:

```python
from moviepy import AudioFileClip

source = AudioFileClip('assets/audio.wav')
try:
    segment = source.subclipped(1.0, 3.0)
    with Audio(segment, wait_at_end=0):
        subject.move(RIGHT)
    result = Scene.save_video('renders/with_audio.mp4')
finally:
    source.close()
```

The clip must remain available until rendering consumes it. Closing it directly
after recording the `Audio` block can be too early because authoring is lazy.
Use exact user-provided timestamps and verify the final stream length and audible
synchronization. Aligning durations does not prove words or sound events match
particular visual changes.

## Speech from an approved source

A `Speech` context asks the scene's speech source for an audio clip corresponding
to the supplied string. It does not decide what should be said:

```python
with Speech('The exact approved line.', wait_at_end=0):
    subject.move(RIGHT)
```

The default source uses pyttsx3 and a system text-to-speech engine. A headless
Linux environment may need eSpeak NG; a successful Algan import does not establish
speech availability. With a transcript and no recording, use this local default
for scratch timing and identify it as scratch speech; a transcript-only task does
not require a WPM estimate or a question about missing audio. Honor an explicit
final voice requirement separately. When recordings are supplied, use them as
the speech source.

## Keep each narration segment beside its animation

Choose Speech boundaries by meaning: the phrase introducing an object, the
sentence explaining a transformation, or the paragraph accompanying one coherent
visual beat. Put the exact words directly beside their actions:

```python
from algan import *

def introduction():
    with Off():
        square = Square().spawn()
    with Speech('The square moves to the right.', wait_at_end=0):
        square.move(RIGHT)
    with Speech('Now it turns.', wait_at_end=0):
        square.rotate(90)
```

Use the user's actual transcript; these lines only demonstrate the API. A block
can contain multiple actions, nested `Seq`/`Sync` contexts and reusable visual
helpers. One block per sentence is not mandatory, and block duration alone does
not align every internal action to an individual word. Split at meaningful cues
when tighter alignment is needed, then verify the actual narration.

Avoid `split_speech(text, count)` plus `tick(index)` or indexed callback lists.
If a splitter returns fewer chunks than planned visual beats, some actions may
never be authored. Writing explicit blocks avoids that indirection; it is not
a workaround for a Speech defect.

Keep the supplied transcript as canonical text. Verify the ordered Speech strings
against it with `project.validate(script=Path("script.txt"))` (from `pathlib`),
or `python project.py --validate --script script.txt` through `run_cli`. A
string is taken as the literal script text. Whitespace is ignored; words,
punctuation and capitalization must match. A mismatch raises
`AlganConfigurationError` naming the first differing word, the scene it falls
in, and the words around it in both texts; only the selected scenes are
compared, in project order. It compares each scene report's unwrapped
`transcript`, so pretty-printed transcript files that wrap a hyphenated word
(`triple-\nnested`) do not cause false mismatches. Do not edit the original
script to satisfy the comparison or silently drop punctuation.

For transcript-aligned recorded narration:

```python
from algan.utils.audio_utils import get_speech_generator_from_file

generator = get_speech_generator_from_file(
    audio_file='assets/narration.wav',
    transcript_file='assets/narration.txt',
)
Scene.current().audio_manager.set_speech_source(generator)

with Speech('An exact segment from that transcript.', wait_at_end=0):
    subject.move(RIGHT)
```

Alignment has optional dependencies, available through `algan[audio]`. Test that
the actual requested words resolve to the correct subclip, especially when words
or phrases repeat. Do not claim a full narration has been aligned when only a
single segment was tested.

A custom source is a callable accepting a script string and returning a MoviePy
audio clip for that string. It belongs to the **Scene's** audio manager:
`Scene.current().audio_manager.set_speech_source(callable)`. It is not a
process-global voice setting. Do not replace this with internal singleton state.
An audio-bearing video can write an accompanying `<stem>_script.txt`; include it
when relevant, but do not mistake a transcript for captions burned into frames.

When effects and speech overlap, express that with nested `Sync`/`Audio` contexts
and verify the resulting overlap. Audio duration, visual duration, and requested
holds must be accounted for before concatenating scenes.

## A Project owns scene rendering

A Project takes zero-argument functions that **author** scenes. Those functions
do not call `Scene.save_video()`; the Project handles the render:

```python
from algan import *

def scene_a():
    with Off():
        subject = Square().spawn()
    with Seq(runtime=1):
        subject.move(RIGHT)

def scene_b():
    with Off():
        subject = Circle().spawn()
    Scene.wait(1)

project = Project([scene_a, scene_b], file_path='renders/complete.mp4')

if __name__ == '__main__':
    project.run_cli()
```

These functions demonstrate mechanics, not a recommended sequence for the user's
video. Put real scene functions in separate modules when that makes the project
easier to manage, and import them into a main script without rendering on import.
Construct fresh Mobs inside each scene function so their ownership is correct.

The Project can render all scenes or a subset:

```python
project.render_video()
project.render_video(1)
project.render_video('scene_b')
project.render_video([0, 'scene_b'])
project.concatenate_videos()
```

Scene IDs are their zero-based list positions, not their order in a subset render.
Names such as `0_scene_a` are stable for a fixed list. Changing the list can change
IDs, so do not concatenate old outputs under an assumed unchanged mapping.
Rendering and concatenation are separate actions.

Typical invocations for the bundled project example:

```bash
python examples/multi_scene_project.py --render-video --video-settings PREVIEW
python examples/multi_scene_project.py --render-video 1 --video-settings HD
python examples/multi_scene_project.py --render-video --video-settings HD
python examples/multi_scene_project.py --concatenate-videos
```

The third command brings all scenes to the same final settings before the fourth.
Do not concatenate a mix of draft and final clips because only the last edited
scene was re-rendered. Check the actual files and overwrite/skip status. Use
`project.render_video(video_settings=...)` or constructor `video_settings=` for
programmatic control; do not change settings after computing screen layout and
assume all authoring choices will update retroactively.

`Project` also accepts a `speech_source` to install across scenes, and directory
options for videos, screenshots, and transcripts. Inspect the installed
signature for the particular paths needed. A filename alone follows Algan's
output-path conventions; an explicit parent directory controls the path directly.

## Still-image iteration

Scene functions may include `Scene.save_frame(...)` calls for deliberate
checkpoints. Project video rendering skips those calls, while
`project.render_screenshots(...)` runs the screenshot workflow rather than
exporting the scene video. The screenshot API supports selecting checkpoints
with `frames=` and optional early stopping. Use it when it reduces iteration
work, not as a substitute for checking time-dependent motion and audio.

Validate authoring before spending time on renders, then select review frames:

```python
report = project.validate()
print(report.duration_seconds)
project.render_screenshots(frames=['initial', 'linearized', 'summary'])
project.render_video('scene_b', video_settings=PREVIEW)
```

The frame names above must match actual named checkpoints. Inspect one useful
storyboard frame per scene, plus extra frames for key transformations. Re-render
only changed scenes/checkpoints once that coverage exists. `stop_early=True`
abandons later authoring and leaves that scene's transcript and later frames stale;
use it for local iteration, then run validation without early stopping.

For review, `project.render_screenshots(contact_sheet=True)` also writes the
selected stills as one labelled image, `contact_sheet.png` in the screenshot
directory (pass a path instead of `True` to choose another, and
`contact_sheet_columns=` for the grid width); `project.last_contact_sheet_path`
holds where it went. Frame and scene selections apply to the sheet too.

CLI equivalents include `--validate [--script FILE]`, `--render-screenshots
--frames NAME [--contact-sheet [PATH]]`, and `--render-video SCENE
--video-settings PREVIEW`. Check installed help first on older Algan versions.
Do not emulate validation with a deliberately nonexistent frame selector: it
produces misleading unmatched-frame warnings.

For an explicitly requested cost estimate, use
`project.estimate_render_time(['scene_a'], video_settings=HD)` or
`--estimate-render-time scene_a --video-settings HD`. This authors all scenes
and renders the selected reference scenes twice by default. Choose short,
representative scenes; it is not a free check or a promise for unsampled effects.
`project.profile('scene_a')` / `--profile scene_a` exposes the existing
`profile_scene` stage report for diagnostics without manually wrapping the scene.
Do not profile an entire long project merely to decide whether to export it.

`Project.run_cli()` handles project actions and reports whether it dispatched
one; invoking a project script without an action is not proof a video was rendered.
Always inspect the command's actual result and deliver existing files only.

When the user's specification replaces the default post-processing passes (for
example a custom bloom), give them to the Project once:
`Project(scenes, post_processes=[partial(bloom_filter, glow_spread=0.015)])`
(`partial` from `functools`, `bloom_filter` from
`algan.rendering.post_processing.bloom`; the values are placeholders). They then apply to videos, stills and profiling, including through `run_cli`.
An explicit `post_processes=` on `render_video(...)` / `render_screenshots(...)`
overrides that default, and an empty sequence disables the passes.

### Project layout

`examples/project_template/` is a runnable multi-scene layout: a driver
(`project.py`, a thin wrapper over `Project.run_cli()` that picks the draft or
final settings), a shared `common.py` (output settings, post-processing passes,
a `say()` wrapper that makes the Speech hold explicit), and one module per scene
under `scenes/`. The render daemon reloads edited helper modules wherever they
live, so helper scripts in subfolders can import `common.py` from the root.

Source basis: [audio and project sources](api-sources.md#audio-and-projects).
