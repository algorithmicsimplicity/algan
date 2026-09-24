# Video-production troubleshooting

Diagnose the user's scene and execution environment. Do not turn a video task
into a renderer-development project. Preserve the requested visual result and
report any workaround that changes it.

## Check the failing stage

Distinguish import/setup errors, authoring errors, materialization/callback errors,
shader compilation, rendering, encoding, and output verification. Record the
actual exception and the smallest relevant source fragment. An error that appears
at export can originate in a callback recorded much earlier.

| Symptom | First checks |
|---|---|
| Algan imports in one shell but not another | Python executable, venv, source location, shadowing files named `algan.py` |
| Missing compiler/import conflict | Matching Algan dependencies; conflicting distributions sharing `quadrants`; use Algan's own diagnostics |
| Text/LaTeX fails | Optional Pango/TeX dependencies, `dvisvgm`, exact font availability, raw LaTeX strings |
| Pango says “error while writing to output stream” | Actual cache path and write access; run `algan check` in the rendering environment. `mkdir(exist_ok=True)` alone does not prove an existing directory is writable. |
| Wireframe edges vanish as the camera rotates | Plain `Line` is planar geometry; use `Line3D` cylinders for connections that need thickness from every angle. |
| No visible objects | Spawned lifespan, opacity, camera pose/clipping, model scale, texture opacity channel |
| Unwanted opening seconds | Sequential default `spawn()` calls outside `Off()` |
| Changes happen sequentially instead of together | Missing `Sync()`; plain Python helpers do not establish timing contexts |
| A whole block has the wrong duration | `runtime` rescales the whole block; distinguish `runtime_per_part`; check enclosing contexts and Speech's end hold |
| Unexpected timing keyword error | Use `runtime`/`easing` on a context, not `run_time`/`rate_func` on a transform |
| Updater fails only during rendering | `(mob, t)` signature; tensor time; broadcasting; `math`/`.item()`/Python conditionals |
| Motion depends on batch size or render order | Incremental callback state, random calls, mutable previous-frame values, CPU-only allocations |
| A follower tracks the wrong object | Loop closure capture or a target position captured outside the callback |
| Updater scene has no duration | Add intended animation or `Scene.wait`; attaching an updater does not advance time |
| Shader/material cannot change | Installation happened after spawn; animate registered parameters or configure a fresh unspawned replacement |
| Vertex shader has a shape mismatch | Nine fixed arguments and `[..., 4]` RGB+glow output; frame/vertex dimensions preserved |
| Fragment stage does not compile | Real source file; exact 16-argument signature; live `ti.template()` annotations; no future-annotations import; no torch/NumPy in the compiled body |
| Shader animation changes nothing | Correct registered Mob parameter name; suffix collisions; static Python closure instead of an animatable parameter |
| Texture is missing or rotated | UV availability, HWC versus WHC entry point, file path versus native tensor, warnings, opacity at native channel 4 |
| Material property accepted but no effect | Unsupported slot warning or wrong renderer assumption; check the actual implementation |
| A mesh lacks expected shadows/light response | Custom vertex shader limitation versus built-in/fragment path; cast/receive flags before spawn |
| Camera does not keep aiming at its orbit center | `orbit` moves position without rotating aim; use the required rotation or targeting updater |
| Caption drifts during camera motion | Screen placement is a snapshot; parent to camera or use an updater |
| Imported character does not deform | Rigid-node animation is not skeletal skinning |
| Audio fails or is silent | Actual audio asset, source lifetime until render, optional synthesis/alignment dependency, encoded audio stream |
| CLI quality appears ignored | Script or Project explicitly selected another preset |
| Output is in a different place | Bare-name output directory versus explicit path; inspect `RenderResult.output_path` |
| Render finishes but output is old | Check returned status and overwrite/skip behavior |
| Alpha or glow looks wrong in a viewer | Correct alpha-capable export and linear premultiplied interpretation; inspect actual decoded data and composite |
| Edits to a helper module have no effect; traceback lines do not match the source | Current daemons reload every edited user module. An older daemon (before this skill's source commit) reloads only modules under the script's directory; a package with a compiled extension is never reloaded. Run with `ALGAN_USE_DAEMON=0` or `algan daemon quit` and retry |
| Small filled shapes render as white rings or blobs | Default outline (`stroke_width=5`, white, screen-space); pass `stroke_width=0` or an explicit stroke |
| A constructor rejects `opacity=`/`glow=` | An `algan.manim` class (Manim's keywords only) or an older Algan; use the root class, or set the attribute after construction |
| An angle comes out tiny or wrong, or warns "looks like radians" | Root angles are degrees (`RegularPolygon(start_angle=90)`, `Line(path_arc=60)`); `algan.manim` classes take radians |
| A translucent box turns opaque after a color change | Color assigned with its own alpha 1 over an alpha set some other way; set `fill_opacity=` / `stroke_opacity=`, which later color assignments keep |
| A lit 3D line or surface looks dark or grey | Default lit material under the current lights; construct it with `unlit=True` (or install `UnlitMaterial()` before spawn) if it should show its own color |
| CUDA out of memory raised from the animation timeline (`_query_row_states`) | Many or large textured Mobs, or an animated `color_texture`; see [performance](performance.md#textures) |
| One scene renders far slower than the others | Profile it; hundreds of `Text` glyphs or many separate Mobs are common causes; see [performance](performance.md) |

## API drift

This skill is pinned to the revision in SKILL.md, not to every PyPI release or
future checkout. Inspect the relevant installed public signature and help text
before replacing a name. Use [API sources](api-sources.md) to find the correct
module when documentation conflicts. Do not infer support from a design proposal
or from an accepted-but-ignored material keyword.

Source documentation at the checked revision has a few inconsistent passages:
an older texture note denies map forwarding, and an older fragment-shader
docstring denies path-tracer pipeline support. The current material implementation
and pipeline contracts support those features. Read the relevant implementation
and verify a small example when encountering such a conflict, rather than copying
the oldest prose into new scene code.

Do not suppress all warnings. Unspawned targets, dropped material maps, missing
lighting support, and output-policy warnings can explain a visibly incorrect
result even when the process exits successfully.

## Renders that are slow or fail on a backend

Confirm the actual renderer and device with Algan's diagnostics. GPU encoding is
not evidence of GPU rendering. Separate initial imports/compilation from a warm
render; do not repeatedly clear caches. Keep custom shader definitions stable,
reuse assets, and re-render only changed Project scenes while iterating.

Reduce draft resolution or render a short representative interval/scene using
supported authoring methods. A CPU fallback can establish whether the scene is
otherwise valid, but it may change runtime substantially; disclose that choice.
Do not silently reduce the final sample count, drop transparency layers, remove
lights, or replace required shaders to meet a runtime goal. Do not guess internal
memory settings, patch kernels, or run unrelated repository CI as a video repair.

If a concrete library defect prevents the requested output, preserve a minimal
reproduction and identify the installed version. Deliver the usable scene source
and explain exactly which requested result is blocked. A code workaround may be
appropriate only when its visual consequences are stated. Do not claim a video
exists when only source code was generated.

## Cancellation and recovery

Use the environment's public cancellation command and wait for the render client
to finish cleanup. On Windows/PowerShell, with the user's interpreter location:

```powershell
& 'D:/algan/.venv/Scripts/algan.exe' daemon cancel
```

The reply acknowledges the request, not completed cleanup. The active script
receives KeyboardInterrupt, workers finish unwinding, and recent Algan versions
reset compiler state before another job runs. Repeated requests are coalesced.
Queued scripts are kept, so coordinate with other work using the same daemon.
Do not invent a socket script or force-kill Python as the normal cancellation path.

If an older daemon lacks cancellation or a subsequent render fails with a
FieldsBuilder/runtime error after interruption, stop the idle daemon and retry
the affected sample in a fresh process:

```powershell
& 'D:/algan/.venv/Scripts/algan.exe' daemon quit
$previousDaemonSetting = $env:ALGAN_USE_DAEMON
try {
    $env:ALGAN_USE_DAEMON = '0'
    & 'D:/algan/.venv/Scripts/python.exe' project.py --render-video intro
} finally {
    if ($null -eq $previousDaemonSetting) {
        Remove-Item Env:ALGAN_USE_DAEMON -ErrorAction SilentlyContinue
    } else {
        $env:ALGAN_USE_DAEMON = $previousDaemonSetting
    }
}
```

Use the actual project filename and scene selector. Verify the newly returned
output; an old MP4 left on disk is not proof the cancelled run succeeded. Preserve
the error and a minimal reproduction if recovery still fails. Do not delete
unrelated caches or kill unrelated jobs.

For denied cache writes, use the exact path in the diagnostic. Configure a
writable `SETTINGS.paths.cache_directory`, or set `ALGAN_CACHE_DIR` before
starting Python. Kernel cache and daemon home are separately controlled by
`TI_OFFLINE_CACHE_FILE_PATH` and `ALGAN_HOME`; changing the content cache does not
move an initialized kernel cache. If sandbox access is the blocker, request the
appropriate grant rather than treating the native writer's error as a text bug.

## Verification levels

Use precise labels: source reviewed, Python syntax checked, tensor behavior
checked, shader compiled on a named backend, scene rendered, output decoded,
frames visually inspected, audio listened to, or alpha composited. These are
different checks. A source-only environment should not report render validation.
The bundled validation document records which checks were possible when creating
this skill, not a blanket certification of every scene made with it.
