# Release audit: what to fix before Algan goes on PyPI

> **Status: partly acted on, on this branch.** Fixed: §1, §3, §4, §6 (all four
> bugs; the auto-spawn default and `~/.algan` were kept by decision, the daemon
> knobs stay environment variables because they are read before `SETTINGS`
> exists), §7 except the Manim root-logger item (kept by decision), §8 except
> the five Three.js material spellings (kept by decision), §9, §10, §11, §12
> (OpenCV became a dev-only dependency: the tests decode frames with it, the
> library no longer imports it), §14. Still open: §2 (which Manim runs — a
> decision), §5 (docs deploy), §13 (repository weight), §15, §16, §17, and one
> leftover from §7: `algan --version` still takes ~3 s because the console
> script is `algan.cli:main`, so importing `algan.cli` runs the package
> `__init__`; the fix is an entry point outside the package. The findings below
> are kept in their original wording as the record of what was wrong.

**What this is.** A pre-release audit of Algan `0.2.2` at `3c03536` (branch
`claude/algan-release-audit-a06q1y`), carried out on 2026-09-02. The brief was to
find what will be *harder to fix after* the first public release: distribution
identity, dependency graph, import-time behaviour, public names and signatures,
default paths, error types, and repository contents. Renderer quality and
performance are out of scope; `RENDERER_WORK_QUEUE.md` already covers them.

**Method.** Seven parallel audits (packaging, import-time behaviour, public API,
core authoring engine, first-run experience, docs and repo hygiene, tests and
CI), each verified by running code where possible, then de-duplicated and
re-checked here. Every finding marked **verified** was reproduced in this
container: Ubuntu 24.04, 4 vCPU, **CPU only**, Python 3.11, torch 2.7.1,
Taichi 1.7.4, system `ffmpeg` and `latex` present. A CPU-only box cannot speak
for CUDA, MPS, Windows or macOS; where that matters the finding says so.

**Baseline.** `uv run -m pytest -q --fast` passes (464 tests). The PyPI name
`algan` is free. Determinism is byte-identical across two renders. Nothing is
written to the package directory or to `$HOME` by a bare `import algan`.

Severity key: **BLOCKER** — fix before the first upload; **HARD-LATER** — not
a blocker, but the fix is a breaking change or a permanent artifact once users
depend on it; **LATER-OK** — worth doing, fixable after release without
breaking anyone.

---

## Ranking

| # | Finding | Severity | Why it is here |
| --- | --- | --- | --- |
| 1 | [`Audio` and `Speech` contexts raise on entry](#1-audio-and-speech-contexts-raise-on-entry) | BLOCKER | A documented feature crashes on line 1; a one-line regression from the `duration`→`runtime` rename. |
| 2 | [Two Manims: the vendored copy is dead code and the real one compiles from source](#2-two-manims) | BLOCKER | The dependency identity of the package. `pip install algan` builds pycairo and manimpango from source on Linux/macOS. |
| 3 | [Vendored third-party code ships without its license notices](#3-vendored-code-ships-without-license-notices) | BLOCKER | MIT requires the notice in every copy; an uploaded PyPI artifact is permanent. |
| 4 | [Internal scratch notes are tracked in the public repo](#4-internal-scratch-notes-are-tracked) | BLOCKER | 173 files of agent briefs, logs and machine paths under `scratch_perf/`. |
| 5 | [The docs URL in the package metadata has no deploy path](#5-docs-url-has-no-deploy-path) | BLOCKER (verify) | `docs.yaml` only uploads an artifact; no `gh-pages` branch exists. Could not reach the site from this sandbox. |
| 6 | [Daemon: fixed port, cross-venv state file, unauthenticated `quit`/`render`](#6-daemon) | HARD-LATER | The default is a product decision; the four bugs inside it are not. |
| 7 | [`import algan` mutates the process: autograd off for ever, root logger reconfigured, stdout noise](#7-import-time-process-mutation) | HARD-LATER | Library consumers (notebooks, test suites, training scripts) will build on whatever ships. |
| 8 | [Public names that violate the one-name rule](#8-public-names-that-violate-the-one-name-rule) | HARD-LATER | Aliases and keyword names are the most expensive things to remove post-release. |
| 9 | [64 root classes show Manim's signature, docstring and radian defaults](#9-manim-inherited-signatures-and-docstrings) | HARD-LATER | `Arc(angle=PI/2)` silently draws a 1.57° sliver. |
| 10 | [Authoring-contract gaps: `bool(mob)`, negative runtimes, the `runtime=` guard](#10-authoring-contract-gaps) | HARD-LATER | Behaviours users will write against. |
| 11 | [Output-path contract inconsistencies](#11-output-path-contract) | HARD-LATER | Relative `output_path`, directory targets, `<stdin>.mp4`. |
| 12 | [Packaging: undeclared and over-declared dependencies, `py.typed`, wheel contents](#12-packaging) | HARD-LATER | Dependency lower bounds and `py.typed` are contracts. |
| 13 | [Repository weight: 149 MiB of video baselines in history](#13-repository-weight) | HARD-LATER | History rewrites get harder with every clone. |
| 14 | [Missing-tool errors: LaTeX, ffmpeg, pydub](#14-missing-tool-errors) | LATER-OK | The first thing a PyPI installer without TeX will hit. |
| 15 | [Metadata and hygiene](#15-metadata-and-hygiene) | LATER-OK | Stale conda recipe, no changelog, placeholder badge and logos, stale branches. |
| 16 | [Smaller correctness and UX items](#16-smaller-items) | LATER-OK | Collected so they are not rediscovered. |
| 17 | [Tests and CI](#17-tests-and-ci) | LATER-OK | Green at HEAD; the gaps are what CI structurally cannot see. |

---

## 1. `Audio` and `Speech` contexts raise on entry

**Verified.**

```
$ python -c "from algan import *
with Audio('t.wav'): Square().spawn()"
AttributeError: 'AudioFileClip' object has no attribute 'runtime'
```

`a837ea7` ("Renamed duration and friends to runtime") renamed two attribute
reads that belong to **moviepy**, not Algan:

- `algan/animation_timeline/animation_contexts.py:1108` — `audio_clip.runtime + wait_at_end`. moviepy clips have `.duration`. `Speech` funnels through `Audio.__init__`, so both contexts are dead, and every example in `docs/source/advanced_user_tutorials/audio_and_speech.rst` raises.
- `algan/scene.py:816` — `audio_clip.runtime = ...original_end`. This *sets* a junk attribute on a `CompositeAudioClip`, so the intended clamp of the audio track to the scene end never applies. No exception; the rendered audio length is simply wrong.

The full unit suite is green because no test enters `Audio` over a real clip:
`tests/unit_tests/test_project.py:14` stands in a `_SilentClip` whose attribute
was renamed to `runtime = 0` along with the code, so the test mirrors the bug
instead of catching it.

**Fix.** Restore `.duration` at both sites. Add one fast-marked smoke test that
enters `Audio` over a generated 1 s wav — the rename touched a foreign
attribute, and only a test that constructs a clip can catch that class of
error. Then re-check the other cross-library hits of the rename:
`algan/mobs/three_d_models/model_mob.py:494,550` read `clip.runtime` on
Algan's own `scene_data.py` dataclass and are fine.

## 2. Two Manims

**Verified** (clean venv from the built wheel, plus import tracing).

`algan/__init__.py:43-45` imports the vendored tree and *then* calls
`sys.modules.setdefault("manim", _vendored_manim)`. The `setdefault` never wins:
51 files under `algan/external_libraries/manim/` use **absolute** `from manim...`
imports (`_config/utils.py:1365` is the first to run, during the vendored
package's own `__init__`), so the *installed* `manim` is imported first and
owns the `manim` key.

```
>>> import algan, sys; sys.modules['manim'].__file__
'.../site-packages/manim/__init__.py'
>>> real Mobject is vendored Mobject -> False; two distinct ManimConfig objects
```

Consequences, in order of cost:

- **`pip install algan` compiles C extensions from source on Linux and macOS.** `manim>=0.18.0` (`pyproject.toml:37`) pulls `pycairo` and `manimpango`; neither publishes Linux wheels (pycairo's 21 wheels are all Windows). A stock `python:3.11-slim` or a fresh Ubuntu without `libcairo2-dev libpango1.0-dev pkg-config` fails with a `pkg-config` error. This is exactly the pain `algan/external_libraries/readme.txt` says the vendoring exists to avoid.
- The vendored copy (1.96 MB of a 7.5 MB wheel) is shadowed at runtime, but still costs ~1.1 s of the 4.1 s import (`python -X importtime`).
- Real Manim's `make_logger` runs at import and installs a `RichHandler` on the **root** logger at level INFO (see §7).
- With `manim` absent the failure is opaque: `RuntimeError: manim_adapters exclusion lists name classes the compatibility layer does not wrap: ['MarkupText', 'Paragraph', 'Text']` from `algan/mobs/manim_adapters.py:401`, not an `ImportError`.
- Only two first-party sites reach the vendored copy on purpose (`algan/mobs/text.py:618`, `algan/utils/manim_svg_cache.py:440`); `algan/mobs/manim_compat.py:19` resolves to the installed one.

**Fix — a decision, not a patch.** Either (a) make the vendored copy real:
rewrite the absolute imports to relative (or register the `sys.modules` shim
*before* line 43), drop `manim` from `dependencies`, and prune the nine
"from manim" entries at `pyproject.toml:28-37` that exist only for it; or (b)
delete `algan/external_libraries/manim`, depend on Manim honestly, and put the
system-package line (`libcairo2-dev libpango1.0-dev pkg-config`) in the README
*above* `pip install algan`. Either way, turn the `manim_adapters` failure into
an `ImportError` naming the missing package. This is the package's dependency
identity; changing it after users have written scenes against real-Manim
behaviour is a breaking change.

## 3. Vendored code ships without license notices

**Verified.** `algan/external_libraries/{manim,ground,sect}` are copied from
MIT-licensed projects. The only license text in the repo, the sdist or the
wheel is Algan's own (`algan-0.2.2.dist-info/licenses/LICENSE`, "Copyright (c)
2025 Algorithmic Simplicity"). `readme.txt` explains *why* code was copied and
carries no copyright line. MIT requires the copyright and permission notice in
all copies or substantial portions; a PyPI upload is a copy and cannot be
edited after the fact (only yanked).

**Fix.** Add `LICENSE` files with upstream text and copyright holders beside
each vendored tree (Manim Community; `ground` and `sect` are by lycantropos),
note the upstream version each was taken from, and list them in
`[project] license-files` so they land in `dist-info/licenses/`.

## 4. Internal scratch notes are tracked

**Verified.** `git ls-files scratch_perf | wc -l` → 173 (3.8 MB): agent briefs,
`chain*.log`, `det_*.log`, patches, probe scripts, absolute container paths
and internal branch names. `.gitignore` has `/scratch_*` ("Session scratch
(briefs for subagents...)") but the files were tracked before the rule was
added, so it does nothing. `benchmarks/renderer_audit/OX_AUDIT.md:3` is the
same category.

**Fix.** `git rm -r --cached scratch_perf` and review `benchmarks/renderer_audit/`.
Consider also whether `API_AUDIT_core_building_blocks.md` and
`RENDERER_WORK_QUEUE.md` belong at the repo root, where they are the first
thing a visitor sees after the README; `agent_guidance/` or `docs/dev/` would
hold them just as well.

## 5. Docs URL has no deploy path

**Verified in the repo; site not reachable from this sandbox.**
`pyproject.toml`, README badges, `CONTRIBUTING.md` and `conf.py`'s
`ogp_site_url` all point at `https://algorithmicsimplicity.github.io/algan`.
`.github/workflows/docs.yaml` builds Sphinx and uploads the HTML as a 7-day CI
artifact; there is no `gh-pages` push or `deploy-pages` step, and the remote has
no `gh-pages` branch. The `documentation` branch is a June 2025 snapshot with
`docs/` and `doctrees/`, not built HTML. If Pages is currently served from
somewhere by hand, that is fine for beta; it is not fine for the URL that goes
into permanent PyPI metadata.

**Fix.** Add a deploy step (`actions/deploy-pages` or `peaceiris/actions-gh-pages`)
to `docs.yaml` on `push: stable` (or `master`), confirm the URL resolves, then
publish. The structural build itself is clean: `--skip-examples --no-open` produced
one warning, a missing `dot` binary in this sandbox.

## 6. Daemon

The auto-spawn default is a product decision and is not questioned here. What
follows are bugs inside the mechanism, all **verified** except where noted.

- **Fixed port + state-file discovery disagree.** `algan/daemon.py:159` binds `ALGAN_DAEMON_PORT` (46711) while clients discover the daemon through `$ALGAN_HOME/daemon.json`. When the port is held — by another user, another `ALGAN_HOME`, or a stale daemon — the new daemon logs `trigger socket unavailable ... Address already in use` and exits, no state file is written, and `_autostart` re-spawns a full `import algan` (~5 s of CPU) on **every** run, for ever. Reproduced here with a fresh `HOME`: `[algan] the background daemon exited early`. Fix: bind port 0 by default and publish the port through the state file, which already carries it.
- **One registration for every virtualenv.** `_StateFile._payload` (`daemon.py:367-375`) is `{protocol, port, pid, token, env}` — no `sys.executable`, `sys.prefix` or `algan.__version__`. `ALGAN_HOME` defaults to `~/.algan` for every project, so `projB/.venv/bin/python scene.py` is executed by project A's interpreter and project A's Algan, resolving the script's imports against the wrong site-packages. The staleness digest (`daemon.py:313`) hashes the daemon's *own* source tree and cannot see this. **Code-read only** (one venv here). Fix: add interpreter, prefix and version to the payload and refuse client-side on mismatch, or key the state file by `sha256(sys.prefix)`.
- **`quit` and `render` skip the token.** `daemon.py:692-699` dispatches `render`, `quit` and `ping` before the `compare_digest` check that `run`/`cancel` get. Reproduced: a raw `b"quit\n"` with no token stopped a daemon. Any local process can stop another user's daemon or re-execute its `--script`. Fix: require the token for all verbs; on POSIX an `AF_UNIX` socket at mode 0600 makes the file mode the whole access control.
- **The prologue runs twice.** Everything above `import algan` executes in the client *and* again in the daemon (`prologue pid=4018` / `prologue pid=3863` in the probe). Exit codes forward correctly; `atexit` handlers do not run; stdin is `/dev/null`. All documented — in `daemon_client.py:48-50`, which no user reads. Fix: state it in the README's CLI section and in the `[algan] starting a background render daemon` line.
- Smaller: `~/.algan` and `daemon.log` ignore `XDG_STATE_HOME`/`XDG_CACHE_HOME` (users will accumulate GBs of Taichi cache there before it can move); the seven `ALGAN_DAEMON_*`/`ALGAN_*_DAEMON` knobs are read live and belong on a `SETTINGS.daemon` section; a script exception under the daemon shows `algan/daemon.py:1107 in execute` and two `runpy` frames above the user's own line.

## 7. Import-time process mutation

**Verified.** `algan/__init__.py:47-67`:

- `torch.inference_mode().__enter__()` is never exited. Inference mode is not merely global, it is irrevocable for tensors created under it, so a notebook that imports Algan can never train afterwards. The code comment acknowledges it; the README does not. Fix: drop the global and wrap Algan's own entry points (`save_video`, `save_frame`, materialization) in `@torch.inference_mode()`; if it stays, it goes in the README's first screen.
- The **root** logger gains a `RichHandler` at level INFO (before: `[]`, level 30; after: `[RichHandler]`, level 20). That is real Manim's `make_logger`, reached through §2; Algan's own logger is well-behaved (`logger.py:82-91`, `propagate=False`). Fix: resolve §2, or snapshot and restore root handlers around the import.
- Taichi's `[Taichi] version 1.7.4, llvm ...` banner goes to **stdout** on every import, including `algan --version` and `algan --help` (3.2–3.4 s each because `cli.py` imports `algan` to read `__version__`), and `Rendering device set to cpu` goes to stderr at INFO. A piped script gets the banner in its data. Fix: `importlib.metadata.version("algan")` for `--version`, import `algan` inside the subcommands, and demote/suppress the banner (set Taichi's log level or defer `_install_render_arch_guard()`'s bring-up to the first render).
- Fine as is: no threads, no `atexit` handlers, no `sys.path` mutation, no directories created by a bare import; the `PYTORCH_ENABLE_MPS_FALLBACK` and `CUDA_CACHE_MAXSIZE` environment writes are benign.

## 8. Public names that violate the one-name rule

`agent_guidance/api_settings.md` allows exactly four alias families. **Verified**
by `id()` comparison and `inspect.signature` over `algan.__all__` (380 names):

| What | Where | Recommendation |
| --- | --- | --- |
| Ten root names for five material classes: `UnlitMaterial`=`MeshBasicMaterial`, `PBRMaterial`=`MeshStandardMaterial`, `AdvancedPBRMaterial`=`MeshPhysicalMaterial`, `SpecularMaterial`=`MeshPhongMaterial`, `DiffuseMaterial`=`MeshLambertMaterial` | `algan/rendering/shaders/materials.py`, exported by `algan/__init__.py:151` | Keep the `Mesh*` names (`CLAUDE.md` already tells authors to) and delete the other five. |
| `mobject=` is the keyword on 16 root callables (`ApplyMatrix`, `ApplyWave`, `Blink`, `Brace`, `Circumscribe`, `Cross`, `Homotopy`, `Indicate`, `MoveAlongPath`, `PhaseFlow`, `ShowPassingFlash`, `Underline`, `Wiggle`, …); `DrawBorderThenFill`/`Group` say `mobs`; nine table classes say `element_to_mobject`; five root classes are spelled `SVGMobject`, `MobjectMatrix`, `MobjectTable`, `DashedVMobject`, `CurvesAsSubmobjects` | `algan/mobs/manim_adapters.py` and `algan/animations/manim_animations.py` | Rename the keyword to `mob` at the root (accept `mobject` only under `algan.manim`); rename the five classes (`SVGMob`, `MobMatrix`, `MobTable`, `DashedMob`, `CurvesAsChildren`). |
| `background=` on `save_video`/`save_frame`, `background_frame=` on `Scene.__init__`, `set_background()`, `SETTINGS.style.background`; `CLAUDE.md:84` and `api_settings.md:12,15` say `background_color=` (a `TypeError` today) | `algan/scene.py` | Keep `background`, annotate it, rename the constructor parameter, fix both docs. |
| `RegularPolygon(n=6, *, num_vertices=None, ...)` — two spellings in one signature | `algan/mobs/shapes_2d.py` | Keep `n`. |
| `Cone(base_radius, radius, u_min, v_range, closed, show_base, checkerboard_colors, direction=OUT)` vs `Cylinder(radius, closed, show_ends, direction=UP)` vs `Sphere(u_range, v_range)` vs `Surface(checkered_color)` | `algan/mobs/shapes_3d.py` | One vocabulary: `radius`, `u_range`/`v_range`, `closed`, `checkered_color`, one `direction` default. |
| `Dot(point=...)` vs `Point(location=...)` vs `Mob.location` | `manim_adapters.py`, `shapes_2d.py` | `location`. |
| `Group(*mobs, _link_children=True, **kwargs)` — an underscore parameter in a public signature | `algan/mobs/group.py:85` (`:208` is the only internal caller) | Make it `link_children` or move it to a classmethod. |
| `DEGREES=1.0`, `RADIANS=57.3`, `RADIANS_TO_DEGREES=RADIANS`, `DEGREES_TO_RADIANS=0.01745` — four names for two factors, and the two that look like synonyms differ by 57× | `algan/constants/math.py:34-45` | Export only `DEGREES`/`RADIANS`. |
| Primitive builders in `__all__` with `Mob`'s generic docstring: `TriangleVertices`, `TriangleTriangulated`, `QuadTriangulated`, `TriangulatedBezierCircuit`; `ManualMemory`, `AudioEffect`, `Color`, `DecimalNumber` have no docstring at all | `algan/__init__.py` | `api_settings.md:164` says primitive builders stay out of the namespace. |
| `SETTINGS.skip_save_frame` — a root attribute outside the five documented sections, read at `scene.py:1194` | `algan/settings/root_settings.py:52-64` | Underscore it or document it. |
| ~25 render-loop methods public on `Scene` (`get_batch_of_primitives`, `render_primitive_batch`, `initialize_frames`, `increment_current_time`, `get_new_id`, `terminate`, `instance`, …) | `algan/scene.py` | Underscore the engine half. |
| `algan.manim.__all__` (200 names) exports the `MANIM_*_NAMES` registry tuples and ~40 `OpenGL*` classes; conversely `manim_fov`, `manim_shader`, `ManimMaterial` sit in `algan.__all__` | `algan/manim/__init__.py`, `algan/__init__.py:113` | Trim both directions. |
| `set_environment_map` is the one module-level Scene-method wrapper, against the documented decision not to have any | `algan/__init__.py:133` | Remove it. |

Keyword and class renames are the single most expensive category to do after
release; everything in this table is a breaking change the day after `0.2.2`
is on PyPI and free the day before.

## 9. Manim-inherited signatures and docstrings

**Verified.** `inspect.signature(algan.Arc)` shows `angle: float = 1.5707963267948966`
while the appended note says the argument is in degrees:

| Call | Result |
| --- | --- |
| `Arc()` | quarter arc (the default is special-cased) |
| `Arc(angle=90)` | quarter arc |
| `Arc(angle=PI/2)` | a **1.57°** sliver, no warning |

64 root classes (`Angle`, `Axes`, `Brace`, `NumberPlane`, `Table`, `Matrix`,
`Vector`, `Variable`, …) carry docstrings with `.. manim::` blocks,
`class X(Scene)` and `self.play(...)` examples that do not run in Algan, and
Manim's types leak into displayed signatures (`Brace(mobject: 'Mobject',
direction: 'Vector3D')`, `arc_center: 'Point3DLike'`). This is what `help()` and
the Sphinx reference show.

**Fix.** For the adapted set, rewrite `__signature__` so displayed defaults are
in Algan units, *replace* rather than append to the inherited docstring, and
warn (or reject) when an angle argument is below 2π and looks like radians.

## 10. Authoring-contract gaps

All **verified** with scripts against the public surface.

- **`bool(Square())` is `False`.** `Mob.__len__` (`algan/animatable_base/mob.py:2214`) returns 0 without `parent_batch_sizes` and there is no `__bool__`, so `if mob:` skips and `mob or fallback` returns the fallback — while `Text("hi")` is truthy. Add `__bool__` returning `True`; consider a named property for the batch count.
- **Negative runtimes rewind the clock silently.** `Scene.wait(-5)` leaves the scene clock at −3, `with Seq(runtime=-1)` measures −1 s, and `save_video()` renders a 2-frame video with no warning. `Scene.wait(target - now)` coming out negative is a realistic pattern. Validate `>= 0` at the context and at `AnimationContext.wait`.
- **The `runtime=` guard only reaches methods with `**kwargs`.** `s.move(RIGHT, runtime=2)` gets the helpful "sets the timing of an animation context... wrap the call" error; `s.rotate(90, OUT, runtime=2)` and `s.scale(2, runtime=3)` get raw `TypeError: ... unexpected keyword argument`. `_reject_context_kwargs` (`animation_contexts.py:76`) is called only from the `**kwargs` path. Route it through `@animated_function`. In the same table, `animation_contexts.py:57` keys `"equialize_runtimes"` (typo) so that arm never fires.
- **Legacy names surface as internal tracebacks.** `Sync(duration=1)` → `AnimationContext.__init__() got an unexpected keyword argument`; `sq.move(RIGHT, duration=1)` dies four frames down in `set_location`, a name the user never typed. The README teaches `runtime`, so `duration`/`run_time`/`rate_func` are the natural wrong guesses; intercept them with a pointer to the new name.
- **`children`/`parents` are documented read-only but assignable.** `p.children = []` succeeds and leaves `k.parents` dangling; the parent then moves without the child. `_GuardedMethod` (`mob.py:1751`) protects exactly one method (`scale`); `mob.rotate = 90` silently shadows the method. Return tuples or guarded views, and wrap the other verb methods.

## 11. Output-path contract

**Verified.** `_resolve_output_destination` and `RenderResult.output_path`:

```
sc.save_video("nope/deeper/a.mp4")  -> output_path=PosixPath('nope/deeper/a.mp4')   # relative
sc.save_video("/abs/adir")          -> writes /abs/adir.mp4 next to the directory
sc.save_video("adir2/")             -> output_path=PosixPath('adir2.mp4'), directory dropped
echo "...save_video()" | python -   -> algan_outputs/<stdin>.mp4                    # illegal name on Windows
sc.save_video("weird.xyz")          -> renders everything, then FileNotFoundError on the temp rename
```

The CLI's `-o` (`algan/cli.py:190`) *does* treat a trailing slash or existing
directory as a directory, so the two halves of the product disagree.
`output_filename_for` (`algan/settings/path_settings.py:59`) special-cases only
`script is None`, not `<stdin>`/`<string>`/`<ipython-input-…>`. The container
extension is never validated up front, although `check_codec_is_available`
(`algan/utils/algan_utils.py:359`) exists precisely so an unusable codec does not
"cost a whole render and then surface as a missing temporary file".

**Fix.** Always return an absolute `output_path`; apply the CLI's directory rule
in the resolver; treat a non-file `__main__.__file__` as `None`; validate the
suffix beside the codec check with the same message shape as the (excellent)
transparent-MP4 error. Also: nothing prints where the file landed
(`Finished rendering quickstart.mp4 in 14.0 s`); print the absolute path.

## 12. Packaging

**Verified** by building with `uv build`, installing the wheel into a clean
Python 3.11 venv and running it from outside the repo; `twine check` passes.

- **Undeclared runtime dependencies.** `pillow` is imported at module scope by first-party code (`algan/rendering/post_processing/anti_aliasing/smaa.py:9`, `algan/mobs/image_compat.py:9`); `pygltflib` (`algan/mobs/three_d_models/gltf_loader.py:350`) is declared nowhere, so glTF animation silently degrades to static on every install; `typing_extensions` (`algan/utils/docbuild/module_parsing.py:11`). All currently arrive only through Manim's graph and vanish with §2(a).
- **Over-declared.** `click` is imported by nothing. `cloup`, `decorator`, `pygments`, `isosurfaces`, `skia-pathops`, `mapbox-earcut`, `beautifulsoup4`, `dendroid`, `prioq`, `reprit`, `decision` serve only the vendored trees. `torchvision` exists for two `read_image` calls (`algan/utils/file_utils.py:50`, `model_mob.py:119`) and pins users to a matching torch; PIL is already loaded. `opencv-python>=3.0.0` resolves to `5.0.0.93`, two majors past anything tested, is the GUI build (X11 libs on headless boxes), and serves one `cv2.imread` (`scene.py:540`).
- **`py.typed` is not backed by annotations.** 13% of functions have a return annotation, 26% of parameters. Shipping the marker makes type checkers trust the package and see `Any` everywhere; removing it later breaks downstream builds. Drop it until `algan.__all__` is annotated.
- **Wheel contents.** 0.72 MB (32%) of the wheel is `DESIGN_*.md` working notes under `algan/rendering/`; `algan/utils/docbuild/` (imports `sphinx`) and `algan/utils/testing/` (imports `pytest`, `matplotlib`) ship with no dependency to satisfy them, and `algan_directive.py:316` resolves `site-packages/docs/rendering_times.csv`. `[tool.hatch.build] exclude` of `/docker`, `/logo`, `/scripts` is dead configuration because `include = ["/algan"]` already excludes them. The sdist has no `tests/` or `taichi_patches/`, so a downstream packager cannot test it.
- **Python support claims.** Every first-party file parses at `feature_version=(3,9)`, `vermin` says 3.9, no 3.10+ runtime constructs, and `uv pip compile` resolves on 3.9 and 3.13 (taichi 1.7.4 has cp39–cp313 wheels). CI runs 3.9 and 3.13 on Linux, so the classifiers are backed. Note that 3.9 resolves a materially different stack (`manim==0.19.0`, `numpy==2.0.2`, `scipy==1.13.1`).
- The installed footprint is ~5.3 GB (nvidia 2.7 GB, torch 1.1 GB, triton 0.7 GB); worth one README line.

## 13. Repository weight

**Verified.** `.git` is 210 MB; 107 media blobs total 149 MiB of 232 MiB of
blobs in history, almost all `tests/full_renders/expected_outputs_{cpu,cuda}/*.mp4`
at ~4 MB each, re-committed on every rebaseline. The working tree's `tests/`
is 38 MB. This grows monotonically and a history rewrite becomes impractical
once the repo is widely cloned. Decide now: Git LFS for `expected_outputs_*`,
or baselines fetched from a release asset, and whether to rewrite history
before the public announcement.

## 14. Missing-tool errors

**Verified** by running with an empty `PATH` inside a subprocess.

- `Tex(r"x^2")` without a TeX distribution: a `rich`-formatted INFO line from vendored Manim, then a raw `FileNotFoundError: 'latex'`. The average PyPI installer has no TeX, and `Text` works without it. Raise `AlganConfigurationError` naming the binary, the install command per platform, and the `Text` alternative.
- `algan check` warns `FFmpeg not found on PATH. Video export may fail` while export succeeds through imageio-ffmpeg's bundled binary (a full render with an empty `PATH` produced a valid mp4). With ffmpeg present it reports `/usr/bin/ffmpeg`, which is not the binary used. Report `imageio_ffmpeg.get_ffmpeg_exe()` and `SETTINGS.paths.ffmpeg_binary`.
- `pydub` emits `RuntimeWarning: Couldn't find ffmpeg or avconv` on `import algan` whenever ffmpeg is off `PATH`, although everything works. Point pydub at the bundled binary or filter the warning.

## 15. Metadata and hygiene

- `recipe/meta.yaml`: version `0.0.2`, `python >=3.10,<3.13` (contradicts `requires-python`), a `pytorch-scatter` dependency that no longer exists, `license_file: LICENSE.md` (file is `LICENSE`), `noarch: python` for a package that needs Taichi and torch, and a literal `<YOUR_SHA256_HASH_HERE>`. Finish it or delete it.
- No `CHANGELOG` and no version policy anywhere; `stable` is 133 commits behind `master` at `0.0.63`, the only tag is `BETA_v0.0.63`, and there are no GitHub releases. Decide the release branch flow (`test.yaml` assumes master→stable PRs) and tag `v0.2.2` at what you upload.
- README: the Discord badge uses server id `1122334455`, a placeholder inherited from Manim's README (real guild ids are 17–19 digit snowflakes); the PyPI badges 404 until upload. `docs/source/_static/algan-logo-sidebar*.svg` are blank placeholders by their own comment.
- Stale internal docs: `agent_guidance/api_settings.md:151` still says "`duration` rather than `run_time`" (now `runtime`); `:104` still lists `ambient_light`/`light_intensity`, which are gone (so `RENDERER_WORK_QUEUE.md` item 8 is done); `CLAUDE.md:104` cites `VALIDATE_render_device_on_cuda.md`, which does not exist; `CLAUDE.md:84` documents `background_color=`.
- `algan/external_libraries/readme.txt` gives Pango build tooling as the reason for vendoring Manim while `pyproject.toml` depends on Manim anyway (§2).
- Environment surface: 236 declared `ALGAN_*` names become public on release; `warn_for_unknown_algan_environment_variables` warns on *any* `ALGAN_`-prefixed variable, so an unrelated tool sharing the prefix warns on every import. Say in the docs that only `_LIVE_VARIABLES` plus the startup tuple are supported, and scope the warning to near-misses of declared names.
- `LICENSE` year is 2025; `conf.py` already uses a `2025-{current}` range.

## 16. Smaller items

Correctness and UX, all **verified**, none contract-breaking to fix later:

- `mob.glow` / `mob.opacity` do not read back a glow or opacity set through `color` (`GREEN.set_glow(0.5)` → `mob.glow` is 0.0); rendering is right, and the two spellings *compose*. One sentence in the tutorial's colour note.
- `Square(location=(1, 2))` → `RuntimeError: The size of tensor a (3) must match...`, while `mob.move(1)` gets the model error message. Run `cast_to_direction` on constructor kwargs; add a constructor-kwarg hint table beside `_MANIM_METHOD_HINTS` (`mob.py:137`) for `side_length` and friends.
- `remove_updater(9999)` → `IndexError`; `become(None)` → `AttributeError` on `.scene`; `save_frame(at=-1)` reports `got -0.8` (rescaled before validation).
- `mob.opacity = 5.0`, `mob.scale(0)`, `mob.scale(-1)` and a NaN location are accepted silently; a NaN guard in `_apply_change` turns a black frame into a named error at the authoring line.
- `algan/constants/spatial.py` docstring says constants are `(1, 1, 3)`; they are `(3,)` and `DEFAULT_BASIS` is `(3, 3)`.
- `algan render nosave.py` exits 0 in silence when the script never renders; `algan check` prints no paths or version despite its help text; the default `SETTINGS.video` (864×486 @ 15, ssaa 2) matches no named preset.
- 121 bare `raise ValueError`, ~30 in user-facing constructors (`surface.py:1085-1091`, `text.py:1033,1102,1165`). `AlganConfigurationError` subclasses `ValueError`, so converting is non-breaking; do it before `except AlganError` becomes a contract users rely on.
- Magic scalars where Algan has types: `Material(side=0)`, `AdvancedPBRMaterial(specular_color=16777215, sheen_color=0, ...)`, `Text(slant='NORMAL', weight='NORMAL')`.
- Viewer: binds `127.0.0.1` only, path-safe static serving, no CORS; but no `Host`-header check and `POST /api/shutdown` is unauthenticated, so a DNS-rebound or CSRF page can stop a running viewer. A per-session token in the URL closes both.

## 17. Tests and CI

**Verified** locally and against the Actions history of `algorithmicsimplicity/algan`.

- `ruff check .` and `ruff format --check .` are clean (313 files). The three per-push workflows (Code Quality, Docs, Test on ubuntu 3.9/3.10/3.13 + macOS 3.10) are green at `3c03536`.
- `tests/unit_tests`: `1 failed, 2641 passed, 139 skipped in 11:06`. The one failure (`test_daemon_run_context.py::test_stdin_is_isolated_from_the_trigger`) captured this sandbox's own shell-startup text on the subprocess's stdin and is green in Actions; not a repo bug.
- The four pushes before HEAD were **red on every matrix leg** with rename fallout (`border_texture_points`, `Model3D.play_animation(runtime=)`, `DEFAULT_DURATION` in the `__all__` snapshot, `Primitive.runtime`) and were fixed by two "Fixed failing tests" commits. Cut the release from a commit that has its own green run, and tag that commit.
- **What CI cannot see.** No matrix leg has a GPU, so the 8 CUDA-only tests (`test_wide_attr_device.py`, `test_scene_arena_upload.py`) never execute anywhere automatically — the class of invisibility that `test_wide_attr_device.py`'s own docstring records causing a staleness bug on 2026-08-28. `tests/full_renders` and `tests/path_traced` are not even collected in CI (`test.yaml` names `tests/unit_tests tests/fast`), so six dense pixel-compared scenes and the path tracer are guarded only by whoever runs the full suite by hand; `tests/path_traced` has no CUDA baseline set yet. A dispatch-only GPU job (there is precedent in `taichi_build.yaml` and `mps_probe.yaml`) would close the first gap before the first release, not after.
- Skips worth a decision: 122 doc-render cases behind `ALGAN_RUN_DOC_RENDERS=1`; two `test_raytracing_unit.py` cases for a megakernel removed in `ceaf3c4` (delete them); one `pygltflib` skip that is really §12's undeclared dependency.
- `tests/README.md` says the fast suite "is now 191"; it collects 464 cases (74 `fast` decorators with parametrization). The cold run here reported 96 s against the 75 s budget; per `CLAUDE.md` that reading is junk before the third consecutive run.

---

## What was checked and found sound

So the next audit does not redo it: transforms (`move` relative, `move_to`
absolute, `about=` arcs, degrees and `degrees=False`, `Sync` composition);
attribute assignment and mid-`Sync` read-back; Scene isolation (nested `with
Scene()`, cross-Scene animation, three back-to-back renders with no
contamination); the `reset=False` contract; `save_frame` not mutating the
timeline; byte-identical re-renders; `video_settings` overrides not leaking into
`SETTINGS`; `Group` indexing/iteration/slicing; `Text` glyph indexing and `Lag`
over glyphs; `arrange_in_line`/`arrange_in_grid`; `become` across 2-D/3-D/Text/
Group; all built-in indication and movement animations; no mutable-literal
defaults anywhere under `algan/`; hierarchy cycle and cross-Scene guards; the
lifecycle warnings (`DespawnedMobWarning`, `NeverSpawnedMobWarning`,
`EmptySceneWarning`); the settings error messages (unknown field with
suggestion, stable section identity, CUDA requested on a CPU box, experimental
switch routing, preset immutability, `snapshot`/`restore`); transparent-MP4 and
bad-codec errors; `overwrite=False`; TTY-aware progress bars; CLI exit codes
and help text; `algan new` scaffold renders; `Scene.view()` serves on an
ephemeral localhost port and stops cleanly; the structural Sphinx build; no
secrets or tokens in the tree; the 3.9 syntax floor.
