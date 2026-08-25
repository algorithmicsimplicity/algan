# Report: hardware (NVENC) video encoding for `save_video`

Task: `scratch_perf/ox/brief_nvenc.md`. All changes left uncommitted in the working tree.

## What changed

- **New `algan/utils/video_encoding.py`** — `select_video_encoder(codec, ffmpeg_params, transparent)` returns the `(codec, ffmpeg_params)` pair the writer gets:
  - Explicit `codec` wins verbatim; transparent output untouched.
  - Caller `ffmpeg_params` containing `-preset`/`-crf` stay on `libx264` with their params verbatim (those flags are x264 rate control; under NVENC they fail or mean something else). Comment in code says why.
  - Otherwise `ALGAN_VIDEO_ENCODER` decides: `software` = today's exact pair (`libx264`, `-crf 17 -preset slower`); `nvenc` forces `h264_nvenc` with `-preset p4 -tune hq -rc vbr -cq 19 -b:v 0 -pix_fmt yuv420p`; `auto` (default) uses NVENC when *usable*, else software. An unusable value warns (`AlganWarning`) and behaves as `auto`.
  - "Usable" = the binary `moviepy.config.FFMPEG_BINARY` resolves to lists `h264_nvenc` in `-encoders` **and** a tiny lavfi test encode exits 0; probed once per process, cached, failures logged at DEBUG.
  - **One deliberate deviation from the brief's literal probe command:** `color=black:s=64x64` fails on this box with `InitializeEncoder failed: invalid param (8): Frame Dimension less than the minimum supported value` — NVENC refuses to open any encoder below its minimum frame dimension, so a 64x64 canary would make `auto` fall back everywhere, defeating the feature. The probe uses `s=256x256` (verified OK here at 256x256 and 1920x1080). Everything else about the probe is exactly as briefed.
  - moviepy-writer interaction checked against the installed moviepy 2.1.2 source: it inserts its own `-preset medium` *before* our params (ours win, FFmpeg takes the last spelling, silently) and appends a trailing `-pix_fmt yuva420p` after them for even-sized frames; verified empirically that FFmpeg then warns `Incompatible pixel format 'yuva420p' for codec 'h264_nvenc', auto-selecting format 'yuv420p'` and encodes fine (exit 0). Params unchanged from the brief.
- **`algan/environment.py`** — `"ALGAN_VIDEO_ENCODER"` declared in `_LIVE_VARIABLES` (alphabetical), read via `env_str` at the point of use inside `select_video_encoder`.
- **`algan/utils/algan_utils.py`** (`_render_scene_to_file`) — selection runs where codec/params defaults used to be chosen, *before* the defaults are filled so caller-supplied values are distinguishable; the default fills remain for the branches selection passes through; logs `Encoding video with {codec}` at INFO once per render.
- **Docs** — new "Choosing the video encoder" section in `docs/source/advanced_user_tutorials/saving_videos_and_images.rst`; `codec/audio_codec/ffmpeg_params` entry of the `save_video` docstring in `algan/scene.py` extended by three sentences pointing at `ALGAN_VIDEO_ENCODER`.

## Verification (required items, outputs verbatim)

### Env-var rule + unit tests
```
$ uv run -m pytest -q tests/unit_tests/test_environment.py tests/unit_tests/test_video_encoding.py
...........................                                              [100%]
27 passed in 20.00s
```
New `tests/unit_tests/test_video_encoding.py` covers every case the brief lists (auto/unusable-probe → today's exact software pair; software never probes; nvenc forces without probing; explicit codec passes through; probe cached across calls — plus x264-flag params, transparent passthrough, bad-value warning). Not marked `fast`.

### End-to-end renders (`scratch_perf/ox/render_hd_encode_check.py`, copy of `nn_scene_UHD.py`'s scene at HD, run_time 0.5, direct `Scene.save_video`, no profiler, one process per arm)

| arm | log line | wall time |
|---|---|---|
| `ALGAN_VIDEO_ENCODER=software` | `Finished rendering nn_HD_software.mp4 in 89.0 s` | 89.0 s |
| `ALGAN_VIDEO_ENCODER=auto` | `Finished rendering nn_HD_auto.mp4 in 77.5 s` | 77.5 s |
| extra: `FFMPEG_BINARY=/usr/bin/ffmpeg ALGAN_VIDEO_ENCODER=auto` | `Encoding video with h264_nvenc` / `Finished rendering nn_HD_auto_sysffmpeg.mp4 in 76.8 s` | 76.8 s |

**The important finding:** on this box `auto` resolved to `libx264` for both required arms, because moviepy does **not** use the system ffmpeg. `moviepy.config.FFMPEG_BINARY` resolves to imageio-ffmpeg's static build
(`.venv/.../imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2`), which contains **zero** NVENC encoders (`-encoders | grep -ci nvenc` → `0`), while `/usr/bin/ffmpeg` has `h264_nvenc`/`hevc_nvenc` and the T4 driver (580.82.07) drives it. Per the brief the probe checks the binary moviepy will actually run, so the fallback is correct behaviour — but it means the two required arms are both software-encoded and their wall times differ only by run noise. The third arm proves the whole path works: repointing moviepy via its own `FFMPEG_BINARY` env var makes `auto` pick `h264_nvenc`, the render completes, and `ALGAN_LOG_LEVEL=DEBUG` shows `NVENC probe: h264_nvenc is usable via /usr/bin/ffmpeg`. At HD the encode tail is small here, so the three times are within noise of each other; the motivating 14.3 s drain was measured at UHD, which I did not re-run (brief specified HD).

### Playability (`ffprobe`)
`ffprobe -v error` parses all three outputs cleanly; full decode (`ffmpeg -v error -i … -f null -`) reports zero errors on all three. `ffprobe -show_streams` of the **auto** output (required):
```
codec_name=h264
codec_tag_string=avc1
width=1920
height=1080
pix_fmt=yuv420p
r_frame_rate=30/1
avg_frame_rate=30/1
duration=0.500000
nb_frames=15
```
Frame count 15 = run_time 0.5 s × HD's 30 fps — correct. The nvenc-arm file reports identically except `profile=Main` (nvenc's default) instead of `High`; also parses and decodes clean.

### Fast suite
```
$ uv run -m pytest -q --fast
fast suite: 66s of its 75s budget (88%)
FAILED tests/fast/test_fast_render.py::test_the_fast_scene_renders_and_matches_its_baseline
1 failed, 275 passed, 1913 deselected in 65.88s (0:01:05)
```
**This failure is pre-existing and not caused by this task.** Evidence: the test passes an explicit `codec="libx264rgb"`, which `select_video_encoder` returns verbatim, so this change cannot alter its pixels; and with my changes entirely removed from the tree (stash round-trip) the same test fails identically — `fast.mp4 differs from its baseline by up to 41 channel values (worst at frame 24)`. The committed baselines don't match current HEAD output; whoever landed `c0c3669` ("Materialize wide attributes…", which touched `surface.py`/`triangle_primitive.py`/`scene_builder.py`/`render_loop.py`) needs to re-baseline or investigate — not this task. First `--fast` run this session printed `100s of its 75s budget (133%)`; per CLAUDE.md the self-reported time is junk until the third consecutive run.

### Lint
`uv run ruff check --no-fix` and `uv run ruff format --check`: clean on all five touched files (`video_encoding.py`, `algan_utils.py`, `environment.py`, `test_video_encoding.py`, `scene.py`). One caveat: `algan/scene.py` carries a pre-existing `I001` (un-sorted import block) — reproduced byte-identically on HEAD's version of the file; my edit touches only its docstring, and I did not reformat unrelated imports.

## Not verified / caveats

- No UHD render (the brief's motivation numbers come from UHD; it asked for HD verification only).
- `hevc_nvenc` untested; only `h264_nvenc` is selected by design.
- The NVENC arm ran via `FFMPEG_BINARY=/usr/bin/ffmpeg` because the venv's moviepy-bundled ffmpeg cannot emit NVENC at all. If NVENC-by-default matters on machines like this one, the durable fix is installing a full ffmpeg and setting `FFMPEG_BINARY` (moviepy honours it) — outside this brief.
- Transparent-output encoding under NVENC untested (transparent paths are excluded from selection by design).
- Daemon interaction: `ALGAN_VIDEO_ENCODER` is declared LIVE so it can flip between renders in one process; the third arm additionally relied on `FFMPEG_BINARY`, which is moviepy's variable, not an Algan-declared one — it must be set before moviepy first resolves its config.
- Housekeeping notes: a concurrent commit (`c0c3669`) landed in this repo mid-session; a stash/pop I used for the pre-existing-failure check interacted with it cleanly and restored every file (verified via `git status` before/after). Untracked leftovers `algan_profile_report_nn_PREVIEW.txt` / `_UHD.txt` predate this session and were left alone.
