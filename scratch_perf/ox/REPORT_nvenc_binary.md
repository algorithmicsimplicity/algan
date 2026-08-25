# Report: `auto` must reach an NVENC-capable ffmpeg binary

Task: `scratch_perf/ox/brief_nvenc_binary.md` (follow-up to `brief_nvenc.md`). All changes left uncommitted in the working tree.

## What changed

- **`algan/utils/video_encoding.py`** — the probe now selects a **binary**, not just a bool:
  - New `_candidate_binaries()`: the `FFMPEG_BINARY` environment variable if set and not `"auto"`, then moviepy's configured binary (`moviepy.config.FFMPEG_BINARY`, read live), then `shutil.which("ffmpeg")`; deduplicated preserving order (the env var usually *is* moviepy's configuration).
  - `_probe_once()` walks candidates in order, first usable wins ("usable" = lists `h264_nvenc` in `-encoders` **and** the 256x256 test encode exits 0 — same two-step check as before, now per candidate). One cache (`_probe_cache`) holds the `(usable, binary)` pair; still at most one probe per process; every rejection logs at DEBUG.
  - New `resolve_encode_binary(codec) -> str | None`: the binary the writer must run, or `None` for "leave moviepy on its configured binary". Only an NVENC codec can move the binary. `software` mode pins today's behaviour including the binary (`None`). In forced `nvenc` mode it probes once to *serve* the forced demand (a codec forced onto a binary that lacks it can only fail); if no candidate is usable it returns `None`, so the writer keeps moviepy's own binary and fails loudly at encode time exactly as before binaries were selectable — forcing never silently falls back to software.
  - New context manager `override_moviepy_ffmpeg_binary(binary)`. **Route taken:** moviepy 2.1.2 has no per-writer binary argument — `FFMPEG_VideoWriter.__init__` builds its command from the module global `moviepy.video.io.ffmpeg_writer.FFMPEG_BINARY`, which that module bound from `moviepy.config` **at import time** (`from moviepy.config import FFMPEG_BINARY`). Setting only `moviepy.config.FFMPEG_BINARY` would therefore not affect an already-imported writer module, so the manager sets **both** attributes for the duration of the block and restores them afterwards. This window is sufficient because moviepy spawns its ffmpeg subprocess at writer construction (verified in the installed source: `close()` only drains stdin/stderr and waits; audio muxing happens inside that same single invocation), so frame writes and close talk to the already-running process while unrelated moviepy use sees its own configuration again immediately after construction returns. `None` is a no-op.
- **`algan/utils/algan_utils.py`** (`_render_scene_to_file`) — after `select_video_encoder`, calls `resolve_encode_binary(codec)` and wraps the single `get_file_writer(...)` call in `with override_moviepy_ffmpeg_binary(encode_binary):`.
- **`tests/unit_tests/test_video_encoding.py`** — all eight original tests keep their assertions verbatim (the probe stub re-points from `_nvenc_usable` to the uncached seam `_probe_once`, which now returns `(usable, binary)`). Nine cases added: moviepy-without-NVENC + system-with-NVENC picks the system binary; nothing usable → software with moviepy's binary (and one probe serves all queries); first usable candidate wins; candidate order env → moviepy → PATH with dedupe; the real usability check skipping an unlaunchable binary, one without the encoder listed, and one whose test encode fails; forced nvenc using the probed binary when one exists / keeping moviepy's when none does; the override context manager setting and restoring both attributes, and doing nothing for `None`. Not marked `fast`.
- **Docs** — the "Choosing the video encoder" section in `docs/source/advanced_user_tutorials/saving_videos_and_images.rst` now describes the binary search and why it matters (moviepy's static build vs system ffmpeg). The `save_video` docstring needed no change ("when the machine's NVIDIA driver exposes NVENC" remains true).

## Verification (required items, outputs verbatim)

### Unit tests
```
$ uv run -m pytest -q tests/unit_tests/test_video_encoding.py tests/unit_tests/test_environment.py
....................................                                     [100%]
36 passed in 12.32s
```
(27 tests before this task; 36 after.)

### End-to-end (`scratch_perf/ox/render_hd_encode_check.py`, `ALGAN_VIDEO_ENCODER=auto`, one process, daemon off)
The stale software-encoded `nn_HD_auto.mp4` from the previous session was moved aside (`nn_HD_auto_prev_libx264.mp4`) so `save_video` would actually render. Log lines verbatim:
```
Encoding video with h264_nvenc
Finished rendering nn_HD_auto.mp4 in 84.9 s
```
The probe's DEBUG walk, captured separately with `ALGAN_LOG_LEVEL=DEBUG` (same process-lifetime probe as the render uses):
```
NVENC probe: /content/algan/.venv/lib/python3.13/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2 does not list h264_nvenc in -encoders
NVENC probe: h264_nvenc is usable via /usr/bin/ffmpeg
selected: ('h264_nvenc', ['-preset', 'p4', '-tune', 'hq', '-rc', 'vbr', '-cq', '19', '-b:v', '0', '-pix_fmt', 'yuv420p'])
binary : /usr/bin/ffmpeg
```

### Playability (`ffprobe -show_streams`, fields of interest)
```
codec_name=h264
profile=Main
width=1920
height=1080
pix_fmt=yuv420p
r_frame_rate=30/1
avg_frame_rate=30/1
duration=0.500000
nb_frames=15
```
Frame count 15 = run_time 0.5 s x HD's 30 fps — correct. `profile=Main` is nvenc's default, matching the nvenc arm of the previous report — which is itself the proof the writer ran `/usr/bin/ffmpeg`: moviepy's imageio build has zero NVENC encoders, so this stream cannot have come from it. A full decode (`ffmpeg -v error -i … -f null -`) reports zero errors.

### Software-arm comparison
From `REPORT_nvenc.md`: `ALGAN_VIDEO_ENCODER=software` rendered the same scene in **89.0 s**; today's auto/NVENC arm took **84.9 s**. At HD the encode tail is small (the motivating 14.3 s drain was measured at UHD), so the difference stays within run-to-run noise on this box — the point of this task was that `auto` reaches NVENC here at all without hand-setting `FFMPEG_BINARY`, not a wall-clock win at HD.

### Lint
```
$ uv run ruff check --no-fix algan/utils/video_encoding.py algan/utils/algan_utils.py tests/unit_tests/test_video_encoding.py
All checks passed!
$ uv run ruff format --check algan/utils/video_encoding.py algan/utils/algan_utils.py tests/unit_tests/test_video_encoding.py
3 files already formatted
```
(7 lint findings in my new test code — D209/C420/SIM300 — fixed before the final run shown above.)

## Not verified / caveats

- No UHD render (unchanged scope from the previous brief: HD only).
- `hevc_nvenc` untested; only `h264_nvenc` is ever selected, by design.
- Transparent-output encoding under NVENC untested (transparent paths are excluded from selection by design).
- On machines where moviepy's own binary already carries NVENC, `resolve_encode_binary` returns that path and the override sets it around construction anyway — semantically identical to leaving moviepy alone, chosen over a special case for uniformity.
- `FFMPEG_BINARY` remains moviepy's variable, not an Algan-declared one: it is read with plain `os.environ` inside `_candidate_binaries` (the env-var rule in `tests/unit_tests/test_environment.py` covers `ALGAN_*` names only), and candidates are resolved at probe time, not import time.
