# Task: hardware (NVENC) video encoding for Algan's save_video, with automatic selection

Read `/content/algan/CLAUDE.md` first (repo rules: `uv run python`, never call `ti.init`,
every `ALGAN_*` env var must be declared in `algan/environment.py` and read via its
accessors, `ruff check --no-fix`, `*_taichi.py` conventions). Set `ALGAN_USE_DAEMON=0`
for every script you run. Work only in the files named below plus new test files.

## Why
Profiling `benchmarks/performance/nn_scene_UHD.py` on this box (Tesla T4, 2 vCPUs) shows
`video encode tail (ffmpeg drain)` = 14.3 s of a 50 s render: the default encoder is
`libx264` with `["-crf", "17", "-preset", "slower"]` (see `render_to_file` in
`algan/utils/algan_utils.py`, ~line 335), which at 3840x2160 cannot keep up with the
renderer on 2 CPU cores, so the bounded frame queue in `RenderLoopMixin.render_to_video`
(`algan/render_loop.py`) stalls the render and the drain waits at the end. This ffmpeg
has `h264_nvenc` / `hevc_nvenc` (`ffmpeg -hide_banner -encoders | grep nvenc`), and the
T4 has an NVENC engine, so encoding can leave the CPU entirely.

## What to build
1. New module `algan/utils/video_encoding.py` with `select_video_encoder(codec, ffmpeg_params, transparent)`
   returning `(codec, ffmpeg_params)`:
   - If the caller passed an explicit `codec`, return it and its params unchanged (explicit wins).
   - Transparent output (`png` codec / `.mov`) is untouched.
   - Otherwise consult env var `ALGAN_VIDEO_ENCODER` (declare it in `algan/environment.py` in
     the LIVE variables tuple; read it with the accessor at the point of use). Values:
     `auto` (default), `nvenc`, `software`. `software` = today's exact behaviour
     (`libx264`, `-crf 17 -preset slower`). `nvenc` = force `h264_nvenc`. `auto` = `h264_nvenc`
     when it is *usable*, else software.
   - "Usable" means: the ffmpeg binary moviepy uses (`moviepy.config.FFMPEG_BINARY`, check the
     installed moviepy's actual attribute) lists `h264_nvenc` in `-encoders` AND a tiny test
     encode succeeds (`ffmpeg -hide_banner -loglevel error -f lavfi -i color=black:s=64x64:d=0.1
     -c:v h264_nvenc -f null -`), because the encoder can be compiled in yet fail without a
     driver. Probe once per process and cache; a failed probe logs at DEBUG and falls back.
   - NVENC params: `["-preset", "p4", "-tune", "hq", "-rc", "vbr", "-cq", "19", "-b:v", "0",
     "-pix_fmt", "yuv420p"]`. Check how moviepy's `FFMPEG_VideoWriter` (in the venv) builds its
     command so pixel format / `-r` / `-s` are not duplicated or contradicted; adjust if needed.
     If the user passed `ffmpeg_params` but no codec, keep their params and only pick the codec
     if the params contain no `-preset`/`-crf` (those are x264-specific); otherwise fall back to
     software. Say in a comment why.
2. Wire it into `render_to_file` in `algan/utils/algan_utils.py` where `codec`/`ffmpeg_params`
   defaults are chosen. Also log the chosen encoder once per render at INFO
   (`logger.info(...)` as the surrounding code does).
3. Document it: a short paragraph in `docs/source/` wherever `save_video`'s codec/ffmpeg_params
   are described (grep for `ffmpeg_params`), and the `save_video` docstring in
   `algan/scene.py` (follow `DOCSTRINGS.md`; keep the edit minimal).

## Verification (all required, report the outputs verbatim)
- `uv run -m pytest -q tests/unit_tests/test_environment.py` (the env-var rule test) passes.
- New `tests/unit_tests/test_video_encoding.py`: with the probe monkeypatched to "unusable",
  `auto` returns the software pair exactly as today; `software` never probes; `nvenc` forces
  `h264_nvenc`; an explicit codec passes through; the probe result is cached (probe called once).
  Do NOT mark tests `fast`.
- End to end on this box: render `benchmarks/performance/nn_scene_UHD.py`'s scene once at `HD`
  (use a copy of that script in `scratch_perf/ox/`, run_time 0.5, `Scene.save_video` directly,
  no profiler) with `ALGAN_VIDEO_ENCODER=software` and once with `auto`; report both wall times
  from the "Finished rendering ... in N s" log line, and `ffprobe -show_streams` of the auto
  output showing `codec_name=h264`. Play-ability is the acceptance criterion: `ffprobe` must
  parse it and report the right frame count.
- `uv run -m pytest -q --fast` passes (report the timing line it prints; it takes 1-3 minutes).
- `uv run ruff check --no-fix` and `uv run ruff format --check` clean on every file you touched.

Write your report to `scratch_perf/ox/REPORT_nvenc.md`: what you changed, the numbers above,
and anything you did NOT verify. Do not commit; leave the changes in the working tree.
