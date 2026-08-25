# Follow-up: `auto` must reach an NVENC-capable ffmpeg binary

Read `/content/algan/CLAUDE.md` first. Set `ALGAN_USE_DAEMON=0` for every script.
This continues your NVENC task (`scratch_perf/ox/brief_nvenc.md`, your report
`scratch_perf/ox/REPORT_nvenc.md`). Your implementation is in
`algan/utils/video_encoding.py`, wired in `algan/utils/algan_utils.py`.

## The gap
Your end-to-end check showed `auto` falling back to software because moviepy's
bundled ffmpeg (imageio-ffmpeg's static build) has no NVENC encoder. That leaves
the feature inert on exactly this machine, where `/usr/bin/ffmpeg` (`ffmpeg
-hide_banner -encoders | grep nvenc`) does have `h264_nvenc` and the T4 has an
NVENC engine. The selector has to choose the **binary** as well as the codec.

## Required behaviour
1. Candidate binaries, probed in order, first usable wins: the `FFMPEG_BINARY`
   environment variable if set and not "auto"; moviepy's configured binary; then
   `shutil.which("ffmpeg")`. "Usable" = lists `h264_nvenc` in `-encoders` AND the
   test encode succeeds (use a 256x256 clip; you found NVENC rejects 64x64).
2. When a candidate other than moviepy's own wins, the writer must run **that**
   binary. `moviepy.video.io.ffmpeg_writer.FFMPEG_VideoWriter` reads
   `FFMPEG_BINARY` from `moviepy.config` at call time -- check the installed
   moviepy (2.x) for the exact attribute and whether it can be overridden per
   writer; if the only route is `moviepy.config.FFMPEG_BINARY = path`, set it
   for the duration of the write (a context manager around `get_file_writer` in
   `algan/utils/algan_utils.py`, restored afterwards) so unrelated moviepy use is
   untouched. Say in a comment which route you took and why.
3. `select_video_encoder` returns the binary alongside the pair (or the writer
   helper takes it); keep the existing tests passing and add cases: a
   moviepy-binary-without-nvenc + system-binary-with-nvenc combination picks the
   system binary; no candidate usable -> software with moviepy's binary.
4. The probe stays once-per-process and its failures stay at DEBUG.

## Verification (report outputs verbatim in `scratch_perf/ox/REPORT_nvenc_binary.md`)
- `uv run -m pytest -q tests/unit_tests/test_video_encoding.py tests/unit_tests/test_environment.py`
- Re-run your `scratch_perf/ox/render_hd_encode_check.py` with
  `ALGAN_VIDEO_ENCODER=auto` and show the log line naming `h264_nvenc`, the
  "Finished rendering ... in N s" line, and `ffprobe -show_streams` of the output
  (`codec_name=h264`, correct `nb_frames`). Also report the software arm's time
  from your previous report for comparison.
- `uv run ruff check --no-fix` and `uv run ruff format --check` on touched files.
Do not commit.
