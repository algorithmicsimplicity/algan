# Algan video skill

A video-authoring skill for an AI agent implementing a user's supplied vision.
It is not a creative-direction guide or a guide to maintaining Algan.

## Use

Keep this entire `algan-video` directory together. Give a skill-capable agent the
directory through its supported skill mechanism, or instruct a file-capable
agent to read `SKILL.md` and open the referenced files as needed. The host must
also provide Python execution and an Algan environment to produce renders.
A Markdown attachment alone does not install dependencies or grant execution.

`algan-video.zip` and `algan-video.skill` contain the same directory and files;
`.skill` is a ZIP archive with a different extension. Whether that extension
can be imported directly depends on the host. `SKILL.md` follows the Agent
Skills directory/frontmatter format.

## Contents

The core instructions route to ten focused references. Seven Python examples
and a runnable multi-scene project template (`examples/project_template/`) show
mechanics without prescribing a visual style. Three utility scripts check the
environment, check video metadata/decodability, and build review contact sheets.
`evals/evals.json` contains behavioral evaluation cases for a host that supports
skill testing.

Run example scripts from a writable project directory with the intended Python
interpreter. Single-scene examples create `renders/` outputs at explicit PREVIEW quality.
The Project example takes its quality from the project CLI and manages separate
scene outputs before concatenation. None is the user's finished design. `text_and_texture.py` additionally needs a
working text/LaTeX environment. Read `VALIDATION.md` for what was actually tested.

Source baseline: Algan master commit
`f9e6d73c12de35f7e14315c2d49df34de570c644`, inspected September 10, 2026.
The installed API takes precedence over this snapshot. No Algan source changes
are part of this package.

## Utility commands

Run the preflight with the same Python that will render:

```bash
python scripts/check_environment.py --json
python scripts/check_environment.py --require-pango --require-latex --require-ffprobe
```

Check an actual exported video, replacing the specifications with the requested
ones. `--decode` checks decodability, not appearance or spoken content:

```bash
python scripts/verify_video.py renders/clip.mp4 --width 1920 --height 1080 --fps 30 --duration 2.5 --decode
python scripts/verify_video.py renders/narrated.mp4 --require-audio --decode
```

For a `Project`, `render_screenshots(contact_sheet=True)` (or
`--render-screenshots --contact-sheet`) writes the storyboard sheet itself. This
script builds review sheets from any stills, or from frames sampled out of a
rendered clip (Pillow required; video mode also needs FFmpeg):

```bash
python scripts/contact_sheet.py stills "renders/stills/s0_*.png" -o review/scene0.png
python scripts/contact_sheet.py video renders/scenes/0_intro.mp4 --fps 1 -o review/scene0_motion.png
```

All utilities accept `--help`. A nonzero exit means a requested check failed;
a preflight failure for missing Algan is not evidence of a skill-format error.
The video checker reports JSON and does not modify the video.
