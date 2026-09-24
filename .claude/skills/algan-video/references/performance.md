# Render performance and memory

Use this reference when one scene renders much slower than the others, a render
runs out of memory, or a full export would take too long. It is about producing
the requested result more cheaply, not about changing it. Do not reduce quality,
remove effects, or change the look to save time without telling the user.

## Measure before optimizing

Separate three costs: fixed per-process startup (imports, kernel cache loading;
tens of seconds on a cold process), per-scene authoring, and per-frame rendering.
The render daemon removes most of the first; see
[setup and rendering](setup-and-rendering.md).

Profile one representative scene rather than the whole project:

```python
from algan.utils.profiling_utils import profile_scene

profile_scene(scene_function, DRAFT_SETTINGS, tag="scene3", runs=1,
              kernel_profiler=False, telemetry=False,
              save_video_kwargs=dict(post_processes=POST),   # only if the project overrides them
              output_directory="profile")
```

`project.profile("scene_name")` and `--profile scene_name` run the same helper
for a Project scene. The report's stage table shows where wall time went. Useful
stage names:

| Stage | Mostly spent on |
|---|---|
| `beziers: _build_circuit_geometry` | 2D shapes and `Text`/`Tex` glyphs (Bezier circuits) that rotate, scale or morph; still or rigidly moving ones are built once per batch |
| `logical PN: ...` | Tessellating curved 3D surfaces |
| `ray traced render total`, `wavefront_*` | Tracing and shading pixels |
| `AttributeTimeline.*`, `set_state_to_times` | Materializing the animation timeline |
| `post-process` | Bloom and other post passes |

To find *which beat* of a long scene is expensive, author the scene up to a
chosen point (for example an early `return` controlled by an environment
variable), time `Scene.save_video` for successive prefixes, and difference the
results. Remove such development hooks afterwards.

Measured data points (GTX 1050, 540x960 draft, Algan 0.0.2). On Algan source up
to `45c1d6a3`, a moving camera made every glyph rebuild every frame: a
five-second beat showing about 500 monospace `Text` glyphs rendered at about 9 s
per frame, two thirds of it in the Bezier stage. Later source builds still or
translating glyph outlines once per render batch, even while the camera moves:
30 lines of monospace text (about 1200 glyphs) under a moving camera, 90 frames,
rendered in 3.8 s warm with 0.3 s in the Bezier stage (7.7 s and 3.5 s before),
and in 3.5 s when the whole text block also slid across the frame.

## Many glyphs of text

Still and rigidly moving text is cheap, as above. Text that rotates, scales,
morphs or writes on is still rebuilt every frame. When profiling shows that
stage dominating, and the text needs no vector edges at extreme zoom,
rasterize the text once and show it as an image. `ImageMob` accepts an
array `[H, W, C]` with C of 3, 4 or 5 (RGB, glow, opacity), is unlit, and is one
world unit tall (scale it to size). Use the user's requested font file:

```python
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def text_image_array(lines, font_path, px_per_line=64, pad=24, rgb=(255, 255, 255), glow=0.0):
    font = ImageFont.truetype(font_path, int(px_per_line * 0.75))
    probe = ImageDraw.Draw(Image.new("L", (1, 1)))
    w = int(max(probe.textlength(s, font=font) for s in lines)) + 2 * pad
    h = px_per_line * len(lines) + 2 * pad
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    for i, s in enumerate(lines):
        draw.text((pad, pad + i * px_per_line), s, font=font, fill=rgb + (255,))
    rgba = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
    return torch.cat([rgba[..., :3], rgba[..., 3:] * glow, rgba[..., 3:]], -1)


with Off():
    block = ImageMob(text_image_array(lines, font_path)).scale(block_height).spawn()
# Later, replace the content instantly (same array shape keeps it cheap):
with Off():
    block.set_color_by_image(text_image_array(other_lines, font_path))
```

Choose the rasterization resolution from the block's final on-screen size so the
text is not soft at the delivery resolution. Raster text is a different
implementation of the same content; keep `Text` when the user's specification
depends on vector behavior.

## Many similar Mobs: pack them

Each actor costs per-frame overhead. `batch_mobs` (exported by
`from algan import *`) packs existing Mobs into one:

```python
members = [Rectangle(width=0.2, height=0.05, stroke_width=0, add_to_scene=False).move(p)
           for p in positions]
pack = batch_mobs(members, add_to_scene=True)
pack.opacity = 0.0            # shared setup on the pack, before spawn
pack.spawn()

with Lag(0.2, runtime=2.0):   # members still animate individually through pack[i]
    for i in range(len(members)):
        pack[i].opacity = 1.0
with Sync(runtime=1.0):
    pack[1].color = RED
    pack[2].move(RIGHT)
```

Build members with `add_to_scene=False` so they are not left behind as separate
actors, configure materials on them before packing, and pack only paths with the
same stroke style. Opacity, color, glow, move and rotate on `pack[i]` were
verified on Algan 0.0.2.

## Textures

A textured Surface's texture is materialized for the frames of a render batch.
Many large textured Mobs therefore multiply memory, and a `color_texture` that
changes over time is heavier still. The symptom of too much is a CUDA
out-of-memory error raised from the animation timeline
(`generate_array_states` / `_query_row_states`) rather than from the renderer.

- Size textures to their on-screen footprint.
- To change what one surface shows, keep one Mob and swap its image
  (`set_color_by_image` inside `Off()`), rather than stacking several textured
  Mobs and toggling them.
- To switch between two fixed looks over time, cross-fading the opacity of two
  static textured Mobs is cheaper than interpolating `color_texture` itself.

## Other levers

- Render stills (`project.render_screenshots`) and only the changed scenes while
  iterating. Keep drafts at the delivery aspect ratio.
- `supersampling` multiplies rendered pixels by its square; a draft can use
  `supersampling=1` with `fxaa=True` (see the performance tutorial in the Algan
  docs). This changes draft appearance only; confirm the final settings.
- Algan chooses frame-batch sizes from memory estimates. An out-of-memory error
  is a reason to reduce the content's texture/geometry footprint or report the
  problem, not to patch the renderer.
