# Authored-appearance light sampling A/B, Kaggle T4, 2026-09-05

Roadmap §6a-bis landed at `0c4e30c`: an authored-appearance material (manim,
toon, matcap, a custom fragment pipeline) under the path tracer samples its
light rows instead of summing every one with a shadow ray each up to
`max_shadow_lights`. `--pt-authored-light-sampling off` is the summing arm
(today's behaviour), `always` the sampling arm; the shipped default `auto`
takes the sampling arm only past the 16-light cap, so on this 64-light scene
it is the `always` arm. Session `pt-authored-1` (`pt-authored-1.txt`, at
`8b58e20`): `pt_baseline.py --scene many_lights_authored`, the lit solids in
authored materials under 64 point lights, five frames, 16 spp ceiling with
adaptive sampling (5.76 mean spp at 720p, 5.74 at 1080p in every arm), 4
bounces, denoiser on, warm RUN 2, device `cuda`.

## Cost

| arm | end-to-end | `pt_shade` | traverse | host |
| --- | --- | --- | --- | --- |
| 1280x720, off (sum all 64) | 3.152 s | 1619 ms | 110 ms | 1.138 s |
| 1280x720, always (sample) | **1.606 s** | **235 ms** | 106 ms | 0.986 s |
| 1920x1080, off | 5.849 s | 3894 ms | 260 ms | 1.177 s |
| 1920x1080, always | **2.550 s** | **598 ms** | 260 ms | 1.164 s |

The shade kernel is **6.9x cheaper at 720p and 6.5x at 1080p**, and the
frame is **2.0x and 2.3x faster end to end**, at the same sample count.
The CPU box measured 7.1x on the kernel and 1.9x on the wall (roadmap
§6a-bis). The summing arm's 1.6–3.9 s of shade is the O(lights) cost model
the fallback renderer exists to escape, and it was also silently wrong:
the summing arm shadows only the first 16 rows, the sampling arm shadows
from all 64.

For scale, the physically-integrated `many_lights` scene (the same solids
in PBR materials, `pt_lighttree_1.md`) shades in 260–275 ms at 720p, so an
authored material now costs about what a lit one does under the path
tracer rather than six times as much.
