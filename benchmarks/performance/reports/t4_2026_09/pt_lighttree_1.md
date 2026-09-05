# Light tree A/B, Kaggle T4, 2026-09-05

Roadmap §6a/§6b landed at `fe74d03`: next-event estimation picks its
emitter by descending a per-frame Conty-Kulla tree (distance, orientation
cone, power) instead of a flat power CDF. `ALGAN_PT_LIGHT_TREE=0` is the
flat arm, byte-identical to the branch before the tree. Session
`pt-lighttree-1` (`pt-lighttree-1.txt`), five frames, 16 spp ceiling with
adaptive sampling (both arms took 5.76 mean spp), 4 bounces, denoiser on,
warm RUN 2, device `cuda`.

## Cost

| arm | end-to-end | `pt_shade` | traverse | host |
| --- | --- | --- | --- | --- |
| many_lights (64) 1280x720, tree | 2.098 s | 260 ms | 109 ms | 1.406 s |
| many_lights (64) 1280x720, flat | 1.616 s | 214 ms | 108 ms | 0.972 s |
| lit (3 lights) 1280x720, tree | 1.628 s | 217 ms | 111 ms | 0.977 s |
| lit (3 lights) 1280x720, flat | 1.582 s | 190 ms | 107 ms | 0.961 s |

The descent costs the shade kernel **21% on 64 lights and 14% on 3**, in
exchange for an 8.7x drop in mean squared error on a 32-light ring at
equal samples (the unit test's figure; the CPU measurement in the roadmap
is 8.8x). At equal error that is roughly 6x less transport work.

The host column is the finding: **+430 ms on the 64-light arm**, which was
the tree being rebuilt for each of five single-frame chunks — host-side
numpy at ~0.2 ms per node, 127 nodes, on Kaggle's slow vCPUs. The rig is
static, so the rebuilds produced byte-identical trees. `light_tree.py` now
memoizes builds by a digest of their inputs (`_tree_cache`), so a static rig
is built once per render; measured on this box the second chunk's build
time drops from 23 ms to hashing only. The re-measurement is
`pt-lighttree-2.txt` when it exists.

## Correctness on this box

`_pt_adaptive_check.py --scene lit` at 720p under the tree: denoiser off,
max 1 count on 2 of 1.84 M pixels (the rounding-count case), 0 interior;
`all arms agree`. The MIS identity (descent probability equals the upward
PMF walk) and the "probabilities sum to one" probes are unit tests and ran
green on the rebased tree.

## With the build memoized (`458e946`, `pt-lighttree-2.txt`)

| arm | end-to-end | `pt_shade` | host |
| --- | --- | --- | --- |
| many_lights 1280x720, tree | 2.029 s | 261 ms | 1.340 s |
| many_lights 1280x720, flat | 1.808 s | 230 ms | 1.132 s |
| many_lights 1920x1080, tree | 2.688 s | 660 ms | 1.243 s |

This session ran slower overall (the flat arm's host went 0.972 to 1.132 s
with no code change on its path — box variance), so the reading is the
gap: tree minus flat on the host fell from 430 ms to **210 ms**. The build
is gone from it; what remains is the tree path's device-to-host copies of
the light tensors at chunk setup, which force a sync that costs the render
loop its prefetch overlap with the previous chunk's kernels.

## Attribution with the PERF setup line (`9391d07`, `pt-lighttree-3.txt`)

Two repetitions of each arm in one session, next-event setup logged per
chunk:

| arm | end-to-end | `pt_shade` device | host | next-event setup per chunk |
| --- | --- | --- | --- | --- |
| tree, run A | 2.055 s | 272 ms | 1.342 s | 8.7–10.0 ms (127 nodes, cache hits) |
| flat, run A | 1.632 s | 215 ms | 0.977 s | 2.6–3.2 ms |
| tree, run B | 1.938 s | 276 ms | 1.190 s | 9–10 ms |
| flat, run B | 1.712 s | 221 ms | 0.991 s | 2.6–3.2 ms |

So the next-event setup is **7 ms per chunk** more with the tree — 35 ms
over five chunks — and every other kernel row is identical between arms
to the millisecond. The remaining **200–365 ms** of host residual on the
tree arm is not attributed by anything the harness measures: it is not
the build (memoized), not the geometry gather or the uploads (inside the
timed line), and not kernel execution (each kernel's wall includes it).
The cold run's `pt_shade` wall is 17.3 s against 14.8 s, so the tree
kernel is a longer compile; RUN 2 is warm, so that is not it either.
Candidates not yet excluded: Taichi launch overhead for the wider
arena-packed argument set (seven more arrays in the offset/shape tables,
75 launches), and the profiler's own per-launch cost growing with that
set. It does not reproduce as a device-time difference and it does not
show on the CPU box. Open; the next step is a `nsys`-style timeline on
the T4, which this harness does not take. At worst it is 12–18% of the
64-light frame's wall clock at 720p for an 8.7x variance reduction.
