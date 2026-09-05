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
