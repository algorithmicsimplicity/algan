# `Mob.become` audit

What follows is the result of trying to break `become` empirically -- 841 ordered
pairs of Mob types, every option axis, chained morphs, and rendered frames
compared against the target rendered alone -- plus a read-only source audit run
in parallel by a second agent (`OX_BECOME_AUDIT.md`, referenced below as "Ox").
The two overlap in one place and are otherwise complementary: the empirical pass
found the crashes and the wrong pictures, the source pass found the endpoint
properties that silently do not travel.

The standard used throughout is **the target**. A morph that finishes has to
leave the Scene holding what spawning the target alone would have held: the same
geometry, the same fill and shading, and no Mob the target would not have
registered. That is measurable in pixels, and where a claim below has a number
attached, it was measured that way rather than reasoned about.

## How it was measured

| Harness | What it does |
| --- | --- |
| `benchmarks/_become_stress.py` | 25x25 pair matrix; bounds, finiteness, mid-flight excursion |
| `benchmarks/_become_stress2.py` | 29x29 matrix with the awkward Mobs, plus `--mode options` (`minimize_movement` x `strategy` x `detach_history`) and `--mode chain` |
| `benchmarks/_become_endstate_check.py` | renders the last frame of a morph and the target alone, compares at the repo's +-2 channel tolerance |
| `benchmarks/_become_pairing_probe.py` | intercepts the assignment and prints which source paired with which target, and why |
| `benchmarks/_become_pairing_aesthetics.py` | renders morph filmstrips under each pairing rule so they can be judged by eye |
| `benchmarks/_become_chain_filmstrip.py` | film-strips `Cylinder -> Sphere -> Arrow`, because nothing else looks at the middle of a morph |
| `tests/unit_tests/test_morph_become_audit.py` | each defect below as a regression test |

Everything ran on CPU in a cloud session. There is no CUDA here, so no claim
below speaks for CUDA.

## What holds up

Worth saying first, because most of `become` is sound:

* **The 25x25 matrix is clean** -- 625 of 625 pairs record, materialize and land
  in the right place.
* **Chaining works, including across primitive families.** Eight chains of 4-6
  morphs each (`Square -> Circle -> Sphere -> Cube -> Text -> Star`,
  `GroupRagged -> GroupMixed -> GroupWide -> GroupEmpty -> GroupRagged`, ...)
  each end on their target at every step.
* **Most endpoints are byte-identical to the target.** Of 26 rendered pairs, 18
  matched exactly before any fix and 23 after -- and the three that still differ
  are all the same one-frame artifact (finding 8), byte-identical 0.05s later.
* Ox proved three things I had flagged as suspects and could not settle by
  inspection: `align_part_lists` **cannot** drop a short-side part (brute-forced
  over every `m <= n < 64`); the `results_by_target[target_index]` lookup in
  `_record_primitive_hierarchy_morph` **cannot** `KeyError`, because the paired
  and unmatched index sets partition the targets; and the five main routes all
  book **exactly one `run_time` unit**, including `_record_pn_morph`'s
  border-phase arithmetic.

## Fixed

### 1. A stroke-only shape could not cross primitive families (crash)

`Axes().become(Sphere())` raised `MorphConversionError`. So did `Cross()`,
`DashedLine()`, `MathTex(...)` and `VGroup(Line(LEFT, RIGHT), Line(UP, DOWN))`,
in either direction, against every 3-D type -- **26 of 841 matrix pairs**.

`_bezier_to_pn_soup` converts a circuit by triangulating each sub-path's
interior. A compound path that encloses no area at all tessellates to no tiles,
and the tiler's packing step cannot `torch.cat` an empty list. But an empty fill
is not a failure: what such a circuit draws is its stroke, and the soup zeroes an
unfilled circuit's opacity a few lines later regardless. It now stands one
degenerate triangle at the path's centroid, which contributes the rows the morph
interpolates and no visible area.

### 2. A morph did not take the target's fill, shading or sidedness

The same-kind path copies `{attr: getattr(theirs, attr) for attr in
mine.animatable_attrs}`. Anything the renderer reads that is *not* on the
timeline was therefore never carried, and the morph ended with the target's
geometry wearing the source's appearance:

* **`filled` / `empty`** -- `Square(filled=True).become(Square(filled=False))`
  ended solid where the target is an outline: peak 255 over **3.66% of the
  frame**. Found empirically.
* **`shader`** -- an unshaded source morphing into a shaded target kept its own
  pipeline (Ox #1, executed).
* **`two_sided` / `closed_shell`** -- a swept-partial `Sphere` is an open
  two-sided shell and a full one is a closed single-sided solid; the renderer
  reads both to decide whether a back-facing hit is shaded as an inside and
  whether `opacity` attenuates once or twice (Ox #3, executed).

These are one defect, so they get one fix: `Mob._MORPH_ADOPTED_ATTRS` names the
plain attributes a morph endpoint must take, `_adopt_structural_attrs` applies
them, and subclasses extend the tuple. `BezierCircuitCubic` adds `filled` and
`empty`.

### 3. Morphing into a Polyhedron grew a wireframe, and drew its faces twice

`Sphere().become(Cube())` ended with eight vertex beads and a wire cage a spawned
Cube does not have, plus a bright rim along every silhouette edge: peak 193 over
**1.04% of the frame**. The four-vertex `Polyhedron -> Cube` case showed exactly
four beads, which is what identified it.

`Polyhedron.get_render_primitives` returns the faces and nothing else, so the
vertex `Dot3D`s and edge Mobs under `self.graph` -- kept for Manim parity, where
`graph_config` styles them -- are geometry the Polyhedron owns but never draws.
Constructing a Cube registers **one** actor; `become` was registering **fifty**.
It walked into the graph, treated each dot as a morph unit, cloned a surrogate
for it, published the surrogate to the Scene so it could grow during the morph,
and spliced it back under the Polyhedron. The faces went the same way, so each
face was then drawn twice -- once by the Polyhedron and once by itself.

Two changes, both stating the same entitlement rule -- *a morph unit has to be
something the renderer actually draws*:

* `_collect_morph_primitives` stops at any Mob that answers
  `get_render_primitives`. A Polyhedron is one unit, however many children it
  keeps, exactly as the renderer already treats it (one `mesh_key`).
* `_register_hierarchy_for_render` does not publish a descendant whose geometry
  an ancestor already draws, and works out that ancestry from what the Mob is
  *already attached to* -- a placeholder face grown by `_expand_n_children` is a
  child of the Polyhedron before it is registered.

`Sphere -> Cube`, `Cube -> Tetrahedron`, `Polyhedron -> Cube` and
`Cube -> Polyhedron` are now byte-identical to the target.

### 4. An empty morph took no time

`Group().become(Group())` recorded **0.0s** where every other route records
exactly 1.0s, so it silently pulled everything after it in a `Seq` a second
early. A context whose block records no event never advances its cursor, and two
empty Groups gave it nothing to record. The roots still have attributes of their
own, so that case now records a root-attribute morph -- which is both the right
thing to animate and what makes it occupy its `run_time`.

## Fixed in the second pass

### 5. `become(ImageMob)` and `strategy="dissolve"` crashed on several types

**16 of 841 matrix pairs**, all through one line -- `_fit_bbox`'s
`mob.scale(scale)` with a non-uniform 3-vector. The bug was in `Mob.scale`, not
in `become`: `Star().scale(torch.tensor([1.5, 0.8, 1.0]))` raised on its own.
Manim-compat Mobs passed the factor into the vendored Manim mobject unconverted,
so NumPy deferred to the tensor and the points came back torch; packed Mobs met
a per-member pivot basis against per-point child rows. Both fixed. Deliberately
**not** through `to_manim`: that converter mirrors z, and a per-axis multiplier
is not a coordinate.

Fixing it exposed a second, latent bug underneath in the dissolve path itself:
`replacement.set(opacity=torch.zeros_like(replacement.opacity))` recurses, so on
a packed Mob a tensor shaped like the root's opacity met a descendant of a
different width. Every dissolve into a `Text` or `Tex` raised. A scalar
broadcasts instead.

### 6. `Arrow3D` and the point-cloud family are one morph unit now

Both build their whole subtree in `get_render_primitives` and none of the parts
is a Scene actor -- the same thing `Polyhedron` says with `draws_descendants` --
but they had no `_morph_family`, so `become` decomposed them into parts it then
had to publish separately. `Sphere -> Arrow3D` missed by 35 channel values over
237 pixels; `Arrow3D -> Sphere` raised. Both are byte-identical to the target
now. A new `"aggregate"` adapter converts each part and concatenates, which is
what makes the two halves agree: without a family, pairing decomposed them while
registration withheld the parts, and declaring `draws_descendants` alone made it
worse rather than better.

### 7. Morphing into a coarser Surface ends on that surface

Reconciliation moves both sides to the finer grid, so the coarser one was
interpolated up -- and interpolating a coarse grid does not reproduce the
surface those samples came from. A 6x6 plane becoming a 4x9 wave ended on a
bilinear resampling of four columns: **0.0258 mean deviation** from the analytic
wave, 0.108 at worst. `Surface._change_resolution` already re-evaluates the
parametric function on a new grid, which is what reconciliation should have been
doing: **0.00064 mean**, a 40x improvement. Packed surfaces keep the
interpolating path.

The endpoint is now sampled at the reconciled resolution -- 6x9 where the target
is 4x9 -- so it renders at a finer tessellation of the same surface and is not
pixel-identical to the target rendered alone. That is a sampling difference, not
a shape one, which is why the guard measures deviation from the surface's own
function rather than distance to the target's sample points: a nearest-point
comparison reports half a grid cell for a perfectly correct morph.

### 8. A morph into a stroke-only shape cross-fades

`_bezier_to_pn_soup` zeroes an unfilled circuit's opacity because there is no
interior to convert, so a cross-family morph into an `Arrow`, `Line`, `Axes` or
unfilled `Square` had nothing to show: the solid faded to nothing, roughly a
third of the frames were empty, and the outline appeared at the end. Such a pair
now cross-fades. `strategy="morph"` still forces the geometric route for a
caller who wants it.

### 9. Cross-family solids no longer tear

The two soups were paired triangle by triangle in build order, so a Cylinder
becoming a Sphere split into visibly separated strips while independent
triangles crossed to unrelated counterparts. `reorder_batch_to_minimize_movement`
already existed but only ran under `minimize_movement`; it now runs by default
under a cap, because it is a Hungarian solve over an N x N matrix. Measured on
this CPU: **0.07s at 1024 triangles, 0.66s at 2048, 0.97s at 2500, 2.2s at 3200,
3.3s at 4096**. The cap is 2500 -- a morph still pays about a second, and it
covers every solid measured (Sphere 462, Torus 1716, a Square's triangulation
2178) while leaving out text-sized soups (`Text("hello")` is 4379) where the
solve runs away. Over the cap it logs at `PERF` rather than degrading silently.

### 10. A Surface takes the target's colour texture

A texture is stored under an attribute name encoding its own `W * H`, so two
surfaces with differently-sized textures share no attribute for the same-kind
morph's `animatable_attrs` intersection to copy: a 4x4 red texture becoming an
8x4 blue one ended red. Assigned through the property rather than the generic
`_MORPH_ADOPTED_ATTRS` list, because the getter hands back the stored
`[1, 1, W*H*5]` row and the setter wants the `[W, H, 5]` image.

Also: `_expand_n_tensor` loses a `counterparts` argument it accepted and ignored
-- deliberately, per its own docstring, but both call sites passed one so it read
as an oversight rather than a decision.

## Found, not fixed

### 11. The frame at exactly the PN swap instant shows the soup

On the cross-family route the soup despawns and the replacement spawns at the
same timestamp. Sampled at that instant, the frame differs from the target by up
to **30 channel values over 132 pixels** -- and is byte-identical a frame later.
Every PN morph into a `Sphere` shows it (`Square`, `Cylinder`, `Cube` sources).
It is the soup's tessellation against the real surface's, at a moment when the
soup has already interpolated to the target's geometry, so it is a sub-frame
seam rather than a wrong endpoint. Left alone: moving the swap risks the double
draw that `test_pn_swap_uses_half_open_lifespans_with_no_gap_or_double_draw`
exists to prevent, for something a viewer sees for at most one frame.

### 12. `_expand_n_batch`'s two `parent_batch_sizes` branches

Reported by Ox (#5): the `objects_per_parent == 1` fast path and the `bincount`
path build lists of different lengths for the same input. Looked for an input
that produces a wrong result and **could not construct one**: across
`Text("ab") -> Text("hello")` and its reverse, every component keeps
`sum(parent_batch_sizes) == location.shape[-2]`, every character view indexes,
and each length matches its Mob's real member structure -- one entry per glyph on
the per-glyph Mobs, one on the circuit. Changing it without a failing case would
be a guess.

### 13. Reported by Ox, resolved or not reproduced

* **#2** `color_texture` not adopted -- confirmed and fixed, finding 10 above.
* **#7** `_expand_n_tensor`'s dead `counterparts` argument -- removed, above.
* **#5** the `parent_batch_sizes` branches -- finding 12 above.
* **#4** `z_index` not adopted -- **did not reproduce**.
  `Square(z_index=2).become(Circle(z_index=5))` ends with the target's `z_index`
  without help, so it was deliberately left out of `_MORPH_ADOPTED_ATTRS`:
  assigning it there would bypass the setter that propagates it to the
  sub-hierarchy.

## The assignment: what changed, and why

`_primitive_pair_cost` builds `compatibility * 1e6 + secondary`, where the
default `secondary` used to be

```python
abs(source_position - target_position) + min(distance, 1e3) * 1e-6
```

The order term spans `[0, 1]`. The distance term is capped at `1e-3`. **So the
default paired by traversal order and the geometry only broke exact ties** --
which is what the comment said it intended, but the consequence is stronger than
it sounds. A Group's child order is very often unrelated to its layout: built by
a loop, by `add()`, from a dict.

`benchmarks/_become_pairing_aesthetics.py`'s `scrambled_children` is that case
reduced to nothing else: four identical squares in the same four positions in
both hierarchies, differing *only* in the order the Group lists them. Nothing
needs to move. Under the old rule all four converged on the centre of the screen
and piled up at the halfway frame before flying back out.

The three things that can distinguish a pairing are now each normalized to
`[0, 1]` and weighted to sum to 1:

| term | weight | what it measures |
| --- | --- | --- |
| order | 0.35 | the gap between the two normalized traversal positions |
| position | 0.50 | centre-to-centre distance over the spread of the parts being paired |
| size | 0.15 | log ratio of extents, saturating at one decade |

Order still leads, because it is what makes `Text("abc") -> Text("abd")` pair a
with a. Position is close behind and is measured against the **spread of the
parts being paired**, not the frame, so a diagram ten units across and a glyph a
tenth of one get the same range of position costs and the same balance. Size was
in neither mode before; it is the nudge that keeps a big part big.

`minimize_movement=True` is untouched: a caller who asks for the least motion
still gets pure proximity.

Measured effects: `scrambled_children` now stands perfectly still, matching what
`minimize_movement=True` always did, and `bar_reorder`, `size_swap` and
`cluster_regroup` are visually unchanged. `_become_pairing_aesthetics.py` keeps
the old rule available as its `order` row, so the two can still be put side by
side.

The cost's subtree walks are also memoized per assignment rather than recomputed
per matrix cell -- the previous code walked both subtrees for every cell of an
`S x T` matrix, with a `.cpu()` sync each (Ox audit #8), and adding a size term
would have doubled that.

**This moves rendered output for any hierarchy morph, and no baseline was
regenerated** -- that was the instruction. `tests/full_renders`'
`complex_hierarchy_become` is the scene that moves; its CPU and CUDA baselines
both need regenerating, and the CUDA set needs a CUDA machine.

One thing deliberately left alone: the compatibility rank dominates by `1e6`, so
a same-type pair is always preferred over a nearer different-type one --
`Square@left + Circle@right` becoming `Circle@left + Square@right` sends both
across the screen rather than changing shape in place. That preserves identity,
which is defensible, and both band scales are safe against the secondary terms
(Ox #9).

## Verification

* `pytest -q tests/unit_tests tests/fast` -- **1841 passed, 93 skipped, 1
  failed**. The failure is `tests/fast`'s pixel-compared render and is
  **pre-existing and unchanged**: checking `HEAD` (`e46264d`) out into a separate
  worktree gives the identical failure, *"up to 5 channel values, worst at frame
  27"*, against a tolerance of 2; the number is still exactly 5 after every
  change here, including one to `Mob.scale`, which is used everywhere.
* `tests/unit_tests/test_morph_become.py` + `test_morph_become_audit.py` -- **42
  passed**, no xfails. Every audit test reproduced its defect before the
  corresponding fix.
* `benchmarks/_become_stress2.py --mode matrix` -- **841 pairs**: 838 ok, 1
  problem and 2 errors before this pass, all three now clean. (The two errors
  were `Image -> Text` and `Image -> Tex`, the latent recursive-set bug that the
  `Mob.scale` fix uncovered; the problem was the harness measuring a surface
  against the target's sample points.)
* `benchmarks/_become_endstate_check.py` -- **23 of 26** default pairs
  byte-identical to the target, up from 18. The three that differ are finding
  11's swap-instant frame and are byte-identical one frame later.
* CPU only. No CUDA machine was available, so nothing here speaks for the CUDA
  baselines.
