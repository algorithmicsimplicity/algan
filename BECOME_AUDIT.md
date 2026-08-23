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

## Found, not fixed

### 5. `become(ImageMob)` and `strategy="dissolve"` crash on several types

**16 of 841 matrix pairs.** Every one funnels through one line --
`_fit_bbox`'s `mob.scale(scale)`, where `scale` is a non-uniform 3-vector -- and
the failure is in `Mob.scale`, not in `become`:

| Type | Failure | Reproduces outside `become`? |
| --- | --- | --- |
| `Star`, `MathTex`, `Axes` (Manim-compat) | `TypeError: expected np.ndarray (got Tensor)` | **Yes** -- `Star().scale(torch.tensor([1.5, 0.8, 1.0]))` fails on its own; a uniform float works |
| a packed circuit (a `Text`'s inner `BezierCircuitCubic`) | basis rows (5) vs point rows (371) | Yes, calling `scale` on the packed circuit directly |
| `Arrow`, `DoubleArrow` | Manim's `get_last_handle` indexes `points[-2]` on a length-1 array | Only via `_fit_bbox` |

Left alone deliberately: the fix belongs in `Mob.scale`'s handling of non-uniform
scale on packed and Manim-compat Mobs, and getting that wrong would break far
more than `become`.

### 6. A Manim-compat container as a morph target does not land on the target

`Sphere().become(VGroup(Square(), Square()))` misses by peak 255 over **7.21% of
the frame**; `Sphere().become(Cross())` by 252 over 0.41%. The native-Algan
equivalent, `Sphere().become(Group(Line(...), Line(...)))`, is **byte-identical**,
as is `Sphere().become(Square(filled=False))`. So this is the `manim_compat`
wrapper as a target, not the geometry and not the PN route -- and it is the same
subsystem as finding 5. Pre-existing; the `Cross` case only became reachable
because finding 1 was fixed.

### 7. Morphing into a coarser Surface ends on an interpolation of it

`_reconcile_grid_pair` resamples both sides to the per-axis maximum grid, so a
target coarser along an axis is resampled *upward* -- and `F.interpolate` over a
coarse grid does not reproduce the surface those samples came from. A 6x6 plane
becoming a 4x9 wave ends **0.161 away** on a surface one unit across: 178 channel
values over 0.70% of an LD frame. Guarded as a strict `xfail` with the mechanism
and the intended fix (re-evaluate `Surface._func` on the finer grid rather than
interpolating its samples, which has to go through the base-grid cache because
the grid holds transformed coordinates).

### 8. One frame at the PN phase boundary shows the soup, not the target

On the cross-family route the triangle soup despawns and the target replacement
spawns at the same instant. At exactly that instant the soup has interpolated to
the target's geometry but is still diced as a soup rather than as the real
surface, so the frame differs from the target by up to **30 channel values over
132 pixels** -- and is byte-identical 0.05s later. Every PN morph into a `Sphere`
shows it (`Square -> Sphere`, `Cylinder -> Sphere`, `Cube -> Sphere`), and a
1-second morph at 30fps lands a frame exactly there. Pre-existing and small; the
fix would be to spawn the replacement a hair before the soup despawns rather than
at the same timestamp, which risks the double-draw
`test_pn_swap_uses_half_open_lifespans_with_no_gap_or_double_draw` guards.

### 9. A morph into a stroke-only shape goes blank in the middle

This is the worst-looking thing in the audit, and no assertion anywhere covers
it, because every check looks at the endpoints. Film-strip
`Cylinder -> Sphere -> Arrow` (`benchmarks/_become_chain_filmstrip.py`) and the
second morph is **empty for roughly a third of its duration**: the sphere fades
away, several frames show nothing at all, and the arrow pops in near the end.

`_bezier_to_pn_soup` zeroes an unfilled circuit's opacity, correctly -- there is
no fill to convert, and the stroke is not in the soup. But that makes the whole
target soup transparent, so the cross-family route tweens the *source's* opacity
down to zero and the morph has nothing to show until the real target spawns.
Every morph into a `Line`, `Arrow`, `Axes`, unfilled `Square` or any other
stroke-only shape does this, and so does the reverse.

Two ways out, neither a small patch: convert the stroke to a ribbon of triangles
so it is genuinely in the soup, or -- much cheaper -- give the target soup the
circuit's own opacity rather than zero, so the morph travels through a filled
silhouette of the target and the border phase then opens it out into an outline.
The second is a choreography change and wants a decision rather than a guess.

The same strip shows the documented seam caveat in its first row: a solid
becoming a solid tears into visibly separated strips mid-flight, because the PN
triangles are paired independently. That one *is* in `become`'s docstring; its
severity is not.

### 10. `Arrow3D` is the same aggregate defect and is not fixed

`Arrow3D.get_render_primitives` builds the shaft, the tip and their end discs
itself -- its own comment says none of them is a Scene actor and it is the only
thing that asks them to build. So it is an aggregator exactly as `Polyhedron`
is, and `become` publishes its parts separately. Both directions are wrong on
`HEAD` and still wrong here, identically: `Sphere -> Arrow3D` misses by peak 35
over 237 pixels, and `Arrow3D -> Sphere` **raises**
(`RuntimeError: The expanded size of the tensor (1) must match the existing
size (0)`). Verified pre-existing by running both against `e46264d` in a
separate worktree. The point-cloud family (`PMobject`, `DotCloud`) aggregates
the same way.

Declaring `draws_descendants` on them is *not* the fix, and was tried and
reverted: their `_morph_family` is `None`, so `_collect_morph_primitives` still
decomposes them into their `Cylinder`/`Cone`/`Dot3D` parts while registration
would withhold exactly those parts. The two halves have to move together, which
means giving these aggregates a morph family so they can convert as one unit --
a feature rather than a fix. `Polyhedron` is the one aggregate where the halves
already agree, because its family is `"mesh"`.

### 11. Reported by Ox, not independently reproduced here

* **#2** the same-kind endpoint keeps the source's `color_texture` when the two
  sides' texel counts differ (the attribute name encodes `W*H`).
* **#5** `_expand_n_batch`'s `objects_per_parent == 1` fast path and its
  `bincount` path build structurally different `parent_batch_sizes`.
* **#7** `_expand_n_tensor` accepts a `counterparts` argument and never reads it,
  though both call sites pass one. Confirmed by reading; harmless today, but the
  docstring of `_expand_n_list` describes counterpart-aware behaviour that this
  sibling does not have.
* **#4** `z_index` not adopted -- **did not reproduce**. `Square(z_index=2).become(Circle(z_index=5))`
  ends with the target's `z_index` without any help, so it was deliberately left
  out of `_MORPH_ADOPTED_ATTRS`: assigning it there would bypass the setter that
  propagates it to the sub-hierarchy.

## The assignment: is there a better way?

Yes, and the evidence is one picture.

`_primitive_pair_cost` builds `compatibility * 1e6 + secondary`, where by default

```python
secondary = abs(source_position - target_position) + min(distance, 1e3) * 1e-6
```

The order term spans `[0, 1]`. The distance term is capped at `1e-3`. **So the
default pairs by traversal order and the geometry only breaks exact ties** --
which is what the comment says it intends ("Distance only breaks otherwise equal
assignments without overriding that order"), but the consequence is stronger than
it sounds. A Group's child order is very often unrelated to its layout: built by
a loop, by `add()`, from a dict.

`benchmarks/_become_pairing_aesthetics.py`'s `scrambled_children` is that case
reduced to nothing else: four identical squares in the same four positions in
both hierarchies, differing *only* in the order the Group lists them. Nothing
needs to move. Under the default, all four converge on the centre of the screen
and pile up on top of each other at the halfway frame before flying back out.
Under `minimize_movement=True` they stand perfectly still.

What I would change, and what I measured:

1. **Make the two terms commensurable.** Normalize the distance by the span of
   the primitive centres so it also lands in `[0, 1]`, then blend, instead of
   adding a term that cannot exceed `1e-3`. A weighting of roughly
   `0.35 * order_gap + 0.5 * position_gap` reproduces the still result on
   `scrambled_children` and leaves `bar_reorder` and `size_swap` looking
   identical to today. Order stays the default semantics; it stops being the
   *only* semantics.
2. **Put size in the cost at all.** It is in neither mode. Two same-type
   candidates at the same distance are equally good matches whether one is the
   source's size and the other ten times it. A saturating log-ratio term is the
   cheap version.

Point 2 is a taste call rather than a defect -- `size_swap` shows a big circle
and a small one exchanging places, and all three rules cross-fade them in place
rather than sliding them past each other. Both readings are defensible (Manim's
`Transform` does the former, `TransformMatchingShapes` the latter), so it wants a
decision, not a patch.

**I did not change the default.** Any change to the pairing moves rendered
output, and `tests/full_renders/complex_hierarchy_become` is precisely the scene
that would move -- which needs both baseline sets regenerated, and the CUDA set
needs a CUDA machine. The harness and the numbers are here so the call can be
made with the filmstrips in view.

Two smaller notes from the same probe:

* `minimize_movement` is all-or-nothing: either order decides or distance does,
  never a blend. The blend above subsumes both.
* The compatibility rank dominates by `1e6`, so a same-type pair is always
  preferred over a nearer different-type one -- `Square@left + Circle@right`
  becoming `Circle@left + Square@right` sends both across the screen rather than
  changing shape in place. That is defensible (it preserves identity) and both
  band scales are safe against the secondary terms, per Ox #9. Noted, not
  changed.

## Verification

* `pytest -q tests/unit_tests tests/fast` -- **1834 passed, 93 skipped, 1 xfailed,
  1 failed**. The failure is `tests/fast`'s pixel-compared render, and it is
  **pre-existing**: checking `HEAD` (`e46264d`) out into a separate worktree and
  running the same suite there gives the identical failure, *"fast.mp4 differs
  from its baseline by up to 5 channel values (worst at frame 27)"*, against a
  tolerance of 2. It also cannot be caused by anything here --
  `tests/fast/scene.py` never calls `become`, and every function this branch
  touches is reachable only from `become`. Re-running with the Manim Tex cache
  warm gives the same 5, so it is not the cold-cache effect `CLAUDE.md`
  describes either; it is the CPU baseline being stale on this machine.
* `tests/unit_tests/test_morph_become.py` -- 26 passed, unchanged by any fix.
* `tests/unit_tests/test_morph_become_audit.py` -- 9 passed, 1 xfail. Every one
  reproduced its defect before the corresponding fix; the two endpoint-property
  tests were re-checked by blanking `_MORPH_ADOPTED_ATTRS` and confirming they
  fail.
* `benchmarks/_become_endstate_check.py` -- of the 26 default pairs, 18 landed
  byte-identically on the target before the fixes and **23** after. All three
  that still differ (`Square->Sphere`, `Cylinder->Sphere`, `Cube->Sphere`) are
  finding 8's one-frame phase-boundary artifact and are byte-identical 0.05s
  later. Findings 6 and 7 are outside this pair list and were measured
  separately (`Sphere:VGroupTwoSquares` at 7.21%, `SurfacePlane:SurfaceWaveCoarse`
  at 0.70%).
* Five pairs went from differing to byte-identical: `Square->SquareUnfilled` and
  its reverse, `Square->Star`, `Sphere->Cube`, `Cube->Tetrahedron`,
  `Polyhedron->Cube`.
* One pair moved that was previously matching, and it is worth stating rather
  than burying: **`Cube->Sphere`** now shows finding 8's phase-boundary frame
  (30 channel values over 132 pixels, byte-identical 0.05s later) because a Cube
  is now one morph unit and the pair takes the cross-family route instead of
  pairing twelve faces against one Sphere. It is the same one-frame artifact
  `Square->Sphere` and `Cylinder->Sphere` already had.
* An earlier iteration of the fix *did* cause a real regression, caught by this
  harness and not by any test: withholding registration under any Mob that
  answers `get_render_primitives` dropped the tip off `Line->Arrow` (peak 255
  over 282 pixels), because a `BezierCircuitCubic` answers it and yet does not
  draw its children. That is what `draws_descendants` exists to distinguish.
* A second review pass by the same agent (`OX_BECOME_FIX_REVIEW.md`, run
  against the fix diff) found two more things worth acting on, both now fixed:
  the empty-tiling catch matched only one of the two torch messages an empty
  tiling can raise (`cat` on one path, `stack` on the other), and an aggregator
  was speaking for a child a *user* had attached to it, so that child was
  withheld from the Scene and vanished from the morph. `Mob.owned_subtrees`
  narrows the claim to what the aggregator built for itself, and
  `test_a_polyhedron_speaks_only_for_the_geometry_it_built` guards it.
* CPU only. No CUDA machine was available, so nothing here speaks for the CUDA
  baselines.
