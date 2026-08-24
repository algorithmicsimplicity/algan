# Audit: how `filled` reaches the renderer, and whether a fill can fade during a morph

Read-only audit per `/home/user/algan/scratch_ox_fill_brief.md`. Nothing under
`algan/`, `tests/` or `benchmarks/` was modified. Probes live in `/tmp/ox/`;
all runs used `.venv/bin/python` with `ALGAN_USE_DAEMON=0`, CPU render device,
PREVIEW settings, one process per arm.

**Provenance warning, read first.** At 08:59–09:01 UTC during this audit,
uncommitted working-tree changes appeared in `mob.py`, `mob_morph.py`,
`bezier_circuit.py`, `complex_hierarchy_become.py` and
`test_morph_become_audit.py`: an in-flight fix that routes fill-crossing pairs
to a cross-fade (§5d). Defect measurements were therefore taken **twice**:
against `HEAD` (commit `327867b`, extracted with `git archive` to
`/tmp/ox/head_tree`) and against the changed working tree. The fix touches no
renderer file, so the Q2/Q3 renderer findings hold for both states. Line
citations marked **(HEAD)** refer to the extract; others are current-tree.

---

## 0. Answers in one paragraph each

1. **Mechanism confirmed, with one refinement.** `filled` is a plain Python
   attribute (bezier_circuit.py:450), adopted by
   `_record_same_kind_morph`'s closing `_adopt_structural_attrs` call
   (mob_morph.py:993-996 HEAD), and read **live, once per render batch**, from
   `get_render_primitives` down into a constant `_rt_circuit_meta` column. So
   the value adopted "at the end" governs every frame the adopting mob owns.
   Refinement: under the default `detach_history=True`, frames *before* the
   morph are owned by a history-holding clone that snapshots the pre-adoption
   attributes (mob.py:1553, 1638), so they keep the fill; **from the morph's
   first frame onward** the fill is gone. Measured directly (§1).
2. **Not equivalent.** `filled=True` + zero fill-texture alpha vs
   `filled=False`, same 10 px stroke: **max channel difference 255 over 4078
   pixels**; at the defect scene's 2 px stroke: **max 253 over 1408 pixels**;
   with `border_width=0` they are byte-identical (both draw nothing). The
   difference is the drawn-region/stroke-placement decision made by the
   `_M_FILLED` column inside the Taichi hit/coverage code (§2).
3. Beyond the boolean readback, `filled` seeds which colour the fill texture
   gets at construction, fixes the non-planar rendering plan, redirects
   texture-grid colour writes, changes merge-block/draw-order bucket, PN-soup
   opacity, morph routing, Flash's clone construction and Manim interop (§3).
4. **`empty` is the same mechanism with a bigger hammer** — adopted by the same
   list, `get_render_primitives` returns `None` outright, and the whole mob
   (stroke included) vanishes from the morph's first frame. Measured. No test
   or package code constructs `empty=True`; only `ManimMob` does, for
   point-less Manim mobjects (§4).
5. Fade options enumerated with costs in §5. Key measured facts: the alpha
   fade of option (a) works and is smooth; option (b)'s end-of-morph swap is
   **not** invisible (it relocates the stroke half a width outward — 1408 px
   at 2 px stroke); option (c) still pops on its own and breaks batch identity.
   The callers that need `_record_same_kind_morph` to return `mine` itself are
   the `detach_history=False` contract holders (grep-listed in §5b).
6. The entitlement problem is real but shared: `shader`, `two_sided`,
   `closed_shell` have the same construction-time-declaration-read-per-frame
   character, and `_nonplanar_plan` is the *reverse* hole (construction-time,
   read per frame, **not** adopted, so `become` cannot cross planarity) (§6).

---

## 1. Q1 — the mechanism, confirmed

### 1.1 The attribute and its adoption

- `self.filled = filled` / `self.empty = empty` are plain instance attributes
  set once in `BezierCircuitCubic.__init__`
  (**algan/mobs/bezier_circuit.py:450-451**).
- They are adopted at the morph endpoint via
  `_MORPH_ADOPTED_ATTRS = (*Mob._MORPH_ADOPTED_ATTRS, "filled", "empty")`
  (**bezier_circuit.py:752-756**; base trio `("shader", "two_sided",
  "closed_shell")` at **algan/animatable_base/mob.py:331**).
- `_record_same_kind_morph` records attribute travel inside a `Sync` window
  and then, in a trailing `Off(...)` + `NoExtra(priority_level=1)` block,
  calls `mine._adopt_structural_attrs(theirs)`
  (**algan/animatable_base/mob_morph.py:992-996** HEAD), which plain-`setattr`s
  each adopted attr (mob.py:333-344). Because the whole timeline is recorded
  before anything renders, this write is not an ending: it is the only value
  the renderer will ever read for those frames.

### 1.2 Every place the renderer reads it (per render batch)

Build path — `filled` is read from the live Python attribute each batch:

| Site | What it does |
| --- | --- |
| **algan/render_loop.py:2071, 2138** | `actor.get_render_primitives()` called per frame window |
| **algan/render_loop.py:1782-1789** and fallback **1673-1682** | merge-group keys read `actor.filled` live (`_bezier_group_key`, `_bezier_block_key`) |
| **algan/mobs/bezier_circuit.py:1197** | `get_render_primitives` passes `filled=self.filled` into the primitive |
| **algan/mobs/bezier_circuit.py:1021-1022** | returns `None` outright when `self.empty` |
| **algan/mobs/bezier_circuit.py:1557** | batched builder: `mega.filled = first.filled` (group precondition "uniform ... filled", :1392-1396) |
| **algan/rendering/primitives/bezier_circuit_primitive.py:92, 101, 112, 213-220** | primitive stores `filled`; `get_batch_identifier` embeds it in the merge key |
| **algan/rendering/raytracing/primitives.py:3082** | `filled = torch.full((Tm, C, 1), 1.0 if self.filled else 0.0)` — a **time-constant** column packed into `_rt_circuit_meta` slot 13 (`_M_FILLED`, raytrace_kernels_taichi.py:372) |
| **algan/rendering/raytracing/primitives.py:3144-3145** | unfilled ⇒ `fill_alpha` forced to 0, feeding frame visibility/AABBs (:3140-3155, 3197-3202) |
| **algan/rendering/raytracing/primitives.py:3194-3195** | unfilled ⇒ opaque-prune flag forced off |
| **algan/rendering/raytracing/raster_taichi.py:1650** | analytic-coverage emission reads `_M_FILLED > 0.5`; drives `_circuit_query_radius` and `_circuit_point_region` |
| **raytrace_kernels_taichi.py:398-410** (`_circuit_query_radius`) | search radius: full stroke width if filled, half if not |
| **raytrace_kernels_taichi.py:413-441** (`_circuit_point_region`) | drawn region: filled = interior dilated by hairline, border = inner band `d ≤ border_w`; unfilled = centred band `\|d\| < border_w/2` |
| **algan/rendering/raytracing/raster_taichi.py:1687-1704** | coverage: unfilled selects the nearer band wall (sub-pixel strokes fade rather than vanish) |
| **algan/rendering/raytracing/raster_taichi.py:1721** | aa=3 oriented wedge gated `if filled` |
| **algan/rendering/raytracing/raster_taichi.py:1805-1818** | `border_frac`: filled computes covered-minus-interior share; unfilled sets 1.0 |
| **raytrace_kernels_taichi.py:1215** (in `_nearest_bezier_hit`, :1102; used at :1800) | wavefront traversal accepts a circuit hit only if `inside` per the same region function |
| **raytrace_kernels_taichi.py:2319, 2578** (latter in `_anyhit_opaque_bez`, :2480) | gather/shadow-prune paths repeat the read |

Renderer-adjacent mob-level readers (not strictly "the renderer" but part of
what the flag decides downstream): PN-soup conversion zeroes an unfilled
circuit's opacity (**algan/animatable_base/morph_conversions.py:372-373**);
stroke-only morph routing (**mob_morph.py:290-300**); draw-order walk skips
`empty` circuits (**render_loop.py:1743**).

### 1.3 The refinement: "all time" starts at the detach boundary

`become` first calls `detach_history()` (mob_morph.py:1621-1625 HEAD), which
hands the mob's recorded history to a hidden clone that **despawns at the
detach instant** and lets the original continue on fresh rows
(**mob.py:1526-1642**; clone at :1553, despawn at :1638, returns `self` at
:1642). The clone snapshots the pre-adoption attributes, so:

- frames before the morph are rendered by the clone → source's `filled=True`;
- every frame from the morph window onward is rendered by the original, whose
  `filled` was flipped at record time → `False`.

**Measured (HEAD, standalone `Circle(radius=0.42, color=BLUE_A).become(
SurroundingRectangle(filled=False))`, morph window [0.40, 3.00]):**

| t | BLUE_A fill pixels |
| --- | --- |
| 0.05–0.34 | 1060 |
| 0.36, 0.38 | 0 |
| 0.40–4.00 | 0 |

`result is src` is `True` (identity kept) and `result.filled` reads `False`.
The flip lands within a frame or two *before* the nominal window start
(attributed, not traced: the clone's despawn event is recorded inside the
`Sync` block and lands at the rescaled window start on `__exit__`). Either
way the brief's description is accurate in substance: **the fill pops off at
the morph's first frame and the remaining 2.6 s plays as an outline.**

Two consequences worth stating:

- The direction is symmetric: an outline becoming a filled endpoint renders
  solid from the morph's first frame (same mechanism; reasoned, not separately
  rendered).
- With `detach_history=False` there is no clone, so the adoption would flip
  even frames recorded before the morph (reasoned, not measured).
- The working tree's in-flight fix changes this pair's route entirely — see §5d.
  My earlier working-tree runs of the same scene showed a gradual cross-fade
  (centre pixel drifting 199→19 over the window, §5a) because the fix was
  already active, which is what prompted the double-state methodology.

---

## 2. Q2 — is `filled=True` + zero fill alpha the same picture as `filled=False`?

**No — measured, decisively.** One `Circle(radius=1.0)`, `border_color=YELLOW`,
identical circuit opacity; the only difference between arms is how "no fill"
is expressed. Frames rendered via `Scene.save_frame(path, PREVIEW, at=0.2)`,
one process per arm.

| Comparison | stroke | max channel diff | pixels > 2 |
| --- | --- | --- | --- |
| `filled=True`, fill-texture alpha forced 0 **vs** `filled=False` | 10 px | **255** | **4078** |
| same pair at the defect scene's stroke width | 2 px | **253** | **1408** |
| same pair, `border_width=0` | 0 px | 0 | 0 |
| control: `filled=True` alpha 1 vs alpha 0 (proves the write landed) | 10 px | 241 | 6978 |

At 10 px the differing pixels form four bands around the rim (measured along
the centre row: outer ~8 px and inner ~11 px on each side) — exactly where the
two stroke models disagree.

### Which code makes them differ — the `filled` column traced

1. **Fragment emission / drawn region (dominant).**
   `_circuit_point_region` (**raytrace_kernels_taichi.py:413-441**): a filled
   circuit draws its whole interior (dilated by the hairline `outline_w`) and
   lays its border **inward**, `[r - w, r]`; an unfilled one draws only the
   centred band `[r - w/2, r + w/2]`. Same stroke width, **different place** —
   the unfilled band's outer half lies outside the outline where the filled
   circuit draws nothing, and the filled border's inner half lies inside where
   the unfilled circuit draws nothing.
2. **Coverage / anti-aliasing.** raster_taichi.py:1687-1704: for unfilled, the
   signed-distance boundary flips past the stroke middle (band wall selection),
   which is also what fades sub-pixel strokes by width instead of letting them
   vanish; the aa=3 oriented-wedge model only applies `if filled` (:1721);
   `border_frac` is computed as covered-minus-interior share for filled and
   hard 1.0 for unfilled (:1805-1818), which changes border-vs-fill colour
   compositing on every straddling pixel.
3. **Nearest-edge query radius.** `_circuit_query_radius`
   (raytrace_kernels_taichi.py:398-410): full stroke width when filled, half
   when not — changes edge classification near corners and notches.

### What does *not* differ in this scenario (host side)

- `primitives.py:3144` forcing `fill_alpha` to 0 when unfilled is a **no-op**
  here: the true texture alpha is already 0, so frame visibility and AABBs
  come out identical either way.
- `primitives.py:3194` forcing the opaque-prune flag off when unfilled is
  likewise a no-op here: with `fill_min < 1` the circuit was already
  non-opaque.

So in the exact configuration the question asks about, the divergence enters
entirely through the kernel-side drawn-region/coverage decision — plus one
silent structural effect: the two arms land in **different merge blocks**
(`get_batch_identifier` embeds `filled`,
bezier_circuit_primitive.py:213-220), which matters only when other coplanar
geometry could be reordered around them.

And the equivalence *does* hold in the degenerate case: `border_width=0` +
zero fill alpha renders byte-identically (both draw nothing at all).

---

## 3. Q3 — everything else `filled` changes on the mob

On the mob itself (all in **algan/mobs/bezier_circuit.py**):

1. **Fill-texture seeding** — `__init__:489`:
   `fill_texture_kwargs["color"] = self.color if self.filled else
   border_color`. An unfilled circuit's *fill grid* holds the stroke colour;
   flipping `filled` later without rewriting colours silently changes what the
   fill samples return.
2. **Non-planar rendering plan** — `classify_circuit(control_points, filled)`
   is fixed at construction (:435) and re-run on repack (:532-534). Filled
   non-planar sub-paths become logical PN patches; unfilled ones split into
   camera-facing run circuits. Construction-time, never updated afterwards.
3. **Texture-grid colour writes** — `_apply_texture_grid_colors`
   (:808-837, read of `filled` at :832) writes the *border* grid too when
   unfilled; reached from `set_color_by_function` (:918-921),
   `set_color_by_image` (:967-982), and `Line.set_color_by_function`
   (**shapes_2d.py:324-327**). **Premise correction: there is no
   `set_fill_colors` anywhere in the package** — the brief's "~line 832"
   method is `_apply_texture_grid_colors`.
4. **Style profiles** seed `filled` per shape class (**shapes_2d.py:99-101**)
   and `Line` forces `filled=False` (:109-111).
5. `empty` additionally skips wave-refinement (:633) and the draw-order walk
   (**render_loop.py:1743**) — `filled` does not affect either.

Outside `algan/mobs/`:

6. **Batching / draw order**: merge identifier embeds `filled`
   (bezier_circuit_primitive.py:213-220); group keys read it per batch
   (render_loop.py:1680-1682, 1782-1789); the vectorized builder requires
   uniform `filled` across a group (bezier_circuit.py:1392-1396).
7. **Morph routing**: `_is_stroke_only` (mob_morph.py:290-300) treats
   unfilled/empty circuits as stroke-only, which is what routes such pairs to
   cross-fades (mob_morph.py:302-315, 1424-1432 HEAD).
8. **PN conversion**: `_bezier_to_pn_soup` zeroes an unfilled/empty circuit's
   soup opacity (morph_conversions.py:372-373).
9. **Flash** (`algan/animations/indication.py:446-453`): the passing-flash
   clone sets `flash.filled = False` and, when the source *was* filled, copies
   the source's **fill** colour onto the clone's border.
10. **Manim interop**: export maps fill opacity through `mob.filled`
    (**manim_compat.py:86**, second site :234); import derives `filled` from
    Manim fill visibility (**manim_mob.py:151-157, 175**).

**What would be wrong about leaving `filled=True` with a zero fill alpha**,
besides the readback: the endpoint never *looks* like the target (stroke stays
inward — §2's 1408 px at 2 px stroke); colour APIs keep writing only the fill
grid while the border grid goes stale; Flash, Manim export, PN conversion and
morph routing all take the filled branches; and the mob sits in the wrong
merge block relative to genuinely-unfilled neighbours. (Render memory /
dicing are unaffected — reasoned; `_get_memory_used_per_timestep`,
bezier_circuit.py:987-1018, does not read the flag.)

---

## 4. Q4 — does `empty` behave the same?

**Yes, and more bluntly.** Adopted by the same list
(bezier_circuit.py:752-756). `get_render_primitives` returns `None` outright
when set (:1021-1022) — stronger than `filled`, which merely changes what is
drawn. The constructor also zeroes both textures' alphas for an `empty` circuit
(:452-453, :485-486), so even a primitive built anyway would be invisible.

**Measured (HEAD)**: `Square(side_length=1.4, color=RED).become(Square(
side_length=1.0, empty=True))`, window [0.40, 3.00]: identity kept
(`res is src` True), `res.empty=True`, `res.get_render_primitives() is None`;
square-coloured pixels: **4624 at t=0.20 (pre-morph, via the clone), 0 at
t=0.45 and t=2.00** — stroke included, total disappearance from the morph's
first frames.

**Does any suite morph cross it? No.** Repo-wide grep: nothing constructs
`empty=True`; the only producer is `ManimMob` when the backing Manim mobject
has zero points (**manim_mob.py:130-133, 176**). The suite's "empty" tests are
about empty `Group`s (test_an_empty_morph_still_spends_its_run_time). So the
flag-crossing hazard is real but currently latent — reachable only by morphing
onto a converted point-less Manim mobject.

---

## 5. Q5 — ways a filled→unfilled same-kind morph could fade its fill

### (a) Keep `filled=True`, animate the fill-texture alpha to 0, never adopt `False`

- **Mechanically works, measured.** The fill grid's colour is a component Mob's
  timeline-backed attribute (`texture_points.get_animated_attribute("color")`,
  bezier_circuit.py:1076/1083; premise checked ✓). Probe: circle with 10 px
  border, `Sync(run_time=1.0)` tween of `texture_points.color` alpha → 0.
  Centre pixel across the window: 199 → 198 → 195 → 189 → 173 → 146 → 72 → 19
  (t = 0 … 0.9) — smooth, monotone; border-pixel count constant at ~3030
  throughout; `c.filled` stays True.
- **Costs / what breaks:**
  - The endstate never equals the target's look: measured difference vs a
    spawned unfilled twin is exactly the §2 table (1408 px at 2 px stroke).
    Any "morph ends on the target" parity check fails.
  - Boolean readback contracts: `test_become_takes_the_targets_fill`
    (tests/unit_tests/test_morph_become_audit.py:231-245) asserts
    `result.filled` flips; it would fail, and so would user expectations set
    by `print(mob.filled)`.
  - Everything in the §3 list keeps taking filled branches: colour writes go
    to a now-invisible grid, Flash copies fill→border, Manim export reports a
    fill, PN conversion carries fill rows, merge block disagrees with the
    target-like neighbourhood.

### (b) As (a), plus swapping in a `filled=False` replacement clone at the end

Pattern reference: `_record_pn_morph` registers soup + replacement, splices
(`_splice_replacement`, mob_morph.py:999-1012; call at :1226 HEAD), and swaps
lifespans inside `Off(spawn_at_end=False)` blocks (:1251-1255 despawn
source/spawn soup; :1264-1269 despawn soup/spawn replacement), returning the
replacement (:1277).

- **Is the swap invisible? No — measured.** At the swap instant the frame goes
  from "filled, alpha 0" to "unfilled", which §2 measured as **1408 differing
  pixels at the 2 px stroke** (max 253). Both states show a ring, but the ring
  relocates by half a stroke width; the viewer sees a pop outward. It shrinks
  toward invisibility only as `border_width → 0`. Secondary (reasoned): the
  swap also moves the mob into the unfilled merge block, so its coplanar
  draw-order bias bucket can change.
- **What depends on `_record_same_kind_morph` returning `mine` itself**
  (grepped):
  - `become()`'s own contract when `detach_history=False`
    (=`replacement_allowed=False`): `test_cross_kind_without_detach_history_
    keeps_identity_and_dissolves` asserts `result is source`
    (tests/unit_tests/test_morph_become.py:244-258); parametrized sibling at
    :392-394.
  - In-package `detach_history=False` callers: `Line.put_start_and_end_on`
    (**shapes_2d.py:355-357**), `Paragraph.set_alignment`
    (**text.py:1047-1053**, continues to mutate `self.children`),
    `_animate_to_manim` (**manim_compat.py:488-498**) and
    `sync_from_manim` (**manim_compat.py:609-632**, rewrites
    `self.children/components` after the call — identity load-bearing).
  - `_record_dissolve` ignores the return (calls for effect,
    mob_morph.py:1138-1144, returns its own replacement/source at :1150).
  - `_record_primitive_hierarchy_morph`'s no-primitives early return
    (:1290-1311 HEAD) uses the result as `final_root` for parent-slot filling;
    there neither side draws anything, so a replacement would be pure churn —
    and per `become`'s docstring (:1591-1594) updaters stay attached to the
    source object, which a silent swap would orphan.
  - Conclusion: any (b)-style swap **must be gated on
    `replacement_allowed`**, exactly as the working-tree patch gates its
    dissolve (§5d).

### (c) Make the renderer's `filled` column time-varying

- Plumbing: `_rt_circuit_meta` is already a per-time tensor
  ([Tm, C, channels], unified by `_unify_time`, primitives.py:3064); the
  column is written constant from the Python bool (:3082). Making it vary
  means making `filled` animatable (new timeline attribute, new materialization
  path) or synthesizing a per-batch ramp in the morph system.
- Kernel cost: none extra — `_circuit_query_radius`/`_circuit_point_region`
  already take runtime bools. But **(c) alone still pops**: the drawn region
  jumps interior↔band at the switch frame whatever the interpolation, so you
  would have to add (a)'s alpha fade anyway.
- Batch identity breaks: the merge identifier embeds `filled`
  (bezier_circuit_primitive.py:213-220), so a mob changing buckets mid-life
  reshuffles merge layout between batches (visible coplanar reordering risk),
  and the vectorized builder's uniform-`filled` precondition
  (bezier_circuit.py:1392-1396) becomes fragile.
- Fixes nothing off-renderer: `.filled` readback, colour-write targeting,
  Flash, Manim export and stroke-placement semantics still see one value.

### (d) What actually appeared mid-audit (working tree, uncommitted)

The concurrent change declares `filled`/`empty` **untravellable**
(`Mob._MORPH_UNTRAVELLABLE_ATTRS` + `_morph_structural_break`, mob.py:346-373
current; extended at bezier_circuit.py:758-769), routes crossing pairs to
`_record_dissolve` when `strategy="auto"` **and** `replacement_allowed`
(mob_morph.py:1539-1557 current), applies the same rule to hierarchy-route
pairs (mob_morph.py:1444-1456 current), and penalizes crossing pairs in the
assignment (`_primitive_compatibility_rank`, mob_morph.py:110-131 current).
This is option (b)'s mechanism reached through the existing dissolve route:
the source fades out holding its own fill, the target arrives pre-formed, no
mid-flight swap needed, and identity contracts are preserved by the
`replacement_allowed` gate. Measured on the working tree, my standalone
Circle→SurroundingRectangle repro cross-fades smoothly instead of popping, and
`pytest -q tests/unit_tests/test_morph_become_audit.py
tests/unit_tests/test_morph_become.py` passes: **48 passed** (including the
two new fill-crossing guards). Costs: a crossing pair no longer travels
geometrically, and the assignment may pair differently than before.

---

## 6. Q6 — entitlement: construction-time data answering per-frame questions

True here, and not unique to `filled`:

- The other adopted attrs — `shader`, `two_sided`, `closed_shell`
  (mob.py:331) — are plain/class declarations read at primitive-pack time each
  batch. Adoption gives them the same whole-life-flip character; they bite
  less often only because same-kind endpoints rarely disagree (a partial-sweep
  Sphere ↔ full Sphere pair does cross `closed_shell`/`two_sided` mid-morph —
  reasoned, not pixel-measured here).
- On `BezierCircuitCubic` specifically, the **reverse hole** also exists:
  `_nonplanar_plan` is construction-time (:435), read every batch (:1056), and
  **not** adopted — so `become` cannot update it, and a planar source morphed
  onto non-planar geometry keeps rendering with the *source's* flattening
  decision for its whole remaining life. Likewise `grid_width/grid_height/
  num_texture_points` are per-frame-read construction data preserved only
  because same-kind pairing picks grid-compatible targets.
- `z_index` shows the intended alternative: excluded from adoption on purpose
  (comment at bezier_circuit.py:747-756) because assigning it would bypass the
  setter that propagates it.
- Framed precisely: `filled` answers three questions — (i) per-frame
  appearance (is the interior drawn), (ii) a geometry convention (where the
  stroke sits), (iii) classification (batching, routing, conversion). Only
  (ii) legitimately belongs to construction; Algan's own convention puts (i)
  on the timeline (every other appearance channel is row-backed), and the
  defect is (i) being answered by construction data. The working-tree patch's
  `_MORPH_UNTRAVELLABLE_ATTRS` formalizes exactly this split for
  `filled`/`empty`.

---

## 7. Premise checks the brief asked for

| Premise | Verdict |
| --- | --- |
| "`filled` is read live" (at primitive-build time, once per render batch) | **Confirmed** (§1.2 sites; measured via §1.3/§2 probes) |
| "`_adopt_structural_attrs` runs at the end of the recorded morph" | **Confirmed** (mob_morph.py:993-996 HEAD) — with the refinement that its effect starts at the detach boundary, so pre-morph frames survive via the clone (§1.3) |
| "the value the renderer reads is the final one for every frame of that mob's life" | **Confirmed for every frame the adopting mob owns** (the whole morph and after); false for pre-morph frames under `detach_history=True`. The defect statement "disappears on the first frame of the morph" is accurate |
| "the fill's alpha comes from `texture_points`' colour, which is timeline-backed" | **Confirmed** (bezier_circuit.py:1076/1083; fade demonstrated in §5a) |
| "`set_fill_colors` (~line 832)" | **Wrong name** — no such method exists; the unfilled dual-grid write is `_apply_texture_grid_colors` (bezier_circuit.py:808-837, flag read at :832) |

## 8. Probe inventory and environment caveats

Scripts (all in `/tmp/ox/`): `probe1_defect_repro.py` (defect census, run
against HEAD and working tree), `probe1b_boundary.py` / `probe1c_centerpix.py`
(boundary and centre-pixel series), `probe1d_who_draws.py` (actor census),
`probe1e_route.py` (route instrumentation), `probe2_alpha_vs_flag.py` +
`probe2_arm.py` (Q2 arms), `probe3_fade.py` (alpha-fade demonstration); images
`q2_*`, `q2w_*`, `fade*`, `out_*`. HEAD measurements ran against a
`git archive` extract (`/tmp/ox/head_tree`, commit `327867b`) with
`PYTHONPATH` override — no tracked file was touched.

Everything above was rendered on **CPU** in this container; nothing here
speaks for CUDA behaviour or CUDA baselines. Pixel counts are PREVIEW
resolution (704×396); tolerances are stated alongside each number. Claims
marked "reasoned" were not executed.

The deliverable stops at diagnosis — nothing was fixed.
