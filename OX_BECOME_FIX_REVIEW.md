# Review: the `Mob.become` bug-fix diff

Adversarial review of the working-tree diff against `HEAD` (branch
`claude/mob-become-audit-3kaz25`), per the brief in `/tmp/ox_brief_review.md`.
Nothing under the repo was written except this file; all probes ran from
`/tmp/opencode` with `/home/user/algan/.venv/bin/python`,
`ALGAN_USE_DAEMON=0`, and a `timeout`. HEAD comparisons used a `git archive`
extract at `/tmp/opencode/head_tree` (no stash, no tracked-file writes).

## 0. The reviewed artifact is not one artifact

**The copy at `/tmp/become_fixes.diff` and the working tree disagree at exactly
the hunk question 3 asks about.** The `/tmp` copy applies the adopted attrs
with `object.__setattr__(self, attr, getattr(target, attr))`; the working tree
(`git diff`, `mob.py:295-306`) uses plain `setattr` plus a docstring explaining
why ("bypassing it with `object.__setattr__` would shadow rather than set
anything that turns out to be a property, and the morph would pass an
attribute check while rendering unchanged"). Everything else matches.

Both spellings are reviewed below (Q3). Verdict: behaviorally identical today,
but the tree's `setattr` is the correct spelling to keep, and the stale `/tmp`
copy should not be treated as the record of what is being committed.

---

## Q1. Which classes does change 2 newly capture, and is that right?

Change 2 (`mob_morph.py:86-88`) stops collection at any Mob with
`_morph_family is not None` **and** (`is_primitive` **or**
`hasattr(mob, "get_render_primitives")`). Two facts bound the blast radius:

- `get_render_primitives` is defined at exactly eight sites:
  `pn_mesh.py:80`, `point_cloud.py:167`, `surfaces/surface.py:2943`,
  `three_d_models/mesh.py:338`, `shapes_2d.py:509` (TriangleVertices),
  `bezier_circuit.py:1020`, `shapes_3d.py:1067` (Arrow3D),
  `shapes_3d.py:1616` (Polyhedron). All other holders inherit.
- Capture additionally requires `_morph_family is not None`. That is set only
  on PNMesh ("pn_soup", pn_mesh.py:22), ImageMob ("image", image_mob.py:54),
  Surface ("grid", surface.py:816), TriangleMesh ("mesh", mesh.py:120),
  TriangleTriangulated/TriangleVertices ("mesh", shapes_2d.py:397,461),
  BezierCircuitCubic ("bezier", bezier_circuit.py:340), Polyhedron ("mesh",
  shapes_3d.py:1468).

| Class(es) | grp? | `is_primitive` at become | family | Newly ONE unit? | Correct? |
| --- | --- | --- | --- | --- | --- |
| **Polyhedron + Prism/Cube/Tetrahedron/Octahedron/Icosahedron/Dodecahedron/ConvexHull3D** | yes | **False** (never set; default False, animatable.py:273) | "mesh" | **YES - intended** | Yes: draws whole subtree under one mesh_key (shapes_3d.py:1616-1631) |
| Arrow3D | yes | **False** | **None** | No - family guard fails; still decomposes to tail/head units as before | n/a (change 3 affects it, see Q2) |
| PMobject, DotCloud, PointCloudDot, PGroup, TrueDot (point_cloud) | yes | **False** | **None** | No - same reason | n/a (registration skip via change 3) |
| Surface, Sphere, Cone, Cylinder, Torus, Dot3D, Line3D, _CapDisc, ImageMob | yes | True (surface.py:1146-1147) | grid/image | already captured pre-diff | n/a |
| TriangleMesh | yes | True (mesh.py:240-241) | mesh | already captured | n/a |
| TriangleVertices | yes | True (shapes_2d.py:475) | mesh | already captured | n/a |
| BezierCircuitCubic (+ ManimMob, manim-compat wrappers incl. Cross/VGroup) | yes | True (bezier_circuit.py:519) | bezier | already captured | n/a |
| PNMesh | yes | True (pn_mesh.py:67) | pn_soup | already captured | n/a |
| ThreeDModelMob | no grp | False | None | transparent container; TriangleMesh child remains the unit | verified: ctor registers 3 actors, new walk identical; `Sphere->TriangleMesh` byte-clean both trees |
| Neuron*/Synapse*/NeuralNetMLP*, plots.AxesMob, NumericDisplay | no grp defined anywhere in those files | - | None | unaffected | n/a |

Answer to the brief's list: of everything answering
`get_render_primitives` while `is_primitive` is False, **only the Polyhedron
family is newly collapsed to one morph unit, which is the intent**. `Arrow3D`
and `PMobject` are the other aggregators; they escape change 2 only because
their `_morph_family` is None - an accident of the guard's conjunction, not of
entitlement (both genuinely draw their subtrees: Arrow3D via
`_renderable_descendants`, shapes_3d.py:1043-1059; PMobject via
`_primitive_children`, point_cloud.py:152-159). If someone later gives Arrow3D
a `_morph_family`, change 2 will silently start collapsing it. That coupling
between family registration and unit-collapse deserves a comment or a
dedicated predicate.

Nothing was found that loses user-visible behaviour by being collapsed.

## Q2. Does change 3 fail to register something that needs registering?

### Constructor vs new walk, all ten constructions (executed)

| Construction | Constructor registers | New walk registers / skips |
| --- | --- | --- |
| Square | 4 (self + 3 component Mobs) | 4 / 0 |
| Circle | 4 | 4 / 0 |
| Text("hi") | 5 (Text + packed glyph circuit + 3 components) | 5 / 0 |
| Sphere | 2 | 2 / 0 |
| Cube | **1** | **29** / skips 12 TriangleVertices + 8 Dot3D |
| Tetrahedron | **1** | 15 / skips 4 TV + 4 Dot3D |
| Group(Square, Circle) | 9 | 9 / 0 |
| TriangleMesh(glb) | 3 | 3 / 0 |
| Arrow3D | **3** (arrow + 2 invisible endpoint markers) | 8 / skips Cylinder, Cone, 3 _CapDiscs |
| Surface | 2 | 2 / 0 |

Findings:

1. **No construction-registered Mob is skipped by the new walk** for any of
   the ten. Skips hit only descendants construction itself never registered
   (they are built `add_to_scene=False` inside aggregates).
2. **The docstring overclaims.** mob_morph.py:289 claims "Publish ``mob`` to
   the Scene the way constructing it would have", but for aggregates the new
   walk still publishes far more than construction ever did (Cube: walk ~29 vs
   constructor 1; real become left 115 actors for `Group(Cube,Arrow3D)` vs ~6
   at construction; HEAD's walk left 136). The extra actors are non-drawing
   wrappers (Groups, TriangleTriangulated, component row-holders), so no
   double-draw today, but the stated invariant is false and any future
   consumer of `scene.actors` that assumes it will be bitten.

### Other consumers of `scene.actors` (read; the skip is safe for each)

- **Memory preflight** - `render_loop.py:1964-1974` sums
  `_get_memory_used_per_timestep()` only over actors answering
  `get_render_primitives`. Every aggregate that now hides its children defines
  the family total itself: Polyhedron (shapes_3d.py:1611-1614),
  Arrow3D (1061-1065), PMobject over unregistered children only
  (point_cloud.py:161-165). Old code double-counted faces (Cube + each TV);
  new code counts once.
- **Materialization** - `render_loop.py:2007`
  `timeline.set_state_to_times(..., active_mobs=actors)`; the working set is
  expanded by `_collect_mob_ids` (timeline.py:2640-2673), which walks
  `children`, so unregistered descendants still materialize. Verified
  empirically: a user Dot3D attached inside a morph target ends up registered,
  materialized and correctly parented (but see defect 2 for its spawn state).
- **Draw order** - `_authored_draw_order` (`render_loop.py:1729-1735`) walks
  actors parent-first; skipped nodes are never circuits in any built-in
  configuration, so no coplanar bias entry is lost.
- **Vertex-bake warning** (1890-1896), **never-spawned-root diagnostic**
  (1936-1944), **window indexing** (365-415): all tolerant of the reduced set;
  the diagnostic's `not actor.parents` clause already collapses containers.

### The parents seed (`drawn_by_any_ancestor`, mob_morph.py:306-316)

- **Parent not in the Scene**: routine and harmless. In the PN/dissolve routes
  registration happens before splicing (mob_morph.py:1055-1057), so clones are
  seeded with no parents at all and register normally; the seed only suppresses
  when attachment already exists, which is precisely the `_expand_n_children`
  placeholder case the docstring cites (and there the ancestor genuinely draws
  the grown face via `_face_primitive_mobs`'s live walk).
- **Cycle**: guarded by the shared `seen` set; recursion terminates.
- **Many parents with different answers**: the seed is an OR - ANY
  `get_render_primitives` ancestor suppresses registration of that node. A Mob
  deliberately multi-parented under both an aggregating ancestor that does not
  draw it (Polyhedron's graph, say) and a plain container that relies on it
  being an actor would lose visibility. Not reachable through built-ins today;
  noted as a hazard of the conservative OR.

---

## Q3. Is `object.__setattr__` right? And the snapshot question

Per attribute (all plain; verified by reading every definition site):

| attr | kind | evidence |
| --- | --- | --- |
| `shader` | plain instance attr | mob.py:258 `self.shader = None`; no property |
| `two_sided` | plain class attr | mob.py:155 |
| `closed_shell` | plain class attr (+ instance overrides in `__init__`, e.g. shapes_3d.py:378) | mob.py:181 |
| `filled` | plain instance attr | bezier_circuit.py:450 |
| `empty` | plain instance attr | bezier_circuit.py:451 |

No property with a setter exists among the five, and no `Mob.__setattr__`
override exists anywhere in `algan/animatable_base/` (grep). So on today's
hierarchy `object.__setattr__` (the `/tmp` copy) and `setattr` (the tree)
behave identically. The tree's spelling is nonetheless the right one to keep:
it degrades gracefully if any of the five ever becomes a property, exactly as
its added docstring says.

**Render checks (executed, not read)** - `benchmarks/_become_endstate_check.py`
methodology, one frame per arm, LD:

- `Square -> SquareUnfilled`: endpoint carries `filled=False, empty=False`;
  morph endstate vs spawned target **peak=0, 0 pixels differ**.
- full `Sphere -> Sphere(partial sweep)` (same-kind route): endpoint declares
  `two_sided=False, closed_shell=False` matching the target; **peak=0,
  0 pixels differ**.

So adoption reaches the render; nothing shadows it.

**Snapshot/staleness**: `BezierCircuitCubic._MORPH_ADOPTED_ATTRS` builds from
`*Mob._MORPH_ADOPTED_ATTRS` at class-definition time (bezier_circuit.py:752-
757). `Mob`'s tuple is defined in the class body (mob.py:293) before any
subclass module can import it, so the name cannot be unresolvable and cannot
be stale within a process. The lookup at mob.py:303 is dynamic
(`self._MORPH_ADOPTED_ATTRS`), so subclass extensions are honored. The one
real hazard is a future subclass *replacing* the tuple instead of extending it
(dropping shader/two_sided/closed_shell); the docstring warns against it, but
nothing enforces it. No defect today.

---

## Q4. Is the `_bezier_to_pn_soup` fallback shaped correctly?

The fallback (`morph_conversions.py:331`) is
`local_2d = projected.reshape(-1, 2).mean(0).expand(1, 3, 2)`.

- **Shape/dtype/device**: matches the non-fallback branch `(N,3,2)`
  convention as `(1,3,2)`; same animation device and dtype as `projected`
  (the normal path reads `triangle_root.corners.location[0]`, same device).
- **Downstream use**: `world = location + local_2d[...,0:1]*e0*scale + ...`
  materializes a fresh contiguous tensor; the winding block and `torch.cat`s
  read only. Nothing writes into `local_2d` or into the expanded view.
  Contiguity concern: none.
- **`rows`**: `rows = world.shape[-2] == 3` in the fallback;
  `batch_corner_counts.append(rows)` (line 353) and the three
  `.expand(-1, rows, -1)` colour/opacity/glow calls plus the shader-param loop
  (368-376) are all consistent with 3 corners. Executed end-to-end:
  `Cross()` converts to a PNMesh of shape `[1,3,3]` where HEAD raises.
- **Winding block**: for the degenerate triangle the cross product is exactly
  zero, the sum is zero, `< 0` is False, `plane_normal` stays as normalized
  `basis[2]`. Leaving it unflipped is correct: the triangle has zero area and
  is culled twice over by the raster path, so the sign can only show up
  mid-morph as a cosmetic normal-blend difference, never at endpoints.
- **Exception matching**: executed on torch 2.7.1. For crossed lines the raise
  actually comes from `packed_reorder` ->
  `tensor_utils.py:42 torch.cat(chunks)` with message
  `'torch.cat(): expected a non-empty list of Tensors'`, which the substring
  catches; the diff comment attributes it to "the packing step", close enough.
  But the match is narrower than the failure space:
  1. If `tile_region` returns `(None, None)` (all perimeter points deduped
     away, triangulated_bezier_circuit.py:223-224 -> `continue` at 950-951),
     `all_tiles` stays empty and line 1004 raises
     `torch.stack([])`: `'stack expects a non-empty TensorList'` - **which the
     substring does NOT match**, so the RuntimeError escapes and the conversion
     crashes anyway. Not reachable through current callers (batches carry >= 1
     segment; point 0 always survives the dedupe mask), but latent.
  2. Message text is torch-version-coupled; a reworded cat/stack message
     silently re-breaks all 26 matrix pairs this fix was for.
  3. The `except` wraps the whole constructor including Mob/timeline setup
     (`super().__init__` at line 998 runs before the raising statements), so a
     half-constructed TBC leaks a timeline row and context registration before
     the fallback substitutes geometry. Harmless today (never spawns,
     `add_to_scene=False`), worth knowing.

## Q5. Is change 4's early return safe?

Reachability (executed over candidate pairs):
`Group()->Group()`, `Group(Mob(),Mob())->Group()`, `Group()->Group(Mob())`,
`Group()->Text("")`, packed-plain-Mobs->Group(): all collect zero primitives
on BOTH sides and hit the early return. In every reachable case both roots'
`morph_kind` is `(None, 1, 0)`: any Mob carrying a family or
`get_render_primitives` gets captured, so both-empty implies family-less,
nppo=1, no components. **A pair with different `morph_kind` cannot reach the
early return through built-ins today.**

Forced dispatch (executed): calling the early return's body with mismatched
kinds (`Group` vs `Square`) does not crash; it silently returns the SOURCE
object wearing copied values - i.e., if kinds could ever differ, the endpoint
would be wrong-class with none of the target's geometry installed, violating
the hierarchy route's replacement contract. Latent, unreachable today.

Does the fix achieve its goal? Yes (executed): `Group()->Group()` now records
`current_time=1.000000` with 10 attribute edits (control `Square->Circle`:
1.000000 / 35 edits). An earlier probe printing 0.000 was my own mis-readout
of a nested timespan after context exit, not a behaviour difference.

One residual nit: the early return hard-codes `replacement_allowed=True`
(mob_morph.py:1134). The hierarchy route is only entered with
`detach_history=True` (mob_morph.py:1416-1425), so this matches today's
callers; fine, but it is an assumption baked in silence.

---

## Q6. Did the diff move anything that was already right?

Executed: `benchmarks/_become_endstate_check.py` default 26 pairs, this
machine, CPU, both trees (same interpreter, same session):

- HEAD: **18/26** land byte-identically.
- Working tree: **23/26**. All six pairs the audit claims fixed went
  differing -> identical (Square<->SquareUnfilled, Square->Star,
  Sphere->Cube, Cube->Tetrahedron, Polyhedron->Cube).
- **REGRESSION: `Cube__Sphere` moved from byte-identical (HEAD, peak=0) to
  peak=30 over 132 px (tree, 0.03% of frame).** Deterministic across five
  independent re-renders; identical signature each time.

Bisect (executed, one variable at a time): blanking `_MORPH_ADOPTED_ATTRS`,
restoring old registration - neither helps. Restoring the OLD
`_collect_morph_primitives` (stop only at `is_primitive`) restores
byte-identity. **Change 2 causes it.**

Mechanism (executed): under the new single-pair pairing, the whole-Cube
PNMesh soup dies exactly at the recorded end time but remains a registered
actor; frames sampled at `end-1e-4` through at least `end+0.005` still show
its contribution (132 px along part of the sphere's surface, bbox
x[383,480] y[205,284]); removing the soup from `scene.actors` or sampling at
`end+0.5` restores byte-identity (peak=0). Keeping only the Sphere actor from
the morph scene renders byte-identical to the fresh target, so the replacement
clone itself is exact. The two pairs that differ identically on BOTH trees
(`Square__Sphere`, `Cylinder__Sphere`: peak=30 / 132 px) show this soup-leak
family pre-exists; change 2 extended it to a pair that used to be clean -
which falsifies BECOME_AUDIT.md's verification claim "No previously-matching
pair moved."

`tests/full_renders/complex_hierarchy_become`: contains Tetrahedra
(lines 97, 109, 119) - Polyhedra, exactly what changes 2 and 3 restructure -
plus Surfaces, circuits and an ImageMob under nested Groups. Pairing around
those Tetrahedra changes from four-face+four-Dot units to one unit, so
mid-flight checkpoints will move even if endpoints do not. The suite would
need CPU and CUDA baseline regeneration if this lands; per CLAUDE.md it skips
under `CI` and its baselines are per-machine, so I ran it neither as proof nor
as disproof.

Guard status on the working tree (executed):
`pytest -q tests/unit_tests/test_morph_become.py
tests/unit_tests/test_morph_become_audit.py` -> **35 passed, 1 xfailed**.

Additional aggregate-pair probes (both trees): `Sphere->Arrow3D` differs
equally pre/post (peak=28/174 px - pre-existing); `Sphere->PMobject` identical
pre/post (peak=35/17 px - pre-existing); `Group(Cube,Arrow3D)->same` improves
from 3529 px wrong (HEAD) to 143 px (tree) - change 3 demonstrably kills the
aggregate double-draw, with a small residue of the same soup-leak family.

### User geometry attached inside a Polyhedron target (Q2 hazard, executed)

`Sphere().become(Cube-with-a-user-Dot3D-child)` vs spawning that cube alone:
1482 px missing on the tree (the dot is registered and parented correctly but
`spawned=False`, so nothing draws it); 16823 px wrong on HEAD. The residual is
pre-existing in mechanism (same unspawned state reproduced on HEAD), so the
diff did not introduce it - but change 3's rule ("a descendant that ancestor
declines to draw is drawn for the first time") does not rescue it either: a
self-drawing descendant of a Polyhedron outside `self.faces` is skipped from
registration although `Polyhedron.get_render_primitives` draws faces only.
The same construction registers and renders it fine without `become`.
Nested-circuit-under-circuit targets are unaffected (verified byte-identical):
circuits take the same-kind identity route, and construction-registered
children keep rendering.

---

## Defects found

| # | Severity | Defect | Where | Trigger |
| --- | --- | --- | --- | --- |
| 1 | **High** | Previously byte-clean pair regressed: morph endstate differs from target by peak=30 over 132 px; despawned whole-Cube PNMesh soup remains visible at the end window. Falsifies "no previously-matching pair moved". | change 2, mob_morph.py:86-88 (bisected; old collect restores identity) | `Cube().become(Sphere())` (Polyhedron<->grid single-pair cross-kind hierarchy morphs generally), final-frame sampling |
| 2 | Medium | Registration walk's entitlement rule is false for non-aggregating grp ancestors: self-drawing descendants of a Polyhedron outside `faces` are skipped yet not drawn by it; combined with the pre-existing unspawned-clone gap they vanish from morph results (1482 px vs spawn-alone). Docstring also overclaims construction-equivalence (Cube: walk 29 actors vs constructor 1). | mob_morph.py:288-328, shapes_3d.py:1604-1631 | become into any Polyhedron carrying user-attached geometry outside `faces`; latent for any future grp Mob with partial aggregation |
| 3 | Low-Med | Empty-tiling catch matches one torch message only; sibling empty path raises `'stack expects a non-empty TensorList'` (triangulated_bezier_circuit.py:1004 via tiles=None->continue) which re-raises; message coupling breaks silently across torch versions; half-constructed TBC leaks a timeline row per fallback hit. | morph_conversions.py:329-331 | crossed-lines-family conversions today OK; degenerate zero-point batches or a torch message change reintroduce the original crash |
| 4 | Low | Change 2's collapse condition hinges on `_morph_family is not None`, so Arrow3D / PMobject escape unit-collapse by accident while satisfying the real entitlement (they draw their subtrees). A later family assignment flips them to collapsed without any signal. | mob_morph.py:86-88 | future maintenance, not current behaviour |
| 5 | Info | Early return hard-codes `replacement_allowed=True` and would return the source-class object if kinds ever diverged (unreachable with built-ins: both-empty forces kind `(None,1,0)`). | mob_morph.py:1121-1135 | latent only |
| 6 | Info | `/tmp/become_fixes.diff` is stale relative to the tree at the `_adopt_structural_attrs` hunk (`object.__setattr__` vs `setattr`). Behaviorally identical today; keep the tree's spelling. | mob.py:295-306 | process hazard, not code |

Not defects, verified clean: Q4 shape/dtype/device/rows/contiguity/winding;
Q3 all five attrs plain and adoption reaches pixels byte-identically
(Square->SquareUnfilled and full->partial Sphere both peak=0);
Q1 no unintended capture; materialization/preflight/draw-order consumers of
`scene.actors` unaffected by the skips.

## Executed vs read

Executed (scripts under `/tmp/opencode`, venv python, ALGAN_USE_DAEMON=0):

- Q1/Q2 enumeration of 23 builders: per-class family/is_primitive/grp,
  constructor registrations, new-walk simulation (q1_enumerate.py).
- Q6 double-tree endstate runs of all 26 default pairs
  (`endstate_wt.log`, `endstate_head.log`): 18/26 vs 23/26; Cube__Sphere
  regression identified.
- Bisect of Cube__Sphere across adoption/registration/collection variants
  (q6_bisect.py); leak attribution by actor-subset renders (q6_debris.py,
  q6_which.py); timing sweep excluding jitter (q6_timing.py, q6_sweep.py);
  lifespan readouts on both trees (q6_lifespan.py).
- Aggregate-pair endstates on both trees: Sphere->Arrow3D, Sphere->PMobject,
  Sphere->TriangleMesh, Group(Cube,Arrow3D)->itself (q2_agg_pairs.py).
- Q3 render parity for filled/empty and two_sided/closed_shell adoption
  (q3_adopted_attrs.py): both peak=0.
- Q4 fallback execution: Cross converts to [1,3,3] soup on tree, raises on
  HEAD (q4_fallback_probe.py); exact raise site + torch 2.7.1 messages for
  stack/cat (q4_raise_site.py).
- Q5 early-return reachability over candidate pairs, run_time occupation
  (current_time=1.0, edit counts), forced kind-mismatch dispatch
  (q5_early_return.py, q5b_timing.py, q5c_forced.py).
- User-extra-in-poly-target probes on both trees incl. actor/spawn-state
  introspection (q2c_extra_dot.py, q2c_dot_state.py, q2c_hier_inner.py,
  q2d_nested_circuit.py).
- Repo guards: pytest test_morph_become.py + test_morph_become_audit.py ->
  35 passed, 1 xfailed.

Read-only (source): consumer analysis of render_loop.py/timeline.py/
scene.py; property-vs-plain audit of the five attrs; import-order reasoning
for `_MORPH_ADOPTED_ATTRS`; tile_region None-return reachability; the
winding-block NaN/flip argument; Arrow3D/PMobject/Polyhedron aggregation
internals beyond what the probes exercised; everything about
`tests/full_renders` baselines (not run, per CLAUDE.md).

Hardware note: all rendering here was CPU in this container. Nothing above
speaks for CUDA behaviour or CUDA baselines.

---

## ADDENDUM - the diff was committed while this review ran

At 11:04, midway through this session, the concurrent author committed the
change as `d147390` ("Make a finished become leave what spawning the target
would have left"), followed by `ab5fe69`/`2580319`/`f1b0c92` recording the
audit files. **The committed version is a third variant**, different from both
snapshots reviewed above:

- The entitlement rule I reviewed as `hasattr(mob, "get_render_primitives")`
  became an explicit boolean: `Mob.draws_descendants = False` (HEAD mob.py:167),
  set True only on `Polyhedron` (shapes_3d.py:1590), consulted in
  `_collect_morph_primitives` (mob_morph.py:88) and in both halves of
  `_register_hierarchy_for_render` (mob_morph.py:318, 330). The commit message
  records that the author hit the same class of failure my Q1/Q2 analysis
  predicted - an earlier cut withheld registration under any Mob answering
  `get_render_primitives` and "dropped the tip off `Line().become(Arrow())`
  at peak 255 over 282 pixels".

Mapping of this report's findings onto current HEAD:

| Finding | Status at HEAD |
| --- | --- |
| Defect 1 (Cube__Sphere moved; despawned soup visible at the end window) | **Persists, re-executed**: peak=30 over 132 px at `end-1e-4` through `end+0.02`; byte-clean from `end+0.05`. The commit message discloses it ("the same one-frame phase-boundary artifact Square->Sphere and Cylinder->Sphere already had"); its "byte-identical 0.05s later" is accurate to within the bracket I measured. |
| Defect 2 (entitlement rule false for non-aggregating grp ancestors) | **Partially designed away**: the predicate is now explicit and only Polyhedron carries it, so circuits can no longer suppress their children's registration. The structural residues stand unchanged: the walk still publishes wrapper Mobs construction never registered (Cube ~29 vs 1), and a self-drawing descendant under a Polyhedron outside `faces` is still skipped although Polyhedron draws faces only (combined with the pre-existing unspawned-clone gap, that input still loses geometry at HEAD). |
| Defect 4 (Arrow3D/PMobject escape collapse only via family None) | Largely addressed: collapse now needs family AND `draws_descendants`; the accidental coupling is gone for flag-less aggregates. |
| Defects 3, 5, 6 and everything in Q3/Q4/Q5 | Unaffected by the redesign - those hunks landed essentially as reviewed (`setattr` spelling included, so the stale `/tmp` copy's `object.__setattr__` never shipped). |

All line citations in the body refer to the pre-commit working tree I was
given; HEAD line numbers differ (e.g. `_register_hierarchy_for_render` is now
at mob_morph.py:289-334). Re-executed against HEAD for the addendum:
Cube__Sphere sweep above; everything else was measured before the commit and
is labelled by snapshot.

