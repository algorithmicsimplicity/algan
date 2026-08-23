# Audit: `Mob.become()` — read-only source review

Scope: `algan/animatable_base/mob_morph.py`, `algan/animatable_base/morph_conversions.py`,
`algan/mobs/text.py` (`Text.become`, line 386), `tests/unit_tests/test_morph_become.py`,
against the conventions in `CLAUDE.md`. All line numbers refer to the tree as of
this audit. Verification scripts ran from `/tmp/opencode`; nothing under the repo
was written except this report.

**Repo-state note.** The tree was **not** byte-identical before this audit began,
and a *concurrent writer* modified the repo while the audit was in progress.
Everything below is documented so my footprint is exactly separable:

- Untracked files present **at audit start**: `Axes`, `Image`, `MathTex`,
  `benchmarks/_become_endstate_check.py`, `benchmarks/_become_pairing_probe.py`,
  `benchmarks/_become_stress.py`, `benchmarks/_become_stress2.py`.
- Untracked files that appeared **during** the session, none authored by this
  audit (every audit write went to `/tmp/opencode`):
  `benchmarks/_become_pairing_aesthetics.py` (10:06:42),
  `tests/unit_tests/test_morph_become_audit.py` (10:13:04).
- **Tracked-file modifications landed at 10:13–10:16**, after all audit probes had
  completed (last probe run ≈10:12): working-tree edits to
  `algan/animatable_base/mob_morph.py`, `algan/animatable_base/morph_conversions.py`,
  `algan/mobs/shapes_3d.py`, `algan/mobs/bezier_circuit.py`. These are another
  session's in-flight work on the same subsystem (its new test file's header says
  it guards "defects found auditing ``Mob.become`` against what the target alone
  renders"). This audit made none of them and reverted none of them — reverting
  another writer's uncommitted work would have destroyed it.

**Citation anchor:** all file:line citations and quoted code refer to git commit
`e46264d` ("Merge pull request #57 …") — i.e. `git show HEAD:<path>` reproduces
exactly what was audited; the concurrent edits are uncommitted drift on top.
All key quotes were re-verified present-and-unchanged in both HEAD and the
drifted worktree (`border_fraction = 0.3`, `unit = am.context.run_time_unit`,
the `align_part_lists` formula, the same-kind values-dict comprehension,
`_morph_center`), but line numbers may shift as that other session lands its work.

I deleted/modified nothing. The only change attributable to this audit is the file
you are reading.

---

## Q1. Does `align_part_lists` do what its docstring claims?

**Yes on retention — no source index can be dropped; the positional pairing is the
intended even-spread correspondence, not an arbitrary shuffle.**

Code (`mob_morph.py:25-57`):

```python
repeat_indices = [(slot * current) // target for slot in range(target)]
seen = set()
aligned = []
for slot, source_index in enumerate(repeat_indices):
    if source_index not in seen:
        aligned.append(short[source_index])
        seen.add(source_index)
    else:
        aligned.append(make_placeholder(long[slot], side=side))
```

### Can an index of `range(m)` be dropped?

No, for any `(m, n)` the function can actually receive. `pad` is only reached with
`current = len(short) <= target = max(len(mine), len(theirs))` (the `current ==
target` and `current == 0` cases return early at lines 41–44; `m == 0` would also
divide by zero and is exactly what the `current == 0` guard prevents). So the
formula always runs with `0 < m ≤ n`.

*Proof:* consecutive terms differ by `⌊(s+1)m/n⌋ − ⌊sm/n⌋ ∈ {0, 1}` because the
step `m/n ≤ 1`. The sequence starts at `⌊0⌋ = 0`, ends at `⌊(n−1)m/n⌋`, and
`(n−1)m/n = n − n/m ≤ n − 1` (since `n/m ≥ 1`), so the last value is at most
`m−1`... and because values only ever hold or increase **by one**, a monotone
sequence from `0` reaching `≥ m−1` without ever jumping must visit **every**
integer in `[0, m−1]`. Hence `set(repeat_indices) == set(range(m))`: every
short-side part survives at its *first* slot, and later occurrences become
placeholders.

*Executed check* (brute force, `/tmp/opencode/verify_arithmetic.py`):

```
pairs (m, n) where some source index is dropped: NONE        # all m ≤ n < 64
max step between consecutive slots over all (m<=n<64): 1     # confirms the step argument
```

So the docstring's "The first occurrence of every original short-side part is
retained" holds exactly; **there is no `(m, n)` that silently deletes a part**, and
therefore no user-visible deletion consequence to report.

### Does a retained part land at a corresponding slot?

The output list replaces the short side at full length `n`; the long side is used
unchanged. Positional zip therefore pairs slot-for-slot:

- retained part `i` sits at its first slot `s_i = min{ s : ⌊s·m/n⌋ = i } ≈ i·n/m`
  — i.e. at its *proportional* position;
- each placeholder slot `s` is built from `long[s]` itself
  (`make_placeholder(long[slot], side=side)`), so placeholders always face their
  own counterpart.

Executed example (`m=2, n=5`): `repeat_indices = [0,0,0,1,1]` → retained pairs
`(slot 0, short[0])`, `(slot 3, short[1])`; the zip gives
`short[0]↔long[0]`, placeholder-of-`long[1]`↔`long[1]`,
placeholder-of-`long[2]`↔`long[2]`, `short[1]↔long[3]`,
placeholder-of-`long[4]`↔`long[4]`.

A retained part is never paired with an *unrelated* long-side part beyond the
coarseness inherent to proportional spreading (with `m=2, n=5`, `short[1]` faces
`long[3]`, skipping two slots that grew their own twins). "Correspondence" here is
by traversal index, not by geometry — if traversal order does not match spatial
order, even a perfect spread pairs by the wrong neighbour — but that is a property
of index-based pairing generally (`minimize_movement=True` exists for it), not of
this padding.

---

## Q2. Is the data each cost term reads entitled to speak for the primitive?

### `_morph_center` on a packed Mob

```python
# mob_morph.py:95-98
center = mob.get_center()
return center.reshape(-1, center.shape[-1]).mean(0).detach().float().cpu()
```

`get_center()` is the **midpoint of the subtree's axis-aligned bounding box** and
returns shape `(*, 8, 3)` internally — eight box corners (`mob_layout.py:105-116,
221-243`). The `mean(0)` averages those eight corners, which is algebraically the
box midpoint again. So `_morph_center` is **one point per primitive: the centre of
one bbox drawn around everything the primitive owns**, regardless of packing. A
packed `Text`'s morph centre is the midpoint of the whole string's box; a
`PNMesh` soup's is the midpoint of the whole soup.

Is that honest for pairing? At the level it is used — choosing which *primitives*
pair — mostly yes, but it is blind in three concrete ways:

1. **Layout-blind.** Two primitives whose boxes agree have distance ≈ 0 however
   different their contents. Executed demo: `Group(Square@LEFT, Circle@RIGHT)` vs
   the mirrored group give centres `5.5e-6` apart → the distance term cannot
   distinguish them and pairing falls entirely to the traversal-order term.
2. **Membership-blind.** A packed batch paired against several single-object
   primitives is summarised by one interior point; nothing in the cost sees the
   individual members (member-level proximity is handled later, and only inside
   the same-kind path, by `reorder_batch_to_minimize_movement`,
   `mob_morph.py:502`).
3. **Distribution-blind.** An L-shaped or hollow arrangement reports the box
   middle, which may sit on empty space.

None of this corrupts correctness — worst case is a cosmetically wrong pairing —
but a single centroid is *not* an honest summary for packed members; it is an
honest summary only of the primitive's extent.

Side note: `.cpu()` per call makes each of the O(S·T) cells sync the animation
device (`mob_morph.py:131-133` calls `_morph_center` twice per cell).

### Scale safety, default route

Compatibility rank ∈ {0,1,2,3,4} (`_primitive_compatibility_rank`,
lines 100-117), multiplied by `1e6` → band boundaries at multiples of `1e6`.
Secondary term (lines 137-143): `abs(source_position − target_position) +
min(distance, 1e3) · 1e-6 ≤ 1 + 1e3·1e-6 = 1.001 ≪ 1e6`. **Safe by three orders
of magnitude; a secondary term can never cross into the next band.** The matrix
is `torch.float64` (line 150), so ties within a band keep precision.

### Scale safety, `minimize_movement=True`

```python
if minimize_movement:
    secondary = distance          # unbounded Euclidean distance
```

Unbounded in principle: a pair of rank `r` with centres ≥ `1e6` world units apart
outranks a rank-`r−1` pair at any distance. Practically unreachable — the frame
is ~14×8 world units and nothing in the engine places mobs near `1e6` — but there
is no clamp mirroring the `min(distance, 1e3)` guard used on the default route.
Verdict: safe in every realistic scene, unbounded by construction.

---

## Q3. Rectangular assignment and the consumer

### What SciPy returns

`scipy.optimize.linear_sum_assignment` on an `S×T` matrix returns index arrays of
length `min(S, T)`, a minimum-total-cost one-to-one assignment; when `S ≠ T` the
shorter side is fully matched and the longer side is partially matched.
Executed (`verify_arithmetic.py`):

```
S=5 T=3: len(pairs)=3 partition_ok=True
S=3 T=5: len(pairs)=3 partition_ok=True
S=4 T=4 / S=1 T=6 / S=6 T=1 : same shape of result
```

### Unmatched-index computation

Lines 162-177 compute `paired_sources`/`paired_targets` as sets from the returned
arrays and take complements over `range(len(...))`. By construction the complements
partition each side (executed `partition_ok=True` above). Both directions handled
symmetrically — `S > T` leaves sources unmatched, `S < T` leaves targets
unmatched, and the early return at line 147-148 handles an empty side by marking
everything unmatched.

### Can `results_by_target[target_index]` KeyError? (line ~1180)

**No.** The guarantee is the partition plus how `pair_specs` is built:

- every pair contributes `(…, target_index)` for each target in the assignment
  (lines 1111-1118);
- **every** unmatched target gets a surrogate pair spec carrying its own index
  (lines 1120-1131);
- unmatched sources get specs keyed `None` and land in `cleanup_results`
  (lines 1133-1151).

Since `{targets of pairs} ⊎ unmatched_targets = range(T)` (disjoint by the
complement construction), and the dispatch loop writes exactly one
`results_by_target[target_index]` per spec or raises (lines 1156-1173), every key
used at line 1181 exists. `_dispatch_become` has no path that returns without
either a result or an exception; an exception aborts the whole `become` before
line 1180 is reached. The only inputs that could break this are ones that make
SciPy return duplicate indices, which cannot happen.

---

## Q4. `_expand_n_batch`: what gets expanded, and the two `parent_batch_sizes` branches

### Ways an attribute is skipped (lines 422-432)

```python
if not hasattr(self, attr):                       continue   # (a)
value = cast_to_tensor(getattr(self, attr))[0]
if value.shape[-2] == 1:                          continue   # (b)
if value.shape[-2] % points_per_object:           continue   # (c)
value_per_object = unsquish(value, -2, points_per_object)
if value_per_object.shape[-3] != current_batch_size: continue # (d)
```

**(a) declared but absent** — nothing exists to expand; correct, and stays
consistent (absent stays absent).

**(b) singleton row** — broadcast row; skipping is correct *provided* consumers
broadcast, and they do: `_expand_rows` explicitly expands 1-row values over any
count (`morph_conversions.py:75-76`), `reorder_batch_to_minimize_movement` skips
`shape[-2] == 1` identically (`mob_morph.py:514`), and renderer packing treats a
1-row attribute as shared. I found **no in-tree site where a 1-row animatable
attribute is consumed positionally against location rows**, so no failure point to
name. In-tree singletons include `border_width`/`portion_of_curve_drawn`
(`bezier_circuit.py:483`, registered at 446-449).

**(c) rows not divisible by `points_per_object`** — the attr lives at a different
granularity than object points (e.g. per-glyph colour rows on a packed Text whose
`num_points_per_object == 4`). Skipping avoids corrupting those rows with
point-level repeats, but leaves them describing the old member count while
location describes the new one. Inside `become` this is reconciled: the
end-of-morph wholesale copy `mine.set_non_recursive(**values)`
(lines 790-796) replaces every shared-name attribute with the target-sized value
on fresh post-detach rows. **Outside `become` (a bare `_expand_n_batch`) nothing
reconciles them** — every current caller is inside the become flow
(lines 671, 673, 715, 717, 983, 985), so today the exposure is latent, not live.

**(d) divisible but count mismatch** — stale/differently-packed data; same story
as (c).

Executed demonstration of the (c)-family consequence surviving to the endpoint
when names do not match: see Q5d — a Surface texture attr
(`color_texture_{W·H}`, `surface.py:1388-1398`) whose name differs on the two
sides is skipped by the values-dict intersection *and* never expanded, so the
result keeps the source's 16-texel texture while the geometry became the
target's.

### The `parent_batch_sizes` branches (lines 453-489)

Three paths: the singleton fast path (455-460), the all-single-object fast path
`index_select` (471-474), and the general `bincount` path (475-489).

**Do fast B and bincount ever disagree? Yes — on precisely the input class fast B
selects, and always (for `n > 0`).** When `objects_per_parent == 1` everywhere,
`parent_of_object = repeat_interleave(arange(P), ones) = arange(P)`, so the
bincount path computes `counts = histogram(repeat_indices)` — length stays `P`,
parents that received duplicates grow — while `index_select` duplicates parent
entries per slot — length becomes `target_batch_size`, uniform one-object entries.
Since `repeat_indices` always repeats something when `target > current`, the two
outputs differ in length and distribution on **every** expansion. Executed
(`verify_arithmetic.py`, replicating lines 462-489 literally):

```
pbs=[3,3,3], ppo=3, 3 objs -> 4:
  fast index_select -> [3, 3, 3, 3]  (len 4)
  bincount          -> [6, 3, 3]     (len 3)      equal? False
pbs=[7,7], ppo=7, 2 objs -> 5:
  fast index_select -> [7, 7, 7, 7, 7] (len 5)
  bincount          -> [21, 14]        (len 2)    equal? False
```

Both are internally sum-consistent (each sums to `target · points_per_object`),
but they describe **different member structures**: fast B says duplicated objects
become new members (`len(mob)` grows); bincount says existing members absorb them
(`len(mob)` unchanged). Which is right depends on packer semantics — fast B looks
deliberate for packs whose members are genuinely one logical object each (e.g.
`TriangleTriangulated` builds `parent_batch_sizes = full(N, 3)` with
`num_points_per_object == 3`, `shapes_2d.py:413,476`), and bincount for container-style
packers. But nothing states that intent, the branch choice is invisible at the
call site, and any consumer assuming one semantics gets the other's structure
whenever the data crosses the branch condition. Severity: medium — a latent
consistency trap rather than a demonstrated live bug.

---

## Q5. Does a completed morph actually equal rendering the target alone?

Short answer: **the PN route and the dissolve route do (their visible endpoint is
a clone of the target); the same-kind route does not.** The same-kind route
copies exactly the intersection of *timeline-backed* attributes and adopts some
plain structural metadata; everything else render-affecting about the target is
silently kept from the source.

### Same-kind route (`_record_same_kind_morph`, lines 721-803)

Copies (inside the Sync, lines 790-796):

```python
values = {attr: getattr(theirs, attr)
          for attr in mine.animatable_attrs
          if hasattr(mine, attr) and hasattr(theirs, attr)}
mine.set_non_recursive(**values)
```

i.e. `location, basis, scale_coefficient, color, opacity, glow` (Mob base,
`mob.py:202-212`) plus whatever both sides registered: `border_width`,
`portion_of_curve_drawn` (circuits, `bezier_circuit.py:446-449`), `normals`
(PNMesh, `pn_mesh.py:46`), shader-specific params *when both sides set shaders
with identical param names* (`mob_materials.py:109-119,188-193` registers them as
animatable attrs), `color_texture_{W·H}` when texel counts match
(`surface.py:1388-1398`).

Then `_adopt_structural_attrs(theirs)` (line 802):

- **Base Mob** (`mob.py:287-289`): adopts **nothing** — returns `self`.
- **TriangleVertices** (`shapes_2d.py:490-492`): adopts `normals` only.
- **TriangleMesh** (`three_d_models/mesh.py:266-276`): adopts `corner_index`,
  `corner_normals`, `corner_uvs`, `num_triangles`, `num_vertices` — **not**
  `texture_map`, `material_texture_map`, `normal_texture_map`,
  `material_texture_flags`, `ignore_normals`, `recompute_normals`.

Not copied anywhere in this route (all plain, non-timeline state):

| Property | Evidence | Endpoint effect vs rendering `b` alone |
| --- | --- | --- |
| `shader` | plain attr (`mob.py:258`), absent from `animatable_attrs` | result renders with **source's** material/pipeline |
| `shader_params` of a shader the source lacks | param attrs registered only on target | tween skipped (`hasattr(mine,…)` false) |
| `color_texture` with a different texel count | attr name embeds `W·H` (`surface.py:1388`) | **source's image persists at the endpoint** |
| `z_index` | plain `_z_index` + property (`bezier_circuit.py:347-394`) | coplanar draw order differs |
| `two_sided` / `closed_shell` | instance attrs computed in `__init__` (e.g. `shapes_3d.py:357-378,1328`, `1560-1564`) | shading sidedness / opacity compositing declared for the *source's* geometry |
| `render_tolerance`, `render_tolerance_pixels`, `geometry_slack_ratio` | plain attrs (`pn_mesh.py:50-66`) | tessellation accuracy of the result is the source's |
| `mesh_key`, `exclude_from_boundary`, `ignore_normals`, textures/material maps on meshes | plain attrs | seam merging, boundary inclusion, lighting response may differ |

`num_points_per_object` needs no copying: `morph_kind` includes it
(`mob.py:261-272`), so same-kind implies equal.

**All four executed confirmations** (probe scripts, CPU, no render):

- `Circle(z_index=2).become(Square(z_index=5))` → `result.z_index == 2.0`
  while `border_width` tweens to 9.0 and colour reaches the target's.
- Target given `set_fragment_shader(cosine_color)` → after morph
  `result.shader is None`; target's param attrs (`frequency`, `phase`) absent.
- `Surface(color_texture 4×4).become(Surface(color_texture 8×4))` →
  `result._color_texture_attr` still `color_texture_16`; result texture ≠ target's.
- `Sphere(u_range=(0, π)).become(Sphere())` (same type ⇒ same-kind route, result
  **is** the source object) → `result.closed_shell == False` while the target it
  now equals declares `True`; and the reverse morph ends declaring `closed_shell
  == True` for half-sphere geometry. Misdeclared closedness changes opacity
  compositing (shell coverage cap, `Mob.closed_shell` docs `mob.py:157-181`);
  misdeclared sidedness changes back-face shading.

Note the interesting split for grids: cross-*type* grid pairs are deliberately
forced through the PN route (`requires_grid_conversion`, lines 1238-1243) so
`Sphere→Torus` ends faithful; it is the **same-type, differently-configured** pair
that falls into the same-kind route and keeps stale declarations.

### Cross-family PN route (`_record_pn_morph`, lines 951-1052)

The visible endpoint is `replacement = target.clone(add_to_scene=False,
spawn=False)` (line 989), spawned only after the soup despawns (lines 1039-1044),
plus `post_animate(replacement, target)` restoring the target's border
(line 1051; hooks at `morph_conversions.py:390-405`). Because it is a clone of
the target, plain state (shader, tolerances, declarations, z-order) rides along.
Endpoint fidelity = clone fidelity. The intermediate soup copies only
`animatable_attrs` (lines 1033-1038) and harmonises the two tolerance families by
taking the min (lines 968-977) — irrelevant to the endpoint. Executed: the
Sphere→Torus probe returned a clone (`result is source == False`) carrying the
target's declarations.

### Dissolve route (`_record_dissolve`, lines 913-949)

Same construction: `replacement = target.clone(...)` (line 922), spawned, faded in
while the source fades out, then `_record_same_kind_morph(replacement, target, …)`
— where `replacement` already *is* the target's twin, so the values-copy is
target→target. Endpoint = clone of target ⇒ faithful. With
`replacement_allowed=False` the despawned *source* is returned for identity
compatibility (line 949) while the clone carries the visuals — documented
behaviour (`become` docstring, lines 1310-1313).

### Bottom line

Properties that make the final frame differ from rendering `b` directly exist
**only on the same-kind route**: `shader`/material, unmatched `shader_params`,
`color_texture` across resolutions, `z_index`, `two_sided`/`closed_shell`
(same-type different-sweep grids), `render_tolerance*`, mesh material maps,
`mesh_key`, `exclude_from_boundary`. Geometry, colour, opacity, glow, border and
basis *do* arrive correctly.

---

## Q6. Entitlement: placeholders, collapse points, zero-area rendering

### `_nearest_geometry_point` (lines 180-201)

It walks `source.get_descendants()` — which includes **self, components and
structural containers** (`mob_hierarchy.py:114-149`, default
`include_self=True`) — and considers every `.location` row. The candidate set is
therefore a *superset* of rendered points: it contains circuit frame anchors and
container locations that appear nowhere directly, and unlike the bounding-box
machinery it does **not** skip `exclude_from_boundary` children (circuits mark
their texture/border grids exactly that way, `bezier_circuit.py:491,500`). In
practice those extra rows sit on or inside the geometry (a circuit's anchor lies
on its own plane), so the chosen point is near-visible geometry; a structurally
far-away pick requires a container location parked away from its children. The
point only seeds where surplus-target geometry *grows from*
(`_record_primitive_hierarchy_morph` lines 1122-1131); a wrong pick misplaces the
birthplace of growing geometry for the duration of the growth, never the
endpoint. Cosmetic severity.

### `_collapse_hierarchy_at` without fading (lines 217-226)

Every descendant's location is written to the single anchor point
(`_setattr_and_rebatch_without_record`, instant), opacity untouched. What renders
is **nothing** — degenerate triangles are culled twice over in the sheet route:

```python
# raster_taichi.py:954  — float barycentric stage
if ti.abs(s) > 1e-30:            # s = perspective-weighted signed area
    ...
# raster_taichi.py:1095-1105 — exact lattice stage
if area2 == 0:
    # Foreshortened to zero area on the lattice. Its edge functions are all
    # zero ... Clearing the set hands it to the sample-less policy below,
    # which under the default drops it -- an error bounded by one lattice
    # unit (1/4096 px), not a hole.
    m = 0
```

and the ray-based fallback rejects the same geometry via the Möller–Trumbore
determinant guards (`|det| > 1e-12`, `raster_taichi.py:1422,1488`). Normals built
from collapsed corners go through `F.normalize(cross, dim=-1)`, which clamps its
denominator and yields the zero vector rather than NaN
(`morph_conversions.py:84-99`). So a collapsed-but-opaque surrogate/sink is
invisible by culling, not by transparency — which is why the code can afford not
to fade it (contrast `_collapsed_child_placeholder`, lines 259-262, which zeroes
opacity anyway, and `_zero_hierarchy_opacity` applied to image sinks only,
lines 1149-1151). Source-read from the kernels; not rendered in this audit.

---

## Q7. Timeline correctness

### Every `with` block in `mob_morph.py`

Complete list (grep, excluding prose): 731, 732, 740, 798 (`_record_same_kind_morph`);
923, 929, 930, 935, 944 (`_record_dissolve`); 1018, 1020, 1026, 1032, 1039, 1046
(`_record_pn_morph`); 1106, 1107, 1155, 1175 (`_record_primitive_hierarchy_morph`);
1340 (`become`). Each is an `AnimationContext` subclass used as a context manager
— entered **and exited**, satisfying CLAUDE.md's rule. All actual event recording
(animated sets/functions, spawns/despawns) happens inside those blocks; the
pre-block manipulation of clones (e.g. lines 989-997) operates on *unspawned*
subtrees, which the `animated_function` fast-path executes without touching any
context (`animatable.py:113-124`). No block records against the top-level context.

### Reading `unit = am.context.run_time_unit` off the current context

Line 1016 reads it **before** any become-owned context is entered, so `am.context`
is the user's ambient context. That is correct: contexts inherit `run_time_unit`
on entry (`animation_contexts.py:357-367`), so inside a user `Sync()`, `Lag()` or
`Seq()` the value equals what every other route implicitly consumes (their child
recordings advance by `run_time_unit` through `increment_times`,
`animation_contexts.py:668-682`, invoked by the animated-function wrapper,
`animatable.py:152`).

### Do the phases sum to exactly `unit`?

Yes, exactly, and the clamp is dead code at current constants:

```python
border_fraction = 0.3                       # line 1011
border_phases = int(source_has_border) + int(target_has_border)   # ∈ {0,1,2}
morph_fraction = 1.0 - border_fraction * border_phases            # ∈ {1.0, 0.7, 0.4}
if morph_fraction <= 0: morph_fraction = 0.4                      # needs phases >= 4: unreachable
```

Total = `0.3p + (1 − 0.3p) = 1.0` for every reachable `p`. Each phase is a
`Sync(run_time=X)` whose content records exactly one `run_time_unit`-long
animation (`set_non_recursive` → `set` opens its own Sync, `mob.py:1698-1705`), so
the phase's authored span is one unit and the exit rescale compresses it to X
(`animation_contexts.py:556-597`); the `Off` phases consume zero time.

### Routes agree — executed

Inside `Sync(run_time=2.5)`, cursor advance per route (CPU probes):

```
same-kind bezier (Circle->Square)           advance=2.500000
PN cross-family (TriangleVertices->Circle)  advance=2.500000
dissolve (ImageMob->ImageMob)               advance=2.500000
hierarchy (Group->Group)                    advance=2.500000
forced dissolve of whole root               advance=2.500000
```

Also `Lag(0.5, run_time=4)` → 4.0; `Seq(run_time=3, run_time_unit=0.5)` → 3.0.
The hypothesised 1.0-vs-1.6 discrepancy does not exist: every route books exactly
one `run_time` unit. The suite's own timing guards corroborate
(`test_pn_swap_uses_half_open_lifespans…` expects soup end at 0.7 with a
0.3-unit border phase; `test_mesh_to_circuit_swaps_borderless_then_grows_the_border`
expects 0.0→0.7 then border growth 0.7→1.0), and all 27 tests in
`tests/unit_tests/test_morph_become.py` pass in this environment
(`27 passed … in 5.22s`).

---

## Q8. Anything else plainly wrong

1. **Dead parameter that misleads callers — low.**
   `_expand_n_tensor(self, value, n, counterparts=None)` (lines 375-402) never
   references `counterparts`; both call sites pass it
   (`counterparts=their_path` / `counterpart=my_path`, lines 652-658) and it is
   silently ignored. The documented behaviour (collapse repeated segments at the
   *source segment's* end) is what the code does, but a reader could reasonably
   believe the counterpart selects the collapse point. Guard tests pin the real
   behaviour (`test_cubic_segment_padding_keeps_new_segments_on_the_source_contour`).

2. **O(S·T) device syncs in pairing — low (perf).**
   `_primitive_pair_cost` evaluates `_morph_center(source/target)` per matrix cell
   (lines 131-133); each call walks the whole subtree bounding hierarchy and ends
   in `.cpu()` (line 98). For large hierarchies this is S·T full bbox traversals
   plus S·T GPU→CPU syncs on the authoring thread. Correct, just quadratic and
   synchronising where a precomputed column/row vector would be neither.

3. **Unbounded secondary term under `minimize_movement` — info.**
   See Q2: crossing a compatibility band requires centres ≥ 1e6 world units
   apart. No clamp mirrors the default route's `min(distance, 1e3)`.

4. **`packed_count or 1` conflation — nit.**
   `_resample_surface_to` sets `surface.grid.batch_size = new_count *
   (packed_count or 1)` (line 568), treating "no parents" like "exactly one";
   harmless today (unreachable with an empty tensor), but `or` is doing type
   coercion, not arithmetic.

5. **Surrogates rely solely on geometric invisibility — info.**
   Fresh-target surrogates spawn at full opacity collapsed to a point
   (lines 1120-1131) with no opacity zeroing, unlike child-placeholders
   (lines 259-262) and image sinks (1149-1151). Safe given the culling evidence in
   Q6, but the defensive posture is inconsistent across three nearly identical
   constructions.

Nothing else rose to reportable: the dissolve-route many-to-one part maps
(lines 889-893, 902-906) leave no part unfitted on either side (both loops cover
their own sides completely), and `Text.become` (`text.py:386-430`) defers wholly
to the mixin and rebuilds glyph views only — it introduces no timing or recording
of its own.

---

## Findings by severity

| # | Severity | Finding | Where | Trigger |
| --- | --- | --- | --- | --- |
| 1 | **High** | Same-kind morph endpoint ≠ rendering the target: `shader`/material and unmatched shader params never adopted | `mob_morph.py:790-796` (intersection copy), `mob.py:258`, `mob_materials.py:109-119` | source unshaded (or differently-named pipeline) → target shaded; executed |
| 2 | **High** | Same-kind endpoint keeps source's `color_texture` when texel counts differ (attr name encodes `W·H`) | `surface.py:1388-1398`, `mob_morph.py:790-794` | `Surface(4×4 tex).become(Surface(8×4 tex))`; executed |
| 3 | **Medium-High** | Same-kind endpoint keeps source's `two_sided`/`closed_shell` instance declarations even as geometry becomes the target's | instance attrs set in `__init__` (`shapes_3d.py:378,1328`); not in `animatable_attrs`; base `_adopt_structural_attrs` no-op (`mob.py:287-289`) | `Sphere(u_range=(0,π)).become(Sphere())` and reverse; executed |
| 4 | **Medium** | Same-kind endpoint keeps source's `z_index` | `bezier_circuit.py:347-394` (plain attr) | `Circle(z_index=2).become(Square(z_index=5))`; executed |
| 5 | **Medium** | `_expand_n_batch`'s all-single-object `index_select` branch and the `bincount` branch produce structurally different `parent_batch_sizes` (member count vs member growth) for the same input | `mob_morph.py:471-489` | any expansion with `objects_per_parent == 1`; arithmetic executed |
| 6 | **Low-Med** | Skipped multi-row attributes (granularity/count mismatches) stay inconsistent until the end-of-morph wholesale copy; bare `_expand_n_batch` callers would persist the inconsistency | `mob_morph.py:422-432`, reconciliation at 790-796 | attr stored at non-point granularity; source-read + texture probe |
| 7 | **Low** | `_expand_n_tensor` ignores its `counterparts` argument | `mob_morph.py:375-402` vs call sites 652-658 | static read |
| 8 | **Low** | Pairing cost is O(S·T) subtree-bbox walks with a `.cpu()` sync per cell | `mob_morph.py:95-98,131-133,150-162` | large hierarchies; perf only |
| 9 | **Info** | `minimize_movement` distance term unbounded (band cross needs ≥1e6 units) | `mob_morph.py:134-135` | absurd coordinates |
| 10 | **Info** | Surrogate placeholders not opacity-zeroed; rely on zero-area culling (which the raster kernel provides, twice) | `mob_morph.py:1120-1131`; `raster_taichi.py:954,1095-1105,1422,1488` | kernel source read |

Positive results worth restating: **Q1** — no input drops a short-side part, and
placeholder slots face their own counterparts; **Q3** — SciPy's rectangular
contract plus the complement-partition construction makes the
`results_by_target[target_index]` lookup unreachable-KeyError; **Q7** — every
`with` block is entered-and-exited, the border-phase arithmetic sums to exactly
one `run_time_unit` on every reachable branch, and all five routes were measured
to book identical durations.

## Executed vs source-reading

**Executed** (scripts in `/tmp/opencode`, run with this repo's `.venv/bin/python`,
no files written into the repo, nothing rendered):
- Q1 coverage brute-force over `(m ≤ n < 64)` and the slot-pairing example.
- Q3 SciPy rectangular shapes and the partition property.
- Q4 literal replication of both `parent_batch_sizes` branch computations.
- Q5a/5b/5d/5e endpoint probes (z_index, fragment shader, colour texture,
  same-type Sphere sweep declarations), Q5c route detection for Sphere↔Torus.
- Q7 duration probes for five routes plus `Lag`/explicit-`run_time_unit` nesting.
- `pytest -q tests/unit_tests/test_morph_become.py` → **27 passed** (baseline
  sanity that the audited tree behaves as its guards claim).

**Source-reading only** (not executed here): the raster-kernel culling quotes and
normal-clamp behaviour (Q6), clone fidelity of the PN/dissolve endpoints (Q5 —
inferred from `clone()` usage plus the passing identity/lifespan tests), the
mid-morph reconciliation story for skipped attributes (Q4/6), the packed-centroid
layout-blindness beyond the two-group demo, and every severity judgement.

**Timing caveat:** the probes and the pytest run executed against the tree as of
commit `e46264d` (no tracked-file modifications existed at 10:00–10:12); the
concurrent session's uncommitted edits postdate them. The drift touches
`_collect_morph_primitives` traversal, `_register_hierarchy_for_render`, adds an
empty-hierarchy early path in `_record_primitive_hierarchy_morph`, and reworks
part of `_bezier_to_pn_soup` — none of the quoted lines this report's findings
rest on — but re-running the probes after that work lands is cheap if numbers
need refreshing.
