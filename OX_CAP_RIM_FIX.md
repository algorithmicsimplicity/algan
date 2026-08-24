# OX cap-rim fix report

Brief: `/tmp/ox_cap_fix.md`. Scope honoured: **`algan/mobs/shapes_3d.py` +
new test file only**; no renderer files, no `surface.py`, no `*_taichi.py`
touched; nothing committed; no baseline variable set; `tests/full_renders`
not run.

## What changed (`algan/mobs/shapes_3d.py`, 115 insertions / 8 deletions)

- `_CapDisc.__init__` (shapes_3d.py:265) no longer pins the rim to the body's
  count. When `grid_width` is not handed in explicitly it calls the new
  `_rimmed_grid_width` (shapes_3d.py:309): search upward over multipliers
  `k = 1, 2, 3, ...`, taking `grid_width = (segments - 1) * k + 1`
  (**vertex** count preserved: the rim closes, so `segments - 1` is the
  body's distinct chord count and the multiplier scales chords, not
  samples). Accept the first `k` whose chord polygon measures within
  `geometry_tolerance` of the true rim; if none fits inside
  `max_grid_resolution`, take the last affordable `k` — degraded but
  building, like every surface search; no raise, no warning. The existing
  `max(3, ...)` floor stays.
- Whole multiples are deliberate and commented: the body-to-cap joint has no
  welding mechanism beyond coincident samples, so every one of the body's
  ring vertices must remain exactly a rim vertex; a multiple adds only
  vertices strictly between them.
- `_rim_chord_deviation` (shapes_3d.py:334) measures generically off
  `rim_function`: sample at `i/m` for `i` in `0..m`, compare each chord
  midpoint against `rim_function((i + 0.5)/m)`; max distance. Exact sagitta
  for a circular rim, no radius assumed anywhere.
- **Tolerance propagation**: `Cone.__init__` (shapes_3d.py:634-635) and
  `Cylinder.add_bases` (shapes_3d.py:867-868) now pass the body's
  `geometry_tolerance` and `max_grid_resolution` into the disc. They stay in
  the kwargs so `Surface.__init__` stores them on the cap like on any
  surface.
- Docstrings rewritten as instructed: the class docstring passage that
  justified the old behaviour ("the search would answer two and leave a
  triangle") and the `segments` parameter entry now state interior-exact /
  rim-not / PN cannot curve a flat patch's boundary / rim sized here;
  `_pn_geometry_deviation` still returns zero but its docstring says true of
  the interior only, with a pointer to `_rimmed_grid_width`.

### Defaults sourcing — choice made

`Surface.__init__` runs *after* the rim must already be sized, so the cap
reads `geometry_tolerance` / `max_grid_resolution` from its own kwargs. When
absent, it falls back to **a module-level table lifted from
`Surface.__init__`'s signature itself** (`inspect.signature`,
`_SURFACE_INIT_DEFAULTS`, shapes_3d.py:198-206) rather than restating
`0.0005` / `200`. I chose this over named constants in `surface.py` because
the brief restricts edits to `shapes_3d.py`; deriving from the signature
keeps a single source of truth without touching that file. Cost: one
`inspect.signature` call at import.

## Expected outcome — confirmed by running

```
cyl body grid: 15 2        cyl caps:     71 71     # was 15
cone body:     4 22        cone base:    85        # was 22
line body:     24          caps:         24 24     # unchanged, as briefed
arrow shaft:   25          cap:          25        # unchanged
arrow head:    25          cap:          49        # MOVES (see below)
```

Measured deviations: cylinder cap 0.011282 → **0.000453** (≤ 5e-4);
cone base 0.006143 → **0.000385**. Both match the brief's predicted numbers
exactly (70 and 84 segments).

Shapes that move beyond the brief's explicit list, reported rather than
suppressed:

- **`Arrow3D`'s conical head** (`base_radius=0.08`, built at
  `resolution=(24,24)` → 24 rim chords): sagitta 6.85e-4 > tolerance, so its
  base disc goes 25 → 49 vertices (k=2). The shaft (thickness 0.02) does not
  move.
- **`Line3D(thickness=0.08)`** and the `thickness=0.2` variant used in an
  existing guard test: k=2. Only the *default* thickness (0.02, sagitta
  0.37× tolerance) is unchanged, as the brief stated.
- Plain `Cylinder()` (r=1.0): its auto-searched ring count is 18, whose
  chords sit 34× over tolerance, so its caps go 18 → 103. Any capped
  cylinder/cone whose inherited count leaves chords over tolerance refines;
  thin ones stay at k=1.

## Verification, verbatim

### `.venv/bin/python -m pytest -q tests/unit_tests/test_cap_disc_rim.py`

```
6 passed, 3 warnings in 1.47s
```

Covers, for both `Cylinder(show_ends=True)` and `Cone(show_base=True)`:
rim deviation measured from the mob's **built vertices**
(`grid.location`) against the analytic circle (not the implementation's
sampling); exact expected counts 71/85; weld (count divisibility + every
body ring vertex coincides with a rim vertex to 1e-5); tolerance propagation
(0.005 < default < 1e-4 widths, coarse cap meets its looser bound);
`max_grid_resolution=40` respected while keeping the whole-multiple weld;
default-thickness `Line3D` untouched. Not marked `fast` (fails only when its
own module changes).

### `.venv/bin/python -m pytest -q --fast`

```
FAILED tests/fast/test_fast_render.py::test_the_fast_scene_renders_and_matches_its_baseline
1 failed, 274 passed, 1698 deselected, 3 warnings in 22.22s
E       AssertionError: fast.mp4 differs from its baseline by up to 5 channel values (worst at frame 27); see /home/user/algan/tests/fast/output_errors/fast.mp4
```

**This failure pre-exists my change; it is not mine.** Evidence: the fast
scene contains no PN geometry (tests/fast/scene.py:32), so no `_CapDisc` is
ever constructed there; and empirically, with `HEAD`'s `shapes_3d.py`
temporarily swapped in (manual file swap — **no stash was used**, the tree's
only tracked modification is my file), the identical failure reproduces:
same 5 channel values, worst at frame 27. My file was then restored
byte-identically (md5 verified). Per CLAUDE.md these baselines are per-
machine; I did not investigate the root cause further — flagging it for the
human. Nothing re-baselined.

### `.venv/bin/python -m pytest -q tests/unit_tests`

```
9 failed, 1865 passed, 91 skipped, 160 warnings in 522.28s (0:08:42)
```

All 9 are caused by my change and all assert the **pre-fix contract** that
this fix intentionally supersedes. Verified on base: with HEAD's
`shapes_3d.py` swapped in, both affected files pass completely
(`62 passed`). Mechanisms:

- `test_normal_orientation.py::test_an_end_discs_rim_sits_on_the_bodys_own_ring`
  (cone, cone-tilted, cylinder, cylinder-tilted, line3d-rebased — 5 cases):
  asserts every *rim* vertex sits within 1e-3 of a *body-ring* vertex. That
  is the converse of the invariant the brief preserves (body ring ⊆ rim):
  refinement legitimately adds rim vertices between body samples, up to half
  a body chord away from the nearest one. The direction of the assertion has
  to flip under the new contract.
- `test_closed_shell_declaration.py::test_a_declared_closed_solid_really_is_one`
  (cylinder-capped, cone-capped, line3d [thickness 0.08 → moves]) and
  `test_the_compound_arrow_declares_per_part` (its head cone moves):
  `_forms_closed_shell` pairs boundary edges after a 2e-3 weld. The cap rim's
  finer polygon no longer edge-pairs 1:1 with the body's coarse ring chords —
  a T-joint whose mid-vertex sits off the body's straight chord by that
  chord's own sagitta (the quantity the fix removes from view).

Per the brief's scope ("In algan/mobs/shapes_3d.py only, plus the two tests
below") I did **not** edit these guard tests. They need a contract update by
someone authorised to touch them: (a) flip the rim test to body-ring ⊆ rim
with position coincidence — my new test file already encodes exactly that;
(b) teach the shell-closure checker that a refined rim meets the body's ring
chords in a T-joint bounded by construction tolerance. Note the *rendered*
joint gets tighter under the fix: previously the cap ended on the inscribed
polygon while the tube's diced silhouette reached the true surface (the
visible notch); now both hug the true curve within construction tolerance.

### ruff

```
.venv/bin/ruff check --no-fix algan/mobs/shapes_3d.py tests/unit_tests/test_cap_disc_rim.py
All checks passed!
.venv/bin/ruff format --check algan/mobs/shapes_3d.py tests/unit_tests/test_cap_disc_rim.py
2 files already formatted
```

## Triangle-count cost (measured on built primitives)

| shape | body tris | caps before | caps after | cap share before | cap share after | share @ tube dice L1 / L2 |
| --- | --- | --- | --- | --- | --- | --- |
| `Cylinder(r=0.45)` capped | 28 | 28 | 140 (×5.0) | 50% | 83% | 56% / 24% |
| `Cone(r=0.55)` capped | 126 | 21 | 84 (×4.0) | 14% | 40% | 14% / 4% |
| `Line3D(t=0.02)` default | 46 | 46 | 46 (×1.0) | 50% | 50% | — |
| `Line3D(t=0.08)` | 46 | 46 | 92 (×2.0) | 50% | 67% | 33% / 11% |
| `Arrow3D()` | 2304 | 72 | 96 (×1.33) | 3% | 4% | 1% / 0.3% |

"At tube dice level L" assumes every tube patch dices uniformly to 4^L
microtriangles while flat cap patches stay at level 0 forever (their criteria
return zero) — illustrative, not measured from a render. Absolute cost is a
few hundred triangles per solid; the share shrinks quickly as the tube dices.

## Rendered output

I expect rendered output to move for every scene containing a capped
solid — the geometry genuinely changes (finer rim polygons hugging the true
circle). Among committed suites that is `tests/full_renders/scenes/
solids_and_camera.py` (the `show_ends` cylinder and `show_base` cone at
lines 64/67 — the defect's origin scene). The other five full-render scenes
contain no capped cylinders/cones/lines/arrows (complex_hierarchy_become's
two cylinders are uncapped), so they should stay byte-identical. Both the
CPU and CUDA baseline sets will need regeneration on their respective
machines; that is deliberately left to you, per the brief. I ran no render
and set no `ALGAN_UPDATE_*` variable.

## What I did not verify

- No render of any kind was executed; the pixel-level effect of the fix is
  inferred from geometry, not seen.
- CUDA behaviour is entirely unverified (CPU-only cloud session).
- The pre-existing `--fast` failure reproduces identically without my change,
  but its root cause (machine variance vs. stale baseline) is undiagnosed.
- The 9 unit-test failures are analysed above, not fixed — editing those
  guard tests was outside my brief. Until their contracts are updated,
  `pytest -q tests/unit_tests` stays red on this branch by design of the fix.
- Long-run cost on pathological inputs (very large radii hitting the grid
  cap) degrades silently by design; I did not measure how common that is.

## Guard-test contract update

Brief: `/tmp/ox_cap_guards.md` (authorised). Scope honoured: only
`tests/unit_tests/test_normal_orientation.py`,
`tests/unit_tests/test_closed_shell_declaration.py` and one sentence of
`_CapDisc`'s class docstring changed; the fix itself untouched; nothing
committed; no `ALGAN_UPDATE_*` variable set; `tests/full_renders` not run;
`--fast` not run.

### 1. `test_an_end_discs_rim_sits_on_the_bodys_own_ring`

- Assertion direction flipped as specified: the invariant is now **body ring
  ⊆ cap rim** — every vertex of the ring the disc closes sits within 1e-3 of
  a rim vertex — plus the whole-multiple check
  `(rim_count - 1) % (ring_count - 1) == 0`. The old rim ⊆ ring direction
  would fail on legitimate refinement (new rim vertices sit up to half a
  body chord from the nearest body sample); a phase-offset independently
  sampled circle fails both new assertions.
- Ring selection no longer reads `body.grid.location.reshape(-1, 3)` (the
  whole grid). Each disc is paired with the row that disc actually closes,
  via `_built_grid` + row indexing **imported from `test_cap_disc_rim.py`
  itself** (`_built_grid`, `_rim_points`) so there is one definition of that
  geometry selection across both files, not a second one. `_RIMS` entries
  now carry `rings_of(body, grid)` returning `(disc, ring_row)` pairs:
  cylinder rows `g[:, 0]` / `g[:, -1]`, cone row `g[0]`.
- Docstring rewritten: states the new direction and that refinement adds
  rim vertices strictly between the body's samples.
- All 5 parametrizations pass, including the tilted and rebased ones.

### 2. `_forms_closed_shell`

Generalised exactly along the specified lines; **`_JOINT_TOL` (1e-3) and
`_JOINT_WELD` (2e-3) are unchanged**:

- Pass 1 (strict near-coincident opposite pairing) is behaviourally the
  original greedy matcher; the flat-faced polyhedra resolve here, untouched.
- New pass 2 consumes a leftover coarse edge with an opposing chain of
  boundary edges: breadth-first over edges whose start coincides with the
  previous edge's end, anchored within `_JOINT_TOL` at BOTH endpoints of
  the coarse edge, every chain edge required to oppose the coarse edge's
  direction (strictly negative dot product), chains kept acyclic by a
  visited set, and each edge consumed at most once (consumption marks are
  shared across passes). Adjacency is checked on float positions between
  welded representatives, so a seam vertex straddling weld cells cannot
  break a legitimate chain.

### Proof the relaxation cannot be fooled

Every verdict below printed explicitly by running all three groups through
`_forms_closed_shell` (probe run once for the record; the same assertions run
permanently in the suite):

```
== _CLOSED entries (expect closed=True) ==
OK  cone-capped                -> closed=True  declares=True
OK  convex-hull                -> closed=True  declares=True
OK  cube                       -> closed=True  declares=True
OK  cylinder-capped            -> closed=True  declares=True
OK  dodecahedron               -> closed=True  declares=True
OK  dot3d                      -> closed=True  declares=True
OK  icosahedron                -> closed=True  declares=True
OK  line3d                     -> closed=True  declares=True
OK  octahedron                 -> closed=True  declares=True
OK  prism                      -> closed=True  declares=True
OK  sphere                     -> closed=True  declares=True
OK  tetrahedron                -> closed=True  declares=True
OK  torus                      -> closed=True  declares=True
== _OPEN entries (expect closed=False) ==
OK  cone-capped-partial        -> closed=False  declares=False
OK  cone-open                  -> closed=False  declares=False
OK  cylinder-open              -> closed=False  declares=False
OK  halfpipe-with-discs        -> closed=False  declares=False
OK  polyhedron-single-quad     -> closed=False  declares=False
OK  sphere-partial             -> closed=False  declares=False
OK  torus-partial              -> closed=False  declares=False
== chain foils (expect closed=False) ==
OK  cap-concentric-but-wrong   -> closed=False
OK  cap-detached               -> closed=False
OK  cap-phase-shifted          -> closed=False
```

All 13 closed entries still close (polyhedra through exact pairing), all 7
open entries still report open, and no negative case passed when it should
fail — nothing to stop and report. The three NEW negative cases live in the
file permanently as `_CHAIN_FOILS` behind
`test_the_chain_rule_cannot_be_fooled` (faults injected after construction,
so only the checker is asserted, not the declaration):

1. **cap-detached** — `remove_child(top_cap)`: a real hole where a chain
   could plausibly be hunted for.
2. **cap-concentric-but-wrong** — top cap scaled to 0.85× (rim radius
   measured 0.2975 vs body 0.35): the concentric-wrong-loop trap.
3. **cap-phase-shifted** — my choice for "most likely to admit": the top cap
   rotated half a segment about the shared axis. Same radius, same plane,
   same winding as the ring it should close — everything about it is right
   except sampling phase, so endpoint anchoring is the ONLY property that
   rejects it. It is the historical scallop fault wearing a perfect
   disguise; measured nearest rim-to-ring distance 1.69e-2 (≈17× tolerance).

### 3. `_CapDisc` docstring

"the two land on each other vertex for vertex and the joint is watertight"
now reads "every one of the body's ring vertices is a rim vertex",
consistent with the whole-multiple paragraph below it.

### Disclosures

- One extra touch inside a scoped file: a clause of
  `test_closed_shell_declaration.py`'s MODULE docstring (it still described
  closure as strictly edge-for-edge) now mentions the anchored-chain rule
  and points at the foils. Section 2 mandated only the function docstring;
  leaving the module docstring stale felt like re-creating the entitlement
  this exercise removes.
- `ruff check --select I001 --fix` was applied to
  `test_normal_orientation.py` to seat the new cross-module import in its
  correct section (import placement only, verified in the diff).

### Verification, verbatim

`.venv/bin/python -m pytest -q tests/unit_tests/test_normal_orientation.py tests/unit_tests/test_closed_shell_declaration.py tests/unit_tests/test_cap_disc_rim.py`

```
71 passed, 3 warnings in 9.96s
```

(re-run after the import fix: `71 passed, 3 warnings in 9.97s`)

`.venv/bin/python -m pytest -q tests/unit_tests`

```
1877 passed, 91 skipped, 160 warnings in 373.86s (0:06:13)
```

**Green with zero survivors** — there is no failure to name or defend. The
9 contract failures recorded in the previous section are resolved (1865 + 9
+ 3 new foil tests = 1877).

`.venv/bin/ruff check --no-fix` / `.venv/bin/ruff format --check` on the
three files:

```
All checks passed!
3 files already formatted
```

Not done, per the brief: `tests/full_renders`, any `ALGAN_UPDATE_*_BASELINE`
variable, `--fast`.

### What I did not verify

- No render of any kind; these are pure tensor assertions, but the
  pixel-level consequence of the refined rims remains inferred (and the
  baselines question stands as documented above).
- CUDA: entirely unverified (CPU-only session).
- The chain rule has a theoretical blind spot I could not close without
  inventing a new tolerance (forbidden): a self-avoiding opposing chain
  could wander away from the coarse edge's straight segment between its two
  anchored endpoints. Per-edge opposition, anchoring and single consumption
  are the only constraints, by design; nothing in the geometry pipeline
  emits such edges and no test exercises them, so the exposure is
  adversarial-mesh-only.
- The whole-multiple assertion relies on the closed-ring seam convention
  (last sample repeats the first), the same reliance `test_cap_disc_rim.py`
  already makes; it is not independently asserted.
- `test_cap_disc_rim.py` was deliberately left unmodified; it passes as-is.

## Chain-rule length bound

Brief: `/tmp/ox_chain_defect.md`. Scope honoured: **only
`tests/unit_tests/test_closed_shell_declaration.py`** changed this round;
the fix (`algan/mobs/shapes_3d.py`), the other test files, and everything
under `algan/` untouched; nothing committed or pushed; no `ALGAN_UPDATE_*`
variable set; `tests/full_renders` and `--fast` not run.

### The defect, reproduced before fixing

The previous disclosure's "adversarial-mesh-only" claim was wrong, as the
brief said. All three shapes built through the public API only:

```
single-triangle      closed=True declares=False     # must be False
single-quad          closed=False declares=False    # correct
tetra-minus-face     closed=True declares=False     # must be False
```

Both wrong verdicts come from the checker: both mobs already declare
`closed_shell=False`, so declaration and geometry agreed and only the chain
consumption lied.

### What changed in `_forms_closed_shell`

- New module constant `_CHAIN_LENGTH_FACTOR = 1.25`, its comment carrying
  the derivation: a refinement's arc/chord is `(theta/2)/sin(theta/2)` --
  ~1.01 on these shapes' rings (measured below), 1.209 for the worst
  constructible 3-segment ring -- while the cheapest hole-detour, two edges
  of a triangle closing the third, measures 2.000 and wandering only adds.
  The bound is scale-free and deliberately not a distance-to-segment test:
  the chain's interior legitimately bows off the chord by up to the ring's
  own sagitta (~1e-2), far more than `_JOINT_TOL`.
- `chain_from` accumulates each candidate path's length and abandons it the
  moment the running total exceeds `1.25 *` the coarse edge's length --
  pruning during the search rather than only on completed paths, which also
  bounds a BFS that previously had nothing stopping it exploring the whole
  boundary.
- `_JOINT_TOL` (1e-3) and `_JOINT_WELD` (2e-3) unchanged; endpoint anchoring
  and strict per-edge opposition untouched -- an additional constraint, not
  a replacement for either.
- Docstrings updated (module, `_forms_closed_shell`, `chain_from`) to state
  the refinement-vs-detour invariant.

### Placement of the two new cases: both in `_OPEN`

Each mob's own `closed_shell` declaration says open (`declares=False` for
both, verified), and `_OPEN` is exactly the group that asserts declaration
and geometry agree. `_CHAIN_FOILS` exists for faults injected AFTER
construction, where no mob can be asked to declare them; these are plain
public-API constructions, not injected faults. Named
`polyhedron-single-triangle` and `polyhedron-tetra-minus-face`, with
comments -- the latter reads as the ordinary hole it is, not an adversarial
construction. The `_OPEN` test's no-descent condition generalised from a
name-string comparison to `not isinstance(mob, Polyhedron)` so a third
polyhedron entry would not need another special case.

### Verdict table, re-run (includes the new entries)

Every line printed by running all three groups plus the new cases through
the real `_forms_closed_shell`:

```
== _CLOSED entries (expect closed=True) ==
OK  cone-capped                -> closed=True  declares=True
OK  convex-hull                -> closed=True  declares=True
OK  cube                       -> closed=True  declares=True
OK  cylinder-capped            -> closed=True  declares=True
OK  dodecahedron               -> closed=True  declares=True
OK  dot3d                      -> closed=True  declares=True
OK  icosahedron                -> closed=True  declares=True
OK  line3d                     -> closed=True  declares=True
OK  octahedron                 -> closed=True  declares=True
OK  prism                      -> closed=True  declares=True
OK  sphere                     -> closed=True  declares=True
OK  tetrahedron                -> closed=True  declares=True
OK  torus                      -> closed=True  declares=True
== _OPEN entries (expect closed=False) ==
OK  cone-capped-partial        -> closed=False declares=False
OK  cone-open                  -> closed=False declares=False
OK  cylinder-open              -> closed=False declares=False
OK  halfpipe-with-discs        -> closed=False declares=False
OK  polyhedron-single-quad     -> closed=False declares=False
OK  polyhedron-single-triangle -> closed=False declares=False
OK  polyhedron-tetra-minus-face -> closed=False declares=False
OK  sphere-partial             -> closed=False declares=False
OK  torus-partial              -> closed=False declares=False
== chain foils (expect closed=False) ==
OK  cap-concentric-but-wrong   -> closed=False
OK  cap-detached               -> closed=False
OK  cap-phase-shifted          -> closed=False
```

All 13 `_CLOSED` entries closed, all 9 `_OPEN` entries open (the original 7
plus the 2 new), all 3 chain foils open.

### Measured chain ratios -- the bound has real margin

Instrumented mirror of the checker recording path/coarse length at every
consumption:

```
cylinder-capped: verdict=True, chains consumed=26
  distinct coarse-edge lengths: [0.167521]
  ratio min=1.009407 max=1.009407 n_edges per chain=[5]
cone-capped: verdict=True, chains consumed=19
  distinct coarse-edge lengths: [0.148135]
  ratio min=1.004285 max=1.004285 n_edges per chain=[4]
```

So the actual legitimate joints measure **1.009407** (cylinder) and
**1.004285** (cone): ~24% relative headroom under the 1.25 bound, ~3%
headroom over the worst constructible legitimate joint (1.209, analytic),
and the cheapest fault the bound must reject (2.000, now permanently in
`_OPEN`) overshoots it by 60%.

### Verification, verbatim

`.venv/bin/python -m pytest -q tests/unit_tests/test_closed_shell_declaration.py tests/unit_tests/test_normal_orientation.py tests/unit_tests/test_cap_disc_rim.py`

```
73 passed, 3 warnings in 9.19s
```

(71 before + the two new `_OPEN` parametrizations.)

`.venv/bin/python -m pytest -q tests/unit_tests`

```
1879 passed, 91 skipped, 160 warnings in 371.65s (0:06:11)
```

Green with zero survivors (1877 + 2 new).

ruff on the file:

```
$ .venv/bin/ruff check --no-fix tests/unit_tests/test_closed_shell_declaration.py
All checks passed!
$ .venv/bin/ruff format --check tests/unit_tests/test_closed_shell_declaration.py
1 file already formatted
```

Not done, per the brief: `tests/full_renders`, `--fast`, any
`ALGAN_UPDATE_*_BASELINE` variable, any commit or push.

### Hunt: the next case of the same kind

Found two -- reported, not fixed, per the brief; the rule was not widened.
Both reachable through the public API (`Polyhedron` takes arbitrary
vertices/faces), and both pass anchoring + opposition + the 1.25 bound while
leaving a genuine hole:

| probe | verdict |
| --- | --- |
| lone isoceles sliver triangle, base 1, apex height h = 0.05 / 0.30 / 0.3749 / 0.3750 | **closed=True (foiled)** |
| same triangle, h = 0.38 / 0.60 | caught |
| shallow 3-leg zigzag closing the unit chord -- legs (-0.3, 0.1), (-0.3, -0.15), (-0.4, 0.05); enclosed area 0.0125 | **closed=True (foiled)** |

Mechanism: an isoceles triangle's two-edge detour measures
`sqrt(1 + (2h/c)^2)` chord-lengths, which drops under 1.25 once h < 0.375c
(measured escape up to h = 0.3750, caught at 0.38); a zigzag whose every leg
has negative projection on the coarse direction sums to as little above the
chord as its slopes allow. Both families approach ratio 1.0 in the limit, so
**no length factor can separate them from a refinement arc** -- the metric
invariant is exhausted, not mis-tuned. What would close them is topological
rather than per-edge-metric: assert closure of each connected component of
the whole consumed shell (Euler characteristic / even edge degree), which
fails for any holed consumption regardless of geometry (a lone triangle is
V-E+F = 1, not 2). Left unimplemented deliberately: out of this brief's
scope, and it changes what the checker asserts rather than tightening what
it already asserts. Built-in mobs are unaffected by both families -- every
partial-sweep/cut/halfpipe entry stays open in the table above, none of them
emits sliver boundary loops.

### Disclosures

- Probes lived in `/tmp/opencode`; nothing probe-related written into the
  repo.
- One early measurement ran through the warm render daemon and came back
  stale: the daemon had imported this test module before the `_OPEN`
  entries were added, so its verdict table printed 9 rows as 7. Every number
  reported here was re-taken with `ALGAN_USE_DAEMON=0` fresh processes; the
  pytest runs spawn their own processes and were never affected.
- The suite being green does NOT mean the checker is hole-proof -- see the
  hunt section for the two shapes that still slip through, documented rather
  than silently widened away.

### What I did not verify

- No render of any kind (per brief); all assertions are pure tensor checks,
  so no rendered-output question arises and no baseline could move.
- CUDA: entirely unverified (CPU-only session) -- moot for this change, as
  no `algan/` code was touched.
- Whether real-world meshes imported through `ThreeDModelMob` contain sliver
  boundary loops that would hit the residual exposure -- untested; the guard
  covers built-ins and explicit public-API constructions only.

## Chain-rule primitive constraint

Brief: `/tmp/ox_chain_primitive.md`. Scope honoured: **only
`tests/unit_tests/test_closed_shell_declaration.py`** changed this round;
`algan/`, the other test files, and `OX_CAP_RIM_FIX.md`'s earlier sections
untouched by it; nothing committed or pushed; no `ALGAN_UPDATE_*` variable
set; `tests/full_renders` and `--fast` not run. All ad-hoc probes ran with
`ALGAN_USE_DAEMON=0` in fresh processes.

### The rule

A chain may only consume boundary edges that come from a **different render
primitive** than the coarse edge it closes. A T-joint is two different parts
of one solid meeting (a cap's rim against its body's ring); a hole is one
part failing to close, and one part's boundary closing over itself -- a lone
triangle, a missing face, a shallow sliver or zigzag -- is now refused
outright, at any shape or scale, with no tolerance involved.

- `_triangle_prims`'s return shape is **unchanged** (flat list of primitive
  objects) -- its existing callers concatenate results, so they needed no
  update. `_forms_closed_shell` derives a per-triangle source index from the
  list position of each primitive (`per_tri_source`), carries it through the
  degenerate-triangle filter alongside each triangle's welded keys, and
  records it per directed edge (`source_of_edge`, first contributor wins for
  an edge key fed by more than one primitive -- pass 1 pairs without looking
  at primitives anyway, so only chain candidates read this).
- The constraint sits inside `chain_from`'s candidate filter next to
  `opposes` and the anchoring test (both seed loop and expansion loop).
  Endpoint anchoring, per-edge opposition, single consumption, and
  `_CHAIN_LENGTH_FACTOR` are all kept.
- Docstrings rewritten as instructed: `_CHAIN_LENGTH_FACTOR`'s comment no
  longer claims the metric separates a refinement from a detour -- it states
  the factor is a search bound (a shallow hole's detour tends to arc/chord
  1.0 exactly as a refinement arc does), and that the primitive constraint
  is the separator. Same correction in `_forms_closed_shell`,
  `chain_from`, and the module docstring.
- `_JOINT_TOL` (1e-3) and `_JOINT_WELD` (2e-3) unchanged.

### Placement of the new permanent cases: `_OPEN`

The sliver family and the zigzag are public-API `Polyhedron` constructions
whose own declarations say open (`closed_shell=False`: one face, so
`orient_faces_outward` returns them unchanged), so they sit in `_OPEN`
beside last round's two, same reasoning as before. Four entries, named for
the family: `polyhedron-shallow-sliver-h005/-h030/-h03749` and
`polyhedron-shallow-zigzag`. Two comment corrections ride along:
`polyhedron-single-triangle`'s note now says the primitive constraint
refuses it outright (the length bound was only ever the weaker guard), and
`polyhedron-tetra-minus-face`'s note documents something I had wrong -- a
Polyhedron emits ONE PRIMITIVE PER FACE TRIANGLE, so its three hole edges
come from three different primitives and chain across them legally; the
retained length bound (~2.0 ratio) is what refuses that entry, not the new
constraint. Last round's closing suggestion (Euler characteristic over
connected components) was NOT what got implemented -- the primitive identity
test is local instead of whole-shell, needs no component enumeration, and
catches the same families.

### Verdict table, re-run

Every line printed by running all three groups plus an ad-hoc sweep through
the real `_forms_closed_shell` (a recording mirror asserted agreement with
it on every verdict):

```
== _CLOSED entries (expect closed=True) ==
OK  cone-capped                  -> closed=True  declares=True
OK  convex-hull                  -> closed=True  declares=True
OK  cube                         -> closed=True  declares=True
OK  cylinder-capped              -> closed=True  declares=True
OK  dodecahedron                 -> closed=True  declares=True
OK  dot3d                        -> closed=True  declares=True
OK  icosahedron                  -> closed=True  declares=True
OK  line3d                       -> closed=True  declares=True
OK  octahedron                   -> closed=True  declares=True
OK  prism                        -> closed=True  declares=True
OK  sphere                       -> closed=True  declares=True
OK  tetrahedron                  -> closed=True  declares=True
OK  torus                        -> closed=True  declares=True
== _OPEN entries (expect closed=False) ==
OK  cone-capped-partial              -> closed=False  declares=False
OK  cone-open                        -> closed=False  declares=False
OK  cylinder-open                    -> closed=False  declares=False
OK  halfpipe-with-discs              -> closed=False  declares=False
OK  polyhedron-shallow-sliver-h005   -> closed=False  declares=False
OK  polyhedron-shallow-sliver-h030   -> closed=False  declares=False
OK  polyhedron-shallow-sliver-h03749 -> closed=False  declares=False
OK  polyhedron-shallow-zigzag        -> closed=False  declares=False
OK  polyhedron-single-quad           -> closed=False  declares=False
OK  polyhedron-single-triangle       -> closed=False  declares=False
OK  polyhedron-tetra-minus-face      -> closed=False  declares=False
OK  sphere-partial                   -> closed=False  declares=False
OK  torus-partial                    -> closed=False  declares=False
== chain foils (expect closed=False) ==
OK  cap-concentric-but-wrong         -> closed=False
OK  cap-detached                     -> closed=False
OK  cap-phase-shifted                -> closed=False
```

All 13 `_CLOSED` closed, all 9 pre-existing `_OPEN` open, all 3 chain foils
open -- no regression -- plus the 4 new shallow-hole entries open.

The two foil families the length bound could not catch, at the heights
measured last round (ad-hoc sweep, including the old escape point and the
old caught points for continuity):

```
sliver h=0.05    closed=False  declares=False  (detour/chord=1.0050)
sliver h=0.3     closed=False  declares=False  (detour/chord=1.1662)
sliver h=0.3749  closed=False  declares=False  (detour/chord=1.2499)
sliver h=0.375   closed=False  declares=False  (detour/chord=1.2500)
sliver h=0.38    closed=False  declares=False  (detour/chord=1.2560)
sliver h=0.6     closed=False  declares=False  (detour/chord=1.5620)
zigzag 3-leg      closed=False  declares=False  (detour/chord=1.0548)
```

All open now -- including h = 0.3750 and below, which the length bound
let through.

### Legitimate joints still consume

Recorded-mirror measurement, same instrument as last round:

```
cylinder-capped: verdict=True, chains consumed=26
  distinct coarse-edge lengths: [0.167521]
  ratio min=1.009407 max=1.009407 n_edges per chain=[5]
  every chain crosses primitives: True; no chain reuses its own coarse primitive: True
cone-capped: verdict=True, chains consumed=19
  distinct coarse-edge lengths: [0.148135]
  ratio min=1.004285 max=1.004285 n_edges per chain=[4]
  every chain crosses primitives: True; no chain reuses its own coarse primitive: True
```

Chain counts and ratios are IDENTICAL to the pre-constraint measurements --
as they must be: body ring chords come from the tube's primitive and rim
fan edges from the caps', so the legitimate joints never touched the new
filter. That is the "does not turn a correct closed verdict into open"
half, shown directly on the shapes whose joints consume chains.

### Verification, verbatim

`.venv/bin/python -m pytest -q tests/unit_tests/test_closed_shell_declaration.py tests/unit_tests/test_normal_orientation.py tests/unit_tests/test_cap_disc_rim.py`

```
77 passed, 3 warnings in 9.63s
```

(73 before + the four new `_OPEN` parametrizations.)

`.venv/bin/python -m pytest -q tests/unit_tests`

```
1883 passed, 91 skipped, 160 warnings in 371.43s (0:06:11)
sys:1: ResourceWarning: unclosed file <_io.TextIOWrapper name=11 mode='w' encoding='utf-8'>
```

Green with zero survivors (1879 + 4 new). The ResourceWarning is pytest's
own teardown noise, present in prior rounds too.

ruff on the file:

```
$ .venv/bin/ruff check --no-fix tests/unit_tests/test_closed_shell_declaration.py
All checks passed!
$ .venv/bin/ruff format --check tests/unit_tests/test_closed_shell_declaration.py
1 file already formatted
```

Not done, per the brief: `tests/full_renders`, `--fast`, any
`ALGAN_UPDATE_*_BASELINE` variable, any commit or push.

### Hunt: cross-primitive consumption still reachable?

Short answer: consumption yes, false closure no -- nothing found that flips
a verdict, and nothing was widened. Every probe below ran through BOTH the
new checker and the previous one (length bound only), extracted verbatim;
verdicts agreed everywhere, so nothing below is newly created or newly
fixed by this round except the sliver family already reported above.

| probe | new / old | reading |
| --- | --- | --- |
| full box / box minus top (hand-wound controls) | True / True; False / False | constructions sane |
| open-top box stacked on open-bottom box, interfaces coincident | True / True | correct: pure pass-1 seal (0 chains), union genuinely watertight |
| smaller sleeve plugged into the opening | False / False | annular gap keeps boundary |
| flush chimney (sleeve open above) on the opening | False / False | interface seals, chimney mouth stays open |
| bare narrow strip cylinder (`v_range=(pi-0.05, pi+0.05)`) | False / False | meridian cuts cannot be closed by the perpendicular arc segments (opposition dot = 0) |
| same strip with end discs | False / False | disc rims are huge leftover loops regardless |
| sphere narrow equatorial band | False / False | parallel latitude loops never anchor on each other |
| `Arrow3D` whole tree (shaft+head+caps together) | True / True | correct; chains 72 = shaft 48 + head 24 exactly -- parts consume independently |
| coarse plate + fine-rim plate, deliberately interlocked T-frame | True / True | see below |
| control: plate + exact reversed copy (no midpoints) | True / True | pure pass-1, zero chains -- glued shells were always "closed" under strict pairing |
| two coincident identical open tubes | False / False | same-direction duplicates neither pair nor oppose |

The T-frame deserves its paragraph because it is the brief's exact question
-- one primitive's hole consumed by another primitive's edges. It exists:
four cross-primitive chains consumed (mirror-verified). But reaching
`closed=True` required the two plates' OUTER rims to coincide exactly and
pair off too -- and the control shows that complete-coincidence closure is
precisely what pass-1 strict pairing has accepted since before chains
existed (two glued shells read as one closed complex, topologically sound:
every directed edge met by exactly one opposite, here a degenerate embedded
torus). The constraint changed none of that: with ANY residual unmatched
edge -- plug, chimney, partial frame, offset rims -- the verdict stays
False. So the answer to "can a primitive's boundary hole be closed by a
different primitive's edges?" is "consumed, yes, but the union then reads
closed only when every edge of both mobs is mutually coincident-and-opposed,
which is the checker's long-standing definition of closure, not a chain-rule
leak". Nothing built-in approaches these configurations; they take
hand-authored coincident `Polyhedron` frames.

### Disclosures

- Probes lived in `/tmp/opencode`; nothing probe-related written into the
  repo. `git show HEAD:...` was useless as the A/B baseline (HEAD predates
  all rounds -- nothing was ever committed), so the old arm was transcribed
  verbatim from this file's pre-edit state into `/tmp/opencode`.
- Two probe-side bugs cost a rerun each and are worth recording: my first
  T-frame fine plate carried the same winding as the coarse one (shoelace-
  checked), and the first "reversed" control reversed BOTH the vertex lists
  and the face indices (cancelling out). Neither touched repo code.
- The suite being green does not mean the checker is hole-proof in the
  adversarial limit -- the T-frame row above is the honest boundary of what
  edge-pairing semantics can express.

### What I did not verify

- No render of any kind (per brief); all assertions are pure tensor checks,
  so no rendered-output question arises and no baseline could move.
- CUDA: entirely unverified (CPU-only session) -- moot, as no `algan/` code
  was touched this round.
- Whether imported real-world meshes (`ThreeDModelMob`) emit one primitive
  per surface or several, which would decide how the constraint treats
  their internal seams -- untested; those mobs were outside every round's
  scope.

## Chain-rule mob grouping

Brief: `/tmp/ox_chain_mob.md`. Scope honoured: **only
`tests/unit_tests/test_closed_shell_declaration.py`** changed this round;
`algan/`, the other test files, and this file's earlier sections untouched by
it; nothing committed or pushed; no `ALGAN_UPDATE_*` variable set;
`tests/full_renders` and `--fast` not run. All ad-hoc probes ran with
`ALGAN_USE_DAEMON=0` in fresh processes and live in `/tmp/opencode`.

### The rule

A chain may only consume boundary edges **emitted by a different MOB** than
the coarse edge it closes -- one level up from last round's per-primitive
constraint, which a `Polyhedron`'s one-primitive-per-face-triangle build
walked straight past. That is what "two parts of the solid meeting" means:
a cap disc and its body are different mobs, so the legitimate joint still
crosses unchanged; every face of a `Polyhedron` comes from that one mob, so
no `Polyhedron` boundary can ever close over itself.

- `_triangle_prims` now returns ``(primitive, emitting mob)`` pairs; the
  emitter is in hand where the primitive is collected and rides alongside it.
  The stable identity used throughout is ``id(mob)`` -- every emitter stays
  alive for the whole call (the Scene holds the tree), so the ids are stable.
- `per_tri_source` carries ``id(emitter)`` per triangle row,
  `source_of_edge` maps each directed edge key to its first-contributing
  MOB (unchanged first-wins rule for coincident seams), and `chain_from`'s
  candidate filter (seed loop and expansion loop) compares against
  ``home_mob``. Endpoint anchoring, per-edge opposition, single consumption,
  and `_CHAIN_LENGTH_FACTOR` as the BFS search bound are all kept.
- `_CHAIN_LENGTH_FACTOR`'s comment now says plainly what it is: a search
  bound, full stop. It no longer even backstops `tetra-minus-face`, so it
  has no correctness role at all -- not even as backstop.
- Docstrings de-primitive-ified: module docstring (which now also states why
  grouping sits at the mob -- a Polyhedron emits ONE PRIMITIVE PER FACE
  TRIANGLE, so primitive identity would still let a face's hole borrow its
  neighbours' edges), `_forms_closed_shell`, `chain_from`, and the
  `polyhedron-tetra-minus-face` comment, which now reads "refused outright
  by the mob grouping" instead of "refused by the retained length bound".
- `_JOINT_TOL` (1e-3) and `_JOINT_WELD` (2e-3) unchanged.

### The new permanent case, and its before-verdict

`polyhedron-shallow-sliver-pyramid`: a three-face pyramid whose base is the
h = 0.05 sliver, named as the multi-face cousin of
`polyhedron-shallow-sliver-h*`. Placed in `_OPEN` by the same rule as every
round before it (public-API construction whose own declaration says open).
Measured BEFORE any edit, in its own fresh process:

```
primitives emitted: 3
verdict closed=True  declares=False
base-edge detour/chord = 1.0050
```

So last round's per-primitive rule called it **closed**: the three base
edges come from three different face primitives and chain across them
legally, at the sliver's ~1.005 detour/chord, far inside the 1.25 bound --
exactly the reachable gap the brief described. Under mob grouping the same
shape reports open (table below); all three faces share one emitting mob
(the sweep prints `[prims=3, emitting mobs=1]` for it).

### Verdict table, re-run

Every line printed by running all three groups through the real checker in
one fresh process; an instrumented transcription of the checker (the
"mirror") was asserted to agree with it on every verdict:

```
== _CLOSED entries (expect closed=True) ==
OK  cone-capped                      -> closed=True  declares=True
OK  convex-hull                      -> closed=True  declares=True
OK  cube                             -> closed=True  declares=True
OK  cylinder-capped                  -> closed=True  declares=True
OK  dodecahedron                     -> closed=True  declares=True
OK  dot3d                            -> closed=True  declares=True
OK  icosahedron                      -> closed=True  declares=True
OK  line3d                           -> closed=True  declares=True
OK  octahedron                       -> closed=True  declares=True
OK  prism                            -> closed=True  declares=True
OK  sphere                           -> closed=True  declares=True
OK  tetrahedron                      -> closed=True  declares=True
OK  torus                            -> closed=True  declares=True
== _OPEN entries (expect closed=False) ==
OK  cone-capped-partial               -> closed=False  declares=False  [prims=2, emitting mobs=2]
OK  cone-open                         -> closed=False  declares=False  [prims=1, emitting mobs=1]
OK  cylinder-open                     -> closed=False  declares=False  [prims=1, emitting mobs=1]
OK  halfpipe-with-discs               -> closed=False  declares=False  [prims=3, emitting mobs=3]
OK  polyhedron-shallow-sliver-h005    -> closed=False  declares=False  [prims=1, emitting mobs=1]
OK  polyhedron-shallow-sliver-h030    -> closed=False  declares=False  [prims=1, emitting mobs=1]
OK  polyhedron-shallow-sliver-h03749  -> closed=False  declares=False  [prims=1, emitting mobs=1]
OK  polyhedron-shallow-sliver-pyramid -> closed=False  declares=False  [prims=3, emitting mobs=1]
OK  polyhedron-shallow-zigzag         -> closed=False  declares=False  [prims=2, emitting mobs=1]
OK  polyhedron-single-quad            -> closed=False  declares=False  [prims=2, emitting mobs=1]
OK  polyhedron-single-triangle        -> closed=False  declares=False  [prims=1, emitting mobs=1]
OK  polyhedron-tetra-minus-face       -> closed=False  declares=False  [prims=3, emitting mobs=1]
OK  sphere-partial                    -> closed=False  declares=False  [prims=1, emitting mobs=1]
OK  torus-partial                     -> closed=False  declares=False  [prims=1, emitting mobs=1]
== chain foils (expect closed=False) ==
OK  cap-concentric-but-wrong          -> closed=False
OK  cap-detached                      -> closed=False
OK  cap-phase-shifted                 -> closed=False
```

All 13 `_CLOSED` closed, all 14 `_OPEN` open (13 pre-existing plus the new
pyramid), all 3 chain foils open -- no regression. The `[prims=k, emitting
mobs=m]` annotations are new diagnostics from the probe; note the Polyhedra:
many primitives, always exactly one emitting mob.

### Legitimate joints still consume

Same instrumented mirror as last round:

```
cylinder-capped: verdict=True, chains consumed=26
  distinct coarse-edge lengths: [0.167521]
  ratio min=1.009407 max=1.009407 n_edges per chain=[5]
  every chain crosses mobs: True; no chain reuses its own coarse mob: True
cone-capped: verdict=True, chains consumed=19
  distinct coarse-edge lengths: [0.148135]
  ratio min=1.004285 max=1.004285 n_edges per chain=[4]
  every chain crosses mobs: True; no chain reuses its own coarse mob: True
```

Chains consumed (26 / 19) and ratios (1.009407 / 1.004285) are IDENTICAL to
both previous rounds', and every chain crosses the new grouping: cap rims
are emitted by `_CapDisc` mobs, ring chords by the tube/cone mob.

### Arrow3D whole-tree probe

```
pooled prims=5, emitting mobs=5
pooled verdict=True, chains=72 (shaft-attributed=48, head-attributed=24, other=0)
shaft alone verdict=True chains=48; head alone verdict=True chains=24
root-published skin: prims=5, emitting mobs=1, verdict=False, chains=0
```

The whole-tree reading pools both part subtrees into ONE checker call --
every triangle of the arrow's skin once, tagged with its true emitting part
-- and still consumes **72 = 48 + 24**, parts independently, as required.
Shaft and head remain separate mobs under the rule, so the split stands.

The last line is a finding worth recording rather than burying: pooling the
arrow's OWN aggregate handover attributes all five primitives to the
Arrow3D publisher (one source), and the checker then rightly refuses to
chain its internal T-joints. It exposed a latent defect in
`_triangle_prims` itself: the walk collected BOTH the root's handover AND
the part subtrees, i.e. every triangle twice -- harmless under per-
primitive indexing (the duplicate copies wore distinct indices and pass-1
paired across them; last round's 72 came out of that doubled pool), but
under mob grouping every edge key's first-contributor became the root and
no chain could ever fire. Fixed inside the helper, mirroring the
publication contract (`draws_descendants`: an aggregate hands the renderer
its whole subtree itself, so the walk stops there). Root-level attribution
is kept deliberately -- a `Polyhedron` publishes every face itself under
one mesh key, and publisher-level attribution is what keeps all its faces
ONE source, which is the brief's central claim. No current caller of the
helper changes behaviour except the double-collection removal; every entry
in the table above was re-run after the fix.

### Does any hole family survive?

**No.** Every family found across the rounds -- the lone triangle, the lone
quad, tetra-minus-face, the shallow slivers (h = 0.05 through 0.3749), the
zigzag, and the sliver-base pyramid -- is refused by the same rule with no
metric involved: each is one mob's boundary trying to close over itself,
and one mob's boundary cannot. What survives is only the honest boundary
stated before: a union of DIFFERENT mobs whose edges coincide exactly can
still read closed (glued shells always could, via strict pairing, since
before chains existed), and a checker fed a single publisher's aggregated
handover sees that publisher's internals as one source. Neither is a hole
in a part; neither was widened away.

### Verification, verbatim

`.venv/bin/python -m pytest -q tests/unit_tests/test_closed_shell_declaration.py tests/unit_tests/test_normal_orientation.py tests/unit_tests/test_cap_disc_rim.py`

```
78 passed, 3 warnings in 9.83s
```

(77 before + the new `_OPEN` parametrization.)

`.venv/bin/python -m pytest -q tests/unit_tests`

```
1884 passed, 91 skipped, 160 warnings in 386.47s (0:06:26)
sys:1: ResourceWarning: unclosed file <_io.TextIOWrapper name=11 mode='w' encoding='utf-8'>
```

Green with zero survivors (1883 + 1 new). The ResourceWarning is pytest's
own teardown noise, present in prior rounds too.

ruff on the file:

```
$ .venv/bin/ruff check --no-fix tests/unit_tests/test_closed_shell_declaration.py
All checks passed!
$ .venv/bin/ruff format --check tests/unit_tests/test_closed_shell_declaration.py
1 file already formatted
```

Not done, per the brief: `tests/full_renders`, `--fast`, any
`ALGAN_UPDATE_*_BASELINE` variable, any commit or push.

### Disclosures

- One mid-round rerun of the whole-tree probe reported False/0-chains and
  cost a diagnosis: it was the double-collection defect above, not the
  grouping misfiring. Both probe-side print bugs it surfaced (a discarded
  shaft record count) were in `/tmp/opencode` scripts, never repo code.
- The helper's return shape changed (pairs, not bare primitives); its four
  in-file call sites were updated with it, including the merged-collection
  test, which now unpacks primitives explicitly.
