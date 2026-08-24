# OX cap-rim audit: why a Cylinder's end cap reads as a polygon

Scene: `tests/full_renders/scenes/solids_and_camera.py:64` —
`Cylinder(radius=0.45, height=1.0, show_ends=True)`.

**What I executed.** No render, no test, no repo file touched. One throwaway
script in `/tmp/opencode/cap_rim_probe.py` (`.venv/bin/python`) replicated
`_compute_pn_geometry_error` + `_find_geometry_resolution` for this exact
cylinder using the *real* `logical_pn_control_points` / `evaluate_logical_pn`;
the only substitution is analytic radial vertex normals (an area-weighted
accumulation over an axially flat, rotationally symmetric closed grid is radial
everywhere, including the merged seam). Every number below comes from that run;
everything else is source-reading and labelled as such.

## 1. The construction grid — the tube's `grid_width` comes from the search

`Cylinder(radius=0.45, height=1.0)` passes neither `resolution` nor grid keys,
so `_surface_resolution_kwargs` (shapes_3d.py:148) sets nothing.
`Surface.__init__` receives `grid_width=None, grid_height=None`
(surface.py:879-913) and therefore enables auto-resolution:
`_geometry_auto_resolution_enabled = (grid_height is None and grid_width is None)`
(surface.py:994-996). Construction then runs the cached search
(surface.py:1051-1061 → `_find_geometry_resolution`, surface.py:2006-2089):
binary-search each axis over `[min_grid_resolution, max_grid_resolution]` =
`[2, 200]` (surface.py:896-897), then a joint trim pass (surface.py:2043-2070).

The acceptance test is `_compute_pn_geometry_error` (surface.py:1729-1793):
build the candidate `[W, H]` grid, compute vertex normals, triangulate, build
logical PN control points, evaluate at **10 fixed barycentric sample points**
(surface.py:1752-1767), and take the max of `Cylinder._pn_geometry_deviation`
= `|radial − radius|` about the mob's location along its axis
(shapes_3d.py:846-858). Accept iff ≤ `geometry_tolerance` = **5e-4 world units**
(surface.py:893).

**Executed result:** the search lands on `(grid_width, grid_height) = (15, 2)`
— max sampled deviation 4.208e-4; width 14 fails at 5.645e-4, so 15 is exactly
the first acceptable count. The height trims to the floor 2 because the surface
is flat along the axis; the binding term is purely azimuthal. It is the
**search**, not a default and not `resolution`.

`Cylinder.add_bases` then hands that count to the caps:
`"segments": self.grid_width` (shapes_3d.py:756-764), and `_CapDisc.__init__`
sets `kwargs.setdefault("grid_width", max(3, int(segments)))`,
`kwargs.setdefault("grid_height", 2)` (shapes_3d.py:256-257). Because grid dims
are now given, the cap's own auto-resolution is off (surface.py:994-996). The
cap docstring says why (shapes_3d.py:216-219): *"a disc's flat interior is
exact at any resolution, so the search would answer 'two'."* The cap's rim is
therefore a polygon with `15 − 1 = 14` distinct chords (`u=0` and `u=1`
coincide on the full sweep).

## 2. The render-time dice levels — what can raise them

Per frame, `LogicalPNTrianglePrimitive._dice_logical_pn` (primitives.py:1998)
calls `_required_subdivision_levels` (primitives.py:1423-1486). The only level
setters in the pipeline are three searches over two error measures:

| Criterion | Where | Looks at | Compares against |
| --- | --- | --- | --- |
| Edge-chord screen error → `_required_edge_levels` (primitives.py:1608-1667) | each patch's 3 boundary curves | BOUNDARY only | the curve's **own** cubic vs its **own** chord polyline, sampled at t=0.25/0.5/0.75 per chord (primitives.py:1240, 1677-1759) |
| Interior flatness → `_required_patch_levels` (primitives.py:1761-1853) | 13 barycentric weights inside each microtriangle (primitives.py:1218-1232; six have a zero component, i.e. sit on microtriangle edges, some on the patch boundary) | INTERIOR (+ edge-straddling samples) | the PN patch vs its **own** piecewise-linear dice (primitives.py:1855-1954) |
| Across-coarsening → `_coarsest_across_levels` (primitives.py:1488-1555) | reuses the flatness measure on anisotropic candidates | same as above | same |

Modifiers: a patch starts at the max of its three boundary levels
(primitives.py:1455-1457, 1774-1777); errors carry a 1.25 safety factor
(primitives.py:1241); a screen guard box clamps off-frame samples
(primitives.py:1217, 1345-1407); `PN_GEOMETRY_SLACK` subtracts the logical
surface's own projected accuracy (primitives.py:2052-2057, 1394-1396); budgets
`max_diced_triangles = 2_000_000` and `max_subdivision_level = 8` cap promotion
(primitives.py:1191, 1208, 1656-1666, 1840-1849). The kernel/torch split
(`pn_criterion_kernel_active`, primitives.py:194-228) changes only where these
run, not what they ask.

There is **no silhouette criterion and no normal-variation criterion anywhere**
in the pipeline, and no criterion ever consults the analytic shape or
`_pn_geometry_deviation` at render time.

## 3. The entitlement question — the zero is true, and beside the point

- `_CapDisc._pn_geometry_deviation` returns zeros (shapes_3d.py:279-281). That
  function is consumed **only** by the construction-time search
  (`_compute_pn_geometry_error`, surface.py:1779); its other would-be caller,
  `_find_screen_space_resolution`, is dead because `_auto_resolution_enabled`
  is hard-set False (surface.py:997-999; gated again in
  `_can_update_resolution`, surface.py:1618-1629). For the cap even that path
  is disabled (§1). And the comment's claim is *true*: planar corners with
  parallel normals make every PN control point a convex combination of the
  corners (`_edge_control`, logical_pn.py:108-114 — the tangent-plane
  projection vanishes), so the patch **is** its own flat triangle and both
  render-time criteria return ~0 exactly. Level stays 0 forever.
- But `_pn_geometry_deviation` measures distance of PN samples to the analytic
  *surface* — the disc **including its interior**. A chord of the rim lies ON
  that surface (the plane is filled to r), so distance-to-surface cannot see
  the boundary's shape for ANY disc, however coarse. The error lives solely in
  which curve bounds the region, and the logical mesh's boundary is fixed at
  construction as straight chords between the 15 sampled ring points.
- The chord-vs-arc gap `r·(1−cos(π/n))` is visible to **no** criterion in the
  pipeline: not to the construction search (which never runs for the cap, and
  whose metric could not see it anyway), not to the edge-chord search (measures
  the cap's cubic, which *is* the chord), not to the flatness search (interior).
  Nothing else reports anything.

**Answer: no.** The quantity that decides the cap's dice level is entitled to
speak only for the fidelity of the PN patch to itself. The cap's silhouette was
discretized once, at construction, inherited from the *tube's* curvature
budget (with credit for the tube's PN bulge, which a flat disc does not get),
and no stage after that ever measures it.

## 4. External invariant — the rim violates the dice budget ~3.2–3.5×

**False premise first:** there is no `SETTINGS.raytracing.render_tolerance_pixels`.
The operative tolerances are per-Surface constructor defaults —
`render_tolerance=0.0005` (frame-height fraction) and
`render_tolerance_pixels=0.5` px (surface.py:894-895) — carried onto the
primitive (surface.py:3106-3107, merged min-over-collection at
primitives.py:1252-1277) and combined per frame by `_pixel_threshold` =
`min(render_tolerance · screen_height, render_tolerance_pixels)`
(primitives.py:1291-1308).

Numbers for this scene (executed arithmetic; projection scale from
primitives.py:1329-1343 with the default camera at `CAMERA_ORIGIN = OUT·7`
(__init__.py:227, constants/spatial.py:33), screen plane 5 units ahead,
half-height 2.5 world units → pixels-per-world ≈ `H_px / depth`; the cylinder
is unscaled all scene, so r_eff = 0.45):

- Rim chord error: `0.45·(1−cos(π/14))` = **0.01128 world units**.
- Suite renders at `PREVIEW` (704×396, tests/full_renders/test_full_renders.py:4-5;
  video_settings.py:94): 396/7 ≈ 56.6 px/world → **≈0.64 px** (0.60–0.69 across
  the cap's z range ±0.5).
- `HD` (1920×1080, video_settings.py:97): 1080/7 ≈ 154.3 px/world → **≈1.74 px**.
- Budget actually in force: PREVIEW `min(0.198, 0.5)` = **0.198 px**; HD
  `min(0.54, 0.5)` = **0.50 px**.
- Verdict: the rim is **outside tolerance by 3.2× (PREVIEW) and 3.5× (HD)**;
  fitting it would need n ≥ 26–27 chords, not 14.

And the more important finding stands: **the pipeline never evaluates this
quantity at all.** The violation is not a near-miss; it is unmeasured.

## 5. Contrast with the tube — and the seam is not welded across the joint

The tube's patches have real curvature, so both render criteria are nonzero and
its edge/interior levels climb per frame until the diced surface is within the
pixel budget (§2). Its rim boundary curves are PN cubics whose controls are
pulled outward toward the circle by the radial endpoint normals
(logical_pn.py:108-114) — executed check: the tube's rim cubic sits within
4.2e-4 world units of the true cylinder, i.e. ≈0.02 px at PREVIEW. So the
tube's silhouette converges to the circle while the cap's face ends on the
inscribed 14-gon, ≈0.0109 world units (~0.62 px at PREVIEW depth 7) inside it.
That divergence is the jagged seam: interpretation (not measured here) is that
near the cap the outer silhouette is the tube's smooth band, and the polygonal
reading comes from the lit break where the flat disc's straight-chord edge
meets it.

Welding: rim *vertices* coincide — the cap's ring is sampled from the same
expression as the tube's rings, live off the shared basis
(shapes_3d.py:752-759, 804-808, 811-823). Beyond endpoints there is **no
crack-free mechanism across the joint**. `snap_boundary_values`
(logical_pn.py:478-550) and the "level is a function of the curve alone"
design (logical_pn.py:12-21, primitives.py:1150-1168) keep two patches
watertight only when they share edge endpoints **and normals**, which holds
within one primitive. Tube and cap are separate Surfaces/primitives (merged
only for mesh identity/AA via `mesh_key`, shapes_3d.py:205-210, 755, 765-766,
surface.py:3126-3134, primitives.py:236-298), and their rim vertices carry
different normals (radial vs axial), so even the shared-curve argument could
not apply. The snap constrains each part internally; the tube↔cap seam relies
on coincident endpoints alone, and the two boundaries legitimately diverge
between them.

## 6. Blast radius

- **Cone `show_base=True`** (scene line 67): affected identically. Its base is
  a `_CapDisc(rim_function=_cap_ring_offsets, segments=self.grid_height)`
  (shapes_3d.py:528-535) — the cone's azimuthal count from the same kind of
  search — so the base disc shows an n-gon rim next to a smoothly diced side.
  Also every `Arrow3D` head (`Cone(..., show_base=True)`,
  shapes_3d.py:1021-1030) and `Line3D` (a capped thin `Cylinder`,
  `resolution=24` → 23-gon caps, shapes_3d.py:1161-1223): present but
  sub-pixel at thickness ≤0.05.
- **Sphere poles**: unaffected. No cap disc exists; pole regions have nonzero
  curvature, so the render criteria raise their levels per frame, and the
  construction search bounds the whole logical mesh (poles included) via the
  exact sphere-distance deviation (shapes_3d.py:412-420).
- **Torus**: unaffected. A closed torus has no caps and a partial sweep cuts
  open without capping (shapes_3d.py:1288-1290, 1341-1343).
- **2-D `Circle`→`Surface` path**: premise mostly false — algan's 2-D
  `Circle` is a cubic Bezier circuit (shapes_2d.py:891), rasterized with
  analytic coverage, never tessellated into a rim polygon. Triangulated circuit
  fills go through the bezier chord criterion, which does measure against the
  true curve (primitives.py:177-191), so they are tolerance-bounded. Only
  *filled non-planar* circuits become PN patches (nonplanar_circuit.py:794),
  which planar circles are not.

So the bug is exactly the `_CapDisc` family: `Cylinder.show_ends`,
`Cone.show_base`, and the thin-cylinder/cone parts of `Line3D`/`Arrow3D`.

## 7. What a fix would have to change (nothing implemented)

1. **Construction-side rim sizing in `_CapDisc`.** Choose
   `segments ≥ π/arccos(1 − tol/r)` (tol = `geometry_tolerance` or a dedicated
   rim tolerance) instead of inheriting the tube's count — here n ≥ 26–27 vs 15.
   Trade-off: deterministic, zero per-frame cost, but ~4–5× more cap triangles
   (28 → ~134 per pair here), and it moves the committed full-render baselines
   for any scene with a visible capped solid (`solids_and_camera` certainly;
   CPU *and* CUDA sets must be re-baselined). Scenes without capped solids stay
   byte-identical. User-side workaround available today: pass
   `resolution=(N, M)` to force the azimuthal count (shapes_3d.py:148-157).
2. **Render-time analytic-boundary criterion.** Give surfaces an optional
   declared boundary curve (a `_pn_boundary_*` hook analogous to
   `_pn_geometry_deviation`) and let `_edge_chord_error` measure the diced
   polyline against *that* rather than against the patch's own cubic, so cap
   edges climb per frame like the tube's do. Trade-off: construction unchanged;
   cost confined to declared edges; baselines move only where caps are on
   screen. Caveat: the cap's knots would then follow the circle while the
   tube's follow its cubic, leaving a residual seam gap bounded by
   construction-tolerance + dice-budget (~sub-pixel) instead of today's exact
   endpoint-only contact.
3. **Cross-part boundary inheritance via `mesh_key`.** Let the cap inherit its
   rim boundary level/polyline from the body's shared-ring edge and snap onto
   the tube's already-smooth boundary. Cheapest in triangles and visually
   complete, but it breaks the deliberate "boundary level is a function of that
   curve alone, no adjacency" invariant (logical_pn.py:12-21) across
   primitives whose endpoint normals disagree — the most invasive to
   established guarantees, and it still moves the same baselines.

Any of the three requires re-baselining `tests/full_renders` on both device
sets and is invisible to `tests/fast` (no PN geometry there — CLAUDE.md §Testing).
