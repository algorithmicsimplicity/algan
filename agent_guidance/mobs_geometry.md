# Mobs and geometry

The Mob model, packing, and how geometry reaches the renderer. Read this before touching
`algan/mobs/`, `algan/animatable_base/`, or surface tessellation.

## Where the implementations live

- `../algan/animatable_base/animatable.py` — `Animatable`: Scene ownership, ids, timeline-backed attribute get/set,
  spawn/despawn, clone, animated functions, updaters.
- `../algan/animatable_base/mob.py` plus the `mob_*.py` mixins — `Mob`: 3D location/basis/color, spatial transforms,
  screen-relative layout, `become` morphing, and the shader/material API.
- `../algan/mobs/shapes_2d.py` and `../algan/mobs/text.py` (`Text`/`Tex`) — cubic bezier circuits, built on
  `../algan/mobs/bezier_circuit.py`.
- `../algan/mobs/shapes_3d.py` — triangle meshes via `Surface` (`../algan/mobs/surfaces/surface.py`).
- `../algan/mobs/three_d_models/` — `Model3D`, which imports .glb/.fbx.
- `ManimMob` wraps Manim mobjects.

## Animatable and Mob model

`Animatable` handles Scene ownership, ids, timeline-backed attributes, lifespans, spawn/despawn, cloning, animated functions, and updaters.

`Mob` adds geometry-independent 3D state and behavior, including location, basis, scale, color, opacity, glow, hierarchy propagation, movement/layout, morphing, and shader/material configuration.

Parent changes normally propagate to descendants through batched timeline row operations. The canonical hierarchy is `children`/`components`; `Group.mobs` is an alias of `children`. Keep hierarchy operations Scene-homogeneous and cycle-safe.

The hierarchy is a **graph read at record time**, not a tree read at playback time. There are no local transforms: `location`/`basis` are world-space rows, and a parent transform is a delta written into the descendant union's rows *when the animation is recorded* — `modify_attribute_and_record` stores the resolved `RowRanges` on the event, so re-parenting after the fact never rewrites an animation already recorded. A Mob may have several parents and accumulates all of their deltas; that is deliberate (overlapping `Group`s arranging the same member), not a bug to guard against. `tests/unit_tests/test_mob_reparenting.py` pins all of it.

Two consequences worth knowing before touching this area:

- **Any mutation of a `children` list owes a `bump_hierarchy_version()`.** `_descendants_cache`, `_attr_inds_cache` and `_subtree_spawn_cache` are all keyed on those counters, so a mutation without the bump does not error — it silently serves the pre-mutation descendant set, and the transform quietly skips the new member while `len()` says it is there. `Group`'s non-owning slice path had exactly that bug.
- **Updaters do not get the record-time freeze.** The recorded-function replay path sets `_active_replay_event` so `replay_inds` hands back the stored rows; the updater loop in `set_state_to_times` does not, so every write inside an updater re-resolves its rows against the hierarchy as it stands at materialization. A hierarchy edit made while an updater is live therefore reaches backwards over frames that updater already covers. Documented and pinned, not endorsed — `AnimationTimeline.note_hierarchy_change` raises `HierarchyChangedDuringUpdaterWarning` at the authoring line that does it. The check is deliberately narrow: it fires only when the edited parent is in the updater's `recursive_dependency_mob_ids` (a Mob the updater addressed *as a subtree*, not merely one it depends on), never while an updater or a replayed function is running, and once per (updater, parent) pair. Widening any of those turns it into noise on every composite Mob.

### Packed mobs

One Mob can stand for many logical objects. Its animatable attributes carry one row per member, its components carry a block of rows per member, and `parent_batch_sizes` maps between the two. `Mob.__getitem__` slices that map to produce a **view** sharing the pack's id, rows and lifespan; `BatchedMobViewSequence` presents those views as a sequence.

Two ways in, differing only in when the packing happens. `from_batches` on a class that can build its geometry for many objects at once (`BezierCircuitCubic`, `Surface`) never constructs the per-member Mobs; `batch_mobs` packs Mobs that already exist. Both are built on `pack_animatable_rows` (the pack itself, one row per member) and `pack_member_rows` (a component, a block of rows per member) in `../algan/utils/mob_utils.py`, and both write through `_setattr_and_rebatch_without_record`, so they are valid only on fresh history.

Two invariants are easy to break and hard to see:

- A recursive write covers the whole subtree, so a value expressed per member must be distributed over each descendant's rows first (`Mob._distribute_over_packed_subtree`). Without it a packed Mob cannot be moved at all.
- The subtree is addressed in **buffer** order, not descendant order — `RowRanges.from_runs` sorts and coalesces the runs. A distribution built in descendant order still matches on row count and silently gives every member a neighbour's value.

Members share one lifespan, because they share one id. Staggered entrances go through opacity, which is what `Tex.write()` does.

Renderable mobs implement `get_render_primitives()`. The primary geometry families consumed by the renderer are:

- flat triangle primitives;
- cubic Bezier circuit primitives.

Important mob implementations include:

- 2D shapes and text, represented primarily as cubic Bezier circuits;
- `Surface` and 3D shapes, represented as flat triangle meshes (curved surfaces are diced from logical PN patches per frame);
- `TriangleMesh` and `Model3D` for imported 3D assets;
- `PointCloud`/point-cloud mobs;
- Manim compatibility wrappers and conversion helpers.

A Bezier circuit is resolved against its own plane, so its control points are projected onto that plane when the geometry is built — the identity for a shape that is genuinely planar, a different shape for one that is not. `../algan/mobs/nonplanar_circuit.py` classifies every circuit once, in `BezierCircuitCubic.__init__`, per sub-path (so a packed circuit is judged on its members, not on their non-planar union): planar circuits are untouched and keep the analytic path; a non-planar **filled** one renders each closed sub-path as logical PN patches, the same primitive `Surface` produces; a non-planar **unfilled** one is split into near-straight runs, each its own circuit whose plane is turned to face the camera about the run's axis (which is what stops a 3-D path's stroke vanishing wherever its osculating plane goes edge-on). The plan is topology only — geometry is rebuilt from the live control points every batch, so animation and transforms follow — but the *decision* is fixed at construction, exactly as the circuit's plane is. `batch_mobs` clones its first member before packing the rest, so anything derived from geometry at construction must be redone in `_after_repack()`; `from_batches` needs no hook because the constructor already sees every member. `ALGAN_NONPLANAR_CIRCUITS=0` restores the flattening.

Shader/material setup that changes primitive layout or registers shader parameters must occur before spawning unless the implementation explicitly supports timeline-safe mutation. Use the Three.js-style material classes (`MeshBasicMaterial`, `MeshStandardMaterial`, `MeshPhysicalMaterial`, and related classes) rather than restoring removed ad-hoc reflectivity/roughness APIs.


`tests/unit_tests/test_nonplanar_circuits.py` is the guard for the circuit-planarity rules above.

## Logical PN patches and dicing

Curved surfaces reach the renderer as *logical PN* patches diced to flat triangles per frame (`algan/rendering/logical_pn.py`); no curved-patch primitive exists in the renderer.

A patch's dice is **per direction** — `2 ** level` rows fanning from one corner, each cut into at most `2 ** across` columns — so a direction the surface is flat along (a cylinder's length) costs one cell however finely the curved direction is cut. Equal levels are the uniform grid exactly.

Both the construction grid (`geometry_tolerance`, per axis) and the render dice (per patch per frame) are chosen by measurement, and the render criteria stop at the logical surface's own accuracy rather than resolving the PN patch's error. The dice budget is the finer of two tolerances at the frame's resolution: `render_tolerance` (a fraction of frame height) and `render_tolerance_pixels` (an absolute pixel count, default 1.0, which is what binds from roughly 1080p up).

## A flat cap's rim is sized at construction, because nothing downstream can refine it

A `Cylinder`'s end discs and a `Cone`'s base are `_CapDisc`s whose vertex normals are one constant, so the PN patch and its PN edge curves *are* the flat triangle and its straight chords. Every render-time criterion measures a diced triangle against the patch's **own** cubic, so all of them return zero at level 0 and whatever polygon the rim was built as is the polygon that ships.

A body's ring count is only adequate because the body's PN patches curve back onto the true surface; a cap inherited that count without inheriting the credit that justified it, which is how a `Cylinder(radius=0.45)` cap rendered as a 14-gon 22.6x outside `geometry_tolerance` while the tube beside it stayed round. The disc now grows its rim in whole multiples of the body's ring until the chord polygon meets that tolerance — whole multiples so every one of the body's ring vertices stays a rim vertex.

`tests/unit_tests/test_cap_disc_rim.py` is the guard, `benchmarks/_cap_rim_probe.py` renders it.

Index a pack (`pack[3]`) for a view sharing the pack's rows and lifespan; members therefore cannot spawn or despawn
independently. `Text` packs its glyphs this way and a point cloud packs its dots.

## The z convention, and the one rule that follows from it

`OUTWARD` is **+z** — out of the screen, towards the viewer — matching Manim, Three.js and
glTF. `(RIGHT, UP, OUTWARD)` is therefore right-handed, and `rotate(90, OUTWARD)` turns
anti-clockwise on screen. `DEFAULT_BASIS` is `(RIGHT, UP, OUTWARD)` — the identity, and
right-handed: a Mob's *forward* axis is the way it **faces**, so a new Mob faces the viewer,
the way a glTF or Three.js model's front faces +z. A **camera**'s forward axis is the way it
*looks*, so it is the one thing built the other way round, at `(RIGHT, UP, INWARD)`
(`camera.py: _CAMERA_BASIS`) — which is what puts its screen between it and the origin.

The pair matters to more than `get_forward_direction()`: geometry built from a Mob's own
basis rows is built in a right-handed frame, and every such expression sweeps its
cross-section from **minus** the forward row (`Cylinder.coord_function` and both
`_cap_ring_offsets`), because the near side is where the forward axis points now. Get that
sign wrong and the (u, v) handedness of every surface of revolution reverses, taking its
vertex normals and its winding with it.

Everything else follows from one fact: **a cross product is a pseudovector.** Mirror a scene
in z and every ordinary vector mirrors with it, but `cross(a, b)` comes out *negated* as well
as mirrored. So each site that derives a direction from a cross product carries an explicit
sign, and each one is commented where it sits:

| Site | What it derives |
| :--- | :--- |
| `surface.py: compute_grid_vertex_normals` | grid vertex normals — returns `+cross`, not `-cross` |
| `shapes_3d.py: Cylinder._move_between_points` | `forward = right x up`, the right-handed frame every Mob is built with |
| `neural_net.py` (batched idle) | the same frame again, batched — the two must agree, and `test_neural_net_idle.py` is what says so |
| `shapes_3d.py: _CapDisc._sweep_faces` | which way a cap's fan winds to face `direction` |
| `geometry.py: get_rotation_around_axis` | returns the **transpose** of Rodrigues, because every call site applies it to row vectors |
| `geometry.py: get_rotation_between_3d_vectors` | pairs with that: `+cross(v1, v2)`, unnegated |
| `geometry.py: get_orthonormal_vector` | seeds from `RIGHT, UP, INWARD`, not the raw identity rows |

Anything that writes **world-space z as a literal** carries the convention too, and the
built-in shapes are where that lives: `Sphere`/`Cone`/`Torus`'s `coord_function` (a surface's
grid is world space — it is never mapped through the Mob's basis), and the vertex tables of
`Prism`, `Tetrahedron`, `Octahedron`, `Icosahedron` and `Dodecahedron`. `Cylinder` writes no z
literal because it builds from `basis_rows` — it carries the convention in the sign of its
forward row instead, as above.

A polygon is triangulated as a fan from `face[0]`, so re-winding a face must hold its first
vertex in place — that is `_rewound`, and it is why the tables can be written outward-wound
without moving a single triangle. `ConvexHull3D` sorts Qhull's simplices into a canonical
order for the same reason: the order the hull comes back in is a function of the input
coordinates, and it reaches the renderer as triangle order.

`TriangleMesh`'s grid needs no basis of its own: its corner positions are already baked into
world space by the loader, so its basis is the transform applied from there on, and that
starts as none — which `DEFAULT_BASIS`, being the identity, already is. Authored normals are
rotated by it while the positions are not, so a basis that was a rotation (or a mirror) would
light baked geometry by normals turned away from it. That is the shape of the bug the
`(RIGHT, UP, INWARD)` default caused here, and it is why an imported glTF model needs no
adjusting: its front faces +z, and so does a Mob's.

### Two things the flip left inconsistent

A **2-D circuit** derives its own frame from its control points
(`bezier_circuit.py: _circuit_location_and_basis`), and row 2 of that frame is
`cross(row 0, row 1)` — a cross product the flip never re-signed. Its control points do not
move when the world mirrors in z (they are in the xy plane), so the normal it yields did not
mirror with everything else. Measured on all three revisions, the NUMBER never moved:
`Square`, `Circle` and `Triangle` have carried `(0, 0, -1)` throughout. What moved is what
that vector is called — it was `OUTWARD`, at the viewer, before the flip and is `INWARD`
after it — so a flat shape now states that it faces **away**. (`Text` is not one of these:
its own basis is the Mob default, its glyphs carrying the circuit frames, so it faces
`OUTWARD` with every other Mob.) Nothing visible depends on it — a circuit is `two_sided`, so `_sided_shading_normal` turns the normal toward
whoever is looking — but it is the one place a Mob does not face the way `DEFAULT_BASIS` says
it does. Re-signing it is not a one-liner: row 1 is `cross(row 2, row 0)`, so the plane
normal's sign also sets the in-plane frame the texture grid and the analytic-AA edge
parameterisation are laid out along.

A **polyhedron's** faces are re-wound by `orient_faces_outward`, whose signed-volume test is
the ordinary right-hand-rule one, so `cross(v1-v0, v2-v0)` on a polyhedron triangle points
**out** of the solid. A **surface's** grid triangles keep the index order they always had,
because that order is a *screen-space* contract — the renderer's backface bit is the
projected winding — and their world-space cross therefore points **in**.

`tests/unit_tests/test_normal_orientation.py::test_revolved_solid_normals_face_outward` is
the test that says so: it fails on the revolved family and passes on the flat one. Reversing
the grid triangulation to agree (`t1 = (idx10, idx01, idx00)`) makes it pass and **regresses
three pixel baselines** — measured: `complex_hierarchy_become` max 1 → 93,
`shapes_and_timeline` 0 → 66, `solids_and_camera` 9 → 163 — so the two consumers want
opposite orders and one of them needs an explicit sign instead. The kernels that read a
geometric face normal mostly pass it straight to `_orient_hit_normals`, which turns it
against the ray, which is why nothing visible depends on it today; the exposure is the
fallback path for a mesh whose vertex normals are degenerate. **Unresolved** — it needs
whoever owns the analytic-AA run rule.
