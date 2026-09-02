"""Every built-in solid's closed-shell declaration matches its geometry.

``Mob.closed_shell`` is what lets ``opacity`` mean one attenuation on a solid
(the sheet compaction caps a declared surface's cumulative coverage per pixel);
an open surface that wrongly declares would lose its legitimately-visible
interior, and a closed one that fails to declare keeps compositing per
crossing. So the declaration is checked against the geometry rather than
trusted: for every built-in solid, the triangles it actually emits must form a
closed, consistently-wound manifold exactly when ``closed_shell`` says so --
every directed edge paired with exactly one opposite (the same proof
``orient_faces_outward`` runs, here applied to the emitted primitives).

Welding tolerance differs by family and is load-bearing: polyhedron faces
share vertices exactly (the 1e-5 lattice ``test_normal_orientation`` uses),
while a cap's rim meets its body's ring only to the ~1e-3 the rim test there
promises, so compound solids weld at 2e-3 -- three orders below the smallest
feature on the unit-scale shapes tested and far above the joint gap. A cap
whose rim refined itself meets its body's coarser ring in a T-joint rather
than edge-for-edge; ``_forms_closed_shell`` consumes such joints through an
explicitly anchored, opposing chain of rim edges. What makes such a chain a
joint rather than a hole to borrow edges around is topological, not metric:
every chain edge must be emitted by a DIFFERENT MOB than the coarse edge it
closes -- a T-joint is two different parts of one solid meeting, while a hole
is one part failing to close and no part can close its own boundary. The
grouping sits at the mob rather than at the render primitive because a
``Polyhedron`` emits ONE PRIMITIVE PER FACE TRIANGLE: primitive identity
would still let a face's hole borrow its neighbours' edges and close over
itself. A length bound only keeps the chain search finite. The negative cases
below hold that generalisation against the openings it must never close over.

Pure tensor assertions on the primitive build -- no render, no Taichi. Feature
tests of the mob geometry, not of anything the timeline or Scene can break, so
they stay out of the fast suite.
"""

from __future__ import annotations

from collections import deque

import pytest
import torch

from algan import (
    ORIGIN,
    RIGHT,
    UP,
    Arrow3D,
    Cone,
    ConvexHull3D,
    Cube,
    Cylinder,
    Dodecahedron,
    Dot3D,
    Icosahedron,
    Line3D,
    Octahedron,
    Off,
    Polyhedron,
    Prism,
    Scene,
    Sphere,
    Square,
    Tetrahedron,
    Torus,
)
from algan.rendering.raytracing.primitives import (
    RayTracedTrianglePrimitive,
    closed_shell_ceiling_flag,
)

_EXACT_WELD = 1e-5
_JOINT_WELD = 2e-3
_JOINT_TOL = 1e-3
# A SEARCH BOUND for the chain BFS, and nothing else -- it keeps the search
# finite and the BFS off the shell's entire boundary, but carries no
# correctness role at all, not even as a backstop. What separates a
# legitimate joint from a hole is topological and lives in ``chain_from``: a
# chain may only consume boundary edges emitted by a DIFFERENT MOB than the
# coarse edge it closes, because a T-joint is two different parts of one
# solid meeting while a hole is one part failing to close itself --
# refused at any shape or scale, no tolerance involved. No length factor can
# do that job: a shallow hole's detour tends to arc/chord 1.0 exactly as a
# refinement arc does (a sliver's two-edge detour measures
# 2*sqrt(0.25 + h^2) chord-lengths, under this bound for every h < 0.375 on
# the unit chord), so the metric here only prunes the breadth-first search --
# paths are abandoned once they exceed 1.25 * the coarse edge's length,
# leaving room for the worst constructible legitimate ring (1.21) beside the
# measured ~1.01.
_CHAIN_LENGTH_FACTOR = 1.25


def _triangle_prims(mob, include_children=True):
    """Every ``(primitive, emitting mob)`` pair of ``mob``'s skin -- each
    triangle collected once.

    A compound solid of revolution is a tree of parts (a capped ``Cylinder``
    is its own tube plus two cap mobs); ``get_render_primitives`` returns only
    a mob's OWN geometry, so the caps are reached through the children. A
    ``Polyhedron`` must NOT be descended into: its children include the
    vertex-and-edge graph (dots, tubes), which is markup on the solid, not
    skin -- its own primitives are already every face triangle.

    An AGGREGATE that draws its own descendants (``draws_descendants`` -- an
    ``Arrow3D``, a point cloud) hands the renderer its whole subtree itself
    and its parts are never published separately, so the walk stops there:
    descending as well would collect every triangle twice, and doubled
    geometry is not what any mob emits.

    Each pair carries the mob that EMITTED the primitive: the chain rule
    groups boundary edges by emitting mob rather than by position in this
    list, so every face triangle of one ``Polyhedron`` -- one primitive per
    face -- reads as a single source.
    """
    out = []
    build = getattr(mob, "get_render_primitives", None)
    got = build() if build is not None else None
    if got is not None:
        for p in got if isinstance(got, list) else [got]:
            if (
                "Triangle" in type(p).__name__
                and getattr(p, "corners", None) is not None
            ):
                out.append((p, mob))
    if include_children and not getattr(mob, "draws_descendants", False):
        for child in getattr(mob, "children", []) or []:
            out.extend(_triangle_prims(child))
    return out


def _forms_closed_shell(prims):
    """True if the emitted ``(primitive, emitter)`` pairs enclose their interior.

    Every boundary edge must be consumed exactly once: either by one
    OPPOSITE boundary edge sitting on it (within ``_JOINT_TOL`` -- strict
    pairing settles the polyhedra outright, whose faces share welded
    vertices exactly), or, for a surface of revolution's cap joint, by a
    CHAIN of opposite-facing boundary edges running along it and anchored
    at its endpoints (each within ``_JOINT_TOL``). The chain is what a
    refined rim makes against its body: whole-multiple refinement keeps
    every ring vertex but adds rim vertices strictly between them, so each
    coarse ring edge is met by a fan of short rim edges whose interiors bow
    off the ring edge's chord by up to that chord's own sagitta (~1e-2 on
    these shapes -- the very quantity refinement removed from view) while
    their endpoints coincide with the chord's to ~1e-5. Judging such an edge
    by anything but its endpoints would take a tolerance big enough to call
    real holes closed; judging it by its endpoints cannot be fooled: a
    genuine opening -- an uncapped rim, a partial sweep's cut, a mis-sized
    or independently sampled cap -- puts no opposing chain between the
    exposed edge's own endpoints (``test_the_chain_rule_cannot_be_fooled``
    holds each of those against this function permanently).

    What separates a legitimate chain from a detour around an opening is
    the MOB GROUPING, not any metric: every chain edge must be emitted by a
    different MOB than the coarse edge it closes. A T-joint is two parts of
    one solid meeting -- a cap's rim refining its body's ring crosses mobs --
    while a hole is one part failing to close, and one mob's boundary
    closing over itself is refused at any shape or scale: every face of a
    ``Polyhedron`` comes from that one mob, so a lone triangle, a missing
    face or a whole pyramid over a slivered base cannot borrow its own
    neighbours' edges however shallow the hole. The grouping deliberately
    sits at the mob rather than at the render primitive -- a ``Polyhedron``
    emits ONE PRIMITIVE PER FACE TRIANGLE, so primitive identity would still
    let all of those holes chain across their own faces legally. The
    constraint needs no tolerance and is strictly narrower than anchoring
    and opposition alone, so it can never call an open shell closed; what
    has to hold -- and what these suites assert -- is that every legitimate
    joint does cross mobs.

    Each chain edge must also oppose the coarse edge's direction, no edge
    is consumed twice, and the SEARCH itself is bounded: a candidate path
    is abandoned once its length exceeds ``_CHAIN_LENGTH_FACTOR *`` the
    coarse edge's. That factor bounds the search and nothing else -- no
    length can tell a shallow hole from a refinement arc -- and it must be
    read together with the mob grouping above.
    """
    if not prims:
        return None
    # One row-block per EMITTING MOB -- keyed by ``id(mob)``, stable because
    # every emitter here stays alive for the whole call -- so each kept
    # triangle remembers which mob emitted it rather than which list slot
    # its primitive sat in; that origin is what the chain rule tests.
    per_tri_source = []
    corner_blocks = []
    for prim, emitter in prims:
        block = prim.corners.reshape(-1, 3, 3)
        corner_blocks.append(block)
        per_tri_source.extend([id(emitter)] * block.shape[0])
    corners = torch.cat(corner_blocks, 0)
    keys = corners.div(_JOINT_WELD).round().to(torch.int64)
    # Degenerate triangles are excluded, not tolerated -- the convention
    # ``test_normal_orientation`` sets: a ``Cone``'s apex row collapses to a
    # point, and a collapsed triangle contributes a self-edge the fill rule
    # never sees rather than boundary the shell lacks.
    tri_keys = keys.tolist()
    kept = [
        (tri, per_tri_source[i])
        for i, tri in enumerate(tri_keys)
        if len({tuple(tri[0]), tuple(tri[1]), tuple(tri[2])}) == 3
    ]
    pts = corners.reshape(-1, 3)
    # Representative float position per welded vertex (first occurrence).
    rep = {}
    flat_keys = keys.reshape(-1, 3)
    for i in range(flat_keys.shape[0]):
        key = tuple(flat_keys[i].tolist())
        if key not in rep:
            rep[key] = pts[i]
    directed = {}
    source_of_edge = {}
    for tri, src in kept:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            e = (*tri[a], *tri[b])
            directed[e] = directed.get(e, 0) + 1
            # An edge key fed by more than one mob's primitives (a
            # coincident seam) keeps its first contributor: pass 1 below
            # pairs without looking at origins anyway, so only chain
            # candidates read this.
            source_of_edge.setdefault(e, src)
    # An interior edge pairs 1:1 with its reverse after welding; every other
    # directed edge is boundary this shell has to account for.
    boundary = [
        e
        for e, n in directed.items()
        if not (n == 1 and directed.get(e[3:] + e[:3], 0) == 1)
    ]
    tail = [rep[e[:3]] for e in boundary]
    head = [rep[e[3:]] for e in boundary]
    consumed = [False] * len(boundary)

    def near(p, q):
        return bool((p - q).norm() <= _JOINT_TOL)

    def opposes(i, coarse_dir):
        return bool(((head[i] - tail[i]) * coarse_dir).sum() < 0)

    # Strict near-coincident opposite pairs: the polyhedra resolve here,
    # untouched by what follows.
    for i in range(len(boundary)):
        if consumed[i]:
            continue
        for j in range(i + 1, len(boundary)):
            if consumed[j]:
                continue
            if near(tail[i], head[j]) and near(head[i], tail[j]):
                consumed[i] = consumed[j] = True
                break

    def chain_from(source, target, coarse_dir, home_mob):
        """An unused opposing chain of boundary edges from source to target.

        Breadth-first over edges anchored where the previous one ended;
        visited welded vertices keep it acyclic. A candidate must oppose
        the coarse edge's direction AND be emitted by a different mob than
        the coarse edge (``home_mob``) -- that mob grouping is what
        separates a refinement from a detour: one mob's boundary cannot
        close over itself, however shallow the hole, while a cap refining
        its body's ring always crosses mobs. The running length is pruned
        past ``_CHAIN_LENGTH_FACTOR *`` the coarse edge's purely to bound
        this search -- no length separates a shallow hole from a
        refinement arc, so nothing here judges legitimacy by size.
        """
        max_length = _CHAIN_LENGTH_FACTOR * float((source - target).norm())
        visited = set()
        queue = deque()
        for i in range(len(boundary)):
            if (
                consumed[i]
                or not near(tail[i], source)
                or not opposes(i, coarse_dir)
                or source_of_edge[boundary[i]] == home_mob
                or near(tail[i], head[i])
            ):
                continue
            length = float((head[i] - tail[i]).norm())
            if length > max_length:
                continue
            visited.add(boundary[i][3:])
            queue.append((i, [i], length))
        while queue:
            i, path, length = queue.popleft()
            if near(head[i], target):
                return path
            for j in range(len(boundary)):
                if consumed[j] or boundary[j][3:] in visited:
                    continue
                if (
                    near(tail[j], head[i])
                    and opposes(j, coarse_dir)
                    and source_of_edge[boundary[j]] != home_mob
                    and not near(tail[j], head[j])
                ):
                    step = float((head[j] - tail[j]).norm())
                    if length + step > max_length:
                        continue
                    visited.add(boundary[j][3:])
                    queue.append((j, [*path, j], length + step))
        return None

    # A leftover coarse edge may be closed by a chain of opposite-facing
    # edges running along it -- the refined-rim T-joint -- anchored at its
    # own two endpoints and emitted by other mobs than its own.
    for i in range(len(boundary)):
        if consumed[i]:
            continue
        path = chain_from(
            head[i], tail[i], head[i] - tail[i], source_of_edge[boundary[i]]
        )
        if path is not None:
            consumed[i] = True
            for j in path:
                consumed[j] = True
    return all(consumed)


# Every entry's declaration must MATCH the geometric verdict.
_CLOSED = {
    "sphere": lambda: Sphere(radius=0.5),
    "dot3d": lambda: Dot3D(radius=0.15),
    "cylinder-capped": lambda: Cylinder(radius=0.35, height=0.9, closed=True),
    "line3d": lambda: Line3D(radius=0.08),
    "cone-capped": lambda: Cone(radius=0.45, height=0.9, closed=True),
    "torus": lambda: Torus(ring_radius=0.55, tube_radius=0.22),
    "prism": lambda: Prism(width=0.9, height=0.6, depth=0.7),
    "cube": lambda: Cube(size=0.8),
    "tetrahedron": lambda: Tetrahedron(edge_length=0.9),
    "octahedron": lambda: Octahedron(edge_length=0.8),
    "icosahedron": lambda: Icosahedron(edge_length=0.6),
    "dodecahedron": lambda: Dodecahedron(edge_length=0.5),
    "convex-hull": lambda: ConvexHull3D(
        (0.4, 0, 0), (-0.4, 0, 0), (0, 0.4, 0), (0, -0.4, 0), (0, 0, 0.5)
    ),
}

_OPEN = {
    "sphere-partial": lambda: Sphere(radius=0.5, v_range=(22.9183, 126.0507)),
    "cylinder-open": lambda: Cylinder(radius=0.35, height=0.9),
    # Whole discs on a half-pipe still leave the cut running along the tube.
    "halfpipe-with-discs": (
        lambda: Cylinder(
            radius=0.35, height=0.9, v_range=(17.1887, 171.8873), closed=True
        )
    ),
    "cone-open": lambda: Cone(radius=0.45, height=0.9),
    # A partial sweep leaves the wedge's cut faces open even with the base on.
    "cone-capped-partial": (
        lambda: Cone(radius=0.45, height=0.9, closed=True, v_range=(22.9183, 171.8873))
    ),
    "torus-partial": (
        lambda: Torus(ring_radius=0.55, tube_radius=0.22, v_range=(28.6479, 229.1831))
    ),
    "polyhedron-single-quad": (
        lambda: Polyhedron([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], [[0, 1, 2, 3]])
    ),
    # Ordinary holes built straight through the public API -- no injected
    # fault, no adversarial mesh. A lone triangle's boundary loop turns
    # through 120 degrees at each corner, so its two remaining edges anchor
    # at the coarse edge's endpoints and oppose it; the length bound once
    # was the only thing refusing them, and now the mob grouping refuses
    # them outright -- one mob cannot close its own boundary.
    "polyhedron-single-triangle": (
        lambda: Polyhedron([[0, 0, 0], [1, 0, 0], [0.5, 0.87, 0]], [[0, 1, 2]])
    ),
    # A tetrahedron missing one face: the most common shape a hole in a
    # triangle mesh has. Its three-edge boundary loop contains three
    # triangular sub-loops, each of which the unbounded chain rule closed.
    # The mob grouping refuses it outright: all three hole edges belong to
    # the one Polyhedron mob. An earlier per-primitive grouping did NOT --
    # a Polyhedron emits one primitive per face triangle, so the three
    # edges came from three different primitives, chained across them
    # legally, and only the length bound (~2.0 ratio) stood guard.
    "polyhedron-tetra-minus-face": (
        lambda: Polyhedron(
            [[0, 0, 0], [1, 0, 0], [0.5, 0.87, 0], [0.5, 0.29, 0.82]],
            [[0, 1, 3], [1, 2, 3], [2, 0, 3]],
        )
    ),
    # -- The shallow-hole family ----------------------------------------
    # Holes the length bound provably could not catch, kept from the round
    # that found them so the constraint that does catch them stays held to
    # them. A lone isoceles sliver's two-edge detour measures
    # 2*sqrt(0.25 + h*h) chord-lengths on the unit chord -- under 1.25 for
    # every h < 0.375 (escape measured up to h = 0.3749, caught only at
    # 0.38) -- and the zigzag's legs each project negatively on the chord,
    # summing to barely more than it; both tend to arc/chord 1.0, the same
    # geometry a genuine refinement arc presents, so no length factor
    # separates them from legitimacy. What refuses them is topological:
    # each is ONE mob whose own boundary tries to close itself.
    "polyhedron-shallow-sliver-h005": (
        lambda: Polyhedron([[0, 0, 0], [1, 0, 0], [0.5, 0.05, 0]], [[0, 1, 2]])
    ),
    "polyhedron-shallow-sliver-h030": (
        lambda: Polyhedron([[0, 0, 0], [1, 0, 0], [0.5, 0.30, 0]], [[0, 1, 2]])
    ),
    "polyhedron-shallow-sliver-h03749": (
        lambda: Polyhedron([[0, 0, 0], [1, 0, 0], [0.5, 0.3749, 0]], [[0, 1, 2]])
    ),
    # The sliver family wearing THREE faces instead of one: a pyramid whose
    # base is the h=0.05 sliver, its three side faces reaching a lifted
    # apex. Any base edge can be closed by the other two -- they anchor on
    # its endpoints, oppose it, and their two-edge detour measures the
    # sliver's own ~1.005 chord-lengths, far inside the search bound --
    # which is exactly why the PREVIOUS per-primitive grouping called this
    # CLOSED: a Polyhedron emits one primitive per face triangle, so the
    # three base edges came from three different primitives and chained
    # across them legally. The hole is real (declares open) and the same
    # single gap as its cousins': all three faces come from one mob, and
    # one mob's boundary cannot close over itself.
    "polyhedron-shallow-sliver-pyramid": (
        lambda: Polyhedron(
            [[0, 0, 0], [1, 0, 0], [0.5, 0.05, 0], [0.5, 0.025, 0.4]],
            [[0, 1, 3], [1, 2, 3], [2, 0, 3]],
        )
    ),
    # A 3-leg zigzag closing the unit chord -- legs (-0.3, 0.1), (-0.3,
    # -0.15), (-0.4, 0.05); enclosed area 0.0125. Detour measures ~1.055
    # chord-lengths, well inside the old bound.
    "polyhedron-shallow-zigzag": (
        lambda: Polyhedron(
            [[0, 0, 0], [1, 0, 0], [0.7, 0.1, 0], [0.4, -0.05, 0]], [[0, 1, 2, 3]]
        )
    ),
}


# Faults injected AFTER construction -- no mob can be asked to declare them,
# so these exist to hold the chain rule of ``_forms_closed_shell`` against
# exactly the openings a sloppier version of that rule would close over. The
# dedicated test below asserts the geometry alone: a checker that calls any
# of these closed is broken, whatever the declaration machinery would have
# said about the unmutated mob.


def _capped_cylinder_detached_cap():
    """A real hole where a chain would plausibly be hunted for."""
    body = Cylinder(radius=0.35, height=0.9, closed=True)
    body.remove_child(body.top_cap)
    return body


def _capped_cylinder_mis_sized_cap():
    """Concentric but wrong: a sloppy loop-matcher welds rim to ring here."""
    body = Cylinder(radius=0.35, height=0.9, closed=True)
    body.top_cap.scale(0.85)
    return body


def _capped_cylinder_phase_shifted_cap():
    """The historical scallop fault in a perfect disguise.

    Same radius, same plane, same winding as the ring it should close --
    rotated half a segment about the shared axis, so its rim interleaves the
    ring's vertices instead of containing them. Only endpoint anchoring
    separates this from closed; that is what makes it the case most likely
    to slip through.
    """
    body = Cylinder(radius=0.35, height=0.9, closed=True)
    body.top_cap.rotate(180.0 / (body.grid_width - 1), UP)
    return body


_CHAIN_FOILS = {
    "cap-detached": _capped_cylinder_detached_cap,
    "cap-concentric-but-wrong": _capped_cylinder_mis_sized_cap,
    "cap-phase-shifted": _capped_cylinder_phase_shifted_cap,
}


@pytest.mark.parametrize("name", sorted(_CHAIN_FOILS))
def test_the_chain_rule_cannot_be_fooled(name):
    """The chain generalisation stays narrower than the holes it must catch.

    Loosening closure is only safe if every fault the strict rule caught is
    still caught; each entry here is a capped cylinder whose cap has been
    sabotaged after construction, which a rule matching loops by anything
    coarser than their actual endpoints would call closed.
    """
    with Scene(), Off():
        mob = _CHAIN_FOILS[name]()
        mob.spawn(animate=False)
        assert _forms_closed_shell(_triangle_prims(mob)) is False, (
            f"{name}: a capped cylinder with {name} was called closed -- "
            "the chain consumption admits a fault it must not"
        )


_FLAT_FACED = {
    "prism",
    "cube",
    "tetrahedron",
    "octahedron",
    "icosahedron",
    "dodecahedron",
    "convex-hull",
}


@pytest.mark.parametrize("name", sorted(_CLOSED))
def test_a_declared_closed_solid_really_is_one(name):
    with Scene(), Off():
        mob = _CLOSED[name]()
        mob.spawn(animate=False)
        assert mob.closed_shell is True, f"{name} fails to declare a closed shell"
        skin = _triangle_prims(mob, include_children=name not in _FLAT_FACED)
        assert _forms_closed_shell(skin) is True, (
            f"{name} declares closed_shell but its emitted triangles do not "
            "enclose their interior"
        )


@pytest.mark.parametrize("name", sorted(_OPEN))
def test_an_open_surface_cannot_declare(name):
    with Scene(), Off():
        mob = _OPEN[name]()
        mob.spawn(animate=False)
        truth = _forms_closed_shell(
            # A Polyhedron's children are its vertex-and-edge markup (dots,
            # tubes), not skin -- see _triangle_prims.
            _triangle_prims(mob, include_children=not isinstance(mob, Polyhedron))
        )
        assert truth is False, f"{name}'s triangles unexpectedly enclose their interior"
        assert mob.closed_shell is False, (
            f"{name} declares closed_shell over geometry with an open boundary"
        )


def test_the_compound_arrow_declares_per_part():
    """Arrow3D is two interpenetrating solids with separate mesh identities;
    each part closes itself (capped shaft, capped head) independently.
    """
    with Scene(), Off():
        arrow = Arrow3D(start=ORIGIN, end=RIGHT * 1.2, shaft_radius=0.06)
        arrow.spawn(animate=False)
        shaft = _forms_closed_shell(_triangle_prims(arrow.tail))
        head = _forms_closed_shell(_triangle_prims(arrow.head))
        assert shaft, "arrow shaft must be a closed manifold"
        assert head, "arrow head must be a closed manifold"
        assert arrow.tail.closed_shell, "shaft must declare closed"
        assert arrow.head.closed_shell, "head must declare closed"


def test_2d_geometry_has_no_shell_to_declare():
    """A circuit carries no triangle primitive and no declaration either way."""
    with Scene(), Off():
        square = Square(size=0.8)
        square.spawn(animate=False)
        assert square.closed_shell is False
        assert _triangle_prims(square) == []


def test_a_merged_collection_keeps_each_mobs_declaration():
    """One batched collection holds a closed solid next to an open one; the
    per-triangle flag must not let either take the other's verdict.
    """
    with Scene(), Off():
        solid = Octahedron(edge_length=0.8)
        open_cone = Cone(radius=0.4, height=0.8)
        for mob in (solid, open_cone):
            mob.spawn(animate=False)
        solid_skin = _triangle_prims(solid)
        open_skin = _triangle_prims(open_cone)
        merged = RayTracedTrianglePrimitive(
            triangle_collection=[p for p, _ in solid_skin + open_skin]
        )
        flag = merged.closed_shell.reshape(merged.closed_shell.shape[1], -1)
        solid_tris = sum(int(p.corners.shape[1]) // 3 for p, _ in solid_skin)
        assert bool((flag[:solid_tris] == 1.0).all())
        assert bool((flag[solid_tris:] == 0.0).all())


def test_transmission_folds_the_declaration_back_open():
    """The pack-time fold: a closed declaration on a TRANSMISSIVE material is
    exempt -- refraction visits both shells as physical transport, so capping
    the second crossing would eat the refracted path. Any authored frame that
    transmits exempts the whole surface.
    """
    ones = torch.ones(1, 4, 3, 1)
    zeros = torch.zeros(1, 4, 3, 1)
    assert bool((closed_shell_ceiling_flag(ones, None) == 1.0).all())
    assert closed_shell_ceiling_flag(None, zeros) is None
    assert bool((closed_shell_ceiling_flag(zeros, None) == 0.0).all())
    assert bool((closed_shell_ceiling_flag(ones, zeros) == 1.0).all())
    # Transmissive at any corner/frame: exempt everywhere.
    partly = zeros.clone()
    partly[0, :, 0, 0] = 0.5
    assert bool((closed_shell_ceiling_flag(ones, partly) == 0.0).all())
