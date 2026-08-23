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
feature on the unit-scale shapes tested and far above the joint gap.

Pure tensor assertions on the primitive build -- no render, no Taichi. Feature
tests of the mob geometry, not of anything the timeline or Scene can break, so
they stay out of the fast suite.
"""

from __future__ import annotations

import pytest
import torch

from algan import (
    ORIGIN,
    RIGHT,
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


def _triangle_prims(mob, include_children=True):
    """Every triangle primitive of ``mob`` -- descendants only asked for.

    A compound solid of revolution is a tree of parts (a capped ``Cylinder``
    is its own tube plus two cap mobs); ``get_render_primitives`` returns only
    a mob's OWN geometry, so the caps are reached through the children. A
    ``Polyhedron`` must NOT be descended into: its children include the
    vertex-and-edge graph (dots, tubes), which is markup on the solid, not
    skin -- its own primitives are already every face triangle.
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
                out.append(p)
    if include_children:
        for child in getattr(mob, "children", []) or []:
            out.extend(_triangle_prims(child))
    return out


def _forms_closed_shell(prims):
    """True if the emitted triangles enclose their interior.

    Every boundary edge must be consumed by exactly one OPPOSITE boundary edge
    sitting on it (within ``_JOINT_TOL``). Strict edge pairing after welding
    settles the polyhedra outright; the surfaces of revolution additionally
    allow a cap's rim fan to close its body's ring -- the two are watertight to
    ~1e-3 (the promise ``test_normal_orientation`` holds them to) without
    sharing vertices, which exact pairing would wrongly call open. Real
    openings cannot pass: an uncapped rim, a partial sweep's cut or a lone quad
    leaves boundary loops with no opposite partner anywhere near them.
    """
    if not prims:
        return None
    corners = torch.cat([p.corners.reshape(-1, 3, 3) for p in prims], 0)
    keys = corners.div(_JOINT_WELD).round().to(torch.int64)
    # Degenerate triangles are excluded, not tolerated -- the convention
    # ``test_normal_orientation`` sets: a ``Cone``'s apex row collapses to a
    # point, and a collapsed triangle contributes a self-edge the fill rule
    # never sees rather than boundary the shell lacks.
    tri_keys = keys.tolist()
    nondegenerate = [
        tri
        for tri in tri_keys
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
    for tri in nondegenerate:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            e = (*tri[a], *tri[b])
            directed[e] = directed.get(e, 0) + 1
    unpaired = []
    seen = set()
    for e, n in directed.items():
        rev = directed.get(e[3:] + e[:3], 0)
        if n == 1 and rev == 1:
            continue
        ek = tuple(sorted((e[:3], e[3:])))
        if ek in seen:
            continue
        seen.add(ek)
        unpaired.append((rep[e[:3]], rep[e[3:]]))
    # Greedily consume near-coincident opposite boundary pairs.
    consumed = [False] * len(unpaired)
    for i, (p, q) in enumerate(unpaired):
        if consumed[i]:
            continue
        for j in range(i + 1, len(unpaired)):
            if consumed[j]:
                continue
            r, s = unpaired[j]
            if torch.allclose(p, s, atol=_JOINT_TOL) and torch.allclose(
                q, r, atol=_JOINT_TOL
            ):
                consumed[i] = consumed[j] = True
                break
    return all(consumed)


# Every entry's declaration must MATCH the geometric verdict.
_CLOSED = {
    "sphere": lambda: Sphere(radius=0.5),
    "dot3d": lambda: Dot3D(radius=0.15),
    "cylinder-capped": lambda: Cylinder(radius=0.35, height=0.9, show_ends=True),
    "line3d": lambda: Line3D(thickness=0.08),
    "cone-capped": lambda: Cone(base_radius=0.45, height=0.9, show_base=True),
    "torus": lambda: Torus(major_radius=0.55, minor_radius=0.22),
    "prism": lambda: Prism(dimensions=(0.9, 0.6, 0.7)),
    "cube": lambda: Cube(side_length=0.8),
    "tetrahedron": lambda: Tetrahedron(edge_length=0.9),
    "octahedron": lambda: Octahedron(edge_length=0.8),
    "icosahedron": lambda: Icosahedron(edge_length=0.6),
    "dodecahedron": lambda: Dodecahedron(edge_length=0.5),
    "convex-hull": lambda: ConvexHull3D(
        (0.4, 0, 0), (-0.4, 0, 0), (0, 0.4, 0), (0, -0.4, 0), (0, 0, 0.5)
    ),
}

_OPEN = {
    "sphere-partial": lambda: Sphere(radius=0.5, v_range=(0.4, 2.2)),
    "cylinder-open": lambda: Cylinder(radius=0.35, height=0.9),
    # Whole discs on a half-pipe still leave the cut running along the tube.
    "halfpipe-with-discs": (
        lambda: Cylinder(radius=0.35, height=0.9, v_range=(0.3, 3.0), show_ends=True)
    ),
    "cone-open": lambda: Cone(base_radius=0.45, height=0.9),
    # A partial sweep leaves the wedge's cut faces open even with the base on.
    "cone-capped-partial": (
        lambda: Cone(base_radius=0.45, height=0.9, show_base=True, v_range=(0.4, 3.0))
    ),
    "torus-partial": (
        lambda: Torus(major_radius=0.55, minor_radius=0.22, v_range=(0.5, 4.0))
    ),
    "polyhedron-single-quad": (
        lambda: Polyhedron([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], [[0, 1, 2, 3]])
    ),
}


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
            _triangle_prims(mob, include_children=name != "polyhedron-single-quad")
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
        arrow = Arrow3D(start=ORIGIN, end=RIGHT * 1.2, thickness=0.06)
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
        square = Square(side_length=0.8)
        square.spawn(animate=False)
        assert square.closed_shell is False
        assert _triangle_prims(square) == []


def test_a_merged_collection_keeps_each_mobs_declaration():
    """One batched collection holds a closed solid next to an open one; the
    per-triangle flag must not let either take the other's verdict.
    """
    with Scene(), Off():
        solid = Octahedron(edge_length=0.8)
        open_cone = Cone(base_radius=0.4, height=0.8)
        for mob in (solid, open_cone):
            mob.spawn(animate=False)
        members = _triangle_prims(solid) + _triangle_prims(open_cone)
        merged = RayTracedTrianglePrimitive(triangle_collection=members)
        flag = merged.closed_shell.reshape(merged.closed_shell.shape[1], -1)
        solid_tris = sum(int(p.corners.shape[1]) // 3 for p in _triangle_prims(solid))
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
