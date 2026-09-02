"""Every built-in solid's normals and winding face OUT.

The renderer shades a mob whose geometry declares an outside
(``Mob.two_sided`` False) one-sided: a back-facing hit keeps its own normal
instead of borrowing the viewer's side. That makes orientation load-bearing.
An inside-out mesh used to be invisible -- the shading normal was flipped
toward the camera, so a solid lit its interior and looked right anyway -- and
two built-ins were inside-out for exactly that reason:

* ``Torus`` inherits Manim's left-handed ``(u, v)``, so ``du x dv`` pointed
  into the tube (``Surface._grid_orientation``).
* ``Cylinder._move_between_points`` built its frame from
  ``get_orthonormal_vector``, which promises orthogonality and determinism but
  not handedness, so a ``Line3D`` came out mirrored.

These are pure tensor assertions on the primitive build -- no render, no
Taichi -- but they are feature tests of the mob geometry rather than of
anything the timeline or the Scene can break, so they stay out of the fast
suite.

Degenerate corners are excluded, not tolerated: a ``Cone``'s apex row collapses
to a point, so its vertex normals are zero there by construction and the kernel
substitutes the geometric normal (``_triangle_normal``). What the tests below
assert is that nothing NON-degenerate points the wrong way.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

# Same helpers test_cap_disc_rim uses to read a surface's built grid and a
# cap's built rim -- one way of selecting that geometry across both files,
# so the two cannot drift apart.
from test_cap_disc_rim import _built_grid, _rim_points

from algan import (
    IN,
    LEFT,
    ORIGIN,
    OUT,
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
    Prism,
    Scene,
    Sphere,
    Tetrahedron,
    Torus,
)
from algan.mobs.shapes_3d import _CapDisc

_DEGENERATE = 1e-6


def _revolved_parts(mob):
    """The surfaces of revolution inside ``mob``, each of which owns its own
    outward reference.

    A compound solid is a tree of them (``Arrow3D`` is a ``Cylinder`` shaft and
    a ``Cone`` head), and asking the compound for its primitives would lose
    which part built which triangle -- the aggregate returns its descendants'
    primitives under its own name.
    """
    parts = [mob] if isinstance(mob, (Sphere, Cylinder, Cone, Torus, _CapDisc)) else []
    # Descend even into a solid that matched: a capped one carries its end
    # discs as children, and they are skin too -- an inward-facing cap lights
    # the inside of the solid exactly as an inside-out tube would.
    return parts + [
        part
        for child in list(getattr(mob, "children", []) or [])
        for part in _revolved_parts(child)
    ]


def _triangles(mob):
    """``mob``'s own render triangles as ``(corners, normals)``, both ``[T, 3,
    3]`` at the first frame. No subtree walk: aggregates would double-count.
    """
    build = getattr(mob, "get_render_primitives", None)
    got = build() if build is not None else None
    if got is None:
        return []
    out = []
    for primitive in got if isinstance(got, list) else [got]:
        # Bezier circuits (the 2-D caps) carry control points, not corners.
        if "Triangle" not in type(primitive).__name__:
            continue
        corners = primitive.corners
        normals = getattr(primitive, "normals", None)
        if corners is None or normals is None:
            continue
        out.append(
            (corners.reshape(-1, 3, 3).float(), normals.reshape(-1, 3, 3).float())
        )
    return out


def _uniform_scale(mob):
    """The mob's scale, read off its basis rows (which carry it)."""
    return float(mob.basis.reshape(-1, 3, 3)[0].norm(dim=-1).mean())


def _outward(mob, points):
    """Analytic outward direction of ``mob``'s surface at world ``points``.

    One expression per solid family rather than a generic oracle, because the
    generic ones are wrong on the shape that matters most here: "away from the
    centre" holds for a sphere and fails on a torus's inner ring.
    """
    centre = mob.location.reshape(-1, 3)[0]
    delta = points - centre
    if isinstance(mob, _CapDisc):
        # A flat disc faces one way everywhere, and it is the way it was told
        # to: the outward normal of the solid it closes.
        return F.normalize(mob.direction.reshape(1, 3), dim=-1).expand_as(points)
    if isinstance(mob, Sphere):  # Dot3D is a Sphere
        return delta
    axis_source = (
        mob.get_forward_direction()
        if isinstance(mob, Torus)
        else mob.get_up_direction()
    )
    axis = F.normalize(axis_source.reshape(-1, 3)[0], dim=-1)
    axial = (delta * axis).sum(-1, keepdim=True) * axis
    radial = delta - axial
    if isinstance(mob, Torus):
        # Away from the nearest point on the major circle. The ring radius is
        # the authored one times whatever the mob has been scaled by, since
        # the grid holds world points.
        ring = float(mob.ring_radius) * _uniform_scale(mob)
        return delta - F.normalize(radial, dim=-1) * ring
    if isinstance(mob, Cone):  # Cone before Cylinder: Line3D is a Cylinder
        return F.normalize(radial, dim=-1) * abs(float(mob.height)) + axis * abs(
            float(mob.radius)
        )
    if isinstance(mob, Cylinder):
        return radial
    raise AssertionError(f"no outward reference for {type(mob).__name__}")


def _assert_faces_outward(mob, name):
    """Winding and vertex normals agree with the analytic outward direction."""
    parts = _revolved_parts(mob)
    assert parts, f"{name} contains no surface of revolution to check"
    batches = [(part, tris) for part in parts for tris in _triangles(part)]
    assert batches, f"{name} built no triangles to check"
    checked_corners = 0
    for owner, (corners, normals) in batches:
        reference = F.normalize(_outward(owner, corners.reshape(-1, 3)), dim=-1)
        reference = reference.reshape(-1, 3, 3)

        winding = torch.cross(
            corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0], dim=-1
        )
        real = winding.norm(dim=-1) > _DEGENERATE  # not a collapsed apex row
        face_dot = (F.normalize(winding[real], dim=-1) * reference[real].mean(1)).sum(
            -1
        )
        assert bool((face_dot > 0).all()), (
            f"{name}: {int((face_dot <= 0).sum())} of {int(real.sum())} triangles "
            "wind the wrong way round, so their geometric normals -- what the "
            "kernel falls back to when vertex normals are degenerate -- point "
            "into the solid"
        )

        flat_normals = normals.reshape(-1, 3)
        lit = flat_normals.norm(dim=-1) > _DEGENERATE
        vertex_dot = (
            F.normalize(flat_normals[lit], dim=-1) * reference.reshape(-1, 3)[lit]
        ).sum(-1)
        assert bool((vertex_dot > 0).all()), (
            f"{name}: {int((vertex_dot <= 0).sum())} of {int(lit.sum())} vertex "
            "normals point into the solid, so one-sided shading lights its "
            "inside"
        )
        checked_corners += int(lit.sum())
    assert checked_corners > 0, f"{name} carries no usable vertex normals"


def _closed_mesh_is_outward(mob, name):
    """A closed mesh's own winding proves its orientation without an oracle.

    The signed volume of a closed triangle soup is positive exactly when the
    faces wind counter-clockwise seen from outside, so this needs no per-shape
    reference -- which is what makes it the right test for the polyhedra, whose
    vertex normals are zero by design (the kernel uses the winding).
    """
    batches = _triangles(mob)
    assert batches, f"{name} built no triangles to check"
    corners = torch.cat([c for c, _n in batches], 0)
    v0, v1, v2 = corners[:, 0], corners[:, 1], corners[:, 2]

    volume = (v0 * torch.cross(v1, v2, dim=-1)).sum(-1).sum() / 6.0
    assert float(volume) > 0, (
        f"{name}: signed volume {float(volume):+.4f} -- the faces wind inward, "
        "so every geometric normal points into the solid"
    )

    # Closed: every directed edge is matched by exactly one reverse edge.
    keys = torch.round(corners / 1e-5).to(torch.int64)
    directed = {}
    for a, b in ((0, 1), (1, 2), (2, 0)):
        starts, ends = keys[:, a], keys[:, b]
        for i in range(starts.shape[0]):
            edge = (*starts[i].tolist(), *ends[i].tolist())
            directed[edge] = directed.get(edge, 0) + 1
    unmatched = [
        e for e, n in directed.items() if n != 1 or directed.get(e[3:] + e[:3], 0) != 1
    ]
    assert not unmatched, (
        f"{name}: {len(unmatched)} directed edges are unpaired, so the mesh is "
        "not a closed, consistently wound manifold"
    )


# The surfaces of revolution, including the frames built by look() and by
# move_between_points, and axes that make get_orthonormal_vector pick each of
# its seeds.
_REVOLVED = {
    "sphere": lambda: Sphere(radius=0.8),
    "sphere-partial": lambda: Sphere(radius=0.8, v_range=(22.9183, 126.0507)),
    "dot3d": lambda: Dot3D(point=RIGHT * 0.3, radius=0.2),
    "cylinder": lambda: Cylinder(radius=0.5, height=1.0),
    "cylinder-capped": lambda: Cylinder(radius=0.5, height=1.0, closed=True),
    "cylinder-x": lambda: Cylinder(radius=0.4, height=1.2, direction=RIGHT),
    "cylinder-z": lambda: Cylinder(radius=0.4, height=1.2, direction=OUT),
    "cylinder-diagonal": lambda: Cylinder(
        radius=0.3, height=1.5, direction=(1.0, 1.0, 1.0)
    ),
    "cylinder-between-points": lambda: Cylinder(
        radius=0.3, height=1.0
    ).move_between_points(LEFT + IN, RIGHT + UP),
    "cylinder-set-direction": lambda: Cylinder(radius=0.3, height=1.0).set_direction(
        (0.2, -1.0, 0.4)
    ),
    "line3d-x": lambda: Line3D(start=LEFT, end=RIGHT, radius=0.12),
    "line3d-y": lambda: Line3D(start=ORIGIN, end=UP * 1.4, radius=0.1),
    "line3d-z": lambda: Line3D(start=IN, end=OUT, radius=0.1),
    "line3d-diagonal": lambda: Line3D(
        start=LEFT + IN * 0.7, end=RIGHT * 1.2 + UP * 0.6, radius=0.09
    ),
    "cone": lambda: Cone(radius=0.6, height=1.0),
    "cone-capped": lambda: Cone(radius=0.6, height=1.0, closed=True),
    "cone-capped-tilted": lambda: Cone(
        radius=0.5, height=1.2, direction=(0.4, -1.0, 0.7), closed=True
    ),
    "cylinder-capped-tilted": lambda: Cylinder(
        radius=0.4, height=1.1, direction=(1.0, 0.3, -0.6), closed=True
    ),
    "line3d-capped-rebased": lambda: Line3D(
        start=LEFT + IN * 0.4, end=RIGHT * 1.3 + UP * 0.5, radius=0.15
    ).move_between_points(LEFT * 0.6 + UP, RIGHT + IN * 0.9),
    "cone-direction": lambda: Cone(radius=0.5, height=1.1, direction=RIGHT),
    "torus": lambda: Torus(ring_radius=0.6, tube_radius=0.25),
    "torus-partial": lambda: Torus(
        ring_radius=0.6,
        tube_radius=0.25,
        u_range=(0, 171.8873),
        v_range=(28.6479, 229.1831),
    ),
    "arrow3d": lambda: Arrow3D(start=ORIGIN, end=RIGHT * 1.1, shaft_radius=0.06),
}

# The flat-sided family: closed, so their own winding is the reference.
_POLYHEDRA = {
    "cube": lambda: Cube(size=1.0),
    "prism": lambda: Prism(width=1.0, height=0.6, depth=0.7),
    "tetrahedron": lambda: Tetrahedron(edge_length=1.0),
    "octahedron": lambda: Octahedron(edge_length=0.9),
    "icosahedron": lambda: Icosahedron(edge_length=0.6),
    "dodecahedron": lambda: Dodecahedron(edge_length=0.5),
    "convex-hull": lambda: ConvexHull3D(
        RIGHT * 0.6, LEFT * 0.6, UP * 0.6, -UP * 0.6, OUT * 0.6, IN * 0.6
    ),
}


@pytest.mark.parametrize("name", sorted(_REVOLVED))
def test_revolved_solid_normals_face_outward(name):
    with Scene(), Off():
        mob = _REVOLVED[name]()
        mob.spawn(animate=False)
        _assert_faces_outward(mob, name)


@pytest.mark.parametrize("name", sorted(_POLYHEDRA))
def test_polyhedron_winds_outward(name):
    with Scene(), Off():
        mob = _POLYHEDRA[name]()
        mob.spawn(animate=False)
        _closed_mesh_is_outward(mob, name)


# The capped joints, with the ring each disc actually closes, selected off
# the built grid by the same row indexing test_cap_disc_rim uses (a body's
# whole grid is only "the ring" for a two-row cylinder; a Cone's base is its
# u=0 row and every other row is something else).
_RIMS = {
    "cylinder": (
        lambda: Cylinder(radius=0.45, height=1.0, closed=True),
        lambda m, g: ((m.bottom_cap, g[:, 0]), (m.top_cap, g[:, -1])),
    ),
    "cylinder-tilted": (
        lambda: Cylinder(
            radius=0.3, height=1.2, direction=(1.0, 0.4, -0.7), closed=True
        ),
        lambda m, g: ((m.bottom_cap, g[:, 0]), (m.top_cap, g[:, -1])),
    ),
    "cone": (
        lambda: Cone(radius=0.55, height=1.1, closed=True),
        lambda m, g: ((m.base_circle, g[0]),),
    ),
    "cone-tilted": (
        lambda: Cone(radius=0.4, height=0.9, direction=(0.2, -1.0, 0.5), closed=True),
        lambda m, g: ((m.base_circle, g[0]),),
    ),
    "line3d-rebased": (
        lambda: Line3D(start=LEFT, end=RIGHT, radius=0.2).move_between_points(
            LEFT * 0.7 + IN * 0.5, RIGHT * 1.2 + UP * 0.8
        ),
        lambda m, g: ((m.bottom_cap, g[:, 0]), (m.top_cap, g[:, -1])),
    ),
}


@pytest.mark.parametrize("name", sorted(_RIMS))
def test_an_end_discs_rim_sits_on_the_bodys_own_ring(name):
    """A cap closes the BODY'S OWN ring: ring ⊆ rim, in whole multiples.

    The direction matters now that rims refine themselves: a cap grows its
    rim in whole multiples of the body's ring count until the chord polygon
    meets ``geometry_tolerance``, so the rim carries MORE vertices than the
    ring -- refinement adds rim vertices strictly BETWEEN the body's
    samples, up to half a ring chord from the nearest one. Holding every RIM
    vertex against the ring (the old direction) would fail on that
    legitimate refinement. What the original fault requires is the converse:
    the cone's base once missed its own ring by half a segment because it
    was sampled independently, which scalloped the rim -- so now every ring
    vertex must sit on the rim (within the joint's 1e-3 construction
    tolerance), and the counts must stand in whole multiple, which is what
    shows the sampling was derived from the ring rather than coinciding with
    it. An independently sampled circle of even the exact radius passes
    neither way: its vertices interleave the ring's instead of containing
    them, and any count relation would be an accident.
    """
    build, rings_of = _RIMS[name]
    with Scene(), Off():
        body = build()
        body.spawn(animate=False)
        grid = _built_grid(body)
        for disc, ring in rings_of(body, grid):
            rim = _rim_points(disc)
            assert len(rim) > 2, f"{name}: no rim found on the disc"
            chords = ring.shape[0] - 1  # closed ring: last sample repeats the first
            assert rim.shape[0] % chords == 0, (
                f"{name}: the cap's {rim.shape[0]}-vertex rim is not a whole "
                f"multiple of the body's {chords}-chord ring, so it was not "
                "sampled off the body's own ring"
            )
            gap = torch.cdist(ring, rim).min(dim=-1).values.max()
            assert float(gap) < 1e-3, (
                f"{name}: a ring vertex sits {float(gap):.2e} from the "
                "nearest rim vertex, so the cap is not a fan over the "
                "body's own ring"
            )


def test_moving_a_solid_does_not_reorient_it():
    """Transforms must not mirror a mob: a rotated, scaled, moved Torus is the
    shape most likely to come back inside-out, since its orientation is the one
    the grid reverses.
    """
    with Scene(), Off():
        torus = Torus(ring_radius=0.7, tube_radius=0.2)
        torus.spawn(animate=False)
        torus.rotate(37, UP + RIGHT).scale(1.4).move(RIGHT * 1.3 + UP * 0.4)
        _assert_faces_outward(torus, "torus after transforms")


def test_surface_grid_orientation_reverses_only_the_v_axis():
    """The reorientation must not move a vertex: it reverses the storage order
    of the v axis, so the rendered grid holds the same points as the authored
    one.
    """
    with Scene(), Off():
        torus = Torus(ring_radius=0.7, tube_radius=0.2)
        torus.spawn(animate=False)
        assert torus._grid_orientation == -1

        rendered = torus._reshape_grid_for_render(torus.grid.location)
        authored = torus.grid.location.reshape(rendered.shape)
        assert torch.equal(rendered, authored.flip(-2))


def test_unit_normals_stay_aligned_with_their_grid_rows():
    """``get_unit_normals`` is a per-vertex API: reorienting the render grid
    must not permute what it returns, only point the vectors outward.
    """
    with Scene(), Off():
        torus = Torus(ring_radius=0.7, tube_radius=0.25)
        torus.spawn(animate=False)
        points = torus.grid.location.reshape(-1, 3)
        normals = torus.get_unit_normals().reshape(-1, 3)
        outward = F.normalize(_outward(torus, points), dim=-1)
        usable = normals.norm(dim=-1) > _DEGENERATE
        dots = (F.normalize(normals[usable], dim=-1) * outward[usable]).sum(-1)
        assert bool((dots > 0).all())
