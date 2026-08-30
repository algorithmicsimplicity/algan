"""Stress `Mob.become` over a cross-product of Mob types and hierarchies.

Read-only diagnostic: every case runs in its own Scene, records a 1-second
morph, then materializes the timeline at several times and checks invariants
that must hold for *any* morph, whatever route it took:

* it does not raise,
* the returned Mob is spawned and registered for rendering,
* at the end of the animation the result's geometry matches what the target
  alone would occupy (the external invariant -- a morph that finishes somewhere
  other than the target is wrong however pretty the path was),
* nothing is NaN at any sampled time,
* no intermediate frame leaves the union of the source and target bounds by
  more than a slack factor (a part flying off screen and back is the classic
  mispairing symptom).

Usage:  <venv-python> benchmarks/_become_stress.py [--filter SUBSTR] [--pairs N]
"""

from __future__ import annotations

import argparse
import itertools
import sys
import traceback

import torch

from algan import (
    BLUE,
    GREEN,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    YELLOW,
    Arrow,
    Circle,
    Cone,
    Cube,
    Cylinder,
    Dot,
    Group,
    Line,
    Off,
    RegularPolygon,
    Scene,
    Sphere,
    Square,
    Star,
    Surface,
    Sync,
    Tetrahedron,
    Text,
    Torus,
    Triangle,
    TriangleVertices,
)

# --------------------------------------------------------------------------
# The catalogue.  Each entry is (name, zero-arg constructor).  Constructors run
# inside an Off() block in a live Scene, so they may spawn.
# --------------------------------------------------------------------------


def _tri_vertices():
    return TriangleVertices(
        torch.tensor([[[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.6, 0.0]]]),
        color=GREEN,
    )


def _surface_plane():
    return Surface(
        lambda u, v: torch.stack((u - 0.5, v - 0.5, torch.zeros_like(u)), -1),
        grid_height=6,
        grid_width=6,
    )


def _group_two():
    return Group(Square(color=BLUE).move(LEFT), Circle(color=YELLOW).move(RIGHT))


def _group_three_mixed():
    return Group(
        Square(color=BLUE).move(LEFT * 2),
        Sphere(radius=0.4).move(ORIGIN),
        Triangle(color=GREEN).move(RIGHT * 2),
    )


def _nested_group():
    return Group(
        Group(Square().move(LEFT), Circle().move(LEFT + UP)),
        Group(Triangle().move(RIGHT), Star().move(RIGHT + UP)),
    )


def _deep_nest():
    return Group(Group(Group(Group(Square(color=BLUE)))))


def _group_of_solids():
    return Group(
        Cube(size=0.6).move(LEFT),
        Sphere(radius=0.35).move(RIGHT),
    )


def _empty_group():
    return Group()


def _group_one():
    return Group(Square(color=BLUE))


CATALOGUE: list[tuple[str, callable]] = [
    # bezier family
    ("Square", lambda: Square(color=BLUE)),
    ("Circle", lambda: Circle(radius=0.6, color=YELLOW)),
    ("Triangle", lambda: Triangle(color=GREEN)),
    ("Star", lambda: Star()),
    ("RegularPolygon7", lambda: RegularPolygon(7)),
    ("Line", lambda: Line(LEFT, RIGHT)),
    ("Arrow", lambda: Arrow(LEFT, RIGHT)),
    ("Dot", lambda: Dot()),
    ("Text_ab", lambda: Text("ab")),
    ("Text_hello", lambda: Text("hello")),
    # grid family (Surface subclasses)
    ("Sphere", lambda: Sphere(radius=0.6)),
    ("Cylinder", lambda: Cylinder(radius=0.4, height=1.0)),
    ("Cone", lambda: Cone()),
    ("Torus", lambda: Torus()),
    ("SurfacePlane", _surface_plane),
    # mesh family
    ("Cube", lambda: Cube(size=0.8)),
    ("Tetrahedron", lambda: Tetrahedron()),
    ("TriangleVertices", _tri_vertices),
    # containers
    ("Group2", _group_two),
    ("Group3Mixed", _group_three_mixed),
    ("GroupNested", _nested_group),
    ("GroupDeep", _deep_nest),
    ("GroupSolids", _group_of_solids),
    ("GroupEmpty", _empty_group),
    ("Group1", _group_one),
]

CATALOGUE_BY_NAME = dict(CATALOGUE)


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------


def _all_points(mob):
    points = []
    seen = set()
    for node in [mob, *mob.get_descendants()]:
        if id(node) in seen:
            continue
        seen.add(id(node))
        location = getattr(node, "location", None)
        if location is None:
            continue
        points.append(location.reshape(*location.shape[:-2], -1, 3))
    return points


def _scene_bounds_at(scene, time_index):
    """Bounds of every *visible* point in the Scene at one materialized time.

    Scanning the Scene rather than the returned Mob's hierarchy is deliberate:
    the PN route renders through a soup Mob that is a sibling of the result, so
    a hierarchy-only walk sees nothing mid-morph.  Visibility is opacity > 0,
    which materialization already zeroes outside a Mob's lifespan.
    """
    lo = None
    hi = None
    seen = set()
    for node in list(scene.actors):
        if id(node) in seen:
            continue
        seen.add(id(node))
        location = getattr(node, "location", None)
        opacity = getattr(node, "opacity", None)
        if location is None or opacity is None:
            continue
        if location.shape[0] <= time_index or opacity.shape[0] <= time_index:
            continue
        pts = location[time_index].reshape(-1, 3)
        op = opacity[time_index].reshape(-1)
        if pts.numel() == 0:
            continue
        if op.numel() == pts.shape[0]:
            keep = op > 1e-3
            if not bool(keep.any()):
                continue
            pts = pts[keep]
        elif op.numel() == 1:
            if float(op) <= 1e-3:
                continue
        node_lo = pts.amin(0)
        node_hi = pts.amax(0)
        lo = node_lo if lo is None else torch.minimum(lo, node_lo)
        hi = node_hi if hi is None else torch.maximum(hi, node_hi)
    return lo, hi


def _static_bounds(mob):
    lo = None
    hi = None
    seen = set()
    for node in [mob, *mob.get_descendants()]:
        if id(node) in seen:
            continue
        seen.add(id(node))
        location = getattr(node, "location", None)
        if location is None:
            continue
        pts = location.reshape(-1, 3)
        if pts.numel() == 0:
            continue
        opacity = getattr(node, "opacity", None)
        if opacity is not None:
            op = opacity.reshape(-1)
            if op.numel() == pts.shape[0]:
                keep = op > 1e-3
                if not bool(keep.any()):
                    continue
                pts = pts[keep]
        node_lo = pts.amin(0)
        node_hi = pts.amax(0)
        lo = node_lo if lo is None else torch.minimum(lo, node_lo)
        hi = node_hi if hi is None else torch.maximum(hi, node_hi)
    return lo, hi


def run_case(source_name, target_name, *, minimize_movement=False, strategy="auto"):
    """Return a dict describing what happened to one become() pair."""
    result = {
        "source": source_name,
        "target": target_name,
        "minimize_movement": minimize_movement,
        "strategy": strategy,
        "status": "ok",
        "problems": [],
        "error": None,
    }
    make_source = CATALOGUE_BY_NAME[source_name]
    make_target = CATALOGUE_BY_NAME[target_name]
    try:
        with Scene() as scene:
            with Off():
                source = make_source().spawn()
                target = make_target().move(RIGHT * 1.5)
            reference_lo, reference_hi = _static_bounds(target)
            source_lo, source_hi = _static_bounds(source)

            start = float(scene.animation_manager.context.timespan.current_time)
            with Sync(duration=1.0):
                morphed = source.become(
                    target,
                    minimize_movement=minimize_movement,
                    strategy=strategy,
                )
            end = float(scene.animation_manager.context.timespan.current_time)
            result["duration"] = end - start

            if morphed is None:
                result["problems"].append("become returned None")
                result["status"] = "problem"
                return result
            if not morphed.is_spawned() or morphed.is_despawned():
                result["problems"].append("returned Mob is not spawned")
            if not any(actor is morphed for actor in scene.actors):
                result["problems"].append("returned Mob is not a scene actor")

            times = [start, start + 0.25, start + 0.5, start + 0.75, end]
            scene.timeline_manager.set_state_to_times(torch.tensor(times))

            # NaN / inf anywhere.
            for points in _all_points(morphed):
                if not bool(torch.isfinite(points).all()):
                    result["problems"].append("non-finite location")
                    break

            # End state must be the target's occupancy.
            end_lo, end_hi = _scene_bounds_at(scene, len(times) - 1)
            if end_lo is None:
                result["problems"].append("nothing visible at the end of the morph")
            elif reference_lo is not None:
                tolerance = 0.08 * float(
                    (reference_hi - reference_lo).abs().max().clamp_min(0.5)
                )
                lo_error = float((end_lo - reference_lo).abs().max())
                hi_error = float((end_hi - reference_hi).abs().max())
                if max(lo_error, hi_error) > max(tolerance, 0.05):
                    result["problems"].append(
                        f"end bounds off target by {max(lo_error, hi_error):.3f} "
                        f"(tol {max(tolerance, 0.05):.3f}); "
                        f"got [{_fmt(end_lo)}]-[{_fmt(end_hi)}] "
                        f"want [{_fmt(reference_lo)}]-[{_fmt(reference_hi)}]"
                    )

            # Intermediate frames must stay near the union of both ends.
            if source_lo is not None and reference_lo is not None:
                union_lo = torch.minimum(source_lo, reference_lo)
                union_hi = torch.maximum(source_hi, reference_hi)
                span = (union_hi - union_lo).abs().max().clamp_min(1.0)
                slack = 0.75 * float(span)
                for index in range(1, len(times) - 1):
                    lo, hi = _scene_bounds_at(scene, index)
                    if lo is None:
                        result["problems"].append(
                            f"nothing visible at t={times[index]:.2f}"
                        )
                        continue
                    excursion = max(
                        float((union_lo - lo).amax()), float((hi - union_hi).amax())
                    )
                    if excursion > slack:
                        result["problems"].append(
                            f"t={times[index]:.2f} leaves the union bounds by "
                            f"{excursion:.2f} (slack {slack:.2f})"
                        )
                        break

            result["actors"] = len(scene.actors)
            if result["problems"]:
                result["status"] = "problem"
    except Exception as exc:  # noqa: BLE001 - the whole point is to catalogue these
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    return result


def _fmt(vector):
    return ",".join(f"{float(value):.2f}" for value in vector)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", default=None)
    parser.add_argument("--pairs", type=int, default=0)
    parser.add_argument("--minimize", action="store_true")
    parser.add_argument("--only", nargs=2, default=None, metavar=("SRC", "DST"))
    parser.add_argument("--traceback", action="store_true")
    args = parser.parse_args()

    names = [name for name, _ in CATALOGUE]
    if args.filter:
        names = [name for name in names if args.filter.lower() in name.lower()]

    if args.only:
        cases = [tuple(args.only)]
    else:
        cases = list(itertools.product(names, names))
        if args.pairs:
            cases = cases[: args.pairs]

    errors = []
    problems = []
    ok = 0
    for source_name, target_name in cases:
        result = run_case(source_name, target_name, minimize_movement=args.minimize)
        if result["status"] == "error":
            errors.append(result)
            print(f"ERROR   {source_name:>14} -> {target_name:<14} {result['error']}")
            if args.traceback:
                print(result["traceback"])
        elif result["status"] == "problem":
            problems.append(result)
            print(f"PROBLEM {source_name:>14} -> {target_name:<14}")
            for problem in result["problems"]:
                print(f"          {problem}")
        else:
            ok += 1
        sys.stdout.flush()

    print()
    print(
        f"{ok} ok, {len(problems)} problems, {len(errors)} errors "
        f"out of {len(cases)} cases"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
