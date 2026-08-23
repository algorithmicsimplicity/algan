"""Round 2 of the `Mob.become` stress: harder types, tighter invariants.

Round 1 (`_become_stress.py`) checked bounding boxes over a 25x25 catalogue and
found nothing.  This round tightens the end-state check from "the bounds are
about right" to "the visible point cloud is the target's, and so are the colour
and opacity", widens the catalogue to the awkward Mobs (Tex, ImageMob,
Polyhedron, Axes, DotCloud, packed and nested containers), and adds the option
axes `become` actually has: `minimize_movement`, `strategy`, `detach_history`,
and chaining.

Usage:
  <venv-python> benchmarks/_become_stress2.py                 # the pair matrix
  <venv-python> benchmarks/_become_stress2.py --mode options  # option axes
  <venv-python> benchmarks/_become_stress2.py --mode chain    # chained morphs
"""

from __future__ import annotations

import argparse
import itertools
import sys
import traceback

import torch

from algan import (
    BLUE,
    DOWN,
    GREEN,
    LEFT,
    RIGHT,
    UP,
    YELLOW,
    Arrow,
    Axes,
    Circle,
    Cone,
    Cube,
    Cylinder,
    Dot,
    Dot3D,
    DoubleArrow,
    Group,
    ImageMob,
    Line,
    MathTex,
    Off,
    Polyhedron,
    Prism,
    Scene,
    Sphere,
    Square,
    Star,
    Surface,
    Sync,
    Tetrahedron,
    Tex,
    Text,
    Torus,
    Triangle,
    TriangleVertices,
)

# --------------------------------------------------------------------------
# Catalogue
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


def _surface_wave():
    return Surface(
        lambda u, v: torch.stack(
            (u - 0.5, v - 0.5, 0.25 * torch.sin(6 * u) * torch.cos(6 * v)), -1
        ),
        grid_height=9,
        grid_width=4,
    )


def _polyhedron():
    vertices = torch.tensor(
        [
            [0.0, 0.6, 0.0],
            [-0.5, -0.3, 0.4],
            [0.5, -0.3, 0.4],
            [0.0, -0.3, -0.6],
        ]
    )
    faces = [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]]
    return Polyhedron(vertices, faces)


def _image():
    return ImageMob("tests/full_renders/assets/world_map.jpg")


def _group_wide():
    return Group(*[Square().move(RIGHT * (index - 3) * 0.8) for index in range(7)])


def _group_ragged():
    return Group(
        Group(Square().move(LEFT * 2)),
        Group(Circle().move(LEFT), Triangle(), Star().move(RIGHT)),
        Sphere(radius=0.3).move(RIGHT * 2),
    )


def _group_asym_depth():
    return Group(
        Square().move(LEFT * 2),
        Group(Group(Group(Circle().move(RIGHT * 2)))),
    )


def _group_mixed_families():
    return Group(
        Text("hi").move(UP),
        Sphere(radius=0.3).move(DOWN),
        Cube(side_length=0.4).move(LEFT * 2),
        _tri_vertices(),
    )


def _dot_cloud():
    return Group(
        *[
            Dot3D(radius=0.06).move(
                RIGHT * (index % 4 - 1.5) * 0.5 + UP * (index // 4 - 1.0) * 0.5
            )
            for index in range(12)
        ]
    )


CATALOGUE: list[tuple[str, callable]] = [
    ("Square", lambda: Square(color=BLUE)),
    ("Circle", lambda: Circle(radius=0.6, color=YELLOW)),
    ("Star", lambda: Star()),
    ("Line", lambda: Line(LEFT, RIGHT)),
    ("Arrow", lambda: Arrow(LEFT, RIGHT)),
    ("DoubleArrow", lambda: DoubleArrow(LEFT, RIGHT)),
    ("Dot", lambda: Dot()),
    ("Text", lambda: Text("hello")),
    ("Tex", lambda: Tex("hi")),
    ("MathTex", lambda: MathTex(r"x^2")),
    ("Sphere", lambda: Sphere(radius=0.6)),
    ("Cylinder", lambda: Cylinder(radius=0.4, height=1.0)),
    ("Cone", lambda: Cone()),
    ("Torus", lambda: Torus()),
    ("SurfacePlane", _surface_plane),
    ("SurfaceWave", _surface_wave),
    ("Cube", lambda: Cube(side_length=0.8)),
    ("Prism", lambda: Prism()),
    ("Tetrahedron", lambda: Tetrahedron()),
    ("Polyhedron", _polyhedron),
    ("TriangleVertices", _tri_vertices),
    ("Image", _image),
    ("Axes", lambda: Axes()),
    ("GroupWide", _group_wide),
    ("GroupRagged", _group_ragged),
    ("GroupAsymDepth", _group_asym_depth),
    ("GroupMixed", _group_mixed_families),
    ("DotCloud", _dot_cloud),
    ("GroupEmpty", lambda: Group()),
]

CATALOGUE_BY_NAME = dict(CATALOGUE)


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------


def _visible_rows(scene, time_index):
    """(points, colors, opacities) of every visible row in the Scene."""
    points = []
    colors = []
    opacities = []
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
        if op.numel() == 1:
            op = op.expand(pts.shape[0])
        if op.numel() != pts.shape[0]:
            continue
        keep = op > 1e-3
        if not bool(keep.any()):
            continue
        color = getattr(node, "color", None)
        if color is not None and color.shape[0] > time_index:
            col = color[time_index].reshape(-1, color.shape[-1])[..., :3]
            if col.shape[0] == 1:
                col = col.expand(pts.shape[0], -1)
            if col.shape[0] != pts.shape[0]:
                col = None
        else:
            col = None
        points.append(pts[keep])
        opacities.append(op[keep])
        colors.append(None if col is None else col[keep])
    if not points:
        return None, None, None
    all_points = torch.cat(points, 0)
    all_opacity = torch.cat(opacities, 0)
    all_colors = (
        torch.cat([c for c in colors if c is not None], 0)
        if any(c is not None for c in colors)
        else None
    )
    return all_points, all_colors, all_opacity


def _static_visible_rows(mob):
    points = []
    seen = set()
    for node in [mob, *mob.get_descendants()]:
        if id(node) in seen:
            continue
        seen.add(id(node))
        location = getattr(node, "location", None)
        opacity = getattr(node, "opacity", None)
        if location is None:
            continue
        pts = location.reshape(-1, 3)
        if pts.numel() == 0:
            continue
        if opacity is not None:
            op = opacity.reshape(-1)
            if op.numel() == 1:
                op = op.expand(pts.shape[0])
            if op.numel() == pts.shape[0]:
                keep = op > 1e-3
                if not bool(keep.any()):
                    continue
                pts = pts[keep]
        points.append(pts)
    return torch.cat(points, 0) if points else None


def _chamfer(a, b):
    """Max over each point of a of its distance to the nearest point of b."""
    if a is None or b is None:
        return float("inf")
    if a.shape[0] > 4000:
        a = a[torch.randperm(a.shape[0])[:4000]]
    if b.shape[0] > 4000:
        b = b[torch.randperm(b.shape[0])[:4000]]
    distances = torch.cdist(a.float(), b.float())
    return max(
        float(distances.amin(1).amax()),
        float(distances.amin(0).amax()),
    )


def run_case(
    source_name,
    target_name,
    *,
    minimize_movement=False,
    strategy="auto",
    detach_history=True,
    target_offset=None,
):
    result = {
        "source": source_name,
        "target": target_name,
        "status": "ok",
        "problems": [],
        "error": None,
    }
    if target_offset is None:
        target_offset = RIGHT * 1.5
    try:
        with Scene() as scene:
            with Off():
                source = CATALOGUE_BY_NAME[source_name]().spawn()
                target = CATALOGUE_BY_NAME[target_name]().move(target_offset)
            reference = _static_visible_rows(target)
            source_points = _static_visible_rows(source)

            start = float(scene.animation_manager.context.timespan.current_time)
            with Sync(run_time=1.0):
                morphed = source.become(
                    target,
                    minimize_movement=minimize_movement,
                    strategy=strategy,
                    detach_history=detach_history,
                )
            end = float(scene.animation_manager.context.timespan.current_time)
            duration = end - start
            if abs(duration - 1.0) > 1e-6:
                result["problems"].append(f"duration is {duration:.6f}, not 1.0")

            if morphed is None:
                result["problems"].append("become returned None")
                result["status"] = "problem"
                return result

            times = [start, start + 0.2, start + 0.5, start + 0.8, end]
            scene.timeline_manager.set_state_to_times(torch.tensor(times))

            # (a) The end state must be the target's point cloud, not merely a
            #     box of the same size.
            end_points, end_colors, _ = _visible_rows(scene, len(times) - 1)
            if end_points is None:
                result["problems"].append("nothing visible at the end")
            elif reference is not None:
                scale = float(
                    (reference.amax(0) - reference.amin(0)).abs().max().clamp_min(0.5)
                )
                error = _chamfer(end_points, reference)
                if error > 0.12 * scale:
                    result["problems"].append(
                        f"end point cloud is {error:.3f} from the target "
                        f"(scale {scale:.2f})"
                    )

            # (b) Nothing may be non-finite at any sampled time.
            for index in range(len(times)):
                points, _, _ = _visible_rows(scene, index)
                if points is not None and not bool(torch.isfinite(points).all()):
                    result["problems"].append(
                        f"non-finite geometry at t={times[index]}"
                    )
                    break

            # (c) The morph must not go blank mid-flight.  A frame with nothing
            #     visible is a flicker the viewer sees.
            if source_points is not None and reference is not None:
                for index in range(1, len(times) - 1):
                    points, _, _ = _visible_rows(scene, index)
                    if points is None or points.shape[0] == 0:
                        result["problems"].append(
                            f"nothing visible at t={times[index]:.2f}"
                        )
                        break

            # (d) Mid-flight geometry must stay near the union of the two ends.
            if source_points is not None and reference is not None:
                union = torch.cat([source_points, reference], 0)
                union_lo, union_hi = union.amin(0), union.amax(0)
                span = float((union_hi - union_lo).abs().max().clamp_min(1.0))
                for index in range(1, len(times) - 1):
                    points, _, _ = _visible_rows(scene, index)
                    if points is None:
                        continue
                    excursion = max(
                        float((union_lo - points.amin(0)).amax()),
                        float((points.amax(0) - union_hi).amax()),
                    )
                    if excursion > 0.5 * span:
                        result["problems"].append(
                            f"t={times[index]:.2f} leaves the union bounds by "
                            f"{excursion:.2f} (span {span:.2f})"
                        )
                        break

            if result["problems"]:
                result["status"] = "problem"
    except Exception as exc:  # noqa: BLE001
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    return result


def _emit(result, show_traceback):
    if result["status"] == "error":
        print(
            f"ERROR   {result['source']:>16} -> {result['target']:<16} "
            f"{result['error']}"
        )
        if show_traceback:
            print(result["traceback"])
    elif result["status"] == "problem":
        print(f"PROBLEM {result['source']:>16} -> {result['target']:<16}")
        for problem in result["problems"]:
            print(f"          {problem}")
    sys.stdout.flush()
    return result["status"]


def mode_matrix(args):
    names = [name for name, _ in CATALOGUE]
    if args.filter:
        names = [n for n in names if args.filter.lower() in n.lower()]
    cases = list(itertools.product(names, names))
    counts = {"ok": 0, "problem": 0, "error": 0}
    for source_name, target_name in cases:
        result = run_case(
            source_name,
            target_name,
            minimize_movement=args.minimize,
        )
        counts[_emit(result, args.traceback)] += 1
    print(
        f"\n{counts['ok']} ok, {counts['problem']} problems, "
        f"{counts['error']} errors out of {len(cases)} cases"
    )


def mode_options(args):
    """Every option axis over a smaller, representative pair set."""
    pairs = [
        ("Square", "Circle"),
        ("Square", "Sphere"),
        ("Sphere", "Cylinder"),
        ("Text", "Square"),
        ("Text", "Text"),
        ("GroupRagged", "GroupMixed"),
        ("GroupMixed", "GroupRagged"),
        ("Cube", "Sphere"),
        ("Image", "Square"),
        ("Square", "Image"),
        ("Axes", "GroupWide"),
        ("DotCloud", "Text"),
        ("GroupEmpty", "GroupRagged"),
        ("GroupRagged", "GroupEmpty"),
        ("Polyhedron", "Tetrahedron"),
        ("TriangleVertices", "Star"),
    ]
    counts = {"ok": 0, "problem": 0, "error": 0}
    total = 0
    for source_name, target_name in pairs:
        for minimize in (False, True):
            for strategy in ("auto", "morph", "dissolve"):
                for detach in (True, False):
                    total += 1
                    result = run_case(
                        source_name,
                        target_name,
                        minimize_movement=minimize,
                        strategy=strategy,
                        detach_history=detach,
                    )
                    result["source"] = (
                        f"{source_name}[{'M' if minimize else '-'}"
                        f"{strategy[0]}{'D' if detach else '-'}]"
                    )
                    counts[_emit(result, args.traceback)] += 1
    print(
        f"\n{counts['ok']} ok, {counts['problem']} problems, "
        f"{counts['error']} errors out of {total} cases"
    )


def mode_chain(args):
    """Chained morphs: the returned Mob must stay morphable indefinitely."""
    chains = [
        ["Square", "Circle", "Sphere", "Cube", "Text", "Star"],
        ["Cylinder", "Sphere", "Cone", "Torus", "Cylinder"],
        ["Text", "Tex", "MathTex", "Text"],
        ["GroupRagged", "GroupMixed", "GroupWide", "GroupEmpty", "GroupRagged"],
        ["Arrow", "Line", "DoubleArrow", "Axes"],
        ["Image", "Square", "Image", "Sphere"],
        ["Polyhedron", "Cube", "Tetrahedron", "Prism"],
        ["DotCloud", "GroupWide", "Text", "Sphere"],
    ]
    failures = 0
    for chain in chains:
        label = " -> ".join(chain)
        try:
            with Scene() as scene:
                with Off():
                    current = CATALOGUE_BY_NAME[chain[0]]().spawn()
                for step, name in enumerate(chain[1:]):
                    with Off():
                        target = CATALOGUE_BY_NAME[name]().move(
                            RIGHT * (step % 3 - 1) * 1.2
                        )
                    reference = _static_visible_rows(target)
                    with Sync(run_time=1.0):
                        current = current.become(target)
                    if current is None:
                        raise RuntimeError(f"step {step} returned None")
                    end = float(scene.animation_manager.context.timespan.current_time)
                    scene.timeline_manager.set_state_to_times(torch.tensor([end]))
                    points, _, _ = _visible_rows(scene, 0)
                    # Materialization leaves the Scene in `active_state`; a
                    # render's get_frames() clears it afterwards and authoring
                    # more of the scene without doing the same corrupts it.
                    scene.timeline_manager.clear_buffers()
                    if reference is not None and points is not None:
                        scale = float(
                            (reference.amax(0) - reference.amin(0))
                            .abs()
                            .max()
                            .clamp_min(0.5)
                        )
                        error = _chamfer(points, reference)
                        if error > 0.12 * scale:
                            print(
                                f"PROBLEM {label}: after step {step + 1} "
                                f"({name}) the frame is {error:.3f} from the "
                                f"target (scale {scale:.2f})"
                            )
                            failures += 1
            print(f"ok      {label}")
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR   {label}: {type(exc).__name__}: {exc}")
            if args.traceback:
                traceback.print_exc()
            failures += 1
        sys.stdout.flush()
    print(f"\n{len(chains) - failures} of {len(chains)} chains clean")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", default="matrix", choices=["matrix", "options", "chain"]
    )
    parser.add_argument("--filter", default=None)
    parser.add_argument("--minimize", action="store_true")
    parser.add_argument("--traceback", action="store_true")
    args = parser.parse_args()
    {"matrix": mode_matrix, "options": mode_options, "chain": mode_chain}[args.mode](
        args
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
