"""Logical PN dice: byte-equality and speed against the pre-dedup reference.

``LogicalPNTrianglePrimitive._dice_logical_pn`` turns each logical PN patch
into flat triangles once per materialized camera frame. The reference arm here
is the version that shipped before the temporal-coherence work: it rebuilt the
control nets on every frame of the batch, evaluated every patch once per
(frame, patch) pair, and interpolated attributes at every microtriangle corner.

Both arms must agree BIT FOR BIT on every diced array -- corners, normals,
colours, the surface/shader parameters, uvs, the padding mask, the levels and
the per-row surface ids. The optimizations are all "compute the same value
fewer times", so anything else is a bug, not a rounding difference.

Arms alternate in one process: cross-process wall clock on this project's
machines swings ~2x with thermal state, and the dice is a few hundred
milliseconds, well inside that noise.

Run: <venv-python> benchmarks/_pn_dice_ab.py [--reps N] [scenario ...]
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from types import SimpleNamespace

os.environ.setdefault("ALGAN_USE_DAEMON", "0")
os.environ.setdefault("ALGAN_PREFETCH_BATCHES", "0")

import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch.nn.functional as F  # noqa: E402

from algan.mobs.shapes_3d import Cone, Cylinder, Sphere, Torus  # noqa: E402
from algan.rendering.logical_pn import (  # noqa: E402
    dice_pattern,
    dice_triangle_count,
    evaluate_logical_pn,
    evaluate_logical_pn_normals,
    interpolate_patch_attribute,
    logical_pn_control_points,
    logical_pn_edge_control_points,
    logical_pn_normal_control_points,
    mean_patch_edge_length,
    snap_boundary_values,
)
from algan.rendering.raytracing.primitives import (  # noqa: E402
    LogicalPNTrianglePrimitive,
    _scatter_diced_rows,
)
from algan.rendering.raytracing.settings import (  # noqa: E402
    mesh_id,
    pn_geometry_slack,
)
from algan.rendering.raytracing.utils import _expand_frames, _flat_frames  # noqa: E402
from algan.scene_manager import SceneManager  # noqa: E402

SHAPES = {"sphere": Sphere, "torus": Torus, "cylinder": Cylinder, "cone": Cone}

SCENARIOS = {
    # name: shape, how many copies batched into one primitive, frames in the
    # batch, camera radius, how far the camera orbits, and how far the mesh
    # deforms over the batch (0 = a mesh that holds still).
    "static": {
        "shape": "sphere",
        "copies": 4,
        "frames": 16,
        "radius": 6.0,
        "orbit": 0.0,
    },
    "orbit": {
        "shape": "sphere",
        "copies": 4,
        "frames": 32,
        "radius": 6.0,
        "orbit": 0.6,
    },
    "near": {"shape": "sphere", "copies": 1, "frames": 24, "radius": 2.2, "orbit": 0.4},
    "torus": {"shape": "torus", "copies": 2, "frames": 16, "radius": 5.0, "orbit": 0.5},
    "cylinder": {
        "shape": "cylinder",
        "copies": 2,
        "frames": 16,
        "radius": 4.0,
        "orbit": 0.3,
    },
    "cone": {"shape": "cone", "copies": 2, "frames": 12, "radius": 3.0, "orbit": 0.2},
    # A genuinely deforming mesh: no frame shares another's geometry, so every
    # dedup in the fast arm has to disable itself and the two arms should land
    # on the same time as well as the same bytes.
    "deforming": {
        "shape": "sphere",
        "copies": 2,
        "frames": 12,
        "radius": 5.0,
        "orbit": 0.3,
        "deform": 0.9,
    },
}


# --------------------------------------------------------------------------
# The reference dice (the shipped-before version), kept verbatim in structure.
# --------------------------------------------------------------------------
def reference_dice(self, camera):
    num_frames = int(camera.ray_origin.shape[0])
    source_corners = self.corners.float()
    source_normals = self.normals.float()
    device = source_corners.device
    dtype = source_corners.dtype
    cam_o = _expand_frames(_flat_frames(camera.ray_origin, (3,)), num_frames).to(device)
    sp = _expand_frames(_flat_frames(camera.screen_point, (3,)), num_frames).to(device)
    sb = _expand_frames(_flat_frames(camera.screen_basis, (3, 3)), num_frames).to(
        device
    )

    expand = self._expanded_frames
    control_points = expand(
        logical_pn_control_points(source_corners, source_normals),
        num_frames,
        "logical PN corners",
    )
    normal_control_points = expand(
        logical_pn_normal_control_points(source_corners, source_normals),
        num_frames,
        "logical PN normals",
    )
    edge_controls = expand(
        logical_pn_edge_control_points(source_corners, source_normals),
        num_frames,
        "logical PN edges",
    )
    output_height = getattr(camera, "output_screen_height", camera.screen_height)
    # Same criterion inputs as production: the arms differ in how the dice is
    # written out, never in which dice the search picks.
    slack = None
    if pn_geometry_slack and self.geometry_slack_ratio > 0:
        slack = _expand_frames(
            mean_patch_edge_length(source_corners) * self.geometry_slack_ratio,
            num_frames,
        )
    levels, edge_levels, apex_levels, across_levels = self._required_subdivision_levels(
        control_points, edge_controls, cam_o, sp, sb, output_height, False, slack
    )

    counts = dice_triangle_count(levels, across_levels)
    offsets = counts.cumsum(1) - counts
    max_triangles = int(counts.sum(1).amax().item()) if counts.numel() else 0

    num_patches = counts.shape[1] if counts.ndim > 1 else 0
    counts_src = getattr(self, "_obj_counts", None)
    obj_ids = getattr(self, "_obj_ids", None) if mesh_id else None
    if obj_ids is not None:
        patch_source = obj_ids.reshape(-1).to(device=device, dtype=torch.int32)
        self._logical_pn_tri_obj_n = int(self._obj_ids_n)
    elif counts_src:
        patch_source = torch.repeat_interleave(
            torch.arange(len(counts_src), dtype=torch.int32, device=device),
            torch.tensor(counts_src, dtype=torch.int64, device=device),
        )
        self._logical_pn_tri_obj_n = len(counts_src)
    else:
        patch_source = torch.zeros((num_patches,), dtype=torch.int32, device=device)
        self._logical_pn_tri_obj_n = 1
    if num_patches and max_triangles:
        ends = counts.cumsum(1)
        cols = torch.arange(max_triangles, device=device)
        patch_of_col = torch.searchsorted(
            ends.contiguous(),
            cols.unsqueeze(0).expand(num_frames, -1).contiguous(),
            right=True,
        ).clamp_max(num_patches - 1)
        self._logical_pn_tri_obj = (
            patch_source[patch_of_col].to(torch.int32).contiguous()
        )
    else:
        self._logical_pn_tri_obj = torch.zeros(
            (num_frames, max_triangles), dtype=torch.int32, device=device
        )

    colors = expand(self.colors.float(), num_frames, "logical PN colors")
    surface_sources = {
        name: expand(getattr(self, name), num_frames, f"logical PN {name}")
        for name in self._surface_params
    }
    shader_sources = [
        expand(value, num_frames, "logical PN shader parameter")
        for value in self.shader_param_values
    ]
    uv_source = expand(self.uvs, num_frames, "logical PN UVs")

    def allocate(values):
        return torch.zeros(
            (num_frames, max_triangles, 3, values.shape[-1]),
            device=values.device,
            dtype=values.dtype,
        )

    diced_corners = allocate(source_corners)
    diced_normals = allocate(source_normals)
    diced_colors = allocate(colors)
    diced_surface_params = {
        name: allocate(source) for name, source in surface_sources.items()
    }
    diced_shader_params = [allocate(v) for v in shader_sources]
    diced_uvs = allocate(uv_source) if uv_source is not None else None
    padding = torch.ones((num_frames, max_triangles), dtype=torch.bool, device=device)

    # One group per dice shape, as production does: the reference arm's job is
    # to reproduce the *write-out* the un-deduped code did, not to second-guess
    # which dice the search chose.
    shape_keys = (
        levels * (self.max_subdivision_level + 1) + across_levels
    ) * 3 + apex_levels
    for key in shape_keys.unique(sorted=True).tolist():
        key = int(key)
        pattern = dice_pattern(
            key // (3 * (self.max_subdivision_level + 1)),
            (key // 3) % (self.max_subdivision_level + 1),
            key % 3,
            device=device,
            dtype=dtype,
        )
        selected = (shape_keys == key).nonzero()
        vertex_uv = pattern.vertex_uv
        triangle_indices = pattern.triangle_indices
        corner_uv = vertex_uv[triangle_indices]
        boundary = pattern.boundary
        num_triangles = triangle_indices.shape[0]
        columns = torch.arange(num_triangles, device=device)
        chunk = max(1, int(self.max_scratch_triangles) // num_triangles)

        for start in range(0, selected.shape[0], chunk):
            rows = selected[start : start + chunk]
            frames, patches = rows[:, 0], rows[:, 1]
            edges = edge_levels[frames, patches]
            positions = snap_boundary_values(
                evaluate_logical_pn(
                    control_points[frames, patches].unsqueeze(0), vertex_uv
                )[0],
                pattern.edge_levels,
                edges,
                boundary,
            )
            vertex_normals = F.normalize(
                snap_boundary_values(
                    evaluate_logical_pn_normals(
                        normal_control_points[frames, patches].unsqueeze(0), vertex_uv
                    )[0],
                    pattern.edge_levels,
                    edges,
                    boundary,
                ),
                p=2,
                dim=-1,
            )
            targets = (
                frames.unsqueeze(1) * max_triangles
                + offsets[frames, patches].unsqueeze(1)
                + columns
            ).reshape(-1)

            _scatter_diced_rows(diced_corners, positions[:, triangle_indices], targets)
            _scatter_diced_rows(
                diced_normals, vertex_normals[:, triangle_indices], targets
            )
            _scatter_diced_rows(
                diced_colors,
                interpolate_patch_attribute(colors[frames, patches], corner_uv),
                targets,
            )
            for name, output in diced_surface_params.items():
                _scatter_diced_rows(
                    output,
                    interpolate_patch_attribute(
                        surface_sources[name][frames, patches], corner_uv
                    ),
                    targets,
                )
            for output, source in zip(diced_shader_params, shader_sources):
                _scatter_diced_rows(
                    output,
                    interpolate_patch_attribute(source[frames, patches], corner_uv),
                    targets,
                )
            if diced_uvs is not None:
                _scatter_diced_rows(
                    diced_uvs,
                    interpolate_patch_attribute(uv_source[frames, patches], corner_uv),
                    targets,
                )
            padding.view(-1).index_fill_(0, targets, False)

    self.corners = diced_corners
    self.normals = diced_normals
    self.colors = diced_colors
    for name, values in diced_surface_params.items():
        setattr(self, name, values)
    self.shader_param_values = diced_shader_params
    self.uvs = diced_uvs
    self._logical_pn_padding = padding
    self._logical_pn_subdivision_levels = levels
    self._logical_pn_edge_levels = edge_levels
    self._logical_pn_across_levels = across_levels
    self._logical_pn_apex = apex_levels
    self._logical_pn_triangle_counts = counts


# --------------------------------------------------------------------------
# Scenario construction
# --------------------------------------------------------------------------
def build_primitive(shape, copies):
    SceneManager.reset()
    mob = SHAPES[shape]()
    mob.spawn()
    primitives = mob.get_render_primitives()
    primitives = (
        list(primitives) if isinstance(primitives, (list, tuple)) else [primitives]
    )
    members = [p for p in primitives if isinstance(p, LogicalPNTrianglePrimitive)]
    if not members:
        raise RuntimeError(f"{shape} produced no logical PN primitive")
    return LogicalPNTrianglePrimitive(triangle_collection=members * copies)


def build_camera(frames, radius, orbit, screen_height=486):
    angle = torch.linspace(0.0, orbit, frames)
    distance = (
        radius * (1.0 + 0.3 * torch.linspace(0.0, 1.0, frames)) if orbit else radius
    )
    origins = torch.stack(
        (
            distance * torch.sin(angle),
            torch.zeros_like(angle),
            -distance * torch.cos(angle),
        ),
        dim=-1,
    ).unsqueeze(1)
    forward = -origins / origins.norm(dim=-1, keepdim=True)
    up = torch.tensor([0.0, 1.0, 0.0]).expand(frames, 3)
    right = torch.cross(up, forward.squeeze(1), dim=-1)
    right = right / right.norm(dim=-1, keepdim=True)
    return SimpleNamespace(
        ray_origin=origins,
        screen_point=origins + forward,
        screen_basis=torch.stack((right, up, forward.squeeze(1)), dim=1),
        screen_width=864,
        screen_height=screen_height,
        output_screen_width=864,
        output_screen_height=screen_height,
        analytic_raster=False,
    )


ATTRIBUTES = ("corners", "normals", "colors", "uvs")


def source_state(primitive, frames, deform=0.0):
    """The primitive's source arrays, materialized one row per frame.

    A real render hands the dice per-frame rows whether or not the mob moved,
    which is the whole point of the collapse the fast arm does; building the
    scenario any other way would measure a case that never happens.
    """
    state = {}
    for name in (*ATTRIBUTES, *primitive._surface_params):
        value = getattr(primitive, name, None)
        if value is None:
            continue
        if value.shape[0] == 1:
            value = value.expand(frames, *value.shape[1:])
        state[name] = value.contiguous()
    state["shader_param_values"] = [
        (v.expand(frames, *v.shape[1:]) if v.shape[0] == 1 else v).contiguous()
        for v in primitive.shader_param_values
    ]
    if deform:
        offsets = torch.linspace(0.0, deform, frames).view(-1, 1, 1, 1)
        state["corners"] = (state["corners"] + offsets).contiguous()
    return state


def restore(primitive, state):
    for name, value in state.items():
        if name == "shader_param_values":
            primitive.shader_param_values = [v.clone() for v in value]
        else:
            setattr(primitive, name, value.clone())


def diced_arrays(primitive):
    out = {
        "corners": primitive.corners,
        "normals": primitive.normals,
        "colors": primitive.colors,
        "uvs": primitive.uvs,
        "padding": primitive._logical_pn_padding,
        "levels": primitive._logical_pn_subdivision_levels,
        "edge_levels": primitive._logical_pn_edge_levels,
        "across_levels": primitive._logical_pn_across_levels,
        "apex": primitive._logical_pn_apex,
        "tri_obj": primitive._logical_pn_tri_obj,
    }
    for name in primitive._surface_params:
        out[name] = getattr(primitive, name)
    for index, value in enumerate(primitive.shader_param_values):
        out[f"shader_param_{index}"] = value
    return {k: v for k, v in out.items() if v is not None}


def bitwise_equal(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype.is_floating_point:
        bits = torch.int32 if a.dtype == torch.float32 else torch.int64
        return torch.equal(a.contiguous().view(bits), b.contiguous().view(bits))
    return torch.equal(a, b)


def time_dice(primitive, state, camera, dice, reps):
    samples = []
    for _ in range(reps):
        restore(primitive, state)
        start = time.perf_counter()
        dice(primitive, camera)
        samples.append(time.perf_counter() - start)
    return samples


def run(name, spec, reps):
    primitive = build_primitive(spec["shape"], spec["copies"])
    frames = spec["frames"]
    state = source_state(primitive, frames, spec.get("deform", 0.0))
    camera = build_camera(frames, spec["radius"], spec["orbit"])
    dice_fast = LogicalPNTrianglePrimitive._dice_logical_pn

    restore(primitive, state)
    reference_dice(primitive, camera)
    expected = {k: v.clone() for k, v in diced_arrays(primitive).items()}
    restore(primitive, state)
    dice_fast(primitive, camera)
    got = diced_arrays(primitive)

    mismatches = [key for key in expected if not bitwise_equal(got[key], expected[key])]
    triangles = expected["corners"].shape[1]

    # Alternate the arms so a thermal drift lands on both.
    reference_times, fast_times = [], []
    for _ in range(reps):
        reference_times += time_dice(primitive, state, camera, reference_dice, 1)
        fast_times += time_dice(primitive, state, camera, dice_fast, 1)

    reference = min(reference_times)
    fast = min(fast_times)
    status = "BIT-IDENTICAL" if not mismatches else f"MISMATCH {mismatches}"
    print(
        f"{name:<10} {frames:>3}f x {expected['levels'].shape[1]:>5}patch "
        f"-> {triangles:>6} tri/frame | "
        f"reference {reference * 1e3:8.1f} ms  fast {fast * 1e3:8.1f} ms  "
        f"{reference / fast:5.2f}x  (median {statistics.median(reference_times) * 1e3:.1f}"
        f" / {statistics.median(fast_times) * 1e3:.1f})  {status}"
    )
    return not mismatches


def main(argv):
    reps = 5
    if "--reps" in argv:
        index = argv.index("--reps")
        reps = int(argv[index + 1])
        del argv[index : index + 2]
    names = argv or list(SCENARIOS)
    ok = True
    for name in names:
        ok &= run(name, SCENARIOS[name], reps)
    print("\nall arms byte-identical" if ok else "\nBYTE MISMATCH -- see above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
