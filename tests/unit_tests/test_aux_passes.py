"""The auxiliary compositing passes ``Scene.get_frames(aux_passes=True)`` hands on.

What a pass records is fixed by one rule: the first surface whose alpha reaches
0.5 along ONE pinhole ray through the pixel centre, at the output resolution.
These pin that rule and everything the render loop must do around it:

* the values -- planar depth (not slant range), a camera-space normal facing
  the viewer, and rows top-down like the frames;
* the threshold -- a surface at opacity 0.3 is seen through, one at 0.7 is not;
* independence from the renderer -- the deterministic and the path-traced
  render of a scene, and the analytic and supersampled routes, produce
  IDENTICAL passes;
* delivery -- one dict per yielded batch, in frame order, through sparse
  ``frame_indices`` renders, background-only windows and the tracer's
  out-of-memory split, and nothing at all when the passes are off;
* the Mob-id gather through the (optional) host tables, and the camera clip
  planes.

Every test renders, so none is ``fast``: what they catch is a change to the
aux trace or its plumbing, which is this feature's own code.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from algan import (
    BLACK,
    BLUE,
    LEFT,
    OUT,
    RED,
    RIGHT,
    SMOKE_TEST,
    UP,
    WHITE,
    CameraView,
    Cube,
    Off,
    RectAreaLight,
    Scene,
    Sphere,
    Square,
)
from algan.rendering.raytracing import aux_passes as aux_module
from algan.rendering.raytracing import scene_builder, tracer
from algan.scene_manager import SceneManager
from algan.settings import SETTINGS
from algan.settings.kernel_settings import KERNEL_REGISTRY
from algan.utils.memory_utils import InsufficientMemoryException

TINY = SMOKE_TEST.set(resolution=(64, 48))
H, W = 48, 64
#: A pixel whose centre ray passes within a fifth of a unit of the axis.
CY, CX = 24, 32


def _render(build, start=0, end=1, *, animate=None, video=TINY, aux=True, **options):
    """Render ``[start, end)`` of the scene ``build`` makes; collect its passes.

    ``build(scene)`` runs under ``Off()`` (immediate), ``animate(scene)`` after
    it is recorded. ``rt`` / ``experimental`` are raytracing settings for the
    render, ``camera`` camera attributes, ``frame_indices`` goes to
    ``get_frames``. Returns ``(frames, auxes, info)`` with ``info`` holding
    the camera location and forward axis.
    """
    snapshot = SETTINGS.snapshot()
    SceneManager.reset()
    auxes = []
    try:
        SETTINGS.raytracing.set(denoise=False, **options.get("rt", {}))
        for key, value in options.get("experimental", {}).items():
            SETTINGS.raytracing.experimental.set(**{key: value})
        with Scene(video_settings=video) as scene:
            with Off():
                build(scene)
            if animate is not None:
                animate(scene)
            for name, value in options.get("camera", {}).items():
                setattr(scene.camera, name, value)
            info = {
                "location": scene.camera.location.reshape(-1)[:3].clone(),
                "forward": scene.camera.get_forward_direction().reshape(-1)[:3].clone(),
            }
            kwargs = {"aux_passes": bool(aux), "aux_sink": auxes.append}
            if options.get("frame_indices") is not None:
                kwargs["frame_indices"] = options["frame_indices"]
            frames = [batch.clone() for batch in scene.get_frames(start, end, **kwargs)]
    finally:
        SceneManager.reset()
        SETTINGS.restore(snapshot)
    return frames, auxes, info


def _joined(auxes):
    return {
        key: torch.cat([aux[key] for aux in auxes], 0)
        for key in aux_module.AUX_PASS_KEYS
    }


def _check_layout(frames, auxes):
    """One dict per batch, CPU tensors of the documented dtypes and shapes."""
    assert len(auxes) == len(frames)
    for batch, aux in zip(frames, auxes):
        assert set(aux) == set(aux_module.AUX_PASS_KEYS)
        n = len(batch)
        assert aux["depth"].shape == (n, H, W)
        assert aux["depth"].dtype == torch.float32
        assert aux["normal"].shape == (n, H, W, 3)
        assert aux["normal"].dtype == torch.float32
        assert aux["mob_id"].shape == (n, H, W)
        assert aux["mob_id"].dtype == torch.int32
        for tensor in aux.values():
            assert tensor.device.type == "cpu"


def _planar(info, point):
    return float(((torch.tensor(point) - info["location"]) * info["forward"]).sum())


def _square_and_cube(scene):
    Square(size=2.0, color=RED).spawn(animate=False)
    Cube().spawn(animate=False).move(RIGHT * 3)


def _front_and_back(front_opacity):
    def build(scene):
        Square(size=4.0, color=BLUE).spawn(animate=False)
        front = Square(size=2.0, color=RED).set_opacity(front_opacity)
        front.spawn(animate=False).move(OUT)

    return build


def test_a_camera_facing_square_records_planar_depth_and_faces_the_camera():
    frames, auxes, info = _render(_square_and_cube)
    _check_layout(frames, auxes)
    aux = _joined(auxes)
    depth, normal, mob = aux["depth"][0], aux["normal"][0], aux["mob_id"][0]

    expected = _planar(info, (0.0, 0.0, 0.0))
    assert depth[CY, CX].item() == pytest.approx(expected, abs=1e-3)
    # Planar, not the slant range: an off-axis pixel of the same plane is at
    # the same depth (its slant range is ~0.06% longer, i.e. > 0.01 here).
    assert depth[CY - 4, CX - 4].item() == pytest.approx(expected, abs=1e-3)
    assert torch.allclose(normal[CY, CX], torch.tensor([0.0, 0.0, 1.0]), atol=1e-5)
    # The cube's near face is one unit closer.
    assert depth[CY, CX + 18].item() == pytest.approx(expected - 1.0, abs=1e-3)
    assert torch.allclose(normal[CY, CX + 18], torch.tensor([0.0, 0.0, 1.0]), atol=1e-5)

    miss = mob == aux_module.MISS_MOB_ID
    assert torch.isinf(depth[miss]).all()
    assert (depth[miss] > 0).all()
    assert (normal[miss] == 0).all()
    assert torch.isfinite(depth[~miss]).all()
    # No source tables on this branch's merge: every hit is "unknown".
    assert (mob[~miss] == aux_module.UNKNOWN_MOB_ID).all()
    assert bool(miss[0, 0])
    assert not bool(miss[CY, CX])
    # 2 x 2 units at 6 px per unit: about 12 x 12 pixels of square.
    square = ~miss[:, : CX + 12]
    assert 120 <= int(square.sum()) <= 170
    # Unit normals on every hit.
    lengths = normal[~miss].norm(dim=-1)
    assert torch.allclose(lengths, torch.ones_like(lengths), atol=1e-4)


def test_rows_are_top_down_like_the_frames():
    def build(scene):
        scene.set_background(BLACK)
        Square(size=1.0, color=RED).spawn(animate=False).move(UP * 1.5)

    frames, auxes, _info = _render(build)
    hit = auxes[0]["mob_id"][0] != aux_module.MISS_MOB_ID
    rows = hit.any(dim=1).nonzero().flatten()
    assert rows.numel() > 0
    assert int(rows.max()) < H // 2, "a square above the axis landed low: rows flipped"
    # The frame agrees about where the square is.
    lit = frames[0][0].int().sum(-1) > 0
    assert (lit & hit).sum() >= 0.8 * hit.sum()


def test_a_sphere_normal_turns_with_the_surface_in_camera_space():
    def build(scene):
        Sphere(radius=1.0).spawn(animate=False)

    frames, auxes, info = _render(build)
    aux = _joined(auxes)
    depth, normal = aux["depth"][0], aux["normal"][0]
    # Pixel (CY, CX)'s centre is half a pixel right of and below the axis,
    # about 0.08 units on the unit sphere: its normal leans that way.
    centre = normal[CY, CX]
    assert centre[0].item() == pytest.approx(0.08, abs=0.02)
    assert centre[1].item() == pytest.approx(-0.08, abs=0.02)
    assert centre[2].item() > 0.98
    # Right of the axis leans right (+x), above it leans up (+y).
    assert normal[CY, CX + 4, 0].item() > 0.4
    assert abs(normal[CY, CX + 4, 1].item()) < 0.2
    assert normal[CY - 4, CX, 1].item() > 0.4
    assert abs(normal[CY - 4, CX, 0].item()) < 0.2
    # Planar depth of the front of the sphere: camera distance minus radius.
    assert depth[CY, CX].item() == pytest.approx(
        _planar(info, (0, 0, 0)) - 1.0, abs=0.03
    )
    # Always toward the viewer (z >= 0 in camera space).
    hit = torch.isfinite(depth)
    assert (normal[hit][:, 2] >= 0).all()


@pytest.mark.parametrize(("opacity", "front"), [(0.3, False), (0.7, True)])
def test_the_alpha_threshold_decides_which_surface_is_recorded(opacity, front):
    frames, auxes, info = _render(_front_and_back(opacity))
    depth = _joined(auxes)["depth"][0]
    back_depth = _planar(info, (0.0, 0.0, 0.0))
    expected = back_depth - 1.0 if front else back_depth
    assert depth[CY, CX].item() == pytest.approx(expected, abs=1e-3)


def _mixed(scene):
    Square(size=2.0, color=RED).spawn(animate=False).move(RIGHT * 2.5)
    Cube().spawn(animate=False).move(RIGHT * -2.5)
    Sphere(radius=1.0).spawn(animate=False)


def test_deterministic_and_path_traced_renders_record_identical_passes():
    frames1, det, _ = _render(_mixed, rt={"samples_per_pixel": 1})
    frames4, traced, _ = _render(_mixed, rt={"samples_per_pixel": 4})
    _check_layout(frames1, det)
    _check_layout(frames4, traced)
    det, traced = _joined(det), _joined(traced)
    for key in aux_module.AUX_PASS_KEYS:
        assert torch.equal(det[key], traced[key]), key
    assert (det["mob_id"] != aux_module.MISS_MOB_ID).sum() > 200


def test_the_flat_and_refit_tree_walks_record_the_same_passes():
    """Both ``refit`` specializations of the trace (the flat walk reads the
    leaf tables through the arena, the refit walk does not).
    """
    _f, refit, _ = _render(_mixed, experimental={"bvh_refit": True})
    _f, flat, _ = _render(_mixed, experimental={"bvh_refit": False})
    refit, flat = _joined(refit), _joined(flat)
    for key in aux_module.AUX_PASS_KEYS:
        assert torch.equal(refit[key], flat[key]), key


def test_path_traced_area_light_panels_are_not_recorded(monkeypatch):
    """Under the path tracer a RectAreaLight is two opaque, camera-visible
    triangles; the deterministic renderer has no such geometry. The passes
    look through the panel, so both renderers still agree.
    """

    def build(scene):
        Square(size=2.0, color=RED).spawn(animate=False)
        RectAreaLight(
            location=OUT * 1.0,
            width=1.0,
            height=1.0,
            samples=4,
            color=WHITE,
            intensity=2.0,
            target=OUT * 5.0,
        ).spawn(animate=False)

    _f, det, info = _render(build, rt={"samples_per_pixel": 1})
    _f, traced, _ = _render(
        build,
        rt={"samples_per_pixel": 4},
        experimental={"pt_area_light_quads": True},
    )
    det, traced = _joined(det), _joined(traced)
    square = _planar(info, (0.0, 0.0, 0.0))
    assert det["depth"][0, CY, CX].item() == pytest.approx(square, abs=1e-3)
    for key in aux_module.AUX_PASS_KEYS:
        assert torch.equal(det[key], traced[key]), key

    # The control: without the merge's ``pt_quad_base`` marker the panel is
    # ordinary geometry, and it IS on the centre ray -- so the skip is what
    # kept it out above.
    real_merge = scene_builder._merge_scene

    def unmarked(*args, **kwargs):
        merged = real_merge(*args, **kwargs)
        assert merged.pop("pt_quad_base", None) is not None
        return merged

    monkeypatch.setattr(scene_builder, "_merge_scene", unmarked)
    _f, control, _ = _render(
        build,
        rt={"samples_per_pixel": 4},
        experimental={"pt_area_light_quads": True},
    )
    centre = _joined(control)["depth"][0, CY, CX].item()
    assert centre == pytest.approx(square - 1.0, abs=1e-3)


def test_a_skipped_panel_does_not_expose_a_surface_hidden_behind_it():
    """The gather that meets an opaque panel stops accepting hits behind it,
    but keeps whatever it buffered before finding the panel -- here a slanted
    triangle whose box is entered first. Draining on past the skipped panel
    would record that triangle over the nearer square; the trace regathers
    from the panel instead, so both renderers still agree.
    """
    from algan.mobs.shapes_2d import TriangleTriangulated

    def build(scene):
        corners = torch.tensor(
            [[[-1.0, -2.0, -2.25], [-1.0, 2.0, -2.25], [3.0, 0.0, 2.75]]]
        )
        TriangleTriangulated(corners).spawn(animate=False)
        Square(size=1.0, color=BLUE).spawn(animate=False)
        RectAreaLight(
            location=OUT * 1.0,
            width=1.0,
            height=1.0,
            samples=4,
            color=WHITE,
            intensity=2.0,
            target=OUT * 5.0,
        ).spawn(animate=False)

    _f, det, info = _render(build, rt={"samples_per_pixel": 1})
    _f, traced, _ = _render(
        build,
        rt={"samples_per_pixel": 4},
        experimental={"pt_area_light_quads": True},
    )
    det, traced = _joined(det), _joined(traced)
    square = _planar(info, (0.0, 0.0, 0.0))
    assert det["depth"][0, CY, CX].item() == pytest.approx(square, abs=1e-3)
    for key in aux_module.AUX_PASS_KEYS:
        assert torch.equal(det[key], traced[key]), key


def test_the_supersampled_route_records_what_the_analytic_route_records(monkeypatch):
    decisions = []
    real_decision = tracer.analytic_raster_route_active

    def spy(*args, **kwargs):
        active = real_decision(*args, **kwargs)
        decisions.append(active)
        return active

    monkeypatch.setattr(tracer, "analytic_raster_route_active", spy)
    video = TINY.set(supersampling=2)
    _f, analytic, _ = _render(_square_and_cube, video=video)
    assert set(decisions) == {True}
    decisions.clear()
    frames, classic, _ = _render(
        _square_and_cube, video=video, rt={"analytic_aa": False}
    )
    assert set(decisions) == {False}
    # Output resolution, whatever the render's supersampled buffer was.
    _check_layout(frames, classic)
    analytic, classic = _joined(analytic), _joined(classic)
    for key in aux_module.AUX_PASS_KEYS:
        assert torch.equal(analytic[key], classic[key]), key


_MOVING = []


def _mover(scene):
    _MOVING[:] = [Square(size=2.0, color=RED).spawn(animate=False)]


def _move_toward_camera(scene):
    _MOVING[0].move(OUT * 2)


def test_an_animated_scene_records_per_frame_depths_in_order_and_sparse_matches():
    frames, auxes, info = _render(_mover, 0, 3, animate=_move_toward_camera)
    _check_layout(frames, auxes)
    dense = _joined(auxes)
    assert dense["depth"].shape[0] == 3
    centre = dense["depth"][:, CY, CX]
    start = _planar(info, (0.0, 0.0, 0.0))
    assert centre[0].item() == pytest.approx(start, abs=1e-3)
    assert centre[2].item() == pytest.approx(start - 2.0, abs=1e-3)
    assert centre[0] > centre[1] > centre[2]

    frames_s, sparse_auxes, _ = _render(
        _mover, 0, 2, animate=_move_toward_camera, frame_indices=(0, 2)
    )
    _check_layout(frames_s, sparse_auxes)
    sparse = _joined(sparse_auxes)
    assert sparse["depth"].shape[0] == 2
    assert torch.allclose(sparse["depth"], dense["depth"][[0, 2]], atol=1e-5)
    assert torch.allclose(sparse["normal"], dense["normal"][[0, 2]], atol=1e-5)
    assert torch.equal(sparse["mob_id"], dense["mob_id"][[0, 2]])


def test_a_background_only_window_delivers_miss_filled_passes():
    frames, auxes, _info = _render(lambda scene: None, 0, 3)
    _check_layout(frames, auxes)
    aux = _joined(auxes)
    assert aux["depth"].shape[0] == sum(len(batch) for batch in frames) == 3
    assert torch.isinf(aux["depth"]).all()
    assert (aux["normal"] == 0).all()
    assert (aux["mob_id"] == aux_module.MISS_MOB_ID).all()


def test_passes_off_change_nothing_and_never_call_the_sink(monkeypatch):
    seen = []
    real_kernel = KERNEL_REGISTRY.render_kernel

    def recording_kernel(*args, **kwargs):
        seen.append(dict(kwargs))
        return real_kernel(*args, **kwargs)

    monkeypatch.setattr(KERNEL_REGISTRY, "render_kernel", recording_kernel)
    # ``aux=False`` still hands get_frames a sink, with ``aux_passes=False``.
    off, never, _ = _render(_square_and_cube, aux=False)
    assert never == [], "the sink was called with the passes off"
    assert seen
    assert all("aux_passes" not in kwargs for kwargs in seen)
    seen.clear()
    on, auxes, _ = _render(_square_and_cube)
    assert seen
    assert all(kwargs.get("aux_passes") is True for kwargs in seen)
    assert len(auxes) == len(on)
    # The passes (and the tree build they force) do not touch the frames.
    assert len(off) == len(on)
    for a, b in zip(off, on):
        assert torch.equal(a, b)


def test_camera_view_captures_trace_no_passes(monkeypatch):
    """A live view's capture passes are textures: only the main pass traces."""
    calls = []
    real_kernel = KERNEL_REGISTRY.render_kernel

    def recording_kernel(*args, **kwargs):
        calls.append(((int(args[2]), int(args[3])), kwargs.get("aux_passes")))
        return real_kernel(*args, **kwargs)

    monkeypatch.setattr(KERNEL_REGISTRY, "render_kernel", recording_kernel)

    def build(scene):
        target = Square(size=1, color=RED).move_to(LEFT * 9).spawn()
        view = CameraView(resolution=(48, 32), height=2).focus_on(target, 0.3)
        view.move_to(OUT).spawn()

    frames, auxes, _info = _render(build)
    captures = [aux for size, aux in calls if size == (48, 32)]
    mains = [aux for size, aux in calls if size == (W, H)]
    assert captures
    assert all(aux is None for aux in captures)
    assert mains
    assert all(aux is True for aux in mains)
    _check_layout(frames, auxes)
    # The view's display panel is geometry of the main pass.
    assert (auxes[0]["mob_id"] != aux_module.MISS_MOB_ID).any()


def test_a_sink_is_required_when_the_passes_are_requested():
    SceneManager.reset()
    try:
        with (
            Scene(video_settings=TINY) as scene,
            pytest.raises(TypeError, match="aux_sink"),
        ):
            next(iter(scene.get_frames(0, 1, aux_passes=True)))
    finally:
        SceneManager.reset()


def _with_tables(monkeypatch, tri_table, circuit_table):
    real_merge = scene_builder._merge_scene

    def merge(*args, **kwargs):
        # What the identity agent's merge will attach: host numpy tables,
        # which the arena upload passes through untouched.
        merged = real_merge(*args, **kwargs)
        if tri_table is not None:
            merged["tri_obj_source_ids"] = np.asarray(tri_table, dtype=np.int32)
        if circuit_table is not None:
            merged["circuit_source_ids"] = np.asarray(circuit_table, dtype=np.int32)
        return merged

    monkeypatch.setattr(scene_builder, "_merge_scene", merge)


def _ids_at(aux):
    mob = aux["mob_id"][0]
    return int(mob[CY, CX]), int(mob[CY, CX + 18]), int(mob[0, 0])


def test_mob_ids_come_from_the_source_tables(monkeypatch):
    # Every global surface id of the cube maps to 1000 + id; the one circuit
    # maps to 777.
    _with_tables(monkeypatch, np.arange(64) + 1000, [777])
    _f, auxes, _ = _render(_square_and_cube)
    square, cube, background = _ids_at(_joined(auxes))
    assert square == 777
    assert cube >= 1000
    assert background == aux_module.MISS_MOB_ID


def test_missing_or_unset_table_entries_read_as_unknown(monkeypatch):
    # A -1 entry, and a circuit table too short to hold circuit 0.
    _with_tables(monkeypatch, np.full(64, -1), np.zeros(0))
    _f, auxes, _ = _render(_square_and_cube)
    square, cube, background = _ids_at(_joined(auxes))
    assert square == aux_module.UNKNOWN_MOB_ID
    assert cube == aux_module.UNKNOWN_MOB_ID
    assert background == aux_module.MISS_MOB_ID


def test_the_near_plane_is_planar_and_the_far_plane_retires_the_ray():
    build = _front_and_back(1.0)
    _f, auxes, info = _render(build)
    back = _planar(info, (0.0, 0.0, 0.0))
    base = _joined(auxes)["depth"][0]
    assert base[CY, CX].item() == pytest.approx(back - 1.0, abs=1e-3)

    # A near plane between the two squares clips the front one everywhere.
    _f, auxes, _ = _render(build, camera={"near": back - 0.5})
    near = _joined(auxes)["depth"][0]
    assert near[CY, CX].item() == pytest.approx(back, abs=1e-3)
    ring = torch.isfinite(base) & (base > back - 0.5)
    assert torch.allclose(near[ring], base[ring], atol=1e-4)

    # A far plane between them drops the back square: its ring goes empty.
    _f, auxes, _ = _render(build, camera={"far": back - 0.5})
    far = _joined(auxes)["depth"][0]
    assert far[CY, CX].item() == pytest.approx(back - 1.0, abs=1e-3)
    assert torch.isinf(far[ring]).all()


def test_the_out_of_memory_split_keeps_passes_in_lockstep(monkeypatch):
    frames, reference, _ = _render(_mover, 0, 6, animate=_move_toward_camera)
    reference = _joined(reference)

    real_trace = aux_module.trace_aux_passes
    failed = []

    def flaky(*args, **kwargs):
        frames_in_chunk = int(args[10]) - int(args[9])
        if frames_in_chunk > 1 and not failed:
            failed.append(frames_in_chunk)
            raise InsufficientMemoryException
        return real_trace(*args, **kwargs)

    monkeypatch.setattr(aux_module, "trace_aux_passes", flaky)
    frames_split, split, _ = _render(_mover, 0, 6, animate=_move_toward_camera)
    assert failed, "no multi-frame chunk was rendered, so nothing was split"
    _check_layout(frames_split, split)
    split = _joined(split)
    assert split["depth"].shape[0] == 6
    assert torch.allclose(split["depth"], reference["depth"], atol=1e-5)
    assert torch.equal(split["mob_id"], reference["mob_id"])
