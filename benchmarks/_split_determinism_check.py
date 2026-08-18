"""Run-to-run determinism of scenes whose pixels SPLIT into several branches.

A PBR (``MeshStandardMaterial``) surface spawns a Fresnel reflection branch, and
under analytic AA a partially-covering fragment additionally spawns a
coverage-miss branch. Every branch of one pixel retires independently and
commits its premultiplied colour into the *same* ``pix_accum`` row with
``ti.atomic_add`` (``wavefront_kernels_taichi`` ~3063, ``raster_taichi`` ~4655).
Float atomic add is not associative, so if the summation order is what varies
between two runs, the same scene renders differently every time.

This script renders one scene twice **in the same process** with identical
settings and reports the pixel-difference distribution, then lets each
hypothesis be tested by turning one thing off:

    solo        every material unlit  -- no Fresnel branch at all
    nobounce    max_bounces = 0       -- PBR materials kept, branches suppressed
    noaa        analytic AA off       -- no coverage-miss branch
    mixed       the full scene        -- the repro

``nobounce`` is the discriminating arm: it keeps the geometry, the materials and
the shading path of ``mixed`` and removes only the continuations, so if it is
byte-identical while ``mixed`` is not, the split is the cause rather than the
materials.

Findings (RTX-class CUDA, commit a7863cd)
-----------------------------------------
``mixed``/``stress``/``pnonly`` all DIFFER run to run; ``nobounce``, ``noaa``
and ``solo`` are byte-identical. With ``SD_HASH_ACCUM=1`` every tile's
``pix_accum`` digest differs across two runs of ``stress`` and every one matches
across two runs of ``nobounce``, which places the divergence in the
accumulation rather than anywhere upstream of it.

The surviving output difference is **|d| = 1** on tens of channel samples out of
165M, because the compositor truncates to ``u8``; the encoded mp4s come out
bit-identical. Reassociation needs >= 3 terms (float atomic add is commutative,
just not associative), so only the many-branch pixels can move at all.

Branches are never DROPPED: starving the pool with ``SD_POOL_FRACTION=0.25``
forced 13 overflow retries in both runs and changed neither the retry count nor
the output, so the discard-and-retry path is deterministic. A run-to-run
difference **larger than 1 is therefore not this mechanism** -- suspect the
change under test.

Run: .venv/Scripts/python.exe benchmarks/_split_determinism_check.py [arms...]
Env: SD_HASH_ACCUM=1      hash each tile's pix_accum before compositing
     SD_POOL_FRACTION=f   shrink the continuation pool to force overflow
     SD_RES=WxH  SD_FPS=n  SD_FRAMES=n
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Benchmarks must never be measured inside a warm daemon: it keeps adaptive
# renderer state (the memory model's batch-size fit) across runs, so one
# benchmark would be timed against whatever ran before it.
os.environ.setdefault("ALGAN_USE_DAEMON", "0")

import algan.render_loop as _render_loop  # noqa: E402
from algan import (  # noqa: E402
    BLUE,
    DOWN,
    GRAY,
    LEFT,
    MD,
    ORIGIN,
    OUT,
    RED,
    RIGHT,
    SETTINGS,
    UP,
    WHITE,
    YELLOW,
    AmbientLight,
    Cube,
    DirectionalLight,
    Off,
    PointLight,
    Scene,
    Sphere,
    Square,
    Sync,
)
from algan.rendering.shaders.materials import (  # noqa: E402
    MeshBasicMaterial,
    MeshStandardMaterial,
)
from algan.scene_manager import SceneManager  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "_split_det_out")
os.makedirs(OUT_DIR, exist_ok=True)

# The user-reported repro pins this so both runs plan the same frame window.
AVAILABLE_MEMORY_OVERRIDE = 2_400_000_000
FRAMES = int(os.environ.get("SD_FRAMES", "60"))
RES = tuple(int(v) for v in os.environ.get("SD_RES", "1280x720").split("x"))
FPS = int(os.environ.get("SD_FPS", "30"))
# <1.0 starves the continuation pool to force overflow/retry (see _starve_pool).
POOL_FRACTION = float(os.environ.get("SD_POOL_FRACTION", "1.0"))
# Hash each tile's pix_accum before compositing, to locate the divergence.
HASH_ACCUM = os.environ.get("SD_HASH_ACCUM") == "1"

_CAPTURED: list[np.ndarray] = []


def _capturing_writer(queue, file_writer):
    """Same as ``render_loop.write_frames_from_queue`` but keeps the frames.

    Comparing the mp4s would fold the encoder's own quantisation into the
    measurement; these are the exact uint8 frames the renderer produced.
    """
    while True:
        frame = queue.get()
        if frame is None:
            break
        array = frame.numpy()
        _CAPTURED.append(array.copy())
        file_writer.write_frame(array)


_render_loop.write_frames_from_queue = _capturing_writer


def _build_scene(mode: str) -> None:
    """3 cubes + 2 spheres over a ground plane, cubes rotating.

    ``solo``   every material unlit -- no Fresnel lobe, hence no branch.
    ``pnonly`` ONLY the spheres (logical PN triangles) are reflective. No
               ``pn_has_reflective`` flag exists, so ``_secondary_split_needed``
               sees an unreflective batch, ``_split_pool_ratio`` returns 1, and
               at ratio 1 the host ignores the pool's overflow flag entirely
               (tracer.py ~232). Any continuation the pool cannot fit is then
               dropped silently, and which ray loses the race is scheduling
               dependent.
    """
    solo = mode == "solo"
    pn_only = mode == "pnonly"
    # Reassociation needs >=3 terms: float atomic add is commutative, so a
    # 2-branch pixel sums identically whatever the order. ``stress`` therefore
    # maximises the population of pixels carrying MANY branches -- big, strong
    # mirrors filling the frame, each covered pixel taking N secondary taps.
    stress = mode == "stress"

    def material(color, roughness=0.15, metalness=0.6, pn=False):
        if solo or (pn_only and not pn):
            return MeshBasicMaterial(color=color)
        if stress:
            roughness, metalness = 0.02, 0.98
        return MeshStandardMaterial(
            color=color, roughness=roughness, metalness=metalness
        )

    with Off():
        AmbientLight(color=WHITE, intensity=0.35).spawn(animate=False)
        DirectionalLight(
            location=RIGHT * 5 + UP * 6 + OUT * 5,
            target=ORIGIN,
            color=WHITE,
            intensity=0.9,
        ).spawn(animate=False)
        PointLight(
            location=LEFT * 4 + UP * 3 + OUT * 4, color=WHITE, intensity=0.7
        ).spawn(animate=False)

        # The reflective floor: this is where the reported differences sit.
        ground = Square(side_length=14).set_material(
            material(
                GRAY,
                roughness=0.02 if stress else 0.08,
                metalness=0.98 if stress else 0.9,
            )
        )
        ground.rotate(-90, RIGHT).move(DOWN * 1.6)

        # One PBR cube, two plain ones, so the mixed batch really is mixed.
        cube_pbr = Cube(side_length=1.1).set_material(material(RED))
        cube_pbr.move(LEFT * 2.2 + UP * 0.2)
        cube_a = Cube(side_length=1.0).set_material(MeshBasicMaterial(color=YELLOW))
        cube_a.move(UP * 0.1)
        cube_b = Cube(side_length=0.9).set_material(MeshBasicMaterial(color=BLUE))
        cube_b.move(RIGHT * 2.2 + UP * 0.05)

        radius_a, radius_b = (1.75, 1.5) if stress else (0.62, 0.52)
        sphere_a = Sphere(radius=radius_a).set_material(
            material(WHITE, roughness=0.05, metalness=0.95, pn=True)
        )
        sphere_a.move(LEFT * 1.1 + DOWN * 0.9 + OUT * 1.4)
        sphere_b = Sphere(radius=radius_b).set_material(
            material(BLUE, roughness=0.05, metalness=0.95, pn=True)
        )
        sphere_b.move(RIGHT * 1.3 + DOWN * 1.0 + OUT * 1.2)

        actors = [ground, cube_pbr, cube_a, cube_b, sphere_a, sphere_b]

    with Off():
        for actor in actors:
            actor.spawn(animate=False)

    # The reported differences GROW as the reflected cubes rotate, so the
    # animation has to move what the floor reflects.
    with Sync(run_time=FRAMES / FPS):
        cube_pbr.rotate(150, UP)
        cube_a.rotate(-120, UP)
        cube_b.rotate(100, RIGHT)


def _instrument_pool_ratio() -> None:
    """Log the pool ratio / reflective flags the tracer actually derived."""
    from algan.rendering.raytracing import tracer as _tracer

    if getattr(_tracer._split_pool_ratio, "_instrumented", False):
        return
    original = _tracer._split_pool_ratio
    seen: set = set()

    def wrapper(splitting, merged, analytic_raster=False, custom_scatter=False):
        ratio = original(splitting, merged, analytic_raster, custom_scatter)
        key = (
            ratio,
            analytic_raster,
            bool(merged.get("tri_has_reflective")),
            bool(merged.get("bez_has_reflective")),
            bool(merged.get("tex_has_reflective")),
            bool(merged.get("has_strong_reflective")),
        )
        if key not in seen:
            seen.add(key)
            print(
                f"    pool_ratio={ratio} analytic_raster={key[1]} "
                f"tri_refl={key[2]} bez_refl={key[3]} tex_refl={key[4]} "
                f"strong={key[5]}"
            )
        return ratio

    wrapper._instrumented = True
    _tracer._split_pool_ratio = wrapper


def _starve_pool(fraction: float) -> None:
    """Shrink the shared continuation pool so tiles actually overflow.

    Overflow is the only path that can DROP a branch rather than merely sum the
    branches in a different order, and no natural arm here reaches it (0
    retries). Starving the pool forces it, which tests whether the
    discard-and-retry path is itself deterministic.
    """
    from algan.rendering.raytracing import tracer as _tracer

    if getattr(_tracer._shared_pool_slots, "_starved", False):
        return
    original = _tracer._shared_pool_slots

    def wrapper(primary_capacity, memory_primary, pool_ratio, analytic_raster):
        full = original(primary_capacity, memory_primary, pool_ratio, analytic_raster)
        return max(1, int(full * fraction))

    wrapper._starved = True
    _tracer._shared_pool_slots = wrapper


_ACCUM_HASHES: list[str] = []


def _instrument_accum_hash() -> None:
    """Hash every tile's ``pix_accum`` just before it is composited.

    This is the direct evidence for *which* buffer is order dependent: the
    merged scene tensors that feed the kernels hash identically run to run, so
    if these digests diverge, the divergence was created by the accumulation
    itself rather than by anything upstream of it.
    """
    import hashlib

    import torch

    from algan.rendering.raytracing import tracer as _tracer

    def wrap(original):
        def wrapper(*args, **kwargs):
            for arg in args:
                # pix_accum is the only (N, 7) f32 tensor in these signatures.
                if torch.is_tensor(arg) and arg.dim() == 2 and arg.shape[1] == 7:
                    data = arg.detach().cpu().numpy().tobytes()
                    _ACCUM_HASHES.append(
                        hashlib.blake2b(data, digest_size=8).hexdigest()
                    )
                    break
            return original(*args, **kwargs)

        wrapper._hashed = True
        return wrapper

    for name in (
        "wf_composite_accum",
        "wf_composite_accum_aa",
        "wf_composite_accum_sparse",
    ):
        original = getattr(_tracer, name)
        if not getattr(original, "_hashed", False):
            setattr(_tracer, name, wrap(original))


def _render(tag: str, mode: str) -> np.ndarray:
    _CAPTURED.clear()
    # Pool overflows are the one way a branch can be DROPPED rather than merely
    # summed in a different order, so the retry counter separates the two
    # candidate mechanisms.
    from algan.rendering.raytracing import tracer as _tracer

    _tracer._WAVEFRONT_POOL_RETRIES[0] = 0
    _instrument_pool_ratio()
    if POOL_FRACTION < 1.0:
        _starve_pool(POOL_FRACTION)
    if HASH_ACCUM:
        _instrument_accum_hash()
        _ACCUM_HASHES.clear()
    SceneManager.reset()
    path = os.path.join(OUT_DIR, f"{tag}.mp4")
    if os.path.exists(path):
        os.remove(path)
    with Scene() as scene:
        _build_scene(mode)
        scene.save_video(
            path,
            # MD unmodified: its default anti_alias_level supersamples, which
            # multiplies both the ray count and the continuation-pool pressure.
            video_settings=MD.set(resolution=RES, frames_per_second=FPS),
            overwrite=True,
        )
    print(f"    [{tag}] pool retries: {_tracer._WAVEFRONT_POOL_RETRIES[0]}")
    if HASH_ACCUM:
        _render.last_accum = list(_ACCUM_HASHES)
    return np.stack(_CAPTURED)


def _decode(path: str) -> np.ndarray:
    import cv2

    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return np.stack(frames)


def _report(tag: str, a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        print(f"  {tag}: SHAPE MISMATCH {a.shape} vs {b.shape}")
        return False
    diff = np.abs(a.astype(np.int32) - b.astype(np.int32))
    total = diff.size
    identical = not diff.any()
    print(f"  {tag}: frames={a.shape[0]} {'IDENTICAL' if identical else 'DIFFERS'}")
    if identical:
        return True
    over = [(t, int((diff > t).sum())) for t in (0, 1, 2, 4, 8, 16, 30)]
    print(f"    max|d| = {diff.max()}")
    print("    " + "  ".join(f">{t}: {c} ({100.0 * c / total:.3f}%)" for t, c in over))
    # Which frames, and does it grow over the animation?
    per_frame = diff.reshape(diff.shape[0], -1).max(axis=1)
    worst = int(per_frame.argmax())
    first = int(np.argmax(per_frame > 2)) if (per_frame > 2).any() else -1
    print(f"    first frame with max|d|>2: {first};  worst frame: {worst}")
    print(f"    per-frame max|d| (every 6th): {per_frame[::6].tolist()}")
    return False


ARMS = {
    "mixed": {"mode": "mixed"},
    "solo": {"mode": "solo"},
    "nobounce": {"mode": "mixed", "max_bounces": 0},
    "noaa": {"mode": "mixed", "analytic_aa": False},
    # Only the (opaque, PN) spheres reflect. Kept because it looks like it
    # should fall to pool ratio 1 -- there is no pn_has_reflective flag -- but
    # constant-property promotion sets tex_has_reflective, which propagates to
    # tri_has_reflective, so it lands on ratio 5 like the rest.
    "pnonly": {"mode": "pnonly"},
    "pnonly_nobounce": {"mode": "pnonly", "max_bounces": 0},
    "stress": {"mode": "stress"},
    "stress_nobounce": {"mode": "stress", "max_bounces": 0},
}


def main(argv: list[str]) -> int:
    requested = argv[1:] or ["mixed"]
    SETTINGS.computing.set(available_memory_override=AVAILABLE_MEMORY_OVERRIDE)
    SETTINGS.raytracing.set(samples_per_pixel=1, shadows=False)
    SETTINGS.raytracing.experimental.set(fragment_shading=True, hybrid_raster=True)

    failures = 0
    for arm in requested:
        if arm not in ARMS:
            print(f"unknown arm {arm!r}; pick from {sorted(ARMS)}")
            return 2
        config = dict(ARMS[arm])
        mode = config.pop("mode")
        snapshot = SETTINGS.snapshot()
        if config:
            SETTINGS.raytracing.set(**config)
        print(
            f"[{arm}] {RES[0]}x{RES[1]} {FRAMES}f  mode={mode} {config or 'defaults'}"
        )
        try:
            first = _render(f"{arm}_run1", mode)
            accum_first = list(getattr(_render, "last_accum", []))
            second = _render(f"{arm}_run2", mode)
            accum_second = list(getattr(_render, "last_accum", []))
            if HASH_ACCUM:
                same = accum_first == accum_second
                differing = sum(1 for x, y in zip(accum_first, accum_second) if x != y)
                print(
                    f"  {arm} pix_accum: {len(accum_first)} tiles hashed, "
                    f"{'ALL MATCH' if same else f'{differing} DIFFER'}"
                )
            ok_raw = _report(f"{arm} RAW frames", first, second)
            # The suites compare the ENCODED videos, so measure that too: a
            # lossy codec can either absorb a small render difference or
            # amplify it by flipping block/mode decisions.
            ok_mp4 = _report(
                f"{arm} DECODED mp4",
                _decode(os.path.join(OUT_DIR, f"{arm}_run1.mp4")),
                _decode(os.path.join(OUT_DIR, f"{arm}_run2.mp4")),
            )
            if not (ok_raw and ok_mp4):
                failures += 1
        finally:
            SETTINGS.restore(snapshot)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
