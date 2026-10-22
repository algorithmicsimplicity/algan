"""The wavefront tile's state fills write whole rows, and only the rows
anything reads (``ALGAN_OPT_DISABLE=exactfills`` restores the strided ones).

Two promises carry the change, and neither is visible in the fills alone:

* only a tile's PRIMARY rows read the nested-IOR stack and shell words the
  host zeroes; every continuation slot has both written where it is spawned,
  and nothing reads a slot before it is spawned;
* only the compactor reads a slot nothing has activated, and it reads the
  status column alone.

So they are held where they are kept: a render through glass, a translucent
closed shell and a rough reflector (nested IOR, shell words, the split pool
and the glossy prefilter, more tiles than frames), on the sheet route and on
the classic one, with every tile's state poisoned before the fills: NaN in the
floats, and in the ints once 0 -- ACTIVE, so a slot whose status went unfilled
is traced -- and once a large value. A read of anything the fills no longer
write would move a pixel. Each arm is compared with its route's unpoisoned
strided-fill render, byte for byte.

The arms run in ONE child process on ONE CPU thread. Several rays of a pixel
deposit into its accumulator with float atomics and race for its glossy
claim, so a multi-threaded render of this scene differs from itself between
runs by a few levels (on CUDA too); serial, it is reproducible to the byte.

Outside the fast suite: only a change to the tile driver or to how a kernel
spawns a ray can break it.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_INT_POISON = 0x3F3F3F3F
_FRAMES = 3
_ROUTES = ("sheets", "classic")
_REPORT = "tile-state-fills report: "


def _render(*, disabled, int_poison):
    """Three frames of the scene; returns them and what the tile driver did.

    The spies are this render's alone: patched in a context of its own, so the
    next render wraps the real functions rather than these.
    """
    import pytest
    import torch

    from algan import (
        DARKER_GRAY,
        GLASS,
        IN,
        LEFT,
        ORIGIN,
        OUT,
        PI,
        RIGHT,
        SMOKE_TEST,
        UP,
        WHITE,
        DirectionalLight,
        MeshLambertMaterial,
        MeshStandardMaterial,
        Off,
        Prism,
        Scene,
        SceneManager,
        Sphere,
    )
    from algan.animation_timeline import timeline as timeline_module
    from algan.rendering.raytracing import raster_pipeline, tracer

    seen = {"widths": [], "pools": [], "ratios": [], "finishes": 0, "sheets": 0}
    alloc = tracer._alloc_wavefront_state
    finish = tracer._gloss_finish_frame
    split_ratio = tracer._split_pool_ratio
    resolve = raster_pipeline.shade_sparse_raster_coverage

    def poisoned_alloc(memory, pool, width, *args, **kwargs):
        state = alloc(memory, pool, width, *args, **kwargs)
        seen["widths"].append(int(width))
        seen["pools"].append(int(pool))
        if int_poison is not None:
            for t in state:
                t.fill_(float("nan") if t.dtype.is_floating_point else int_poison)
        return state

    def counted_finish(*args, **kwargs):
        seen["finishes"] += 1
        return finish(*args, **kwargs)

    def recorded_ratio(*args, **kwargs):
        ratio = split_ratio(*args, **kwargs)
        seen["ratios"].append(int(ratio))
        return ratio

    def counted_resolve(*args, **kwargs):
        seen["sheets"] += 1
        return resolve(*args, **kwargs)

    retries = tracer._WAVEFRONT_POOL_RETRIES[0]
    SceneManager.reset()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(timeline_module, "_OPT_DISABLED", frozenset(disabled))
        mp.setattr(tracer, "_alloc_wavefront_state", poisoned_alloc)
        mp.setattr(tracer, "_gloss_finish_frame", counted_finish)
        mp.setattr(tracer, "_split_pool_ratio", recorded_ratio)
        mp.setattr(raster_pipeline, "shade_sparse_raster_coverage", counted_resolve)
        try:
            video = SMOKE_TEST.set(resolution=(48, 36), frames_per_second=6)
            with Scene(video_settings=video) as scene:
                with Off():
                    scene.set_background(DARKER_GRAY)
                    Scene.clear_lights()
                    DirectionalLight(
                        location=RIGHT * 3 + UP * 4 + OUT * 5,
                        target=ORIGIN,
                        color=WHITE,
                        intensity=PI,
                    ).spawn(animate=False)
                    wall = Prism(width=7, height=5, depth=0.2)
                    wall.set_material(
                        MeshStandardMaterial(color=WHITE, metalness=1, roughness=0.3)
                    )
                    wall.move(IN * 1.2).spawn(animate=False)
                    glass = Sphere(radius=0.8).set_material(GLASS)
                    glass.move(LEFT * 0.9 + OUT * 0.6).spawn(animate=False)
                    veil = Sphere(radius=0.6, opacity=0.5)
                    veil.set_material(MeshLambertMaterial(color=(0.9, 0.3, 0.2)))
                    veil.move(RIGHT * 1.0 + OUT * 0.3).spawn(animate=False)
                glass.move(RIGHT * 0.6)
                frames = [
                    b.clone() for b in scene.get_frames(0, _FRAMES, post_processes=())
                ]
        finally:
            SceneManager.reset()
    seen["retries"] = tracer._WAVEFRONT_POOL_RETRIES[0] - retries
    return torch.cat(frames, 0), seen


def _run_arms():
    """Every arm against its route's unpoisoned strided-fill reference."""
    import pytest
    import torch

    from algan import SETTINGS
    from algan.rendering.raytracing import settings as rt_module

    strided = {"exactfills", "glossfuse"}
    arms = {
        # The poison is a probe the strided fills already pass ...
        "strided": strided,
        # ... and the exact fills read nothing more, with and without the
        # fused glossy finish.
        "exact": {"glossfuse"},
        "exact_fused": set(),
    }
    report = {}
    with pytest.MonkeyPatch.context() as mp:
        SETTINGS.raytracing.set(samples_per_pixel=1, shadows=True, max_bounces=3)
        # A fixed tile, so the window takes more tiles than it has frames.
        SETTINGS.raytracing.experimental.set(wavefront_tile_rays=2048)
        mp.setattr(rt_module, "wavefront_tile_auto", False)
        mp.setattr(rt_module, "nested_ior", True)
        mp.setattr(rt_module, "glossy_reflection", True)
        mp.setattr(rt_module, "glossy_prefilter", True)
        for route in _ROUTES:
            SETTINGS.raytracing.experimental.set(hybrid_raster=route == "sheets")
            reference, seen = _render(disabled=strided, int_poison=None)
            route_report = {"reference": seen, "std": float(reference.float().std())}
            for int_poison in (0, _INT_POISON):
                for name, disabled in arms.items():
                    frames, arm_seen = _render(disabled=disabled, int_poison=int_poison)
                    arm_seen["equal"] = bool(torch.equal(frames, reference))
                    arm_seen["differing"] = int((frames != reference).sum())
                    route_report[f"{name}@{int_poison}"] = arm_seen
            report[route] = route_report
    return report


def test_the_exact_fills_read_nothing_the_strided_fills_wrote():
    from algan.rendering.raytracing.wavefront_kernels_taichi import SCA_WIDTH_NESTED

    env = dict(os.environ)
    for name in list(env):
        if name.startswith(("ALGAN_OPT_DISABLE", "ALGAN_GLOSSY", "ALGAN_NESTED")):
            env.pop(name)
    env.update(
        ALGAN_RENDER_DEVICE="cpu",
        ALGAN_USE_DAEMON="0",
        ALGAN_PRECOMPILE_JOBS="0",
        PYTHONPATH=os.pathsep.join(
            p for p in (str(_REPO_ROOT), env.get("PYTHONPATH")) if p
        ),
    )
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve())],
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
        cwd=str(_REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith(_REPORT)]
    assert lines, proc.stdout[-3000:] + proc.stderr[-3000:]
    report = json.loads(lines[-1][len(_REPORT) :])

    for route in _ROUTES:
        route_report = report[route]
        seen = route_report["reference"]
        # The scene reaches every path the fills serve, or this proves
        # nothing: the route, the nested-IOR stack, a split pool, more tiles
        # than frames, and (on the sheet route) a glossy finish per frame.
        assert (seen["sheets"] > 0) == (route == "sheets"), (route, seen)
        assert min(seen["widths"]) >= SCA_WIDTH_NESTED, (route, seen)
        assert max(seen["ratios"]) > 1, (route, seen)
        assert len(seen["widths"]) - seen["retries"] > _FRAMES, (route, seen)
        if route == "sheets":
            assert seen["finishes"] == _FRAMES, (route, seen)
        assert route_report["std"] > 1.0, route
        for name, arm in route_report.items():
            if name in ("reference", "std"):
                continue
            assert arm["equal"], (route, name, arm)
            assert arm["widths"] == seen["widths"], (route, name, arm)


if __name__ == "__main__":
    # One CPU thread (module docstring); set before the first render brings
    # the Taichi program up.
    import algan.rendering.taichi_runtime as _runtime

    _init_kwargs = _runtime.taichi_init_kwargs

    def _one_thread():
        kwargs = _init_kwargs()
        kwargs["cpu_max_num_threads"] = 1
        return kwargs

    _runtime.taichi_init_kwargs = _one_thread
    print(_REPORT + json.dumps(_run_arms()))
