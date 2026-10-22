"""The bounce drain's shadow events are compacted and traced on the device
(``ALGAN_OPT_DISABLE=devshadow`` restores the host glue).

``tracer._device_wavefront_shadows`` replaces a ``nonzero`` readback, a dense
all-lit visibility table and a dozen count-sized gathers per bounce with a
binned compaction (``shadow_event_bins`` / ``shadow_event_scatter``), a trace
sized by the row count with the event count on the device
(``deferred_shadow_trace``) and a row -> slot map the shade kernel reads
(``deferred_shadows == 2``). Two promises carry it, held here:

* the compaction gives every accepted row exactly one slot, grouped by bin in
  bin order, and maps every other row to -1;
* the image is the one the host glue renders, byte for byte -- each arm in a
  child process of its own on one CPU thread, where a render through glass
  and over a mirror is reproducible (``agent_guidance/memory_perf.md``), at
  the defaults and with the event sort, identity-aware rejection and the
  shadow terminator all switched.

Outside the fast suite: only a change to the drain's shadow path can break it.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REPORT = "device-shadow-glue report: "
_FRAMES = 2


@pytest.mark.parametrize(
    ("frames", "rows", "bits"),
    [(1, 1 << 30, 18), (2, 1 << 30, 17), (30, 1 << 30, 13), (1, 1000, 10), (4, 9, 2)],
)
def test_bins_never_outnumber_the_rows_or_the_cap(frames, rows, bits):
    from algan.rendering.raytracing import tracer

    assert tracer._shadow_event_bin_bits(frames, rows) == bits
    nbins = frames << bits
    assert nbins <= 1 << tracer._SHADOW_EVENT_BIN_LOG2
    assert nbins <= max(2 * rows, frames)


@pytest.mark.parametrize("sort_keys", [0, 1], ids=["unsorted", "binned"])
@pytest.mark.parametrize("rows", [1, 37, 4096])
def test_every_accepted_row_gets_its_own_slot_in_bin_order(sort_keys, rows):
    from algan.rendering.raytracing import tracer
    from algan.rendering.raytracing.wavefront_kernels_taichi import (
        shadow_event_bins,
        shadow_event_scatter,
    )
    from algan.rendering.taichi_runtime import init_taichi
    from algan.settings._startup import render_device
    from algan.taichi_compat import ti

    init_taichi()
    device = render_device()
    gen = torch.Generator().manual_seed(rows)
    frames, time_start = 3, 5
    accept = (torch.rand(rows, generator=gen) < 0.35).to(torch.int32)
    frame = torch.randint(time_start, time_start + frames, (rows,), generator=gen)
    code = torch.randint(0, 1 << 30, (rows,), generator=gen)
    keys = (frame.to(torch.int64) << 32) | code.to(torch.int64)
    bits = tracer._shadow_event_bin_bits(frames, rows)
    nbins = (frames << bits) if sort_keys else 1
    expected_bin = (
        ((frame - time_start) << bits) | (code >> (30 - bits))
        if sort_keys
        else torch.zeros(rows, dtype=torch.int64)
    )

    counts = torch.zeros(nbins, dtype=torch.int32, device=device)
    accept_d, keys_d = accept.to(device), keys.to(device)
    shadow_event_bins(
        rows, accept_d, sort_keys, keys_d, time_start, frames, bits, counts
    )
    ti.sync()
    accepted = accept.bool()
    assert torch.equal(
        counts.cpu().long(),
        torch.bincount(expected_bin[accepted], minlength=nbins),
    )

    ends = torch.cumsum(counts, 0, dtype=torch.int32)
    ev_list = torch.full((rows,), -7, dtype=torch.int32, device=device)
    ev_slot = torch.full((rows,), -7, dtype=torch.int32, device=device)
    shadow_event_scatter(
        rows,
        accept_d,
        sort_keys,
        keys_d,
        time_start,
        frames,
        bits,
        counts,
        ends,
        ev_list,
        ev_slot,
    )
    ti.sync()
    n = int(accepted.sum())
    ev_list, ev_slot = ev_list.cpu().long(), ev_slot.cpu().long()
    assert int(ends[-1]) == n
    # Counted back down to zero: nothing reads them after, but it is the
    # arithmetic that gave each row a distinct slot.
    assert int(counts.abs().sum()) == 0
    rows_with_events = torch.nonzero(accepted).flatten()
    assert torch.equal(torch.sort(ev_list[:n]).values, rows_with_events)
    assert torch.equal(ev_list[ev_slot[rows_with_events]], rows_with_events)
    assert bool((ev_slot[~accepted] == -1).all())
    # Grouped by bin, in bin order: the event sort, cut short at the bin.
    binned = expected_bin[ev_list[:n]]
    assert bool((binned[1:] >= binned[:-1]).all())


def _scene(scene_cls):
    """Shadows from four kinds of light on a mirror floor, through glass."""
    from algan import (
        BLUE_A,
        DOWN,
        GRAY_D,
        IN,
        LEFT,
        ORANGE,
        ORIGIN,
        OUT,
        PI,
        RIGHT,
        UP,
        WHITE,
        DirectionalLight,
        MeshLambertMaterial,
        MeshPhysicalMaterial,
        MeshStandardMaterial,
        Off,
        PointLight,
        Prism,
        RectAreaLight,
        Sphere,
        SpotLight,
    )

    with Off():
        scene_cls.clear_lights()
        DirectionalLight(
            location=RIGHT * 3 + UP * 4 + OUT * 5, target=ORIGIN, intensity=0.9
        ).spawn(animate=False)
        SpotLight(
            location=LEFT * 3 + UP * 4,
            target=DOWN,
            color=ORANGE,
            intensity=1.4,
            cone_angle=40.0,
        ).spawn(animate=False)
        RectAreaLight(
            location=UP * 4 + IN * 2,
            target=ORIGIN,
            width=2.0,
            height=1.5,
            samples=2,
            color=BLUE_A,
            intensity=2.0,
        ).spawn(animate=False)
        PointLight(location=RIGHT * 2 + UP * 2 + OUT * 2, intensity=PI).spawn(
            animate=False
        )
        floor = Prism(width=8, height=0.2, depth=8, color=GRAY_D)
        floor.set_material(MeshStandardMaterial(metalness=0.6, roughness=0.05))
        floor.move(DOWN * 1.2).spawn(animate=False)
        glass = Sphere(radius=0.7).set_material(
            MeshPhysicalMaterial(transmission=1.0, ior=1.5, roughness=0.0)
        )
        glass.move(LEFT * 0.8).spawn(animate=False)
        box = Prism(width=0.8, height=0.8, depth=0.8)
        box.set_material(MeshLambertMaterial(color=WHITE))
        box.move(RIGHT * 1.0 + DOWN * 0.6).spawn(animate=False)
    glass.move(RIGHT * 0.3)


def _render(disabled):
    """The scene's frames, and what the drain's shadow path did."""
    import pytest
    import torch

    from algan import SMOKE_TEST, Scene, SceneManager
    from algan.animation_timeline import timeline as timeline_module
    from algan.rendering.raytracing import raster_taichi, tracer

    seen = {"device": 0, "host": 0, "events": 0}
    gate = tracer._device_shadow_glue
    trace = raster_taichi.deferred_shadow_trace
    params = raster_taichi._DEFERRED_SHADOW_TRACE_PARAMS

    def counted_gate(device):
        on = gate(device)
        seen["device" if on else "host"] += 1
        return on

    def counted_trace(*args):
        trace(*args)
        seen["events"] += int(args[params.index("event_count")][0])

    SceneManager.reset()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(timeline_module, "_OPT_DISABLED", frozenset(disabled))
        mp.setattr(tracer, "_device_shadow_glue", counted_gate)
        mp.setattr(raster_taichi, "deferred_shadow_trace", counted_trace)
        try:
            video = SMOKE_TEST.set(resolution=(64, 48), frames_per_second=6)
            with Scene(video_settings=video) as scene:
                _scene(Scene)
                frames = [
                    b.clone() for b in scene.get_frames(0, _FRAMES, post_processes=())
                ]
        finally:
            SceneManager.reset()
    return torch.cat(frames, 0), seen


#: The settings each pair of arms renders under: the defaults (the event sort,
#: identity-aware rejection and the shadow terminator all on), and all three
#: flipped.
_CONFIGS = {
    "default": {},
    "flipped": {
        "wf_shadow_event_sort": False,
        "shadow_identity_reject": False,
        "shadow_terminator": False,
    },
}


def _render_arm(glue, config, out_path):
    """One arm, in this (child) process; the frames go to ``out_path``."""
    import pytest
    import torch

    from algan import SETTINGS
    from algan.rendering.raytracing import settings as rt_module

    with pytest.MonkeyPatch.context() as mp:
        SETTINGS.raytracing.set(samples_per_pixel=1, shadows=True, max_bounces=3)
        for knob, value in _CONFIGS[config].items():
            mp.setattr(rt_module, knob, value)
        frames, seen = _render({"devshadow"} if glue == "host" else set())
    torch.save(frames, out_path)
    return seen


def _child(glue, config, out_path):
    """Render one arm in a child process of its own.

    One render a process: compiling a second set of kernel variants after a
    render in the same CPU process corrupts the heap on Windows (Quadrants
    1.3.3; reproduced with this branch's switches off and at its base), and
    the host and device arms compile different variants.
    """
    env = dict(os.environ)
    for name in list(env):
        if name.startswith(("ALGAN_OPT_DISABLE", "ALGAN_WF_", "ALGAN_SHADOW")):
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
        [sys.executable, str(Path(__file__).resolve()), glue, config, str(out_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
        cwd=str(_REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith(_REPORT)]
    assert lines, proc.stdout[-3000:] + proc.stderr[-3000:]
    return json.loads(lines[-1][len(_REPORT) :])


@pytest.mark.parametrize("config", sorted(_CONFIGS))
def test_the_device_glue_renders_what_the_host_glue_renders(config, tmp_path):
    host_seen = _child("host", config, tmp_path / "host.pt")
    device_seen = _child("device", config, tmp_path / "device.pt")
    host = torch.load(tmp_path / "host.pt")
    device = torch.load(tmp_path / "device.pt")

    # The scene reaches the path, or this proves nothing: the drain built
    # events on the host in one arm and on the device in the other, and the
    # device traced some.
    assert host_seen["host"] > 0, host_seen
    assert host_seen["device"] == 0, host_seen
    assert device_seen["device"] > 0, device_seen
    assert device_seen["host"] == 0, device_seen
    assert device_seen["events"] > 0, device_seen
    assert float(host.float().std()) > 1.0
    assert torch.equal(host, device), int((host != device).sum())


if __name__ == "__main__":
    # One CPU thread (module docstring); set before the render brings the
    # Taichi program up.
    import algan.rendering.taichi_runtime as _runtime

    _init_kwargs = _runtime.taichi_init_kwargs

    def _one_thread():
        kwargs = _init_kwargs()
        kwargs["cpu_max_num_threads"] = 1
        return kwargs

    _runtime.taichi_init_kwargs = _one_thread
    print(_REPORT + json.dumps(_render_arm(*sys.argv[1:4])))
