"""Measure the tonemap's linear-HDR -> uint8 transfer curve directly.

This bypasses the renderer: it hands ``_finalize_on_device`` a synthetic
linear-HDR frame holding a known ramp and reads back what each value quantizes
to, once with ``tonemapping=True`` (the default) and once with it off. That
isolates the tonemap arithmetic from every other stage, and lets the ramp reach
values above 1.0, which a flat authored fill never can.

Both shipping implementations are probed -- the standalone Taichi kernel
(``post_tonemap_kernel=True``, the default) and the torch pipeline it replaced
-- so a divergence between them shows up here rather than in a render.

    <venv-python> benchmarks/_tonemap_transfer_probe.py
"""

from __future__ import annotations

import torch

import algan  # noqa: F401  -- initialises torch/taichi and inference_mode
from algan.rendering.post_processing.post_process import _finalize_on_device
from algan.rendering.raytracing import settings as rt_settings
from algan.settings._startup import render_device
from algan.utils.memory_utils import ManualMemory

# Values are in the units the post stage works in: linear HDR where 1.0 is the
# top of the display range. Below 1.0 is what an authored colour produces; above
# it is what a bright light or an emissive surface produces.
RAMP = [
    0.0,
    0.01,
    0.02,
    0.04,
    0.06,
    0.08,
    0.10,
    0.125,
    0.25,
    0.375,
    0.5,
    0.625,
    0.75,
    0.76,
    0.8,
    0.875,
    0.95,
    1.0,
    1.25,
    1.5,
    2.0,
    3.0,
    4.0,
    8.0,
]


def _probe(values, *, tonemapping, kernel):
    """Return the uint8 each ramp value quantizes to, as a list of ints."""
    frame = torch.zeros(
        (1, 1, len(values), 4), dtype=torch.float32, device=render_device()
    )
    for i, v in enumerate(values):
        frame[0, 0, i, 0] = v
        frame[0, 0, i, 1] = v
        frame[0, 0, i, 2] = v

    memory = ManualMemory(0.0, device=render_device(), managed=False, num_bytes=1 << 22)
    was_kernel = rt_settings.post_tonemap_kernel
    rt_settings.set_post_tonemap_kernel(kernel)
    try:
        out = _finalize_on_device(
            frame,
            4,
            memory,
            tonemap_enabled=True,  # post-process tonemap stage runs
            tonemapping=tonemapping,  # ...applying a curve, or just clamping
            tonemap_method=rt_settings.tonemap_method,
            exposure=rt_settings.tonemap_exposure,
        )
        return [int(x) for x in out[0, 0, :, 0].tolist()]
    finally:
        rt_settings.set_post_tonemap_kernel(was_kernel)


def main():
    on_k = _probe(RAMP, tonemapping=True, kernel=True)
    off_k = _probe(RAMP, tonemapping=False, kernel=True)
    on_t = _probe(RAMP, tonemapping=True, kernel=False)
    off_t = _probe(RAMP, tonemapping=False, kernel=False)

    print(
        f"method={rt_settings.tonemap_method!r} exposure={rt_settings.tonemap_exposure}"
    )
    print()
    print(
        f"{'linear in':>10} {'ideal u8':>9} | {'ON kern':>8} {'OFF kern':>9} "
        f"{'ON torch':>9} {'OFF torch':>10} | {'ON-OFF':>7}"
    )
    print("-" * 76)
    for i, v in enumerate(RAMP):
        ideal = min(255, round(v * 255.0))
        print(
            f"{v:10.3f} {ideal:9d} | {on_k[i]:8d} {off_k[i]:9d} "
            f"{on_t[i]:9d} {off_t[i]:10d} | {on_k[i] - off_k[i]:+7d}"
        )

    print()
    sdr = [i for i, v in enumerate(RAMP) if v <= 1.0]
    worst = max(sdr, key=lambda i: abs(on_k[i] - off_k[i]))
    print(
        f"largest SDR (input <= 1.0) shift from tonemapping: "
        f"{on_k[worst] - off_k[worst]:+d} at input {RAMP[worst]}"
    )
    unchanged = [RAMP[i] for i in sdr if on_k[i] == off_k[i]]
    print(f"SDR inputs the tonemap leaves alone: {unchanged}")
    div = [(RAMP[i], on_k[i], on_t[i]) for i in range(len(RAMP)) if on_k[i] != on_t[i]]
    print(f"kernel/torch divergence (tonemapping on): {div if div else 'none'}")


if __name__ == "__main__":
    main()
