"""Hardware ray tracing (NVIDIA OptiX, RT cores) as a side path of the renderer.

``SETTINGS.raytracing.experimental.optix`` (``ALGAN_OPTIX``, default off)
moves two of the deterministic renderer's traversals onto RT cores:

* shadow rays (``shadows.py``), as an arm of the light-major shadow queue;
* the reflection/refraction gather (``gather.py``), filling the same k-buffer
  ``wavefront_traverse_events`` fills.

Both trace triangles only; Bezier circuits stay on Algan's software walk (a
shadow ray's circuit transmittance is marched in software and multiplied on;
a gather over a batch with circuits stays in software). A device without RT
cores, a driver without OptiX 9.1 (ABI 118), or anything else this package
cannot take keeps the software traversal, so the switch never fails a
render; the reason is logged once.

Licensing. OptiX's SDK headers are NVIDIA-proprietary and Algan is MIT, so
none is vendored. The host side (``_abi.py``) re-declares the ABI subset it
calls -- enum values, struct layouts, the function-table order -- the way
Mitsuba's BSD ``optix_api.h`` does; the device side
(``algan_optix_device.h``) re-declares the ``_optix_*`` intrinsic calls its
programs make, as Dr.Jit's PTX generator does. The runtime is the display
driver's own ``nvoptix`` library, and the device programs are compiled at
first use by the NVRTC that PyTorch's CUDA wheels ship. Nothing here needs a
CUDA toolkit or the OptiX SDK, at build time or at run time.
"""

from __future__ import annotations

import torch


def optix_requested() -> bool:
    """Whether the switch is on (read live)."""
    from algan.rendering.raytracing import settings as rt_settings

    return bool(rt_settings.optix)


def gather_kbuf():
    """The k-buffer width the device programs are compiled for."""
    from algan.rendering.raytracing.raytrace_kernels_taichi import kbuf

    return min(int(kbuf), 4)


def runtime_for(device):
    """The process's OptiX runtime on ``device``, or ``None`` (logged once)."""
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    from algan.rendering.raytracing.optix.runtime import get_runtime
    from algan.rendering.raytracing.raytrace_kernels_taichi import (
        TRIANGLE_EDGE_EPSILON,
    )

    return get_runtime(device, gather_kbuf(), TRIANGLE_EDGE_EPSILON)
