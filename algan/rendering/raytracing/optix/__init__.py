"""Hardware ray tracing (NVIDIA OptiX, RT cores) as a side path of the renderer.

``SETTINGS.raytracing.experimental.optix`` (``ALGAN_OPTIX``, default off)
moves two of the deterministic renderer's traversals onto RT cores:

* shadow rays (``shadows.py``), as an arm of the light-major shadow queue;
* the reflection/refraction gather (``gather.py``), filling the same k-buffer
  ``wavefront_traverse_events`` fills.

Both trace triangles only; Bezier circuits stay on Algan's software walk (a
shadow ray's circuit transmittance is marched in software and multiplied on;
a gather's circuit hits are gathered in software and the two k-buffers
merged). A device without RT
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


def gather_kbuf():
    """The k-buffer width the device programs are compiled for."""
    from algan.rendering.raytracing.raytrace_kernels_taichi import kbuf

    return min(int(kbuf), 4)


#: Why ``runtime_for`` declined, for the reasons :func:`get_runtime` does not
#: log itself; each is logged once per process.
_NOTED = set()


def _note_unavailable(reason):
    if reason in _NOTED:
        return
    _NOTED.add(reason)
    from algan.logging.logger import get_logger

    get_logger(__name__).info(
        "SETTINGS.raytracing.experimental.optix is on, but %s, so the software "
        "traversal renders instead.",
        reason,
    )


def runtime_for(device):
    """The process's OptiX runtime on ``device``, or ``None`` (logged once).

    A runtime that fails to start says why itself (``runtime.get_runtime``).
    The two declines here used to say nothing, so a user who had set OptiX up
    wrongly got the slower path silently: a device that is not CUDA, and a
    second device in a process whose one runtime lives on the first.
    """
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        _note_unavailable(f"OptiX needs a CUDA device and this render is on {device}")
        return None
    from algan.rendering.raytracing.optix import runtime as runtime_module
    from algan.rendering.raytracing.raytrace_kernels_taichi import (
        TRIANGLE_EDGE_EPSILON,
    )

    runtime = runtime_module.get_runtime(device, gather_kbuf(), TRIANGLE_EDGE_EPSILON)
    if runtime is None and runtime_module._RUNTIME is not None:
        live = runtime_module._RUNTIME
        _note_unavailable(
            f"this process's OptiX runtime was built for {live.device} "
            f"(k-buffer {live.kbuf}) and this render is on {device} "
            f"(k-buffer {gather_kbuf()}); a process keeps one runtime"
        )
    return runtime
