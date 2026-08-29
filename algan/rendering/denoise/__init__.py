"""Path-tracer output denoising (Open Image Denoise's RT filter, in torch).

The ``samples_per_pixel > 1`` renderer produces Monte Carlo noise wherever
transport is actually stochastic (lit surfaces, GGX, GI); this package removes
it with the pre-trained U-Net from Intel Open Image Denoise -- the
``rt_hdr_alb_nrm`` filter: linear HDR color guided by albedo and normal
auxiliary images -- re-implemented as a handful of ``torch.nn.functional``
calls so no native OIDN dependency exists.

Pieces:

``tza``
    Parser for OIDN's ``.tza`` tensor-archive format (the official weight
    files).
``weights``
    Resolves the weight file: an explicit path (``denoise_weights``), the
    on-disk cache under ``SETTINGS.paths.cache_directory/oidn/``, or a
    one-time download of the official file (sha256-pinned). Every failure
    path degrades to "denoising off" with one warning, never an error --
    an offline machine still renders.
``oidn_unet``
    The RT U-Net itself, built from the parsed tensors (plain functional
    convolutions; no ``nn.Module``, so the process-global
    ``torch.inference_mode`` cannot trip over parameter registration).
``denoise``
    The end-to-end filter: autoexposure, the PU transfer function, 16-pixel
    alignment padding, tiled inference with overlap, and the inverse
    transform back to linear HDR.

:func:`get_denoiser` is the one entry point the render loop calls; it caches
the loaded network per (process, device) and returns ``None`` when weights
cannot be had.
"""

from __future__ import annotations

from algan.rendering.denoise.denoise import Denoiser, get_denoiser

__all__ = ["Denoiser", "get_denoiser"]
