"""Exact ManualMemory planning for the built-in post-processing pipeline."""

from __future__ import annotations

import math
from contextlib import contextmanager
from functools import partial

import torch


def _numel(shape):
    return math.prod(int(x) for x in shape)


def _itemsize(dtype):
    if dtype in (torch.float32, torch.int32, torch.complex32):
        return 4
    if dtype in (torch.float64, torch.int64, torch.complex64):
        return 8
    if dtype in (torch.float16, torch.bfloat16, torch.int16):
        return 2
    return 1


class PostProcessMemorySizer:
    """Forward-pointer subset of ``ManualMemory`` exposed to custom planners.

    A custom post-process callable used with automatic batching may attach an
    ``algan_memory_planner`` callable.  The planner receives this object and
    must mirror its stage's persistent and temporary allocations with
    :meth:`alloc` and :meth:`temp`, then return the output ``(shape, dtype)``.
    This keeps custom stages extensible without guessing at CUDA allocations.
    """

    def __init__(self, initial_pointer=0):
        self.pointer = int(initial_pointer)
        self.maximum = self.pointer

    def alloc(self, shape, dtype):
        alignment = _itemsize(dtype)
        self.pointer += (-self.pointer) % alignment
        self.pointer += _numel(shape) * alignment
        self.maximum = max(self.maximum, self.pointer)

    @contextmanager
    def temp(self):
        pointer = self.pointer
        try:
            yield
        finally:
            self.pointer = pointer


# Private compatibility name used by the original built-in planner helpers.
_ArenaSizer = PostProcessMemorySizer


def _plan_fxaa(sizer, shape):
    b, c, h, w = shape
    # Persistent float result.
    sizer.alloc(shape, torch.float32)
    with sizer.temp():
        sizer.alloc((b, 3, h, w), torch.float32)       # product / 3 scratch planes
        sizer.alloc((b, 1, h, w), torch.float32)       # luma / range / x offset
        sizer.alloc((1, 3, 1, 1), torch.float32)       # weights / constants
        sizer.alloc((b, 1, h + 2, w + 2), torch.float32)
        sizer.alloc((b, 1, h, w), torch.bool)          # edge mask
        sizer.alloc((b, 1, h, w), torch.bool)          # horizontal mask
        sizer.alloc((b, 1, h, w), torch.bool)          # gradient direction
        sizer.alloc((b, h, w, 2), torch.float32)       # sampling grid
        sizer.alloc((w,), torch.float32)
        sizer.alloc((h,), torch.float32)


def _bloom_args(process):
    from algan.rendering.post_processing.bloom import bloom_filter
    if process is bloom_filter:
        return {}
    if isinstance(process, partial) and process.func is bloom_filter:
        return dict(process.keywords or {})
    return None


def _custom_plan(process, sizer, shape, dtype, device):
    """Run an opt-in exact planner attached to a custom post-process."""
    if isinstance(process, partial):
        target = process.func
        args = tuple(process.args)
        kwargs = dict(process.keywords or {})
    else:
        target = process
        args = ()
        kwargs = {}
    planner = getattr(process, "algan_memory_planner", None)
    if planner is None:
        planner = getattr(target, "algan_memory_planner", None)
    if planner is None:
        return None
    result = planner(
        sizer=sizer,
        frame_shape=shape,
        frame_dtype=dtype,
        device=device,
        args=args,
        kwargs=kwargs,
    )
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError(
            "algan_memory_planner must return (output_shape, output_dtype)"
        )
    output_shape, output_dtype = result
    return tuple(int(x) for x in output_shape), output_dtype


def _plan_bloom(sizer, shape, dtype, kwargs, device):
    b, h_full, w_full, c = shape
    work_dtype = torch.float32 if dtype == torch.uint8 else dtype
    channels = 4 if c == 5 else 3
    scale_factor = kwargs.get("scale_factor", 8)
    scale_factor = max(int(scale_factor * h_full / 2160), 1)
    iterations = int(kwargs.get("num_iterations", 1))
    glow_spread = float(kwargs.get("glow_spread", 0.10))
    rim_frac = float(kwargs.get("rim_frac", 0.004))
    from algan.rendering.post_processing.bloom_kernels_taichi import (
        can_use_bloom_taichi,
    )
    use_taichi = (
        work_dtype == torch.float32 and can_use_bloom_taichi(device)
    )

    # Persistent bloom result.
    sizer.alloc(shape, work_dtype)
    with sizer.temp():
        if dtype == torch.uint8:
            sizer.alloc(shape, torch.float32)
        if c == 5:
            sizer.alloc((b, h_full, w_full, channels), work_dtype)

        h = int(h_full / scale_factor)
        w = int(w_full / scale_factor)
        color_shape = (b, channels, h, w)
        sizer.alloc(color_shape, work_dtype)  # downsample
        if (h, w) != (h_full, w_full):
            if use_taichi and c == 4:
                with sizer.temp():
                    # Pack the non-contiguous RGB view for Taichi interop.
                    sizer.alloc(
                        (b, h_full, w_full, channels), work_dtype
                    )
            elif not use_taichi:
                with sizer.temp():
                    sizer.alloc((b, channels, h_full, w), work_dtype)
        sizer.alloc(color_shape, work_dtype)  # accumulator

        sigma_rim = max(rim_frac * h, 1.0)
        sigma_tail = max(glow_spread * h, sigma_rim * 1.5)
        for sigma in (sigma_rim, sigma_tail):
            with sizer.temp():
                radius = max(1, int(math.ceil(3.0 * sigma)))
                k = 2 * radius + 1
                sizer.alloc((k,), work_dtype)
                if iterations > 0:
                    sizer.alloc(color_shape, work_dtype)  # horizontal result
                    sizer.alloc(color_shape, work_dtype)  # vertical result
                    if not use_taichi:
                        sizer.alloc(color_shape, work_dtype)  # fallback scratch

        sizer.alloc((b, channels, h_full, w_full), work_dtype)
        if not use_taichi and (h, w) != (h_full, w_full):
            with sizer.temp():
                sizer.alloc((b, channels, h, w_full), work_dtype)
    return shape, work_dtype


def _plan_final(sizer, shape, dtype, original_channels, *, tonemap_enabled,
                tonemapping, tonemap_method):
    b, h, w, channels = shape
    same_layout = channels == original_channels
    stripped_channels = (
        4 if same_layout and channels == 5
        else (channels - 1 if same_layout else channels)
    )
    output_channels = (
        (4 if stripped_channels == 4 else 3)
        if tonemap_enabled else stripped_channels
    )
    output_shape = (b, h, w, output_channels)

    if not tonemap_enabled and dtype == torch.uint8:
        if same_layout and channels == 5:
            sizer.alloc(output_shape, torch.uint8)
        return output_shape, torch.uint8

    sizer.alloc(output_shape, torch.uint8)
    with sizer.temp():
        if same_layout and channels == 5:
            sizer.alloc(output_shape, dtype)

        pixels = b * h * w
        if tonemap_enabled:
            if dtype == torch.uint8:
                sizer.alloc((b, h, w, 3), torch.float32)
            tone_dtype = torch.float32 if dtype == torch.uint8 else dtype
            if tonemapping and tonemap_method == "neutral":
                sizer.alloc((pixels, 3), tone_dtype)  # exposed
                sizer.alloc((pixels, 1), tone_dtype)  # minimum
                sizer.alloc((pixels, 1), torch.bool)
                sizer.alloc((pixels, 1), tone_dtype)  # offset
                sizer.alloc((1,), tone_dtype)
                sizer.alloc((pixels, 3), tone_dtype)  # color offset
                sizer.alloc((pixels, 1), tone_dtype)  # peak
                sizer.alloc((pixels, 1), torch.bool)
                sizer.alloc((pixels, 1), tone_dtype)  # new peak
                sizer.alloc((pixels, 1), tone_dtype)  # scalar scratch
                sizer.alloc((pixels, 1), tone_dtype)  # scale
                sizer.alloc((pixels, 3), tone_dtype)  # compressed color
                sizer.alloc((pixels, 3), tone_dtype)  # compression rhs
            elif tonemapping and tonemap_method == "agx":
                sizer.alloc((pixels, 3), tone_dtype)  # ping
                sizer.alloc((pixels, 3), tone_dtype)  # pong
                sizer.alloc((pixels,), tone_dtype)    # transform scratch
                sizer.alloc((pixels,), tone_dtype)    # x2
                sizer.alloc((pixels,), tone_dtype)    # x4
            elif tonemapping:
                raise ValueError(f"Unknown tonemapping method: {tonemap_method}")
            else:
                sizer.alloc((b, h, w, 3), tone_dtype)

            sizer.alloc((b, h, w, 3), tone_dtype)     # uint8 conversion scratch
            if output_channels == 4 and dtype != torch.uint8:
                sizer.alloc((b, h, w, 1), dtype)
        else:
            sizer.alloc(output_shape, dtype)
    return output_shape, torch.uint8


def get_post_process_memory_required(
        frame_shape, frame_dtype, anti_alias_level, post_processes=(),
        apply_fxaa=False, *, tonemap_enabled=None, tonemapping=None,
        tonemap_method=None, initial_pointer=0, device=None):
    """Return the exact additional ManualMemory peak for built-in stages.

    ``frame_shape`` is the ray tracer's input layout ``[T, H, W, C]`` before
    post-process downsampling.  The result excludes that existing input tensor
    and includes every arena allocation made by downsampling, FXAA, built-in
    bloom, channel stripping, and tonemapping. ``initial_pointer`` makes dtype
    alignment exact when the live arena prefix is not naturally aligned. Bloom
    is planned for its active (nonzero-glow) path, which is the only safe choice
    before pixels exist.

    Unknown custom post-process callables are rejected because their tensor
    lifetime cannot be inferred safely; they should expose a built-in-style
    planner before participating in automatic batching.
    """
    from algan.rendering.raytracing import settings as rt_settings
    if tonemap_enabled is None:
        tonemap_enabled = rt_settings.is_post_process_tonemap_enabled()
    if tonemapping is None:
        tonemapping = rt_settings.TONEMAPPING
    if tonemap_method is None:
        tonemap_method = rt_settings.TONEMAP_METHOD
    if device is None:
        from algan.settings.defaults import COMPUTING_DEFAULTS
        device = COMPUTING_DEFAULTS.render_device
    device = torch.device(device)

    shape = tuple(int(x) for x in frame_shape)
    dtype = frame_dtype
    original_channels = shape[-1]
    aa = max(1, int(anti_alias_level))
    initial_pointer = int(initial_pointer)
    sizer = PostProcessMemorySizer(initial_pointer)

    if aa > 1:
        shape = (shape[0], shape[1] // aa, shape[2] // aa, shape[3])
        sizer.alloc(shape, torch.uint8)
        with sizer.temp():
            sizer.alloc(shape, torch.float32)
        dtype = torch.uint8

    if apply_fxaa:
        sizer.alloc(shape, torch.uint8)
        with sizer.temp():
            sizer.alloc(shape, torch.float32)  # explicit NHWC cast
            _plan_fxaa(sizer, (shape[0], shape[3], shape[1], shape[2]))
        dtype = torch.uint8

    for process in post_processes:
        bloom_kwargs = _bloom_args(process)
        if bloom_kwargs is not None:
            shape, dtype = _plan_bloom(
                sizer, shape, dtype, bloom_kwargs, device
            )
            continue
        custom_result = _custom_plan(
            process, sizer, shape, dtype, device
        )
        if custom_result is None:
            raise ValueError(
                "Automatic batching cannot size custom post-process "
                f"{process!r}; attach an exact algan_memory_planner."
            )
        shape, dtype = custom_result

    _plan_final(
        sizer, shape, dtype, original_channels,
        tonemap_enabled=bool(tonemap_enabled),
        tonemapping=bool(tonemapping), tonemap_method=str(tonemap_method),
    )
    return sizer.maximum - initial_pointer
