"""Portable device-count plans with explicit commit and lifetime boundaries.

The int32 header stores published count, reservations, overflow, error.
Capacity-guarded consumers never turn overflow into a truncated result.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
from contextvars import ContextVar
from dataclasses import dataclass

import torch

from algan.rendering.arena_regions import TypedArenaView, lease_regions

COUNT, RESERVED, OVERFLOW, ERROR = range(4)
_INT32_MAX = (1 << 31) - 1
_ACTIVE_DISPATCH = ContextVar("algan_device_dispatch", default=None)
_QUARANTINED_DISPATCHES = []


class DeviceDispatchError(RuntimeError):
    """A speculative dispatch must not be committed."""


class DeviceDispatchOverflow(DeviceDispatchError):
    """Retry the complete batch with sufficient capacity."""


@dataclass(frozen=True)
class DeviceCount:
    header: TypedArenaView
    capacity: int

    def __post_init__(self):
        if isinstance(self.capacity, bool) or not isinstance(self.capacity, int):
            raise ValueError("Device capacity must be an integer")
        if not 0 <= self.capacity <= _INT32_MAX:
            raise ValueError("Device capacity must fit signed int32")
        self.validate()

    def validate(self):
        self.header.validate()
        if self.header.tensor.dtype != torch.int32 or self.header.tensor.shape != (4,):
            raise ValueError("Device-count header must be contiguous int32[4]")

    def launch_capacity(self, multiplier=1):
        if (
            isinstance(multiplier, bool)
            or not isinstance(multiplier, int)
            or multiplier < 0
        ):
            raise ValueError("Dispatch multiplier must be a nonnegative integer")
        total = self.capacity * multiplier
        if total > _INT32_MAX:
            raise ValueError("Flattened dispatch extent exceeds signed int32")
        return total

    def __int__(self):
        raise TypeError("DeviceCount is not a host count; use the guarded dispatch ABI")


@dataclass(frozen=True)
class DispatchStage:
    """Reusable submission function; must not capture per-batch tensors."""

    name: str
    submit: Callable


@dataclass(frozen=True)
class DispatchPlan:
    """Rebind stages each submission; first stage initializes the device header.

    Stages and consumers write scratch only. Final output is committed after
    a consolidated status read. Leases include late-bound consumer arguments.
    """

    stages: tuple[DispatchStage, ...]

    def run(self, extent, views, bindings, *, consume, synchronize, commit=None):
        extent.validate()
        if not any(v is extent.header for v in views):
            raise ValueError("A dispatch must lease its device-count header")
        if any(v.tensor.device != extent.header.tensor.device for v in views):
            raise ValueError("Dispatch regions must share the header's device")
        with ExitStack() as stack:
            metadata = stack.enter_context(lease_regions(views))
            token = _ACTIVE_DISPATCH.set(stack)
            try:
                for stage in self.stages:
                    stage.submit(extent, bindings)
                result = consume(extent, bindings, metadata)
                synchronize()
                count, reserved, overflow, error = extent.header.tensor.cpu().tolist()
                if error:
                    raise DeviceDispatchError(f"Invalid device queue (error={error})")
                if overflow:
                    raise DeviceDispatchOverflow(
                        f"Device queue needs {reserved} slots; capacity is {extent.capacity}"
                    )
                if not (0 <= count == reserved <= extent.capacity):
                    raise DeviceDispatchError(
                        "Device queue published an inconsistent count"
                    )
                if commit is not None:
                    result = commit(result)
                    synchronize()
            except BaseException:
                try:
                    synchronize()
                except BaseException:
                    # Unknown completion is not permission to reuse memory.
                    _QUARANTINED_DISPATCHES.append(stack.pop_all())
                    raise
                raise
            finally:
                _ACTIVE_DISPATCH.reset(token)
            return result


def retain_dispatch_regions(views, resources=()):
    """Hold a late-bound consumer's regions through the plan's completion fence."""
    stack = _ACTIVE_DISPATCH.get()
    if stack is None:
        raise DeviceDispatchError(
            "Region-aware asynchronous launch requires a dispatch plan"
        )
    metadata = stack.enter_context(lease_regions(views))
    stack.callback(lambda held=resources: None)
    return metadata
