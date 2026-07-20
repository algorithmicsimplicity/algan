import gc
import os
import time

import torch

from algan.settings.defaults import COMPUTING_DEFAULTS
from algan.constants.math import GIGABYTES


class InsufficientMemoryException(Exception):
    pass


def get_num_available_bytes(device=torch.device("cuda")):
    device = torch.device(device)
    if device.type == "cuda":
        # ``empty_cache`` acts on PyTorch's current CUDA device.  The render
        # arena may target a different indexed device, so make that device
        # current while reclaiming its cached blocks before measuring it.
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            free_bytes, _ = torch.cuda.mem_get_info(device)
        return free_bytes
    elif device.type == "mps":
        allocated_bytes = torch.mps.driver_allocated_memory()
        total_bytes = torch.mps.recommended_max_memory()
        free_bytes = total_bytes - allocated_bytes
        free_bytes = min(free_bytes, 1 * GIGABYTES)
        return free_bytes
    else:
        return COMPUTING_DEFAULTS.max_cpu_memory_used


def _gpu_memory_pressure(threshold=0.8):
    """True when the CUDA device is using more than ``threshold`` of its memory
    (driver-level, so it accounts for Taichi + torch + everything)."""
    if not torch.cuda.is_available():
        return True  # No CUDA telemetry; keep the original (always-gc) behavior.
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return (total_bytes - free_bytes) > threshold * total_bytes
    except Exception:
        return True


# Opportunistic (force_gc=False) reclaim throttles. On a card where the render
# arena reserves most of VRAM, the driver-level pressure test above is
# chronically true, so the "conditional" gc fired on every call and the
# per-batch reclaim added up to ~5% of render wall time. A full gc pass only
# matters for breaking reference cycles, which accumulate slowly -- rate-limit
# it. ``torch.cuda.empty_cache()`` is not free either (each cached-block
# cudaFree synchronizes the device, serializing the async render pipeline), so
# it is skipped when the allocator holds too little reclaimable cache to be
# worth returning. Explicit ``force_gc=True`` callers (OOM retries, arena
# sizing) always reclaim everything.
_GC_MIN_INTERVAL_S = float(os.environ.get("ALGAN_GC_MIN_INTERVAL_S", "5.0"))
_EMPTY_CACHE_MIN_BYTES = int(float(os.environ.get(
    "ALGAN_EMPTY_CACHE_MIN_MB", "64")) * 1024 * 1024)
_last_gc_time = 0.0


def _cuda_reclaimable_bytes():
    """Bytes torch's CUDA caching allocator holds but is not using."""
    try:
        return (torch.cuda.memory_reserved()
                - torch.cuda.memory_allocated())
    except Exception:
        return _EMPTY_CACHE_MIN_BYTES  # No telemetry; keep reclaiming.


def empty_cache(force_gc=True):
    """Reclaim freed memory back to the allocators.

    ``gc.collect()`` walks the entire Python object graph and dominates this
    call (~0.2s each on a large scene; it was costing ~40% of a small render
    when called several times per frame batch). It is only needed to break
    *reference cycles* -- reference counting already frees the (explicitly
    nulled) geometry tensors immediately -- so opportunistic callers
    (``force_gc=False``) run it only under GPU memory pressure and at most once
    per ``ALGAN_GC_MIN_INTERVAL_S`` seconds, and skip the device cache release
    when the allocator holds under ``ALGAN_EMPTY_CACHE_MIN_MB`` of reclaimable
    cache (each cached-block cudaFree synchronizes the device). ``force_gc=True``
    (OOM retries, arena sizing) always reclaims everything.
    """
    global _last_gc_time
    if force_gc:
        gc.collect()
        _last_gc_time = time.monotonic()
    elif _gpu_memory_pressure():
        now = time.monotonic()
        if now - _last_gc_time >= _GC_MIN_INTERVAL_S:
            gc.collect()
            _last_gc_time = now
    if torch.cuda.is_available():
        if force_gc or _cuda_reclaimable_bytes() >= _EMPTY_CACHE_MIN_BYTES:
            torch.cuda.empty_cache()
    if torch.mps.is_available():
        torch.mps.empty_cache()


class TempMemoryContext:
    def __init__(self, memory, clear_persist):
        self.memory = memory
        self.clear_persist = clear_persist

    def __enter__(self):
        self.initial_pointer = self.memory.current_pointer
        self.initial_reverse_pointer = self.memory.current_reverse_pointer
        return self.memory

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.memory.current_pointer = self.initial_pointer
        if self.clear_persist:
            self.memory.current_reverse_pointer = self.initial_reverse_pointer
        # Never suppress an exception.  Pointer restoration is especially
        # important on the error path: callers use a temp scope around an
        # arena-backed operation and then retry a smaller frame window.
        return False


class ManualMemory:
    def __init__(
        self,
        portion_of_available_memory_used,
        device=None,
        managed=True,
        *,
        num_bytes=None,
    ):
        if device is None:
            device = COMPUTING_DEFAULTS.render_device
        self.current_pointer = 0
        self.max_pointer = 0
        self.stack = []
        self.managed = managed

        if num_bytes is None:
            num_bytes = (
                int(get_num_available_bytes(device)
                    * portion_of_available_memory_used)
                if managed else 1
            )
        num_bytes = max(0, int(num_bytes))
        self.data = torch.empty((num_bytes,), device=device, dtype=torch.uint8)
        self.length = len(self.data)
        self.current_reverse_pointer = self.length

    def __len__(self):
        return self.length

    def get_pointers(self):
        return self.current_pointer, self.current_reverse_pointer

    def set_pointers(self, pointers):
        pointers = [*pointers]
        self.current_pointer = pointers[0]
        self.current_reverse_pointer = pointers[1]

    def get_percent_used(self):
        if not len(self):
            return 0.0
        return 1.0 - self.get_num_bytes_remaining() / len(self)

    def get_num_bytes_remaining(self):
        return self.current_reverse_pointer - self.current_pointer

    def clone(self, x, **kwargs):
        new_x = self.get_tensor(x.shape, x.dtype, **kwargs)
        new_x[:] = x
        return new_x

    def cast(self, x, dtype, **kwargs):
        new_x = self.get_tensor(x.shape, dtype=dtype, **kwargs)
        new_x[:] = x
        return new_x

    def get_tensor(self, shape, dtype=torch.float, persist=False):
        if not self.managed:
            return torch.empty(shape, dtype=dtype, device=self.data.device)
        reverse = persist

        def get_shape(shape):
            shape = [int(_.item()) if hasattr(_, "item") else int(_)
                     for _ in shape]
            # Scalars have no last dimension to widen into bytes. Represent
            # them as one element; callers still receive a scalar view below.
            scalar = not shape
            if scalar:
                shape = [1]
            element_size = dtype.itemsize
            byte_shape = list(shape)
            byte_shape[-1] *= element_size
            return shape, byte_shape, element_size, scalar

        logical_shape, byte_shape, num_bytes, scalar = get_shape(shape)

        pointer = self.current_pointer if not reverse else self.current_reverse_pointer

        def get_bap():
            remainder = pointer % num_bytes
            if not reverse:
                byte_align_offset = (num_bytes - remainder) if (remainder > 0) else 0
            else:
                byte_align_offset = -remainder
            return byte_align_offset

        byte_align_offset = get_bap()

        def get_numel():
            # return np.prod(shape) +  byte_align_offset
            nu = byte_shape[0]
            for x in byte_shape[1:]:
                nu = nu * x
            if reverse:
                nu = nu * -1
            return nu

        numel = get_numel()
        pointer = pointer + byte_align_offset
        new_pointer = pointer + numel

        def error_check():
            if ((new_pointer < self.current_pointer) if reverse else (new_pointer > self.current_reverse_pointer)):
                raise InsufficientMemoryException

        error_check()

        def get_x():
            if reverse:
                x = self.data[new_pointer:pointer]
            else:
                x = self.data[pointer:new_pointer]
            return x

        def get_data():
            x = get_x()
            if reverse:
                self.current_reverse_pointer = new_pointer
            else:
                self.current_pointer = new_pointer
            #old_max = self.max_pointer
            self.max_pointer = max(self.max_pointer, self.current_pointer + (self.length - self.current_reverse_pointer))
            #if self.max_pointer > old_max:
            #    LoggerManager.instance().log_message(f'Reached {self.max_pointer} bytes, {self.max_pointer / len(self)}%')
            x = x.view(byte_shape).view(dtype).view(logical_shape)
            if scalar:
                x = x.view(())
            return x

        return get_data()

    def reset(self):
        self.current_pointer = 0
        self.current_reverse_pointer = self.length
        self.max_pointer = 0
        self.stack = []

    def save_pointer(self):
        self.stack.append(self.current_pointer)

    def reset_pointer(self):
        self.current_pointer = self.stack[-1]
        self.stack = self.stack[:-1]

    def temp(self, clear_persist=False):
        return TempMemoryContext(self, clear_persist)
