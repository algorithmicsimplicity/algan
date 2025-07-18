import math

import torch

from algan import DEFAULT_CPU_MEMORY_USED
from algan.defaults.device_defaults import DEFAULT_RENDER_DEVICE

class InsufficientMemoryException(Exception):
    pass

def get_num_available_bytes(cuda=True):
    if cuda:
        torch.cuda.empty_cache()
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return free_bytes#torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)
    else:
        return DEFAULT_CPU_MEMORY_USED

class ManualMemory:
    def __init__(self, portion_of_available_memory_used):
        self.is_cpu = DEFAULT_RENDER_DEVICE == torch.device('cpu')
        self.current_pointer = 0
        self.max_pointer = 0
        self.stack = []
        if self.is_cpu:
            return

        num_bytes = int(get_num_available_bytes() * portion_of_available_memory_used)
        self.data = torch.empty((1 if self.is_cpu else num_bytes,), device=DEFAULT_RENDER_DEVICE, dtype=torch.bool)

    def __len__(self):
        return len(self.data)

    def get_num_bytes_remaining(self):
        if self.is_cpu:
            return DEFAULT_CPU_MEMORY_USED
        return len(self) - self.current_pointer

    def get_tensor(self, shape, dtype=torch.float):
        if self.is_cpu:
            return torch.empty(shape, dtype=dtype)
        shape = [_ for _ in shape]
        num_bytes = 1
        if dtype in [torch.int, torch.float]:
            num_bytes = 4
        elif dtype in [torch.long, torch.double]:
            num_bytes = 8
        shape[-1] = shape[-1] * num_bytes

        remainder = (self.current_pointer % num_bytes)
        byte_align_offset = (num_bytes - remainder) if (remainder > 0) else 0

        numel = torch.tensor(shape).prod() + byte_align_offset
        if (self.current_pointer + numel) >= len(self):
            raise InsufficientMemoryException

        x = self.data[self.current_pointer+byte_align_offset:self.current_pointer + numel]
        self.current_pointer = self.current_pointer + numel
        self.max_pointer = max(self.max_pointer, self.current_pointer)
        x = x.view(shape).view(dtype)
        return x

    def reset(self):
        self.current_pointer = 0
        self.max_pointer = 0
        self.stack = []

    def save_pointer(self):
        self.stack.append(self.current_pointer)

    def reset_pointer(self):
        self.current_pointer = self.stack[-1]
        self.stack = self.stack[:-1]
