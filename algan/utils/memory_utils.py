import math
import traceback

import torch
import numpy as np

from algan import not_compiled
from algan.logging.logger import LoggerManager
from algan.settings.defaults import COMPUTING_DEFAULTS
from algan.constants.math import GIGABYTES


class InsufficientMemoryException(Exception):
    pass


def get_num_available_bytes(device=torch.device("cuda")):
    if device == torch.device("cuda"):
        torch.cuda.empty_cache()
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return free_bytes
    elif device == torch.device("mps"):
        allocated_bytes = torch.mps.driver_allocated_memory()
        total_bytes = torch.mps.recommended_max_memory()
        free_bytes = total_bytes - allocated_bytes
        free_bytes = min(free_bytes, 1 * GIGABYTES)
        return free_bytes
    else:
        return COMPUTING_DEFAULTS.max_cpu_memory_used


def empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.mps.is_available():
        torch.mps.empty_cache()


class TempMemoryContext:
    def __init__(self, memory):
        self.memory = memory

    def __enter__(self):
        self.initial_pointer = self.memory.current_pointer
        self.initial_reverse_pointer = self.memory.current_reverse_pointer

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return False
        self.memory.current_pointer = self.initial_pointer
        #self.memory.current_reverse_pointer = self.initial_reverse_pointer
        return True


class ManualMemory:
    def __init__(self, portion_of_available_memory_used, device=None):
        if device is None:
            device = COMPUTING_DEFAULTS.render_device
        self.current_pointer = 0
        self.max_pointer = 0
        self.stack = []

        num_bytes = int(
            get_num_available_bytes(device) * portion_of_available_memory_used
        )
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
        return self.get_num_bytes_remaining() / len(self)

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

    @not_compiled
    def get_tensor(self, shape, dtype=torch.float, persist=False):
        #return torch.empty(shape, dtype=dtype, device=self.data.device)
        reverse = persist
        def get_shape(shape):
            shape = [_ for _ in shape]
            num_bytes = 1
            if dtype in [torch.int, torch.float]:
                num_bytes = 4
            elif dtype in [torch.long, torch.double]:
                num_bytes = 8
            shape[-1] = shape[-1] * num_bytes
            return shape, num_bytes

        shape, num_bytes = get_shape(shape)

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
            nu = shape[0]
            for x in shape[1:]:
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
            x = x.view(shape).view(dtype)
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

    def temp(self):
        return TempMemoryContext(self)
