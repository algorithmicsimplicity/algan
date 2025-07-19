import math
import traceback

import torch

from algan.logging.logger import LoggerManager
from algan.settings.defaults import COMPUTING_DEFAULTS


class InsufficientMemoryException(Exception):
    pass

def get_num_available_bytes(device=torch.device('cuda')):
    logger = LoggerManager.instance().set_class('memory')
    if device == torch.device('cuda'):
        torch.cuda.empty_cache()
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        logger.log_message(f'get_num_available_bytes device: {device}, total_bytes: {total_bytes}, free_bytes: {free_bytes}')
        return free_bytes
    elif device == torch.device('mps'):
        allocated_bytes = torch.mps.driver_allocated_memory()
        total_bytes = torch.mps.recommended_max_memory()
        free_bytes = total_bytes - allocated_bytes
        logger.log_message(f'get_num_available_bytes device: {device}, allocated_bytes: {allocated_bytes}, total_bytes:'
                           f' {total_bytes}, free_bytes: {free_bytes}, free_portion: {free_bytes / total_bytes}')
        return free_bytes
    else:
        return COMPUTING_DEFAULTS.max_cpu_memory_used

def empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.mps.is_available():
        torch.mps.empty_cache()

class ManualMemory:
    def __init__(self, portion_of_available_memory_used, device=None):
        if device is None:
            device = COMPUTING_DEFAULTS.render_device
        self.current_pointer = 0
        self.max_pointer = 0
        self.stack = []

        num_bytes = int(get_num_available_bytes(device) * portion_of_available_memory_used)
        self.data = torch.empty((num_bytes,), device=device, dtype=torch.bool)

    def __len__(self):
        return len(self.data)

    def get_percent_used(self):
        return self.current_pointer / len(self)

    def get_num_bytes_remaining(self):
        return len(self) - self.current_pointer

    def get_tensor(self, shape, dtype=torch.float):
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
            logger = LoggerManager.instance().set_class('memory')
            logger.log_message(f'Manual Memory OOM, tried to allocate {numel} bytes, memory is already {self.get_percent_used()} full.')

            raise InsufficientMemoryException

        x = self.data[self.current_pointer+byte_align_offset:self.current_pointer + numel]
        self.current_pointer = self.current_pointer + numel.item()
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
