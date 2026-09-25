"""No Algan, Quadrants, native hooks, or asynchronous copy operations."""
import time

import psutil
import torch


def main():
    assert torch.backends.mps.is_available()
    process = psutil.Process()
    size = 1202590840
    assert size < 0.5 * torch.mps.recommended_max_memory()
    start = time.monotonic()
    print('PLAIN_ALLOCATOR', torch.__version__, size, flush=True)
    for iteration in range(768):
        assert psutil.virtual_memory().available > 1536 * 1024**2, 'memory safety stop before allocation'
        data = torch.empty(size, dtype=torch.uint8, device='mps')
        data[:8].fill_(7)
        data[-8:].fill_(13)
        head = data[:8].cpu().tolist()
        tail = data[-8:].cpu().tolist()
        assert head == [7] * 8 and tail == [13] * 8, (iteration, head, tail)
        del data
        torch.mps.empty_cache()
        small = torch.ones(8, device='mps')
        values = small.cpu().tolist()
        assert values == [1.0] * 8, (iteration, values)
        del small
        torch.mps.empty_cache()
        if iteration % 32 == 31:
            print('PLAIN_CHURN', iteration + 1, 'rss', process.memory_info().rss, 'driver', torch.mps.driver_allocated_memory(), 'elapsed', time.monotonic() - start, flush=True)
    print('PLAIN_CHURN_PASS', 768, flush=True)


if __name__ == '__main__':
    main()
