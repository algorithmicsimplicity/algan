"""Temporary offset-write diagnostic; never enters the implementation tree."""
import json
import torch
from algan import SETTINGS
from algan.rendering.taichi_runtime import init_taichi, _live_arch
from algan.rendering.raytracing.array_copy_taichi import gather_rows_into
from algan.utils.memory_utils import ManualMemory
init_taichi()
device = SETTINGS.computing.render_device
print('PROBE_ENV', torch.__version__, device, _live_arch(), flush=True)
for dtype in (torch.float32, torch.float16, torch.int32, torch.int64, torch.bool):
    for width in (1, 4):
        for mode in ('index_select', 'copy_indexing', 'local_copy'):
            memory = ManualMemory(0, device=device, num_bytes=1 << 16)
            memory.data.fill_(219)
            head = memory.get_tensor((64,), torch.int64); head.fill_(719)
            out = memory.get_tensor((6, width), dtype)
            tail = memory.get_tensor((32,), torch.int64); tail.fill_(827)
            source = torch.arange(32 * width, device=device).reshape(32, width).to(dtype)
            index = torch.tensor([3, 7, 11, 3, 20, 1], dtype=torch.int64, device=device)
            expected = source.cpu()[index.cpu()]
            before = memory.data.cpu().clone()
            start = out.data_ptr() - memory.data.data_ptr()
            end = start + out.numel() * out.element_size()
            error = None
            try:
                if mode == 'index_select':
                    torch.index_select(source, 0, index, out=out)
                elif mode == 'copy_indexing':
                    out.copy_(source[index])
                else:
                    # Byte-preserving same-width payloads for bool/half.
                    sd = source.view(torch.uint8) if dtype == torch.bool else source
                    od = out.view(torch.uint8) if dtype == torch.bool else out
                    gather_rows_into(sd.reshape(-1), index, od.reshape(-1), 6, width)
                actual = out.cpu()
            except Exception as exc:
                error = repr(exc)
                actual = None
            after = memory.data.cpu()
            delta = torch.nonzero(before != after).flatten()
            outside = delta[(delta < start) | (delta >= end)]
            print('OFFSET_PROBE', json.dumps({'dtype': str(dtype), 'width': width, 'mode': mode, 'offset': start, 'exact': actual is not None and torch.equal(actual, expected), 'outside_bytes': outside.numel(), 'outside_first': outside[:16].tolist(), 'head': head.cpu().tolist()[:6], 'error': error}), flush=True)
