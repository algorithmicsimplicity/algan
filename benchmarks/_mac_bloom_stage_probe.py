"""Small, fixed-size mathematical references for both MPS bloom resize paths."""
import json
import os
from pathlib import Path

os.environ.update(ALGAN_RENDER_DEVICE="mps", ALGAN_ANIMATION_DEVICE="cpu",
                  ALGAN_TORCH_COMPILE="0", ALGAN_USE_DAEMON="0")
import numpy as np
import torch
import torch.nn.functional as F
from algan.rendering.post_processing import bloom as bloom
from algan.rendering.post_processing import bloom_kernels_taichi as kernels
from algan.rendering import mps_zero_copy as zc
from algan.rendering.taichi_runtime import init_taichi
from algan.utils.memory_utils import ManualMemory

init_taichi()
assert torch.backends.mps.is_available() and zc.zero_copy_available()
output = Path("algan_outputs/bloom_stage_probe")
output.mkdir(parents=True, exist_ok=True)
rows = []
arrays = {}


def comparison(name, actual, reference):
    a, b = actual.float().cpu(), reference.float().cpu()
    difference = (a-b).abs()
    where = (difference > 1e-4).nonzero()
    examples = []
    for index in where[:8].tolist():
        ix = tuple(index)
        examples.append({"index": index, "actual": a[ix].item(), "reference": b[ix].item()})
    row = {"name": name, "shape": list(a.shape), "max": difference.max().item(),
           "mean": difference.mean().item(), "over_1e4": len(where), "examples": examples}
    rows.append(row)
    arrays[name.replace("/", "_")] = a.numpy()
    print("BLOOM_STAGE " + json.dumps(row), flush=True)


def allocation(device, shape, dtype=torch.float32):
    memory = ManualMemory(0, device=device, num_bytes=64*2**20)
    prefix = memory.get_tensor((137,), torch.uint8)
    prefix.fill_(91)
    tensor = memory.get_tensor(shape, dtype)
    return memory, tensor, prefix


for height, width, scale, channels in ((127, 193, 3, 3), (64, 80, 4, 4), (270, 480, 8, 3)):
    torch.manual_seed(76)
    source = torch.rand((1, height, width, channels))
    reference_down = F.interpolate(source.permute(0, 3, 1, 2), scale_factor=1/scale,
                                   mode="bilinear", align_corners=False, antialias=True)
    reference_up = F.interpolate(reference_down, size=(height, width), mode="bilinear", align_corners=False)
    for arm in ("fallback", "metal"):
        kernels.can_use_bloom_taichi = lambda device, arm=arm: arm == "metal" and torch.device(device).type == "mps"
        mem, tensor, prefix = allocation("mps", source.shape)
        tensor.copy_(source)
        down = mem.get_tensor(reference_down.shape)
        down.fill_(-91)
        before = dict(zc.STATS)
        bloom._downsample_bloom(tensor, down, mem, scale)
        comparison(f"{height}x{width}/{arm}/down", down, reference_down)
        assert (prefix.cpu() == 91).all()
        # Independent reference input for the upsampler, so a downsample error
        # cannot contaminate the attribution to the second kernel.
        up_input = mem.get_tensor(reference_down.shape)
        up_input.copy_(reference_down)
        up = mem.get_tensor(reference_up.shape)
        up.fill_(-91)
        bloom._upsample_bloom(up_input, up, mem)
        comparison(f"{height}x{width}/{arm}/up", up, reference_up)
        assert (prefix.cpu() == 91).all()
        print("ENGAGEMENT", arm, zc.STATS["converted_launches"]-before["converted_launches"], flush=True)

# Capture each whole-bloom intermediate, including FFT results, to distinguish
# an arithmetic discrepancy from arena-layout-sensitive corruption elsewhere.
original_down, original_up, original_fft = bloom._downsample_bloom, bloom._upsample_bloom, bloom.fft_conv1d
captures = {}
arm = "cpu"
fft_index = 0


def down_capture(*args, **kwargs):
    result = original_down(*args, **kwargs)
    captures[arm+"/down"] = args[1].cpu().clone()
    return result


def up_capture(*args, **kwargs):
    result = original_up(*args, **kwargs)
    captures[arm+"/up_input"] = args[0].cpu().clone()
    captures[arm+"/up"] = args[1].cpu().clone()
    return result


def fft_capture(*args, **kwargs):
    global fft_index
    result = original_fft(*args, **kwargs)
    captures[arm+f"/fft{fft_index}"] = result.cpu().clone()
    fft_index += 1
    return result


bloom._downsample_bloom, bloom._upsample_bloom, bloom.fft_conv1d = down_capture, up_capture, fft_capture
bloom._should_bypass_bloom = lambda: False
torch.manual_seed(1238)
frames = torch.randint(1, 220, (1, 127, 193, 4), dtype=torch.uint8)
frames[..., 3] = 100
for arm in ("cpu", "fallback", "metal"):
    kernels.can_use_bloom_taichi = lambda device: arm == "metal" and torch.device(device).type == "mps"
    mem, tensor, prefix = allocation("cpu" if arm == "cpu" else "mps", frames.shape, torch.uint8)
    tensor.copy_(frames)
    fft_index = 0
    captures[arm+"/final"] = bloom.bloom_filter(tensor, memory=mem, scale_factor=64).cpu().clone()
    assert (prefix.cpu() == 91).all()
for arm in ("fallback", "metal"):
    for stage in ("down", "fft0", "fft1", "fft2", "fft3", "up_input", "up", "final"):
        comparison("whole/"+arm+"/"+stage, captures[arm+"/"+stage], captures["cpu/"+stage])
np.savez_compressed(output/"arrays.npz", **arrays)
(output/"comparisons.json").write_text(json.dumps(rows, indent=2))
