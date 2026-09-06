"""Open Image Denoise's RT U-Net, as plain functional torch calls.

The topology is the ``UNet`` of OIDN's training repository (``model.py``):
3x3 convolutions with padding 1, ReLU after every convolution except the
last, 2x2 max-pooling on the way down, 2x nearest-neighbour upsampling on
the way up, and skip connections that concatenate the POOLED encoder
activations (and finally the raw input). The ``rt_hdr_alb_nrm`` weights pin
every channel count; :func:`_check_shapes` verifies the parsed archive
against them so a wrong or truncated file fails at load, not at inference.

Held as a plain object over ``torch.nn.functional`` rather than an
``nn.Module``: algan runs under a process-global ``torch.inference_mode``,
and functional convolutions over ordinary tensors are indifferent to it.

Images must be padded to multiples of 16 (four pooling levels) before
:meth:`OidnUNet.__call__`; the caller does that (``denoise.py``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

ALIGNMENT = 16

#: name -> (out_channels, in_channels) of every convolution, in forward
#: order. 9 input channels: color, albedo and normal, 3 each.
RT_HDR_ALB_NRM_LAYERS = {
    "enc_conv0": (32, 9),
    "enc_conv1": (32, 32),
    "enc_conv2": (48, 32),
    "enc_conv3": (64, 48),
    "enc_conv4": (80, 64),
    "enc_conv5a": (96, 80),
    "enc_conv5b": (96, 96),
    "dec_conv4a": (112, 160),  # 96 upsampled + 64 skip (pool3)
    "dec_conv4b": (112, 112),
    "dec_conv3a": (96, 160),  # 112 upsampled + 48 skip (pool2)
    "dec_conv3b": (96, 96),
    "dec_conv2a": (64, 128),  # 96 upsampled + 32 skip (pool1)
    "dec_conv2b": (64, 64),
    "dec_conv1a": (64, 73),  # 64 upsampled + the 9-channel input
    "dec_conv1b": (32, 64),
    "dec_conv0": (3, 32),
}


class WeightShapeError(RuntimeError):
    """The parsed archive does not carry the RT U-Net's tensors."""


def _check_shapes(tensors: dict[str, torch.Tensor]) -> None:
    for name, (oc, ic) in RT_HDR_ALB_NRM_LAYERS.items():
        w = tensors.get(f"{name}.weight")
        b = tensors.get(f"{name}.bias")
        if w is None or b is None:
            raise WeightShapeError(f"missing tensors for layer {name}")
        if tuple(w.shape) != (oc, ic, 3, 3) or tuple(b.shape) != (oc,):
            raise WeightShapeError(
                f"layer {name} has shape {tuple(w.shape)}/{tuple(b.shape)}, "
                f"expected {(oc, ic, 3, 3)}/{(oc,)}"
            )


class OidnUNet:
    """The RT filter network. Input/output are ``[N, C, H, W]`` tensors in
    the transfer-function domain; H and W must be multiples of 16.

    ``dtype`` is the arithmetic precision the convolutions run at (the input
    is cast on the way in and the output is returned as float32 whatever it
    is), and ``channels_last`` selects NHWC activations and weights, which is
    the layout cuDNN's half-precision tensor-core kernels want. Both are
    chosen by ``denoise.get_denoiser`` from ``denoise_precision``.
    """

    def __init__(
        self,
        tensors: dict[str, torch.Tensor],
        device,
        dtype=torch.float32,
        channels_last=False,
    ):
        _check_shapes(tensors)
        self.device = device
        self.dtype = dtype
        self.channels_last = bool(channels_last)
        self.weights = {
            name: self._place(tensors[name])
            for name in (
                f"{layer}.{kind}"
                for layer in RT_HDR_ALB_NRM_LAYERS
                for kind in ("weight", "bias")
            )
        }

    def _place(self, tensor):
        tensor = tensor.to(device=self.device, dtype=self.dtype)
        if self.channels_last and tensor.dim() == 4:
            return tensor.contiguous(memory_format=torch.channels_last)
        return tensor.contiguous()

    def _conv(self, x, name, activate=True):
        x = F.conv2d(
            x, self.weights[f"{name}.weight"], self.weights[f"{name}.bias"], padding=1
        )
        return F.relu(x, inplace=True) if activate else x

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] % ALIGNMENT or x.shape[-2] % ALIGNMENT:
            raise ValueError(f"input {tuple(x.shape)} is not {ALIGNMENT}-aligned")
        x = self._place(x)
        pool = lambda t: F.max_pool2d(t, 2, 2)  # noqa: E731
        up = lambda t: F.interpolate(t, scale_factor=2, mode="nearest")  # noqa: E731

        y = self._conv(x, "enc_conv0")
        pool1 = pool(self._conv(y, "enc_conv1"))
        pool2 = pool(self._conv(pool1, "enc_conv2"))
        pool3 = pool(self._conv(pool2, "enc_conv3"))
        y = pool(self._conv(pool3, "enc_conv4"))
        y = self._conv(y, "enc_conv5a")
        y = self._conv(y, "enc_conv5b")

        y = torch.cat((up(y), pool3), 1)
        y = self._conv(y, "dec_conv4a")
        y = self._conv(y, "dec_conv4b")
        y = torch.cat((up(y), pool2), 1)
        y = self._conv(y, "dec_conv3a")
        y = self._conv(y, "dec_conv3b")
        y = torch.cat((up(y), pool1), 1)
        y = self._conv(y, "dec_conv2a")
        y = self._conv(y, "dec_conv2b")
        y = torch.cat((up(y), x), 1)
        y = self._conv(y, "dec_conv1a")
        y = self._conv(y, "dec_conv1b")
        return self._conv(y, "dec_conv0", activate=False).float()
