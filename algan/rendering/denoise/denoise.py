"""The end-to-end RT denoise filter and its process-level cache.

Faithful to OIDN's own inference pipeline (``training/infer.py`` +
``training/color.py``): the color input is scaled by an autoexposure value
(0.18 middle grey over the log-average of downsampled luminance), taken
through the PU transfer function normalised at the half-float ceiling, and
concatenated with the auxiliary albedo (in [0, 1], no transfer) and normal
(rescaled from [-1, 1] to [0, 1], matching the training dataset); the
network output is clamped non-negative and taken back through the inverse
transfer and exposure. The transfer arithmetic stays in float32; the
network itself runs at ``denoise_precision`` (half on CUDA by default,
:func:`resolve_precision`), and its output comes back as float32.

Large frames run as overlapping tiles (``denoise_tile_size``, 32-pixel
overlap; only each tile's core is kept) so peak activation memory is
bounded by the tile, not the frame; every tile is zero-padded up to the
network's 16-pixel alignment exactly as OIDN pads whole images.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from algan.logging.logger import get_logger
from algan.rendering.denoise.oidn_unet import ALIGNMENT, OidnUNet
from algan.rendering.denoise.tza import parse_tza
from algan.rendering.denoise.weights import weights_path
from algan.rendering.mps_compat import clamp_floor
from algan.rendering.raytracing import settings as rt_settings

logger = get_logger("raytracing")

# The PU transfer function: a fit of the PU2 curve normalised at 100 cd/m^2
# [Aydin et al. 2008], constants verbatim from OIDN training/color.py.
_HDR_Y_MAX = 65504.0
_PU_A = 1.41283765e03
_PU_B = 1.64593172e00
_PU_C = 4.31384981e-01
_PU_D = -2.94139609e-03
_PU_E = 1.92653254e-01
_PU_F = 6.26026094e-03
_PU_G = 9.98620152e-01
_PU_Y0 = 1.57945760e-06
_PU_Y1 = 3.22087631e-02
_PU_X0 = 2.23151711e-03
_PU_X1 = 3.70974749e-01

_TILE_OVERLAP = 32
_MIN_TILE = 4 * _TILE_OVERLAP


def _pu_forward(y):
    return torch.where(
        y <= _PU_Y0,
        _PU_A * y,
        torch.where(
            y <= _PU_Y1,
            _PU_B * torch.pow(y.clamp_min(_PU_Y0), _PU_C) + _PU_D,
            _PU_E * torch.log(y + _PU_F) + _PU_G,
        ),
    )


def _pu_inverse(x):
    return torch.where(
        x <= _PU_X0,
        x / _PU_A,
        torch.where(
            x <= _PU_X1,
            torch.pow(clamp_floor((x - _PU_D) / _PU_B, 1e-12), 1.0 / _PU_C),
            torch.exp((x - _PU_G) / _PU_E) - _PU_F,
        ),
    )


_PU_NORM_SCALE = 1.0 / float(_pu_forward(torch.tensor(_HDR_Y_MAX, dtype=torch.float64)))


def _luminance(rgb):
    return 0.212671 * rgb[..., 0] + 0.715160 * rgb[..., 1] + 0.072169 * rgb[..., 2]


def autoexposure(rgb: torch.Tensor) -> float:
    """OIDN's autoexposure: 0.18 over the log2-average of the luminance of
    ~16x16-downsampled bins, ignoring near-black bins. ``rgb`` is one linear
    ``[H, W, 3]`` frame.
    """
    lum = _luminance(rgb.float())
    h, w = lum.shape
    bins_h = max(1, (h + 15) // 16)
    bins_w = max(1, (w + 15) // 16)
    binned = F.adaptive_avg_pool2d(lum.unsqueeze(0).unsqueeze(0), (bins_h, bins_w))
    binned = binned.flatten()
    binned = binned[binned > 1e-8]
    if binned.numel() == 0:
        return 1.0
    return float(0.18 / torch.exp2(torch.log2(binned).mean()))


class Denoiser:
    """The RT HDR filter over one loaded network."""

    def __init__(self, net: OidnUNet):
        self.net = net

    def _run_aligned(self, x: torch.Tensor) -> torch.Tensor:
        """Run the network on one ``[1, 9, h, w]`` tile, zero-padded up to
        the 16-pixel alignment and cropped back (OIDN pads whole images the
        same way).
        """
        h, w = int(x.shape[2]), int(x.shape[3])
        pad_h = (-h) % ALIGNMENT
        pad_w = (-w) % ALIGNMENT
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        y = self.net(x)
        return y[:, :, :h, :w]

    def __call__(
        self,
        color: torch.Tensor,
        albedo: torch.Tensor,
        normal: torch.Tensor,
    ) -> torch.Tensor:
        """Denoise linear HDR ``color`` (``[F, H, W, 3]``, values in
        [0, inf)) guided by ``albedo`` (``[F, H, W, 3]`` in [0, 1]) and
        ``normal`` (``[F, H, W, 3]``, roughly unit, any world frame).
        Returns denoised linear HDR of the same shape and dtype float32.
        """
        frames, height, width = color.shape[0], color.shape[1], color.shape[2]
        tile = max(_MIN_TILE, int(rt_settings.denoise_tile_size))
        core = tile - 2 * _TILE_OVERLAP
        out = torch.empty(
            (frames, height, width, 3), dtype=torch.float32, device=color.device
        )
        albedo9 = albedo.float().clamp(0.0, 1.0)
        normal9 = normal.float().clamp(-1.0, 1.0) * 0.5 + 0.5
        for f in range(frames):
            frame = color[f].float().clamp_min(0.0)
            exposure = autoexposure(frame)
            transferred = _pu_forward(frame * exposure) * _PU_NORM_SCALE
            image = (
                torch.cat((transferred, albedo9[f], normal9[f]), -1)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .contiguous()
            )
            result = torch.empty(
                (3, height, width), dtype=torch.float32, device=color.device
            )
            for ty in range(0, height, core):
                for tx in range(0, width, core):
                    y0 = max(ty - _TILE_OVERLAP, 0)
                    x0 = max(tx - _TILE_OVERLAP, 0)
                    y1 = min(ty + core + _TILE_OVERLAP, height)
                    x1 = min(tx + core + _TILE_OVERLAP, width)
                    piece = self._run_aligned(image[:, :, y0:y1, x0:x1])[0]
                    cy1 = min(ty + core, height)
                    cx1 = min(tx + core, width)
                    result[:, ty:cy1, tx:cx1] = piece[
                        :, ty - y0 : cy1 - y0, tx - x0 : cx1 - x0
                    ]
            restored = _pu_inverse(
                result.clamp_min(0.0).permute(1, 2, 0) / _PU_NORM_SCALE
            )
            out[f] = restored / exposure
        return out


#: Per-(device, precision) cache. "" marks a load that failed (stay off).
_denoisers: dict[str, Denoiser | str] = {}


def resolve_precision(device) -> tuple[torch.dtype, bool]:
    """``(dtype, channels_last)`` the network runs with on ``device`` under
    the live ``denoise_precision`` setting: ``"auto"`` is half precision with
    channels-last activations on CUDA and float32 elsewhere (a CPU half
    convolution is slower than float32, and MPS half support is uneven).
    """
    choice = str(rt_settings.denoise_precision).strip().lower()
    if choice == "auto":
        choice = "fp16" if torch.device(device).type == "cuda" else "fp32"
    if choice == "fp16":
        return torch.float16, True
    return torch.float32, False


def get_denoiser(device) -> Denoiser | None:
    """The cached :class:`Denoiser` for ``device`` at the live
    ``denoise_precision``, or ``None`` when the weights cannot be had or
    loaded (warned once; the render continues without denoising).
    """
    dtype, channels_last = resolve_precision(device)
    key = f"{device}:{'fp16' if dtype == torch.float16 else 'fp32'}"
    cached = _denoisers.get(key)
    if cached is not None:
        return cached if isinstance(cached, Denoiser) else None
    path = weights_path()
    if path is None:
        _denoisers[key] = ""
        return None
    try:
        with open(path, "rb") as f:
            tensors = parse_tza(f.read())
        net = OidnUNet(tensors, device, dtype=dtype, channels_last=channels_last)
    except Exception as exc:  # TzaError, WeightShapeError, IO, device errors
        logger.warning(
            f"Could not load the denoiser weights at {path} ({exc}); "
            f"rendering without denoising."
        )
        _denoisers[key] = ""
        return None
    denoiser = Denoiser(net)
    _denoisers[key] = denoiser
    return denoiser


def _reset_for_tests() -> None:
    _denoisers.clear()
