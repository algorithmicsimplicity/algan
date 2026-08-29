"""Parser for Open Image Denoise's ``.tza`` tensor-archive format.

The format (OIDN ``core/tza.cpp``, version 2.x): a little-endian header of
``uint16`` magic ``0x41D7``, ``uint8`` major, ``uint8`` minor and a ``uint64``
offset to the tensor table; the table is a ``uint32`` tensor count followed by
one record per tensor -- ``uint16`` name length, the name bytes, ``uint8``
ndims, ``uint32`` dims, one layout character per dim (``"oihw"`` for
convolution weights, ``"x"`` for biases), one data-type character (``'f'``
float32, ``'h'`` float16) and a ``uint64`` byte offset to the raw data.

Only what the RT weights actually use is supported; anything else raises
:class:`TzaError`, which the caller treats as "no weights" rather than a
render failure.
"""

from __future__ import annotations

import struct

import torch

_MAGIC = 0x41D7
_DTYPES = {"f": (torch.float32, 4), "h": (torch.float16, 2)}


class TzaError(RuntimeError):
    """The file is not a tza archive this parser understands."""


def parse_tza(data: bytes) -> dict[str, torch.Tensor]:
    """Parse a ``.tza`` archive into ``{name: float32 CPU tensor}``.

    Weights come back in the layout the archive declares (``oihw`` matches
    ``torch.nn.functional.conv2d`` directly), converted to float32.
    """
    if len(data) < 12:
        raise TzaError("truncated tza header")
    magic, major, _minor = struct.unpack_from("<HBB", data, 0)
    if magic != _MAGIC:
        raise TzaError(f"bad tza magic {magic:#x}")
    if major != 2:
        raise TzaError(f"unsupported tza version {major}")
    (table_offset,) = struct.unpack_from("<Q", data, 4)
    if not 12 <= table_offset <= len(data) - 4:
        raise TzaError("tza table offset out of range")

    pos = table_offset
    (count,) = struct.unpack_from("<I", data, pos)
    pos += 4
    tensors: dict[str, torch.Tensor] = {}
    for _ in range(count):
        (name_len,) = struct.unpack_from("<H", data, pos)
        pos += 2
        name = data[pos : pos + name_len].decode("ascii")
        pos += name_len
        (ndims,) = struct.unpack_from("<B", data, pos)
        pos += 1
        dims = struct.unpack_from(f"<{ndims}I", data, pos)
        pos += 4 * ndims
        layout = data[pos : pos + ndims].decode("ascii")
        pos += ndims
        type_char = chr(data[pos])
        pos += 1
        (offset,) = struct.unpack_from("<Q", data, pos)
        pos += 8

        if layout not in ("oihw", "x"):
            raise TzaError(f"unsupported tensor layout {layout!r} for {name}")
        if type_char not in _DTYPES:
            raise TzaError(f"unsupported tensor type {type_char!r} for {name}")
        dtype, item = _DTYPES[type_char]
        numel = 1
        for d in dims:
            numel *= int(d)
        end = offset + numel * item
        if end > len(data):
            raise TzaError(f"tensor {name} overruns the archive")
        raw = torch.frombuffer(
            bytearray(data[offset:end]), dtype=dtype
        )  # bytearray: owned, writable memory
        tensors[name] = raw.reshape(tuple(int(d) for d in dims)).float()
    return tensors
