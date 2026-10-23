"""NVRTC through ctypes: compile the device programs to PTX at runtime.

The compiler is the copy PyTorch's CUDA wheels already ship (``nvrtc64_*.dll``
in ``torch/lib`` on Windows, ``libnvrtc.so.*`` from the ``nvidia-cuda-nvrtc``
wheel or ``torch/lib`` on Linux), so no CUDA toolkit is needed. Compiling never
touches the GPU.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import sys
from pathlib import Path


class NvrtcError(RuntimeError):
    """NVRTC is missing, or a compile failed."""


_LIB = None


def _candidate_dirs():
    try:
        import torch

        yield Path(torch.__file__).parent / "lib"
    except Exception:  # pragma: no cover - torch is a hard dependency
        pass
    for entry in sys.path:
        yield Path(entry) / "torch" / "lib"
        yield Path(entry) / "nvidia" / "cuda_nvrtc" / "lib"
        yield Path(entry) / "nvidia" / "cuda_nvrtc" / "bin"


def _load():
    if sys.platform == "win32":
        for directory in _candidate_dirs():
            libraries = sorted(directory.glob("nvrtc64_*.dll"))
            if not libraries:
                continue
            os.add_dll_directory(str(directory))
            # nvrtc loads its builtins DLL by bare name through the classic
            # search order, which ignores add_dll_directory: map it first.
            for builtins in sorted(directory.glob("nvrtc-builtins64_*.dll")):
                ctypes.CDLL(str(builtins))
            return ctypes.CDLL(str(libraries[-1]))
        raise NvrtcError("NVRTC (nvrtc64_*.dll) not found beside PyTorch")
    names = ["libnvrtc.so.12", "libnvrtc.so.13", "libnvrtc.so"]
    for directory in _candidate_dirs():
        for library in sorted(directory.glob("libnvrtc.so*")):
            for builtins in sorted(directory.glob("libnvrtc-builtins.so*")):
                with contextlib.suppress(OSError):
                    ctypes.CDLL(str(builtins), mode=ctypes.RTLD_GLOBAL)
            try:
                return ctypes.CDLL(str(library))
            except OSError:
                continue
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    raise NvrtcError("NVRTC (libnvrtc.so) not found")


def lib():
    global _LIB
    if _LIB is None:
        loaded = _load()
        loaded.nvrtcGetErrorString.restype = ctypes.c_char_p
        loaded.nvrtcGetErrorString.argtypes = [ctypes.c_int]
        _LIB = loaded
    return _LIB


def _check(result, what):
    if result != 0:
        raise NvrtcError(f"{what}: {lib().nvrtcGetErrorString(result).decode()}")


def version() -> tuple[int, int]:
    major, minor = ctypes.c_int(), ctypes.c_int()
    _check(lib().nvrtcVersion(ctypes.byref(major), ctypes.byref(minor)), "version")
    return major.value, minor.value


def supported_architectures() -> list[int]:
    count = ctypes.c_int()
    _check(lib().nvrtcGetNumSupportedArchs(ctypes.byref(count)), "arch count")
    archs = (ctypes.c_int * count.value)()
    _check(lib().nvrtcGetSupportedArchs(archs), "archs")
    return list(archs)


def compile_ptx(
    source: str,
    name: str,
    options: list[str],
    headers: dict[str, str] | None = None,
) -> bytes:
    """Compile CUDA C++ ``source`` to PTX text (no trailing NUL)."""
    nvrtc = lib()
    headers = headers or {}
    program = ctypes.c_void_p()
    header_src = (ctypes.c_char_p * max(1, len(headers)))(
        *[text.encode() for text in headers.values()]
    )
    header_names = (ctypes.c_char_p * max(1, len(headers)))(
        *[key.encode() for key in headers]
    )
    _check(
        nvrtc.nvrtcCreateProgram(
            ctypes.byref(program),
            source.encode(),
            name.encode(),
            len(headers),
            header_src if headers else None,
            header_names if headers else None,
        ),
        "nvrtcCreateProgram",
    )
    try:
        encoded = [option.encode() for option in options]
        array = (ctypes.c_char_p * len(encoded))(*encoded)
        result = nvrtc.nvrtcCompileProgram(program, len(encoded), array)
        log_size = ctypes.c_size_t()
        nvrtc.nvrtcGetProgramLogSize(program, ctypes.byref(log_size))
        log = ctypes.create_string_buffer(max(1, log_size.value))
        nvrtc.nvrtcGetProgramLog(program, log)
        if result != 0:
            raise NvrtcError(
                f"compiling {name} failed: "
                f"{nvrtc.nvrtcGetErrorString(result).decode()}\n"
                f"{log.value.decode(errors='replace')}"
            )
        size = ctypes.c_size_t()
        _check(nvrtc.nvrtcGetPTXSize(program, ctypes.byref(size)), "nvrtcGetPTXSize")
        buffer = ctypes.create_string_buffer(size.value)
        _check(nvrtc.nvrtcGetPTX(program, buffer), "nvrtcGetPTX")
        return buffer.raw[: size.value].rstrip(b"\x00")
    finally:
        nvrtc.nvrtcDestroyProgram(ctypes.byref(program))
