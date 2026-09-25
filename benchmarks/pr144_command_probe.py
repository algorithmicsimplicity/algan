"""Diagnose command-buffer errors without adding synchronization or retries."""
from __future__ import annotations

import ctypes
import json
import os
import traceback
from pathlib import Path

import pytest

_native = None


def pytest_sessionstart(session):
    global _native
    import torch
    from algan.taichi_compat import submodule

    assert torch.__version__.split("+")[0] == "2.13.0"
    assert torch.backends.mps.is_available()
    output = Path("algan_outputs")
    output.mkdir(exist_ok=True)
    _native = ctypes.CDLL(str(Path(os.environ["RUNNER_TEMP"]) / "pr144_command_trace.dylib"))
    _native.pr144_install.argtypes = [ctypes.c_char_p]
    _native.pr144_install.restype = ctypes.c_int
    _native.pr144_context.argtypes = [ctypes.c_char_p]
    _native.pr144_context.restype = None
    _native.pr144_enable.argtypes = [ctypes.c_int]
    _native.pr144_enable.restype = None
    _native.pr144_errors.argtypes = []
    _native.pr144_errors.restype = ctypes.c_uint
    assert _native.pr144_install(str(output / f"commands-{os.getpid()}.jsonl").encode()) == 1
    session.config._pr144_native = _native
    kernel_type = submodule("lang.kernel_impl").Kernel
    original = kernel_type.__call__

    def traced(self, *args, **kwargs):
        name = getattr(getattr(self, "func", None), "__name__", "unknown")
        value = f"quadrants:{name}"
        _native.pr144_context(value.encode())
        os.environ["PR144_OPERATION"] = value
        try:
            return original(self, *args, **kwargs)
        finally:
            _native.pr144_context(b"")
            os.environ.pop("PR144_OPERATION", None)

    kernel_type.__call__ = traced


def pytest_collection_modifyitems(config, items):
    if os.environ.get("PR144_HISTORY") == "updaters":
        selected = [item for item in items if "test_updater_mob_creation.py" in item.nodeid]
        config.hook.pytest_deselected(items=[item for item in items if item not in selected])
        items[:] = selected


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    os.environ["PR144_CURRENT_TEST"] = item.nodeid
    enabled = "test_updater_mob_creation.py" in item.nodeid
    _native.pr144_enable(int(enabled))
    output = Path("algan_outputs") / f"python-operations-{os.getpid()}.jsonl"

    class Operations(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            def describe(value):
                if isinstance(value, torch.Tensor):
                    return {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device), "stride": list(value.stride()), "offset": value.storage_offset()}
                if isinstance(value, (tuple, list)):
                    return [describe(x) for x in value[:12]]
                return str(value)[:200]
            details = {"op": str(func), "args": describe(args), "kwargs": {k: describe(v) for k, v in (kwargs or {}).items()}}
            value = json.dumps(details)
            _native.pr144_context(value.encode())
            os.environ["PR144_OPERATION"] = value
            previous = _native.pr144_errors()
            try:
                result = func(*args, **(kwargs or {}))
                if _native.pr144_errors() != previous:
                    details["test"] = item.nodeid
                    details["python_stack"] = traceback.format_stack()
                    with output.open("a") as stream:
                        stream.write(json.dumps(details) + "\n")
                    raise RuntimeError("First Metal command-buffer failure; see commands and python-operations artifacts")
                return result
            finally:
                _native.pr144_context(b"")
                os.environ.pop("PR144_OPERATION", None)

    if enabled:
        with Operations():
            yield
    else:
        yield
    if _native.pr144_errors():
        item.session.shouldstop = "Native GPU error observed; stop before secondary pipeline failures"
    _native.pr144_enable(0)
