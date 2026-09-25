"""Tag native calls without changing Torch's tensor dispatch/subclass handling."""
from __future__ import annotations

import ctypes
import functools
import json
import os
import traceback
from pathlib import Path

import pytest

_native = None


def pytest_sessionstart(session):
    global _native
    import torch

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


def pytest_collection_modifyitems(config, items):
    if os.environ.get("PR144_HISTORY") == "updaters":
        selected = [item for item in items if "test_updater_mob_creation.py" in item.nodeid]
        config.hook.pytest_deselected(items=[item for item in items if item not in selected])
        items[:] = selected


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    os.environ["PR144_CURRENT_TEST"] = item.nodeid
    enabled = "test_updater_mob_creation.py" in item.nodeid
    _native.pr144_enable(int(enabled))
    output = Path("algan_outputs") / f"python-operations-{os.getpid()}.jsonl"

    def wrapped(original, make_label):
        @functools.wraps(original)
        def call(*args, **kwargs):
            label = make_label(args, kwargs)
            previous = os.environ.get("PR144_OPERATION", "")
            _native.pr144_context(label.encode())
            os.environ["PR144_OPERATION"] = label
            before = _native.pr144_errors()
            try:
                result = original(*args, **kwargs)
                if _native.pr144_errors() != before:
                    with output.open("a") as stream:
                        stream.write(json.dumps({"test": item.nodeid, "op": label, "python_stack": traceback.format_stack()}) + "\n")
                    raise RuntimeError("Native GPU command-buffer failure; see commands and python-operations artifacts")
                return result
            finally:
                _native.pr144_context(previous.encode())
                os.environ["PR144_OPERATION"] = previous
        return call

    if enabled:
        import torch
        from algan.taichi_compat import submodule
        from algan.utils.memory_utils import ManualMemory

        kernel_type = submodule("lang.kernel_impl").Kernel
        # Plain wrappers forward the exact original calls and return objects.
        # Do not use TorchDispatchMode: it breaks Color's Tensor subclass
        # construction, even during pytest's ordinary settings snapshot.
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(torch, "empty", wrapped(torch.empty, lambda args, kwargs: "torch.empty " + str(args)[:200] + " " + str(kwargs)[:200]))
            patch.setattr(ManualMemory, "__init__", wrapped(ManualMemory.__init__, lambda args, kwargs: "ManualMemory.__init__"))
            patch.setattr(kernel_type, "__call__", wrapped(kernel_type.__call__, lambda args, kwargs: "quadrants:" + getattr(getattr(args[0], "func", None), "__name__", "unknown")))
            yield
    else:
        yield
    if _native.pr144_errors():
        item.session.shouldstop = "Native GPU error observed; stop before secondary pipeline failures"
    _native.pr144_enable(0)
