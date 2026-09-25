"""Test/allocation attribution without intercepting Tensor subclass construction."""
from __future__ import annotations

import ctypes
import functools
import os
import threading
import traceback
from pathlib import Path

_LIB = None
_LOCAL = threading.local()
_ACTIVE = False
_FACTORIES = {}


def _set_op(value):
    _LOCAL.operation = value
    _LIB.pr144_op(value.encode('utf-8', errors='replace'))


def _description(arg):
    import torch
    if isinstance(arg, torch.Tensor):
        return f'{type(arg).__name__}{tuple(arg.shape)}:{arg.dtype}:{arg.device}'
    if isinstance(arg, (int, float, bool, str, type(None))):
        return repr(arg)[:150]
    if isinstance(arg, (tuple, list)) and len(arg) <= 8:
        return '(' + ','.join(_description(x) for x in arg) + ')'
    return str(arg) if isinstance(arg, (torch.dtype, torch.device)) else type(arg).__name__


def _factory_label(name, previous):
    @functools.wraps(previous)
    def labelled(*args, **kwargs):
        old = getattr(_LOCAL, 'operation', 'unlabelled')
        frames = [f'{Path(f.filename).name}:{f.lineno}:{f.name}'
                  for f in traceback.extract_stack(limit=16) if '/algan/' in f.filename]
        text = name + '(' + ','.join(_description(x) for x in args) + ')'
        text += ' kwargs=' + repr({k: _description(v) for k, v in kwargs.items()})
        _set_op(text + ' at ' + ';'.join(frames[-5:]))
        try:
            return previous(*args, **kwargs)
        finally:
            _set_op(old)
    return labelled


def pytest_sessionstart(session):
    import torch
    from algan.taichi_compat import submodule
    global _LIB
    assert torch.__version__.split('+', 1)[0] == '2.13.0'
    assert torch.backends.mps.is_available()
    _LIB = ctypes.CDLL(str(Path(os.environ['RUNNER_TEMP']) / 'pr144_command_provenance.dylib'))
    _LIB.pr144_install.argtypes = []
    _LIB.pr144_install.restype = ctypes.c_int
    _LIB.pr144_test.argtypes = [ctypes.c_char_p, ctypes.c_int]
    _LIB.pr144_test.restype = None
    _LIB.pr144_op.argtypes = [ctypes.c_char_p]
    _LIB.pr144_op.restype = None
    assert _LIB.pr144_install() == 1
    Kernel = submodule('lang.kernel_impl').Kernel
    previous = Kernel.__call__

    @functools.wraps(previous)
    def labelled(kernel, *args, **kwargs):
        if not _ACTIVE:
            return previous(kernel, *args, **kwargs)
        old = getattr(_LOCAL, 'operation', 'unlabelled')
        _set_op('quadrants:' + kernel.func.__name__ + '(' + ','.join(_description(x) for x in args) + ')')
        try:
            return previous(kernel, *args, **kwargs)
        finally:
            _set_op(old)
    Kernel.__call__ = labelled


def pytest_collection_modifyitems(config, items):
    mode = os.environ.get('PR144_HISTORY', 'full')
    selected = [i for i in items if '/unit_tests/' in str(i.path)
                and (Path(str(i.path)).name <= 'test_updater_mob_creation.py'
                     if mode == 'full' else Path(str(i.path)).name == 'test_updater_mob_creation.py')]
    config.hook.pytest_deselected(items=[i for i in items if i not in selected])
    items[:] = selected


def pytest_runtest_setup(item):
    global _ACTIVE
    _ACTIVE = Path(str(item.path)).name == 'test_updater_mob_creation.py'
    _LIB.pr144_test(item.nodeid.encode(), int(_ACTIVE))
    if _ACTIVE:
        import torch
        import psutil
        # Wrap only ordinary factory entry points. TorchDispatchMode is not
        # safe around Tensor.__new__(Color, ...): materializing its intermediate
        # Tensor object changes subclass construction before the test runs.
        if not _FACTORIES:
            for name in ('empty', 'empty_like', 'zeros', 'zeros_like', 'ones'):
                original = getattr(torch, name)
                _FACTORIES[name] = original
                setattr(torch, name, _factory_label('torch.' + name, original))
        print('PR144_BEFORE_TEST', item.nodeid, 'rss', psutil.Process().memory_info().rss,
              'mps', torch.mps.current_allocated_memory(), torch.mps.driver_allocated_memory(), flush=True)


def pytest_sessionfinish(session, exitstatus):
    import torch
    for name, original in _FACTORIES.items():
        setattr(torch, name, original)
    _FACTORIES.clear()
