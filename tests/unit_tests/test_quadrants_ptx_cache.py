"""Algan's half of Quadrants patches 0010 and 0011 (``quadrants_patches/README.md``).

* 0010 adds ``offline_cache_ptx_by_content_hash``: on CUDA a cached kernel's
  PTX is found by the SHA-256 its ``.qdc`` already stores, and its IR is
  parsed only on a PTX miss. ``taichi_init_kwargs`` asks for it on CUDA when
  the installed wheel has the field, unless ``ALGAN_PTX_CACHE_BY_CONTENT_HASH=0``.
* 0011 adds ``Program.prepare_kernel``: a first launch's work without the
  launch. A CUDA precompile worker calls it on every spec, so the PTX cache
  and the driver's compute cache are warm before the render reaches the
  kernel, unless ``ALGAN_PRECOMPILE_PTX=0``.

Neither switch may move a key: both arms of each A/B must find the same
index entries and artifacts, or the A/B would time a cache miss. And neither
may break a wheel without the patches, which is every wheel published so far.

Neither can be seen working here -- both act in CUDA's launch path. The GPU
checks are ``quadrants_patches/verify_ptx_cache.py`` on a patched wheel and a
fresh-process / cold-cache render A/B per switch.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import algan.taichi_compat
from algan.rendering import kernel_precompile as kp
from algan.rendering import taichi_runtime
from algan.taichi_compat import BACKEND, program, ti

quadrants_only = pytest.mark.skipif(
    BACKEND != "quadrants", reason="the patches are Quadrants-only"
)

_FIELD = "offline_cache_ptx_by_content_hash"


# --- 0010: the init keyword ---------------------------------------------------


@pytest.mark.parametrize(
    ("device", "flag", "supported", "passed"),
    [
        ("cuda", None, True, True),
        ("cuda", "1", True, True),
        ("cuda", "0", True, False),
        ("cuda", None, False, False),
        ("cpu", None, True, False),
        ("cpu", "1", True, False),
    ],
)
def test_the_content_hash_keyword_is_passed_on_cuda_where_the_wheel_has_it(
    monkeypatch, device, flag, supported, passed
):
    """On by default; never passed to a wheel that would reject it, nor off CUDA.

    Left out rather than passed as ``False`` in the off arm, so that arm's
    ``init`` is exactly what it was before the switch existed.
    """
    monkeypatch.setattr(taichi_runtime, "render_device", lambda: torch.device(device))
    monkeypatch.setattr(
        taichi_runtime, "ptx_cache_by_content_hash_supported", lambda: supported
    )
    if flag is None:
        monkeypatch.delenv("ALGAN_PTX_CACHE_BY_CONTENT_HASH", raising=False)
    else:
        monkeypatch.setenv("ALGAN_PTX_CACHE_BY_CONTENT_HASH", flag)
    kwargs = taichi_runtime.taichi_init_kwargs()
    if passed:
        assert kwargs[_FIELD] is True
    else:
        assert _FIELD not in kwargs


def test_support_is_read_off_the_compilers_config_type(monkeypatch):
    """Asked of the type, which exists before ``init``; Taichi never has it."""
    from algan.taichi_compat import submodule

    real = submodule

    def with_config(config_type):
        def fake(name):
            if name == "_lib.core.quadrants_python":
                return SimpleNamespace(CompileConfig=config_type)
            return real(name)

        monkeypatch.setattr(algan.taichi_compat, "submodule", fake)

    monkeypatch.setattr(taichi_runtime, "BACKEND", "quadrants")
    with_config(type("Patched", (), {_FIELD: property(lambda self: False)}))
    assert taichi_runtime.ptx_cache_by_content_hash_supported() is True
    with_config(type("Published", (), {"offline_cache": None}))
    assert taichi_runtime.ptx_cache_by_content_hash_supported() is False
    monkeypatch.setattr(taichi_runtime, "BACKEND", "taichi")
    with_config(type("Patched", (), {_FIELD: property(lambda self: False)}))
    assert taichi_runtime.ptx_cache_by_content_hash_supported() is False


@quadrants_only
def test_support_matches_the_installed_wheel():
    """True exactly when the installed build carries 0010's field."""
    from algan.taichi_compat import submodule

    config_type = submodule("_lib.core.quadrants_python").CompileConfig
    assert taichi_runtime.ptx_cache_by_content_hash_supported() is hasattr(
        config_type, _FIELD
    )


# --- neither switch moves a key -----------------------------------------------


def test_the_switches_reach_neither_the_source_key_nor_a_precompile_context(
    monkeypatch,
):
    from algan.utils import taichi_source_key as sk

    before = sk._environment_fingerprint()
    context_before = kp.context_id(kp.current_context())
    monkeypatch.setenv("ALGAN_PTX_CACHE_BY_CONTENT_HASH", "0")
    monkeypatch.setenv("ALGAN_PRECOMPILE_PTX", "0")
    assert sk._environment_fingerprint() == before
    assert kp.context_id(kp.current_context()) == context_before


def test_the_config_field_is_out_of_both_source_keys():
    """By its ``offline_cache`` prefix, in Algan's source key and Quadrants' own.

    The field decides which file serves a kernel's PTX, never what the PTX is;
    the compiler's C++ key lists its fields explicitly and leaves it out too.
    """
    from algan.utils import taichi_source_key as sk

    assert _FIELD.startswith(sk._CONFIG_EXCLUDE_PREFIXES)
    if BACKEND == "quadrants":
        from algan.taichi_compat import submodule

        hasher = submodule("lang._fast_caching.config_hasher")
        assert any(_FIELD.startswith(prefix) for prefix in hasher.EXCLUDE_PREFIXES)


# --- 0011: preparing in the precompile worker ---------------------------------


class _Program:
    def __init__(self, fail=False):
        self.prepared = []
        self.fail = fail

    def prepare_kernel(self, data):
        if self.fail:
            raise RuntimeError("ptxas said no")
        self.prepared.append(data)


def _on_arch(monkeypatch, arch):
    from algan.taichi_compat import submodule

    impl = submodule("lang.impl")
    monkeypatch.setattr(impl, "current_cfg", lambda: SimpleNamespace(arch=arch))


def test_a_kernel_is_prepared_only_on_cuda_with_the_entry_point_and_the_switch(
    monkeypatch,
):
    data = object()
    monkeypatch.delenv("ALGAN_PRECOMPILE_PTX", raising=False)

    prog = _Program()
    _on_arch(monkeypatch, ti.cuda)
    assert kp.prepare_for_launch(prog, data) >= 0.0
    assert prog.prepared == [data]

    # The CPU arch keeps no machine code across processes: nothing to prepare.
    prog = _Program()
    _on_arch(monkeypatch, ti.x64)
    assert kp.prepare_for_launch(prog, data) is None
    assert prog.prepared == []

    # A wheel without 0011.
    _on_arch(monkeypatch, ti.cuda)
    assert kp.prepare_for_launch(SimpleNamespace(), data) is None

    prog = _Program()
    monkeypatch.setenv("ALGAN_PRECOMPILE_PTX", "0")
    assert kp.prepare_for_launch(prog, data) is None
    assert prog.prepared == []


def test_a_failed_prepare_is_reported_not_raised(monkeypatch):
    """The worker keeps the ``.qdc`` it built; the render prepares again and reports."""
    _on_arch(monkeypatch, ti.cuda)
    monkeypatch.delenv("ALGAN_PRECOMPILE_PTX", raising=False)
    outcome = {}
    assert kp._prepare(_Program(fail=True), object(), outcome) is False
    assert outcome == {"ptx_error": "RuntimeError: ptxas said no"}
    outcome = {}
    assert kp._prepare(_Program(), object(), outcome) is True
    assert set(outcome) == {"ptx_seconds"}


class _Kernel:
    """Enough of a Quadrants ``Kernel`` for :func:`kp.compile_spec`."""

    def __init__(self, cached=None):
        self.compiled_kernel_data_by_key = {"key": cached} if cached else {}
        self.materialized_kernels = {"key": object()}
        self.fast_checksum = None

    def ensure_compiled(self, *args):
        return "key"


@pytest.fixture
def worker(monkeypatch):
    """``compile_spec`` against a stub kernel on the live program, recording its writes.

    Returns ``(kernel_for, calls, prepared)``: ``kernel_for(cached)`` installs a
    stub kernel, ``calls`` is the order of the writes and the preparation.
    """
    taichi_runtime.ensure_taichi_for_render()
    calls = []
    prepared = []

    def prepare(prog, data):
        calls.append("prepare")
        prepared.append(data)
        return 0.25

    monkeypatch.setattr(kp, "prepare_for_launch", prepare)
    monkeypatch.setattr(kp, "placeholder_args", lambda spec, device: ())
    monkeypatch.setattr(
        type(program()),
        "dump_cache_data_to_disk",
        lambda self: calls.append("dump"),
    )

    def kernel_for(cached):
        kernel = _Kernel(cached)
        monkeypatch.setattr(kp, "resolve_kernel", lambda spec: kernel)
        return kernel

    return kernel_for, calls, prepared


def test_a_cached_kernel_is_prepared_and_then_dumped(worker):
    """Its ``.qdc`` may predate the PTX tier, so the index hit is no reason to skip."""
    kernel_for, calls, prepared = worker
    data = object()
    kernel_for(cached=data)
    outcome = kp.compile_spec({}, before_writing=lambda: calls.append("enter"))
    assert outcome["status"] == "cached"
    assert outcome["ptx_seconds"] == 0.25
    assert prepared == [data]
    # The dump takes the cache's metadata lock: only inside a granted section.
    assert calls == ["prepare", "enter", "dump"]


def test_a_cached_kernel_that_was_not_prepared_writes_nothing(worker, monkeypatch):
    kernel_for, calls, _ = worker
    monkeypatch.setattr(kp, "prepare_for_launch", lambda prog, data: None)
    kernel_for(cached=object())
    outcome = kp.compile_spec({}, before_writing=lambda: calls.append("enter"))
    assert outcome["status"] == "cached"
    assert "ptx_seconds" not in outcome
    assert calls == []


def test_a_compiled_kernel_is_prepared_before_the_one_dump(worker, monkeypatch):
    """The PTX goes to disk in the same dump as the ``.qdc``, outside any compile."""
    kernel_for, calls, prepared = worker
    kernel = kernel_for(cached=None)
    data = object()
    monkeypatch.setattr(
        type(program()),
        "compile_kernel",
        lambda self, *args: SimpleNamespace(
            compiled_kernel_data=data, cache_hit=False, cache_key="T0"
        ),
    )
    outcome = kp.compile_spec({}, before_writing=lambda: calls.append("enter"))
    assert outcome["status"] == "compiled"
    assert outcome["ptx_seconds"] == 0.25
    assert prepared == [data]
    assert kernel.compiled_kernel_data_by_key["key"] is data
    assert calls == ["prepare", "enter", "dump"]
