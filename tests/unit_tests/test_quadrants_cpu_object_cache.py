"""Algan's half of Quadrants patch 0012 (``quadrants_patches/README.md``).

0012 adds ``offline_cache_cpu_object_by_content_hash``: on the CPU arch the
machine code the JIT generates for a kernel is kept in an object file named by
the content hash its ``.qdc`` stores plus the host CPU, and a later process
links that file instead of parsing the IR and running LLVM's codegen at the
kernel's first launch. ``taichi_init_kwargs`` asks for it on the CPU arch when
the installed wheel has the field, unless ``ALGAN_CPU_OBJECT_CACHE=0``; a CPU
precompile worker then calls ``prepare_kernel`` on each spec, which writes the
object for the render.

The switch must not move a key (both A/B arms share every artifact), and must
not break a wheel without the patch. Where the installed wheel has it, the
cross-process behaviour itself is checked: a second process links the first
one's objects without rewriting them and computes the same answer, and a
damaged object is recompiled rather than linked.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import algan.taichi_compat
from algan.rendering import kernel_precompile as kp
from algan.rendering import taichi_runtime
from algan.taichi_compat import BACKEND

_FIELD = "offline_cache_cpu_object_by_content_hash"


@pytest.mark.parametrize(
    ("device", "flag", "supported", "passed"),
    [
        ("cpu", None, True, True),
        ("cpu", "1", True, True),
        ("cpu", "0", True, False),
        ("cpu", None, False, False),
        ("cuda", None, True, False),
        ("mps", "1", True, False),
    ],
)
def test_the_object_cache_keyword_is_passed_on_cpu_where_the_wheel_has_it(
    monkeypatch, device, flag, supported, passed
):
    """On by default; never passed to a wheel that would reject it, nor off CPU.

    Left out rather than passed as ``False`` in the off arm, so that arm's
    ``init`` is exactly what it was before the switch existed.
    """
    monkeypatch.setattr(taichi_runtime, "render_device", lambda: torch.device(device))
    monkeypatch.setattr(taichi_runtime, "cpu_object_cache_supported", lambda: supported)
    monkeypatch.setattr(taichi_runtime, "_torch_mps_command_queue", lambda: 0)
    if flag is None:
        monkeypatch.delenv("ALGAN_CPU_OBJECT_CACHE", raising=False)
    else:
        monkeypatch.setenv("ALGAN_CPU_OBJECT_CACHE", flag)
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
    assert taichi_runtime.cpu_object_cache_supported() is True
    with_config(type("Published", (), {"offline_cache": None}))
    assert taichi_runtime.cpu_object_cache_supported() is False
    monkeypatch.setattr(taichi_runtime, "BACKEND", "taichi")
    with_config(type("Patched", (), {_FIELD: property(lambda self: False)}))
    assert taichi_runtime.cpu_object_cache_supported() is False


def test_the_switch_reaches_neither_the_source_key_nor_a_precompile_context(
    monkeypatch,
):
    from algan.utils import taichi_source_key as sk

    before = sk._environment_fingerprint()
    context_before = kp.context_id(kp.current_context())
    monkeypatch.setenv("ALGAN_CPU_OBJECT_CACHE", "0")
    assert sk._environment_fingerprint() == before
    assert kp.context_id(kp.current_context()) == context_before


def test_the_config_field_is_out_of_the_source_key():
    """By its ``offline_cache`` prefix, as 0010's field is."""
    from algan.utils import taichi_source_key as sk

    assert _FIELD.startswith(sk._CONFIG_EXCLUDE_PREFIXES)


# --- the cache itself, on a wheel that has it -------------------------------------

_PROBE = textwrap.dedent(
    """
    import os
    import sys

    import algan  # noqa: F401
    import torch
    from algan.rendering import taichi_runtime
    from algan.taichi_compat import program, ti

    @ti.kernel
    def probe(src: ti.types.ndarray(), dst: ti.types.ndarray(), scale: ti.f32):
        for i in range(src.shape[0]):
            dst[i] = ti.sin(src[i]) * scale + ti.sqrt(ti.abs(src[i]))

    taichi_runtime.ensure_taichi_for_render()
    assert program().config().{field} is True, "the switch did not reach init"
    src = torch.linspace(-3.0, 3.0, 257)
    dst = torch.zeros(257)
    probe(src, dst, 2.5)
    expected = torch.sin(src) * 2.5 + torch.sqrt(torch.abs(src))
    assert torch.allclose(dst, expected, atol=1e-5), (dst - expected).abs().max()
    ti.sync()
    print("PROBE-OK", flush=True)
    os._exit(0)
    """
).format(field=_FIELD)


def _run_probe(tmp_path):
    script = tmp_path / "probe.py"
    script.write_text(_PROBE, encoding="utf-8")
    env = dict(
        os.environ,
        ALGAN_CACHE_DIR=str(tmp_path / "cache"),
        ALGAN_RENDER_DEVICE="cpu",
        ALGAN_USE_DAEMON="0",
        ALGAN_AUTO_DAEMON="0",
        ALGAN_PRECOMPILE_JOBS="0",
        ALGAN_FIRST_RUN_POOL="0",
        ALGAN_STARTUP_WARMUP="0",
    )
    env.pop("ALGAN_CPU_OBJECT_CACHE", None)
    result = subprocess.run(
        [sys.executable, str(script)],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert "PROBE-OK" in result.stdout, result.stdout + result.stderr


def _objects(tmp_path):
    return {
        path: (path.stat().st_ino, path.stat().st_mtime_ns)
        for path in Path(tmp_path / "cache").rglob("cpu_object_cache/*.o")
    }


@pytest.mark.skipif(
    BACKEND != "quadrants" or not taichi_runtime.cpu_object_cache_supported(),
    reason="needs a Quadrants wheel with patch 0012",
)
def test_a_second_process_links_the_first_ones_machine_code(tmp_path):
    _run_probe(tmp_path)
    written = _objects(tmp_path)
    assert written, "the first process stored no object"
    _run_probe(tmp_path)
    assert _objects(tmp_path) == written, "the second process compiled again"

    # A damaged object is a miss: recompiled, rewritten, and the answer right.
    damaged = next(iter(written))
    damaged.write_bytes(damaged.read_bytes()[:64])
    _run_probe(tmp_path)
    assert damaged.stat().st_size > 64
