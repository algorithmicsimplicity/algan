"""Inductor's CPU instruction-set probe, carried between processes.

``algan/utils/torch_compile.py`` (``_remember_vec_isa_probe``) stores the list
of SIMD sets Inductor's probe verified, so a later process skips the four
torch-importing subprocesses the probe starts (7.2 s measured on a 4-core x86
box). Held here against stand-ins for torch's modules, so nothing is built:
a full pass is stored and handed back in the same order without probing, a
partial pass is never stored, a changed compiler or CPU probes again, a
damaged file costs only the saving, and an explicit ``vec_isa_ok`` is left
alone. The last test holds that torch still has the internals this relies
on, so an upgrade that moves them fails here instead of quietly costing the
seven seconds again.
"""

from __future__ import annotations

import functools
import types

import pytest

from algan.utils import torch_compile as tc


class _ISA:
    def __init__(self, name, passes=True):
        self.name = name
        self.passes = passes

    def __str__(self):
        return self.name


def _modules(*, flags=("avx2", "avx512"), compiler="gcc 13", failing=(), ok=None):
    """Fresh stand-ins for ``cpu_vec_isa`` and Inductor's ``config``, as a new process sees them."""
    supported = [
        _ISA("avx512 amx_tile"),
        _ISA("avx512", passes="avx512" not in failing),
        _ISA("avx2", passes="avx2" not in failing),
        _ISA("asimd"),
    ]
    calls = []

    @functools.lru_cache(None)
    def valid_vec_isa_list():
        calls.append(1)
        return [
            isa
            for isa in supported
            if all(flag in flags for flag in str(isa).split()) and isa.passes
        ]

    cpu_vec_isa = types.SimpleNamespace(
        valid_vec_isa_list=valid_vec_isa_list,
        supported_vec_isa_list=supported,
        x86_isa_checker=lambda: list(flags),
        _get_isa_dry_compile_fingerprint=lambda isa_flags: f"{compiler}={isa_flags}",
    )
    config = types.SimpleNamespace(
        cpp=types.SimpleNamespace(vec_isa_ok=ok), is_fbcode=lambda: False
    )
    return cpu_vec_isa, config, calls


def _process(path, **kwargs):
    """Seed, then ask for the list the way Inductor does; return names and probe count."""
    cpu_vec_isa, config, calls = _modules(**kwargs)
    tc._seed_vec_isa_probe(cpu_vec_isa, config, path)
    return [str(isa) for isa in cpu_vec_isa.valid_vec_isa_list()], len(calls)


def test_a_full_pass_is_handed_to_the_next_process_without_probing(tmp_path):
    path = tmp_path / "vec_isa.json"
    assert _process(path) == (["avx512", "avx2"], 1)
    assert path.exists()
    assert _process(path) == (["avx512", "avx2"], 0)


def test_a_partial_pass_is_never_stored(tmp_path):
    path = tmp_path / "vec_isa.json"
    assert _process(path, failing=("avx512",)) == (["avx2"], 1)
    assert not path.exists()
    # ... and the next process, whose probe passes everything, probes again.
    assert _process(path) == (["avx512", "avx2"], 1)


def test_a_different_compiler_or_cpu_probes_again(tmp_path):
    path = tmp_path / "vec_isa.json"
    _process(path)
    assert _process(path, compiler="gcc 14") == (["avx512", "avx2"], 1)
    assert _process(path, flags=("avx2",)) == (["avx2"], 1)
    # Each environment keeps its own entry.
    assert _process(path) == (["avx512", "avx2"], 0)
    assert _process(path, compiler="gcc 14")[1] == 0


def test_a_damaged_file_costs_only_the_saving(tmp_path):
    path = tmp_path / "vec_isa.json"
    path.write_text("{not json", encoding="utf-8")
    assert _process(path) == (["avx512", "avx2"], 1)
    assert _process(path) == (["avx512", "avx2"], 0)


def test_an_explicit_vec_isa_ok_is_left_alone(tmp_path):
    path = tmp_path / "vec_isa.json"
    cpu_vec_isa, config, _ = _modules(ok=True)
    original = cpu_vec_isa.valid_vec_isa_list
    tc._seed_vec_isa_probe(cpu_vec_isa, config, path)
    assert cpu_vec_isa.valid_vec_isa_list is original
    assert not path.exists()


def test_torch_still_has_the_internals_the_memo_relies_on():
    cpu_vec_isa = pytest.importorskip("torch._inductor.cpu_vec_isa")
    from torch._inductor import config

    assert callable(getattr(cpu_vec_isa.valid_vec_isa_list, "cache_info", None))
    assert callable(cpu_vec_isa.x86_isa_checker)
    assert callable(cpu_vec_isa._get_isa_dry_compile_fingerprint)
    assert all(str(isa) for isa in cpu_vec_isa.supported_vec_isa_list)
    assert hasattr(config.cpp, "vec_isa_ok")
    assert callable(config.is_fbcode)
    # Inductor reaches the list through the module attribute at call time,
    # which is what replacing it relies on.
    assert "valid_vec_isa_list" in cpu_vec_isa.pick_vec_isa.__code__.co_names
