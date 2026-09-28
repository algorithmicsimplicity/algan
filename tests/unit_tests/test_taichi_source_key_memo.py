"""The source memo must never change a key, only how fast it is built.

``taichi_source_key._SourceMemo`` persists what source retrieval and the
bytecode walk produce, keyed by file *content*, so a fresh process skips the
tokenizing and disassembly a key otherwise repeats. Its whole claim is that a
key built through it is byte-identical to one built without it, and that it
never persists anything that is not a function of the bytes it is keyed by --
a file edited under a running process must not leave an entry behind that a
later, healthy process would trust. These tests hold both halves: identity
across three processes (memo off, cold, warm) over every Algan kernel body,
and the rejection of an entry the live code no longer matches.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import types

import pytest

from algan.taichi_compat import BACKEND
from algan.utils import taichi_source_key as sk

quadrants_only = pytest.mark.skipif(
    BACKEND != "quadrants", reason="the source-keyed index is Quadrants-only"
)


@pytest.fixture
def memo(tmp_path, monkeypatch):
    """A fresh memo writing to ``tmp_path``, active for the test."""
    fresh = sk._SourceMemo()
    monkeypatch.setattr(sk, "_MEMO", fresh)
    monkeypatch.setattr(sk, "_MEMO_ACTIVE", True)
    monkeypatch.setattr(
        sk._SourceMemo, "path", classmethod(lambda cls: str(tmp_path / "memo.json"))
    )
    fresh.begin_key()
    return fresh


def _load_module(path, name):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Each test formats its own constant into the body. Code objects compare by
# bytecode, not by file, so two tests loading identical text would share the
# in-process chain memo and never reach the persistent one.
_MODULE_V1 = """
LIMIT = {tag}
OTHER = 2


def reader(x):
    return x + LIMIT + {tag}
"""

# Same line count and the same position for `reader`; a different body and a
# different global read, so both the source hash and the chains move.
_MODULE_V2 = """
LIMIT = {tag}
OTHER = 2


def reader(x):
    return x * OTHER * {tag}
"""


def test_an_entry_is_stored_and_served_for_a_file_that_matches(tmp_path, memo):
    path = tmp_path / "memo_probe_match.py"
    path.write_text(_MODULE_V1.format(tag=101), encoding="utf-8")
    module = _load_module(path, "memo_probe_match")

    computed = sk._function_source_hash(module.reader)
    chains = sk._function_chains(module.reader)
    assert memo.stats == {"hits": 0, "stored": 2, "rejected": 0}
    assert memo.flush()

    served = sk._SourceMemo()
    served.begin_key()
    code = module.reader.__code__
    _, value = served.lookup("src", str(path), code.co_firstlineno, sk._code_name(code))
    assert tuple(value) == computed
    _, value = served.lookup(
        "chains", str(path), code.co_firstlineno, sk._code_name(code)
    )
    assert tuple((k, r, tuple(a)) for k, r, a in value) == chains


def test_a_file_edited_after_import_is_never_persisted(tmp_path, memo):
    """The live function came from v1; the file on disk is now v2.

    Whatever this process computes for ``reader`` is what the non-memo path
    would compute, and it is used -- but it is not an answer about v2's bytes,
    so nothing may be stored under v2's hash for a later process to find.
    """
    path = tmp_path / "memo_probe_edit.py"
    path.write_text(_MODULE_V1.format(tag=102), encoding="utf-8")
    module = _load_module(path, "memo_probe_edit")
    # Warm-start style: the process captured v1's source before the edit.
    sk._source_info_and_src()(module.reader)

    path.write_text(_MODULE_V2.format(tag=102), encoding="utf-8")
    os.utime(path, ns=(os.stat(path).st_mtime_ns + 10**9,) * 2)
    memo.begin_key()

    sk._function_source_hash(module.reader)
    sk._function_chains(module.reader)
    assert memo.stats["stored"] == 0
    assert memo.stats["rejected"] == 2
    assert not memo.flush()
    assert not (tmp_path / "memo.json").exists()


def test_code_without_a_file_is_never_memoized(memo):
    namespace = {}
    exec(compile("def made():\n    return 1\n", "<string>", "exec"), namespace)
    code = namespace["made"].__code__
    key, value = memo.lookup("src", sk._code_path(code), code.co_firstlineno, "made")
    assert (key, value) == (None, None)


def test_entries_from_another_setup_are_ignored(tmp_path, memo, monkeypatch):
    path = tmp_path / "memo_probe_header.py"
    path.write_text(_MODULE_V1.format(tag=103), encoding="utf-8")
    module = _load_module(path, "memo_probe_header")
    sk._function_source_hash(module.reader)
    assert memo.flush()

    monkeypatch.setattr(
        sk._SourceMemo,
        "_header",
        classmethod(lambda cls: {"schema": cls._SCHEMA, "compiler": ["other", [9]]}),
    )
    other = sk._SourceMemo()
    other.begin_key()
    code = module.reader.__code__
    _, value = other.lookup("src", str(path), code.co_firstlineno, sk._code_name(code))
    assert value is None


def test_a_write_prunes_entries_for_superseded_versions_of_a_file(tmp_path, memo):
    path = tmp_path / "memo_probe_prune.py"
    path.write_text(_MODULE_V1.format(tag=104), encoding="utf-8")
    first = _load_module(path, "memo_probe_prune_v1")
    sk._function_source_hash(first.reader)
    assert memo.flush()
    old_hash = memo._file(str(path))[0]

    path.write_text(_MODULE_V2.format(tag=104), encoding="utf-8")
    os.utime(path, ns=(os.stat(path).st_mtime_ns + 10**9,) * 2)
    memo.begin_key()
    second = _load_module(path, "memo_probe_prune_v2")
    sk._function_source_hash(second.reader)
    assert memo.flush()

    entries = json.loads((tmp_path / "memo.json").read_text())["entries"]
    hashes = {key.split("\x1f")[2] for key in entries if str(path) in key}
    assert old_hash not in hashes
    assert hashes == {memo._file(str(path))[0]}


_WALK_CHILD = textwrap.dedent(
    """
    import hashlib, importlib, json, pkgutil
    import algan  # noqa: F401
    import algan.rendering.raytracing as package
    from algan.taichi_compat import BACKEND, submodule
    from algan.utils import taichi_source_key as sk

    for info in pkgutil.iter_modules(package.__path__):
        if info.name.endswith("_taichi"):
            importlib.import_module(f"{package.__name__}.{info.name}")
    function_hasher = submodule("lang._fast_caching.function_hasher")
    hash_strings = submodule("lang._fast_caching.hash_utils").hash_iterable_strings
    sk._MEMO_ACTIVE = sk.env_flag("ALGAN_TAICHI_SOURCE_KEY_MEMO", True)
    sk._MEMO.begin_key()
    digest = hashlib.sha256()
    seen = set()
    for kernel in submodule("lang.impl").get_runtime().kernels:
        function = kernel.func
        if function in seen or not function.__module__.startswith("algan."):
            continue
        seen.add(function)
        ctx = sk._KeyContext(BACKEND)
        ctx.visited.add(id(function))
        sk._hash_references(function, ctx, 0)
        ctx.out.append(sk._kernel_fragment(function, function_hasher, hash_strings))
        digest.update("\\x1e".join(ctx.out).encode())
    sk._MEMO.flush()
    print("REPORT " + json.dumps({"digest": digest.hexdigest(), "kernels": len(seen),
                                  **sk._MEMO.stats}))
    """
)


@quadrants_only
def test_every_kernel_walk_is_identical_with_the_memo_off_cold_and_warm(tmp_path):
    """Three processes over one cache directory: off, cold (stores), warm (serves).

    The digest covers every chain rendered and every source hash and kernel
    fragment emitted for every Algan kernel body, so any divergence the memo
    introduced anywhere in the walk shows up as a different digest.
    """
    env = {
        **os.environ,
        "ALGAN_USE_DAEMON": "0",
        "ALGAN_AUTO_DAEMON": "0",
        "ALGAN_CACHE_DIR": str(tmp_path / "cache"),
        "ALGAN_HOME": str(tmp_path / "home"),
    }

    def run(memo_on):
        result = subprocess.run(
            [sys.executable, "-c", _WALK_CHILD],
            env={**env, "ALGAN_TAICHI_SOURCE_KEY_MEMO": "1" if memo_on else "0"},
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        line = next(
            line for line in result.stdout.splitlines() if line.startswith("REPORT ")
        )
        return json.loads(line[len("REPORT ") :])

    off = run(False)
    cold = run(True)
    warm = run(True)
    assert off["kernels"] > 40
    assert cold["digest"] == off["digest"]
    assert warm["digest"] == off["digest"]
    assert cold["stored"] > 100
    assert cold["rejected"] == 0
    assert warm["stored"] == 0
    assert warm["hits"] >= cold["stored"]


def test_memo_state_is_a_module_global_the_render_flushes():
    """``flush_source_memo`` is the name ``taichi_runtime`` calls at job end."""
    assert isinstance(sk._MEMO, sk._SourceMemo)
    assert isinstance(sk.flush_source_memo, types.FunctionType)
