"""The kernel-compiler compatibility layer (:mod:`algan.taichi_compat`).

The layer's whole job is that exactly one Taichi-language implementation is live
in a process. The test that earns its keep is
:func:`test_the_package_never_imports_a_backend_by_name` -- the others pin the
small API the engine reaches it through.
"""

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from algan.taichi_compat import (
    BACKEND,
    BACKENDS,
    KERNEL_SPECIALIZATIONS_ATTR,
    backend_version,
    describe_backend,
    kernel_specializations,
    submodule,
    ti,
)

SOURCE_ROOT = Path(__file__).parents[2] / "algan"


def _imported_backend_names(tree):
    """The backend module names ``tree`` names in an import statement."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in BACKENDS:
                    yield alias.name
        elif isinstance(node, ast.ImportFrom):
            # ``from . import x`` has no module; a relative import cannot name a
            # backend anyway.
            root = (node.module or "").split(".")[0]
            if node.level == 0 and root in BACKENDS:
                yield node.module


def test_the_package_never_imports_a_backend_by_name():
    """No module under ``algan/`` may name taichi or quadrants in an import.

    Naming one defeats the layer: the module would bind that implementation
    whatever ``ALGAN_TAICHI_BACKEND`` selected, and a process that also renders
    through the other ends up with two runtimes, two CUDA contexts and two
    kernel caches -- which presents as kernels compiled by one backend being
    launched against the other's runtime, not as an import error.

    Use ``from algan.taichi_compat import ti``, and ``submodule("lang.impl")``
    where a submodule is needed.
    """
    violations = []
    for source_path in SOURCE_ROOT.rglob("*.py"):
        if "external_libraries" in source_path.parts:
            continue
        if source_path.name == "taichi_compat.py":
            continue  # the one module allowed to bind a backend
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for name in _imported_backend_names(tree):
            violations.append(f"{source_path.relative_to(SOURCE_ROOT.parent)}: {name}")

    assert not violations, (
        "import the kernel compiler from algan.taichi_compat instead:\n"
        + "\n".join(sorted(violations))
    )


def test_the_bound_module_is_the_selected_backend():
    assert BACKEND in BACKENDS
    assert ti.__name__ == BACKEND


def test_submodule_resolves_against_the_bound_backend():
    impl = submodule("lang.impl")
    assert impl.__name__ == f"{BACKEND}.lang.impl"
    assert impl is submodule("lang.impl")


def test_kernel_specializations_names_an_attribute_the_backend_has():
    """The dict is set per instance, so the class is checked by its source."""
    kernel_impl = submodule("lang.kernel_impl")
    source = inspect.getsource(kernel_impl.Kernel)
    assert f"self.{KERNEL_SPECIALIZATIONS_ATTR}" in source


def test_kernel_specializations_reads_that_attribute():
    kernel = SimpleNamespace(**{KERNEL_SPECIALIZATIONS_ATTR: {"key": "compiled"}})
    assert kernel_specializations(kernel) == {"key": "compiled"}


def test_describe_backend_names_the_implementation_not_just_a_version():
    """The two report unrelated version numbers, so the name has to be there."""
    described = describe_backend()
    assert described.startswith(BACKEND)
    assert ".".join(str(part) for part in backend_version()) in described


def test_an_unknown_backend_is_refused_rather_than_defaulted(monkeypatch):
    from algan import taichi_compat

    monkeypatch.setenv("ALGAN_TAICHI_BACKEND", "warp")
    with pytest.raises(ValueError, match="warp"):
        taichi_compat._select_backend()


@pytest.mark.parametrize("spelling", ["QUADRANTS", " quadrants "])
def test_the_backend_name_is_case_and_space_insensitive(monkeypatch, spelling):
    from algan import taichi_compat

    monkeypatch.setenv("ALGAN_TAICHI_BACKEND", spelling)
    assert taichi_compat._select_backend() == "quadrants"


def test_each_backend_gets_its_own_kernel_cache_directory(monkeypatch):
    """Sharing one directory would let each backend's LRU prune the other's."""
    from algan.settings import _startup

    monkeypatch.delenv("TI_OFFLINE_CACHE_FILE_PATH", raising=False)
    assert _startup._TAICHI_CACHE_DIRECTORY.name == BACKEND
