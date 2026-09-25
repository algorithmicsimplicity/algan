"""MPS CI partitions preserve coverage and do not turn earlier failures green."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def suite_plugin(pytestconfig):
    """Use the loaded conftest, without importing its side effects twice."""
    target = ROOT / "tests" / "conftest.py"
    return next(
        plugin
        for plugin in pytestconfig.pluginmanager.get_plugins()
        if getattr(plugin, "__file__", None)
        and Path(plugin.__file__).resolve() == target
    )


@pytest.mark.parametrize(
    ("nodeid", "expected"),
    [
        ("tests/unit_tests/test_animation.py::test_a", "early"),
        ("tests/unit_tests/test_mps_zero_copy.py::test_a", "early"),
        ("tests/unit_tests/test_nested_ior.py::test_a", "middle"),
        ("tests/unit_tests/test_render_batch_sizing.py::test_a", "middle"),
        ("tests/unit_tests/test_scene.py::test_a", "late"),
        ("tests/unit_tests/test_ux_regressions.py::test_a[refraction]", "late"),
        ("tests/fast/test_fast_render.py::test_a", "late"),
        ("tests/unit_tests/test_a.py::TestGroup::test_a[test_z.py]", "early"),
        (r"tests\unit_tests\test_nested_ior.py::test_a", "middle"),
        ("tests/new_suite/test_a.py::test_a[unit_tests/test_m.py]", "late"),
    ],
)
def test_batch_uses_the_module_not_the_test_name(suite_plugin, nodeid, expected):
    assert suite_plugin._ci_batch(nodeid) == expected


def test_every_portable_module_belongs_to_exactly_one_batch(suite_plugin):
    modules = sorted((ROOT / "tests" / "unit_tests").glob("test_*.py"))
    modules += sorted((ROOT / "tests" / "fast").glob("test_*.py"))
    assert modules
    for module in modules:
        nodeid = module.relative_to(ROOT).as_posix()
        assert (
            sum(suite_plugin._ci_batch(nodeid) == b for b in suite_plugin.CI_BATCHES)
            == 1
        )


@pytest.mark.parametrize("batch", [None, "early", "middle", "late"])
def test_selection_preserves_order_and_marks_only_selected_items(
    suite_plugin, monkeypatch, batch
):
    paths = [
        "tests/unit_tests/test_a.py::test_x[0]",
        "tests/unit_tests/test_a.py::test_x[1]",
        "tests/unit_tests/test_n.py::test_x",
        "tests/unit_tests/test_s.py::test_x",
        "tests/fast/test_fast_render.py::test_x",
    ]
    original = [SimpleNamespace(nodeid=path) for path in paths]
    selected = original.copy()
    removed, marked = [], []
    config = SimpleNamespace(
        getoption={"ci_batch": batch, "fast": False}.__getitem__,
        hook=SimpleNamespace(pytest_deselected=lambda *, items: removed.extend(items)),
    )
    monkeypatch.setattr(suite_plugin, "_mark_known_mps_failures", marked.extend)
    suite_plugin.pytest_collection_modifyitems(config, selected)
    expected = [
        item
        for item in original
        if batch is None or suite_plugin._ci_batch(item.nodeid) == batch
    ]
    assert selected == expected
    assert marked == expected
    assert removed == [item for item in original if item not in expected]
    assert len(selected) + len(removed) == len(original)


def _mps_step():
    workflow = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "test.yaml").read_text()
    )
    steps = workflow["jobs"]["build"]["steps"]
    mps = next(
        step
        for step in steps
        if step.get("name") == "Run MPS tests in isolated batches"
    )
    cpu = next(step for step in steps if step.get("name") == "Run tests")
    assert mps["if"] == "matrix.render_device == 'mps'"
    assert cpu["if"] == "matrix.render_device != 'mps'"
    assert "--ci-batch" not in cpu["run"]
    return mps["run"]


@pytest.mark.skipif(
    os.name == "nt" or not shutil.which("bash"), reason="CI uses a POSIX shell"
)
@pytest.mark.parametrize("failed_batch", ["", "early", "middle", "late"])
def test_ci_runs_every_batch_and_preserves_any_failure(
    tmp_path, suite_plugin, failed_batch
):
    """Execute the real workflow shell with a stand-in interpreter."""
    interpreter = tmp_path / ".venv" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text(
        f"#!{sys.executable} -S\n"
        "import json, os, sys\n"
        "with open(os.environ['CI_BATCH_TRACE'], 'a') as out:\n"
        "    out.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "bad = '--ci-batch=' + os.environ['CI_BATCH_FAILURE']\n"
        "sys.exit(int(bad in sys.argv))\n"
    )
    interpreter.chmod(0o755)
    trace = tmp_path / "calls.jsonl"
    env = dict(os.environ, CI_BATCH_TRACE=str(trace), CI_BATCH_FAILURE=failed_batch)
    result = subprocess.run(
        [shutil.which("bash"), "-c", _mps_step()],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == int(bool(failed_batch)), result.stderr
    calls = [json.loads(line) for line in trace.read_text().splitlines()]
    assert calls[0] == ["-m", "coverage", "erase"]
    assert len(calls) == len(suite_plugin.CI_BATCHES) + 1
    for batch, call in zip(suite_plugin.CI_BATCHES, calls[1:], strict=True):
        assert call[:4] == ["-m", "pytest", "tests/unit_tests", "tests/fast"]
        assert f"--ci-batch={batch}" in call
        assert "--cov-append" in call
        assert "--cov=algan" in call
        assert "--cov-report=xml" in call
        assert "--fast" not in call
        assert "--forked" not in call
