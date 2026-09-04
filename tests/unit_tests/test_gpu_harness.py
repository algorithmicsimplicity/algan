"""The GPU measurement harnesses' offline halves.

Neither harness can be exercised end to end from here -- one needs an Apple
GPU, the other a Kaggle session -- so what is testable is everything that runs
*before* the expensive part: the request resolver that decides a run's matrix,
and the notebook generator whose output is transmitted inline and then executed
on a box that costs a quota slot to reach.

Both failure modes these guard against are silent and late. A resolver that
emits an empty matrix produces a green run that measured nothing; a notebook
body with a formatting error fails after apt, clone and install have already
been paid for.

See `agent_guidance/gpu_harnesses.md`.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(path: Path, name: str):
    """Import a module that is not on the import path (a workflow helper)."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def resolver():
    return _load(
        REPO_ROOT / ".github" / "workflows" / "scripts" / "resolve_gpu_request.py",
        "resolve_gpu_request",
    )


@pytest.fixture(scope="module")
def make_notebook():
    return _load(REPO_ROOT / "scripts" / "kaggle" / "make_notebook.py", "make_notebook")


@pytest.fixture(scope="module")
def runner():
    return _load(REPO_ROOT / "scripts" / "kaggle" / "runner.py", "kaggle_runner")


class TestResolveGpuRequest:
    def test_dispatch_inputs_win_over_the_request_file(self, resolver):
        out = resolver.resolve(
            {"IN_COMMAND": "echo dispatched", "IN_ARMS": "mac-cpu"},
            {"command": "echo from-file", "arms": ["mac-mps"]},
        )
        assert out["command"] == "echo dispatched"
        assert json.loads(out["matrix"]) == [
            {"os": "macos-latest", "device": "cpu", "label": "mac-cpu"}
        ]

    def test_the_request_file_is_used_when_no_input_was_given(self, resolver):
        out = resolver.resolve(
            {},
            {
                "command": "uv run python benchmarks/_foo.py",
                "arms": ["mac-mps", "linux-cpu"],
                "env": {"ALGAN_VIDEO_ENCODER": "software"},
                "latex": True,
                "timeout_minutes": 90,
            },
        )
        assert out["command"] == "uv run python benchmarks/_foo.py"
        assert [a["label"] for a in json.loads(out["matrix"])] == [
            "mac-mps",
            "linux-cpu",
        ]
        assert out["env"] == "ALGAN_VIDEO_ENCODER=software"
        assert out["latex"] == "true"
        assert out["timeout"] == "90"

    def test_a_run_with_no_command_is_refused(self, resolver):
        with pytest.raises(SystemExit):
            resolver.resolve({}, None)

    def test_an_unknown_arm_is_refused_rather_than_dropped(self, resolver):
        # Dropping it would leave a smaller matrix that still runs and reports
        # green, which is the expensive way to learn about a typo.
        with pytest.raises(SystemExit):
            resolver.resolve({"IN_COMMAND": "echo hi", "IN_ARMS": "mac-metal"}, None)

    def test_an_unticked_latex_checkbox_does_not_override_the_file(self, resolver):
        # A checkbox arrives as the string "false", not as an empty value, so a
        # naive "input wins if non-empty" would make the file's `latex: true`
        # unreachable from the push entry point.
        out = resolver.resolve({"IN_LATEX": "false"}, {"command": "x", "latex": True})
        assert out["latex"] == "false"
        out = resolver.resolve({}, {"command": "x", "latex": True})
        assert out["latex"] == "true"

    def test_a_non_numeric_timeout_is_refused(self, resolver):
        # `timeout-minutes: ${{ fromJSON(...) }}` would fail the whole workflow
        # with a parse error naming neither the field nor the value.
        with pytest.raises(SystemExit):
            resolver.resolve({"IN_COMMAND": "x", "IN_TIMEOUT": "an hour"}, None)

    def test_the_wheel_defaults_but_can_be_opted_out_of(self, resolver):
        assert resolver.resolve({"IN_COMMAND": "x"}, None)["wheel"].isdigit()
        assert (
            resolver.resolve({"IN_COMMAND": "x", "IN_WHEEL": "none"}, None)["wheel"]
            == "none"
        )

    def test_the_quadrants_wheel_is_empty_unless_asked_for(self, resolver):
        # No default, unlike the Taichi wheel: an empty value means the arm
        # runs whatever Quadrants `uv sync` installed, and a run id or a
        # release-asset URL passes through from either entry point untouched.
        assert resolver.resolve({"IN_COMMAND": "x"}, None)["quadrants_wheel"] == ""
        out = resolver.resolve({"IN_COMMAND": "x", "IN_QUADRANTS_WHEEL": "123"}, None)
        assert out["quadrants_wheel"] == "123"
        url = "https://github.com/o/r/releases/download/t/quadrants-1.3.0.post1.whl"
        out = resolver.resolve({}, {"command": "x", "quadrants_wheel": url})
        assert out["quadrants_wheel"] == url

    def test_multiline_values_use_the_heredoc_form(self, resolver):
        text = resolver.format_outputs({"env": "A=1\nB=2", "timeout": "60"})
        assert "env<<ghadelim_" in text
        assert "timeout=60" in text
        # A bare `env=A=1\nB=2` would silently truncate to `A=1`.
        assert "env=A=1" not in text


class TestKaggleNotebook:
    def _body(self, make_notebook, tmp_path, extra=()):
        out = tmp_path / "body.py"
        make_notebook.main(
            [
                "--tag",
                "smoke",
                "--branch",
                "some/branch",
                "--step",
                "one:python -c 'print(1)'",
                "--out",
                str(out),
                *extra,
            ]
        )
        return out.read_text(encoding="utf-8")

    def test_the_generated_body_is_valid_python(self, make_notebook, tmp_path):
        # It is executed on a box that costs a GPU quota slot to reach, after
        # ~75 s of apt and pip; a SyntaxError there is the dearest possible
        # place to find one.
        ast.parse(self._body(make_notebook, tmp_path))

    def test_the_body_carries_the_spec_and_the_branch(self, make_notebook, tmp_path):
        body = self._body(make_notebook, tmp_path)
        assert "some/branch" in body
        tree = ast.parse(body)
        spec = next(
            ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and getattr(node.targets[0], "id", None) == "SPEC"
        )
        assert spec["tag"] == "smoke"
        assert spec["steps"] == [{"name": "one", "command": "python -c 'print(1)'"}]

    def test_the_body_hands_over_to_the_in_repo_runner(self, make_notebook, tmp_path):
        # The split is the design: if the body ever grows the run logic back,
        # every launch pays for it inline and nobody reviews it.
        body = self._body(make_notebook, tmp_path)
        assert "scripts" in body
        assert "runner.py" in body
        assert len(body) < 4000, (
            "the bootstrap body has grown; keep the logic in runner.py"
        )

    def test_a_step_needs_a_name_and_a_command(self, make_notebook):
        import argparse

        assert make_notebook.parse_step("uhd:python x.py") == {
            "name": "uhd",
            "command": "python x.py",
        }
        # The name becomes a log filename.
        for bad in ("no-colon", ":command", "name:", "a/b:cmd"):
            with pytest.raises(argparse.ArgumentTypeError):
                make_notebook.parse_step(bad)

    def test_a_command_containing_a_colon_keeps_it(self, make_notebook):
        step = make_notebook.parse_step("t:python -c 'a:b'")
        assert step["command"] == "python -c 'a:b'"


class TestKaggleRunner:
    def test_the_results_prefix_is_a_single_greppable_token(self, runner):
        assert runner.RESULTS_PREFIX == "RESULTS "

    def test_a_non_cuda_device_aborts_unless_it_was_asked_for(
        self, runner, monkeypatch
    ):
        # The trap this exists for: a wrong `machineShape` is dropped silently
        # and Kaggle hands back a P100, on which torch.cuda.is_available() is
        # True but Algan renders on the CPU. Every number in such a run is a CPU
        # number on a notebook titled "t4"; two rounds were collected that way.
        class Completed:
            returncode = 0
            stdout = "ALGAN_DEVICE cpu\nTORCH 2.7.1 cuda=True\n"
            stderr = ""

        monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: Completed())
        with pytest.raises(SystemExit) as excinfo:
            runner.check_render_device(Path("."), allow_cpu=False)
        assert "NvidiaTeslaT4" in str(excinfo.value)
        assert runner.check_render_device(Path("."), allow_cpu=True) == "cpu"

    def test_a_cuda_device_is_accepted(self, runner, monkeypatch):
        class Completed:
            returncode = 0
            stdout = "ALGAN_DEVICE cuda\nGPU Tesla T4\n"
            stderr = ""

        monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: Completed())
        assert runner.check_render_device(Path("."), allow_cpu=False) == "cuda"


# ---------------------------------------------------------------------------
# runner.run_step's step timeout
# ---------------------------------------------------------------------------
def test_step_timeout_kills_a_hung_step(runner, tmp_path):
    """A step that stops producing output must still hit its deadline.

    The timeout used to be applied to ``process.wait()`` *after* draining the
    child's stdout to EOF -- which is to say after the child had already
    exited -- so it could never fire on a hung step. One Taichi compile that
    wedged then ran until the Kaggle session itself timed out, taking every
    later step of the sweep with it and costing the whole session.
    """
    started = time.monotonic()
    result = runner.run_step(
        name="hang",
        command=f"{sys.executable} -c 'import time; time.sleep(120)'",
        repo=REPO_ROOT,
        out_dir=tmp_path,
        env={},
        timeout=2,
    )
    elapsed = time.monotonic() - started

    assert result["status"] == "failed"
    assert result["returncode"] == -9
    # Generous, but far below the 120 s the child asked for: the point is that
    # the deadline fired at all, not how precisely.
    assert elapsed < 60, f"step ran {elapsed:.1f}s despite a 2s timeout"


def test_step_timeout_kills_the_whole_process_tree(runner, tmp_path):
    """Killing the shell is not enough -- the grandchild holds the GPU.

    ``shell=True`` means the immediate child is a shell and the render is its
    grandchild, so a kill aimed at ``process.pid`` alone leaves the render
    running: it keeps its VRAM, and the next step of the sweep then measures a
    card that is still busy. The kill goes to the process group instead.
    """
    marker = tmp_path / "grandchild.pid"
    script = tmp_path / "grandchild.py"
    script.write_text(
        "import os, sys, time\n"
        "open(sys.argv[1], 'w').write(str(os.getpid()))\n"
        "time.sleep(120)\n"
    )
    result = runner.run_step(
        name="tree",
        command=f"sh -c '{sys.executable} {script} {marker}'",
        repo=REPO_ROOT,
        out_dir=tmp_path,
        env={},
        timeout=5,
    )
    assert result["status"] == "failed"
    assert result["returncode"] == -9
    assert marker.exists(), "the grandchild never started; the test proves nothing"

    pid = int(marker.read_text())
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except OSError:
            break
        time.sleep(0.1)
    else:
        pytest.fail(f"grandchild {pid} survived the step timeout")


def test_a_quick_step_is_unaffected_by_the_timeout(runner, tmp_path):
    """The ordinary path still captures output and reports success."""
    result = runner.run_step(
        name="quick",
        command=f"{sys.executable} -c 'print(\"hello from the step\")'",
        repo=REPO_ROOT,
        out_dir=tmp_path,
        env={},
        timeout=60,
    )
    assert result["status"] == "ok"
    assert result["returncode"] == 0
    assert "hello from the step" in (tmp_path / "quick.log").read_text()
