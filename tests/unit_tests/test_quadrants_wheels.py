"""The Quadrants wheel build's offline halves.

The build itself cannot run from here -- it needs three runners, an Apple GPU
box among them, and about twenty minutes per wheel. What *is* testable is
everything that decides what gets built and what comes back: the matrix
resolver the workflow obeys, and the driver script's contract with the
artifacts the workflow uploads.

Both failure modes these guard are silent and late, in the way the GPU
harness's are (`tests/unit_tests/test_gpu_harness.py`). A resolver that drops a
platform produces a green run that built two wheels instead of three and says
so nowhere. An artifact name the driver cannot parse produces a download that
lands the wheels but files them under `unknown`, which then reaches the
manifest as the provenance of a measurement.

See `scripts/build_quadrants_wheels.py`.
"""

from __future__ import annotations

import importlib.util
import json
import re
import urllib.request
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "quadrants_build.yaml"


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
        REPO_ROOT / ".github" / "workflows" / "scripts" / "resolve_wheel_matrix.py",
        "resolve_wheel_matrix",
    )


@pytest.fixture(scope="module")
def driver():
    return _load(
        REPO_ROOT / "scripts" / "build_quadrants_wheels.py", "build_quadrants_wheels"
    )


@pytest.fixture(scope="module")
def workflow():
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


class TestResolveWheelMatrix:
    def test_the_default_is_every_platform(self, resolver):
        out = resolver.resolve({})
        assert out["linux"] == "true"
        assert out["macos"] == "true"
        assert out["windows"] == "true"
        assert json.loads(out["pythons"]) == ["3.11"]

    def test_one_platform_leaves_the_others_off(self, resolver):
        out = resolver.resolve({"IN_PLATFORMS": "macos"})
        assert out["macos"] == "true"
        assert out["linux"] == "false"
        assert out["windows"] == "false"

    def test_all_expands(self, resolver):
        out = resolver.resolve({"IN_PLATFORMS": "all"})
        assert all(out[name] == "true" for name in resolver.PLATFORMS)

    def test_a_full_python_matrix(self, resolver):
        out = resolver.resolve({"IN_PYTHONS": "3.10,3.11,3.12,3.13"})
        assert json.loads(out["pythons"]) == ["3.10", "3.11", "3.12", "3.13"]
        assert "12 wheel(s)" in out["summary"]

    def test_whitespace_and_newlines_are_separators(self, resolver):
        out = resolver.resolve(
            {"IN_PLATFORMS": "linux,\n macos ", "IN_PYTHONS": "3.12"}
        )
        assert out["linux"] == "true"
        assert out["macos"] == "true"
        assert out["windows"] == "false"
        assert json.loads(out["pythons"]) == ["3.12"]

    def test_a_repeat_does_not_make_two_jobs(self, resolver):
        # Two matrix entries for one platform would race to upload the same
        # artifact name, and the second upload is the one that errors.
        out = resolver.resolve(
            {"IN_PLATFORMS": "macos,macos", "IN_PYTHONS": "3.11,3.11"}
        )
        assert json.loads(out["pythons"]) == ["3.11"]
        assert "1 wheel(s)" in out["summary"]

    def test_an_unknown_platform_is_refused(self, resolver):
        with pytest.raises(SystemExit, match="unknown platform"):
            resolver.resolve({"IN_PLATFORMS": "linux,solaris"})

    @pytest.mark.parametrize("version", ["3.9", "3.14", "2.7", "311"])
    def test_an_unsupported_python_is_refused(self, resolver, version):
        with pytest.raises(SystemExit, match="unsupported Python"):
            resolver.resolve({"IN_PYTHONS": version})

    def test_an_empty_list_is_refused_rather_than_defaulted(self, resolver):
        # A blank input means "use the default"; a list of separators means the
        # caller asked for nothing, and building nothing should not look green.
        with pytest.raises(SystemExit, match="no platforms"):
            resolver.resolve({"IN_PLATFORMS": ",,"})
        with pytest.raises(SystemExit, match="no Python"):
            resolver.resolve({"IN_PYTHONS": ", ,"})

    def test_every_platform_names_a_runner(self, resolver):
        out = resolver.resolve({})
        for name, spec in resolver.PLATFORMS.items():
            assert out[f"runner_{name}"] == spec["runner"]
            assert spec["runner"], f"{name} has no runner image"

    def test_no_output_is_multiline(self, resolver):
        # `format_outputs` writes plain `name=value` lines; a value containing a
        # newline would silently truncate in $GITHUB_OUTPUT.
        rendered = resolver.format_outputs(resolver.resolve({"IN_PLATFORMS": "all"}))
        for line in rendered.splitlines():
            assert "=" in line
        assert rendered.count("\n") == len(resolver.resolve({"IN_PLATFORMS": "all"}))


class TestWorkflowMatchesTheResolver:
    """The YAML and the resolver are one contract; these are its two halves."""

    def test_the_plan_job_only_reads_outputs_the_resolver_emits(
        self, resolver, workflow
    ):
        emitted = set(resolver.resolve({})) | {"summary"}
        declared = workflow["jobs"]["plan"]["outputs"]
        referenced = {
            match
            for value in declared.values()
            for match in re.findall(r"steps\.resolve\.outputs\.(\w+)", str(value))
        }
        assert referenced, "the plan job stopped reading the resolver"
        missing = referenced - emitted
        assert not missing, (
            f"the plan job reads outputs the resolver never sets: {missing}"
        )

    def test_every_platform_has_a_job_gated_on_its_output(self, resolver, workflow):
        for name in resolver.PLATFORMS:
            assert name in workflow["jobs"], f"no job builds {name}"
            job = workflow["jobs"][name]
            assert f"needs.plan.outputs.{name}" in job["if"]
            assert f"needs.plan.outputs.runner_{name}" in job["runs-on"]

    def test_the_artifact_names_are_the_ones_the_driver_parses(self, driver, workflow):
        # The workflow writes these names and the driver reads them back into
        # (platform, python); a change to one alone is the bug this catches.
        found = 0
        for name, job in workflow["jobs"].items():
            for step in job.get("steps", []):
                artifact = (step.get("with") or {}).get("name", "")
                if not artifact.startswith("quadrants-wheel-"):
                    continue
                found += 1
                # Substitute what the matrix expands to at run time.
                rendered = artifact.replace("${{ matrix.python-version }}", "3.11")
                match = driver.ARTIFACT_RE.match(rendered)
                assert match, (
                    f"{job} uploads {rendered!r}, which the driver cannot parse"
                )
                assert match.group("platform") == name
                assert match.group("python") == "3.11"
        assert found == len(driver.PLATFORMS), "a platform stopped uploading a wheel"

    def test_the_workflow_is_dispatch_only(self, workflow):
        # A push trigger on this would build wheels on every commit; it is a
        # ~20-minute-per-wheel job that guards no regression.
        assert set(workflow[True]) == {"workflow_dispatch"}

    def test_the_run_name_carries_the_tag_the_driver_searches_for(self, workflow):
        assert "inputs.run_tag" in workflow["run-name"]


class TestDriver:
    def test_artifact_names_parse(self, driver):
        match = driver.ARTIFACT_RE.match("quadrants-wheel-windows-py3.13")
        assert match is not None
        assert match.group("platform") == "windows"
        assert match.group("python") == "3.13"
        assert driver.ARTIFACT_RE.match("invariant-load-arms-py3.11") is None

    def test_it_reads_the_platform_table_from_the_resolver(self, driver, resolver):
        # Not a copy: the driver validates --platforms against what the runner
        # will actually obey.
        assert driver.PLATFORMS == resolver.PLATFORMS
        assert driver.PYTHONS == resolver.PYTHONS

    def test_the_token_precedence(self, driver, monkeypatch):
        monkeypatch.setenv("GH_TOKEN", "from-gh")
        monkeypatch.setenv("GITHUB_TOKEN", "from-github")
        assert driver.resolve_token() == "from-gh"
        monkeypatch.delenv("GH_TOKEN")
        assert driver.resolve_token() == "from-github"

    def test_a_missing_token_says_what_to_do(self, driver, monkeypatch):
        monkeypatch.delenv("GH_TOKEN", raising=False)
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.setattr(
            driver.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError())
        )
        with pytest.raises(SystemExit, match="gh auth login"):
            driver.resolve_token()

    @pytest.mark.parametrize(
        "url",
        [
            "https://github.com/algorithmicsimplicity/algan",
            "https://github.com/algorithmicsimplicity/algan.git",
            "git@github.com:algorithmicsimplicity/algan.git",
        ],
    )
    def test_owner_and_repo_come_off_the_remote(self, driver, monkeypatch, url):
        class Result:
            stdout = url + "\n"

        monkeypatch.setattr(driver.subprocess, "run", lambda *a, **k: Result())
        assert driver.detect_repo() == ("algorithmicsimplicity", "algan")

    def test_the_redirect_handler_drops_the_token_off_host(self, driver):
        handler = driver._StripAuthOnRedirect()

        def redirect(url: str, new_url: str):
            request = urllib.request.Request(url)
            request.add_header("Authorization", "Bearer secret")
            return handler.redirect_request(request, None, 302, "Found", {}, new_url)

        off_host = redirect(
            "https://api.github.com/repos/o/r/actions/artifacts/1/zip",
            "https://productionresultssa.blob.core.windows.net/abc",
        )
        assert off_host.get_header("Authorization") is None

        same_host = redirect(
            "https://api.github.com/repos/o/r/actions/runs/1",
            "https://api.github.com/repos/o/r/actions/runs/2",
        )
        assert same_host.get_header("Authorization") == "Bearer secret"

    def test_install_refuses_when_nothing_matches_this_interpreter(
        self, driver, tmp_path
    ):
        rows = [{"file": "quadrants-1.3.1-cp99-cp99-nonesuch.whl"}]
        with pytest.raises(SystemExit, match="no downloaded wheel matches"):
            driver.install_matching(tmp_path, rows)

    def test_install_picks_the_wheel_for_this_interpreter(
        self, driver, monkeypatch, tmp_path
    ):
        import sys

        key = {"linux": "linux", "darwin": "macos", "win32": "windows"}[sys.platform]
        tag = driver.PLATFORMS[key]["wheel_tag"]
        cp = f"cp{sys.version_info.major}{sys.version_info.minor}"
        mine = f"quadrants-1.3.1.dev0+gabc-{cp}-{cp}-{tag}.whl"
        rows = [{"file": "quadrants-1.3.1-cp39-cp39-nonesuch.whl"}, {"file": mine}]

        called: dict = {}

        class Result:
            returncode = 0

        def fake_run(command, *a, **k):
            called["command"] = command
            return Result()

        monkeypatch.setattr(driver.subprocess, "run", fake_run)
        driver.install_matching(tmp_path, rows)
        assert str(tmp_path / mine) in called["command"]
        # Without this the installer can consider the requirement already
        # satisfied by the PyPI wheel and do nothing.
        assert "--reinstall-package" in called["command"]

    def test_it_extracts_wheels_out_of_the_artifact_zips(
        self, driver, monkeypatch, tmp_path
    ):
        # The real download cannot run from here (it redirects to blob storage,
        # which this container's egress policy refuses), so the network is the
        # one thing stubbed: everything downstream of it -- selecting the wheel
        # artifacts, unzipping, hashing, and the rows that become the manifest
        # -- is the real code.
        import io
        import zipfile

        def zip_of(*names: str) -> bytes:
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w") as archive:
                for name in names:
                    archive.writestr(name, b"wheel bytes for " + name.encode())
            return buffer.getvalue()

        artifacts = {
            "artifacts": [
                {
                    "id": 1,
                    "name": "quadrants-wheel-linux-py3.11",
                    "size_in_bytes": 10,
                    "expired": False,
                },
                {
                    "id": 2,
                    "name": "quadrants-wheel-macos-py3.11",
                    "size_in_bytes": 10,
                    "expired": False,
                },
                # Not a wheel artifact, and an expired one: neither should be
                # downloaded, and the expired one must not abort the rest.
                {
                    "id": 3,
                    "name": "invariant-load-arms-py3.11",
                    "size_in_bytes": 10,
                    "expired": False,
                },
                {
                    "id": 4,
                    "name": "quadrants-wheel-windows-py3.11",
                    "size_in_bytes": 10,
                    "expired": True,
                },
            ]
        }
        blobs = {
            1: zip_of("quadrants-1.3.1-cp311-cp311-manylinux_2_27_x86_64.whl"),
            2: zip_of("quadrants-1.3.1-cp311-cp311-macosx_13_0_arm64.whl", "build.log"),
        }

        def fake_api(path, token, **kwargs):
            if path.endswith("/artifacts?per_page=100"):
                return artifacts
            match = re.search(r"/artifacts/(\d+)/zip", path)
            assert match, path
            return blobs[int(match.group(1))]

        monkeypatch.setattr(driver, "api", fake_api)
        rows = driver.download_wheels("o", "r", "tok", 7, tmp_path)

        assert {row["file"] for row in rows} == {
            "quadrants-1.3.1-cp311-cp311-manylinux_2_27_x86_64.whl",
            "quadrants-1.3.1-cp311-cp311-macosx_13_0_arm64.whl",
        }
        assert {row["platform"] for row in rows} == {"linux", "macos"}
        assert all(row["python"] == "3.11" for row in rows)
        assert all(row["version"] == "1.3.1" for row in rows)
        # Non-wheel members stay out of the wheelhouse.
        assert not (tmp_path / "build.log").exists()
        # The digest is of the file that actually landed, not of the zip.
        landed = tmp_path / "quadrants-1.3.1-cp311-cp311-macosx_13_0_arm64.whl"
        assert rows[1]["sha256"] == driver._sha256(landed)
        assert rows[1]["bytes"] == landed.stat().st_size

    def test_a_run_with_no_wheels_says_what_it_did_have(
        self, driver, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(
            driver,
            "api",
            lambda *a, **k: {
                "artifacts": [
                    {"id": 9, "name": "quadrants-macos-logs-py3.11", "expired": False}
                ]
            },
        )
        with pytest.raises(SystemExit, match="uploaded no wheel artifacts"):
            driver.download_wheels("o", "r", "tok", 7, tmp_path)

    def test_the_manifest_records_provenance(self, driver, tmp_path):
        run = {
            "id": 42,
            "html_url": "https://github.com/o/r/actions/runs/42",
            "display_title": "Quadrants wheels: linux py3.11 [algan-deadbeef]",
            "conclusion": "success",
            "head_branch": "some-branch",
            "head_sha": "0" * 40,
        }
        rows = [
            {"file": "w.whl", "platform": "linux", "python": "3.11", "sha256": "ab"}
        ]
        path = driver.write_manifest(tmp_path, run, rows)
        manifest = json.loads(path.read_text())
        assert manifest["run_id"] == 42
        assert manifest["head_sha"] == "0" * 40
        assert manifest["wheels"] == rows
        # The patch digests are the other half of "which wheel is this": a
        # patch set that changed makes a differently-behaving wheel of the
        # same version.
        assert manifest["patches"], "no patch digests recorded"
        assert {row["name"] for row in manifest["patches"]} == {
            p.name for p in (REPO_ROOT / "quadrants_patches").glob("[0-9]*.patch")
        }

    def test_the_manifest_is_what_gitignore_keeps(self, driver):
        # The wheels are ignored and the manifest is not; if that inverts, the
        # provenance record silently stops being committed.
        text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        assert "/quadrants_wheels/*" in text
        assert "!/quadrants_wheels/manifest.json" in text
