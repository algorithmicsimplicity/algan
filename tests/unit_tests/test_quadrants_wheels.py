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
import platform
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
        assert all(out[name] == "true" for name in resolver.PLATFORMS)
        assert json.loads(out["selected"]) == list(resolver.PLATFORMS)
        assert json.loads(out["pythons"]) == ["3.11"]

    def test_one_platform_leaves_the_others_off(self, resolver):
        out = resolver.resolve({"IN_PLATFORMS": "macos"})
        assert out["macos"] == "true"
        assert json.loads(out["selected"]) == ["macos"]
        assert all(
            out[name] == "false" for name in resolver.PLATFORMS if name != "macos"
        )

    def test_the_two_linux_platforms_are_distinct_wheels(self, resolver):
        # Both legs build a manylinux wheel and the release validator files them
        # by substring, so a shared tag would let one leg satisfy the other's
        # slot and a 16-wheel release would go out with 8 of one architecture.
        tags = {name: spec["wheel_tag"] for name, spec in resolver.PLATFORMS.items()}
        assert len(set(tags.values())) == len(tags)
        assert tags["linux"] not in tags["linux_arm64"]
        assert tags["linux_arm64"] not in tags["linux"]

    def test_no_linux_tag_promises_more_than_its_container(self, resolver):
        """The tag may exceed the image's floor, never undercut it.

        `quay.io/pypa/manylinux_2_28_x86_64` guarantees glibc 2.28 and nothing
        older, so stamping `manylinux_2_27_x86_64` on its output would be a
        promise the container does not make.

        The other direction is legitimate and aarch64 is living proof: it
        builds in `manylinux_2_34` and stamps `manylinux_2_35`, because the
        prebuilt LLVM it links carries a GLOBAL `_dl_find_object@GLIBC_2.35`
        that the image's own glibc happens to satisfy. The container sets a
        floor; something it links can still push the wheel above it, and
        `verify_wheel_tag.py` is what notices when it does.
        """
        for name, spec in resolver.PLATFORMS.items():
            if "container" not in spec:
                continue
            image = spec["container"].rsplit("/", 1)[-1]
            image_parts = image.split("_")
            tag_parts = spec["wheel_tag"].split("_")
            assert image_parts[0] == tag_parts[0] == "manylinux", name
            assert image_parts[3:] == tag_parts[3:], (
                f"{name}: builds for {image_parts[3:]} but stamps {tag_parts[3:]}"
            )
            image_glibc = (int(image_parts[1]), int(image_parts[2]))
            tag_glibc = (int(tag_parts[1]), int(tag_parts[2]))
            assert tag_glibc >= image_glibc, (
                f"{name}: stamps {spec['wheel_tag']}, which claims to run on "
                f"something older than {image} guarantees"
            )

    def test_only_the_linux_platforms_build_in_a_container(self, resolver):
        # macOS and Windows have no container story: their wheels' floor is the
        # SDK and the runner, and `container:` on those jobs would be ignored.
        containerised = {n for n, s in resolver.PLATFORMS.items() if "container" in s}
        assert containerised == {"linux", "linux_arm64"}
        out = resolver.resolve({})
        for name in resolver.PLATFORMS:
            assert (f"container_{name}" in out) == (name in containerised)
            assert out[f"wheel_tag_{name}"] == resolver.PLATFORMS[name]["wheel_tag"]

    def test_all_expands(self, resolver):
        out = resolver.resolve({"IN_PLATFORMS": "all"})
        assert all(out[name] == "true" for name in resolver.PLATFORMS)

    def test_a_full_python_matrix(self, resolver):
        out = resolver.resolve({"IN_PYTHONS": "3.10,3.11,3.12,3.13"})
        assert json.loads(out["pythons"]) == ["3.10", "3.11", "3.12", "3.13"]
        expected = len(resolver.PLATFORMS) * len(resolver.PYTHONS)
        assert f"{expected} wheel(s)" in out["summary"]

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

    def test_a_platform_key_can_be_a_job_id_and_an_artifact_name(self, resolver):
        # Three things consume a key verbatim: a job id, a `$GITHUB_OUTPUT`
        # name (`runner_<key>`, read as `needs.plan.outputs.runner_<key>`, where
        # a hyphen would parse as subtraction), and the artifact name the driver
        # parses back. `[a-z0-9_]` is the intersection.
        for name in resolver.PLATFORMS:
            assert re.fullmatch(r"[a-z][a-z0-9_]*", name), name


class TestThePublishGate:
    """A PyPI version cannot be amended, so "complete" has to mean the table."""

    def complete(self, resolver, **overrides):
        env = {
            "APPLY_PATCHES": "true",
            "SELECTED": json.dumps(list(resolver.PLATFORMS)),
            "PYTHONS": json.dumps(list(resolver.PYTHONS)),
        }
        env.update(overrides)
        return env

    def test_the_complete_patched_matrix_passes(self, resolver):
        assert resolver.publish_failures(self.complete(resolver)) == []

    def test_every_platform_in_the_table_is_required(self, resolver):
        # The point of reading the table: this is what a hardcoded list of
        # platform names in the YAML stopped doing when the matrix grew.
        for name in resolver.PLATFORMS:
            others = [p for p in resolver.PLATFORMS if p != name]
            failures = resolver.publish_failures(
                self.complete(resolver, SELECTED=json.dumps(others))
            )
            assert failures == [f"publish requires {name}"]

    def test_a_stock_build_cannot_publish(self, resolver):
        failures = resolver.publish_failures(
            self.complete(resolver, APPLY_PATCHES="false")
        )
        assert failures == ["publish requires apply_patches=true"]

    def test_a_partial_python_matrix_cannot_publish(self, resolver):
        failures = resolver.publish_failures(
            self.complete(resolver, PYTHONS=json.dumps(["3.11"]))
        )
        assert len(failures) == 1
        assert "requires Python" in failures[0]

    def test_the_cli_flag_refuses_loudly(self, resolver, monkeypatch):
        for key, value in self.complete(resolver, SELECTED="[]").items():
            monkeypatch.setenv(key, value)
        with pytest.raises(SystemExit, match="Refusing a partial/stock"):
            resolver.main(["--check-publish"])

    def test_the_workflow_gate_calls_it_with_what_it_reads(self, resolver, workflow):
        steps = workflow["jobs"]["plan"]["steps"]
        gate = [s for s in steps if "--check-publish" in str(s.get("run", ""))]
        assert len(gate) == 1, "the plan job stopped gating the publish inputs"
        assert gate[0]["if"] == "${{ inputs.publish }}"
        # The env the script reads and the env the YAML sets are one contract.
        assert set(gate[0]["env"]) == {"APPLY_PATCHES", "SELECTED", "PYTHONS"}
        assert "steps.resolve.outputs.selected" in gate[0]["env"]["SELECTED"]


class TestVerifyWheelTag:
    """`scripts/gate/verify_wheel_tag.py` -- the thing that makes the tag true.

    Every case here is fed saved `auditwheel show` text, because the failure
    this guards is a release that goes out with a tag nobody measured, and a
    test that needed auditwheel and a real wheel would not run in this suite at
    all.
    """

    # auditwheel hard-wraps its prose. This sample keeps the wrap in the middle
    # of the verdict sentence on purpose: it is where a line-oriented regex
    # stops matching, which is a silently passing gate rather than a failing
    # one.
    WRAPPED = """\
quadrants-1.3.1.dev0-cp311-cp311-linux_x86_64.whl is consistent with the
following platform tag: "manylinux_2_28_x86_64".

The wheel references external versioned symbols in these system-provided
shared libraries: libc.so.6 with versions {'GLIBC_2.14', 'GLIBC_2.2.5'}.

This constrains the platform tag to "manylinux_2_28_x86_64".
"""

    @pytest.fixture(scope="class")
    def gate(self):
        return _load(
            REPO_ROOT / "scripts" / "gate" / "verify_wheel_tag.py", "verify_wheel_tag"
        )

    def test_it_reads_the_verdict_through_auditwheels_line_wrap(self, gate):
        assert gate.auditwheel_verdict(self.WRAPPED) == "manylinux_2_28_x86_64"

    def test_it_reads_an_unwrapped_and_unquoted_verdict_too(self, gate):
        text = "w.whl is consistent with the following platform tag: manylinux_2_34_aarch64."
        assert gate.auditwheel_verdict(text) == "manylinux_2_34_aarch64"

    def test_output_it_cannot_parse_is_a_failure_not_a_pass(self, gate):
        with pytest.raises(SystemExit, match="could not find auditwheel's verdict"):
            gate.auditwheel_verdict("auditwheel: error: cannot read wheel")

    @pytest.mark.parametrize(
        ("tag", "expected"),
        [
            ("manylinux_2_28_x86_64", ((2, 28), "x86_64")),
            ("manylinux_2_34_aarch64", ((2, 34), "aarch64")),
            ("manylinux2014_x86_64", ((2, 17), "x86_64")),
            ("manylinux1_x86_64", ((2, 5), "x86_64")),
            ("linux_x86_64", None),
            ("macosx_13_0_arm64", None),
        ],
    )
    def test_tag_parsing(self, gate, tag, expected):
        assert gate.parse_tag(tag) == expected

    def test_a_wheel_that_matches_its_tag_passes(self, gate):
        assert gate.check("manylinux_2_28_x86_64", "manylinux_2_28_x86_64") == []

    def test_a_wheel_older_than_its_tag_passes(self, gate):
        # It would run on more systems than the tag admits, which costs reach,
        # not correctness -- and it is what the x86-64 container actually does
        # (its LLVM tops out at GLIBC_2.14).
        assert gate.check("manylinux2014_x86_64", "manylinux_2_28_x86_64") == []

    def test_a_wheel_newer_than_its_tag_is_refused(self, gate):
        problems = gate.check("manylinux_2_34_x86_64", "manylinux_2_28_x86_64")
        assert len(problems) == 1
        assert "needs glibc 2.34" in problems[0]
        assert "pip would install it on systems it cannot run on" in problems[0]

    def test_the_wrong_architecture_is_refused(self, gate):
        # A leg pointed at the wrong runner builds a real wheel with a real
        # tag; only the architecture gives it away.
        problems = gate.check("manylinux_2_28_x86_64", "manylinux_2_28_aarch64")
        assert any("architecture mismatch" in p for p in problems)

    def test_a_wheel_no_policy_accepts_is_refused_with_the_reason(self, gate):
        problems = gate.check("linux_x86_64", "manylinux_2_28_x86_64")
        assert len(problems) == 1
        assert "not a manylinux policy at all" in problems[0]

    def test_the_matrix_tags_are_ones_this_gate_understands(self, gate, resolver):
        # The workflow passes `wheel_tag_<platform>` straight in as --expect.
        for name, spec in resolver.PLATFORMS.items():
            if "container" not in spec:
                continue
            assert gate.parse_tag(spec["wheel_tag"]) is not None, name

    def test_end_to_end_through_the_cli(self, gate, tmp_path):
        saved = tmp_path / "auditwheel.txt"
        saved.write_text(self.WRAPPED, encoding="utf-8")
        wheel = tmp_path / "quadrants-1.3.1-cp311-cp311-linux_x86_64.whl"
        argv = [str(wheel), "--from-file", str(saved), "--expect"]
        assert gate.main([*argv, "manylinux_2_28_x86_64"]) == 0
        with pytest.raises(SystemExit, match="refusing to stamp a tag"):
            gate.main([*argv, "manylinux_2_17_x86_64"])

    def test_a_refusal_carries_auditwheels_evidence(self, gate, tmp_path):
        """The verdict says what; only auditwheel's own output says why.

        The aarch64 leg refused with "needs glibc 2.35" and nothing about
        which library asked for it (run 34032726212), which is a diagnosis
        that costs another 13-minute build to obtain.
        """
        saved = tmp_path / "auditwheel.txt"
        saved.write_text(self.WRAPPED, encoding="utf-8")
        wheel = tmp_path / "quadrants-1.3.1-cp311-cp311-linux_x86_64.whl"
        with pytest.raises(SystemExit) as excinfo:
            gate.main(
                [
                    str(wheel),
                    "--from-file",
                    str(saved),
                    "--expect",
                    "manylinux_2_17_x86_64",
                ]
            )
        message = str(excinfo.value)
        assert "--- auditwheel show ---" in message
        # The section that names the libraries and their symbol versions.
        assert "references external versioned symbols" in message
        assert "libc.so.6 with versions" in message


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

    def test_the_two_linux_legs_are_one_recipe(self):
        """x86-64 and aarch64 build identically, or the difference is a bug.

        GitHub Actions has no way to share a job body between two `runs-on`
        values -- no anchors, and one job per platform is what
        `test_every_platform_has_a_job_gated_on_its_output` requires -- so the
        two Linux legs are a copy. This is what keeps the copy honest: the two
        blocks, character for character, with the platform key rewritten out.
        Everything that genuinely varies with the architecture reaches them as
        a `needs.plan.outputs.*` value (runner, container, wheel tag), so a
        difference here is a fix applied to one leg and forgotten on the other.

        It reads the **raw text**, not `yaml.safe_load`'s output, because the
        parser drops comments -- and comments are most of what a reader of this
        workflow is relying on. Generating the aarch64 twin by substitution
        mangled `manylinux` into `manylinux_arm64` in four of them, and a
        parsed comparison saw nothing at all.
        """
        import difflib

        # Only where the key stands as its own token: a plain `.replace()` also
        # rewrites the `linux` inside `manylinux`, so a mangled
        # `manylinux_arm64_2_27` would normalize to the same text as a correct
        # `manylinux_2_27` -- the legs comparing equal exactly where one is
        # wrong. Longest alternative first, so `linux` cannot eat the head of
        # `linux_arm64`.
        token = re.compile(
            r"(?<![A-Za-z0-9])(linux_arm64|linux-arm64|linux)(?![A-Za-z0-9])"
        )
        sibling = re.compile(r"^  [A-Za-z_][A-Za-z0-9_-]*:")

        def one_recipe(name: str) -> str:
            lines = WORKFLOW.read_text(encoding="utf-8").splitlines()
            start = next(i for i, ln in enumerate(lines) if ln.startswith(f"  {name}:"))
            end = next(
                (i for i in range(start + 1, len(lines)) if sibling.match(lines[i])),
                len(lines),
            )
            block = "\n".join(lines[start:end]).rstrip()
            return token.sub("PLATFORM", block)

        x86 = one_recipe("linux")
        arm = one_recipe("linux_arm64")
        if x86 != arm:
            diff = "\n".join(
                difflib.unified_diff(
                    x86.splitlines(),
                    arm.splitlines(),
                    fromfile="jobs.linux",
                    tofile="jobs.linux_arm64",
                    lineterm="",
                )
            )
            raise AssertionError(
                "the two Linux legs have drifted apart; they build the same "
                f"wheel on two architectures and must stay one recipe:\n{diff}"
            )

    def test_the_linux_cache_keys_are_keyed_by_architecture(self, workflow):
        # `runner.os` is 'Linux' on both legs. Without `runner.arch` the arm
        # leg restores the x86-64 LLVM archive the other leg cached, and the
        # failure surfaces much later, inside clang.
        for name in ("linux", "linux_arm64"):
            caches = [
                step
                for step in workflow["jobs"][name]["steps"]
                if str(step.get("uses", "")).startswith("actions/cache")
            ]
            assert caches, f"{name} caches nothing"
            for step in caches:
                assert "runner.arch" in step["with"]["key"], name
                assert "runner.arch" in step["with"]["restore-keys"], name

    def test_the_linux_legs_build_in_the_container_the_table_names(
        self, resolver, workflow
    ):
        for name in resolver.PLATFORMS:
            job = workflow["jobs"][name]
            if "container" in resolver.PLATFORMS[name]:
                assert f"needs.plan.outputs.container_{name}" in str(
                    job.get("container", "")
                ), f"{name} does not build in its container"
            else:
                assert "container" not in job, f"{name} should not have a container"

    def test_no_job_reads_another_platforms_outputs(self, resolver, workflow):
        """A leg must read its own row of the table, and only its own.

        This is the blind spot in `test_the_two_linux_legs_are_one_recipe` and
        it has already bitten: the aarch64 leg was given the x86-64 leg's
        `cc_linux`/`cxx_linux` by a global substitution, and the drift test saw
        nothing, because rewriting the platform key out maps `cc_linux` and
        `cc_linux_arm64` to the same token. The two tests are complementary --
        one says the legs are identical, this one says each is identical to
        *itself*.
        """
        for name in resolver.PLATFORMS:
            text = yaml.safe_dump(workflow["jobs"][name])
            for reference in set(re.findall(r"needs\.plan\.outputs\.(\w+)", text)):
                for other in resolver.PLATFORMS:
                    if reference == other or reference.endswith(f"_{other}"):
                        assert other == name, (
                            f"job {name!r} reads {reference!r}, which belongs "
                            f"to {other!r}"
                        )

    def test_the_containerised_legs_do_not_reach_for_the_host(self, workflow):
        # The manylinux images are RPM-based and run as root: `sudo` is not
        # installed and `apt-get` does not exist, so either one is a step that
        # cannot run. `setup-python` would install an interpreter built against
        # a different libc than the wheel, which is worse -- it would work.
        for name in ("linux", "linux_arm64"):
            text = yaml.safe_dump(workflow["jobs"][name])
            for forbidden in ("sudo ", "apt-get", "actions/setup-python"):
                assert forbidden not in text, f"{name} still uses {forbidden!r}"

    def test_the_wheel_is_installed_before_it_is_stamped(self, resolver, workflow):
        """The stamp can put a tag on the wheel that its own container rejects.

        The tag is what the wheel *measures*, and that can exceed the image's
        glibc: the aarch64 leg stamps `manylinux_2_35` inside a 2.34 image, and
        pip then declines its own build with "not a supported wheel on this
        platform" (run 34034938922). Installing first tests the same bits under
        the one tag the container will accept.
        """
        for name in resolver.PLATFORMS:
            if "container" not in resolver.PLATFORMS[name]:
                continue
            names = [
                str(step.get("name", "")) for step in workflow["jobs"][name]["steps"]
            ]
            install = next(i for i, n in enumerate(names) if n.startswith("Install it"))
            stamp = next(
                i for i, n in enumerate(names) if n.startswith("Verify and stamp")
            )
            assert install < stamp, (
                f"{name} installs after stamping, which its own container may refuse"
            )

    def test_every_built_wheel_is_verified_before_it_is_stamped(
        self, resolver, workflow
    ):
        """A stamp without the check is exactly the bug this whole thing fixes."""
        for name in resolver.PLATFORMS:
            if "container" not in resolver.PLATFORMS[name]:
                continue
            steps = workflow["jobs"][name]["steps"]
            runs = [str(step.get("run", "")) for step in steps]
            verify = [i for i, r in enumerate(runs) if "verify_wheel_tag.py" in r]
            stamp = [i for i, r in enumerate(runs) if "wheel tags --platform-tag" in r]
            assert len(verify) == 1, f"{name} does not verify its tag"
            assert len(stamp) == 1, f"{name} does not stamp its tag"
            assert verify[0] <= stamp[0], f"{name} stamps before it verifies"
            # The tag it checks against and the tag it stamps are one value,
            # read from the table rather than written twice.
            step = steps[stamp[0]]
            assert f"needs.plan.outputs.wheel_tag_{name}" in str(step.get("env", ""))

    def test_no_two_jobs_upload_the_same_artifact_name(self, workflow):
        # Two jobs uploading one name is not a merge: the second upload fails.
        names: list[str] = []
        for job in workflow["jobs"].values():
            for step in job.get("steps", []):
                name = (step.get("with") or {}).get("name", "")
                if name and str(step.get("uses", "")).startswith(
                    "actions/upload-artifact"
                ):
                    names.append(name)
        assert len(names) == len(set(names)), sorted(names)

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

    def test_no_expression_uses_an_empty_true_branch(self, workflow):
        # `a && b || c` in GitHub expressions is not a ternary: it falls through
        # to `c` whenever `b` is falsy, and '' is falsy. So `x && '' || 'MARK'`
        # renders MARK unconditionally. The first run of this workflow labelled
        # a patched build "(STOCK)" that way.
        text = WORKFLOW.read_text(encoding="utf-8")
        offenders = re.findall(r"\$\{\{[^}]*&&\s*''\s*\|\|[^}]*\}\}", text)
        assert not offenders, (
            "these render their fallback unconditionally, because the true "
            f"branch is the empty string: {offenders}"
        )


class TestDriver:
    def test_artifact_names_parse(self, driver):
        match = driver.ARTIFACT_RE.match("quadrants-wheel-windows-py3.13")
        assert match is not None
        assert match.group("platform") == "windows"
        assert match.group("python") == "3.13"
        assert driver.ARTIFACT_RE.match("invariant-load-arms-linux-py3.11") is None

    def test_an_underscored_platform_key_survives_the_round_trip(self, driver):
        # The name the workflow writes for every platform in the table has to
        # come back as that platform; `linux_arm64` is the one with a `_` in it.
        for name in driver.PLATFORMS:
            match = driver.ARTIFACT_RE.match(f"quadrants-wheel-{name}-py3.11")
            assert match is not None, name
            assert match.group("platform") == name
            assert match.group("python") == "3.11"

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

    @pytest.mark.parametrize(
        ("system", "machine", "expected"),
        [
            ("linux", "x86_64", "linux"),
            ("linux", "aarch64", "linux_arm64"),
            ("linux", "arm64", "linux_arm64"),
            ("darwin", "arm64", "macos"),
            ("win32", "AMD64", "windows"),
            # Nothing here builds these, and saying so is the answer. An Intel
            # Mac used to be handed the arm64 tag, whose failure reads as a
            # download that came back short rather than as a wheel that does
            # not exist.
            ("darwin", "x86_64", None),
            ("freebsd14", "x86_64", None),
        ],
    )
    def test_the_machine_decides_which_wheel_belongs_to_it(
        self, driver, monkeypatch, system, machine, expected
    ):
        monkeypatch.setattr(driver.sys, "platform", system)
        monkeypatch.setattr(driver.platform, "machine", lambda: machine)
        assert driver.platform_key_for_this_machine() == expected

    def test_every_platform_in_the_table_is_reachable_from_some_machine(
        self, driver, monkeypatch
    ):
        # A key nothing maps to is a wheel the build produces and `--install`
        # can never put anywhere.
        reached = set()
        for system, machine in [
            ("linux", "x86_64"),
            ("linux", "aarch64"),
            ("darwin", "arm64"),
            ("win32", "AMD64"),
        ]:
            monkeypatch.setattr(driver.sys, "platform", system)
            monkeypatch.setattr(driver.platform, "machine", lambda m=machine: m)
            reached.add(driver.platform_key_for_this_machine())
        assert reached == set(driver.PLATFORMS)

    def test_install_refuses_when_nothing_matches_this_interpreter(
        self, driver, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(driver, "platform_key_for_this_machine", lambda: "linux")
        rows = [{"file": "quadrants-1.3.1-cp99-cp99-nonesuch.whl"}]
        with pytest.raises(SystemExit, match="no downloaded wheel matches"):
            driver.install_matching(tmp_path, rows)

    def test_install_says_so_where_no_wheel_is_built(self, driver, monkeypatch):
        monkeypatch.setattr(driver, "platform_key_for_this_machine", lambda: None)
        with pytest.raises(SystemExit, match="does not know what to do"):
            driver.install_matching(Path("."), [])

    def test_install_picks_the_wheel_for_this_interpreter(
        self, driver, monkeypatch, tmp_path
    ):
        import sys

        key = driver.platform_key_for_this_machine()
        if key is None:
            pytest.skip(f"no wheel is built for {sys.platform}/{platform.machine()}")
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
