"""Cleanup preserves changed, active and unmerged branches and leases deletions."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[2] / "scripts" / "cleanup_mac_branches.py"
SPEC = importlib.util.spec_from_file_location("mac_cleanup", PATH)
cleanup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cleanup)


def test_cleanup_only_selects_unchanged_merged_inactive_branches():
    names = [
        f"codex/mac-{name}"
        for name in ("done", "changed", "active", "unmerged", "absent")
    ]
    records = [{"branch": name, "sha": str(i + 1) * 40} for i, name in enumerate(names)]
    current = {row["branch"]: row["sha"] for row in records[:-1]}
    current[names[1]] = "f" * 40
    plan = cleanup.deletion_plan(
        records, current, {names[2]}, "merged", lambda sha, _: sha != "4" * 40
    )
    assert [r["action"] for r in plan] == [
        "delete",
        "head changed; preserve",
        "open pull request; preserve",
        "history not merged; preserve",
        "already absent",
    ]
    assert cleanup.delete_command(plan) == [
        "git",
        "push",
        "--atomic",
        f"--force-with-lease=refs/heads/{names[0]}:{'1' * 40}",
        "origin",
        f":refs/heads/{names[0]}",
    ]


@pytest.mark.parametrize(
    "name",
    [
        "master",
        "stable",
        "codex/mps-normal-release-gate",
        "codex/mac-../master",
        "--all",
    ],
)
def test_cleanup_refuses_unrelated_or_invalid_names(name):
    with pytest.raises(ValueError, match="Invalid"):
        cleanup.deletion_plan(
            [{"branch": name, "sha": "a" * 40}], {}, set(), "merged", lambda *_: True
        )


def test_cleanup_refuses_duplicate_entries_and_invalid_hashes():
    row = {"branch": "codex/mac-done", "sha": "a" * 40}
    with pytest.raises(ValueError, match="duplicate"):
        cleanup.deletion_plan([row, row], {}, set(), "merged", lambda *_: True)
    with pytest.raises(ValueError, match="Invalid"):
        cleanup.deletion_plan(
            [dict(row, sha="HEAD")], {}, set(), "merged", lambda *_: True
        )


def test_cleanup_has_no_push_when_nothing_can_be_removed():
    row = {"branch": "claude/eager-cori-h61xc8", "sha": "a" * 40}
    plan = cleanup.deletion_plan([row], {}, set(), "merged", lambda *_: True)
    assert cleanup.delete_command(plan) is None


def test_atomic_cleanup_preserves_every_branch_if_one_moves(tmp_path):
    """Exercise real Git leases and atomicity against a disposable local remote."""
    remote, client = tmp_path / "remote", tmp_path / "client"

    def run(*args, check=True, **kwargs):
        return subprocess.run(
            ["git", *args], text=True, capture_output=True, check=check, **kwargs
        )

    run("init", "--bare", str(remote))
    run("init", "--bare", str(client))
    run("-C", str(client), "remote", "add", "origin", str(remote))
    tree = run("-C", str(client), "mktree", input="").stdout.strip()
    commit_args = [
        "-C",
        str(client),
        "-c",
        "user.name=Cleanup test",
        "-c",
        "user.email=cleanup-test@example.invalid",
        "commit-tree",
        tree,
    ]
    original = run(*commit_args, "-m", "original").stdout.strip()
    newer = run(*commit_args, "-p", original, "-m", "new work").stdout.strip()
    names = ["codex/mac-done", "codex/mac-changed"]
    run(
        "-C",
        str(client),
        "push",
        "origin",
        *[f"{original}:refs/heads/{n}" for n in names],
    )
    records = [{"branch": name, "sha": original} for name in names]
    plan = cleanup.deletion_plan(
        records, dict.fromkeys(names, original), set(), original, lambda *_: True
    )
    # Another writer publishes after the cleanup inventory was read.
    run("-C", str(client), "push", "origin", f"{newer}:refs/heads/{names[1]}")
    result = run(*cleanup.delete_command(plan)[1:], cwd=client, check=False)
    assert result.returncode != 0
    for name, expected in zip(names, (original, newer)):
        actual = run(
            "-C", str(remote), "rev-parse", f"refs/heads/{name}"
        ).stdout.strip()
        assert actual == expected
