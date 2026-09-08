"""Remove the recorded Mac investigation branches after PR 115 is merged.

Dry-run by default. Deletions use explicit Git leases in one atomic push;
new commits, open pull requests and unmerged history keep their branches.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import urllib.request
from pathlib import Path

REPOSITORY = "algorithmicsimplicity/algan"
INTEGRATION_BRANCH = "codex/mac-depth-buffer-reuse"
MANIFEST = Path("benchmarks/performance/reports/mac_2026_09/INTEGRATION_BRANCHES.json")


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def is_ancestor(sha, merged):
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", sha, merged], check=False
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(f"Cannot verify ancestry of {sha}")
    return result.returncode == 0


def deletion_plan(records, current, active, merged, ancestor):
    seen = set()
    plan = []
    for record in records:
        name, sha = record["branch"], record["sha"]
        allowed = (
            re.fullmatch(r"codex/mac-[a-z0-9-]+", name)
            or name == "claude/eager-cori-h61xc8"
        )
        if not allowed or name in seen or not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise ValueError(f"Invalid or duplicate cleanup entry: {name}")
        seen.add(name)
        if name not in current:
            reason = "already absent"
        elif current[name] != sha:
            reason = "head changed; preserve"
        elif name in active:
            reason = "open pull request; preserve"
        elif not ancestor(sha, merged):
            reason = "history not merged; preserve"
        else:
            reason = "delete"
        plan.append({"branch": name, "sha": sha, "action": reason})
    return plan


def delete_command(plan):
    selected = [row for row in plan if row["action"] == "delete"]
    if not selected:
        return None
    return [
        "git",
        "push",
        "--atomic",
        *[f"--force-with-lease=refs/heads/{r['branch']}:{r['sha']}" for r in selected],
        "origin",
        *[f":refs/heads/{r['branch']}" for r in selected],
    ]


def api(path):
    request = urllib.request.Request(
        f"https://api.github.com/repos/{REPOSITORY}/{path}",
        headers={
            "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
            "Accept": "application/vnd.github+json",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if os.environ.get("GITHUB_REPOSITORY") != REPOSITORY:
        raise RuntimeError("Cleanup is scoped to the Algan repository")
    if os.environ.get("GITHUB_REF") != "refs/heads/master":
        raise RuntimeError("Cleanup requires the merged master workflow")
    merged = os.environ["GITHUB_SHA"]
    pr = api("pulls/115")
    if not (
        pr["merged"]
        and pr["base"]["ref"] == "master"
        and pr["head"]["ref"] == INTEGRATION_BRANCH
        and pr["head"]["repo"]["full_name"] == REPOSITORY
        and is_ancestor(pr["merge_commit_sha"], merged)
        and is_ancestor(pr["head"]["sha"], merged)
    ):
        raise RuntimeError(
            "PR 115 must be merged with its complete history into master"
        )
    manifest = json.loads(MANIFEST.read_text())
    if manifest["repository"] != REPOSITORY or manifest["pull_request"] != 115:
        raise ValueError("Unexpected cleanup manifest")
    records = manifest["branches"]
    # The integration head cannot be embedded in its own commit's manifest.
    # Resolve only this one entry from the successfully merged PR.
    for record in records:
        if record["branch"] == INTEGRATION_BRANCH:
            if not is_ancestor(record["sha"], pr["head"]["sha"]):
                raise RuntimeError(
                    "Integration no longer contains the recorded candidate"
                )
            record["sha"] = pr["head"]["sha"]
    current = {}
    for line in git("ls-remote", "--heads", "origin").splitlines():
        sha, ref = line.split()
        current[ref.removeprefix("refs/heads/")] = sha
    active = set()
    for page in range(1, 101):
        prs = api(f"pulls?state=open&per_page=100&page={page}")
        for item in prs:
            for side in ("base", "head"):
                repo = item[side].get("repo")
                if repo and repo["full_name"] == REPOSITORY:
                    active.add(item[side]["ref"])
        if len(prs) < 100:
            break
    else:
        raise RuntimeError("Open PR inventory exceeded the page limit")
    plan = deletion_plan(records, current, active, merged, is_ancestor)
    print(json.dumps(plan, indent=2), flush=True)
    Path("mac-branch-cleanup.json").write_text(json.dumps(plan, indent=2) + "\n")
    command = delete_command(plan)
    if args.apply and command:
        subprocess.run(command, check=True)
        print("Deleted the unchanged, fully merged branches in one atomic push.")


if __name__ == "__main__":
    main()
