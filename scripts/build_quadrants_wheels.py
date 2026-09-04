r"""Build Algan's patched Quadrants wheels on GitHub's runners, and bring them home.

Algan depends on ``quadrants>=1.3.0,<1.4`` and patches it: ``quadrants_patches/``
carries four patches against v1.3.0 that the PyPI wheel does not have, and
three of Algan's platforms need a different subset of them. There is no machine
here that can build all of that -- the Metal patches need an Apple GPU box and
the CUDA ones need a toolchain this container does not have -- so the build
happens on GitHub's runners and the wheels come back over the API.

This script is the front door to ``.github/workflows/quadrants_build.yaml``.
One command dispatches the build, waits for it, downloads every wheel it
produced into ``quadrants_wheels/``, and writes a manifest recording exactly
which commit and which patches each wheel came from::

    uv run python scripts/build_quadrants_wheels.py

That builds all three platforms for the default Python (3.11), which is four
wheels' worth of runner time and about 20-40 minutes of waiting. Narrower::

    uv run python scripts/build_quadrants_wheels.py --platforms macos
    uv run python scripts/build_quadrants_wheels.py --python 3.10,3.11,3.12,3.13

Already have a run? Skip the dispatch and just fetch it::

    uv run python scripts/build_quadrants_wheels.py --run-id 12345678
    uv run python scripts/build_quadrants_wheels.py --list

And to put the wheel for *this* machine into the current environment::

    uv run python scripts/build_quadrants_wheels.py --run-id 12345678 --install

Where the wheels land
---------------------
``quadrants_wheels/`` in the repository root, which is **gitignored** except
for its manifest. The wheels are 20-30 MiB each and a full matrix is twelve of
them; this repository already refuses to carry binaries that size in git
(``tests/README.md``, "Where the heavy baselines live" -- the render baselines
are release assets for the same reason). What *is* committed is
``quadrants_wheels/manifest.json``: the run id, the commit, and a sha256 per
wheel and per patch, so which wheel a measurement was taken on stays a fact in
git even though the bytes are not. That is the same split
``scripts/package_baselines.py`` makes with ``tests/baselines.json``.

To share the wheels themselves, attach them to a release -- the manifest's
digests are what makes an uploaded asset verifiable.

Authentication
--------------
``GH_TOKEN`` or ``GITHUB_TOKEN`` if either is set, otherwise ``gh auth token``.
The token needs ``actions:write`` on the repository (dispatching a workflow)
and ``actions:read`` (reading runs and downloading artifacts).

Why the run tag
---------------
``POST /actions/workflows/{id}/dispatches`` answers ``204 No Content``: it does
not tell you which run it just created. So the dispatch generates a random tag,
passes it as an input, and the workflow echoes it into ``run-name``; this then
finds the run whose ``display_title`` carries that tag. Matching on "the newest
run on my branch" instead would pick up somebody else's run started in the same
few seconds, and the failure would look like a build that ignored its inputs.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_FILE = "quadrants_build.yaml"
DEFAULT_OUT = REPO_ROOT / "quadrants_wheels"
PATCH_DIR = REPO_ROOT / "quadrants_patches"
API = "https://api.github.com"

# Artifact names the workflow uploads wheels under, and the shape this parses
# them back into. Kept as one pattern in one place: the workflow writes it, this
# reads it, and a change to either without the other is the bug.
ARTIFACT_GLOB = "quadrants-wheel-*"
ARTIFACT_RE = re.compile(
    r"^quadrants-wheel-(?P<platform>[a-z0-9]+)-py(?P<python>[\d.]+)$"
)


def _load_platform_table() -> tuple[dict, tuple[str, ...]]:
    """Read the platform/Python vocabulary out of the workflow's own resolver.

    The resolver is the thing the runner actually obeys, so importing it here
    rather than restating it keeps this script from validating against a table
    the workflow no longer uses. It is not on the import path (it lives under
    ``.github/``), hence the by-path load that ``tests/`` does for the same file.
    """
    path = REPO_ROOT / ".github" / "workflows" / "scripts" / "resolve_wheel_matrix.py"
    spec = importlib.util.spec_from_file_location("resolve_wheel_matrix", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PLATFORMS, module.PYTHONS


PLATFORMS, PYTHONS = _load_platform_table()


# -----------------------------------------------------------------------------
# GitHub plumbing


class _StripAuthOnRedirect(urllib.request.HTTPRedirectHandler):
    """Drop the ``Authorization`` header when a redirect leaves api.github.com.

    An artifact download is a 302 from the API to blob storage, and that
    storage rejects a request carrying a GitHub bearer token outright ("Only
    one auth mechanism allowed"). Python's default redirect handler re-sends
    every header it was given, so without this the download fails with a 400
    that says nothing about the cause. It is the safer default in its own right
    too: a token should not follow a redirect to a host that did not issue it.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urlparse(req.full_url).netloc != urlparse(newurl).netloc:
            new.remove_header("Authorization")
        return new


_OPENER = urllib.request.build_opener(_StripAuthOnRedirect())


def resolve_token() -> str:
    """``GH_TOKEN``/``GITHUB_TOKEN``, else whatever ``gh`` is logged in as."""
    for name in ("GH_TOKEN", "GITHUB_TOKEN"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    try:
        out = subprocess.run(
            ["gh", "auth", "token"], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        out = None
    if out is not None and out.returncode == 0 and out.stdout.strip():
        return out.stdout.strip()
    raise SystemExit(
        "no GitHub token: set GH_TOKEN or GITHUB_TOKEN, or run `gh auth login`.\n"
        "It needs actions:write (to dispatch) and actions:read (to download)."
    )


def api(
    path: str,
    token: str,
    *,
    method: str = "GET",
    body: dict | None = None,
    raw: bool = False,
):
    """One GitHub REST call. Returns parsed JSON, or bytes when ``raw``."""
    url = path if path.startswith("http") else f"{API}{path}"
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Authorization", f"Bearer {token}")
    request.add_header("Accept", "application/vnd.github+json")
    request.add_header("X-GitHub-Api-Version", "2022-11-28")
    request.add_header("User-Agent", "algan-build-quadrants-wheels")
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with _OPENER.open(request, timeout=300) as response:
            payload = response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:2000]
        hint = ""
        # Dispatching needs a scope that reading does not, so this is the one
        # 403 that arrives *after* several successful calls -- which reads as
        # "the script is broken" rather than "the token is too narrow".
        if error.code == 403 and url.endswith("/dispatches"):
            hint = (
                "\n\nThis token can read the repository but not start a workflow:\n"
                "dispatching needs `actions: write` (classic tokens: the `repo`\n"
                "scope; fine-grained: Actions -> Read and write). Either widen it,\n"
                "or start the run from the Actions tab and come back with\n"
                "`--run-id <id>` to download the wheels."
            )
        raise SystemExit(
            f"GitHub API {method} {url} -> {error.code} {error.reason}\n{detail}{hint}"
        ) from error
    except urllib.error.URLError as error:
        # An artifact download is a redirect off api.github.com to blob
        # storage, so this is where a restricted network shows up -- and it
        # shows up as a transport error with no HTTP status to explain it.
        # Worth naming, because the API calls before it all succeeded and the
        # obvious reading of that is "the script is broken".
        raise SystemExit(
            f"transport error on {method} {url}: {error.reason}\n"
            "Note the host in that error may not be the one that refused: an\n"
            "artifact download redirects off api.github.com to blob storage, and\n"
            "a proxy or egress policy that allows the API can still refuse the\n"
            "redirect target. Downloading the artifact from the run page in a\n"
            "browser is the fallback."
        ) from error
    if raw:
        return payload
    return json.loads(payload) if payload else {}


def detect_repo() -> tuple[str, str]:
    """``owner, repo`` from origin's URL."""
    url = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "remote", "get-url", "origin"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    match = re.search(r"[:/]([^/:]+)/([^/]+?)(?:\.git)?$", url)
    if not match:
        raise SystemExit(f"cannot parse owner/repo out of origin url {url!r}")
    return match.group(1), match.group(2)


def current_branch() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


# -----------------------------------------------------------------------------
# Dispatch and wait


def dispatch(
    owner: str,
    repo: str,
    token: str,
    *,
    ref: str,
    platforms: str,
    pythons: str,
    patched: bool,
) -> str:
    """Ask for a build. Returns the run tag that will identify it."""
    tag = f"algan-{uuid.uuid4().hex[:10]}"
    api(
        f"/repos/{owner}/{repo}/actions/workflows/{WORKFLOW_FILE}/dispatches",
        token,
        method="POST",
        body={
            "ref": ref,
            "inputs": {
                "platforms": platforms,
                "python_versions": pythons,
                # Workflow inputs are strings over the API even when the
                # workflow declares them as booleans.
                "apply_patches": "true" if patched else "false",
                "run_tag": tag,
            },
        },
    )
    return tag


def find_run(owner: str, repo: str, token: str, *, tag: str, ref: str) -> dict | None:
    """The run whose name carries ``tag``, or None if it has not appeared yet."""
    # `branch=` narrows the page but is not what identifies the run: it is
    # ignored when `ref` is a tag rather than a branch, and two dispatches of
    # the same branch are told apart only by the run tag.
    runs = api(
        f"/repos/{owner}/{repo}/actions/workflows/{WORKFLOW_FILE}/runs"
        f"?event=workflow_dispatch&per_page=30&branch={quote(ref, safe='')}",
        token,
    ).get("workflow_runs", [])
    if not runs:
        runs = api(
            f"/repos/{owner}/{repo}/actions/workflows/{WORKFLOW_FILE}/runs"
            f"?event=workflow_dispatch&per_page=30",
            token,
        ).get("workflow_runs", [])
    for run in runs:
        if tag in (run.get("display_title") or "") or tag in (run.get("name") or ""):
            return run
    return None


def wait_for_run(
    owner: str, repo: str, token: str, run_id: int, *, poll: int, timeout_s: int
) -> dict:
    """Poll until the run is finished, reporting each job as it settles.

    `agent_guidance/gpu_harnesses.md` records that GitHub's run status can stay
    ``in_progress`` long after a run has actually finished. So this does not
    trust ``run.status`` alone: a run whose every job has completed is treated
    as finished regardless of what the run object says.
    """
    started = time.monotonic()
    seen: dict[str, str] = {}
    while True:
        run = api(f"/repos/{owner}/{repo}/actions/runs/{run_id}", token)
        jobs = api(
            f"/repos/{owner}/{repo}/actions/runs/{run_id}/jobs?per_page=100", token
        ).get("jobs", [])
        for job in jobs:
            state = f"{job['status']}/{job.get('conclusion') or '-'}"
            if seen.get(job["name"]) != state:
                seen[job["name"]] = state
                print(f"  [{_hhmm()}] {job['name']}: {state}")
        run_done = run.get("status") == "completed"
        jobs_done = bool(jobs) and all(job["status"] == "completed" for job in jobs)
        if run_done or jobs_done:
            if jobs_done and not run_done:
                print("  (every job is completed while the run still reads in_progress")
                print("   -- the documented stale-status case; proceeding)")
            return run
        if time.monotonic() - started > timeout_s:
            raise SystemExit(
                f"gave up after {timeout_s // 60} minutes; the run is still going:\n"
                f"  {run.get('html_url')}\n"
                f"Re-run this script with --run-id {run_id} to pick it back up."
            )
        time.sleep(poll)


def _hhmm() -> str:
    return datetime.now().strftime("%H:%M:%S")


# -----------------------------------------------------------------------------
# Download


def download_wheels(
    owner: str, repo: str, token: str, run_id: int, out_dir: Path
) -> list[dict]:
    """Fetch every wheel artifact of a run into ``out_dir``. Returns manifest rows."""
    artifacts = api(
        f"/repos/{owner}/{repo}/actions/runs/{run_id}/artifacts?per_page=100", token
    ).get("artifacts", [])
    wanted = [a for a in artifacts if fnmatch.fnmatch(a["name"], ARTIFACT_GLOB)]
    if not wanted:
        names = ", ".join(a["name"] for a in artifacts) or "(none)"
        raise SystemExit(
            f"run {run_id} uploaded no wheel artifacts. It has: {names}\n"
            "A build that failed still uploads its logs, so check the run first."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for artifact in sorted(wanted, key=lambda a: a["name"]):
        if artifact.get("expired"):
            print(f"  {artifact['name']}: EXPIRED, skipping")
            continue
        match = ARTIFACT_RE.match(artifact["name"])
        platform = match.group("platform") if match else "unknown"
        python = match.group("python") if match else "unknown"
        print(f"  {artifact['name']} ({artifact['size_in_bytes'] / 1048576:.1f} MiB)")
        blob = api(
            f"/repos/{owner}/{repo}/actions/artifacts/{artifact['id']}/zip",
            token,
            raw=True,
        )
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            for member in archive.namelist():
                if not member.endswith(".whl"):
                    continue
                target = out_dir / Path(member).name
                with archive.open(member) as source, open(target, "wb") as sink:
                    shutil.copyfileobj(source, sink)
                digest = _sha256(target)
                rows.append(
                    {
                        "file": target.name,
                        "platform": platform,
                        "python": python,
                        "version": target.name.split("-")[1]
                        if "-" in target.name
                        else "",
                        "bytes": target.stat().st_size,
                        "sha256": digest,
                    }
                )
                print(f"    -> {_display(target)}  {digest[:16]}...")
    return rows


def _display(path: Path) -> str:
    """Repo-relative when it can be, absolute when it cannot.

    ``--out`` is free-form, so it is routinely somewhere else entirely (a
    scratch directory, another disk); `Path.relative_to` raises on those rather
    than falling back, which turned a successful download into a traceback.
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def patch_digests(head_sha: str | None) -> dict:
    """sha256 of each patch, taken from the commit the run built where possible.

    A wheel is only as identifiable as the patches that went into it, and the
    working tree here is not necessarily what the runner checked out. So this
    hashes the blobs at the run's own commit when that commit is available
    locally, and says so when it has had to fall back to the working tree.
    """
    names = sorted(p.name for p in PATCH_DIR.glob("[0-9]*.patch"))
    if head_sha:
        try:
            rows = []
            for name in names:
                blob = subprocess.run(
                    [
                        "git",
                        "-C",
                        str(REPO_ROOT),
                        "show",
                        f"{head_sha}:quadrants_patches/{name}",
                    ],
                    capture_output=True,
                    check=True,
                )
                rows.append(
                    {"name": name, "sha256": hashlib.sha256(blob.stdout).hexdigest()}
                )
            return {"patches_from": f"commit {head_sha[:12]}", "patches": rows}
        except subprocess.CalledProcessError:
            pass
    return {
        "patches_from": "working tree (the run's commit is not available locally)",
        "patches": [
            {"name": name, "sha256": _sha256(PATCH_DIR / name)} for name in names
        ],
    }


def write_manifest(out_dir: Path, run: dict, rows: list[dict]) -> Path:
    manifest = {
        "_comment": (
            "Written by scripts/build_quadrants_wheels.py. The wheels beside this "
            "file are gitignored; this manifest is committed so that which wheel a "
            "measurement used stays recoverable."
        ),
        "workflow": WORKFLOW_FILE,
        "run_id": run.get("id"),
        "run_url": run.get("html_url"),
        "run_name": run.get("display_title"),
        "conclusion": run.get("conclusion"),
        "ref": run.get("head_branch"),
        "head_sha": run.get("head_sha"),
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **patch_digests(run.get("head_sha")),
        "wheels": rows,
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


# -----------------------------------------------------------------------------
# Install


def install_matching(out_dir: Path, rows: list[dict]) -> None:
    """Install the one downloaded wheel this interpreter can actually use."""
    key = {"linux": "linux", "darwin": "macos", "win32": "windows"}.get(sys.platform)
    if key is None:
        raise SystemExit(f"--install does not know what to do on {sys.platform}")
    want_python = f"cp{sys.version_info.major}{sys.version_info.minor}"
    want_tag = PLATFORMS[key]["wheel_tag"]

    candidates = [
        row for row in rows if want_python in row["file"] and want_tag in row["file"]
    ]
    if not candidates:
        have = ", ".join(row["file"] for row in rows) or "(nothing)"
        raise SystemExit(
            f"no downloaded wheel matches {want_python}/{want_tag}. Downloaded: {have}\n"
            f"Build one with --platforms {key} --python "
            f"{sys.version_info.major}.{sys.version_info.minor}"
        )
    wheel = out_dir / candidates[0]["file"]
    # `--reinstall-package` because the patched wheel usually carries the same
    # *name* as the one uv already resolved from PyPI; without it the installer
    # can consider the requirement already satisfied and do nothing at all.
    command = [
        "uv",
        "pip",
        "install",
        "--reinstall-package",
        "quadrants",
        str(wheel),
    ]
    print(f"\n$ {' '.join(command)}")
    if subprocess.run(command).returncode != 0:
        raise SystemExit("install failed")
    print(f"installed {wheel.name}")


# -----------------------------------------------------------------------------


def list_runs(owner: str, repo: str, token: str, limit: int) -> int:
    runs = api(
        f"/repos/{owner}/{repo}/actions/workflows/{WORKFLOW_FILE}/runs?per_page={limit}",
        token,
    ).get("workflow_runs", [])
    if not runs:
        print(f"no runs of {WORKFLOW_FILE} yet")
        return 0
    print(f"{'run id':>12}  {'status':<12} {'conclusion':<10} {'branch':<28} name")
    for run in runs:
        print(
            f"{run['id']:>12}  {run['status']:<12} {str(run.get('conclusion')):<10} "
            f"{(run.get('head_branch') or '')[:28]:<28} {run.get('display_title', '')}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--platforms",
        default=",".join(PLATFORMS),
        help=f"comma separated, any of {','.join(PLATFORMS)} (default: all of them)",
    )
    parser.add_argument(
        "--python",
        dest="pythons",
        default="3.11",
        help=f"comma separated, any of {','.join(PYTHONS)} (default: 3.11)",
    )
    parser.add_argument(
        "--ref",
        default=None,
        help="branch or tag to build (default: the current branch)",
    )
    parser.add_argument(
        "--stock",
        action="store_true",
        help="build unpatched v1.3.0 -- the control arm, for when a build breaks "
        "and the question is whether the patches are at fault",
    )
    parser.add_argument(
        "--run-id",
        type=int,
        default=None,
        help="skip the dispatch and download this existing run instead",
    )
    parser.add_argument("--list", action="store_true", help="list recent runs and exit")
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="dispatch, print the run url, and exit without waiting",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT, help="where wheels land"
    )
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--timeout-minutes", type=int, default=240)
    parser.add_argument(
        "--install",
        action="store_true",
        help="after downloading, install the wheel matching this interpreter",
    )
    args = parser.parse_args(argv)

    owner, repo = detect_repo()
    token = resolve_token()

    if args.list:
        return list_runs(owner, repo, token, 20)

    if args.run_id is None:
        # Validate before spending a dispatch: the workflow would reject these
        # too, but only after a runner has been allocated.
        bad = [
            p
            for p in args.platforms.split(",")
            if p.strip() and p.strip() not in PLATFORMS
        ]
        if bad:
            raise SystemExit(f"unknown platform(s) {bad}; known: {sorted(PLATFORMS)}")
        bad = [
            v for v in args.pythons.split(",") if v.strip() and v.strip() not in PYTHONS
        ]
        if bad:
            raise SystemExit(
                f"unsupported Python version(s) {bad}; known: {list(PYTHONS)}"
            )

        ref = args.ref or current_branch()
        print(
            f"dispatching {WORKFLOW_FILE} on {owner}/{repo}@{ref}\n"
            f"  platforms: {args.platforms}\n"
            f"  pythons:   {args.pythons}\n"
            f"  patched:   {not args.stock}"
        )
        tag = dispatch(
            owner,
            repo,
            token,
            ref=ref,
            platforms=args.platforms,
            pythons=args.pythons,
            patched=not args.stock,
        )
        print(f"  run tag:   {tag}")

        run = None
        for _ in range(30):
            time.sleep(4)
            run = find_run(owner, repo, token, tag=tag, ref=ref)
            if run is not None:
                break
        if run is None:
            raise SystemExit(
                "dispatched, but no run carrying the tag appeared within ~2 minutes.\n"
                "`--list` will show what the workflow actually started."
            )
        run_id = run["id"]
        print(f"  run:       {run['html_url']}")
        if args.no_wait:
            print(f"\nnot waiting. Come back with:\n  --run-id {run_id}")
            return 0
    else:
        run_id = args.run_id
        run = api(f"/repos/{owner}/{repo}/actions/runs/{run_id}", token)
        print(f"using existing run {run_id}: {run['html_url']}")

    if run.get("status") != "completed":
        print("\nwaiting (each job reports as it settles):")
        run = wait_for_run(
            owner,
            repo,
            token,
            run_id,
            poll=args.poll_seconds,
            timeout_s=args.timeout_minutes * 60,
        )

    conclusion = run.get("conclusion")
    print(f"\nrun finished: {conclusion}")
    # Not fatal: `fail-fast: false` plus per-platform `if: always()` uploads mean
    # a run where one leg failed still carries usable wheels from the others.
    if conclusion != "success":
        print("some jobs did not succeed -- downloading whatever wheels exist anyway.")

    print(f"\ndownloading into {_display(args.out)}/:")
    rows = download_wheels(owner, repo, token, run_id, args.out)
    manifest = write_manifest(args.out, run, rows)
    print(f"\n{len(rows)} wheel(s); manifest at {_display(manifest)}")

    if args.install:
        install_matching(args.out, rows)
    return 0 if conclusion == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
