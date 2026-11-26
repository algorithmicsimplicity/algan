# Releasing Algan

Operational instructions for [`.github/workflows/release.yaml`](.github/workflows/release.yaml),
checked against source on 2026-09-09; §1, §3 and §5 re-checked on 2026-11-26. [RELEASE_AUDIT.md](docs/reports/RELEASE_AUDIT.md) is the
historical pre-release audit, not a list of current blockers.

A source checkout cannot establish whether a PyPI name or version is available,
whether Pages is configured, or whether the release identity can update a
protected branch. Verify those external prerequisites for each release; do not
reuse a previous run's account-status notes as proof.

## 1. Choose and validate the release commit

Merge the intended changes before dispatching the workflow. Record the exact
commit SHA and use the workflow run's `head_sha` to confirm what it will release.
A green run on an earlier commit is not sufficient.

`pyproject.toml` is the source of the package version. At this document's source
check it declares `0.0.3`; that is a snapshot, not a standing instruction to
publish that version. Choose an unused version, update the metadata and lockfile
as needed in a PR, and let that exact commit complete CI. The workflow does not
edit package versions or regenerate the lockfile for you.

The release gate checks that:

- the `version` input exactly matches `project.version`, without a leading `v`;
- `v<version>` is not already a Git tag;
- a run that publishes (anything but `dry_run`) was dispatched from `master`;
- the selected commit has at least one run of the Test workflow (`test.yaml`),
  and every Test run on it, whatever event started it, has completed with
  `success`;
- every archive named by `tests/baselines.json` downloads and matches its recorded
  SHA-256, via `scripts/gate/verify_baseline_pointer.py`.

The Test gate is the cheap check, not the release's test run. A push to master
runs only the CI test group on two arms. The release runs the rest itself: its
`test` job calls the Test workflow at the full scope (every unit test and the
fast render, on Linux 3.10 and 3.14 and on macOS CPU and MPS), a rehearsal
runs it too, and `promote` waits for it. Review its skips as well as its
conclusion. The gate reads
only the Test workflow; inspect code-quality and documentation checks as well. See [tests/README.md](tests/README.md) for
portable tests, GPU coverage and separately hosted render baselines.

Before choosing the commit, check that the built-in kernel list is current:
`scripts/generate_kernel_specs.py --check` (a few minutes of cold renders). It is
what a new installation's first render and `algan warmup` precompile in parallel,
so a stale list ships a slower first run rather than a broken one. Regenerate it
without `--check` and commit the result if it reports differences.

## 2. Check publication prerequisites

**Branch and tag permissions.** `promote` uses the workflow's `GITHUB_TOKEN` to
fast-forward `stable` and create `v<version>`. Confirm that the release identity
is allowed to make both changes under the repository's current rules. A
maintainer's successful push does not demonstrate the bot has the same rights.
Do not turn off branch protection or force-update `stable` as a routine release
step.

Fetch the relevant refs in a full clone and check the ancestry before spending a
release build:

```bash
git fetch origin master stable --tags
RELEASE_SHA=$(git rev-parse origin/master)
git merge-base --is-ancestor origin/stable "$RELEASE_SHA"
```

A nonzero result needs investigation. Reconcile the histories through a reviewed
repository-maintenance change; a normal release must remain a fast-forward.
The workflow's ancestry check also runs in `promote`, after the build and docs
jobs, so checking it in advance avoids a late failure.

**Pages.** Configure GitHub Pages to use GitHub Actions and ensure the
`github-pages` environment permits deployment from the selected ref. The
`publish_docs` job consumes the built HTML artifact; it does not commit generated
videos to a documentation branch. Rehearse with `docs_only` before relying on a
new or changed Pages configuration.

**PyPI.** Verify project ownership, the intended version's availability, the
Trusted Publishing configuration for this repository/workflow, and any approvals
required by the `pypi` GitHub environment. Publication uses OpenID Connect, not a
PyPI API token stored in this workflow. Existing credentials for the separate
`algan-quadrants` project do not by themselves prove Algan's publisher is ready.
Treat an uploaded distribution as immutable; plan a new version rather than
trying to replace a published wheel.

## 3. Keep compiler publication separate

The current `pyproject.toml` and `uv.lock` consume
`algan-quadrants==1.3.3.post3`. Its Python import remains `quadrants`.
The compiler build workflow may target a newer version without changing what
Algan installs. Read [quadrants_patches/PYPI.md](quadrants_patches/PYPI.md) before
changing that dependency.

Publishing Algan does not require rebuilding the compiler. A compiler upgrade is
a separate change: publish its wheels, update the dependency and lockfile,
validate the platform matrix, then release the Algan commit that consumes them.
Do not switch a dependency to a version whose required wheels are not available.

## 4. Rehearse without publishing

The workflow is manually dispatched. These are its inputs:

| Input | Default | Meaning |
| --- | --- | --- |
| `version` | Empty | Required for a package release; ignored by `docs_only`. |
| `dry_run` | `true` | Build, test and validate, but do not publish refs, docs or distributions. |
| `docs_only` | `false` | Build docs only; skip package-version, tag, Test and baseline gates and the full test run. |
| `skip_docs_examples` | `false` | Omit rendered examples; useful for a structural check, not a complete example-video build. |

For a release rehearsal, select the intended ref, supply its declared version,
leave `dry_run=true`, and leave `docs_only=false`. Review the `distributions` and
`docs-html` artifacts, not only the job conclusions.

The build job creates the wheel and sdist, runs `twine check`, and verifies the
vendored license notices in the distributions. The docs job installs the locked
dependencies and native tools, then builds Sphinx with warnings treated as errors.
By default it renders embedded example videos. Its rendered and skipped-example
builds use separate doctree caches.

A structural docs build is useful during development:

```bash
# Use .venv/Scripts/python.exe on Windows.
.venv/bin/python docs/make_and_open_docs.py --skip-examples --no-open
```

This does not validate the example renders. Avoid bare `uv run` in a development
environment containing a locally patched compiler; see [AGENTS.md](AGENTS.md).

## 5. Publish

After a successful rehearsal and a fresh prerequisite check, dispatch the same
release commit and version with `dry_run=false`. The dependency graph is:

```text
gate -> test + build + build_macos + docs -> promote
                                       |-> publish_docs
                                       |-> github_release (also needs build and build_macos)
                                       `-> PyPI waits for both publication jobs
```

`test` is the slow one: about 90 minutes an arm on four arms, two of them on
hosted Macs. It runs in a rehearsal as well, so a rehearsal is also the
release's full test run on that commit.

`build_macos` builds the macOS abi3 wheel on a hosted Mac. `promote`,
`github_release` and PyPI all need it, so a macOS runner failure, or the account's
macOS slots being busy, blocks or delays the whole release.

`publish_docs` and `github_release` can run independently after promotion;
this is not a strictly sequential transaction. `promote` first fast-forwards
`stable`, then creates and pushes the annotated version tag. The GitHub release
attaches the wheel and sdist and uses server-generated release notes. Its job
sets `GH_REPO` explicitly because it downloads artifacts without checking out a
Git repository.

Heavy render baselines are **not** repackaged or attached to the package release.
They have their own baseline release tag and a committed manifest. Update and
upload them before the package release when necessary, following
[the baseline instructions](tests/README.md#where-the-heavy-baselines-live).
The gate verifies the bytes at those pointers; it does not create missing assets.

After the run, check that the Git tag and `stable` point to the intended commit,
the GitHub release contains the expected distributions, the deployed docs include
the intended examples, and PyPI serves the intended version. Test installation in
a clean environment using the supported Python/platform combination. The workflow
builds artifacts before upload; successful upload alone is not an installation
test.

## 6. Documentation-only deployment

Use `docs_only=true, dry_run=false` to deploy a documentation correction without
publishing a package or moving `stable`. `version` is ignored. Because this mode
skips the package release gates, select and review the documentation commit
explicitly; do not interpret a green docs-only run as a package-release approval.

For a complete site, leave `skip_docs_examples=false`. A deployment with examples
skipped can remove videos from the published site, even when the HTML build is
green.

## 7. Recover from a partial release

The workflow has a single-release concurrency group and does not cancel an
in-progress release automatically. Nevertheless, a job can fail after another job
has published. Inspect the failed run, its exact SHA and its artifacts before
retrying anything.

Record which steps actually completed: stable promotion, tag creation, Pages,
GitHub release, and PyPI. The jobs are not idempotent as a group: for example, a
new dispatch fails the free-tag gate if the earlier attempt already pushed the
tag. Do not delete or move a public tag, overwrite a distribution, or assume
that rerunning the whole workflow is safe. Resolve the failed step through a
reviewed recovery plan, preserving the connection between source and artifacts.

Two earlier incidents explain checks retained in the workflow:

- A baseline asset returned successfully but contained an obsolete archive. A
  successful HTTP request is insufficient; the gate now checks SHA-256.
- The GitHub-release job had no checkout and could not infer the repository.
  Explicit `GH_REPO` fixes that source-level issue. It does not establish that
  any particular release attempt subsequently completed.

For remaining engineering work rather than release procedure, see [TODO.md](TODO.md).
