# Releasing Algan to PyPI

The operational companion to `RELEASE_AUDIT.md`. That file is the record of
*what was wrong* before the first public release; this one is the ordered list
of what to *do*, and what is currently in the way.

`.github/workflows/release.yaml` already automates the release itself. Almost
everything below is either a one-time account setting the workflow cannot make
for itself, or a precondition its gate job checks and currently fails.

State as of 2026-09-09, verified against the live repository and PyPI.

---

## What is already done

| Thing | State |
| --- | --- |
| `algan` name on PyPI | **Free** — `https://pypi.org/pypi/algan/json` is 404 |
| `algan-quadrants` on PyPI | Published: `1.3.0.post1` (12 wheels), `1.3.0.post2` (16 wheels) |
| `pyproject.toml` / `uv.lock` | Already consume `algan-quadrants==1.3.0.post2`, with real PyPI hashes |
| `pypi` GitHub environment | Exists and has published before (`algan-quadrants` post2, run 34064942236) |
| Release workflow | `.github/workflows/release.yaml`, with `dry_run` and `docs_only` rehearsal modes |
| Build + metadata | `uv build`, `twine check` and `scripts/gate/verify_license_notices.py` all pass locally |
| Audit §15 hygiene | `recipe/` deleted, README badges real, `LICENSE` year `2025-2026` |

## Do the quadrants wheels need rebuilding?

**No. Reuse `algan-quadrants==1.3.0.post2`.**

It is published as 16 wheels — 4 platforms x Python 3.10/3.11/3.12/3.13 — which
exactly covers `requires-python = ">=3.10,<3.14"`, and `uv.lock` already pins it
with the hashes PyPI serves. Nothing about releasing Algan requires touching it.

The build side of `quadrants_build.yaml` is staged for `1.3.0.post3` (the
portable aarch64 LLVM that lowers the ARM Linux floor from glibc 2.35 to 2.34).
Leave it staged. Publishing post3 now would pull in the whole two-stage
consumer-side dance from `quadrants_patches/PYPI.md` — publish, then bump
`pyproject.toml`, then regenerate `uv.lock`, then re-green CI — for a fix that
is not on the critical path for a first release. Ship post2, and let post3 land
in 0.2.3.

## The version is `0.0.0`

`pyproject.toml` declares `0.0.0`, and the release tags `v0.0.0`. The internal
`0.2.x` numbering was never published, so the public history starts at zero.

Two consequences worth knowing rather than discovering:

- **`0.0.0` is the permanent floor.** PyPI versions cannot be deleted or reused,
  and nothing can ever be published below it. Every future release is an
  upgrade, which is what you want here, but there is no room underneath.
- **`>=` ranges behave.** `0.0.0` is a valid PEP 440 version and sorts below
  everything, so a user pinning `algan>=0.1` simply will not match this release.

Nothing else needs editing. `algan.__version__` and `algan --version` both read
package metadata lazily, so `pyproject.toml` is the single source of truth, and
no test hardcodes a version. The gate compares the declared version against the
workflow's `version` input exactly — dispatch with `0.0.0`, and note that
`v0.0.0` is free (the only legacy tag is `BETA_v0.0.63`).

You do not need to touch `stable` or the tag by hand. `promote` fast-forwards
`stable` to the released commit and tags `v0.0.0` on it, in that order, in the
same run — no ruleset covers tags, and `stable`'s blocking rules are currently
switched off (§1b).

---

## Blockers

All cleared for this release — though §1b is cleared by switching protection
off rather than by fixing it. In the order the release hits them, plus one step
that is configured but has never run.

### 1. `stable` and `master` had no common ancestor — *resolved*

The `promote` job requires a fast-forward:

```
git merge-base --is-ancestor origin/stable ${{ github.sha }}
```

This failed, and not because `stable` was merely behind. The two branches had
**no merge base at all** — `git merge-base origin/stable origin/master` exited 1
with no output. `stable` was 133 commits of a disjoint history ending at
`2a0555a`, the commit tagged `BETA_v0.0.63`; `master` is 406 commits from an
unrelated root. Master's history was evidently rewritten at some point after
`stable` was cut, and nothing had reconciled them since.

`RELEASE_AUDIT.md` §15 recorded this as "`stable` is 133 commits behind
`master`". That reading is too generous: no merge will ever fast-forward it.

**Done, 2026-09-09.** `stable` was reset to `master` (`1be534d`):

```bash
git push --force-with-lease origin origin/master:refs/heads/stable
```

`stable` is now an ancestor of every future release commit, so the `promote`
job's fast-forward holds normally from here on. The `BETA_v0.0.63` tag keeps the
old history reachable, so nothing is lost.

### 1b. `stable` is protected, and the release bot is not you — *unblocked, temporarily*

> **Current state (2026-09-09): the blocking rules are switched off.** Only
> `deletion` (from "Default Dev Ruleset") still applies to `stable`, so
> `promote`'s push will go through as designed. This was done to get the first
> release out on a single-maintainer repository.
>
> **It is meant to be restored.** Until it is, `stable` has no required checks,
> no PR requirement and no force-push protection — the branch that is supposed
> to mean "the latest released version" is the least protected in the
> repository. The rest of this section is the record of what was there and how
> to bring it back without re-blocking the release; the deploy-key route below
> is the version that lets both hold at once.

The reset earlier printed this, and still went through:

```
remote: - Required status check "ubuntu-latest / Python 3.10" is in progress.
remote: - Cannot update this protected ref.
```

It succeeded because the push carried *your* credentials and your account can
bypass the rule. The `promote` job pushes as `GITHUB_TOKEN`, which by default
cannot — and it runs after `gate`, `build` and `docs`, so a real release would
spend up to three hours before hitting it, with the tag unpushed and the PyPI
upload never reached.

**What is actually configured.** Ruleset **"Stable Rules"** (id `20899556`),
`enforcement: active`, targeting `refs/heads/stable`, created 2026-08-16 and
last updated 2026-08-28 — so it was in force during the reset above, which went
through only because that push carried the repository owner's credentials.
Its **`bypass_actors` list is empty**.

Six rules apply, and the status checks are the *least* of them:

| Rule | Effect on `promote`'s `git push origin <sha>:refs/heads/stable` |
| --- | --- |
| `update` | Restricts updates to bypass actors only — there are none |
| `pull_request` | Requires changes to arrive via a PR; this is a direct push |
| `required_status_checks` | The five contexts below must be green |
| `non_fast_forward` | Blocks force pushes only — `promote` fast-forwards, so this is fine |
| `deletion`, `creation` | Not relevant to a fast-forward |

The five required contexts (all from the GitHub Actions app, integration
`15368`):

```
Sphinx Build                    (docs.yaml)
Ruff Format                     (code_quality.yaml)
lock_file                       (code_quality.yaml)
macos-latest / Python 3.10      (test.yaml)
ubuntu-latest / Python 3.10     (test.yaml)
```

Worth noticing that `ubuntu-latest / Python 3.13` and
`macos-latest / Python 3.10 / render=mps` are **not** required — the gate on
`stable` is narrower than the matrix `test.yaml` actually runs, and the MPS leg
in particular is the one that was red all morning.

**So green checks alone will not unblock it.** Even with all five passing, the
`update` and `pull_request` rules reject a direct push from Actions.

**Only the branch push is blocked.** Both rulesets are `target: branch`, and
`Stable Rules` matches `refs/heads/stable` alone — so `promote`'s *other* push,
the `v0.0.0` tag, is unaffected. The problem is exactly one line of the job.

**The clean fix is not available on this repository.** Adding the GitHub Actions
app as a bypass actor is the right answer in principle, but the bypass list on a
user-owned repository offers only *Deploy keys*, *Maintain* and *Write* — no
apps. Repository-role bypass does not help: `GITHUB_TOKEN` acts as the GitHub
Actions app, not as a collaborator holding a role, so a Write/Maintain entry
never matches it.

It costs one command to find out whether the API accepts what the UI does not
offer:

```bash
gh api -X PUT repos/algorithmicsimplicity/algan/rulesets/20899556 \
  -H "Accept: application/vnd.github+json" --input - <<'JSON'
{"bypass_actors":[{"actor_id":15368,"actor_type":"Integration","bypass_mode":"always"}]}
JSON
```

`15368` is the GitHub Actions app — the integration that reports all five
required checks. `bypass_mode` must be `always`; `pull_request` mode only
applies inside a PR. If it returns 422 the UI was telling the truth, and
nothing is changed by a failed call.

If it *does* work, know what it grants: a bypass actor bypasses the **whole**
ruleset, not just the rule in the way, for **every** workflow running with
`contents: write` — not only `release.yaml`.

### The workaround that needs no settings change

Advance `stable` yourself, as the owner, *before* dispatching the release. Then
`promote`'s push has nothing to do:

```bash
git push origin master:stable      # after the merge, before dispatching
```

`promote` runs

```bash
git merge-base --is-ancestor origin/stable ${{ github.sha }}   # true: equal
git push origin ${{ github.sha }}:refs/heads/stable            # no-op
```

A commit is its own ancestor, so the guard passes, and a push that updates
nothing reports `Everything up-to-date` and exits 0 **without a ref update, so
no rule is evaluated**. That is not a guess: the second `git push` during the
2026-09-09 reset did exactly this — `Everything up-to-date`, `rc=0`, and none of
the protected-ref output the first one produced.

The cost is that `stable` moves a few minutes before the release rather than
during it. If the release then fails, `stable` is ahead of the last published
version until the next one — an inconsistency to be aware of, not a hazard.

### The durable fix, when there is time

Give `promote` a **deploy key** with write access and add *Deploy keys* to the
bypass list — that option *is* offered on this repository. It needs an SSH key
in secrets and a push over SSH rather than `GITHUB_TOKEN`, so it is a workflow
change, not a settings change; not one to make on release day, but it is the
version of this that keeps the release a single unattended run.

None of this can be done from a Claude Code session: the agent proxy refuses
writes to the GitHub API (`403 Write access to this GitHub API path is not
permitted through this proxy`) and no MCP tool covers rulesets. Reads work,
which is how the rules above were established rather than guessed.

The alternative is to stop having `promote` push at all and advance `stable`
through the `master → stable` PR that `development.rst` describes — but that is
a workflow change, not a settings change, and not one to make on release day.

The dry run **cannot** catch any of this: `promote` is skipped under `dry_run`,
so the first time that push is attempted for real is the real release.

Worth deciding at the same time whether the `master → stable` PR flow
`test.yaml` assumes is something you still want. The reset makes that flow
possible rather than replacing it.

### 2. The baseline pointer does not match what was uploaded — *resolved*

**Done, 2026-09-09.** `baselines-2026-09-09.2` was published with all five
assets correctly named, and every one verifies by content:
`scripts/gate/verify_baseline_pointer.py` passes against the committed pointer
(31.2 MB, 5/5). The release gate now runs that script instead of a HEAD check.
The account of what went wrong is kept below, because the failure shape is the
reason the gate downloads.

The gate job used to HEAD-check every archive in `tests/baselines.json`. Two of
the five were wrong under tag `baselines-2026-09-09.1`, because two assets had
been uploaded with a **hyphen missing from the filename**:

| Pointer expects | sha256 | Asset actually on the release |
| --- | --- | --- |
| `full_renders-cpu.tar.gz` | `63426f88…` | ok |
| `full_renders-cuda.tar.gz` | `3103f3a8…` | ok |
| `path_traced-cuda.tar.gz` | `ef80f96c…` | ok |
| `full_renders-cpu_eager.tar.gz` | `cf0516e9…` | **`full_renderscpu_eager.tar.gz`** — 404s |
| `path_traced-cpu.tar.gz` | `7b3282b9…` | **`path_tracedcpu.tar.gz`** — and a *different* file occupies the expected name |

The second row of that pair is the nastier one. `path_traced-cpu.tar.gz` does
exist, so the gate's HEAD request returns 200 — but it is a different archive
(`adcb2818…`, 21577 bytes) than the pointer's `7b3282b9…` (21523 bytes). The
gate passes; `tests/baseline_store.py` then rejects it on the sha256 and the
suite **skips**. That is exactly the silent-stop-testing failure the pointer
mechanism exists to prevent.

The archive occupying the expected `path_traced-cpu.tar.gz` name is not junk —
it is the *superseded* baseline. `7cbd375` ("Rebaseline the CPU render suites
and pin baselines-2026-09-09.1") rewrote that entry from `adcb2818…` to
`7b3282b9…`, so `adcb2818…` is what the pointer used to expect under tag
`baselines-2026-09-09`. Both ended up on the `.1` release, and the newer one is
the one wearing the mangled name.

**The fix, as carried out — five assets under `baselines-2026-09-09.2`**, with
`tests/baselines.json` pointing at that tag. Three files carried over unchanged;
two were renamed:

| Upload as | sha256 | Take from `.1`'s asset |
| --- | --- | --- |
| `full_renders-cpu.tar.gz` | `63426f88…` | `full_renders-cpu.tar.gz` — unchanged |
| `full_renders-cuda.tar.gz` | `3103f3a8…` | `full_renders-cuda.tar.gz` — unchanged |
| `path_traced-cuda.tar.gz` | `ef80f96c…` | `path_traced-cuda.tar.gz` — unchanged |
| `full_renders-cpu_eager.tar.gz` | `cf0516e9…` | **`full_renderscpu_eager.tar.gz`** — rename |
| `path_traced-cpu.tar.gz` | `7b3282b9…` | **`path_tracedcpu.tar.gz`** — rename |

Do **not** carry `adcb2818…` forward under any name; it is the superseded
`path_traced/cpu` and nothing references it any more.

**Verifying it** — this is now `scripts/gate/verify_baseline_pointer.py`, which
the gate runs, and it reports 5/5 against the committed pointer. The equivalent
by hand:

```bash
python - <<'PY'
import json, hashlib, pathlib, urllib.request
p = json.loads(pathlib.Path("tests/baselines.json").read_text())
for key, e in p["archives"].items():
    url = f"{p['base_url']}/{p['tag']}/{e['file']}"
    got = hashlib.sha256(urllib.request.urlopen(url, timeout=120).read()).hexdigest()
    print(("ok  " if got == e["sha256"] else "BAD "), key, e["file"], got[:12])
PY
```

Pointed at the old `.1` tag the gate reports both failure shapes and lists what
the tag actually carries, so the two mangled names are visible side by side with
the ones the pointer expects. `tests/unit_tests/test_verify_baseline_pointer.py`
pins both shapes offline over `file://`.

### 3. Green CI on the exact release commit

The gate requires *this commit* to have its own green `Test` run — the §17
lesson, that sitting after a green run is not the same as being green.

The last five completed `Test` runs on `master` were red. All of them failed the
same single test, on the two macOS legs only, with both Ubuntu legs green:

```
tests/unit_tests/test_baseline_store.py::test_an_unbaselined_macos_key_names_the_opt_out
```

CI exports `ALGAN_ALLOW_UNBASELINED_MACOS=1` for the whole macOS job, which
suppresses the very sentence the test asserts on. `9c420ef` fixes it by clearing
the variable in the file's autouse fixture, and is already on `master`.

Run [34317795058](https://github.com/algorithmicsimplicity/algan/actions/runs/34317795058)
is in flight on `1be534d` and carries that fix. **Confirm it goes green before
releasing.** If it does, `1be534d` is a releasable commit.

### 4. The docs deploy has never actually run — *not a blocker, but unproven*

**Settings → Pages → Source is already set to "GitHub Actions"** (confirmed
2026-09-09), so `publish_docs` will not fail on the setting. This is no longer a
blocker; it is the one remaining step in the release that has never executed.

What is still true is that the live site is stale:
`https://algorithmicsimplicity.github.io/algan/` serves
`<title>Algan v0.1.0</title>` with zero example videos. Changing the source does
not retract the previous deployment — Pages keeps serving the last thing that
was deployed until something new replaces it. So the stale content is expected,
and the first successful Actions deploy is what clears it.

That URL is in `pyproject.toml`, the README badges and `conf.py`'s
`ogp_site_url`, and PyPI metadata is permanent, so the site wants to be correct
*before* the upload rather than after. Rehearse with `docs_only` (Step 3) rather
than letting the first-ever deploy happen during the real release. The
`github-pages` environment is created by `actions/deploy-pages` on first run;
there is nothing to set up by hand.

---

## The release, in order

### Step 0 — Create the PyPI pending publisher

There is no way to create an empty project on PyPI, and you do not need one. A
**pending publisher** reserves the name and lets the first Trusted Publishing
upload create the project.

PyPI → Account settings → Publishing → *Add a new pending publisher* → GitHub:

| Field | Value |
| --- | --- |
| PyPI Project Name | `algan` |
| Owner | `algorithmicsimplicity` |
| Repository name | `algan` |
| Workflow name | `release.yaml` |
| Environment name | `pypi` |

Every field must match `release.yaml` exactly — the workflow *filename*, not its
`name:`, and the environment its `pypi` job declares.

No API token is involved. This is a second, independent publisher from the one
`algan-quadrants` uses; they share the `pypi` GitHub environment but each names
its own workflow file, so they do not collide.

### Step 1 — Clear the blockers

Done: `stable` reset (§1) · `baselines-2026-09-09.2` published and verified by
content (§2) · PyPI pending publisher created (Step 0).

Left: confirm the `Test` run on the release commit is green (§3). Pages (§4)
needs nothing but the rehearsal in Step 3, and `stable`'s blocking rules are
switched off for the release (§1b) — which is a loan, not a fix.

### Step 2 — Write the release notes

`github_release` calls `gh release create --generate-notes`. Generated notes are
computed against the most recent previous release, which is
`baselines-2026-09-09.1` — a baseline asset drop from earlier today. For a
*first public release* that produces notes covering a few hours of commits, not
the project.

Write `v0.0.0`'s notes by hand, or add a `CHANGELOG.md` (still absent, and
`RELEASE_AUDIT.md` §15 wants one) and point at it. This is the release most
likely to be read by someone who has never seen Algan; it is the wrong one to
let a tag-diff heuristic write.

### Step 3 — Rehearse the docs deploy

Dispatch **Release** from `master` with `docs_only: true`, `dry_run: false`.

Publishes the docs and nothing else — no version gate, no wheel, no tag, no
PyPI. This is what proves the Pages setting from §4 actually took, in the one
place where discovering otherwise is cheap. Confirm the site serves `0.0.0` and
that example videos are present (`find docs/build/html -name '*.mp4' | wc -l` is
reported in the run summary; the current live site has none).

Budget up to three hours — the release docs build renders every `.. algan::`
example on CPU.

### Which ref do I dispatch from?

**`master`, and the release commit has to be on it first.** Releasing directly
from a feature branch does not work here, for three independent reasons — the
first is not a judgement call but a measured fact about this repository:

1. **The `pypi` environment refuses it.** `quadrants_patches/PYPI.md` records
   it from experience: publishing `algan-quadrants` post2 succeeded from
   `master`, while "an earlier attempt from a feature branch was rejected at
   the `pypi` environment gate in two seconds with zero steps and uploaded
   nothing". `release.yaml`'s `pypi` job uses that same environment, so a
   branch dispatch would run gate, build, docs, promote, publish_docs and the
   GitHub release — tagging and deploying for real — and then fail at the one
   job the whole ordering exists to protect.
2. **It would immediately recreate blocker §1.** `promote` pushes the released
   commit to `stable`. From a feature branch that leaves `stable` at a commit
   `master` does not contain, so the next release's
   `merge-base --is-ancestor origin/stable master` fails again — the exact
   state the reset just undid.
3. **The released version would not be on `master`.** "`stable` carries the
   latest released version" stops being true, and the next release PR diffs
   against a fiction.

So the order is: **merge, wait for green, then release.**

```
merge claude/admiring-brown-dy0qk3 -> master
      ↓  (this is the commit that gets released)
wait for Test on the new master HEAD to go green
      ↓  (the gate demands this commit's own green run, not a later one)
dispatch Release from master
```

The merge keeps the fast-forward invariant: `stable` is at `1be534d`, and any
merge of this branch descends from it, so `promote` still fast-forwards.

Two things that are easy to get wrong here:

- **The green run must be on the merge commit itself**, not on the branch tip
  before it and not on an earlier master commit. That is §17's whole lesson,
  and the gate enforces it.
- **Nothing in the release edits the tree it releases.** The `0.0.0` bump and
  everything else must already be merged; the workflow will not do it for you.

### Step 4 — Full dry run

Dispatch **Release** from `master` with:

| Input | Value |
| --- | --- |
| version | `0.0.0` — a text field, not a checkbox, and the gate hard-fails if it is blank |
| dry_run | **checked** (this is the default) |
| docs_only | unchecked |
| skip_docs_examples | **checked** — see below |

Runs `gate`, `build` and `docs`, publishes nothing. This is where anything
missed gets caught. The wheel, the sdist and the built HTML land as downloadable
artifacts — pull the wheel and install it into a clean venv on a machine that is
not this one.

**Check `skip_docs_examples` for this run.** The example renders are the whole
cost of the docs job (27 minutes of a 27-minute build), the deploy was already
proven end to end by Step 3, and a dry run publishes nothing — so rendering them
again validates nothing new. Skipping turns a ~30-minute rehearsal into a few
minutes, and what you actually want from it is the gate and the artifacts:

- `verify_baseline_pointer.py`, which has never run in CI,
- `twine check` and the vendored-license gate on the real distributions,
- the version/tag/green-run checks against the actual release commit.

Leave it unchecked only if the docs source changed since Step 3.

### Step 5 — Release

Dispatch **Release** from `master` with `version: 0.0.0` and **all three
checkboxes unchecked**. Note that `dry_run` defaults to *checked*, so this is
the one that has to be actively cleared; `skip_docs_examples` must go back to
unchecked too, or you publish a videoless site.

The jobs run in order of how hard each is to undo:

```
gate → build → docs → promote (ff stable, push v0.0.0)
                    → publish_docs
                    → github_release
                    → pypi
```

PyPI is last on purpose. A tag can be moved, a docs deploy re-run, a GitHub
release edited; a version on PyPI can be yanked but never replaced. If anything
is going to fail, it fails before that.

### If the release fails partway

It did, on the first `v0.0.0` attempt (2026-09-09). `github_release` ran
`gh release create` in a job that deliberately has no `actions/checkout`, so
`gh` had no git remote to infer the repository from and exited with
`failed to run git: fatal: not a git repository`. Fixed by giving that step
`GH_REPO: ${{ github.repository }}`.

The ordering did its job. What had already happened:

| | |
| --- | --- |
| `promote` | ✅ `stable` fast-forwarded, tag `v0.0.0` pushed |
| `publish_docs` | ✅ live docs serving the new version |
| `github_release` | ❌ failed |
| `pypi` | ⏭️ skipped — it `needs: github_release` |
| PyPI | **nothing uploaded** |

So the one step that cannot be undone was never reached, and everything that
*had* happened was reversible. That is the whole reason for the job order.

**The workflow has no resume mode**, and two things make a naive re-run fail:

- **`gh release create` is not idempotent.** A second run against an existing
  release errors.
- **The gate's "tag is free" check now fails**, because `promote` pushed the
  tag before the failure. Worse, once the fix is merged the tag points at the
  *previous* commit, so it is not even the same commit being re-released.

The clean recovery is to **delete the tag and re-run from the top**:

```bash
git push --delete origin v0.0.0     # only while nothing consumes it
```

That is safe exactly as long as the tag is young and unreferenced — no GitHub
release points at it, nothing is on PyPI, and no one has fetched it. Check all
three before deleting; past that point the tag is a promise and the recovery is
a new version number instead.

`promote` is otherwise re-run-safe: a `stable` push that changes nothing reports
`Everything up-to-date`, and a fresh tag on the new commit is a normal tagging.
`publish_docs` simply redeploys.

### Step 6 — Verify

```bash
pip install algan          # in a clean venv, on a machine that never built it
algan check
```

Then confirm the README's PyPI badges resolve (they 404 until the first upload)
and that the docs site serves the new build.

---

## Known limitations to state publicly

Not blockers — things a first release should be honest about, because each one
turns into an issue report otherwise.

- **No sdist for `algan-quadrants`.** 16 wheels and nothing else, so
  `pip install algan` fails outright on any platform outside that set: macOS
  Intel (already documented at `docs/source/installation.rst:168`), musl Linux,
  Windows on ARM, and Python 3.9 or 3.14.
- **ARM Linux needs glibc ≥ 2.35.** post2's aarch64 wheel is tagged
  `manylinux_2_35_aarch64`. post3 lowers this to 2.34; until then, ARM users on
  older distributions cannot install.
- **No GPU leg in CI** (`RELEASE_AUDIT.md` §17). The 8 CUDA-only tests run
  nowhere automatically, and `tests/full_renders` / `tests/path_traced` are not
  collected in CI at all — six dense pixel-compared scenes and the path tracer
  are guarded only by whoever runs the full suite by hand.
- **Repository weight** (`RELEASE_AUDIT.md` §13, still open). `.git` is 210 MB,
  mostly historical render baselines. Hosting them as release assets stopped the
  growth; it did not shrink what is already there. A history rewrite gets
  harder the moment the repository is widely cloned — which is what this release
  causes. Decide before the announcement, not after.
