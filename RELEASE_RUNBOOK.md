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

You do not need to tag `stable` by hand. The `promote` job fast-forwards
`stable` to the released commit and tags `v0.0.0` on it, in that order, in the
same run.

---

## Blockers

Three things will fail, in the order the release hits them, plus one step that
is configured but has never run.

### 1. `stable` and `master` have no common ancestor

The `promote` job requires a fast-forward:

```
git merge-base --is-ancestor origin/stable ${{ github.sha }}
```

This fails today, and not because `stable` is merely behind. The two branches
have **no merge base at all** — `git merge-base origin/stable origin/master`
exits 1 with no output. `stable` is 133 commits of a disjoint history ending at
`2a0555a`, the commit tagged `BETA_v0.0.63`; `master` is 406 commits from an
unrelated root. Master's history was evidently rewritten at some point after
`stable` was cut, and nothing has reconciled them since.

`RELEASE_AUDIT.md` §15 recorded this as "`stable` is 133 commits behind
`master`". That reading is too generous: no merge will ever fast-forward it.

**Fix — a one-time reset, and it needs your explicit go-ahead** because it
discards a published branch's history:

```bash
git push --force-with-lease origin origin/master:refs/heads/stable
```

After that `stable` is an ancestor of every future release commit and the
`promote` job's fast-forward holds normally, forever. The `BETA_v0.0.63` tag
keeps the old history reachable, so nothing is actually lost.

Check first whether any branch protection covers `stable`, and whether the
`master → stable` PR flow `test.yaml` assumes is something you want to keep —
if it is, this reset is the act that makes it possible rather than replacing it.

### 2. The baseline pointer does not match what was uploaded

The gate job HEAD-checks every archive in `tests/baselines.json`. Two of the
five are wrong under tag `baselines-2026-09-09.1`, because two assets were
uploaded with a **hyphen missing from the filename**:

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

**Fix — five assets under `baselines-2026-09-09.2`.** `tests/baselines.json` is
already updated to that tag on this branch. Three files carry over unchanged;
two need renaming:

| Upload as | sha256 | Take from `.1`'s asset |
| --- | --- | --- |
| `full_renders-cpu.tar.gz` | `63426f88…` | `full_renders-cpu.tar.gz` — unchanged |
| `full_renders-cuda.tar.gz` | `3103f3a8…` | `full_renders-cuda.tar.gz` — unchanged |
| `path_traced-cuda.tar.gz` | `ef80f96c…` | `path_traced-cuda.tar.gz` — unchanged |
| `full_renders-cpu_eager.tar.gz` | `cf0516e9…` | **`full_renderscpu_eager.tar.gz`** — rename |
| `path_traced-cpu.tar.gz` | `7b3282b9…` | **`path_tracedcpu.tar.gz`** — rename |

Do **not** carry `adcb2818…` forward under any name; it is the superseded
`path_traced/cpu` and nothing references it any more.

**Verify after uploading**, because a HEAD check would not have caught the
wrong-content case that created this mess:

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

**Also worth doing:** the gate job only issues a HEAD request, which is exactly
what let the wrong-content asset through. Folding the check above into it costs
about 20 MB and a few seconds in the cheapest job of the release.

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

Reset `stable` (§1, needs your go-ahead) · publish `baselines-2026-09-09.2` with
the five correctly named assets (§2 — the pointer is already updated) · confirm
run 34317795058 is green (§3). Pages (§4) needs nothing but the rehearsal in
Step 3.

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

Dispatch **Release** with `docs_only: true`, `dry_run: false`.

Publishes the docs and nothing else — no version gate, no wheel, no tag, no
PyPI. This is what proves the Pages setting from §4 actually took, in the one
place where discovering otherwise is cheap. Confirm the site serves `0.0.0` and
that example videos are present (`find docs/build/html -name '*.mp4' | wc -l` is
reported in the run summary; the current live site has none).

Budget up to three hours — the release docs build renders every `.. algan::`
example on CPU.

### Step 4 — Full dry run

Dispatch **Release** with `version: 0.0.0`, `dry_run: true`.

Runs `gate`, `build` and `docs`, publishes nothing. This is where §1–§3 get
caught if you have missed one. The wheel, the sdist and the built HTML land as
downloadable artifacts — pull the wheel and install it into a clean venv on a
machine that is not this one.

### Step 5 — Release

Dispatch **Release** with `version: 0.0.0`, `dry_run: false`.

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
