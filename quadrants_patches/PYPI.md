# Publishing Algan's patched Quadrants wheels

Algan's patched Quadrants build is published to PyPI under the **distribution**
name `algan-quadrants`. The installed Python package is still named
`quadrants`, so Algan continues to use `import quadrants` and
`algan.taichi_compat` does not change.

The repository consumes `algan-quadrants==1.3.0.post3`, the first revision
with cp314 wheels (see "Python 3.14" below). post3 is twenty wheels; the
recorded post2 publication on 2026-09-06 contained sixteen; post1 contained
twelve. Check actual PyPI availability before changing the consumer pin or
attempting a new publication. The
`post` suffix identifies Algan's patched build of upstream Quadrants v1.3.0
and keeps it distinct from the upstream `quadrants==1.3.0` release.

## One-time PyPI setup

1. Create or reserve the `algan-quadrants` project on PyPI.
2. Configure a PyPI Trusted Publisher for:
   - repository: `algorithmicsimplicity/algan`
   - workflow: `.github/workflows/quadrants_build.yaml`
   - environment: `pypi`
3. Create the `pypi` GitHub environment. Protect it with required reviewers if
   desired; publication is already opt-in at workflow dispatch time.

No PyPI API token is required. The publish job requests GitHub's OIDC token with
`id-token: write` and uses `pypa/gh-action-pypi-publish`.

## Publishing a release

Dispatch **Quadrants wheels (patched)** with all of the following:

- platforms: `linux,linux_arm64,macos,windows` (or `all`)
- Python versions: `3.10,3.11,3.12,3.13,3.14`
- apply patches: enabled
- publish: enabled

The workflow refuses a stock or partial matrix when `publish` is enabled, and
what it counts as complete is `resolve_wheel_matrix.py`'s platform table rather
than a list of names in the YAML — so a platform (or Python) added there is
required here without this document or that gate being edited. Each platform builds the
ordinary `quadrants` wheel with `SETUPTOOLS_SCM_PRETEND_VERSION` set from
the workflow's `ALGAN_QUADRANTS_VERSION` (currently `1.3.0.post3`), so
Quadrants' Python metadata and its native build see the same version. After all
twenty wheels succeed, the publish job rewrites only the distribution metadata
from `quadrants` to `algan-quadrants`, validates the complete matrix, and
uploads it to PyPI. The `quadrants/` import package is not renamed.

Normal diagnostic builds keep `publish` disabled. Their artifacts stay named
`quadrants-...whl`, which preserves `scripts/build_quadrants_wheels.py --install`
and the existing wheel provenance workflow.

## Historical bootstrap order for Algan

This bootstrap has completed; do not repeat the post1 pin change below. It
records why publication must precede consumption. Do **not** change a
dependency to a new compiler distribution/version before the first
release exists: `uv sync --locked` would otherwise become intentionally
unresolvable while the package is being bootstrapped.

The bootstrap order is therefore:

1. Land the publication-capable wheel workflow while Algan still resolves the
   upstream `quadrants` dependency.
2. Publish `algan-quadrants==1.3.0.post1` using the complete matrix above.
3. Change `pyproject.toml` from `quadrants>=1.3.0,<1.4` to
   `algan-quadrants==1.3.0.post1`.
4. Run `uv lock` and commit the real PyPI file hashes in `uv.lock`.
5. Run the normal Algan test/release pipeline.
6. Publish Algan. A user can then install the patched compiler with only:

   ```bash
   pip install algan
   ```

Never hand-edit the lock entry: it must be generated from the files PyPI
actually serves.

## Releasing another downstream revision

For another patch-only revision of upstream v1.3.0, increment the downstream
version (for example `1.3.0.post2`) — but **in two stages, not three files at
once**. The build side and the consumer side cannot move together, because the
consumer side cannot be locked against a release that does not exist yet:

1. **Before publishing**, bump the two files that decide what gets built and
   what it is branded as. They are one contract — the build stamps
   `SETUPTOOLS_SCM_PRETEND_VERSION` from the first and `rebrand_wheel` refuses
   any wheel that is not the second — and
   `test_the_workflow_and_the_rebrand_script_agree_on_the_version` holds them
   to it:
   - `.github/workflows/quadrants_build.yaml` (`ALGAN_QUADRANTS_VERSION`)
   - `scripts/rebrand_quadrants_wheel.py` (`DOWNSTREAM_VERSION`)
2. Dispatch the complete matrix with `publish` enabled.
3. **Only then** bump `pyproject.toml` and regenerate `uv.lock` from what PyPI
   actually serves.

Doing step 3 early is the same mistake the bootstrap section above warns
about: `uv.lock` pins the old version with its file hashes, `uv lock --locked`
runs in `code_quality.yaml`, and the session-start hook runs
`uv sync --locked` — so the repository stops resolving and cannot be fixed
until the release lands. Do not overwrite an existing PyPI version either;
PyPI releases are immutable.

**Adding a platform is one of those revisions.** `1.3.0.post1` was published as
twelve wheels and cannot grow four more: the publish step uploads the whole
directory with no `skip-existing`, so re-running it against that version fails
on the twelve files that already exist. The `linux_arm64` wheels therefore
reached users as `1.3.0.post2` — the full sixteen, built and published
together, followed by the `pyproject.toml` bump and a regenerated `uv.lock`.
That is done: [run 34064942236](https://github.com/algorithmicsimplicity/algan/actions/runs/34064942236)
published all sixteen from `master`. An earlier attempt from a feature branch
was rejected at the `pypi` environment gate in two seconds with zero steps
and uploaded nothing — publish from the default branch.

**The Linux wheel filenames changed in post2 too.** Both Linux legs build
inside manylinux containers and are gated by `scripts/gate/verify_wheel_tag.py`
before the stamp goes on. post2 uses `manylinux_2_28_x86_64` and
`manylinux_2_35_aarch64`; the aarch64 2.35 value is a real historical
measurement caused by a single GLOBAL `_dl_find_object@GLIBC_2.35` from the
upstream prebuilt LLVM archive. The next-revision section below describes the
source rebuild that removes it. For
x86-64 users this was a *narrowing on paper and a fix in fact* — the
`1.3.0.post1` x86-64 wheel claims 2.27 and actually needs 2.34, so systems
between those two versions (RHEL 8, Ubuntu 20.04, Debian 11) install it and
fail at `import quadrants`. `post2`'s x86-64 wheels measure 2.27 for real, so
those three now work rather than being told a comfortable lie.

## post3: portable aarch64 LLVM and glibc 2.34

`algan-quadrants==1.3.0.post3` followed the two-stage procedure above:
`.github/workflows/quadrants_build.yaml` and `scripts/rebrand_quadrants_wheel.py`
targeted post3 while `pyproject.toml` and `uv.lock` stayed on the published
post2 release, and the consumer side moved only once post3 was on PyPI.
Diagnostic validation uses `publish=false`; do not publish a revision merely to
test a build path.

Run 34036846316 proved that using GCC 11 only for the Quadrants compile/link
still leaves `_dl_find_object@GLIBC_2.35`, isolating the reference to the
upstream aarch64 LLVM archive. The post3 workflow fixes that binary input rather
than lowering the tag cosmetically:

1. `portable_llvm_arm64` builds LLVM 22.1.0 commit
   `4434dabb69916856b824f68a64b029c67175e532` once per workflow run in
   `quay.io/pypa/manylinux_2_34_aarch64@sha256:effc0e17319a56b2c7eaff0cb5dd81a2d2c2850410841d3f017378f52ead6442`.
2. The LLVM build uses AlmaLinux GCC/G++/libstdc++-static
   `11.5.0-14.el9.alma.1`, CMake 3.31.10 and Ninja 1.13.2.
3. `scripts/gate/build_portable_quadrants_llvm.sh` measures linked ELF files with
   `readelf`, rejects any maximum GLIBC symbol above 2.34, and scans static LLVM
   archives with `llvm-nm` for the unresolved `_dl_find_object` fingerprint. It
   writes a SHA-256 and records the exact provenance.
4. The aarch64 wheel job verifies that SHA-256, extracts the portable install
   into upstream Quadrants v1.3.0's existing
   `llvm-22.1.0-aarch64-202603120808` cache slot, and pins the final link to the
   same GCC 11 generation. x86-64 continues using the upstream prebuilt LLVM.
5. `verify_wheel_tag.py` must measure a floor no newer than
   `manylinux_2_34_aarch64` before the wheel is stamped.
6. A separate fresh, pinned glibc-2.34 aarch64 job downloads the final stamped
   artifact, reruns the ABI gate, checks that `_dl_find_object@GLIBC_2.35` is
   absent, imports `quadrants`, and executes a CPU kernel. Publishing waits for
   this runtime gate.

The portable LLVM ZIP is an Actions artifact and is also cached by a key that
includes its version/commit/toolchain recipe plus the builder-script hash, so a
normal four-Python aarch64 matrix builds LLVM once and reuses it. To regenerate
or update it, change the pinned values in
`.github/workflows/scripts/resolve_wheel_matrix.py`; the builder will fail
loudly if the userspace, compiler generation, commit, checksum, or measured ABI
no longer matches the contract.

## Python 3.14

post3 is also the revision that adds **cp314**. Upstream Quadrants publishes
cp310–cp313 only; `0009-python-3.14.patch` is what makes a cp314 wheel correct
(`README.md`, "0009"), and `resolve_wheel_matrix.py`'s `PYTHONS` is what makes
the publish gate demand one — five Pythons × four platforms, twenty wheels.
Nothing about publishing changes: dispatch the list above, from the default
branch, with `publish` enabled.

The consumer side followed the same two-stage rule as every other revision.
Two parts of it landed before post3 existed, because they are inert until
`requires-python` admits 3.14:

- the `taichi` extra is marked `python_version < '3.14'` (taichi 1.7.4 has no
  cp314 wheel and no sdist, so unmarked it makes `algan[dev]` uninstallable
  there);
- `[tool.uv] constraint-dependencies` in `pyproject.toml` carries a floor for
  every locked binary package whose pinned version has no cp314 wheel, scoped
  to 3.14 by marker. That marker is what makes uv fork the resolution at 3.14,
  so 3.10–3.13 keep exactly the versions they were locked to before.

What waited for post3 to be on PyPI landed as one change. It is also the
recipe for the next ceiling:

1. `pyproject.toml`: `requires-python = ">=3.10,<3.15"`, the
   `Programming Language :: Python :: 3.14` classifier, and
   `algan-quadrants==1.3.0.post3`. Rewrite the comment above `requires-python`
   too — it says the ceiling waits on this release.
2. `uv lock`, from PyPI. Then check two things before committing: that every
   `(name, version)` in the old lock except `algan-quadrants` is still in the
   new one (the 3.10–3.13 pins did not move), and `uv lock --locked` with the
   uv that `.github/actions/install-uv` pins.
3. `.github/workflows/test.yaml`: move the Linux ceiling arm from `3.13` to
   `3.14`, and rewrite the comment that explains the ceiling (its `audioop-lts`
   reasoning still holds — the marker is `>= '3.13'`). `code_quality.yaml`'s
   `macos_packaging` job reinstalls the built abi3 wheel on "the supported
   ceiling" too; move that step with it.
4. The user-facing ranges: `docs/source/installation.rst` ("3.10 through
   3.13", and its `brew install python@3.13` / `uv python install 3.13` hints),
   `.claude/skills/algan-video/references/setup-and-rendering.md` and
   `.claude/skills/algan-video/scripts/check_environment.py`.

**Rehearsed, not performed** (2026-09-30), with a locally built cp314 wheel
re-versioned to `1.3.0.post3`, run through `rebrand_quadrants_wheel.py`, and
served to uv with `--find-links` in place of PyPI: step 1's edits plus
`uv lock` resolved; the only `(name, version)` the new lock lost was
`algan-quadrants 1.3.0.post2`, so no 3.10–3.13 pin moved;
`uv sync --locked --all-extras --dev --python 3.14` built the environment with
nothing compiled from source except `manimpango` (which has no Linux wheel on
any Python); and every 3.14-reachable package in that lock has a cp314, abi3 or
pure wheel for each platform it installs on, apart from the platform-gated
pyobjc/pywin32/nvidia sets and `taichi`, which the marker excludes. In that
environment `pytest -q --fast` passed, 645 of 645, fast render included. The lock
itself was not kept: it records the local `--find-links` directory rather than
PyPI's file hashes, which is exactly what step 2 must not commit.

**Performed** (2026-10-01).
[Run 36802622527](https://github.com/algorithmicsimplicity/algan/actions/runs/36802622527)
published all twenty post3 wheels from `master`; the aarch64 ones are tagged
`manylinux_2_34_aarch64`. `uv lock` against PyPI then lost only
`algan-quadrants 1.3.0.post2`. Each of the 33 `(name, version)` pairs it added
is reached only through an edge marked `python_full_version >= '3.14'` (apart
from `algan-quadrants 1.3.0.post3` itself), so no 3.10–3.13 pin moved. The
3.14 fork took torch 2.14.1 with torchaudio 2.11.0, which declares no torch
requirement at all. `uv lock --locked` passes with both uv 0.4.25 (the
`install-uv` pin) and 0.7.9. The 3.14 test arm in `test.yaml` is the first run
of this lock on 3.14; it was not synced locally before pushing.
