# Publishing Algan's patched Quadrants wheels

Algan's patched Quadrants build is published to PyPI under the **distribution**
name `algan-quadrants`. The installed Python package is still named
`quadrants`, so Algan continues to use `import quadrants` and
`algan.taichi_compat` does not change.

The current downstream release is `algan-quadrants==1.3.0.post2`, sixteen
wheels published 2026-09-06; `1.3.0.post1` was the first, twelve wheels. The
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
- Python versions: `3.10,3.11,3.12,3.13`
- apply patches: enabled
- publish: enabled

The workflow refuses a stock or partial matrix when `publish` is enabled, and
what it counts as complete is `resolve_wheel_matrix.py`'s platform table rather
than a list of names in the YAML — so a platform added there is required here
without this document or that gate being edited. Each platform builds the
ordinary `quadrants` wheel with `SETUPTOOLS_SCM_PRETEND_VERSION=1.3.0.post2`, so
Quadrants' Python metadata and its native build see the same version. After all
sixteen wheels succeed, the publish job rewrites only the distribution metadata
from `quadrants` to `algan-quadrants`, validates the complete matrix, and
uploads it to PyPI. The `quadrants/` import package is not renamed.

Normal diagnostic builds keep `publish` disabled. Their artifacts stay named
`quadrants-...whl`, which preserves `scripts/build_quadrants_wheels.py --install`
and the existing wheel provenance workflow.

## Bootstrap order for Algan

Do **not** change Algan's dependency to `algan-quadrants` before the first
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

**The Linux wheel filenames change in that revision too.** Both Linux legs now
build inside manylinux containers and are stamped with the tag the wheel is
measured to earn rather than the `manylinux_2_27` constant upstream's
`build_wheel` writes: `manylinux_2_28_x86_64` and `manylinux_2_35_aarch64`,
verified per wheel by `scripts/gate/verify_wheel_tag.py` before the stamp goes
on. (The aarch64 tag is a version above its own container, and that is
measured rather than chosen — a single GLOBAL `_dl_find_object@GLIBC_2.35`
coming out of the prebuilt LLVM it links, not out of any local toolchain
choice; pinning an older compiler was tried and changed nothing.
`resolve_wheel_matrix.py` carries the readings.) For
x86-64 users this was a *narrowing on paper and a fix in fact* — the
`1.3.0.post1` x86-64 wheel claims 2.27 and actually needs 2.34, so systems
between those two versions (RHEL 8, Ubuntu 20.04, Debian 11) install it and
fail at `import quadrants`. `post2`'s x86-64 wheels measure 2.27 for real, so
those three now work rather than being told a comfortable lie.
