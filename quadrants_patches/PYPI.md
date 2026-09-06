# Publishing Algan's patched Quadrants wheels

Algan's patched Quadrants build is published to PyPI under the **distribution**
name `algan-quadrants`. The installed Python package is still named
`quadrants`, so Algan continues to use `import quadrants` and
`algan.taichi_compat` does not change.

The first downstream release is `algan-quadrants==1.3.0.post1`. The `post1`
suffix identifies Algan's patched build of upstream Quadrants v1.3.0 and keeps
it distinct from the upstream `quadrants==1.3.0` release.

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
ordinary `quadrants` wheel with `SETUPTOOLS_SCM_PRETEND_VERSION=1.3.0.post1`, so
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
version (for example `1.3.0.post2`) consistently in:

- `.github/workflows/quadrants_build.yaml`
- `scripts/rebrand_quadrants_wheel.py`
- `pyproject.toml`

Build and publish the complete wheel matrix, then regenerate `uv.lock` from the
new PyPI release. Do not overwrite an existing PyPI version; PyPI releases are
immutable.

**Adding a platform is one of those revisions.** `1.3.0.post1` was published as
twelve wheels and cannot grow four more: the publish step uploads the whole
directory with no `skip-existing`, so re-running it against that version fails
on the twelve files that already exist. The `linux_arm64` wheels therefore
reach users as `1.3.0.post2` — the full sixteen, built and published together,
followed by the `pyproject.toml` bump and a regenerated `uv.lock`. Until that
happens the aarch64 leg is a wheel the workflow can build and
`scripts/build_quadrants_wheels.py --install` can install, and `pip install
algan` on aarch64 Linux still resolves nothing.
