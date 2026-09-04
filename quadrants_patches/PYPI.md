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

- platforms: `linux,macos,windows`
- Python versions: `3.10,3.11,3.12,3.13`
- apply patches: enabled
- publish: enabled

The workflow refuses a stock or partial matrix when `publish` is enabled. Each
platform builds the ordinary `quadrants` wheel with
`SETUPTOOLS_SCM_PRETEND_VERSION=1.3.0.post1`, so Quadrants' Python metadata and
its native CMake build see the same version. After all twelve wheels succeed,
the publish job rewrites only the distribution metadata from `quadrants` to
`algan-quadrants`, validates the complete matrix, and uploads it to PyPI. The
`quadrants/` import package is not renamed.

Normal diagnostic builds keep `publish` disabled. Their artifacts stay named
`quadrants-...whl`, which preserves `scripts/build_quadrants_wheels.py --install`
and the existing wheel provenance workflow.

## Bootstrap order for Algan

`pyproject.toml` now declares:

```toml
"algan-quadrants==1.3.0.post1",
```

That dependency cannot be resolved from PyPI until the first downstream wheel
release exists. Therefore the bootstrap order is:

1. Merge or otherwise dispatch the wheel workflow containing the publication
   changes.
2. Publish `algan-quadrants==1.3.0.post1` using the complete matrix above.
3. Run `uv lock` from the Algan repository and commit the resulting `uv.lock`.
4. Run the normal Algan test/release pipeline.
5. Publish Algan. A user can then install the patched compiler with only:

   ```bash
   pip install algan
   ```

The branch intentionally does **not** hand-edit `uv.lock` before step 2. A lock
entry should be generated from the actual PyPI files and hashes, not guessed in
advance.

## Releasing another downstream revision

For another patch-only revision of upstream v1.3.0, increment the downstream
version (for example `1.3.0.post2`) consistently in:

- `.github/workflows/quadrants_build.yaml`
- `scripts/rebrand_quadrants_wheel.py`
- `pyproject.toml`

Build and publish the complete wheel matrix, then regenerate `uv.lock` from the
new PyPI release. Do not overwrite an existing PyPI version; PyPI releases are
immutable.
