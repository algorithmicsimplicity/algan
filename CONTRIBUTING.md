# Thanks for your interest in contributing!

Please read our contributing guidelines, which are hosted at
<https://algorithmicsimplicity.github.io/algan/contributing.html>.

To set up a development environment — cloning the repository, installing the
system and Python dependencies, and running the tests, documentation build and
linters — see
<https://algorithmicsimplicity.github.io/algan/contributing/development.html>,
whose source lives in `docs/source/contributing/development.rst`.

In short:

```bash
git clone https://github.com/algorithmicsimplicity/algan
cd algan
uv venv
uv sync --locked --all-extras --dev
<venv-python> -m pytest -q --fast    # the curated fast suite; interpreter path: see the development guide
```

That skips the system dependencies, which differ per platform and which the
install will fail without — the development guide lists them.

Versioning, the `master` → `stable` release flow and how a release is cut and
tagged are in the "Versioning and releases" section of that same development
guide.

Before making a large change, open or review an issue on the
[issue tracker](https://github.com/algorithmicsimplicity/algan/issues) so the
design and scope can be discussed.
