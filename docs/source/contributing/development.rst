###########################
Development
###########################

This page sets up Algan **from source**, for working on Algan itself. To
install a released Algan and write animations with it, follow
:doc:`../installation/uv` instead.

Ask questions on the `Discord server <https://discord.gg/NvarFmvXKm>`__ and
report bugs on the `issue tracker
<https://github.com/algorithmicsimplicity/algan/issues>`__.

System dependencies
===================

Install these before the Python packages; two of Algan's dependencies build
from source and will fail without them.

.. tab-set::

   .. tab-item:: Debian / Ubuntu

      .. code-block:: bash

         sudo apt update
         sudo apt install build-essential python3-dev pkg-config \
                          libcairo2-dev libpango1.0-dev \
                          texlive-latex-base texlive-latex-extra \
                          texlive-fonts-recommended latexmk \
                          ffmpeg

   .. tab-item:: Fedora

      .. code-block:: bash

         sudo dnf install gcc gcc-c++ python3-devel pkg-config \
                          cairo-devel pango-devel ffmpeg
         # plus a LaTeX installation -- see the installation guide

   .. tab-item:: macOS

      .. code-block:: bash

         brew install cairo pkg-config ffmpeg
         # plus MacTeX -- see the installation guide

   .. tab-item:: Windows

      Install `MiKTeX <https://miktex.org>`__ and add an `FFmpeg build
      <https://ffmpeg.org/download.html>`__ to ``PATH``. The Cairo and Pango
      headers are not needed: ``pycairo`` and ``manimpango`` both ship Windows
      wheels.

Why each of these is needed:

* **A C compiler, Python headers, pkg-config, Cairo and Pango headers** —
  ``manimpango`` publishes no Linux wheels at all and ``pycairo`` may also
  build from source, so both are compiled during the install. Without the
  Pango headers the install fails with ``Package 'pangocairo' was not found``.
* **LaTeX** — required for ``Tex`` and ``Text``. See
  :ref:`the LaTeX step <installation-optional-latex>` for what a minimal
  installation needs.
* **FFmpeg** — *not* needed to render video, which uses the binary bundled
  with ``imageio-ffmpeg``, but the documentation build shells out to a system
  ``ffmpeg``.

Getting the source
==================

.. code-block:: bash

   git clone https://github.com/algorithmicsimplicity/algan
   cd algan
   uv venv
   uv sync --locked --all-extras --dev

``uv sync`` installs Algan itself in editable mode alongside every runtime,
optional and development dependency, pinned to the versions in ``uv.lock``.

.. important::

   Use ``--locked``. Resolving fresh instead of installing the lockfile picks
   up newer releases of Manim, Torch and OpenCV than the ones Algan is tested
   against, and those have broken Algan before.

   If you change a dependency in ``pyproject.toml``, re-run ``uv lock`` and
   commit the updated ``uv.lock`` with your change.

Torch and your GPU
------------------

The lockfile installs a CUDA build of Torch. Algan runs without a GPU — it
falls back to the CPU automatically, and the test suite passes there — but
rendering is much slower. For a ROCm or a CPU-only build, install the wheel
for your platform from https://pytorch.org/get-started/locally/ over the
top, replacing ``pip3`` with ``uv pip`` in the command PyTorch gives you.

Running the interpreter
=======================

Every command below is written as ``<venv-python>``, meaning the virtual
environment's interpreter:

* Linux / macOS: ``.venv/bin/python``
* Windows: ``.venv\Scripts\python.exe``

The system ``python`` will not do — it has none of the pinned dependencies.
Alternatively, prefix commands with ``uv run`` (``uv run pytest -q --fast``),
which resolves the environment for you on every platform.

Testing
=======

.. code-block:: bash

   <venv-python> -m pytest -q --fast   # the development loop, ~1 minute
   <venv-python> -m pytest -q          # everything, ~12 minutes

Run ``--fast`` after every change and the full suite before opening a pull
request. ``--fast`` runs a hand-curated set of about 190 tests -- those marked
``fast``, and nothing else -- covering the machinery every animation and render
goes through, plus one pixel-compared render. It reports where it landed
against its budget. Pass no path: it uses ``testpaths`` from ``pyproject.toml``.

A test you add is outside that set unless you mark it, which is deliberate:
mark ``fast`` only when a change elsewhere in the codebase is liable to break
it. ``tests/README.md`` documents what is in the fast suite, what is left out,
and where each omission is covered instead.

.. note::

   Render tests compare frames pixel-wise against baselines checked in per
   device, under ``expected_outputs_cuda/`` or ``expected_outputs_cpu/``. Both
   sets are committed, so the comparison runs on a CPU-only machine too. CPU and
   CUDA renders are not bit-identical, though, so re-baseline only for the device
   you are on -- and a device with no baseline directory renders the scene and
   silently skips the comparison.

Re-baselining a render
----------------------

When a change legitimately alters rendered output, regenerate the baselines for
your device and **look at the result** before committing it:

.. code-block:: bash

   ALGAN_UPDATE_FAST_BASELINE=1 .venv/bin/python -m pytest -q tests/fast
   ALGAN_UPDATE_FULL_RENDER_BASELINES=1 .venv/bin/python -m pytest -q tests/full_renders

Diff videos for a failing comparison land in that suite's ``output_errors/``.
Small deviations (a channel or two) across runs are expected and tolerated;
anything larger is a real change and needs an explanation in the pull request.

Documentation
=============

.. code-block:: bash

   <venv-python> docs/make_and_open_docs.py

This renders every embedded example video, so it is slow and needs a system
``ffmpeg``. For a structural or autodoc-only check:

.. code-block:: bash

   <venv-python> docs/make_and_open_docs.py --skip-examples --no-open

Docstrings on the public API follow ``DOCSTRINGS.md``; read it before writing
or editing one.

Documented code is tested
-------------------------

``tests/unit_tests/test_doc_examples.py`` extracts every Python block in
``docs/source`` and checks it, so a renamed API cannot quietly leave the
tutorials behind. Two of its three tiers run whenever the suite does: a static
pass over every block, and an execution pass over the blocks that are complete
scripts with rendering stubbed out. The third tier actually renders them and is
opt-in behind ``ALGAN_RUN_DOC_RENDERS=1``.

Prefer ``.. algan::`` over ``.. code-block:: python`` when an example is a
complete script *and* its rendered result teaches the reader something -- the
directive renders it during a full build and embeds the video. Keep
``code-block`` for fragments, for anti-examples, and for anything needing an
asset the repository does not carry; mark those last two so the test skips them:

.. code-block:: rst

   .. algan-doc-check: skip -- needs an asset that does not ship with the docs

   .. code-block:: python

The marker is an reStructuredText comment, so it never reaches the rendered
page. ``tests/README.md`` documents the tiers and when to reach for each.

Linting
=======

.. code-block:: bash

   <venv-python> -m ruff check --no-fix
   <venv-python> -m ruff format --check

.. warning::

   Ruff is configured with ``fix = true``, so a bare ``ruff check`` **rewrites
   your files**. Pass ``--no-fix`` unless you mean to apply the fixes.

   Never let a formatter touch ``*_taichi.py``. The ``from __future__ import
   annotations`` it inserts breaks Taichi kernel compilation. Those files are
   excluded in the Ruff configuration, which is why every kernel module's name
   has to end in ``_taichi``.

Repository conventions
======================

``AGENTS.md`` records the repository-specific rules a change is expected to
follow, and ``AGENTS_DETAILED.md`` is the architecture reference. In
particular: preserve Scene containment, keep Taichi sources named
``*_taichi.py``, and validate rendering changes against a small deterministic
scene before running long benchmarks.
