===========
Development
===========

This page explains how to set up Algan **from source** to work on Algan itself.
If you just want to install a released version to make animations, follow
:doc:`../installation/uv` instead.

Have questions or want to discuss changes? Head over to our `Discord server
<https://discord.gg/NvarFmvXKm>`__ or the `GitHub issue tracker
<https://github.com/algorithmicsimplicity/algan/issues>`__.

System Dependencies
===================

Install these before installing Python dependencies, as two of Algan's packages
compile native extensions from source:

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
      <https://ffmpeg.org/download.html>`__ to your ``PATH``. Cairo and Pango
      headers are not required because ``pycairo`` and ``manimpango`` ship
      pre-built Windows wheels.

Why each dependency is needed:

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

Getting the Source Code
=======================

Clone the repository and set up an editable environment using ``uv``:

.. code-block:: bash

   git clone https://github.com/algorithmicsimplicity/algan
   cd algan
   uv venv
   uv sync --locked --all-extras --dev

``uv sync`` installs Algan in editable mode along with all runtime, optional, and
development dependencies pinned to the versions in ``uv.lock``.

.. important::

   Always pass ``--locked``. Resolving without the lockfile may pull in newer
   releases of upstream dependencies (such as Torch or Manim) that have not been
   tested and could introduce breaking changes.

   If you change a dependency in ``pyproject.toml``, re-run ``uv lock`` and
   commit the updated ``uv.lock`` with your change.

PyTorch and Hardware Acceleration
---------------------------------

The lockfile installs a CUDA build of PyTorch by default. Algan works without a
GPU by falling back to CPU execution (and all tests pass on CPU), but rendering
will be significantly slower. For a ROCm or a CPU-only build, install the wheel
for your platform from https://pytorch.org/get-started/locally/ over the
top, replacing ``pip3`` with ``uv pip`` in the command PyTorch gives you.`.

Running the Interpreter
=======================

Run commands through ``uv run`` (e.g. ``uv run pytest -q --fast``) or use the
virtual environment's Python interpreter directly:

* Linux / macOS: ``.venv/bin/python``
* Windows: ``.venv\Scripts\python.exe``

Do not use your system Python, as it will not have the locked virtual environment
packages.

Testing
=======

We provide two testing loops:

.. code-block:: bash

   uv run pytest -q --fast   # Fast development loop (~1 minute)
   uv run pytest -q          # Full test suite (~12 minutes)

Run ``--fast`` after every code change, and run the full test suite before opening
a pull request.

The ``--fast`` suite runs a curated set of ~190 tests covering the core animation
pipeline, scene management, timeline materialization, and a deterministic
pixel-compared render test.

.. note::

   Render tests compare generated frames against baselines committed in
   ``expected_outputs_cuda/`` or ``expected_outputs_cpu/``. Because CPU and GPU
   rasterization differences are expected, baseline files are maintained
   separately for each backend. macOS is keyed separately again
   (``expected_outputs_macos_cpu/``), because Apple Silicon is a different
   instruction set: whether a path tracer's ``float32`` arithmetic reproduces
   across the two closely enough for the two-channel tolerance is what the
   macOS CI job is there to find out. If it does not, that directory is
   emptied and a Mac renders without comparing.

Updating Baseline Videos
------------------------

When a change you've made legitimately and intentionally alters rendered output,
regenerate the baselines for your device and **look at the result** before committing it:

.. code-block:: bash

   ALGAN_UPDATE_FAST_BASELINE=1 uv run pytest -q tests/fast
   ALGAN_UPDATE_FULL_RENDER_BASELINES=1 uv run pytest -q tests/full_renders

If a comparison fails, a diff video will be produced in that suite's ``output_errors/`` dir.
Small deviations (a channel or two) across runs are expected and tolerated;
anything larger is a real change and needs an explanation in the pull request.

Documentation
=============

.. code-block:: bash

   uv run python docs/make_and_open_docs.py

This renders every embedded example video, so it is quite slow.
For a structural or autodoc-only check:

.. code-block:: bash

   uv run python docs/make_and_open_docs.py --skip-examples --no-open

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
complete script and the video will get the point across faster than text can.
Keep ``code-block`` for fragments, for anti-examples, and for anything needing an
asset the repository does not carry; mark those last two so the test skips them:

.. code-block:: rst

   .. algan-doc-check: skip -- needs an asset that does not ship with the docs

   .. code-block:: python

The marker is an reStructuredText comment, so it never reaches the rendered
page. ``tests/README.md`` documents the tiers and when to reach for each.

Linting
=======

.. code-block:: bash

   uv run ruff check --no-fix
   uv run ruff format --check

.. warning::

   Ruff is configured with ``fix = true``, so a bare ``ruff check`` **rewrites
   your files**. Pass ``--no-fix`` unless you mean to apply the fixes.

   Never let a formatter touch ``*_taichi.py``. The ``from __future__ import
   annotations`` it inserts breaks Taichi kernel compilation. Those files are
   excluded in the Ruff configuration, which is why every kernel module's name
   has to end in ``_taichi``.

Opening a pull request
======================

``.github/pull_request_template.md`` is the layout, and it asks for the things a
diff cannot show: what the change is for, whether rendered output moved, which
suites you ran and on what hardware, and which documentation pages moved with
it.
