============
Installation
============

.. _installation:

Algan is available in **Python 3.10 through 3.13**.
Algan can be installed from `PyPI <https://pypi.org/project/algan/>`__ with ``pip``,
like any other Python package:

.. code-block:: bash

   pip install algan

This will install a working version of Algan for CPU, however
depending on your operating system it may be missing some
features such as GPU acceleration, LaTeX type setting, speech synchronization,
and extra 3-D asset formats. To make sure you installation is fully functional
select the tab for your operating system below and follow the instructions there.

.. dropdown:: Never used Python before? Start here.
   :icon: light-bulb

   The instructions below assume that you are fimiliar with
   basic programming terminology, so they are spelled out here:

   **Python** is the programming language you will write your animations in,
   and Algan is a package you add to it: Python comes first, and the
   ``pip`` command that installs Algan comes with it. Your operating system's
   tab below starts with installing Python.

   **A terminal** is where you type the commands on this page. Open one with:

   * *Windows* -- press Start, type ``PowerShell``, press Enter.
   * *macOS* -- press Cmd-Space, type ``Terminal``, press Enter.
   * *Linux* -- your desktop's terminal application, usually Ctrl-Alt-T.

   "Run this command" means: copy the line, paste it into that window, press
   Enter, and wait for it to finish before running the next one. Every code
   block on this page has a copy button in its top-right corner.

   **A virtual environment** is a folder that holds the packages for one
   project, so that two projects can use different versions of the same package
   without fighting. ``python -m venv .venv`` creates one, and *activating* it
   points ``python`` and ``pip`` at that folder instead of at the whole
   machine. You activate it once per terminal window -- if you close the
   terminal and come back tomorrow, activate it again before running your
   scripts. Nothing about it is permanent: deleting the project folder deletes
   the environment with it, leaving Python itself untouched.

.. important::

   If you run into trouble, do not spend the evening on it: ask on our `Discord
   server <https://discord.gg/NvarFmvXKm>`__ or open an issue on the `GitHub
   issue tracker <https://github.com/algorithmicsimplicity/algan/issues>`__.

.. tip::

   Already use `uv <https://docs.astral.sh/uv/>`__, Poetry, conda or another
   environment manager? Nothing here is special: ``uv add algan``,
   ``uv pip install algan`` or ``poetry add algan`` work just the same.
   The instructions below use the ``venv`` module that comes with Python, so
   that they work on a machine with nothing extra installed.

.. tab-set::

   .. tab-item:: Windows

      .. dropdown:: Don't have Python yet? Install it first.
         :icon: download

         Download Python 3.13 from `python.org/downloads
         <https://www.python.org/downloads/>`__ and run the installer.
         **Tick "Add python.exe to PATH"** on the first screen.

         Check it from a fresh PowerShell window:

         .. code-block:: powershell

            py --version

      .. rubric:: 1. Install Algan

      Make a folder for your animations, create a virtual environment inside
      it, and install Algan into that environment:

      .. code-block:: powershell

         mkdir alganimations
         cd alganimations
         py -m venv .venv
         .venv\Scripts\activate
         pip install algan

      Your prompt now starts with ``(.venv)``. It will not in a *new* terminal:
      run ``.venv\Scripts\activate`` again each time you come back to the
      project.

      .. note::

         If activating fails with *"running scripts is disabled on this
         system"*, PowerShell's execution policy is blocking the script. Either
         allow it for this window only (
         ``Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`` ) or
         use ``cmd``, where the command is ``.venv\Scripts\activate.bat``.

      .. rubric:: 2. Check the installation

      .. code-block:: powershell

         algan check

      This prints the Algan and PyTorch versions, the device renders will run
      on, and where output and cache files will go. If it runs, you can make
      videos.

      .. rubric:: 3. Enable your GPU (recommended)

      If ``algan check`` says ``[INFO] Running on CPU`` even though you have an
      NVIDIA card, that is expected: **the PyTorch published on PyPI for
      Windows is built without CUDA.** Replace it with the CUDA build from
      PyTorch's own package index:

      .. code-block:: powershell

         pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128

      ``cu128`` is CUDA 12.8. If your driver is older, pick the build that
      matches it on `pytorch.org/get-started/locally
      <https://pytorch.org/get-started/locally/>`__ (choose *Stable*,
      *Windows*, *Pip*, *Python*, then your CUDA version) and use the index URL
      it gives you. Then run ``algan check`` again -- it should report
      ``[OK] CUDA acceleration`` and your card's name.

      For an AMD card, follow `AMD's PyTorch-on-Windows instructions
      <https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html>`__.

      Algan runs on the CPU perfectly well if you have no supported GPU; it is
      just slower.

      .. rubric:: Optional: mathematical formulas

      Prose works out of the box: :class:`~algan.mobs.text.Text` typesets with
      your system fonts, through Pango, which is installed with Algan on
      Windows.

      Mathematical formulas are a different matter. ``Tex`` and ``MathTex`` are
      typeset by LaTeX, which is a separate installation: get the `MiKTeX
      distribution <https://miktex.org/download>`__ and let it install packages
      on the fly. ``algan check`` reports whether it found one, and
      :ref:`installation-optional-latex` says what Algan actually asks of it.

      .. rubric:: Optional: speech and 3-D models

      * Speech **Generation** works out of the box, through the Windows SAPI5 voices.
        Aligning animations to a *pre-recorded* audio file additionally needs
        ``pip install "algan[audio]"``.
      * **3-D models** in glTF, GLB, OBJ and PLY are supported out of the box.
        FBX needs ``pip install "algan[fbx]"`` plus an ``assimp`` DLL on your
        ``PATH``.

   .. tab-item:: macOS

      .. rubric:: Requirements

      **Apple Silicon (M1 or newer) running macOS 13 Ventura or later.** Intel
      Macs are not supported: Algan's kernel compiler publishes arm64 wheels
      only, so ``pip install algan`` cannot resolve on an Intel machine.

      .. dropdown:: Don't have Python yet? Install it first.
         :icon: download

         macOS ships a Python that is best left alone. Install your own 3.13
         from `python.org/downloads <https://www.python.org/downloads/>`__, or
         with `Homebrew <https://brew.sh>`__:

         .. code-block:: bash

            brew install python@3.13

      .. rubric:: 1. Install Algan

      Make a folder for your animations, create a virtual environment inside
      it, and install Algan into that environment:

      .. code-block:: bash

         mkdir alganimations
         cd alganimations
         python3 -m venv .venv
         source .venv/bin/activate
         pip install algan

      Your prompt now starts with ``(.venv)``. It will not in a *new* terminal:
      run ``source .venv/bin/activate`` again each time you come back to the
      project.

      .. rubric:: 2. Check the installation

      .. code-block:: bash

         algan check

      This prints the Algan and PyTorch versions, the device renders will run
      on, and where output and cache files will go. It should report
      ``[OK] Apple Silicon MPS acceleration available``: your GPU is used
      automatically, through Metal, with nothing to configure.

      .. rubric:: Optional: mathematical formulas

      Prose works out of the box: :class:`~algan.mobs.text.Text` typesets with
      your system fonts, through Pango, which is installed with Algan on macOS.

      Mathematical formulas are a different matter. ``Tex`` and ``MathTex`` are
      typeset by LaTeX, which is a separate installation: get the `MacTeX
      distribution <https://www.tug.org/mactex/mactex-download.html>`__ and
      follow the standard installer. It is a large download; the much smaller
      BasicTeX plus the packages Algan uses also works:

      .. code-block:: bash

         brew install --cask basictex
         sudo tlmgr update --self
         sudo tlmgr install standalone preview dvisvgm

      ``algan check`` reports whether it found a LaTeX installation, and
      :ref:`installation-optional-latex` says what Algan actually asks of one.

      .. rubric:: Optional: speech and 3-D models

      * Speech **Generation** works out of the box, through the macOS speech
        synthesizer. Aligning animations to a *pre-recorded* audio file
        additionally needs ``pip install "algan[audio]"``.
      * **3-D models** in glTF, GLB, OBJ and PLY are supported out of the box.
        FBX needs ``brew install assimp`` and ``pip install "algan[fbx]"``.

   .. tab-item:: Linux

      .. rubric:: Requirements

      **x86-64 with glibc 2.27 or newer** -- Ubuntu 18.04+, Debian 10+, Fedora,
      RHEL 8+, Arch, and anything of similar vintage. ARM (aarch64) machines
      are not supported: Algan's kernel compiler publishes no aarch64 wheel.

      .. dropdown:: Don't have Python (or ``python3-venv``) yet? Install it first.
         :icon: download

         Most distributions ship a suitable Python 3. Debian and Ubuntu split
         the ``venv`` module into its own package, and the next step needs it,
         so install that too:

         .. code-block:: bash

            sudo apt install python3 python3-venv     # Debian / Ubuntu
            sudo dnf install python3                  # Fedora
            sudo pacman -S python                     # Arch

         If your distribution's Python is older than 3.10 or newer than 3.13,
         get a supported one from `deadsnakes
         <https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa>`__, `pyenv
         <https://github.com/pyenv/pyenv>`__ or ``uv python install 3.13``, and
         use it in place of ``python3`` below.

      .. rubric:: 1. Install Algan

      Make a folder for your animations, create a virtual environment inside
      it, and install Algan into that environment:

      .. code-block:: bash

         mkdir alganimations
         cd alganimations
         python3 -m venv .venv
         source .venv/bin/activate
         pip install algan

      Your prompt now starts with ``(.venv)``. It will not in a *new* terminal:
      run ``source .venv/bin/activate`` again each time you come back to the
      project.

      .. note::

         This pulls in around 6 GB, far more than on the other two platforms,
         because PyPI's Linux PyTorch bundles the entire CUDA runtime. If you
         have no NVIDIA GPU and would rather not carry it, install the CPU
         build of PyTorch *first* and Algan will use it, for around 1.5 GB
         in total:

         .. code-block:: bash

            pip install torch --index-url https://download.pytorch.org/whl/cpu
            pip install algan

      .. rubric:: 2. Check the installation

      .. code-block:: bash

         algan check

      This prints the Algan and PyTorch versions, the device renders will run
      on, and where output and cache files will go.

      With an NVIDIA GPU and a driver new enough for the bundled CUDA runtime,
      it reports ``[OK] CUDA acceleration`` and your card's name -- there is
      nothing to install for that, unlike on Windows. If it says
      ``[INFO] Running on CPU``, your driver is likely too old: check
      ``nvidia-smi``, then pick the PyTorch build matching your CUDA version at
      `pytorch.org/get-started/locally
      <https://pytorch.org/get-started/locally/>`__ and install it with the
      index URL given there. For an AMD card, install a ROCm build of PyTorch
      the same way.

      Algan runs on the CPU perfectly well if you have no supported GPU; it is
      just slower.

      .. rubric:: Optional: text and mathematical formulas

      Linux is the one platform where text needs something installed first.
      Algan draws prose with :class:`~algan.mobs.text.Text` and formulas with
      :class:`~algan.mobs.text.Tex`; on Windows and macOS ``Text`` uses system
      fonts through Pango, which is installed with Algan there, but Pango
      publishes no Linux wheel and so is not. Until you install one of the two
      backends below, neither class can draw anything.

      LaTeX is much the easier of the two here, and it covers both classes, so
      install it first:

      .. code-block:: bash

         # Debian / Ubuntu -- the same set Algan's own CI installs
         sudo apt install texlive-latex-base texlive-latex-extra \
                          texlive-fonts-recommended latexmk

      ``texlive-latex-extra`` pulls in ``texlive-latex-recommended``, and the
      ``dvisvgm`` converter that turns LaTeX output into glyph outlines arrives
      with ``texlive-binaries``, so neither needs naming separately.

      On Fedora (``dnf``) or Arch (``pacman``), install the equivalent *TeX
      Live* packages for your distribution. If you would rather not work out
      the mapping, the complete distribution (``texlive-scheme-full`` on
      Fedora, ``texlive-meta`` on Arch) always works -- it is a multi-gigabyte
      download, which is the only reason not to recommend it first. See
      :ref:`installation-optional-latex` for the small set Algan actually uses.

      With LaTeX alone, ``Text`` typesets through LaTeX's text mode: it works,
      but it cannot use your system fonts and ignores the font, weight and
      slant arguments. To get those, install the ``pango`` extra as well. This
      is the one part of Algan that compiles from source on Linux -- which is
      why it is opt-in here and automatic elsewhere -- so install a compiler
      and the development headers first:

      .. code-block:: bash

         # Debian / Ubuntu
         sudo apt install build-essential python3-dev libpango1.0-dev pkg-config
         # Fedora
         sudo dnf install gcc python3-devel pango-devel pkg-config
         # Arch
         sudo pacman -S base-devel pango

      .. code-block:: bash

         pip install "algan[pango]"

      .. rubric:: Optional: speech and 3-D models

      * Speech **Generation** needs a system speech engine, which Linux does not ship
        by default:

        .. code-block:: bash

           sudo apt install espeak-ng    # Debian / Ubuntu
           sudo dnf install espeak-ng    # Fedora
           sudo pacman -S espeak-ng      # Arch

        Aligning animations to a *pre-recorded* audio file additionally needs
        ``pip install "algan[audio]"``.

      * **3-D models** in glTF, GLB, OBJ and PLY are supported out of the box.
        FBX needs ``pip install "algan[fbx]"`` plus the native assimp library
        (``sudo apt install libassimp5``, ``sudo dnf install assimp``,
        ``sudo pacman -S assimp``).

Your first animation
====================

With the environment activated, put the following code into ``my_first_animation.py`` inside
your project folder:

.. code-block:: python

   from algan import *

   square = Square().spawn()

   Scene.save_video("example")

and run it:

.. code-block:: bash

   python my_first_animation.py

If there is now a video at ``algan_outputs/example.mp4`` beside the script,
your installation is complete! Continue on to :doc:`new_user_tutorials/getting_started`
to learn how to use Algan.

.. _installation-optional-latex:

What Algan needs from LaTeX
===========================

Algan's default TeX template is deliberately small. It needs only the
``standalone``, ``babel``, ``amsmath`` and ``amssymb`` packages, plus the
``latex`` and ``dvisvgm`` binaries. Any distribution providing those renders
every ``Tex`` and ``MathTex`` Algan builds on its own, which is why a minimal
distribution such as `TinyTeX <https://yihui.org/tinytex/>`__ or BasicTeX is
enough.

.. dropdown:: The wider package list, for custom Manim templates

   The list below is what the *Manim* templates Algan can be pointed at may
   reach for. Install it only if you supply your own ``TexTemplate`` with extra
   ``\usepackage`` lines:

   .. code-block:: text

      amsmath babel-english cbfonts-fd cm-super count1to ctex doublestroke dvisvgm everysel
      fontspec frcursive fundus-calligra gnu-freefont jknapltx latex-bin
      mathastext microtype multitoc physics preview prelim2e ragged2e relsize rsfs
      setspace standalone tipa wasy wasysym xcolor xetex xkeyval

Installing from source
======================

Everything above installs a released Algan for *writing animations*. To work on
Algan itself, or to run a version newer than the latest release, clone the
repository and install it from source instead, as described in :doc:`contributing/development`.

