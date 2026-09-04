# Installing Algan locally

The standard way of installing Algan is by using
Python's package manager `pip` to install the latest
release from [PyPI](https://pypi.org/project/algan/).

To make it easier for you to follow best practices when it
comes to setting up a Python project for your Algan animations,
we strongly recommend using a tool for managing Python environments
and dependencies. In particular,
[we strongly recommend using `uv`](https://docs.astral.sh/uv/#getting-started).

For the main way of installing Algan described below, we assume
that `uv` is available; we think it is particularly helpful if you are
new to Python or programming in general. It is not a hard requirement
whatsoever; if you know what you are doing you can just use `pip` to
install Algan directly.

:::::{admonition} Installing the Python management tool `uv`
:class: seealso

One way to install `uv` is via the dedicated console installer supporting
all large operating systems. Simply paste the following snippet into
your terminal / PowerShell -- or
[consult `uv`'s documentation](https://docs.astral.sh/uv/#getting-started)
for alternative ways to install the tool.

::::{tab-set}
:::{tab-item} MacOS and Linux
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
:::
:::{tab-item} Windows
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
:::
::::

:::::

Of course, if you know what you are doing and prefer to setup a virtual
environment yourself, feel free to do so!

:::{important}
If you run into issues when following our instructions below, do not worry:
you can ask for help on our [Discord server](https://discord.gg/NvarFmvXKm),
or report the problem on our
[GitHub issue tracker](https://github.com/algorithmicsimplicity/algan/issues).
:::


## Installation

### Step 1: Installing Python

We first need to check that an appropriate version of Python is available
on your machine. Algan supports Python 3.10 through 3.13 (Python 3.14 is not
supported yet -- both ends are the kernel compiler's wheel range). Open a
terminal to run
```bash
uv python install 3.13
```
to install Python 3.13.

Once installed, we can create a new folder 'alganimations' and instantiate a uv virtual 
environment by running these commands

```bash
uv init --python 3.13 alganimations
cd alganimations
uv venv
```

uv will install packages to the virtual environment contained in the current working directory,
so make sure you stay in the same working directory throughout the installation process!

(installation-optional-latex)=
### Step 2 (optional): Installing LaTeX

[LaTeX](https://en.wikibooks.org/wiki/LaTeX/Mathematics) is a very well-known
and widely used typesetting system allowing you to write formulas like

\begin{equation*}
\frac{1}{2\pi i} \oint_{\gamma} \frac{f(z)}{(z - z_0)^{n+1}}~dz
= \frac{f^{(n)}(z_0)}{n!}.
\end{equation*}

Algan provides two ways to render text:
- **`Text`**: Prose. Typeset through LaTeX's text mode by default, or through
  system fonts and Pango if you install the optional `pango` extra (see
  [Optional: Pango text](#optional-pango-text) below).
- **`Tex` and `MathTex`**: High-quality mathematical equations and formulas rendered using LaTeX.

If you only plan to draw geometric graphics, you can skip this step. If you plan to render text or mathematical formulas, follow the instructions below for your operating system.

:::::{tab-set}

::::{tab-item} Windows
For Windows we recommend installing LaTeX via the
[MiKTeX distribution](https://miktex.org). Simply grab
the Windows installer available from their download page,
<https://miktex.org/download> and run it.
::::

::::{tab-item} MacOS
If you are running MacOS, we recommend installing the
[MacTeX distribution](https://www.tug.org/mactex/). The latest
available PKG file can be downloaded from
<https://www.tug.org/mactex/mactex-download.html>.
Get it and follow the standard installation procedure.
::::

::::{tab-item} Linux
Given the large number of Linux distributions with different ways
of installing packages, we cannot give detailed instructions for
all package managers. What you need is a *TeX Live* distribution
(<https://www.tug.org/texlive/>), which every major distribution
packages.

On Debian-based systems with the package manager `apt`, this is
enough — it is the same set Algan's CI installs:
```bash
sudo apt install texlive-latex-base texlive-latex-extra \
                 texlive-fonts-recommended latexmk
```
`texlive-latex-extra` pulls in `texlive-latex-recommended`, and the
`dvisvgm` converter Algan uses to turn LaTeX output into glyph outlines
arrives with `texlive-binaries`, so those do not need naming separately.

For Fedora (managed via `dnf`) or Arch (`pacman`), install the
equivalent LaTeX packages for your distribution; if you would rather not
work out the mapping, the complete distribution (`texlive-scheme-full` on
Fedora, `texlive-meta` on Arch) always works. It is a multi-gigabyte
download that takes a while, which is the only reason we do not
recommend it first.

As soon as LaTeX is installed, continue with actually installing Algan
itself.

::::

:::::

:::{dropdown} I know what I am doing and I would like to setup a minimal LaTeX installation
You are welcome to use a smaller, more customizable LaTeX distribution like
[TinyTeX](https://yihui.org/tinytex/).

Algan's default TeX template is deliberately small — it needs only
`standalone`, `babel`, `amsmath` and `amssymb`, plus the `latex` and
`dvisvgm` binaries. Any distribution providing those renders every
`Tex` and `MathTex` Algan builds on its own.

The wider list below is what the *Manim* templates Algan can be pointed
at may reach for. Install it if you supply your own
`TexTemplate` with extra `\usepackage` lines, not otherwise:
```text
amsmath babel-english cbfonts-fd cm-super count1to ctex doublestroke dvisvgm everysel
fontspec frcursive fundus-calligra gnu-freefont jknapltx latex-bin
mathastext microtype multitoc physics preview prelim2e ragged2e relsize rsfs
setspace standalone tipa wasy wasysym xcolor xetex xkeyval
```
:::

### Step 3: Installing PyTorch

Algan uses PyTorch for GPU-accelerated array operations.
If you do not have a GPU you can also run Algan on CPU,
but you will still need to install PyTorch.
PyTorch can be installed by running:
```bash
uv pip install torch torchvision
```

Check that the installation was succesful by running the
appropriate command:
:::::{tab-set}

::::{tab-item} Windows & Linux
```bash
uv run python -c "import torch; print(f'GPU is available: {torch.cuda.is_available()}')"
```

::::

::::{tab-item} Mac
```bash
uv run python -c "import torch; print(f'GPU is available: {torch.mps.is_available()}')"
```

::::

:::::
If it says `GPU is available: True` (or you want Algan to run on CPU),
the installation was successful and you can go to the next step.

Otherwise, in order to use your GPU you will need to go to [pytorch.org](https://pytorch.org/get-started/locally/) 
and follow the instructions there. Select the following options
PyTorch Build: Stable
Your OS: Your OS
Package: Pip
Language: Python
Compute Platform: Your hardware

Then run the command given in the "Run this Command" box, with "uv pip" instead of "pip3".
e.g. the box says:
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

you run:
```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

If you are using an AMD (ROCm) GPU on Windows, you will need to follow the instructions at
[AMD's documentation](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html/)

### Step 4: Installing Algan 

:::{note}
Algan's installed footprint is large -- around 5 GB, mostly PyTorch's CUDA
wheels and their NVIDIA dependencies pulled in by Step 3 above.
:::

Follow the instructions for your operating system below:

::::::{tab-set}

:::::{tab-item} Windows

Run this command
```powershell
uv pip install algan
```

:::::

:::::{tab-item} MacOS
Nothing to prepare: every Algan dependency ships a macOS wheel.

```bash
uv pip install algan
```
:::::

:::::{tab-item} Linux
Nothing to prepare: every Algan dependency ships a Linux wheel, so there is
nothing to compile and no system package to install first.

```bash
uv pip install algan
```

:::::

::::::

:::{note}
If you prefer to manage your dependencies through your project's
`pyproject.toml` (the recommended `uv` workflow), you can run
`uv add algan` instead of `uv pip install algan`.
:::

(optional-pango-text)=
### Optional: Pango text

`Text` typesets through LaTeX's text mode by default, which needs no extra
install. Pango gives it system fonts instead, plus per-span font, weight and
slant styling, and adds Manim's `MarkupText` and `Paragraph` to
`algan.manim`. To use it, install the `pango` extra:

```bash
uv pip install "algan[pango]"
```

It is an extra rather than a dependency because
[ManimPango](https://github.com/ManimCommunity/ManimPango) publishes no Linux
wheel. On Windows and macOS it installs a wheel and there is nothing else to
do; on Linux it builds from source and wants a C compiler, Python's
development headers, `pkg-config`, and Pango's development headers first:

::::{tab-set}

:::{tab-item} Debian-based / apt
```bash
sudo apt update
sudo apt install build-essential python3-dev libpango1.0-dev pkg-config
```
:::

:::{tab-item} Fedora / dnf
```bash
sudo dnf install python3-devel pkg-config pango-devel
```
:::

:::{tab-item} Arch Linux / pacman
```bash
sudo pacman -Syu base-devel pango
```
:::

::::

If you completed the installation instructions with no errors, then you are ready to Alganimate!

At this point, you can also open your project folder with the
IDE of your choice. All modern Python IDEs (for example VS Code
with the Python extension, or PyCharm) should automatically detect
the local environment created by `uv` such that if you put
```py
import algan
```
into a new file `my-first-animation.py`, the import is resolved
correctly and autocompletion is available.

## Installing from source

The instructions above install a released Algan for *writing animations*. If
you want to work on Algan itself, or run a version newer than the latest
release, clone the repository and install it from source instead. That setup,
including the test suite and the documentation build, is covered in
{doc}`../contributing/development`.
