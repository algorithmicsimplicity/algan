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
If you run into issues when following our instructions below, do
not worry: check our [installation FAQs](<project:/faq/installation.md>) to
see whether the problem is already addressed there -- and otherwise go and
check [how to contact our community](<project:/faq/help.md>) to get help.
:::


## Installation

### Step 1: Installing Python

We first need to check that an appropriate version of Python is available
on your machine. Open a terminal to run
```bash
uv python install
```
to install the latest version of Python. 

Once installed, we can create a new folder 'alganimations' and create a uv virtual 
environment by running these commands

::::{tab-set}
:::{tab-item} MacOS and Linux
```bash
uv init alganimations
cd alganimations
uv venv
source .venv/bin/activate
```
:::
:::{tab-item} Windows
```powershell
uv init alganimations
cd alganimations
uv venv
.venv/Scripts/activate
```
:::
::::

The final command activates the virtual environment we just created.
This means that any commands to install Python packages will install to this folder,
and the Python command will
use the interpreter installed to this folder. Ensure that this environment is active for the rest
of the installation process.

(installation-optional-latex)=
### Step 2 (optional): Installing LaTeX

[LaTeX](https://en.wikibooks.org/wiki/LaTeX/Mathematics) is a very well-known
and widely used typesetting system allowing you to write formulas like

\begin{equation*}
\frac{1}{2\pi i} \oint_{\gamma} \frac{f(z)}{(z - z_0)^{n+1}}~dz
= \frac{f^{(n)}(z_0)}{n!}.
\end{equation*}

Algan uses LaTeX to generate its text. If you never intend to render text, then
you can technically skip this step. Otherwise select your operating system from the tab 
list below and follow the instructions.

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
all package managers.

In general we recommend to install a *TeX Live* distribution
(<https://www.tug.org/texlive/>). For most Linux distributions,
TeX Live has already been packaged such that it can be installed
easily with your system package manager. Search the internet and
your usual OS resources for detailed instructions.

For example, on Debian-based systems with the package manager `apt`,
a full TeX Live distribution can be installed by running
```bash
sudo apt install texlive-full
```
For Fedora (managed via `dnf`), the corresponding command is
```bash
sudo dnf install texlive-scheme-full
```
As soon as LaTeX is installed, continue with actually installing Algan
itself.

::::

:::::

:::{dropdown} I know what I am doing and I would like to setup a minimal LaTeX installation
You are welcome to use a smaller, more customizable LaTeX distribution like
[TinyTeX](https://yihui.org/tinytex/). Algan overall requires the following
LaTeX packages to be installed in your distribution:
```text
amsmath babel-english cbfonts-fd cm-super count1to ctex doublestroke dvisvgm everysel
fontspec frcursive fundus-calligra gnu-freefont jknapltx latex-bin
mathastext microtype multitoc physics preview prelim2e ragged2e relsize rsfs
setspace standalone tipa wasy wasysym xcolor xetex xkeyval
```
:::

### Step 3: Installing PyTorch

Algan is built on top of PyTorch, to provide GPU accelerated animations and rendering.
Depending on your operating system and GPU hardware, you will need to install different
versions of PyTorch.

Head over to [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) 
and you should see a bunch of options to choose from.

* For the PyTorch build select `Stable`.

* Select your operating system.

* For the package select `pip`.

* For the language select `Python`.

* For the compute platform, you need to select the right version for your
 computer's GPU hardware. If you are using an NVIDIA GPU you want to select
 the latest version of CUDA. If you are using an AMD GPU you will want to
 select ROCm. Otherwise, you can select CPU to run without GPU acceleration.

There should now be a pip3 command shown in the "Run this Command" box. Since we are using
uv to manage our packages, we need to modify the this command to use uv.
Simply replace the `pip3` at the start of the command with `uv pip`.
For example, PyTorch shows me the command

`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128`

So I would run

```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

If no error message is shown, then installation was successful and you can move on
to installing Algan.

### Step 4 (optional): Installing Manim 

Algan is a work in progress, and currently our selection of built-in objects is
relatively limited. To make up for this, we provide functionality to import
objects created in Manim into Algan, allowing you to make use of Manim's much
more extensive library of built in objects. In order to import Manim objects,
you need to install Manim. If Algan's built in library is enough for needs,
you can skip this step.

Otherwise, follow the below instructions for your
operating system.

::::::{tab-set}

:::::{tab-item} Windows

Run this command
```powershell
uv add manim
```

:::::

:::::{tab-item} MacOS
Before we can install Manim, we need to make sure that the system utilities
`cairo` and `pkg-config` are present. They are needed for the [`pycairo` Python
package](https://pycairo.readthedocs.io/en/latest/), a dependency of Manim.

The easiest way of installing these utilities is by using [Homebrew](https://brew.sh/),
a fairly popular 3rd party package manager for MacOS. Check whether Homebrew is
already installed by running

```bash
brew --version
```

which will report something along the lines of `Homebrew 4.4.15-54-...`
if it is installed, and a message `command not found: brew` otherwise. In this
case, use the shell installer [as instructed on Homebrew's website](https://brew.sh/),
or get a `.pkg`-installer from
[their GitHub release page](https://github.com/Homebrew/brew/releases). Make sure to
follow the instructions of the installer carefully, especially when prompted to
modify your `.zprofile` to add Homebrew to your system's PATH.

With Homebrew available, the required utilities can be installed by running

```bash
brew install cairo pkg-config
```

With all of this preparation out of the way, now it is time to actually install
Manim itself!

```bash
uv add manim
```
:::::

:::::{tab-item} Linux
Linux requires some additional dependencies to build 
[ManimPango](https://github.com/ManimCommunity/ManimPango)
(and potentially [pycairo](https://pycairo.readthedocs.io/en/latest/))
from source. More specifically, this includes:

- A C compiler,
- Python's development headers,
- the `pkg-config` tool,
- Pango and its development headers,
- and Cairo and its development headers.

Instructions for popular systems / package managers are given below.

::::{tab-set}

:::{tab-item} Debian-based / apt
```bash
sudo apt update
sudo apt install build-essential python3-dev libcairo2-dev libpango1.0-dev
```
:::

:::{tab-item} Fedora / dnf
```bash
sudo dnf install python3-devel pkg-config cairo-devel pango-devel
```
:::

:::{tab-item} Arch Linux / pacman
```bash
sudo pacman -Syu base-devel cairo pango
```
:::

::::

As soon as the required dependencies are installed, you can run
```bash
uv add manim
```

:::::

::::::

### Step 5: Installing Algan

Once PyTorch has successfully been installed,
you can proceed to install Algan by entering the following command in your terminal/powershell:

```bash
uv pip install --extra-index-url https://test.pypi.org/simple/ algan --index-strategy unsafe-best-match
```

If it completes with no errors, then you are ready to Alganimate!

At this point, you can also open your project folder with the
IDE of your choice. All modern Python IDEs (for example VS Code
with the Python extension, or PyCharm) should automatically detect
the local environment created by `uv` such that if you put
```py
import algan
```
into a new file `my-first-animation.py`, the import is resolved
correctly and autocompletion is available.

*Happy Alganimating!*
