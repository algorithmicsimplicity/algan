=====================
Using PyTorch Compile
=====================

If you are are running on Linux or MacOS, then you can skip this section, Pytorch compile
is already activated by default.

If you are running on Windows, read on!

PyTorch offers the ability to compile code written in PyTorch. Compiled code
will combine multiple GPU kernels into one, as well as several other optimizations,
resulting in much faster code execution.

Unfortunately, PyTorch compile does not work on Windows.
But there is a work around for Windows users: you just need to run PyTorch
from within the Windows Subsystem Linux (WSL).

In order to do this, you will first need to install the WSL by following the instructions
at https://learn.microsoft.com/en-us/windows/wsl/install

Then you must install Algan from scratch to your WSL. To do so, open the WSL terminal,
then follow the :doc:`installation instructions <../installation/uv>` , making sure
you follow the instructions for Linux OS.

Once you have successfully installed Algan to your WSL, you can then use it to run python
files on your windows machine. And when run from the WSL, Algan will automatically enable
PyTorch compile.


`pytorch_scatter <https://github.com/rusty1s/pytorch_scatter>`_ is a project
which implements highly optimized pre-compiled wheels for some advanced scatter operations
that the regular PyTorch package lacks.
Algan uses one such scatter operation in its rendering pipeline. If you don't have
pytorch_scatter installed, Algan will perform the operation by combining many un-optimized
operations. If you install pytorch_scatter, Algan will automatically detect it and use it instead,
resulting in 10-30% faster rendering times.

To install it, follow the instructions at `pytorch_scatter <https://github.com/rusty1s/pytorch_scatter>`_
Make sure you select the same hardware version as your original PyTorch installation.
