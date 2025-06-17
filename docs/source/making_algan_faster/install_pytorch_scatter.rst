==========================
Installing pytorch_scatter
==========================

`pytorch_scatter <https://github.com/rusty1s/pytorch_scatter>`_ is a project
which implements highly optimized pre-compiled wheels for some advanced scatter operations
that the regular PyTorch package lacks.
Algan uses one such scatter operation in its rendering pipeline. If you don't have
pytorch_scatter installed, Algan will perform the operation by combining many un-optimized
operations. If you install pytorch_scatter, Algan will automatically detect it and use it instead,
resulting in 10-30% faster rendering times.

To install it, follow the instructions at `pytorch_scatter <https://github.com/rusty1s/pytorch_scatter>`_
Make sure you select the same hardware version as your original Pytorch installation.
