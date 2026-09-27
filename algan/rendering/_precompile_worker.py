"""Entry point of a precompile worker process (``rendering/kernel_precompile.py``).

A module of its own because ``python -m algan.rendering.kernel_precompile``
would execute a second copy of that module as ``__main__`` beside the one
``import algan`` has already loaded, and the two would disagree about every
piece of module state.
"""

from __future__ import annotations

from algan.rendering.kernel_precompile import worker_main

worker_main()
