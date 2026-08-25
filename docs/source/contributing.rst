============
Contributing
============

We welcome contributions to Algan! Whether you want to fix a bug, improve
documentation, add a new Mob, or optimize GPU render kernels, we'd love your
help.

If you are planning a large change or new feature, it is always a good idea to
open an issue on the `GitHub issue tracker
<https://github.com/algorithmicsimplicity/algan/issues>`_ or mention it on
`Discord <https://discord.gg/NvarFmvXKm>`_ first, so we can align on design and
approach before you write lots of code.

What makes a great pull request:

* **Clear motivation and approach:** A short summary of what you are solving and
  why you chose this implementation.
* **Tests:** Unit or render tests covering any behavioral changes.
* **Documentation:** Updated docstrings, tutorials, and examples for any public
  API changes.
* **Deterministic validation:** For renderer optimizations or fixes, verify
  rendered outputs on a small scene before running full benchmarks.

Getting Started with Development
================================

To set up a local development environment, install system dependencies, run the
test suite, and build the documentation locally, follow the guide below:

.. toctree::
   :maxdepth: 2

   contributing/development
