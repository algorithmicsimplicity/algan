############
Contributing
############

Algan welcomes focused contributions to the renderer, animation system, mobs,
documentation, tests, and developer tooling. Before making a large change, open
or review an issue on the `Algan issue tracker
<https://github.com/algorithmicsimplicity/algan/issues>`__ so the design and
scope can be discussed.

Repository-specific development rules are recorded in ``AGENTS.md``. In
particular, contributors should preserve Scene containment, keep Taichi source
files named ``*_taichi.py``, avoid formatting those kernel files automatically,
and validate rendering changes with a small deterministic scene before running
long benchmarks.

A useful pull request normally contains:

* a clear explanation of the problem and chosen design;
* tests for behavioral changes;
* updated tutorials, reference stubs, and docstrings for public API changes;
* output-parity evidence for renderer optimizations, or an explicit explanation
  of intended visual differences;
* any new settings, fallbacks, or compatibility behavior required by the change.

Development setup and validation
================================

.. toctree::
   :maxdepth: 2

   contributing/development
