About Algan
===========

Algan (ALGorithmic ANimation) is a Python library for building 2-D and 3-D
animations. Algan is inspired by `Manim
<https://docs.manim.community/en/stable/>`_, and aims to keep the same ease of use,
while providing full-fledged GPU-oriented raytraced renderer for complex moving 3-D
scenes and realistic lighting.

Key capabilities include:

* **Lazy animation recording.** Algan separates scene authoring
  from rendering, meaning that a scene can be authored once and rendered many times,
  and scenes can be authored out of source order.
* **Composable animation contexts.** ``Seq``, ``Sync``, ``Lag``, and ``Off``
  are context wrappers which automatically compose with eachother,
  meaning animation code can be written without thinking about how it will be
  used in a video. This promotes writing modular, reusable animation code.
* **2-D and 3-D geometry.** Native cubic-Bezier shapes, text and TeX, triangle
  surfaces, point clouds, imported 3D asset models, and compatible Manim vector objects
  share one scene model.
* **Physically based rendering.** Materials, fragment stages, lighting,
  shadows, reflection/refraction, hybrid rasterization, and ray-tracing paths
  are implemented through Torch and Taichi.
* **Synchronized audio.** Audio and speech contexts are recorded on the same
  Scene timeline as visual animation, making it easy to line up animations
  with voice-over narration.

First steps
===========

* Follow :doc:`installation` to install Algan and its system dependencies.
* Build your first animation in :doc:`new_user_tutorials/getting_started`.
* Learn how animations are composed in
  :doc:`new_user_tutorials/controlling_animations`.
* Browse what you can put on screen in :doc:`new_user_tutorials/mob_gallery`.
* Move into three dimensions with :doc:`new_user_tutorials/three_d_basics`.
* Configure quality, paths, memory, and ray tracing through
  :doc:`advanced_user_tutorials/settings`, and keep renders fast with
  :doc:`advanced_user_tutorials/performance_and_quality`.
* Manim users can start with :doc:`manim_user_quickstart/index`.

Finding help
============

For API details, use the :doc:`reference manual <reference>` and documentation
search. Installation and usage bugs can be reported on the Algan GitHub issue
tracker. Include a minimal script, platform, Python version, Torch/Taichi device,
and the complete traceback or renderer log.

Documentation index
===================

.. toctree::
   :maxdepth: 3

   installation
   tutorials_guides
   reference
   faq/index
   contributing
   changelog
