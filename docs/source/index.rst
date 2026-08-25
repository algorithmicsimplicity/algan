===========
About Algan
===========

Algan (ALGorithmic ANimation) is a Python library for making 2-D and 3-D
animations. Algan is inspired by `Manim
<https://docs.manim.community/en/stable/>`_, and aims to keep the same ease of use,
while providing a full-fledged GPU-oriented raytraced renderer for complex moving 3-D
scenes and realistic lighting.

Key capabilities include:

* **Lazy animation recording.** Algan separates scene authoring
  from rendering, meaning that a scene can be authored once and rendered many times,
  and scenes can be authored out of animation order.
* **Composable animation contexts.** ``Seq``, ``Sync``, ``Lag``, and ``Off``
  are context wrappers which automatically compose with eachother,
  meaning animation code can be written without thinking about how it will be
  used in a video. This promotes writing modular and reusable animation code.
* **2-D and 3-D geometry.** Native cubic-Bezier shapes, text and TeX, triangle
  surfaces, point clouds, imported 3D asset models, and compatible Manim vector objects
  are handled natively.
* **Physically based rendering.** Materials, fragment shading, lighting,
  shadows, reflection/refraction, and ray-tracing paths are supported.
  Custom fragment shading and scattering can be written as Taichi functions.
* **Synchronized audio.** Audio and speech contexts are recorded on the same
  Scene timeline as visual animation, making it easy to line up animations
  with voice-over narration.

First Steps
===========

* Follow :doc:`installation` to install Algan and its system dependencies.
* Learn how to use Algan at :doc:`new_user_tutorials/getting_started` and subsequent tutorials.
* Browse the catalogue of available objects at :doc:`galleries/mob_gallery`, and
  the ready-made animations at :doc:`galleries/built_in_animations`.
* Manim users can start with :doc:`manim_migration_guide`.

Getting Help
============

* For individual classes, methods, and functions, check the
  :doc:`reference manual <reference>` or use the documentation search.
* Found a bug or have a suggestion? Open an issue on our `GitHub issue tracker
  <https://github.com/algorithmicsimplicity/algan/issues>`_.
* Join our `Discord server <https://discord.gg/NvarFmvXKm>`_ to chat with the
  developers and fellow Algan users.

Documentation Index
===================

.. toctree::
   :maxdepth: 3

   installation
   tutorials_guides
   reference
   faq/index
   contributing
   changelog
