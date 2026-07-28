About Algan
===========

Algan (ALGorithmic ANimation) is a Python library for building 2-D and 3-D
explanatory animations. It is inspired by `Manim
<https://docs.manim.community/en/stable/>`_, but uses a Scene-contained lazy
animation system and a GPU-oriented renderer designed for complex moving 3-D
scenes.

Authoring code records animations instead of rendering them immediately. Each
:class:`~algan.scene.Scene` owns its mobs, camera, lights, animation timeline,
animation contexts, audio state, and render loop. At output time Algan
materializes the Scene in batches of frame times and renders the resulting
triangles and cubic-Bezier circuits.

Key capabilities include:

* **Lazy animation recording.** Animations can be rescaled, nested, synchronized,
  or written out of source order before any frame is computed.
* **Independent Scenes.** Timeline, animation, and audio managers are regular
  per-Scene objects; only active-Scene selection is process-global.
* **Composable animation contexts.** ``Seq``, ``Sync``, ``Lag``, and ``Off``
  provide structured timing without an imperative ``play`` loop.
* **2-D and 3-D geometry.** Native cubic-Bezier shapes, text and TeX, triangle
  surfaces, point clouds, imported models, and compatible Manim vector objects
  share one scene model.
* **Physically based rendering.** Materials, fragment stages, lighting,
  shadows, reflection/refraction, hybrid rasterization, and ray-tracing paths
  are implemented through Torch and Taichi.
* **Unified output APIs.** ``Scene.save_frame`` and ``Scene.save_video`` share
  path resolution and accept explicit video presets.
* **Synchronized audio.** Audio and speech contexts are recorded on the same
  Scene timeline as visual animation.

First steps
===========

* Follow :doc:`installation` to install Algan and its system dependencies.
* Build your first animation in :doc:`new_user_tutorials/getting_started`.
* Learn Scene ownership and output behavior in
  :doc:`new_user_tutorials/scenes_and_rendering`.
* Configure quality, paths, memory, and ray tracing through
  :doc:`new_user_tutorials/settings`.
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
