For Manim Users
===============

If you already know Manim, you do not need the tutorial series to get started.
:doc:`migrating_from_manim` is a single page that maps what you know onto Algan:
the ``self.play`` model becomes lazy, Scene-owned recording; angles become
degrees; mobjects become Mobs. It ends with a step-by-step sequence for porting a
larger project.

Two things worth knowing before you read it:

* Algan records animations rather than playing them. Nothing renders until
  :meth:`~algan.scene.Scene.save_video` or
  :meth:`~algan.scene.Scene.save_frame` materializes the timeline.
* Manim's geometry is still available. Algan bundles Manim and exposes
  :class:`~algan.mobs.manim_mob.ManimMob` plus compatibility classes for ``Axes``,
  ``NumberPlane``, ``Brace`` and the rest, so a diagram you already have keeps
  working. :doc:`../new_user_tutorials/importing_from_manim` covers that route.

.. toctree::
   :caption: Table of Contents
   :maxdepth: 2

   migrating_from_manim
