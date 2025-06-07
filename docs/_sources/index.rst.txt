Algan
=====

Algan (ALGorithmic ANimation) is an animation library inspired by `Manim <https://docs.manim.community/en/stable/>`_.
Manim is a popular Python library for animating mathematical concepts, featuring
a simple and intuitive interface and an expansive collection of built in functionality.
However, Manim has some key design flaws that make it unsuited for orchestrating
complex animations involving many moving parts, and unsuited for developing modular animation software.
Algan seeks to improve upon Manim by providing a similarly simple interface, but a much
more robust and scalable animation framework.

To this end, Algan features a fully complete animation and rendering backend, built from
the ground up in `Pytorch <https://pytorch.org/>`_. Because the entire pipeline is purpose-built for algorithmic animating,
Algan enjoys several key benefits over Manim:

- Algan does not render animations as they are created in the program, but instead records
  all defined animations until a command to render is given. This allows programs to define animations
  out-of-order (i.e. not in the order they appear in the video timeline). This makes animating some complex scenes
  far simpler.
- Algan features a robust AnimationContext system to control animations. This system makes writing modular animation code trivial.
- Because both the animation and rendering pipelines are built in Pytorch, Algan allows for the use of GPU
  acceleration during both rendering AND animating. Note that while Manim does support GPU acceleration for rendering,
  it does not for animating. This means that for scenes involving the animation of many thousands of actors, Algan will
  be far quicker.
- The Algan rendering pipeline is fully customizable. If you need to implement a new type of rendering operation,
  or unique lighting and shading, you can do so by simply writing a Pytorch program to implement them. You don't need
  to bother trying to learn how to use shader languages like OpenGL, just write your rendering pipeline in good old
  familiar Python with numpy-like arrays.

First Steps
-----------

Are you new to Algan and are looking for where to get started? Then you are
in the right place!

- The :doc:`Installation <installation>` section has the latest and
  up-to-date installation instructions for Windows, macOS, and Linux.
- In our :doc:`Tutorials <new_user_tutorials/index>` section you will find a
  collection of resources that will teach you how to use Algan.


Finding Help
------------

Are you struggling with installing or using Algan? Don't worry, we've all been
there. Here are some good resources to help you out:

- Perhaps your problem is one that occurs frequently, then chances are it is
  addressed in our :doc:`collection of FAQs <faq/index>`.
- If you are looking for information on some specific class, look for it
  in the :doc:`reference manual <reference>` and/or use the search feature
  of the documentation.
- Still no luck? Then you are welcome to ask the community for help, together
  we usually manage to find a solution for your problem! Consult the
  :doc:`FAQ page on getting help <faq/help>` for instructions.

Index
-----

.. toctree::
   :maxdepth: 3

   installation
   tutorials_guides
   reference

.. image:: _static/crowdin-badge.svg
  :align: center
  :alt: Localized with Crowdin
  :target: https://translate.manim.community
