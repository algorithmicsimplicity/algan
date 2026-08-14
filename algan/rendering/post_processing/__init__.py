"""Passes that run over rendered frames before they reach the video file.

A post-process is a callable ``process(frames, memory=arena)`` that takes the
frame batch as a torch tensor on the render device and returns it modified.
:meth:`~algan.scene.Scene.save_video` accepts any number of them through its
``post_processes`` argument and runs them in order.

The built-in passes are bloom/glow (:mod:`~algan.rendering.post_processing.bloom`),
anti-aliasing (:mod:`~algan.rendering.post_processing.anti_aliasing`) and
tonemapping.
"""
