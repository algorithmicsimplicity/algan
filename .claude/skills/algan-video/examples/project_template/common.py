"""Shared, style-free helpers for a multi-scene narrated project.

Everything here is mechanics. The settings values are placeholders: replace them with
the user's requested resolution/frame rate. Put the user's own palette, fonts and
reusable props in this module (or a sibling one) as the project needs them.
"""

from algan import *

# Output formats. Fix the aspect ratio before authoring: screen-relative layout and
# camera framing are computed against it. Keep the draft at the same aspect ratio.
DRAFT = PREVIEW                                  # e.g. PREVIEW.set(resolution=(540, 960)) for 9:16
FINAL = HD                                       # e.g. HD.set(resolution=(1080, 1920))

# None keeps Algan's default post-processing. A tuple replaces it for every video,
# still and profile the Project renders, e.g.
# (partial(bloom_filter, glow_spread=0.02),) with functools.partial and
# algan.rendering.post_processing.bloom.bloom_filter.
POST = None


def say(text, hold=1.0):
    """A Speech block with an explicit hold after the clip (Speech's default is 1 s)."""
    return Speech(text, wait_at_end=hold)


# Camera moves need no helper: ``Scene.get_camera().fly_to(position, look_at=target)``
# moves position and aim together with a level horizon (``via=`` / ``look_at_via=``
# curve the paths). Give it a timing context like any other change, and set the
# opening shot inside ``Off()``.
