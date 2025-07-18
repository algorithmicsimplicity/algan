from dataclasses import dataclass

from algan.constants.color import *

__all__ = ['STYLE_DEFAULTS']

@dataclass
class StyleDefaults:
    background_color = BLACK
    frame = background_color
    text_color = WHITE
    buffer = 0.6

STYLE_DEFAULTS = StyleDefaults()
