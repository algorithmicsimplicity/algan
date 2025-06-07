from typing import overload, Tuple

import torch
import re

from algan import DEFAULT_DEVICE

re_hex = re.compile("((?<=#)|(?<=0x))[A-F0-9]{6,8}", re.IGNORECASE)

class Color(torch.Tensor):
    def __new__(cls, rgb:str|Tuple[float], glow=0, opacity=1, *args, **kwargs):
        if isinstance(rgb, str):
            hex_code = re_hex.search(rgb).group()
            if len(hex_code) == 6:
                hex_code += "00"
            tmp = int(hex_code, 16)
            rgb = (((tmp >> 24) & 0xFF) / 255,
                   ((tmp >> 16) & 0xFF) / 255,
                   ((tmp >> 8) & 0xFF) / 255)
        return super().__new__(cls, (*rgb, glow, opacity), *args, **kwargs).to(DEFAULT_DEVICE)

    def __init__(self, rgb, glow=0, opacity=1, *args, **kwargs):
        pass#super().__init__((red, green, blue, glow, opacity))

    @property
    def rgb(self):
        return self.data[...,:3]

    @rgb.setter
    def rgb(self, value):
        self.data[...,:3] = value

    def mult_rgb(self, other):
        orgb = other.rgb if isinstance(other, Color) else other
        out = self.new_empty()
        out.data = self.data.clone()
        out.rgb = self.rgb * orgb
        return out

    def convert_to_uint8(self):
        return (self * 255).to(torch.uint8)

    def new_empty(self, *args, **kwargs):
        return Color((0,0,0), **kwargs)

    @staticmethod
    def add_defaults(color):
        if color.shape[-1] < 4:
            color = torch.cat((color, torch.ones_like(color[...,:1])), -1)
        if color.shape[-1] < 5:
            color = torch.cat((color[...,:-1], torch.zeros_like(color[...,:1]), color[...,-1:]), -1)
        return color


def color_to_texture_map(color):
    return lambda coords: color.view(([1] * (coords.dim()-1)) + [-1]).expand(list(coords.shape[:-1]) + [-1])

GLOW = Color((0,0,0),1,0)

#REDS = [Color(*[__ / 255 for __ in _]) for _ in ((249, 113, 123), (225, 69, 81), (213, 27, 41), (172, 13, 24), (139, 0, 10))]
#YELLOWS = [Color(*[__ / 255 for __ in _]) for _ in ((255,230, 116), (231, 202, 71), (219, 184, 28), (177, 147, 13), (142, 116, 0))]
#BLUES = [Color(*[__ / 255 for __ in _]) for _ in ((110, 92, 178), (82, 62, 159), (59+20, 35+20, 151+20), (43, 22, 122), (20, 11, 98))]
#GREENS = [Color(*[__ / 255 for __ in _]) for _ in ((112, 212, 96), (77, 191, 59), (45, 181, 23), (29, 146, 11), (16, 118, 0))]

GRAY_A = Color("#DDDDDD")
GREY_A = Color("#DDDDDD")
GRAY_B = Color("#BBBBBB")
GREY_B = Color("#BBBBBB")
GRAY_C = Color("#888888")
GREY_C = Color("#888888")
GRAY_D = Color("#444444")
GREY_D = Color("#444444")
GRAY_E = Color("#222222")
GREY_E = Color("#222222")
BLACK = Color("#000000")
WHITE = Color("#FFFFFF")
LIGHTER_GRAY = Color("#DDDDDD")
LIGHTER_GREY = Color("#DDDDDD")
LIGHT_GRAY = Color("#BBBBBB")
LIGHT_GREY = Color("#BBBBBB")
GRAY = Color("#888888")
GREY = Color("#888888")
DARK_GRAY = Color("#444444")
DARK_GREY = Color("#444444")
DARKER_GRAY = Color("#222222")
DARKER_GREY = Color("#222222")
BLUE_A = Color("#C7E9F1")
BLUE_B = Color("#9CDCEB")
BLUE_C = Color("#58C4DD")
BLUE_D = Color("#29ABCA")
BLUE_E = Color("#236B8E")
PURE_BLUE = Color("#0000FF")
BLUE = Color("#58C4DD")
DARK_BLUE = Color("#236B8E")
TEAL_A = Color("#ACEAD7")
TEAL_B = Color("#76DDC0")
TEAL_C = Color("#5CD0B3")
TEAL_D = Color("#55C1A7")
TEAL_E = Color("#49A88F")
TEAL = Color("#5CD0B3")
GREEN_A = Color("#C9E2AE")
GREEN_B = Color("#A6CF8C")
GREEN_C = Color("#83C167")
GREEN_D = Color("#77B05D")
GREEN_E = Color("#699C52")
PURE_GREEN = Color("#00FF00")
GREEN = Color("#83C167")
YELLOW_A = Color("#FFF1B6")
YELLOW_B = Color("#FFEA94")
YELLOW_C = Color("#FFFF00")
YELLOW_D = Color("#F4D345")
YELLOW_E = Color("#E8C11C")
YELLOW = Color("#FFFF00")
GOLD_A = Color("#F7C797")
GOLD_B = Color("#F9B775")
GOLD_C = Color("#F0AC5F")
GOLD_D = Color("#E1A158")
GOLD_E = Color("#C78D46")
GOLD = Color("#F0AC5F")
RED_A = Color("#F7A1A3")
RED_B = Color("#FF8080")
RED_C = Color("#FC6255")
RED_D = Color("#E65A4C")
RED_E = Color("#CF5044")
PURE_RED = Color("#FF0000")
RED = Color("#FC6255")
MAROON_A = Color("#ECABC1")
MAROON_B = Color("#EC92AB")
MAROON_C = Color("#C55F73")
MAROON_D = Color("#A24D61")
MAROON_E = Color("#94424F")
MAROON = Color("#C55F73")
PURPLE_A = Color("#CAA3E8")
PURPLE_B = Color("#B189C6")
PURPLE_C = Color("#9A72AC")
PURPLE_D = Color("#715582")
PURPLE_E = Color("#644172")
PURPLE = Color("#9A72AC")
PINK = Color("#D147BD")
LIGHT_PINK = Color("#DC75CD")
ORANGE = Color("#FF862F")
LIGHT_BROWN = Color("#CD853F")
DARK_BROWN = Color("#8B4513")
GRAY_BROWN = Color("#736357")
GREY_BROWN = Color("#736357")