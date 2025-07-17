from algan.constants.color import *


RIGHT = torch.tensor((1,0,0), dtype=torch.get_default_dtype())
LEFT = -RIGHT
UP = torch.tensor((0,1,0), dtype=torch.get_default_dtype())
DOWN = -UP
IN = torch.tensor((0,0,1), dtype=torch.get_default_dtype())
OUT = -IN

DEFAULT_BASIS = torch.stack((RIGHT, UP, OUT))

ORIGIN = torch.zeros_like(OUT)
CAMERA_ORIGIN = ORIGIN + OUT * 7

NUM_DIMENSIONS = 3
