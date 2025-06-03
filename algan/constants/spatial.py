from algan.constants.color import *


#CAMERA_ORIGIN = torch.tensor((0,0,0), dtype=torch.get_default_dtype())
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

RADIANS_TO_DEGREES = 180 / torch.pi
DEGREES_TO_RADIANS = torch.pi / 180
