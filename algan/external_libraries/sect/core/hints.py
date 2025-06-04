from typing import (Callable,
                    TypeVar)

from algan.external_libraries.ground.base import Orientation
from algan.external_libraries.ground.hints import Point

Domain = TypeVar('Domain')
Orienteer = Callable[[Point, Point, Point], Orientation]
