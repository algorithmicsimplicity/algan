from typing import (Callable,
                    Tuple)

from algan.external_libraries.ground.base import (Location,
                         Relation)
from algan.external_libraries.ground.hints import (Point,
                          Segment)

PointInCircleLocator = Callable[[Point, Point, Point, Point], Location]
SegmentEndpoints = Tuple[Point, Point]
SegmentContainmentChecker = Callable[[Segment, Point], bool]
SegmentsRelater = Callable[[Segment, Segment], Relation]
