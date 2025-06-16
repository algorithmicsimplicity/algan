#!/usr/bin/env python
from __future__ import annotations


# isort: off

# Importing the config module should be the first thing we do, since other
# modules depend on the global config dict for initialization.
from ._config import *

# many scripts depend on this -> has to be loaded first

# isort: on
import numpy as np

from .constants import *
from .mobject.frame import *
from .mobject.geometry.arc import *
from .mobject.geometry.boolean_ops import *
from .mobject.geometry.labeled import *
from .mobject.geometry.line import *
from .mobject.geometry.polygram import *
from .mobject.geometry.shape_matchers import *
from .mobject.geometry.tips import *
from .mobject.graph import *
from .mobject.graphing.coordinate_systems import *
from .mobject.graphing.functions import *
from .mobject.graphing.number_line import *
from .mobject.graphing.probability import *
from .mobject.graphing.scale import *
from .mobject.logo import *
from .mobject.matrix import *
from .mobject.mobject import *
from .mobject.opengl.dot_cloud import *
from .mobject.opengl.opengl_point_cloud_mobject import *
from .mobject.svg.brace import *
from .mobject.svg.svg_mobject import *
from .mobject.text.tex_mobject import *
from .mobject.text.code_mobject import *
from .mobject.text.numbers import *
from .mobject.table import *
#from .mobject.text.text_mobject import *
from .mobject.three_d.polyhedra import *
from .mobject.three_d.three_d_utils import *
from .mobject.three_d.three_dimensions import *
from .mobject.types.image_mobject import *
from .mobject.types.point_cloud_mobject import *
from .mobject.types.vectorized_mobject import *
from .mobject.value_tracker import *
from .mobject.vector_field import *

