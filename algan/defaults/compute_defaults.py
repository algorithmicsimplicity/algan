from dataclasses import dataclass


@dataclass
class ComputeDefaults:
    compiled: bool = False


COMPUTE_DEFAULTS = ComputeDefaults()