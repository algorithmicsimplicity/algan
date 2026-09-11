"""Immutable, tensor-free decisions shared by batch preflight and execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class WavefrontPolicy:
    """Allocation and specialization decisions for deterministic ray state."""

    refraction: bool
    ior_stack: bool
    state_scalar_width: int
    pool_ratio: int
    tile_rays: int
    memory_trim: bool
    opaque_closest: bool
    opaque_prepass: bool
    fused_generation: bool
    triangle_pipeline_mask: int
    triangle_aa: bool


@dataclass(frozen=True, slots=True)
class BatchExecutionPolicy:
    """One prepared batch's route, AA, shading, and continuation contract.

    Resolve on the render thread after scene preparation, then reuse through
    preflight, upload and all frame chunks. Do not cache on a module or a live
    scene across render jobs. BVH objects are deliberately absent: an on-demand
    build may replace them; their concrete type remains the launch authority.
    """

    primary_route: Literal["analytic_sheets", "classic_wavefront", "path_tracer"]
    fallback_reasons: tuple[str, ...]
    samples_per_pixel: int
    requested_aa: int
    effective_aa: int
    inplace_aa: bool
    fragment_shading: bool
    shadow_mode: int
    shadows: bool
    custom_scatter: bool
    lights_extended: bool
    has_triangles: bool
    has_beziers: bool
    max_bounces: int
    near_clip: float
    far_clip: float
    wavefront: WavefrontPolicy | None

    @property
    def analytic_raster(self):
        return self.primary_route == "analytic_sheets"

    @property
    def frame_scale(self):
        """Output buffer scale, distinct from in-place jittered sample count."""
        return 1 if self.inplace_aa else self.effective_aa

    @property
    def kernel_aa(self):
        return self.effective_aa if self.inplace_aa else 1
