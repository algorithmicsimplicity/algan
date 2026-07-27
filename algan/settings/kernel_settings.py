"""Render-kernel registry (runtime service, not user configuration)."""


class KernelRegistry:
    def __init__(self):
        self.render_kernel = None


KERNEL_REGISTRY = KernelRegistry()
# Compatibility alias.
KERNEL_SETTINGS = KERNEL_REGISTRY

__all__ = ["KernelRegistry", "KERNEL_REGISTRY", "KERNEL_SETTINGS"]
