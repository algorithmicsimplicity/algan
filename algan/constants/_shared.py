"""Internal: the pristine values of Algan's shared constant tensors.

``BLUE``, ``ORIGIN``, ``RIGHT`` and the rest are ordinary tensors, one object
each for the whole process, and every Mob, animation and script that names one
holds that same object. That is what makes them cheap to pass around and to
compare, and it is also why an in-place edit through an alias reaches everyone:
``o = ORIGIN; o += 1.0`` moves ``ORIGIN`` itself, for the rest of the script
and -- on the warm daemon, which serves one script after another in one
process -- for the next script too.

Making the constants immutable would change their type on the engine's hot
paths, so this module keeps them as they are and records a copy of each when
it is defined. :func:`restore_shared_constants` writes those copies back in
place, which is what a process that runs several scripts calls between them.
The documented channel setters (``c.glow = 2`` on a :class:`Color`) refuse a
shared constant outright, through :func:`is_shared_constant`; in-place tensor
arithmetic cannot be refused without wrapping every tensor operation, so a
single script that writes ``ORIGIN += ...`` still changes its own ``ORIGIN``.
"""

from __future__ import annotations

import torch

# (live constant, pristine copy) for every registered constant, in the order
# they were registered. Names do not matter here: ``IN is INWARD``, and the
# same object is recorded once.
_PRISTINE: list[tuple[torch.Tensor, torch.Tensor]] = []
_REGISTERED_IDS: set[int] = set()
# The storages the constants live in. A channel setter checks against these
# rather than against object identity, so a view or ``as_subclass`` alias of a
# constant -- the same memory under another Python object -- is refused too.
_SHARED_STORAGES: set[int] = set()


def _register_shared_constants(namespace: dict) -> None:
    """Record every tensor constant in a module namespace as shared.

    Called at the bottom of the modules that define them, before any user code
    can have run. Only upper-case names are constants.
    """
    for name, value in namespace.items():
        if not name.isupper() or not isinstance(value, torch.Tensor):
            continue
        if id(value) in _REGISTERED_IDS:
            continue
        _REGISTERED_IDS.add(id(value))
        _PRISTINE.append((value, value.detach().clone()))
        _SHARED_STORAGES.add(value.untyped_storage().data_ptr())


def is_shared_constant(tensor: torch.Tensor) -> bool:
    """Whether ``tensor`` is, or shares memory with, a shared constant."""
    return tensor.untyped_storage().data_ptr() in _SHARED_STORAGES


def restore_shared_constants() -> None:
    """Put every shared constant back to the value it was defined with, in place.

    The objects keep their identity, so every module and Mob that holds one
    sees the restored value. Cheap -- a few hundred five-element copies -- and
    a no-op for a constant nothing changed.
    """
    for live, pristine in _PRISTINE:
        if (
            is_shared_constant(live)
            and live.shape == pristine.shape
            and live.dtype == pristine.dtype
            and live.device == pristine.device
        ):
            if not torch.equal(live.data, pristine):
                live.data.copy_(pristine)
            continue
        # ``BLUE.data = ...`` or a ``resize_`` replaced the storage itself.
        # Copying into the replacement would write into memory the script may
        # still hold under another name, so give the constant a fresh storage
        # of its own instead, and record that as the shared one.
        live.data = pristine.clone()
        _SHARED_STORAGES.add(live.untyped_storage().data_ptr())
