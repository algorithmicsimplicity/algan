from __future__ import annotations

import dataclasses
import difflib
import typing
from contextlib import contextmanager
from copy import deepcopy

from algan.errors import AlganConfigurationError


def _is_special_var(annotation) -> bool:
    """Return whether an annotation is not a normal instance setting field."""
    if isinstance(annotation, str):
        return any(
            marker in annotation for marker in ("ClassVar", "InitVar", "KW_ONLY")
        )

    if annotation is getattr(dataclasses, "KW_ONLY", None):
        return True

    origin = typing.get_origin(annotation) or annotation
    if origin is typing.ClassVar:
        return True

    init_var = getattr(dataclasses, "InitVar", None)
    return bool(
        origin is init_var or init_var is not None and isinstance(annotation, init_var)
    )


class Settings:
    """Base class for validated settings sections and immutable presets.

    Normal settings sections are mutable: :meth:`set` validates a complete
    replacement value and then updates the existing object in place.  This
    keeps section identity stable for code that imported ``SETTINGS`` once.

    Presets use the same concrete classes but are marked immutable.  Calling
    :meth:`set` on a preset returns another preset and leaves the original
    untouched.  Direct assignment to a preset field is rejected.
    """

    _is_preset = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        def make_setter(field_name: str):
            def setter(self, value):
                return self.set(**{field_name: value})

            setter.__name__ = f"set_{field_name}"
            setter.__doc__ = f"Set '{field_name}', or return a modified preset copy."
            return setter

        for base in reversed(cls.__mro__):
            for name, annotation in getattr(base, "__annotations__", {}).items():
                if _is_special_var(annotation) or name.startswith("_"):
                    continue
                setter_name = f"set_{name}"
                if not hasattr(cls, setter_name):
                    setattr(cls, setter_name, make_setter(name))

    def __setattr__(self, name, value):
        declared = type(self)._declared_field_names()
        if getattr(self, "_is_preset", False) and name in declared:
            raise AlganConfigurationError(
                f"{type(self).__name__} preset fields are immutable; use "
                f"preset.set({name}=...) to create a modified copy"
            )
        if not name.startswith("_") and name not in declared:
            # Without this, ``SETTINGS.video.fps = 60`` would quietly attach a
            # junk attribute and the real setting would keep its old value.
            type(self)._check_keys({name: value})
        object.__setattr__(self, name, value)

    @classmethod
    def _declared_field_names(cls) -> set[str]:
        names: set[str] = set()
        for base in reversed(cls.__mro__):
            for name, annotation in getattr(base, "__annotations__", {}).items():
                if not _is_special_var(annotation) and not name.startswith("_"):
                    names.add(name)
        return names

    @classmethod
    def _check_keys(cls, kwargs):
        valid = cls._declared_field_names()
        unknown = [name for name in kwargs if name not in valid]
        if unknown:
            name = unknown[0]
            suggestion = difflib.get_close_matches(name, sorted(valid), n=1)
            hint = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
            raise AlganConfigurationError(
                f"Unknown {cls.__name__} setting '{name}'.{hint}"
            )

    def _validate(self):
        """Hook for non-dataclass subclasses that need semantic validation."""
        return None

    def _validated_replacement(self, **kwargs):
        type(self)._check_keys(kwargs)
        if dataclasses.is_dataclass(self):
            try:
                return dataclasses.replace(self, **kwargs)
            except (TypeError, ValueError) as exc:
                raise AlganConfigurationError(str(exc)) from exc

        result = deepcopy(self)
        object.__setattr__(result, "_is_preset", False)
        for key, value in kwargs.items():
            object.__setattr__(result, key, value)
        result._validate()
        return result

    def set(self, source=None, **kwargs):
        """Validate and apply settings.

        ``source`` may be another instance of the same settings class. Its
        fields are copied first, then any keyword overrides are applied.
        Mutable sections are changed in place and returned. Presets return a
        new immutable copy.
        """
        values = {}
        if source is not None:
            if type(source) is not type(self):
                raise AlganConfigurationError(
                    f"{type(self).__name__}.set expected another "
                    f"{type(self).__name__} object, got "
                    f"{type(source).__name__}"
                )
            values.update(source.to_dict())
        values.update(kwargs)
        replacement = self._validated_replacement(**values)
        if getattr(self, "_is_preset", False):
            object.__setattr__(replacement, "_is_preset", True)
            return replacement

        for name in type(self)._declared_field_names():
            object.__setattr__(self, name, deepcopy(getattr(replacement, name)))
        return self

    @property
    def is_preset(self) -> bool:
        return bool(getattr(self, "_is_preset", False))

    def clone(self, *, preset: bool | None = None):
        result = deepcopy(self)
        if preset is not None:
            object.__setattr__(result, "_is_preset", bool(preset))
        return result

    def as_preset(self):
        return self.clone(preset=True)

    def as_mutable(self):
        return self.clone(preset=False)

    def to_dict(self) -> dict:
        return {
            name: deepcopy(getattr(self, name))
            for name in type(self)._declared_field_names()
        }

    @contextmanager
    def override(self, **kwargs):
        """Temporarily override a mutable section and restore it reliably."""
        if self.is_preset:
            raise AlganConfigurationError(
                "Cannot override an immutable preset in place"
            )
        previous = self.to_dict()
        self.set(**kwargs)
        try:
            yield self
        finally:
            self.set(**previous)
