"""The base class every settings section is built from.

:class:`Settings` gives a dataclass of configuration fields three things Algan
relies on: ``set()``, which mutates in place so a section keeps its identity;
``override()``, a context manager that restores the previous values on exit --
including when the body raises; and validation at construction, so a bad value is
rejected where it is written rather than deep inside a render.

**Assignment is ``set()``.** ``SETTINGS.video.frames_per_second = 60`` and
``SETTINGS.video.set(frames_per_second=60)`` are the same operation, down to
validation and normalization, so the shorter spelling cannot be the one that
skips the checks. ``RayTracingSettings`` -- which is not a subclass, and has by
far the most fields -- has always routed assignment this way; the dataclass
sections do too.

**A write only touches what changed.** ``set()`` compares each field against
the validated replacement by *identity* and leaves the untouched ones exactly
as they were. Without that, setting any one field replaced every other with an
equal-but-different deepcopy, so anything holding a ``Color`` from
``SETTINGS.style`` by reference silently stopped tracking the setting.

It also supports **immutable presets**. A frozen section (``HD``, ``PREVIEW``)
answers ``set()`` with a modified copy instead of mutating, which is what lets
``HD.set(frames_per_second=60)`` be a safe expression.

Sections declare initialization-only fields so that assigning one raises a
message naming the environment variable to set instead of a generic unknown-key
error.
"""

from __future__ import annotations

import dataclasses
import difflib
import typing
from contextlib import contextmanager
from copy import deepcopy

from algan.errors import AlganConfigurationError

#: Sentinel for "this field is not set yet", so the identity test in
#: :meth:`Settings.set` cannot accidentally match a real value the way ``None``
#: would (``None`` is a legitimate setting value -- ``available_memory_override``
#: and ``default_material`` both default to it).
_MISSING = object()


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
    Assigning a field routes through :meth:`set`, so the two spellings validate
    and normalize identically.

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
        if name in declared and self.__dict__.keys() >= declared:
            # Assignment and ``set`` are the same operation, so a field cannot
            # be written unvalidated and unnormalized by picking the shorter
            # spelling: ``SETTINGS.video.frames_per_second = 0`` used to store
            # a zero that only failed much later, deep in a render.
            # ``RayTracingSettings`` -- the largest section, and the one users
            # touch most -- has always worked this way; this brings the
            # dataclass sections in line with it.
            #
            # The subset test is what keeps construction working: the dataclass
            # ``__init__`` assigns fields one at a time, and ``set`` replaces
            # the whole object, which a half-built one cannot survive. Once
            # every declared field exists, the object is complete.
            # ``__post_init__`` normalizes through ``object.__setattr__`` and so
            # never lands here, and a preset has already been refused above.
            self.set(**{name: value})
            return
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
                f"Unknown {cls.__name__} setting '{name}'.{hint} "
                f"Valid settings are: {', '.join(sorted(valid))}."
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

        Every field is validated -- the replacement is built by re-running the
        section's own construction -- but only the fields whose value actually
        changed are written back, so an unrelated field keeps its identity.
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
            new = getattr(replacement, name)
            if new is getattr(self, name, _MISSING):
                # Untouched. ``dataclasses.replace`` passes the fields the
                # caller did not name straight through, so identity here means
                # the value is the one already stored -- writing a deepcopy of
                # it back would swap a live object for an equal stranger. That
                # is not hypothetical: ``SETTINGS.style.set(buffer=0.7)`` used
                # to replace ``background_color`` too, so anything holding that
                # Color by reference silently stopped tracking the setting.
                #
                # Identity, deliberately, not equality: a ``Color`` is a torch
                # tensor, where ``==`` is elementwise and ``bool()`` of the
                # result raises. Identity can only be true when nothing
                # changed, so this narrows the write set without ever skipping
                # a real change.
                continue
            object.__setattr__(self, name, deepcopy(new))
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
