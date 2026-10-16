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

A section may also declare **aliases** with :func:`settings_aliases` -- a second
spelling for a field, honoured everywhere the declared name is (construction,
``set()``, ``override()``, attribute read and attribute write). An alias is not a
second setting: ``to_dict`` and snapshots always answer with the declared name,
so a save/restore round-trips through one spelling no matter which one was
written.
"""

from __future__ import annotations

import dataclasses
import difflib
import functools
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

    #: ``alias -> declared field name``, filled in by :func:`settings_aliases`
    #: and empty for a section that declares none.
    _ALIASES: dict[str, str] = {}

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
        # Resolved up front so an alias is the field for every branch below --
        # a preset refuses it, and a mutable section validates it, exactly as
        # the declared spelling does.
        name = type(self)._ALIASES.get(name, name)
        declared = type(self)._declared_field_names()
        if getattr(self, "_is_preset", False) and name in declared:
            raise AlganConfigurationError(
                f"{type(self).__name__} preset fields are immutable; use "
                f"preset.set({name}=...) to create a modified copy"
            )
        if not name.startswith("_") and name not in declared:
            # Without this, ``SETTINGS.video.frame_rate = 60`` would quietly
            # attach a junk attribute and the real setting would keep its old
            # value.
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

    def __getattr__(self, name):
        # Only reached when normal lookup has already failed, so an alias can
        # never shadow a real attribute of the same name.
        target = type(self)._ALIASES.get(name)
        if target is None:
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}"
            )
        return getattr(self, target)

    def __dir__(self):
        # Aliases exist as behaviour rather than as attributes, so without this
        # they are invisible to ``dir()`` and to tab completion.
        return sorted(set(super().__dir__()) | set(type(self)._ALIASES))

    @classmethod
    def _declared_field_names(cls) -> set[str]:
        names: set[str] = set()
        for base in reversed(cls.__mro__):
            for name, annotation in getattr(base, "__annotations__", {}).items():
                if not _is_special_var(annotation) and not name.startswith("_"):
                    names.add(name)
        return names

    @classmethod
    def _canonical_keys(cls, kwargs):
        """Return ``kwargs`` with every alias replaced by the field it names.

        Everything downstream -- validation, ``dataclasses.replace``, the
        write-back loop in :meth:`set` -- works in declared names only, so the
        aliases are resolved once, here, at each entry point.
        """
        if not any(key in cls._ALIASES for key in kwargs):
            return kwargs
        canonical: dict = {}
        written_as: dict = {}
        for key, value in kwargs.items():
            name = cls._ALIASES.get(key, key)
            if name in canonical:
                raise AlganConfigurationError(
                    f"{cls.__name__} setting '{name}' was given twice, as "
                    f"'{written_as[name]}' and '{key}'"
                )
            written_as[name] = key
            canonical[name] = value
        return canonical

    @classmethod
    def _check_keys(cls, kwargs):
        valid = cls._declared_field_names()
        unknown = [
            name for name in kwargs if name not in valid and name not in cls._ALIASES
        ]
        if unknown:
            name = unknown[0]
            suggestion = difflib.get_close_matches(
                name, sorted(valid | set(cls._ALIASES)), n=1
            )
            hint = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
            aliases = (
                f" Aliases: {', '.join(sorted(cls._ALIASES))}." if cls._ALIASES else ""
            )
            raise AlganConfigurationError(
                f"Unknown {cls.__name__} setting '{name}'.{hint} "
                f"Valid settings are: {', '.join(sorted(valid))}.{aliases}"
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

        Mutable sections are changed in place and returned. Presets return a
        new immutable copy, which is what makes ``HD.set(frames_per_second=60)``
        a safe expression.

        Every field is validated -- the replacement is built by re-running the
        section's own construction -- but only the fields whose value actually
        changed are written back, so an unrelated field keeps its identity.

        Parameters
        ----------
        source
            Another instance of the same settings class -- a preset such as
            ``HD``, typically -- whose fields are copied first. Defaults to
            ``None``, meaning start from this object's own values.
        **kwargs
            Fields to change, by name or alias, applied after ``source``.
            Naming one field by two spellings in the same call is an error
            rather than a silent last-one-wins.

        Returns
        -------
        :class:`Settings`
            This section, changed in place, so calls can be chained; or, on a
            preset, the modified copy.

        Raises
        ------
        :class:`.AlganConfigurationError`
            If a name is not a field of this section, a value is invalid, or
            ``source`` is a different settings class.
        """
        kwargs = type(self)._canonical_keys(kwargs)
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
                # to replace ``background`` too, so anything holding that
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
        """Whether this is an immutable preset rather than a live section.

        ``True`` for ``HD``, ``PREVIEW`` and the other presets, and for whatever
        :meth:`as_preset` returns; ``False`` for the sections of ``SETTINGS``.
        """
        return bool(getattr(self, "_is_preset", False))

    def clone(self, *, preset: bool | None = None):
        """Get an independent copy of these settings.

        Parameters
        ----------
        preset
            ``True`` to make the copy an immutable preset, ``False`` to make it
            mutable. Defaults to ``None``, meaning whichever this one is.

        Returns
        -------
        :class:`Settings`
            A deep copy of the same class: changing either never changes the
            other.
        """
        result = deepcopy(self)
        if preset is not None:
            object.__setattr__(result, "_is_preset", bool(preset))
        return result

    def as_preset(self):
        """Get an immutable copy of these settings, the kind ``HD`` is.

        Use it to keep the current values safe from later changes:
        ``saved = SETTINGS.video.as_preset()``.

        Returns
        -------
        :class:`Settings`
            A preset: :meth:`set` on it returns a modified copy instead of
            changing it.
        """
        return self.clone(preset=True)

    def as_mutable(self):
        """Get a mutable copy of these settings, detached from the original.

        Returns
        -------
        :class:`Settings`
            A copy that :meth:`set` changes in place. It is not ``SETTINGS``'s
            section: changing it does not change what Algan renders with.
        """
        return self.clone(preset=False)

    def to_dict(self) -> dict:
        """Get every field as a plain dict, under its declared name.

        Aliases (``fps`` for ``frames_per_second``) are left out, so
        ``section.set(**section.to_dict())`` round-trips.

        Returns
        -------
        dict
            Each field's name mapped to a deep copy of its value.
        """
        return {
            name: deepcopy(getattr(self, name))
            for name in type(self)._declared_field_names()
        }

    @contextmanager
    def override(self, **kwargs):
        """Change fields for the duration of a ``with`` block.

        Every value is put back on exit, including when the block raises::

            with SETTINGS.video.override(frames_per_second=60):
                Scene.save_video("smooth")

        Parameters
        ----------
        **kwargs
            Fields to change, as for :meth:`set`.

        Yields
        ------
        :class:`Settings`
            This section.

        Raises
        ------
        :class:`.AlganConfigurationError`
            If this is a preset, which cannot change in place, or :meth:`set`
            refuses the values.
        """
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


def _alias_setter(alias: str, target: str):
    def setter(self, value):
        return self.set(**{target: value})

    setter.__name__ = f"set_{alias}"
    setter.__qualname__ = f"set_{alias}"
    setter.__doc__ = (
        f"Set '{target}' (spelled '{alias}'), or return a modified preset copy."
    )
    return setter


def settings_aliases(**aliases: str):
    """Give a settings section a second spelling for some of its fields.

    Apply it *outside* ``@dataclass``, so that it wraps the ``__init__`` that
    decorator generates::

        @settings_aliases(fps="frames_per_second")
        @dataclass
        class VideoSettings(Settings):
            frames_per_second: int = 30

    ``fps`` then works wherever ``frames_per_second`` does -- as a constructor
    keyword, in ``set()`` and ``override()``, as ``set_fps(...)``, and for
    reading and assigning the attribute. It stays a spelling rather than
    becoming a field: it is absent from ``to_dict()``, from snapshots and from
    ``dataclasses.fields``, so state that round-trips through those cannot end
    up carrying the same value twice under two names.
    """

    def decorate(cls):
        declared = cls._declared_field_names()
        merged = dict(cls._ALIASES)
        for alias, target in aliases.items():
            if target not in declared:
                raise AlganConfigurationError(
                    f"{cls.__name__} has no setting '{target}' for alias "
                    f"'{alias}' to point at"
                )
            if alias in declared:
                raise AlganConfigurationError(
                    f"{cls.__name__} alias '{alias}' is already the name of a setting"
                )
            merged[alias] = target
        cls._ALIASES = merged

        # The dataclass __init__ knows only the declared names, so the aliases
        # are translated on the way in. Everything else on the class routes
        # through set/__setattr__, which resolve them themselves.
        original_init = cls.__init__

        @functools.wraps(original_init)
        def __init__(self, *args, **kwargs):
            original_init(self, *args, **type(self)._canonical_keys(kwargs))

        cls.__init__ = __init__

        # A distinct function per alias rather than a second reference to the
        # declared field's setter, so that ``set_fps.__name__`` is ``set_fps``
        # and the generated API reference does not list one method twice under
        # two headings.
        for alias, target in aliases.items():
            setattr(cls, f"set_{alias}", _alias_setter(alias, target))
        return cls

    return decorate
