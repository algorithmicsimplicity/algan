import dataclasses
import difflib
import typing
from copy import deepcopy

from algan.errors import AlganConfigurationError


def _is_special_var(annotation) -> bool:
    """Return whether an annotation is not a normal instance setting field."""
    if isinstance(annotation, str):
        return any(
            marker in annotation
            for marker in ("ClassVar", "InitVar", "KW_ONLY")
        )

    if annotation is getattr(dataclasses, "KW_ONLY", None):
        return True

    origin = typing.get_origin(annotation) or annotation
    if origin is typing.ClassVar:
        return True

    init_var = getattr(dataclasses, "InitVar", None)
    if origin is init_var or (init_var is not None and isinstance(annotation, init_var)):
        return True

    return False


class Settings:
    """Base for non-mutating, validated setting objects.

    Subclasses normally use :func:`dataclasses.dataclass`.  ``set`` and the
    generated ``set_<field>`` methods return a copy and reject unknown names,
    so misspelled options fail at the point of use instead of becoming inert
    attributes.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        def make_setter(field_name: str):
            def setter(self, value):
                return self.set(**{field_name: value})

            setter.__name__ = f"set_{field_name}"
            setter.__doc__ = f"Return a copy with '{field_name}' set to value."
            return setter

        for base in reversed(cls.__mro__):
            for name, annotation in getattr(base, "__annotations__", {}).items():
                if _is_special_var(annotation):
                    continue
                setter_name = f"set_{name}"
                if not hasattr(cls, setter_name):
                    setattr(cls, setter_name, make_setter(name))

    @classmethod
    def _declared_field_names(cls) -> set[str]:
        names: set[str] = set()
        for base in reversed(cls.__mro__):
            for name, annotation in getattr(base, "__annotations__", {}).items():
                if not _is_special_var(annotation):
                    names.add(name)
        return names

    def _validate(self):
        """Hook for non-dataclass subclasses that need semantic validation."""
        return None

    def set(self, **kwargs):
        valid = self._declared_field_names()
        unknown = [name for name in kwargs if name not in valid]
        if unknown:
            name = unknown[0]
            suggestion = difflib.get_close_matches(name, sorted(valid), n=1)
            hint = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
            raise AlganConfigurationError(
                f"Unknown {type(self).__name__} setting '{name}'.{hint}"
            )

        if dataclasses.is_dataclass(self):
            try:
                result = dataclasses.replace(self, **kwargs)
            except (TypeError, ValueError) as exc:
                raise AlganConfigurationError(str(exc)) from exc
        else:
            result = self.clone()
            for key, value in kwargs.items():
                setattr(result, key, value)
            result._validate()
        return result

    replace = set

    def clone(self):
        return deepcopy(self)
