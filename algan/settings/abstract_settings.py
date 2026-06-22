import dataclasses
import typing
from copy import deepcopy


def _is_special_var(annotation) -> bool:
    """
    Determines if an annotation is a ClassVar, InitVar, or KW_ONLY marker,
    meaning it is not a standard instance field and shouldn't get a setter.
    """
    # 1. Handle string/deferred annotations (e.g. x: 'ClassVar[int]')
    if isinstance(annotation, str):
        return any(marker in annotation for marker in ('ClassVar', 'InitVar', 'KW_ONLY'))

    # 2. Check for dataclasses.KW_ONLY (Python 3.10+)
    if annotation is getattr(dataclasses, 'KW_ONLY', None):
        return True

    # Get the origin type (e.g., ClassVar[int] -> ClassVar)
    origin = typing.get_origin(annotation) or annotation

    # 3. Check for typing.ClassVar
    if origin is typing.ClassVar:
        return True

    # 4. Check for dataclasses.InitVar
    if origin is getattr(dataclasses, 'InitVar', None) or isinstance(annotation,
                                                                     getattr(dataclasses, 'InitVar', type(None))):
        return True

    return False


class Settings:
    """
    A base class that automatically generates a set_<field> method
    for every dataclass field defined in its subclasses, and makes
    all set operations non-inplace.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Helper function to generate the setter method dynamically.
        # This prevents late-binding closure bugs in loops.
        def make_setter(field_name: str):
            def setter(self, value):
                self = self.clone()
                setattr(self, field_name, value)
                return self  # Return self to allow method chaining

            setter.__name__ = f"set_{field_name}"
            setter.__doc__ = f"Automatically generated setter for '{field_name}'."
            return setter

        # Walk through the MRO (Method Resolution Order) in reverse.
        # This ensures we handle inherited fields just like @dataclass does.
        for base in reversed(cls.__mro__):
            if hasattr(base, '__annotations__'):
                for name, annotation in base.__annotations__.items():
                    if _is_special_var(annotation):
                        continue

                    setter_name = f"set_{name}"

                    # Only add the method if it wasn't manually defined by the user
                    if not hasattr(cls, setter_name):
                        setattr(cls, setter_name, make_setter(name))

    def set(self, **kwargs):
        self = self.clone()
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self

    def clone(self):
        return deepcopy(self)
