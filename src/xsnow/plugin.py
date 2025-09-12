"""
Defines the plugin system for registering xsnowDatset methods.

This is used in xsnow to keep the xsnowDataset class definition tidy and 
register class methods from other modules. It can also be used to register completely custom
functions that do not ship with this package.
"""

from typing import Dict, Callable, Optional
import warnings

# The global registry for custom analysis functions.
# This dictionary maps a function name (str) to the function object (Callable).
_posthoc_registered_methods: Dict[str, Callable] = {}

def register_xsnowDataset_method(func: Optional[Callable] = None, *, name: Optional[str] = None):
    """
    Decorator to attach a free function as a method on xsnowDataset.

    The decorated function must take an `xsnowDataset` object as its first
    argument, followed by any other parameters. When the method is called on an
    `xsnowDataset` instance, the instance itself is automatically passed as this
    first argument.

    Parameters:
    ----------
        func (Callable): The function to be registered
        name (str), optional: An alias for the registered method. Defaults to the function name.

    Examples:
    ---------
        from xsnow.extensions import register_xsnowDataset_method
        @register_xsnowDataset_method
        def mytool(self, ...): ...

        @register_xsnowDataset_method(name="alias")
        def mytool(self, ...): ...

        # Usage:
        >>> import xsnow
        >>> ds = xsnow.read('path/to/data')
        >>> ds.mytool(...)

    Returns:
    --------
        Callable: The original, unmodified function.
    """
    def _attach(f: Callable) -> Callable:
        meth_name = name or f.__name__
        _posthoc_registered_methods[meth_name] = f

        from xsnow import xsnowDataset  # late import to avoid circulars
        if hasattr(xsnowDataset, meth_name):
            warnings.warn(f"xsnowDataset already has attribute '{meth_name}', overwriting.", stacklevel=2)
        setattr(xsnowDataset, meth_name, f)
        return f

    return _attach(func) if callable(func) else _attach


def list_registered_methods() -> Dict[str, Callable]:
    """Return a copy of the registered extension methods (for docs/CLI)."""
    return dict(_posthoc_registered_methods)
