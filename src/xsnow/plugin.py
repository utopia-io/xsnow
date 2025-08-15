# xsnow/plugin.py
"""
Defines the plugin system for registering custom analysis functions.

This module provides the `register_analysis` decorator, which allows
user-defined functions to be attached to the `xsnowDataset` class at runtime.
This makes the package extensible, allowing users to create and share their
own post-processing and analysis routines that integrate seamlessly with the
core data object.
"""

from typing import Dict, Callable, Any
import warnings

# The global registry for custom analysis functions.
# This dictionary maps a function name (str) to the function object (Callable).
_analysis_plugins: Dict[str, Callable] = {}


def register_analysis(func: Callable) -> Callable:
    """
    A decorator to register a function as a custom xsnowDataset method.

    The decorated function must take an `xsnowDataset` object as its first
    argument, followed by any other parameters. When the method is called on an
    `xsnowDataset` instance, the instance itself is automatically passed as this
    first argument.

    Example:
        import xsnow

        @xsnow.register_analysis
        def get_surface_temperature(ds):
            \"\"\"Returns the temperature of the top snow layer.\"\"\"
            # Find the index of the highest layer for each profile
            surface_idx = ds.data['height'].argmax(dim='layer')
            # Use the index to select the corresponding temperature
            return ds.data['temperature'].isel(layer=surface_idx)

        # Usage:
        my_dataset = xsnow.xsnowDataset('path/to/data')
        surface_temps = my_dataset.get_surface_temperature()

    Args:
        func (Callable): The function to be registered.

    Returns:
        Callable: The original, unmodified function.
    """
    func_name = func.__name__
    if func_name in _analysis_plugins:
        warnings.warn(
            f"Overwriting existing analysis plugin '{func_name}'.",
            stacklevel=2
        )

    _analysis_plugins[func_name] = func
    return func