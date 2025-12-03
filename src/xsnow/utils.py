# utils.py
"""
This module provides common utility functions for the xsnow package.

These functions are designed to be used across different modules to reduce
code duplication and improve maintainability.
"""

import logging
import os
from typing import Union, Any, Optional

import numpy as np
import xarray as xr

from .core import xsnowDataset

LEVEL_PREFIX = {
    logging.DEBUG: "[d]",
    logging.INFO: "[i]",
    logging.WARNING: "[w]",
    logging.ERROR: "[E]",
    logging.CRITICAL: "[C]",
}


class _LevelPrefixFormatter(logging.Formatter):
    """Formatter that injects a short level prefix (e.g., [i], [w], [E])."""

    def format(self, record: logging.LogRecord) -> str:
        record.levelprefix = LEVEL_PREFIX.get(record.levelno, f"[{record.levelname[:1]}]")
        return super().format(record)


def get_logger(
    name: str,
    to_console: bool = True,
    level_console: int = logging.INFO,
    to_file: bool = False,
    logfile: Optional[str] = None,
    level_file: int = logging.DEBUG,
    logger: Optional[logging.Logger] = None,
    alter_logger: bool = False,
) -> logging.Logger:
    """
    Create or adjust a logger with optional console and file handlers.

    Args:
        name: Logger name to use if a logger instance is not provided.
        to_console: Whether to log to the console (stream handler).
        level_console: Logging level for console output.
        to_file: Whether to log to a file.
        logfile: Path to the logfile when `to_file` is True.
        level_file: Logging level for file output.
        logger: An existing logger to configure. If provided and
            `alter_logger` is False, the logger is returned unchanged.
        alter_logger: When True, update the provided logger to match the
            requested handlers and levels. When False, the logger is
            returned as-is.

    Returns:
        A configured `logging.Logger` instance.

    Raises:
        ValueError: If `to_file` is True but neither `logfile` nor an
        existing file handler on `logger` is available.
    """
    log = logger or logging.getLogger(name)
    target_path = os.path.abspath(logfile) if logfile else None

    def _find_console_handler() -> Optional[logging.Handler]:
        for handler in log.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                return handler
        return None

    def _find_file_handler(target: Optional[str]) -> Optional[logging.FileHandler]:
        for handler in log.handlers:
            if isinstance(handler, logging.FileHandler):
                if target is None:
                    return handler
                try:
                    if handler.baseFilename == target:
                        return handler
                except Exception:
                    continue
        return None

    if logger is not None and not alter_logger:
        if to_file and _find_file_handler(target_path) is None:
            raise ValueError("to_file=True requires a logfile or an existing file handler on the provided logger.")
        return log

    handlers_added = False

    if alter_logger and not to_console:
        # Remove console handlers when explicitly disabled
        for h in list(log.handlers):
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                log.removeHandler(h)

    if alter_logger and not to_file:
        # Remove file handlers when explicitly disabled
        for h in list(log.handlers):
            if isinstance(h, logging.FileHandler):
                log.removeHandler(h)

    console_handler: Optional[logging.Handler] = None
    if to_console:
        console_handler = _find_console_handler()
        if console_handler is None:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(_LevelPrefixFormatter("%(levelprefix)s %(name)s: %(message)s"))
            log.addHandler(console_handler)
            handlers_added = True
        console_handler.setLevel(level_console)

    file_handler: Optional[logging.FileHandler] = None
    if to_file:
        file_handler = _find_file_handler(target_path)
        if file_handler is None and logfile is None:
            raise ValueError("logfile must be provided or logger must already have a file handler when to_file is True.")
        if file_handler is None and logfile:
            file_handler = logging.FileHandler(target_path)
            file_handler.setFormatter(
                _LevelPrefixFormatter("%(asctime)s %(levelprefix)s %(name)s - %(message)s")
            )
            log.addHandler(file_handler)
            handlers_added = True
        if file_handler:
            file_handler.setLevel(level_file)

    # Ensure logger level does not prevent handler output
    active_levels = []
    if console_handler:
        active_levels.append(level_console)
    if file_handler:
        active_levels.append(level_file)
    if active_levels:
        log.setLevel(min(active_levels))

    if handlers_added or console_handler or file_handler:
        # Prevent duplicate propagation to root when custom handlers are attached
        log.propagate = False

    return log


def get_backend(array_like: Union[xsnowDataset, xr.Dataset, xr.DataArray]) -> Any:
    """
    Determines the appropriate numerical backend (NumPy or CuPy) for a given object.

    This function inspects the underlying data in an xarray object to see if it's
    a CuPy array (on a GPU) or a NumPy array (on the CPU) and returns the
    corresponding module.

    Args:
        array_like: An xsnowDataset, xarray.Dataset, or xarray.DataArray.

    Returns:
        The `numpy` or `cupy` module.
    """
    # If it's an xsnowDataset, get the underlying xarray.Dataset
    if isinstance(array_like, xsnowDataset):
        if array_like.data is None:
            return np
        data = array_like.data
    else:
        data = array_like

    # Find the first data variable to inspect
    if isinstance(data, xr.Dataset):
        first_var_name = next(iter(data.data_vars), None)
        if first_var_name is None:
            return np # No data variables, default to numpy
        array = data[first_var_name].data
    else: # It's a DataArray
        array = data.data

    # Check for the 'device' attribute, which is characteristic of a CuPy array
    if hasattr(array, 'device'):
        try:
            import cupy
            return cupy
        except ImportError:
            # This case is unlikely if a cupy array exists, but handled for safety
            return np
            
    return np
