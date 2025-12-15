# utils.py
"""
This module provides common utility functions for the xsnow package.

These functions are designed to be used across different modules to reduce
code duplication and improve maintainability.
"""

import builtins
import logging
import os
from typing import Union, Any, Optional, List

import numpy as np
import xarray as xr
import warnings

from .core import xsnowDataset
from .plugin import register_xsnowDataset_method

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

@register_xsnowDataset_method
def print(*values: Any, force_details: bool = False, **kwargs: Any) -> None:
    """
    xsnow-aware print that can force detailed __repr__ on xsnowDataset objects.

    Args:
        *values: Values to print.
        force_details: If True, call xsnowDataset.__repr__(force_details=True)
            on any xsnowDataset arguments.
        **kwargs: Passed through to built-in print (sep, end, file, flush).
    """
    rendered = []
    for val in values:
        if isinstance(val, xsnowDataset):
            try:
                rendered.append(val.__repr__(force_details=force_details))
            except TypeError:
                rendered.append(val.__repr__())
        else:
            rendered.append(val)
    builtins.print(*rendered, **kwargs)


def normalize_slope_coords(
    ds: xr.Dataset,
    slope_coord_names: Optional[List[str]] = None,
    verify_consistency: bool = True,
    allow_per_location: bool = False,
    logger: Optional[logging.Logger]=None,
) -> xr.Dataset:
    """
    Normalize slope coordinates to have only 'slope' dimension.
    
    When combining datasets from multiple locations, slope metadata like
    'azimuth' and 'inclination' may end up with dimensions (location, slope)
    instead of just (slope). This function fixes that by:
    
    1. Detecting coordinates that have both 'location' and 'slope' dimensions
    2. Verifying values are consistent across locations (if requested)
    3. Reducing to just the 'slope' dimension (or keeping per-location if allowed)
    
    Parameters
    ----------
    ds : xr.Dataset
        Input dataset potentially with multi-dimensional slope coords
    slope_coord_names : list of str, optional
        Explicit list of coordinate names to normalize. If None, will
        auto-detect coordinates with 'slope' dimension that also have
        other dimensions.
    verify_consistency : bool, default True
        If True, verify that slope coordinate values are identical across
        all locations before reducing dimensions. Raises ValueError if
        inconsistent and allow_per_location=False.
    allow_per_location : bool, default False
        If True, when slopes vary by location, preserve the (location, slope)
        dimensions instead of forcing normalization. This is useful when
        different locations legitimately have different slope angles.
    logger : logging.Logger, optional
        Provide user configured logger, or create new one per default
    
    Returns
    -------
    xr.Dataset
        Dataset with normalized slope coordinates
        
    Raises
    ------
    ValueError
        If verify_consistency=True, allow_per_location=False, and slope 
        coordinates vary across locations
        
    Examples
    --------
    >>> ds = xsnow.read("data/pros/gridded/")
    >>> # azimuth might have dims (location, slope)
    >>> ds = normalize_slope_coords(ds)
    >>> # Now azimuth has only (slope,)
    
    >>> # Allow per-location slopes to be preserved
    >>> ds = normalize_slope_coords(ds, allow_per_location=True)
    >>> # If slopes differ, keeps (location, slope) dimensions
    
    Notes
    -----
    This function modifies coordinate dimensions but preserves all data
    variables unchanged. It's designed to be called after merging datasets
    from multiple locations.
    """
    logger = logger or get_logger(__name__)
    if 'slope' not in ds.dims:
        # No slope dimension, nothing to normalize
        return ds
    
    # Auto-detect slope coordinates if not specified
    if slope_coord_names is None:
        slope_coord_names = []
        for coord_name in ds.coords:
            coord = ds.coords[coord_name]
            # Look for coords with slope + other dims
            if 'slope' in coord.dims and len(coord.dims) > 1:
                slope_coord_names.append(coord_name)
    
    if not slope_coord_names:
        # Nothing to normalize
        return ds
    
    # Process each coordinate
    coords_to_update = {}
    for coord_name in slope_coord_names:
        if coord_name not in ds.coords:
            warnings.warn(
                f"Coordinate '{coord_name}' not found in dataset, skipping",
                UserWarning
            )
            continue
            
        coord = ds.coords[coord_name]
        
        if 'slope' not in coord.dims:
            warnings.warn(
                f"Coordinate '{coord_name}' does not have 'slope' dimension, skipping",
                UserWarning
            )
            continue
        
        # Identify extra dimensions
        extra_dims = [d for d in coord.dims if d != 'slope']
        
        if not extra_dims:
            # Already normalized
            continue
        
        # Handle the most common case: (location, slope) -> (slope)
        if 'location' in extra_dims:
            # Check consistency across locations if requested
            is_consistent = True  # Track consistency
            
            if verify_consistency and ds.sizes['location'] > 1:
                reference = coord.isel(location=0)
                
                for loc_idx in range(1, ds.sizes['location']):
                    comparison = coord.isel(location=loc_idx)
                    if not np.allclose(
                        comparison.values,
                        reference.values,
                        rtol=1e-9,
                        atol=1e-9,
                        equal_nan=True
                    ):
                        is_consistent = False  # Mark as inconsistent
                        
                        # If we allow per-location slopes, skip normalization
                        if allow_per_location:
                            logger.info(
                                f"Slope coordinate '{coord_name}' varies by location. "
                                f"Preserving (location, slope) dimensions as allow_per_location=True."
                            )
                            # Don't add to coords_to_update - keep original dimensions
                            break
                        
                        # Otherwise, raise error as before
                        # Find which slopes differ
                        diff_mask = ~np.isclose(
                            comparison.values,
                            reference.values,
                            rtol=1e-9,
                            atol=1e-9,
                            equal_nan=True
                        )
                        diff_indices = np.where(diff_mask)[0]
                        
                        raise ValueError(
                            f"Slope coordinate '{coord_name}' is inconsistent across locations.\n"
                            f"Location 0 vs Location {loc_idx} differ at slope indices: {diff_indices}\n"
                            f"Location 0 values: {reference.values[diff_indices]}\n"
                            f"Location {loc_idx} values: {comparison.values[diff_indices]}\n"
                            f"Slope coordinates must be identical across all locations.\n"
                            f"Set verify_consistency=False to use first location's values anyway, "
                            f"or allow_per_location=True to preserve per-location slopes."
                        )
            
            # Only normalize if consistent or if we're not checking
            if is_consistent or not verify_consistency:
                # Take first location, drop its attached coords (lat/lon/alt), keep only slope dim
                normalized = (
                    coord
                    .isel(location=0, drop=True)
                    .reset_coords(drop=True)
                )
                coords_to_update[coord_name] = normalized
            
        else:
            # For other dimension combinations, warn and take first index
            warnings.warn(
                f"Coordinate '{coord_name}' has unexpected dimensions {coord.dims}. "
                f"Normalizing by taking first index along {extra_dims}.",
                UserWarning
            )
            
            # Iteratively reduce extra dimensions
            normalized = coord
            for dim in extra_dims:
                normalized = normalized.isel({dim: 0})
                # Drop the coordinate to avoid scalar dimension conflicts
                normalized = normalized.drop_vars(dim, errors='ignore')
            
            # Remove any remaining non-dimension coords (e.g., latitude) before reassigning
            normalized = normalized.reset_coords(drop=True)
            coords_to_update[coord_name] = normalized
    
    # Apply updates if any
    if coords_to_update:
        # Drop the old coordinates first to avoid dimension conflicts
        ds = ds.drop_vars(list(coords_to_update.keys()))
        
        # Now assign the new coordinates
        ds = ds.assign_coords(coords_to_update)
        
        # Log what was done
        coord_list = ", ".join(f"'{c}'" for c in coords_to_update.keys())
        logger.debug(
            f"Normalized slope coordinates {coord_list} to have only 'slope' dimension"
        )
    
    return ds


def detect_slope_coord_issues(ds: xr.Dataset) -> dict:
    """
    Detect potential issues with slope coordinate dimensions.
    
    Returns a diagnostic dictionary describing any problems found.
    
    Parameters
    ----------
    ds : xr.Dataset
        Dataset to check
        
    Returns
    -------
    dict
        Diagnostic information with keys:
        - 'has_issues': bool, True if problems detected
        - 'multidim_coords': list of coordinate names with multiple dimensions
        - 'inconsistent_coords': list of coordinates that vary across locations
        - 'recommendations': list of suggested fixes
    """
    diagnostics = {
        'has_issues': False,
        'multidim_coords': [],
        'inconsistent_coords': [],
        'recommendations': []
    }
    
    if 'slope' not in ds.dims:
        return diagnostics
    
    # Check each coordinate
    for coord_name in ds.coords:
        coord = ds.coords[coord_name]
        
        # Check for multi-dimensional slope coords
        if 'slope' in coord.dims and len(coord.dims) > 1:
            diagnostics['multidim_coords'].append(coord_name)
            diagnostics['has_issues'] = True
            
            # Check for inconsistency across locations
            if 'location' in coord.dims and ds.sizes['location'] > 1:
                reference = coord.isel(location=0)
                inconsistent = False
                
                for loc_idx in range(1, ds.sizes['location']):
                    comparison = coord.isel(location=loc_idx)
                    if not np.allclose(
                        comparison.values,
                        reference.values,
                        rtol=1e-9,
                        atol=1e-9,
                        equal_nan=True
                    ):
                        inconsistent = True
                        break
                
                if inconsistent:
                    diagnostics['inconsistent_coords'].append(coord_name)
    
    # Generate recommendations
    if diagnostics['multidim_coords']:
        diagnostics['recommendations'].append(
            f"Run normalize_slope_coords() to fix {len(diagnostics['multidim_coords'])} "
            f"multi-dimensional slope coordinates"
        )
    
    if diagnostics['inconsistent_coords']:
        diagnostics['recommendations'].append(
            f"WARNING: {len(diagnostics['inconsistent_coords'])} slope coordinates "
            f"have inconsistent values across locations. This should be fixed at "
            f"the data generation stage."
        )
    
    return diagnostics


@register_xsnowDataset_method
def compute_HS(self) -> 'xsnowDataset':
    """
    Compute total snow height (HS) from the height coordinate.
    
    HS is defined as the maximum height value in each profile,
    representing the total depth of the snowpack from ground to surface.
    
    Returns
    -------
    xsnowDataset
        A new dataset with the HS variable added. If height data is not
        available, returns the original dataset unchanged.
    
    Examples
    --------
    >>> import xsnow
    >>> ds = xsnow.read('path/to/data.pro')
    >>> ds = ds.compute_HS()
    >>> print(ds['HS'])  # Total snow height for each profile
    
    Notes
    -----
    - HS is computed as max(height) along the layer dimension for each profile
    - If all height values are NaN for a profile, HS will be NaN for that profile
    - HS has dimensions (location, time, slope, realization) - no layer dimension
    """
    if 'height' not in self.data:
        warnings.warn(
            "Cannot compute HS: 'height' variable not found in dataset",
            UserWarning
        )
        return self
    
    # HS is the maximum height in each profile (maximum along layer dimension)
    # This represents the snow surface height above ground
    hs = self.data['height'].max(dim='layer', skipna=True)
    
    # Add metadata from registry if available, otherwise use defaults
    from .registry import registry
    reg_entry = registry.get('HS')
    if reg_entry:
        hs.attrs = {
            'standard_name': reg_entry.get('standard_name', 'snow_height'),
            'units': reg_entry.get('si_units', 'm'),
            'long_name': reg_entry.get('long_name', 'Total snow height'),
        }
    else:
        hs.attrs = {
            'standard_name': 'snow_height',
            'long_name': 'Total snow height',
            'units': 'm',
        }
    
    hs.attrs['description'] = 'Total depth of snowpack (maximum height above ground)'
    
    # Create new dataset with HS
    new_data = self.data.copy(deep=False)
    new_data['HS'] = hs
    
    return self.__class__(new_data)
