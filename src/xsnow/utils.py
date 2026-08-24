# utils.py
"""
This module provides common utility functions for the xsnow package.

These functions are designed to be used across different modules to reduce
code duplication and improve maintainability.
"""

import builtins
import logging
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Union, Any, Literal, Optional, List, Sequence, overload

import numpy as np
import pandas as pd
import xarray as xr
import warnings

from .core import xsnowDataset
from .plugin import register_xsnowDataset_method
from .registry import registry

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

    # CuPy arrays live in the 'cupy' package. (NumPy >= 2.0 arrays also
    # expose a 'device' attribute, so the type's module is the reliable
    # signal.)
    if type(array).__module__.split('.')[0] == 'cupy':
        try:
            import cupy
            return cupy
        except ImportError:
            # This case is unlikely if a cupy array exists, but handled for safety
            return np

    return np


def require_vars(ds: xsnowDataset, names: Sequence[str], context: str) -> None:
    """Raise ``ValueError`` if ``ds`` is empty or misses any of ``names``."""
    if ds.data is None:
        raise ValueError("Cannot compute on an empty dataset.")
    for var in names:
        if var not in ds.data:
            raise ValueError(
                f"Missing required variable for {context}: '{var}'"
            )


def registry_attrs(name: str, default_long_name: str,
                   default_units: str) -> dict:
    """Variable attrs from the registry with a fallback."""
    if name in registry:
        entry = registry[name]
        return {
            'long_name': entry.get('long_name', default_long_name),
            'units': entry.get('xsnow_units', default_units),
            'standard_name': entry.get('standard_name', name),
        }
    return {
        'long_name': default_long_name,
        'units': default_units,
        'standard_name': name,
    }


def wrap_like(arr: Any, template_da: xr.DataArray,
              attrs: dict) -> xr.DataArray:
    """Wrap a backend array into a DataArray shaped like ``template_da``."""
    da = xr.DataArray(arr, dims=template_da.dims, coords=template_da.coords)
    da.attrs = attrs
    return da


def finite_max(a: Any, xp: Any, axis: int = -1) -> Any:
    """Maximum over the finite entries of ``a``; NaN where there are none."""
    m = xp.max(xp.where(xp.isfinite(a), a, -xp.inf), axis=axis)
    return xp.where(xp.isfinite(m), m, xp.nan)


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


def _is_missing_slope_metadata(value: Any) -> bool:
    """Return whether a scalar slope-metadata value is missing."""
    if value is None or np.ma.is_masked(value):
        return True
    try:
        missing = np.asarray(pd.isna(value))
        return bool(missing) if missing.ndim == 0 else False
    except (TypeError, ValueError):
        return False


def _slope_metadata_values_match(left: Any, right: Any) -> bool:
    """Compare two scalar slope-metadata values with the consistency tolerance."""
    left_missing = _is_missing_slope_metadata(left)
    right_missing = _is_missing_slope_metadata(right)
    if left_missing or right_missing:
        return left_missing and right_missing

    try:
        return bool(
            np.isclose(
                left,
                right,
                rtol=1e-9,
                atol=1e-9,
                equal_nan=True,
            )
        )
    except (TypeError, ValueError):
        try:
            return bool(left == right)
        except (TypeError, ValueError):
            return False


def _present_slope_metadata_mask(values: np.ndarray) -> np.ndarray:
    """Return a vectorized non-missing mask for slope metadata."""
    values = np.asanyarray(values)
    return (
        np.asarray(pd.notna(np.ma.getdata(values)), dtype=bool)
        & ~np.ma.getmaskarray(values)
    )


def _slope_signatures_match(left: tuple, right: tuple) -> bool:
    """Return whether two composite slope-metadata signatures match."""
    return all(
        _slope_metadata_values_match(left_value, right_value)
        for left_value, right_value in zip(left, right)
    )


def _slope_metadata_sort_key(value: Any) -> tuple:
    """Return a deterministic sort key for one slope-metadata value."""
    if _is_missing_slope_metadata(value):
        return (2, 0.0, "")
    try:
        return (0, float(value), "")
    except (TypeError, ValueError):
        return (1, 0.0, str(value))


def _slope_signature_sort_key(
    signature: tuple,
    slope_coord_names: Sequence[str],
) -> tuple:
    """Sort standard signatures by inclination then azimuth, then custom fields."""
    preferred_order = {"inclination": 0, "azimuth": 1}
    coord_positions = sorted(
        range(len(slope_coord_names)),
        key=lambda pos: (
            preferred_order.get(slope_coord_names[pos], 2),
            pos,
        ),
    )
    return tuple(
        _slope_metadata_sort_key(signature[pos])
        for pos in coord_positions
    )


def normalize_slope_coords(
    ds: xr.Dataset,
    slope_coord_names: Sequence[str] = ("azimuth", "inclination"),
    logger: Optional[logging.Logger] = None,
    *,
    strategy: Literal["union", "auto"] = "auto",
) -> xr.Dataset:
    """
    Normalize local slope positions using slope-metadata signatures.

    When datasets from multiple locations are combined, recognized slope
    metadata such as ``azimuth`` and ``inclination`` may have dimensions
    ``(location, slope)``. This function can construct a global catalogue of
    their metadata signatures and remap every slope-bearing variable onto it.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset potentially containing per-location slope coordinates.
    slope_coord_names : sequence of str, default ("azimuth", "inclination")
        Coordinate names used to identify slope configurations. Names absent
        from the dataset are ignored. Pass another sequence to recognize
        additional slope metadata coordinates.
    logger : logging.Logger, optional
        Provide user configured logger, or create new one per default.
    strategy : {"union", "auto"}, default "auto"
        ``"union"`` constructs the global union of recognized slope-metadata
        signatures and remaps all slope-bearing data onto it. ``"auto"`` does
        the same only when the union does not widen the existing slope
        dimension; otherwise it preserves the existing layout.

    Returns
    -------
    xr.Dataset
        Dataset with normalized slope coordinates, or the original dataset
        when normalization is not possible or ``"auto"`` would widen it.

    Raises
    ------
    ValueError
        If a union cannot be constructed losslessly because recognized
        metadata has unsupported dimensions or a location contains duplicate
        slope signatures.

    Examples
    --------
    >>> ds = xsnow.read("data/pros/gridded/")
    >>> # Normalize compatible local slope subsets without widening the data
    >>> ds = normalize_slope_coords(ds)

    >>> # Always build a global slope catalogue
    >>> ds = normalize_slope_coords(ds, strategy="union")

    Notes
    -----
    ``"union"`` may widen and sparsify slope-bearing variables. Both
    strategies preserve data by remapping local slope positions to recognized
    metadata signatures. Only the recognized metadata coordinates are
    materialized; other slope-bearing data remain lazy. If signatures cannot
    be identified, the dataset is returned unchanged. A ``UserWarning`` is
    emitted for an explicit ``"union"`` request or an ambiguous layout with
    multiple slope positions.
    """
    logger = logger or get_logger(__name__)
    if strategy not in {"union", "auto"}:
        raise ValueError(
            f"Unsupported slope normalization strategy: {strategy!r}. "
            "Use 'union' or 'auto'."
        )

    if "location" not in ds.dims or "slope" not in ds.dims:
        return ds

    location_size = ds.sizes["location"]
    slope_size = ds.sizes["slope"]
    if location_size == 0 or slope_size == 0 or not slope_coord_names:
        return ds

    allow_slope_dimension_growth = strategy == "union"
    metadata_arrays = {}
    metadata_coords = {}
    per_location_names = []

    for coord_name in slope_coord_names:
        if coord_name not in ds.coords:
            continue

        coord = ds.coords[coord_name]
        if "slope" not in coord.dims:
            warnings.warn(
                f"Coordinate '{coord_name}' does not have 'slope' dimension, skipping",
                UserWarning,
            )
            continue

        metadata_coords[coord_name] = coord
        if coord.dims == ("slope",):
            values = np.asarray(coord.values)
            metadata_arrays[coord_name] = np.broadcast_to(
                values[np.newaxis, :],
                (location_size, slope_size),
            )
            continue

        if set(coord.dims) == {"location", "slope"} and len(coord.dims) == 2:
            metadata_arrays[coord_name] = np.asarray(
                coord.transpose("location", "slope").values
            )
            per_location_names.append(coord_name)
            continue

        raise ValueError(
            f"Cannot normalize slope coordinate {coord_name!r} losslessly: "
            f"expected dimensions ('slope',) or ('location', 'slope'), "
            f"got {coord.dims!r}."
        )

    if not metadata_arrays:
        if allow_slope_dimension_growth or slope_size > 1:
            warnings.warn(
                "Slope normalization skipped: no recognized slope metadata "
                f"coordinates {tuple(slope_coord_names)!r} with a 'slope' "
                "dimension were found. Preserving the existing slope layout.",
                UserWarning,
                stacklevel=3,
            )
        return ds

    metadata_present = np.zeros((location_size, slope_size), dtype=bool)
    for coord_name in metadata_arrays:
        coord_values = metadata_arrays[coord_name]
        metadata_present |= _present_slope_metadata_mask(coord_values)

    if not metadata_present.any():
        if allow_slope_dimension_growth or slope_size > 1:
            warnings.warn(
                "Slope normalization skipped: the recognized slope metadata "
                f"coordinates {tuple(metadata_arrays)!r} contain no usable "
                "values. Preserving the existing slope layout.",
                UserWarning,
                stacklevel=3,
            )
        return ds

    unidentified_locations = ~metadata_present.any(axis=1)
    if unidentified_locations.any():
        if allow_slope_dimension_growth or slope_size > 1:
            warnings.warn(
                "Slope normalization skipped: the recognized slope metadata "
                f"coordinates {tuple(metadata_arrays)!r} do not identify any "
                f"slope for {int(unidentified_locations.sum())} location(s). "
                "Preserving the existing slope layout.",
                UserWarning,
                stacklevel=3,
            )
        return ds

    if not per_location_names:
        return ds

    # Missing cells in an otherwise identified location are outer-join
    # padding, not additional slope configurations.
    valid_source = metadata_present

    metadata_names = tuple(metadata_arrays)
    unique_signatures = []
    target_by_source = np.full(
        (location_size, slope_size),
        -1,
        dtype=np.int64,
    )

    for location_index in range(location_size):
        targets_at_location = {}
        for slope_index in np.flatnonzero(valid_source[location_index]):
            signature = tuple(
                metadata_arrays[coord_name][location_index, slope_index]
                for coord_name in metadata_names
            )
            target_index = next(
                (
                    index
                    for index, existing in enumerate(unique_signatures)
                    if _slope_signatures_match(signature, existing)
                ),
                None,
            )
            if target_index is None:
                target_index = len(unique_signatures)
                unique_signatures.append(signature)
                if (
                    not allow_slope_dimension_growth
                    and len(unique_signatures) > slope_size
                ):
                    logger.info(
                        "Preserving per-location slope coordinates because "
                        "their global union would grow the slope dimension "
                        f"beyond its current size of {slope_size}."
                    )
                    return ds

            if target_index in targets_at_location:
                first_slope_index = targets_at_location[target_index]
                location_value = ds["location"].values[location_index]
                raise ValueError(
                    f"Location {location_value!r} contains duplicate slope "
                    f"metadata at local slope indices {first_slope_index} and "
                    f"{slope_index}: {signature!r}."
                )

            targets_at_location[target_index] = int(slope_index)
            target_by_source[location_index, slope_index] = target_index

    # Canonical ordering makes the global slope index independent of location
    # order while retaining the familiar inclination-then-azimuth ordering.
    signature_order = sorted(
        range(len(unique_signatures)),
        key=lambda index: _slope_signature_sort_key(
            unique_signatures[index],
            metadata_names,
        ),
    )
    old_to_new = {
        old_index: new_index
        for new_index, old_index in enumerate(signature_order)
    }
    unique_signatures = [
        unique_signatures[old_index]
        for old_index in signature_order
    ]

    remapped_targets = np.full_like(target_by_source, -1)
    for old_index, new_index in old_to_new.items():
        remapped_targets[target_by_source == old_index] = new_index
    target_by_source = remapped_targets

    union_size = len(unique_signatures)
    source_by_target = np.zeros(
        (location_size, union_size),
        dtype=np.int64,
    )
    valid_target = np.zeros(
        (location_size, union_size),
        dtype=bool,
    )
    for location_index in range(location_size):
        for slope_index in np.flatnonzero(target_by_source[location_index] >= 0):
            target_index = target_by_source[location_index, slope_index]
            source_by_target[location_index, target_index] = slope_index
            valid_target[location_index, target_index] = True

    normalized_dim = "_normalized_slope"
    indexer_coords = {
        "location": ds["location"],
        normalized_dim: np.arange(union_size),
    }
    source_indexer = xr.DataArray(
        source_by_target,
        dims=("location", normalized_dim),
        coords=indexer_coords,
    )
    valid_targets = xr.DataArray(
        valid_target,
        dims=("location", normalized_dim),
        coords=indexer_coords,
    )

    normalized = ds.drop_vars(list(metadata_coords)).isel(
        slope=source_indexer
    )

    # isel uses source index zero as a placeholder for absent signatures.
    # Replace those selected values with the dtype-appropriate missing value,
    # but do not broadcast the mask onto variables unrelated to slope.
    has_missing_targets = not valid_target.all()
    for var_name in list(normalized.data_vars):
        if normalized_dim not in normalized[var_name].dims:
            continue
        if not has_missing_targets:
            continue
        if (
            var_name == "profile_status"
            and np.issubdtype(normalized[var_name].dtype, np.integer)
        ):
            normalized[var_name] = normalized[var_name].where(
                valid_targets,
                other=0,
            )
        else:
            normalized[var_name] = normalized[var_name].where(valid_targets)

    coord_updates = {}
    for coord_name in normalized.coords:
        if (
            has_missing_targets
            and coord_name not in normalized.dims
            and coord_name != "slope"
            and normalized_dim in normalized[coord_name].dims
        ):
            coord_updates[coord_name] = normalized[coord_name].where(
                valid_targets
            )
    if coord_updates:
        normalized = normalized.assign_coords(coord_updates)

    if "slope" in normalized.coords:
        normalized = normalized.drop_vars("slope")
    normalized = normalized.rename({normalized_dim: "slope"})
    normalized = normalized.assign_coords(
        slope=xr.DataArray(
            np.arange(union_size),
            dims=("slope",),
            attrs=ds["slope"].attrs,
        )
    )

    signature_coords = {}
    for coord_position, coord_name in enumerate(metadata_names):
        values = [
            (
                np.nan
                if _is_missing_slope_metadata(signature[coord_position])
                else signature[coord_position]
            )
            for signature in unique_signatures
        ]
        signature_coords[coord_name] = xr.DataArray(
            np.asarray(values),
            dims=("slope",),
            attrs=metadata_coords[coord_name].attrs,
        )
    normalized = normalized.assign_coords(signature_coords)

    logger.debug(
        "Normalized per-location slope coordinates onto a global union of "
        f"{union_size} slope metadata signatures."
    )
    return normalized


def detect_slope_coord_issues(
    ds: xr.Dataset,
    slope_coord_names: Sequence[str] = ("azimuth", "inclination"),
) -> dict:
    """
    Detect potential issues with slope coordinate dimensions.
    
    Returns a diagnostic dictionary describing any problems found.
    
    Parameters
    ----------
    ds : xr.Dataset
        Dataset to check
    slope_coord_names : sequence of str, default ("azimuth", "inclination")
        Recognized slope metadata coordinates to inspect. Names absent from the
        dataset are ignored.
        
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
    
    # Check each recognized slope metadata coordinate
    for coord_name in slope_coord_names:
        if coord_name not in ds.coords:
            continue
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


@overload
def reorder_dims(
    ds: xsnowDataset,
    preferred_order: Sequence[str] = ("location", "time", "slope", "realization", "layer"),
    last_dims: Optional[Sequence[str]] = None,
) -> xsnowDataset: ...
@overload
def reorder_dims(
    ds: xr.Dataset,
    preferred_order: Sequence[str] = ("location", "time", "slope", "realization", "layer"),
    last_dims: Optional[Sequence[str]] = None,
) -> xr.Dataset: ...
@register_xsnowDataset_method
def reorder_dims(
    ds: Union[xsnowDataset, xr.Dataset],
    preferred_order: Sequence[str] = ("location", "time", "slope", "realization", "layer"),
    last_dims: Optional[Sequence[str]] = None,
) -> Union[xsnowDataset, xr.Dataset]:
    """
    Reorder dataset dimensions without dropping data, coords, or attrs.
    
    Parameters
    ----------
    ds : xsnowDataset or xr.Dataset
        Dataset to reorder.
    preferred_order : sequence of str
        Dimensions to place first in the given order.
    last_dims : sequence of str, optional
        Dimensions to force to the end in the given order.

    Notes
    -----
    - Any dimensions not in ``preferred_order`` or ``last_dims`` keep their
      original relative order and are placed between the two groups.
    - All coordinates, data variables, attrs, and encoding are preserved.
    - Rebuilds the dataset to enforce deterministic discovery order for ``ds.dims``.
    """
    input_is_xsnow = isinstance(ds, xsnowDataset)
    base_ds: xr.Dataset = ds.data if input_is_xsnow else ds

    tail = list(last_dims) if last_dims is not None else []
    dims = list(base_ds.dims)

    final: List[str] = [d for d in preferred_order if d in dims]
    seen = set(final)

    tail_set = set(tail)
    extras = [d for d in dims if d not in seen and d not in tail_set]
    final.extend(extras)

    for d in tail:
        if d in dims and d not in final:
            final.append(d)

    transposed = base_ds.transpose(*final, missing_dims="ignore")

    def _var_sort_key(var: xr.Variable) -> int:
        idxs = [final.index(d) for d in var.dims if d in final]
        return min(idxs) if idxs else len(final) + 1

    ordered_vars = OrderedDict(
        sorted(transposed.variables.items(), key=lambda item: (_var_sort_key(item[1]), item[0]))
    )
    rebuilt = xr.Dataset(ordered_vars, attrs=transposed.attrs)
    rebuilt = rebuilt.set_coords(transposed.coords.keys())

    final_ds = rebuilt.transpose(*final, missing_dims="ignore")
    return xsnowDataset(final_ds) if input_is_xsnow else final_ds


def cite(fmt: str = "bibtex", display: bool = True) -> Optional[str]:
    """
    Render the package citation from CITATION.cff without external converters.

    Args:
        fmt: Output format, one of {"biblatex", "bibtex", "apa", "copernicus"}.
            Defaults to "bibtex". "copernicus" follows the citation style
            used in Copernicus journals for software/data references.
        display: When True (default), print the rendered citation (BibTeX/BibLaTeX
            is aligned for readability). When False, return the string.

    Returns:
        The rendered citation string if display is False; otherwise None.

    Raises:
        FileNotFoundError: If CITATION.cff is missing.
        ValueError: If an unsupported format is requested.
    """
    fmt_norm = fmt.lower()
    supported = {"biblatex", "bibtex", "apa", "copernicus"}
    if fmt_norm not in supported:
        raise ValueError(f"Unsupported format '{fmt}'. Choose from {sorted(supported)}.")

    cff_path = Path(__file__).resolve().parents[2] / "CITATION.cff"
    if not cff_path.exists():
        raise FileNotFoundError(f"CITATION.cff not found at {cff_path}")

    metadata = _load_cff_metadata(cff_path)
    authors = metadata.get("authors", [])
    if not isinstance(authors, list):
        authors = []
    title = metadata.get("title")
    version = str(metadata.get("version", "")).strip() or None
    doi = metadata.get("doi")
    url = metadata.get("url")
    publisher = metadata.get("publisher")
    date_released = metadata.get("date-released")
    year, month = _year_month(date_released)

    if fmt_norm in {"bibtex", "biblatex"}:
        entry_type = "misc" if fmt_norm == "bibtex" else "software"
        bib_key = _bib_key_from_version(version)
        fields: list[tuple[str, str]] = []
        author_str = _format_author_bibtex(authors)
        if author_str:
            fields.append(("author", author_str))
        if title:
            fields.append(("title", title))
        if year:
            fields.append(("year", year))
        if month:
            fields.append(("month", month))
        if version:
            fields.append(("version", version))
        if publisher:
            fields.append(("publisher", publisher))
        if doi:
            fields.append(("doi", doi))
        if url:
            fields.append(("url", url))

        rendered = f"@{entry_type}{{{bib_key},\n"
        rendered += "\n".join(f"  {k} = {{{v}}}," for k, v in fields)
        rendered += "\n}"

        if display:
            builtins.print(_format_bib_block(rendered))
            return None
        return rendered

    if fmt_norm == "apa":
        author_str = _format_author_apa(authors)
        parts = []
        if author_str:
            parts.append(author_str)
        if year:
            parts.append(f"({year}).")
        if title:
            parts.append(f"{title}")
        if version:
            parts.append(f"(version {version}).")
        if publisher:
            parts.append(f"{publisher}.")
        url_parts = []
        if doi:
            url_parts.append(f"https://doi.org/{doi}")
        if url:
            url_parts.append(url)
        if url_parts:
            parts.append(", ".join(url_parts))
        rendered = " ".join(parts).strip()
        if not rendered.endswith("."):
            rendered += "."
        if display:
            builtins.print(rendered)
            return None
        return rendered

    if fmt_norm == "copernicus":
        author_str = _format_author_copernicus(authors)
        title_part = title or "xsnow"
        if version:
            title_part = f"{title_part} (version {version})"
        title_part = f"{title_part} [software]"

        rest_parts = [title_part]
        if publisher:
            rest_parts.append(publisher)
        url_parts = []
        if doi:
            url_parts.append(f"https://doi.org/{doi}")
        if url:
            url_parts.append(url)
        if url_parts:
            rest_parts.append(", ".join(url_parts))
        if year:
            rest_parts.append(year)

        rest = ", ".join(rest_parts)
        if author_str:
            rendered = f"{author_str}: {rest}"
        else:
            rendered = rest
        if not rendered.endswith("."):
            rendered += "."
        if display:
            builtins.print(rendered)
            return None
        return rendered


def _format_bib_block(bib_str: str) -> str:
    """
    Align BibTeX/BibLaTeX key/value pairs for nicer display.
    """
    lines = bib_str.splitlines()
    if len(lines) < 3:
        return bib_str

    header = lines[0]
    footer = lines[-1]
    body = lines[1:-1]

    key_vals = []
    for line in body:
        stripped = line.strip()
        if "=" in stripped:
            key, val = stripped.split("=", 1)
            key_vals.append((key.strip(), val.strip()))
        else:
            key_vals.append((None, stripped))

    max_key = max((len(k) for k, v in key_vals if k is not None), default=0)
    formatted_body = []
    for key, val in key_vals:
        if key is None:
            formatted_body.append(f"  {val}")
        else:
            formatted_body.append(f"  {key.ljust(max_key)} = {val}")

    return "\n".join([header, *formatted_body, footer])


def _bib_key_from_version(version: str | None) -> str:
    """
    Build a BibTeX key from the package name and version.
    Example: version 0.1.2 -> xsnow_v0_1_2
    """
    if not version:
        return "xsnow"
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", version).strip("_")
    if not normalized:
        return "xsnow"
    return f"xsnow_v{normalized}"


def _strip_value(value: str) -> str:
    stripped = value.strip()
    if (stripped.startswith('"') and stripped.endswith('"')) or (
        stripped.startswith("'") and stripped.endswith("'")
    ):
        return stripped[1:-1]
    return stripped


def _parse_cff_text(text: str) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    authors: list[dict[str, str]] = []
    in_authors = False
    current: dict[str, str] | None = None

    for line in text.splitlines():
        if not line.strip():
            continue

        if not line.startswith(" "):  # top-level key
            in_authors = False
            key, sep, value = line.partition(":")
            if not sep:
                continue
            key = key.strip()
            if key == "authors":
                in_authors = True
                meta["authors"] = authors
                current = None
            else:
                meta[key] = _strip_value(value)
            continue

        if in_authors:
            stripped = line.strip()
            if stripped.startswith("- "):
                current = {}
                authors.append(current)
                stripped = stripped[2:]
            if stripped.startswith("family-names:"):
                if current is None:
                    current = {}
                    authors.append(current)
                current["family-names"] = _strip_value(stripped.split(":", 1)[1])
            elif stripped.startswith("given-names:"):
                if current is None:
                    current = {}
                    authors.append(current)
                current["given-names"] = _strip_value(stripped.split(":", 1)[1])

    return meta


def _load_cff_metadata(cff_path: Path) -> dict[str, Any]:
    text = cff_path.read_text(encoding="utf-8")
    for loader in ("ruamel.yaml", "yaml"):
        try:
            if loader == "ruamel.yaml":
                from ruamel.yaml import YAML  # type: ignore

                yaml = YAML()
                data = yaml.load(text)
            else:
                import yaml  # type: ignore

                data = yaml.safe_load(text)
            if isinstance(data, dict):
                return data
        except Exception:
            continue
    return _parse_cff_text(text)


def _format_author_bibtex(authors: list[dict[str, str]]) -> str:
    names = []
    for author in authors:
        family = author.get("family-names")
        given = author.get("given-names")
        if family and given:
            names.append(f"{family}, {given}")
        elif family:
            names.append(family)
        elif given:
            names.append(given)
    return " and ".join(names)


def _given_to_initials(given: str) -> str:
    parts = []
    for word in given.split():
        subparts = [p for p in word.split("-") if p]
        initials = "-".join(f"{p[0].upper()}." for p in subparts if p)
        if initials:
            parts.append(initials)
    return " ".join(parts)


def _format_author_apa(authors: list[dict[str, str]]) -> str:
    formatted = []
    for author in authors:
        family = author.get("family-names")
        given = author.get("given-names")
        if family and given:
            formatted.append(f"{family}, {_given_to_initials(given)}")
        elif family:
            formatted.append(family)
        elif given:
            formatted.append(given)

    if not formatted:
        return ""
    if len(formatted) == 1:
        return formatted[0]
    if len(formatted) == 2:
        return f"{formatted[0]}, & {formatted[1]}"
    return ", ".join(formatted[:-1]) + f", & {formatted[-1]}"


def _format_author_copernicus(authors: list[dict[str, str]]) -> str:
    formatted = []
    for author in authors:
        family = author.get("family-names")
        given = author.get("given-names")
        if family and given:
            formatted.append(f"{family}, {_given_to_initials(given)}")
        elif family:
            formatted.append(family)
        elif given:
            formatted.append(given)

    if not formatted:
        return ""
    if len(formatted) == 1:
        return formatted[0]
    if len(formatted) == 2:
        return f"{formatted[0]} and {formatted[1]}"
    return ", ".join(formatted[:-1]) + f", and {formatted[-1]}"


def _year_month(date_value: str | None) -> tuple[Optional[str], Optional[str]]:
    if not date_value:
        return None, None
    parts = date_value.split("-")
    year = parts[0] if parts else None
    month = None
    if len(parts) >= 2:
        try:
            month = str(int(parts[1]))
        except ValueError:
            month = parts[1]
    return year, month


def latex_style() -> str:
    """Path of the packaged paper-ready matplotlib style."""
    from importlib.resources import files

    return str(files('xsnow') / 'resources' / 'latex.mplstyle')
