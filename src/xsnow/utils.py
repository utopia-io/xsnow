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
from typing import Union, Any, Optional, List, Sequence

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
    Render the package citation from CITATION.cff using cffconvert.

    Args:
        fmt: Output format, one of {"biblatex", "bibtex", "apa", "csljson"}.
            Defaults to "bibtex" to avoid BibLaTeX warnings; use "biblatex"
            when available for @software entries.
        display: When True (default), print the rendered citation (BibTeX/BibLaTeX
            is aligned for readability). When False, return the string.

    Returns:
        The rendered citation string if display is False; otherwise None.

    Raises:
        RuntimeError: If cffconvert is not installed (with install hint).
        FileNotFoundError: If CITATION.cff is missing.
        ValueError: If an unsupported format is requested.
    """
    fmt_norm = fmt.lower()
    preferred_methods = {
        "biblatex": ("as_biblatex", "as_bibtex"),  # fall back to bibtex if biblatex is unavailable
        "bibtex": ("as_bibtex",),
        "apa": ("as_apalike",),
        "csljson": ("as_csljson", "as_csl"),  # cffconvert versions vary on naming
    }
    if fmt_norm not in preferred_methods:
        raise ValueError(f"Unsupported format '{fmt}'. Choose from {sorted(preferred_methods)}.")

    try:
        from cffconvert import Citation
    except ImportError as exc:
        raise RuntimeError(
            "cffconvert is required for cite(). Install with `pip install .[release]` "
            "or `pip install cffconvert ruamel.yaml`. Citation info is also available at "
            "https://xsnow.avacollabra.org."
        ) from exc

    cff_path = Path(__file__).resolve().parents[2] / "CITATION.cff"
    if not cff_path.exists():
        raise FileNotFoundError(f"CITATION.cff not found at {cff_path}")

    cff_text = cff_path.read_text(encoding="utf-8")
    citation = Citation(cff_text)
    bib_key = _bib_key_from_cff(cff_text)

    for method_name in preferred_methods[fmt_norm]:
        if hasattr(citation, method_name):
            if fmt_norm == "biblatex" and method_name != "as_biblatex":
                warnings.warn(
                    "cffconvert does not expose biblatex in this environment; falling back to BibTeX (@misc).",
                    RuntimeWarning,
                )
            if fmt_norm == "csljson" and method_name != "as_csljson":
                warnings.warn(
                    "cffconvert does not expose csljson; falling back to csl.",
                    RuntimeWarning,
                )
            formatter = getattr(citation, method_name)
            rendered = formatter()
            if fmt_norm in {"bibtex", "biblatex"}:
                rendered = _override_bib_key(rendered, key=bib_key)
            if display:
                if fmt_norm in {"bibtex", "biblatex"}:
                    builtins.print(_format_bib_block(rendered))
                else:
                    builtins.print(rendered)
                return None
            return rendered

    raise RuntimeError(
        f"cffconvert does not support '{fmt_norm}' with available methods. "
        f"Found methods: {', '.join(sorted(m for m in dir(citation) if m.startswith('as_')))}"
    )


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

def _override_bib_key(bib_str: str, key: str = "xsnow") -> str:
    """
    Replace the default cffconvert key (YourReferenceHere) with a stable one.
    """
    return bib_str.replace("YourReferenceHere", key, 1)

def _bib_key_from_cff(cff_text: str) -> str:
    """
    Build a BibTeX key from the package name and the CFF version.
    Example: version 0.1.2 -> xsnow_v0_1_2
    """
    match = re.search(r"^version:\s*[\"']?([^\s\"']+)", cff_text, re.MULTILINE)
    if not match:
        return "xsnow"
    version = match.group(1)
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", version).strip("_")
    if not normalized:
        return "xsnow"
    return f"xsnow_v{normalized}"

