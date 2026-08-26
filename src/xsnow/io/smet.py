# io/smet.py
"""
SMET file I/O operations for xsnow.

SMET (Swiss Meteo Exchange Text) is a text-based format for meteorological
time series data. This module handles reading and writing SMET files.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

from ..core import xsnowDataset
from ..plugin import register_xsnowDataset_method
from ..registry import registry
from ..units import UnitConversionError, UnitLike, convert_units, normalize_unit
from ..utils import get_logger

from .base import (
    _apply_float_dtype,
    _open_any_as_text,
    _round_coord,
    _sanitize_station_name,
    FloatDtypeLike,
)


@dataclass(frozen=True)
class SmetUnitSpec:
    """Describe unit metadata inferred for one SMET field."""

    field: str
    plot_unit: Optional[str]
    units_multiplier: float
    units_offset: float
    storage_unit: Optional[str]
    ambiguous: bool = False


_SMET_MULTIPLIER_UNIT_MAP = {
    ("m", 1.0): "m",
    ("m", 0.01): "cm",
    ("m", 0.001): "mm",
    ("K", 1.0): "K",
    ("W m-2", 1.0): "W m-2",
    ("MJ m-2", 1.0): "MJ m-2",
    ("kg m-2", 1.0): "kg m-2",
    ("kg m-2 h-1", 1.0): "kg m-2 h-1",
    ("m s-1", 1.0): "m s-1",
    ("degree", 1.0): "degree",
}
_SMET_UNIT_VALIDATION_MODES = {"light", "strict"}
SmetUnitType = Literal["input", "forcing", "output"]


_SMET_UNIT_TOKENS = {
    "ISO": "ISO8601",
    "1": "1",
    "%": "%",
    "degree": "deg",
    "degC": "degC",
    "K": "K",
    "m": "m",
    "cm": "cm",
    "mm": "mm",
    "Pa": "Pa",
    "kPa": "kPa",
    "W m-2": "W/m2",
    "J m-2": "J/m2",
    "MJ m-2": "MJ/m2",
    "kg m-2": "kg/m2",
    "kg m-2 h-1": "kg/m2/h",
    "kg m-2 s-1": "kg/m2/s",
    "m s-1": "m/s",
}
_SMET_INTEGER_VARS = {
    "profile_status",
}


def _normalize_unitlike(unit: UnitLike) -> UnitLike:
    """Normalize one unit string or a list of unit strings."""
    if isinstance(unit, list):
        return [normalize_unit(item) or "1" for item in unit]
    return normalize_unit(unit) or "1"


def _unit_allowed(unit: str, allowed_units: UnitLike) -> bool:
    """Return True when a unit matches one of the registry-declared SMET units."""
    normalized_unit = normalize_unit(unit)
    normalized_allowed = _normalize_unitlike(allowed_units)
    if isinstance(normalized_allowed, list):
        return normalized_unit in normalized_allowed
    return normalized_unit == normalized_allowed


def _fallback_smet_unit(reg_entry: Mapping[str, Any]) -> Optional[str]:
    """Return an unambiguous registry SMET unit, otherwise None."""
    smet_units = reg_entry.get("smet_units")
    if isinstance(smet_units, list):
        return smet_units[0] if len(smet_units) == 1 else None
    return normalize_unit(smet_units)


def _validate_smet_unit_type(unit_type: str) -> None:
    """Raise for unsupported SMET writer unit types."""
    if unit_type not in ("input", "forcing", "output"):
        raise ValueError(
            "unit_type must be 'input', 'forcing', or 'output' for SMET writing, "
            f"got {unit_type!r}."
        )


def _normalize_smet_unit_type(unit_type: SmetUnitType) -> Literal["input", "output"]:
    """Normalize SMET writer unit-type aliases."""
    return "input" if unit_type == "forcing" else unit_type


def _smet_unit_token(unit: Optional[str]) -> str:
    """Return a whitespace-free SMET unit token for a canonical unit."""
    normalized = normalize_unit(unit) or "1"
    return _SMET_UNIT_TOKENS.get(normalized, str(normalized).replace(" ", "_"))


def _source_data_unit(var_name: str, data_array: xr.DataArray) -> Optional[str]:
    """Return the current data unit from attrs, falling back to xsnow_units."""
    attr_unit = data_array.attrs.get("units")
    if attr_unit is not None:
        return normalize_unit(attr_unit)
    reg_entry = registry.get(var_name, {})
    xsnow_unit = reg_entry.get("xsnow_units")
    if isinstance(xsnow_unit, list):
        return None
    return normalize_unit(xsnow_unit)


def _target_smet_unit(var_name: str, data_array: xr.DataArray, unit_type: SmetUnitType) -> str:
    """Return the storage unit to write for one SMET variable."""
    reg_entry = registry.get(var_name)
    if not reg_entry:
        return _source_data_unit(var_name, data_array) or "1"

    typed_unit = reg_entry.get(f"smet_units_{_normalize_smet_unit_type(unit_type)}")
    if typed_unit is not None:
        return normalize_unit(typed_unit) or "1"

    smet_units = reg_entry.get("smet_units")
    if isinstance(smet_units, list):
        current_unit = _source_data_unit(var_name, data_array)
        normalized_allowed = _normalize_unitlike(smet_units)
        if current_unit is not None and isinstance(normalized_allowed, list):
            if normalize_unit(current_unit) in normalized_allowed:
                return normalize_unit(current_unit) or "1"
        xsnow_unit = reg_entry.get("xsnow_units")
        if isinstance(normalized_allowed, list) and not isinstance(xsnow_unit, list):
            normalized_xsnow = normalize_unit(xsnow_unit)
            if normalized_xsnow in normalized_allowed:
                return normalized_xsnow or "1"
        return normalize_unit(smet_units[0]) or "1"

    fallback_unit = smet_units or reg_entry.get("xsnow_units")
    if isinstance(fallback_unit, list):
        return "1"
    return normalize_unit(fallback_unit) or "1"


def _convert_series_for_smet_write(
    var_name: str,
    data_array: xr.DataArray,
    values: np.ndarray,
    target_unit: str,
) -> np.ndarray:
    """Convert one SMET column from current units to selected storage units."""
    source_unit = _source_data_unit(var_name, data_array)
    if source_unit is None:
        return values
    try:
        return convert_units(values, source_unit, target_unit)
    except UnitConversionError as exc:
        raise UnitConversionError(
            f"Cannot convert SMET variable {var_name!r} from {source_unit!r} "
            f"to SMET unit {target_unit!r} for writing."
        ) from exc


def _format_compact_smet_float(value: float) -> str:
    """Format one continuous SMET value with compact, Snowpack-like precision."""
    abs_value = abs(value)
    if abs_value and (abs_value < 1e-3 or abs_value >= 1e5):
        return f"{value:.6e}"
    return f"{value:.3f}"


def _format_smet_value(var_name: str, value: Any, nodata: float) -> str:
    """Format one SMET data value with variable-aware precision."""
    if pd.isna(value):
        return f"{nodata:.1f}"
    if isinstance(value, str):
        return value

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)

    if var_name in _SMET_INTEGER_VARS:
        return str(int(round(numeric)))
    return _format_compact_smet_float(numeric)


def _timestamp_from_smet_index(index_value: Any, index_names: List[Optional[str]]) -> Any:
    """Extract the time-like value from a SMET DataFrame index entry."""
    if isinstance(index_value, tuple):
        for time_name in ("time", "timestamp"):
            if time_name in index_names:
                return index_value[index_names.index(time_name)]
        return index_value[0]
    return index_value


def _write_smet_data_rows(path: Union[str, Path], df: pd.DataFrame, nodata: float) -> None:
    """Append SMET data rows using column-aware value formatting."""
    with _open_any_as_text(path, 'at') as f:
        for timestamp, row in df.iterrows():
            timestamp = _timestamp_from_smet_index(timestamp, list(df.index.names))
            timestamp_str = pd.to_datetime(timestamp).strftime('%Y-%m-%dT%H:%M:%S')
            values = [
                _format_smet_value(name, row[name], nodata)
                for name in df.columns
            ]
            f.write(f"{timestamp_str}\t" + "\t".join(values) + "\n")


def _validate_smet_unit_validation_mode(unit_validation: str) -> None:
    """Raise for unsupported SMET unit validation modes."""
    if unit_validation not in _SMET_UNIT_VALIDATION_MODES:
        raise ValueError(
            "unit_validation must be 'light' or 'strict' for SMET files, "
            f"got {unit_validation!r}."
        )


def _convert_smet_var_to_xsnow_units(
    ds: xr.Dataset,
    var_name: str,
    spec: Optional[SmetUnitSpec],
    reg_entry: Mapping[str, Any],
    filepath: Path,
    logger: logging.Logger,
) -> xr.Dataset:
    """Validate and convert one SMET variable to xsnow units when possible."""
    source_unit = spec.storage_unit if spec is not None else None
    assumed_from_registry = False
    if source_unit is None:
        source_unit = _fallback_smet_unit(reg_entry)
        assumed_from_registry = source_unit is not None
    if source_unit is None:
        logger.warning(
            "SMET variable %s in %s has no usable unit metadata; leaving "
            "values unchanged with units='unknown'.",
            var_name,
            filepath.name,
        )
        return ds

    if assumed_from_registry:
        logger.warning(
            "SMET variable %s in %s has no usable unit metadata; assuming "
            "registry SMET unit %r before conversion.",
            var_name,
            filepath.name,
            source_unit,
        )

    expected_smet_units = reg_entry.get("smet_units")
    if expected_smet_units is not None and not _unit_allowed(source_unit, expected_smet_units):
        raise UnitConversionError(
            f"SMET variable {var_name!r} in {filepath.name} declares storage "
            f"unit {source_unit!r}, but xsnow expects {expected_smet_units!r}."
        )

    target_unit = reg_entry.get("xsnow_units")
    if target_unit is None:
        return ds
    if isinstance(target_unit, list):
        raise UnitConversionError(
            f"Cannot convert SMET variable {var_name!r} to vector xsnow units "
            f"{target_unit!r}."
        )

    try:
        converted = convert_units(ds[var_name].values, source_unit, target_unit)
    except UnitConversionError as exc:
        raise UnitConversionError(
            f"Cannot convert SMET variable {var_name!r} in {filepath.name} "
            f"from {source_unit!r} to xsnow unit {target_unit!r}."
        ) from exc

    ds[var_name] = xr.DataArray(
        converted,
        coords=ds[var_name].coords,
        dims=ds[var_name].dims,
        attrs=dict(ds[var_name].attrs),
    )
    ds[var_name].attrs["units"] = target_unit
    return ds


def _parse_smet_header(filepath: Union[str, Path]) -> Dict[str, str]:
    """Parse key/value lines from a SMET [HEADER] section."""
    header: Dict[str, str] = {}
    file_path = Path(filepath)
    in_header = False

    with _open_any_as_text(file_path, "rt", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line == "[HEADER]":
                in_header = True
                continue
            if line == "[DATA]":
                break
            if in_header and "=" in line:
                key, value = line.split("=", 1)
                header[key.strip()] = value.strip()

    return header


def _split_header_values(value: Optional[str]) -> list[str]:
    """Split a SMET header list value into tokens."""
    return [] if value is None else value.split()


def _split_numeric_header_values(value: Optional[str], default: float, n: int) -> list[float]:
    """Split a SMET numeric list, padding missing metadata with defaults."""
    if value is None:
        return [default] * n
    values = [float(part) for part in value.split()]
    if len(values) == n:
        return values
    if len(values) < n:
        return values + [default] * (n - len(values))
    raise ValueError(f"Expected at most {n} unit values, got {len(values)}.")


def _get_smet_units_declaration(header: Mapping[str, str]) -> list[str]:
    """Return units declared by a SMET units header or commented #units line."""
    return _split_header_values(header.get("units") or header.get("#units"))


def infer_smet_storage_unit(
    field: str,
    plot_unit: Optional[str],
    units_multiplier: float = 1.0,
    units_offset: float = 0.0,
    registry_entry: Optional[Mapping[str, Any]] = None,
    use_registry_for_ambiguous: bool = False,
) -> tuple[Optional[str], bool]:
    """
    Infer a SMET data-storage unit from plot metadata.

    SMET plot metadata describes how a value should be displayed. In common
    SNOWPACK files, displayed value = stored value * units_multiplier +
    units_offset. Dimensionless fields with ``plot_unit = -`` are treated as
    fractions when the multiplier is 1 and as percentages when the multiplier
    is 0.01.
    """
    normalized_plot_unit = normalize_unit(plot_unit)
    if field == "timestamp" or normalized_plot_unit == "ISO":
        return "ISO", False
    if normalized_plot_unit is None:
        return None, True

    if normalized_plot_unit == "1":
        if np.isclose(units_offset, 0.0):
            if np.isclose(units_multiplier, 0.01):
                return "%", False
            if np.isclose(units_multiplier, 1.0):
                return "1", False
        if use_registry_for_ambiguous and registry_entry is not None:
            expected = registry_entry.get("smet_units")
            if expected is not None:
                return normalize_unit(expected), True
        return None, True

    if np.isclose(units_offset, 273.15) and normalized_plot_unit == "K":
        return "degC", False

    key = (normalized_plot_unit, float(units_multiplier))
    if key in _SMET_MULTIPLIER_UNIT_MAP:
        return _SMET_MULTIPLIER_UNIT_MAP[key], False

    if np.isclose(units_multiplier, 1.0) and np.isclose(units_offset, 0.0):
        return normalized_plot_unit, False

    return None, True


def parse_smet_unit_specs(
    filepath: Union[str, Path],
    registry_map: Optional[Mapping[str, Mapping[str, Any]]] = None,
    use_registry_for_ambiguous: bool = False,
) -> Dict[str, SmetUnitSpec]:
    """Parse SMET unit metadata and infer storage units per field."""
    header = _parse_smet_header(filepath)
    fields = _split_header_values(header.get("fields"))
    if not fields:
        raise ValueError(f"SMET file {filepath} is missing a fields header.")

    plot_units = _split_header_values(header.get("plot_unit"))
    declared_units = _get_smet_units_declaration(header)

    if declared_units and len(declared_units) != len(fields):
        raise ValueError(
            f"SMET field/unit metadata length mismatch in {filepath}: "
            f"{len(fields)} fields, {len(declared_units)} declared units."
        )

    if not plot_units:
        plot_units = [None] * len(fields)
    if len(plot_units) != len(fields):
        raise ValueError(
            f"SMET field/unit metadata length mismatch in {filepath}: "
            f"{len(fields)} fields, {len(plot_units)} plot units."
        )

    multipliers = _split_numeric_header_values(
        header.get("units_multiplier"), default=1.0, n=len(fields)
    )
    offsets = _split_numeric_header_values(
        header.get("units_offset"), default=0.0, n=len(fields)
    )
    registry_to_use = registry if registry_map is None else registry_map

    specs: Dict[str, SmetUnitSpec] = {}
    for index, (field, plot_unit, multiplier, offset) in enumerate(
        zip(fields, plot_units, multipliers, offsets)
    ):
        if declared_units:
            storage_unit = normalize_unit(declared_units[index])
            ambiguous = False
        else:
            storage_unit, ambiguous = infer_smet_storage_unit(
                field=field,
                plot_unit=plot_unit,
                units_multiplier=multiplier,
                units_offset=offset,
                registry_entry=registry_to_use.get(field),
                use_registry_for_ambiguous=use_registry_for_ambiguous,
            )
        specs[field] = SmetUnitSpec(
            field=field,
            plot_unit=normalize_unit(plot_unit),
            units_multiplier=multiplier,
            units_offset=offset,
            storage_unit=storage_unit,
            ambiguous=ambiguous,
        )

    return specs


def _read_smet_files(
    smet_files: List[Path],
    datetime_start: Optional[str],
    datetime_end: Optional[str],
    logger: logging.Logger,
    unit_validation: str = "strict",
) -> List[xr.Dataset]:
    """
    Reads a list of .smet files into xarray.Dataset objects.

    Args:
        smet_files: A list of file paths to .smet files.
        datetime_start: An optional start timestamp for filtering.
        datetime_end: An optional end timestamp for filtering.
        logger: A configured logger instance.
        unit_validation: SMET unit validation mode. Both `'strict'` and
            `'light'` parse each file's SMET unit metadata, warn for missing
            unit metadata, and raise for incompatible declared units.

    Returns:
        A list of xarray.Dataset objects, one for each successfully
        processed .smet file.
    """
    _validate_smet_unit_validation_mode(unit_validation)

    smet_datasets = []
    for fp in smet_files:
        try:
            if ds := read_smet(
                fp,
                datetime_start,
                datetime_end,
                logger=logger,
                unit_validation=unit_validation,
                # read()'s float_dtype policy is applied once, at the merge
                # funnel
                float_dtype=None,
            ):
                smet_datasets.append(ds)
        except UnitConversionError:
            raise
        except Exception as e:
            logger.error(f"Failed to read .smet file {fp.name}", exc_info=e)
    return smet_datasets


def read_smet(
    filepath: Union[str, Path],
    datetime_start: Optional[str] = None,
    datetime_end: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    unit_validation: str = "strict",
    float_dtype: FloatDtypeLike = "float32",
) -> Optional[xr.Dataset]:
    """
    Parses a single SMET file into an xarray.Dataset.

    Reads a SMET-formatted text file, parsing the header for metadata and the
    data section into a pandas DataFrame, which is then converted into an
    xarray.Dataset. The data is filtered by the specified time range.

    Slope metadata from the SMET header is stored as ``inclination`` and
    ``azimuth`` coordinates when available. If no slope metadata is present,
    the returned dataset contains one provisional slope with NaN slope
    coordinates. During a later ``xsnow.read()`` merge with profile data, such
    no-slope SMET data can be broadcast to all profile slopes at the same
    location.

    When SMET unit metadata is present, storage units are inferred from
    ``plot_unit`` plus ``units_multiplier``/``units_offset`` or from explicit
    ``units`` declarations, validated against the registry, and converted to
    xsnow units before returning.

    Args:
        filepath: Path to the SMET file.
        datetime_start: The start of the time range to read. Data before
            this time is dropped.
        datetime_end: The end of the time range to read. Data after
            this time is dropped.
        logger: An optional, pre-configured logger instance. If None,
            the default module logger is used.
        unit_validation: SMET unit validation mode. Both `'strict'` (default)
            and `'light'` parse this file's unit metadata, warn when registered
            variables have no usable unit metadata, and raise for incompatible
            declared units.
        float_dtype: Floating-point storage dtype; see `read()`. None
            preserves parsed dtypes (float64).

    Returns:
        An xarray.Dataset containing the SMET data, or None if the file
        cannot be parsed or contains no data in the specified time range.
    """
    logger_input = None if isinstance(logger, bool) else logger
    logger = get_logger(__name__, logger=logger_input)
    _validate_smet_unit_validation_mode(unit_validation)

    file_path = Path(filepath)
    if not file_path.exists():
        logger.error(f"File not found: {filepath}")
        return None

    header = {}
    data_start_line = -1
    with _open_any_as_text(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if line.strip() == '[DATA]':
                data_start_line = i + 1
                break
            if '=' in line:
                key, value = line.split('=', 1)
                header[key.strip()] = value.strip()

    if 'fields' not in header or data_start_line == -1:
        logger.error(f"SMET file {filepath} is missing 'fields' or '[DATA]' section.")
        return None

    unit_specs = parse_smet_unit_specs(file_path)

    # Normalize coordinates early
    for coord_key in ("latitude", "longitude"):
        if coord_key in header:
            header[coord_key] = _round_coord(header[coord_key])

    variable_heights = {
        key.replace('_height', ''): value.strip()
        for key, value in header.items() if '_height' in key
    }

    original_names = header['fields'].split()
    new_column_names = []
    column_specs: Dict[str, Optional[SmetUnitSpec]] = {}
    for name in original_names:
        if name in variable_heights:
            new_name = f"{name}_{variable_heights[name]}m"
        else:
            new_name = name
        new_column_names.append(new_name)
        column_specs[new_name] = unit_specs.get(name)
    
    df = pd.read_csv(
        file_path,
        skiprows=data_start_line,
        sep=r'\s+',
        header=None,
        names=new_column_names,
        na_values=header.get('nodata', -999)
    )

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')

    if datetime_start:
        df = df[df.index >= pd.to_datetime(datetime_start, format='mixed')]
    if datetime_end:
        df = df[df.index <= pd.to_datetime(datetime_end, format='mixed')]
        
    if df.empty:
        return None

    ds = xr.Dataset.from_dataframe(df)
    ds = ds.rename({'timestamp': 'time'})
    
    station_name = header.get('station_name', header.get('station_id', file_path.stem))
    station_name = _sanitize_station_name(station_name)
    
    # Check if dimensions already exist before expanding
    dims_to_add = {}
    if 'location' not in ds.dims and 'location' not in ds.coords:
        dims_to_add['location'] = [station_name]
    if 'slope' not in ds.dims and 'slope' not in ds.coords:
        dims_to_add['slope'] = [0]
    if 'realization' not in ds.dims and 'realization' not in ds.coords:
        dims_to_add['realization'] = [0]
    
    if dims_to_add:
        ds = ds.expand_dims(dims_to_add)
    
    tz_val = header.get('tz')
    if tz_val is not None:
        try:
            tz_val = float(tz_val)
        except (ValueError, TypeError):
            pass  # allow string input of timezone names
        ds['time'].tz.set_attr(tz_val)
            
    # Add coordinates if they don't exist
    if 'latitude' not in ds.coords:
        ds = ds.assign_coords({
            'latitude': ('location', [_round_coord(header.get('latitude', np.nan))]),
            'longitude': ('location', [_round_coord(header.get('longitude', np.nan))]),
            'altitude': ('location', [float(header.get('altitude', np.nan))]),
        })
    
    # Try to get slope values from SMET header before defaulting to NaN
    if 'inclination' not in ds.coords:
        slope_incl = header.get('SlopeAngle', header.get('slopeAngle', 
                    header.get('slope_angle', header.get('SlopeAzi='))))
        slope_azi = header.get('SlopeAzi', header.get('slopeAzi', 
                    header.get('slope_azi', header.get('SlopeAzi='))))
        
        # Convert to float, defaulting to NaN if not available or invalid
        try:
            slope_incl = float(slope_incl) if slope_incl is not None else np.nan
        except (ValueError, TypeError):
            slope_incl = np.nan
            
        try:
            slope_azi = float(slope_azi) if slope_azi is not None else np.nan
        except (ValueError, TypeError):
            slope_azi = np.nan

        ds = ds.assign_coords({
            'inclination': ('slope', [slope_incl]),
            'azimuth': ('slope', [slope_azi])
        })
    
    # Attach metadata from registry to all variables
    for var_name in ds.data_vars:
        if reg_entry := registry.get(str(var_name)):
            ds = _convert_smet_var_to_xsnow_units(
                ds=ds,
                var_name=str(var_name),
                spec=column_specs.get(str(var_name)),
                reg_entry=reg_entry,
                filepath=file_path,
                logger=logger,
            )
            ds[var_name].attrs['standard_name'] = reg_entry.get('standard_name', str(var_name))
            ds[var_name].attrs.setdefault('units', 'unknown')
            ds[var_name].attrs['long_name'] = reg_entry.get('standard_name', str(var_name)).replace('_', ' ')
        else:
            # Set default attributes for variables not in registry
            spec = column_specs.get(str(var_name))
            if 'units' not in ds[var_name].attrs:
                ds[var_name].attrs['units'] = (
                    spec.storage_unit
                    if spec is not None and spec.storage_unit is not None
                    else '1'
                )
            if 'standard_name' not in ds[var_name].attrs:
                ds[var_name].attrs['standard_name'] = str(var_name)
            if 'long_name' not in ds[var_name].attrs:
                ds[var_name].attrs['long_name'] = str(var_name).replace('_', ' ')

    return _apply_float_dtype(ds, float_dtype)


def _require_single_smet_series(ds: xsnowDataset) -> None:
    """Reject dimensions that cannot be represented by one SMET time series."""
    for dim in ('location', 'slope', 'realization'):
        size = ds.sizes.get(dim)
        if size is not None and size > 1:
            raise ValueError(
                "to_smet only supports datasets with a single location, slope, "
                f"and realization. Use ds.sel({dim}=...) to select one {dim} first."
            )


def _scalar_slope_value(ds: xsnowDataset, *coord_names: str) -> Optional[float]:
    """Return the first present finite scalar slope descriptor, or None."""
    for coord_name in coord_names:
        if coord_name not in ds.coords:
            continue
        values = np.asarray(ds[coord_name].values).ravel()
        if values.size == 0:
            continue
        if values.size != 1:
            raise ValueError(
                f"SMET slope coordinate {coord_name!r} must contain exactly one "
                "value. Select a single location, slope, and realization first."
            )
        try:
            value = float(values.item())
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"SMET slope coordinate {coord_name!r} must contain a numeric value."
            ) from exc
        if np.isfinite(value):
            return value
    return None


@register_xsnowDataset_method
def to_smet(
    ds: xsnowDataset,
    path: Union[str, Path],
    max_files: Optional[int] = None,
    unit_type: SmetUnitType = "output",
    write_slope_metadata: bool = True,
    **kwargs
) -> None:
    """
    Saves time-series data from the dataset to a SMET file.

    This function extracts data variables that do not have a 'layer' dimension
    (e.g., meteorological data, total snow height) and writes them into the
    SMET format. It only supports writing data for a single location, slope,
    and realization.
    If the destination path ends in '.gz', '.bz2', or '.xz', the output is
    compressed with the corresponding compressor.

    Args:
        ds: The dataset instance to save.
        path: The destination file path for the .smet file. Supported suffixes
            are '.smet', '.smet.gz', '.smet.bz2', and '.smet.xz'.
        max_files: The maximum number of locations allowed.
            If the dataset contains more locations, a ValueError is raised.
            Defaults to None (no limit).
        unit_type: SMET storage-unit convention to write. `'output'`
            (default) writes SNOWPACK output-style units where known, while
            `'input'` and its alias `'forcing'` write forcing/input-style units
            where known.
        write_slope_metadata: Whether to write available ``SlopeAngle`` and
            ``SlopeAzi`` header fields. Defaults to True. Set to False when one
            forcing SMET time series per location is valid for all slopes. The
            dataset must still contain at most one slope; select the shared
            forcing time series first if necessary.
        **kwargs: Reserved for future filtering options.
    """
    if ds.data is None:
        raise ValueError("Cannot save an empty dataset.")
    _validate_smet_unit_type(unit_type)
    
    if max_files is not None:
        num_locations = ds.sizes.get('location')
        if num_locations and num_locations > max_files:
            raise ValueError(
                f"Dataset contains {num_locations} locations, which exceeds the "
                f"specified maximum of {max_files} files to save."
            )
    
    _require_single_smet_series(ds)

    smet_vars = [name for name, var in ds.data.data_vars.items() if 'layer' not in var.dims]
    if not smet_vars:
        logging.warning("Warning: No time-series variables found to write to SMET file.")
        return

    slope_angle = None
    slope_azi = None
    if write_slope_metadata:
        slope_angle = _scalar_slope_value(ds, "inclination", "SlopeAngle")
        slope_azi = _scalar_slope_value(ds, "azimuth", "SlopeAzi")
        if slope_azi is not None and slope_angle is None:
            raise ValueError("Cannot write SlopeAzi without a finite SlopeAngle.")

    # Squeeze and convert to dataframe, keeping only data variables
    df = ds.data[smet_vars].squeeze(drop=True).to_dataframe()[smet_vars]
    smet_units = {
        name: _target_smet_unit(name, ds.data[name], unit_type)
        for name in smet_vars
    }
    for name, target_unit in smet_units.items():
        if name in registry:
            df[name] = _convert_series_for_smet_write(
                name,
                ds.data[name],
                df[name].to_numpy(),
                target_unit,
            )
    nodata = -999.0

    with _open_any_as_text(path, 'wt') as f:
        f.write("SMET 1.1 ASCII\n")
        f.write("[HEADER]\n")
        station_id = str(ds.location.item())
        f.write(f"station_id = {station_id}\n")
        f.write(f"station_name = {station_id}\n")
        # Handle coordinates that may have extra dimensions (e.g., slope)
        lat = float(ds.latitude.values.flat[0]) if ds.latitude.size > 1 else ds.latitude.item()
        lon = float(ds.longitude.values.flat[0]) if ds.longitude.size > 1 else ds.longitude.item()
        alt = float(ds.altitude.values.flat[0]) if ds.altitude.size > 1 else ds.altitude.item()
        f.write(f"latitude = {lat:.4f}\n")
        f.write(f"longitude = {lon:.4f}\n")
        f.write(f"altitude = {alt:.0f}\n")
        if slope_angle is not None:
            f.write(f"SlopeAngle = {slope_angle:.1f}\n")
        if slope_azi is not None:
            f.write(f"SlopeAzi = {slope_azi:.1f}\n")
        f.write(f"nodata = {nodata:.1f}\n")
        fields = "timestamp " + " ".join(df.columns)
        f.write(f"fields = {fields}\n")
        unit_tokens = " ".join(_smet_unit_token(smet_units[name]) for name in df.columns)
        f.write(f"units = ISO8601 {unit_tokens}\n")
        f.write("[DATA]\n")

    df.index.name = 'timestamp'
    _write_smet_data_rows(path, df, nodata)
    logging.info(f"Dataset successfully saved to {path}")
    
