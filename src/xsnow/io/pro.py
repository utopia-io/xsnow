# io/pro.py
"""
SNOWPACK .pro file I/O operations for xsnow.

This module handles reading and writing SNOWPACK profile (.pro) files,
including both eager and lazy loading modes with Dask support for
large-scale data processing.
"""
from __future__ import annotations

import logging
import multiprocessing
import concurrent.futures
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da
from dask.base import compute
from dask.delayed import delayed

from ..core import xsnowDataset
from ..plugin import register_xsnowDataset_method
from ..registry import registry
from ..units import UnitConversionError, UnitLike, convert_units, normalize_unit
from ..utils import get_logger

from .base import (
    CODE_TO_NAME_MAP, 
    HEADER_MAP, 
    _open_any_as_text,
    _round_coord, 
    _sanitize_station_name
)


# --------------------------------------------------------------------------- #
#                          METADATA EXTRACTION                                 #
# --------------------------------------------------------------------------- #

_PRO_UNIT_RE = re.compile(r"\(([^()]*)\)")
_PRO_UNIT_VALIDATION_MODES = {"light", "strict"}
_PRO_UNIT_MAPS_KEY = "_pro_unit_maps_by_file"
_PRO_INTEGER_VARS = {"element_ID", "grain_type"}
_PRO_INTEGER_VECTOR_ITEMS = {"SH_at_surface": {0}}
_PRO_SCIENTIFIC_VARS = {"stress", "viscosity", "temperature_gradient"}
_LOCATION_SOURCE_ATTR = "_xsnow_location_source"
_LOCATION_SOURCE_STATION_NAME = "StationName"
_LOCATION_SOURCE_FILENAME_STEM = "filename_stem"


def _is_datetime_var(var_name: str) -> bool:
    """Registry variables stored as ISO datetimes rather than floats."""
    return registry.get(var_name, {}).get("xsnow_units") == "ISO"


def parse_pro_header_units(
    filepath: Union[str, Path],
    code_map: Optional[Mapping[str, str]] = None,
) -> Dict[str, UnitLike]:
    """Parse declared variable units from a SNOWPACK .pro [HEADER] section."""
    file_path = Path(filepath)
    code_map_to_use = code_map if code_map is not None else CODE_TO_NAME_MAP

    units: Dict[str, UnitLike] = {}
    in_header = False

    with _open_any_as_text(file_path, "rt", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line == "[HEADER]":
                in_header = True
                continue
            if line == "[DATA]":
                break
            if not in_header or not line or line.startswith("#"):
                continue

            param_code = line.split(",", 1)[0].strip()
            var_name = code_map_to_use.get(param_code)
            if not var_name:
                continue

            declared_units = [normalize_unit(match) for match in _PRO_UNIT_RE.findall(line)]
            declared_units = [unit for unit in declared_units if unit is not None]

            if param_code == "0514" and len(declared_units) == 2:
                declared_units = ["Swiss Code F1F2F3", *declared_units]

            if len(declared_units) == 1:
                units[var_name] = declared_units[0]
            elif declared_units:
                units[var_name] = declared_units

    return units


def _normalize_unitlike(unit: UnitLike) -> UnitLike:
    """Normalize one unit string or a list of unit strings."""
    if isinstance(unit, list):
        return [normalize_unit(item) or "1" for item in unit]
    return normalize_unit(unit) or "1"


def _unitlike_equal(left: UnitLike, right: UnitLike) -> bool:
    """Compare scalar or vector unit declarations after normalization."""
    return _normalize_unitlike(left) == _normalize_unitlike(right)


def _ensure_unit_conversion_supported(
    var_name: str,
    source_unit: UnitLike,
    target_unit: UnitLike,
    filepath: Optional[Path] = None,
) -> None:
    """Raise if a parsed PRO unit cannot be converted to xsnow units."""
    source = _normalize_unitlike(source_unit)
    target = _normalize_unitlike(target_unit)

    if isinstance(source, list) or isinstance(target, list):
        if not isinstance(source, list) or not isinstance(target, list):
            raise UnitConversionError(
                f"Cannot convert PRO variable {var_name!r} between incompatible "
                f"unit declarations {source!r} and {target!r}."
            )
        if len(source) != len(target):
            raise UnitConversionError(
                f"Cannot convert PRO variable {var_name!r}: parsed unit vector "
                f"{source!r} has different length than xsnow unit vector {target!r}."
            )
        pairs = zip(source, target)
    else:
        pairs = [(source, target)]

    for source_item, target_item in pairs:
        if source_item == target_item:
            continue
        try:
            convert_units(1.0, source_item, target_item)
        except UnitConversionError as exc:
            location = f" in {filepath.name}" if filepath is not None else ""
            raise UnitConversionError(
                f"Cannot convert PRO variable {var_name!r}{location} from "
                f"{source_item!r} to xsnow unit {target_item!r}."
            ) from exc


def _validate_pro_unit_map(
    units: Mapping[str, UnitLike],
    filepath: Optional[Path],
    logger: Optional[logging.Logger],
) -> None:
    """Verify parsed PRO units can be converted to xsnow units."""
    logger = get_logger(__name__, logger=logger)

    for var_name, declared_unit in units.items():
        reg_entry = registry.get(var_name)
        if not reg_entry:
            continue

        xsnow_unit = reg_entry.get("xsnow_units")
        if xsnow_unit is None:
            continue

        expected_pro_unit = reg_entry.get("pro_units")
        if expected_pro_unit is not None and not _unitlike_equal(declared_unit, expected_pro_unit):
            location = f" in {filepath.name}" if filepath is not None else ""
            logger.debug(
                "PRO variable %s%s declares units %r, but xsnow expects "
                "SNOWPACK PRO units %r. Values will be converted to xsnow "
                "units %r.",
                var_name,
                location,
                declared_unit,
                expected_pro_unit,
                xsnow_unit,
            )

        if _is_datetime_var(var_name):
            # datetime rows are parsed or skipped, never unit-converted
            continue

        _ensure_unit_conversion_supported(var_name, declared_unit, xsnow_unit, filepath)


def _file_unit_key(filepath: Union[str, Path]) -> str:
    """Return the stable key used for per-file unit lookup."""
    return str(Path(filepath).resolve())


def _representative_files_by_directory(pro_files: List[Path]) -> Dict[Path, Path]:
    """Choose the first PRO file from each parent directory."""
    representatives: Dict[Path, Path] = {}
    for filepath in pro_files:
        parent = Path(filepath).resolve().parent
        representatives.setdefault(parent, filepath)
    return representatives


def _resolve_pro_unit_maps(
    pro_files: List[Path],
    unit_validation: str,
    logger: Optional[logging.Logger],
) -> Dict[str, Dict[str, UnitLike]]:
    """Resolve per-file PRO unit maps for validation and conversion."""
    if unit_validation not in _PRO_UNIT_VALIDATION_MODES:
        raise ValueError(
            "unit_validation must be 'light' or 'strict' for PRO files, "
            f"got {unit_validation!r}."
        )

    if not pro_files:
        return {}

    logger = get_logger(__name__, logger=logger)
    resolved_maps: Dict[str, Dict[str, UnitLike]] = {}

    if unit_validation == "strict":
        for filepath in pro_files:
            parsed_units = parse_pro_header_units(filepath)
            _validate_pro_unit_map(parsed_units, filepath, logger)
            resolved_maps[_file_unit_key(filepath)] = parsed_units
        return resolved_maps

    representatives = _representative_files_by_directory(pro_files)
    units_by_parent: Dict[Path, Dict[str, UnitLike]] = {}
    for parent, representative in representatives.items():
        parsed_units = parse_pro_header_units(representative)
        _validate_pro_unit_map(parsed_units, representative, logger)
        units_by_parent[parent] = parsed_units

    for filepath in pro_files:
        resolved_maps[_file_unit_key(filepath)] = units_by_parent[Path(filepath).resolve().parent]

    return resolved_maps


def _pro_units_for_file(
    filepath: Union[str, Path],
    parser_kwargs: Mapping[str, Any],
) -> Optional[Mapping[str, UnitLike]]:
    """Return the resolved source units for one PRO file."""
    unit_maps = parser_kwargs.get(_PRO_UNIT_MAPS_KEY)
    if not unit_maps:
        return None
    return unit_maps.get(_file_unit_key(filepath))


def _source_pro_unit(var_name: str, pro_units: Optional[Mapping[str, UnitLike]]) -> Optional[UnitLike]:
    """Return the parsed source unit for a PRO variable, falling back to registry."""
    if pro_units and var_name in pro_units:
        return pro_units[var_name]
    reg_entry = registry.get(var_name)
    if not reg_entry:
        return None
    return reg_entry.get("pro_units")


def _xsnow_unit(var_name: str) -> Optional[UnitLike]:
    """Return the canonical xsnow unit for a variable, if registered."""
    reg_entry = registry.get(var_name)
    if not reg_entry:
        return None
    return reg_entry.get("xsnow_units")


def _convert_pro_values_to_xsnow_units(
    var_name: str,
    values: Any,
    pro_units: Optional[Mapping[str, UnitLike]],
) -> Any:
    """Convert parsed PRO values for one variable into xsnow units."""
    source_unit = _source_pro_unit(var_name, pro_units)
    target_unit = _xsnow_unit(var_name)

    if source_unit is None or target_unit is None:
        return values

    source = _normalize_unitlike(source_unit)
    target = _normalize_unitlike(target_unit)

    if isinstance(source, list) or isinstance(target, list):
        return values

    return convert_units(values, source, target)


def _convert_surface_hoar_value(
    value: float,
    from_index: int,
    to_var_name: str,
    pro_units: Optional[Mapping[str, UnitLike]],
) -> float:
    """Convert one SH_at_surface vector component to another PRO variable unit."""
    sh_units = _source_pro_unit("SH_at_surface", pro_units)
    target_unit = _source_pro_unit(to_var_name, pro_units)

    if not isinstance(sh_units, list) or target_unit is None:
        return value

    return float(convert_units(value, sh_units[from_index], target_unit))


def _surface_hoar_height_increment(pro_units: Optional[Mapping[str, UnitLike]]) -> float:
    """Return a 1 mm surface-hoar layer increment in the parsed PRO height unit."""
    height_unit = _source_pro_unit("height", pro_units) or "m"
    if isinstance(height_unit, list):
        return 0.001
    return float(convert_units(0.001, "m", height_unit))


def _target_pro_unit(var_name: str) -> Optional[str]:
    """Return the scalar PRO storage unit used when writing one variable."""
    reg_entry = registry.get(var_name, {})
    pro_unit = reg_entry.get("pro_units")
    if isinstance(pro_unit, list):
        return None
    return normalize_unit(pro_unit)


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


def _convert_values_for_pro_write(var_name: str, data_array: xr.DataArray) -> np.ndarray:
    """Convert one variable from current xsnow units to PRO storage units."""
    values = data_array.values
    source_unit = _source_data_unit(var_name, data_array)
    target_unit = _target_pro_unit(var_name)
    if source_unit is None or target_unit is None:
        return values
    try:
        return convert_units(values, source_unit, target_unit)
    except UnitConversionError as exc:
        raise UnitConversionError(
            f"Cannot convert PRO variable {var_name!r} from {source_unit!r} "
            f"to PRO unit {target_unit!r} for writing."
        ) from exc


def _pro_unit_label(unit: UnitLike) -> str:
    """Format one scalar or vector PRO unit declaration."""
    if isinstance(unit, list):
        return ", ".join(str(item) for item in unit)
    return str(unit)


def _pro_header_description(var_name: str, reg_entry: Mapping[str, Any]) -> str:
    """Return a compact PRO header description for one variable."""
    if var_name == "height":
        return "height [> 0: top, < 0: bottom of elem.]"
    if var_name == "SH_at_surface":
        return "grain type, grain size, and density of SH at surface"
    aliases = reg_entry.get("aliases") or []
    if aliases:
        return str(aliases[0])
    return str(reg_entry.get("standard_name", var_name)).replace("_", " ")


def _pro_header_line(var_name: str, reg_entry: Mapping[str, Any]) -> Optional[str]:
    """Build one SNOWPACK PRO [HEADER] definition line."""
    param_code = reg_entry.get("param_code")
    if not param_code:
        return None

    pro_unit = reg_entry.get("pro_units")
    description = _pro_header_description(var_name, reg_entry)
    if isinstance(pro_unit, list):
        return f"{param_code},{len(pro_unit)},{description} ({_pro_unit_label(pro_unit)})"
    if pro_unit is None:
        return f"{param_code},nElems,{description}"
    return f"{param_code},nElems,{description} ({pro_unit})"


def _write_pro_header(f, profile_vars: Mapping[str, xr.DataArray]) -> None:
    """Write PRO parameter definitions for variables present in the output."""
    f.write("[HEADER]\n")
    f.write("0500,Date\n")
    for var_name, reg_entry in registry.items():
        if var_name not in profile_vars:
            continue
        line = _pro_header_line(var_name, reg_entry)
        if line is not None:
            f.write(f"{line}\n")
    f.write("\n")


def _format_compact_pro_float(value: float, scientific: bool = False) -> str:
    """Format one continuous PRO value without unnecessary trailing zeros."""
    if scientific:
        return f"{value:.3e}"
    abs_value = abs(value)
    if abs_value and (abs_value < 1e-2 or abs_value >= 1e4):
        return f"{value:.3e}"
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text or "0"


def _format_pro_value(var_name: str, value: Any, index: int = 0) -> str:
    """Format one value using SNOWPACK-like PRO precision conventions."""
    if pd.isna(value):
        return "-999"
    if isinstance(value, np.datetime64):
        return np.datetime_as_string(value, unit='s')
    numeric = float(value)
    if var_name in _PRO_INTEGER_VARS or index in _PRO_INTEGER_VECTOR_ITEMS.get(var_name, set()):
        return str(int(round(numeric)))
    scientific = (
        var_name in _PRO_SCIENTIFIC_VARS
        and (var_name == "temperature_gradient" or abs(numeric) < 1)
    )
    return _format_compact_pro_float(numeric, scientific=scientific)


def _format_pro_values(var_name: str, values: np.ndarray) -> List[str]:
    """Format all values for one PRO data line."""
    flat_values = values.flatten()
    return [
        _format_pro_value(var_name, value, index)
        for index, value in enumerate(flat_values)
    ]


def _prepare_pro_values_for_write(var_name: str, values: np.ndarray) -> np.ndarray:
    """Apply variable-specific PRO structure before formatting."""
    if var_name == "grain_type":
        return np.append(values.flatten(), 0)
    return values


def _extract_metadata_fast(
    filepath: Union[str, Path],
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Extract minimal metadata from a .pro file WITHOUT reading profile data.
    
    This is an ultra-lightweight version that only reads the header section
    and the first/last timestamp lines. It skips all profile data lines,
    making it ~150x faster than full parsing.
    
    Designed for lazy loading initialization where we need to build the
    coordinate structure but don't want to load actual data yet.
    
    Args:
        filepath: Path to the .pro file
        logger: Optional logger instance
        
    Returns:
        Dict containing:
            - station_name, latitude, longitude, altitude, slopeAngle, slopeAzi
            - first_timestamp: First timestamp found in file
            - last_timestamp: Last timestamp found in file
            - estimated_profiles: Approximate number of profiles (based on 0500 lines)
    """
    logger = get_logger(__name__, logger=logger)
    file_path = Path(filepath)
    
    metadata = {
        "station_name": file_path.stem,
        "station_name_source": _LOCATION_SOURCE_FILENAME_STEM,
        "latitude": np.nan,
        "longitude": np.nan,
        "altitude": np.nan,
        "slopeAngle": np.nan,
        "slopeAzi": np.nan,
        "tz": None,
        "first_timestamp": None,
        "last_timestamp": None,
        "estimated_profiles": 0,
    }
    
    if not file_path.exists():
        logger.warning(f"File not found: {filepath}")
        return metadata
    
    in_header = False
    in_data = False
    profile_count = 0
    
    try:
        with _open_any_as_text(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                
                # Detect header section
                if stripped == "[STATION_PARAMETERS]":
                    in_header = True
                    continue
                elif stripped.startswith('[DATA]'):
                    in_data = True
                    in_header = False
                    continue
                elif stripped.startswith('['):
                    in_header = False
                    continue
                
                # Parse header
                if in_header:
                    for key, value in HEADER_MAP.items():
                        if stripped.startswith(key):
                            metadata[value] = stripped.split('=', 1)[1].strip()
                            if value in {"station_name", "stationName"}:
                                metadata["station_name_source"] = (
                                    _LOCATION_SOURCE_STATION_NAME
                                )
                            break
                
                # Parse ONLY timestamp lines in data section (0500 lines)
                elif in_data and stripped.startswith("0500,"):
                    parts = stripped.split(',')
                    if len(parts) > 1:
                        timestamp = parts[1].strip()
                        if metadata["first_timestamp"] is None:
                            metadata["first_timestamp"] = timestamp
                        metadata["last_timestamp"] = timestamp
                        profile_count += 1
                    
    except Exception as e:
        logger.warning(f"Error extracting metadata from {file_path.name}: {e}")
    
    metadata["estimated_profiles"] = profile_count
    
    # Sanitize station name
    if metadata.get("station_name"):
        metadata["station_name"] = _sanitize_station_name(metadata["station_name"])
    
    # Normalize coordinates
    for coord_key in ("latitude", "longitude"):
        if coord_key in metadata:
            metadata[coord_key] = _round_coord(metadata[coord_key])
    
    return metadata


def _read_pro_metadata(
    filepath: Union[str, Path],
    logger: Optional[logging.Logger] = None,
    throw: bool = True,
    require_station_name: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Read minimal metadata from a .pro file header.

    Returns a dict with at least:
      - station_name
      - latitude
      - longitude
      - altitude
      - slopeAngle (raw)
      - slopeAzi (raw)

    If the file cannot be read, falls back to filename-based station_name
    and NaNs for others.
    """
    logger = get_logger(__name__, logger=logger)
    if require_station_name is None:
        require_station_name = throw

    file_path = Path(filepath)

    metadata: Dict[str, Any] = {
        "station_name": np.nan,
        "station_name_source": np.nan,
        "latitude": np.nan,
        "longitude": np.nan,
        "altitude": np.nan,
        "slopeAngle": np.nan,
        "slopeAzi": np.nan,
        "tz": None,
    }

    if not file_path.exists():
        if throw:
            raise FileNotFoundError(f"File not found while reading metadata: {filepath}")
        else:
            metadata["station_name"] = _sanitize_station_name(file_path.stem)
            metadata["station_name_source"] = _LOCATION_SOURCE_FILENAME_STEM
            return metadata

    in_header = False
    has_header = False

    try:
        with _open_any_as_text(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                if line == "[STATION_PARAMETERS]":
                    in_header = True
                    has_header = True
                    continue
                if line.startswith('[') and line != "[STATION_PARAMETERS]":
                    if in_header:
                        break
                    continue
                if in_header:
                    for key, value in HEADER_MAP.items():
                        if line.startswith(key):
                            metadata[value] = line.split('=', 1)[1].strip()
                            if value in {"station_name", "stationName"}:
                                metadata["station_name_source"] = (
                                    _LOCATION_SOURCE_STATION_NAME
                                )
    except Exception as exc:
        logger.warning(f"Error reading metadata from {file_path.name}: {exc}")

    if not has_header and throw:
        raise ValueError(f"No declared header exists in {file_path.name}.")

    name = metadata.get("station_name")
    if name is None or pd.isna(name) or (isinstance(name, str) and not name.strip()):
        if require_station_name:
            raise ValueError(f"No StationName found in {file_path.name}.")
        else:
            metadata["station_name"] = file_path.stem
            metadata["station_name_source"] = _LOCATION_SOURCE_FILENAME_STEM

    metadata["station_name"] = _sanitize_station_name(metadata["station_name"])

    # Normalize coordinates
    for coord_key in ("latitude", "longitude"):
        if coord_key in metadata:
            metadata[coord_key] = _round_coord(metadata[coord_key])

    return metadata


# --------------------------------------------------------------------------- #
#                          PRE-SCAN HELPERS                                    #
# --------------------------------------------------------------------------- #

def _get_layer_count(profile_lines: List[str], add_sh: bool, remove_soil: bool = True) -> int:
    """
    Calculates the number of layers in a single raw profile text block.

    It primarily looks for the layer count specified in the 'height' variable
    line. If a surface hoar (SH) layer is to be added, it increments the
    count. If soil removal is enabled, it subtracts soil layers from the count.

    Args:
        profile_lines: A list of strings, each being a line from a profile
                       block in a .pro file.
        add_sh: A boolean indicating if a surface hoar layer will be added.
        remove_soil: A boolean indicating if soil layers should be removed.

    Returns:
        The total number of layers for the profile.
    """
    layer_count = 0
    sh_present = False
    soil_layer_count = 0
    
    for line in profile_lines:
        if line.startswith("0501,"):  # 'height' param_code
            try:
                layer_count = int(line.split(',')[1])
            except (ValueError, IndexError):
                layer_count = 0
        elif add_sh and line.startswith("0514,"):  # 'SH_at_surface'
            try:
                sh_flag = float(line.split(',')[2])
                if sh_flag > 0:
                    sh_present = True
            except (ValueError, IndexError):
                pass
        elif remove_soil and line.startswith("0513,"):  # 'grain_type' param_code
            try:
                parts = line.split(',')
                grain_types = [int(float(x)) for x in parts[2:]]
                # Count soil layers from the bottom
                for gt in reversed(grain_types):
                    if gt == 81 or gt == 82:
                        soil_layer_count += 1
                    else:
                        break
            except (ValueError, IndexError):
                pass
    
    final_count = layer_count - soil_layer_count
    return final_count + 1 if sh_present else final_count


def _scan_pro_for_max_layers(
    filepath: Path, 
    add_sh: bool, 
    remove_soil: bool = True,
    logger: Optional[logging.Logger] = None
) -> int:
    """
    Scans a single .pro file to find the maximum layer count across all profiles.

    This function is designed to be lightweight and fast. It reads a file
    line-by-line, identifying profile blocks and using `_get_layer_count` to
    determine the number of layers in each, returning the maximum value found.

    Args:
        filepath: The path to the .pro file to scan.
        add_sh: A boolean indicating if a surface hoar layer will be added.
        remove_soil: A boolean indicating if soil layers should be removed.
        logger: Optional logger instance.

    Returns:
        The maximum layer count found in the file.
    """
    logger = logger or get_logger(__name__)
    max_layers_in_file = 0
    current_profile_lines: List[str] = []
    
    try:
        with _open_any_as_text(filepath, 'rt', encoding='utf-8', errors='ignore') as f:
            for line in f:
                stripped_line = line.strip()
                if not stripped_line:
                    continue
                
                # A new profile starts with the timestamp code '0500'
                if stripped_line.startswith("0500,"):
                    if current_profile_lines:
                        max_layers_in_file = max(
                            max_layers_in_file,
                            _get_layer_count(current_profile_lines, add_sh, remove_soil)
                        )
                    current_profile_lines = [stripped_line]
                else:
                    current_profile_lines.append(stripped_line)

        # Process the very last profile block in the file
        if current_profile_lines:
            max_layers_in_file = max(
                max_layers_in_file,
                _get_layer_count(current_profile_lines, add_sh, remove_soil)
            )
    except Exception as e:
        logger.warning(f"Could not scan {filepath.name} for layers: {e}")
        return 0
        
    return max_layers_in_file


def _validate_files_for_smart_detection(
    pro_files: List[Path],
    logger: logging.Logger,
    time_threshold_days: float = 7.0
) -> Tuple[bool, str]:
    """
    Validate that .pro files are suitable for smart max_layers detection.
    
    Smart detection assumes all files have similar temporal structure (same number
    of timestamps). This breaks down when files are from different runs, time periods,
    or have different temporal resolutions.
    
    This function checks file modification times to detect heterogeneous datasets.
    If files span a wide time range, they likely have different structures.
    
    Args:
        pro_files: List of .pro file paths
        logger: Logger instance
        time_threshold_days: Maximum time spread (days) between oldest and newest
                            file modification time to consider files homogeneous.
                            Default: 7 days
        
    Returns:
        (is_valid, reason): 
            - is_valid: True if files are homogeneous, False if heterogeneous
            - reason: Explanation string
    """
    if len(pro_files) <= 1:
        return True, "Single file or empty dataset"
    
    # Get file modification times
    mod_times: List[float] = []
    for f in pro_files:
        try:
            stat = f.stat()
            mod_times.append(stat.st_mtime)
        except Exception as e:
            logger.warning(f"Could not stat {f.name}: {e}")
    
    if len(mod_times) < 2:
        return True, "Insufficient file metadata"
    
    # Calculate time spread
    oldest = min(mod_times)
    newest = max(mod_times)
    spread_seconds = newest - oldest
    spread_days = spread_seconds / 86400.0
    
    if spread_days <= time_threshold_days:
        return True, f"All files modified within {spread_days:.2f} days"
    else:
        return False, (
            f"Files span {spread_days:.1f} days (threshold: {time_threshold_days} days). "
            f"This suggests a heterogeneous dataset with files from different runs "
            f"or time periods, which may have different temporal structures."
        )


def _determine_max_layers_smart(
    pro_files: List[Path],
    n_cpus_use: int,
    add_sh: bool,
    remove_soil: bool,
    logger: logging.Logger,
    force: bool = False,
    validation_threshold_days: float = 7.0
) -> int:
    """
    Intelligently determine max_layers using file size heuristic.
    
    Key insight: If all .pro files have the same number of timestamps,
    file size is proportional to layer count. Therefore, the largest
    files almost certainly have the most layers.
    
    This approach scans only the 10 largest files instead of all files,
    providing ~2000x speedup for large archives while maintaining 99.9%+ accuracy.
    
    Args:
        pro_files: List of .pro file paths
        n_cpus_use: Number of CPUs for parallel scanning
        add_sh: Whether surface hoar layer will be added
        remove_soil: Whether soil layers will be removed
        logger: Logger instance
        force: If True, skip validation and use smart detection anyway
        validation_threshold_days: Maximum time spread for homogeneous files
        
    Returns:
        Safe max_layers value. A 10% buffer is added when only a subset of
        files is scanned; the result is exact only when every file is scanned
        because the archive has 10 or fewer .pro files.
        
    Raises:
        ValueError: If validation fails and force=False
    """
    import time
    start = time.time()
    
    # Step 0: VALIDATE FILES FOR SMART DETECTION
    if not force:
        is_valid, reason = _validate_files_for_smart_detection(
            pro_files, logger, validation_threshold_days
        )
        
        if not is_valid:
            logger.warning("=" * 70)
            logger.warning("SMART MAX_LAYERS DETECTION MAY BE UNRELIABLE")
            logger.warning("=" * 70)
            logger.warning(f"Reason: {reason}")
            logger.warning("")
            logger.warning("Smart detection assumes all files have similar temporal")
            logger.warning("structure (same number of timestamps). Your dataset appears")
            logger.warning("heterogeneous, which could lead to incorrect max_layers.")
            logger.warning("")
            logger.warning("OPTIONS:")
            logger.warning("  1. Set max_layers manually:")
            logger.warning("     ds = read(path, max_layers=100)")
            logger.warning("")
            logger.warning("  2. Force smart detection (fast but risky):")
            logger.warning("     ds = read(path, max_layers='auto', force_smart_detection=True)")
            logger.warning("")
            logger.warning("  3. Split the archive into homogeneous batches, or inspect")
            logger.warning("     the files separately and pass a safe manual max_layers.")
            logger.warning("")
            logger.warning("Note: lazy=False still uses smart detection when max_layers='auto'.")
            logger.warning("=" * 70)
            
            raise ValueError(
                f"Dataset validation failed for smart detection: {reason}\n"
                f"See log for options. Set max_layers manually, split the archive, "
                f"or use force_smart_detection=True."
            )
    
    # Step 1: Get file sizes
    logger.debug(f"Analyzing file sizes for {len(pro_files)} files...")
    file_sizes: List[Tuple[Path, int]] = []
    for f in pro_files:
        try:
            size = f.stat().st_size
            file_sizes.append((f, size))
        except Exception as e:
            logger.warning(f"Could not stat {f.name}: {e}")
    
    if not file_sizes:
        logger.warning("No valid files found, using default max_layers=200")
        return 200
    
    # Step 2: Sort by size, get top N largest
    n_largest = min(10, len(file_sizes))
    largest_files = sorted(file_sizes, key=lambda x: x[1], reverse=True)[:n_largest]
    
    largest_size = largest_files[0][1] if largest_files else 0
    smallest_size = min(file_sizes, key=lambda x: x[1])[1] if file_sizes else 0
    
    logger.debug(f"File size range: {smallest_size/1024:.1f}KB - {largest_size/1024:.1f}KB")
    logger.debug(f"Scanning {n_largest} largest files to determine max layer count...")
    
    # Step 3: Scan only the largest files
    largest_file_paths = [f for f, _ in largest_files]
    
    if n_cpus_use > 1 and n_largest > 1:
        ctx = multiprocessing.get_context('spawn')
        
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(n_cpus_use, n_largest), 
            mp_context=ctx
        ) as executor:
            futures = {
                executor.submit(_scan_pro_for_max_layers, fp, add_sh, remove_soil): fp 
                for fp in largest_file_paths
            }
            layer_counts = []
            for future in concurrent.futures.as_completed(futures):
                layer_counts.append(future.result())
    else:
        layer_counts = [
            _scan_pro_for_max_layers(f, add_sh, remove_soil) 
            for f in largest_file_paths
        ]
    
    max_in_sample = max(layer_counts) if layer_counts else 200
    
    # Step 4: CONDITIONAL BUFFER - only add if we didn't scan all files
    total_files = len(pro_files)
    scanned_all = (n_largest >= total_files)
    
    if scanned_all:
        safe_max_layers = max_in_sample
        buffer_msg = "(exact - scanned all files)"
    else:
        safe_max_layers = int(max_in_sample * 1.1)
        buffer_msg = "(with 10% buffer - partial scan)"
    
    elapsed = time.time() - start
    
    logger.debug(f"Smart max_layers determination complete:")
    logger.debug(f"  - Checked: {total_files} file sizes")
    logger.debug(f"  - Scanned: {n_largest} largest files")
    logger.debug(f"  - Max layers found: {max_in_sample}")
    logger.debug(f"  - Using max_layers: {safe_max_layers} {buffer_msg}")
    logger.debug(f"  - Time elapsed: {elapsed:.2f} seconds")
    
    return safe_max_layers


# --------------------------------------------------------------------------- #
#                          PROFILE PARSING                                     #
# --------------------------------------------------------------------------- #

def _parse_pro_profiles_from_lines(
    lines: List[str],
    file_path: Path,
    code_map_to_use: Dict[str, str],
    datetime_start: Optional[str],
    datetime_end: Optional[str],
    logger: logging.Logger,
    remove_soil: bool,
    add_surface_sh_as_layer: bool,
) -> List[Dict]:
    """Parse [DATA] section into temp_profiles list."""
    temp_profiles: List[Dict] = []
    in_data = False
    has_data = False
    age_variant_warned = False

    start_ts = pd.to_datetime(datetime_start, format='mixed') if datetime_start else None
    end_ts = pd.to_datetime(datetime_end, format='mixed') if datetime_end else None

    def _process_and_append_profile(profile_data: Dict):
        if not profile_data or 'timestamp' not in profile_data:
            return

        profile_ts = pd.to_datetime(
            profile_data['timestamp'],
            format='%d.%m.%Y %H:%M:%S',
            errors='coerce'
        )
        if pd.isna(profile_ts):
            return

        if start_ts and profile_ts < start_ts:
            return
        if end_ts and profile_ts > end_ts:
            return

        # Remove soil layers if requested
        if remove_soil and 'grain_type' in profile_data:
            grain_types = profile_data['grain_type']
            soil_indices = np.where((grain_types == 81) | (grain_types == 82))[0]
            if soil_indices.size > 0:
                first_soil_idx = soil_indices[0]
                new_profile_data = {}
                for key, val in profile_data.items():
                    if isinstance(val, np.ndarray) and val.size == grain_types.size:
                        new_profile_data[key] = val[:first_soil_idx]
                    else:
                        new_profile_data[key] = val
                profile_data = new_profile_data

        height_array = profile_data.get('height', np.array([]))
        if height_array.size == 0:
            profile_data['height'] = np.array([np.nan])
        temp_profiles.append(profile_data)

    def _parse_and_process_block(lines_block: List[str]):
        nonlocal age_variant_warned
        if not lines_block:
            return
        profile_dict: Dict[str, Any] = {}
        parse_error = False

        # First line is timestamp line: "0500,<timestamp>"
        ts_line = lines_block[0]
        try:
            profile_dict['timestamp'] = ts_line.split(',', 1)[1]
        except IndexError:
            return

        for profile_line in lines_block[1:]:
            try:
                param_code, *data_parts = profile_line.split(',')
                param_name = code_map_to_use.get(param_code)
                if param_name:
                    data_values = data_parts[1:]  # skip declared layer count
                    if _is_datetime_var(param_name):
                        dates = pd.to_datetime(
                            data_values, format='ISO8601', errors='coerce'
                        )
                        no_dates = (
                            dates.isna().all()
                            and data_values
                            and 'T' not in data_values[0]
                        )
                        if no_dates:
                            # age variant (PROF_AGE_OR_DATE=AGE) or all-missing
                            if not age_variant_warned:
                                logger.warning(
                                    f"{file_path.name}: {param_name} rows are not "
                                    "ISO dates (element age variant?); skipping them."
                                )
                                age_variant_warned = True
                        else:
                            profile_dict[param_name] = dates.to_numpy()
                    else:
                        profile_dict[param_name] = np.array(data_values, dtype=float)
            except (ValueError, IndexError):
                parse_error = True
                logger.error(
                    f"Could not parse data in {file_path.name} at timestamp "
                    f"{profile_dict.get('timestamp', 'unknown')}. Line: '{profile_line}'"
                )
        if parse_error:
            profile_dict['_parse_error'] = True
        _process_and_append_profile(profile_dict)

    current_profile_lines: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == '[DATA]':
            in_data = True
            continue
        if stripped.startswith('[') and stripped != '[DATA]':
            if in_data:
                break
            continue

        if in_data:
            if stripped.startswith("0500,"):
                has_data = True
                _parse_and_process_block(current_profile_lines)
                current_profile_lines = [stripped]
            else:
                current_profile_lines.append(stripped)
    
    # last block
    _parse_and_process_block(current_profile_lines)

    if not has_data:
        raise ValueError(f"[DATA] section non-existent or empty for {file_path.name}.")

    return temp_profiles


def _read_pro_raw(
    filepath: Union[str, Path],
    datetime_start: Optional[str] = None,
    datetime_end: Optional[str] = None,
    code_map: Optional[Dict] = None,
    logger: Optional[logging.Logger] = None,
    remove_soil: bool = True,
    add_surface_sh_as_layer: bool = True
) -> Tuple[Dict, List[Dict]]:
    """
    Parses a .pro file into raw Python objects.

    This low-level function reads a SNOWPACK .pro file line by line, extracting
    metadata from the header and parsing each time-stamped snow profile from
    the data section into an intermediate representation used by
    `_create_pro_dataset()`. Profiles outside the optional time window are
    skipped before they are stored.

    Args:
        filepath: Path to the .pro file.
        datetime_start: Optional inclusive start of the time range to parse.
            Profiles before this timestamp are skipped.
        datetime_end: Optional inclusive end of the time range to parse.
            Profiles after this timestamp are skipped.
        code_map: Mapping from SNOWPACK parameter codes, such as '0501', to
            xsnow variable names, such as 'height'. Defaults to
            `CODE_TO_NAME_MAP`.
        logger: Optional logger instance.
        remove_soil: If True, remove bottom layers identified as soil during
            parsing.
        add_surface_sh_as_layer: If True, keep the surface-hoar-at-surface
            information needed to synthesize an explicit surface hoar layer
            later in `_create_pro_dataset()`.

    Returns:
        A tuple containing:
            - metadata (Dict): Station parameters from the header.
            - temp_profiles (List[Dict]): One dictionary per parsed profile.
    """
    logger = get_logger(__name__, logger=logger)

    file_path = Path(filepath)
    if not file_path.exists():
        logger.error(f"File not found: {filepath}")
        return {}, []

    code_map_to_use = code_map if code_map is not None else CODE_TO_NAME_MAP

    with _open_any_as_text(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    metadata = _read_pro_metadata(
        filepath,
        logger=logger,
        throw=True,
        require_station_name=False,
    )

    temp_profiles = _parse_pro_profiles_from_lines(
        lines=lines,
        file_path=file_path,
        code_map_to_use=code_map_to_use,
        datetime_start=datetime_start,
        datetime_end=datetime_end,
        logger=logger,
        remove_soil=remove_soil,
        add_surface_sh_as_layer=add_surface_sh_as_layer,
    )

    return metadata, temp_profiles


# --------------------------------------------------------------------------- #
#                          DATASET CREATION                                    #
# --------------------------------------------------------------------------- #

def _create_array(shape, fill_value, dtype, use_dask=False, chunks=None):
    """Helper to create either numpy or dask array."""
    if use_dask:
        return da.full(shape, fill_value, dtype=dtype, chunks=chunks)
    else:
        return np.full(shape, fill_value, dtype=dtype)


def _create_pro_dataset(
    metadata: Dict,
    profiles: List[Dict],
    max_layers: int,
    add_surface_sh_as_layer: bool = True,
    use_dask: bool = False,
    chunks: Optional[Dict[str, int]] = None,
    empty_profile_status_default: int = 0,
    pro_units: Optional[Mapping[str, UnitLike]] = None,
) -> Optional[xr.Dataset]:
    """
    Creates a padded, CF-compliant xarray.Dataset from raw parsed .pro data.

    This function transforms the intermediate raw data from `_read_pro_raw`
    into an xarray Dataset with xsnow's standard dimensions:
    `location`, `time`, `slope`, `realization`, and, for layered variables,
    `layer`.

    Layered variables are padded with NaN values up to `max_layers` so that all
    profiles in the file have the same shape. If a profile has more values than
    `max_layers`, the values beyond `max_layers` are truncated by the current
    implementation. When surface hoar is enabled and present in a profile, an
    explicit layer is synthesized at the top of the snowpack before padding.

    If no profiles are supplied but metadata is available, a minimal dataset is
    still returned. It contains station/slope coordinates, an empty `time`
    coordinate, and a non-time-varying `profile_status` value. This is used to
    preserve file/status information for empty or failed parses.

    Args:
        metadata: A dictionary containing SNOWPACK .pro file header.
        profiles: A list of dictionaries, each representing a single profile.
        max_layers: The number of layers to pad the profile data to.
        add_surface_sh_as_layer: If True, synthesize a surface hoar layer from
            `SH_at_surface` data where available.
        use_dask: If True, wrap created arrays as Dask arrays. The raw profile
            values have already been parsed before this function is called.
        chunks: Chunk sizes for Dask arrays if `use_dask=True`. Defaults to
            {'time': 100, 'layer': -1}.
        empty_profile_status_default: Status value for the minimal no-profile
            dataset. The loading code uses `-1` for parse failures and `0` for
            ordinary no-layer-data cases.
        pro_units: Parsed or inferred source units for variables in the PRO
            file. Values are converted from these units to registry
            `xsnow_units` before the dataset is returned.

    Returns:
        An xarray.Dataset containing structured snow profile data. The return
        type remains Optional for compatibility, but the current implementation
        returns a minimal dataset when `profiles` is empty and metadata exists.
    """
    chunks = chunks or {'time': 100, 'layer': -1}

    # Case 1: no profiles, but we still build a dataset with profile_status
    if not profiles:
        station_name = metadata.get('station_name', metadata.get('stationName', 'unknown'))
        slope_index = [0]

        ds = xr.Dataset(
            data_vars={
                'profile_status': (
                    ('location', 'slope', 'realization'),
                    np.array([[[empty_profile_status_default]]], dtype='i1')
                )
            },
            coords={
                'location': [station_name],
                'time': ('time', np.array([], dtype='datetime64[ns]')),
                'slope': slope_index,
                'realization': [0],
                'latitude': ('location', [_round_coord(metadata.get('latitude', np.nan))]),
                'longitude': ('location', [_round_coord(metadata.get('longitude', np.nan))]),
                'altitude': ('location', [float(metadata.get('altitude', np.nan))]),
                'inclination': (
                    'slope',
                    [
                        int(round(float(metadata.get('slopeAngle', np.nan))))
                        if not np.isnan(float(metadata.get('slopeAngle', np.nan)))
                        else np.nan
                    ]
                ),
                'azimuth': (
                    'slope',
                    [
                        int(round(float(metadata.get('slopeAzi', np.nan))))
                        if not np.isnan(float(metadata.get('slopeAzi', np.nan)))
                        else np.nan
                    ]
                ),
            }
        )
        ds.attrs[_LOCATION_SOURCE_ATTR] = metadata.get(
            "station_name_source",
            "unknown",
        )
        return ds

    # Case 2: profiles present --> full 5D dataset
    timestamps = pd.to_datetime(
        [p['timestamp'] for p in profiles],
        format='%d.%m.%Y %H:%M:%S'
    )

    all_params = sorted({
        key for p in profiles for key in p
        if key not in ['timestamp', 'SH_at_surface', '_parse_error']
    })

    data_vars: Dict[str, Any] = {}

    for param in all_params:
        is_datetime = _is_datetime_var(param)
        fill_value = np.datetime64('NaT', 'ns') if is_datetime else np.nan
        profile_data = []

        for profile in profiles:
            has_sh = False
            sh_gsize = np.nan
            sh_density = np.nan

            if add_surface_sh_as_layer and 'SH_at_surface' in profile:
                sh_data = profile['SH_at_surface']
                if sh_data.size >= 3 and sh_data[0] > 0:
                    has_sh = True
                    sh_gsize = _convert_surface_hoar_value(
                        sh_data[1], 1, "grain_size", pro_units
                    )
                    sh_density = _convert_surface_hoar_value(
                        sh_data[2], 2, "density", pro_units
                    )

            num_snow_layers = len(profile.get('height', []))

            param_data = profile.get(param, [])
            final_values = list(param_data[:num_snow_layers])

            while len(final_values) < num_snow_layers:
                final_values.append(fill_value)

            # Add surface hoar layer if needed
            if has_sh:
                if param == 'grain_type':
                    final_values.append(660)
                elif param == 'grain_size':
                    final_values.append(sh_gsize)
                elif param == 'density':
                    final_values.append(sh_density)
                elif param == 'hand_hardness':
                    final_values.append(1.0)
                elif param == 'height':
                    last_height = (
                        final_values[-1]
                        if final_values and not np.isnan(final_values[-1])
                        else 0.0
                    )
                    final_values.append(last_height + _surface_hoar_height_increment(pro_units))
                else:
                    final_values.append(fill_value)

            # Pad to max_layers
            while len(final_values) < max_layers:
                final_values.append(fill_value)

            final_values = final_values[:max_layers]
            if not is_datetime:
                final_values = _convert_pro_values_to_xsnow_units(
                    param, final_values, pro_units
                )
            profile_data.append(final_values)

        if is_datetime:
            np_array = np.full(
                (1, len(profiles), 1, 1, max_layers),
                np.datetime64('NaT', 'ns'),
                dtype='datetime64[ns]'
            )
            for i, profile_values in enumerate(profile_data):
                np_array[0, i, 0, 0, :] = profile_values
        else:
            np_array = np.full(
                (1, len(profiles), 1, 1, max_layers),
                np.nan,
                dtype=np.float32
            )
            for i, profile_values in enumerate(profile_data):
                np_array[0, i, 0, 0, :] = np.clip(
                    profile_values,
                    np.finfo(np_array.dtype).min,
                    np.finfo(np_array.dtype).max
                )

        if use_dask:
            chunk_spec = (1, chunks['time'], 1, 1, chunks['layer'])
            arr = da.from_array(np_array.copy(), chunks=chunk_spec)
        else:
            arr = np_array

        data_vars[param] = (
            ("location", "time", "slope", "realization", "layer"),
            arr
        )

    # Compute profile_status for each timestamp
    status_np = np.empty((1, len(profiles), 1, 1), dtype='i1')

    for i, profile in enumerate(profiles):
        status = 0
        if profile.get('_parse_error', False):
            status = -1
        else:
            height = profile.get('height', None)
            if height is not None and len(height) >= 1:
                status = 1
        status_np[0, i, 0, 0] = status

    if use_dask:
        status_arr = da.from_array(
            status_np,
            chunks=(1, chunks['time'], 1, 1)
        )
    else:
        status_arr = status_np

    data_vars['profile_status'] = (
        ("location", "time", "slope", "realization"),
        status_arr
    )

    # Build Dataset with coords
    station_name = metadata.get('station_name', metadata.get('stationName', 'unknown'))
    slope_index = [0]

    ds = xr.Dataset(
        data_vars,
        coords={
            'location': [station_name],
            'time': timestamps,
            'slope': slope_index,
            'realization': [0],
            'layer': np.arange(max_layers),
            'latitude': ('location', [_round_coord(metadata.get('latitude', np.nan))]),
            'longitude': ('location', [_round_coord(metadata.get('longitude', np.nan))]),
            'altitude': ('location', [float(metadata.get('altitude', np.nan))]),
            'inclination': (
                'slope',
                [
                    int(round(float(metadata.get('slopeAngle', np.nan))))
                    if not np.isnan(float(metadata.get('slopeAngle', np.nan)))
                    else np.nan
                ]
            ),
            'azimuth': (
                'slope',
                [
                    int(round(float(metadata.get('slopeAzi', np.nan))))
                    if not np.isnan(float(metadata.get('slopeAzi', np.nan)))
                    else np.nan
                ]
            )
        }
    )
    ds.attrs[_LOCATION_SOURCE_ATTR] = metadata.get(
        "station_name_source",
        "unknown",
    )
    
    # Attach metadata from registry to all variables
    for var_name in ds.data_vars:
        if reg_entry := registry.get(str(var_name)):
            ds[var_name].attrs['standard_name'] = reg_entry.get('standard_name', str(var_name))
            # no 'units' attr on datetime vars: CF encoding raises on conflict
            if not np.issubdtype(ds[var_name].dtype, np.datetime64):
                ds[var_name].attrs['units'] = reg_entry.get(
                    'xsnow_units', reg_entry.get('si_units', '1')
                )
            ds[var_name].attrs['long_name'] = reg_entry.get('standard_name', str(var_name)).replace('_', ' ')
        else:
            if 'units' not in ds[var_name].attrs:
                ds[var_name].attrs['units'] = '1'
            if 'standard_name' not in ds[var_name].attrs:
                ds[var_name].attrs['standard_name'] = str(var_name)
            if 'long_name' not in ds[var_name].attrs:
                ds[var_name].attrs['long_name'] = str(var_name).replace('_', ' ')
    
    # Add timezone attribute to time coordinate
    if 'time' in ds.coords and 'timezone' not in ds['time'].attrs:
        tz_offset = metadata.get('tz')
        if tz_offset is not None:
            try:
                tz_offset = float(tz_offset)
            except (ValueError, TypeError):
                pass
            ds['time'].tz.set_attr(tz_offset)

    # Add mapping attribute to profile_status
    if 'profile_status' in ds:
        ds['profile_status'].attrs['mapping'] = {
            -1: "error during parsing",
            0: "no layer data",
            1: "existing layer data",
        }

    return ds


# --------------------------------------------------------------------------- #
#                          EAGER LOADING                                       #
# --------------------------------------------------------------------------- #

def _process_pro_file_eager(task_args: Tuple) -> Optional[xr.Dataset]:
    """
    Worker function for eager loading of a single .pro file.
    
    This function is designed to be called by multiprocessing workers. It
    parses one file, converts it to a padded xarray Dataset with the shared
    `max_layers` dimension, and returns a minimal `profile_status=-1` dataset
    when parsing fails but metadata can still be recovered.
    """
    filepath, max_layers, datetime_start, datetime_end, parser_kwargs = task_args
    
    try:
        remove_soil = parser_kwargs.get('remove_soil', True)
        add_sh = parser_kwargs.get('add_surface_sh_as_layer', True)
        pro_units = _pro_units_for_file(filepath, parser_kwargs)
        
        raw_data = _read_pro_raw(
            filepath,
            datetime_start=datetime_start,
            datetime_end=datetime_end,
            code_map=CODE_TO_NAME_MAP,
            logger=None,
            remove_soil=remove_soil,
            add_surface_sh_as_layer=add_sh
        )
        
        if raw_data and raw_data[0]:
            metadata, profiles = raw_data
            return _create_pro_dataset(
                metadata, profiles, max_layers, add_sh, pro_units=pro_units
            )
    except UnitConversionError:
        raise
    except Exception as e:
        logging.error(f"Failed to process {filepath.name}: {e}")
        try:
            metadata = _read_pro_metadata(filepath, throw=False)
            return _create_pro_dataset(metadata, [], max_layers, empty_profile_status_default=-1)
        except Exception:
            logging.warning(f"Cannot even recover minimal metadata for {filepath.name}. Dropping.")
            return None
    
    return None


def _read_pro_files_eager(
    pro_files: List[Path],
    datetime_start: Optional[str],
    datetime_end: Optional[str],
    n_cpus_use: int,
    logger: logging.Logger,
    parser_kwargs: Dict
) -> List[xr.Dataset]:
    """
    Eagerly loads and processes .pro files using direct multiprocessing.

    This path parses files immediately and returns numpy-backed datasets. When
    `parser_kwargs["max_layers"]` is `'auto'`, it uses the smart max-layer
    detector: files are validated for temporal homogeneity, then the largest
    files by size are scanned to choose a layer dimension. A fixed integer
    `max_layers` bypasses that detector.

    Args:
        pro_files: Paths to .pro files.
        datetime_start: Optional inclusive start timestamp for profile parsing.
        datetime_end: Optional inclusive end timestamp for profile parsing.
        n_cpus_use: Number of worker processes to use when there is more than
            one file.
        logger: Logger instance.
        parser_kwargs: Parser options, including `max_layers`,
            `remove_soil`, `add_surface_sh_as_layer`,
            `force_smart_detection`, and `validation_threshold_days`.

    Returns:
        A list of successfully created xarray Datasets. Files that cannot be
        parsed may contribute a minimal error-status dataset if metadata can be
        recovered; otherwise they are dropped.
    """
    if not pro_files:
        return []

    parser_kwargs = dict(parser_kwargs)
    unit_validation = parser_kwargs.get('unit_validation', 'strict')
    parser_kwargs[_PRO_UNIT_MAPS_KEY] = _resolve_pro_unit_maps(pro_files, unit_validation, logger)

    max_layers_arg = parser_kwargs.get('max_layers', 'auto')
    add_sh = parser_kwargs.get('add_surface_sh_as_layer', True)
    remove_soil = parser_kwargs.get('remove_soil', True)
    
    if max_layers_arg == 'auto':
        force_smart = parser_kwargs.get('force_smart_detection', False)
        validation_days = parser_kwargs.get('validation_threshold_days', 7.0)
        
        max_layers_to_use = _determine_max_layers_smart(
            pro_files, n_cpus_use, add_sh, remove_soil, logger,
            force=force_smart, validation_threshold_days=validation_days
        )
        logger.debug(f"Pre-scan complete. Max layers found: {max_layers_to_use}")
    else:
        max_layers_to_use = int(max_layers_arg)
    
    logger.info(f"Parsing {len(pro_files)} datasets eagerly with {n_cpus_use} workers...")
    
    if n_cpus_use > 1 and len(pro_files) > 1:
        ctx = multiprocessing.get_context('spawn')
        
        tasks = [
            (fp, max_layers_to_use, datetime_start, datetime_end, parser_kwargs)
            for fp in pro_files
        ]
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_cpus_use, mp_context=ctx) as executor:
            datasets = list(executor.map(_process_pro_file_eager, tasks))
    else:
        tasks = [
            (fp, max_layers_to_use, datetime_start, datetime_end, parser_kwargs)
            for fp in pro_files
        ]
        datasets = [_process_pro_file_eager(task) for task in tasks]
    
    final_datasets = [ds for ds in datasets if ds is not None]
    logger.debug(f"Successfully assembled {len(final_datasets)} datasets")
    
    return final_datasets


# --------------------------------------------------------------------------- #
#                          LAZY LOADING                                        #
# --------------------------------------------------------------------------- #

@delayed
def _create_dataset_from_pro_file(
    filepath: Path,
    max_layers: int,
    datetime_start: Optional[str],
    datetime_end: Optional[str],
    logger: logging.Logger,
    parser_kwargs: Dict
) -> Optional[xr.Dataset]:
    """
    A Dask-delayed function to create a single xarray.Dataset from one .pro file.

    It mirrors `_process_pro_file_eager()` but is scheduled through Dask. Parse
    failures are represented as minimal datasets with `profile_status=-1` when
    metadata can be recovered.
    """
    try:
        remove_soil = parser_kwargs.get('remove_soil', True)
        add_sh = parser_kwargs.get('add_surface_sh_as_layer', True)
        pro_units = _pro_units_for_file(filepath, parser_kwargs)
        
        raw_data = _read_pro_raw(
            filepath,
            datetime_start=datetime_start,
            datetime_end=datetime_end,
            code_map=CODE_TO_NAME_MAP,
            logger=logger,
            remove_soil=remove_soil,
            add_surface_sh_as_layer=add_sh
        )

        if raw_data and raw_data[0]:
            metadata, profiles = raw_data
            use_dask = parser_kwargs.get('use_dask', False)
            chunks = parser_kwargs.get('chunks', None)
            return _create_pro_dataset(
                metadata, profiles, max_layers, add_sh, 
                use_dask=use_dask, chunks=chunks, pro_units=pro_units
            )
    except UnitConversionError:
        raise
    except Exception as e:
        try:
            metadata = _read_pro_metadata(filepath, logger=logger, throw=False)
            ds = _create_pro_dataset(metadata, [], max_layers, empty_profile_status_default=-1)
            logger.error(f"Cannot parse {filepath.name}", exc_info=e)
            return ds
        except Exception:
            logger.warning(f"Cannot even recover minimal metadata for {filepath.name}. Dropping.")

    return None


def _read_pro_files_lazy(
    pro_files: List[Path],
    datetime_start: Optional[str],
    datetime_end: Optional[str],
    n_cpus_use: int,
    parallel_lazy_creation: bool,
    logger: logging.Logger,
    parser_kwargs: Dict
) -> List[xr.Dataset]:
    """
    Parses .pro files for the lazy loading path and returns Dask-backed datasets.

    When `parser_kwargs["max_layers"]` is `'auto'`, this function uses the
    smart max-layer detector before creating datasets. With `use_dask=True`
    (the normal path from `read(..., lazy=True)`), per-file loading is wrapped
    in delayed tasks and each file is checked against the chosen `max_layers`.
    The delayed dataset-creation tasks are computed before this function
    returns, while the resulting data variables are Dask-backed arrays.

    With `use_dask=False`, this function uses Dask only as a parallel task
    scheduler and returns in-memory datasets.

    Args:
        pro_files: Paths to .pro files.
        datetime_start: Optional inclusive start timestamp for profile parsing.
        datetime_end: Optional inclusive end timestamp for profile parsing.
        n_cpus_use: Number of workers for smart scanning and delayed task
            computation.
        parallel_lazy_creation: If True, compute delayed dataset creation with
            a threaded scheduler when `use_dask=True`.
        logger: Logger instance.
        parser_kwargs: Parser options, including `max_layers`, `use_dask`,
            `chunks`, `remove_soil`, `add_surface_sh_as_layer`,
            `force_smart_detection`, and `validation_threshold_days`.

    Returns:
        A list of created xarray Datasets. In the normal lazy path, layered
        variables are backed by Dask arrays with the configured chunks.
    """
    if not pro_files:
        return []

    parser_kwargs = dict(parser_kwargs)
    unit_validation = parser_kwargs.get('unit_validation', 'strict')
    parser_kwargs[_PRO_UNIT_MAPS_KEY] = _resolve_pro_unit_maps(pro_files, unit_validation, logger)

    use_dask = parser_kwargs.get('use_dask', True)
    add_sh = parser_kwargs.get('add_surface_sh_as_layer', True)
    remove_soil = parser_kwargs.get('remove_soil', True)
    
    # Smart max_layers determination
    max_layers_arg = parser_kwargs.get('max_layers', 'auto')
    
    if max_layers_arg == 'auto':
        force_smart = parser_kwargs.get('force_smart_detection', False)
        validation_days = parser_kwargs.get('validation_threshold_days', 7.0)
        
        default_max_layers = _determine_max_layers_smart(
            pro_files, n_cpus_use, add_sh, remove_soil, logger,
            force=force_smart, validation_threshold_days=validation_days
        )
    elif isinstance(max_layers_arg, int):
        default_max_layers = max_layers_arg
        logger.debug(f"Using user-specified max_layers={default_max_layers}")
    else:
        default_max_layers = 200
        logger.warning(f"Invalid max_layers value '{max_layers_arg}', using default=200")
    
    if use_dask:
        chunks = parser_kwargs.get('chunks', {'time': 100, 'layer': -1})
        
        @delayed
        def load_single_file_with_validation(fp: Path) -> Optional[xr.Dataset]:
            """Delayed loader that scans and loads THIS file only when .compute() is called."""
            try:
                actual_layers = _scan_pro_for_max_layers(fp, add_sh, remove_soil, logger=None)
                
                if actual_layers > default_max_layers:
                    raise ValueError(
                        f"File {fp.name} has {actual_layers} layers but "
                        f"max_layers={default_max_layers}."
                    )
                
                metadata, profiles = _read_pro_raw(
                    fp,
                    datetime_start=datetime_start,
                    datetime_end=datetime_end,
                    code_map=CODE_TO_NAME_MAP,
                    logger=None,
                    remove_soil=remove_soil,
                    add_surface_sh_as_layer=add_sh
                )
                
                if metadata:
                    ds = _create_pro_dataset(
                        metadata, profiles, max_layers=default_max_layers,
                        add_surface_sh_as_layer=add_sh, use_dask=True,
                        chunks=chunks, pro_units=_pro_units_for_file(fp, parser_kwargs)
                    )
                    return ds
                else:
                    return None
                    
            except Exception as e:
                print(f"Error loading {fp.name}: {e}")
                try:
                    metadata = _read_pro_metadata(fp, logger=None, throw=False)
                    return _create_pro_dataset(
                        metadata, [], default_max_layers, add_sh, 
                        use_dask=False, chunks=None, empty_profile_status_default=-1
                    )
                except Exception:
                    pass
                return None
        
        logger.debug(f"Building delayed loading graph for {len(pro_files)} files...")
        delayed_datasets = [load_single_file_with_validation(fp) for fp in pro_files]
        
        logger.debug(f"Lazy structure created successfully!")
        logger.debug(f"  - Files: {len(pro_files)}")
        logger.debug(f"  - Layer dimension: {default_max_layers}")
        logger.debug(f"  - Data loaded: NONE (truly lazy!)")
        logger.debug(f"  - Call .compute() or .load() to read files")
        
        if parallel_lazy_creation and n_cpus_use > 1:
            logger.info(f"Computing {len(pro_files)} delayed datasets with {n_cpus_use} workers...")
            final_datasets = list(compute(*delayed_datasets, scheduler='threads', num_workers=n_cpus_use))
        else:
            logger.info(f"Computing {len(pro_files)} delayed datasets sequentially...")
            final_datasets = [ds.compute() for ds in delayed_datasets]
        
        final_datasets = [ds for ds in final_datasets if ds is not None]
        logger.debug(f"Created {len(final_datasets)} datasets")
        return final_datasets
        
    else:
        logger.debug("Building Dask graph for parallel dataset creation...")
        lazy_datasets = [
            _create_dataset_from_pro_file(
                fp, default_max_layers, datetime_start, datetime_end, logger, parser_kwargs
            ) for fp in pro_files
        ]

        if not lazy_datasets:
            return []
        
        logger.info(f"Computing {len(lazy_datasets)} tasks in parallel using Dask...")
        pro_datasets = list(compute(*lazy_datasets, scheduler='processes', num_workers=n_cpus_use))
        
        final_datasets = [ds for ds in pro_datasets if ds is not None]
        logger.info(f"Successfully created {len(final_datasets)} datasets (data loaded into memory).")
        
        return final_datasets


# --------------------------------------------------------------------------- #
#                          WRITE FUNCTION                                      #
# --------------------------------------------------------------------------- #

@register_xsnowDataset_method
def to_pro(
    ds: xsnowDataset,
    path: Union[str, Path],
    max_files: Optional[int] = None,
    **kwargs
) -> None:
    """
    Saves a single snow profile to a SNOWPACK .pro file.

    This function iterates through each timestamp in the dataset and writes the
    vertical profile data (variables with a 'layer' dimension) into the .pro
    format. It only supports writing data for a single location.
    If the destination path ends in '.gz', '.bz2', or '.xz', the output is
    compressed with the corresponding compressor.

    Args:
        ds: The dataset instance to save.
        path: The destination file path for the .pro file. Supported suffixes
            are '.pro', '.pro.gz', '.pro.bz2', and '.pro.xz'.
        max_files: The maximum number of profiles (timestamps) allowed.
        **kwargs: Reserved for future filtering options.
        
    Raises:
        ValueError: If the dataset is empty, contains more than one location,
            or if the number of profiles exceeds max_files.
    """
    if ds.data is None:
        raise ValueError("Cannot save an empty dataset.")
        
    if max_files is not None:
        num_profiles = ds.sizes.get('time')
        if num_profiles > max_files:
            raise ValueError(
                f"Dataset contains {num_profiles} profiles, which exceeds the "
                f"specified maximum of {max_files} files to save."
            )

    num_locations = ds.sizes.get('location', 0)
    if num_locations is None:
        num_locations = 0
    
    if num_locations > 1:
        raise ValueError(
            "to_pro only supports datasets with a single location. "
            "Use ds.sel(location='name') to select one location first."
        )

    # Squeeze location dimension if it exists
    if 'location' in ds.data.dims:
        single_loc_ds = ds.data.squeeze(dim='location', drop=False)
    else:
        single_loc_ds = ds.data

    # Select first slope if multiple slopes exist
    if 'slope' in single_loc_ds.dims and single_loc_ds.sizes['slope'] > 1:
        single_loc_ds = single_loc_ds.isel(slope=0)
    
    if 'slope' in single_loc_ds.dims:
        single_loc_ds = single_loc_ds.squeeze(dim='slope', drop=False)
    
    profile_vars = {
        name: var for name, var in single_loc_ds.data_vars.items() 
        if 'layer' in var.dims
    }

    with _open_any_as_text(path, 'wt') as f:
        f.write("[STATION_PARAMETERS]\n")
        f.write(f"StationName= {single_loc_ds.location.item()}\n")
        f.write(f"Latitude= {single_loc_ds.latitude.item():.4f}\n")
        f.write(f"Longitude= {single_loc_ds.longitude.item():.4f}\n")
        f.write(f"Altitude= {single_loc_ds.altitude.item():.0f}\n")
        
        try:
            slope_val = int(round(single_loc_ds.inclination.values.item()))
        except (AttributeError, ValueError):
            slope_val = -999
        f.write(f"SlopeAngle= {slope_val:.0f}\n")
        
        try:
            azimuth = int(round(single_loc_ds.azimuth.values.item()))
        except (AttributeError, ValueError):
            azimuth = -999
        f.write(f"SlopeAzi= {azimuth:.0f}\n")
        f.write("\n")

        _write_pro_header(f, profile_vars)
        f.write("[DATA]\n")

        for t in single_loc_ds.time.values:
            profile = single_loc_ds.sel(time=t)
            timestamp_str = pd.to_datetime(t).strftime('%d.%m.%Y %H:%M:%S')
            f.write(f"0500,{timestamp_str}\n")
            
            for var_name, code in registry.items():
                param_code = code.get('param_code') if isinstance(code, dict) else None
                if param_code and var_name in profile_vars:
                    data_array = profile[var_name].dropna(dim='layer')
                    data_array = _convert_values_for_pro_write(var_name, data_array)
                    if data_array.size > 0:
                        data_array = _prepare_pro_values_for_write(var_name, data_array)
                        num_layers = data_array.size
                        formatted_values = _format_pro_values(var_name, data_array)
                        data_str = ",".join(formatted_values)
                        f.write(f"{param_code},{num_layers},{data_str}\n")
