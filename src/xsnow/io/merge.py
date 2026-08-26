# io/merge.py
"""
Dataset merging and file resolution utilities for xsnow I/O.

This module contains functions for:
- Resolving file paths from various source types
- Merging multiple datasets into a unified xsnowDataset
- Applying final filters and coordinate normalization
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union, Literal

import numpy as np
import pandas as pd
import xarray as xr

from ..core import xsnowDataset
from ..registry import registry
from ..selectors import TimeSelector, LocationSelector, SlopeSelector, RealizationSelector
from ..utils import get_logger

from .base import (
    _apply_norm_slopes,
    _apply_float_dtype,
    _handle_duplicate_names,
    FloatDtypeLike,
    PRO_FILE_REGEX,
    SMET_FILE_REGEX,
)


_LOCATION_LEVEL_COORDS = ("latitude", "longitude", "altitude", "location")
_SLOPE_COORDS = ("inclination", "azimuth")
_LOCATION_SOURCE_ATTR = "_xsnow_location_source"
_LOCATION_SOURCE_FILENAME_STEM = "filename_stem"


def _is_missing(value) -> bool:
    """Return True for scalar missing values, but not for ordinary strings."""
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _values_equal(left, right) -> bool:
    """Compare scalar or array-like values, treating paired NaNs as equal."""
    try:
        return bool(np.array_equal(left, right, equal_nan=True))
    except TypeError:
        try:
            return bool(left == right)
        except Exception:
            return False


def _merge_attrs_no_conflicts(attrs_list: Sequence[Dict], context: str) -> Dict:
    """Merge attrs using xarray-style no-conflicts semantics."""
    merged: Dict = {}
    for attrs in attrs_list:
        for key, value in attrs.items():
            if key not in merged:
                merged[key] = value
                continue
            if not _values_equal(merged[key], value):
                raise ValueError(
                    f"Conflicting attributes while {context}: "
                    f"{key!r}: {merged[key]!r} vs {value!r}"
                )
    return merged


def _strict_merge(datasets: Sequence[xr.Dataset], context: str) -> xr.Dataset:
    """Merge datasets without allowing overlapping non-null conflicts."""
    try:
        return xr.merge(
            datasets,
            compat="no_conflicts",
            join="outer",
            combine_attrs="no_conflicts",
        )
    except (xr.MergeError, ValueError) as exc:
        raise ValueError(f"Conflicting data while {context}: {exc}") from exc


def _strict_concat(
    datasets: Sequence[xr.Dataset],
    *,
    dim,
    context: str,
    **kwargs,
) -> xr.Dataset:
    """Concatenate datasets without allowing conflicting attrs or values."""
    try:
        return xr.concat(
            datasets,
            dim=dim,
            compat="no_conflicts",
            combine_attrs="no_conflicts",
            **kwargs,
        )
    except (xr.MergeError, ValueError) as exc:
        raise ValueError(f"Conflicting data while {context}: {exc}") from exc


def _non_null_scalar_coord_values(ds: xr.Dataset, coord_name: str) -> List:
    """Return non-missing scalar coordinate values for conflict checks."""
    if coord_name not in ds.coords:
        return []

    values = np.asarray(ds[coord_name].values).reshape(-1)
    out = []
    for value in values:
        scalar = value.item() if hasattr(value, "item") else value
        if not _is_missing(scalar):
            out.append(scalar)
    return out


def _first_non_null_scalar_coord_value(ds: xr.Dataset, coord_name: str):
    """Return the first non-missing scalar value of a coordinate, if any."""
    values = _non_null_scalar_coord_values(ds, coord_name)
    return values[0] if values else None


def _slope_signature(ds: xr.Dataset) -> Optional[Tuple]:
    """Represent slope metadata as an inclination/azimuth tuple for matching."""
    signature = tuple(
        _first_non_null_scalar_coord_value(ds, coord_name)
        for coord_name in _SLOPE_COORDS
    )
    if all(value is None for value in signature):
        return None
    return signature


def _slope_signatures_equal(left: Tuple, right: Tuple) -> bool:
    """Return True when two slope metadata signatures identify the same slope."""
    for left_value, right_value in zip(left, right):
        if left_value is None or right_value is None:
            if left_value is not None or right_value is not None:
                return False
        elif not _values_equal(left_value, right_value):
            return False
    return True


def _slope_signature_matches(candidate: Tuple, target: Tuple) -> bool:
    """Match a partial slope signature against a fully known template slope."""
    matched_at_least_one_coord = False
    for candidate_value, target_value in zip(candidate, target):
        if candidate_value is None or target_value is None:
            continue
        matched_at_least_one_coord = True
        if not _values_equal(candidate_value, target_value):
            return False
    return matched_at_least_one_coord


def _slope_template_signatures(template: xr.Dataset) -> List[Tuple]:
    """Extract slope coordinate values and metadata signatures from a template."""
    signatures: List[Tuple] = []
    if "slope" not in template.dims:
        return signatures

    slope_values = template["slope"].values
    for pos in range(template.sizes["slope"]):
        slope_coord = slope_values[pos]
        slope_coord = slope_coord.item() if hasattr(slope_coord, "item") else slope_coord
        signature = _slope_signature(template.isel(slope=pos))
        if signature is not None:
            signatures.append((slope_coord, signature))
    return signatures


def _assign_single_slope_coord(ds: xr.Dataset, slope_coord) -> xr.Dataset:
    """Assign one slope coordinate to a single-slope dataset."""
    if "slope" not in ds.dims or ds.sizes.get("slope") != 1:
        return ds
    return ds.assign_coords(slope=[slope_coord])


def _sortable_slope_value(value):
    """Return a stable sort key for numeric or string slope metadata values."""
    if value is None:
        return (1, 0.0)
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (0, str(value))


def _slope_sort_key(ds: xr.Dataset):
    """Sort datasets by slope metadata so slope order is deterministic."""
    signature = _slope_signature(ds)
    if signature is None:
        return tuple(_sortable_slope_value(None) for _ in _SLOPE_COORDS)
    return tuple(_sortable_slope_value(value) for value in signature)


def _merge_profile_datasets_by_slope_signature(
    datasets: Sequence[xr.Dataset],
    loc_name: str,
) -> List[xr.Dataset]:
    """Strictly merge same-slope profile files before slope concatenation."""
    grouped: List[Tuple[Optional[Tuple], List[xr.Dataset]]] = []

    for ds in datasets:
        signature = _slope_signature(ds)
        matching_group = None
        for group_signature, group_datasets in grouped:
            if (
                signature is None
                and group_signature is None
            ) or (
                signature is not None
                and group_signature is not None
                and _slope_signatures_equal(signature, group_signature)
            ):
                matching_group = group_datasets
                break

        if matching_group is None:
            grouped.append((signature, [ds]))
        else:
            matching_group.append(ds)

    merged: List[xr.Dataset] = []
    for signature, group_datasets in grouped:
        if len(group_datasets) == 1:
            merged.append(group_datasets[0])
        else:
            merged.append(
                _strict_merge(
                    group_datasets,
                    context=(
                        f"merging profile files for location {loc_name!r} "
                        f"and slope metadata {signature!r}"
                    ),
                )
            )

    merged.sort(key=_slope_sort_key)
    return merged


def _broadcast_unknown_slope_to_template(
    ds: xr.Dataset,
    profile_template: xr.Dataset,
) -> xr.Dataset:
    """Broadcast no-slope-metadata time series to every profile slope."""
    if "slope" not in ds.dims or ds.sizes.get("slope") != 1:
        return ds
    if "slope" not in profile_template.dims:
        return ds

    slope_values = profile_template["slope"].values
    base = ds.isel(slope=0, drop=True).drop_vars(_SLOPE_COORDS, errors="ignore")
    return base.expand_dims(slope=slope_values)


def _align_dataset_to_profile_slope_template(
    ds: xr.Dataset,
    profile_template: xr.Dataset,
    loc_name: str,
) -> xr.Dataset:
    """Align a time-series dataset to profile slope indices before strict merge."""
    if "slope" not in ds.dims or ds.sizes.get("slope") != 1:
        return ds

    signature = _slope_signature(ds)
    if signature is None:
        return _broadcast_unknown_slope_to_template(ds, profile_template)

    matches = [
        slope_coord
        for slope_coord, template_signature in _slope_template_signatures(profile_template)
        if _slope_signature_matches(signature, template_signature)
    ]

    if len(matches) == 1:
        return _assign_single_slope_coord(ds, matches[0])
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous slope metadata while aligning time-series data for "
            f"location {loc_name!r}: {signature!r} matches slope coordinates "
            f"{matches!r}."
        )

    raise ValueError(
        f"Conflicting slope metadata while aligning time-series data for "
        f"location {loc_name!r}: {signature!r} does not match any profile slope."
    )


def _align_timeseries_slope_indices(
    datasets: Sequence[xr.Dataset],
    profile_template: Optional[xr.Dataset],
    loc_name: str,
) -> List[xr.Dataset]:
    """Align slope-specific SMET files and broadcast generic SMET files."""
    if not datasets:
        return []

    if profile_template is not None and "slope" in profile_template.dims:
        return [
            _align_dataset_to_profile_slope_template(ds, profile_template, loc_name)
            for ds in datasets
        ]

    known_signatures: List[Tuple[Tuple, int]] = []
    next_slope_index = 0
    aligned = []

    for ds in datasets:
        if "slope" not in ds.dims or ds.sizes.get("slope") != 1:
            aligned.append(ds)
            continue

        signature = _slope_signature(ds)
        if signature is None:
            aligned.append(ds)
            continue

        matching_indices = [
            slope_index
            for known_signature, slope_index in known_signatures
            if _slope_signatures_equal(signature, known_signature)
        ]

        if matching_indices:
            slope_index = matching_indices[0]
        else:
            slope_index = next_slope_index
            known_signatures.append((signature, slope_index))
            next_slope_index += 1

        aligned.append(_assign_single_slope_coord(ds, slope_index))

    return aligned


def _merged_location_coord_for_slope_concat(
    datasets: Sequence[xr.Dataset],
    coord_name: str,
    loc_name: str,
) -> Optional[xr.DataArray]:
    """
    Merge a scalar location-level coordinate before slope concat drops it.

    Missing values are ignored. Multiple distinct non-null values are treated as
    a real conflict for the same station, because otherwise slope files from
    different physical locations could be silently combined.
    """
    coords = [ds[coord_name] for ds in datasets if coord_name in ds.coords]
    if not coords:
        return None

    candidate = None
    for ds in datasets:
        for value in _non_null_scalar_coord_values(ds, coord_name):
            if candidate is None:
                candidate = value
            elif not _values_equal(candidate, value):
                raise ValueError(
                    f"Conflicting location coordinate {coord_name!r} while "
                    f"concatenating profile slopes for location {loc_name!r}: "
                    f"{candidate!r} vs {value!r}"
                )

    first_coord = coords[0]
    if candidate is None:
        merged = first_coord.copy()
    else:
        template = np.asarray(first_coord.values)
        if template.shape == ():
            data = np.asarray(candidate)
        else:
            data = np.full(template.shape, candidate, dtype=np.asarray(candidate).dtype)
        merged = xr.DataArray(data, dims=first_coord.dims, name=coord_name)

    merged.attrs = _merge_attrs_no_conflicts(
        [dict(coord.attrs) for coord in coords],
        (
            f"combining attributes of location coordinate {coord_name!r} "
            f"for location {loc_name!r}"
        ),
    )
    return merged


def _merged_location_coords_for_slope_concat(
    datasets: Sequence[xr.Dataset],
    loc_name: str,
) -> Dict[str, xr.DataArray]:
    """Collect validated location-level coordinates for slope concatenation."""
    location_coords: Dict[str, xr.DataArray] = {}
    for coord_name in _LOCATION_LEVEL_COORDS:
        coord = _merged_location_coord_for_slope_concat(datasets, coord_name, loc_name)
        if coord is not None:
            location_coords[coord_name] = coord
    return location_coords


def _resolve_filepaths(
    source: Union[str, Path, Sequence[Union[str, Path]]],
    recursive: bool = False, 
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[Path], List[Path]]:
    """
    Resolves the source argument into separate lists of .pro and .smet files.

    Args:
        source: The data source, which can be a directory path, a single
            file path, or a list of paths.
        recursive: If True, search subdirectories recursively.
        logger: Optional logger instance.

    Returns:
        A tuple containing two lists: one for .pro file paths and one for
        .smet file paths.
        
    Raises:
        TypeError: If the source is of an unsupported type.
    """
    logger = logger or get_logger(__name__)
    paths_to_process: List[Path] = []
    
    if isinstance(source, (str, Path)):
        paths_to_process.append(Path(source))
    elif isinstance(source, (list, tuple)):
        paths_to_process = [Path(p) for p in source]
    else:
        raise TypeError(f"Unsupported source type: {type(source)}")

    resolved_files: List[Path] = []
    for p in paths_to_process:
        if not p.exists():
            logger.warning(f"Source path does not exist and will be skipped: {p}")
            continue
        
        if p.is_dir():
            pattern = '**/*.pro*' if recursive else '*.pro*'
            resolved_files.extend(f for f in p.glob(pattern) if re.search(PRO_FILE_REGEX, str(f)))
            pattern = '**/*.smet*' if recursive else '*.smet*'
            resolved_files.extend(f for f in p.glob(pattern) if re.search(SMET_FILE_REGEX, str(f)))
        else:
            resolved_files.append(p)

    unique_files = sorted(list(dict.fromkeys(resolved_files)))

    pro_files = [f for f in unique_files if re.search(PRO_FILE_REGEX, str(f))]
    smet_files = [f for f in unique_files if re.search(SMET_FILE_REGEX, str(f))]
    
    return pro_files, smet_files


def _fold_aspect_locations(
    grouped_by_location: Dict[str, List[xr.Dataset]],
    logger: Optional[logging.Logger] = None,
) -> Dict[str, List[xr.Dataset]]:
    """
    Fold SNOWPACK per-aspect station names into their base location.

    SNOWPACK writes one output file per simulated slope: the flat field as
    ``<station>`` and the virtual slopes as ``<station><n>`` (n = 1..9, the
    NUMBER_SLOPES index). This regroups those datasets under the base station
    name so downstream slope-signature merging concatenates them along the
    slope dimension instead of treating each aspect as its own location.

    Folding is conservative: a location is only folded when its name ends in a
    single nonzero digit, the base name is present in the same batch, the
    suffixed profile location was inferred from the filename rather than an
    explicit ``StationName``, the base profile is a compatible flat field, and
    every dataset of the candidate carries a finite nonzero inclination. Snow-
    and wind stations (also ending in 2 and 1) should be on the flat.

    Args:
        grouped_by_location: Datasets grouped by their location string.
        logger: Logger instance.

    Returns:
        The regrouped mapping; iteration order keeps base locations before
        their folded slopes so slope order stays deterministic.
    """
    logger = logger or get_logger(__name__)

    def _all_from_filename_fallback(dsets: List[xr.Dataset]) -> bool:
        return all(
            ds.attrs.get(_LOCATION_SOURCE_ATTR) == _LOCATION_SOURCE_FILENAME_STEM
            for ds in dsets
        )

    def _all_inclined(dsets: List[xr.Dataset]) -> bool:
        for ds in dsets:
            if "inclination" not in ds.coords:
                return False
            inclination = np.asarray(ds["inclination"].values, dtype=float).reshape(-1)
            if inclination.size == 0:
                return False
            if not (np.isfinite(inclination).all() and (inclination > 0).all()):
                return False
        return True

    def _is_flat_filename_fallback_profile(ds: xr.Dataset) -> bool:
        if "layer" not in ds.dims:
            return False
        if ds.attrs.get(_LOCATION_SOURCE_ATTR) != _LOCATION_SOURCE_FILENAME_STEM:
            return False
        if "inclination" not in ds.coords:
            return False
        inclination = np.asarray(ds["inclination"].values, dtype=float).reshape(-1)
        return bool(
            inclination.size
            and np.isfinite(inclination).all()
            and np.allclose(inclination, 0.0)
        )

    def _has_compatible_flat_base(
        candidate_dsets: List[xr.Dataset],
        base_dsets: List[xr.Dataset],
    ) -> bool:
        flat_base_dsets = [
            ds for ds in base_dsets if _is_flat_filename_fallback_profile(ds)
        ]
        if not flat_base_dsets:
            return False

        for coord_name in ("latitude", "longitude", "altitude"):
            base_values = [
                value
                for ds in flat_base_dsets
                for value in _non_null_scalar_coord_values(ds, coord_name)
            ]
            candidate_values = [
                value
                for ds in candidate_dsets
                for value in _non_null_scalar_coord_values(ds, coord_name)
            ]
            if not base_values or not candidate_values:
                continue
            if any(
                not _values_equal(base_value, candidate_value)
                for base_value in base_values
                for candidate_value in candidate_values
            ):
                return False

        return True

    folded: Dict[str, List[xr.Dataset]] = {}
    for loc in sorted(grouped_by_location):  # base sorts before <base><n>
        dsets = grouped_by_location[loc]
        base = loc[:-1]
        is_aspect_slope = (
            len(loc) >= 2
            and loc[-1].isdigit()
            and loc[-1] != "0"
            and base in grouped_by_location
            and _all_from_filename_fallback(dsets)
            and _all_inclined(dsets)
            and _has_compatible_flat_base(dsets, grouped_by_location[base])
        )
        if not is_aspect_slope:
            folded.setdefault(loc, []).extend(dsets)
            continue
        logger.debug(
            "Folding per-aspect location %r into %r as an additional slope", loc, base)
        for ds in dsets:
            renamed = ds.copy()
            renamed["location"] = [base]
            folded.setdefault(base, []).append(renamed)
    return folded


def _merge_and_finalize(
    all_datasets: List[xr.Dataset],
    time: Optional[TimeSelector],
    location: Optional[LocationSelector],
    slope: Optional[SlopeSelector],
    realization: Optional[RealizationSelector],
    logger: logging.Logger,
    norm_slopes: Union[bool, Literal['auto']] = 'auto',
    float_dtype: FloatDtypeLike = "float32",
) -> Optional[xsnowDataset]:
    """
    Merges, re-indexes, and applies final filters to the datasets.

    This function combines a list of xarray.Dataset objects into a single,
    unified xsnowDataset. Merge and concat steps use strict xarray
    ``compat="no_conflicts"`` and ``combine_attrs="no_conflicts"`` semantics,
    while allowing missing values to be filled from another source. Locations
    remain station-name coordinates, duplicate station names are made unique,
    final selectors are applied, and the final z-depth coordinate is computed.

    Args:
        all_datasets: A list of xarray.Dataset objects to be merged.
        time: A TimeSelector for final time-based filtering.
        location: A LocationSelector for final location-based filtering.
        slope: A SlopeSelector for final slope-based filtering.
        realization: A RealizationSelector for final realization-based filtering.
        logger: Logger instance.
        norm_slopes: Controls slope normalization:
            - 'auto' (default): remap local slopes to their global metadata
              union when doing so does not widen the slope dimension; otherwise
              preserve per-location slope coordinates.
            - True: always build the global slope-metadata union and remap all
              slope-bearing data, accepting a wider, sparse slope dimension.
            - False: do not call `normalize_slope_coords`.
            If recognized slope metadata cannot identify the local slope
            positions, normalization preserves the existing layout. It warns
            for True or an ambiguous multi-slope layout.
        float_dtype: Floating-point storage dtype applied to the merged
            dataset; see `read()`. None preserves source dtypes.

    Returns:
        A unified and filtered xsnowDataset, or None if no datasets were provided.
    """
    logger = get_logger(__name__, logger=logger)
    
    if not all_datasets:
        return None

    # Enforce timezone consistency
    final_timezone = None
    for ds in all_datasets:
        if 'time' in ds and 'timezone' in ds.time.attrs:
            current_tz = ds.time.attrs['timezone']
            if final_timezone is None:
                final_timezone = current_tz
            elif final_timezone != current_tz:
                raise ValueError(
                    f"Conflicting timezones: '{final_timezone}' vs '{current_tz}'.")

    # Group datasets by location (location is already a string)
    grouped_by_location: Dict[str, List[xr.Dataset]] = {}
    for ds in all_datasets:
        loc = str(ds.location.values[0])  # Get string location name
        grouped_by_location.setdefault(loc, []).append(ds)

    # Fold SNOWPACK per-aspect station names (<station><n>) into their base
    # location, so their profiles concatenate along the slope dimension below.
    grouped_by_location = _fold_aspect_locations(grouped_by_location, logger)
    for dsets in grouped_by_location.values():
        for ds in dsets:
            ds.attrs.pop(_LOCATION_SOURCE_ATTR, None)

    # HANDLE DUPLICATE STATION NAMES
    station_names = list(grouped_by_location.keys())
    unique_names = _handle_duplicate_names(station_names, logger=logger)
    
    # Rename locations if duplicates were found
    if unique_names != station_names:
        new_grouped: Dict[str, List[xr.Dataset]] = {}
        for old_name, new_name in zip(station_names, unique_names):
            new_grouped[new_name] = grouped_by_location[old_name]
            # Update location coordinate in datasets
            for ds in new_grouped[new_name]:
                ds['location'] = [new_name]
        grouped_by_location = new_grouped
    
    merged_datasets: List[xr.Dataset] = []
    for loc_name, dsets in grouped_by_location.items():    
        profile_dsets = [ds for ds in dsets if 'layer' in ds.dims]
        timeseries_dsets = [ds for ds in dsets if 'layer' not in ds.dims]

        combined_profiles_ds = None
        if profile_dsets:
            # Extract .data if any caller provided wrapped xsnowDataset objects.
            profile_xr_datasets = [
                ds.data if hasattr(ds, 'data') else ds
                for ds in profile_dsets
            ]

            # Multiple profile files for the same physical slope are merged
            # along their coordinates first. Only distinct slope metadata is
            # represented as separate slope entries.
            profile_dsets_by_slope = _merge_profile_datasets_by_slope_signature(
                profile_xr_datasets,
                loc_name,
            )

            if len(profile_dsets_by_slope) > 1:
                # Location-level coordinates must not vary between slope files.
                # They are dropped for the slope concat, so validate and merge
                # them explicitly before xarray loses the chance to check them.
                location_coords = _merged_location_coords_for_slope_concat(
                    profile_dsets_by_slope,
                    loc_name,
                )
                
                # Drop location-level coords before concatenating along slope
                profile_dsets_cleaned = [
                    ds.drop_vars(
                        [c for c in _LOCATION_LEVEL_COORDS if c in ds.coords],
                        errors='ignore'
                    )
                    for ds in profile_dsets_by_slope
                ]
                combined_profiles_ds = _strict_concat(
                    profile_dsets_cleaned,
                    dim='slope',
                    context=f"concatenating profile slopes for location {loc_name!r}",
                    join='outer',
                    coords='all',
                )
                combined_profiles_ds = combined_profiles_ds.assign_coords(
                    slope=np.arange(combined_profiles_ds.sizes['slope'])
                )
                
                # Restore location-level coordinates (without slope dimension)
                combined_profiles_ds = combined_profiles_ds.assign_coords(location_coords)
            else:
                combined_profiles_ds = profile_dsets_by_slope[0]

        # Extract .data from all datasets before merge
        all_to_merge: List[xr.Dataset] = []
        if combined_profiles_ds is not None:
            all_to_merge.append(combined_profiles_ds)
        # Also extract .data from timeseries datasets and align slope-specific
        # SMET files to the matching profile slope before strict merging.
        timeseries_xr_datasets = [
            ds.data if hasattr(ds, 'data') else ds
            for ds in timeseries_dsets
        ]
        all_to_merge.extend(
            _align_timeseries_slope_indices(
                timeseries_xr_datasets,
                combined_profiles_ds,
                loc_name,
            )
        )
        
        if not all_to_merge: 
            continue

        if len(all_to_merge) > 1:
            merged_loc_ds = _strict_merge(
                all_to_merge,
                context=f"merging data for location {loc_name!r}",
            )
            merged_datasets.append(merged_loc_ds)
        else:
            merged_datasets.append(all_to_merge[0])

    if not merged_datasets: 
        return None

    # Sort by station name for consistent ordering
    merged_datasets.sort(key=lambda ds: str(ds.location.values[0]))
    
    # Extract .data from merged datasets (xr.concat needs plain xarray Datasets)
    merged_xr_datasets = [ds.data if hasattr(ds, 'data') else ds for ds in merged_datasets]
    final_xr_dataset = _strict_concat(
        merged_xr_datasets,
        context="concatenating locations",
        dim='location', 
        join='outer', 
        coords='all',
    )
    final_xr_dataset.attrs.pop(_LOCATION_SOURCE_ATTR, None)
    
    # Preserve CF-compliant metadata
    if not final_xr_dataset.attrs:
        final_xr_dataset.attrs = {
            'Conventions': 'CF-1.8',
            'crs': 'EPSG:4326'
        }

    # Add variable metadata
    for var_name_hashable in list(final_xr_dataset.variables):
        var_name = str(var_name_hashable)
        if reg_entry := registry.get(var_name):
            final_xr_dataset[var_name].attrs.setdefault(
                'standard_name',
                reg_entry.get('standard_name', 'unknown'),
            )
            final_xr_dataset[var_name].attrs.setdefault(
                'units',
                reg_entry.get('xsnow_units', reg_entry.get('si_units', 'unknown')),
            )
            final_xr_dataset[var_name].attrs.setdefault(
                'long_name',
                reg_entry.get('standard_name', var_name).replace('_', ' '),
            )
        else:
            # Set defaults for variables not in registry if not already set
            if var_name in final_xr_dataset.data_vars:
                if 'units' not in final_xr_dataset[var_name].attrs:
                    final_xr_dataset[var_name].attrs['units'] = '1'
                if 'standard_name' not in final_xr_dataset[var_name].attrs:
                    final_xr_dataset[var_name].attrs['standard_name'] = var_name
                if 'long_name' not in final_xr_dataset[var_name].attrs:
                    final_xr_dataset[var_name].attrs['long_name'] = var_name.replace('_', ' ')

    # Normalize slope coordinates
    try:
        final_xr_dataset = _apply_norm_slopes(
            final_xr_dataset,
            norm_slopes=norm_slopes,
            logger=logger,
        )
    except ValueError as e:
        logger.error(
            f"Failed to normalize slope coordinates: {e}\n"
            f"This indicates a data structure issue that needs investigation."
        )
        raise
    
    # Apply final filters using selectors
    if time:
        final_xr_dataset = time.apply(final_xr_dataset)
    if location:
        final_xr_dataset = location.apply(final_xr_dataset)
    if slope:
        final_xr_dataset = slope.apply(final_xr_dataset)
    if realization:
        final_xr_dataset = realization.apply(final_xr_dataset)
    
    from ..utils import reorder_dims
    final_xr_dataset = reorder_dims(final_xr_dataset)
    final_xr_dataset = _apply_float_dtype(final_xr_dataset, float_dtype)
    final_dataset = xsnowDataset(final_xr_dataset)
    
    # Compute z-depth if not present
    if 'z' not in final_dataset and 'height' in final_dataset:
        final_dataset = final_dataset.compute_z_depth()
    
    # Compute HS if not present
    if 'HS' not in final_dataset and 'height' in final_dataset:
        final_dataset = final_dataset.compute_hs()
    
    # Update profile status
    final_dataset = final_dataset.update_profile_status()
    
    return final_dataset
