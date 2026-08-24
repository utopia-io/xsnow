# io/__init__.py
"""
xsnow I/O subpackage.

This subpackage handles all data parsing and I/O for the xsnow package,
enabling scalable and memory-efficient processing of large datasets using Dask.

Supported formats:
- SNOWPACK .pro files (snow profile data)
- SMET files (meteorological time series)
- NetCDF files
- Zarr archives
"""
from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import xarray as xr

from ..core import xsnowDataset
from ..selectors import (
    TimeSelector, LocationSelector, SlopeSelector, RealizationSelector,
    TimeLike, LocationLike, SlopeLike, RealizationLike
)
from ..utils import get_logger
from ..units import UnitValidationError

from .base import (
    CODE_TO_NAME_MAP,
    HEADER_MAP,
    _coerce_time,
    _coerce_location,
    _coerce_slope,
    _coerce_realization,
    _round_coord,
    _sanitize_station_name,
    _handle_duplicate_names,
    _validate_norm_slopes,
    _apply_float_dtype,
    _validate_float_dtype,
    FloatDtypeLike,
)
from .memory import (
    MemoryLimitExceeded,
    MemoryLimitLike,
    MemorySizeUnknown,
    check_memory_budget,
    resolve_memory_limit,
)
from .convert import ArchiveAppendError, build_archive
from .merge import _resolve_filepaths, _merge_and_finalize
from .pro import _read_pro_files_eager, _read_pro_files_lazy, to_pro
from .smet import _read_smet_files, read_smet, to_smet
from .netcdf import _read_xsnow_netcdf, to_netcdf
from .zarr import _read_zarr, to_zarr


__all__ = [
    'read',
    'read_smet',
    'to_pro',
    'to_smet',
    'to_netcdf',
    'to_zarr',
    'build_archive',
    'check_memory_budget',
    'MemoryLimitExceeded',
    'MemorySizeUnknown',
    'ArchiveAppendError',
]


def read(
    source: Union[str, List[str], Path, List[Path]],
    *,
    recursive: bool = False,
    time: Optional[TimeLike] = None,
    location: Optional[LocationLike] = None,
    slope: Optional[SlopeLike] = None,
    realization: Optional[RealizationLike] = None,
    lazy: Optional[bool] = None,
    lazy_threshold: int = 30,
    parallel_lazy_creation: bool = True,
    n_cpus_use: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    chunks: Optional[Dict[str, int]] = None,
    memory_limit: MemoryLimitLike = "auto",
    norm_slopes: Union[bool, Literal["auto"]] = "auto",
    float_dtype: FloatDtypeLike = "float32",
    **parser_kwargs: Any
) -> Optional[xsnowDataset]:
    """
    Reads, parses, and combines snowpack data into a unified xsnowDataset.

    This is the primary user-facing function for loading data. It orchestrates
    file discovery, optional parallel processing, and merging of snow profile
    (.pro) and meteorological time series (.smet) files into one dataset.
    NetCDF and Zarr are currently opened only as one direct source at a time:
    a single .nc file or a single .zarr archive path. Direct multi-file
    NetCDF/Zarr expansion and merging is not supported here.

    PRO/SMET sources are merged conservatively. Merge and concat steps use
    xarray's ``compat="no_conflicts"`` and
    ``combine_attrs="no_conflicts"`` semantics: missing values can be filled
    from another source, but conflicting non-null values or conflicting
    attributes raise an exception. Files with identical SNOWPACK station names
    are interpreted as belonging to the same xsnow location.

    SMET files with slope metadata are treated as slope-specific and aligned
    to matching profile slopes. SMET files without slope metadata are treated
    as location-wide time series; when profile data defines multiple slopes
    for that location, these generic SMET values are broadcast to all slopes.

    Args:
        source: The data source. Can be a path to a single file (e.g.,
            '.pro', '.smet', '.nc', '.zarr'), a list of paths (.pro and .smet),
            or a directory (of .pro and .smet). PRO/SMET files can have extensions
            (.gz, .bz2, or .xz), and will be read with the appropriate
            decompressor.
        recursive: If True, search subdirectories recursively for .pro and .smet
            files when source is a directory. Default is False for backward
            compatibility and to avoid accidentally including unwanted files.
        time: A `TimeSelector` object for filtering data by time.
        location: A `LocationSelector` object for filtering by station ID.
        slope: A `SlopeSelector` for filtering by slope angle.
        realization: A `RealizationSelector` for filtering by model realization.
        lazy: Loading policy. None (default) makes a conservative adaptive
            choice for NetCDF and Zarr, while PRO/SMET collections use
            ``lazy_threshold``. True always returns Dask-backed data and is the
            recommended archive mode for large datasets.
            False explicitly requests in-memory data; eager NetCDF and Zarr
            reads remain protected by ``memory_limit``.
        memory_limit: Pre-flight memory budget for eager NetCDF and Zarr reads.
            The selected decoded size is checked before eager data loading, and
            `MemoryLimitExceeded` is raised when it is too large.
            `MemorySizeUnknown` is raised when an eager load cannot be checked
            reliably. `'auto'` (default) uses 70% of detected available
            memory; a size string (`'8GB'`, `'512MiB'`) or byte count sets an
            absolute cap, and None disables the guard. Selectors are applied
            before this check. For containers and scheduled HPC jobs, prefer
            an explicit cap based on assigned memory, because automatic
            detection may report the host or whole node rather than the cgroup
            or scheduler limit.
        lazy_threshold: Number of files above which lazy loading is enabled
            for PRO/SMET collections when ``lazy=None``. Default: 30 files.
        parallel_lazy_creation: If True (default), parallelize the creation of lazy
            datasets using ThreadPoolExecutor. This speeds up the metadata parsing
            phase by 5-15x when working with hundreds or thousands of files.
        n_cpus_use: The number of CPU cores to use. If not provided, defaults
            to min(32, total_cores - 1), with a minimum of 1.
        logger: An optional, pre-configured logger instance.
        chunks: Chunk sizes for Dask arrays. For NetCDF and Zarr, the mapping
            is forwarded to xarray when ``lazy`` is not False, including during
            automatic size detection. None uses xarray/Dask automatic chunking,
            while ``{}`` requests backend-preferred/native chunks. Ignored when
            ``lazy=False``. PRO/SMET sources default to
            ``{'time': 100, 'layer': -1}``.
        norm_slopes: Controls slope-coordinate normalization for every source
            format. `'auto'` (default) normalizes slope coords (inclination and
            azimuth) to dimensions `(slope,)` when they are consistent across
            locations or a union of signatures does not widen the existing slope
            dimension; otherwise it preserves per-location dimensions `(location, slope,)`,
            which is identical to False. True builds the global union of slope
            coordinate signatures and remaps all slope-bearing data onto it. The
            resulting coords have dimensions `(slope,)`. If usable recognized
            slope metadata is unavailable, normalization preserves the existing
            layout. It warns for `True` or an ambiguous multi-slope layout.
        float_dtype: Floating-point storage dtype for every source format.
            `'float32'` (default) casts all float variables and coordinates to
            single precision at read-in; `latitude`/`longitude`/`altitude` and
            datetime variables keep their source precision. `'float64'` casts
            to double precision, None preserves source dtypes. Dask-backed
            reads stay lazy; the memory guard and automatic archive mode
            account for the cast size on Dask-backed archive reads.
        **parser_kwargs: Additional keyword arguments passed to the underlying
            parser functions:
            - `max_layers`: Max layer dimension. `'auto'` (default) uses the
              smart detector to scan the largest .pro files and add a buffer
              when only a subset is scanned. Set an int to force a fixed layer
              dimension. For compressed .pro archives, compressed byte size is
              a weaker proxy for layer count than uncompressed text size; use a
              manual integer value for heterogeneous compressed archives when a
              conservative layer dimension is important.
            - `force_smart_detection`: If True, skip the homogeneity validation
              before smart max-layer detection. This is faster for known-safe
              archives, but can be wrong for heterogeneous file sets.
            - `validation_threshold_days`: Maximum file modification time
              spread, in days, for files to be considered homogeneous by smart
              detection. Default: 7.0 days.
            - `trim_padding`: If True, automatically trim trailing all-NaN
              padding after loading. For lazy datasets this computes a validity
              reduction over selected variables to detect padding, while the
              trimmed payload remains Dask-backed. Computing it later may read
              the payload again.
            - `remove_soil`: If True (default), drops soil layers during parsing.
            - `add_surface_sh_as_layer`: If True (default), inserts a surface hoar
              layer explicitly as layer when indicated by the PRO file.
            - `unit_validation`: Unit handling/validation mode. `'strict'`
              (default) parses every PRO file's declared units and raises when
              a declared unit cannot be converted to xsnow units. `'light'`
              parses one representative PRO file per parent directory and
              applies those units to all PRO files in that directory; this is
              faster for homogeneous PRO file collections. SMET files parse
              each file's unit metadata in both modes, warn for missing unit
              metadata, and raise for incompatible units. NetCDF and Zarr
              datasets preserve stored attrs and validate present
              `attrs["units"]` against xsnow units: `'strict'` raises on
              mismatches, while `'light'` warns and leaves attrs unchanged.
            - `chunk_cache_size`: NetCDF sources only. Size in bytes of each
              netCDF4/HDF5 per-variable chunk cache while the file is open.
              By default, xsnow leaves the process's current netCDF4 setting
              untouched. On datasets with many variables, these caches can
              collectively consume gigabytes. Set a smaller value (e.g. 1 MiB:
              `1024**2`, or `1_048_576` bytes) to limit aggregate chunk-cache
              memory. Smaller caches may reduce I/O performance.
    Returns:
        A unified `xsnowDataset` object containing the data from all sources,
        or `None` if no valid data files are found.

    Raises:
        MemoryLimitExceeded: If an eager NetCDF or Zarr load is known to
            exceed ``memory_limit``.
        MemorySizeUnknown: If an eager NetCDF or Zarr load's decoded size
            cannot be estimated reliably.

    Notes:
        Automatic archive mode estimates decoded size after applying selectors;
        it does not use compressed size on disk. Estimates exclude Python and
        index overhead and temporary allocations. Archives containing object or
        variable-length string data, or whose size cannot be estimated, remain
        lazy.

        Lazy NetCDF and Zarr reads keep their backing store available for
        deferred Dask reads. Call ``dataset.close()`` after the dataset and any
        derived lazy objects have been computed or are no longer needed. Eager
        archive reads are loaded and closed before ``read()`` returns.

    Examples:
        >>> # Let xsnow choose a conservative loading mode
        >>> ds = read("archive.nc")
        
        >>> # Force lazy loading for large datasets
        >>> ds = read("large_archive/", lazy=True, n_cpus_use=8)
        
        >>> # Force eager loading with trimmed padding
        >>> ds = read("small_dataset/", lazy=False, trim_padding=True)

        >>> # Use a fixed layer dimension when the smart detector is unsuitable
        >>> ds = read("mixed_archive/", max_layers=120)

        >>> # Cap an explicit eager read; raises rather than risking an OOM kill
        >>> ds = read("season.zarr", lazy=False, memory_limit="8GB")

        >>> # Fall back to lazy instead of failing when the archive is too big
        >>> from xsnow import MemoryLimitExceeded
        >>> try:
        ...     ds = read("season.zarr", lazy=False)
        ... except MemoryLimitExceeded:
        ...     ds = read("season.zarr", lazy=True)

        >>> # Narrowing the read shrinks what the guard measures
        >>> ds = read("season.zarr", lazy=False, time=slice("2024-01", "2024-02"))

        >>> # Opt out when you know the box can take it
        >>> ds = read("season.zarr", lazy=False, memory_limit=None)

    See Also
    --------
    to_netcdf : Save datasets to NetCDF with proper formatting
    to_zarr : Save datasets to Zarr archives
    """
    # --- Step 1: Setup ---
    logger = get_logger(__name__, logger=logger)
    _validate_norm_slopes(norm_slopes)
    _validate_float_dtype(float_dtype)

    if n_cpus_use is None:
        cpu_count = os.cpu_count()
        n_cpus_use = min(32, cpu_count - 1) if cpu_count and cpu_count > 1 else 1

    # Resolved once, up front, so a malformed budget fails fast on every source
    # type rather than being swallowed by a per-format read handler.
    budget_bytes = resolve_memory_limit(memory_limit, logger=logger)

    time_sel = _coerce_time(time)
    location_sel = _coerce_location(location)
    slope_sel = _coerce_slope(slope)
    realization_sel = _coerce_realization(realization)
    
    # --- Step 2: Handle special file types directly ---
    
    # NetCDF files
    if isinstance(source, (str, Path)) and Path(source).suffix == '.nc':
        try:
            netcdf_kwargs = {}
            if 'chunk_cache_size' in parser_kwargs:
                netcdf_kwargs['chunk_cache_size'] = parser_kwargs['chunk_cache_size']
            return _read_xsnow_netcdf(
                source,
                unit_validation=parser_kwargs.get('unit_validation', 'strict'),
                logger=logger,
                lazy=lazy,
                chunks=chunks,
                time_sel=time_sel,
                location_sel=location_sel,
                slope_sel=slope_sel,
                realization_sel=realization_sel,
                memory_limit=budget_bytes,
                norm_slopes=norm_slopes,
                float_dtype=float_dtype,
                **netcdf_kwargs,
            )
        except (UnitValidationError, MemoryLimitExceeded, MemorySizeUnknown):
            # Deliberate refusals, not read failures: a swallowed memory guard
            # would hand back None and leave the caller none the wiser.
            raise
        except Exception as e:
            logger.warning(f"Failed to read NetCDF file {source}: {e}")
            return None
    
    # Zarr archives
    if isinstance(source, (str, Path)) and (
        Path(source).suffix == '.zarr' or 
        (Path(source).is_dir() and (Path(source) / 'zarr.json').exists())
    ):
        try:
            return _read_zarr(
                source,
                time_sel,
                location_sel,
                slope_sel,
                realization_sel,
                logger,
                unit_validation=parser_kwargs.get('unit_validation', 'strict'),
                lazy=lazy,
                chunks=chunks,
                memory_limit=budget_bytes,
                norm_slopes=norm_slopes,
                float_dtype=float_dtype,
            )
        except (UnitValidationError, MemoryLimitExceeded, MemorySizeUnknown):
            # See the NetCDF branch: refusals must not degrade to None.
            raise
        except Exception as e:
            logger.warning(f"Failed to read Zarr archive {source}: {e}")
            return None
    
    # --- Step 3: Resolve file paths for .pro/.smet files ---
    try:
        pro_files, smet_files = _resolve_filepaths(source, recursive, logger)
    except (FileNotFoundError, TypeError) as e:
        logger.error(e)
        return None

    if not pro_files and not smet_files:
        logger.warning("No data files found in the specified source.")
        return None

    # --- Step 4: Auto-detect lazy mode if not specified ---
    if lazy is None:
        total_files = len(pro_files) + len(smet_files)
        lazy = total_files > lazy_threshold
        logger.debug(f"Auto-detected lazy={lazy} for {total_files} files (threshold={lazy_threshold})")

    # --- Step 5: Read and process data files ---
    datetime_start, datetime_end = (
        (time_sel.sel.start, time_sel.sel.stop) 
        if time_sel and isinstance(time_sel.sel, slice) 
        else (None, None)
    )
    
    if lazy:
        chunks = chunks or {'time': 100, 'layer': -1}
        parser_kwargs['use_dask'] = lazy
        parser_kwargs['chunks'] = chunks

        logger.debug("Using lazy loading with Dask")
        pro_datasets = _read_pro_files_lazy(
            pro_files, datetime_start, datetime_end, 
            n_cpus_use, parallel_lazy_creation, logger, parser_kwargs
        )
    else:
        pro_datasets = _read_pro_files_eager(
            pro_files, datetime_start, datetime_end, 
            n_cpus_use, logger, parser_kwargs
        )
    
    smet_datasets = _read_smet_files(
        smet_files,
        datetime_start,
        datetime_end,
        logger,
        unit_validation=parser_kwargs.get('unit_validation', 'strict'),
    )

    # --- Step 6: Merge and finalize datasets ---
    all_datasets = pro_datasets + smet_datasets
    final_dataset = _merge_and_finalize(
        all_datasets,
        time_sel,
        location_sel,
        slope_sel,
        realization_sel,
        logger=logger,
        norm_slopes=norm_slopes,
        float_dtype=float_dtype,
    )

    # Keep the public lazy contract consistent for every text-source result.
    # SMET is parsed eagerly because it is a row-oriented text format, but a
    # lazy request still returns Dask-backed arrays after merging.
    if final_dataset is not None:
        if lazy:
            requested_chunks = chunks or {}
            applicable_chunks = {
                dim: size
                for dim, size in requested_chunks.items()
                if dim in final_dataset.dims
            }
            final_dataset = final_dataset.chunk(applicable_chunks or "auto")
        else:
            final_dataset.load()
    
    # --- Step 7: Trim padding if requested ---
    if final_dataset is not None and parser_kwargs.get('trim_padding', False):
        if lazy:
            logger.warning(
                """
                WARNING: trim_padding=True scans all selected variables that use the
                layer dimension to find trailing all-NaN slices. The trimmed result
                remains Dask-backed, so computing it later may read the data again.

                For large lazy datasets, use this workflow instead:
                    # Load lazily
                    ds = read('huge/', lazy=True)
                    
                    # Subset first
                    ds_sub = ds.sel(location=['A', 'B'], time='2024')
                    
                    # Compute only the subset
                    ds_sub = ds_sub.compute()
                    
                    # Trim the small subset (fast!)
                    ds_sub = ds_sub.trim_padding()
                """
            )
        logger.debug("Trimming NaN padding from layer dimension...")
        final_dataset = final_dataset.trim_padding()
    
    return final_dataset
