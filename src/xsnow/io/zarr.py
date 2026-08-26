# io/zarr.py
"""
Zarr I/O operations for xsnow.

Zarr archives provide a directory-based storage format that offers:
- Better stability (pure Python, no HDF5 crashes)
- Lower memory usage (streaming reads)
- Cloud-friendly (works with S3, GCS)
- Same xarray interface as NetCDF

Two archive layouts are read here. The xsnow layout written by
:func:`to_zarr` is opened directly. The SNOWPACK ZarrIO layout -- Zarr v2,
profile arrays on ``(location, time, layer)``, scalar time series on
``(location, time)``, station labels in ``station_id``/``station_name``,
SNOWPACK storage units and one location slot per virtual slope -- is
turned into the xsnow layout by the ``snowpack`` flavour below, without
materializing any layered variable.
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
import warnings
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Union,
)

import dask
import numpy as np
import xarray as xr

from ..core import xsnowDataset
from ..plugin import register_xsnowDataset_method
from ..registry import registry
from ..selectors import TimeSelector, LocationSelector, SlopeSelector, RealizationSelector
from ..units import (
    UnitConversionError,
    UnitValidationError,
    normalize_unit,
    validate_xsnow_unit_attrs,
)
from ..utils import get_logger

from .base import (
    _apply_norm_slopes,
    _apply_float_dtype,
    _promote_slope_metadata_coords,
    _resolve_archive_lazy,
    FloatDtypeLike,
)
from .memory import (
    MemoryLimitExceeded,
    MemoryLimitLike,
    MemorySizeUnknown,
    check_memory_budget,
    resolve_memory_limit,
)

__all__ = [
    'ZarrIOLayoutError',
    'adapt_snowpack_zarrio',
    'is_snowpack_zarrio',
    'open_snowpack_zarrio',
    'read_snowpack_archives',
    'to_zarr',
]

ZarrFlavour = Literal['auto', 'xsnow', 'snowpack']

# Define the allowed modes matching xarray/zarr requirements
ZarrModes = Literal['w', 'w-', 'a', 'a-', 'r+', 'r']


class ZarrIOLayoutError(RuntimeError):
    """The archive cannot be adapted to the xsnow layout."""


# --------------------------------------------------------------------------- #
#                       CORE: THE XSNOW ARCHIVE LAYOUT                         #
# --------------------------------------------------------------------------- #


def _read_zarr(
    path: Union[str, Path],
    time_sel: Optional[TimeSelector] = None,
    location_sel: Optional[LocationSelector] = None,
    slope_sel: Optional[SlopeSelector] = None,
    realization_sel: Optional[RealizationSelector] = None,
    logger: Optional[logging.Logger] = None,
    unit_validation: str = "strict",
    lazy: Optional[bool] = None,
    chunks: Optional[Dict[str, int]] = None,
    memory_limit: MemoryLimitLike = "auto",
    norm_slopes: Union[bool, Literal["auto"]] = "auto",
    float_dtype: FloatDtypeLike = "float32",
    flavour: ZarrFlavour = 'auto',
    **flavour_kwargs,
) -> Optional[xsnowDataset]:
    """
    Internal function to read a Zarr archive.
    
    Called by read() when it detects a .zarr file or directory.
    Similar to _read_xsnow_netcdf() for NetCDF files.
    
    Args:
        path: Path to the .zarr directory
        time_sel: TimeSelector for filtering by time (already coerced)
        location_sel: LocationSelector for filtering by location (already coerced)
        slope_sel: SlopeSelector for filtering by slope (already coerced)
        realization_sel: RealizationSelector for filtering by realization (already coerced)
        logger: Logger instance
        unit_validation: Stored-unit validation mode. `'strict'` raises when
            present unit attrs differ from xsnow units; `'light'` warns.
        lazy: If True, return Dask-backed data backed by the open archive; the
            caller must close the returned dataset. If False, load and close
            the archive before returning. If None (default), load selected data
            below the adaptive threshold derived from ``memory_limit`` eagerly.
        chunks: Dask chunk sizes forwarded to ``xarray.open_zarr`` when
            ``lazy`` is not False. None uses xarray/Dask automatic chunking;
            ``{}`` requests the Zarr archive's native chunks.
        memory_limit: Pre-flight budget for an eager load. `'auto'` (default)
            uses a fraction of detected available memory; a size string or byte
            count sets an absolute cap; None disables the guard. In automatic
            mode, half this budget, capped at 4 GiB, is the eager threshold.
        norm_slopes: Slope-coordinate normalization policy. True builds the
            global union of recognized slope-metadata signatures. `'auto'`
            remaps to that union only when it does not widen the slope
            dimension. False preserves per-location dimensions.
        float_dtype: Floating-point storage dtype; see `read()`. None
            preserves stored dtypes. On Dask-backed opens the cast is lazy
            and applied before the memory guard; with ``lazy=False`` it is
            applied after loading.
        flavour: ``'xsnow'`` for archives written by :func:`to_zarr`,
            ``'snowpack'`` for SNOWPACK ZarrIO archives (adapted lazily to
            the xsnow layout), ``'auto'`` (default) detects the layout.
        **flavour_kwargs: Options of
            :func:`adapt_snowpack_zarrio`
            (``realization_label``, ``slope_metadata``, ``geo``, ``station_index``, ``layers``,
            ``mask_stale_layers``, ``time_chunk``).

    Returns:
        xsnowDataset with the loaded data, or None if loading fails

    Raises:
        MemoryLimitExceeded: If an eager load would exceed ``memory_limit``.
        MemorySizeUnknown: If an eager load's decoded size cannot be estimated
            reliably.
    """
    logger = get_logger(__name__, logger=logger)
    opened: Optional[xr.Dataset] = None

    # Resolved up front so a malformed budget raises instead of being caught
    # by this function's "return None on read failure" contract below.
    budget_bytes = resolve_memory_limit(memory_limit, logger=logger)

    try:
        path = Path(path)
        
        # Verify it's a Zarr directory
        if not path.exists():
            raise FileNotFoundError(f"Zarr archive not found: {path}")
        
        if not path.is_dir():
            raise ValueError(f"Path is not a directory: {path}")
            
        # Check for Zarr metadata (v2 or v3)
        has_zarr_v3 = (path / 'zarr.json').exists()
        has_zarr_v2 = any(
            (path / name).exists()
            for name in ('.zarray', '.zattrs', '.zgroup')
        )
        
        if not (has_zarr_v3 or has_zarr_v2):
            raise ValueError(f"Directory does not appear to be a Zarr archive: {path}")
        
        logger.debug(f"Reading Zarr archive: {path}")
        
        # A ZarrIO archive is always opened Dask-backed so the adapter's
        # index and arithmetic operations stay lazy; an eager request is
        # honoured by the guarded load below.
        open_lazily = lazy is not False
        consolidated = (
            None if (path / '.zmetadata').exists() or has_zarr_v3 else False
        )
        if flavour == 'snowpack':
            opened = open_snowpack_zarrio(path, chunks=chunks)
        else:
            open_chunks = (
                (chunks if chunks is not None else "auto")
                if open_lazily else None
            )
            opened = xr.open_zarr(
                path, chunks=open_chunks, consolidated=consolidated
            )
            if flavour == 'auto' and is_snowpack_zarrio(opened):
                opened.close()
                flavour = 'snowpack'
                opened = open_snowpack_zarrio(path, chunks=chunks)
        ds_raw = opened
        if flavour == 'snowpack':
            ds_raw = adapt_snowpack_zarrio(
                ds_raw, path=path, logger=logger, **flavour_kwargs
            )
            open_lazily = True
        elif flavour_kwargs:
            raise ZarrIOLayoutError(
                "ZarrIO reader options given for an xsnow-layout archive: "
                f"{sorted(flavour_kwargs)}"
            )
        ds_raw = _promote_slope_metadata_coords(ds_raw)
        ds_raw = _apply_norm_slopes(
            ds_raw,
            norm_slopes=norm_slopes,
            logger=logger,
        )
        validate_xsnow_unit_attrs(
            ds_raw,
            unit_validation=unit_validation,
            logger=logger,
            source=f"Zarr archive {path}",
        )
        
        logger.debug(
            f"  Loaded: {len(ds_raw.location):,} locations, "
            f"{len(ds_raw.time):,} timesteps, "
            f"{len(ds_raw.data_vars)} variables"
        )
        
        # Apply selectors if provided (already coerced by read())
        if time_sel:
            ds_raw = time_sel.apply(ds_raw)
        if location_sel:
            ds_raw = location_sel.apply(ds_raw)
        if slope_sel:
            ds_raw = slope_sel.apply(ds_raw)
        if realization_sel:
            ds_raw = realization_sel.apply(ds_raw)

        # Lazy cast on Dask-backed opens, so the eager-mode resolution and
        # memory guard below measure the post-cast size
        if open_lazily:
            ds_raw = _apply_float_dtype(ds_raw, float_dtype)

        resolved_lazy = _resolve_archive_lazy(
            ds_raw,
            lazy,
            memory_budget_bytes=budget_bytes,
        )
        if lazy is None:
            logger.debug(
                f"Auto-detected archive lazy={resolved_lazy} from decoded size"
            )

        if not resolved_lazy:
            # Check the narrowed selection, not the whole archive, and check it
            # before the allocation an OOM kill would make uncatchable.
            check_memory_budget(
                ds_raw,
                budget_bytes,
                raise_on_exceed=True,
                raise_on_unknown=True,
                source=f"Zarr archive {path}",
                logger=logger,
            )
            ds_raw.load()
            opened.close()

        # Non-Dask opens materialize on .astype(), so cast only after the
        # guarded load
        if not open_lazily:
            ds_raw = _apply_float_dtype(ds_raw, float_dtype)
        elif not resolved_lazy and lazy is False:
            ds_raw = _apply_float_dtype(ds_raw, float_dtype)

        # Wrap in xsnowDataset
        return xsnowDataset(ds_raw)

    except (UnitValidationError, MemoryLimitExceeded, MemorySizeUnknown,
            ZarrIOLayoutError):
        # Deliberate refusals, not read failures: surface them to the caller
        # rather than degrading to the None "could not read" contract below.
        if opened is not None:
            opened.close()
        raise
    except (FileNotFoundError, ValueError, OSError) as e:
        if opened is not None:
            opened.close()
        # Expected failures: missing path, not a Zarr archive, or an I/O error
        # while reading. Per this function's contract, report and return None.
        logger.error(f"Failed to read Zarr archive: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None
    except Exception:
        if opened is not None:
            opened.close()
        # Unexpected errors (e.g. a programming bug like a wrong xarray call)
        # must surface rather than be masked as a None "read failure".
        logger.error(f"Unexpected error reading Zarr archive: {path}")
        raise


@register_xsnowDataset_method
def to_zarr(
    ds: xsnowDataset,
    path: Union[str, Path],
    mode: ZarrModes = 'w',
    consolidated: bool = True,
    logger: Optional[logging.Logger] = None,
    **kwargs
) -> None:
    """
    Saves the dataset to a Zarr archive.

    Zarr is a directory-based format that provides several advantages over
    NetCDF for large datasets:
    - Better stability (pure Python, no HDF5 crashes)
    - Lower memory usage (streaming reads)
    - Cloud-friendly (works with S3, GCS out of the box)
    - Same xarray interface

    This method is particularly useful for very large datasets that may
    cause memory issues with NetCDF.

    Args:
        ds: The dataset instance to save.
        path: The destination path for the .zarr directory.
        mode: Write mode for Zarr. See xarray.Dataset.to_zarr() for details.
            'w' (default): Create new archive (overwrite if exists)
            'w-': Create new archive (fail if exists)
            'a': Append to existing archive
            'a-': Append (create if doesn't exist)
            'r+': Read/write existing archive
        consolidated: Whether to consolidate metadata for faster opening.
            Default is True, which is recommended for most use cases.
        logger: An optional preconfigured logger instance.
        **kwargs: Additional keyword arguments passed to `xarray.Dataset.to_zarr()`.

    Examples
    --------
    >>> ds = xsnow.read('large_data/')
    >>> ds.to_zarr('output.zarr')

    >>> # Append new data to existing archive
    >>> new_ds = xsnow.read('new_data/')
    >>> new_ds.to_zarr('output.zarr', mode='a')
    """
    if ds.data is None:
        raise ValueError("Cannot save an empty dataset.")
    
    logger = get_logger(__name__, logger=logger)
    
    # Ensure we're working with the underlying xarray Dataset
    ds_to_save = ds.to_xarray() if hasattr(ds, 'to_xarray') else ds.data
    
    # Ensure .zarr extension
    path = Path(path)
    if not str(path).endswith('.zarr'):
        path = Path(str(path) + '.zarr')
        logger.warning(f"Added .zarr extension: {path}")
    
    # Save to Zarr
    logger.debug(f"Saving to Zarr: {path}")
    
    try:
        # Note: Zarr v3 uses default compression automatically
        # Custom encoding can be passed via kwargs if needed
        ds_to_save.to_zarr(
            path,
            mode=mode,
            consolidated=consolidated,
            **kwargs
        )
        
        logger.info(f"Dataset saved to {path}")
        
        # Report size
        result = subprocess.run(
            ['du', '-sh', str(path)],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            size_str = result.stdout.split()[0]
            logger.info(f"   Size: {size_str}")
        
    except Exception as e:
        logger.error(f"Failed to save Zarr archive: {e}")
        raise


# --------------------------------------------------------------------------- #
#                     FLAVOUR: SNOWPACK ZARRIO ARCHIVES                        #
# --------------------------------------------------------------------------- #

#: Days between the Julian day epoch and 1970-01-01T00:00Z.
_JULIAN_UNIX_EPOCH = 2440587.5
_ARCHIVE_RENAMES = {'critical_cut_length': 'rc_flat'}
_SLOT_LABEL_VARS = ('station_id', 'station_name')
_CHECKPOINT_GROUP = 'checkpoint'
_DEFAULT_TIME_CHUNK = 720
_CORE_DIMS = ('location', 'time', 'slope', 'realization', 'layer')

SlopeMetadata = Mapping[int, tuple[float, float]]
GeoMetadata = Mapping[str, tuple[float, float, float]]


def is_snowpack_zarrio(ds: xr.Dataset) -> bool:
    """Whether an opened Zarr store carries the SNOWPACK ZarrIO layout."""
    return (
        'snowpack_version' in ds.attrs
        and 'location' in ds.dims
        and 'location' not in ds.coords
        and 'slope' not in ds.dims
    )


StationIndex = Union[str, Path, Mapping[str, int]]


def _slot_labels(
    ds: xr.Dataset, station_index: Optional[StationIndex]
) -> np.ndarray:
    """Label of every location slot: from the archive's ``station_id`` array,
    else from a station index file (``<label> <slot>`` per line) or mapping.

    The plugin writes ``station_id`` only when it also writes the snow cover
    (``[Output] SNOW = ZARR``); runs keeping ``.sno`` state need the index
    file they pinned the slots with.
    """
    n_slots = ds.sizes['location']
    labels = np.full(n_slots, '', dtype='U64')
    if 'station_id' in ds.data_vars:
        labels[:] = _decode_labels(ds['station_id'].values)
    if station_index is not None:
        if isinstance(station_index, (str, Path)):
            mapping: dict[str, int] = {}
            with open(station_index, 'r', encoding='utf-8') as fh:
                for line in fh:
                    parts = line.split()
                    if not parts or parts[0].startswith('#'):
                        continue
                    mapping[parts[0]] = (
                        int(parts[1]) if len(parts) > 1 else len(mapping)
                    )
        else:
            mapping = {str(k): int(v) for k, v in station_index.items()}
        for label, slot in mapping.items():
            if 0 <= slot < n_slots and not labels[slot]:
                labels[slot] = label
    return labels


def open_snowpack_zarrio(
    path: Union[str, Path],
    chunks: Any = None,
    group: Optional[str] = None,
) -> xr.Dataset:
    """Open a ZarrIO archive (or one of its groups) lazily, without the
    consolidated-metadata fallback warning and zarr's complaint about the
    plugin's ``station_index.json`` side file."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore', message='.*station_index.json.*'
        )
        warnings.filterwarnings(
            'ignore', message='.*_WRITE_INCOMPLETE.*'
        )
        return xr.open_zarr(
            str(path),
            group=group,
            consolidated=False,
            chunks={} if chunks is None else chunks,
        )


def _decode_labels(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind == 'S':
        return np.char.decode(values, 'utf-8')
    return values.astype('U')


def _split_slots(labels: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Base station id and sector number of every slot label.

    A label equal to another label plus trailing digits is a virtual slope
    of that station; everything else is a main station (sector 0).
    """
    known = set(labels)
    base = []
    sector = []
    for label in labels:
        digits = re.search(r'\d+$', label)
        split = None
        if digits:
            # longest known base wins: 'MST964' -> ('MST96', 4)
            for n in range(1, len(digits.group(0)) + 1):
                if label[:-n] in known:
                    split = (label[:-n], int(label[-n:]))
                    break
        if split:
            base.append(split[0])
            sector.append(split[1])
        else:
            base.append(label)
            sector.append(0)
    return np.asarray(base, dtype='U'), np.asarray(sector, dtype='int64')


def _discover_side_files(
    path: Optional[Union[str, Path]],
    station_index: Optional[StationIndex],
    slope_metadata: Optional[SlopeMetadata],
) -> tuple[Optional[StationIndex], Optional[SlopeMetadata]]:
    """Pick up the index and layout files profsnow writes next to an archive
    (``<archive>.stations.txt``, ``<archive>.layout.json``) unless the caller
    supplied the information."""
    if path is None:
        return station_index, slope_metadata
    if station_index is None:
        index_file = Path(f'{path}.stations.txt')
        if index_file.is_file():
            station_index = index_file
    if slope_metadata is None:
        layout_file = Path(f'{path}.layout.json')
        if layout_file.is_file():
            try:
                recorded = json.loads(layout_file.read_text()).get(
                    'slope_metadata') or {}
                slope_metadata = {
                    int(k): (float(v[0]), float(v[1]))
                    for k, v in recorded.items()
                }
            except (ValueError, TypeError, IndexError, OSError):
                slope_metadata = None
    return station_index, slope_metadata


def _read_checkpoint(path: Optional[Union[str, Path]]) -> Optional[xr.Dataset]:
    if path is None:
        return None
    if not (Path(path) / _CHECKPOINT_GROUP / '.zgroup').exists():
        return None
    try:
        return open_snowpack_zarrio(path, group=_CHECKPOINT_GROUP)
    except Exception:  # noqa: BLE001 - terrain is optional
        return None


def _slot_terrain(
    n_slots: int,
    sector: np.ndarray,
    base: np.ndarray,
    checkpoint: Optional[xr.Dataset],
    slope_metadata: Optional[SlopeMetadata],
    geo: Optional[GeoMetadata],
) -> dict[str, np.ndarray]:
    """Per-slot inclination/azimuth/latitude/longitude/altitude.

    Checkpoint values win where finite, then caller-provided metadata,
    otherwise NaN (sector 0 defaults to flat).
    """
    out = {
        name: np.full(n_slots, np.nan)
        for name in ('inclination', 'azimuth',
                     'latitude', 'longitude', 'altitude')
    }
    out['inclination'][sector == 0] = 0.0
    out['azimuth'][sector == 0] = 0.0
    if slope_metadata:
        for i, k in enumerate(sector):
            if k in slope_metadata:
                out['inclination'][i], out['azimuth'][i] = slope_metadata[k]
    if geo:
        for i, b in enumerate(base):
            if b in geo:
                (out['latitude'][i], out['longitude'][i],
                 out['altitude'][i]) = geo[b]
    if checkpoint is not None:
        for name, var in (('inclination', 'slope_angle'),
                          ('azimuth', 'slope_azi'),
                          ('latitude', 'latitude'),
                          ('longitude', 'longitude'),
                          ('altitude', 'altitude')):
            if var not in checkpoint:
                continue
            values = np.asarray(checkpoint[var].values, dtype='float64')
            if values.shape[0] != n_slots:
                continue
            finite = np.isfinite(values)
            out[name][finite] = values[finite]
    return out


def _scan_n_layers(ds: xr.Dataset) -> tuple[np.ndarray, Optional[int]]:
    """Active time slots and the deepest profile, from one pass over the
    ``(location, time)`` array ``n_layers``.

    A slot is active when its timestamp decodes and, if the archive holds
    profile output, ``n_layers`` is finite at any location: pinned time
    grids fill the whole coordinate, so NaT alone does not mark unused
    slots. Without ``n_layers`` the deepest profile is unknown (None).
    """
    valid = ~np.isnat(ds['time'].values)
    if 'n_layers' not in ds.data_vars:
        return valid, None
    finite = np.isfinite(ds['n_layers'])
    written, deepest = dask.compute(
        finite.any('location'), ds['n_layers'].max()
    )
    valid &= np.asarray(written, dtype=bool)
    return valid, int(np.nan_to_num(np.asarray(deepest), nan=0.0))


def _julian_to_datetime(da: xr.DataArray) -> xr.DataArray:
    ns = (da - _JULIAN_UNIX_EPOCH) * 86400.0e9
    out = ns.astype('datetime64[ns]')
    out.attrs = {k: v for k, v in da.attrs.items() if k != 'units'}
    return out


def _convert_to_registry_units(ds: xr.Dataset) -> xr.Dataset:
    for name, da in list(ds.data_vars.items()):
        if name == 'deposition_date':
            ds[name] = _julian_to_datetime(da)
            continue
        entry = registry.get(name)
        if not entry:
            continue
        target = entry.get('xsnow_units')
        source = da.attrs.get('units')
        if not isinstance(target, str) or source is None:
            continue
        if normalize_unit(source) == normalize_unit(target):
            da.attrs['units'] = normalize_unit(target)
            continue
        try:
            ds[name] = da.units.to(target)
        except UnitConversionError as exc:
            raise ZarrIOLayoutError(
                f"Cannot convert archive variable {name!r} from "
                f"{source!r} to xsnow units {target!r}: {exc}"
            ) from exc
    return ds


def _attach_registry_attrs(ds: xr.Dataset) -> xr.Dataset:
    for name in ds.data_vars:
        entry = registry.get(name)
        if entry and 'standard_name' in entry:
            ds[name].attrs.setdefault('standard_name', entry['standard_name'])
    return ds


def adapt_snowpack_zarrio(
    ds: xr.Dataset,
    *,
    path: Optional[Union[str, Path]] = None,
    realization_label: Optional[str] = None,
    slope_metadata: Optional[SlopeMetadata] = None,
    geo: Optional[GeoMetadata] = None,
    station_index: Optional[StationIndex] = None,
    layers: Optional[int] = None,
    mask_stale_layers: bool = False,
    time_chunk: Optional[int] = _DEFAULT_TIME_CHUNK,
    logger: Optional[logging.Logger] = None,
) -> xr.Dataset:
    """
    Turn an opened ZarrIO archive into the canonical xsnow layout, lazily.

    Parameters
    ----------
    ds
        The archive as returned by :func:`open_snowpack_zarrio`.
    path
        Archive path; used to read the optional ``checkpoint/`` group for
        slope and position metadata.
    realization_label
        Label of the realization dimension; defaults to the archive's
        ``experiment`` attribute.
    slope_metadata
        ``{sector: (inclination, azimuth)}`` in degrees for the virtual
        slopes, used where the archive carries no checkpoint.
    geo
        ``{station_id: (latitude, longitude, altitude)}``, likewise.
    station_index
        Station index file (``<label> <slot>`` lines, the run's
        ``ZARR_STATION_INDEX_FILE``)
        or mapping naming the slots; needed when the archive carries no
        ``station_id`` array (runs with ``[Output] SNOW = SMET``).
    layers
        Length of the returned ``layer`` axis. Default: the deepest profile
        in the archive. Longer pads with NaN, shorter truncates (with a
        warning when data is lost).
    mask_stale_layers
        Mask layers at or beyond ``n_layers`` with NaN. Only needed for
        archives whose slots were rewritten with fewer layers than before.
    time_chunk
        Coarsen the Dask chunks along ``time`` to this size when the store
        was written with smaller chunks. None keeps the stored chunking.

    Returns
    -------
    xarray.Dataset
        Dims ``(location, time, slope, realization, layer)``, registry
        units, ``z`` coordinate, ``profile_status`` and ``n_layers``. Every
        multi-dimensional variable stays Dask-backed.

    Raises
    ------
    ZarrIOLayoutError
        If the archive is not a ZarrIO archive or a variable's storage unit
        has no conversion to xsnow units.
    """
    logger = get_logger(__name__, logger=logger)
    if not is_snowpack_zarrio(ds):
        raise ZarrIOLayoutError(
            "Dataset does not carry the SNOWPACK ZarrIO layout "
            "(expected root attr 'snowpack_version' and a 'location' axis)."
        )
    station_index, slope_metadata = _discover_side_files(
        path, station_index, slope_metadata
    )

    labels = _slot_labels(ds, station_index)
    written = np.array([bool(s) for s in labels])
    if not written.any():
        raise ZarrIOLayoutError(
            "Archive holds no labelled station slots: it carries no "
            "'station_id' array (written only with [Output] SNOW = ZARR); "
            "pass station_index= (the ZARR_STATION_INDEX_FILE of the run)."
        )
    checkpoint = _read_checkpoint(path)
    base, sector = _split_slots([s for s in labels])
    terrain = _slot_terrain(
        len(labels), sector, base, checkpoint, slope_metadata, geo
    )

    ds = ds.drop_vars([v for v in _SLOT_LABEL_VARS if v in ds])
    ds = ds.assign_coords(location=('location', labels))

    active, deepest = _scan_n_layers(ds)
    if not active.all():
        ds = ds.isel(time=np.flatnonzero(active))
    if not ds.indexes['time'].is_monotonic_increasing:
        ds = ds.sortby('time')
    ds['time'].attrs.setdefault('timezone', 'UTC')

    # terrain coordinates only when known: NaN slope coordinates would make
    # xsnow's slope normalization fold the sectors together
    have_slopes = bool(np.isfinite(terrain['inclination'][written]).all())
    have_geo = any(np.isfinite(terrain[n][written]).any()
                   for n in ('latitude', 'longitude', 'altitude'))
    parts = []
    for k in sorted(set(sector[written])):
        idx = np.flatnonzero((sector == k) & written)
        part = ds.isel(location=idx)
        part = part.assign_coords(location=('location', base[idx]))
        part = part.expand_dims(slope=[k])
        if have_slopes:
            part = part.assign_coords(
                inclination=(('location', 'slope'),
                             terrain['inclination'][idx][:, None]),
                azimuth=(('location', 'slope'),
                         terrain['azimuth'][idx][:, None]),
            )
        parts.append(part)
    ds = xr.concat(
        parts, dim='slope', join='outer', coords='minimal',
        compat='override', combine_attrs='override',
    )
    # stations in name order, like the PRO reader, whatever the slot order
    ds = ds.sortby('location')
    if have_geo:
        first = np.array(
            [np.flatnonzero(base == b)[0] for b in ds['location'].values]
        )
        ds = ds.assign_coords(
            latitude=('location', terrain['latitude'][first]),
            longitude=('location', terrain['longitude'][first]),
            altitude=('location', terrain['altitude'][first]),
        )

    label = realization_label or str(ds.attrs.get('experiment', 'NO_EXP'))
    ds = ds.expand_dims(realization=1)
    # a numpy unicode array, not a pandas index of objects, so the eager
    # memory guard can size the coordinate
    ds = ds.assign_coords(realization=np.array([label], dtype='U'))
    ds = ds.transpose(*_CORE_DIMS, missing_dims='ignore')

    ds = ds.rename({k: v for k, v in _ARCHIVE_RENAMES.items() if k in ds})
    ds = _convert_to_registry_units(ds)
    ds = _attach_registry_attrs(ds)

    if 'layer' in ds.dims:
        ds = _finish_layer_axis(ds, deepest, layers, mask_stale_layers, logger)
        if 'height' in ds and 'HS' in ds:
            z = ds['height'] - ds['HS']
            entry = registry.get('z', {})
            z.attrs = {
                'long_name': entry.get('long_name',
                                       'Depth Below Snow Surface'),
                'units': entry.get('xsnow_units', 'm'),
                'standard_name': entry.get('standard_name', 'depth'),
                'positive': 'up',
            }
            ds = ds.assign_coords(z=z)

    ds = _attach_profile_status(ds)
    ds.attrs.setdefault('crs', 'EPSG:4326')
    # the source's Zarr v2 encodings (numcodecs compressor, chunks, units)
    # would be reused by xarray when writing the adapted data to a store
    for var in list(ds.variables.values()):
        var.encoding = {}

    if time_chunk and ds['time'].size:
        ds = _regularize_time_chunks(ds, time_chunk)
    return ds


def _regularize_time_chunks(ds: xr.Dataset, time_chunk: int) -> xr.Dataset:
    """Give every Dask-backed data variable uniform ``time`` chunks (the last
    one may be shorter), coarsened to ``time_chunk``.

    Dropping inactive slots leaves ragged chunks that Zarr refuses to write;
    only data variables are rechunked so the small numpy coordinates stay
    numpy (``Dataset.chunk`` would wrap them in Dask too).
    """
    n = ds.sizes['time']
    size = min(time_chunk, n)
    regular = (size,) * (n // size) + ((n % size,) if n % size else ())
    for name, var in ds.data_vars.items():
        chunks = getattr(var.data, 'chunks', None)
        if chunks is None or 'time' not in var.dims:
            continue
        if tuple(chunks[var.get_axis_num('time')]) != regular:
            ds[name] = var.chunk({'time': size})
    return ds


def _finish_layer_axis(
    ds: xr.Dataset,
    deepest: Optional[int],
    layers: Optional[int],
    mask_stale_layers: bool,
    logger: logging.Logger,
) -> xr.Dataset:
    if 'n_layers' in ds:
        ds['n_layers'] = ds['n_layers'].fillna(0).astype('int16')
        ds['n_layers'].attrs = {
            'long_name': 'number of snow layers holding values',
            'units': '1',
            'standard_name': 'number_of_snow_layers',
        }
    if deepest is None:
        deepest = ds.sizes['layer']
    target = layers if layers is not None else max(deepest, 1)
    if deepest > target:
        logger.warning(
            f"Truncating the layer axis to {target} layers; the archive "
            f"holds profiles with up to {deepest} layers."
        )
    if target < ds.sizes['layer']:
        ds = ds.isel(layer=slice(0, target))
    ds = ds.assign_coords(layer=np.arange(ds.sizes['layer']))
    if target > ds.sizes['layer']:
        ds = ds.reindex(layer=np.arange(target))
    if mask_stale_layers and 'n_layers' in ds:
        valid = ds['layer'] < ds['n_layers']
        for name, da in ds.data_vars.items():
            if 'layer' in da.dims and da.dtype.kind in 'fM':
                ds[name] = da.where(valid).assign_attrs(da.attrs)
    return ds


def _attach_profile_status(ds: xr.Dataset) -> xr.Dataset:
    """``profile_status`` = 1 where the archive wrote a profile (``n_layers``
    finite, bare ground included), else 0; derived without a layer scan."""
    core = [d for d in ('location', 'time', 'slope', 'realization')
            if d in ds.dims]
    if 'n_layers' in ds:
        # n_layers is int16 here; the fill of unwritten slots is 0, which
        # cannot be told from bare ground, so use HS when available
        if 'HS' in ds and 'layer' not in ds['HS'].dims:
            status = np.isfinite(ds['HS']).astype('int8')
        else:
            status = (ds['n_layers'] > 0).astype('int8')
    elif core:
        status = xr.DataArray(
            np.zeros([ds.sizes[d] for d in core], dtype='int8'),
            dims=tuple(core),
            coords={d: ds.coords[d] for d in core},
        )
    else:
        return ds
    status = status.transpose(*core)
    status.attrs = {
        'long_name': 'profile status (snow layer data)',
        'units': '1',
        'standard_name': 'profile_status',
        'mapping': {-1: 'error during parsing', 0: 'no layer data',
                    1: 'existing layer data'},
    }
    ds['profile_status'] = status
    return ds


def read_snowpack_archives(
    stores: Iterable[Union[str, Path]],
    *,
    exclude: Iterable[str] = (),
    join: str = 'exact',
    logger: Optional[logging.Logger] = None,
    **read_kwargs: Any,
):
    """
    Open several ZarrIO archives as one lazy xsnowDataset.

    Parameters
    ----------
    stores
        Archive paths, e.g. one per station.
    exclude
        Station ids to drop (failed runs); an archive contributing no
        remaining station is skipped.
    join
        ``xarray.concat`` join along ``time``; ``'exact'`` requires the
        identical time grid in every archive.
    **read_kwargs
        Forwarded to :func:`xsnow.read` (``realization_label``,
        ``slope_metadata``, ``geo``, ``layers``, ...).
    """
    from . import read  # noqa: PLC0415 - avoid an import cycle

    logger = get_logger(__name__, logger=logger)
    excluded = set(exclude)
    datasets = []
    for store in stores:
        xs = read(store, flavour='snowpack', lazy=True, logger=logger,
                  **read_kwargs)
        if xs is None:
            raise ZarrIOLayoutError(f"Could not read archive {store}")
        data = xs.data
        keep = [loc for loc in data['location'].values
                if str(loc) not in excluded]
        if not keep:
            continue
        if len(keep) < data.sizes['location']:
            data = data.sel(location=keep)
        datasets.append(data)
    if not datasets:
        raise ZarrIOLayoutError("No archive contributed any station.")
    if len(datasets) == 1:
        return xsnowDataset(datasets[0])
    combined = xr.concat(
        datasets, dim='location', join=join, coords='minimal',
        compat='override', combine_attrs='drop_conflicts',
    )
    return xsnowDataset(combined)
