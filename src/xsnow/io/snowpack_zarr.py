"""
Reader for SNOWPACK ZarrIO archives.

SNOWPACK's ZarrIO plugin writes one Zarr v2 archive per run with profile
arrays on ``(location, time, layer)``, scalar time series on
``(location, time)``, station labels in ``station_id``/``station_name``,
SNOWPACK storage units (cm, %, kPa, ...) and one location slot per virtual
slope (``<station><sector>``). :func:`adapt_snowpack_zarrio` turns such an
archive, opened lazily with xarray, into the canonical xsnow layout without
materializing any layered variable; :func:`read_snowpack_archives` opens
several archives as one dataset.
"""
from __future__ import annotations

import logging
import re
import warnings
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

import dask
import numpy as np
import xarray as xr

from ..core import xsnowDataset
from ..registry import registry
from ..units import UnitConversionError, normalize_unit
from ..utils import get_logger

__all__ = [
    'ZarrIOLayoutError',
    'adapt_snowpack_zarrio',
    'is_snowpack_zarrio',
    'open_snowpack_zarrio',
    'read_snowpack_archives',
]

#: Days between the Julian day epoch and 1970-01-01T00:00Z.
_JULIAN_UNIX_EPOCH = 2440587.5
_ARCHIVE_RENAMES = {'critical_cut_length': 'rc_flat'}
_SLOT_LABEL_VARS = ('station_id', 'station_name')
_CHECKPOINT_GROUP = 'checkpoint'
_DEFAULT_TIME_CHUNK = 720
_CORE_DIMS = ('location', 'time', 'slope', 'realization', 'layer')

SlopeMetadata = Mapping[int, tuple[float, float]]
GeoMetadata = Mapping[str, tuple[float, float, float]]


class ZarrIOLayoutError(RuntimeError):
    """The archive cannot be adapted to the xsnow layout."""


def is_snowpack_zarrio(ds: xr.Dataset) -> bool:
    """Whether an opened Zarr store carries the SNOWPACK ZarrIO layout."""
    return (
        'snowpack_version' in ds.attrs
        and 'station_id' in ds.data_vars
        and 'location' not in ds.coords
        and 'slope' not in ds.dims
    )


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
            "(expected root attr 'snowpack_version' and a 'station_id' array)."
        )

    labels = _decode_labels(ds['station_id'].values)
    written = np.array([bool(s) for s in labels])
    if not written.any():
        raise ZarrIOLayoutError("Archive holds no written station slots.")
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

    parts = []
    for k in sorted(set(sector[written])):
        idx = np.flatnonzero((sector == k) & written)
        part = ds.isel(location=idx)
        part = part.assign_coords(location=('location', base[idx]))
        part = part.expand_dims(slope=[k])
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
