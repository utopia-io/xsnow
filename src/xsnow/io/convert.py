# io/convert.py
"""
Batch-append converter: build a Zarr season archive from PRO/SMET files.

A season archive is written in batches rather than in one pass. A fully lazy
conversion of a large PRO/SMET collection builds a Dask graph big enough to
stall before any data is written, so each batch is read *eagerly* and appended
to the store, keeping peak memory bounded by the batch size instead of the
collection size.

PRO files are per-station and each spans the whole season, so a batch is a
disjoint set of stations covering the full time range: the append axis is
``location``, not ``time``. That makes the store's non-append dimensions
(``time`` and ``layer``) fixed at creation, which is why this module pre-scans
the inputs before writing anything:

- **Layer dimension.** A store created with full-layer chunks fixes the layer
  size permanently. A later batch with a deeper snowpack would shape-mismatch
  on append -- silently fine early season, broken mid-winter. The pre-scan
  finds the deepest profile across every input file and pads all batches to it.
- **Time axis.** Every batch must land on the same time coordinate, so the
  pre-scan collects the global time span and each batch is reindexed onto one
  canonical axis before it is appended.

This is also where the archive's *physical* chunk layout is chosen, which is a
write-time decision that cannot be cheaply undone on read.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

from ..utils import get_logger

from .base import _sanitize_station_name
from .memory import MemoryLimitLike, resolve_memory_limit
from .merge import _resolve_filepaths
from .pro import _extract_metadata_fast, _scan_pro_for_max_layers
from .smet import _parse_smet_header


#: Initial guess for bytes of decoded data per (timestep, layer) cell, used to
#: size the first batch. Recalibrated from the first batch's real size, so the
#: guess only has to be the right order of magnitude.
_DEFAULT_BYTES_PER_CELL = 240

#: Default per-batch memory target when the caller does not specify one.
_DEFAULT_BATCH_BYTES = "2GB"


class ArchiveAppendError(RuntimeError):
    """Raised when a batch cannot be appended to the archive as-is."""


@dataclass
class _FileRecord:
    """Pre-scan facts about one input file, gathered without parsing payload."""

    path: Path
    station: str
    kind: str  # 'pro' or 'smet'
    n_timesteps: int = 0
    max_layers: int = 0
    first_time: Optional[pd.Timestamp] = None
    last_time: Optional[pd.Timestamp] = None

    @property
    def cells(self) -> int:
        """Cell count driving this file's decoded memory footprint."""
        return max(self.n_timesteps, 1) * max(self.max_layers, 1)


@dataclass
class PrescanResult:
    """The canonical grid the whole archive is written onto."""

    records: List[_FileRecord] = field(default_factory=list)
    max_layers: int = 0
    time_start: Optional[pd.Timestamp] = None
    time_end: Optional[pd.Timestamp] = None

    @property
    def stations(self) -> List[str]:
        return sorted({record.station for record in self.records})

    def by_station(self) -> Dict[str, List[_FileRecord]]:
        """Group records by station so a station's files stay in one batch."""
        grouped: Dict[str, List[_FileRecord]] = {}
        for record in self.records:
            grouped.setdefault(record.station, []).append(record)
        return grouped


def _prescan(
    pro_files: Sequence[Path],
    smet_files: Sequence[Path],
    *,
    add_surface_sh_as_layer: bool = True,
    remove_soil: bool = True,
    logger: Optional[logging.Logger] = None,
) -> PrescanResult:
    """
    Collect the canonical layer depth and time span without parsing payload.

    Reads only headers, timestamp lines, and layer counts. This is the pass
    that makes the mid-winter layer mismatch impossible rather than latent.
    """
    log = get_logger(__name__, logger=logger)
    result = PrescanResult()
    times: List[pd.Timestamp] = []

    for path in pro_files:
        metadata = _extract_metadata_fast(path, logger=log)
        layers = _scan_pro_for_max_layers(
            path, add_surface_sh_as_layer, remove_soil, logger=log
        )
        first = _as_timestamp(metadata.get("first_timestamp"))
        last = _as_timestamp(metadata.get("last_timestamp"))
        record = _FileRecord(
            path=path,
            station=_sanitize_station_name(str(metadata.get("station_name", path.stem))),
            kind="pro",
            n_timesteps=int(metadata.get("estimated_profiles", 0) or 0),
            max_layers=layers,
            first_time=first,
            last_time=last,
        )
        result.records.append(record)
        result.max_layers = max(result.max_layers, layers)
        times.extend(t for t in (first, last) if t is not None)

    for path in smet_files:
        try:
            header = _parse_smet_header(path)
        except Exception as exc:  # pragma: no cover - unreadable header
            log.warning(f"Could not read SMET header for {path.name}: {exc}")
            header = {}
        station = header.get("station_name") or header.get("station_id") or path.stem
        result.records.append(
            _FileRecord(
                path=path,
                station=_sanitize_station_name(str(station)),
                kind="smet",
            )
        )

    if times:
        result.time_start = min(times)
        result.time_end = max(times)

    if result.max_layers <= 0 and pro_files:
        raise ArchiveAppendError(
            "Pre-scan found no snow layers in any input .pro file. The archive "
            "layer dimension cannot be determined; pass max_layers explicitly."
        )

    log.info(
        f"Pre-scan: {len(result.records)} files, {len(result.stations)} stations, "
        f"max_layers={result.max_layers}, "
        f"time {result.time_start} -> {result.time_end}"
    )
    return result


#: PRO timestamp layout. Day-first, so a bare ``pd.to_datetime`` would read
#: ``05.01.2025`` as 5 May instead of 5 January.
_PRO_TIME_FORMAT = "%d.%m.%Y %H:%M:%S"


def _as_timestamp(value: Any) -> Optional[pd.Timestamp]:
    """
    Coerce a pre-scan timestamp to pandas, tolerating unparseable values.

    ``_extract_metadata_fast`` returns the raw ``0500`` field, which is a
    day-first PRO timestamp string, so the format is given explicitly to match
    how the PRO parser itself reads it.
    """
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value

    for parse in (
        lambda v: pd.to_datetime(v, format=_PRO_TIME_FORMAT),
        lambda v: pd.to_datetime(v, dayfirst=True),
    ):
        try:
            stamp = parse(value)
        except (ValueError, TypeError, OverflowError):
            continue
        return None if pd.isna(stamp) else stamp
    return None


_StationItem = Tuple[str, List[_FileRecord]]


def _station_cells(records: Sequence[_FileRecord]) -> int:
    """Cells contributed by one station, driving its decoded footprint."""
    return sum(record.cells for record in records if record.kind == "pro")


def _group_into_batches(
    stations: Sequence[_StationItem],
    target_bytes: int,
    bytes_per_cell: float = _DEFAULT_BYTES_PER_CELL,
) -> List[List[_StationItem]]:
    """
    Pack stations into batches under a decoded-bytes target.

    A station's files always travel together: its PRO and SMET data have to be
    read in the same ``read()`` call to merge into one location. A station
    that exceeds the target on its own still gets its own batch rather than
    being split, since splitting it would break that merge.
    """
    batches: List[List[_StationItem]] = []
    current: List[_StationItem] = []
    current_bytes = 0.0

    for station, records in stations:
        station_bytes = _station_cells(records) * bytes_per_cell

        if current and current_bytes + station_bytes > target_bytes:
            batches.append(current)
            current, current_bytes = [], 0.0

        current.append((station, records))
        current_bytes += station_bytes

    if current:
        batches.append(current)
    return batches


def _batch_paths(batch: Sequence[_StationItem]) -> List[Path]:
    """Flatten a batch of stations into the file paths ``read()`` receives."""
    return [record.path for _, records in batch for record in records]


def _canonical_time_axis(
    first_batch: xr.Dataset,
    prescan: PrescanResult,
    time_axis: Optional[pd.DatetimeIndex],
    logger: logging.Logger,
) -> pd.DatetimeIndex:
    """
    Build the single time coordinate every batch is reindexed onto.

    Derived from the first batch's observed frequency spanning the pre-scan's
    global time range, so batches whose own coverage is shorter still line up.
    """
    if time_axis is not None:
        return pd.DatetimeIndex(time_axis)

    observed = pd.DatetimeIndex(first_batch["time"].values)
    if len(observed) < 2:
        return observed

    freq = pd.infer_freq(observed)
    if freq is None:
        # Irregular sampling: fall back to the observed stride. A wrong guess
        # here would silently drop timestamps, so verify it covers the span.
        deltas = np.diff(observed.values)
        step = pd.Timedelta(np.median(deltas))
        if step <= pd.Timedelta(0):
            raise ArchiveAppendError(
                "Could not infer a time step from the first batch. Pass an "
                "explicit time_axis to define the archive's time coordinate."
            )
        logger.warning(
            f"Irregular timestamps in the first batch; assuming a {step} step. "
            f"Pass time_axis explicitly if that is wrong."
        )
        freq = step

    start = prescan.time_start or observed[0]
    end = prescan.time_end or observed[-1]
    return pd.date_range(start=min(start, observed[0]), end=max(end, observed[-1]), freq=freq)


def _align_to_grid(
    ds: xr.Dataset,
    canonical_time: pd.DatetimeIndex,
    max_layers: int,
    *,
    batch_label: str,
    on_time_mismatch: str,
    logger: logging.Logger,
) -> xr.Dataset:
    """
    Reindex a batch onto the archive's canonical time and layer grid.

    Timestamps that fall off the canonical axis would be dropped silently by
    ``reindex``, so they are counted and surfaced instead.
    """
    if "time" in ds.dims:
        incoming = pd.DatetimeIndex(ds["time"].values)
        lost = incoming.difference(canonical_time)
        if len(lost):
            message = (
                f"{batch_label}: {len(lost)} timestamp(s) fall outside the "
                f"archive's time axis and would be dropped "
                f"(first: {lost[0]}, last: {lost[-1]}). Widen the archive with "
                f"an explicit time_axis, or pass on_time_mismatch='warn' to "
                f"drop them deliberately."
            )
            if on_time_mismatch == "raise":
                raise ArchiveAppendError(message)
            logger.warning(message)
        ds = ds.reindex(time=canonical_time)

    if "layer" in ds.dims and ds.sizes["layer"] != max_layers:
        if ds.sizes["layer"] > max_layers:
            raise ArchiveAppendError(
                f"{batch_label}: has {ds.sizes['layer']} layers but the archive "
                f"layer dimension is {max_layers}. The pre-scan should have "
                f"caught this; pass a larger max_layers to rebuild the archive."
            )
        ds = ds.reindex(layer=np.arange(max_layers))

    return ds


@dataclass
class _ArchiveSchema:
    """The shape an appended batch has to match."""

    sizes: Dict[str, int] = field(default_factory=dict)
    var_dims: Dict[str, Tuple[str, ...]] = field(default_factory=dict)

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> "_ArchiveSchema":
        return cls(
            sizes=dict(ds.sizes),
            var_dims={
                str(name): tuple(str(d) for d in var.dims)
                for name, var in ds.variables.items()
            },
        )


def _validate_appendable(
    reference: _ArchiveSchema,
    ds: xr.Dataset,
    append_dim: str,
    batch_label: str,
) -> None:
    """
    Check a batch against the archive's shape before handing it to Zarr.

    Two things have to line up. Every non-append dimension must have the same
    size, and every shared variable must have the same *dimension names* --
    a variable that is ``('slope',)`` in one batch and ``('location', 'slope')``
    in another cannot be appended, and Zarr reports that only from deep inside
    the writer.
    """
    mismatches = [
        f"dimension {dim}: archive has {size}, batch has {ds.sizes[dim]}"
        for dim, size in reference.sizes.items()
        if dim != append_dim and dim in ds.sizes and ds.sizes[dim] != size
    ]

    missing = [
        dim for dim in reference.sizes if dim != append_dim and dim not in ds.sizes
    ]
    if missing:
        mismatches.append(f"batch is missing dimension(s): {', '.join(sorted(missing))}")

    batch_dims = {
        str(name): tuple(str(d) for d in var.dims)
        for name, var in ds.variables.items()
    }
    for name, dims in sorted(reference.var_dims.items()):
        if name in batch_dims and batch_dims[name] != dims:
            mismatches.append(
                f"variable {name!r}: archive has dims {dims}, batch has "
                f"{batch_dims[name]}"
            )

    if mismatches:
        raise ArchiveAppendError(
            f"{batch_label}: cannot append, batch does not match the archive.\n  "
            + "\n  ".join(mismatches)
            + "\n\nBatches must have identical structure. If slope coordinates "
            "differ, ensure norm_slopes=False so every batch keeps "
            "per-location slope coordinates."
        )


def build_archive(
    source: Union[str, Path, Sequence[Union[str, Path]]],
    target: Union[str, Path],
    *,
    recursive: bool = False,
    batch_bytes: MemoryLimitLike = _DEFAULT_BATCH_BYTES,
    max_layers: Optional[int] = None,
    time_axis: Optional[pd.DatetimeIndex] = None,
    chunks: Optional[Dict[str, int]] = None,
    encoding: Optional[Dict[str, Dict[str, Any]]] = None,
    mode: str = "w-",
    append_dim: str = "location",
    consolidated: bool = True,
    on_time_mismatch: str = "raise",
    logger: Optional[logging.Logger] = None,
    **read_kwargs: Any,
) -> Path:
    """
    Build or extend a Zarr season archive from a PRO/SMET collection in batches.

    Each batch is read eagerly and appended, so peak memory tracks the batch
    size rather than the size of the whole collection. This is the point where
    the archive's physical chunk layout is decided, which is a write-time
    choice that cannot be cheaply undone on read.

    Args:
        source: Directory, file, or list of paths holding .pro/.smet files.
        target: Destination .zarr archive path.
        recursive: Search subdirectories when ``source`` is a directory.
        batch_bytes: Decoded-bytes target per batch. Accepts the same forms as
            ``read()``'s ``memory_limit`` (`'2GB'`, a byte count, or `'auto'`
            for a fraction of available RAM). The estimate is recalibrated
            from the first batch's real size.
        max_layers: Archive layer dimension. None (default) pre-scans every
            input file for the deepest profile, which is what prevents a
            later, deeper batch from shape-mismatching on append.
        time_axis: Explicit archive time coordinate. None (default) derives it
            from the pre-scanned time span at the first batch's frequency.
        chunks: Physical chunk sizes for the store, e.g.
            ``{'location': 200, 'time': 24, 'layer': -1}``. ``-1`` means the
            full extent of that dimension in one chunk.
        encoding: Per-variable Zarr encoding, applied when the store is
            created. Ignored for appends, which inherit the store's encoding.
        mode: ``'w-'`` (default) creates and fails if the archive exists,
            ``'w'`` overwrites, ``'a'`` appends every batch into an existing
            archive.
        append_dim: Dimension batches are appended along. Defaults to
            ``'location'`` because PRO files are per-station and each spans
            the full season.
        consolidated: Consolidate Zarr metadata for faster opening.
        on_time_mismatch: ``'raise'`` (default) rejects a batch holding
            timestamps outside the archive's time axis; ``'warn'`` drops them
            with a warning.
        logger: Optional preconfigured logger.
        **read_kwargs: Forwarded to ``read()`` for every batch (for example
            ``remove_soil`` or ``unit_validation``).

    Returns:
        The path to the written archive.

    Raises:
        ArchiveAppendError: If a batch cannot be aligned to the archive grid.
        FileNotFoundError: If no .pro/.smet files are found in ``source``.

    Examples:
        >>> # Build a season archive with a station-chunked layout
        >>> build_archive(
        ...     "season_runs/",
        ...     "season.zarr",
        ...     chunks={"location": 200, "time": 24, "layer": -1},
        ...     batch_bytes="4GB",
        ... )

        >>> # Extend an existing archive with a new set of stations
        >>> build_archive("more_stations/", "season.zarr", mode="a")

    See Also
    --------
    to_zarr : Write an already-loaded dataset to a Zarr archive
    read : Load PRO/SMET/NetCDF/Zarr data into an xsnowDataset
    """
    from . import read  # local import: read() lives in this subpackage's __init__

    log = get_logger(__name__, logger=logger)
    target = Path(target)
    if not str(target).endswith(".zarr"):
        target = Path(str(target) + ".zarr")
        log.warning(f"Added .zarr extension: {target}")

    if on_time_mismatch not in {"raise", "warn"}:
        raise ValueError(
            f"on_time_mismatch must be 'raise' or 'warn', got {on_time_mismatch!r}"
        )

    pro_files, smet_files = _resolve_filepaths(source, recursive, log)
    if not pro_files and not smet_files:
        raise FileNotFoundError(f"No .pro or .smet files found in {source}")

    # read()'s default norm_slopes='auto' decides whether slope coords collapse
    # to ('slope',) or stay ('location', 'slope') based on whether the stations
    # it happens to see share a slope. Across batches that is not stable: a
    # batch of same-slope stations would produce a different structure from a
    # batch of differing ones and fail to append. False keeps the general
    # per-location form for every batch.
    if "norm_slopes" not in read_kwargs:
        read_kwargs["norm_slopes"] = False
    elif read_kwargs["norm_slopes"] is not False:
        log.warning(
            f"norm_slopes={read_kwargs['norm_slopes']!r} makes a batch's slope "
            f"coordinates depend on which stations it contains, which can make "
            f"batches fail to append. norm_slopes=False is recommended."
        )

    target_bytes = resolve_memory_limit(batch_bytes, logger=log)
    if target_bytes is None:
        target_bytes = resolve_memory_limit(_DEFAULT_BATCH_BYTES, logger=log)
        log.debug(f"Falling back to a {_DEFAULT_BATCH_BYTES} batch target")

    prescan = _prescan(
        pro_files,
        smet_files,
        add_surface_sh_as_layer=read_kwargs.get("add_surface_sh_as_layer", True),
        remove_soil=read_kwargs.get("remove_soil", True),
        logger=log,
    )
    archive_layers = max_layers if max_layers is not None else prescan.max_layers
    if max_layers is not None and max_layers < prescan.max_layers:
        raise ArchiveAppendError(
            f"max_layers={max_layers} is smaller than the deepest profile found "
            f"by the pre-scan ({prescan.max_layers}). Deeper batches would "
            f"shape-mismatch on append."
        )

    station_items: List[_StationItem] = sorted(prescan.by_station().items())
    bytes_per_cell: float = _DEFAULT_BYTES_PER_CELL
    batches = _group_into_batches(station_items, int(target_bytes), bytes_per_cell)
    log.info(
        f"Writing {target} in ~{len(batches)} batch(es), "
        f"target {int(target_bytes) / 1024**3:.2f} GiB each"
    )

    canonical_time: Optional[pd.DatetimeIndex] = (
        pd.DatetimeIndex(time_axis) if time_axis is not None else None
    )
    reference_schema: Optional[_ArchiveSchema] = None
    n_written = 0
    index = 0
    pending: List[_StationItem] = list(station_items)

    while pending:
        batch = batches[0] if batches else [pending[0]]
        batch_files = _batch_paths(batch)
        label = f"batch {index + 1}"
        log.info(
            f"{label}: reading {len(batch_files)} file(s) "
            f"across {len(batch)} station(s)"
        )

        batch_result = read(
            list(batch_files),
            lazy=False,
            max_layers=archive_layers,
            logger=log,
            **read_kwargs,
        )

        # Advance past the stations this batch consumed, however it turns out.
        pending = pending[len(batch):]
        index += 1

        if batch_result is None or batch_result.data is None:
            log.warning(f"{label}: no readable data, skipping")
            batches = _group_into_batches(pending, int(target_bytes), bytes_per_cell)
            continue

        data = batch_result.data

        # Calibrate the size model from the first batch's real decoded size,
        # then re-plan the remaining batches. The initial bytes-per-cell guess
        # only has to be the right order of magnitude.
        if n_written == 0:
            observed_cells = sum(_station_cells(records) for _, records in batch)
            if observed_cells > 0:
                measured = int(data.nbytes) / observed_cells
                if measured > 0:
                    log.debug(
                        f"Calibrated batch sizing: {measured:.1f} bytes/cell "
                        f"(initial estimate {bytes_per_cell:.1f})"
                    )
                    bytes_per_cell = measured
        batches = _group_into_batches(pending, int(target_bytes), bytes_per_cell)

        if canonical_time is None:
            canonical_time = _canonical_time_axis(data, prescan, None, log)
            log.info(
                f"Archive time axis: {len(canonical_time)} steps, "
                f"{canonical_time[0]} -> {canonical_time[-1]}"
            )

        data = _align_to_grid(
            data,
            canonical_time,
            archive_layers,
            batch_label=label,
            on_time_mismatch=on_time_mismatch,
            logger=log,
        )

        if chunks:
            applicable = {
                dim: (data.sizes[dim] if size == -1 else size)
                for dim, size in chunks.items()
                if dim in data.dims
            }
            if applicable:
                data = data.chunk(applicable)

        if n_written == 0 and mode != "a":
            write_kwargs: Dict[str, Any] = {"mode": mode}
            if encoding:
                write_kwargs["encoding"] = encoding
            data.to_zarr(target, consolidated=consolidated, **write_kwargs)
            reference_schema = _ArchiveSchema.from_dataset(data)
        else:
            if reference_schema is None:
                reference_schema = _existing_schema(target)
            _validate_appendable(reference_schema, data, append_dim, label)
            data.to_zarr(
                target,
                mode="a",
                append_dim=append_dim,
                consolidated=consolidated,
            )

        n_written += 1
        log.info(f"{label}: appended {data.sizes.get(append_dim, 0)} {append_dim}(s)")

    if n_written == 0:
        raise ArchiveAppendError(
            f"No batches were written to {target}; every batch was empty."
        )

    log.info(f"Archive written: {target} ({n_written} batch(es))")
    return target


def _existing_schema(target: Path) -> _ArchiveSchema:
    """Read an existing archive's shape without loading its data."""
    with xr.open_zarr(target) as existing:
        return _ArchiveSchema.from_dataset(existing)
