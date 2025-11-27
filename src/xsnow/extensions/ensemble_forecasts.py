"""
Ensemble and/or forecast analysis extension

This extension aims to 
 1. support working with ensemble simulations and 
 2. facilitate explicit forecast analysis in the sense of comparing model runs. 

While the extension focuses on the ensemble context, analyzing the performance of different
forecast model runs for a deterministic model is just a simple special case of the full functionality.

Objectives of this extension
----
Extend the ``xsnowDataset`` class with additional dimensions 
(`realization` for the ensemble members, `run` for different forecast model runs).


Expected layout of data source file structure
---------------
``source/{run}/{member}/{location}.{smet|pro|nc}``
Examples:
    deterministic-forecast-analysis/
        nowcast/            # run: this could be a stacked pseudo-run that spans the entire season
            det/            # member: this example has only the deterministic model
                VIR1A.smet  # one specific location, time spans entire season
                VIR2A.smet
        2025-12-01T00:00Z/  # run: specific model run, this directory contains only few timesteps 
            det/
                VIR1A.smet  # only includes forecast data from 2025-12-01T00:00Z--full leadtime (~72h)
                VIR2A.smet
        2025-12-01T09:00Z/
            det/
                VIR1A.smet
                VIR2A.smet


    ensemble-forecast-analysis/
        nowcast/            # run: as above
            det/            # member: deterministic control member
                VIR1A.smet  # location: as above
                VIR2A.smet
        2025-12-01T00:00Z/
            det/            # specific forecast of control member
                VIR1A.smet
                VIR2A.smet
            p01/            # specific forecast of ensemble member
                VIR1A.smet
                VIR2A.smet
        2025-12-01T09:00Z/
            det/
                VIR1A.smet
                VIR2A.smet
            p01/
                VIR1A.smet
                VIR2A.smet

What this module adds
---------------------
* ``run`` dimension (string), populated from the run folder names, with a timezone attribute.
* ``realization`` dimension (string), populated from the member folder names.
* ``run_start`` coordinate (datetime64[ns], tz in attrs) for each run.
* ``leadtime`` coordinate (float hours) computed from run_start to valid time.

Key APIs
--------
* ``read_ensemble_fx``: read (requested) runs/members/files into an
  ``EnsembleFX`` decorator (subclasses ``DatasetDecorator``).
* ``scan_source``: list available runs, members, and file bases up front.
"""
from __future__ import annotations

import fnmatch
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

import xsnow as _xsnow
from xsnow import DatasetDecorator, read as xsnow_read
from xsnow.core import xsnowDataset
from xsnow.accessors import _coerce_tzinfo, _normalize_tz_attr_value

logger = logging.getLogger(__name__)

DEFAULT_EXTS = (".smet", ".pro", ".nc")
# Prefer xsnow.concat if available; fallback to xarray.concat.
_xs_concat = getattr(_xsnow, "concat", xr.concat)


def _normalize_selection(values: Optional[Union[str, Iterable[str]]]) -> Optional[set[str]]:
    if values is None:
        return None
    if isinstance(values, str):
        return {values}
    return {str(v) for v in values}


def _normalize_patterns(patterns: Optional[Union[str, Sequence[str]]]) -> Optional[List[str]]:
    if patterns is None:
        return None
    if isinstance(patterns, str):
        return [patterns]
    return [str(p) for p in patterns]


def _parse_run_timestamp(run_label: str, run_timezone: Optional[Union[str, int, float]]) -> Optional[pd.Timestamp]:
    """
    Return a tz-aware Timestamp if the run label can be parsed; otherwise None.
    If the label is tz-naive and a run_timezone is provided, localize to it.
    """
    try:
        ts = pd.to_datetime(run_label)
    except Exception:
        return None

    try:
        if ts.tzinfo is None:
            if run_timezone is None:
                return None
            return ts.tz_localize(_coerce_tzinfo(run_timezone))
        if run_timezone is not None:
            return ts.tz_convert(_coerce_tzinfo(run_timezone))
        return ts
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Failed to interpret run timestamp %s: %s", run_label, exc)
        return None


def _ensure_realization_dim(ds: xr.Dataset, realization_label: str) -> xr.Dataset:
    """
    Guarantee a string-valued realization coordinate carrying the member label.
    If realization already exists and is length 1, overwrite its value.
    """
    label = str(realization_label)
    if "realization" in ds.dims:
        if ds.sizes.get("realization", 0) == 1:
            ds = ds.assign_coords(realization=("realization", np.array([label], dtype=object)))
        return ds
    return ds.expand_dims(realization=[label])


def _collect_files(
    member_dir: Path,
    patterns: Optional[List[str]],
    allowed_exts: Sequence[str],
    fname_filter: Optional[set[str]],
) -> List[Path]:
    files: List[Path] = []
    for f in member_dir.iterdir():
        if not f.is_file():
            continue
        if f.suffix.lower() not in allowed_exts:
            continue
        base = f.stem
        if fname_filter and base not in fname_filter:
            continue
        if patterns and not any(fnmatch.fnmatch(base, pat) for pat in patterns):
            continue
        files.append(f)
    return sorted(files)


def _compute_leadtime_hours(
    ds: xr.Dataset,
    run_ts: Optional[pd.Timestamp],
    run_tz: Optional[str],
) -> Optional[xr.DataArray]:
    """
    Compute leadtime in hours as float. Returns None if no time coordinate exists.
    """
    if "time" not in ds:
        return None

    time_coord = ds["time"]
    source_tz = time_coord.attrs.get("timezone", run_tz or "UTC")

    if run_ts is None:
        values = np.full(time_coord.shape, np.nan, dtype=float)
        note = "Run timestamp missing; leadtime set to NaN."
    else:
        time_idx = time_coord.tz.localize(tz=source_tz).tz_convert(run_ts.tzinfo)
        deltas = (time_idx - run_ts) / pd.Timedelta(hours=1)
        values = deltas.to_numpy()
        note = None

    lead = xr.DataArray(
        values,
        dims=time_coord.dims,
        coords={dim: ds[dim] for dim in time_coord.dims},
        name="leadtime",
        attrs={
            "units": "hours",
            "description": "Lead time between run start and valid time.",
        },
    )
    if run_tz:
        lead.attrs["run_timezone"] = run_tz
    if note:
        lead.attrs["note"] = note
    return lead


def scan_source(
    source: Union[str, Path],
    *,
    run: Optional[Union[str, Iterable[str]]] = None,
    member: Optional[Union[str, Iterable[str]]] = None,
    fname_base: Optional[Union[str, Iterable[str]]] = None,
    filename_pattern: Optional[Union[str, Sequence[str]]] = None,
    allowed_exts: Sequence[str] = DEFAULT_EXTS,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Discover available runs, members, and station file bases in a forecast tree.

    Expected layout: ``source/{run}/{member}/{station}.{ext}`` where ``ext`` is in
    ``allowed_exts``. This is a lightweight scan (no file reads) to drive UI
    selection or preflight checks.

    Parameters
    ----------
    source : str or Path
        Root directory that contains run subfolders.
    run, member, fname_base : str or iterable of str, optional
        If provided, restrict the scan to these run labels, member labels, or
        file bases (filename without extension).
    filename_pattern : str or sequence of str, optional
        fnmatch-style patterns applied to the filename base (e.g., ``"VIR*"``).
    allowed_exts : sequence of str, default (".smet", ".pro", ".nc")
        File extensions considered during the scan (case-insensitive).

    Returns
    -------
    (runs, members, fname_bases) : tuple[list[str], list[str], list[str]]
        Sorted unique values discovered under the given constraints.
    """
    src = Path(source)
    if not src.exists():
        raise FileNotFoundError(f"Source path does not exist: {source}")

    run_sel = _normalize_selection(run)
    member_sel = _normalize_selection(member)
    fname_sel = _normalize_selection(fname_base)
    patterns = _normalize_patterns(filename_pattern)

    runs, members, bases = set(), set(), set()

    for run_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        if run_sel and run_dir.name not in run_sel:
            continue
        runs.add(run_dir.name)

        for member_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
            if member_sel and member_dir.name not in member_sel:
                continue
            members.add(member_dir.name)

            for f in member_dir.iterdir():
                if not f.is_file():
                    continue
                if f.suffix.lower() not in allowed_exts:
                    continue
                base = f.stem
                if fname_sel and base not in fname_sel:
                    continue
                if patterns and not any(fnmatch.fnmatch(base, pat) for pat in patterns):
                    continue
                bases.add(base)

    return sorted(runs), sorted(members), sorted(bases)


class EnsembleFX(DatasetDecorator):
    """
    Decorator that enriches an xsnowDataset with run context and leadtimes.

    Dimensions/coordinates guaranteed after ``read_ensemble_fx``:
    - ``run`` (string) with attrs including optional ``timezone``.
    - ``realization`` (string) describing the ensemble member label.
    - ``run_start`` coordinate on ``run`` (datetime64[ns], tz in attrs; NaT if unknown).
    - ``leadtime`` coordinate on ``time`` (float hours from run_start to valid time).

    All existing xsnowDataset API remains available via inheritance.
    """
    name = "ensemble_fx"
    priority = 30


def read_ensemble_fx(
    source: Union[str, Path],
    *,
    run: Optional[Union[str, Iterable[str]]] = None,
    realization: Optional[Union[str, Iterable[str]]] = None,
    fname_base: Optional[Union[str, Iterable[str]]] = None,
    filename_pattern: Optional[Union[str, Sequence[str]]] = None,
    run_timezone: Optional[Union[str, int, float]] = "UTC",
    allowed_exts: Sequence[str] = DEFAULT_EXTS,
    squeeze_run: bool = True,
    **read_kwargs,
) -> Optional[EnsembleFX]:
    """
    Read a forecast collection into an ``EnsembleFX`` dataset.

    Layout: ``source/{run}/{member}/{station}.{smet|pro|nc}``. Runs and members are
    derived from folder names; station IDs from filenames. Only requested runs,
    members, and filename bases are read to keep I/O minimal.

    Time handling:
      - ``run_timezone`` (default ``"UTC"``) is stored on the ``run`` coord and used
        to interpret run labels that parse as datetimes (e.g., ``2024-01-17T09Z``).
      - ``run_start`` is always present; if parsing fails it is NaT with a timezone
        attribute preserved.
      - ``leadtime`` (float hours) is computed as valid time minus run start using
        timezone-aware math; NaN if run start is missing.

    Concatenation strategy:
      - Members are concatenated along ``realization`` with ``join='outer'``.
      - Runs are concatenated along ``run`` with ``join='outer'``.
      - ``squeeze_run`` drops the run dimension when only one run is present.

    Parameters
    ----------
    source : str or Path
        Root directory of the forecast collection.
    run, realization, fname_base : str or iterable of str, optional
        Filters to limit which runs, members, or filename bases are read.
    filename_pattern : str or sequence of str, optional
        fnmatch patterns applied to filename bases (e.g., ``"VIR*"``).
    run_timezone : str | int | float | None, default "UTC"
        Timezone used to localize/convert run labels; stored in ``run.attrs["timezone"]``.
    allowed_exts : sequence of str, default (".smet", ".pro", ".nc")
        File extensions that will be read.
    squeeze_run : bool, default True
        If True, drop the run dimension when only one run is present.
    **read_kwargs :
        Forwarded to ``xsnow.read`` for parsing options (lazy, selectors, etc.).

    Returns
    -------
    EnsembleFX or None
        Decorated dataset when data were found; otherwise ``None``.
    """
    src = Path(source)
    if not src.exists():
        raise FileNotFoundError(f"Source path does not exist: {source}")

    run_sel = _normalize_selection(run)
    realization_sel = _normalize_selection(realization)
    fname_sel = _normalize_selection(fname_base)
    patterns = _normalize_patterns(filename_pattern)
    run_tz_str = _normalize_tz_attr_value(run_timezone) if run_timezone is not None else None

    run_datasets: List[xr.Dataset] = []

    for run_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        run_label = run_dir.name
        if run_sel and run_label not in run_sel:
            continue

        member_datasets: List[xr.Dataset] = []
        run_ts = _parse_run_timestamp(run_label, run_timezone)

        for member_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
            member_label = member_dir.name
            if realization_sel and member_label not in realization_sel:
                continue

            files = _collect_files(member_dir, patterns, allowed_exts, fname_sel)
            if not files:
                continue

            ds = xsnow_read(files, recursive=False, **read_kwargs)
            if ds is None or ds.data is None:
                continue

            xr_ds = _ensure_realization_dim(ds.data, member_label)
            lead = _compute_leadtime_hours(xr_ds, run_ts, run_tz_str)

            xr_ds = xr_ds.expand_dims(run=[run_label])
            xr_ds["run"].attrs["description"] = "Model run identifier"
            if run_tz_str:
                xr_ds["run"].attrs["timezone"] = run_tz_str

            # Ensure every run has a run_start coordinate for concat compatibility.
            if run_ts is None:
                run_start_val = np.array([np.datetime64("NaT")], dtype="datetime64[ns]")
                run_start_tz = run_tz_str
            else:
                run_start_val = np.array([run_ts.tz_localize(None).to_datetime64()], dtype="datetime64[ns]")
                run_start_tz = run_tz_str or str(run_ts.tzinfo)

            run_start = xr.DataArray(
                run_start_val,
                dims=("run",),
                coords={"run": [run_label]},
                name="run_start",
                attrs={
                    "timezone": run_start_tz,
                    "description": "Run start time localized to run timezone (tz info stored in attrs).",
                },
            )
            xr_ds = xr_ds.assign_coords(run_start=run_start)

            if lead is not None:
                xr_ds = xr_ds.assign_coords(leadtime=lead.expand_dims(run=[run_label]))

            member_datasets.append(xr_ds)

        if member_datasets:
            run_ds = _xs_concat(
                member_datasets,
                dim="realization",
                combine_attrs="drop_conflicts",
                join="outer",
            )
            run_datasets.append(run_ds)

    if not run_datasets:
        return None

    combined = _xs_concat(
        run_datasets,
        dim="run",
        combine_attrs="drop_conflicts",
        join="outer",
    )

    if squeeze_run and combined.sizes.get("run", 0) == 1:
        combined = combined.squeeze("run")

    return EnsembleFX(xsnowDataset(combined))
