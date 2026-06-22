"""
Custom xarray accessors for human-readable labels, units, and timezones.

This module provides custom accessors including:

1. ``.labels`` accessor
   - Generic mechanism to attach and retrieve human-readable labels for any
     DataArray that uses integer codes internally.
   - Requires that the DataArray has an attribute ``.attrs["mapping"]``
     containing an integer-to-string dictionary.
   - Example:
       >>> import xarray as xr
       >>> da = xr.DataArray([0], dims="realization", name="realization")
       >>> da.attrs["mapping"] = {0: "manual profile"}
       >>> da.labels.get().item()
       'manual profile'

2. ``.icssg`` accessor
   - Specialized accessor for snow grain type codes according to the
     *International Classification for Seasonal Snow on the Ground* (ICSSG).
   - Operates on DataArrays that store three-digit SLF-style codes
     ("Swiss Code F1F2F3") in their ``.attrs["units"]`` metadata.
   - Provides convenience methods:
       * ``.icssg.primary()`` → returns primary ICSSG label
     * ``.icssg.secondary()`` → returns secondary ICSSG label

3. ``.units`` accessor
   - Converts DataArray values while preserving coordinates and metadata.
   - Exposes current and preferred plotting units through ``.current`` and
     ``.plot``.

4. ``.tz`` accessor
   - Provides timezone-aware operations for timezone-naive time coordinates.

Together, these accessors allow xsnow datasets to remain efficient by storing
compact integer codes internally, while exposing user-friendly string labels
on demand for analysis, visualization, or export.
"""

from __future__ import annotations
import re
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from datetime import timezone, timedelta, tzinfo
from typing import Optional, Union, Literal
from zoneinfo import ZoneInfo
from xsnow.graincode_converter import slf_graincode_to_ICSSG_primary, slf_graincode_to_ICSSG_secondary
from xsnow.units import UnitConversionError, UnitLike, convert_units, normalize_unit


TZLike = Union[str, int, float, tzinfo]
_UTC_OFFSET_RE = re.compile(r"^UTC([+-])(\d{1,2})(?::?(\d{2}))?$")  # UTC+01:00 or UTC+1 or UTC+0100


@xr.register_dataarray_accessor("units")
class UnitsAccessor:
    """Inspect and convert units attached to an xarray DataArray."""

    def __init__(self, xarray_obj: xr.DataArray):
        self._obj = xarray_obj

    @property
    def current(self) -> Optional[str]:
        """Return the unit stored in ``attrs["units"]``, if present."""
        unit = self._obj.attrs.get("units")
        return str(unit) if unit is not None else None

    @property
    def plot(self) -> Optional[UnitLike]:
        """Return the variable's preferred plotting unit from the registry."""
        if self._obj.name is None:
            return None

        from xsnow.registry import registry

        return registry.get(str(self._obj.name), {}).get("plot_units")

    def to(self, unit: str) -> xr.DataArray:
        """
        Return a copy converted to an explicit target unit.

        The source unit is read from ``attrs["units"]``. Coordinates, name,
        and other attributes are preserved, while ``attrs["units"]`` is
        updated to the normalized target unit.

        Parameters
        ----------
        unit : str
            Target unit expression supported by
            :func:`xsnow.units.convert_units`.

        Returns
        -------
        xarray.DataArray
            A converted copy of the DataArray.

        Raises
        ------
        UnitConversionError
            If source-unit metadata is missing or the conversion is unknown.
        """
        source_unit = self.current
        if source_unit is None:
            raise UnitConversionError(
                f"Cannot convert DataArray {self._obj.name!r}: "
                "attrs['units'] is missing."
            )

        converted_data = convert_units(self._obj.data, source_unit, unit)
        out = self._obj.copy(data=converted_data)
        out.attrs = dict(self._obj.attrs)
        out.attrs["units"] = normalize_unit(unit)
        return out

    def to_plot_unit(self) -> xr.DataArray:
        """
        Return a copy converted to the registered preferred plotting unit.

        If the variable has no single plotting unit in the registry, a shallow
        copy is returned unchanged. This includes variables with compound
        values whose registry entry contains multiple plotting units.
        """
        plot_unit = self.plot
        if not isinstance(plot_unit, str):
            return self._obj.copy(deep=False)
        return self.to(plot_unit)


@xr.register_dataarray_accessor("labels")
class LabelsAccessor:
    """
    A custom xarray accessor to retrieve the human-readable names for variables
    that use integer codes for efficiency.

    This accessor will work on any coordinate or data variable that owns an attribute called
    `"mapping"`, which contains the integer-to-string dictionary.
    """
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._mapping = xarray_obj.attrs.get("mapping", None)

    def get(self, as_dict=False) -> xr.DataArray | dict:
        """
        Return a labeled version of the DataArray using the integer-to-label mapping stored
        in `.attrs["mapping"]`.

        Parameters:
        ----------
            as_dict (bool): If True, return the raw mapping dict (or {} if none exists).

        Returns:
        --------
            DataArray of string labels with dataset dimensions preserved or dict with the 
            integer-to-labels mapping (empty dict if no mapping exists).

        Examples:
        ---------
            >>> from xsnow.sample_data import single_profile
            >>> ds = single_profile()
            >>> ds["realization"]
            >>> ds["realization"].attrs["mapping"] = {0: "manual profile"}
            >>> ds["realization"].labels.get().item()
            'manual profile'

            >>> ds["location"].labels.get().item()
            'Kasererwinkl C6'

        Raises:
        -------
            AttributeError
                If no 'mapping' attribute is present **and** `as_dict=False`.
        """
        if self._mapping is None:
            if as_dict:
                return {}
            raise AttributeError("No 'mapping' attribute found on this DataArray.")

        if as_dict:
            return self._mapping

        def _lookup(v):
            try:
                if np.isnan(v):
                    return np.nan
            except Exception:
                pass
            if isinstance(v, np.integer):
                v = int(v)
            return self._mapping.get(v, np.nan)

        return xr.apply_ufunc(
            np.vectorize(_lookup, otypes=[object]),
            self._obj,
            dask="allowed",
            output_dtypes=[object]
        )



@xr.register_dataarray_accessor("icssg")
class ICSSGAccessor:
    """
    An accessor to convert integer graincodes to human-readable grain type labels following the
    convention by the *International Classification for Seasonal Snow on the Ground* (ICSSG).

    Expected: DataArray of int/float codes; NaN allowed.
    """

    def __init__(self, xarray_obj: xr.DataArray):
        self._obj = xarray_obj
        self._unit = self._obj.attrs.get("units", None)

    def primary(self) -> xr.DataArray:
        """
        Convert an integer grain code to the ICSSG label of the **primary** grain type.

        Returns:
        --------
            DataArray of string labels broadcast over input dimensions.

        Examples:
        ---------
            >>> from xsnow.sample_data import single_profile
            >>> ds = single_profile()
            >>> ds["grain_type"].squeeze().values
            array([772.,   3.,   3., 772.,   4., 772.,   4.,   4.,   6.,   2.],
                dtype=float32)
            >>> ds["primary"] = ds["grain_type"].icssg.primary()
            >>> ds["primary"].squeeze().values
            array(['MFcr', 'RG', 'RG', 'MFcr', 'FC', 'MFcr', 'FC', 'FC', 'SH', 'DF'],
                dtype=object)

        Raises:
        -------
            AttributeError
                If the attribute in 'units' is not a known or implemented grain code
        """
        if self._unit == 'Swiss Code F1F2F3':
            return slf_graincode_to_ICSSG_primary(self._obj)
        else:
            raise AttributeError("Unknown grain code unit in `.attrs['units']`. "
                                 "Needs to be like `Swiss Code F1F2F3`."
                                 )

    def secondary(self) -> xr.DataArray:
        """
        Convert an integer grain code to the ICSSG label of the **secondary** grain type.

        Returns:
        --------
            DataArray of string labels broadcast over input dimensions.

        See Also:
        ---------
        primary : Convert to primary grain types (see Example section there).

        Raises:
        -------
            AttributeError
                If the attribute in 'units' is not a known or implemented grain code
        """
        if self._unit == 'Swiss Code F1F2F3':
            return slf_graincode_to_ICSSG_secondary(self._obj)
        else:
            raise AttributeError("Unknown grain code unit in `.attrs['units']`. "
                                 "Needs to be like `Swiss Code F1F2F3`."
                                 )




@xr.register_dataarray_accessor("tz")
class TimezoneAccessor:
    """
    Timezone utilities for a timezone-naive time coordinate.

    Expected target: a 1D datetime64[ns] DataArray (usually the 'time' coord).
    The intended timezone is stored as a string in the DataArray's `.attrs['timezone']` (e.g. 'UTC').

    Core idea:
      - Keep on-disk times tz-naive for portability.
      - Record the associated tz as attribute.
      - Provide helpers to get a tz-aware index for selection/plotting.
    """

    def __init__(self, xarray_obj: xr.DataArray):
        self._obj = xarray_obj
        if not np.issubdtype(self._obj.dtype, np.datetime64):
            raise AttributeError("`.tz` accessor only applies to datetime64 arrays.")

    # ----- metadata helpers -----
    def get_attr(self, default: Optional[str] = None) -> Optional[str]:
        """Return timezone string from attrs (or `default` if not present)."""
        tz = self._obj.attrs.get("timezone", default)
        return tz

    def set_attr(self, tz: Union[str, int, float, ZoneInfo]) -> None:
        """Set/overwrite the timezone attribute (validation included)."""
        self._obj.attrs["timezone"] = _normalize_tz_attr_value(tz)

    # ----- main conversions -----
    def localize(
        self,
        tz: Optional[TZLike] = None,
        *,
        ambiguous: Literal["raise", "infer", "NaT"] = "raise",
        nonexistent: Literal["raise", "shift_forward", "shift_backward", "NaT"] = "raise",
        as_series: bool = False,
    ) -> pd.DatetimeIndex | pd.Series:
        """
        Convert the tz-naive datetime64 array to a tz-aware DatetimeIndex (or Series),
        using the tz provided or the value in `.attrs['timezone']`.

        Returns a **pandas** object because xarray does not store tz-aware dtypes
        in coords/data yet in a portable way.

        Parameters
        ----------
        tz : str, optional
            Timezone like 'UTC' or 'Europe/Vienna'. Defaults to the stored 'timezone' attribute.
        ambiguous, nonexistent : handling for DST edge cases (forwarded to pandas).
        as_series : bool
            If True, return a pandas.Series aligned to the same, unchanged index labels as the
            input DataArray.

        Examples
        --------
        >>> time_tz = ds['time'].tz.localize()  # use stored attr
        >>> time_tz_utc = time_tz.tz_convert("UTC")
        """
        attr_tz = self.get_attr()
        tzinfo = _coerce_tzinfo(tz if tz is not None else attr_tz)
        if tzinfo is None:
            raise ValueError("No timezone provided and no '.attrs[\"timezone\"]' set.")

        if tz is not None and attr_tz is not None:
            stored = _coerce_tzinfo(attr_tz)
            if str(tzinfo) != str(stored):
                warnings.warn(
                    f"Requested timezone '{tzinfo}' differs from stored attrs['timezone'] '{stored}'. This changes important time information!",
                    UserWarning,
                )

        values = self._obj.values
        idx = _as_datetime_index(values).tz_localize(tzinfo, ambiguous=ambiguous, nonexistent=nonexistent)
        if as_series:
            is_1d = (self._obj.ndim == 1)
            ser_index = self._obj.to_index() if is_1d else pd.RangeIndex(len(idx))
            return pd.Series(idx, index=ser_index, name=self._obj.name or "time")
        return idx


    def to_datetimeindex_in(
        self,
        target_tz: TZLike,
        *,
        as_series: bool = False,
        ambiguous: Literal["raise", "infer", "NaT"] = "raise",
        nonexistent: Literal["raise", "shift_forward", "shift_backward", "NaT"] = "raise",
    ) -> pd.DatetimeIndex | pd.Series:
        """
        Localize to the stored timezone (see `.get_attr()`), then convert to `target_tz`
        and return a pandas DatetimeIndex (or Series).

        Parameters
        ----------
        target_tz : str, default "UTC"
            Timezone to convert the localized index into.
        as_series : bool, default False
            If True, return a pandas.Series aligned to the input coordinate.
        ambiguous, nonexistent :
            Passed to pandas tz_localize when localizing the naive values.

        Examples
        --------
        >>> idx_vienna = ds['time'].tz.to_datetimeindex_in("Europe/Vienna")
        >>> idx_utc_series = ds['time'].tz.to_datetimeindex_in("UTC", as_series=True)
        """
        target_tzinfo = _coerce_tzinfo(target_tz)
        aware = self.localize(ambiguous=ambiguous, nonexistent=nonexistent, as_series=as_series)
        if as_series:
            return aware.dt.tz_convert(target_tzinfo)
        return aware.tz_convert(target_tzinfo)


    def to_naive_in(
        self,
        target_tz: TZLike,
        *,
        as_series: bool=False,
        ambiguous: Literal["raise", "infer", "NaT"] = "raise",
        nonexistent: Literal["raise", "shift_forward", "shift_backward", "NaT"] = "raise",
    ) -> xr.DataArray | pd.Series | np.ndarray:
        """
        Localize to the stored timezone, then convert to `target_tz`,
        drop tz info (make naive), and return as a DataArray (default) or Series.

        This is useful when you want to *store* or *compare* times in a single
        canonical tz-naive representation.

        Parameters
        ----------
        target_tz : str, default "UTC"
            Timezone to convert into before dropping tz info.
        as_series : bool, default False
            Instead of an xarray.DataArray, return a pandas.Series?
        ambiguous, nonexistent :
            Passed to pandas tz_localize when localizing the naive values.

        Examples
        --------
        >>> # UTC-naive view for storage/comparison
        >>> t_utc_naive = ds['time'].tz.to_naive_in("UTC")
        >>> # Vienna-naive view
        >>> t_vie_naive_series = ds['time'].tz.to_naive_in("Europe/Vienna", return_type="series")
        """
        target_tzinfo = _coerce_tzinfo(target_tz)
        idx_aware = self.localize(ambiguous=ambiguous, nonexistent=nonexistent)
        idx_in_target = idx_aware.tz_convert(target_tzinfo)
        np_naive = idx_in_target.tz_localize(None).to_numpy(dtype="datetime64[ns]")
        tz_attr_str = _normalize_tz_attr_value(target_tz)

        var_name = f"time_{str(target_tzinfo).replace('/','_').lower()}_naive"
        if as_series:
            is_1d = (self._obj.ndim == 1)
            idx = self._obj.to_index() if is_1d else pd.RangeIndex(len(np_naive))
            return pd.Series(np_naive, index=idx, name=var_name)

        elif self._obj.ndim == 0:
            # Return a true 0-D DataArray (scalar)
            return xr.DataArray(
                np_naive[0],
                name=var_name,
                attrs={"timezone": tz_attr_str, "note": "dtype is tz-naive"},
            )
        else:
            return xr.DataArray(
                np_naive,
                dims=("time",) if self._obj.ndim == 1 else self._obj.dims,
                coords={k: v for k, v in self._obj.coords.items()},
                name=var_name,
                attrs={"timezone": tz_attr_str, "note": "dtype is tz-naive"},
            )


    # ----- selection helpers -----
    def between(
        self,
        start: Optional[str | pd.Timestamp] = None,
        end: Optional[str | pd.Timestamp] = None,
        tz: Optional[str] = None,
        *,
        as_series: bool = False,
    ) -> xr.DataArray | pd.Series:
        """
        Boolean mask selecting a time range expressed in `tz`, without mutating the stored
        (timezone-naive) coordinate.

        Parameters
        ----------
        start : str | pandas.Timestamp, optional
            Start datetime in `tz`. If tz-naive, it will be localized to the target tz.
            Defaults to the earliest timestamp.
        end : str | pandas.Timestamp, optional
            End datetime in `tz`. If tz-naive, it will be localized to the target tz.
            Defaults to the latest timestamp.
        tz : str, optional
            Timezone name (e.g. "UTC", "Europe/Vienna"). If `start`/`end` are tz-aware,
            their timezone takes priority. If omitted, uses `.attrs["timezone"]`.
        as_series : bool, default False
            If True, return a pandas.Series mask. Otherwise return an xarray.DataArray aligned
            to the 'time' dimension.

        Returns
        -------
        xarray.DataArray | pandas.Series
            Boolean mask aligned to the time coordinate.

        Examples
        --------
        >>> mask = ds['time'].tz.between("2025-01-01 06:00", "2025-01-01 12:00", tz="Europe/Vienna")
        >>> ds_sub = ds.sel(time=mask)
        """
        attr_tz = self.get_attr()
        if attr_tz is None:
            raise ValueError("No '.attrs[\"timezone\"]' set on self.")
        attr_tzinfo = _coerce_tzinfo(attr_tz)
        user_tzinfo = _coerce_tzinfo(tz) if tz is not None else None

        idx_naive = _as_datetime_index(self._obj.values)  # keep stored values as-naive

        # Empty guard
        if idx_naive.empty:
            if as_series:
                is_1d = (self._obj.ndim == 1)
                ser_index = self._obj.to_index() if is_1d else pd.RangeIndex(0)
                return pd.Series([], index=ser_index, dtype=bool, name=f"{self._obj.name or 'time'}_mask")
            return xr.DataArray(
                np.array([], dtype=bool),
                dims=("time",),
                coords={"time": self._obj},
                name=f"{self._obj.name or 'time'}_mask",
            )

        def _normalize_bound(ts_like, label: str) -> Optional[pd.Timestamp]:
            if ts_like is None:
                return None
            ts = pd.Timestamp(ts_like)
            if ts.tzinfo is not None:
                if user_tzinfo is not None and str(ts.tzinfo) != str(user_tzinfo):
                    raise ValueError(
                        f"{label} timezone ({ts.tzinfo}) does not match provided tz ({user_tzinfo})."
                    )
                return ts.tz_convert(attr_tzinfo).tz_localize(None)
            else:
                if user_tzinfo is None:
                    raise ValueError(
                        f"{label} is timezone-naive. Please provide `tz` to localize it."
                    )
                return (
                    ts.tz_localize(user_tzinfo)
                    .tz_convert(attr_tzinfo)
                    .tz_localize(None)
                )

        # Defaults to full span if start/end not given
        start_ts = _normalize_bound(start, "start") or idx_naive.min()
        end_ts   = _normalize_bound(end, "end") or idx_naive.max()

        # Normalize ordering
        if start_ts > end_ts:
            start_ts, end_ts = end_ts, start_ts

        mask = (idx_naive >= start_ts) & (idx_naive <= end_ts)

        if as_series:
            is_1d = (self._obj.ndim == 1)
            ser_index = self._obj.to_index() if is_1d else pd.RangeIndex(len(mask))
            return pd.Series(mask, index=ser_index, name=f"{self._obj.name or 'time'}_mask")
        return xr.DataArray(
            mask.astype(bool),
            dims=("time",),
            coords={"time": self._obj},
            name=f"{self._obj.name or 'time'}_mask",
        )

## --- private helper functions outside of TimezoneAccessor class ----- ##

def _int_offset_to_tzstring(offset_hours: Union[int, float]) -> str:
    sign = "+" if offset_hours >= 0 else "-"
    ah = abs(float(offset_hours))
    hours = int(ah)
    minutes = int(round((ah - hours) * 60))
    if hours==0 and minutes==0:
        return "UTC"
    return f"UTC{sign}{hours:02d}:{minutes:02d}"

def _parse_tzinfo_from_string(tz: str):
    """
    Return tzinfo from a tz string. Supports:
      - IANA zone names (e.g., 'Europe/Vienna') via ZoneInfo
      - Fixed offsets like 'UTC+01:00', 'UTC-7', 'UTC+0530' via datetime.timezone
    """
    m = _UTC_OFFSET_RE.match(tz)
    if m:
        sign, hh, mm = m.groups()
        hours = int(hh)
        minutes = int(mm or "00")
        delta = timedelta(hours=hours, minutes=minutes)
        if sign == "-":
            delta = -delta
        return timezone(delta)

    # fall back to IANA zone
    if ZoneInfo is None:
        raise ValueError(
            f"Unknown timezone: {tz!r} and no IANA database available. "
            "Use an offset like 'UTC+01:00' or install tzdata."
        )
    try:
        return ZoneInfo(tz)
    except Exception as e:
        raise ValueError(f"Unknown timezone: {tz!r}") from e

def _coerce_tzinfo(tz: TZLike):
    """
    Accept tz as str/int/float/tzinfo and return a tzinfo.
    """
    if hasattr(tz, "utcoffset"):  # tzinfo-like
        return tz  # type: ignore[return-value]
    if isinstance(tz, (int, float)):
        return timezone(timedelta(hours=float(tz)))
    if isinstance(tz, str):
        return _parse_tzinfo_from_string(tz)
    raise ValueError("timezone must be str/int/float/tzinfo, e.g. 'Europe/Vienna', 'UTC+01:00', or 1")

def _normalize_tz_attr_value(tz: Union[int, float, ZoneInfo, str]) -> str:
    """
    Return canonical string to store in attrs['timezone'].
    - int/float  -> 'UTC±HH:MM'
    - str offset -> normalized 'UTC±HH:MM'
    - str IANA   -> validate and keep as-is
    - ZoneInfo   -> extract IANA string name
    """
    if isinstance(tz, (int, float)):
        return _int_offset_to_tzstring(tz)

    if isinstance(tz, ZoneInfo):
        _ = _parse_tzinfo_from_string(tz.key)  # raises on invalid
        return tz.key  # type: ignore[attr-defined]

    if isinstance(tz, str):
        # If it's already an offset-like string, normalize its formatting
        m = _UTC_OFFSET_RE.match(tz)
        if m:
            sign, hh, mm = m.groups()
            return f"UTC{sign}{int(hh):02d}:{int(mm or '0'):02d}"
        # validate IANA name
        _ = _parse_tzinfo_from_string(tz)  # raises on invalid
        return tz
    raise ValueError(
            "Pass timezone as a string (e.g., 'Europe/Vienna' or 'UTC+01:00') "
            "or numeric offset (e.g., -7)."
        )

def _as_datetime_index(values) -> pd.DatetimeIndex:
    """
    Return a pandas.DatetimeIndex for either scalar or array-like datetime64 values.
    """
    if np.ndim(values) == 0:
        return pd.DatetimeIndex([values])
    return pd.DatetimeIndex(np.asarray(values))
