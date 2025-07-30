import pandas as pd
import xarray as xr
from datetime import datetime
from typing import Union, Sequence, Optional, Any
from xsnow import xsnowDataset


class Selector:
    def __init__(self, dim: str, sel: Any):
        self.dim = dim
        self.sel = sel

    def apply(self, ds: xr.Dataset) -> xr.Dataset:
        """By default, route through .sel if it looks label‑based, else .isel"""
        s = self.sel
        if s is None:
            return ds

        # Boolean mask or DataArray: always label‐based
        if isinstance(s, (xr.DataArray, xr.Dataset)) or (
            isinstance(s, Sequence) and hasattr(s, "dtype")
            and getattr(s, "dtype", None) == bool
        ):
            return ds.sel({self.dim: s})

        # If slice of strings or datetimes => sel
        if isinstance(s, slice) and (
            isinstance(s.start, (str, pd.Timestamp, type(None))) or
            isinstance(s.stop,  (str, pd.Timestamp, type(None)))
        ):
            return ds.sel({self.dim: s})

        # If datetime, string, list of those => sel
        if isinstance(s, (str, pd.Timestamp)) or (
            isinstance(s, Sequence) and s and
            all(isinstance(x, (str, pd.Timestamp)) for x in s)
        ):
            return ds.sel({self.dim: s})

        # Otherwise assume integer/indexing => isel
        return ds.isel({self.dim: s})
    

class TimeSelector(Selector):
    def __init__(self, sel: Union[str, pd.Timestamp, slice, Sequence[pd.Timestamp]]):
        super().__init__("time", sel)


class LocationSelector(Selector):
    def __init__(self, sel: Union[str, Sequence[str], int, slice]):
        super().__init__("location", sel)


class SlopeSelector(Selector):
    def __init__(self, sel: Union[int, Sequence[int], slice, xr.DataArray]):
        super().__init__("slope", sel)


class RealizationSelector(Selector):
    def __init__(self, sel: Union[int, Sequence[int], slice]):
        super().__init__("realization", sel)


def read_pro(
    filepath: str,
    datetime_start: Optional[datetime] = None,
    remove_soil: bool = True,
    include_smet: bool = True) -> xsnowDataset:
    """
    Reads a SNOWPACK PRO file into an xsnowDataset.

    Parameters:
    -----------
        filepath : str
            Path to the PRO file.

        datetime_start : datetime.datetime, optional
            The PRO file will be read starting with this datetime. Data prior to 
            this timestamp will be ignored. This feature can optimize efficiency
            when reading only recent timestamps in an operational day-to-day
            setting by reading the PRO file backwards.

        remove_soil : bool
            If True, removes soil layers from the profiles.

        include_smet : bool
            If True, also try reading the corresponding SMET file by swapping
            file extensions from .pro to .smet.
    """
    pass


def read_smet(
    filepath: str) -> xsnowDataset:
    """
    Reads a SNOWPACK SMET file into an xsnowDataset.

    Parameters:
    -----------
        filepath (str):
            Path to the SMET file.
    """
    pass


def read(
    filepaths: Optional[Union[str, Sequence[str]]] = None,
    *,
    directories: Optional[Union[str, Sequence[str]]] = None,
    time: Optional[TimeSelector] = None,
    location: Optional[LocationSelector] = None,
    slope: Optional[SlopeSelector] = None,
    realization: Optional[RealizationSelector] = None,
    **parser_kwargs) -> xr.Dataset:
    """
    Reads multiple snow cover simulation outputs into a single xarray.Dataset.

    This function iterates through all specified `filepaths` and all supported
    files found in each `directories` path, auto-detects the appropriate parser
    for each format, and merges the results into one consolidated dataset.
    The output can be subsetted along time, location, slope configuration, or
    realization. If multiple profiles exist at the same (location, time, slope), they 
    are stored as separate realizations.

    Parameters
    ----------
    filepaths : str or list of str, optional
        Path or list of paths to individual simulation files to read.
    directories : str or list of str, optional
        Path or list of directories whose supported files will all be read.
    time : TimeSelector, optional
        Selector for the time dimension (e.g., datetime, slice, list of
        datetimes, or pandas.DatetimeIndex).
    location : LocationSelector, optional
        Selector for the location dimension (e.g., single ID string, slice,
        or list of strings).
    slope : SlopeSelector, optional
        Selector for the slope dimension (e.g., integer index, slice,
        list of indices, boolean mask or DataArray).
    realization : RealizationSelector, optional
        Selector for the realization dimension (e.g., integer index, slice,
        or list of integers).
    **parser_kwargs
        Additional keyword arguments forwarded to the individual parsers.

    Returns
    -------
    xsnowDataset
        Consolidated and optionally subsetted dataset containing snow cover
        simulations, with dimensions for location, time, slope, realization, and layer.

    Examples
    --------
    >>> ds = read("profile.pro")

    >>> ds = read(
    ...     directories=["./snowpack_A/", "./snowpack_B/"],
    ...     time=slice("2024-01-01", "2024-02-01")
    ... )
    """
    pass
