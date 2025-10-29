# selectors.py
"""
This module provides a structured, object-oriented interface for selecting
and subsetting data from an xsnowDataset object.

The Selector classes are designed to provide a more explicit and reusable
way to define complex queries compared to using dictionaries directly with
xarray's .sel() or .isel() methods.

The base `Selector` class intelligently routes the selection to either `.sel()`
(for label-based indexing) or `.isel()` (for integer-based indexing) based
on the type of the input.

---
Usage Example
---

from xsnow.dataset import xsnowDataset
from selectors import TimeSelector, LocationSelector

# 1. Load a dataset
ds = xsnowDataset('path/to/your/data')

# 2. Create selector objects
# By station ID
location_selector_id = LocationSelector(sel='station_A')

# By geographic bounding box, specifying the CRS
location_selector_bbox = LocationSelector(
    lat_range=(45.0, 46.0),
    lon_range=(10.0, 11.0),
    crs='EPSG:4326'  # WGS 84
)

# 3. Apply the selectors to the dataset
subset_by_id = location_selector_id.apply(ds.data)
subset_by_bbox = location_selector_bbox.apply(ds.data)

print(subset_by_id)
print(subset_by_bbox)
"""

from typing import Any, Union, Sequence, Optional, Tuple, Mapping
import pandas as pd
import xarray as xr
import warnings
import numpy as np

IndexLike = Union[int, slice, Sequence[int]]

TimeIndexScalar = Union[str, pd.Timestamp, np.datetime64]
TimeIndexSeq    = Sequence[TimeIndexScalar]
TimeSlice       = slice
TimeLike = Union["TimeSelector", TimeIndexScalar, TimeSlice, TimeIndexSeq]

LocationIndexScalar = str
LocationIndexSeq    = Sequence[LocationIndexScalar]
# - NOTE: bbox args (lat_range/lon_range/crs) cannot be expressed
#   positionally here, so callers that want bbox still need to pass
#   an actual LocationSelector(...)
LocationLike = Union["LocationSelector", LocationIndexScalar, LocationIndexSeq]

SlopeIndexLike = Union[int, slice, Sequence[int], xr.DataArray]  # xr <-> boolean mask
SlopeQueryMapping = Mapping[str, Any]  # e.g., {"inclination": 30, "azimuth": 180}
SlopeLike = Union["SlopeSelector", SlopeIndexLike, SlopeQueryMapping]

RealIndexLike = Union[int, str, slice, Sequence[int], Sequence[str]]
RealizationLike = Union["RealizationSelector", RealIndexLike]


class Selector:
    """
    A base class for creating dimension-specific data selectors.

    This class provides the core logic to intelligently apply a selection
    to an xarray.Dataset, automatically choosing between label-based (`.sel`)
    and integer-based (`.isel`) indexing.
    """
    def __init__(self, dim: str, sel: Any):
        self.dim = dim
        self.sel = sel

    def apply(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Applies the selector to a given xarray.Dataset.

        By default, it routes the selection through .sel if the input looks
        like a label (e.g., a string, timestamp, or slice of those), and
        through .isel otherwise (e.g., an integer or slice of integers).

        Args:
            ds (xr.Dataset): The dataset to apply the selection to.

        Returns:
            xr.Dataset: The subsetted dataset.
        """
        s = self.sel
        if s is None:
            return ds

        # Convert numpy datetime64 to pandas Timestamp
        if isinstance(s, np.datetime64):
            s = pd.Timestamp(s)
        elif isinstance(s, slice):
            start = s.start
            stop = s.stop
            step = s.step
            
            # Convert numpy datetime64 to pandas Timestamp
            if start is not None and isinstance(start, np.datetime64):
                start = pd.Timestamp(start)
            if stop is not None and isinstance(stop, np.datetime64):
                stop = pd.Timestamp(stop)
            
            # Reconstruct slice with converted values
            s = slice(start, stop, step)

        # Boolean mask or DataArray: always label-based
        if isinstance(s, (xr.DataArray, xr.Dataset)) or (
            isinstance(s, Sequence) and hasattr(s, "dtype")
            and np.issubdtype(getattr(s, "dtype", None), np.bool_)
        ):
            return ds.sel({self.dim: s})

        # If slice with any datetime-like values => sel
        if isinstance(s, slice):
            if (s.start is not None and isinstance(s.start, (str, pd.Timestamp))) or \
            (s.stop is not None and isinstance(s.stop, (str, pd.Timestamp))):
                return ds.sel({self.dim: s})

        # If datetime, string, or a list of those => sel
        if isinstance(s, (str, pd.Timestamp)) or (
            isinstance(s, Sequence) and s and
            all(isinstance(x, (str, pd.Timestamp)) for x in s)
        ):
            return ds.sel({self.dim: s})
            
        # Otherwise, assume integer/indexing => isel
        try:
            return ds.isel({self.dim: s})
        except IndexError:
            return ds.isel({self.dim: slice(0, 0)})


class TimeSelector(Selector):
    """A selector specifically for the 'time' dimension."""
    def __init__(self, sel: Union[TimeIndexScalar, TimeSlice, TimeIndexSeq]):
        super().__init__("time", sel)


class LocationSelector(Selector):
    """
    A selector for the 'location' dimension that can filter by station ID
    or by a geographic bounding box.
    """
    def __init__(
        self,
        sel: Optional[Union[str, Sequence[str], int, slice]] = None,
        lat_range: Optional[Tuple[float, float]] = None,
        lon_range: Optional[Tuple[float, float]] = None,
        crs: Optional[str] = "EPSG:4326"
    ):
        super().__init__("location", sel)
        self.lat_range = lat_range
        self.lon_range = lon_range
        self.crs = crs

    def apply(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Apply location selection - NOW MUCH SIMPLER!
        """
        # Geographic bounding box selection
        if self.lat_range is not None or self.lon_range is not None:
            return self._apply_bbox(ds)
        
        # ✨ STRING SELECTION JUST WORKS!
        # Base class handles str, list[str], int, slice automatically
        return super().apply(ds)
    
    def _apply_bbox(self, ds: xr.Dataset) -> xr.Dataset:
        """Apply geographic bounding box filter."""
        dataset_crs = ds.attrs.get('crs')
        if dataset_crs and dataset_crs != self.crs:
            warnings.warn(
                f"Query CRS ('{self.crs}') does not match dataset CRS ('{dataset_crs}'). "
                "Results may be inaccurate."
            )

        mask = True
        if self.lat_range:
            if 'latitude' not in ds.coords:
                raise ValueError("Dataset must have 'latitude' coordinates")
            mask &= (ds.latitude >= self.lat_range[0]) & (ds.latitude <= self.lat_range[1])
        
        if self.lon_range:
            if 'longitude' not in ds.coords:
                raise ValueError("Dataset must have 'longitude' coordinates")
            mask &= (ds.longitude >= self.lon_range[0]) & (ds.longitude <= self.lon_range[1])

        return ds.sel(location=mask)


class SlopeSelector(Selector):
    """
    Selector for the 'slope' dimension.

    Supports:
    - int / list[int] / slice / xr.DataArray (existing positional/boolean logic)
    - dict query like {'inclination': 30, 'azimuth': 180} to match slope
      metadata coords/vars.
    """
    def __init__(self, sel: Union[SlopeIndexLike, SlopeQueryMapping]):
        super().__init__("slope", sel)

    def apply(self, ds: xr.Dataset) -> xr.Dataset:
        s = self.sel

        # Case 1: dict-style metadata query for slope properties
        if isinstance(s, dict):
            # Build a cumulative mask over the slope dimension
            # Start with all True
            if "slope" not in ds.dims:
                raise ValueError("Dataset has no 'slope' dimension")

            mask = xr.DataArray(
                np.ones(ds.sizes["slope"], dtype=bool),
                dims=("slope",),
            )

            for key, expected in s.items():
                # Look up candidate array (coord or data var)
                if key in ds.coords:
                    arr = ds.coords[key]
                elif key in ds:
                    arr = ds[key]
                else:
                    raise ValueError(
                        f"SlopeSelector: cannot filter by '{key}'; "
                        f"not found in dataset coords or variables."
                    )

                if "slope" not in arr.dims:
                    raise ValueError(
                        f"SlopeSelector: field '{key}' is not defined per-slope "
                        f"(dims={arr.dims})"
                    )

                cond = (arr == expected)
                # align to slope dim just in case of broadcasting
                cond = cond.broadcast_like(mask)
                mask = mask & cond

            # Check if anything matched
            if not bool(mask.any()):
                # No slope satisfies all criteria. HARD FAIL.
                criteria = ", ".join(f"{k} == {v}" for k, v in s.items())
                raise ValueError(
                    f"No slope matches criteria: {criteria}"
                )

            # Convert boolean mask -> integer indices
            # (This avoids the ambiguity of .sel on integer coords.)
            matching_idx = np.nonzero(mask.values)[0]

            return ds.isel(slope=matching_idx)

        # Case 2: everything else (ints, slices, xr.DataArray, etc.)
        return super().apply(ds)


class RealizationSelector(Selector):
    """A selector specifically for the 'realization' dimension."""
    def __init__(self, sel: RealIndexLike):
        super().__init__("realization", sel)