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

from typing import Any, Union, Sequence, Optional, Tuple
import pandas as pd
import xarray as xr
import warnings
import numpy as np

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

        # Boolean mask or DataArray: always label-based
        if isinstance(s, (xr.DataArray, xr.Dataset)) or (
            isinstance(s, Sequence) and hasattr(s, "dtype")
            and np.issubdtype(getattr(s, "dtype", None), np.bool_)
        ):
            return ds.sel({self.dim: s})

        # If slice of strings or datetimes => sel
        if isinstance(s, slice) and (
            isinstance(s.start, (str, pd.Timestamp)) or
            isinstance(s.stop,  (str, pd.Timestamp))
        ):
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
            # If the index is out of bounds, return an empty dataset
            # by selecting a zero-length slice of the dimension.
            return ds.isel({self.dim: slice(0, 0)})


class TimeSelector(Selector):
    """A selector specifically for the 'time' dimension."""
    def __init__(self, sel: Union[str, pd.Timestamp, slice, Sequence[pd.Timestamp]]):
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
        Applies the location selection.

        If lat_range or lon_range are provided, it performs a geographic
        query. Otherwise, it falls back to the default ID-based selection.
        """
        if self.lat_range is None and self.lon_range is None:
            # If no geographic range is given, use the default behavior
            return super().apply(ds)

        # Check for CRS consistency
        dataset_crs = ds.attrs.get('crs')
        if dataset_crs and dataset_crs != self.crs:
            warnings.warn(
                f"Query CRS ('{self.crs}') does not match dataset CRS ('{dataset_crs}'). "
                "Results may be inaccurate. Consider reprojecting your data."
            )

        # Build a boolean mask for the geographic query
        mask = True
        if self.lat_range:
            if 'latitude' not in ds.coords:
                raise ValueError("Dataset must have 'latitude' coordinates for geographic selection.")
            mask &= (ds.latitude >= self.lat_range[0]) & (ds.latitude <= self.lat_range[1])
        
        if self.lon_range:
            if 'longitude' not in ds.coords:
                raise ValueError("Dataset must have 'longitude' coordinates for geographic selection.")
            mask &= (ds.longitude >= self.lon_range[0]) & (ds.longitude <= self.lon_range[1])

        return ds.sel(location=mask)


class SlopeSelector(Selector):
    """A selector specifically for the 'slope' dimension."""
    def __init__(self, sel: Union[int, Sequence[int], slice, xr.DataArray]):
        super().__init__("slope", sel)


class RealizationSelector(Selector):
    """A selector specifically for the 'realization' dimension."""
    def __init__(self, sel: Union[int, Sequence[int], slice]):
        super().__init__("realization", sel)