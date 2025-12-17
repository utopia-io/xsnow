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
      metadata coords/vars with optional tolerance.
    """
    def __init__(
        self, 
        sel: Union[SlopeIndexLike, SlopeQueryMapping],
        tolerance: Optional[float] = None
    ):
        """
        Parameters
        ----------
        sel : int, slice, list, DataArray, or dict
            Selection criteria for slopes
        tolerance : float, optional
            When using dict-style queries with numerical values, allow matches
            within this tolerance. Useful for floating-point comparisons.
            If None, uses exact equality for integers and small default tolerance
            (1e-6) for floats.
        """
        super().__init__("slope", sel)
        self.tolerance = tolerance

    def apply(self, ds: xr.Dataset) -> xr.Dataset:
        s = self.sel

        # Case 1: dict-style metadata query for slope properties
        if isinstance(s, dict):
            # Build a cumulative mask over the slope dimension
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

                # Handle case where arr has extra dimensions (e.g., location)
                if "slope" not in arr.dims:
                    raise ValueError(
                        f"SlopeSelector: field '{key}' is not defined per-slope "
                        f"(dims={arr.dims})"
                    )
                
                # If arr has extra dimensions beyond 'slope', reduce them
                # This handles the common case of (location, slope) -> (slope)
                extra_dims = [d for d in arr.dims if d != 'slope']
                if extra_dims:
                    # Check if values are consistent across extra dimensions
                    # If so, take the first value; otherwise warn
                    if 'location' in extra_dims:
                        # Take first location, but check consistency
                        first_loc = arr.isel(location=0)
                        
                        # Check if all locations have the same values
                        if arr.sizes['location'] > 1:
                            # Compare all locations to first
                            all_same = True
                            for loc_idx in range(1, arr.sizes['location']):
                                if not np.allclose(
                                    arr.isel(location=loc_idx).values,
                                    first_loc.values,
                                    rtol=1e-9,
                                    atol=1e-9,
                                    equal_nan=True
                                ):
                                    all_same = False
                                    break
                            
                            if not all_same:
                                warnings.warn(
                                    f"Slope coordinate '{key}' varies across locations. "
                                    f"Using values from first location for selection. "
                                    f"This may indicate a data structure issue - slope "
                                    f"coordinates should typically be consistent across locations.",
                                    UserWarning
                                )
                        
                        arr = first_loc
                    else:
                        # For other extra dimensions, just take the first value
                        warnings.warn(
                            f"Slope coordinate '{key}' has unexpected dimensions {arr.dims}. "
                            f"Reducing by taking first value along {extra_dims}.",
                            UserWarning
                        )
                        for dim in extra_dims:
                            arr = arr.isel({dim: 0})

                # Determine tolerance for this comparison
                tol = self.tolerance
                if tol is None:
                    # Auto-determine: exact for integers, small tolerance for floats
                    if np.issubdtype(arr.dtype, np.integer):
                        tol = 0
                    else:
                        tol = 1e-6
                
                # Apply condition with tolerance
                if tol == 0:
                    cond = (arr == expected)
                else:
                    cond = np.abs(arr - expected) <= tol
                
                # Ensure cond has the same shape as mask
                cond = cond.broadcast_like(mask)
                mask = mask & cond

            # Check if anything matched
            if not bool(mask.any()):
                # No slope satisfies all criteria. HARD FAIL with helpful message.
                criteria = ", ".join(f"{k} == {v}" for k, v in s.items())
                
                # Provide helpful diagnostics
                avail_info = []
                for key in s.keys():
                    if key in ds.coords or key in ds:
                        arr = ds.coords.get(key, ds.get(key))
                        # Reduce to slope dimension for reporting
                        if 'location' in arr.dims:
                            arr = arr.isel(location=0)
                        unique_vals = np.unique(arr.values)
                        if len(unique_vals) <= 10:
                            avail_info.append(f"  {key}: {list(unique_vals)}")
                        else:
                            avail_info.append(
                                f"  {key}: range [{arr.min().values}, {arr.max().values}]"
                            )
                
                avail_str = "\n".join(avail_info) if avail_info else "  (no diagnostic info available)"
                
                raise ValueError(
                    f"No slope matches criteria: {criteria}\n"
                    f"Available values:\n{avail_str}\n"
                    f"Consider using tolerance parameter if values are close."
                )

            # Convert boolean mask -> integer indices
            matching_idx = np.nonzero(mask.values)[0]

            return ds.isel(slope=matching_idx)

        # Case 2: everything else (ints, slices, xr.DataArray, etc.)
        return super().apply(ds)


class RealizationSelector(Selector):
    """A selector specifically for the 'realization' dimension."""
    def __init__(self, sel: RealIndexLike):
        super().__init__("realization", sel)
