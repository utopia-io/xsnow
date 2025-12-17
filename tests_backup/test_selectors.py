# test_selectors.py
"""
Unit tests for the selectors.py module.

This module uses the pytest framework to test the functionality of the
Selector classes, ensuring they correctly build and apply queries to
xarray.Dataset objects.
"""

import pytest
import xarray as xr
import pandas as pd
import numpy as np

# Import the selector classes to be tested
from xsnow.selectors import (
    TimeSelector,
    LocationSelector,
    SlopeSelector,
    RealizationSelector
)

# --- Fixtures ---

@pytest.fixture
def mock_dataset():
    """
    Creates a standard, predictable xarray.Dataset for testing selectors.
    It includes multiple locations with distinct coordinates.
    """
    ds = xr.Dataset(
        data_vars={
            'density': (('location', 'time'), np.random.rand(3, 5))
        },
        coords={
            'location': ['station_A', 'station_B', 'station_C'],
            'time': pd.to_datetime(pd.date_range('2025-01-01', periods=5)),
            'latitude': ('location', [45.0, 45.5, 46.0]),
            'longitude': ('location', [10.0, 10.5, 11.0])
        }
    )
    # Add CRS metadata for testing geographic queries
    ds.attrs['crs'] = 'EPSG:4326'
    return ds

# --- Tests for LocationSelector ---

def test_location_selector_by_id(mock_dataset):
    """Tests selecting a single location by its string ID."""
    selector = LocationSelector(sel='station_B')
    result = selector.apply(mock_dataset)
    # FIX: When selecting a single label, the dimension is dropped.
    # We verify the coordinate value instead of the dimension size.
    assert 'location' not in result.sizes
    assert result.location.item() == 'station_B'

def test_location_selector_by_list(mock_dataset):
    """Tests selecting multiple locations using a list of IDs."""
    selector = LocationSelector(sel=['station_A', 'station_C'])
    result = selector.apply(mock_dataset)
    assert result.sizes['location'] == 2
    assert 'station_B' not in result.location.values

def test_location_selector_by_bbox(mock_dataset):
    """Tests selecting locations within a geographic bounding box."""
    # This should select station_A and station_B
    selector = LocationSelector(lat_range=(44.9, 45.6), lon_range=(9.9, 10.6))
    result = selector.apply(mock_dataset)
    assert result.sizes['location'] == 2
    assert 'station_C' not in result.location.values

    # This should select only station_C
    selector_single = LocationSelector(lat_range=(45.9, 46.1))
    result_single = selector_single.apply(mock_dataset)
    assert result_single.sizes['location'] == 1
    assert result_single.location.item() == 'station_C'

def test_location_selector_crs_warning(mock_dataset):
    """Tests that a warning is raised if the query CRS mismatches the dataset CRS."""
    selector = LocationSelector(lat_range=(45.0, 46.0), crs='EPSG:3857')
    with pytest.warns(UserWarning, match="Query CRS .* does not match dataset CRS"):
        selector.apply(mock_dataset)

# --- Tests for Other Selectors ---

def test_time_selector(mock_dataset):
    """Tests selecting a time slice."""
    selector = TimeSelector(slice('2025-01-02', '2025-01-04'))
    result = selector.apply(mock_dataset)
    assert result.sizes['time'] == 3
    assert pd.to_datetime(result.time.min().values).day == 2
    assert pd.to_datetime(result.time.max().values).day == 4

def test_slope_selector(mock_dataset):
    """Tests a simple integer-based selection on a dummy slope dimension."""
    ds = mock_dataset.expand_dims(slope=[0, 1])
    selector = SlopeSelector(sel=0)
    result = selector.apply(ds)
    assert 'slope' not in result.dims # Dimension is dropped after selection
    assert result.coords['slope'].item() == 0

def test_realization_selector(mock_dataset):
    """Tests a simple slice selection on a dummy realization dimension."""
    ds = mock_dataset.expand_dims(realization=np.arange(5))
    selector = RealizationSelector(sel=slice(1, 4))
    result = selector.apply(ds)
    assert result.sizes['realization'] == 3
