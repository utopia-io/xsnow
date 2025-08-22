# tests/test_accessors.py

import pytest
import xarray as xr
import numpy as np
import xsnow
from xsnow import accessors

@pytest.fixture
def indexed_coord():
    """Creates a DataArray with an integer index and a name mapping."""
    mapping = {0: 'station_A', 1: 'station_B'}
    
    return xr.DataArray(
        [0, 1], dims=['location'], name='location', attrs={'mapping': mapping}
    )

@pytest.fixture
def non_indexed_coord():
    """Creates a standard DataArray without a name mapping."""
    return xr.DataArray([10, 20], dims=['x'], name='x')

def test_labels_accessor_get(indexed_coord):
    """
    Tests that the .labels.get() method successfully retrieves the mapping.
    """
    expected_mapping = {0: 'station_A', 1: 'station_B'}
    actual_mapping = indexed_coord.labels.get()
    
    assert isinstance(actual_mapping, dict)
    assert actual_mapping == expected_mapping

def test_labels_accessor_no_mapping(non_indexed_coord):
    """
    Tests that the accessor returns an empty dict when no mapping exists.
    """
    actual_mapping = non_indexed_coord.labels.get()
    
    assert isinstance(actual_mapping, dict)
    assert actual_mapping == {}