# tests/test_accessors.py

import pytest
import xarray as xr
import numpy as np
import xsnow
from xsnow import accessors
from xsnow.sample_data import snp_snowobs_dir

@pytest.fixture
def indexed_coord():
    """Creates a DataArray with an integer index and a name mapping."""
    mapping = {0: 'station_A', 1: 'station_B'}

    return xr.DataArray(
        [0, 0, 1, 2], dims=['location'], name='location', attrs={'mapping': mapping}
    )

@pytest.fixture
def non_indexed_coord():
    """Creates a standard DataArray without a name mapping."""
    return xr.DataArray([10, 20], dims=['x'], name='x')

def test_labels_accessor_get(indexed_coord):
    """
    Tests that the .labels.get() method successfully retrieves the DataArray of labels.
    """
    expected_da = xr.DataArray(['station_A', 'station_A', 'station_B', None],
                               dims=['location'], name='location')
    actual_da = indexed_coord.labels.get()

    assert isinstance(actual_da, xr.DataArray)
    assert (actual_da == expected_da).all()

def test_labels_accessor_get_dict(indexed_coord):
    """
    Tests that the .labels.get(as_dict=True) method successfully retrieves the mapping.
    """
    expected_mapping = {0: 'station_A', 1: 'station_B'}
    actual_mapping = indexed_coord.labels.get(as_dict=True)

    assert isinstance(actual_mapping, dict)
    assert actual_mapping == expected_mapping

def test_labels_accessor_no_mapping_raise(non_indexed_coord):
    """
    Tests that the accessor raises an exception when no mapping exists.
    """
    with pytest.raises(AttributeError):
        non_indexed_coord.labels.get()

def test_labels_accessor_no_mapping_empty_dict(non_indexed_coord):
    """
    Tests that the accessor returns an empty dict for a non-existant mapping
    """
    assert non_indexed_coord.labels.get(as_dict=True) == {}

def test_icssg_accessor_on_snowobs_data():
    snowobs_dir = snp_snowobs_dir()
    ds = xsnow.read(f"{snowobs_dir}/pros/snowobs")
    primary = ds.grain_type.icssg.primary().values
    secondary = ds.grain_type.icssg.secondary().values
    codes = ds.grain_type.values
    unique_codes, unique_indices = np.unique(codes, return_index=True, equal_nan=True)
    unique_primary = primary.flatten()[unique_indices]
    unique_secondary = secondary.flatten()[unique_indices]

    expected_codes = np.array([110., 121., 210., 211., 220., 230., 231., 240., 241., 321., 330.,
       341., 421., 430., 431., 440., 450., 490., 550., 591., 660., 770.,
       772., 951., 990.,  np.nan])
    expected_primary = np.array(['PP', 'PP', 'DF', 'DF', 'DF', 'DF', 'DF', 'DF', 'DF', 'RG', 'RG',
       'RG', 'FC', 'FC', 'FC', 'FC', 'FC', 'FC', 'DH', 'DH', 'SH', 'MF',
       'MFcr', 'FCxr', 'FCxr', None])
    expected_secondary = np.array(['PP', 'DF', 'PP', 'PP', 'DF', 'RG', 'RG', 'FC', 'FC', 'DF', 'RG',
       'FC', 'DF', 'RG', 'RG', 'FC', 'DH', 'FCxr', 'DH', 'FCxr', 'SH',
       'MF', 'MF', 'DH', 'FCxr', None])

    assert np.array_equal(unique_codes, expected_codes, equal_nan=True)
    assert np.array_equal(unique_primary, expected_primary)
    assert np.array_equal(unique_secondary, expected_secondary)
