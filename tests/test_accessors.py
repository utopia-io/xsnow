# tests/test_accessors.py

import pytest
import numpy as np
import pandas as pd
import xarray as xr
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

@pytest.fixture
def ds() -> xr.Dataset:
    """
    Dataset mock-up (for tz-accessor tests).

    - time: hourly stamps (datetime64[ns])
    - dims: location x time x slope x realization
    - var: HS (snow height) with simple, non-constant values
    - includes: time.attrs["timezone"] for tz-accessor

    Shape: (location=2, time=48, slope=1, realization=1)
    """

    # # replace the mock-up ds with packaged ds:
    # ds = xsnow.single_profile_timeseries()
    # ds["time"].attrs["timezone"] = "Europe/Vienna"
    # return ds

    # 48 hours starting 2024-01-17 00:00 (keeps DST out of the way for most tests)
    time = pd.date_range("2024-01-20 00:00", periods=48, freq="H")

    # Coordinates
    location = ["station_A", "station_B"]
    slope = [0]
    realization = [0]

    # Simple, deterministic HS field: linear + small sine to avoid constant series
    locN = len(location)
    tN = len(time)
    slN = len(slope)
    rN = len(realization)

    # base shape (location, time, slope, realization)
    base = np.arange(tN, dtype=float)  # 0..47
    hs = (
        base[None, :, None, None]               # broadcast along time
        + np.array([0.0, 5.0])[:, None, None, None]  # offset per location
        + 2.0 * np.sin(np.linspace(0, 2*np.pi, tN))[None, :, None, None]
    )
    hs = np.broadcast_to(hs, (locN, tN, slN, rN)).astype("float32")

    ds = xr.Dataset(
        data_vars={
            "HS": (("location", "time", "slope", "realization"), hs),
        },
        coords={
            "time": ("time", time.values.astype("datetime64[ns]")),
            "location": ("location", location),
            "slope": ("slope", slope),
            "realization": ("realization", realization),
        },
        attrs={"title": "mock xsnow test dataset"},
    )

    ds["time"].attrs["timezone"] = "UTC+01:00"
    return ds



## ------- LabelsAccessor ----------------------------------- ##
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



## ------- ICSSGAccessor ----------------------------------- ##
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



## ------- TimezoneAccessor ----------------------------------- ##
def test_set_get_attr(ds):
    t = ds["time"]
    t.tz.set_attr("America/Denver")
    assert t.tz.get_attr() == "America/Denver"


def test_set_attr_invalid_raises(ds):
    with pytest.raises(ValueError):
        ds["time"].tz.set_attr("")  # empty
    with pytest.raises(ValueError):
        ds["time"].tz.set_attr("Not/A/Timezone")


def test_localize_uses_attr_returns_index(ds):
    ds["time"].tz.set_attr("Europe/Vienna")
    idx = ds["time"].tz.localize()
    assert isinstance(idx, pd.DatetimeIndex)
    # tz repr may vary (Europe/Vienna resolves to CET/CEST internally)
    assert str(idx.tz) in ("Europe/Vienna", "CET", "CEST")
    assert len(idx) == ds.sizes["time"]


def test_localize_as_series(ds):
    ser = ds["time"].tz.localize(as_series=True)
    assert isinstance(ser, pd.Series)
    assert len(ser) == ds.sizes["time"]
    # index alignment with original coordinate
    pd.testing.assert_index_equal(ser.index, ds.indexes["time"])


def test_localize_with_explicit_tz(ds):
    idx = ds["time"].tz.localize(tz="UTC")
    assert isinstance(idx, pd.DatetimeIndex)
    assert str(idx.tz) == "UTC"


def test_to_datetimeindex_in_and_to_naive_in_roundtrip(ds):
    # Vienna-aware → UTC-aware → UTC-naive and compare with manual path
    idx_vie = ds["time"].tz.to_datetimeindex_in("Europe/Vienna")
    assert isinstance(idx_vie, pd.DatetimeIndex)
    assert str(idx_vie.tz) in ("Europe/Vienna", "CET", "CEST")

    # UTC-naive via helper
    utc_naive = ds["time"].tz.to_naive_in("UTC")
    assert isinstance(utc_naive, xr.DataArray)
    assert np.issubdtype(utc_naive.dtype, np.datetime64)

    # Manual equivalent: localize (attr), convert to UTC, drop tz
    idx_manual = ds["time"].tz.localize().tz_convert("UTC").tz_localize(None)
    np.testing.assert_array_equal(
        utc_naive.values.astype("datetime64[ns]"),
        idx_manual.to_numpy(dtype="datetime64[ns]"),
    )


def test_to_naive_in_series(ds):
    ser = ds["time"].tz.to_naive_in("Europe/Vienna", as_series=True)
    assert isinstance(ser, pd.Series)
    assert len(ser) == ds.sizes["time"]
    # dtype is numpy datetime64[ns] wrapped in Series
    assert np.issubdtype(ser.values.dtype, np.datetime64)
    pd.testing.assert_index_equal(ser.index, ds.indexes["time"])


def test_between_defaults_full_span_da(ds):
    mask = ds["time"].tz.between()  # default: DataArray mask
    assert isinstance(mask, xr.DataArray)
    assert mask.dims == ("time",)
    assert mask.dtype == bool
    assert bool(mask.all()) is True  # full span


def test_between_range_selection_da(ds):
    # choose a narrow Vienna time window
    start = "2024-01-20 06:00"
    end   = "2024-01-21 12:00"
    mask = ds["time"].tz.between(start, end, tz="Europe/Vienna")
    ds_sub = ds.sel(time=mask)
    # sanity: non-empty and all times within requested window in Vienna
    ser_vienna = ds_sub["time"].tz.localize(as_series=True)
    assert len(ser_vienna) > 0
    s = pd.Timestamp(start).tz_localize("Europe/Vienna")
    e = pd.Timestamp(end).tz_localize("Europe/Vienna")
    assert (ser_vienna >= s).all() and (ser_vienna <= e).all()


def test_between_series_mask(ds):
    mask_series = ds["time"].tz.between(
        "2024-01-20", "2024-01-22", tz="UTC", as_series=True
    )
    assert isinstance(mask_series, pd.Series)
    hs = ds["HS"].isel(location=0, slope=0, realization=0).to_pandas()
    assert len(mask_series) == len(hs)  # aligned lengths


def test_between_swapped_bounds_normalized(ds):
    m1 = ds["time"].tz.between("2024-01-22", "2024-01-20", tz="UTC")  # reversed
    m2 = ds["time"].tz.between("2024-01-20", "2024-01-22", tz="UTC")
    xr.testing.assert_equal(m1, m2)


def test_scalar_localize_and_to_naive_in(ds):
    t0 = ds["time"].isel(time=0)

    # localize returns a DatetimeIndex even for scalar (length-1)
    idx = t0.tz.localize()
    assert isinstance(idx, pd.DatetimeIndex)
    assert len(idx) == 1

    # to_naive_in returns a 0-D DataArray by default → use .item() for scalar
    v = t0.tz.to_naive_in("UTC")
    arr = v.to_numpy()
    assert isinstance(v, xr.DataArray)
    assert v.ndim == 0
    assert np.issubdtype(arr.dtype, np.datetime64)


def test_accessor_rejects_non_datetime():
    arr = xr.DataArray([1, 2, 3], dims=("time",))
    # Since your accessor raises in __init__, xarray wraps it as RuntimeError on access
    with pytest.raises(RuntimeError, match="error initializing 'tz' accessor"):
        _ = arr.tz


def test_localize_requires_tz(ds):
    ds["time"].attrs.pop("timezone", None)
    with pytest.raises(ValueError):
        ds["time"].tz.localize()
