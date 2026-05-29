# tests/test_accessors.py

import pytest
import numpy as np
import pandas as pd
import warnings
import xarray as xr
import pytz
import xsnow
from zoneinfo import ZoneInfo
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
    time = pd.date_range("2024-01-20 00:00", periods=48, freq="h")

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

@pytest.fixture
def ds(tmp_path) -> xsnow.xsnowDataset:
    """
    Dataset created by reading a temporary SMET file with a timezone
    attribute set in the header (tz = 1).
    """
    # This SMET content includes a timezone and covers the date ranges
    # needed by the accessor tests.
    smet_content = """
SMET 1.1 ASCII
[HEADER]
station_name = station_A
tz = 1
fields = timestamp HS
[DATA]
2024-01-20T00:00:00 0.50
2024-01-20T06:00:00 0.56
2024-01-21T12:00:00 0.62
2024-01-22T00:00:00 0.64
"""
    smet_file = tmp_path / "test.smet"
    smet_file.write_text(smet_content)
    
    # The read() function will now automatically handle attaching the timezone.
    return xsnow.read(smet_file)

@pytest.fixture
def dst_gap_time_coord():
    """Naive times that hit the DST spring-forward gap in America/New_York."""
    times = pd.date_range("2024-03-10 01:30", periods=3, freq="30min")
    return xr.DataArray(
        times.values.astype("datetime64[ns]"),
        dims=["time"],
        name="time",
        attrs={"timezone": "America/New_York"},
    )

@pytest.fixture
def dst_fall_time_coord():
    """Naive times that become ambiguous during DST fall-back in Europe/Vienna."""
    times = pd.date_range("2024-10-27 01:30", periods=3, freq="30min")
    return xr.DataArray(
        times.values.astype("datetime64[ns]"),
        dims=["time"],
        name="time",
        attrs={"timezone": "Europe/Vienna"},
    )

@pytest.fixture
def empty_time_coord():
    """Empty datetime coordinate to exercise edge-case paths."""
    return xr.DataArray(
        np.array([], dtype="datetime64[ns]"),
        dims=["time"],
        name="time",
        attrs={"timezone": "UTC"},
    )

## ------- LabelsAccessor ----------------------------------- ##
def test_labels_accessor_get(indexed_coord):
    """
    Tests that the .labels.get() method successfully retrieves the DataArray of labels.
    """
    expected_vals = ['station_A', 'station_A', 'station_B', None]
    actual_da = indexed_coord.labels.get()

    assert isinstance(actual_da, xr.DataArray)
    # Use pandas for comparison since None == None is False in numpy
    pd.testing.assert_series_equal(
        pd.Series(actual_da.values),
        pd.Series(expected_vals),
        check_names=False,
    )

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
    ds = xsnow.read(f"{snowobs_dir}/pros/snowobs", n_cpus_use=1)
    primary = ds.grain_type.icssg.primary().values
    secondary = ds.grain_type.icssg.secondary().values
    codes = ds.grain_type.values
    unique_codes, unique_indices = np.unique(codes, return_index=True, equal_nan=True)
    unique_primary = primary.flatten()[unique_indices]
    unique_secondary = secondary.flatten()[unique_indices]

    expected_codes = np.array([
        110., 121., 210., 211., 220., 230., 231., 240., 241., 321., 330.,
        341., 421., 430., 431., 440., 450., 490., 550., 591., 660., 770.,
        772., 951., 990., np.nan
    ])
    expected_primary = np.array([
        'PP', 'PP', 'DF', 'DF', 'DF', 'DF', 'DF', 'DF', 'DF', 'RG', 'RG',
        'RG', 'FC', 'FC', 'FC', 'FC', 'FC', 'FC', 'DH', 'DH', 'SH', 'MF',
        'MFcr', 'FCxr', 'FCxr', np.nan
    ], dtype=object)
    expected_secondary = np.array([
        'PP', 'DF', 'PP', 'PP', 'DF', 'RG', 'RG', 'FC', 'FC', 'DF', 'RG',
        'FC', 'DF', 'RG', 'RG', 'FC', 'DH', 'FCxr', 'DH', 'FCxr', 'SH',
        'MF', 'MF', 'DH', 'FCxr', np.nan
    ], dtype=object)

    def _array_eq_nan_none(a, b):
        """Compare object arrays treating nan and None as equal missing values."""
        a_null = np.array([x is None or (isinstance(x, float) and np.isnan(x)) for x in a])
        b_null = np.array([x is None or (isinstance(x, float) and np.isnan(x)) for x in b])
        return (np.array_equal(a_null, b_null) and
                np.array_equal(np.asarray(a)[~a_null], np.asarray(b)[~b_null]))

    assert np.array_equal(unique_codes, expected_codes, equal_nan=True)
    assert _array_eq_nan_none(unique_primary, expected_primary)
    assert _array_eq_nan_none(unique_secondary, expected_secondary)

## ------- TimezoneAccessor ----------------------------------- ##
def test_set_get_attr(ds):
    t = ds["time"]
    t.tz.set_attr("America/Denver")
    assert t.tz.get_attr() == "America/Denver"

def test_set_attr_normalizes_offsets(ds):
    t = ds["time"]
    t.tz.set_attr(-7.5)
    assert t.tz.get_attr() == "UTC-07:30"
    t.tz.set_attr("UTC+1")
    assert t.tz.get_attr() == "UTC+01:00"
    t.tz.set_attr("Europe/Paris")
    assert t.tz.get_attr() == "Europe/Paris"


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

def test_localize_handles_nonexistent_times(dst_gap_time_coord):
    with pytest.raises((pytz.NonExistentTimeError, ValueError)):
        dst_gap_time_coord.tz.localize()

    shifted = dst_gap_time_coord.tz.localize(nonexistent="shift_forward")
    assert isinstance(shifted, pd.DatetimeIndex)
    assert shifted[1].hour == 3  # clock jumps to 03:00 after missing hour
    assert str(shifted.tz) == "America/New_York"

    nat_idx = dst_gap_time_coord.tz.localize(nonexistent="NaT")
    assert nat_idx.isna().sum() == 2  # nonexistent times become NaT

def test_localize_handles_ambiguous_times(dst_fall_time_coord):
    with pytest.raises((pytz.AmbiguousTimeError, ValueError)):
        dst_fall_time_coord.tz.localize()

    ambiguous_as_nat = dst_fall_time_coord.tz.localize(ambiguous="NaT")
    assert ambiguous_as_nat.isna().sum() == 2  # two repeated times during fallback
    assert not pd.isna(ambiguous_as_nat[0])


def test_localize_with_explicit_tz(ds):
    with pytest.warns(UserWarning):
        idx = ds["time"].tz.localize(tz="UTC")
    assert isinstance(idx, pd.DatetimeIndex)
    assert str(idx.tz) == "UTC"

def test_localize_without_conflict_no_warning(ds):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ds["time"].tz.localize()


def test_to_datetimeindex_in_and_to_naive_in_roundtrip(ds):
    # Vienna-aware → UTC-aware → UTC-naive and compare with manual path
    idx_utc = ds["time"].tz.to_datetimeindex_in("UTC")
    assert isinstance(idx_utc, pd.DatetimeIndex)
    assert str(idx_utc.tz) in ("UTC")

    naive_utc = ds['time'].tz.to_naive_in("UTC")
    idx_naive_utc = idx_utc.tz_localize(None)
    np.testing.assert_array_equal(
        naive_utc.values.astype("datetime64[ns]"),
        idx_naive_utc.to_numpy(dtype="datetime64[ns]"),
    )

def test_to_datetimeindex_in_series_alignment(ds):
    ser = ds["time"].tz.to_datetimeindex_in(0, as_series=True)
    assert isinstance(ser, pd.Series)
    pd.testing.assert_index_equal(ser.index, ds.indexes["time"])
    assert str(ser.dt.tz) == "UTC"
    assert ser.iloc[0] == pd.Timestamp("2024-01-19 23:00:00+00:00")

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

def test_to_naive_in_preserves_coords_and_attrs(ds):
    naive = ds["time"].tz.to_naive_in("UTC")
    assert naive.name == "time_utc_naive"
    assert naive.attrs["timezone"] == "UTC"
    assert naive.attrs["note"] == "dtype is tz-naive"
    xr.testing.assert_equal(naive["time"], ds["time"])
    assert not np.array_equal(naive.values, ds["time"].values)


def test_between_defaults_full_span_da(ds):
    mask = ds["time"].tz.between()  # default: DataArray mask
    assert isinstance(mask, xr.DataArray)
    assert mask.dims == ("time",)
    assert mask.dtype == bool
    assert bool(mask.all()) is True  # full span

def test_between_respects_tzaware_bounds(ds):
    start = pd.Timestamp("2024-01-19 23:00:00", tz="UTC")
    end = pd.Timestamp("2024-01-20 05:00:00", tz="UTC")
    mask = ds["time"].tz.between(start, end)
    assert mask.sum().item() == 2  # first two timestamps fall in this UTC window

def test_between_tzaware_with_user_tz_mismatch_raises(ds):
    start = pd.Timestamp("2024-01-19 23:00:00", tz="UTC")
    with pytest.raises(ValueError):
        ds["time"].tz.between(start, tz="Europe/Vienna")

def test_between_tznaive_requires_user_tz(ds):
    with pytest.raises(ValueError):
        ds["time"].tz.between("2024-01-19")

def test_between_does_not_alter_coordinate(ds):
    original = ds["time"].copy()
    _ = ds["time"].tz.between("2024-01-20", "2024-01-21", tz="UTC")
    xr.testing.assert_equal(ds["time"], original)

def test_between_empty_coord(empty_time_coord):
    mask = empty_time_coord.tz.between("2024-01-01", "2024-01-02")
    assert isinstance(mask, xr.DataArray)
    assert mask.size == 0
    assert mask.dtype == bool
    assert mask.dims == ("time",)


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
    # This failed because the 'slope' and 'realization' dimensions
    # are not present in the dataset loaded from a SMET file.
    # hs = ds["HS"].isel(location=0, slope=0, realization=0).to_pandas()

    #Remove the non-existent dimensions from the selection.
    hs = ds["HS"].isel(location=0, slope=0, realization=0).to_pandas()
    assert len(mask_series) == len(hs)  # aligned lengths


def test_between_swapped_bounds_normalized(ds):
    m1 = ds["time"].tz.between("2024-01-22", "2024-01-20", tz="UTC")  # reversed
    m2 = ds["time"].tz.between("2024-01-20", "2024-01-22", tz="UTC")
    xr.testing.assert_equal(m1, m2)


def test_reassign_time_with_converted_naive_keeps_attrs(ds):
    original_attrs = ds.attrs.copy()
    other_coord_attrs = {name: ds[name].attrs.copy() for name in ds.coords if name != "time"}

    converted = ds["time"].tz.to_naive_in("UTC")
    ds_converted = ds.assign_coords(time=converted)

    # values converted, attrs updated
    np.testing.assert_array_equal(ds_converted["time"].values, converted.values)
    assert ds_converted["time"].attrs["timezone"] == "UTC"
    assert ds_converted["time"].attrs["note"] == "dtype is tz-naive"

    # dataset-level attrs preserved
    assert ds_converted.attrs == original_attrs

    # other coordinate attrs preserved
    for name, attrs in other_coord_attrs.items():
        assert ds_converted[name].attrs == attrs


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
