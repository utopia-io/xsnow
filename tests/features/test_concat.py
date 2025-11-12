import numpy as np
import xarray as xr
import pandas as pd
import pytest

from xsnow.core import xsnowDataset
from xsnow.utils import concat as xsnow_concat


def _make_profile_ds(time_vals, layer_vals, value_offset=0.0, name="v"):
    """Helper: small deterministic Dataset with time, layer, and one data var."""
    time_vals = np.asarray(time_vals)
    layer_vals = np.asarray(layer_vals)

    data = (
        np.arange(time_vals.size * layer_vals.size, dtype=float)
        .reshape(time_vals.size, layer_vals.size)
        + float(value_offset)
    )

    return xr.Dataset(
        {name: (("time", "layer"), data)},
        coords={"time": time_vals, "layer": layer_vals},
    )


# =====================================================================
# 1. Basic behavior: outer / exact should delegate to xarray.concat
# =====================================================================

def test_concat_outer_matches_xarray_behavior():
    ds1 = _make_profile_ds([0, 1], [0, 1], value_offset=0)
    ds2 = _make_profile_ds([1, 2], [1, 2], value_offset=10)

    expected = xr.concat([ds1, ds2], dim="realization", join="outer")
    result = xsnow_concat([ds1, ds2], dim="realization", join="outer")

    assert isinstance(result, xr.Dataset)
    xr.testing.assert_identical(result, expected)


def test_concat_outer_with_xsnowdataset_wraps_output():
    ds1 = xsnowDataset(_make_profile_ds([0, 1], [0, 1], value_offset=0))
    ds2 = _make_profile_ds([1, 2], [1, 2], value_offset=10)

    result = xsnow_concat([ds1, ds2], dim="realization", join="outer")

    # If any input is xsnowDataset -> result should be xsnowDataset
    assert isinstance(result, xsnowDataset)
    # Underlying data should match pure xarray.concat
    expected = xr.concat([ds1.data, ds2], dim="realization", join="outer")
    xr.testing.assert_identical(result.data, expected)


def test_concat_exact_delegates_and_errors_like_xarray():
    ds1 = _make_profile_ds([0, 1], [0, 1], value_offset=0)
    ds2 = _make_profile_ds([0, 1], [0, 1], value_offset=10)

    # Same coords: ok
    xsnow_concat([ds1, ds2], dim="realization", join="exact")

    # Different coords: both xarray.concat and xsnow_concat should fail
    ds3 = _make_profile_ds([0, 2], [0, 1], value_offset=20)
    with pytest.raises(Exception):
        xr.concat([ds1, ds3], dim="realization", join="exact")
    with pytest.raises(Exception):
        xsnow_concat([ds1, ds3], dim="realization", join="exact")


# =====================================================================
# 2. Special behavior: inner join keeps union of 'layer'
# =====================================================================

def test_concat_inner_uses_inner_for_time_but_union_for_layer():
    # ds1: time 0,1; layer 0,1
    ds1 = _make_profile_ds([0, 1], [0, 1], value_offset=0)
    # ds2: time 1,2; layer 1,2
    ds2 = _make_profile_ds([1, 2], [1, 2], value_offset=10)

    result = xsnow_concat([ds1, ds2], dim="realization", join="inner")

    # time should be intersection: only 1
    assert "time" in result.dims
    assert list(result["time"].values) == [1]

    # layer should be UNION, not intersection: 0,1,2
    assert "layer" in result.dims
    assert list(result["layer"].values) == [0, 1, 2]

    # Shape: realization x time x layer
    assert result.dims["realization"] == 2
    assert result.dims["time"] == 1
    assert result.dims["layer"] == 3

    # Check that where there was no data, we have NaNs (from reindex)
    v = result["v"]
    # For time=1 and layer=1 both have data, so both realizations have non-NaN
    assert not np.isnan(v.isel(realization=0, time=0, layer=1))
    assert not np.isnan(v.isel(realization=1, time=0, layer=1))
    # For layers that didn't exist originally, at least some entries should be NaN
    assert np.isnan(v.isel(realization=0, time=0, layer=0))
    assert np.isnan(v.isel(realization=1, time=0, layer=2))


# =====================================================================
# 3. Special behavior: left/right joins + union of 'layer'
# =====================================================================

def test_concat_left_preserves_left_time_and_union_layer():
    # Left ds: time 0,1; layer 0,1
    ds_left = _make_profile_ds([0, 1], [0, 1], value_offset=0)
    # Right ds: time 1,2; layer 1,2
    ds_right = _make_profile_ds([1, 2], [1, 2], value_offset=10)

    result = xsnow_concat([ds_left, ds_right], dim="realization", join="left")

    # Left join on time: use left's time coords (0,1)
    assert list(result["time"].values) == [0, 1]

    # BUT: layer must be union: 0,1,2
    assert list(result["layer"].values) == [0, 1, 2]

    # For time=0, only left dataset contributed
    v = result["v"]
    # realization=0 is left dataset:
    assert not np.isnan(v.isel(realization=0, time=0, layer=0))
    # realization=1 (right) should be NaN for time=0 because it's outside its domain
    assert np.isnan(v.isel(realization=1, time=0, layer=0))


def test_concat_right_preserves_right_time_and_union_layer():
    # Left ds: time 0,1; layer 0,1
    ds_left = _make_profile_ds([0, 1], [0, 1], value_offset=0)
    # Right ds: time 1,2; layer 1,2
    ds_right = _make_profile_ds([1, 2], [1, 2], value_offset=10)

    result = xsnow_concat([ds_left, ds_right], dim="realization", join="right")

    # Right join on time: use right's time coords (1,2)
    assert list(result["time"].values) == [1, 2]

    # layer must be union: 0,1,2
    assert list(result["layer"].values) == [0, 1, 2]

    # At time=2, only right dataset has data; left should be NaN there
    v = result["v"]
    assert not np.isnan(v.isel(realization=1, time=1, layer=1))
    assert np.isnan(v.isel(realization=0, time=1, layer=1))


# =====================================================================
# 4. No 'layer' dimension → behave like xarray.concat
# =====================================================================

def test_concat_without_layer_falls_back_to_xarray():
    ds1 = xr.Dataset(
        {"a": ("time", [1, 2])},
        coords={"time": [0, 1]},
    )
    ds2 = xr.Dataset(
        {"a": ("time", [3, 4])},
        coords={"time": [1, 2]},
    )

    expected = xr.concat([ds1, ds2], dim="realization", join="inner")
    result = xsnow_concat([ds1, ds2], dim="realization", join="inner")

    xr.testing.assert_identical(result, expected)


# =====================================================================
# 5. Single input and type behavior
# =====================================================================

def test_concat_single_input_passthrough():
    ds = _make_profile_ds([0, 1], [0, 1])
    xds = xsnowDataset(ds)

    out1 = xsnow_concat([ds], dim="realization")
    out2 = xsnow_concat([xds], dim="realization")

    assert isinstance(out1, xr.Dataset)
    xr.testing.assert_identical(out1, ds)

    assert isinstance(out2, xsnowDataset)
    xr.testing.assert_identical(out2.data, ds)
