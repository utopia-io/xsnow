# tests/test_concat_merge.py
import numpy as np
import pandas as pd
import xarray as xr
import pytest

from typing import Union

import xsnow
from xsnow import xsnowDataset


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _loc_names(ds: xsnowDataset) -> list[str]:
    vals = ds.data["location"].values
    return [v.decode() if isinstance(v, (bytes, bytearray)) else str(v) for v in vals]

def _copy_xs(xs: xsnowDataset) -> xsnowDataset:
    return xsnowDataset(xs.data.copy(deep=True))

def _first_two_locs(xs: xsnowDataset) -> tuple[str, str]:
    names = _loc_names(xs)
    assert len(names) >= 2
    return names[0], names[1]

def _make_profile_ds(time_vals, layer_vals, value_offset=0.0, name="v"):
    """
    Helper: small deterministic Dataset with 'time', 'layer',
    and one data variable.

    This is intentionally minimal and does NOT use xsnowDataset so that
    we can test type/behaviour differences between pure xarray and xsnow.
    """
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


def _make_simple_scalar_ds(value: float, time: int, attrs=None) -> xr.Dataset:
    attrs = attrs or {}
    return xr.Dataset(
        {"v": (("location", "time", "slope", "realization"), np.array([[[[value]]]]))},
        coords={
            "location": ["A"],
            "time": [time],
            "slope": [0],
            "realization": [0],
        },
        attrs=attrs,
    )


# =========================================================================
# xsnow.concat tests -- SYNTHETIC DATA
# =========================================================================

@pytest.mark.synthetic
def test_concat_outer_matches_xarray_behavior():
    """
    For simple non-xsnow Datasets with a 'layer' dimension and join='outer',
    xsnow.concat should behave identically to xarray.concat.
    """
    ds1 = _make_profile_ds([0, 1], [0, 1], value_offset=0)
    ds2 = _make_profile_ds([1, 2], [1, 2], value_offset=10)

    expected = xr.concat([ds1, ds2], dim="realization", join="outer")
    result = xsnow.concat([ds1, ds2], dim="realization", join="outer")

    assert isinstance(result, xr.Dataset)
    xr.testing.assert_identical(result, expected)


@pytest.mark.synthetic
def test_concat_outer_with_xsnowdataset_wraps_output():
    """
    If at least one input is xsnowDataset, xsnow.concat should return an
    xsnowDataset, with .data matching the pure xarray.concat result.
    """
    ds1 = xsnowDataset(_make_profile_ds([0, 1], [0, 1], value_offset=0))
    ds2 = _make_profile_ds([1, 2], [1, 2], value_offset=10)

    result = xsnow.concat([ds1, ds2], dim="realization", join="outer")

    # If any input is xsnowDataset -> result should be xsnowDataset
    assert isinstance(result, xsnowDataset)

    # Underlying data should match pure xarray.concat
    expected = xr.concat([ds1.data, ds2], dim="realization", join="outer")
    xr.testing.assert_identical(result.data, expected)


@pytest.mark.synthetic
def test_concat_inner_uses_inner_for_time_but_union_for_layer():
    """
    join='inner' should use intersection for 'time' but union for 'layer'.

    This is the synthetic counterpart of the real-data test where we check
    that xsnow.concat preserves union of 'layer' while other dims reflect
    the inner join.
    """
    # ds1: time 0,1; layer 0,1
    ds1 = _make_profile_ds([0, 1], [0, 1], value_offset=0)
    # ds2: time 1,2; layer 1,2
    ds2 = _make_profile_ds([1, 2], [1, 2], value_offset=10)

    result = xsnow.concat([ds1, ds2], dim="realization", join="inner")

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

    # assert explicit values
    v = result["v"]
    xr.testing.assert_allclose(v.isel(realization=0, time=0, layer=1), ds1['v'].isel(time=1, layer=1))
    xr.testing.assert_allclose(v.isel(realization=1, time=0, layer=1), ds2['v'].isel(time=0, layer=0))


@pytest.mark.synthetic
def test_concat_left_preserves_left_time_and_union_layer():
    """
    join='left': time domain from the left dataset, but 'layer' should
    still be the union across inputs.
    """
    # Left ds: time 0,1; layer 0,1
    ds_left = _make_profile_ds([0, 1], [0, 1], value_offset=0)
    # Right ds: time 1,2; layer 1,2
    ds_right = _make_profile_ds([1, 2], [1, 2], value_offset=10)

    result = xsnow.concat([ds_left, ds_right], dim="realization", join="left")

    # Left join on time: use left's time coords (0,1)
    assert list(result["time"].values) == [0, 1]

    # BUT: layer must be union: 0,1,2
    assert list(result["layer"].values) == [0, 1, 2]

    v = result["v"]
    # For time=0, only left dataset contributed
    # realization=0 is left dataset:
    assert not np.isnan(v.isel(realization=0, time=0, layer=0))
    # realization=1 (right) should be NaN for time=0 because it's outside its domain
    assert np.isnan(v.isel(realization=1, time=0, layer=0))


@pytest.mark.synthetic
def test_concat_right_preserves_right_time_and_union_layer():
    """
    join='right': time domain from the right dataset, but 'layer' should
    still be the union across inputs.
    """
    # Left ds: time 0,1; layer 0,1
    ds_left = _make_profile_ds([0, 1], [0, 1], value_offset=0)
    # Right ds: time 1,2; layer 1,2
    ds_right = _make_profile_ds([1, 2], [1, 2], value_offset=10)

    result = xsnow.concat([ds_left, ds_right], dim="realization", join="right")

    # Right join on time: use right's time coords (1,2)
    assert list(result["time"].values) == [1, 2]

    # layer must be union: 0,1,2
    assert list(result["layer"].values) == [0, 1, 2]

    v = result["v"]
    # At time=2, only right dataset has data; left should be NaN there
    assert not np.isnan(v.isel(realization=1, time=1, layer=1))
    assert np.isnan(v.isel(realization=0, time=1, layer=1))


@pytest.mark.synthetic
@pytest.mark.parametrize("combine_attrs_mode", ["no_conflicts", "no_conflicts_rlx"])
def test_concat_time_timezone_conflict_raises(combine_attrs_mode):
    ds1 = _make_profile_ds([0], [0])
    ds1 = ds1.assign_coords(time=ds1.time.assign_attrs(timezone="UTC"))
    ds2 = _make_profile_ds([1], [0])
    ds2 = ds2.assign_coords(time=ds2.time.assign_attrs(timezone="CET"))

    with pytest.raises(ValueError):
        xsnow.concat([ds1, ds2], dim="time", combine_attrs=combine_attrs_mode)


@pytest.mark.synthetic
def test_concat_time_axis_join_left_keeps_time_union():
    """
    When concatenating along 'time', join='left' must not drop time coords;
    it should still produce the union of times while applying join to
    non-concat dims only. Layer stays unioned.
    """
    ds_left = _make_profile_ds([0], [0, 1], value_offset=0)
    ds_right = _make_profile_ds([10], [1, 2], value_offset=10)

    result = xsnow.concat([ds_left, ds_right], dim="time", join="left")

    assert list(result["time"].values) == [0, 10]
    assert list(result["layer"].values) == [0, 1, 2]


@pytest.mark.synthetic
def test_concat_time_axis_join_right_keeps_time_union():
    """
    When concatenating along 'time', join='right' must not drop time coords;
    it should still produce the union of times while applying join to
    non-concat dims only. Layer stays unioned.
    """
    ds_left = _make_profile_ds([0], [0, 1], value_offset=0)
    ds_right = _make_profile_ds([10], [1, 2], value_offset=10)

    result = xsnow.concat([ds_left, ds_right], dim="time", join="right")

    assert list(result["time"].values) == [0, 10]
    assert list(result["layer"].values) == [0, 1, 2]


@pytest.mark.synthetic
def test_concat_without_layer_falls_back_to_xarray():
    """
    If there is no 'layer' dimension at all, xsnow.concat should essentially
    behave like xarray.concat (no special layer union semantics).
    """
    ds1 = xr.Dataset(
        {"a": ("time", [1, 2])},
        coords={"time": [0, 1]},
    )
    ds2 = xr.Dataset(
        {"a": ("time", [3, 4])},
        coords={"time": [1, 2]},
    )

    expected = xr.concat([ds1, ds2], dim="realization", join="inner")
    result = xsnow.concat([ds1, ds2], dim="realization", join="inner")

    xr.testing.assert_identical(result, expected)


@pytest.mark.synthetic
def test_concat_single_input_passthrough():
    """
    With a single input:
    - if it's an xarray.Dataset, xsnow.concat returns an xarray.Dataset,
    - if it's an xsnowDataset, xsnow.concat returns an xsnowDataset.
    """
    ds = _make_profile_ds([0, 1], [0, 1])
    xds = xsnowDataset(ds)

    out1 = xsnow.concat([ds], dim="realization")
    out2 = xsnow.concat([xds], dim="realization")

    assert isinstance(out1, xr.Dataset)
    xr.testing.assert_identical(out1, ds)

    assert isinstance(out2, xsnowDataset)
    xr.testing.assert_identical(out2.data, ds)


# =========================================================================
# xsnow.concat tests -- REAL DATA
# =========================================================================

@pytest.mark.real_data
def test_concat_two_locations_non_overlapping_layers_times_slopes(snp_gridded_xs):
    """
    xs1: 1st location, slopes {0,1}, layers 0..L1
    xs2: 2nd location, slopes {1}, layers 0..L2 (L2 > L1), +1 extra timestamp
    Outer join: union across non-concat coords; concat along 'location'
    """
    xs = snp_gridded_xs
    loc0, loc1 = _first_two_locs(xs)

    # Build xs1 and xs2 from the single dataset
    base = xs.data
    times = base["time"].values
    assert times.size >= 3, "Needs at least 3 timestamps for the test."

    xs1 = xsnowDataset(
        base.sel(location=loc0, slope=[0, 1], time=times[:-1], layer=slice(0, 20))
    )
    xs2 = xsnowDataset(
        base.sel(location=loc1, slope=[1], time=times, layer=slice(0, 25))
    )

    out = xsnow.concat([xs1, xs2], dim="location", join="outer")
    assert isinstance(out, xsnowDataset)
    ds = out.data

    # Location dimension: 2
    assert ds.sizes["location"] == 2
    # Slope union {0,1}
    assert set(ds["slope"].values.tolist()) == {0, 1}
    # Time union (xs2 has one more)
    assert ds.sizes["time"] == times.size
    # Layer union
    assert ds.sizes["layer"] == max(xs1.sizes['layer'], xs2.sizes['layer'])

    # Check values exist where expected:
    # For (loc0, slope=0) last time should be NaN (only xs2 had last time)
    arr = ds["density"]
    assert np.isnan(arr.sel(location=loc0, slope=0).isel(time=-1).values).all()


@pytest.mark.real_data
@pytest.mark.skip(reason="Skipped due to timezone conflicts in snowobs sample data")
def test_concat_realization_outer(snp_snowobs_sim_xs, snp_snowobs_obs_xs):
    """snp_snowobs_obs_xs
    Concat sim+obs along realization with custom Index labels: join='outer'
    """
    xs_sim = snp_snowobs_sim_xs
    xs_obs = snp_snowobs_obs_xs

    rdim = pd.Index(["simulated", "observed"], name="realization")
    out = xsnow.concat([xs_sim, xs_obs], dim=rdim, join="outer")
    assert isinstance(out, xsnowDataset)
    ds = out.data

    assert ds.sizes["realization"] == 2
    assert ds.sizes["location"] == xs_sim.sizes["location"]
    assert ds.sizes["slope"] == 1
    unique_times = np.unique(np.concatenate([xs_sim.time.values, xs_obs.time.values]))
    assert ds.sizes["time"] == unique_times.shape[0]
    assert ds.sizes["layer"] == xs_sim.sizes["layer"]


@pytest.mark.real_data
@pytest.mark.skip(reason="Skipped due to timezone conflicts in snowobs sample data")
def test_concat_realization_inner_keep_layer_union(snp_snowobs_sim_xs, snp_snowobs_obs_xs):
    """
    join='inner' but xsnow.concat preserves union of 'layer',
    while other dims reflect the inner join.
    """
    xs_sim = snp_snowobs_sim_xs
    xs_obs = snp_snowobs_obs_xs

    # Require that there are exactly 2 overlapping times (as per fixture note)
    t_sim = set(xs_sim.data["time"].values)
    t_obs = set(xs_obs.data["time"].values)
    overlap = sorted(t_sim & t_obs)
    assert len(overlap) >= 1, "Fixtures need to overlap in some timestamps."

    rdim = pd.Index(["simulated", "observed"], name="realization")
    out = xsnow.concat([xs_sim, xs_obs], dim=rdim, join="inner")
    ds = out.data

    assert ds.sizes["realization"] == 2
    assert ds.sizes["time"] == len(overlap)
    assert ds.sizes["location"] == 2
    assert ds.sizes["slope"] == 1
    assert ds.sizes["layer"] == xs_sim.sizes["layer"]


# ---------------------------------------------------------------------
# xsnow.combine compat / combine_attrs tests (synthetic)
# ---------------------------------------------------------------------


@pytest.mark.synthetic
def test_combine_compat_no_conflicts_raises_on_value_diff():
    ds_left = _make_simple_scalar_ds(1.0, time=0)
    ds_right = _make_simple_scalar_ds(2.0, time=0)

    with pytest.raises(ValueError):
        xsnow.combine([ds_left, ds_right], compat="no_conflicts")


@pytest.mark.synthetic
def test_combine_compat_override_prefers_left_even_if_different():
    ds_left = _make_simple_scalar_ds(1.0, time=0)
    ds_right = _make_simple_scalar_ds(2.0, time=0)

    out = xsnow.combine([ds_left, ds_right], compat="override")
    assert float(out["v"].values.squeeze()) == 1.0


@pytest.mark.synthetic
def test_combine_attrs_drop_conflicts_and_no_conflicts():
    ds_left = _make_simple_scalar_ds(1.0, time=0, attrs={"a": 1, "b": 2})
    ds_right = _make_simple_scalar_ds(1.0, time=1, attrs={"a": 1, "b": 3})

    out_drop = xsnow.combine(
        [ds_left, ds_right],
        compat="override",
        combine_attrs="drop_conflicts",
    )
    assert out_drop.attrs == {"a": 1}

    with pytest.raises(ValueError):
        xsnow.combine(
            [ds_left, ds_right],
            compat="override",
            combine_attrs="no_conflicts",
        )


@pytest.mark.synthetic
def test_combine_time_timezone_conflict_raises():
    ds_left = _make_simple_scalar_ds(1.0, time=0)
    ds_left = ds_left.assign_coords(time=ds_left.time.assign_attrs(timezone="UTC"))

    ds_right = _make_simple_scalar_ds(2.0, time=1)
    ds_right = ds_right.assign_coords(time=ds_right.time.assign_attrs(timezone="CET"))

    with pytest.raises(ValueError):
        xsnow.combine([ds_left, ds_right], compat="override", combine_attrs="override")


@pytest.mark.synthetic
def test_combine_crs_conflict_raises():
    ds_left = _make_simple_scalar_ds(1.0, time=0, attrs={"crs": "EPSG:4326"})
    ds_right = _make_simple_scalar_ds(2.0, time=1, attrs={"crs": "EPSG:3857"})

    with pytest.raises(ValueError):
        xsnow.combine([ds_left, ds_right], compat="override", combine_attrs="override")


@pytest.mark.synthetic
def test_combine_data_var_attr_conflict_modes():
    ds_left = _make_simple_scalar_ds(1.0, time=0)
    ds_left["v"].attrs = {"units": "K"}
    ds_right = _make_simple_scalar_ds(2.0, time=1)
    ds_right["v"].attrs = {"units": "C"}

    with pytest.raises(ValueError):
        xsnow.combine([ds_left, ds_right], compat="override", combine_attrs="no_conflicts")

    out_drop = xsnow.combine([ds_left, ds_right], compat="override", combine_attrs="drop_conflicts")
    assert "units" not in out_drop["v"].attrs

    out_override = xsnow.combine([ds_left, ds_right], compat="override", combine_attrs="override")
    assert out_override["v"].attrs.get("units") == "K"


@pytest.mark.synthetic
def test_combine_reports_multiple_conflicts_together():
    # Attr conflict on data_var and compat conflict on data values
    ds_left = _make_simple_scalar_ds(1.0, time=0)
    ds_left["v"].attrs = {"units": "K"}

    ds_right = _make_simple_scalar_ds(2.0, time=0)
    ds_right["v"].attrs = {"units": "C"}

    with pytest.raises(ValueError) as exc:
        xsnow.combine([ds_left, ds_right], compat="no_conflicts", combine_attrs="no_conflicts")

    msg = str(exc.value)
    assert "compat='no_conflicts'" in msg
    assert "'v'" in msg
    assert "combine_attrs='no_conflicts'" in msg
    assert "'units'" in msg
    assert "'K'" in msg



# =========================================================================
# Complex xsnow.combine behaviors on layered xsnowDatasets
# =========================================================================

def _prep_two_mergable_datasets(xs: xsnowDataset) -> tuple[xsnowDataset, xsnowDataset]:
    """
    Create 'old' and 'new' layered xsnowDatasets with overlapping time ranges and
    controlled manipulations for testing merge semantics.

    - old:  earlier time window
    - new:  later/longer time window, overlapping with old
    - various manipulations to old:
        * remove parts of two locations in time (set to NaN)
        * remove one slope globally
        * remove another slope only for one location
        * remove several layers at one overlap time
        * overwrite top layer at another overlap time
        * set all non-null density values to -999 during the overlap
    """
    
    reduce_dims = [d for d in xs.HS.dims if d != "time"]
    mask = (xs.HS > 0).all(dim=reduce_dims)
    first_all = xs["time"].where(mask, drop=True).min()
    xs = xs.sel(time=slice(first_all, None))

    base = xs.data
    times = base["time"].values
    assert times.size >= 48, "Need at least 48 timestamps to create overlap windows."

    # Define old/new windows (index-based, ensures overlap)
    t0 = int((base.HS >= 0).any(dim=["slope", "realization"])\
                                       .all(dim="location")\
                                       .argmax())  # first ts all locations have data
    if base.sizes['time'] < t0+38:
        raise ValueError("Longer timeseries required!")
    t_old = slice(times[t0],  times[t0+36])   # earlier window
    t_new = slice(times[t0+24], times[-1])   # later window, overlapping

    old = base.sel(time=t_old).copy(deep=True)
    new = base.sel(time=t_new).copy(deep=True)

    # Compute ordered overlap times between old and new
    old_times = old["time"].values
    new_times = new["time"].values
    overlap_times = np.intersect1d(old_times, new_times)
    if overlap_times.size < 2:
        pytest.skip("Not enough overlapping timestamps for complex merge tests.")

    # ------------------------------------------------------------------
    # Pick two locations
    # ------------------------------------------------------------------
    locs = _loc_names(xs)
    if len(locs) < 2:
        pytest.skip("Need at least two locations for complex merge tests.")
    l0, l1 = locs[0], locs[1]

    # ------------------------------------------------------------------
    # Remove data after two cut times at two locations in 'old' (set to NaN)
    # ------------------------------------------------------------------
    cut0 = pd.Timestamp(old["time"].values[-4])
    cut1 = pd.Timestamp(old["time"].values[-2])
    for loc, cut in ((l0, cut0), (l1, cut1)):
        mask_time = old["time"].values > np.datetime64(cut)
        if not mask_time.any():
            continue
        t_sel = old["time"].values[mask_time]
        for var in old.data_vars:
            if "time" in old[var].dims and "location" in old[var].dims:
                old[var].loc[dict(location=loc, time=t_sel)] = np.nan

    # ------------------------------------------------------------------
    # Remove one slope entirely from 'old'
    # ------------------------------------------------------------------
    if old.sizes.get("slope", 0) >= 3:
        # More robust: drop the last slope by its coordinate value
        last_slope_val = int(old["slope"].values[-1])
        keep_slopes = [s for s in old["slope"].values if int(s) != last_slope_val]
        old = old.sel(slope=keep_slopes)

    # ------------------------------------------------------------------
    # Remove another slope from 'old' only for one location
    # ------------------------------------------------------------------
    if old.sizes.get("slope", 0) >= 2:
        srm = int(old["slope"].values[0])
        for var in old.data_vars:
            if "slope" in old[var].dims and "location" in old[var].dims:
                old[var].loc[dict(location=l0, slope=srm)] = np.nan
        old['z'].loc[dict(location=l0, slope=srm)] = np.nan
        old['HS'].loc[dict(location=l0, slope=srm)] = np.nan

    # ------------------------------------------------------------------
    # In one overlap timestamp: remove several layers from 'old'
    # ------------------------------------------------------------------
    if old.sizes.get("layer", 0) >= 5:
        t_remove = overlap_times[0]
        rem_slice = slice(0, 3)
        for var in old.data_vars:
            if "layer" in old[var].dims and "time" in old[var].dims:
                old[var].loc[dict(time=t_remove, layer=rem_slice)] = np.nan

    # ------------------------------------------------------------------
    # In another overlap timestamp: modify the top layer in 'old'
    # ------------------------------------------------------------------
    if old.sizes.get("layer", 0) >= 1 and overlap_times.size >= 2:
        t_modify = overlap_times[1]
        slope_modify = np.arange(1, old.sizes['slope'])
        top_idx = old.sizes["layer"] - 1
        for var in old.data_vars:
            if "layer" in old[var].dims and "time" in old[var].dims:
                arr = old[var].sel(time=t_modify, slope=slope_modify)
                vals = arr.values
                if np.issubdtype(vals.dtype, np.number):
                    vals[..., top_idx] = 123.0
                    old[var].loc[dict(time=t_modify, slope=slope_modify)] = vals

    # ------------------------------------------------------------------
    # Set all non-null density values of 'old' to -999 during overlap
    # ------------------------------------------------------------------
    dens = old["density"]
    # restrict to overlap times where old actually has data
    overlap_in_old = np.intersect1d(dens["time"].values, overlap_times)
    if overlap_in_old.size > 0:
        dens_overlap = dens.sel(time=overlap_in_old)
        vals = dens_overlap.values
        mask_notnull = np.isfinite(vals)
        vals[mask_notnull] = -999
        dens.loc[dict(time=overlap_in_old)] = vals

    # ------------------------------------------------------------------
    # Drop profile_status and recompute
    # ------------------------------------------------------------------
    old, new = xsnowDataset(old), xsnowDataset(new)
    old = old.drop_vars("profile_status")
    old = old.update_profile_status()
    new = new.drop_vars("profile_status")
    new = new.update_profile_status()

    return old, new

def _assert_complex_merge_invariants(
    ds_old: Union[xsnowDataset, xr.Dataset],
    ds_new: Union[xsnowDataset, xr.Dataset],
) -> None:
    """
    Hard precondition checks for complex xsnow.combine tests.

    These assertions are *required* for the tests to be meaningful. If any of
    them fails, either the sample data layout changed, or the manipulations in
    _prep_two_mergable_datasets no longer do what the tests assume.

    Assumes the following `profile_status` semantics:

        < 0 : error / special flag (preserve as-is)
          0 : no usable vertical profile (no layer data)
          1 : valid vertical profile WITHOUT snow
          2 : valid vertical profile WITH snow
    """
    # Unwrap xsnowDataset to bare xr.Dataset if needed
    if isinstance(ds_old, xsnowDataset):
        ds_old = ds_old.data
    if isinstance(ds_new, xsnowDataset):
        ds_new = ds_new.data

    # --- required variables & core dims ---
    for label, ds in (("old", ds_old), ("new", ds_new)):
        assert "density" in ds.data_vars, f"{label} dataset must have variable 'density'"
        assert "profile_status" in ds.data_vars, f"{label} dataset must have 'profile_status'"

        for dim in ("time", "location", "slope", "layer", "realization"):
            assert dim in ds.dims, f"{label} dataset must have dimension {dim!r}"

    # --- overlapping times must exist and be non-trivial ---
    overlap = np.intersect1d(ds_old["time"].values, ds_new["time"].values)
    assert overlap.size >= 2, "Expected at least two overlapping timestamps."
    t_remove = overlap[0]

    # --- manipulations in _prep_two_mergable_datasets should have taken effect ---

    # 1) Old density in the overlap must contain -999 somewhere
    dens_old_overlap = ds_old["density"].sel(time=overlap)
    assert np.isfinite(dens_old_overlap.values).any(), (
        "Old density has no finite values in overlap; unexpected for test fixture."
    )
    assert (dens_old_overlap.values == -999).any(), (
        "Expected some -999 values in old density over overlap; "
        "looks like _prep_two_mergable_datasets did not apply the -999 overwrite."
    )

    # 2) New density should have at least some finite data in the overlap
    dens_new_overlap = ds_new["density"].sel(time=overlap)
    assert np.isfinite(dens_new_overlap.values).any(), (
        "New density has no finite values in overlap; combine tests would be meaningless."
    )

    # 3) At the first overlap time, the layered variable should have some NaNs in
    #    old *inside valid profiles* (partial layer deletions, no full-profile gap).
    layer_var = "density"
    old_at_t_remove = ds_old[layer_var].sel(time=t_remove)
    ps_old_t = ds_old["profile_status"].sel(time=t_remove)

    # Align & broadcast profile_status to the layered variable
    old_at_t_remove, ps_old_t = xr.align(old_at_t_remove, ps_old_t, join="inner")
    ps_old_b = ps_old_t.broadcast_like(old_at_t_remove)

    vals_old_t = old_at_t_remove.values
    ps_vals = ps_old_b.values

    # Valid profiles (status > 0)
    mask_valid_prof = ps_vals > 0
    # NaNs inside valid profiles
    mask_nan_valid = np.isnan(vals_old_t) & mask_valid_prof
    assert mask_nan_valid.any(), (
        "Expected some NaNs in old[density] at t_remove inside valid profiles; "
        "the 'remove several layers' manipulation may not have stuck."
    )
    # Also ensure there is some finite data inside valid profiles so they are not
    # degenerate "no-layer" profiles.
    assert (np.isfinite(vals_old_t) & mask_valid_prof).any(), (
        "Expected some finite density values inside valid profiles at t_remove."
    )

    # 4) Slope dropping: if new has more slopes than old, the last slope in new
    #    must not be present in old (this matches the 'drop last slope' logic).
    if ds_old.sizes.get("slope", 0) >= 1 and ds_new.sizes.get("slope", 0) > ds_old.sizes["slope"]:
        last_new = int(ds_new["slope"].values[-1])
        old_slopes = {int(s) for s in ds_old["slope"].values}
        assert last_new not in old_slopes, (
            "Expected last slope in 'new' to be absent in 'old' after slope-dropping."
        )

    # 5) Slope removed only for one location in 'old' should show:
    #    - no layer data in old (all NaN),
    #    - profile_status == 0 in old (no usable profile),
    #    - some finite density + profile_status > 0 in new.
    locs = list(ds_old["location"].values)
    if ds_old.sizes.get("slope", 0) >= 2 and len(locs) >= 1:
        l0 = locs[0]
        srm = ds_old["slope"].values[0]

        dens_old_gap = ds_old["density"].sel(location=l0, slope=srm, time=overlap)
        assert np.isnan(dens_old_gap.values).all(), (
            "Expected full NaN profiles in old at (location=l0, slope=srm) over overlap; "
            "looks like the 'remove slope for one location' manipulation changed."
        )

        ps_old_gap = ds_old["profile_status"].sel(location=l0, slope=srm, time=overlap)
        assert (ps_old_gap.values == 0).all(), (
            "Expected profile_status == 0 where there is no layer data "
            "for the removed slope at one location."
        )

        ps_new_gap = ds_new["profile_status"].sel(location=l0, slope=srm, time=overlap)
        assert (ps_new_gap.values > 0).any(), (
            "Expected new to have valid profiles (profile_status > 0) "
            "for the removed slope/location."
        )

@pytest.mark.real_data
def test_combine_prefer_old_left_join(snp_gridded_xs):
    """
    Prefer old data, fill gaps from new, domain from old (join='left').

    We test that:
    - partial layer deletions in old are NOT refilled from new (no layer mixing)
      inside valid profiles (profile_status > 0),
    - full-profile gaps (profile_status==0) in old are filled from new,
    - density values overwritten to -999 in old remain -999,
    - a slope dropped entirely from old does NOT appear in the combined dataset,
    - a slope removed only for one location in old is restored from new,
    - timestamps only in new (after overlap) do NOT appear in combined.
    """
    old, new = _prep_two_mergable_datasets(snp_gridded_xs)
    _assert_complex_merge_invariants(old, new)

    combined = xsnow.combine([old, new], join="left")
    ds_old, ds_new, ds_comb = old.data, new.data, combined.data

    # --- basic domain checks: left join => domain from old on core dims ---
    assert set(ds_comb.time.values) == set(ds_old.time.values)
    assert set(ds_comb.location.values) == set(ds_old.location.values)
    assert set(ds_comb.slope.values) == set(ds_old.slope.values)
    assert set(ds_comb.realization.values) == set(ds_old.realization.values)

    # --- overlap times and layered variable ---
    overlap_times = np.intersect1d(ds_old.time.values, ds_new.time.values)
    assert overlap_times.size >= 2
    t_remove = overlap_times[0]
    t_modify = overlap_times[1]

    layer_var = "density"
    nlayer = ds_comb.sizes["layer"]
    top_idx = nlayer - 1

    # 1) Several layers removed in old at t_remove should stay NaN (no refill)
    #    inside valid profiles (profile_status > 0).
    old_da = ds_old[layer_var].sel(time=t_remove)
    new_da = ds_new[layer_var].sel(time=t_remove)
    comb_da = ds_comb[layer_var].sel(time=t_remove)
    ps_old = ds_old["profile_status"].sel(time=t_remove)

    # Align everything
    old_da, new_da, comb_da, ps_old = xr.align(
        old_da, new_da, comb_da, ps_old, join="inner"
    )

    ps_b = ps_old.broadcast_like(old_da)
    old_vals = old_da.values
    new_vals = new_da.values
    comb_vals = comb_da.values
    ps_vals = ps_b.values

    mask_valid_prof = ps_vals > 0

    nan_mask = np.isnan(old_vals) & mask_valid_prof
    assert nan_mask.any(), "Expected some NaNs in old at t_remove inside valid profiles"

    # Combined should still have NaNs where old had NaNs *inside valid profiles*
    assert np.all(np.isnan(comb_vals[nan_mask]))

    # If new has data there, ensure it was NOT used (no layer mixing)
    if np.any(~np.isnan(new_vals[nan_mask])):
        assert not np.allclose(
            comb_vals[nan_mask],
            new_vals[nan_mask],
            equal_nan=True,
        )

    # Where old had data inside valid profiles, combined must match old
    mask_old_data = mask_valid_prof & ~np.isnan(old_vals)
    if mask_old_data.any():
        assert np.allclose(
            comb_vals[mask_old_data],
            old_vals[mask_old_data],
            equal_nan=True,
        )

    # 2) At t_modify, combined should still follow old *inside valid old profiles*,
    #    and top layer density should be -999 wherever finite.
    old_da2 = ds_old[layer_var].sel(time=t_modify)
    comb_da2 = ds_comb[layer_var].sel(time=t_modify)
    ps_old2 = ds_old["profile_status"].sel(time=t_modify)

    old_da2, comb_da2, ps_old2 = xr.align(old_da2, comb_da2, ps_old2, join="inner")

    ps2_b = ps_old2.broadcast_like(old_da2)
    mask_valid2 = ps2_b.values > 0

    old_top = old_da2.isel(layer=top_idx)
    comb_top = comb_da2.isel(layer=top_idx)

    finite_old = np.isfinite(old_top.values) & (ps2_b.isel(layer=top_idx).values > 0)
    if finite_old.any():
        # by construction in _prep_two_mergable_datasets
        assert np.all(old_top.values[finite_old] == -999)
        assert np.all(comb_top.values[finite_old] == -999)

    # More generally: within valid profiles, combined must follow old
    if mask_valid2.any():
        np.testing.assert_allclose(
            comb_da2.values[mask_valid2],
            old_da2.values[mask_valid2],
        )

    # 3) density values -999 in old remain -999 in combined over overlap
    dens_old = ds_old["density"].sel(time=overlap_times)
    dens_comb = ds_comb["density"].sel(time=overlap_times)

    # Align so dims (e.g., slope) match
    dens_old, dens_comb = xr.align(dens_old, dens_comb, join="inner")

    mask_999 = dens_old.values == -999
    assert mask_999.any(), "Expected some -999 in old density over overlap."
    assert np.all(dens_comb.values[mask_999] == -999)

    # 4) Slope dropped entirely from old does NOT appear in combined
    all_slopes_new = list(ds_new.slope.values)
    all_slopes_old = list(ds_old.slope.values)
    all_slopes_comb = list(ds_comb.slope.values)
    if len(all_slopes_new) > len(all_slopes_old):
        # Last slope present only in new
        last_slope_new = all_slopes_new[-1]
        assert last_slope_new not in all_slopes_old
        assert last_slope_new not in all_slopes_comb
    # And slopes in combined should exactly match old
    assert set(all_slopes_comb) == set(all_slopes_old)

    # 5) For the slope removed only for one location in old, combined can use new
    locs = list(ds_old.location.values)
    if ds_old.sizes.get("slope", 0) >= 1 and len(locs) >= 1:
        l0 = locs[0]
        srm = ds_old.slope.values[0]  # this is the one masked in _prep
        t_any = overlap_times[0]

        old_profile = ds_old[layer_var].sel(location=l0, slope=srm, time=t_any)
        new_profile = ds_new[layer_var].sel(location=l0, slope=srm, time=t_any)
        comb_profile = ds_comb[layer_var].sel(location=l0, slope=srm, time=t_any)

        # old is a gap here by construction
        assert np.isnan(old_profile.values).all(), "Expected full gap in old"
        has_new_data = np.any(np.isfinite(new_profile.values))
        if has_new_data:
            # When new has data, combined is allowed to use it
            assert np.allclose(
                comb_profile.values,
                new_profile.values,
                equal_nan=True,
            )
    # If new is also all-NaN, there's nothing to fill; no further assertion needed.

    # 6) Timestamps only in new (after overlap) must NOT appear with join='left'
    only_in_new_times = set(ds_new.time.values) - set(ds_old.time.values)
    assert all(t not in ds_comb.time.values for t in only_in_new_times)


@pytest.mark.real_data
def test_combine_prefer_old_outer_join(snp_gridded_xs):
    """
    Prefer old data, fill gaps from new, domain is union (join='outer').

    Additional expectations vs left-join:
    - combined has union of timestamps from old and new,
    - combined has all slopes from both old and new,
    - partial layer deletions in old are still NOT patched from new inside valid profiles,
    - density -999 in old kept, etc.
    """
    old, new = _prep_two_mergable_datasets(snp_gridded_xs)
    _assert_complex_merge_invariants(old, new)

    combined = xsnow.combine([old, new], join="outer")
    ds_old, ds_new, ds_comb = old.data, new.data, combined.data

    # domain checks...
    assert set(ds_comb.time.values) == set(ds_old.time.values) | set(ds_new.time.values)
    assert set(ds_comb.location.values) == set(ds_old.location.values) | set(ds_new.location.values)
    assert set(ds_comb.realization.values) == set(ds_old.realization.values) | set(ds_new.realization.values)
    assert set(ds_comb.slope.values) == set(ds_old.slope.values) | set(ds_new.slope.values)

    layer_var = "density"

    overlap_times = np.intersect1d(ds_old.time.values, ds_new.time.values)
    assert overlap_times.size >= 2
    t_remove = overlap_times[0]
    t_modify = overlap_times[1]
    nlayer = ds_comb.sizes["layer"]
    top_idx = nlayer - 1

    # 1) Partial layer deletions in old stay NaN inside valid profiles
    old_da = ds_old[layer_var].sel(time=t_remove)
    new_da = ds_new[layer_var].sel(time=t_remove)
    comb_da = ds_comb[layer_var].sel(time=t_remove)
    ps_old = ds_old["profile_status"].sel(time=t_remove)

    old_da, new_da, comb_da, ps_old = xr.align(
        old_da, new_da, comb_da, ps_old, join="inner"
    )

    ps_b = ps_old.broadcast_like(old_da)
    old_vals = old_da.values
    new_vals = new_da.values
    comb_vals = comb_da.values
    ps_vals = ps_b.values

    mask_valid_prof = ps_vals > 0
    nan_mask = np.isnan(old_vals) & mask_valid_prof
    if nan_mask.any():
        # combined must keep NaNs where old had NaNs
        assert np.all(np.isnan(comb_vals[nan_mask]))

        # if new has any valid values there, assert they were NOT used
        if np.any(~np.isnan(new_vals[nan_mask])):
            assert not np.allclose(
                comb_vals[nan_mask],
                new_vals[nan_mask],
                equal_nan=True,
            )
        mask_old_data = mask_valid_prof & ~np.isnan(old_vals)
        if mask_old_data.any():
            assert np.allclose(
                comb_vals[mask_old_data],
                old_vals[mask_old_data],
                equal_nan=True,
            )

    # 2) At t_modify, combined should still follow old inside valid profiles.
    old_da2 = ds_old[layer_var].sel(time=t_modify)
    comb_da2 = ds_comb[layer_var].sel(time=t_modify)
    ps_old2 = ds_old["profile_status"].sel(time=t_modify)

    old_da2, comb_da2, ps_old2 = xr.align(old_da2, comb_da2, ps_old2, join="inner")

    ps2_b = ps_old2.broadcast_like(old_da2)
    mask_valid2 = ps2_b.values > 0

    old_top = old_da2.isel(layer=top_idx)
    comb_top = comb_da2.isel(layer=top_idx)

    finite_old = np.isfinite(old_top.values) & (ps2_b.isel(layer=top_idx).values > 0)
    if finite_old.any():
        assert np.all(old_top.values[finite_old] == -999)
        assert np.all(comb_top.values[finite_old] == -999)

    if mask_valid2.any():
        np.testing.assert_allclose(
            comb_da2.values[mask_valid2],
            old_da2.values[mask_valid2],
        )

    # 3) density -999 in old kept in overlap region
    dens_old = ds_old["density"].sel(time=overlap_times)
    dens_comb = ds_comb["density"].sel(time=overlap_times)

    # Align so we compare only common coords (e.g. slopes)
    dens_old, dens_comb = xr.align(dens_old, dens_comb, join="inner")
    mask_999 = dens_old.values == -999
    if mask_999.any():
        assert np.all(dens_comb.values[mask_999] == -999)

    # 4) slope only in new present in combined
    all_slopes_new = list(ds_new.slope.values)
    all_slopes_old = list(ds_old.slope.values)
    all_slopes_comb = list(ds_comb.slope.values)
    if len(all_slopes_new) > len(all_slopes_old):
        extra_slope = all_slopes_new[-1]
        assert extra_slope not in all_slopes_old
        assert extra_slope in all_slopes_comb

    # 5) slope removed only for one location in old can be filled from new
    locs = list(ds_old.location.values)
    if ds_old.sizes.get("slope", 0) >= 1 and len(locs) >= 1:
        l0 = locs[0]
        srm = ds_old.slope.values[0]
        t_any = overlap_times[0]

        old_profile = ds_old[layer_var].sel(location=l0, slope=srm, time=t_any)
        new_profile = ds_new[layer_var].sel(location=l0, slope=srm, time=t_any)
        comb_profile = ds_comb[layer_var].sel(location=l0, slope=srm, time=t_any)

        if np.isnan(old_profile.values).all():
            has_new_data = np.any(np.isfinite(new_profile.values))
            if has_new_data:
                # combined is allowed to follow new here
                assert np.allclose(
                    comb_profile.values,
                    new_profile.values,
                    equal_nan=True,
                )
            # If new is also NaN, nothing to assert.


@pytest.mark.real_data
@pytest.mark.parametrize("join_mode", ["left", "right", "inner", "outer"])
def test_combine_prefer_old_other_joins_basic_invariants(snp_gridded_xs, join_mode):
    """
    Smoke-test other joins (left/right/inner/outer) in the prefer-old ordering
    for basic domain invariants and check that profile_status is present and aligned.

    Also implicitly checks that the input datasets satisfy the complex-merge
    invariants (including the updated profile_status semantics).
    """
    old, new = _prep_two_mergable_datasets(snp_gridded_xs)
    _assert_complex_merge_invariants(old, new)
    combined = xsnow.combine([old, new], join=join_mode)
    ds_old, ds_new, ds_comb = old.data, new.data, combined.data

    # profile_status should exist and have core dims consistent with combined
    assert "profile_status" in ds_comb
    ps = ds_comb["profile_status"]
    for dim in ("location", "time", "slope", "realization"):
        if dim in ds_comb.dims:
            assert dim in ps.dims
            assert ps.sizes[dim] == ds_comb.sizes[dim]

    # Basic sanity: profile_status codes should be within {-1, 0, 1}
    assert np.isin(ps.values, [-1, 0, 1]).all()

    # Domain invariants
    if join_mode == "left":
        assert set(ds_comb.time.values) == set(ds_old.time.values)
    elif join_mode == "right":
        assert set(ds_comb.time.values) == set(ds_new.time.values)
    elif join_mode == "inner":
        assert set(ds_comb.time.values) == (
            set(ds_old.time.values) & set(ds_new.time.values)
        )
    elif join_mode == "outer":
        assert set(ds_comb.time.values) == (
            set(ds_old.time.values) | set(ds_new.time.values)
        )


# --------------------------------------------------------------------
# Case 2: prefer new (overwrite old where new has data)
# --------------------------------------------------------------------


@pytest.mark.real_data
def test_combine_prefer_new_overwrite_modes(snp_gridded_xs):
    """
    Prefer new data (leftmost), overwrite old where both have valid profiles
    using different join modes.

    We check that:
    - on overlaps, new overwrites old (density and modified layers),
    - earlier-only times (in old) still come from old when join!='left',
    - join modes control the domain on time as expected.
    """
    old, new = _prep_two_mergable_datasets(snp_gridded_xs)
    _assert_complex_merge_invariants(old, new)

    ds_old, ds_new = old.data, new.data

    # overlap for checks
    overlap_times = np.intersect1d(ds_old.time.values, ds_new.time.values)
    assert overlap_times.size >= 2
    t_modify = overlap_times[1]

    layer_var = "density"

    # convenience to check density and layered vars under different joins
    def _run(join_mode: str):
        combined = xsnow.combine([new, old], join=join_mode)
        return combined.data

    # LEFT JOIN: domain = new, prefer new
    ds_left = _run("left")
    assert set(ds_left.time.values) == set(ds_new.time.values)

    # On overlap, density should match new, not old (-999), wherever new has
    # a valid profile (profile_status > 0).
    dens_old = ds_old["density"].sel(time=overlap_times)
    dens_new = ds_new["density"].sel(time=overlap_times)
    dens_left = ds_left["density"].sel(time=overlap_times)

    ps_old_overlap = ds_old["profile_status"].sel(time=overlap_times)
    ps_new_overlap = ds_new["profile_status"].sel(time=overlap_times)

    # Align on intersection of coords so masks and data align
    dens_old, dens_new, dens_left, ps_old_overlap, ps_new_overlap = xr.align(
        dens_old, dens_new, dens_left, ps_old_overlap, ps_new_overlap, join="inner"
    )

    vals_old = dens_old.values
    vals_new = dens_new.values
    vals_left = dens_left.values

    ps_new_b = ps_new_overlap.broadcast_like(dens_new).values
    ps_old_b = ps_old_overlap.broadcast_like(dens_old).values

    # Only consider positions where new has a valid profile
    mask_new_valid_prof = ps_new_b > 0

    # Wherever old == -999 and new has finite data in a valid profile, left
    # should follow new (overwrite the -999).
    mask_999_and_new = (vals_old == -999) & np.isfinite(vals_new) & mask_new_valid_prof
    if mask_999_and_new.any():
        assert np.allclose(
            vals_left[mask_999_and_new],
            vals_new[mask_999_and_new],
        )

    # At t_modify, we want to assert *profile-level* semantics (no layer mixing):
    # - where new has a valid profile (profile_status > 0), left should take the
    #   entire profile from new (all layers),
    # - where new has no valid profile but old does, left should take the entire
    #   profile from old,
    # - if both are invalid, we don't constrain the result.
    if "layer" in ds_left[layer_var].dims:
        var_old = ds_old[layer_var].sel(time=t_modify)
        var_new = ds_new[layer_var].sel(time=t_modify)
        var_left = ds_left[layer_var].sel(time=t_modify)

        ps_old = ds_old["profile_status"].sel(time=t_modify)
        ps_new = ds_new["profile_status"].sel(time=t_modify)

        # Align everything on common core coords (location, slope, realization)
        var_old, var_new, var_left, ps_old, ps_new = xr.align(
            var_old, var_new, var_left, ps_old, ps_new, join="inner"
        )

        mask_new_valid = (ps_new > 0)
        mask_only_old_valid = (ps_old > 0) & (ps_new == 0)

        if bool(mask_new_valid.any()):
            xr.testing.assert_allclose(
                var_left.where(mask_new_valid, drop=True),
                var_new.where(mask_new_valid, drop=True),
            )

        if bool(mask_only_old_valid.any()):
            xr.testing.assert_allclose(
                var_left.where(mask_only_old_valid, drop=True),
                var_old.where(mask_only_old_valid, drop=True),
            )

    # RIGHT JOIN: domain = old, prefer new on overlaps
    ds_right = _run("right")
    assert set(ds_right.time.values) == set(ds_old.time.values)

    # times only in old (before new) must still come from old (value-wise)
    only_old = sorted(set(ds_old.time.values) - set(ds_new.time.values))
    if only_old:
        t0 = only_old[0]

        da_right = ds_right[layer_var].sel(time=t0)
        da_old = ds_old[layer_var].sel(time=t0)
        ps_old_t0 = ds_old["profile_status"].sel(time=t0)

        da_right, da_old, ps_old_t0 = xr.align(
            da_right, da_old, ps_old_t0, join="inner"
        )

        ps_b = ps_old_t0.broadcast_like(da_old)
        mask_valid = ps_b.values > 0

        # Only require equality where old has a valid profile; for status==0
        # we don't constrain (combine may treat it as no layer data).
        if mask_valid.any():
            np.testing.assert_allclose(
                da_right.values[mask_valid],
                da_old.values[mask_valid],
            )

    # INNER JOIN: domain = intersection, prefer new
    ds_inner = _run("inner")
    assert set(ds_inner.time.values) == set(overlap_times)
    # On overlap, wherever new has finite density in a valid profile,
    # inner result should match new.
    dens_inner = ds_inner["density"].sel(time=overlap_times)
    dens_new_sel = ds_new["density"].sel(time=overlap_times)
    ps_new_sel = ds_new["profile_status"].sel(time=overlap_times)

    dens_inner_a, dens_new_a, ps_new_a = xr.align(
        dens_inner, dens_new_sel, ps_new_sel, join="inner"
    )

    vals_inner = dens_inner_a.values
    vals_new = dens_new_a.values
    ps_new_b = ps_new_a.broadcast_like(dens_new_a).values

    mask_new_valid = np.isfinite(vals_new) & (ps_new_b > 0)
    if mask_new_valid.any():
        assert np.allclose(
            vals_inner[mask_new_valid],
            vals_new[mask_new_valid],
        )

    # OUTER JOIN: domain = union, prefer new
    ds_outer = _run("outer")
    assert set(ds_outer.time.values) == (
        set(ds_old.time.values) | set(ds_new.time.values)
    )

    # On overlap, wherever new has finite density in a valid profile,
    # outer result should match new.
    dens_outer = ds_outer["density"].sel(time=overlap_times)
    dens_new_sel = ds_new["density"].sel(time=overlap_times)
    ps_new_sel = ds_new["profile_status"].sel(time=overlap_times)

    dens_outer_a, dens_new_a, ps_new_a = xr.align(
        dens_outer, dens_new_sel, ps_new_sel, join="inner"
    )

    vals_outer = dens_outer_a.values
    vals_new = dens_new_a.values
    ps_new_b = ps_new_a.broadcast_like(dens_new_a).values

    mask_new_valid = np.isfinite(vals_new) & (ps_new_b > 0)
    if mask_new_valid.any():
        assert np.allclose(
            vals_outer[mask_new_valid],
            vals_new[mask_new_valid],
        )
    # Where new is NaN or has no valid profile, we don't constrain what combine did
    # (old may have -999 or NaNs).


# --------------------------------------------------------------------
# Convenience methods: fill_gaps_with & overwrite_with
# --------------------------------------------------------------------


@pytest.mark.real_data
def test_fill_gaps_with_is_combine_left(snp_gridded_xs):
    """
    fill_gaps_with(other) must be equivalent to combine([self, other], join='left').
    """
    old, new = _prep_two_mergable_datasets(snp_gridded_xs)

    combined_fn = old.fill_gaps_with(new)
    combined_direct = xsnow.combine([old, new], join="left")

    xr.testing.assert_identical(combined_fn.data, combined_direct.data)


@pytest.mark.real_data
def test_overwrite_with_is_combine_outer_prefer_other(snp_gridded_xs):
    """
    overwrite_with(other) must be equivalent to combine([other, self], join='outer').
    """
    old, new = _prep_two_mergable_datasets(snp_gridded_xs)

    combined_fn = old.overwrite_with(new)
    combined_direct = xsnow.combine([new, old], join="outer")

    xr.testing.assert_identical(combined_fn.data, combined_direct.data)


@pytest.mark.synthetic
def test_xsnowdataset_update_raises():
    ds = xsnowDataset(_make_simple_scalar_ds(1.0, time=0))
    with pytest.raises(NotImplementedError):
        ds.update(None)


@pytest.mark.synthetic
def test_xsnowdataset_combine_first_raises():
    ds = xsnowDataset(_make_simple_scalar_ds(1.0, time=0))
    with pytest.raises(NotImplementedError):
        ds.combine_first(None)




# =========================================================================
# xsnow.combine behaviors on layered + non-layered xsnowDatasets
# =========================================================================

@pytest.fixture(scope="session")
def snp_gridded_smets_xs() -> xsnow.xsnowDataset:
    """
    Gridded SMET nowcast inputs corresponding to snp_gridded_xs.

    This is a non-layered dataset (no 'layer' dim) that contains forcing / meta
    variables such as TA, HS, etc., on the same spatial grid as snp_gridded_xs.
    """
    path = f"{xsnow.sample_data.snp_gridded_dir()}/smets/gridded/nowcast"
    return xsnow.read(path)


# 1) Augment layered dataset with non-layered SMET inputs
@pytest.mark.real_data
@pytest.mark.parametrize("join_mode", ["outer", "left", "right", "inner"])
def test_combine_augment_with_smets_preserves_layers_and_adds_TA(
    snp_gridded_xs: xsnow.xsnowDataset,
    snp_gridded_smets_xs: xsnow.xsnowDataset,
    join_mode: str,
):
    """
    Combine layered snp_gridded_xs with non-layered SMET inputs:

    - Drop TA (and a few other met vars) from the layered dataset.
    - Subset SMETs to a single location (VIR2A) so it contributes only there.
    - Combine with different join modes.

    Expectations
    ------------
    * Layered variables (e.g. density) at VIR2A are identical to snp_gridded_xs
      after combine (no layer mixing / no changes from SMETs).
    * TA at VIR2A in the combined dataset matches the SMET dataset (augmentation).
    * HS in the combined dataset at VIR2A comes from the layered dataset, not
      from SMETs, wherever the layered profile is valid (profile_status > 0).
    """
    xs_layered = snp_gridded_xs
    xs_smets = snp_gridded_smets_xs

    # Sanity: need at least one of the met vars present to drop later
    met_vars = ["TA", "RH", "VW", "DW", "ISWR"]
    have_any_met = any(v in xs_layered.data_vars for v in met_vars)
    if not have_any_met:
        pytest.skip(
            "Layered dataset does not contain any of the expected met variables."
        )

    # Pick locations: we expect VIR1A and VIR2A to exist, but guard with skips
    locs_layer = [str(l) for l in xs_layered.data["location"].values]
    if "VIR2A" not in locs_layer:
        pytest.skip("Location 'VIR2A' not found in layered dataset.")
    if "VIR1A" not in locs_layer:
        pytest.skip("Location 'VIR1A' not found in layered dataset.")

    locs_smets = [str(l) for l in xs_smets.data["location"].values]
    if "VIR2A" not in locs_smets:
        pytest.skip("Location 'VIR2A' not found in SMET dataset.")

    # ------------------------------------------------------------------
    # 1) Build a modified layered dataset:
    #    - drop VIR1A entirely
    #    - drop some met vars so SMETs can provide them uniquely
    # ------------------------------------------------------------------
    base_ds = xs_layered.data
    ds_layered_mod = base_ds.sel(location=[l for l in locs_layer if l != "VIR1A"])
    ds_layered_mod = ds_layered_mod.drop_vars(
        [v for v in met_vars if v in ds_layered_mod.data_vars]
    )

    xs_layered_mod = xsnow.xsnowDataset(ds_layered_mod)

    # ------------------------------------------------------------------
    # 2) Subset SMETs to VIR2A only, but KEEP the location dimension
    # ------------------------------------------------------------------
    ds_smets_sub = xs_smets.data.sel(location=["VIR2A"])
    xs_smets_sub = xsnow.xsnowDataset(ds_smets_sub)

    # ------------------------------------------------------------------
    # 3) Combine layered + SMETs
    # ------------------------------------------------------------------
    combined = xsnow.combine([xs_layered_mod, xs_smets_sub], join=join_mode)
    ds_comb = combined.data

    # ----------------------------
    # a) Layers at VIR2A unchanged
    # ----------------------------
    if "density" in xs_layered_mod.data_vars:
        layer_var = "density"
    else:
        candidates = [
            n for n, v in xs_layered_mod.data_vars.items() if "layer" in v.dims
        ]
        if not candidates:
            pytest.skip("No layered variable available in layered dataset.")
        layer_var = candidates[0]

    old_layer = xs_layered_mod.data[layer_var].sel(location="VIR2A")
    new_layer = ds_comb[layer_var].sel(location="VIR2A")

    # Align on common coords (time, slope, realization, layer)
    old_layer_a, new_layer_a = xr.align(old_layer, new_layer, join="inner")
    xr.testing.assert_allclose(new_layer_a, old_layer_a)

    # ----------------------------
    # b) TA at VIR2A matches SMETs
    # ----------------------------
    if "TA" not in xs_smets_sub.data_vars:
        pytest.skip("SMET dataset does not contain TA variable.")

    ta_smets = xs_smets_sub.data["TA"].sel(location="VIR2A")
    ta_comb = ds_comb["TA"].sel(location="VIR2A")

    ta_smets_a, ta_comb_a = xr.align(ta_smets, ta_comb, join="inner")
    
    # We only require that the *values* of TA match the SMETs input.
    # Coordinate metadata on the slope dimension (inclination, azimuth, etc.)
    # may come from whichever dataset is on the left of the combine call.
    np.testing.assert_allclose(ta_comb_a.values, ta_smets_a.values)

    # ----------------------------
    # c) HS at VIR2A comes from the layered dataset
    # ----------------------------
    if "HS" in xs_layered_mod.data_vars and "HS" in ds_comb.data_vars:
        hs_layered = xs_layered_mod.data["HS"].sel(location="VIR2A")

        # profile_status from layered dataset (recompute if needed)
        if "profile_status" in xs_layered_mod.data_vars:
            ps_layered = xs_layered_mod.data["profile_status"].sel(location="VIR2A")
        else:
            ps_layered = (
                xs_layered_mod.update_profile_status()
                .data["profile_status"]
                .sel(location="VIR2A")
            )

        hs_comb = ds_comb["HS"].sel(location="VIR2A")

        hs_layered_a, ps_layered_a, hs_comb_a = xr.align(
            hs_layered, ps_layered, hs_comb, join="inner"
        )

        mask_valid = ps_layered_a > 0
        if bool(mask_valid.any()):
            np.testing.assert_allclose(
                hs_comb_a.values[mask_valid.values],
                hs_layered_a.values[mask_valid.values],
            )


# 2) Precedence & gap-filling for non-layer variables (TA),
#    HS consistency w.r.t. layered dataset
@pytest.mark.real_data
@pytest.mark.parametrize("join_mode", ["outer", "left", "right", "inner"])
def test_combine_nonlayer_TA_precedence_and_HS_consistency(
    snp_gridded_xs: xsnow.xsnowDataset,
    snp_gridded_smets_xs: xsnow.xsnowDataset,
    join_mode: str,
):
    """
    Check prefer-left semantics and NaN-filling for non-layer variables, and
    ensure HS stays consistent with the layered dataset.

    Setup
    -----
    - Modify TA in both snp_gridded_xs and SMETs on a shared
      (location, time[, slope, realization]) index:
        * t0: left TA = 999, right TA = 42      (both finite)
        * t1: left TA = NaN, right TA = 42      (gap to be filled)
    - Combine in both orders: [left, right] and [right, left].

    Expectations
    ------------
    - combine([left, right], ...) : prefer left where finite, otherwise fill
      from right:
        * at t0 → 999
        * at t1 → 42
    - combine([right, left], ...) : same rule, but with swapped roles:
        * at t0 → 42
        * at t1 → 999
    - Layered variables (e.g. density) are identical to the original
      layered dataset in both combined outputs (no layer mixing).
    - HS in both combined outputs comes from the layered dataset wherever
      the original profile_status > 0.
    """
    xs_layered = snp_gridded_xs
    xs_smets = snp_gridded_smets_xs

    if "TA" not in xs_layered.data_vars or "TA" not in xs_smets.data_vars:
        pytest.skip("Both datasets must contain TA for this precedence test.")

    # Choose a common location and two overlapping times
    locs_layer = set(str(l) for l in xs_layered.data["location"].values)
    locs_smets = set(str(l) for l in xs_smets.data["location"].values)
    common_locs = sorted(locs_layer & locs_smets)
    if not common_locs:
        pytest.skip("No common locations between layered and SMET datasets.")
    loc = common_locs[0]

    times_layer = set(xs_layered.data["time"].values)
    times_smets = set(xs_smets.data["time"].values)
    overlap_times = sorted(times_layer & times_smets)
    if len(overlap_times) < 2:
        pytest.skip("Need at least two overlapping timestamps for TA precedence test.")

    t0, t1 = overlap_times[:2]

    # Pick first slope/realization indices where present
    dsL = xs_layered.data
    dsS = xs_smets.data

    sel_common = {"location": loc}
    for dim in ("slope", "realization"):
        if dim in dsL.dims and dsL.sizes[dim] > 0:
            sel_common[dim] = dsL.coords[dim].values[0]
        elif dim in dsS.dims and dsS.sizes[dim] > 0:
            sel_common[dim] = dsS.coords[dim].values[0]

    # Create modified copies
    dsL_mod = dsL.copy(deep=True)
    dsS_mod = dsS.copy(deep=True)

    # Helper to build selection dict for a given time
    def _sel_for_time(t):
        d = dict(sel_common)
        d["time"] = t
        return d

    # t0: both finite, different values
    dsL_mod["TA"].loc[_sel_for_time(t0)] = 999.0
    dsS_mod["TA"].loc[_sel_for_time(t0)] = 42.0

    # t1: left NaN, right finite
    dsL_mod["TA"].loc[_sel_for_time(t1)] = np.nan
    dsS_mod["TA"].loc[_sel_for_time(t1)] = 42.0

    xsL_mod = xsnow.xsnowDataset(dsL_mod)
    xsS_mod = xsnow.xsnowDataset(dsS_mod)

    # Combine in both orders with the same join mode
    comb_LR = xsnow.combine([xsL_mod, xsS_mod], join=join_mode)
    comb_RL = xsnow.combine([xsS_mod, xsL_mod], join=join_mode)

    ds_LR = comb_LR.data
    ds_RL = comb_RL.data

    # Build selection dicts for reading back TA
    def _sel(ds, t):
        d = {"location": loc, "time": t}
        ta = ds["TA"]
        for dim in ("slope", "realization"):
            if dim in ta.dims:
                d[dim] = sel_common[dim]
        return d

    # Prefer-left semantics with NaN fill: [left, right]
    val_LR_t0 = float(ds_LR["TA"].sel(_sel(ds_LR, t0)))
    val_LR_t1 = float(ds_LR["TA"].sel(_sel(ds_LR, t1)))
    assert val_LR_t0 == pytest.approx(999.0)
    assert val_LR_t1 == pytest.approx(42.0)

    # Same rule, swapped roles: [right, left]
    val_RL_t0 = float(ds_RL["TA"].sel(_sel(ds_RL, t0)))
    val_RL_t1 = float(ds_RL["TA"].sel(_sel(ds_RL, t1)))
    assert val_RL_t0 == pytest.approx(42.0)
    assert val_RL_t1 == pytest.approx(42.0)

    # ----------------------------
    # Layers unchanged in both combined outputs
    # ----------------------------
    if "density" in xs_layered.data_vars:
        layer_var = "density"
    else:
        candidates = [
            n for n, v in xs_layered.data_vars.items() if "layer" in v.dims
        ]
        if not candidates:
            pytest.skip("No layered variable available for layer-invariance check.")
        layer_var = candidates[0]

    dens_orig = xs_layered.data[layer_var]
    dens_LR = ds_LR[layer_var]
    dens_RL = ds_RL[layer_var]

    dens_orig_a, dens_LR_a = xr.align(dens_orig, dens_LR, join="inner")
    dens_orig_b, dens_RL_b = xr.align(dens_orig, dens_RL, join="inner")

    # LR: we expect a fully “xsnow-native” result, including coords
    xr.testing.assert_allclose(dens_LR_a, dens_orig_a)

    # RL: only require that the *values* of the layered variable are unchanged.
    # Coordinate metadata (inclination, azimuth, etc.) may differ depending
    # on which dataset appears on the left during combine.
    np.testing.assert_allclose(dens_RL_b.values, dens_orig_b.values)

    # ----------------------------
    # HS always comes from the layered dataset wherever it has valid profiles
    # ----------------------------
    if "HS" in xs_layered.data_vars:
        hs_orig = xs_layered.data["HS"]
        # profile_status from layered dataset (recompute if needed)
        if "profile_status" in xs_layered.data_vars:
            ps_orig = xs_layered.data["profile_status"]
        else:
            ps_orig = xs_layered.update_profile_status().data["profile_status"]

        for ds_comb in (ds_LR, ds_RL):
            if "HS" not in ds_comb.data_vars:
                continue

            hs_comb = ds_comb["HS"]

            hs_orig_a, ps_orig_a, hs_comb_a = xr.align(
                hs_orig, ps_orig, hs_comb, join="inner"
            )

            mask_valid = ps_orig_a > 0
            if bool(mask_valid.any()):
                np.testing.assert_allclose(
                    hs_comb_a.values[mask_valid.values],
                    hs_orig_a.values[mask_valid.values],
                )
