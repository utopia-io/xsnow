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


# =========================================================================
# xsnow.concat tests
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
def test_concat_realization_outer(snp_snowobs_sim_xs, snp_snowobs_obs_xs):
    """
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
    base = xs.data
    times = base["time"].values
    assert times.size >= 48, "Need at least 48 timestamps to create overlap windows."

    # Define old/new windows (index-based, ensures overlap)
    t0 = int((base.profile_status >= 1).any(dim=["slope", "realization"])\
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
        top_idx = old.sizes["layer"] - 1
        for var in old.data_vars:
            if "layer" in old[var].dims and "time" in old[var].dims:
                arr = old[var].sel(time=t_modify)
                vals = arr.values
                if np.issubdtype(vals.dtype, np.number):
                    vals[..., top_idx] = 123.0
                    old[var].loc[dict(time=t_modify)] = vals

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

def _assert_complex_merge_invariants(ds_old: Union[xsnowDataset, xr.Dataset], 
                                     ds_new: Union[xsnowDataset, xr.Dataset]) -> None:
    """
    Hard precondition checks for complex xsnow.combine tests.

    These assertions are *required* for the tests to be meaningful. If any of
    them fails, either the sample data layout changed, or the manipulations in
    _prep_two_mergable_datasets no longer do what the tests assume.
    """
    # --- required variables ---
    for label, ds in (("old", ds_old), ("new", ds_new)):
        assert "density" in ds.data_vars, f"{label} dataset must have variable 'density'"
        assert "profile_status" in ds.data_vars, f"{label} dataset must have 'profile_status'"

        # --- required core dims ---
        for dim in ("time", "location", "slope", "layer", "realization"):
            assert dim in ds.dims, f"{label} dataset must have dimension {dim!r}"

    # --- overlapping times must exist and be non-trivial ---
    overlap = np.intersect1d(ds_old["time"].values, ds_new["time"].values)
    assert overlap.size >= 2, "Expected at least two overlapping timestamps."

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

    # 3) At the first overlap time, the layered variable should have some NaNs in old
    layer_var = "density"
    t_remove = overlap[0]
    old_at_t_remove = ds_old[layer_var].sel(time=t_remove)
    assert np.isnan(old_at_t_remove.values).any(), (
        f"Expected some NaNs in old[{layer_var!r}] at t_remove={t_remove}; "
        "looks like the 'remove several layers' manipulation did not stick."
    )

    # 4) Slope dropping: if new has more slopes than old, the last slope in new
    #    must not be present in old (this matches the 'drop last slope' logic).
    if ds_old.sizes.get("slope", 0) >= 1 and ds_new.sizes.get("slope", 0) > ds_old.sizes["slope"]:
        last_new = int(ds_new["slope"].values[-1])
        old_slopes = {int(s) for s in ds_old["slope"].values}
        assert last_new not in old_slopes, (
            "Expected last slope in 'new' to be absent in 'old' after slope-dropping."
        )


@pytest.mark.real_data
def test_combine_prefer_old_left_join(snp_gridded_xs):
    """
    Prefer old data, fill gaps from new, domain from old (join='left').

    We test that:
    - partial layer deletions in old are NOT refilled from new (no layer mixing),
    - full-profile gaps (profile_status==0) in old are filled from new,
    - density values overwritten to -999 in old remain -999,
    - a slope dropped entirely from old does NOT appear in the combined dataset,
    - a slope removed only at one location in old is restored from new,
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
    old_da = ds_old[layer_var].sel(time=t_remove)
    new_da = ds_new[layer_var].sel(time=t_remove)
    comb_da = ds_comb[layer_var].sel(time=t_remove)

    # Align on the intersection of coords so shapes match
    old_da, new_da, comb_da = xr.align(old_da, new_da, comb_da, join="inner")

    old_vals = old_da.values
    new_vals = new_da.values
    comb_vals = comb_da.values

    nan_mask = np.isnan(old_vals)
    assert nan_mask.any(), "Expected some NaNs in old at t_remove"

    # Combined should still have NaNs where old had NaNs
    # (no partial backfill of individual layers)
    assert np.all(np.isnan(comb_vals[nan_mask]))

    # If new actually has data at some of those positions, make sure
    # it was NOT used (we want to ensure no layer mixing).
    if np.any(~np.isnan(new_vals[nan_mask])):
        assert not np.allclose(
            comb_vals[nan_mask],
            new_vals[nan_mask],
            equal_nan=True,
        )

    # Where old had data, combined must match old
    assert np.allclose(
        comb_vals[~nan_mask], old_vals[~nan_mask], equal_nan=True
    )

    # 2) At t_modify, combined should still follow old, and top layer density
    #    should be -999 wherever finite (because old's overlap density was set).
    old_da2 = ds_old[layer_var].sel(time=t_modify)
    comb_da2 = ds_comb[layer_var].sel(time=t_modify)

    # Align in case of slight coord differences
    old_da2, comb_da2 = xr.align(old_da2, comb_da2, join="inner")

    old_top = old_da2.isel(layer=top_idx)
    comb_top = comb_da2.isel(layer=top_idx)

    finite_old = np.isfinite(old_top.values)
    if finite_old.any():
        # by construction in _prep_two_mergable_datasets
        assert np.all(old_top.values[finite_old] == -999)
        # combined must match old
        assert np.all(comb_top.values[finite_old] == -999)

    # More generally, for this variable at t_modify, combined should follow old
    xr.testing.assert_allclose(comb_da2, old_da2)

    # 3) density values -999 in old remain -999 in combined over overlap
    dens_old = ds_old["density"].sel(time=overlap_times)
    dens_comb = ds_comb["density"].sel(time=overlap_times)

    # Align so dims (e.g., slope) match
    dens_old, dens_comb = xr.align(dens_old, dens_comb, join="inner")

    mask_999 = dens_old.values == -999
    assert mask_999.any(), "Expected some -999 in old density over overlap."

    # Combined must keep -999 where old had it
    assert np.all(dens_comb.values[mask_999] == -999)

    # 4) Slope dropped entirely from old does NOT appear in combined
    #    (Recall _prep_two_mergable_datasets dropped last slope from old)
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
    #    *if* new actually has data there. If new is NaN as well, there's nothing to fill.
    locs = list(ds_old.location.values)
    if ds_old.sizes.get("slope", 0) >= 1 and len(locs) >= 1:
        l0 = locs[0]
        srm = ds_old.slope.values[0]  # this is the one masked in _prep
        # pick an overlap time
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
    - partial layer deletions in old are still NOT patched from new,
    - density -999 in old kept, etc.
    """
    old, new = _prep_two_mergable_datasets(snp_gridded_xs)
    _assert_complex_merge_invariants(old, new)

    combined = xsnow.combine([old, new], join="outer")
    ds_old, ds_new, ds_comb = old.data, new.data, combined.data

    # --- domain checks: union of coords on core dims ---
    assert set(ds_comb.time.values) == set(ds_old.time.values) | set(ds_new.time.values)
    assert set(ds_comb.location.values) == set(ds_old.location.values) | set(
        ds_new.location.values
    )
    assert set(ds_comb.realization.values) == set(ds_old.realization.values) | set(
        ds_new.realization.values
    )
    assert set(ds_comb.slope.values) == set(ds_old.slope.values) | set(
        ds_new.slope.values
    )

    layer_var = "density"

    # Overlap and special times
    overlap_times = np.intersect1d(ds_old.time.values, ds_new.time.values)
    assert overlap_times.size >= 2
    t_remove = overlap_times[0]
    t_modify = overlap_times[1]
    nlayer = ds_comb.sizes["layer"]
    top_idx = nlayer - 1

    # 1) Partial layer deletions in old should still be NaN, no refill from new
    old_da = ds_old[layer_var].sel(time=t_remove)
    new_da = ds_new[layer_var].sel(time=t_remove)
    comb_da = ds_comb[layer_var].sel(time=t_remove)

    old_da, new_da, comb_da = xr.align(old_da, new_da, comb_da, join="inner")

    old_vals = old_da.values
    new_vals = new_da.values
    comb_vals = comb_da.values

    nan_mask = np.isnan(old_vals)
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

        # and match old where old had values
        assert np.allclose(
            comb_vals[~nan_mask], old_vals[~nan_mask], equal_nan=True
        )

    # 2) At t_modify, combined should still follow old for this variable.
    #    We align first because combined may have extra slopes from 'new'.
    old_da2 = ds_old[layer_var].sel(time=t_modify)
    comb_da2 = ds_comb[layer_var].sel(time=t_modify)

    old_da2, comb_da2 = xr.align(old_da2, comb_da2, join="inner")

    old_top = old_da2.isel(layer=top_idx)
    comb_top = comb_da2.isel(layer=top_idx)

    finite_old = np.isfinite(old_top.values)
    if finite_old.any():
        # by construction, density in old over overlap is -999 wherever finite
        assert np.all(old_top.values[finite_old] == -999)
        assert np.all(comb_top.values[finite_old] == -999)

    xr.testing.assert_allclose(comb_da2, old_da2)


    # 3) density -999 in old kept in overlap region
    dens_old = ds_old["density"].sel(time=overlap_times)
    dens_comb = ds_comb["density"].sel(time=overlap_times)

    # Align so we compare only common coords (e.g. slopes)
    dens_old, dens_comb = xr.align(dens_old, dens_comb, join="inner")

    mask_999 = dens_old.values == -999
    if mask_999.any():
        assert np.all(dens_comb.values[mask_999] == -999)

    # 4) Slope that existed only in new (dropped from old) should now be present
    all_slopes_new = list(ds_new.slope.values)
    all_slopes_old = list(ds_old.slope.values)
    all_slopes_comb = list(ds_comb.slope.values)
    if len(all_slopes_new) > len(all_slopes_old):
        extra_slope = all_slopes_new[-1]
        assert extra_slope not in all_slopes_old
        assert extra_slope in all_slopes_comb

    # 5) For the slope removed only for one location in old, combined can use new
    #    if new actually has data there.
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
    t_remove = overlap_times[0]
    t_modify = overlap_times[1]

    layer_var = "density"

    # convenience to check density and layered vars under different joins
    def _run(join_mode: str):
        combined = xsnow.combine([new, old], join=join_mode)
        return combined.data

    # LEFT JOIN: domain = new, prefer new
    ds_left = _run("left")
    assert set(ds_left.time.values) == set(ds_new.time.values)

    # On overlap, density should match new, not old (-999)
    dens_old = ds_old["density"].sel(time=overlap_times)
    dens_new = ds_new["density"].sel(time=overlap_times)
    dens_left = ds_left["density"].sel(time=overlap_times)

    # Align on intersection of coords so boolean mask applies safely
    dens_old, dens_new, dens_left = xr.align(dens_old, dens_new, dens_left, join="inner")

    vals_old = dens_old.values
    vals_new = dens_new.values
    vals_left = dens_left.values

    # Only consider positions where old == -999 AND new has finite data
    mask_999_and_new = (vals_old == -999) & np.isfinite(vals_new)
    if mask_999_and_new.any():
        # After combine([new, old], join='left'), new should win there
        assert np.allclose(
            vals_left[mask_999_and_new],
            vals_new[mask_999_and_new],
        )
    # Where new is NaN, it's fine (and expected) that left still holds -999.

    # At t_modify, we want to assert *profile-level* semantics (no layer mixing):
    # - where new has a valid profile (profile_status > 0), left should take the
    #   entire profile from new (all layers),
    # - where new has no valid profile but old does, left should take the entire
    #   profile from old,
    # - if both are invalid, we don't constrain the result.
    layer_var = "density"
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

        # Broadcast profile_status masks to the layered variable's shape
        # (xarray will handle broadcasting along the 'layer' dimension).
        mask_new_valid = (ps_new > 0)
        mask_only_old_valid = (ps_old > 0) & (ps_new == 0)

        # Where new has a valid profile, left must match new for the full profile
        if bool(mask_new_valid.any()):
            xr.testing.assert_allclose(
                var_left.where(mask_new_valid, drop=True),
                var_new.where(mask_new_valid, drop=True),
            )

        # Where only old has a valid profile, left must match old for the full profile
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

        # Align on common coords (location, slope, realization, layer)
        da_right, da_old = xr.align(da_right, da_old, join="inner")

        # We only require that the *data values* match; coordinates like 'z'
        # may be recomputed by combine and are not part of the merge semantics.
        np.testing.assert_allclose(da_right.values, da_old.values)
        # If you still want a bit of safety on coords, you could add e.g.:
        for coord in ("location", "slope", "realization", "layer"):
            assert np.array_equal(da_right[coord], da_old[coord])

    # INNER JOIN: domain = intersection, prefer new
    ds_inner = _run("inner")
    assert set(ds_inner.time.values) == set(overlap_times)
    # On overlap, wherever new has finite density, inner result should match new
    dens_inner = ds_inner["density"].sel(time=overlap_times)
    dens_new_sel = ds_new["density"].sel(time=overlap_times)

    # Align on common coords (e.g., slope) before comparing
    dens_inner_a, dens_new_a = xr.align(dens_inner, dens_new_sel, join="inner")

    vals_inner = dens_inner_a.values
    vals_new = dens_new_a.values

    mask_new_valid = np.isfinite(vals_new)
    if mask_new_valid.any():
        assert np.allclose(
            vals_inner[mask_new_valid],
            vals_new[mask_new_valid],
        )
    # Where new is NaN, we don't constrain what combine did (old may have -999 or NaNs).

    # OUTER JOIN: domain = union, prefer new
    ds_outer = _run("outer")
    assert set(ds_outer.time.values) == (
        set(ds_old.time.values) | set(ds_new.time.values)
    )

    # On overlap, wherever new has finite density, outer result should match new.
    dens_outer = ds_outer["density"].sel(time=overlap_times)
    dens_new_sel = ds_new["density"].sel(time=overlap_times)

    # Align on common coords (esp. slope) before comparing
    dens_outer_a, dens_new_a = xr.align(dens_outer, dens_new_sel, join="inner")

    vals_outer = dens_outer_a.values
    vals_new = dens_new_a.values

    mask_new_valid = np.isfinite(vals_new)
    if mask_new_valid.any():
        assert np.allclose(
            vals_outer[mask_new_valid],
            vals_new[mask_new_valid],
        )
    # Where new is NaN, we don't constrain what combine did
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
