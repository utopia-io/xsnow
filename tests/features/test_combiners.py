# tests/test_concat_merge.py
import numpy as np
import pandas as pd
import xarray as xr
import pytest

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
    v = "density" if "density" in ds.data_vars else list(ds.data_vars)[0]
    arr = ds[v]
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
    t_old = slice(times[0],  times[36])   # earlier window
    t_new = slice(times[24], times[-1])   # later window, overlapping

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
    if "density" in old:
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


def _get_layer_var(ds: xr.Dataset) -> str:
    """Pick a representative layered variable, preferring 'density' if present."""
    if "density" in ds:
        return "density"
    for name, var in ds.data_vars.items():
        if "layer" in var.dims:
            return name
    raise RuntimeError("No layered variable found for testing.")


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

    layer_var = _get_layer_var(ds_comb)
    nlayer = ds_comb.sizes["layer"]
    top_idx = nlayer - 1

    # 1) Several layers removed in old at t_remove should stay NaN (no refill)
    old_vals = ds_old[layer_var].sel(time=t_remove).values
    new_vals = ds_new[layer_var].sel(time=t_remove).values
    comb_vals = ds_comb[layer_var].sel(time=t_remove).values

    nan_mask = np.isnan(old_vals)
    assert nan_mask.any(), "Expected some NaNs in old at t_remove"
    # New has real data there (to prove we truly didn't mix):
    assert np.any(~np.isnan(new_vals[nan_mask]))
    # Combined should still have NaNs there (no backfill from new):
    assert np.all(np.isnan(comb_vals[nan_mask]))
    # Where old had data, combined must match old:
    assert np.allclose(
        comb_vals[~nan_mask], old_vals[~nan_mask], equal_nan=True
    )

    # 2) Modified top layer at t_modify in old should be kept
    old_top = ds_old[layer_var].sel(time=t_modify).isel(layer=top_idx)
    new_top = ds_new[layer_var].sel(time=t_modify).isel(layer=top_idx)
    comb_top = ds_comb[layer_var].sel(time=t_modify).isel(layer=top_idx)

    # We know old wrote 123.0 there
    assert np.allclose(old_top, 123.0)
    # combined must match old, not new
    assert np.allclose(comb_top, old_top)
    # Just to be safe, new should differ somewhere
    assert not np.allclose(new_top, old_top)

    # 3) density values -999 in old remain -999 in combined over overlap
    if "density" in ds_old and "density" in ds_comb:
        dens_old = ds_old["density"].sel(time=overlap_times)
        dens_new = ds_new["density"].sel(time=overlap_times)
        dens_comb = ds_comb["density"].sel(time=overlap_times)

        mask_999 = dens_old.values == -999
        assert mask_999.any(), "Expected some -999 in old density over overlap."
        # Combined must keep -999 where old had it
        assert np.all(dens_comb.values[mask_999] == -999)
        # And new density there should typically differ
        assert np.any(dens_new.values[mask_999] != -999)

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

    # 5) For the slope removed only for one location in old, combined uses new
    locs = list(ds_old.location.values)
    if ds_old.sizes.get("slope", 0) >= 1 and len(locs) >= 1:
        l0 = locs[0]
        srm = ds_old.slope.values[0]  # this is the one masked in _prep
        # pick an overlap time
        t_any = overlap_times[0]
        # old is NaN here (gap)
        old_profile = ds_old[layer_var].sel(location=l0, slope=srm, time=t_any)
        new_profile = ds_new[layer_var].sel(location=l0, slope=srm, time=t_any)
        comb_profile = ds_comb[layer_var].sel(location=l0, slope=srm, time=t_any)

        assert np.isnan(old_profile.values).all(), "Expected full gap in old"
        # new must have some valid data
        assert np.any(~np.isnan(new_profile.values))
        # combined should take new, not remain all-NaN
        assert np.any(~np.isnan(comb_profile.values))
        assert np.allclose(
            comb_profile.values,
            new_profile.values,
            equal_nan=True,
        )

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

    layer_var = _get_layer_var(ds_comb)

    # Overlap and special times
    overlap_times = np.intersect1d(ds_old.time.values, ds_new.time.values)
    assert overlap_times.size >= 2
    t_remove = overlap_times[0]
    t_modify = overlap_times[1]
    nlayer = ds_comb.sizes["layer"]
    top_idx = nlayer - 1

    # 1) Partial layer deletions in old should still be NaN, no refill from new
    old_vals = ds_old[layer_var].sel(time=t_remove).values
    new_vals = ds_new[layer_var].sel(time=t_remove).values
    comb_vals = ds_comb[layer_var].sel(time=t_remove).values

    nan_mask = np.isnan(old_vals)
    if nan_mask.any():
        assert np.any(~np.isnan(new_vals[nan_mask]))
        assert np.all(np.isnan(comb_vals[nan_mask]))
        assert np.allclose(
            comb_vals[~nan_mask], old_vals[~nan_mask], equal_nan=True
        )

    # 2) Modified top layer remains from old
    old_top = ds_old[layer_var].sel(time=t_modify).isel(layer=top_idx)
    comb_top = ds_comb[layer_var].sel(time=t_modify).isel(layer=top_idx)
    assert np.allclose(comb_top, old_top)

    # 3) density -999 in old kept in overlap region
    if "density" in ds_old and "density" in ds_comb:
        dens_old = ds_old["density"].sel(time=overlap_times)
        dens_new = ds_new["density"].sel(time=overlap_times)
        dens_comb = ds_comb["density"].sel(time=overlap_times)

        mask_999 = dens_old.values == -999
        if mask_999.any():
            assert np.all(dens_comb.values[mask_999] == -999)
            assert np.any(dens_new.values[mask_999] != -999)

    # 4) Slope that existed only in new (dropped from old) should now be present
    all_slopes_new = list(ds_new.slope.values)
    all_slopes_old = list(ds_old.slope.values)
    all_slopes_comb = list(ds_comb.slope.values)
    if len(all_slopes_new) > len(all_slopes_old):
        extra_slope = all_slopes_new[-1]
        assert extra_slope not in all_slopes_old
        assert extra_slope in all_slopes_comb

    # 5) For the slope removed only for one location in old, combined uses new
    locs = list(ds_old.location.values)
    if ds_old.sizes.get("slope", 0) >= 1 and len(locs) >= 1:
        l0 = locs[0]
        srm = ds_old.slope.values[0]
        t_any = overlap_times[0]
        old_profile = ds_old[layer_var].sel(location=l0, slope=srm, time=t_any)
        new_profile = ds_new[layer_var].sel(location=l0, slope=srm, time=t_any)
        comb_profile = ds_comb[layer_var].sel(location=l0, slope=srm, time=t_any)

        if np.isnan(old_profile.values).all():
            assert np.any(~np.isnan(new_profile.values))
            assert np.any(~np.isnan(comb_profile.values))
            assert np.allclose(
                comb_profile.values,
                new_profile.values,
                equal_nan=True,
            )


@pytest.mark.real_data
@pytest.mark.parametrize("join_mode", ["left", "right", "inner", "outer"])
def test_combine_prefer_old_other_joins_basic_invariants(snp_gridded_xs, join_mode):
    """
    Smoke-test other joins (left/right/inner/outer) in the prefer-old ordering
    for basic domain invariants and check that profile_status is present and aligned.
    """
    old, new = _prep_two_mergable_datasets(snp_gridded_xs)
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

    ds_old, ds_new = old.data, new.data

    # overlap for checks
    overlap_times = np.intersect1d(ds_old.time.values, ds_new.time.values)
    assert overlap_times.size >= 2
    t_remove = overlap_times[0]
    t_modify = overlap_times[1]

    layer_var = _get_layer_var(ds_old)

    # convenience to check density and layered vars under different joins
    def _run(join_mode: str):
        combined = xsnow.combine([new, old], join=join_mode)
        return combined.data

    # LEFT JOIN: domain = new, prefer new
    ds_left = _run("left")
    assert set(ds_left.time.values) == set(ds_new.time.values)

    # On overlap, density should match new, not old (-999)
    if "density" in ds_old and "density" in ds_new and "density" in ds_left:
        dens_old = ds_old["density"].sel(time=overlap_times)
        dens_new = ds_new["density"].sel(time=overlap_times)
        dens_left = ds_left["density"].sel(time=overlap_times)
        # Where old had -999, combined should equal new, not -999
        mask_999 = dens_old.values == -999
        if mask_999.any():
            assert np.allclose(dens_left.values[mask_999], dens_new.values[mask_999])

    # At t_modify, top layer should match new, not old(=123)
    nlayer = ds_old.sizes["layer"]
    top_idx = nlayer - 1
    if "layer" in ds_left[layer_var].dims:
        old_top = ds_old[layer_var].sel(time=t_modify).isel(layer=top_idx)
        new_top = ds_new[layer_var].sel(time=t_modify).isel(layer=top_idx)
        left_top = ds_left[layer_var].sel(time=t_modify).isel(layer=top_idx)
        assert np.allclose(left_top, new_top)
        assert not np.allclose(left_top, old_top)

    # RIGHT JOIN: domain = old, prefer new on overlaps
    ds_right = _run("right")
    assert set(ds_right.time.values) == set(ds_old.time.values)
    # times only in old (before new) must still come from old
    only_old = set(ds_old.time.values) - set(ds_new.time.values)
    if only_old:
        t0 = sorted(only_old)[0]
        xr.testing.assert_allclose(
            ds_right[layer_var].sel(time=t0), ds_old[layer_var].sel(time=t0)
        )

    # INNER JOIN: domain = intersection, prefer new
    ds_inner = _run("inner")
    assert set(ds_inner.time.values) == set(overlap_times)
    # On overlap, density again should follow new
    if "density" in ds_inner and "density" in ds_new:
        dens_inner = ds_inner["density"].sel(time=overlap_times)
        dens_new = ds_new["density"].sel(time=overlap_times)
        xr.testing.assert_allclose(dens_inner, dens_new)

    # OUTER JOIN: domain = union, prefer new
    ds_outer = _run("outer")
    assert set(ds_outer.time.values) == (
        set(ds_old.time.values) | set(ds_new.time.values)
    )
    # On overlap, again follow new
    if "density" in ds_outer and "density" in ds_new:
        dens_outer = ds_outer["density"].sel(time=overlap_times)
        dens_new = ds_new["density"].sel(time=overlap_times)
        xr.testing.assert_allclose(dens_outer, dens_new)


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
