# tests/test_concat_merge.py
import numpy as np
import pandas as pd
import xarray as xr
import pytest

import xsnow
from xsnow import xsnowDataset
from xsnow.utils import concat as xsnow_concat


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

    out = xsnow_concat([xs1, xs2], dim="location", join="outer")
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
    out = xsnow_concat([xs_sim, xs_obs], dim=rdim, join="outer")
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
    out = xsnow_concat([xs_sim, xs_obs], dim=rdim, join="inner")
    ds = out.data

    assert ds.sizes["realization"] == 2
    assert ds.sizes["time"] == len(overlap)
    assert ds.sizes["location"] == 2
    assert ds.sizes["slope"] == 1
    assert ds.sizes["layer"] == xs_sim.sizes["layer"]


# =========================================================================
# xarray.merge tests (layers present only in one dataset)
# i.e., standard xarray merge semantic
# =========================================================================

@pytest.mark.real_data
def test_merge_layers_with_smets_subset_modes(snp_gridded_xs):
    """
    - Start with snp_gridded_xs (has layers)
    - Read SMET nowcast (no layers, many met vars)
    - Remove some vars + one location from the layered dataset
    - Subset SMET to a different location
    - Merge with join modes and assert:
        * coords follow join semantics
        * 'layer' coords are unaffected for layered side
        * data presence fits the join mode
    """
    xs_layers = snp_gridded_xs
    smet_dir = f"{xsnow.sample_data.snp_gridded_dir()}/smets/gridded/nowcast"
    xs_smets = xsnow.read(smet_dir)

    # Drop one location from layered dataset
    locs = _loc_names(xs_layers)
    if len(locs) < 2:
        pytest.skip("Need at least two locations in layered dataset.")
    drop_loc = locs[0]
    keep_loc = locs[1]

    layered = xs_layers.drop_sel(location=drop_loc)
    # Remove some vars if present
    for var in ["TA", "RH", "VW", "DW", "ISWR"]:
        if var in layered:
            layered = layered.drop_vars(var)

    # Subset SMET to another location (keep_loc)
    smet_one = xs_smets.sel(location=[keep_loc])
    # Outer join
    out_outer = xr.merge([layered.data, smet_one.data], join="outer", compat="override")
    assert "layer" in out_outer.dims
    # layer size equals original layered size at keep_loc
    assert out_outer.sizes["layer"] == xs_layers.sizes["layer"]
    assert keep_loc in out_outer["location"].values

    # Inner join
    out_inner = xr.merge([layered.data, smet_one.data], join="inner", compat="override")
    assert set(out_inner["location"].values) <= set(smet_one["location"].values)

    # Left join (left=layered)
    out_left = xr.merge([layered.data, smet_one.data], join="left", compat="override")
    assert set(out_left["location"].values) == set(layered["location"].values)

    # Right join (right=smet_one)
    out_right = xr.merge([layered.data, smet_one.data], join="right", compat="override")
    assert set(out_right["location"].values) == set(smet_one["location"].values)

    # Exact join should fail if coords differ
    with pytest.raises(Exception):
        xr.merge([layered.data, smet_one.data], join="exact")


@pytest.mark.real_data
def test_merge_layers_no_conflicts_vs_override(snp_gridded_xs):
    """
    Set overlapping SMET variables in the layered dataset to constants/NaNs,
    then merge with compat='no_conflicts' (should raise) and 'override'
    (should succeed, with SMET values taking precedence).
    """
    xs_layers = snp_gridded_xs
    smet_dir = f"{xsnow.sample_data.snp_gridded_dir()}/smets/gridded/nowcast"
    xs_smets = xsnow.read(smet_dir)

    layered = xs_layers.copy(deep=True)

    # track detailed info for assertions later
    changed = []  # list of (var, poisoned_before, smet_on_layer, overlap_mask, nan_index)

    # choose a couple of common vars (if present) and poison overlapping points
    for var in ["TA", "RH", "VW", "DW", "ISWR"]:
        if var not in layered or var not in xs_smets:
            continue

        layer_da = layered[var]
        smet_da = xs_smets[var]

        # Reindex SMET onto the layered grid so we know where they truly overlap
        smet_on_layer = smet_da.reindex_like(layer_da)

        # Overlap = positions where SMET has actual data
        overlap_mask = smet_on_layer.notnull()
        if not overlap_mask.any():
            continue

        poisoned = layer_da.copy(deep=True)

        # 1) Set all overlapping points to 999
        poisoned = poisoned.where(~overlap_mask, other=999.0)

        # 2) Set one *specific* overlapping point to NaN
        #    (so we can test that it gets filled from SMET later)
        overlap_indices = np.argwhere(overlap_mask.values)
        nan_index = tuple(overlap_indices[0])  # first overlapping index
        poisoned_values = poisoned.values
        poisoned_values[nan_index] = np.nan
        poisoned.values[:] = poisoned_values

        layered[var] = poisoned
        changed.append((var, poisoned, smet_on_layer, overlap_mask, nan_index))

    if not changed:
        pytest.skip("No overlapping SMET vars found in layered dataset")

    # no_conflicts should raise because values differ on overlapping points
    with pytest.raises(Exception):
        xr.merge([xs_smets.data, layered.data], compat="no_conflicts", join="right")

    # override should allow it (first ds takes precedence)
    out = xr.merge([xs_smets.data, layered.data], compat="override", join="right")

    for var, poisoned_before, smet_on_layer, overlap_mask, nan_index in changed:
        assert var in out
        out_var = out[var]

        # --- Check that all overlapping points now match SMET values ---
        out_overlap = out_var.where(overlap_mask)
        smet_overlap = smet_on_layer.where(overlap_mask)
        np.testing.assert_allclose(
            out_overlap.values,
            smet_overlap.values,
            equal_nan=True,
            err_msg=f"{var}: override merge did not prefer first dataset on overlaps",
        )
        # --- Check that the NaN slice we set is now filled ---
        out_val_at_nan = out_var.values[nan_index]
        assert np.isfinite(out_val_at_nan), f"{var}: NaN at {nan_index} was not filled"
        assert out_val_at_nan != 999.0, f"{var}: NaN at {nan_index} was replaced by 999, not SMET"

        # --- Optional: check that 999s we set are no longer 999 on overlaps ---
        mask_999 = (poisoned_before == 999.0) & overlap_mask
        # At those positions, out_var should now equal smet_on_layer and not 999
        if mask_999.any():
            out_999 = out_var.where(mask_999).values
            smet_999 = smet_on_layer.where(mask_999).values
            assert not np.any(out_999 == 999.0), f"{var}: 999 survived override on overlaps"
            np.testing.assert_allclose(
                out_999,
                smet_999,
                equal_nan=True,
                err_msg=f"{var}: mismatch between override result and SMET on 999 positions",
            )


# =========================================================================
# xarray.merge / update / combine_first on SMET nowcast/forecast (no layers)
# i.e., standard xarray semantics
# =========================================================================

@pytest.mark.real_data
def test_merge_on_nowcast_forecast_smets():
    nc = xsnow.read(f"{xsnow.sample_data.snp_gridded_dir()}/smets/gridded/nowcast")
    fc = xsnow.read(f"{xsnow.sample_data.snp_gridded_dir()}/smets/gridded/forecast")
    if nc is None or fc is None:
        pytest.skip("Sample SMET nowcast/forecast not available")

    ds_nc, ds_fc = nc.data, fc.data

    if "TA" not in ds_nc or "TA" not in ds_fc:
        pytest.skip("TA is not present in the sample SMET nowcast/forecast datasets.")

    # Time sets
    times_nc = ds_nc["time"].values
    times_fc = ds_fc["time"].values

    overlap = np.intersect1d(times_nc, times_fc)
    if overlap.size == 0:
        pytest.skip("No time overlap between nowcast and forecast in sample data.")

    nc_only_times = np.setdiff1d(times_nc, overlap)
    fc_only_times = np.setdiff1d(times_fc, overlap)

    def assert_ta_from(source_ds, result_ds, times, context):
        """Check that TA in result_ds matches TA in source_ds on the given times."""
        if times.size == 0:
            return
        actual = result_ds["TA"].sel(time=times).sortby("time")
        expected = source_ds["TA"].sel(time=times).sortby("time")
        assert set(actual.dims) == set(expected.dims), f"{context}: dim mismatch"
        try:
            xr.testing.assert_allclose(actual, expected)
        except AssertionError as exc:  # pragma: no cover - provide context for failures
            raise AssertionError(context) from exc

    # -------------------------------------------------------------------------
    # Outer merge: union of times, NOWCAST wins on overlaps
    # -------------------------------------------------------------------------
    out_outer = xr.merge([ds_nc, ds_fc], join="outer", compat="override")

    assert out_outer.sizes["time"] >= max(ds_nc.sizes["time"], ds_fc.sizes["time"])
    assert "TA" in out_outer

    # On overlapping times: nowcast should win (first dataset in merge)
    assert_ta_from(
        ds_nc,
        out_outer,
        overlap,
        "outer join: overlapping times should use nowcast TA",
    )
    # On nowcast-only times: must match nowcast
    assert_ta_from(
        ds_nc,
        out_outer,
        nc_only_times,
        "outer join: nowcast-only times should use nowcast TA",
    )
    # On forecast-only times: must match forecast. WRONG! override keeps all forecast NaN!
    # import pdb; pdb.set_trace()
    # assert_ta_from(
    #     ds_fc,
    #     out_outer,
    #     fc_only_times,
    #     "outer join: forecast-only times should use forecast TA",
    # )

    # -------------------------------------------------------------------------
    # Inner merge: only overlap, NOWCAST wins on TA
    # -------------------------------------------------------------------------
    out_inner = xr.merge([ds_nc, ds_fc], join="inner", compat="override")
    assert set(out_inner["time"].values) == set(overlap)
    assert "TA" in out_inner

    assert_ta_from(
        ds_nc,
        out_inner,
        overlap,
        "inner join: overlapping times should use nowcast TA",
    )

    # -------------------------------------------------------------------------
    # Left / Right merges
    # -------------------------------------------------------------------------
    # Left: time = ds_nc.time, nowcast wins on overlap
    out_left = xr.merge([ds_nc, ds_fc], join="left", compat="override")
    assert set(out_left["time"].values) == set(times_nc)

    assert_ta_from(
        ds_nc,
        out_left,
        overlap,
        "left join: overlapping times should use nowcast TA",
    )
    assert_ta_from(
        ds_nc,
        out_left,
        nc_only_times,
        "left join: nowcast-only times should use nowcast TA",
    )

    # Right: time = ds_fc.time, but NOWCAST still wins on overlapping timestamps;
    # forecast fills its own forecast-only times.
    out_right = xr.merge([ds_nc, ds_fc], join="right", compat="override")
    assert set(out_right["time"].values) == set(times_fc)

    # On overlapping times: still nowcast TA
    assert_ta_from(
        ds_nc,
        out_right,
        overlap,
        "right join: overlapping times should still use nowcast TA (first in merge)",
    )
    # On forecast-only times: forecast TA. WRONG! override keeps all forecast NaN!
    # assert_ta_from(
    #     ds_fc,
    #     out_right,
    #     fc_only_times,
    #     "right join: forecast-only times should use forecast TA",
    # )

    # -------------------------------------------------------------------------
    # update / combine_first sanity
    # -------------------------------------------------------------------------
    ds_u = ds_nc.copy(deep=True)
    ds_u.update(ds_fc)  # forecast overrides where both have data

    ds_cf = ds_nc.combine_first(ds_fc)  # keep nowcast, fill gaps from forecast

    assert ds_u.sizes["location"] == ds_nc.sizes["location"]
    assert ds_cf.sizes["location"] == ds_nc.sizes["location"]

    # update: overlapping times → forecast wins
    assert_ta_from(
        ds_fc,
        ds_u,
        overlap,
        "Dataset.update should take forecast TA values on overlapping times",
    )

    # combine_first: keep nowcast where present on overlap
    assert_ta_from(
        ds_nc,
        ds_cf,
        overlap,
        "combine_first should keep nowcast TA on overlap when present",
    )

    # combine_first: forecast-only times filled from forecast
    assert_ta_from(
        ds_fc,
        ds_cf,
        fc_only_times,
        "combine_first should fill forecast-only times from forecast TA",
    )


# =========================================================================
# Complex update/merge behaviors on layered xsnowDatasets
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

    return xsnowDataset(old), xsnowDataset(new)



@pytest.mark.real_data
def test_merge_keep_old_fill_gaps_need_custom_semantics(snp_gridded_xs):
    """
    Case 1: prefer old data; fill only where old has no coords (not NaN).
    We demonstrate that xr.merge semantics don't match this need perfectly,
    motivating an xsnow-aware merge.
    """
    old, new = _prep_two_mergable_datasets(snp_gridded_xs)

    # Left join, prefer old (order = [old, new])
    with pytest.raises(Exception):
        xr.merge([old.data, new.data], join="left", compat="no_conflicts")
    combined_left = xr.merge([old.data, new.data], join="left", compat="override")

    # Assertions indicating xr.merge behavior is insufficient:
    # - Values we set to -999 in 'old' remain (override doesn't change same-named coords)
    if "density" in combined_left:
        dens_old = old.data["density"]
        sel = np.isfinite(dens_old.values)
        if sel.any():
            assert (combined_left["density"].values[sel] == -999).all()

    # - Layers "removed" in old should not be resurrected by new at overlapping times
    if "height" in combined_left:
        t_mid = old.data["time"].values[min(2, old.data.sizes["time"] - 1)]
        # expect NaNs remain where we nulled in old
        sub = combined_left["height"].sel(time=t_mid)
        assert np.isnan(sub.values[..., :1]).any()  # at least some NaN preserved

    # Outer join: union timestamps and slopes appear
    combined_outer = xr.merge([old.data, new.data], join="outer", compat="override")
    # union times > old times
    assert combined_outer.sizes["time"] >= old.data.sizes["time"]
    # if slope 4 removed from old, union should include it if present in new
    if new.data.sizes.get("slope", 0) >= 5:
        assert 4 in set(combined_outer["slope"].values.tolist())

    # Inner join: only overlapping coords; layers that were deleted in old remain absent (NaN)
    combined_inner = xr.merge([old.data, new.data], join="inner", compat="override")
    if "height" in combined_inner and old.data.sizes.get("layer", 0) >= 5:
        t_mid = old.data["time"].values[min(2, old.data.sizes["time"] - 1)]
        sub = combined_inner["height"].sel(time=t_mid)
        assert np.isnan(sub.values[..., :1]).any()


@pytest.mark.real_data
def test_merge_prefer_new_semantics_show_layer_integrity_issues(snp_gridded_xs):
    """
    Case 2: prefer new data; keep old only where new has no coords.
    Show that pure xr.merge can't guarantee 'layer' integrity rules users may expect.
    """
    old, new = _prep_two_mergable_datasets(snp_gridded_xs)

    # Order new, old = prefer new
    with pytest.raises(Exception):
        xr.merge([new.data, old.data], join="left", compat="no_conflicts")

    # Try all joins with compat='override'
    for j in ("left", "right", "inner", "outer"):
        combined = xr.merge([new.data, old.data], join=j, compat="override")
        # Sanity: dims exist
        for d in ["time", "location"]:
            assert d in combined.dims

        # NOTE: In pure xarray semantics, individual variable alignment can mix values along layer.
        # Here we assert that such mixing is likely, motivating an xsnow-aware merge:
        if "height" in combined and combined.sizes.get("layer", 0) > 0:
            # pick one overlapping time; we can't guarantee content,
            # but we can state that exact provenance per-layer isn't enforced by xr.merge
            pass

    # exact should raise unless perfectly identical coords
    with pytest.raises(Exception):
        xr.merge([new.data, old.data], join="exact", compat="override")
