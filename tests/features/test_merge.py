import numpy as np
import pandas as pd
import xarray as xr
import pytest

from xsnow import xsnowDataset
from xsnow.utils import concat as xsnow_concat


# ======================================================================
# Helpers: synthetic xsnow-like datasets
# ======================================================================

def make_layered_dataset(
    name: str,
    times,
    locations,
    slopes,
    layers,
    base: float = 0.0,
    with_profile_status: bool = False,
    status_value: int = 1,
) -> xr.Dataset:
    """
    Create a synthetic xsnow-like layered Dataset.

    Dims:
      - location
      - time
      - slope
      - realization (fixed to [0])
      - layer

    Data:
      - density: deterministic values (base + linear index)
      - optional: profile_status(location, time, slope, realization)

    Also adds simple inclination/azimuth coords based on slope index.
    """
    times = np.array(times, dtype="datetime64[ns]")
    locations = np.array(locations, dtype=str)
    slopes = np.array(slopes, dtype=int)
    layers = np.array(layers, dtype=int)
    realization = np.array([0], dtype=int)

    shape = (
        locations.size,
        times.size,
        slopes.size,
        realization.size,
        layers.size,
    )

    density_values = base + np.arange(np.prod(shape), dtype=float).reshape(shape)

    ds = xr.Dataset(
        {
            "density": (
                ("location", "time", "slope", "realization", "layer"),
                density_values,
            )
        },
        coords={
            "location": locations,
            "time": times,
            "slope": slopes,
            "realization": realization,
            "layer": layers,
        },
    )

    if with_profile_status:
        ps_shape = (
            locations.size,
            times.size,
            slopes.size,
            realization.size,
        )
        ds["profile_status"] = (
            ("location", "time", "slope", "realization"),
            np.full(ps_shape, status_value, dtype=int),
        )

    if slopes.size > 0:
        ds = ds.assign_coords(
            inclination=("slope", slopes * 10),
            azimuth=("slope", (slopes * 45) % 360),
        )

    return ds


def make_timeseries_dataset_no_layer(
    times,
    locations,
    slopes=None,
    base: float = 0.0,
) -> xr.Dataset:
    """
    Dataset WITHOUT 'layer' dimension (e.g. surface time series).

    Dims:
      - location
      - time
      - (optional) slope
      - realization ([0])

    Data:
      - surface_temp
    """
    times = np.array(times, dtype="datetime64[ns]")
    locations = np.array(locations, dtype=str)
    realization = np.array([0], dtype=int)

    if slopes is not None:
        slopes = np.array(slopes, dtype=int)
        shape = (locations.size, times.size, slopes.size, realization.size)
        dims = ("location", "time", "slope", "realization")
        coords = {
            "location": locations,
            "time": times,
            "slope": slopes,
            "realization": realization,
        }
    else:
        shape = (locations.size, times.size, realization.size)
        dims = ("location", "time", "realization")
        coords = {
            "location": locations,
            "time": times,
            "realization": realization,
        }

    vals = base + np.arange(np.prod(shape), dtype=float).reshape(shape)

    ds = xr.Dataset(
        {"surface_temp": (dims, vals)},
        coords=coords,
    )
    return ds


def make_disjoint_time_datasets():
    """Case: concat/merge datasets with NO time overlap."""
    times_a = pd.date_range("2024-01-01T00", periods=2, freq="H")
    times_b = pd.date_range("2024-01-02T00", periods=2, freq="H")

    ds_a = make_layered_dataset(
        "A",
        times=times_a,
        locations=["LOC1"],
        slopes=[0],
        layers=[0, 1],
        base=0.0,
    )
    ds_b = make_layered_dataset(
        "B",
        times=times_b,
        locations=["LOC1"],
        slopes=[0],
        layers=[0, 1],
        base=100.0,
    )
    return ds_a, ds_b


def make_layer_vs_no_layer_datasets():
    """Case: merging when only one dataset has 'layer' dimension."""
    times = pd.date_range("2024-01-01T00", periods=3, freq="H")

    ds_layered = make_layered_dataset(
        "L",
        times=times,
        locations=["LOC1"],
        slopes=[0],
        layers=[0, 1],
        base=0.0,
    )

    ds_nolayer = make_timeseries_dataset_no_layer(
        times=times,
        locations=["LOC1"],
        slopes=None,
        base=50.0,
    )

    return ds_layered, ds_nolayer


def make_overlap_with_profile_status():
    """
    Case: time overlap + profile_status.

    ds_old: baseline (status=1, finite density).
    ds_new:
      - same coords
      - t0: status=1, distinct density (trusted)
      - t1: status=0, density=NaN (untrusted in xsnow semantics)

    Used to show what xr.merge can do (override, conflicts) and where it
    CANNOT implement xsnow's planned prefer_layers logic.
    """
    times = pd.date_range("2024-01-01T00", periods=2, freq="H")
    locations = ["LOC1"]
    slopes = [0]
    layers = [0, 1]

    ds_old = make_layered_dataset(
        "old",
        times=times,
        locations=locations,
        slopes=slopes,
        layers=layers,
        base=0.0,
        with_profile_status=True,
        status_value=1,
    )

    ds_new = make_layered_dataset(
        "new",
        times=times,
        locations=locations,
        slopes=slopes,
        layers=layers,
        base=100.0,
        with_profile_status=True,
        status_value=1,
    )

    # At t1: mark status=0 and density=NaN in ds_new
    t1 = 1
    ps = ds_new["profile_status"].values
    ps[:, t1, :, :] = 0
    ds_new["profile_status"].values[...] = ps

    dens = ds_new["density"].values
    dens[:, t1, :, :, :] = np.nan
    ds_new["density"].values[...] = dens

    return ds_old, ds_new


def make_new_locations_and_slopes():
    """
    Case: merging datasets with new locations and new slopes.

    base_ds:
      - LOC1, slope 0
    ext_ds:
      - LOC1, slopes 0 & 1
      - LOC2, slope 0
    """
    times = pd.date_range("2024-01-01T00", periods=2, freq="H")
    layers = [0, 1]

    base_ds = make_layered_dataset(
        "base",
        times=times,
        locations=["LOC1"],
        slopes=[0],
        layers=layers,
        base=0.0,
        with_profile_status=True,
        status_value=1,
    ).assign_coords(
        inclination=("slope", [30]),
        azimuth=("slope", [0]),
    )

    ext_ds = make_layered_dataset(
        "ext",
        times=times,
        locations=["LOC1", "LOC2"],
        slopes=[0, 1],
        layers=layers,
        base=200.0,
        with_profile_status=True,
        status_value=1,
    ).assign_coords(
        inclination=("slope", [30, 38]),
        azimuth=("slope", [0, 180]),
    )

    return base_ds, ext_ds


# ======================================================================
# 1. Concat: disjoint times
# ======================================================================

def test_concat_no_time_overlap_outer():
    ds_a, ds_b = make_disjoint_time_datasets()

    out = xsnow_concat([ds_a, ds_b], dim="time", join="outer")

    # Expect union of times:
    expected_times = list(ds_a.time.values) + list(ds_b.time.values)
    assert list(out.time.values) == expected_times

    # Spot-check densities from both inputs present
    assert float(out["density"].sel(time=ds_a.time.values[0]).isel(layer=0)) == float(
        ds_a["density"].isel(time=0, layer=0)
    )
    assert float(out["density"].sel(time=ds_b.time.values[-1]).isel(layer=-1)) == float(
        ds_b["density"].isel(time=-1, layer=-1)
    )


# ======================================================================
# 2. Merge: only one dataset has 'layer'
# ======================================================================

def test_merge_layer_and_nolayer_outer_no_conflicts():
    ds_layered, ds_nolayer = make_layer_vs_no_layer_datasets()

    # surface_temp gets broadcast over extra dims, standard xarray semantics
    merged = xr.merge([ds_layered, ds_nolayer], compat="no_conflicts", join="outer")

    # 'density' keeps its 5D shape
    assert set(merged["density"].dims) == {
        "location",
        "time",
        "slope",
        "realization",
        "layer",
    }

    # 'surface_temp' is broadcast across slope/layer
    assert "surface_temp" in merged
    assert "layer" in merged["surface_temp"].dims or "slope" in merged["surface_temp"].dims


# ======================================================================
# 3. Overlap + profile_status:
#    What xr.merge can/can't do (motivates xsnow.merge)
# ======================================================================

def test_merge_overlap_left_join_override_prefers_first_dataset():
    """
    With [new, old] and compat='override', xr.merge uses FIRST dataset's values
    where coords overlap, regardless of NaN. That matches a "new wins always"
    policy, but NOT the xsnow 'prefer_layers with profile_status' logic.
    """
    ds_old, ds_new = make_overlap_with_profile_status()

    merged = xr.merge(
        [ds_new, ds_old],
        join="left",       # coords from ds_new only
        compat="override", # ds_new wins on conflicts
    )

    # Same coords as new:
    xr.testing.assert_identical(merged["time"], ds_new["time"])

    # At all coords, density should equal ds_new (including NaNs at t1)
    xr.testing.assert_identical(merged["density"], ds_new["density"])

    # This explicitly shows: xr.merge cannot "fallback to old when status=0".


def test_merge_overlap_left_join_no_conflicts_raises():
    """
    If we try to be strict with compat='no_conflicts', overlapping but different
    densities from old/new cause a MergeError. This also shows why we can't use
    plain xr.merge for conditional precedence.
    """
    ds_old, ds_new = make_overlap_with_profile_status()

    with pytest.raises(xr.MergeError):
        xr.merge(
            [ds_new, ds_old],
            join="left",
            compat="no_conflicts",
        )


# ======================================================================
# 4. New locations & slopes
# ======================================================================

def test_merge_new_locations_and_slopes_outer_override():
    """
    Demonstrate what xr.merge can do:
    - With compat='override', later datasets in the list can overwrite.
    - New locations and slopes are naturally included with join='outer'.
    """
    base_ds, ext_ds = make_new_locations_and_slopes()

    # Order: [ext, base] → ext has precedence on overlaps
    merged = xr.merge(
        [ext_ds, base_ds],
        join="outer",
        compat="override",
    )

    # LOC2 from ext must be present
    assert "LOC2" in merged["location"].values

    # For LOC1 & slope=0, values should match ext_ds due to override
    sel = dict(location="LOC1", slope=0)
    xr.testing.assert_identical(
        merged["density"].sel(**sel),
        ext_ds["density"].sel(**sel),
    )


def test_merge_new_locations_and_slopes_no_conflicts_fails_on_overlap():
    """
    With compat='no_conflicts', differing overlapping values cause xr.merge
    to fail. This highlights that naive use of merge is not enough for
    xsnow-aware precedence semantics.
    """
    base_ds, ext_ds = make_new_locations_and_slopes()

    with pytest.raises(xr.MergeError):
        xr.merge(
            [ext_ds, base_ds],
            join="outer",
            compat="no_conflicts",
        )


# ======================================================================
# 5. Expert-only: merging layer info from separate datasets
# ======================================================================

def test_expert_merge_layers_multiloc_multislope_exact_no_conflicts():
    """
    Expert scenario:
    - Two datasets contribute different layer-wise variables over identical
      coords.
    - Use join='exact', compat='no_conflicts' to ensure strict alignment.
    """
    times = pd.date_range("2024-01-01T00", periods=2, freq="H")
    locs = ["LOC1", "LOC2"]
    slopes = [0, 1]
    layers = [0, 1, 2]

    base = make_layered_dataset(
        "base",
        times=times,
        locations=locs,
        slopes=slopes,
        layers=layers,
        base=0.0,
    )

    # top_only: same coords, but defines a different var for illustration
    top_only = base.copy()
    top_only = top_only.drop_vars("density")
    top_only["density_top"] = xr.full_like(
        base["density"], fill_value=999.0
    )

    merged = xr.merge(
        [base, top_only],
        join="exact",
        compat="no_conflicts",
    )

    # All coords must match exactly
    xr.testing.assert_identical(merged["time"], base["time"])
    xr.testing.assert_identical(merged["location"], base["location"])
    xr.testing.assert_identical(merged["layer"], base["layer"])

    # Vars from both must be present without conflict
    assert "density" in merged
    assert "density_top" in merged


def test_expert_merge_layers_single_profile_after_manual_align():
    """
    Variant:
    - Restrict to a single (location, slope) profile in each dataset.
    - Align explicitly, then merge with join='exact', compat='no_conflicts'.
    - This pattern is what advanced users can apply for precise layer fusion.
    """
    times = pd.date_range("2024-01-01T00", periods=2, freq="H")
    locs = ["LOC1", "LOC2"]
    slopes = [0, 1]
    layers = [0, 1, 2]

    ds1 = make_layered_dataset(
        "ds1",
        times=times,
        locations=locs,
        slopes=slopes,
        layers=layers,
        base=0.0,
    )
    ds2 = make_layered_dataset(
        "ds2",
        times=times,
        locations=locs,
        slopes=slopes,
        layers=layers,
        base=100.0,
    )

    # Select a single profile for each
    prof1 = ds1.sel(location="LOC1", slope=0)
    prof2 = ds2.sel(location="LOC1", slope=0)

    # Ensure consistent dim order: (time, realization, layer)
    prof1, prof2 = xr.align(prof1, prof2, join="exact")

    merged = xr.merge(
        [prof1, prof2],
        join="exact",
        compat="no_conflicts",
    )

    # Both density fields exist separately (if we rename before merge),
    # or if we keep same name, compat='no_conflicts' guarantees equality.
    assert set(merged.dims.keys()) == set(prof1.dims.keys())
    xr.testing.assert_identical(merged["time"], prof1["time"])
    xr.testing.assert_identical(merged["layer"], prof1["layer"])
