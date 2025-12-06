import numpy as np
import pandas as pd
import xarray as xr
import pytest

from xsnow import xsnowDataset


@pytest.fixture
def base_coords():
    """Canonical coords used in most tests."""
    time = pd.to_datetime(
        ["2024-01-01T00:00:00", "2024-01-01T01:00:00"]
    )
    return {
        "location": ["TEST"],
        "time": time,
        "slope": [0],
        "realization": [0],
        "layer": np.arange(3),
    }


@pytest.fixture
def make_xsnow(base_coords):
    """
    Factory to create an xsnowDataset with shared coords,
    plus arbitrary data_vars overrides.
    """

    def _factory(data_vars=None, drop_layer: bool = False, single_time: bool = False):
        coords = dict(base_coords)

        if drop_layer:
            coords.pop("layer", None)

        if single_time:
            coords["time"] = coords["time"][:1]

        ds = xr.Dataset(coords=coords)
        if data_vars:
            ds = ds.assign(data_vars)

        return xsnowDataset(ds)

    return _factory


# ----------------------------------------------------------------------
# z-based behaviour
# ----------------------------------------------------------------------


@pytest.mark.synthetic
def test_update_profile_status_z_basic(make_xsnow, base_coords):
    """
    With a vertical z field on 'layer':
    - profiles with any z < 0 → status 1 (existing layer data),
    - profiles with no z < 0 and no HS==0 → status 0 (no layer data).
    """
    time = base_coords["time"]

    # dims: (location=1, time=2, slope=1, realization=1, layer=3)
    z = np.empty((1, 2, 1, 1, 3), dtype=float)
    z[0, 0, 0, 0, :] = [-0.1, 0.0, 0.1]  # t0: has negative → existing layer data
    z[0, 1, 0, 0, :] = [0.0, 0.1, 0.2]   # t1: all >= 0, no HS → no layer data

    ds = make_xsnow(
        data_vars={
            "z": (("location", "time", "slope", "realization", "layer"), z),
        }
    ).update_profile_status()

    ps = ds["profile_status"]

    assert set(ps.dims) == {"location", "time", "slope", "realization"}

    # t0: has z<0 → status 1
    assert ps.sel(time=time[0]).item() == 1
    # t1: only non-negative z, no HS==0 and no prior positive status → 0
    assert ps.sel(time=time[1]).item() == 0

    mapping = ps.attrs["mapping"]
    assert mapping[-1] == "error during parsing"
    assert mapping[0] == "no layer data"
    assert mapping[1] == "existing layer data"


@pytest.mark.synthetic
def test_update_profile_status_z_nan_with_HS_zero(make_xsnow, base_coords):
    """
    If z exists but is all-NaN at a profile and HS==0 there, this still counts as
    'existing layer data' (status 1).

    At a profile where z is NaN and HS!=0 (or missing), status must be 0.
    """
    time = base_coords["time"]

    # z all NaN at both times
    z = np.full((1, 2, 1, 1, 3), np.nan, dtype=float)

    # HS is 0 at t0 and positive at t1
    hs = np.empty((1, 2, 1, 1), dtype=float)
    hs[0, 0, 0, 0] = 0.0   # t0: HS == 0 → counts as existing layer info
    hs[0, 1, 0, 0] = 0.2   # t1: HS > 0 → no special treatment in new logic

    ds = make_xsnow(
        data_vars={
            "z": (("location", "time", "slope", "realization", "layer"), z),
            "HS": (("location", "time", "slope", "realization"), hs),
        }
    ).update_profile_status()

    ps = ds["profile_status"]

    # t0: z NaN, HS == 0 → status 1
    assert ps.sel(time=time[0]).item() == 1
    # t1: z NaN, HS > 0 and no prior positive status → status 0
    assert ps.sel(time=time[1]).item() == 0


# ----------------------------------------------------------------------
# Behaviour when z is missing
# ----------------------------------------------------------------------


@pytest.mark.synthetic
def test_update_profile_status_missing_z_resets_nonnegative_to_zero(make_xsnow):
    """
    If z does not exist at all, every non-negative profile_status must become 0
    (no layer data), regardless of HS or other non-layer variables.
    Negative codes are preserved.
    """
    # Create a dataset without 'layer' and without 'z'
    hs = np.array([[[[0.0]]]], dtype=float)  # would have implied "no snow" previously

    ds = make_xsnow(
        data_vars={
            "HS": (("location", "time", "slope", "realization"), hs),
        },
        drop_layer=True,
        single_time=True,
    )

    # Inject a custom profile_status with 1 (positive) and -1 (error)
    base_status = xr.DataArray(
        data=[[[[1]]]],
        coords={k: ds.coords[k] for k in ("location", "time", "slope", "realization")},
        dims=("location", "time", "slope", "realization"),
        name="profile_status",
    )
    ds = ds.assign(profile_status=base_status)

    ds = ds.update_profile_status()
    ps = ds["profile_status"]

    # Non-negative entries are reset to 0 because z is missing
    assert ps.item() == 0


# ----------------------------------------------------------------------
# Preservation of negative / positive pre-existing status
# ----------------------------------------------------------------------


@pytest.mark.synthetic
def test_update_profile_status_preserves_negative(make_xsnow):
    """Existing negative profile_status must not be overridden."""
    ds = make_xsnow(single_time=True)

    # dims: (location=1, time=1, slope=1, realization=1, layer=3)
    z = np.array([[[[[-0.1, -0.2, -0.3]]]]], dtype=float)

    base_status = xr.DataArray(
        data=[[[[-1]]]],
        coords={k: ds.coords[k] for k in ("location", "time", "slope", "realization")},
        dims=("location", "time", "slope", "realization"),
        name="profile_status",
    )

    ds = ds.assign(
        z=(("location", "time", "slope", "realization", "layer"), z),
        profile_status=base_status,
    )

    ds = ds.update_profile_status()
    ps = ds["profile_status"]

    # Still -1 despite snow-like z
    assert ps.item() == -1


@pytest.mark.synthetic
def test_update_profile_status_preserves_existing_positive_without_HS(make_xsnow):
    """
    If z exists but has no negative values and HS does not force layer data,
    an existing positive profile_status should keep that profile as 1.
    """
    ds = make_xsnow(single_time=True)

    # z with all non-negative values (no snow signal)
    z = np.array([[[[[0.0, 0.1, 0.2]]]]], dtype=float)

    base_status = xr.DataArray(
        data=[[[[1]]]],
        coords={k: ds.coords[k] for k in ("location", "time", "slope", "realization")},
        dims=("location", "time", "slope", "realization"),
        name="profile_status",
    )

    ds = ds.assign(
        z=(("location", "time", "slope", "realization", "layer"), z),
        profile_status=base_status,
    )

    ds = ds.update_profile_status()
    ps = ds["profile_status"]

    # No z<0 and no HS==0, but prior status was positive → stays 1
    assert ps.item() == 1
