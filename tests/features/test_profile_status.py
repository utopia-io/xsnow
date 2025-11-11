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


@pytest.mark.synthetic
def test_update_profile_status_z_and_data(make_xsnow, base_coords):
    """z < 0 → 2, otherwise 1 when data present."""
    time = base_coords["time"]

    # dims: (location=1, time=2, slope=1, realization=1, layer=3)
    z = np.empty((1, 2, 1, 1, 3), dtype=float)
    z[0, 0, 0, 0, :] = [-0.1, 0.0, 0.1]  # t0: has negative → snow
    z[0, 1, 0, 0, :] = [0.0, 0.1, 0.2]   # t1: all >= 0 → no snow
    density = np.ones_like(z)

    ds = make_xsnow(
        data_vars={
            "z": (("location", "time", "slope", "realization", "layer"), z),
            "density": (("location", "time", "slope", "realization", "layer"), density),
        }
    ).update_profile_status()

    ps = ds["profile_status"]

    assert set(ps.dims) == {"location", "time", "slope", "realization"}

    # t0: snow
    assert ps.sel(time=time[0]).item() == 2
    # t1: no snow but data present → 1
    assert ps.sel(time=time[1]).item() == 1

    mapping = ps.attrs["mapping"]
    assert mapping[2] == "valid, with snow"
    assert mapping[1] == "valid, no snow"
    assert mapping[0] == "unknown / no data"
    assert mapping[-1] == "error during parsing"


@pytest.mark.synthetic
def test_update_profile_status_HS_fallback(make_xsnow):
    """If z missing, HS > 0 → 2, HS == 0 with data → 1."""
    # dims: (location=1, time=2, slope=1, realization=1)
    hs = np.empty((1, 2, 1, 1), dtype=float)
    hs[0, 0, 0, 0] = 0.0   # t0: no snow
    hs[0, 1, 0, 0] = 0.2   # t1: snow

    T = np.empty((1, 2, 1, 1), dtype=float)
    T[0, :, 0, 0] = 273.15  # data both times

    ds = make_xsnow(
        data_vars={
            "HS": (("location", "time", "slope", "realization"), hs),
            "T": (("location", "time", "slope", "realization"), T),
        },
        drop_layer=True,
    ).update_profile_status()

    time = ds.coords["time"].values
    ps = ds["profile_status"]

    # t0: HS==0 but has data → 1
    assert ps.sel(time=time[0]).item() == 1
    # t1: HS>0 → 2
    assert ps.sel(time=time[1]).item() == 2


@pytest.mark.synthetic
def test_update_profile_status_preserves_negative(make_xsnow):
    """Existing negative profile_status must not be overridden by snow detection."""
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

    assert ps.item() == -1  # unchanged despite snow


@pytest.mark.synthetic
def test_update_profile_status_respects_data_vars_hint(make_xsnow):
    """
    Hinted vars are used first, but remaining candidate slots are filled
    with other suitable vars up to max_data_vars.
    Even if the hinted var is all-NaN, another suitable var with data
    should still upgrade status from 0 to 1.
    """
    # one time step
    hinted = np.array([[[[np.nan]]]])  # no info
    other = np.array([[[[1.0]]]])      # real data

    ds = make_xsnow(
        data_vars={
            "hint_var": (("location", "time", "slope", "realization"), hinted),
            "other_var": (("location", "time", "slope", "realization"), other),
        },
        drop_layer=True,
        single_time=True,
    )

    # With max_data_vars=2, we use: ['hint_var', 'other_var']
    ds = ds.update_profile_status(data_vars_hint=["hint_var"], max_data_vars=2)
    ps = ds["profile_status"]

    # other_var is considered and has data → should become 1
    assert ps.item() == 1


@pytest.mark.synthetic
def test_update_profile_status_unknown_when_no_useful_data(make_xsnow):
    """
    If no snow and no data vars with core dims, status remains 0.
    """
    ds = make_xsnow(
        data_vars={
            "meta_only": (("location",), [1.0]),  # missing time/slope/realization dims
        },
        drop_layer=True,
        single_time=True,
    ).update_profile_status()

    ps = ds["profile_status"]

    assert ps.item() == 0
    assert ps.attrs["mapping"][0] == "unknown / no data"
