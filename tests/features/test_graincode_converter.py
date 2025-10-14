import pytest
import xarray as xr
import numpy as np
import xsnow
from xsnow.graincode_converter import slf_graincode_to_ICSSG_primary, slf_graincode_to_ICSSG_secondary


@pytest.mark.parametrize(
    "codes, expected_primary, expected_secondary",
    [
        # --- 1-digit codes: primary == secondary, cycle defaults to 'dry' ---
        ([0], ["PPgp"], ["PPgp"]),
        ([3], ["RG"],   ["RG"]),
        ([9], ["FCxr"], ["FCxr"]),

        # --- 2-digit codes: (d0,d1), cycle defaults to 'dry' ---
        ([33], ["RG"],  ["RG"]),
        ([45], ["FC"],  ["DH"]),
        ([78], ["MF"],  ["IF"]),

        # --- 3-digit codes: (d0,d1,d2), d2 is cycle: 0='dry', 1='mel', 2='rfr' ---
        # cycle=0 → primary unaffected
        ([330], ["RG"], ["RG"]),
        ([120], ["PP"], ["DF"]),

        # cycle=1 ('mel') → primary unaffected
        ([331], ["RG"], ["RG"]),
        ([241], ["DF"], ["FC"]),

        # cycle=2 ('rfr') → SPECIAL RULE: primary forced to 'MFcr'
        ([332], ["MFcr"], ["RG"]),
        ([742], ["MFcr"], ["FC"]),
    ],
)
def test_primary_secondary_basic_cases(codes, expected_primary, expected_secondary):
    da = xr.DataArray(np.array(codes))
    prim = slf_graincode_to_ICSSG_primary(da)
    sec  = slf_graincode_to_ICSSG_secondary(da)

    assert prim.dtype == object
    assert sec.dtype == object
    assert prim.shape == da.shape
    assert sec.shape == da.shape
    assert prim.to_series().tolist() == expected_primary
    assert sec.to_series().tolist() == expected_secondary


def test_nan_and_invalid_handling():
    # Includes NaN, None, a valid 3-digit with rfr, and an out-of-range code (1005)
    da = xr.DataArray(np.array([np.nan, None, 332, 1005], dtype=object))

    prim = slf_graincode_to_ICSSG_primary(da)
    sec  = slf_graincode_to_ICSSG_secondary(da)

    # Expect None for missing/invalid; 'MFcr' for 332 primary; 'RG' for 332 secondary
    assert prim.to_series().tolist() == [None, None, "MFcr", None]
    assert sec.to_series().tolist()  == [None, None, "RG",   None]


def test_shape_and_coords_preserved():
    # 2x2 grid with coordinates
    codes = xr.DataArray(
        np.array([[33, 332], [45, 120]]),
        dims=("location", "time"),
        coords={"location": ["A", "B"], "time": [0, 1]},
        name="grain_code",
    )

    prim = slf_graincode_to_ICSSG_primary(codes)
    sec  = slf_graincode_to_ICSSG_secondary(codes)

    assert prim.dims == codes.dims
    assert sec.dims == codes.dims
    assert all(prim.coords[d].equals(codes.coords[d]) for d in codes.dims)
    assert all(sec.coords[d].equals(codes.coords[d]) for d in codes.dims)
    assert prim.values.tolist() == [["RG", "MFcr"], ["FC", "PP"]]
    assert sec.values.tolist()  == [["RG", "RG"],   ["DH", "DF"]]


@pytest.mark.parametrize(
    "arr_like",
    [
        np.array([33, 332, 45], dtype=float),
        [33, 332, 45],
        xr.DataArray([33, 332, 45]),
    ],
)
def test_accepts_various_array_likes(arr_like):
    prim = slf_graincode_to_ICSSG_primary(arr_like)
    sec  = slf_graincode_to_ICSSG_secondary(arr_like)

    # Return type should be DataArray in all entry points (per implementation)
    assert isinstance(prim, xr.DataArray)
    assert isinstance(sec, xr.DataArray)
    assert prim.values.tolist() == ["RG", "MFcr", "FC"]
    assert sec.values.tolist()  == ["RG", "RG",   "DH"]


def test_broadcasting_over_extra_dims():
    # (location=2, time=2, layer=3)
    data = np.array([
        [[33, 332, 45], [120, 742, 9]],
        [[33,  33,  33], [241, 121,  0]],
    ])
    da = xr.DataArray(
        data,
        dims=("location", "time", "layer"),
        coords={
            "location": ["L1", "L2"],
            "time": [0, 1],
            "layer": np.arange(3),
        },
    )

    prim = slf_graincode_to_ICSSG_primary(da)
    sec  = slf_graincode_to_ICSSG_secondary(da)

    # da[0,0,:] = [33,332,45]
    # 33 → ('RG','RG','dry')        → prim 'RG',   sec 'RG'
    # 332→ ('RG','RG','rfr')        → prim 'MFcr', sec 'RG'  (special rule)
    # 45 → ('FC','DH','dry')        → prim 'FC',   sec 'DH'
    assert prim.isel(location=0, time=0).values.tolist() == ["RG", "MFcr", "FC"]
    assert sec.isel(location=0, time=0).values.tolist()  == ["RG", "RG",   "DH"]

    # da[1,1,:] = [241,121,0]
    # 241 → ('DF','FC','mel')       → prim 'DF',   sec 'FC'
    # 121 → ('PP','DF','mel')       → prim 'PP',   sec 'DF'
    # 0   → ('PPgp','PPgp','dry')   → prim 'PPgp', sec 'PPgp'
    assert prim.isel(location=1, time=1).values.tolist() == ["DF", "PP", "PPgp"]
    assert sec.isel(location=1, time=1).values.tolist()  == ["FC", "DF", "PPgp"]
