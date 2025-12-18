from __future__ import annotations

import importlib.util
import json
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xsnow.extensions import classification
from xsnow import xsnowDataset


# ---------------------- fixtures ---------------------------------------------
@pytest.fixture
def tiny_ds_xr() -> xr.Dataset:
    """
    Build a very small, fully in-memory xarray.Dataset suitable for classification tests.
    Dims: location(2), time(3), slope(1), realization(1), layer(4).
    Vars:
      - HS: (loc, time, slope, realization)         -> scalar per profile
      - density, grain_size: (..., layer)           -> per-layer
      - temp: (..., layer)                          -> per-layer
    """
    loc = ["A", "B"]
    time = pd.date_range("2025-01-01", periods=3, freq="D").values.astype("datetime64[ns]")
    slope = [0]
    realization = [0]
    layer = np.arange(4)

    shape_scalar = (len(loc), len(time), len(slope), len(realization))
    shape_layer  = (len(loc), len(time), len(slope), len(realization), len(layer))

    rng = np.random.default_rng(42)
    HS = xr.DataArray(  # snow height ~ [0.5..2.0]
        0.5 + 1.5 * rng.random(shape_scalar),
        dims=("location", "time", "slope", "realization"),
        coords={"location": loc, "time": time, "slope": slope, "realization": realization},
        name="HS",
    )
    density = xr.DataArray(  # [150..450] kg/m^3
        150.0 + 300.0 * rng.random(shape_layer),
        dims=("location", "time", "slope", "realization", "layer"),
        coords={"location": loc, "time": time, "slope": slope, "realization": realization, "layer": layer},
        name="density",
    )
    grain_size = xr.DataArray(  # [0.1..1.0] mm
        0.1 + 0.9 * rng.random(shape_layer),
        dims=("location", "time", "slope", "realization", "layer"),
        coords={"location": loc, "time": time, "slope": slope, "realization": realization, "layer": layer},
        name="grain_size",
    )
    temp = xr.DataArray(  # [-15 .. 0] C
        -15.0 + 15.0 * rng.random(shape_layer),
        dims=("location", "time", "slope", "realization", "layer"),
        coords={"location": loc, "time": time, "slope": slope, "realization": realization, "layer": layer},
        name="temperature",
    )

    ds = xr.Dataset({"HS": HS, "density": density, "grain_size": grain_size, "temperature": temp})
    # Add a timezone attr for completeness
    ds["time"].attrs["timezone"] = "UTC"
    return ds


@pytest.fixture
def tiny_xsnow(tiny_ds_xr) -> xsnowDataset:
    """Wrap the tiny xr.Dataset in xsnowDataset."""
    return xsnowDataset(tiny_ds_xr)


@pytest.fixture
def tiny_xsnow_geo(tiny_ds_xr) -> xsnowDataset:
    """Tiny dataset with latitude/longitude coords for spatial classification."""
    ds = tiny_ds_xr.assign_coords(
        latitude=("location", np.array([0.0, 10.0], dtype=float)),
        longitude=("location", np.array([0.0, 10.0], dtype=float)),
    )
    ds.attrs["crs"] = "EPSG:4326"
    return xsnowDataset(ds)


@pytest.fixture
def tiny_xsnow_proj(tiny_ds_xr) -> xsnowDataset:
    """Tiny dataset with projected easting/northing coords for spatial classification."""
    ds = tiny_ds_xr.assign_coords(
        easting=("location", np.array([0.0, 1000.0], dtype=float)),
        northing=("location", np.array([0.0, 1000.0], dtype=float)),
    )
    ds.attrs["crs"] = "EPSG:3857"
    return xsnowDataset(ds)


# ---------------------- tests: classify (callable-based) ----------------------

def test_classify_attach_coord_from_dataset(tiny_xsnow: xsnowDataset):
    # Classify using HS from the *dataset* (func receives xr.Dataset)
    def by_hs(xds: xr.Dataset) -> xr.DataArray:
        da = (xds["HS"] > 1.25)
        # convert to int labels {0,1} and attach mapping
        out = da.astype(int).rename("hs_class")
        out.attrs["mapping"] = {0: "shallow", 1: "deep"}
        return out

    out = tiny_xsnow.classify(by_hs, output_name="hs_class", output_kind="coord")
    assert isinstance(out, xsnowDataset)
    assert 'mapping' in out.hs_class.attrs  # attrs were injected through function by_hs()
    assert "hs_class" in out.data.coords
    assert set(np.unique(out.data["hs_class"].values)).issubset({0, 1})


def test_classify_attach_data_from_variable(tiny_xsnow: xsnowDataset):
    # Classify passing a variable (func receives xr.DataArray)
    def by_density(da: xr.DataArray) -> xr.DataArray:
        # per-layer classification
        lab = xr.apply_ufunc(np.where, da > 300.0, 2, 1, dask="allowed", output_dtypes=[int])
        lab = xr.DataArray(lab, dims=da.dims, coords=da.coords, name="rho_class")
        lab.attrs["mapping"] = {1: "low", 2: "high"}
        return lab

    out = tiny_xsnow.classify(by_density, output_name="rho_class", output_kind="data", 
                              input_var="density", attrs={"mapping": {1: "low", 2: "high"}})
    assert isinstance(out, xsnowDataset)
    assert "rho_class" in out.data
    assert 'mapping' in out.rho_class.attrs  # attrs were injected through argument
    assert out.data["rho_class"].sizes["layer"] == tiny_xsnow.data.sizes["layer"]


def test_classify_by_bins_helper_works_with_classify(tiny_xsnow: xsnowDataset):
    bins = np.array([0.75, 1.25])
    out = tiny_xsnow.classify(
        classification.classify_by_bins,
        output_name="HS_bin",
        output_kind="coord",
        input_var="HS",
        bins=bins,
        right=True,
    )
    assert isinstance(out, xsnowDataset)
    assert "HS_bin" in out.data.coords
    vals = out.data["HS_bin"].values
    assert np.issubdtype(vals.dtype, np.integer)
    assert vals.min() >= 0
    assert vals.max() <= len(bins)


def test_classify_raises_on_wrong_return_type(tiny_xsnow: xsnowDataset):
    def bad_func(xds: xr.Dataset):
        return 123  # not a DataArray

    with pytest.raises(TypeError):
        _ = tiny_xsnow.classify(bad_func, output_name="oops", output_kind="data")


# ---------------------- tests: classify_criteria (string parser) --------------

def test_classify_criteria_return_mask(tiny_xsnow: xsnowDataset):
    # Per-layer criteria mask
    mask = tiny_xsnow.classify_criteria("density < 200 & grain_size > 0.8", kind="mask",
                                        attrs={"mapping": {0: "dense or small grained",
                                                           1: "low density and large grains"}})
    assert isinstance(mask, xr.DataArray)
    assert mask.dtype == bool
    assert "layer" in mask.dims
    assert "mapping" in mask.attrs
    # sanity: mask aligns with data dims subset
    assert set(mask.dims).issubset(set(tiny_xsnow.data["density"].dims))


def test_classify_criteria_attach_coord_and_use(tiny_xsnow: xsnowDataset):
    ds2 = tiny_xsnow.classify_criteria("HS > 1.0", name="hs_gt_1", kind="coord")
    assert isinstance(ds2, xsnowDataset)
    assert "hs_gt_1" in ds2.data.coords

    # Reduce the multi-d mask to a 1D mask along time
    mask = ds2.data["hs_gt_1"]  # dims: location,time,slope,realization
    reduce_dims = [d for d in mask.dims if d != "time"]
    time_mask = mask.any(dim=reduce_dims)

    # Now the boolean indexer matches the time dimension
    sel = ds2.data.sel(time=time_mask)
    assert sel.sizes["time"] <= ds2.data.sizes["time"]
    # Ensure at least one True exists to make this a meaningful selection
    assert bool(time_mask.any())


def test_classify_criteria_invalid_var_raises(tiny_xsnow: xsnowDataset):
    with pytest.raises(ValueError):
        _ = tiny_xsnow.classify_criteria("nope > 3", kind="mask")


def test_classify_criteria_invalid_op_raises(tiny_xsnow: xsnowDataset):
    with pytest.raises(ValueError):
        _ = tiny_xsnow.classify_criteria("density >> 300", kind="mask")


# ---------------------- tests: mask_by_criteria -------------------------------

def test_mask_by_criteria_preserves_shape_and_masks_layer_vars(tiny_xsnow: xsnowDataset):
    ds_masked = tiny_xsnow.mask_by_criteria("density < 200 & grain_size > 0.8")
    assert isinstance(ds_masked, xsnowDataset)
    # shape preserved
    for d in tiny_xsnow.data.dims:
        assert ds_masked.data.sizes[d] == tiny_xsnow.data.sizes[d]

    # Variables with layer dim should have NaNs where mask is False
    mask = tiny_xsnow.classify_criteria("density < 200 & grain_size > 0.8", kind="mask")
    dens_masked = ds_masked.data["density"]
    # Collect a position that is False in the mask and check becomes NaN in data
    # (flatten to find one quickly)
    flat_mask = mask.values.ravel()
    if (~flat_mask).any():
        idx_false = np.where(~flat_mask)[0][0]
        # unravel idx into dims
        unr = np.unravel_index(idx_false, mask.shape)
        # build indexers
        indexers = {dim: coord.values[i] for dim, i, coord in zip(mask.dims, unr, [mask.coords[d] for d in mask.dims])}
        # Focus on same indexers for density (aligning dims)
        val = dens_masked.sel(**indexers).item()
        assert np.isnan(val)  # masked out


def test_mask_by_criteria_leaves_scalars_intact(tiny_xsnow: xsnowDataset):
    # HS has no 'layer' dim; mask should not nuke HS values
    ds_masked = tiny_xsnow.mask_by_criteria("density > 350")
    # Values remain finite (not all NaN)
    assert np.isfinite(ds_masked.data["HS"].values).all()


# ---------------------- roundtrip sanity & chaining ---------------------------

def test_chaining_returns_xsnowDataset(tiny_xsnow: xsnowDataset):
    ds2 = tiny_xsnow.classify_criteria("HS > 0.6", name="hs_hi", kind="coord")
    assert isinstance(ds2, xsnowDataset)

    # Reduce to 1D time mask for selection
    mask = ds2.data["hs_hi"]
    reduce_dims = [d for d in mask.dims if d != "time"]
    time_mask = mask.any(dim=reduce_dims)

    ds3 = ds2.sel(time=time_mask)
    assert isinstance(ds3, xsnowDataset)


# ---------------------- tests: classify_spatially -----------------------------

def test_classify_spatially_attach_coord_from_wkt_mapping(tiny_xsnow_geo: xsnowDataset):
    polys = {
        "zoneA": "POLYGON ((-1 -1, -1 1, 1 1, 1 -1, -1 -1))",
    }
    ds2 = tiny_xsnow_geo.classify_spatially(polys, name="zone", kind="coord", polygons_crs="EPSG:4326")
    assert isinstance(ds2, xsnowDataset)
    assert "zone" in ds2.data.coords
    assert ds2.zone.sel(location="A").item() == "zoneA"
    assert ds2.zone.sel(location="B").item() == "outside"

    grouped = ds2.data.groupby("zone").mean(dim="location")
    assert "zone" in grouped.dims


def test_classify_spatially_return_mask(tiny_xsnow_geo: xsnowDataset):
    polys = {
        "zoneA": "POLYGON ((-1 -1, -1 1, 1 1, 1 -1, -1 -1))",
        "zoneB": "POLYGON ((9 9, 9 11, 11 11, 11 9, 9 9))",
    }
    mask = tiny_xsnow_geo.classify_spatially(polys, name="in_zone", kind="mask", polygons_crs="EPSG:4326")
    assert isinstance(mask, xr.DataArray)
    assert mask.dtype == bool
    assert mask.dims[0] == "polygon"
    assert set(mask.coords["polygon"].values.tolist()) == {"zoneA", "zoneB"}
    assert bool(mask.sel(polygon="zoneA", location="A").item()) is True
    assert bool(mask.sel(polygon="zoneA", location="B").item()) is False
    assert bool(mask.sel(polygon="zoneB", location="B").item()) is True


def test_classify_spatially_from_geojson_file(tmp_path, tiny_xsnow_geo: xsnowDataset):
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "zoneA"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[-1, -1], [-1, 1], [1, 1], [1, -1], [-1, -1]],
                    ],
                },
            }
        ],
    }
    p = tmp_path / "zones.geojson"
    p.write_text(json.dumps(geojson), encoding="utf-8")

    ds2 = tiny_xsnow_geo.classify_spatially(str(p), name="zone", kind="coord", polygons_crs="EPSG:4326")
    assert ds2.zone.sel(location="A").item() == "zoneA"
    assert ds2.zone.sel(location="B").item() == "outside"


def test_classify_spatially_projected_uses_easting_northing(tiny_xsnow_proj: xsnowDataset):
    polys = {
        "zoneA": "POLYGON ((-10 -10, -10 10, 10 10, 10 -10, -10 -10))",
    }
    ds2 = tiny_xsnow_proj.classify_spatially(polys, name="zone", kind="coord", polygons_crs="EPSG:3857")
    assert ds2.zone.sel(location="A").item() == "zoneA"
    assert ds2.zone.sel(location="B").item() == "outside"


def test_classify_spatially_crs_transform_if_pyproj_available(tiny_xsnow_geo: xsnowDataset):
    pyproj = pytest.importorskip("pyproj")
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    xs, ys = transformer.transform([-1, -1, 1, 1, -1], [-1, 1, 1, -1, -1])
    coords = ", ".join(f"{x} {y}" for x, y in zip(xs, ys))
    wkt = f"POLYGON (({coords}))"

    ds2 = tiny_xsnow_geo.classify_spatially(
        {"zoneA": wkt},
        name="zone",
        kind="coord",
        polygons_crs="EPSG:3857",
    )
    assert ds2.zone.sel(location="A").item() == "zoneA"
    assert ds2.zone.sel(location="B").item() == "outside"


def test_classify_spatially_raises_when_transform_needed_but_pyproj_missing(tiny_xsnow_geo: xsnowDataset):
    """
    Meaningful guardrail: if polygon CRS != dataset CRS, we must either transform
    with pyproj or fail with a clear error.
    """
    if importlib.util.find_spec("pyproj") is not None:
        pytest.skip("pyproj installed; transformation is available in this environment.")

    polys = {
        "zoneA": "POLYGON ((-1 -1, -1 1, 1 1, 1 -1, -1 -1))",
    }
    with pytest.raises(ValueError) as exc:
        _ = tiny_xsnow_geo.classify_spatially(polys, name="zone", kind="coord", polygons_crs="EPSG:3857")
    assert "pyproj" in str(exc.value).lower()
