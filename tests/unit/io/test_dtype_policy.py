# tests/io/test_dtype_policy.py
"""Tests for the float_dtype read policy."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xsnow import read, to_netcdf, to_zarr
from xsnow.combiners import _nan_like
from xsnow.io.base import GEO_COORD_NAMES, _apply_float_dtype
from xsnow.io.smet import read_smet


def _synthetic_dataset() -> xr.Dataset:
    time = pd.date_range("2024-01-01", periods=3)
    dims = ("location", "time")
    ds = xr.Dataset(
        data_vars={
            "density": (dims, np.ones((1, 3), dtype="float64")),
            "temperature": (dims, np.ones((1, 3), dtype="float32")),
            "profile_status": (dims, np.ones((1, 3), dtype="i1")),
        },
        coords={
            "location": ("location", ["STAT1"]),
            "time": ("time", time),
            "latitude": ("location", np.array([46.5], dtype="float64")),
            "longitude": ("location", np.array([11.3], dtype="float64")),
            "altitude": ("location", np.array([2500.0], dtype="float64")),
            "z": (("location", "time"), np.zeros((1, 3), dtype="float64")),
        },
    )
    ds["density"].attrs["units"] = "kg/m3"
    return ds


class TestApplyFloatDtype:
    def test_casts_float64_vars_and_coords(self):
        result = _apply_float_dtype(_synthetic_dataset(), "float32")

        assert result["density"].dtype == np.float32
        assert result["temperature"].dtype == np.float32
        assert result["z"].dtype == np.float32
        assert result["profile_status"].dtype == np.int8
        assert result["time"].dtype.kind == "M"
        for name in GEO_COORD_NAMES:
            assert result[name].dtype == np.float64

    def test_none_is_noop(self):
        ds = _synthetic_dataset()
        assert _apply_float_dtype(ds, None) is ds

    def test_preserves_attrs(self):
        result = _apply_float_dtype(_synthetic_dataset(), "float32")
        assert result["density"].attrs["units"] == "kg/m3"

    def test_does_not_mutate_input(self):
        ds = _synthetic_dataset()
        _apply_float_dtype(ds, "float32")
        assert ds["density"].dtype == np.float64

    def test_float64_upcasts(self):
        result = _apply_float_dtype(_synthetic_dataset(), "float64")
        assert result["temperature"].dtype == np.float64

    def test_invalid_dtype_raises(self):
        with pytest.raises(ValueError, match="float_dtype"):
            _apply_float_dtype(_synthetic_dataset(), "int32")


class TestNanLike:
    def test_preserves_float32(self):
        ref = xr.DataArray(np.ones(3, dtype="float32"))
        filled = _nan_like(ref)
        assert filled.dtype == np.float32
        assert filled.isnull().all()

    def test_int_ref_uses_float64(self):
        ref = xr.DataArray(np.ones(3, dtype="int64"))
        assert _nan_like(ref).dtype == np.float64


@pytest.mark.real_data
class TestReadFloatDtype:
    def test_smet_defaults_to_float32(self, gridded_pros_dir):
        ds = read(gridded_pros_dir / "VIR1A4.smet")

        float_vars = {
            name: da.dtype
            for name, da in ds.data.data_vars.items()
            if da.dtype.kind == "f"
        }
        assert float_vars
        assert all(dt == np.float32 for dt in float_vars.values())
        for name in GEO_COORD_NAMES:
            assert ds.data[name].dtype == np.float64

    def test_none_preserves_source(self, gridded_pros_dir):
        ds = read(gridded_pros_dir / "VIR1A4.smet", float_dtype=None)
        assert ds.data["TA"].dtype == np.float64

    def test_float64_upcasts_pro(self, gridded_pros_dir):
        ds = read(
            gridded_pros_dir / "VIR1A4.pro",
            unit_validation="light",
            float_dtype="float64",
        )
        assert ds.data["density"].dtype == np.float64

    def test_merged_pro_smet_is_float32(self, gridded_pros_dir):
        ds = read(
            [
                gridded_pros_dir / "VIR1A4.pro",
                gridded_pros_dir / "VIR1A4.smet",
            ],
            unit_validation="light",
        )
        assert ds.data["density"].dtype == np.float32
        assert ds.data["TA"].dtype == np.float32
        assert ds.data["z"].dtype == np.float32

    def test_invalid_dtype_fails_fast(self, gridded_pros_dir):
        with pytest.raises(ValueError, match="float_dtype"):
            read(gridded_pros_dir / "VIR1A4.smet", float_dtype="int32")

    def test_read_smet_standalone(self, gridded_pros_dir):
        ds = read_smet(gridded_pros_dir / "VIR1A4.smet")
        assert ds["TA"].dtype == np.float32
        assert ds["latitude"].dtype == np.float64

        ds64 = read_smet(gridded_pros_dir / "VIR1A4.smet", float_dtype=None)
        assert ds64["TA"].dtype == np.float64


@pytest.mark.real_data
class TestArchiveFloatDtype:
    @pytest.fixture()
    def float64_archive_source(self, gridded_pros_dir):
        return read(gridded_pros_dir / "VIR1A4.smet", float_dtype=None)

    def test_netcdf_roundtrip(self, float64_archive_source, tmp_path):
        path = tmp_path / "rt.nc"
        to_netcdf(float64_archive_source, path)

        assert read(path).data["TA"].dtype == np.float32
        assert read(path, float_dtype=None).data["TA"].dtype == np.float64
        assert read(path, lazy=False).data["TA"].dtype == np.float32

    def test_zarr_roundtrip_stays_lazy(self, float64_archive_source, tmp_path):
        path = tmp_path / "rt.zarr"
        to_zarr(float64_archive_source, path, mode="w")

        lazy_ds = read(path, lazy=True)
        assert lazy_ds.data["TA"].chunks is not None
        assert lazy_ds.data["TA"].dtype == np.float32

        eager = read(path, lazy=False, memory_limit=None)
        assert eager.data["TA"].dtype == np.float32
        assert eager.data["TA"].chunks is None
