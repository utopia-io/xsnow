# tests/io/test_netcdf_zarr_units.py
"""Tests for NetCDF/Zarr stored unit metadata validation."""

import logging
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from xsnow import read
from xsnow.units import UnitValidationError


def test_read_netcdf_preserves_matching_unit_attrs(tmp_path, stored_units_xr_dataset):
    """NetCDF read preserves attrs when stored units match xsnow units."""
    nc_file = tmp_path / "matching_units.nc"
    stored_units_xr_dataset("m").to_netcdf(nc_file)

    loaded = read(nc_file)

    assert loaded["height"].attrs["units"] == "m"
    assert loaded["height"].attrs["custom_attr"] == "preserve-me"
    np.testing.assert_allclose(loaded["height"].values, [[[[[0.1, 0.2]]]]])


def test_read_netcdf_strict_raises_on_mismatched_unit_attrs(
    tmp_path,
    stored_units_xr_dataset,
):
    """Strict NetCDF validation fails when stored units differ from xsnow units."""
    nc_file = tmp_path / "mismatched_units.nc"
    stored_units_xr_dataset("cm").to_netcdf(nc_file)

    with pytest.raises(UnitValidationError, match="declares units 'cm'.*expects 'm'"):
        read(nc_file)


def test_read_netcdf_light_warns_and_preserves_mismatched_unit_attrs(
    tmp_path,
    stored_units_xr_dataset,
    caplog,
):
    """Light NetCDF validation warns but keeps legacy unit attrs unchanged."""
    nc_file = tmp_path / "legacy_units.nc"
    stored_units_xr_dataset("cm").to_netcdf(nc_file)
    logger = logging.getLogger("tests.io.test_netcdf_zarr_units.netcdf")

    with caplog.at_level(logging.WARNING, logger=logger.name):
        loaded = read(nc_file, unit_validation="light", logger=logger)

    assert "declares units 'cm'" in caplog.text
    assert loaded["height"].attrs["units"] == "cm"
    assert loaded["height"].attrs["custom_attr"] == "preserve-me"
    np.testing.assert_allclose(loaded["height"].values, [[[[[0.1, 0.2]]]]])


def _mock_zarr_store(tmp_path, monkeypatch, ds: xr.Dataset) -> Path:
    """Create a minimal Zarr path and mock xarray's backend reader."""
    zarr_path = tmp_path / "stored_units.zarr"
    zarr_path.mkdir()
    (zarr_path / "zarr.json").write_text("{}")
    monkeypatch.setattr(xr, "open_zarr", lambda path, **kwargs: ds.copy())
    return zarr_path


def test_read_zarr_preserves_matching_unit_attrs(
    tmp_path,
    monkeypatch,
    stored_units_xr_dataset,
):
    """Zarr read preserves attrs when stored units match xsnow units."""
    zarr_path = _mock_zarr_store(tmp_path, monkeypatch, stored_units_xr_dataset("m"))

    loaded = read(zarr_path)

    assert loaded["height"].attrs["units"] == "m"
    assert loaded["height"].attrs["custom_attr"] == "preserve-me"
    np.testing.assert_allclose(loaded["height"].values, [[[[[0.1, 0.2]]]]])


def test_read_zarr_strict_raises_on_mismatched_unit_attrs(
    tmp_path,
    monkeypatch,
    stored_units_xr_dataset,
):
    """Strict Zarr validation fails when stored units differ from xsnow units."""
    zarr_path = _mock_zarr_store(tmp_path, monkeypatch, stored_units_xr_dataset("cm"))

    with pytest.raises(UnitValidationError, match="declares units 'cm'.*expects 'm'"):
        read(zarr_path)


def test_read_zarr_light_warns_and_preserves_mismatched_unit_attrs(
    tmp_path,
    monkeypatch,
    stored_units_xr_dataset,
    caplog,
):
    """Light Zarr validation warns but keeps legacy unit attrs unchanged."""
    zarr_path = _mock_zarr_store(tmp_path, monkeypatch, stored_units_xr_dataset("cm"))
    logger = logging.getLogger("tests.io.test_netcdf_zarr_units.zarr")

    with caplog.at_level(logging.WARNING, logger=logger.name):
        loaded = read(zarr_path, unit_validation="light", logger=logger)

    assert "declares units 'cm'" in caplog.text
    assert loaded["height"].attrs["units"] == "cm"
    assert loaded["height"].attrs["custom_attr"] == "preserve-me"
    np.testing.assert_allclose(loaded["height"].values, [[[[[0.1, 0.2]]]]])
