# tests/core/test_units.py
"""Tests for generic unit normalization and conversion helpers."""

from unittest.mock import Mock

import numpy as np
import pytest
import xarray as xr
import xsnow

from xsnow.units import (
    UnitConversionError,
    convert_units,
    normalize_unit,
    validate_xsnow_unit_attrs,
)


def test_normalize_unit_handles_common_spellings():
    """Unit normalization accepts SMET-style and CF-like spellings."""
    assert normalize_unit("W/m2") == "W m-2"
    assert normalize_unit("kg/m2/h") == "kg m-2 h-1"
    assert normalize_unit("m/s") == "m s-1"
    assert normalize_unit("°") == "degree"
    assert normalize_unit("-") == "1"
    assert normalize_unit("ISO8601") == "ISO"


def test_convert_units_numeric_values():
    """Unit conversion handles offset and multiplicative conversions."""
    np.testing.assert_allclose(convert_units(np.array([0.0, 100.0]), "cm", "m"), [0.0, 1.0])
    np.testing.assert_allclose(convert_units(np.array([0.0, 1.0]), "1", "%"), [0.0, 100.0])
    np.testing.assert_allclose(convert_units(np.array([0.0, -10.0]), "degC", "K"), [273.15, 263.15])
    np.testing.assert_allclose(convert_units(np.array([3600.0]), "kg m-2 h-1", "kg m-2 s-1"), [1.0])
    assert convert_units(10.0, "kPa", "Pa") == 10000.0


def test_convert_units_raises_for_unknown_conversion():
    """Unsupported conversions raise explicitly."""
    with pytest.raises(UnitConversionError):
        convert_units([1.0], "kg m-2", "m")


def test_dataarray_units_to_explicit_unit():
    """DataArray.units.to converts values from attrs['units']."""
    da = xr.DataArray([0.01, 0.02], dims="layer", name="grain_size", attrs={"units": "m"})

    out = da.units.to("mm")

    np.testing.assert_allclose(out.values, [10.0, 20.0])
    assert out.attrs["units"] == "mm"
    assert da.attrs["units"] == "m"


def test_dataarray_units_exposes_current_and_plot_units():
    """The units accessor exposes metadata and registered plot units."""
    da = xr.DataArray([0.01, 0.02], dims="layer", name="grain_size", attrs={"units": "m"})

    assert da.units.current == "m"
    assert da.units.plot == xsnow.registry["grain_size"]["plot_units"]


def test_dataarray_units_to_registered_plot_unit():
    """DataArray.units.to_plot_unit uses registry plot_units."""
    da = xr.DataArray([0.01, 0.02], dims="layer", name="grain_size", attrs={"units": "m"})

    out = da.units.to_plot_unit()

    np.testing.assert_allclose(out.values, [10.0, 20.0])
    assert out.attrs["units"] == xsnow.registry["grain_size"]["plot_units"]


def test_units_accessor_on_xsnow_dataset_variable(xsnow_dataset):
    """Variables selected from an xsnowDataset expose the units accessor."""
    out = xsnow_dataset["height"].units.to_plot_unit()

    assert isinstance(xsnow_dataset, xsnow.xsnowDataset)
    assert isinstance(out, xr.DataArray)
    np.testing.assert_allclose(
        out.values,
        xsnow_dataset["height"].values * 100.0,
    )
    assert out.attrs["units"] == "cm"
    assert xsnow_dataset["height"].attrs["units"] == "m"


def test_dataarray_units_missing_plot_unit_leaves_values_unchanged():
    """Unknown variables without plot_units are returned unchanged."""
    da = xr.DataArray([1.0, 2.0], dims="layer", name="unknown_var", attrs={"units": "m"})

    out = da.units.to_plot_unit()

    np.testing.assert_allclose(out.values, da.values)
    assert out.attrs["units"] == "m"
    assert da.units.plot is None
    assert out is not da


def test_dataarray_units_to_requires_source_unit():
    """Explicit DataArray conversion requires attrs['units']."""
    da = xr.DataArray([1.0, 2.0], dims="layer", name="height")

    with pytest.raises(UnitConversionError, match=r"attrs\['units'\] is missing"):
        da.units.to("cm")


def test_validate_xsnow_unit_attrs_accepts_xsnow_dataset(xsnow_dataset):
    """Unit validation accepts an xsnowDataset and returns True when valid."""
    is_valid = validate_xsnow_unit_attrs(xsnow_dataset)

    assert is_valid is True


def test_validate_xsnow_unit_attrs_checks_xsnow_dataset(xsnow_dataset):
    """Strict validation detects mismatched attrs on an xsnowDataset."""
    xsnow_dataset["height"].attrs["units"] = "cm"

    with pytest.raises(xsnow.units.UnitValidationError, match="height"):
        validate_xsnow_unit_attrs(xsnow_dataset)


def test_validate_xsnow_unit_attrs_light_returns_false_and_logs(xsnow_dataset):
    """Light validation reports strict incompatibility without raising."""
    xsnow_dataset["height"].attrs["units"] = "cm"
    logger = Mock()

    is_valid = validate_xsnow_unit_attrs(
        xsnow_dataset,
        unit_validation="light",
        logger=logger,
    )

    assert is_valid is False
    logger.warning.assert_called_once()
