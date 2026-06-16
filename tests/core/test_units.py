# tests/core/test_units.py
"""Tests for generic unit normalization and conversion helpers."""

import numpy as np
import pytest

from xsnow.units import UnitConversionError, convert_units, normalize_unit


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
