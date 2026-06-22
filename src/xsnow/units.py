# units.py
"""Generic unit normalization and conversion helpers for xsnow."""
from __future__ import annotations

import re
import logging
from typing import Any, Optional, Union

import numpy as np
import xarray as xr

UnitLike = Union[str, list[str]]


class UnitConversionError(ValueError):
    """Raised when xsnow does not know how to convert between two units."""


class UnitValidationError(ValueError):
    """Raised when stored unit metadata is incompatible with xsnow units."""


_UNIT_VALIDATION_MODES = {"light", "strict"}


UNIT_ALIASES = {
    "": "1",
    "-": "1",
    "1": "1",
    "%": "%",
    "degree": "degree",
    "degrees": "degree",
    "deg": "degree",
    "°": "degree",
    "c": "degC",
    "degc": "degC",
    "°c": "degC",
    "k": "K",
    "m": "m",
    "cm": "cm",
    "mm": "mm",
    "pa": "Pa",
    "kpa": "kPa",
    "gpa s": "GPa s",
    "pa s": "Pa s",
    "w/m2": "W m-2",
    "w m-2": "W m-2",
    "mj/m2": "MJ m-2",
    "mj m-2": "MJ m-2",
    "j/m2": "J m-2",
    "j m-2": "J m-2",
    "kg/m2": "kg m-2",
    "kg m-2": "kg m-2",
    "kg/m2/h": "kg m-2 h-1",
    "kg m-2 h-1": "kg m-2 h-1",
    "kg/m2/s": "kg m-2 s-1",
    "kg m-2 s-1": "kg m-2 s-1",
    "m/s": "m s-1",
    "m s-1": "m s-1",
    "s-1": "s-1",
    "1.e-6 s-1": "1.e-6 s-1",
    "1.0e-6 s-1": "1.e-6 s-1",
    "iso": "ISO",
    "iso8601": "ISO",
    "time": "ISO",
    "swiss code f1f2f3": "Swiss Code F1F2F3",
}

CONVERSION_FACTORS = {
    ("cm", "m"): 0.01,
    ("m", "cm"): 100.0,
    ("mm", "m"): 0.001,
    ("m", "mm"): 1000.0,
    ("mm", "cm"): 0.1,
    ("cm", "mm"): 10.0,
    ("%", "1"): 0.01,
    ("1", "%"): 100.0,
    ("kPa", "Pa"): 1000.0,
    ("Pa", "kPa"): 0.001,
    ("GPa s", "Pa s"): 1.0e9,
    ("Pa s", "GPa s"): 1.0e-9,
    ("MJ m-2", "J m-2"): 1.0e6,
    ("J m-2", "MJ m-2"): 1.0e-6,
    ("kg m-2 h-1", "kg m-2 s-1"): 1.0 / 3600.0,
    ("kg m-2 s-1", "kg m-2 h-1"): 3600.0,
    ("1.e-6 s-1", "s-1"): 1.0e-6,
    ("s-1", "1.e-6 s-1"): 1.0e6,
}


def normalize_unit(unit: Optional[str]) -> Optional[str]:
    """Return the canonical spelling xsnow uses for a unit string."""
    if unit is None:
        return None
    text = str(unit).strip()
    if not text:
        return "1"
    text = (
        text.replace("²", "2")
        .replace("⁻¹", "-1")
        .replace("⁻", "-")
        .replace("−", "-")
    )
    compact = re.sub(r"\s+", " ", text).strip()
    return UNIT_ALIASES.get(compact.lower(), compact)


def convert_units(values: Any, from_unit: str, to_unit: str) -> Any:
    """Convert numeric values between supported xsnow units."""
    source = normalize_unit(from_unit)
    target = normalize_unit(to_unit)

    if source == target:
        return values

    arr = np.asarray(values)
    scalar_input = np.isscalar(values)

    if source == "degC" and target == "K":
        converted = arr + 273.15
    elif source == "K" and target == "degC":
        converted = arr - 273.15
    else:
        try:
            factor = CONVERSION_FACTORS[(source, target)]
        except KeyError as exc:
            raise UnitConversionError(
                f"Cannot convert from {source!r} to {target!r}."
            ) from exc
        converted = arr * factor

    return converted.item() if scalar_input else converted


def validate_xsnow_unit_attrs(
    ds: xr.Dataset,
    *,
    unit_validation: str = "strict",
    logger: Optional[logging.Logger] = None,
    source: str = "dataset",
) -> xr.Dataset:
    """Validate stored variable unit attrs against xsnow's unit convention."""
    if unit_validation not in _UNIT_VALIDATION_MODES:
        raise UnitValidationError(
            "unit_validation must be 'light' or 'strict' for stored datasets, "
            f"got {unit_validation!r}."
        )

    from .registry import registry

    for var_name, data_array in ds.data_vars.items():
        reg_entry = registry.get(var_name)
        if not reg_entry:
            continue
        expected_unit = reg_entry.get("xsnow_units")
        if expected_unit is None or isinstance(expected_unit, list):
            continue

        stored_unit = data_array.attrs.get("units")
        if stored_unit is None:
            message = (
                f"{source} variable {var_name!r} has no units attribute; "
                f"expected xsnow unit {expected_unit!r}."
            )
            if logger is not None:
                logger.warning(message)
            continue

        normalized_stored = normalize_unit(stored_unit)
        normalized_expected = normalize_unit(expected_unit)
        if normalized_stored == normalized_expected:
            continue

        message = (
            f"{source} variable {var_name!r} declares units {stored_unit!r}, "
            f"but xsnow expects {normalized_expected!r}."
        )
        if unit_validation == "strict":
            raise UnitValidationError(message)
        if logger is not None:
            logger.warning(message)

    return ds
