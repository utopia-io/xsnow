# units.py
"""Generic unit normalization and conversion helpers for xsnow."""
from __future__ import annotations

import re
import logging
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from .core import xsnowDataset

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
    """
    Convert numeric values between units supported by xsnow.

    Unit spellings are normalized through :func:`normalize_unit` before the
    conversion is selected. Conversions may be multiplicative, such as metres
    to millimetres, or include an offset, such as degrees Celsius to kelvin.
    Supported multiplicative conversions are listed in
    :data:`CONVERSION_FACTORS`.

    Parameters
    ----------
    values : scalar or array-like
        Numeric values to convert. NumPy, xarray, and lazy array-like objects
        are preserved where their arithmetic operations support the requested
        conversion. Python lists and tuples are returned as NumPy arrays.
    from_unit : str
        Unit currently represented by ``values``.
    to_unit : str
        Unit into which ``values`` should be converted.

    Returns
    -------
    scalar or array-like
        Converted values. The input object is not modified. Scalar inputs
        produce scalar outputs.

    Raises
    ------
    UnitConversionError
        If xsnow has no conversion from ``from_unit`` to ``to_unit``.

    Notes
    -----
    This function converts values only; it does not update unit metadata. For
    an xarray DataArray, use the xsnow `UnitsAccessor` ``da.units.to(...)`` to preserve coordinates and
    attributes while also updating ``attrs["units"]``.

    Examples
    --------
    >>> from xsnow.units import convert_units
    >>> convert_units(25.0, "cm", "m")
    0.25
    >>> convert_units([0.0, 1.0], "1", "%")
    array([  0., 100.])
    """
    source = normalize_unit(from_unit)
    target = normalize_unit(to_unit)

    if source == target:
        return values

    scalar_input = np.isscalar(values)
    if scalar_input or isinstance(values, (list, tuple)):
        arr = np.asarray(values)
    else:
        arr = values

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
    ds: Union[xr.Dataset, "xsnowDataset"],
    *,
    unit_validation: str = "strict",
    logger: Optional[logging.Logger] = None,
    source: str = "dataset",
) -> bool:
    """
    Validate dataset unit attributes against xsnow's unit convention.

    Each registered data variable with a scalar ``xsnow_units`` entry is
    compared with its stored ``attrs["units"]`` value after normalizing common
    unit spellings. This function validates metadata only: it neither converts
    values nor changes attributes.

    Variables absent from the registry, registry entries without
    ``xsnow_units``, and compound variables whose ``xsnow_units`` entry is a
    list are skipped. Missing ``attrs["units"]`` metadata does not raise, but
    is reported through ``logger`` when one is provided.

    Parameters
    ----------
    ds : xarray.Dataset or xsnow.xsnowDataset
        Dataset whose data-variable unit attributes should be validated. An
        ``xsnowDataset`` is validated through its underlying xarray Dataset.
    unit_validation : {"strict", "light"}, default "strict"
        In ``"strict"`` mode, a stored unit that differs from the registered
        xsnow unit raises :class:`UnitValidationError`. In ``"light"`` mode,
        the mismatch is reported through ``logger`` and validation continues.
        Missing unit metadata never raises in either mode.
    logger : logging.Logger, optional
        Logger used for missing-unit and light-mode mismatch warnings. If no
        logger is supplied, these non-fatal conditions remain silent.
    source : str, default "dataset"
        Description of the dataset source included in warnings and exceptions,
        for example a NetCDF or Zarr path.

    Returns
    -------
    bool
        ``True`` when no incompatible stored units are found. In light mode,
        ``False`` is returned when a mismatch is logged. The same mismatch
        raises :class:`UnitValidationError` in strict mode instead of returning.

    Raises
    ------
    UnitValidationError
        If ``unit_validation`` is invalid, or if strict validation finds a
        stored unit that differs from the registered xsnow unit.

    Examples
    --------
    ``import xsnow`` makes the units module available as ``xsnow.units``:

    >>> import xarray as xr
    >>> import xsnow
    >>> ds = xr.Dataset({
    ...     "height": xr.DataArray([0.1, 0.2], dims="layer", attrs={"units": "m"})
    ... })
    >>> xsnow.units.validate_xsnow_unit_attrs(ds)
    True
    >>> xs = xsnow.xsnowDataset(ds)
    >>> xsnow.units.validate_xsnow_unit_attrs(xs)
    True

    Use ``DataArray.units.to(...)`` when values should be converted rather than
    merely validated:

    >>> height_cm = ds["height"].units.to("cm")
    >>> height_cm.attrs["units"]
    'cm'
    """
    if unit_validation not in _UNIT_VALIDATION_MODES:
        raise UnitValidationError(
            "unit_validation must be 'light' or 'strict' for stored datasets, "
            f"got {unit_validation!r}."
        )

    from .core import xsnowDataset
    from .registry import registry

    dataset = ds.to_xarray() if isinstance(ds, xsnowDataset) else ds
    is_valid = True

    for var_name, data_array in dataset.data_vars.items():
        reg_entry = registry.get(var_name)
        if not reg_entry:
            continue
        expected_unit = reg_entry.get("xsnow_units")
        if expected_unit is None or isinstance(expected_unit, list):
            continue
        if np.issubdtype(data_array.dtype, np.datetime64):
            # datetime vars carry no units attr (CF encoding owns it)
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
        is_valid = False
        if unit_validation == "strict":
            raise UnitValidationError(message)
        if logger is not None:
            logger.warning(message)

    return is_valid
