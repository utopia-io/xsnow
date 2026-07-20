# tests/io/test_smet_units.py
"""Tests for SMET-specific unit inference helpers."""

import logging

import numpy as np
import pytest

from xsnow import read
from xsnow.io.smet import infer_smet_storage_unit, parse_smet_unit_specs, read_smet
from xsnow.registry import registry


@pytest.mark.real_data
def test_parse_smet_unit_specs_from_snowpack_output(gridded_pros_dir):
    """SMET unit inference derives storage units from plot metadata."""
    specs = parse_smet_unit_specs(
        gridded_pros_dir / "VIR1A.smet",
        use_registry_for_ambiguous=True,
    )

    assert specs["timestamp"].storage_unit == "ISO"
    assert specs["TA"].plot_unit == "K"
    assert specs["TA"].storage_unit == "degC"
    assert specs["HS_mod"].plot_unit == "m"
    assert specs["HS_mod"].storage_unit == "cm"
    assert specs["HN24"].storage_unit == "cm"
    assert specs["hoar_size"].storage_unit == "mm"
    assert specs["ski_pen"].storage_unit == "m"
    assert specs["ColdContentSnow"].storage_unit == "MJ m-2"

    assert specs["RH"].storage_unit == "%"
    assert not specs["RH"].ambiguous


@pytest.mark.real_data
def test_parse_smet_unit_specs_uses_multiplier_for_dimensionless_fields(gridded_pros_dir):
    """Dimensionless SMET fields infer percent storage from multiplier metadata."""
    specs = parse_smet_unit_specs(gridded_pros_dir / "VIR1A.smet")

    assert specs["RH"].plot_unit == "1"
    assert specs["RH"].units_multiplier == 0.01
    assert specs["RH"].storage_unit == "%"
    assert not specs["RH"].ambiguous


@pytest.mark.real_data
def test_parse_smet_unit_specs_from_forcing_file(gridded_pros_dir):
    """SMET forcing files without multiplier metadata still parse units."""
    output_specs = parse_smet_unit_specs(gridded_pros_dir / "VIR1A.smet")
    forcing_specs = parse_smet_unit_specs(
        gridded_pros_dir.parents[1] / "smets" / "gridded" / "nowcast" / "VIR1A.smet"
    )

    assert output_specs["TA"].storage_unit == "degC"
    assert forcing_specs["TA"].storage_unit == "K"
    assert forcing_specs["VW"].storage_unit == "m s-1"
    assert forcing_specs["DW"].storage_unit == "degree"
    assert forcing_specs["RH"].storage_unit == "1"
    assert not forcing_specs["RH"].ambiguous


@pytest.mark.real_data
def test_parse_smet_unit_specs_from_commented_units_declaration(gridded_pros_dir):
    """Station SMET files can declare storage units in a commented #units line."""
    station_file = gridded_pros_dir.parents[1] / "smets" / "stations" / "ISEE1.smet"
    specs = parse_smet_unit_specs(station_file)

    assert specs["timestamp"].storage_unit == "ISO"
    assert specs["RH"].storage_unit == "1"
    assert specs["TA"].storage_unit == "K"
    assert specs["VW"].storage_unit == "m s-1"
    assert specs["VW_MAX"].storage_unit == "m s-1"
    assert specs["DW"].storage_unit == "degree"
    assert not any(spec.ambiguous for spec in specs.values())


@pytest.mark.real_data
def test_parse_smet_unit_specs_from_commented_units_declaration_with_snow_vars(gridded_pros_dir):
    """Commented #units declarations also handle station snow variables."""
    station_file = gridded_pros_dir.parents[1] / "smets" / "stations" / "ISEE2.smet"
    specs = parse_smet_unit_specs(station_file)

    assert specs["ISWR"].storage_unit == "W m-2"
    assert specs["HS"].storage_unit == "m"
    assert specs["RH"].storage_unit == "1"
    assert specs["TA"].storage_unit == "K"
    assert specs["TSG"].storage_unit == "K"
    assert specs["TSS"].storage_unit == "K"
    assert not any(spec.ambiguous for spec in specs.values())


@pytest.mark.synthetic
def test_parse_smet_unit_specs_prefers_units_declaration_over_plot_metadata(tmp_path):
    """Explicit SMET units declarations override plot metadata inference."""
    smet_file = tmp_path / "declared_units.smet"
    smet_file.write_text(
        """SMET 1.1 ASCII
[HEADER]
fields = timestamp HS_mod
units = ISO cm
plot_unit = - kg/m2
[DATA]
2024-01-01T00:00:00 25
"""
    )

    specs = parse_smet_unit_specs(smet_file)

    assert specs["HS_mod"].storage_unit == "cm"
    assert not specs["HS_mod"].ambiguous


def test_registry_allows_observed_smet_storage_units():
    """Registry whitelists SMET variables with observed input/output unit variants."""
    assert registry["TA"]["smet_units"] == ["K", "degC"]
    assert registry["TSG"]["smet_units"] == ["K", "degC"]
    assert registry["RH"]["smet_units"] == ["1", "%"]
    assert registry["TSS"]["smet_units"] == "K"
    assert registry["VW_MAX"]["smet_units"] == "m s-1"
    assert registry["DW_MAX"]["smet_units"] == "degree"
    assert registry["zSn"]["smet_units"] == "m"


@pytest.mark.synthetic
def test_read_smet_converts_target_snow_fields_to_xsnow_units(tmp_path):
    """SMET reads convert snow-height fields to xsnow units and attrs."""
    smet_file = tmp_path / "snow_units.smet"
    smet_file.write_text(
        """SMET 1.1 ASCII
[HEADER]
station_name = UNIT_TEST
latitude = 46.5
longitude = 10.5
altitude = 2500
nodata = -999
fields = timestamp HS_mod HN24 ski_pen zSn
plot_unit = - m m m m
units_multiplier = 1 0.01 0.01 1 1
units_offset = 0 0 0 0 0
[DATA]
2024-01-01T00:00:00 25 5 0.4 12
"""
    )

    ds = read_smet(smet_file)

    np.testing.assert_allclose(ds["HS_mod"].values, [[[[0.25]]]])
    np.testing.assert_allclose(ds["HN24"].values, [[[[0.05]]]])
    np.testing.assert_allclose(ds["ski_pen"].values, [[[[0.4]]]])
    np.testing.assert_allclose(ds["zSn"].values, [[[[12.0]]]])

    assert ds["HS_mod"].attrs["units"] == "m"
    assert ds["HN24"].attrs["units"] == "m"
    assert ds["ski_pen"].attrs["units"] == "m"
    assert ds["zSn"].attrs["units"] == "m"


@pytest.mark.synthetic
def test_read_smet_raises_for_registered_field_with_unexpected_units(tmp_path):
    """SMET reads fail when a registered field declares incompatible units."""
    smet_file = tmp_path / "bad_units.smet"
    smet_file.write_text(
        """SMET 1.1 ASCII
[HEADER]
station_name = UNIT_TEST
fields = timestamp HS_mod
plot_unit = - kg/m2
[DATA]
2024-01-01T00:00:00 25
"""
    )

    with pytest.raises(ValueError, match="HS_mod.*expects"):
        read_smet(smet_file)


@pytest.mark.synthetic
def test_read_raises_for_smet_file_with_unexpected_units(tmp_path):
    """Top-level read() does not hide SMET unit validation failures."""
    smet_file = tmp_path / "bad_units.smet"
    smet_file.write_text(
        """SMET 1.1 ASCII
[HEADER]
station_name = UNIT_TEST
fields = timestamp HS_mod
plot_unit = - kg/m2
[DATA]
2024-01-01T00:00:00 25
"""
    )

    with pytest.raises(ValueError, match="HS_mod.*expects"):
        read(smet_file)


@pytest.mark.synthetic
def test_read_smet_warns_on_missing_usable_unit_metadata(tmp_path, caplog):
    """Strict SMET validation warns when a registered variable has unknown units."""
    logger = logging.getLogger("tests.io.test_smet_units.strict")
    smet_file = tmp_path / "missing_units.smet"
    smet_file.write_text(
        """SMET 1.1 ASCII
[HEADER]
station_name = UNIT_TEST
fields = timestamp TA
[DATA]
2024-01-01T00:00:00 273.15
"""
    )

    with caplog.at_level(logging.WARNING, logger=logger.name):
        ds = read_smet(smet_file, logger=logger)

    assert ds["TA"].attrs["units"] == "unknown"
    assert "no usable unit metadata" in caplog.text


@pytest.mark.synthetic
def test_read_smet_light_validation_warns_on_missing_units(tmp_path, caplog):
    """Light SMET validation still warns when registered variable units are unknown."""
    logger = logging.getLogger("tests.io.test_smet_units.light")
    smet_file = tmp_path / "missing_units.smet"
    smet_file.write_text(
        """SMET 1.1 ASCII
[HEADER]
station_name = UNIT_TEST
fields = timestamp TA
[DATA]
2024-01-01T00:00:00 273.15
"""
    )

    with caplog.at_level(logging.WARNING, logger=logger.name):
        ds = read_smet(smet_file, logger=logger, unit_validation="light")

    assert ds["TA"].attrs["units"] == "unknown"
    assert "no usable unit metadata" in caplog.text


@pytest.mark.synthetic
def test_read_passes_unit_validation_to_smet_stream(tmp_path):
    """Top-level read() validates unit_validation for SMET-only sources."""
    smet_file = tmp_path / "units.smet"
    smet_file.write_text(
        """SMET 1.1 ASCII
[HEADER]
station_name = UNIT_TEST
fields = timestamp TA
[DATA]
2024-01-01T00:00:00 273.15
"""
    )

    with pytest.raises(ValueError, match="unit_validation"):
        read(smet_file, unit_validation="off")


@pytest.mark.synthetic
def test_parse_smet_unit_specs_without_plot_units_marks_units_ambiguous(tmp_path):
    """SMET fields without unit metadata are explicitly unknown."""
    smet_file = tmp_path / "missing_units.smet"
    smet_file.write_text(
        """SMET 1.1 ASCII
[HEADER]
fields = timestamp TA RH
[DATA]
2024-01-01T00:00:00 273.15 0.5
"""
    )

    specs = parse_smet_unit_specs(smet_file)

    assert specs["TA"].storage_unit is None
    assert specs["TA"].ambiguous
    assert specs["RH"].storage_unit is None
    assert specs["RH"].ambiguous


@pytest.mark.parametrize(
    "plot_unit,multiplier,offset,expected,ambiguous",
    [
        ("m", 0.01, 0.0, "cm", False),
        ("m", 0.001, 0.0, "mm", False),
        ("K", 1.0, 273.15, "degC", False),
        ("K", 1.0, 0.0, "K", False),
        ("kg/m2/h", 1.0, 0.0, "kg m-2 h-1", False),
        ("-", 0.01, 0.0, "%", False),
        ("-", 1.0, 0.0, "1", False),
        ("-", 2.0, 0.0, None, True),
    ],
)
def test_infer_smet_storage_unit(plot_unit, multiplier, offset, expected, ambiguous):
    """Single-field SMET unit inference handles common SNOWPACK cases."""
    storage_unit, is_ambiguous = infer_smet_storage_unit(
        field="field",
        plot_unit=plot_unit,
        units_multiplier=multiplier,
        units_offset=offset,
    )

    assert storage_unit == expected
    assert is_ambiguous is ambiguous
