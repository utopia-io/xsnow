# tests/io/test_smet_units.py
"""Tests for SMET-specific unit inference helpers."""

import pytest

from xsnow.io.smet import infer_smet_storage_unit, parse_smet_unit_specs
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


def test_registry_allows_observed_smet_storage_units():
    """Registry whitelists SMET variables with observed input/output unit variants."""
    assert registry["TA"]["smet_units"] == ["K", "degC"]
    assert registry["TSG"]["smet_units"] == ["K", "degC"]
    assert registry["RH"]["smet_units"] == ["1", "%"]
    assert registry["TSS"]["smet_units"] == "K"
    assert registry["VW_MAX"]["smet_units"] == "m s-1"
    assert registry["DW_MAX"]["smet_units"] == "degree"


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
