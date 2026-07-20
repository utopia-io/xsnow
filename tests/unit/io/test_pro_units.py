# tests/io/test_pro_units.py
"""Tests for PRO-specific unit parsing helpers."""

import numpy as np
import pytest

from xsnow import read
from xsnow.io.pro import _file_unit_key, _resolve_pro_unit_maps, parse_pro_header_units


@pytest.mark.real_data
def test_parse_pro_header_units_from_real_file(sample_pro_file):
    """PRO header parsing extracts declared units without reading data blocks."""
    units = parse_pro_header_units(sample_pro_file)

    assert units["height"] == "cm"
    assert units["density"] == "kg m-3"
    assert units["temperature"] == "degC"
    assert units["grain_size"] == "mm"
    assert units["stress"] == "kPa"
    assert units["SH_at_surface"] == ["Swiss Code F1F2F3", "mm", "kg m-3"]


@pytest.mark.synthetic
def test_parse_pro_header_units_ignores_data_section(tmp_path):
    """PRO unit parsing stops at [DATA] and ignores data lines."""
    pro_file = tmp_path / "units.pro"
    pro_file.write_text(
        """[HEADER]
0501,nElems,height [> 0: top, < 0: bottom of elem.] (cm)
[DATA]
0501,1,2.5 (not a header unit)
"""
    )

    assert parse_pro_header_units(pro_file) == {"height": "cm"}


@pytest.mark.synthetic
def test_read_pro_converts_declared_units_to_xsnow_units(tmp_path, synthetic_pro):
    """PRO values are converted to xsnow units and attrs describe current data."""
    pro_file = synthetic_pro.with_declared_units(tmp_path / "units.pro")

    xs = read(pro_file, lazy=False, max_layers=2)
    ds = xs.data

    np.testing.assert_allclose(ds["height"].values[0, 0, 0, 0, :], [0.1, 0.2])
    np.testing.assert_allclose(ds["grain_size"].values[0, 0, 0, 0, :], [0.0005, 0.001])
    np.testing.assert_allclose(ds["lwc"].values[0, 0, 0, 0, :], [0.05, 0.1])
    np.testing.assert_allclose(ds["stress"].values[0, 0, 0, 0, :], [1200, 2400])

    assert ds["height"].attrs["units"] == "m"
    assert ds["grain_size"].attrs["units"] == "m"
    assert ds["lwc"].attrs["units"] == "1"
    assert ds["stress"].attrs["units"] == "Pa"


@pytest.mark.synthetic
def test_read_pro_strict_unit_validation_converts_each_file_units(tmp_path, synthetic_pro):
    """Strict validation allows mixed units when each file can be converted."""
    synthetic_pro.with_declared_units(
        tmp_path / "station_a.pro",
        station_name="A",
        height_unit="cm",
        height_values=(10.0, 20.0),
    )
    synthetic_pro.with_declared_units(
        tmp_path / "station_b.pro",
        station_name="B",
        height_unit="m",
        height_values=(10.0, 20.0),
    )

    xs = read(tmp_path, lazy=False, max_layers=2, n_cpus_use=1)
    ds = xs.data

    np.testing.assert_allclose(
        ds["height"].sel(location="A").values[0, 0, 0, :],
        [0.1, 0.2],
    )
    np.testing.assert_allclose(
        ds["height"].sel(location="B").values[0, 0, 0, :],
        [10.0, 20.0],
    )


@pytest.mark.synthetic
def test_read_pro_strict_unit_validation_raises_on_unconvertible_units(tmp_path, synthetic_pro):
    """Strict PRO unit validation fails when declared units cannot be converted."""
    pro_file = synthetic_pro.with_declared_units(
        tmp_path / "station_a.pro",
        station_name="A",
        height_unit="furlong",
    )

    with pytest.raises(ValueError, match="Cannot convert PRO variable 'height'"):
        read(pro_file, lazy=False, max_layers=2, n_cpus_use=1)


@pytest.mark.synthetic
def test_read_pro_light_unit_validation_checks_one_file_per_directory(tmp_path, synthetic_pro):
    """Light validation uses the first PRO file from each parent directory."""
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()

    first_a = synthetic_pro.with_declared_units(dir_a / "first.pro", height_unit="cm")
    second_a = synthetic_pro.with_declared_units(dir_a / "second.pro", height_unit="furlong")
    first_b = synthetic_pro.with_declared_units(dir_b / "first.pro", height_unit="m")

    unit_maps = _resolve_pro_unit_maps([first_a, second_a, first_b], "light", logger=None)

    assert unit_maps[_file_unit_key(first_a)]["height"] == "cm"
    assert unit_maps[_file_unit_key(second_a)]["height"] == "cm"
    assert unit_maps[_file_unit_key(first_b)]["height"] == "m"


@pytest.mark.synthetic
def test_read_pro_rejects_unknown_unit_validation_mode(tmp_path, synthetic_pro):
    """PRO reads accept only explicit unit validation modes."""
    pro_file = synthetic_pro.with_declared_units(tmp_path / "units.pro")

    with pytest.raises(ValueError, match="unit_validation"):
        read(pro_file, lazy=False, max_layers=2, unit_validation="off")
