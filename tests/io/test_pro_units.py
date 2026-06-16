# tests/io/test_pro_units.py
"""Tests for PRO-specific unit parsing helpers."""

import pytest

from xsnow.io.pro import parse_pro_header_units


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
