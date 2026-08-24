import logging

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xsnow import read
from xsnow.io.merge import _merge_and_finalize


def _write_pro(
    path,
    *,
    station_name="CONFLICT_STATION",
    latitude=46.5,
    longitude=10.5,
    altitude=2500,
    slope_angle=35,
    slope_azi=180,
    density=200,
):
    station_lines = [f"StationName= {station_name}"]
    if latitude is not None:
        station_lines.append(f"Latitude= {latitude}")
    if longitude is not None:
        station_lines.append(f"Longitude= {longitude}")
    if altitude is not None:
        station_lines.append(f"Altitude= {altitude}")
    if slope_angle is not None:
        station_lines.append(f"SlopeAngle= {slope_angle}")
    if slope_azi is not None:
        station_lines.append(f"SlopeAzi= {slope_azi}")

    path.write_text(
        "\n".join(
            [
                "[STATION_PARAMETERS]",
                *station_lines,
                "",
                "[DATA]",
                "0500,01.01.2025 12:00:00",
                "0501,2,0.1,0.2",
                f"0502,2,{density},{density + 50}",
                "",
            ]
        )
    )


def _write_smet(
    path,
    *,
    station_name="CONFLICT_STATION",
    latitude=46.5,
    longitude=10.5,
    altitude=2500,
    slope_angle=None,
    slope_azi=None,
    ta=270.0,
):
    header_lines = [
        "SMET 1.1 ASCII",
        "",
        "[HEADER]",
        f"station_name = {station_name}",
    ]
    if latitude is not None:
        header_lines.append(f"latitude = {latitude}")
    if longitude is not None:
        header_lines.append(f"longitude = {longitude}")
    if altitude is not None:
        header_lines.append(f"altitude = {altitude}")
    if slope_angle is not None:
        header_lines.append(f"slope_angle = {slope_angle}")
    if slope_azi is not None:
        header_lines.append(f"slope_azi = {slope_azi}")
    header_lines.extend(
        [
            "nodata = -999",
            "fields = timestamp TA",
            "",
            "[DATA]",
            f"2025-01-01T12:00:00 {ta}",
            "",
        ]
    )
    path.write_text("\n".join(header_lines))


def test_read_raises_on_conflicting_pro_smet_location_coordinates(tmp_path):
    _write_pro(tmp_path / "station.pro", latitude=46.5)
    _write_smet(tmp_path / "station.smet", latitude=47.0)

    with pytest.raises(ValueError, match="conflict|Conflicting"):
        read(tmp_path, lazy=False, n_cpus_use=1, max_layers=2)


def test_read_allows_missing_coordinate_when_other_file_has_value(tmp_path):
    _write_pro(tmp_path / "station.pro", latitude=46.5, longitude=10.5, altitude=2500)
    _write_smet(tmp_path / "station.smet", latitude=None, longitude=None, altitude=None)

    ds = read(tmp_path, lazy=False, n_cpus_use=1, max_layers=2)

    assert ds is not None
    assert float(ds.latitude.sel(location="CONFLICT_STATION")) == pytest.approx(46.5)
    assert float(ds.longitude.sel(location="CONFLICT_STATION")) == pytest.approx(10.5)
    assert float(ds.altitude.sel(location="CONFLICT_STATION")) == pytest.approx(2500)
    assert "density" in ds.data_vars
    assert "TA" in ds.data_vars


def test_read_raises_on_conflicting_overlapping_smet_values(tmp_path):
    _write_smet(tmp_path / "left.smet", ta=270.0)
    _write_smet(tmp_path / "right.smet", ta=271.0)

    with pytest.raises(ValueError, match="conflict|Conflicting"):
        read(tmp_path, lazy=False, n_cpus_use=1)


def test_read_keeps_smet_files_with_distinct_slope_metadata_separate(tmp_path):
    _write_smet(tmp_path / "flat.smet", slope_angle=0, slope_azi=0, ta=270.0)
    _write_smet(tmp_path / "steep.smet", slope_angle=35, slope_azi=180, ta=271.0)

    ds = read(tmp_path, lazy=False, n_cpus_use=1)

    assert ds is not None
    assert ds.sizes["slope"] == 2
    assert set(ds.inclination.values.tolist()) == {0.0, 35.0}
    assert set(ds.azimuth.values.tolist()) == {0.0, 180.0}
    assert float(ds.TA.sel(slope=0).values.squeeze()) == pytest.approx(270.0)
    assert float(ds.TA.sel(slope=1).values.squeeze()) == pytest.approx(271.0)


def test_read_broadcasts_smet_without_slope_metadata_to_all_profile_slopes(tmp_path):
    _write_pro(tmp_path / "flat.pro", slope_angle=0, slope_azi=0)
    _write_pro(tmp_path / "steep.pro", slope_angle=35, slope_azi=180)
    _write_smet(tmp_path / "meteo.smet", slope_angle=None, slope_azi=None, ta=270.0)

    ds = read(tmp_path, lazy=False, n_cpus_use=1, max_layers=2)

    assert ds is not None
    assert ds.sizes["slope"] == 2
    assert set(ds.inclination.values.tolist()) == {0, 35}
    assert set(ds.azimuth.values.tolist()) == {0, 180}
    np.testing.assert_allclose(
        ds.TA.sel(location="CONFLICT_STATION").values.squeeze(),
        [270.0, 270.0],
    )


def test_read_raises_on_conflicting_location_coordinates_across_slopes(tmp_path):
    _write_pro(tmp_path / "slope_0.pro", latitude=46.5, slope_angle=0)
    _write_pro(tmp_path / "slope_35.pro", latitude=47.0, slope_angle=35)

    with pytest.raises(ValueError, match="conflict|Conflicting"):
        read(tmp_path, lazy=False, n_cpus_use=1, max_layers=2)


def test_merge_and_finalize_raises_on_dataset_attribute_conflicts():
    time = [pd.Timestamp("2025-01-01T12:00:00")]
    left = xr.Dataset(
        {"TA": (("location", "time", "slope", "realization"), np.array([[[[270.0]]]]))},
        coords={
            "location": ["LEFT"],
            "time": time,
            "slope": [0],
            "realization": [0],
            "latitude": ("location", [46.5]),
            "longitude": ("location", [10.5]),
            "altitude": ("location", [2500.0]),
            "inclination": ("slope", [np.nan]),
            "azimuth": ("slope", [np.nan]),
        },
        attrs={"crs": "EPSG:4326"},
    )
    right = left.assign_coords(location=["RIGHT"]).copy()
    right.attrs = {"crs": "EPSG:3857"}

    with pytest.raises(ValueError, match="crs|attrs|Conflicting"):
        _merge_and_finalize(
            [left, right],
            time=None,
            location=None,
            slope=None,
            realization=None,
            logger=logging.getLogger(__name__),
        )


@pytest.mark.synthetic
class TestStationCoordPrecision:
    """PRO and SMET headers can disagree in the 6th coordinate decimal."""

    PRO = (
        "[STATION_PARAMETERS]\n"
        "StationName= SNAP\n"
        "Latitude= 47.07318724\n"
        "Longitude= 12.45937850\n"
        "Altitude= 2000\n"
        "SlopeAngle= 0.00\n"
        "SlopeAzi= 0.00\n"
        "\n"
        "[HEADER]\n"
        "0500,Date\n"
        "0501,nElems,height [> 0: top, < 0: bottom of elem.] (cm)\n"
        "0502,nElems,element density (kg m-3)\n"
        "\n"
        "[DATA]\n"
        "0500,01.12.2024 12:00:00\n"
        "0501,1,10.0\n"
        "0502,1,150.0\n"
    )
    SMET = (
        "SMET 1.1 ASCII\n"
        "[HEADER]\n"
        "station_id = SNAP\n"
        "station_name = SNAP\n"
        "latitude = 47.073187\n"
        "longitude = 12.459379\n"
        "altitude = 2000.0\n"
        "nodata = -999\n"
        "tz = 0\n"
        "slope_angle = 0\n"
        "slope_azi = 0\n"
        "fields = timestamp HS_mod\n"
        "plot_unit = - m\n"
        "units_multiplier = 1 1\n"
        "[DATA]\n"
        "2024-12-01T12:00:00 0.1\n"
    )

    def test_near_equal_coords_snap_to_profile(self, tmp_path):
        (tmp_path / 'SNAP.pro').write_text(self.PRO)
        (tmp_path / 'SNAP.smet').write_text(self.SMET)
        ds = read([tmp_path / 'SNAP.pro', tmp_path / 'SNAP.smet'],
                  unit_validation='light')
        assert float(ds.data['longitude'].values[0]) == pytest.approx(
            12.459378, abs=1e-9)
        assert 'HS_mod' in ds.data
