# tests/io/test_writers.py
"""
Tests for output functions.

Focus: Data serialization and format conversion.

Strategy:
  - Use REAL data to ensure realistic export scenarios
  - Use SYNTHETIC data for edge cases (empty, extreme values)
  - Test roundtrip workflows (read → write → read)
"""

import pytest
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

import netCDF4

import xsnow.io.netcdf as netcdf_io
from tests.conftest import ALL_FILE_EXTS, COMPRESSION_MODULES
from xsnow import read, single_profile, xsnowDataset
from xsnow.io.netcdf import _bounded_chunk_cache


# ============================================================================
# NETCDF OUTPUT TESTS
# ============================================================================

@pytest.mark.real_data
def test_to_netcdf_basic(real_file_subset, tmp_path):
    """
    Primary test: Save real data to NetCDF.
    
    Most common use case for caching and sharing.
    """
    ds = read(real_file_subset)
    
    output_file = tmp_path / "output.nc"
    ds.to_netcdf(output_file)
    
    # File should exist
    assert output_file.exists()
    assert output_file.stat().st_size > 0

@pytest.mark.real_data
def test_to_netcdf_preserves_metadata(real_file_subset, tmp_path):
    """
    Test that NetCDF output preserves important metadata and
    the 'mapping' attribute on profile_status.
    """
    ds_original = read(real_file_subset)
    
    output_file = tmp_path / "with_metadata.nc"
    ds_original.to_netcdf(output_file)
    
    # Read back
    ds_loaded = read(output_file)
    
    # --- Global metadata ---
    assert 'Conventions' in ds_loaded.data.attrs
    assert 'crs' in ds_loaded.data.attrs
    
    # --- Coordinates ---
    assert 'latitude' in ds_loaded.coords
    assert 'longitude' in ds_loaded.coords
    assert 'altitude' in ds_loaded.coords

    # --- profile_status.mapping roundtrip ---
    # Ensure the original dataset had profile_status + mapping
    assert "profile_status" in ds_original
    orig_ps = ds_original["profile_status"]
    assert "mapping" in orig_ps.attrs
    orig_mapping = orig_ps.attrs["mapping"]
    assert isinstance(orig_mapping, dict)

    # After NetCDF roundtrip, mapping should be restored identically
    assert "profile_status" in ds_loaded
    loaded_ps = ds_loaded["profile_status"]
    assert "mapping" in loaded_ps.attrs

    loaded_mapping = loaded_ps.attrs["mapping"]
    assert isinstance(loaded_mapping, dict)
    assert loaded_mapping == orig_mapping



@pytest.mark.real_data
def test_to_netcdf_roundtrip(real_file_subset, tmp_path, compare_datasets):
    """
    Test that data survives NetCDF roundtrip unchanged.
    
    Critical for caching workflows.
    """
    ds_original = read(real_file_subset)
    
    # Save
    nc_file = tmp_path / "roundtrip.nc"
    ds_original.to_netcdf(nc_file)
    
    # Load
    ds_loaded = read(nc_file)
    
    # Should be identical (within floating point tolerance)
    compare_datasets(ds_original, ds_loaded, rtol=1e-6)


@pytest.mark.real_data
def test_to_netcdf_roundtrip_preserves_slope_metadata_as_coords(real_file_subset, tmp_path):
    """
    Slope metadata remains coordinates in the stored file and after roundtrip.
    """
    ds_original = read(real_file_subset)
    assert "inclination" in ds_original.data.coords
    assert "azimuth" in ds_original.data.coords

    nc_file = tmp_path / "slope_coords.nc"
    ds_original.to_netcdf(nc_file)

    with xr.open_dataset(nc_file) as raw:
        raw_loaded = raw.load()

    for coord in ("inclination", "azimuth"):
        assert coord in raw_loaded.coords
        assert coord not in raw_loaded.data_vars

    ds_loaded = read(nc_file)
    for coord in ("inclination", "azimuth"):
        assert coord in ds_loaded.data.coords
        assert coord not in ds_loaded.data.data_vars
        np.testing.assert_allclose(
            ds_loaded.data[coord].values,
            ds_original.data[coord].values,
            equal_nan=True,
        )


def test_read_netcdf_promotes_legacy_slope_metadata_data_vars(tmp_path):
    """
    Older files with demoted slope metadata are restored to coordinates.
    """
    legacy = xr.Dataset(
        data_vars={
            "HS": (
                ("location", "time", "slope", "realization"),
                np.array([[[[1.2], [1.3]]]]),
            ),
            "inclination": (("location", "slope"), np.array([[0.0, 35.0]])),
            "azimuth": (("location", "slope"), np.array([[0.0, 180.0]])),
        },
        coords={
            "location": ["A"],
            "time": [pd.Timestamp("2025-01-01")],
            "slope": [0, 1],
            "realization": [0],
        },
    )
    nc_file = tmp_path / "legacy_demoted_slope_coords.nc"
    legacy.to_netcdf(nc_file)

    loaded = read(nc_file)

    for coord in ("inclination", "azimuth"):
        assert coord in loaded.data.coords
        assert coord not in loaded.data.data_vars
        assert loaded.data[coord].dims == ("slope",)

    subset = loaded.sel(slope=(loaded.azimuth == 180.0))
    assert subset.sizes["slope"] == 1
    assert float(subset.inclination.values[0]) == 35.0


def _normalize_location_string(loc):
    """Helper to normalize location values that may be bytes or strings."""
    if isinstance(loc, bytes):
        return loc.decode('utf-8')
    return str(loc)


@pytest.mark.real_data
def test_to_netcdf_location_mapping(real_file_subset, tmp_path):
    """
    Test that location coordinate values are correctly saved and restored.
    
    Updated: Now tests string-based coordinate preservation instead of
    the legacy integer-index + mapping approach.
    """
    ds_original = read(real_file_subset)
    
    # Get original location values (strings)
    original_locations = [_normalize_location_string(loc) for loc in ds_original.location.values]
    assert len(original_locations) > 0
    
    # Save and reload
    nc_file = tmp_path / "with_locations.nc"
    ds_original.to_netcdf(nc_file)
    ds_loaded = read(nc_file)
    
    # Location values should be preserved
    loaded_locations = [_normalize_location_string(loc) for loc in ds_loaded.location.values]
    
    # Should have same locations
    assert len(loaded_locations) == len(original_locations)
    assert loaded_locations == original_locations

@pytest.mark.synthetic
@pytest.mark.edge_case
def test_to_netcdf_empty_dataset_raises_error(tmp_path):
    """
    Edge case: Cannot save empty dataset.
    """
    from xsnow import xsnowDataset
    
    # Create empty dataset wrapper
    empty_ds = xsnowDataset.__new__(xsnowDataset)
    empty_ds.data = None
    
    with pytest.raises(ValueError, match="Cannot save an empty dataset"):
        empty_ds.to_netcdf(tmp_path / "empty.nc")


# ============================================================================
# .PRO OUTPUT TESTS  
# ============================================================================

@pytest.mark.real_data
def test_to_pro_basic(sample_pro_file, tmp_path):
    """
    Test writing .pro file from real data.
    
    Useful for converting NetCDF back to SNOWPACK format.
    """
    # Read a single station
    ds = read(sample_pro_file)
    
    for ext in ALL_FILE_EXTS:
        output_file = tmp_path / ("output.pro" + ext)
        ds.to_pro(output_file)
    
        # File should exist and have content
        assert output_file.exists()
        assert output_file.stat().st_size > 0
        
        # Should have proper structure
        content = read_compressed(output_file)
        assert '[STATION_PARAMETERS]' in content
        assert '[DATA]' in content


@pytest.mark.real_data
def test_to_pro_roundtrip_structure(sample_pro_file, tmp_path):
    """
    Test that .pro file structure is preserved in roundtrip.
    
    Note: Exact values may differ due to rounding, but structure should match.
    """
    ds_original = read(sample_pro_file)
    
    for ext in ALL_FILE_EXTS:
        # Write to .pro
        output_file = tmp_path / ("roundtrip.pro" + ext)
        ds_original.to_pro(output_file)
        
        # Read back
        ds_loaded = read(output_file)
        
        # Check dimensions match
        assert ds_loaded.sizes['location'] == ds_original.sizes['location']
        assert ds_loaded.sizes['time'] == ds_original.sizes['time']
        
        # Check variables present
        for var in ['height', 'density']:
            if var in ds_original.data:
                assert var in ds_loaded.data


@pytest.mark.synthetic
def test_to_pro_converts_xsnow_units_to_pro_units(tmp_path, synthetic_pro):
    """PRO writer converts xsnow-unit data back to SNOWPACK PRO units."""
    source_file = synthetic_pro.with_declared_units(tmp_path / "source.pro")
    ds = read(source_file, lazy=False, max_layers=2)

    output_file = tmp_path / "written.pro"
    ds.to_pro(output_file)

    content = output_file.read_text()
    assert "0501,2,10,20" in content
    assert "0506,2,5,10" in content
    assert "0512,2,0.5,1" in content
    assert "0513,3,550,440,0" in content
    assert "0517,2,1.2,2.4" in content


@pytest.mark.synthetic
def test_to_pro_writes_parameter_header_definitions(tmp_path, synthetic_pro):
    """PRO writer emits parameter definitions required by external readers."""
    source_file = synthetic_pro.with_declared_units(tmp_path / "source.pro")
    ds = read(source_file, lazy=False, max_layers=2)

    output_file = tmp_path / "with_header.pro"
    ds.to_pro(output_file)

    content = output_file.read_text()
    assert "[STATION_PARAMETERS]" in content
    assert "[HEADER]" in content
    assert "[DATA]" in content
    assert content.index("[HEADER]") < content.index("[DATA]")
    assert "0500,Date" in content
    assert "0501,nElems,height [> 0: top, < 0: bottom of elem.] (cm)" in content
    assert "0502,nElems,density (kg m-3)" in content
    assert "0512,nElems,grain size (mm)" in content
    assert "0513,nElems,grain type (Swiss Code F1F2F3)" in content
    assert "0517,nElems,stress (kPa)" in content

    loaded = read(output_file, lazy=False, max_layers=2)
    assert loaded is not None
    np.testing.assert_allclose(loaded["height"].values[0, 0, 0, 0, :], [0.1, 0.2])
    np.testing.assert_allclose(loaded["grain_type"].values[0, 0, 0, 0, :], [550, 440])
    np.testing.assert_allclose(loaded["stress"].values[0, 0, 0, 0, :], [1200, 2400])


@pytest.mark.synthetic
@pytest.mark.edge_case
def test_to_pro_max_files_limit(tmp_path, synthetic_pro):
    """
    Test that max_files parameter enforces limit.
    
    Prevents accidentally writing thousands of files.
    """
    # Create dataset with many timesteps
    synthetic_pro.multi_station(tmp_path, n_stations=1, n_profiles_each=10)
    ds = read(tmp_path)
    
    # Try to write with max_files=5
    output_file = tmp_path / "output.pro"
    with pytest.raises(ValueError, match="exceeds the specified maximum"):
        ds.to_pro(output_file, max_files=5)


@pytest.mark.real_data
def test_to_pro_single_location_requirement(real_file_subset, tmp_path):
    """
    Test that to_pro requires single location.
    
    .pro files are per-station, not multi-station.
    """
    ds = read(real_file_subset)
    
    if ds.sizes['location'] > 1:
        output_file = tmp_path / "multi_location.pro"
        with pytest.raises(ValueError, match="only supports datasets with a single location"):
            ds.to_pro(output_file)


# ============================================================================
# .SMET OUTPUT TESTS
# ============================================================================

@pytest.mark.real_data
def test_to_smet_basic(real_file_subset, tmp_path):
    """
    Test writing SMET file from real data.
    
    Exports time series data (no layer dimension).
    """
    ds = read(real_file_subset)
    
    # Select one time series that can be represented by a single SMET file.
    for dim in ("location", "slope", "realization"):
        if ds.sizes.get(dim, 1) > 1:
            ds = ds.isel({dim: 0})
    
    for ext in ALL_FILE_EXTS:
        output_file = tmp_path / ("output.smet" + ext)
        ds.to_smet(output_file)

        # File should exist
        assert output_file.exists()

        # Should have SMET structure
        content = read_compressed(output_file)
        assert 'SMET' in content
        assert '[HEADER]' in content
        assert '[DATA]' in content


@pytest.fixture
def smet_writer_dataset():
    """Single-location dataset with missing values and string data for SMET export."""
    times = pd.date_range("2025-01-01", periods=2, freq="D")
    ds = xr.Dataset(
        data_vars={
            "TA": (("location", "time", "slope"), [[[270.123456], [np.nan]]]),
            "weather": (("location", "time", "slope"), [[["clear"], ["storm"]]]),
            "density": (("location", "time", "slope", "layer"), [[[[100.0, 200.0]], [[np.nan, np.nan]]]]),
        },
        coords={
            "location": ["TEST"],
            "time": times,
            "slope": [0],
            "layer": [0, 1],
            "latitude": ("location", [46.5]),
            "longitude": ("location", [10.5]),
            "altitude": ("location", [2500.0]),
            "azimuth": ("slope", [180.0]),
            "inclination": ("slope", [30.0]),
        },
    )
    return xsnowDataset(ds)


@pytest.mark.synthetic
def test_to_smet_writes_and_roundtrips_slope_metadata(smet_writer_dataset, tmp_path):
    """Finite scalar slope coordinates are written to and read from the header."""
    from xsnow.io.smet import read_smet

    output_file = tmp_path / "slope_metadata.smet"
    smet_writer_dataset.to_smet(output_file)

    content = output_file.read_text()
    assert "SlopeAngle = 30.0" in content
    assert "SlopeAzi = 180.0" in content

    loaded = read_smet(output_file)
    np.testing.assert_allclose(loaded.inclination.values, [30.0])
    np.testing.assert_allclose(loaded.azimuth.values, [180.0])


@pytest.mark.synthetic
def test_to_smet_writes_slope_angle_without_azimuth(smet_writer_dataset, tmp_path):
    """SlopeAngle is useful on its own and does not require SlopeAzi."""
    dataset = xsnowDataset(smet_writer_dataset.data.drop_vars("azimuth"))
    output_file = tmp_path / "slope_angle_only.smet"

    dataset.to_smet(output_file)

    content = output_file.read_text()
    assert "SlopeAngle = 30.0" in content
    assert "SlopeAzi" not in content


@pytest.mark.synthetic
def test_to_smet_can_omit_slope_metadata(smet_writer_dataset, tmp_path):
    """Slope headers can be omitted for forcing that is valid for all slopes."""
    output_file = tmp_path / "no_slope_metadata.smet"

    smet_writer_dataset.to_smet(output_file, write_slope_metadata=False)

    content = output_file.read_text()
    assert "SlopeAngle" not in content
    assert "SlopeAzi" not in content


@pytest.mark.synthetic
def test_to_smet_rejects_azimuth_without_slope_angle(smet_writer_dataset, tmp_path):
    """SlopeAzi cannot be represented meaningfully without SlopeAngle."""
    dataset = xsnowDataset(smet_writer_dataset.data.drop_vars("inclination"))
    output_file = tmp_path / "azimuth_only.smet"

    with pytest.raises(ValueError, match="SlopeAzi.*SlopeAngle"):
        dataset.to_smet(output_file)

    assert not output_file.exists()


@pytest.mark.synthetic
def test_to_smet_omits_nonfinite_slope_metadata(smet_writer_dataset, tmp_path):
    """NaN and infinity are missing metadata, not valid SMET header values."""
    dataset = xsnowDataset(
        smet_writer_dataset.data.assign_coords(
            inclination=("slope", [np.inf]),
            azimuth=("slope", [np.nan]),
        )
    )
    output_file = tmp_path / "nonfinite_slope.smet"

    dataset.to_smet(output_file)

    content = output_file.read_text()
    assert "SlopeAngle" not in content
    assert "SlopeAzi" not in content


@pytest.mark.synthetic
@pytest.mark.parametrize("dimension", ["location", "slope", "realization"])
def test_to_smet_rejects_multiple_output_members(
    smet_writer_dataset,
    tmp_path,
    dimension,
):
    """One SMET file cannot identify multiple locations, slopes, or realizations."""
    base = smet_writer_dataset.data
    if dimension == "location":
        other = base.assign_coords(location=["SECOND"])
        dataset = xr.concat([base, other], dim="location")
    elif dimension == "slope":
        other = base.assign_coords(
            slope=[1],
            inclination=("slope", [35.0]),
            azimuth=("slope", [200.0]),
        )
        dataset = xr.concat([base, other], dim="slope")
    else:
        first = base.expand_dims(realization=[0])
        other = first.assign_coords(realization=[1])
        dataset = xr.concat([first, other], dim="realization")
    output_file = tmp_path / f"multiple_{dimension}.smet"

    with pytest.raises(ValueError, match=rf"ds\.sel\({dimension}="):
        xsnowDataset(dataset).to_smet(output_file)

    assert not output_file.exists()


@pytest.mark.synthetic
def test_to_smet_rejects_nonscalar_slope_coordinate(smet_writer_dataset, tmp_path):
    """The metadata extractor must not silently take the first coordinate value."""
    dataset = xsnowDataset(
        smet_writer_dataset.data.drop_vars(["inclination", "azimuth"]).assign_coords(
            SlopeAngle=("time", [20.0, 30.0]),
        )
    )
    output_file = tmp_path / "nonscalar_slope.smet"

    with pytest.raises(ValueError, match="SlopeAngle.*exactly one"):
        dataset.to_smet(output_file)

    assert not output_file.exists()


@pytest.mark.synthetic
def test_to_smet_output_units_convert_from_xsnow_units(smet_units_xsnow_dataset, tmp_path):
    """Output-style SMET writing converts xsnow units to output storage units."""
    from xsnow.io.smet import read_smet

    output_file = tmp_path / "output_units.smet"
    smet_units_xsnow_dataset.to_smet(output_file)

    content = output_file.read_text()
    assert "station_id = TEST" in content
    assert "station_name = TEST" in content
    assert "units = ISO8601 degC % cm" in content
    assert "0.000\t50.000\t25.000" in content

    loaded = read_smet(output_file)
    np.testing.assert_allclose(loaded["TA"].isel(time=0).values, [[[0.0]]])
    np.testing.assert_allclose(loaded["RH"].isel(time=0).values, [[[0.5]]])
    np.testing.assert_allclose(loaded["HS_mod"].isel(time=0).values, [[[0.25]]])


@pytest.mark.synthetic
def test_to_smet_input_units_convert_from_xsnow_units(smet_units_xsnow_dataset, tmp_path):
    """Input-style SMET writing uses forcing units where registry declares them."""
    from xsnow.io.smet import read_smet

    output_file = tmp_path / "input_units.smet"
    xsnowDataset(smet_units_xsnow_dataset.data[["TA", "RH"]]).to_smet(
        output_file,
        unit_type="input",
    )

    content = output_file.read_text()
    assert "units = ISO8601 K 1" in content
    assert "273.150\t0.500" in content

    loaded = read_smet(output_file)
    np.testing.assert_allclose(loaded["TA"].isel(time=0).values, [[[0.0]]])
    np.testing.assert_allclose(loaded["RH"].isel(time=0).values, [[[0.5]]])


@pytest.mark.synthetic
def test_to_smet_forcing_unit_type_aliases_input(smet_units_xsnow_dataset, tmp_path):
    """Forcing is accepted as an alias for input-style SMET units."""
    output_file = tmp_path / "forcing_units.smet"
    xsnowDataset(smet_units_xsnow_dataset.data[["TA", "RH"]]).to_smet(
        output_file,
        unit_type="forcing",
    )

    content = output_file.read_text()
    assert "units = ISO8601 K 1" in content
    assert "273.150\t0.500" in content


@pytest.mark.synthetic
def test_to_smet_rejects_unknown_unit_type(smet_units_xsnow_dataset, tmp_path):
    """SMET writer validates the selected unit convention."""
    with pytest.raises(ValueError, match="unit_type"):
        smet_units_xsnow_dataset.to_smet(tmp_path / "bad.smet", unit_type="archive")


@pytest.mark.synthetic
def test_to_smet_writes_nodata_precision_and_unquoted_strings(smet_writer_dataset, tmp_path):
    """SMET writer should emit rectangular rows even when values are missing."""
    output_file = tmp_path / "nodata.smet"
    smet_writer_dataset.to_smet(output_file)

    content = output_file.read_text()
    assert "nodata = -999.0" in content
    assert "fields = timestamp TA weather" in content
    assert '"' not in content
    assert "270.123" in content
    assert "-999.0" in content

    data_lines = content.split("[DATA]\n", maxsplit=1)[1].splitlines()
    assert data_lines
    column_counts = {len(line.split()) for line in data_lines}
    assert column_counts == {3}


@pytest.mark.synthetic
def test_to_smet_roundtrip_preserves_missing_values(smet_writer_dataset, tmp_path):
    """A SMET written with nodata tokens should parse back to NaN values."""
    from xsnow.io.smet import read_smet

    output_file = tmp_path / "roundtrip_missing.smet"
    smet_writer_dataset.to_smet(output_file)

    loaded = read_smet(output_file)
    assert loaded is not None
    assert "TA" in loaded
    assert np.isclose(loaded["TA"].isel(time=0).values, 270.123).all()
    assert np.isnan(loaded["TA"].isel(time=1).values).all()
    assert "layer" not in loaded["TA"].dims


@pytest.mark.synthetic
@pytest.mark.edge_case
def test_to_smet_max_files_limit(tmp_path, synthetic_pro):
    """
    Test that max_files parameter works for SMET.
    """
    # Create multi-station dataset
    synthetic_pro.multi_station(tmp_path, n_stations=5)
    ds = read(tmp_path)
    
    # Try to write with max_files=2
    output_file = tmp_path / "output.smet"
    with pytest.raises(ValueError, match="exceeds the specified maximum"):
        ds.to_smet(output_file, max_files=2)


@pytest.mark.real_data
def test_to_smet_single_location_requirement(real_file_subset, tmp_path):
    """
    Test that to_smet requires single location.
    """
    ds = read(real_file_subset)
    
    if ds.sizes['location'] > 1:
        output_file = tmp_path / "multi_location.smet"
        with pytest.raises(ValueError, match="only supports datasets with a single location"):
            ds.to_smet(output_file)


@pytest.mark.real_data
def test_to_smet_only_exports_time_series(real_file_subset, tmp_path):
    """
    Test that SMET output only includes time series variables (no layers).
    """
    ds = read(real_file_subset)
    
    # Select one time series that can be represented by a single SMET file.
    for dim in ("location", "slope", "realization"):
        if ds.sizes.get(dim, 1) > 1:
            ds = ds.isel({dim: 0})
    
    output_file = tmp_path / "timeseries.smet"
    ds.to_smet(output_file)
    
    # Read back and check
    from xsnow.io.smet import read_smet
    ds_smet = read_smet(output_file)
    
    # Should have no layer dimension
    assert 'layer' not in ds_smet.dims


# ============================================================================
# FORMAT CONVERSION TESTS
# ============================================================================

@pytest.mark.real_data
def test_convert_pro_to_netcdf(sample_pro_file, tmp_path):
    """
    Integration: .pro → NetCDF conversion workflow.
    """
    # Read .pro
    ds = read(sample_pro_file)
    
    # Write to NetCDF
    nc_file = tmp_path / "converted.nc"
    ds.to_netcdf(nc_file)
    
    # Verify by reading back
    ds_nc = read(nc_file)
    assert ds_nc is not None
    assert ds_nc.sizes['location'] == ds.sizes['location']


@pytest.mark.real_data  
def test_convert_netcdf_to_pro(real_file_subset, tmp_path):
    """
    Integration: NetCDF → .pro conversion workflow.
    """
    # Read original
    ds_original = read(real_file_subset)
    
    # Convert to NetCDF
    nc_file = tmp_path / "intermediate.nc"
    ds_original.to_netcdf(nc_file)
    
    # Read NetCDF
    ds_from_nc = read(nc_file)
    
    # Convert back to .pro (single location)
    if ds_from_nc.sizes['location'] > 1:
        first_location = str(ds_from_nc.data.location.values[0])
        ds_from_nc = ds_from_nc.sel(location=first_location)

    
    pro_file = tmp_path / "from_nc.pro"
    ds_from_nc.to_pro(pro_file)
    
    assert pro_file.exists()


# ============================================================================
# DATA INTEGRITY TESTS
# ============================================================================

@pytest.mark.real_data
def test_netcdf_preserves_data_types(real_file_subset, tmp_path):
    """
    Test that NetCDF preserves data types correctly.
    """
    ds_original = read(real_file_subset)
    
    # Save and reload
    nc_file = tmp_path / "types.nc"
    ds_original.to_netcdf(nc_file)
    ds_loaded = read(nc_file)
    
    # Check data types
    for var in ds_original.data.data_vars:
        if var in ds_loaded.data:
            original_dtype = ds_original.data[var].dtype
            loaded_dtype = ds_loaded.data[var].dtype
            
            # Should be same or compatible (e.g., float32 vs float64)
            assert np.issubdtype(loaded_dtype, np.floating) == np.issubdtype(original_dtype, np.floating)


@pytest.mark.real_data
def test_netcdf_preserves_nan_values(real_file_subset, tmp_path):
    """
    Test that NaN values are correctly preserved in NetCDF.
    
    Important for padded layer data.
    """
    ds_original = read(real_file_subset)
    
    # Save and reload
    nc_file = tmp_path / "with_nans.nc"
    ds_original.to_netcdf(nc_file)
    ds_loaded = read(nc_file)
    
    # Check NaN preservation
    for var in ['height', 'density']:
        if var in ds_original.data and var in ds_loaded.data:
            orig_nans = np.isnan(ds_original.data[var].values)
            load_nans = np.isnan(ds_loaded.data[var].values)
            
            # NaN patterns should match
            np.testing.assert_array_equal(orig_nans, load_nans)


@pytest.mark.real_data
def test_netcdf_preserves_time_encoding(real_file_subset, tmp_path):
    """
    Test that time coordinate is correctly encoded/decoded.
    """
    ds_original = read(real_file_subset)
    
    # Save and reload
    nc_file = tmp_path / "times.nc"
    ds_original.to_netcdf(nc_file)
    ds_loaded = read(nc_file)
    
    # Times should match exactly
    np.testing.assert_array_equal(
        ds_original.time.values,
        ds_loaded.time.values
    )


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

@pytest.mark.real_data
@pytest.mark.slow
def test_netcdf_write_performance(gridded_pros_dir, tmp_path, timer):
    """
    Benchmark NetCDF writing performance.
    """
    # Read large dataset
    ds = read(gridded_pros_dir)
    
    # Time the write
    timer.start('netcdf_write')
    output_file = tmp_path / "large.nc"
    ds.to_netcdf(output_file)
    timer.stop('netcdf_write')
    
    timer.report()
    
    # Should complete in reasonable time
    elapsed = timer.get('netcdf_write')
    if elapsed:
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"\nWrote {file_size_mb:.1f} MB in {elapsed:.2f}s")


@pytest.mark.real_data
@pytest.mark.slow
def test_netcdf_vs_pro_write_speed(real_file_subset, tmp_path, timer):
    """
    Compare NetCDF vs .pro write speeds.
    """
    ds = read(real_file_subset)
    
    # NetCDF write
    timer.start('netcdf')
    nc_file = tmp_path / "test.nc"
    ds.to_netcdf(nc_file)
    timer.stop('netcdf')
    
    # .pro write (single location)
    if ds.sizes['location'] > 1:
        first_location = str(ds.data.location.values[0])
        ds_single = ds.sel(location=first_location)
    else:
        ds_single = ds
    
    timer.start('pro')
    pro_file = tmp_path / "test.pro"
    ds_single.to_pro(pro_file)
    timer.stop('pro')
    
    timer.report()
    
    # NetCDF is typically faster for large datasets


# ============================================================================
# REGRESSION TESTS
# ============================================================================

@pytest.mark.real_data
def test_netcdf_compression_enabled(real_file_subset, tmp_path):
    """
    Regression: NetCDF should use compression by default.
    """
    ds = read(real_file_subset)
    
    # Write with default settings
    output_file = tmp_path / "compressed.nc"
    ds.to_netcdf(output_file)
    
    # File should be reasonably compressed
    # (Hard to test without uncompressed comparison, but file should exist)
    assert output_file.exists()


@pytest.mark.real_data
def test_writer_methods_chainable(real_file_subset, tmp_path):
    """
    Regression: Writer methods should not break method chaining.
    """
    ds = read(real_file_subset)
    
    # These should not raise errors
    nc_file = tmp_path / "test.nc"
    ds.to_netcdf(nc_file)
    
    # Can still access data after writing
    assert ds.sizes['location'] > 0


# ============================================================================
# NOT YET IMPLEMENTED TESTS
# ============================================================================

@pytest.mark.skip(reason="to_json not yet implemented")
def test_to_json_placeholder():
    """Placeholder for future to_json implementation."""
    pass


@pytest.mark.skip(reason="to_caaml not yet implemented")
def test_to_caaml_placeholder():
    """Placeholder for future to_caaml implementation."""
    pass


@pytest.mark.skip(reason="to_crocus not yet implemented")
def test_to_crocus_placeholder():
    """Placeholder for future to_crocus implementation."""
    pass

@pytest.mark.synthetic
class TestNetCDFStringEncoding:
    """Test NetCDF encoding of string coordinates."""
    
    def test_save_and_load_string_coords(self, synthetic_pro, tmp_path):
        """Test round-trip save/load preserves string coords."""
        from xsnow import read
        
        filepath = synthetic_pro.basic(tmp_path, station_name='TEST_STATION', n_layers=3)
        ds = read(filepath)
        
        # Save to NetCDF
        nc_path = tmp_path / "test.nc"
        ds.to_netcdf(nc_path)
        
        # Load back
        ds_loaded = read(nc_path)
        
        # Verify location dtype is string-like
        assert ds_loaded.data.location.dtype.kind in ('U', 'S', 'O')
        
        # Get location value and normalize (may be bytes from NetCDF)
        loaded_str = _normalize_location_string(ds_loaded.data.location.values[0])
        
        assert loaded_str == 'TEST_STATION'
    
    def test_long_station_names(self, tmp_path):
        """Test that long station names are preserved."""
        from xsnow import read
        
        long_name = "Very_Long_Station_Name_With_Many_Characters"
        
        content = f"""[STATION_PARAMETERS]
StationName= {long_name}

[DATA]
0500,01.01.2025 12:00:00
0501,2,0.1,0.2
0502,2,100,200
"""
        filepath = tmp_path / "test.pro"
        filepath.write_text(content)
        
        ds = read(filepath)
        nc_path = tmp_path / "test.nc"
        ds.to_netcdf(nc_path)
        
        ds_loaded = read(nc_path)
        
        # Get locations and normalize
        original = _normalize_location_string(ds.data.location.values[0])
        loaded = _normalize_location_string(ds_loaded.data.location.values[0])
        
        # Should be preserved (up to 64 chars)
        assert original[:64] == loaded[:64]

# ============================================================================
# NETCDF CHUNK CACHE
# ============================================================================

@pytest.mark.synthetic
def test_bounded_chunk_cache_sets_and_restores():
    outside = netCDF4.get_chunk_cache()
    current = (outside[0], outside[1] + 1, 0.5)
    try:
        netCDF4.set_chunk_cache(*current)
        with _bounded_chunk_cache(4 * 1024 * 1024):
            assert netCDF4.get_chunk_cache() == (
                4 * 1024 * 1024,
                current[1],
                current[2],
            )
        assert netCDF4.get_chunk_cache() == current
    finally:
        netCDF4.set_chunk_cache(*outside)


@pytest.mark.synthetic
def test_bounded_chunk_cache_none_is_noop():
    outside = netCDF4.get_chunk_cache()
    with _bounded_chunk_cache(None):
        assert netCDF4.get_chunk_cache() == outside


@pytest.mark.synthetic
def test_chunk_cache_size_roundtrip_kwarg(tmp_path, compare_datasets):
    """to_netcdf and read accept chunk_cache_size and still roundtrip."""
    ds = single_profile()
    nc_path = tmp_path / "chunk_cache.nc"
    ds.to_netcdf(nc_path, chunk_cache_size=2 * 1024 * 1024)

    ds_loaded = read(nc_path, chunk_cache_size=2 * 1024 * 1024)
    compare_datasets(ds, ds_loaded)

    ds_unclamped = read(nc_path, chunk_cache_size=None)
    compare_datasets(ds, ds_unclamped)


@pytest.mark.synthetic
def test_chunk_cache_size_lazy_read(tmp_path, monkeypatch):
    """A lazy read applies the cache at open and remains usable afterward."""
    ds = single_profile()
    nc_path = tmp_path / "lazy_chunk_cache.nc"
    ds.to_netcdf(nc_path)
    expected_height = ds["height"].values.copy()

    outside = netCDF4.get_chunk_cache()
    cache_size = 2 * 1024 * 1024
    observed_cache = []
    open_dataset = netcdf_io.xr.open_dataset

    def recording_open_dataset(*args, **kwargs):
        observed_cache.append(netCDF4.get_chunk_cache())
        return open_dataset(*args, **kwargs)

    monkeypatch.setattr(netcdf_io.xr, "open_dataset", recording_open_dataset)

    loaded = read(nc_path, lazy=True, chunk_cache_size=cache_size)
    try:
        assert loaded is not None
        assert observed_cache == [(cache_size, outside[1], outside[2])]
        assert netCDF4.get_chunk_cache() == outside
        assert loaded["height"].chunks is not None
        np.testing.assert_allclose(loaded["height"].values, expected_height)
    finally:
        if loaded is not None:
            loaded.close()


# ============================================================================
# UTILITY
# ============================================================================

def read_compressed(filepath: Path):
    if filepath.suffix in COMPRESSION_MODULES.keys():
        module = COMPRESSION_MODULES[filepath.suffix]
        with module.open(filepath, "rt") as f:
            content = f.read()
    else:
        content = filepath.read_text()
    return content

if __name__ == '__main__':
    pytest.main([__file__, '-v'])


def test_smet_column_formatting_matches_scalar_path():
    """The whole-column formatter must reproduce _format_smet_value exactly."""
    from xsnow.io.smet import _format_smet_column, _format_smet_value

    values = np.array([np.nan, 0.0, -0.0, 1e-4, -1e-4, 5e-4, 1e-3, 0.999, 1.0, 2.5, 3.5,
        99999.999, 1e5, -1e5, 1e6, 270.123456])
    nodata = -999.0
    for name in ("TA", "profile_status"):
        column = _format_smet_column(name, values, nodata)
        expected = [_format_smet_value(name, value, nodata) for value in values]
        assert list(column) == expected, name
    infinite = np.array([np.inf, -np.inf, 1.0])
    assert list(_format_smet_column("TA", infinite, nodata)) == \
        [_format_smet_value("TA", value, nodata) for value in infinite]
    with pytest.raises(OverflowError):
        _format_smet_value("profile_status", np.inf, nodata)
    with pytest.raises(OverflowError):
        _format_smet_column("profile_status", infinite, nodata)
    flags = np.array([True, False])
    assert list(_format_smet_column("flag", flags, nodata)) == \
        [_format_smet_value("flag", value, nodata) for value in flags]
    assert _format_smet_column("weather", np.array(["clear", "storm"], dtype=object), nodata) is None


def test_to_smet_ignores_scalar_coordinates(smet_writer_dataset, tmp_path):
    """Scalar coordinates (region labels, geometry text) must not change the output."""
    plain = tmp_path / "plain.smet"
    smet_writer_dataset.to_smet(plain)
    decorated = xsnowDataset(smet_writer_dataset.data.assign_coords(
        region_id="AT-07-01", location_geometry="{" + "x" * 100000 + "}"))
    labelled = tmp_path / "labelled.smet"
    decorated.to_smet(labelled)
    assert labelled.read_text() == plain.read_text()


def test_to_smet_reports_nat_timestamps_as_value_error(smet_writer_dataset, tmp_path):
    times = smet_writer_dataset.data.time.values.astype("datetime64[ns]").copy()
    times[1] = np.datetime64("NaT")
    broken = xsnowDataset(smet_writer_dataset.data.assign_coords(time=times))
    with pytest.raises(ValueError, match="NaT"):
        broken.to_smet(tmp_path / "nat.smet")
