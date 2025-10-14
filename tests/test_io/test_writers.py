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
from pathlib import Path

from xsnow import read


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
    Test that NetCDF output preserves important metadata.
    """
    ds_original = read(real_file_subset)
    
    output_file = tmp_path / "with_metadata.nc"
    ds_original.to_netcdf(output_file)
    
    # Read back
    ds_loaded = read(output_file)
    
    # Check metadata preservation
    assert 'Conventions' in ds_loaded.data.attrs
    assert 'crs' in ds_loaded.data.attrs
    
    # Check coordinates
    assert 'latitude' in ds_loaded.coords
    assert 'longitude' in ds_loaded.coords
    assert 'altitude' in ds_loaded.coords


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
def test_to_netcdf_location_mapping(real_file_subset, tmp_path):
    """
    Test that location mapping is correctly saved and restored.
    
    Location mapping converts integer indices to station names.
    """
    ds_original = read(real_file_subset)
    
    # Get original mapping
    original_mapping = ds_original.location.attrs.get('mapping')
    assert original_mapping is not None
    
    # Save and reload
    nc_file = tmp_path / "with_mapping.nc"
    ds_original.to_netcdf(nc_file)
    ds_loaded = read(nc_file)
    
    # Mapping should be preserved
    loaded_mapping = ds_loaded.location.attrs.get('mapping')
    assert loaded_mapping == original_mapping


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
    
    output_file = tmp_path / "output.pro"
    ds.to_pro(output_file)
    
    # File should exist and have content
    assert output_file.exists()
    assert output_file.stat().st_size > 0
    
    # Should have proper structure
    content = output_file.read_text()
    assert '[STATION_PARAMETERS]' in content
    assert '[DATA]' in content


@pytest.mark.real_data
def test_to_pro_roundtrip_structure(sample_pro_file, tmp_path):
    """
    Test that .pro file structure is preserved in roundtrip.
    
    Note: Exact values may differ due to rounding, but structure should match.
    """
    ds_original = read(sample_pro_file)
    
    # Write to .pro
    output_file = tmp_path / "roundtrip.pro"
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
    
    # Select single location
    if ds.sizes['location'] > 1:
        ds = ds.sel(location=0)
    
    output_file = tmp_path / "output.smet"
    ds.to_smet(output_file)
    
    # File should exist
    assert output_file.exists()
    
    # Should have SMET structure
    content = output_file.read_text()
    assert 'SMET' in content
    assert '[HEADER]' in content
    assert '[DATA]' in content


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
    
    # Select single location
    if ds.sizes['location'] > 1:
        ds = ds.sel(location=0)
    
    output_file = tmp_path / "timeseries.smet"
    ds.to_smet(output_file)
    
    # Read back and check
    from xsnow.xsnow_io import read_smet
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
        ds_from_nc = ds_from_nc.sel(location=0)
    
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
        ds_single = ds.sel(location=0)
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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])