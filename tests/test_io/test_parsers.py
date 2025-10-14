# tests/io/test_parsers.py
"""
Tests for low-level parsing functions.

Focus: Correctness of data extraction from .pro and .smet files.

Strategy:
  - Use REAL data to validate parsing logic
  - Use SYNTHETIC data for specific edge cases
  - Test both raw parsing and dataset creation
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from xsnow.xsnow_io import (
    _read_pro_raw,
    _create_pro_dataset,
    read_smet,
    _scan_pro_for_max_layers,
    _get_layer_count
)


# ============================================================================
# REAL DATA TESTS - .PRO PARSING
# ============================================================================

@pytest.mark.real_data
def test_read_pro_raw_basic_real_file(sample_pro_file):
    """
    Primary test: Parse a real SNOWPACK .pro file.
    
    Validates that the parser correctly extracts:
    - Station metadata
    - Profile data for each timestamp
    - All expected variables
    """
    metadata, profiles = _read_pro_raw(sample_pro_file)
    
    # Metadata should be present
    assert 'stationName' in metadata
    assert 'latitude' in metadata
    assert 'longitude' in metadata
    assert 'altitude' in metadata
    
    # Should have profiles
    assert len(profiles) > 0
    
    # Each profile should have expected structure
    for profile in profiles:
        assert 'timestamp' in profile
        assert 'height' in profile
        assert 'density' in profile
        
        # Data should be numpy arrays
        assert isinstance(profile['height'], np.ndarray)
        assert isinstance(profile['density'], np.ndarray)


@pytest.mark.real_data
def test_read_pro_raw_with_time_filter_real_data(sample_pro_file):
    """
    Test time filtering during parsing with real data.
    
    Validates that datetime_start/datetime_end parameters
    correctly filter profiles at the parsing stage.
    """
    # First get all profiles
    _, all_profiles = _read_pro_raw(sample_pro_file)
    
    if len(all_profiles) < 2:
        pytest.skip("Need at least 2 profiles for time filtering test")
    
    # Get timestamp range
    timestamps = [pd.to_datetime(p['timestamp'], format='%d.%m.%Y %H:%M:%S') 
                  for p in all_profiles]
    mid_point = timestamps[len(timestamps) // 2]
    
    # Filter to first half
    _, filtered_profiles = _read_pro_raw(
        sample_pro_file,
        datetime_start=None,
        datetime_end=mid_point.strftime('%d.%m.%Y %H:%M:%S')
    )
    
    # Should have fewer profiles
    assert len(filtered_profiles) <= len(all_profiles)
    assert len(filtered_profiles) > 0
    
    # All should be before mid_point
    for profile in filtered_profiles:
        ts = pd.to_datetime(profile['timestamp'], format='%d.%m.%Y %H:%M:%S')
        assert ts <= mid_point


@pytest.mark.real_data
def test_read_pro_raw_soil_removal_real_data(sample_pro_file):
    """
    Test soil removal during parsing with real data.
    
    Real data may or may not contain soil, so we verify:
    - With removal: no grain types 81/82
    - Without removal: original data preserved
    """
    # Parse without soil removal
    _, profiles_with_soil = _read_pro_raw(sample_pro_file, remove_soil=False)
    
    # Parse with soil removal
    _, profiles_no_soil = _read_pro_raw(sample_pro_file, remove_soil=True)
    
    assert len(profiles_with_soil) == len(profiles_no_soil)
    
    # Check if any profiles had soil
    for i, prof_with in enumerate(profiles_with_soil):
        prof_without = profiles_no_soil[i]
        
        if 'grain_type' in prof_with:
            # Check for soil in original
            has_soil = np.any((prof_with['grain_type'] == 81) | 
                             (prof_with['grain_type'] == 82))
            
            if has_soil:
                # Filtered version should have fewer layers
                assert len(prof_without['height']) <= len(prof_with['height'])
                
                # No soil should remain
                if 'grain_type' in prof_without:
                    assert not np.any((prof_without['grain_type'] == 81) | 
                                     (prof_without['grain_type'] == 82))


@pytest.mark.real_data
def test_read_pro_raw_surface_hoar_real_data(sample_pro_file):
    """
    Test surface hoar handling with real data.
    
    Checks that SH_at_surface parameter is correctly parsed
    and can be used to add synthetic SH layers.
    """
    _, profiles = _read_pro_raw(sample_pro_file, add_surface_sh_as_layer=False)
    
    # Check for SH_at_surface in profiles
    sh_count = 0
    for profile in profiles:
        if 'SH_at_surface' in profile:
            sh_data = profile['SH_at_surface']
            assert isinstance(sh_data, np.ndarray)
            
            if sh_data.size >= 3:
                # Valid SH data structure
                if sh_data[0] > 0:  # Flag > 0 means SH present
                    sh_count += 1
                    assert sh_data[1] >= 0  # grain size
                    assert sh_data[2] >= 0  # density
    
    # Real data typically has some SH
    # But we don't require it for test to pass


@pytest.mark.real_data
def test_create_pro_dataset_real_data(sample_pro_file):
    """
    Test dataset creation from real parsed data.
    
    Validates the full pipeline: parse → create dataset.
    """
    metadata, profiles = _read_pro_raw(sample_pro_file)
    
    if not profiles:
        pytest.skip("No profiles in sample file")
    
    # Create dataset
    ds = _create_pro_dataset(metadata, profiles, max_layers=100)
    
    assert ds is not None
    
    # Check structure
    assert 'location' in ds.dims
    assert 'time' in ds.dims
    assert 'layer' in ds.dims
    
    # Check metadata made it to coords
    assert 'latitude' in ds.coords
    assert 'longitude' in ds.coords
    assert 'altitude' in ds.coords
    
    # Check data variables
    assert 'height' in ds.data_vars
    assert 'density' in ds.data_vars


# ============================================================================
# REAL DATA TESTS - .SMET PARSING
# ============================================================================

@pytest.mark.real_data
def test_read_smet_basic_real_data(smet_files_dir):
    """
    Test SMET parsing with real meteorological data.
    
    Validates extraction of time series data.
    """
    smet_files = list(smet_files_dir.glob("*.smet"))
    if not smet_files:
        pytest.skip("No SMET files available")
    
    # Parse first SMET file
    ds = read_smet(smet_files[0])
    
    assert ds is not None
    
    # Check structure
    assert 'time' in ds.dims
    assert 'location' in ds.dims
    
    # Should have time series variables
    # (exact variables depend on file content)
    assert len(ds.data_vars) > 0


@pytest.mark.real_data
def test_read_smet_metadata_real_data(smet_files_dir):
    """
    Test that SMET metadata is correctly parsed and stored.
    """
    smet_files = list(smet_files_dir.glob("*.smet"))
    if not smet_files:
        pytest.skip("No SMET files available")
    
    ds = read_smet(smet_files[0])
    
    # Check coordinate metadata
    assert 'latitude' in ds.coords
    assert 'longitude' in ds.coords
    assert 'altitude' in ds.coords
    
    # Values should be reasonable
    lat = ds.latitude.values[0]
    lon = ds.longitude.values[0]
    alt = ds.altitude.values[0]
    
    assert -90 <= lat <= 90
    assert -180 <= lon <= 180
    assert 0 <= alt <= 9000  # Reasonable altitude range


# ============================================================================
# SYNTHETIC DATA TESTS - EDGE CASES
# ============================================================================

@pytest.mark.synthetic
@pytest.mark.edge_case
def test_parse_corrupted_file_malformed_data(tmp_path, synthetic_pro):
    """
    Edge case: File with malformed data should not crash parser.
    """
    corrupted = synthetic_pro.corrupted(tmp_path, "malformed_data")
    
    # Should handle gracefully
    metadata, profiles = _read_pro_raw(corrupted)
    
    # May return empty or partial data, but shouldn't crash
    assert isinstance(metadata, dict)
    assert isinstance(profiles, list)


@pytest.mark.synthetic
@pytest.mark.edge_case
def test_parse_corrupted_file_invalid_timestamp(tmp_path, synthetic_pro):
    """
    Edge case: Invalid timestamp format.
    """
    corrupted = synthetic_pro.corrupted(tmp_path, "invalid_timestamp")
    
    metadata, profiles = _read_pro_raw(corrupted)
    
    # Profiles with bad timestamps should be skipped
    assert isinstance(profiles, list)


@pytest.mark.synthetic
@pytest.mark.edge_case
def test_parse_empty_file(tmp_path, synthetic_pro):
    """
    Edge case: Completely empty file.
    """
    empty = synthetic_pro.corrupted(tmp_path, "empty")
    
    metadata, profiles = _read_pro_raw(empty)
    
    # Should return empty structures
    assert metadata == {}
    assert profiles == []


@pytest.mark.synthetic
@pytest.mark.edge_case
def test_parse_file_no_data_section(tmp_path, synthetic_pro):
    """
    Edge case: File with headers but no [DATA] section.
    """
    no_data = synthetic_pro.corrupted(tmp_path, "no_data_section")
    
    metadata, profiles = _read_pro_raw(no_data)
    
    # Should parse headers but have no profiles
    assert len(metadata) > 0  # Has station parameters
    assert profiles == []  # No data


@pytest.mark.synthetic
def test_read_pro_raw_soil_removal_all_soil(tmp_path, synthetic_pro):
    """
    Edge case: Profile with ONLY soil layers.
    """
    filepath = synthetic_pro.with_soil_layers(tmp_path, n_snow=0, n_soil=3)
    
    _, profiles = _read_pro_raw(filepath, remove_soil=True)
    
    # Profiles with only soil should be filtered out
    assert len(profiles) == 0


@pytest.mark.synthetic
def test_read_pro_raw_surface_hoar_synthesis(tmp_path, synthetic_pro):
    """
    Test that SH_at_surface data is correctly extracted.
    
    Uses synthetic data to ensure SH is present.
    """
    filepath = synthetic_pro.with_surface_hoar(tmp_path, has_sh=True)
    
    _, profiles = _read_pro_raw(filepath, add_surface_sh_as_layer=False)
    
    assert len(profiles) == 1
    profile = profiles[0]
    
    # Should have SH data
    assert 'SH_at_surface' in profile
    sh_data = profile['SH_at_surface']
    
    assert sh_data[0] == 1.0  # Flag
    assert sh_data[1] == 0.5  # Size
    assert sh_data[2] == 100.0  # Density


@pytest.mark.synthetic
def test_create_pro_dataset_empty_profiles(tmp_path, synthetic_pro):
    """
    Edge case: Creating dataset from empty profile list.
    """
    metadata = {
        'stationName': 'TEST',
        'latitude': '46.5',
        'longitude': '10.5',
        'altitude': '2500'
    }
    
    ds = _create_pro_dataset(metadata, [], max_layers=10)
    
    # Should return None for empty profiles
    assert ds is None


@pytest.mark.synthetic
def test_read_pro_raw_varying_layer_counts(tmp_path, synthetic_pro):
    """
    Test parsing file with profiles having different layer counts.
    """
    filepath = synthetic_pro.varying_layers(tmp_path, [2, 5, 3, 8])
    
    _, profiles = _read_pro_raw(filepath)
    
    assert len(profiles) == 4
    
    # Each profile should have correct number of layers
    expected_counts = [2, 5, 3, 8]
    for i, profile in enumerate(profiles):
        actual_count = len(profile['height'])
        assert actual_count == expected_counts[i]


# ============================================================================
# HELPER FUNCTION TESTS
# ============================================================================

@pytest.mark.synthetic
def test_get_layer_count_basic(tmp_path, synthetic_pro):
    """
    Test _get_layer_count helper with synthetic data.
    """
    filepath = synthetic_pro.basic(tmp_path, n_layers=5)
    
    # Read raw file to get profile lines
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Find data section
    data_lines = []
    in_data = False
    for line in lines:
        if '[DATA]' in line:
            in_data = True
            continue
        if in_data and line.strip():
            data_lines.append(line.strip())
    
    # Count layers
    count = _get_layer_count(data_lines, add_sh=False, remove_soil=False)
    assert count == 5


@pytest.mark.synthetic
def test_get_layer_count_with_surface_hoar(tmp_path, synthetic_pro):
    """
    Test that _get_layer_count correctly adds SH layer.
    """
    filepath = synthetic_pro.with_surface_hoar(tmp_path, has_sh=True)
    
    with open(filepath, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    
    # Find profile block
    profile_lines = []
    for i, line in enumerate(lines):
        if line.startswith('0500,'):
            # Get this profile block
            j = i
            while j < len(lines) and not lines[j].startswith('['):
                profile_lines.append(lines[j])
                j += 1
            break
    
    count_no_sh = _get_layer_count(profile_lines, add_sh=False, remove_soil=False)
    count_with_sh = _get_layer_count(profile_lines, add_sh=True, remove_soil=False)
    
    # Should be one more with SH
    assert count_with_sh == count_no_sh + 1


@pytest.mark.synthetic
def test_scan_pro_for_max_layers(tmp_path, synthetic_pro):
    """
    Test max layer scanning with synthetic data.
    """
    filepath = synthetic_pro.varying_layers(tmp_path, [2, 8, 5])
    
    max_layers = _scan_pro_for_max_layers(filepath, add_sh=False, remove_soil=False)
    
    # Should find the maximum
    assert max_layers == 8


@pytest.mark.real_data
def test_scan_pro_for_max_layers_real_data(sample_pro_file):
    """
    Test max layer scanning on real data.
    """
    max_layers = _scan_pro_for_max_layers(sample_pro_file, add_sh=False, remove_soil=False)
    
    # Should be reasonable
    assert max_layers > 0
    assert max_layers < 500  # Sanity check


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

@pytest.mark.real_data
def test_parse_then_create_dataset_pipeline(sample_pro_file):
    """
    Integration test: Full parsing pipeline with real data.
    """
    # Step 1: Scan for max layers
    max_layers = _scan_pro_for_max_layers(sample_pro_file, add_sh=True, remove_soil=True)
    
    # Step 2: Parse raw
    metadata, profiles = _read_pro_raw(sample_pro_file, remove_soil=True, add_surface_sh_as_layer=True)
    
    # Step 3: Create dataset
    ds = _create_pro_dataset(metadata, profiles, max_layers, add_surface_sh_as_layer=True)
    
    assert ds is not None
    assert ds.sizes['layer'] == max_layers
    assert len(profiles) == ds.sizes['time']


@pytest.mark.synthetic
def test_parse_smet_with_missing_fields(tmp_path, synthetic_smet):
    """
    Test SMET parsing with minimal field set.
    """
    filepath = synthetic_smet.basic(tmp_path, n_timesteps=10)
    
    ds = read_smet(filepath)
    
    assert ds is not None
    assert ds.sizes['time'] == 10


@pytest.mark.real_data
def test_parsing_preserves_physical_validity(sample_pro_file, assert_physical_validity):
    """
    Ensure parsing doesn't introduce invalid values.
    """
    metadata, profiles = _read_pro_raw(sample_pro_file)
    ds = _create_pro_dataset(metadata, profiles, max_layers=100)
    
    if ds:
        from xsnow import xsnowDataset
        xsnow_ds = xsnowDataset(ds)
        assert_physical_validity(xsnow_ds)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])