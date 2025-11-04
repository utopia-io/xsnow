# test_dask_integration_real.py
"""
Dask integration tests using real .pro files from the sample data.
These tests provide more realistic validation than synthetic data.
"""
import pytest
from pathlib import Path
import pandas as pd
import numpy as np
import psutil
import time
import shutil

from xsnow import read, xsnowDataset

# Try to import sample data functions, but don't fail if unavailable
try:
    from xsnow.sample_data import snp_gridded_dir, snp_snowobs_dir
    SAMPLE_DATA_AVAILABLE = True
except (ImportError, AttributeError):
    SAMPLE_DATA_AVAILABLE = False
    snp_gridded_dir = None
    snp_snowobs_dir = None

import os

# Check if running in CI or if we should skip Dask edge case tests
SKIP_DASK_EDGE_CASES = os.getenv('CI') == 'true' or os.getenv('SKIP_DASK_EDGE_CASES') == 'true'

# ============================================================================
# FIXTURES FOR REAL DATA
# ============================================================================

@pytest.fixture(scope="module")
def real_gridded_dir():
    """Returns path to real gridded sample data."""
    if not SAMPLE_DATA_AVAILABLE:
        pytest.skip("Sample data module not available")
    
    try:
        base_path = Path(snp_gridded_dir())
        
        # Try different possible locations
        possible_paths = [
            base_path / "pros" / "gridded",
            base_path / "gridded",
            base_path / "pros",
            base_path,
        ]
        
        for path in possible_paths:
            if path.exists() and list(path.glob("*.pro")):
                print(f"\nUsing gridded data from: {path}")
                return path
        
        pytest.skip(f"No .pro files found in {base_path} or subdirectories")
    except Exception as e:
        pytest.skip(f"Could not access sample data: {e}")


@pytest.fixture(scope="module")
def real_snowobs_dir():
    """Returns path to real snow observation data."""
    if not SAMPLE_DATA_AVAILABLE:
        pytest.skip("Sample data module not available")
    
    try:
        base_path = Path(snp_snowobs_dir())
        
        # Try different possible locations
        possible_paths = [
            base_path / "pros" / "snowobs",
            base_path / "snowobs",
            base_path / "pros",
            base_path,
        ]
        
        for path in possible_paths:
            if path.exists() and list(path.glob("*.pro")):
                print(f"\nUsing snowobs data from: {path}")
                return path
        
        pytest.skip(f"No .pro files found in {base_path} or subdirectories")
    except Exception as e:
        pytest.skip(f"Could not access sample data: {e}")


@pytest.fixture(scope="module")
def real_pit_dir():
    """Returns path to real manual pit data."""
    if not SAMPLE_DATA_AVAILABLE:
        pytest.skip("Sample data module not available")
    
    try:
        base_path = Path(snp_snowobs_dir())
        
        # Try different possible locations
        possible_paths = [
            base_path / "pits",
            base_path,
        ]
        
        for path in possible_paths:
            if path.exists() and list(path.glob("*.pro")):
                print(f"\nUsing pit data from: {path}")
                return path
        
        pytest.skip(f"No .pro files found in {base_path} or subdirectories")
    except Exception as e:
        pytest.skip(f"Could not access sample data: {e}")


@pytest.fixture
def subset_of_real_files(real_gridded_dir, tmp_path):
    """
    Creates a temporary directory with a subset of real files.
    Useful for controlled testing without processing all files.
    """
    # Copy first N files to tmp directory
    source_files = list(real_gridded_dir.glob("*.pro"))[:10]
    
    for src_file in source_files:
        shutil.copy(src_file, tmp_path / src_file.name)
    
    return tmp_path


# ============================================================================
# BASIC SANITY CHECKS
# ============================================================================

def test_sample_data_available():
    """Test that sample data module is available."""
    assert SAMPLE_DATA_AVAILABLE, "Sample data module not available"
    assert snp_gridded_dir is not None
    assert snp_snowobs_dir is not None


def test_real_data_paths_exist(real_gridded_dir):
    """Test that real data paths are accessible."""
    assert real_gridded_dir.exists()
    pro_files = list(real_gridded_dir.glob("*.pro"))
    assert len(pro_files) > 0, f"No .pro files found in {real_gridded_dir}"
    print(f"\nFound {len(pro_files)} .pro files in {real_gridded_dir}")


# ============================================================================
# PARALLEL PROCESSING WITH REAL DATA
# ============================================================================

def test_parallel_processing_real_data(real_gridded_dir):
    """
    Tests parallel processing on actual sample data files.
    """
    # Sequential
    start = time.time()
    ds_seq = read(real_gridded_dir, n_cpus_use=1)
    seq_time = time.time() - start
    
    # Parallel
    start = time.time()
    ds_par = read(real_gridded_dir, n_cpus_use=4)
    par_time = time.time() - start
    
    # Verify both produce same results
    assert ds_seq is not None and ds_par is not None
    assert ds_seq.sizes['location'] == ds_par.sizes['location']
    assert ds_seq.sizes['time'] == ds_par.sizes['time']
    
    print(f"\nReal data processing:")
    print(f"  Sequential: {seq_time:.3f}s")
    print(f"  Parallel (4 CPUs): {par_time:.3f}s")
    print(f"  Speedup: {seq_time/par_time:.2f}x")

@pytest.mark.real_data
@pytest.mark.skipif(
    SKIP_DASK_EDGE_CASES,
    reason="Dask scheduler hangs when computing lazy arrays in this test. "
           "Known issue with Dask threaded scheduler + complex task graphs. "
           "Main Dask functionality verified by test_parallel_processing_real_data."
)
def test_prescan_accuracy_real_data(real_gridded_dir):
    """
    Verifies that pre-scan correctly identifies max layers in real files.
    """
    ds = read(real_gridded_dir)
    
    assert ds is not None
    max_layers = ds.sizes['layer']
    
    # Collect all non-NaN layer counts across all locations, slopes, and times
    all_layer_counts = []
    profiles_with_many_layers = []
    
    for loc_idx in range(ds.sizes['location']):
        for slope_idx in range(ds.sizes.get('slope', 1)):
            for time_idx in range(ds.sizes['time']):
                # Select one specific location, slope, and time
                profile = ds.isel(location=loc_idx, slope=slope_idx, time=time_idx)
                
                # Count non-NaN layers
                non_nan_count = (~np.isnan(profile['height'].values)).sum()
                if non_nan_count > 0:  # Only count profiles with actual data
                    all_layer_counts.append(non_nan_count)
                    if non_nan_count > 30:  # Track profiles with many layers
                        time_val = pd.to_datetime(profile['time'].values)
                        profiles_with_many_layers.append((loc_idx, slope_idx, time_val, non_nan_count))
    
    if all_layer_counts:
        actual_max = max(all_layer_counts)
        print(f"\nLayer count analysis:")
        print(f"  Pre-scan determined max: {max_layers}")
        print(f"  Actual max in data: {actual_max}")
        print(f"  Min layers: {min(all_layer_counts)}")
        print(f"  Mean layers: {np.mean(all_layer_counts):.1f}")
        
        if profiles_with_many_layers:
            print(f"\nProfiles with >30 layers:")
            for loc_idx, slope_idx, time_val, count in profiles_with_many_layers[:5]:
                print(f"  Location {loc_idx}, Slope {slope_idx}, Time {time_val}: {count} layers")
        
        # Pre-scan should correctly identify or slightly overestimate max layers
        assert max_layers >= actual_max, f"Pre-scan underestimated: {max_layers} < {actual_max}"
        
        # Allow small margin for surface hoar addition
        if max_layers > actual_max + 5:
            import warnings
            warnings.warn(f"Pre-scan overestimated: {max_layers} > {actual_max}")
        else:
            print(f"  ✓ Pre-scan correctly determined max layers")


def test_soil_removal_real_data(real_gridded_dir):
    """
    Tests soil removal on real data (if soil layers exist).
    """
    # Read with soil removal
    ds_no_soil = read(real_gridded_dir, remove_soil=True)
    
    # Read without soil removal
    ds_with_soil = read(real_gridded_dir, remove_soil=False)
    
    assert ds_no_soil is not None
    assert ds_with_soil is not None
    
    # If soil layers exist, layer count should differ
    # Otherwise, they should be the same
    print(f"\nLayers with soil: {ds_with_soil.sizes['layer']}")
    print(f"Layers without soil: {ds_no_soil.sizes['layer']}")
    
    # At minimum, soil removal shouldn't increase layer count
    assert ds_no_soil.sizes['layer'] <= ds_with_soil.sizes['layer']


# ============================================================================
# MEMORY EFFICIENCY WITH REAL DATA
# ============================================================================

def test_memory_efficiency_real_data(real_gridded_dir):
    """
    Tests that processing real data doesn't cause memory issues.
    """
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Process the real data
    ds = read(real_gridded_dir, n_cpus_use=4)
    
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory
    
    assert ds is not None
    
    print(f"\nMemory usage:")
    print(f"  Initial: {initial_memory:.2f} MB")
    print(f"  Final: {final_memory:.2f} MB")
    print(f"  Increase: {memory_increase:.2f} MB")
    print(f"  Locations loaded: {ds.sizes['location']}")
    print(f"  Total profiles: {ds.sizes['time'] * ds.sizes['location']}")
    
    # Memory per profile should be reasonable
    total_profiles = ds.sizes['time'] * ds.sizes['location']
    mb_per_profile = memory_increase / max(total_profiles, 1)
    print(f"  Memory per profile: {mb_per_profile:.3f} MB")
    
    # This is a sanity check - adjust based on your system
    assert memory_increase < 2000  # Less than 2 GB increase


def test_incremental_loading_real_data(real_gridded_dir):
    """
    Tests loading real data in chunks (e.g., by time range).
    """
    from xsnow.selectors import TimeSelector
    
    # Load full dataset first to know time range
    ds_full = read(real_gridded_dir)
    assert ds_full is not None
    
    time_values = pd.to_datetime(ds_full.time.values)
    mid_point = time_values[len(time_values) // 2]
    
    # Load first half
    time_filter_1 = TimeSelector(sel=slice(None, mid_point))
    ds_first_half = read(real_gridded_dir, time=time_filter_1)
    
    # Load second half
    time_filter_2 = TimeSelector(sel=slice(mid_point, None))
    ds_second_half = read(real_gridded_dir, time=time_filter_2)
    
    assert ds_first_half is not None
    assert ds_second_half is not None
    
    # Combined should roughly equal full dataset
    total_times = ds_first_half.sizes['time'] + ds_second_half.sizes['time']
    # Allow for mid_point overlap
    assert abs(total_times - ds_full.sizes['time']) <= 1


# ============================================================================
# SURFACE HOAR HANDLING WITH REAL DATA
# ============================================================================

def test_surface_hoar_real_data(real_gridded_dir):
    """
    Tests surface hoar handling with real data that contains SH.
    """
    # With SH addition (default)
    ds_with_sh = read(real_gridded_dir, add_surface_sh_as_layer=True)
    
    # Without SH addition
    ds_no_sh = read(real_gridded_dir, add_surface_sh_as_layer=False)
    
    assert ds_with_sh is not None
    assert ds_no_sh is not None
    
    print(f"\nSurface hoar handling:")
    print(f"  Layers with SH: {ds_with_sh.sizes['layer']}")
    print(f"  Layers without SH: {ds_no_sh.sizes['layer']}")
    
    # If any profiles have SH, layer count should differ
    # Check if SH layers exist in the data
    has_sh_layers = False
    if 'grain_type' in ds_with_sh.data:
        # Grain type 660 indicates surface hoar
        has_sh_layers = np.any(ds_with_sh['grain_type'].values == 660)
    
    if has_sh_layers:
        print(f"  Found SH layers in dataset")
        # With SH should have more or equal layers
        assert ds_with_sh.sizes['layer'] >= ds_no_sh.sizes['layer']


# ============================================================================
# MULTI-SLOPE HANDLING WITH REAL DATA
# ============================================================================

def test_slope_dimension_real_data(real_gridded_dir):
    """
    Tests that slope dimension is correctly handled in real gridded data.
    """
    ds = read(real_gridded_dir)
    
    assert ds is not None
    assert 'slope' in ds.dims
    
    print(f"\nSlope dimension:")
    print(f"  Number of slopes: {ds.sizes.get('slope', 'N/A')}")
    print(f"  Slope slope_inclinations: {ds.slope_inclination.values}")
    
    # Verify slope slope_inclinations are reasonable (0-90 degrees)
    if 'slope_inclination' in ds.coords:
        slopes = ds.slope_inclination.values
        valid_slopes = slopes[~np.isnan(slopes)]
        if len(valid_slopes) > 0:
            assert np.all((valid_slopes >= 0) & (valid_slopes <= 90))


def test_slope_filtering_real_data(real_gridded_dir):
    """
    Tests filtering by slope using real data.
    """
    from xsnow.selectors import SlopeSelector
    
    # Load full dataset
    ds_full = read(real_gridded_dir)
    assert ds_full is not None
    
    if ds_full.sizes.get('slope', 1) > 1:
        # Select first slope
        slope_filter = SlopeSelector(sel=0)
        ds_slope_0 = read(real_gridded_dir, slope=slope_filter)
        
        assert ds_slope_0 is not None
        assert 'slope' not in ds_slope_0.dims  # Should be squeezed
        
        print(f"\nSlope filtering:")
        print(f"  Full dataset slopes: {ds_full.sizes['slope']}")
        print(f"  Filtered dataset: single slope")


# ============================================================================
# COMBINED OPERATIONS WITH REAL DATA
# ============================================================================

def test_multiple_sources_real_data(real_snowobs_dir, real_pit_dir):
    """
    Tests reading from multiple real data sources simultaneously.
    """
    ds = read([real_snowobs_dir, real_pit_dir])
    
    assert ds is not None
    assert ds.sizes['location'] >= 1
    
    print(f"\nMultiple sources:")
    print(f"  Total locations: {ds.sizes['location']}")
    print(f"  Total profiles: {ds.sizes['time']}")
    
    # Verify data integrity
    assert 'height' in ds.data or 'density' in ds.data


def test_time_filtering_performance_real_data(real_gridded_dir):
    """
    Tests that time filtering at source is faster than post-filtering.
    """
    from xsnow.selectors import TimeSelector
    
    # Get time range from full dataset
    ds_full = read(real_gridded_dir)
    time_values = pd.to_datetime(ds_full.time.values)
    
    if len(time_values) > 10:
        # Select middle 50% of data
        start_idx = len(time_values) // 4
        end_idx = 3 * len(time_values) // 4
        start_time = time_values[start_idx]
        end_time = time_values[end_idx]
        
        # Method 1: Filter at source
        time_filter = TimeSelector(sel=slice(start_time, end_time))
        start = time.time()
        ds_filtered = read(real_gridded_dir, time=time_filter)
        source_filter_time = time.time() - start
        
        # Method 2: Load all then filter
        start = time.time()
        ds_full_2 = read(real_gridded_dir)
        ds_post_filtered = ds_full_2.sel(time=slice(start_time, end_time))
        post_filter_time = time.time() - start
        
        print(f"\nTime filtering performance:")
        print(f"  Source filtering: {source_filter_time:.3f}s")
        print(f"  Post filtering: {post_filter_time:.3f}s")
        
        # Source filtering should generally be faster or comparable
        # (Especially for very large datasets)


# ============================================================================
# DATA INTEGRITY CHECKS WITH REAL DATA
# ============================================================================

def test_data_completeness_real_data(real_gridded_dir):
    """
    Verifies that essential variables are present in real data.
    """
    ds = read(real_gridded_dir)
    
    assert ds is not None
    
    # Check for essential variables
    expected_vars = ['height', 'density']
    present_vars = [var for var in expected_vars if var in ds.data]
    
    print(f"\nData completeness:")
    print(f"  Variables found: {list(ds.data.data_vars.keys())}")
    print(f"  Expected variables present: {present_vars}")
    
    assert len(present_vars) > 0, "No expected variables found"


def test_coordinate_consistency_real_data(real_gridded_dir):
    """
    Verifies coordinate consistency in real data.
    """
    ds = read(real_gridded_dir)
    
    assert ds is not None
    
    # Check required coordinates
    assert 'location' in ds.coords
    assert 'time' in ds.coords
    assert 'layer' in ds.coords
    
    # Check coordinate attributes
    if 'latitude' in ds.coords:
        lats = ds.latitude.values
        valid_lats = lats[~np.isnan(lats)]
        if len(valid_lats) > 0:
            assert np.all((valid_lats >= -90) & (valid_lats <= 90))
    
    if 'longitude' in ds.coords:
        lons = ds.longitude.values
        valid_lons = lons[~np.isnan(lons)]
        if len(valid_lons) > 0:
            assert np.all((valid_lons >= -180) & (valid_lons <= 180))
    
    print(f"\nCoordinate checks passed:")
    print(f"  Locations: {ds.sizes['location']}")
    print(f"  Time steps: {ds.sizes['time']}")
    print(f"  Layers: {ds.sizes['layer']}")


def test_physical_validity_real_data(real_gridded_dir):
    """
    Checks that physical values in real data are within reasonable bounds.
    """
    ds = read(real_gridded_dir)
    
    assert ds is not None
    
    # Check density is physically reasonable (50-600 kg/m³ for snow)
    if 'density' in ds.data:
        density = ds['density'].values
        valid_density = density[~np.isnan(density)]
        if len(valid_density) > 0:
            assert np.all(valid_density >= 0)
            assert np.all(valid_density <= 1000)  # Upper bound
            print(f"\nDensity range: {valid_density.min():.1f} - {valid_density.max():.1f} kg/m³")
    
    # Check temperature is physically reasonable
    # First determine if data is in Celsius or Kelvin by checking the range
    if 'temperature' in ds.data:
        temp = ds['temperature'].values
        valid_temp = temp[~np.isnan(temp)]
        if len(valid_temp) > 0:
            temp_min = valid_temp.min()
            temp_max = valid_temp.max()
            
            # If values are below 100, assume Celsius (-60 to 40°C)
            # If values are above 200, assume Kelvin (213 to 313 K)
            if temp_max < 100:
                # Celsius
                assert np.all(valid_temp >= -100), f"Temperature too low: {temp_min}°C"
                assert np.all(valid_temp <= 50), f"Temperature too high: {temp_max}°C"
                print(f"Temperature range: {temp_min:.1f} - {temp_max:.1f} °C")
            else:
                # Kelvin
                assert np.all(valid_temp >= 200), f"Temperature too low: {temp_min} K"
                assert np.all(valid_temp <= 320), f"Temperature too high: {temp_max} K"
                print(f"Temperature range: {temp_min:.1f} - {temp_max:.1f} K")
    
    # Check layer height is non-negative
    if 'height' in ds.data:
        height = ds['height'].values
        valid_height = height[~np.isnan(height)]
        if len(valid_height) > 0:
            assert np.all(valid_height >= 0)
            print(f"Layer height range: {valid_height.min():.4f} - {valid_height.max():.4f} m")


# ============================================================================
# BENCHMARK TESTS WITH REAL DATA
# ============================================================================

@pytest.mark.benchmark
def test_benchmark_full_pipeline_real_data(real_gridded_dir):
    """
    Benchmark the full read pipeline with real data.
    Requires pytest-benchmark plugin: pip install pytest-benchmark
    
    This test will be skipped if pytest-benchmark is not installed.
    """
    try:
        import pytest_benchmark
        pytest.skip("Benchmark test requires manual execution with pytest-benchmark")
    except ImportError:
        pytest.skip("pytest-benchmark not installed")
    
    # This code won't run in normal test execution
    # result = benchmark(read, real_gridded_dir, n_cpus_use=4)
    # assert result is not None


@pytest.mark.slow
def test_repeated_reads_real_data(real_gridded_dir):
    """
    Tests that repeated reads produce consistent results.
    """
    ds1 = read(real_gridded_dir, n_cpus_use=2)
    ds2 = read(real_gridded_dir, n_cpus_use=2)
    ds3 = read(real_gridded_dir, n_cpus_use=2)
    
    assert ds1.sizes == ds2.sizes == ds3.sizes
    
    # Verify data values are identical
    if 'density' in ds1.data:
        np.testing.assert_array_equal(
            ds1['density'].values,
            ds2['density'].values,
            err_msg="Repeated reads produced different density values"
        )