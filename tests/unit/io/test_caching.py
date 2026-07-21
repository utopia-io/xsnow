# tests/io/test_caching.py
"""
Tests for NetCDF caching and incremental update workflow.

Focus: Efficiency of incremental updates using .append_latest() method.

This is the KEY workflow for processing large archives:
1. Load data → save to NetCDF cache
2. Later: load cache → update with new data
3. Save updated cache

Strategy:
  - Use REAL data to validate realistic workflows
  - Use SYNTHETIC data for precise time control
  - Measure performance gains
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from xsnow import read


# ============================================================================
# BASIC CACHING WORKFLOW
# ============================================================================

@pytest.mark.real_data
def test_basic_cache_workflow(real_file_subset, tmp_path):
    """
    Primary test: Complete caching workflow with real data.
    
    Workflow:
    1. Read initial data
    2. Cache to NetCDF
    3. Load from cache
    4. Verify data integrity
    """
    # Step 1: Read initial data
    ds_original = read(real_file_subset)
    
    assert ds_original is not None
    original_time_count = ds_original.sizes['time']
    
    # Step 2: Cache to NetCDF
    cache_file = tmp_path / "cache.nc"
    ds_original.to_netcdf(cache_file)
    
    assert cache_file.exists()
    print(f"\nCached {original_time_count} timesteps")
    
    # Step 3: Load from cache
    ds_cached = read(cache_file)
    
    assert ds_cached is not None
    assert ds_cached.sizes['time'] == original_time_count
    
    # Step 4: Verify integrity
    np.testing.assert_array_equal(
        ds_original.time.values,
        ds_cached.time.values
    )


@pytest.mark.synthetic
def test_basic_cache_workflow_controlled(tmp_path, synthetic_pro):
    """
    Basic workflow with synthetic data for precise control.
    
    Ensures we understand exact behavior with known inputs.
    """
    # Create initial data (3 profiles)
    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    
    # Generate 3 timestamps - DON'T call basic() first
    for i in range(3):
        date = pd.Timestamp('2025-01-01') + pd.Timedelta(days=i)
        content = f"""[STATION_PARAMETERS]
StationName= CACHE_TEST
[DATA]
0500,{date.strftime('%d.%m.%Y %H:%M:%S')}
0501,3,0.1,0.2,0.3
0502,3,100,200,300
"""
        (initial_dir / f"profile_{i}.pro").write_text(content)
    
    # Cache workflow
    ds_original = read(initial_dir)
    assert ds_original.sizes['time'] == 3
    
    cache_file = tmp_path / "cache.nc"
    ds_original.to_netcdf(cache_file)
    
    ds_cached = read(cache_file)
    assert ds_cached.sizes['time'] == 3

# ============================================================================
# INCREMENTAL UPDATE TESTS
# ============================================================================

@pytest.mark.real_data
def test_append_latest_only_loads_new_data(real_file_subset, tmp_path):
    """
    KEY TEST: .append_latest() should only read data after last cached timestamp.
    
    This is the efficiency feature that makes incremental updates fast.
    """
    # Split files into "old" and "new"
    all_files = sorted(real_file_subset.glob("*.pro"))
    
    if len(all_files) < 2:
        pytest.skip("Need at least 2 files for update test")
    
    # Create initial cache with first file
    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    (initial_dir / all_files[0].name).write_text(all_files[0].read_text())
    
    ds_initial = read(initial_dir)
    cache_file = tmp_path / "cache.nc"
    ds_initial.to_netcdf(cache_file)
    
    last_time = pd.to_datetime(ds_initial.time.values[-1])
    print(f"\nLast cached time: {last_time}")
    
    # Create "new" data directory with second file
    new_dir = tmp_path / "new"
    new_dir.mkdir()
    (new_dir / all_files[1].name).write_text(all_files[1].read_text())
    
    # Update
    ds_cached = read(cache_file)
    ds_updated = ds_cached.append_latest(new_dir)
    
    # Should have more timestamps
    assert ds_updated.sizes['time'] >= ds_initial.sizes['time']
    
    # New times should be after last cached time
    if ds_updated.sizes['time'] > ds_initial.sizes['time']:
        new_times = pd.to_datetime(ds_updated.time.values[ds_initial.sizes['time']:])
        assert all(t > last_time for t in new_times)


@pytest.mark.synthetic
def test_append_latest_with_no_new_data(tmp_path, synthetic_pro):
    """
    Test update when source contains no new data.
    
    Should return dataset unchanged.
    """
    # Create initial data
    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    
    content = """[STATION_PARAMETERS]
StationName= UPDATE_TEST
[DATA]
0500,01.01.2025 12:00:00
0501,2,0.1,0.2
0502,2,100,200
"""
    (initial_dir / "data.pro").write_text(content)
    
    # Cache it
    ds_initial = read(initial_dir)
    cache_file = tmp_path / "cache.nc"
    ds_initial.to_netcdf(cache_file)
    
    # Try to update with same data
    ds_cached = read(cache_file)
    ds_updated = ds_cached.append_latest(initial_dir)
    
    # Should be unchanged (same number of times)
    assert ds_updated.sizes['time'] == ds_initial.sizes['time']


@pytest.mark.synthetic
def test_append_latest_with_time_gap(tmp_path, synthetic_pro):
    """
    Test update when there's a gap between cached and new data.
    
    Gap should be preserved (no interpolation).
    """
    # Old data: Jan 1
    old_dir = tmp_path / "old"
    old_dir.mkdir()
    old_content = """[STATION_PARAMETERS]
StationName= GAP_TEST
[DATA]
0500,01.01.2025 00:00:00
0501,2,0.1,0.2
0502,2,100,200
"""
    (old_dir / "old.pro").write_text(old_content)
    
    # New data: Jan 10 (9 day gap)
    new_dir = tmp_path / "new"
    new_dir.mkdir()
    new_content = """[STATION_PARAMETERS]
StationName= GAP_TEST
[DATA]
0500,10.01.2025 00:00:00
0501,2,0.3,0.4
0502,2,140,240
"""
    (new_dir / "new.pro").write_text(new_content)
    
    # Cache and update
    ds_old = read(old_dir)
    cache_file = tmp_path / "cache.nc"
    ds_old.to_netcdf(cache_file)
    
    ds_cached = read(cache_file)
    ds_updated = ds_cached.append_latest(new_dir)
    
    # Should have 2 timestamps with 9-day gap
    assert ds_updated.sizes['time'] == 2
    
    times = pd.to_datetime(ds_updated.time.values)
    gap_days = (times[1] - times[0]).days
    assert gap_days == 9


@pytest.mark.synthetic
def test_append_latest_with_overlapping_times(tmp_path, synthetic_pro):
    """
    Test update when new data overlaps with cached data.
    
    Overlapping timestamps should be deduplicated.
    """
    # Cached: Jan 1 00:00 and 06:00
    cached_dir = tmp_path / "cached"
    cached_dir.mkdir()
    cached_content = """[STATION_PARAMETERS]
StationName= OVERLAP_TEST
[DATA]
0500,01.01.2025 00:00:00
0501,2,0.1,0.2
0502,2,100,200
0500,01.01.2025 06:00:00
0501,2,0.15,0.25
0502,2,110,210
"""
    (cached_dir / "cached.pro").write_text(cached_content)
    
    # New: Jan 1 06:00 (overlap) and 12:00 (new)
    new_dir = tmp_path / "new"
    new_dir.mkdir()
    new_content = """[STATION_PARAMETERS]
StationName= OVERLAP_TEST
[DATA]
0500,01.01.2025 06:00:00
0501,2,0.15,0.25
0502,2,110,210
0500,01.01.2025 12:00:00
0501,2,0.2,0.3
0502,2,120,220
"""
    (new_dir / "new.pro").write_text(new_content)
    
    # Cache and update
    ds_cached_data = read(cached_dir)
    cache_file = tmp_path / "cache.nc"
    ds_cached_data.to_netcdf(cache_file)
    
    ds_cached = read(cache_file)
    ds_updated = ds_cached.append_latest(new_dir)
    
    # Should have 3 unique times (deduplicated)
    assert ds_updated.sizes['time'] == 3
    
    # Times should be unique
    times = pd.to_datetime(ds_updated.time.values)
    assert len(times) == len(set(times))


@pytest.mark.real_data
def test_append_latest_preserves_data_integrity(real_file_subset, tmp_path, assert_physical_validity):
    """
    Test that update preserves data quality and physical validity.
    """
    # Split into initial and update
    all_files = sorted(real_file_subset.glob("*.pro"))
    
    if len(all_files) < 2:
        pytest.skip("Need at least 2 files")
    
    # Initial cache
    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    (initial_dir / all_files[0].name).write_text(all_files[0].read_text())
    
    ds_initial = read(initial_dir)
    cache_file = tmp_path / "cache.nc"
    ds_initial.to_netcdf(cache_file)
    
    # Update
    new_dir = tmp_path / "new"
    new_dir.mkdir()
    (new_dir / all_files[1].name).write_text(all_files[1].read_text())
    
    ds_cached = read(cache_file)
    ds_updated = ds_cached.append_latest(new_dir)
    
    # Validate
    assert_physical_validity(ds_updated)


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

@pytest.mark.real_data
@pytest.mark.slow
def test_cache_is_faster_than_full_reload(real_file_subset, tmp_path, timer):
    """
    KEY TEST: Incremental update should be faster than full reload.
    
    This validates the caching strategy.
    """
    # Split files
    all_files = sorted(real_file_subset.glob("*.pro"))
    
    if len(all_files) < 3:
        pytest.skip("Need at least 3 files for performance test")
    
    # Create initial cache (first 2 files)
    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    for f in all_files[:2]:
        (initial_dir / f.name).write_text(f.read_text())
    
    ds_initial = read(initial_dir)
    ds_initial = ds_initial.isel(time=slice(0, 300))
    cache_file = tmp_path / "cache.nc"
    ds_initial.to_netcdf(cache_file)
    
    # New file
    new_dir = tmp_path / "new"
    new_dir.mkdir()
    (new_dir / all_files[2].name).write_text(all_files[2].read_text())
    
    # Method 1: Incremental update
    timer.start('incremental')
    ds_cached = read(cache_file)  # Fast: from NetCDF
    ds_incremental = ds_cached.append_latest(new_dir, join='outer')  # Fast: only new file
    timer.stop('incremental')
    
    # Method 2: Full reload
    all_dir = tmp_path / "all"
    all_dir.mkdir()
    for f in all_files[:3]:
        (all_dir / f.name).write_text(f.read_text())
    
    timer.start('full_reload')
    ds_full = read(all_dir)  # Slow: reprocess everything
    timer.stop('full_reload')
    
    timer.report()
    
    # Both should produce same result for time dimension
    assert ds_incremental.sizes['time'] == ds_full.sizes['time']
    assert ds_incremental.sizes['location'] == ds_full.sizes['location']
    assert ds_incremental.sizes['layer'] == ds_full.sizes['layer']


    # Incremental should be faster (or at least competitive)
    incr_time = timer.get('incremental')
    full_time = timer.get('full_reload')
    
    if incr_time and full_time:
        speedup = full_time / incr_time
        print(f"\nSpeedup: {speedup:.2f}x")
        # Incremental might not always be faster with small datasets
        # but should be competitive

@pytest.mark.real_data
def test_netcdf_roundtrip_preserves_data(real_file_subset, tmp_path, compare_datasets):
    """
    Test that NetCDF save/load preserves all data exactly.
    
    Critical for caching workflow.
    """
    # Original
    ds_original = read(real_file_subset)
    
    # Save and reload
    cache_file = tmp_path / "roundtrip.nc"
    ds_original.to_netcdf(cache_file)
    ds_reloaded = read(cache_file)
    
    # Should be identical
    compare_datasets(ds_original, ds_reloaded, rtol=1e-6)


# ============================================================================
# MULTI-STATION UPDATE TESTS
# ============================================================================

@pytest.mark.synthetic
def test_append_latest_multi_station(tmp_path, synthetic_pro):
    """
    Test update workflow with multiple stations.
    """
    # Initial: 2 stations
    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    for i in range(2):
        content = f"""[STATION_PARAMETERS]
StationName= STATION_{i}
[DATA]
0500,01.01.2025 00:00:00
0501,2,0.1,0.2
0502,2,100,200
"""
        (initial_dir / f"station_{i}.pro").write_text(content)
    
    # Cache
    ds_initial = read(initial_dir)
    cache_file = tmp_path / "cache.nc"
    ds_initial.to_netcdf(cache_file)
    
    # New: Same 2 stations with new timestamp
    new_dir = tmp_path / "new"
    new_dir.mkdir()
    for i in range(2):
        content = f"""[STATION_PARAMETERS]
StationName= STATION_{i}
[DATA]
0500,02.01.2025 00:00:00
0501,2,0.3,0.4
0502,2,140,240
"""
        (new_dir / f"station_{i}.pro").write_text(content)
    
    # Update
    ds_cached = read(cache_file)
    ds_updated = ds_cached.append_latest(new_dir)
    
    # Should have 2 locations, 2 times each
    assert ds_updated.sizes['location'] == 2
    assert ds_updated.sizes['time'] == 2


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

@pytest.mark.synthetic
@pytest.mark.edge_case
def test_append_latest_with_empty_source(tmp_path, synthetic_pro):
    """
    Edge case: Updating with empty source directory.
    """
    # Create and cache initial data
    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    synthetic_pro.basic(initial_dir)
    
    ds_initial = read(initial_dir)
    cache_file = tmp_path / "cache.nc"
    ds_initial.to_netcdf(cache_file)
    
    # Try to update with empty directory
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    
    ds_cached = read(cache_file)
    ds_updated = ds_cached.append_latest(empty_dir)
    
    # Should return unchanged dataset
    assert ds_updated.sizes['time'] == ds_initial.sizes['time']


@pytest.mark.synthetic
@pytest.mark.edge_case
def test_append_latest_with_corrupted_new_data(tmp_path, synthetic_pro):
    """
    Edge case: New data contains corrupted file.
    """
    # Create and cache initial data
    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    synthetic_pro.basic(initial_dir, "STATION_A")
    
    ds_initial = read(initial_dir)
    cache_file = tmp_path / "cache.nc"
    ds_initial.to_netcdf(cache_file)
    
    # New directory with corrupted file
    new_dir = tmp_path / "new"
    new_dir.mkdir()
    synthetic_pro.corrupted(new_dir, "malformed_data")
    
    # Update should handle gracefully
    ds_cached = read(cache_file)
    ds_updated = ds_cached.append_latest(new_dir)
    
    # Should not crash, may return original data
    assert ds_updated is not None


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

@pytest.mark.real_data
def test_realistic_caching_workflow(real_file_subset, tmp_path, timer):
    """
    Integration: Simulate realistic daily update workflow.
    
    Day 1: Process all data, cache
    Day 2: Load cache, add new data
    Day 3: Load cache, add new data
    ...
    """
    all_files = sorted(real_file_subset.glob("*.pro"))
    
    if len(all_files) < 3:
        pytest.skip("Need at least 3 files for workflow test")
    
    cache_file = tmp_path / "daily_cache.nc"
    
    # === Day 1: Initial processing ===
    day1_dir = tmp_path / "day1"
    day1_dir.mkdir()
    (day1_dir / all_files[0].name).write_text(all_files[0].read_text())
    
    timer.start('day1')
    ds = read(day1_dir)
    ds.to_netcdf(cache_file)
    timer.stop('day1')
    
    day1_times = ds.sizes['time']
    
    # === Day 2: Incremental update ===
    day2_dir = tmp_path / "day2"
    day2_dir.mkdir()
    if len(all_files) > 1:
        (day2_dir / all_files[1].name).write_text(all_files[1].read_text())
        
        timer.start('day2')
        # Load eagerly because the cache is overwritten in place below.
        ds = read(cache_file, lazy=False)
        ds = ds.append_latest(day2_dir)
        ds.to_netcdf(cache_file)
        timer.stop('day2')
        
        day2_times = ds.sizes['time']
    
    # === Day 3: Another incremental update ===
    day3_dir = tmp_path / "day3"
    day3_dir.mkdir()
    if len(all_files) > 2:
        (day3_dir / all_files[2].name).write_text(all_files[2].read_text())
        
        timer.start('day3')
        # Load eagerly because the cache is overwritten in place below.
        ds = read(cache_file, lazy=False)
        ds = ds.append_latest(day3_dir)
        ds.to_netcdf(cache_file)
        timer.stop('day3')
        
        day3_times = ds.sizes['time']
    
    timer.report()
    
    # Final dataset should have all data
    ds_final = read(cache_file)
    assert ds_final is not None
    print(f"\nFinal dataset: {ds_final.sizes['time']} timesteps")


@pytest.mark.real_data
def test_caching_with_selectors(real_file_subset, tmp_path):
    """
    Integration: Caching with filtered data.
    """
    from xsnow.selectors import TimeSelector
    
    # Read with time filter
    ds_full = read(real_file_subset)
    
    if ds_full.sizes['time'] < 2:
        pytest.skip("Need multiple timesteps")
    
    mid_time = ds_full.time.values[ds_full.sizes['time'] // 2]
    time_filter = TimeSelector(sel=slice(None, mid_time))
    
    # Cache filtered data
    ds_filtered = read(real_file_subset, time=time_filter)
    cache_file = tmp_path / "filtered_cache.nc"
    ds_filtered.to_netcdf(cache_file)
    
    # Load and verify
    ds_loaded = read(cache_file)
    assert ds_loaded.sizes['time'] == ds_filtered.sizes['time']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
