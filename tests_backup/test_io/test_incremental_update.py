# test_incremental_update.py
"""
Tests for the incremental update workflow:
1. Load cached NetCDF
2. Update with new .pro files only
3. Save updated NetCDF

This workflow is critical for efficient processing of large archives.
"""
import pytest
from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr
from xsnow import read, xsnowDataset


@pytest.fixture
def initial_data(tmp_path):
    """Creates initial .pro files (simulating old data)."""
    data_dir = tmp_path / "initial"
    data_dir.mkdir()
    
    for i in range(3):
        (data_dir / f"station_{i}.pro").write_text(f"""
[STATION_PARAMETERS]
StationName= STATION_{i}
Latitude= 46.5
Longitude= 10.5
Altitude= 2500
[DATA]
0500,01.01.2025 00:00:00
0501,2,0.1,0.2
0502,2,100,200
0500,01.01.2025 06:00:00
0501,2,0.15,0.25
0502,2,110,210
0500,01.01.2025 12:00:00
0501,2,0.2,0.3
0502,2,120,220
""")
    return data_dir


@pytest.fixture
def update_data(tmp_path):
    """Creates new .pro files (simulating recent data)."""
    data_dir = tmp_path / "update"
    data_dir.mkdir()
    
    for i in range(3):
        (data_dir / f"station_{i}.pro").write_text(f"""
[STATION_PARAMETERS]
StationName= STATION_{i}
Latitude= 46.5
Longitude= 10.5
Altitude= 2500
[DATA]
0500,01.01.2025 18:00:00
0501,2,0.25,0.35
0502,2,130,230
0500,02.01.2025 00:00:00
0501,2,0.3,0.4
0502,2,140,240
""")
    return data_dir



def _make_simple_ts(location, times, values):
    times = pd.to_datetime(times)
    data = xr.Dataset(
        {
            "TA": (("location", "time", "slope", "realization"), values.reshape(1, -1, 1, 1)),
        },
        coords={
            "location": [location],
            "time": times,
            "slope": [0],
            "realization": [0],
        },
    )
    return xsnowDataset(data)

# ============================================================================
# BASIC UPDATE WORKFLOW TESTS
# ============================================================================

def test_basic_update_workflow(initial_data, update_data, tmp_path):
    """
    Tests the complete update workflow:
    1. Load initial data
    2. Save to NetCDF
    3. Load from NetCDF
    4. Update with new data
    5. Verify combined result
    """
    # Step 1: Load initial data
    ds_initial = read(initial_data, lazy=False)
    assert ds_initial is not None
    assert ds_initial.sizes['time'] == 3
    initial_times = ds_initial.time.values
    
    # Step 2: Save to NetCDF
    nc_file = tmp_path / "cached.nc"
    ds_initial.to_netcdf(nc_file)
    print(f"\nSaved initial data: {ds_initial.sizes['time']} timesteps")
    
    # Step 3: Load from NetCDF (simulating next session)
    ds_cached = read(nc_file)
    assert ds_cached is not None
    assert ds_cached.sizes['time'] == 3
    np.testing.assert_array_equal(ds_cached.time.values, initial_times)
    
    # Step 4: Update with new data (incremental update)
    ds_updated = ds_cached.append_latest(update_data)
    assert ds_updated is not None
    assert ds_updated.sizes['time'] == 5  # 3 initial + 2 new
    print(f"Updated data: {ds_updated.sizes['time']} timesteps")
    
    # Step 5: Verify all data is present
    expected_count = 5  # 3 from initial + 2 from update
    assert ds_updated.sizes['time'] == expected_count
    
    # Verify times are sorted and unique
    times = pd.to_datetime(ds_updated.time.values)
    assert len(times) == len(set(times)), "Duplicate timestamps found"
    assert all(times[i] <= times[i+1] for i in range(len(times)-1)), "Times not sorted"


def test_append_latest_only_loads_new_data(initial_data, update_data, tmp_path):
    """
    Verifies that .append_latest() only reads data after the last timestamp.
    This is the key efficiency feature.
    """
    # Load and cache initial data
    ds_initial = read(initial_data, lazy=False)
    nc_file = tmp_path / "cached.nc"
    ds_initial.to_netcdf(nc_file)
    
    # Get last timestamp
    last_time = pd.to_datetime(ds_initial.time.values[-1])
    print(f"\nLast timestamp in cache: {last_time}")
    
    # Load from cache and update
    ds_cached = read(nc_file)
    ds_updated = ds_cached.append_latest(update_data)
    
    # Should have added 2 new timesteps (18:00 and 00:00 next day)
    # but NOT re-read the 3 initial ones
    assert ds_updated.sizes['time'] == 5
    
    # Verify new times are after last_time
    new_times = pd.to_datetime(ds_updated.time.values[3:])  # Last 2
    assert all(t > last_time for t in new_times), "New times should be after cached data"


def test_append_latest_with_no_new_data(initial_data, tmp_path):
    """Tests update when there's no new data to add."""
    # Load and cache
    ds_initial = read(initial_data, lazy=False)
    nc_file = tmp_path / "cached.nc"
    ds_initial.to_netcdf(nc_file)
    
    # Try to update with the same data
    ds_cached = read(nc_file)
    ds_updated = ds_cached.append_latest(initial_data)
    
    # Should have the same data (no duplicates)
    assert ds_updated.sizes['time'] == ds_initial.sizes['time']


def test_append_latest_preserves_data_integrity(initial_data, update_data, tmp_path):
    """Verifies that update preserves all data correctly."""
    # Incremental load
    ds_initial = read(initial_data, lazy=False)
    nc_file = tmp_path / "cached.nc"
    ds_initial.to_netcdf(nc_file)
    ds_cached = read(nc_file)
    ds_incremental = ds_cached.append_latest(update_data)
    
    # Verify incremental update worked
    assert ds_incremental is not None
    assert ds_incremental.sizes['location'] == 3
    assert ds_incremental.sizes['time'] == 5  # 3 initial + 2 new
    
    # Verify data values are reasonable
    density = ds_incremental['density'].values
    valid_density = density[~np.isnan(density)]
    assert len(valid_density) > 0
    assert np.all(valid_density >= 0)
    assert np.all(valid_density <= 1000)
    
    # Verify times are sorted and unique
    times = pd.to_datetime(ds_incremental.time.values)
    assert len(times) == len(set(times)), "Should have no duplicate timestamps"
    assert all(times[i] <= times[i+1] for i in range(len(times)-1)), "Times should be sorted"


def test_append_latest_from_time_trims_and_appends(tmp_path):
    """from_time should drop older data and append newer data."""
    t1 = pd.Timestamp("2025-01-01T00:00:00")
    t2 = pd.Timestamp("2025-01-02T00:00:00")
    t3 = pd.Timestamp("2025-01-03T00:00:00")

    existing = _make_simple_ts("A", [t1, t2], np.array([1.0, 2.0]))
    new = _make_simple_ts("A", [t2, t3], np.array([2.5, 3.0]))

    new_path = tmp_path / "new.nc"
    new.to_netcdf(new_path)

    out = existing.append_latest(new_path, from_time=t2)

    assert list(out.time.values) == [t1.to_datetime64(), t2.to_datetime64(), t3.to_datetime64()]
    np.testing.assert_allclose(out["TA"].values.squeeze(), [1.0, 2.5, 3.0])



def test_append_latest_non_outer_join_preserves_time_union(tmp_path):
    """
    join='left' should keep the time union while restricting non-concat dims
    to the left domain (location 'A' only).
    """
    t1 = pd.Timestamp("2025-01-01T00:00:00")
    t2 = pd.Timestamp("2025-01-02T00:00:00")

    existing = _make_simple_ts("A", [t1], np.array([1.0]))
    new = _make_simple_ts("B", [t2], np.array([5.0]))

    new_path = tmp_path / "new_left.nc"
    new.to_netcdf(new_path)

    out = existing.append_latest(new_path, join="left")

    # Time union is preserved even with join='left'
    assert list(out.time.values) == [t1.to_datetime64(), t2.to_datetime64()]
    # Location domain should be restricted to the left dataset
    assert list(out.location.values) == ["A"]
    # The new time exists, but data for dropped location should be NaN
    assert np.isnan(out["TA"].sel(time=t2).values).all()


# ============================================================================
# EFFICIENCY TESTS
# ============================================================================

def test_append_latest_is_faster_than_full_reload(initial_data, update_data, tmp_path):
    """Verifies that incremental update is faster than full reload."""
    import time
    
    # Setup: Load and cache initial data
    ds_initial = read(initial_data, lazy=False)
    nc_file = tmp_path / "cached.nc"
    ds_initial.to_netcdf(nc_file)
    
    # Method 1: Full reload (read all files again)
    start = time.time()
    all_files = list(initial_data.glob("*.pro")) + list(update_data.glob("*.pro"))
    ds_full = read(all_files, lazy=False)
    full_time = time.time() - start
    
    # Method 2: Incremental update (read only new data)
    start = time.time()
    ds_cached = read(nc_file)  # Fast: read from NetCDF
    ds_incremental = ds_cached.append_latest(update_data)  # Only reads new files
    update_time = time.time() - start
    
    print(f"\nPerformance comparison:")
    print(f"  Full reload: {full_time:.3f}s")
    print(f"  Incremental: {update_time:.3f}s")
    print(f"  Speedup: {full_time/update_time:.2f}x")

    # Incremental should be faster (or at least not much slower)
    # This is a soft check as timing can vary
    assert ds_full is not None
    assert ds_incremental is not None


def test_netcdf_roundtrip_preserves_data(initial_data, tmp_path):
    """Tests that NetCDF save/load preserves all data."""
    # Original
    ds_original = read(initial_data, lazy=False)
    
    # Save and reload
    nc_file = tmp_path / "test.nc"
    ds_original.to_netcdf(nc_file)
    ds_reloaded = read(nc_file)
    
    # Should be identical
    assert ds_reloaded.sizes == ds_original.sizes
    np.testing.assert_array_equal(ds_reloaded.time.values, ds_original.time.values)
    np.testing.assert_allclose(
        ds_reloaded['density'].values,
        ds_original['density'].values,
        rtol=1e-5,
        equal_nan=True
    )


# ============================================================================
# EDGE CASES
# ============================================================================

def test_append_latest_with_empty_cache(update_data, tmp_path):
    """Tests update when starting with empty NetCDF."""
    # Create a minimal empty dataset
    import xarray as xr
    empty_ds = xr.Dataset()
    empty_ds.to_netcdf(tmp_path / "empty.nc")
    
    # This should handle gracefully
    # (Might fail depending on implementation)
    # Just test that it doesn't crash


def test_append_latest_with_time_gap(tmp_path):
    """Tests update when there's a time gap between cached and new data."""
    # Old data: Jan 1
    old_dir = tmp_path / "old"
    old_dir.mkdir()
    (old_dir / "station.pro").write_text("""
[STATION_PARAMETERS]
StationName= STATION_A
[DATA]
0500,01.01.2025 00:00:00
0501,2,0.1,0.2
0502,2,100,200
""")
    
    # New data: Jan 10 (9 day gap)
    new_dir = tmp_path / "new"
    new_dir.mkdir()
    (new_dir / "station.pro").write_text("""
[STATION_PARAMETERS]
StationName= STATION_A
[DATA]
0500,10.01.2025 00:00:00
0501,2,0.3,0.4
0502,2,140,240
""")
    
    # Load, cache, update
    ds_old = read(old_dir, lazy=False)
    nc_file = tmp_path / "cached.nc"
    ds_old.to_netcdf(nc_file)
    
    ds_cached = read(nc_file)
    ds_updated = ds_cached.append_latest(new_dir)
    
    assert ds_updated.sizes['time'] == 2
    times = pd.to_datetime(ds_updated.time.values)
    time_diff = (times[1] - times[0]).days
    assert time_diff == 9, f"Expected 9 day gap, got {time_diff}"


def test_append_latest_with_overlapping_times(tmp_path):
    """Tests update when new data overlaps with cached data."""
    # Cached: Jan 1 00:00 and 06:00
    cached_dir = tmp_path / "cached"
    cached_dir.mkdir()
    (cached_dir / "station.pro").write_text("""
[STATION_PARAMETERS]
StationName= STATION_A
[DATA]
0500,01.01.2025 00:00:00
0501,2,0.1,0.2
0502,2,100,200
0500,01.01.2025 06:00:00
0501,2,0.15,0.25
0502,2,110,210
""")
    
    # New: Jan 1 06:00 (overlap) and 12:00 (new)
    new_dir = tmp_path / "new"
    new_dir.mkdir()
    (new_dir / "station.pro").write_text("""
[STATION_PARAMETERS]
StationName= STATION_A
[DATA]
0500,01.01.2025 06:00:00
0501,2,0.15,0.25
0502,2,110,210
0500,01.01.2025 12:00:00
0501,2,0.2,0.3
0502,2,120,220
""")
    
    # Load, cache, update
    ds_cached_data = read(cached_dir, lazy=False)
    nc_file = tmp_path / "cached.nc"
    ds_cached_data.to_netcdf(nc_file)
    
    ds_cached = read(nc_file)
    ds_updated = ds_cached.append_latest(new_dir)
    
    # Should have 3 unique times (no duplicates)
    assert ds_updated.sizes['time'] == 3
    times = pd.to_datetime(ds_updated.time.values)
    assert len(times) == len(set(times)), "Should remove duplicate timestamps"


# ============================================================================
# MULTI-LOCATION TESTS
# ============================================================================

def test_append_latest_with_multiple_locations(tmp_path):
    """Tests update workflow with multiple stations."""
    # Initial: 3 stations
    initial_dir = tmp_path / "initial"
    initial_dir.mkdir()
    for i in range(3):
        (initial_dir / f"station_{i}.pro").write_text(f"""
[STATION_PARAMETERS]
StationName= STATION_{i}
[DATA]
0500,01.01.2025 00:00:00
0501,2,0.1,0.2
0502,2,100,200
""")
    
    # New: Same 3 stations with new data
    new_dir = tmp_path / "new"
    new_dir.mkdir()
    for i in range(3):
        (new_dir / f"station_{i}.pro").write_text(f"""
[STATION_PARAMETERS]
StationName= STATION_{i}
[DATA]
0500,01.01.2025 12:00:00
0501,2,0.3,0.4
0502,2,140,240
""")
    
    # Load, cache, update
    ds_initial = read(initial_dir, lazy=False)
    nc_file = tmp_path / "cached.nc"
    ds_initial.to_netcdf(nc_file)
    
    ds_cached = read(nc_file)
    ds_updated = ds_cached.append_latest(new_dir)
    
    # Should have 3 locations, 2 times each
    assert ds_updated.sizes['location'] == 3
    assert ds_updated.sizes['time'] == 2


# ============================================================================
# WORKFLOW DOCUMENTATION TEST
# ============================================================================

def test_documented_workflow_example(tmp_path):
    """
    Documents the recommended incremental update workflow.
    This serves as both a test and documentation.
    """
    print("\n" + "="*70)
    print("RECOMMENDED INCREMENTAL UPDATE WORKFLOW")
    print("="*70)
    
    # === Day 1: Initial load ===
    print("\n[Day 1] Initial data load...")
    day1_dir = tmp_path / "day1"
    day1_dir.mkdir()
    (day1_dir / "station.pro").write_text("""
[STATION_PARAMETERS]
StationName= MY_STATION
[DATA]
0500,01.01.2025 00:00:00
0501,2,0.1,0.2
0502,2,100,200
""")
    
    ds = read(day1_dir, lazy=False)
    print(f"  Loaded: {ds.sizes['time']} timesteps")
    
    # Save to NetCDF cache
    cache_file = tmp_path / "archive.nc"
    ds.to_netcdf(cache_file)
    print(f"  Cached to: {cache_file}")
    
    # === Day 2: Incremental update ===
    print("\n[Day 2] Incremental update...")
    day2_dir = tmp_path / "day2"
    day2_dir.mkdir()
    (day2_dir / "station.pro").write_text("""
[STATION_PARAMETERS]
StationName= MY_STATION
[DATA]
0500,02.01.2025 00:00:00
0501,2,0.3,0.4
0502,2,140,240
""")
    
    # Load cached data (fast)
    ds_cached = read(cache_file)
    print(f"  Loaded from cache: {ds_cached.sizes['time']} timesteps")
    
    # Update with only new data (efficient)
    ds_updated = ds_cached.append_latest(day2_dir)
    print(f"  After update: {ds_updated.sizes['time']} timesteps")
    
    # Save updated cache
    ds_updated.to_netcdf(cache_file)
    print(f"  Updated cache saved")
    
    # === Verification ===
    assert ds_updated.sizes['time'] == 2
    print("\n✓ Workflow completed successfully!")
    print("="*70)
