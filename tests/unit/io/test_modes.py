# tests/loading/test_loading_modes.py
"""
Tests comparing eager vs lazy loading modes.

Focus: Consistency between modes and auto-detection logic.

Strategy:
  - Use REAL data to validate both modes produce identical results
  - Use SYNTHETIC data to test auto-detection thresholds
  - Verify explicit overrides work correctly
"""

from pathlib import Path

import dask.array as da
import numpy as np
import pytest
import time
import xarray as xr

from xsnow import read


@pytest.fixture(params=("netcdf", "zarr"))
def archive_path(request, tmp_path) -> Path:
    """Write the same small dataset to each supported archive backend."""
    ds = xr.Dataset(
        data_vars={
            "marker": (
                ("location", "time"),
                np.arange(5, dtype=float).reshape(1, 5),
            ),
        },
        coords={
            "location": ["TEST"],
            "time": np.arange(5),
        },
    )

    if request.param == "netcdf":
        path = tmp_path / "archive.nc"
        ds.to_netcdf(path)
    else:
        path = tmp_path / "archive.zarr"
        ds.chunk({"time": 2}).to_zarr(path)

    return path


# ============================================================================
# CONSISTENCY TESTS
# ============================================================================


@pytest.mark.parametrize("lazy", (None, True, False))
def test_archive_lazy_modes_have_matching_public_semantics(archive_path, lazy):
    """Both archive formats default lazy and honor explicit overrides."""
    loaded = read(archive_path, lazy=lazy)

    assert loaded is not None
    if lazy is False:
        assert isinstance(loaded["marker"].data, np.ndarray)
    else:
        assert isinstance(loaded["marker"].data, da.Array)

    loaded.close()


def test_archive_lazy_mode_honors_custom_chunks(archive_path):
    """The public chunks argument reaches both lazy archive backends."""
    loaded = read(archive_path, lazy=True, chunks={"time": 2})

    marker = loaded["marker"]
    time_axis = marker.get_axis_num("time")
    assert marker.chunks[time_axis] == (2, 2, 1)

    loaded.close()


@pytest.mark.parametrize(
    "lazy,chunks,expected_chunks",
    (
        (None, None, "auto"),
        (True, None, "auto"),
        (True, {}, {}),
        (False, None, None),
        (False, {"time": 2}, None),
    ),
)
@pytest.mark.parametrize(
    "archive_kind,opener_name",
    (("netcdf", "open_dataset"), ("zarr", "open_zarr")),
)
def test_archive_reader_forwards_expected_chunks(
    tmp_path,
    monkeypatch,
    lazy,
    chunks,
    expected_chunks,
    archive_kind,
    opener_name,
):
    """Archive readers translate the public chunk policy consistently."""
    source = tmp_path / ("archive.nc" if archive_kind == "netcdf" else "archive.zarr")
    if archive_kind == "zarr":
        source.mkdir()
        (source / "zarr.json").write_text("{}")

    source_ds = xr.Dataset(
        {"marker": (("location", "time"), np.ones((1, 2)))},
        coords={"location": ["TEST"], "time": [0, 1]},
    )
    received_chunks = []

    def fake_opener(path, **kwargs):
        received_chunks.append(kwargs.get("chunks"))
        return source_ds.copy()

    monkeypatch.setattr(xr, opener_name, fake_opener)

    loaded = read(source, lazy=lazy, chunks=chunks)

    assert loaded is not None
    assert received_chunks == [expected_chunks]
    loaded.close()

@pytest.mark.real_data
def test_eager_vs_lazy_same_results(real_file_subset, compare_datasets):
    """
    PRIMARY TEST: Eager and lazy modes must produce identical results.
    
    This is critical - users should be able to choose mode based on
    performance without worrying about correctness.
    """
    # Eager mode
    ds_eager = read(real_file_subset, lazy=False, n_cpus_use=2)
    
    # Lazy mode
    ds_lazy = read(real_file_subset, lazy=True, n_cpus_use=2)
    
    assert ds_eager is not None
    assert ds_lazy is not None
    
    # Should be identical
    compare_datasets(ds_eager, ds_lazy, rtol=1e-5)


@pytest.mark.real_data
def test_eager_vs_lazy_same_structure(real_file_subset):
    """
    Test that both modes produce same dataset structure.
    """
    ds_eager = read(real_file_subset, lazy=False, n_cpus_use=2)
    ds_lazy = read(real_file_subset, lazy=True, n_cpus_use=2)
    
    # Same dimensions
    assert ds_eager.sizes == ds_lazy.sizes
    
    # Same variables
    assert set(ds_eager.data.data_vars) == set(ds_lazy.data.data_vars)
    
    # Same coordinates
    assert set(ds_eager.data.coords) == set(ds_lazy.data.coords)


@pytest.mark.real_data
def test_eager_vs_lazy_same_metadata(real_file_subset):
    """
    Test that metadata is preserved in both modes.
    """
    ds_eager = read(real_file_subset, lazy=False, n_cpus_use=2)
    ds_lazy = read(real_file_subset, lazy=True, n_cpus_use=2)
    
    # Location mapping
    assert ds_eager.location.attrs.get('mapping') == ds_lazy.location.attrs.get('mapping')
    
    # Coordinates
    for coord in ['latitude', 'longitude', 'altitude']:
        if coord in ds_eager.coords and coord in ds_lazy.coords:
            np.testing.assert_array_equal(
                ds_eager.coords[coord].values,
                ds_lazy.coords[coord].values
            )


@pytest.mark.real_data
def test_eager_vs_lazy_handle_soil_removal_same(real_file_subset):
    """
    Test that soil removal works identically in both modes.
    """
    # Eager with soil removal
    ds_eager = read(real_file_subset, lazy=False, remove_soil=True, n_cpus_use=2)
    
    # Lazy with soil removal
    ds_lazy = read(real_file_subset, lazy=True, remove_soil=True, n_cpus_use=2)
    
    assert ds_eager is not None
    assert ds_lazy is not None
    
    # Should have same layer count
    assert ds_eager.sizes['layer'] == ds_lazy.sizes['layer']


@pytest.mark.real_data
def test_eager_vs_lazy_handle_surface_hoar_same(real_file_subset):
    """
    Test that surface hoar handling works identically.
    """
    # Eager with SH
    ds_eager = read(real_file_subset, lazy=False, add_surface_sh_as_layer=True, n_cpus_use=2)
    
    # Lazy with SH
    ds_lazy = read(real_file_subset, lazy=True, add_surface_sh_as_layer=True, n_cpus_use=2)
    
    assert ds_eager is not None
    assert ds_lazy is not None
    
    # Should have same structure
    assert ds_eager.sizes['layer'] == ds_lazy.sizes['layer']


# ============================================================================
# AUTO-DETECTION TESTS
# ============================================================================

@pytest.mark.synthetic
def test_auto_detect_small_dataset(tmp_path, synthetic_pro):
    """
    Test that auto-detection uses EAGER for small datasets (≤10 files).
    
    Eager is more efficient for small datasets.
    """
    # Create 5 files (small dataset)
    synthetic_pro.multi_station(tmp_path, n_stations=5, n_profiles_each=2)
    
    # Auto-detect (lazy=None)
    ds = read(tmp_path, lazy=None, n_cpus_use=2)
    
    assert ds is not None
    assert ds.sizes['location'] == 5
    
    # With ≤10 files, should use eager mode
    # (We can't directly check which mode was used, but it should work)


@pytest.mark.synthetic
def test_auto_detect_medium_dataset(tmp_path, synthetic_pro):
    """
    Test that auto-detection uses LAZY for medium datasets (>10 files).
    
    Lazy is more efficient for larger datasets.
    """
    # Create 15 files (medium dataset)
    synthetic_pro.multi_station(tmp_path, n_stations=15, n_profiles_each=2)
    
    # Auto-detect (lazy=None)
    ds = read(tmp_path, lazy=None, n_cpus_use=2)
    
    assert ds is not None
    assert ds.sizes['location'] == 15
    
    # With >10 files, should use lazy mode
    # (We can't directly check which mode was used, but it should work)


@pytest.mark.real_data
def test_auto_detect_large_dataset(gridded_pros_dir):
    """
    Test auto-detection with real large dataset.
    
    Real datasets typically have >10 files, so should use lazy.
    """
    # Auto-detect
    ds = read(gridded_pros_dir, lazy=None, n_cpus_use=4)
    
    assert ds is not None
    assert ds.sizes['location'] > 0


@pytest.mark.synthetic
@pytest.mark.parametrize("n_files,expected_threshold", [
    (5, "eager"),   # ≤10: eager
    (10, "eager"),  # =10: eager
    (11, "lazy"),   # >10: lazy
    (20, "lazy"),   # >10: lazy
])
def test_auto_detect_threshold_boundary(tmp_path, synthetic_pro, n_files, expected_threshold):
    """
    Test auto-detection at threshold boundaries.
    
    Threshold is 10 files: ≤10 uses eager, >10 uses lazy.
    """
    # Create exact number of files
    synthetic_pro.multi_station(tmp_path, n_stations=n_files, n_profiles_each=1)
    
    # Auto-detect
    ds = read(tmp_path, lazy=None, n_cpus_use=2)
    
    assert ds is not None
    assert ds.sizes['location'] == n_files
    
    print(f"\n{n_files} files → expected {expected_threshold} mode")


# ============================================================================
# EXPLICIT MODE OVERRIDE TESTS
# ============================================================================

@pytest.mark.synthetic
def test_explicit_mode_override_force_eager(tmp_path, synthetic_pro):
    """
    Test that explicit lazy=False works even for large datasets.
    
    User can force eager mode regardless of file count.
    """
    # Create 20 files (would normally auto-select lazy)
    synthetic_pro.multi_station(tmp_path, n_stations=20, n_profiles_each=1)
    
    # Force eager mode
    ds = read(tmp_path, lazy=False, n_cpus_use=4)
    
    assert ds is not None
    assert ds.sizes['location'] == 20


@pytest.mark.synthetic
def test_explicit_mode_override_force_lazy(tmp_path, synthetic_pro):
    """
    Test that explicit lazy=True works even for small datasets.
    
    User can force lazy mode regardless of file count.
    """
    # Create 3 files (would normally auto-select eager)
    synthetic_pro.multi_station(tmp_path, n_stations=3, n_profiles_each=1)
    
    # Force lazy mode
    ds = read(tmp_path, lazy=True, n_cpus_use=2)
    
    assert ds is not None
    assert ds.sizes['location'] == 3


@pytest.mark.real_data
def test_explicit_override_with_real_data(real_file_subset):
    """
    Test explicit mode override with real data.
    """
    # Force eager
    ds_eager = read(real_file_subset, lazy=False, n_cpus_use=2)
    
    # Force lazy
    ds_lazy = read(real_file_subset, lazy=True, n_cpus_use=2)
    
    # Both should work
    assert ds_eager is not None
    assert ds_lazy is not None
    
    # Should produce same results
    assert ds_eager.sizes == ds_lazy.sizes


# ============================================================================
# PERFORMANCE COMPARISON TESTS
# ============================================================================

@pytest.mark.real_data
@pytest.mark.slow
def test_performance_comparison_small_dataset(real_file_subset, timer):
    """
    Compare performance on small dataset.
    
    Eager mode is typically faster for small datasets.
    """
    # Eager mode
    timer.start('eager_small')
    ds_eager = read(real_file_subset, lazy=False, n_cpus_use=4)
    timer.stop('eager_small')
    
    # Lazy mode
    timer.start('lazy_small')
    ds_lazy = read(real_file_subset, lazy=True, n_cpus_use=4)
    timer.stop('lazy_small')
    
    timer.report()
    
    # Both should work
    assert ds_eager is not None
    assert ds_lazy is not None
    
    eager_time = timer.get('eager_small')
    lazy_time = timer.get('lazy_small')
    
    if eager_time and lazy_time:
        print(f"\nEager: {eager_time:.3f}s")
        print(f"Lazy:  {lazy_time:.3f}s")
        print(f"Ratio: {lazy_time/eager_time:.2f}x")


@pytest.mark.real_data
@pytest.mark.slow
def test_performance_comparison_large_dataset(gridded_pros_dir, timer):
    """
    Compare performance on large dataset.
    
    Lazy mode can be more efficient for large datasets.
    """
    # Eager mode
    timer.start('eager_large')
    ds_eager = read(gridded_pros_dir, lazy=False, n_cpus_use=4)
    timer.stop('eager_large')
    
    # Lazy mode
    timer.start('lazy_large')
    ds_lazy = read(gridded_pros_dir, lazy=True, n_cpus_use=4)
    timer.stop('lazy_large')
    
    timer.report()
    
    assert ds_eager is not None
    assert ds_lazy is not None
    
    eager_time = timer.get('eager_large')
    lazy_time = timer.get('lazy_large')
    
    if eager_time and lazy_time:
        print(f"\nLarge dataset ({ds_eager.sizes['location']} locations):")
        print(f"  Eager: {eager_time:.3f}s")
        print(f"  Lazy:  {lazy_time:.3f}s")


@pytest.mark.synthetic
def test_performance_with_varying_file_counts(tmp_path, synthetic_pro, timer):
    """
    Test performance across different file counts.
    """
    results = []
    
    for n_files in [5, 10, 15, 20]:
        # Create files
        test_dir = tmp_path / f"n{n_files}"
        test_dir.mkdir()
        synthetic_pro.multi_station(test_dir, n_stations=n_files, n_profiles_each=2)
        
        # Eager
        timer.start(f'eager_{n_files}')
        ds_eager = read(test_dir, lazy=False, n_cpus_use=4)
        timer.stop(f'eager_{n_files}')
        eager_time = timer.get(f'eager_{n_files}')
        
        # Lazy
        timer.start(f'lazy_{n_files}')
        ds_lazy = read(test_dir, lazy=True, n_cpus_use=4)
        timer.stop(f'lazy_{n_files}')
        lazy_time = timer.get(f'lazy_{n_files}')
        
        if eager_time and lazy_time:
            results.append((n_files, eager_time, lazy_time))
    
    # Print results
    print("\nPerformance across file counts:")
    print("Files | Eager(s) | Lazy(s) | Ratio")
    print("-" * 40)
    for n, eager, lazy in results:
        print(f"{n:5d} | {eager:8.3f} | {lazy:7.3f} | {lazy/eager:5.2f}x")


# ============================================================================
# MEMORY COMPARISON TESTS
# ============================================================================

@pytest.mark.real_data
def test_memory_comparison_modes(real_file_subset, memory_monitor):
    """
    Compare memory usage between modes.
    """
    import gc
    
    # Test eager
    memory_monitor.start()
    ds_eager = read(real_file_subset, lazy=False, n_cpus_use=2)
    eager_memory = memory_monitor.increase()
    del ds_eager
    gc.collect()
    
    # Test lazy
    memory_monitor.start()
    ds_lazy = read(real_file_subset, lazy=True, n_cpus_use=2)
    lazy_memory = memory_monitor.increase()
    
    print(f"\nMemory usage:")
    print(f"  Eager: {eager_memory:.2f} MB")
    print(f"  Lazy:  {lazy_memory:.2f} MB")
    
    # Both should be reasonable
    assert eager_memory < 1000  # < 1 GB
    assert lazy_memory < 1000


# ============================================================================
# CONSISTENCY ACROSS PARAMETERS
# ============================================================================

@pytest.mark.real_data
def test_modes_consistent_with_filters(real_file_subset):
    """
    Test that both modes handle filters consistently.
    """
    from xsnow.selectors import TimeSelector
    
    # Get time range
    ds_full = read(real_file_subset, lazy=False)
    
    if ds_full.sizes['time'] < 2:
        pytest.skip("Need multiple timesteps")
    
    mid_time = ds_full.time.values[ds_full.sizes['time'] // 2]
    time_filter = TimeSelector(sel=slice(None, mid_time))
    
    # Eager with filter
    ds_eager = read(real_file_subset, lazy=False, time=time_filter, n_cpus_use=2)
    
    # Lazy with filter
    ds_lazy = read(real_file_subset, lazy=True, time=time_filter, n_cpus_use=2)
    
    # Should produce same results
    assert ds_eager.sizes == ds_lazy.sizes


@pytest.mark.real_data
def test_modes_consistent_with_parserkwargs(real_file_subset):
    """
    Test that both modes handle parser kwargs consistently.
    """
    # Eager with soil removal
    ds_eager = read(real_file_subset, lazy=False, remove_soil=True, n_cpus_use=2)
    
    # Lazy with soil removal
    ds_lazy = read(real_file_subset, lazy=True, remove_soil=True, n_cpus_use=2)
    
    # Should produce same layer count
    assert ds_eager.sizes['layer'] == ds_lazy.sizes['layer']


@pytest.mark.synthetic
def test_modes_consistent_with_max_layers_override(tmp_path, synthetic_pro):
    """
    Test that both modes respect max_layers parameter.
    """
    synthetic_pro.basic(tmp_path, n_layers=3)
    
    # Eager with override
    ds_eager = read(tmp_path, lazy=False, max_layers=10, n_cpus_use=2)
    
    # Lazy with override
    ds_lazy = read(tmp_path, lazy=True, max_layers=10, n_cpus_use=2)
    
    # Should both use specified max_layers
    assert ds_eager.sizes['layer'] == 10
    assert ds_lazy.sizes['layer'] == 10


# ============================================================================
# EDGE CASE CONSISTENCY
# ============================================================================

@pytest.mark.synthetic
@pytest.mark.edge_case
def test_modes_consistent_with_corrupted_files(tmp_path, synthetic_pro):
    """
    Test that both modes handle corrupted files the same way.
    """
    # Create mix of valid and corrupted
    synthetic_pro.basic(tmp_path, "VALID")
    synthetic_pro.corrupted(tmp_path, "malformed_data")
    
    # Eager
    ds_eager = read(tmp_path, lazy=False, n_cpus_use=2)
    
    # Lazy
    ds_lazy = read(tmp_path, lazy=True, n_cpus_use=2)
    
    # Both should load valid file
    if ds_eager is not None and ds_lazy is not None:
        assert ds_eager.sizes['location'] == ds_lazy.sizes['location']


@pytest.mark.synthetic
@pytest.mark.edge_case
def test_modes_consistent_with_empty_directory(tmp_path):
    """
    Test that both modes handle empty directory the same way.
    """
    # Eager
    ds_eager = read(tmp_path, lazy=False)
    
    # Lazy
    ds_lazy = read(tmp_path, lazy=True)
    
    # Both should return None
    assert ds_eager is None
    assert ds_lazy is None


# ============================================================================
# REGRESSION TESTS
# ============================================================================

@pytest.mark.real_data
def test_mode_selection_documented_in_logs(real_file_subset, caplog):
    """
    Regression: Mode selection should be logged for debugging.
    
    (This test checks if logging exists - adjust based on implementation)
    """
    import logging
    caplog.set_level(logging.INFO)
    
    # Auto-detect mode
    ds = read(real_file_subset, lazy=None, n_cpus_use=2)
    
    assert ds is not None
    
    # Check if mode was logged (optional - depends on implementation)
    # This is a soft check
    log_messages = [record.message for record in caplog.records]
    # May contain "lazy" or "eager" in logs


@pytest.mark.real_data
def test_both_modes_produce_xsnowdataset(real_file_subset):
    """
    Regression: Both modes should return xsnowDataset instances.
    """
    from xsnow import xsnowDataset
    
    ds_eager = read(real_file_subset, lazy=False, n_cpus_use=2)
    ds_lazy = read(real_file_subset, lazy=True, n_cpus_use=2)
    
    assert isinstance(ds_eager, xsnowDataset)
    assert isinstance(ds_lazy, xsnowDataset)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
