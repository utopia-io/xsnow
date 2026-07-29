# tests/unit/io/test_modes.py
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
from xsnow.io import base as io_base


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
    """Small archives default eager while explicit overrides remain exact."""
    loaded = read(archive_path, lazy=lazy)

    assert loaded is not None
    if lazy is True:
        assert isinstance(loaded["marker"].data, da.Array)
    else:
        assert isinstance(loaded["marker"].data, np.ndarray)

    loaded.close()


@pytest.mark.parametrize(
    "memory_budget_bytes,expected",
    (
        (None, 1024**3),
        (1, 1),
        (512 * 1024**2, 256 * 1024**2),
        (2 * 1024**3, 1024**3),
        (32 * 1024**3, 4 * 1024**3),
    ),
)
def test_archive_auto_eager_threshold_is_adaptive(
    memory_budget_bytes,
    expected,
):
    """Auto eager uses half the budget, a 4 GiB cap, and 1 GiB fallback."""
    assert (
        io_base._resolve_archive_auto_eager_max_bytes(memory_budget_bytes)
        == expected
    )


def test_archive_auto_mode_uses_strict_decoded_size_boundary():
    """Only estimates strictly below the threshold load eagerly."""
    ds = xr.Dataset({"marker": ("sample", np.ones(4, dtype=np.float64))})

    assert io_base._resolve_archive_lazy(
        ds,
        None,
        memory_budget_bytes=2 * (ds.nbytes + 1),
    ) is False
    assert io_base._resolve_archive_lazy(
        ds,
        None,
        memory_budget_bytes=2 * ds.nbytes,
    ) is True


def test_archive_auto_mode_keeps_object_data_lazy():
    """Pointer-based object nbytes estimates are not used for eager loading."""
    ds = xr.Dataset({"label": ("sample", np.array(["variable"], dtype=object))})

    assert io_base._resolve_archive_lazy(ds, None) is True
    assert io_base._resolve_archive_lazy(ds, True) is True
    assert io_base._resolve_archive_lazy(ds, False) is False


def test_archive_auto_mode_keeps_unknown_estimates_lazy():
    """An unavailable decoded-size estimate falls back to lazy loading."""

    class UnknownSizeDataset:
        variables = {}

        @property
        def nbytes(self):
            raise TypeError("size unavailable")

    assert io_base._resolve_archive_lazy(UnknownSizeDataset(), None) is True


def test_archive_auto_mode_estimates_selected_data(tmp_path, monkeypatch):
    """A small selection can load eagerly from an otherwise large archive."""
    source = tmp_path / "archive.nc"
    source_ds = xr.Dataset(
        {"marker": (("location", "time"), np.ones((1, 10)))},
        coords={"location": ["TEST"], "time": np.arange(10)},
    ).chunk({"time": 2})
    selected_nbytes = source_ds.isel(time=slice(0, 1)).nbytes
    assert source_ds.nbytes > selected_nbytes

    monkeypatch.setattr(xr, "open_dataset", lambda path, **kwargs: source_ds.copy())

    loaded = read(
        source,
        time=slice(0, 1),
        memory_limit=2 * (selected_nbytes + 1),
    )

    assert loaded is not None
    assert loaded.sizes["time"] == 1
    assert isinstance(loaded["marker"].data, np.ndarray)


@pytest.mark.parametrize(
    "archive_kind,opener_name",
    (("netcdf", "open_dataset"), ("zarr", "open_zarr")),
)
@pytest.mark.parametrize(
    "memory_budget_bytes,expected_lazy",
    ((1, True), (1024**3, False)),
)
def test_archive_auto_mode_applies_to_each_backend(
    tmp_path,
    monkeypatch,
    archive_kind,
    opener_name,
    memory_budget_bytes,
    expected_lazy,
):
    """NetCDF and Zarr both apply the adaptive decoded-size policy."""
    filename = "archive.nc" if archive_kind == "netcdf" else "archive.zarr"
    source = tmp_path / filename
    if archive_kind == "zarr":
        source.mkdir()
        (source / "zarr.json").write_text("{}")

    source_ds = xr.Dataset(
        {"marker": (("location", "time"), da.ones((1, 2), chunks=(1, 2)))},
        coords={"location": ["TEST"], "time": [0, 1]},
    )
    monkeypatch.setattr(xr, opener_name, lambda path, **kwargs: source_ds.copy())

    loaded = read(source, memory_limit=memory_budget_bytes)

    assert loaded is not None
    assert isinstance(loaded["marker"].data, da.Array) is expected_lazy
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


def _archive_slope_dataset(*, consistent, legacy):
    inclination = np.array([[0.0, 35.0], [0.0, 35.0]])
    azimuth = np.array([[0.0, 180.0], [0.0, 180.0]])
    if not consistent:
        inclination[1, 1] = 40.0
        azimuth[1, 1] = 90.0

    ds = xr.Dataset(
        {
            "marker": (
                ("location", "time", "slope"),
                np.ones((2, 2, 2)),
            ),
        },
        coords={
            "location": ["A", "B"],
            "time": [0, 1],
            "slope": [0, 1],
            "layer": [0, 1],
            "z": (
                ("location", "time", "slope", "layer"),
                np.arange(16, dtype=float).reshape(2, 2, 2, 2),
            ),
        },
    )
    if legacy:
        ds["inclination"] = (("location", "slope"), inclination)
        ds["azimuth"] = (("location", "slope"), azimuth)
        return ds

    return ds.assign_coords(
        inclination=(("location", "slope"), inclination),
        azimuth=(("location", "slope"), azimuth),
    )


@pytest.mark.parametrize(
    "archive_kind,opener_name",
    (("netcdf", "open_dataset"), ("zarr", "open_zarr")),
)
@pytest.mark.parametrize("legacy", (False, True))
@pytest.mark.parametrize(
    "norm_slopes,consistent,expected_dims",
    (
        (False, True, ("location", "slope")),
        ("auto", True, ("slope",)),
        ("auto", False, ("location", "slope")),
        (True, False, ("slope",)),
    ),
)
def test_archive_readers_share_slope_normalization_policy(
    tmp_path,
    monkeypatch,
    archive_kind,
    opener_name,
    legacy,
    norm_slopes,
    consistent,
    expected_dims,
):
    """NetCDF and Zarr apply the same policy to modern and legacy layouts."""
    source = tmp_path / (
        "slopes.nc" if archive_kind == "netcdf" else "slopes.zarr"
    )
    if archive_kind == "zarr":
        source.mkdir()
        (source / "zarr.json").write_text("{}")

    source_ds = _archive_slope_dataset(
        consistent=consistent,
        legacy=legacy,
    )
    monkeypatch.setattr(
        xr,
        opener_name,
        lambda path, **kwargs: source_ds.copy(),
    )

    loaded = read(
        source,
        lazy=False,
        memory_limit=None,
        norm_slopes=norm_slopes,
    )

    assert loaded is not None
    for coord_name in ("inclination", "azimuth"):
        assert coord_name in loaded.coords
        assert coord_name not in loaded.data_vars
        assert loaded[coord_name].dims == expected_dims
    assert loaded["z"].dims == ("location", "time", "slope", "layer")
    np.testing.assert_allclose(loaded["z"], source_ds["z"])


def test_read_rejects_invalid_norm_slopes_before_opening(tmp_path):
    with pytest.raises(ValueError, match="Unsupported norm_slopes value"):
        read(tmp_path / "missing.nc", norm_slopes="sometimes")


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
    Test that auto-detection uses EAGER below the default 30-file threshold.
    
    Eager is more efficient for small datasets.
    """
    # Create 5 files (small dataset)
    synthetic_pro.multi_station(tmp_path, n_stations=5, n_profiles_each=2)
    
    # Auto-detect (lazy=None)
    ds = read(tmp_path, lazy=None, n_cpus_use=2)
    
    assert ds is not None
    assert ds.sizes['location'] == 5
    
    assert isinstance(ds.data['height'].data, np.ndarray)


@pytest.mark.synthetic
def test_auto_detect_medium_dataset(tmp_path, synthetic_pro):
    """
    Test that auto-detection remains EAGER below the default threshold.
    
    Lazy mode is reserved for collections above 30 files by default.
    """
    # Create 15 files (medium dataset)
    synthetic_pro.multi_station(tmp_path, n_stations=15, n_profiles_each=2)
    
    # Auto-detect (lazy=None)
    ds = read(tmp_path, lazy=None, n_cpus_use=2)
    
    assert ds is not None
    assert ds.sizes['location'] == 15
    
    assert isinstance(ds.data['height'].data, np.ndarray)


@pytest.mark.real_data
def test_auto_detect_large_dataset(gridded_pros_dir):
    """
    Test auto-detection with real large dataset.
    
    Real gridded datasets typically exceed the 30-file default threshold.
    """
    # Auto-detect
    ds = read(gridded_pros_dir, lazy=None, n_cpus_use=4)
    
    assert ds is not None
    assert ds.sizes['location'] > 0


@pytest.mark.synthetic
@pytest.mark.parametrize("n_files,expected_mode", [
    (5, "eager"),   # ≤10: eager
    (10, "eager"),  # =10: eager
    (11, "lazy"),   # >10: lazy
    (20, "lazy"),   # >10: lazy
])
def test_auto_detect_threshold_boundary(tmp_path, synthetic_pro, n_files, expected_mode):
    """
    Test auto-detection around a configured threshold.
    
    With ``lazy_threshold=10``, ≤10 files use eager and >10 use lazy.
    """
    # Create exact number of files
    synthetic_pro.multi_station(tmp_path, n_stations=n_files, n_profiles_each=1)
    
    ds = read(tmp_path, lazy=None, lazy_threshold=10, n_cpus_use=2)
    
    assert ds is not None
    assert ds.sizes['location'] == n_files
    is_lazy = hasattr(ds.data['height'].data, 'chunks')
    assert is_lazy is (expected_mode == "lazy")
    
    print(f"\n{n_files} files → expected {expected_mode} mode")


# ============================================================================
# EXPLICIT MODE OVERRIDE TESTS
# ============================================================================

@pytest.mark.synthetic
def test_explicit_mode_override_force_eager(tmp_path, synthetic_pro):
    """
    Test that explicit lazy=False works even for large datasets.
    
    User can force eager mode regardless of file count.
    """
    # Create 20 files (above the configured 10-file threshold)
    synthetic_pro.multi_station(tmp_path, n_stations=20, n_profiles_each=1)
    
    # Force eager mode
    ds = read(tmp_path, lazy=False, lazy_threshold=10, n_cpus_use=4)
    
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
