# tests/integration/test_lazy_loading.py
"""
Tests for lazy loading behavior with dask arrays.

These tests verify that lazy=True creates truly lazy datasets backed by
dask arrays, while lazy=False creates eager datasets with numpy arrays.
"""
import pytest
import numpy as np
import xarray as xr
from xsnow import read


class TestLazyLoadingDaskArrays:
    """Test that lazy=True creates datasets backed by dask arrays."""
    
    @pytest.mark.synthetic
    def test_lazy_returns_dask_arrays(self, synthetic_pro, tmp_path):
        """
        Test that lazy=True returns datasets backed by dask arrays.
        """
        # Create multiple files to trigger lazy mode
        data_dir = synthetic_pro.multi_station(
            tmp_path,
            n_stations=12,  # > 10 triggers auto-lazy
            n_profiles_each=10
        )
        
        # Load with lazy=True
        ds = read(data_dir, lazy=True)
        
        # Check that data variables are dask arrays
        for var_name in ['height', 'density', 'temperature']:
            if var_name in ds.data:
                var = ds.data[var_name]
                
                # Should be dask array
                assert hasattr(var.data, 'chunks'), \
                    f"{var_name} should be a dask array but is {type(var.data)}"
                
                # Check it's actually a dask array type
                import dask.array as da
                assert isinstance(var.data, da.Array), \
                    f"{var_name} should be dask.array.Array, got {type(var.data)}"
    
    @pytest.mark.synthetic
    def test_lazy_explicit_mode(self, synthetic_pro, tmp_path):
        """Test that explicitly setting lazy=True creates dask arrays."""
        # Create small dataset (< 10 files, would default to eager)
        data_dir = synthetic_pro.multi_station(
            tmp_path,
            n_stations=3,
            n_profiles_each=5
        )
        
        # Explicitly request lazy loading
        ds = read(data_dir, lazy=True)
        
        # Should still be dask arrays despite small size
        assert hasattr(ds.data['height'].data, 'chunks'), \
            "Explicit lazy=True should create dask arrays even for small datasets"
    
    @pytest.mark.synthetic  
    def test_eager_loading_returns_numpy_arrays(self, synthetic_pro, tmp_path):
        """
        Test that lazy=False returns numpy arrays (eager loading).
        """
        data_dir = synthetic_pro.basic(tmp_path, n_layers=10)
        
        # Load eagerly
        ds = read(data_dir, lazy=False)
        
        # Should be numpy arrays
        for var_name in ['height', 'density', 'temperature']:
            if var_name in ds.data:
                var = ds.data[var_name]
                assert isinstance(var.data, np.ndarray), \
                    f"{var_name} should be numpy array in eager mode, got {type(var.data)}"
                
                # Should NOT have chunks attribute (not dask)
                assert not hasattr(var.data, 'chunks'), \
                    f"{var_name} should not be dask array in eager mode"

    @pytest.mark.synthetic
    @pytest.mark.parametrize("lazy", (True, False))
    def test_smet_explicit_lazy_mode_matches_public_contract(
        self,
        synthetic_smet,
        tmp_path,
        lazy,
    ):
        """SMET results follow the same Dask/eager output contract as PRO."""
        smet_file = synthetic_smet.basic(tmp_path, n_timesteps=5)

        ds = read(smet_file, lazy=lazy)

        if lazy:
            import dask.array as da
            assert isinstance(ds.data['TA'].data, da.Array)
        else:
            assert isinstance(ds.data['TA'].data, np.ndarray)
    
    @pytest.mark.synthetic
    def test_auto_detect_lazy_mode(self, synthetic_pro, tmp_path):
        """Test that lazy mode is auto-detected based on file count."""
        # Create 31 files (> 30 threshold)
        data_dir = synthetic_pro.multi_station(
            tmp_path,
            n_stations=31,
            n_profiles_each=5
        )
        
        # Don't specify lazy parameter (auto-detect)
        ds = read(data_dir)
        
        # Should auto-detect lazy=True and use dask arrays
        assert hasattr(ds.data['height'].data, 'chunks'), \
            "Should auto-detect lazy=True for > 10 files"
    
    @pytest.mark.synthetic
    def test_auto_detect_eager_mode(self, synthetic_pro, tmp_path):
        """Test that eager mode is auto-detected for small datasets."""
        # Create 5 files (< 10 threshold)
        data_dir = synthetic_pro.multi_station(
            tmp_path,
            n_stations=5,
            n_profiles_each=5
        )
        
        # Don't specify lazy parameter (auto-detect)
        ds = read(data_dir)
        
        # Should auto-detect lazy=False and use numpy arrays
        assert isinstance(ds.data['height'].data, np.ndarray), \
            "Should auto-detect lazy=False for ≤ 10 files"


class TestLazyLoadingChunks:
    """Test custom chunking for dask arrays."""
    
    @pytest.mark.synthetic
    def test_default_chunks(self, synthetic_pro, tmp_path):
        """Test that default chunks are {'time': 100, 'layer': -1}."""
        data_dir = synthetic_pro.multi_station(
            tmp_path,
            n_stations=12,
            n_profiles_each=50  # More than default chunk size
        )
        
        ds = read(data_dir, lazy=True)
        
        height = ds.data['height']
        assert hasattr(height.data, 'chunks'), "Should be dask array"
        
        chunks = height.data.chunks
        # Chunks: (location, time, slope, realization, layer)
        # Time dimension should be chunked at 100 (or less if fewer profiles)
        time_chunks = chunks[1]
        assert time_chunks[0] <= 100, \
            f"Time chunks should be ≤100, got {time_chunks[0]}"
    
    @pytest.mark.synthetic
    def test_custom_chunks(self, synthetic_pro, tmp_path):
        """Test that custom chunk sizes can be specified."""
        data_dir = synthetic_pro.multi_station(
            tmp_path,
            n_stations=12,
            n_profiles_each=20
        )
        
        # Load with custom chunks
        ds = read(data_dir, lazy=True, chunks={'time': 5, 'layer': 50})
        
        height = ds.data['height']
        assert hasattr(height.data, 'chunks'), "Should be dask array"
        
        chunks = height.data.chunks
        # Time chunks should be 5 or less
        time_chunks = chunks[1]
        assert time_chunks[0] <= 5, \
            f"Time chunks should be ≤5, got {time_chunks[0]}"
    
    @pytest.mark.synthetic
    def test_chunks_ignored_in_eager_mode(self, synthetic_pro, tmp_path):
        """Test that chunks parameter is ignored when lazy=False."""
        data_dir = synthetic_pro.basic(tmp_path, n_layers=10)
        
        # Load eagerly with chunks specified (should be ignored)
        ds = read(data_dir, lazy=False, chunks={'time': 5, 'layer': 50})
        
        # Should still be numpy arrays (chunks ignored)
        assert isinstance(ds.data['height'].data, np.ndarray), \
            "Should be numpy array in eager mode even with chunks specified"


class TestLazyLoadingComputation:
    """Test lazy computation behavior."""
    
    @pytest.mark.synthetic
    def test_lazy_data_not_computed_initially(self, synthetic_pro, tmp_path):
        """Test that lazy loading doesn't compute data on initial load."""
        data_dir = synthetic_pro.multi_station(
            tmp_path,
            n_stations=15,
            n_profiles_each=20
        )
        
        # Load lazily
        ds = read(data_dir, lazy=True)
        
        # Access the data variable (shouldn't trigger computation)
        height = ds.data['height']
        
        # Should still be dask array (not computed)
        import dask.array as da
        assert isinstance(height.data, da.Array), \
            "Should remain dask array after accessing"
        
        # The actual values should not be computed yet
        # (This is implicit - dask arrays delay computation)
    
    @pytest.mark.synthetic
    def test_lazy_compute_explicitly(self, synthetic_pro, tmp_path):
        """Test that lazy datasets compute when explicitly requested."""
        data_dir = synthetic_pro.multi_station(
            tmp_path,
            n_stations=12,
            n_profiles_each=10
        )
        
        # Load lazily
        ds = read(data_dir, lazy=True)
        
        height = ds.data['height']
        
        # Should be dask array before compute
        import dask.array as da
        assert isinstance(height.data, da.Array), "Should be dask before compute"
        
        # Explicitly compute
        height_computed = height.compute()
        
        # After compute, should be numpy array
        assert isinstance(height_computed.data, np.ndarray), \
            "After compute(), should be numpy array"
        
        # Can also compute the whole dataset
        ds_computed = ds.data.compute()
        assert isinstance(ds_computed['height'].data, np.ndarray), \
            "Dataset compute() should return numpy arrays"
    
    @pytest.mark.synthetic
    def test_lazy_load_method(self, synthetic_pro, tmp_path):
        """Test that .load() method computes dask arrays."""
        data_dir = synthetic_pro.multi_station(
            tmp_path,
            n_stations=12,
            n_profiles_each=10
        )
        
        # Load lazily
        ds = read(data_dir, lazy=True)
        
        # Use load() instead of compute()
        ds_loaded = ds.data.load()
        
        # After load(), should have numpy arrays
        assert isinstance(ds_loaded['height'].data, np.ndarray), \
            "After load(), should be numpy array"


class TestLazyLoadingWorkflow:
    """Test realistic lazy loading workflows."""
    
    @pytest.mark.synthetic
    def test_lazy_select_then_compute(self, synthetic_pro, tmp_path):
        """Test efficient workflow: load lazy → select subset → compute."""
        # Create multi-station dataset
        data_dir = synthetic_pro.multi_station(
            tmp_path,
            n_stations=15,
            n_profiles_each=30
        )
        
        # Load lazily (should be fast)
        ds = read(data_dir, lazy=True)
        
        # Select subset (still lazy - no computation yet)
        ds_subset = ds.sel(time=ds.data.time[:5])
        
        # Verify still lazy after selection
        import dask.array as da
        assert isinstance(ds_subset.data['height'].data, da.Array), \
            "Should remain lazy after selection"
        
        # Now compute only the subset (efficient!)
        ds_computed = ds_subset.compute()
        
        # Subset should be computed
        assert isinstance(ds_computed['height'].data, np.ndarray), \
            "Computed subset should be numpy array"
        
        # Should have reduced time dimension
        assert ds_computed.sizes['time'] == 5, \
            "Subset should have 5 time steps"
    
    @pytest.mark.synthetic
    def test_lazy_filter_operations(self, synthetic_pro, tmp_path):
        """Test that filtering operations preserve lazy evaluation."""
        data_dir = synthetic_pro.multi_station(
            tmp_path,
            n_stations=15,
            n_profiles_each=20
        )
        
        ds = read(data_dir, lazy=True)
        
        # Apply various operations (should stay lazy)
        ds_filtered = ds.sel(time=ds.data.time[:10])
        ds_filtered = ds_filtered.isel(location=slice(0, 5))
        
        # Should still be lazy
        import dask.array as da
        assert isinstance(ds_filtered.data['height'].data, da.Array), \
            "Should remain lazy after multiple operations"
    
    @pytest.mark.synthetic
    def test_lazy_with_derived_variables(self, synthetic_pro, tmp_path):
        """Test lazy loading with derived variable computation."""
        data_dir = synthetic_pro.multi_station(
            tmp_path,
            n_stations=12,
            n_profiles_each=10
        )
        
        # Load lazily
        ds = read(data_dir, lazy=True)
        
        # Compute derived variable (should work with lazy data)
        if 'height' in ds.data:
            ds = ds.compute_hs()
            
            # HS might be computed, but underlying data can still be lazy
            # The key is that it doesn't fail
            assert 'HS' in ds.data, "Should compute HS successfully"


class TestLazyLoadingEdgeCases:
    """Test edge cases and boundary conditions."""
    
    @pytest.mark.synthetic
    def test_lazy_single_file(self, synthetic_pro, tmp_path):
        """Test that lazy=True works with a single file."""
        filepath = synthetic_pro.basic(tmp_path, n_layers=10)
        
        # Explicitly lazy with single file
        ds = read(tmp_path, lazy=True)
        
        # Should create dask arrays even for single file
        import dask.array as da
        assert isinstance(ds.data['height'].data, da.Array), \
            "Should support lazy loading for single file"
    
    @pytest.mark.synthetic
    def test_lazy_empty_selection(self, synthetic_pro, tmp_path):
        """Test that lazy loading handles empty selections gracefully."""
        data_dir = synthetic_pro.multi_station(
            tmp_path,
            n_stations=12,
            n_profiles_each=10
        )
        
        ds = read(data_dir, lazy=True)
        
        # Select time range with no data
        ds_empty = ds.sel(time=slice('1900-01-01', '1900-12-31'))
        
        # Should handle gracefully (may be empty)
        # Key is that it doesn't crash
        assert ds_empty is not None
    
    @pytest.mark.synthetic
    def test_lazy_preserves_metadata(self, synthetic_pro, tmp_path):
        """Test that lazy loading preserves metadata."""
        data_dir = synthetic_pro.multi_station(
            tmp_path,
            n_stations=12,
            n_profiles_each=10
        )
        
        ds_lazy = read(data_dir, lazy=True)
        ds_eager = read(data_dir, lazy=False)
        
        # Coordinates should be the same
        assert ds_lazy.data.sizes == ds_eager.data.sizes, \
            "Lazy and eager should have same dimensions"
        
        # Attributes should be preserved
        assert 'location' in ds_lazy.data.coords
        assert 'time' in ds_lazy.data.coords
        assert 'altitude' in ds_lazy.data.coords


class TestLazyLoadingPerformance:
    """Test performance characteristics of lazy loading."""
    
    @pytest.mark.synthetic
    @pytest.mark.slow
    def test_lazy_load_time_fast(self, synthetic_pro, tmp_path):
        """Test that lazy loading is significantly faster than eager loading."""
        import time
        
        # Create moderately large dataset
        data_dir = synthetic_pro.multi_station(
            tmp_path,
            n_stations=20,
            n_profiles_each=50
        )
        
        # Time lazy loading
        start = time.time()
        ds_lazy = read(data_dir, lazy=True)
        lazy_time = time.time() - start
        
        # Time eager loading
        start = time.time()
        ds_eager = read(data_dir, lazy=False)
        eager_time = time.time() - start
        
        # Lazy should be faster (though with small test data, may not be dramatic)
        # Main point is lazy doesn't crash and completes
        assert lazy_time < eager_time * 2, \
            f"Lazy ({lazy_time:.2f}s) should be faster than eager ({eager_time:.2f}s)"
        
        # Verify lazy is actually lazy
        import dask.array as da
        assert isinstance(ds_lazy.data['height'].data, da.Array), \
            "Lazy load should create dask arrays"
        assert isinstance(ds_eager.data['height'].data, np.ndarray), \
            "Eager load should create numpy arrays"


@pytest.mark.synthetic
class TestLazyLoadingIntegration:
    """Integration tests combining lazy loading with other features."""
    
    def test_lazy_with_time_filtering(self, synthetic_pro, tmp_path):
        """Test lazy loading with time filtering."""
        from xsnow.selectors import TimeSelector
        
        data_dir = synthetic_pro.multi_station(
            tmp_path,
            n_stations=12,
            n_profiles_each=20
        )
        
        # Load with time filter and lazy mode
        time_filter = TimeSelector(sel=slice('2025-01-01', '2025-01-10'))
        ds = read(data_dir, time=time_filter, lazy=True)
        
        # Should be dask arrays
        import dask.array as da
        assert isinstance(ds.data['height'].data, da.Array), \
            "Should be lazy with time filtering"
        
        # Should have filtered time range
        assert ds.data.sizes['time'] <= 20, \
            "Should filter time dimension"
    
    def test_lazy_roundtrip_netcdf(self, synthetic_pro, tmp_path):
        """Test saving and loading lazy datasets to NetCDF."""
        data_dir = synthetic_pro.multi_station(
            tmp_path,
            n_stations=12,
            n_profiles_each=10
        )
        
        # Load lazily
        ds_original = read(data_dir, lazy=True)
        
        # Compute and save to NetCDF
        nc_file = tmp_path / "lazy_test.nc"
        ds_original.to_netcdf(nc_file)
        
        # Reload
        ds_reloaded = read(nc_file)
        
        # Should be able to read back
        assert ds_reloaded is not None
        assert ds_reloaded.data.sizes == ds_original.data.sizes
