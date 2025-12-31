#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test suite for Zarr reading/writing with xsnow

Tests:
- Writing Zarr files with proper encoding
- Reading Zarr files (eager and lazy)
- Data integrity verification
- Adaptive chunking
- Fill values
- Coordinate preprocessing
- Round-trip consistency
"""

import pytest
import xarray as xr
import xsnow
from xsnow.sample_data import snp_gridded_dir
import shutil
from pathlib import Path
import numpy as np
import tempfile


class ZarrTestHelper:
    """Helper class with Zarr encoding and preprocessing methods."""
    
    @staticmethod
    def get_zarr_encoding(ds):
        """
        Get Zarr encoding with fill values for integer types.
        
        Uses SNOWPACK standard (-999) when possible, dtype-specific otherwise.
        """
        FILL_VALUES = {
            'int8': -128,
            'int16': -999,
            'int32': -999,
            'int64': -999,
            'uint8': 255,
            'uint16': 65535,
            'uint32': 4294967295,
            'uint64': 18446744073709551615,
        }
        
        encoding = {}
        for var in ds.data_vars:
            dtype_str = str(ds[var].dtype)
            if 'int' in dtype_str:
                fill_value = FILL_VALUES.get(dtype_str, -999)
                encoding[var] = {'_FillValue': fill_value}
        
        return encoding
    
    @staticmethod
    def get_optimal_chunks(ds):
        """
        Get optimal chunk sizes based on dataset dimensions.
        
        Only chunks dimensions that exist.
        """
        desired_chunks = {
            'location': 200,
            'time': 100,
            'layer': -1,
            'slope': -1,
            'realization': -1
        }
        
        chunks = {}
        for dim, chunk_size in desired_chunks.items():
            if dim in ds.dims:
                chunks[dim] = chunk_size
        
        return chunks
    
    @staticmethod
    def preprocess_for_merge(ds):
        """
        Preprocess datasets for merging.
        
        Converts slope coordinates and drops derived coordinates.
        """
        # Convert slope coordinates to data variables
        slope_coords = ['azimuth', 'inclination', 'aspect']
        
        for coord in slope_coords:
            if coord in ds.coords:
                ds = ds.reset_coords(coord)
                
                # Broadcast 1D to 2D if needed
                coord_var = ds[coord]
                if coord_var.dims == ('slope',) and 'location' in ds.dims and 'slope' in ds.dims:
                    template = None
                    for var_name in ds.data_vars:
                        if var_name == coord:
                            continue
                        var = ds[var_name]
                        if 'location' in var.dims and 'slope' in var.dims:
                            template = var
                            for dim in var.dims:
                                if dim not in ['location', 'slope']:
                                    template = template.isel({dim: 0})
                            break
                    
                    if template is not None:
                        ds[coord] = ds[coord].broadcast_like(template)
        
        # Drop derived and non-essential coordinates
        essential_coords = [
            'location', 'time', 'layer', 'slope', 'realization',
            'latitude', 'longitude', 'altitude', 'elevation',
            'lat', 'lon'
        ]
        
        coords_to_drop = [c for c in ds.coords if c not in essential_coords]
        if coords_to_drop:
            ds = ds.drop_vars(coords_to_drop, errors='ignore')
        
        return ds


@pytest.fixture
def sample_data_dir():
    """Fixture providing sample data directory."""
    return snp_gridded_dir()


@pytest.fixture
def temp_zarr_dir():
    """Fixture providing temporary directory for Zarr files."""
    temp_dir = tempfile.mkdtemp(prefix='zarr_test_')
    yield Path(temp_dir)
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def helper():
    """Fixture providing ZarrTestHelper instance."""
    return ZarrTestHelper()


# =============================================================================
# BASIC READING TESTS
# =============================================================================

def test_read_sample_data_eager(sample_data_dir):
    """Test reading sample data with eager loading."""
    print(f"\n{'='*70}")
    print("TEST: Read Sample Data (Eager)")
    print(f"{'='*70}")
    
    # Read with eager loading
    ds = xsnow.read(sample_data_dir, lazy=False)
    
    assert ds is not None, "Dataset is None"
    assert ds.data is not None, "Dataset.data is None"
    
    data = ds.data
    
    # Check dimensions exist
    assert 'location' in data.dims, "Missing location dimension"
    assert 'time' in data.dims, "Missing time dimension"
    
    # Check some data
    print(f"✅ Loaded eagerly")
    print(f"  Dimensions: {dict(data.dims)}")
    print(f"  Variables: {len(data.data_vars)}")
    print(f"  Coordinates: {list(data.coords)}")


def test_read_sample_data_lazy(sample_data_dir):
    """Test reading sample data with lazy loading."""
    print(f"\n{'='*70}")
    print("TEST: Read Sample Data (Lazy)")
    print(f"{'='*70}")
    
    # Read with lazy loading
    ds = xsnow.read(sample_data_dir, lazy=True)
    
    assert ds is not None, "Dataset is None"
    assert ds.data is not None, "Dataset.data is None"
    
    data = ds.data
    
    # Check dimensions exist
    assert 'location' in data.dims, "Missing location dimension"
    assert 'time' in data.dims, "Missing time dimension"
    
    # Verify data is lazy (dask arrays)
    for var in data.data_vars:
        if hasattr(data[var].data, 'chunks'):
            print(f"✅ Variable '{var}' is lazy (has chunks)")
            break
    
    print(f"  Dimensions: {dict(data.dims)}")
    print(f"  Variables: {len(data.data_vars)}")


# =============================================================================
# BASIC WRITING TESTS
# =============================================================================

def test_write_zarr_basic(sample_data_dir, temp_zarr_dir, helper):
    """Test basic Zarr writing with encoding."""
    print(f"\n{'='*70}")
    print("TEST: Write Zarr (Basic)")
    print(f"{'='*70}")
    
    # Read sample data
    ds = xsnow.read(sample_data_dir, lazy=False)
    data = ds.data
    
    # Prepare for save
    data_preprocessed = helper.preprocess_for_merge(data)
    chunks = helper.get_optimal_chunks(data_preprocessed)
    data_chunked = data_preprocessed.chunk(chunks)
    
    # Get encoding
    encoding = helper.get_zarr_encoding(data_chunked)
    
    # Save to Zarr
    output_path = temp_zarr_dir / 'test_basic.zarr'
    print(f"  Saving to: {output_path}")
    
    if encoding:
        data_chunked.to_zarr(output_path, mode='w', encoding=encoding)
    else:
        data_chunked.to_zarr(output_path, mode='w')
    
    assert output_path.exists(), "Zarr directory not created"
    assert (output_path / 'zarr.json').exists(), "Zarr metadata missing"
    
    print(f"✅ Saved successfully")
    print(f"  Size: {sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file()) / 1024 / 1024:.1f} MB")


def test_write_zarr_with_int8(sample_data_dir, temp_zarr_dir, helper):
    """Test Zarr writing handles int8 variables with correct fill values."""
    print(f"\n{'='*70}")
    print("TEST: Write Zarr (int8 Fill Values)")
    print(f"{'='*70}")
    
    # Read sample data
    ds = xsnow.read(sample_data_dir, lazy=False)
    data = ds.data
    
    # Add a test int8 variable
    data['test_int8'] = xr.DataArray(
        np.array([[1, 2, -128]], dtype='int8'),
        dims=['location', 'time'],
        attrs={'description': 'Test int8 variable'}
    )
    
    # Get encoding
    encoding = helper.get_zarr_encoding(data)
    
    # Check int8 fill value
    if 'test_int8' in encoding:
        fill_val = encoding['test_int8']['_FillValue']
        print(f"  int8 fill value: {fill_val}")
        assert fill_val == -128, f"Wrong fill value: {fill_val}, expected -128"
        print(f"✅ Correct int8 fill value (-128)")
    
    # Save to Zarr
    output_path = temp_zarr_dir / 'test_int8.zarr'
    data.to_zarr(output_path, mode='w', encoding=encoding)
    
    assert output_path.exists(), "Zarr directory not created"
    print(f"✅ Saved with int8 variables")


# =============================================================================
# READING TESTS
# =============================================================================

def test_read_zarr_eager(sample_data_dir, temp_zarr_dir, helper):
    """Test reading Zarr files with eager loading."""
    print(f"\n{'='*70}")
    print("TEST: Read Zarr (Eager)")
    print(f"{'='*70}")
    
    # Create Zarr file first
    ds = xsnow.read(sample_data_dir, lazy=False)
    data = ds.data
    data_preprocessed = helper.preprocess_for_merge(data)
    chunks = helper.get_optimal_chunks(data_preprocessed)
    data_chunked = data_preprocessed.chunk(chunks)
    
    output_path = temp_zarr_dir / 'test_read_eager.zarr'
    encoding = helper.get_zarr_encoding(data_chunked)
    
    if encoding:
        data_chunked.to_zarr(output_path, mode='w', encoding=encoding)
    else:
        data_chunked.to_zarr(output_path, mode='w')
    
    # Read back (eager)
    print(f"  Reading from: {output_path}")
    ds_read = xr.open_zarr(output_path)
    
    # Force computation
    ds_read = ds_read.compute()
    
    assert ds_read is not None, "Failed to read Zarr"
    assert len(ds_read.dims) > 0, "No dimensions"
    
    print(f"✅ Read successfully (eager)")
    print(f"  Dimensions: {dict(ds_read.dims)}")
    print(f"  Variables: {len(ds_read.data_vars)}")


def test_read_zarr_lazy(sample_data_dir, temp_zarr_dir, helper):
    """Test reading Zarr files with lazy loading."""
    print(f"\n{'='*70}")
    print("TEST: Read Zarr (Lazy)")
    print(f"{'='*70}")
    
    # Create Zarr file first
    ds = xsnow.read(sample_data_dir, lazy=False)
    data = ds.data
    data_preprocessed = helper.preprocess_for_merge(data)
    chunks = helper.get_optimal_chunks(data_preprocessed)
    data_chunked = data_preprocessed.chunk(chunks)
    
    output_path = temp_zarr_dir / 'test_read_lazy.zarr'
    encoding = helper.get_zarr_encoding(data_chunked)
    
    if encoding:
        data_chunked.to_zarr(output_path, mode='w', encoding=encoding)
    else:
        data_chunked.to_zarr(output_path, mode='w')
    
    # Read back (lazy)
    print(f"  Reading from: {output_path}")
    ds_read = xr.open_zarr(output_path)
    
    # Check it's lazy (has chunks)
    has_chunks = False
    for var in ds_read.data_vars:
        if hasattr(ds_read[var].data, 'chunks'):
            has_chunks = True
            print(f"✅ Variable '{var}' is lazy")
            break
    
    assert has_chunks, "Data not lazy loaded"
    print(f"  Dimensions: {dict(ds_read.dims)}")


# =============================================================================
# ROUND-TRIP TESTS
# =============================================================================

def test_roundtrip_eager(sample_data_dir, temp_zarr_dir, helper):
    """Test write-read round-trip with eager loading."""
    print(f"\n{'='*70}")
    print("TEST: Round-trip (Eager)")
    print(f"{'='*70}")
    
    # Read original
    ds_orig = xsnow.read(sample_data_dir, lazy=False)
    data_orig = ds_orig.data
    
    # Save to Zarr
    data_preprocessed = helper.preprocess_for_merge(data_orig)
    chunks = helper.get_optimal_chunks(data_preprocessed)
    data_chunked = data_preprocessed.chunk(chunks)
    
    output_path = temp_zarr_dir / 'test_roundtrip_eager.zarr'
    encoding = helper.get_zarr_encoding(data_chunked)
    
    if encoding:
        data_chunked.to_zarr(output_path, mode='w', encoding=encoding)
    else:
        data_chunked.to_zarr(output_path, mode='w')
    
    # Read back
    ds_read = xr.open_zarr(output_path)
    ds_read = ds_read.compute()
    
    # Compare dimensions
    assert set(ds_read.dims) == set(data_preprocessed.dims), "Dimensions mismatch"
    print(f"✅ Dimensions preserved: {list(ds_read.dims)}")
    
    # Compare variables (preprocessed data may have fewer)
    common_vars = set(ds_read.data_vars) & set(data_preprocessed.data_vars)
    print(f"✅ Common variables: {len(common_vars)}")
    
    # Check a key variable if present
    if 'HS' in common_vars:
        orig_hs = data_preprocessed['HS'].values
        read_hs = ds_read['HS'].values
        
        # Allow for NaN differences
        orig_valid = ~np.isnan(orig_hs)
        read_valid = ~np.isnan(read_hs)
        
        if orig_valid.any():
            np.testing.assert_allclose(
                orig_hs[orig_valid],
                read_hs[orig_valid],
                rtol=1e-5,
                err_msg="HS values don't match"
            )
            print(f"✅ HS values match (within tolerance)")


def test_roundtrip_lazy(sample_data_dir, temp_zarr_dir, helper):
    """Test write-read round-trip with lazy loading."""
    print(f"\n{'='*70}")
    print("TEST: Round-trip (Lazy)")
    print(f"{'='*70}")
    
    # Read original (lazy)
    ds_orig = xsnow.read(sample_data_dir, lazy=True)
    data_orig = ds_orig.data
    
    # Save to Zarr
    data_preprocessed = helper.preprocess_for_merge(data_orig)
    chunks = helper.get_optimal_chunks(data_preprocessed)
    data_chunked = data_preprocessed.chunk(chunks)
    
    output_path = temp_zarr_dir / 'test_roundtrip_lazy.zarr'
    encoding = helper.get_zarr_encoding(data_chunked)
    
    if encoding:
        data_chunked.to_zarr(output_path, mode='w', encoding=encoding)
    else:
        data_chunked.to_zarr(output_path, mode='w')
    
    # Read back (lazy)
    ds_read = xr.open_zarr(output_path)
    
    # Verify lazy
    has_chunks = any(hasattr(ds_read[var].data, 'chunks') for var in ds_read.data_vars)
    assert has_chunks, "Data not lazy after reading"
    print(f"✅ Lazy loading preserved")
    
    # Compare dimensions
    assert set(ds_read.dims) == set(data_preprocessed.dims), "Dimensions mismatch"
    print(f"✅ Dimensions preserved: {list(ds_read.dims)}")


# =============================================================================
# CHUNKING TESTS
# =============================================================================

def test_adaptive_chunking_profile_data(sample_data_dir, helper):
    """Test adaptive chunking with profile data (.pro files)."""
    print(f"\n{'='*70}")
    print("TEST: Adaptive Chunking (Profile Data)")
    print(f"{'='*70}")
    
    # Read sample data (likely has layer dimension)
    ds = xsnow.read(sample_data_dir, lazy=False)
    data = ds.data
    
    # Get chunks
    chunks = helper.get_optimal_chunks(data)
    
    print(f"  Dataset dimensions: {list(data.dims)}")
    print(f"  Chunk specification: {chunks}")
    
    # Verify only existing dimensions are chunked
    for dim in chunks:
        assert dim in data.dims, f"Chunk dimension '{dim}' not in dataset"
    
    # Apply chunks
    data_chunked = data.chunk(chunks)
    
    print(f"✅ Adaptive chunking works")
    print(f"  Chunked dimensions: {list(chunks.keys())}")


def test_adaptive_chunking_timeseries_data(temp_zarr_dir, helper):
    """Test adaptive chunking with time-series data (no layer)."""
    print(f"\n{'='*70}")
    print("TEST: Adaptive Chunking (Time-series Data)")
    print(f"{'='*70}")
    
    # Create synthetic time-series data (no layer dimension)
    data = xr.Dataset({
        'temperature': (['location', 'time', 'slope'], np.random.rand(10, 100, 3)),
        'HS': (['location', 'time', 'slope'], np.random.rand(10, 100, 3))
    })
    
    print(f"  Dataset dimensions: {list(data.dims)}")
    print(f"  No 'layer' dimension (simulates .smet files)")
    
    # Get chunks
    chunks = helper.get_optimal_chunks(data)
    
    print(f"  Chunk specification: {chunks}")
    
    # Verify 'layer' not in chunks (doesn't exist)
    assert 'layer' not in chunks, "'layer' should not be in chunks"
    
    # Apply chunks (should work without error)
    data_chunked = data.chunk(chunks)
    
    print(f"✅ Adaptive chunking works without 'layer'")
    print(f"  Chunked dimensions: {list(chunks.keys())}")


# =============================================================================
# PREPROCESSING TESTS
# =============================================================================

def test_preprocess_slope_coordinates(sample_data_dir, helper):
    """Test preprocessing converts slope coordinates."""
    print(f"\n{'='*70}")
    print("TEST: Preprocess Slope Coordinates")
    print(f"{'='*70}")
    
    # Read sample data
    ds = xsnow.read(sample_data_dir, lazy=False)
    data = ds.data
    
    print(f"  Original coordinates: {list(data.coords)}")
    
    # Preprocess
    data_preprocessed = helper.preprocess_for_merge(data)
    
    print(f"  After preprocessing: {list(data_preprocessed.coords)}")
    
    # Check that slope coords moved to data vars
    slope_coords = ['azimuth', 'inclination', 'aspect']
    for coord in slope_coords:
        if coord in data.coords:
            # Should now be data variable
            assert coord in data_preprocessed.data_vars or coord not in data_preprocessed.coords, \
                f"{coord} still a coordinate after preprocessing"
            print(f"✅ '{coord}' converted from coordinate")


def test_preprocess_drops_derived_coords(temp_zarr_dir, helper):
    """Test preprocessing drops derived coordinates like z."""
    print(f"\n{'='*70}")
    print("TEST: Preprocess Drops Derived Coordinates")
    print(f"{'='*70}")
    
    # Create data with derived coordinates
    data = xr.Dataset({
        'temperature': (['location', 'time'], np.random.rand(10, 100))
    })
    
    # Add derived coordinates (like z, x, y)
    data = data.assign_coords({
        'z': ('location', np.arange(10)),
        'x': ('location', np.random.rand(10)),
        'y': ('location', np.random.rand(10)),
        'latitude': ('location', np.random.rand(10)),  # Essential
        'longitude': ('location', np.random.rand(10))  # Essential
    })
    
    print(f"  Before preprocessing:")
    print(f"    Coordinates: {list(data.coords)}")
    
    # Preprocess
    data_preprocessed = helper.preprocess_for_merge(data)
    
    print(f"  After preprocessing:")
    print(f"    Coordinates: {list(data_preprocessed.coords)}")
    
    # Check derived coords dropped
    assert 'z' not in data_preprocessed.coords, "'z' not dropped"
    assert 'x' not in data_preprocessed.coords, "'x' not dropped"
    assert 'y' not in data_preprocessed.coords, "'y' not dropped"
    
    # Check essential coords kept
    assert 'latitude' in data_preprocessed.coords, "'latitude' was dropped"
    assert 'longitude' in data_preprocessed.coords, "'longitude' was dropped"
    
    print(f"✅ Derived coordinates dropped")
    print(f"✅ Essential coordinates preserved")


# =============================================================================
# PERFORMANCE COMPARISON
# =============================================================================

def test_compare_eager_vs_lazy_write(sample_data_dir, temp_zarr_dir, helper):
    """Compare eager vs lazy loading for writing."""
    print(f"\n{'='*70}")
    print("TEST: Compare Eager vs Lazy (Write)")
    print(f"{'='*70}")
    
    import time
    
    # Test eager
    print("  Testing EAGER loading...")
    start = time.time()
    ds_eager = xsnow.read(sample_data_dir, lazy=False)
    data_eager = ds_eager.data
    data_preprocessed = helper.preprocess_for_merge(data_eager)
    chunks = helper.get_optimal_chunks(data_preprocessed)
    data_chunked = data_preprocessed.chunk(chunks)
    encoding = helper.get_zarr_encoding(data_chunked)
    
    output_eager = temp_zarr_dir / 'test_eager.zarr'
    if encoding:
        data_chunked.to_zarr(output_eager, mode='w', encoding=encoding)
    else:
        data_chunked.to_zarr(output_eager, mode='w')
    
    time_eager = time.time() - start
    print(f"    Eager time: {time_eager:.2f}s")
    
    # Test lazy
    print("  Testing LAZY loading...")
    start = time.time()
    ds_lazy = xsnow.read(sample_data_dir, lazy=True)
    data_lazy = ds_lazy.data
    data_preprocessed = helper.preprocess_for_merge(data_lazy)
    chunks = helper.get_optimal_chunks(data_preprocessed)
    data_chunked = data_preprocessed.chunk(chunks)
    encoding = helper.get_zarr_encoding(data_chunked)
    
    output_lazy = temp_zarr_dir / 'test_lazy.zarr'
    if encoding:
        data_chunked.to_zarr(output_lazy, mode='w', encoding=encoding)
    else:
        data_chunked.to_zarr(output_lazy, mode='w')
    
    time_lazy = time.time() - start
    print(f"    Lazy time: {time_lazy:.2f}s")
    
    print(f"\n✅ Both methods completed")
    print(f"  Eager: {time_eager:.2f}s")
    print(f"  Lazy:  {time_lazy:.2f}s")
    print(f"  Difference: {abs(time_eager - time_lazy):.2f}s")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    """Run all tests."""
    print(f"\n{'='*70}")
    print("ZARR TEST SUITE")
    print(f"{'='*70}")
    print(f"Sample data: {snp_gridded_dir()}")
    print()
    
    # Run with pytest
    pytest.main([__file__, '-v', '-s'])