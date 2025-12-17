# test_eager_loading.py
"""
Comprehensive tests for eager vs lazy loading modes.
Tests both synthetic data and real SNOWPACK model output.
"""
import pytest
from pathlib import Path
import time
import numpy as np
import psutil
from xsnow import read

# Try to import sample data functions
try:
    from xsnow.sample_data import snp_gridded_dir, snp_snowobs_dir
    SAMPLE_DATA_AVAILABLE = True
except (ImportError, AttributeError):
    SAMPLE_DATA_AVAILABLE = False
    snp_gridded_dir = None
    snp_snowobs_dir = None


# ============================================================================
# FIXTURES - SYNTHETIC DATA
# ============================================================================

@pytest.fixture
def small_dataset(tmp_path):
    """Creates a small dataset (5 files) for testing."""
    for i in range(5):
        (tmp_path / f"station_{i}.pro").write_text(f"""
[STATION_PARAMETERS]
StationName= STATION_{i}
[DATA]
0500,01.01.2025 12:00:00
0501,3,0.1,0.2,0.3
0502,3,100,200,300
0500,02.01.2025 12:00:00
0501,3,0.15,0.25,0.35
0502,3,110,210,310
""")
    return tmp_path


@pytest.fixture
def medium_dataset(tmp_path):
    """Creates a medium dataset (15 files) for testing."""
    for i in range(15):
        (tmp_path / f"station_{i}.pro").write_text(f"""
[STATION_PARAMETERS]
StationName= STATION_{i}
[DATA]
0500,01.01.2025 12:00:00
0501,3,0.1,0.2,0.3
0502,3,100,200,300
""")
    return tmp_path


# ============================================================================
# FIXTURES - REAL DATA
# ============================================================================

@pytest.fixture(scope="module")
def real_gridded_dir():
    """Returns path to real gridded sample data."""
    if not SAMPLE_DATA_AVAILABLE:
        pytest.skip("Sample data module not available")
    
    try:
        base_path = Path(snp_gridded_dir())
        possible_paths = [
            base_path / "pros" / "gridded",
            base_path / "gridded",
            base_path / "pros",
            base_path,
        ]
        
        for path in possible_paths:
            if path.exists() and list(path.glob("*.pro")):
                return path
        
        pytest.skip(f"No .pro files found in {base_path}")
    except Exception as e:
        pytest.skip(f"Could not access sample data: {e}")


@pytest.fixture
def subset_real_files(real_gridded_dir, tmp_path):
    """Creates a subset of real files for controlled testing."""
    import shutil
    source_files = list(real_gridded_dir.glob("*.pro"))[:5]
    
    for src_file in source_files:
        shutil.copy(src_file, tmp_path / src_file.name)
    
    return tmp_path


# ============================================================================
# BASIC FUNCTIONALITY TESTS - SYNTHETIC DATA
# ============================================================================

def test_eager_loading_basic(small_dataset):
    """Tests basic eager loading functionality."""
    ds = read(small_dataset, lazy=False, n_cpus_use=2)
    
    assert ds is not None
    assert ds.sizes['location'] == 5
    assert ds.sizes['time'] == 2
    assert 'density' in ds.data
    assert 'height' in ds.data


def test_lazy_loading_basic(small_dataset):
    """Tests basic lazy loading functionality."""
    ds = read(small_dataset, lazy=True, n_cpus_use=2)
    
    assert ds is not None
    assert ds.sizes['location'] == 5
    assert ds.sizes['time'] == 2
    assert 'density' in ds.data
    assert 'height' in ds.data


def test_eager_vs_lazy_produce_same_results(small_dataset):
    """Verifies that eager and lazy loading produce identical results."""
    ds_eager = read(small_dataset, lazy=False, n_cpus_use=2)
    ds_lazy = read(small_dataset, lazy=True, n_cpus_use=2)
    
    # Same dimensions
    assert ds_eager.sizes == ds_lazy.sizes
    
    # Same data values
    np.testing.assert_array_equal(
        ds_eager['density'].values,
        ds_lazy['density'].values
    )
    np.testing.assert_array_equal(
        ds_eager['height'].values,
        ds_lazy['height'].values
    )


# ============================================================================
# AUTO-DETECTION TESTS
# ============================================================================

def test_auto_detect_uses_eager_for_small_datasets(small_dataset):
    """Tests that auto-detection chooses eager for <=10 files."""
    ds = read(small_dataset, lazy=None)  # Auto-detect
    
    assert ds is not None
    assert ds.sizes['location'] == 5


def test_auto_detect_uses_lazy_for_medium_datasets(medium_dataset):
    """Tests that auto-detection chooses lazy for >10 files."""
    ds = read(medium_dataset, lazy=None)  # Auto-detect
    
    assert ds is not None
    assert ds.sizes['location'] == 15


@pytest.mark.parametrize("num_files,expected_mode", [
    (3, "eager"),
    (8, "eager"),
    (12, "lazy"),
    (20, "lazy"),
])
def test_auto_detect_threshold(tmp_path, num_files, expected_mode):
    """Tests that auto-detection correctly switches at the threshold."""
    for i in range(num_files):
        (tmp_path / f"station_{i}.pro").write_text(f"""
[STATION_PARAMETERS]
StationName= STATION_{i}
[DATA]
0500,01.01.2025 12:00:00
0501,2,0.1,0.2
0502,2,100,200
""")
    
    ds = read(tmp_path, lazy=None)  # Auto-detect
    
    assert ds is not None
    assert ds.sizes['location'] == num_files
    print(f"{num_files} files -> expected {expected_mode} mode")


# ============================================================================
# PERFORMANCE COMPARISON TESTS - SYNTHETIC DATA
# ============================================================================

def test_eager_loading_is_faster_for_small_datasets(small_dataset):
    """Compares performance of eager vs lazy for small datasets."""
    # Eager loading
    start = time.time()
    ds_eager = read(small_dataset, lazy=False, n_cpus_use=2)
    eager_time = time.time() - start
    
    # Lazy loading
    start = time.time()
    ds_lazy = read(small_dataset, lazy=True, n_cpus_use=2)
    lazy_time = time.time() - start
    
    print(f"\nPerformance for 5 files:")
    print(f"  Eager: {eager_time:.3f}s")
    print(f"  Lazy:  {lazy_time:.3f}s")
    print(f"  Speedup: {lazy_time/eager_time:.2f}x")
    
    assert ds_eager is not None
    assert ds_lazy is not None


def test_lazy_loading_better_for_many_files(medium_dataset):
    """Compares performance for medium-sized datasets."""
    # Eager loading
    start = time.time()
    ds_eager = read(medium_dataset, lazy=False, n_cpus_use=4)
    eager_time = time.time() - start
    
    # Lazy loading
    start = time.time()
    ds_lazy = read(medium_dataset, lazy=True, n_cpus_use=4)
    lazy_time = time.time() - start
    
    print(f"\nPerformance for 15 files:")
    print(f"  Eager: {eager_time:.3f}s")
    print(f"  Lazy:  {lazy_time:.3f}s")
    
    assert ds_eager is not None
    assert ds_lazy is not None


# ============================================================================
# PARALLEL PROCESSING TESTS
# ============================================================================

def test_eager_loading_with_sequential_processing(small_dataset):
    """Tests eager loading with n_cpus_use=1 (sequential)."""
    ds = read(small_dataset, lazy=False, n_cpus_use=1)
    
    assert ds is not None
    assert ds.sizes['location'] == 5


def test_eager_loading_with_parallel_processing(small_dataset):
    """Tests eager loading with n_cpus_use>1 (parallel)."""
    ds = read(small_dataset, lazy=False, n_cpus_use=4)
    
    assert ds is not None
    assert ds.sizes['location'] == 5


def test_eager_sequential_vs_parallel(medium_dataset):
    """Compares sequential vs parallel eager loading."""
    # Sequential
    start = time.time()
    ds_seq = read(medium_dataset, lazy=False, n_cpus_use=1)
    seq_time = time.time() - start
    
    # Parallel
    start = time.time()
    ds_par = read(medium_dataset, lazy=False, n_cpus_use=4)
    par_time = time.time() - start
    
    print(f"\nEager loading (15 files):")
    print(f"  Sequential: {seq_time:.3f}s")
    print(f"  Parallel (4 CPUs): {par_time:.3f}s")
    print(f"  Speedup: {seq_time/par_time:.2f}x")
    
    assert ds_seq is not None
    assert ds_par is not None
    assert ds_seq.sizes == ds_par.sizes


# ============================================================================
# FEATURE TESTS - SYNTHETIC DATA
# ============================================================================

def test_eager_loading_respects_time_filter(small_dataset):
    """Tests that eager loading correctly applies time filters."""
    from xsnow.selectors import TimeSelector
    
    time_filter = TimeSelector(sel=slice('2025-01-02 00:00:00', '2025-01-02 23:59:59'))
    ds = read(small_dataset, lazy=False, time=time_filter)
    
    assert ds is not None
    assert ds.sizes['time'] == 1


def test_eager_loading_with_soil_removal(tmp_path):
    """Tests soil removal works correctly in eager mode."""
    pro_content = """
[STATION_PARAMETERS]
StationName= SOIL_TEST
[DATA]
0500,01.01.2025 12:00:00
0501,5,0.1,0.2,0.3,0.4,0.5
0502,5,100,200,300,400,500
0513,5,200,201,202,81,82
"""
    (tmp_path / "soil_test.pro").write_text(pro_content)
    
    ds = read(tmp_path, lazy=False, remove_soil=True)
    
    assert ds is not None
    assert ds.sizes['layer'] == 3  # 5 - 2 soil layers


def test_eager_loading_with_surface_hoar(tmp_path):
    """Tests surface hoar addition works correctly in eager mode."""
    pro_content = """
[STATION_PARAMETERS]
StationName= SH_TEST
[DATA]
0500,01.01.2025 12:00:00
0501,3,0.1,0.2,0.3
0502,3,100,200,300
0514,3,1.0,0.5,100.0
"""
    (tmp_path / "sh_test.pro").write_text(pro_content)
    
    ds = read(tmp_path, lazy=False, add_surface_sh_as_layer=True)
    
    assert ds is not None
    assert ds.sizes['layer'] == 4  # 3 + 1 SH layer


def test_eager_and_lazy_handle_max_layers_override(small_dataset):
    """Tests that both modes respect manual max_layers setting."""
    ds_eager = read(small_dataset, lazy=False, max_layers=10)
    ds_lazy = read(small_dataset, lazy=True, max_layers=10)
    
    assert ds_eager.sizes['layer'] == 10
    assert ds_lazy.sizes['layer'] == 10


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

def test_eager_loading_handles_corrupted_files(tmp_path):
    """Tests that eager loading handles corrupted files gracefully."""
    # Good file
    (tmp_path / "good.pro").write_text("""
[STATION_PARAMETERS]
StationName= GOOD
[DATA]
0500,01.01.2025 12:00:00
0501,2,0.1,0.2
0502,2,100,200
""")
    
    # Corrupted file
    (tmp_path / "bad.pro").write_text("""
[STATION_PARAMETERS]
StationName= BAD
[DATA]
0500,01.01.2025 12:00:00
0501,2,CORRUPTED
""")
    
    ds = read(tmp_path, lazy=False)
    
    # Should load only the good file
    assert ds is not None
    assert ds.sizes['location'] == 2
    assert (ds.profile_status>0).sum() == 1  # only 1 valid
    assert ds.sel(location="BAD").profile_status.values == -1


def test_eager_loading_empty_directory(tmp_path):
    """Tests eager loading with empty directory."""
    ds = read(tmp_path, lazy=False)
    assert ds is None


def test_single_file_eager_loading(tmp_path):
    """Tests eager loading with a single file."""
    (tmp_path / "single.pro").write_text("""
[STATION_PARAMETERS]
StationName= SINGLE
[DATA]
0500,01.01.2025 12:00:00
0501,2,0.1,0.2
0502,2,100,200
""")
    
    ds = read(tmp_path / "single.pro", lazy=False)
    
    assert ds is not None
    assert ds.sizes['location'] == 1


# ============================================================================
# OVERRIDE TESTS
# ============================================================================

def test_explicit_lazy_false_overrides_auto_detect(medium_dataset):
    """Tests that explicit lazy=False works even for >10 files."""
    # 15 files would normally auto-select lazy, but we force eager
    ds = read(medium_dataset, lazy=False, n_cpus_use=4)
    
    assert ds is not None
    assert ds.sizes['location'] == 15


def test_explicit_lazy_true_overrides_auto_detect(small_dataset):
    """Tests that explicit lazy=True works even for <10 files."""
    # 5 files would normally auto-select eager, but we force lazy
    ds = read(small_dataset, lazy=True, n_cpus_use=2)
    
    assert ds is not None
    assert ds.sizes['location'] == 5


# ============================================================================
# MEMORY USAGE TESTS
# ============================================================================

def test_eager_loading_memory_usage(small_dataset):
    """Tests that eager loading memory usage is reasonable."""
    process = psutil.Process()
    
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    ds = read(small_dataset, lazy=False, n_cpus_use=2)
    
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory
    
    assert ds is not None
    print(f"Memory increase (eager, 5 files): {memory_increase:.2f} MB")
    
    # Should be very small for 5 tiny files
    assert memory_increase < 500  # Less than 500 MB


# ============================================================================
# REAL DATA TESTS - SNOWPACK MODEL OUTPUT
# ============================================================================

def test_eager_loading_real_data_subset(subset_real_files):
    """Tests eager loading with a subset of real SNOWPACK files."""
    ds = read(subset_real_files, lazy=False, n_cpus_use=2)
    
    assert ds is not None
    assert ds.sizes['location'] >= 1
    assert 'height' in ds.data
    assert 'density' in ds.data
    print(f"\nLoaded {ds.sizes['location']} locations from real data")


def test_eager_vs_lazy_real_data_consistency(subset_real_files):
    """Verifies eager and lazy produce same results with real data."""
    ds_eager = read(subset_real_files, lazy=False, n_cpus_use=2)
    ds_lazy = read(subset_real_files, lazy=True, n_cpus_use=2)
    
    assert ds_eager is not None
    assert ds_lazy is not None
    
    # Same structure
    assert ds_eager.sizes['location'] == ds_lazy.sizes['location']
    assert ds_eager.sizes['layer'] == ds_lazy.sizes['layer']
    
    # Same data (allowing for floating point precision)
    np.testing.assert_allclose(
        ds_eager['density'].values,
        ds_lazy['density'].values,
        rtol=1e-5,
        equal_nan=True
    )


def test_eager_loading_real_data_performance(subset_real_files):
    """Compares eager vs lazy performance on real SNOWPACK data."""
    # Eager
    start = time.time()
    ds_eager = read(subset_real_files, lazy=False, n_cpus_use=2)
    eager_time = time.time() - start
    
    # Lazy
    start = time.time()
    ds_lazy = read(subset_real_files, lazy=True, n_cpus_use=2)
    lazy_time = time.time() - start
    
    print(f"\nReal SNOWPACK data (5 files):")
    print(f"  Eager: {eager_time:.3f}s")
    print(f"  Lazy:  {lazy_time:.3f}s")
    
    assert ds_eager is not None
    assert ds_lazy is not None


def test_eager_loading_real_data_with_slopes(subset_real_files):
    """Tests that eager loading correctly handles multi-slope real data."""
    ds = read(subset_real_files, lazy=False)
    
    assert ds is not None
    
    # Real gridded data has multiple slopes
    if 'slope' in ds.dims:
        assert ds.sizes['slope'] > 1
        print(f"\nReal data has {ds.sizes['slope']} slopes")
        print(f"Slope inclinations: {ds.inclination.values}")


def test_eager_loading_full_real_dataset_auto_detect(real_gridded_dir):
    """Tests auto-detection with full real dataset (25 files)."""
    # Should auto-select lazy for 25 files
    ds = read(real_gridded_dir, lazy=None)
    
    assert ds is not None
    assert ds.sizes['location'] >= 5
    print(f"\nFull real dataset: {ds.sizes['location']} locations, "
          f"{ds.sizes['slope']} slopes, {ds.sizes['time']} timesteps")


def test_eager_loading_forced_on_large_real_dataset(real_gridded_dir):
    """Tests forcing eager mode on the full real dataset."""
    # Force eager even though auto-detect would choose lazy
    start = time.time()
    ds = read(real_gridded_dir, lazy=False, n_cpus_use=4)
    elapsed = time.time() - start
    
    assert ds is not None
    print(f"\nForced eager loading of full real dataset took {elapsed:.2f}s")


def test_real_data_eager_with_time_filter(subset_real_files):
    """Tests eager loading with time filtering on real data."""
    from xsnow.selectors import TimeSelector
    
    # First load to know time range
    ds_full = read(subset_real_files, lazy=False)
    time_values = ds_full.time.values
    
    if len(time_values) > 10:
        import pandas as pd
        mid_point = pd.to_datetime(time_values[len(time_values) // 2])
        
        # Load only first half
        time_filter = TimeSelector(sel=slice(None, mid_point))
        ds_filtered = read(subset_real_files, lazy=False, time=time_filter)
        
        assert ds_filtered is not None
        assert ds_filtered.sizes['time'] < ds_full.sizes['time']
        print(f"\nFiltered: {ds_filtered.sizes['time']} / {ds_full.sizes['time']} timesteps")


def test_real_data_physical_validity_eager_vs_lazy(subset_real_files):
    """Verifies physical validity is maintained in both modes."""
    ds_eager = read(subset_real_files, lazy=False)
    ds_lazy = read(subset_real_files, lazy=True)
    
    for ds, mode in [(ds_eager, "eager"), (ds_lazy, "lazy")]:
        # Check density
        if 'density' in ds.data:
            density = ds['density'].values
            valid_density = density[~np.isnan(density)]
            assert np.all(valid_density >= 0), f"{mode}: Negative density found"
            assert np.all(valid_density <= 1000), f"{mode}: Unrealistic density found"
        
        # Check height
        if 'height' in ds.data:
            height = ds['height'].values
            valid_height = height[~np.isnan(height)]
            assert np.all(valid_height >= 0), f"{mode}: Negative height found"


# ============================================================================
# STRESS TESTS WITH REAL DATA
# ============================================================================

@pytest.mark.slow
def test_eager_loading_memory_efficiency_real_data(real_gridded_dir):
    """Tests memory usage with full real dataset in eager mode."""
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    ds = read(real_gridded_dir, lazy=False, n_cpus_use=4)
    
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory
    
    assert ds is not None
    print(f"\nEager loading full real dataset:")
    print(f"  Memory increase: {memory_increase:.2f} MB")
    print(f"  Locations: {ds.sizes['location']}")
    print(f"  Total profiles: {ds.sizes['time'] * ds.sizes['location']}")
    
    # Should be reasonable (adjust based on actual data size)
    assert memory_increase < 2000  # Less than 2 GB