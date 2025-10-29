# tests/conftest.py
"""
Pytest configuration and shared fixtures for xsnow test suite.

Real Data Strategy:
  - Clones external test data repository on first run
  - Provides fixtures for accessing real SNOWPACK model output
  - Test data versioned separately from package

Synthetic Data Strategy:
  - Generates edge case data on-the-fly
  - Used ONLY for specific error conditions and edge cases
  - Each test gets isolated tmp_path
"""

import os
import subprocess
import pytest
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import shutil

from xsnow import xsnowDataset

# ============================================================================
# TEST DATA REPOSITORY MANAGEMENT
# ============================================================================

HERE = Path(__file__).parent
TESTDATA_DIR = HERE / "data"
TESTDATA_REPO = "https://gitlab.com/avalanche-warning/universal/archive/test-data"


def _ensure_testdata_repo():
    """
    Clone external testdata repo into TESTDATA_DIR.
    Only downloads if TESTDATA_DIR does not already exist.
    """
    if not TESTDATA_DIR.is_dir():
        print(f"\nCloning testdata repo into {TESTDATA_DIR} ...")
        subprocess.run(
            ["git", "clone", "--depth", "1", TESTDATA_REPO, str(TESTDATA_DIR)],
            check=True
        )
        print("✓ Test data cloned successfully")
    else:
        print(f"✓ Test data already available at {TESTDATA_DIR}")


def _get_file_path(relpath: str) -> Path:
    """
    Ensure testdata is present, then return full path to file inside it.
    
    Args:
        relpath: Relative path within test data repo
        
    Returns:
        Absolute Path to the file
        
    Raises:
        FileNotFoundError: If file doesn't exist after cloning
    """
    _ensure_testdata_repo()
    full_path = TESTDATA_DIR / relpath
    if not full_path.is_file():
        raise FileNotFoundError(f"Expected test file not found: {full_path}")
    return full_path


def _get_dir_path(relpath: str) -> Path:
    """
    Ensure testdata is present, then return full path to directory inside it.
    
    Args:
        relpath: Relative path within test data repo
        
    Returns:
        Absolute Path to the directory
        
    Raises:
        FileNotFoundError: If directory doesn't exist after cloning
    """
    _ensure_testdata_repo()
    full_path = TESTDATA_DIR / relpath
    if not full_path.is_dir():
        raise FileNotFoundError(f"Expected test directory not found: {full_path}")
    return full_path


# ============================================================================
# REAL SNOWPACK DATA FIXTURES (PRIMARY)
# ============================================================================

@pytest.fixture(scope="session")
def testdata_dir() -> Path:
    """Root directory of cloned test data repository."""
    _ensure_testdata_repo()
    return TESTDATA_DIR


@pytest.fixture(scope="session")
def sample_pro_file() -> Path:
    """Path to a single sample SNOWPACK .pro file."""
    return _get_file_path("pros/gridded/VIR1A1.pro")


@pytest.fixture(scope="session")
def gridded_pros_dir() -> Path:
    """Directory containing gridded SNOWPACK .pro files (multi-slope)."""
    return _get_dir_path("pros/gridded")


@pytest.fixture(scope="session")
def snowobs_pros_dir() -> Path:
    """Directory containing snow observation SNOWPACK .pro files."""
    try:
        return _get_dir_path("pros/snowobs")
    except FileNotFoundError:
        pytest.skip("Snow observation data not available in test repo")


@pytest.fixture(scope="session")
def pit_pros_dir() -> Path:
    """Directory containing manual snow pit .pro files."""
    try:
        return _get_dir_path("pits")
    except FileNotFoundError:
        pytest.skip("Pit data not available in test repo")


@pytest.fixture(scope="session")
def smet_files_dir() -> Path:
    """Directory containing SMET meteorological forcing files."""
    try:
        return _get_dir_path("smets")
    except FileNotFoundError:
        pytest.skip("SMET data not available in test repo")


@pytest.fixture
def real_file_subset(gridded_pros_dir, tmp_path) -> Path:
    """
    Copy a small subset (5 files) of real data to tmp directory.
    Use this for tests that need real data but should run fast.
    """
    source_files = list(gridded_pros_dir.glob("*.pro"))[:5]
    
    if not source_files:
        pytest.skip("No source files available in gridded data")
    
    for src in source_files:
        shutil.copy(src, tmp_path / src.name)
    
    return tmp_path


@pytest.fixture
def single_real_pro(sample_pro_file, tmp_path) -> Path:
    """
    Copy a single real .pro file to tmp directory for focused testing.
    Isolated from original to allow modifications without affecting other tests.
    """
    dest = tmp_path / sample_pro_file.name
    shutil.copy(sample_pro_file, dest)
    return dest


# ============================================================================
# SYNTHETIC DATA GENERATOR (FOR EDGE CASES ONLY)
# ============================================================================

class SyntheticProGenerator:
    """Factory for generating synthetic .pro files."""
    
    @staticmethod
    def basic(
        tmp_path: Path,
        station_name: str = "SYNTHETIC",
        n_layers: int = 3,
        timestamp: str = "01.01.2025 12:00:00"
    ) -> Path:
        """Generate a minimal valid .pro file."""
        heights = ','.join(str(0.1 * i) for i in range(1, n_layers + 1))
        densities = ','.join(str(100 * i) for i in range(1, n_layers + 1))
        
        content = f"""[STATION_PARAMETERS]
StationName= {station_name}
Latitude= 46.5
Longitude= 10.5
Altitude= 2500
SlopeAngle= 35
SlopeAzi= 180

[DATA]
0500,{timestamp}
0501,{n_layers},{heights}
0502,{n_layers},{densities}
"""
        filepath = tmp_path / f"{station_name}.pro"
        filepath.write_text(content)
        return filepath
    
    @staticmethod
    def with_soil_layers(tmp_path: Path, n_snow: int = 3, n_soil: int = 2) -> Path:
        """Generate profile with soil layers at bottom."""
        total = n_snow + n_soil
        heights = ','.join(str(0.1 * i) for i in range(1, total + 1))
        densities = ','.join(str(100 + i * 50) for i in range(total))
        
        snow_types = ['200'] * n_snow
        soil_types = []
        for i in range(n_soil):
            soil_types.append('81' if i % 2 == 0 else '82')
        grain_types = ','.join(snow_types + soil_types)
        
        content = f"""[STATION_PARAMETERS]
StationName= WITH_SOIL
SlopeAngle= 35
SlopeAzi= 180

[DATA]
0500,01.01.2025 12:00:00
0501,{total},{heights}
0502,{total},{densities}
0513,{total},{grain_types}
"""
        filepath = tmp_path / "with_soil.pro"
        filepath.write_text(content)
        return filepath
    
    @staticmethod
    def with_surface_hoar(
        tmp_path: Path, 
        station_name: str = "SH_TEST",
        has_sh: bool = True,
        sh_grain_size: float = 0.5,
        sh_density: float = 100.0
    ) -> Path:
        """Generate profile with surface hoar data."""
        n_layers = 3
        heights = ','.join(str(0.1 * i) for i in range(1, n_layers + 1))
        densities = ','.join(str(100 * i) for i in range(1, n_layers + 1))
        grain_types = ','.join(['200'] * n_layers)
        
        sh_flag = 1.0 if has_sh else 0.0
        sh_data = f"{sh_flag},{sh_grain_size},{sh_density}"
        
        content = f"""[STATION_PARAMETERS]
StationName= {station_name}
Latitude= 46.5
Longitude= 10.5
Altitude= 2500
SlopeAngle= 35
SlopeAzi= 180

[DATA]
0500,01.01.2025 12:00:00
0501,{n_layers},{heights}
0502,{n_layers},{densities}
0513,{n_layers},{grain_types}
0514,3,{sh_data}
"""
        filepath = tmp_path / f"{station_name}.pro"
        filepath.write_text(content)
        return filepath
    
    @staticmethod
    def corrupted(tmp_path: Path, corruption_type: str = "malformed_data") -> Path:
        """Generate intentionally corrupted files."""
        if corruption_type == "malformed_data":
            content = """[STATION_PARAMETERS]
StationName= CORRUPTED
[DATA]
0500,01.01.2025 12:00:00
0501,INVALID_DATA
"""
        elif corruption_type == "empty":
            content = ""
            
        elif corruption_type == "no_data_section":
            content = """[STATION_PARAMETERS]
StationName= NO_DATA
Latitude= 46.5
"""
        elif corruption_type == "invalid_timestamp":
            content = """[STATION_PARAMETERS]
StationName= BAD_TIME
[DATA]
0500,NOT_A_DATE
0501,2,0.1,0.2
0502,2,100,200
"""
        elif corruption_type == "missing_headers":
            content = """[STATION_PARAMETERS]
StationName= MISSING_HEADERS
[DATA]
0500,01.01.2025 12:00:00
0501,2,0.1,0.2
"""
        else:
            raise ValueError(f"Unknown corruption type: {corruption_type}")
        
        if tmp_path.suffix:
            tmp_path = tmp_path.parent
        
        filepath = tmp_path / f"corrupted_{corruption_type}.pro"
        filepath.write_text(content)
        return filepath    
    
    @staticmethod
    def empty_profiles(tmp_path: Path, n_profiles: int = 3) -> Path:
        """Generate file with profiles that have no snow."""
        profiles = []
        for i in range(n_profiles):
            date = pd.Timestamp('2025-01-01') + pd.Timedelta(days=i)
            profiles.append(f"""0500,{date.strftime('%d.%m.%Y %H:%M:%S')}
0501,0,
0502,0,""")
        
        content = f"""[STATION_PARAMETERS]
StationName= EMPTY_PROFILES

[DATA]
{chr(10).join(profiles)}
"""
        filepath = tmp_path / "empty_profiles.pro"
        filepath.write_text(content)
        return filepath
    
    @staticmethod
    def varying_layers(tmp_path: Path, layer_counts: list = None) -> Path:
        """Generate file with profiles having different layer counts."""
        if layer_counts is None:
            layer_counts = [2, 5, 3, 8, 4]
        
        profiles = []
        for i, n_layers in enumerate(layer_counts):
            date = pd.Timestamp('2025-01-01') + pd.Timedelta(days=i)
            heights = ','.join(str(0.1 * j) for j in range(1, n_layers + 1))
            densities = ','.join(str(100 * j) for j in range(1, n_layers + 1))
            profiles.append(f"""0500,{date.strftime('%d.%m.%Y %H:%M:%S')}
0501,{n_layers},{heights}
0502,{n_layers},{densities}""")
        
        content = f"""[STATION_PARAMETERS]
SlopeAngle=35
SlopeAzi=180
StationName= VARYING_LAYERS

[DATA]
{chr(10).join(profiles)}
"""
        filepath = tmp_path / "varying_layers.pro"
        filepath.write_text(content)
        return filepath
    
    @staticmethod
    def multi_station(
        tmp_path: Path,
        n_stations: int = 3,
        n_profiles_each: int = 2
    ) -> Path:
        """Generate multiple station files in a directory."""
        for i in range(n_stations):
            station_name = f"STATION_{i:03d}"  # NEW: Consistent naming
            
            profiles = []
            for j in range(n_profiles_each):
                date = pd.Timestamp('2025-01-01') + pd.Timedelta(days=j)
                profiles.append(f"""0500,{date.strftime('%d.%m.%Y %H:%M:%S')}
0501,2,0.1,0.2
0502,2,{100 + i * 10},{200 + i * 10}""")
            
            # Add unique slope values for each station
            slope_angle = 30 + i * 5  # Station 0: 30°, Station 1: 35°, etc.
            slope_azi = 180 + i * 10   # Station 0: 180°, Station 1: 190°, etc.
            
            content = f"""[STATION_PARAMETERS]
StationName= {station_name}
Latitude= {46.0 + i * 0.1}
Longitude= {10.0 + i * 0.1}
Altitude= {2000 + i * 100}
SlopeAngle= {slope_angle}
SlopeAzi= {slope_azi}

[DATA]
{chr(10).join(profiles)}
"""
            (tmp_path / f"{station_name}.pro").write_text(content)
        
        return tmp_path
    
    @staticmethod
    def extreme_layers(tmp_path: Path, n_layers: int = 500) -> Path:
        """Generate profile with extreme layer count."""
        heights = ','.join(str(0.01 * i) for i in range(1, n_layers + 1))
        densities = ','.join(str(100 + i) for i in range(n_layers))
        
        content = f"""[STATION_PARAMETERS]
slopeAngle=35
SlopeAzi=180
StationName= EXTREME_LAYERS

[DATA]
0500,01.01.2025 12:00:00
0501,{n_layers},{heights}
0502,{n_layers},{densities}
"""
        filepath = tmp_path / "extreme_layers.pro"
        filepath.write_text(content)
        return filepath
    
    @staticmethod
    def multi_slope(tmp_path: Path, station_name: str, slopes: list = None) -> Path:
        """Generate multiple .pro files for same station with different slopes."""
        if slopes is None:
            slopes = [0, 15, 30, 45]
        
        for slope_angle in slopes:
            content = f"""[STATION_PARAMETERS]
StationName= {station_name}
Latitude= 46.5
Longitude= 10.5
Altitude= 2500
SlopeAngle= {slope_angle}
SlopeAzi= 180

[DATA]
0500,01.01.2025 12:00:00
0501,3,0.1,0.2,0.3
0502,3,100,200,300
"""
            filename = f"{station_name}_slope{slope_angle}.pro"
            (tmp_path / filename).write_text(content)
        
        return tmp_path

class SyntheticSmetGenerator:
    """Factory for generating synthetic SMET files."""
    
    @staticmethod
    def basic(
        tmp_path: Path,
        station_name: str = "SYNTHETIC",
        n_timesteps: int = 24,
        start_time: str = "2025-01-01T00:00:00"
    ) -> Path:
        """Generate a basic SMET file with temperature and humidity."""
        timestamps = pd.date_range(start_time, periods=n_timesteps, freq='h')
        data_lines = []
        
        for ts in timestamps:
            ta = 270 + np.random.randn() * 5  # ~-3°C ± 5°C
            rh = 80 + np.random.randn() * 10  # 80% ± 10%
            rh = np.clip(rh, 0, 100)
            data_lines.append(f"{ts.isoformat()} {ta:.2f} {rh:.2f}")
        
        content = f"""SMET 1.1 ASCII

[HEADER]
station_name = {station_name}
latitude = 46.5
longitude = 10.5
altitude = 2500
nodata = -999
fields = timestamp TA RH

[DATA]
{chr(10).join(data_lines)}
"""
        filepath = tmp_path / f"{station_name}.smet"
        filepath.write_text(content)
        return filepath


@pytest.fixture
def synthetic_pro():
    """Provide access to synthetic .pro generator in tests."""
    return SyntheticProGenerator


@pytest.fixture
def synthetic_smet():
    """Provide access to synthetic SMET generator in tests."""
    return SyntheticSmetGenerator


# ============================================================================
# XARRAY/XSNOW DATASET FIXTURES
# ============================================================================

@pytest.fixture
def minimal_xr_dataset():
    """Minimal xarray.Dataset for basic functionality tests."""
    return xr.Dataset(
        {'temp': (['x'], [1, 2, 3])},
        coords={'x': [0, 1, 2]}
    )


@pytest.fixture
def snow_profile_xr_dataset():
    """xarray.Dataset with realistic snow profile structure."""
    ds = xr.Dataset(
        {
            'height': (['location', 'time', 'layer'],
                      np.array([[[0.1, 0.2, 0.3], [0.15, 0.25, 0.35]]])),
            'density': (['location', 'time', 'layer'],
                       np.array([[[100, 200, 300], [110, 210, 310]]]))
        },
        coords={
            'location': ['TestStation'],  # ✨ CHANGED: String instead of [0]
            'time': pd.date_range('2025-01-01', periods=2, freq='D'),
            'layer': [0, 1, 2],
            'latitude': ('location', [46.5]),
            'longitude': ('location', [10.5]),
            'altitude': ('location', [2500.0])
        }
    )
    return ds


@pytest.fixture
def xsnow_dataset(snow_profile_xr_dataset):
    """xsnowDataset wrapper around snow profile data."""
    return xsnowDataset(snow_profile_xr_dataset)


# ============================================================================
# HELPER/UTILITY FIXTURES
# ============================================================================

@pytest.fixture
def assert_physical_validity():
    """
    Helper fixture for validating physical constraints in datasets.
    
    Usage:
        def test_my_function(real_file_subset, assert_physical_validity):
            ds = read(real_file_subset)
            assert_physical_validity(ds)
    """
    def validator(ds):
        """Validate that dataset values are physically reasonable."""
        # Unwrap if xsnowDataset
        data = ds.data if isinstance(ds, xsnowDataset) else ds
        
        if 'density' in data:
            density = data['density'].values
            valid = density[~np.isnan(density)]
            if len(valid) > 0:
                assert np.all(valid >= 0), "Negative density found"
                assert np.all(valid <= 1000), "Unrealistic density (>1000 kg/m³)"
        
        if 'temperature' in data:
            temp = data['temperature'].values
            valid = temp[~np.isnan(temp)]
            if len(valid) > 0:
                # Handle both Kelvin and Celsius
                if valid.max() > 100:  # Likely Kelvin
                    assert np.all(valid >= 200), "Temperature too low (K)"
                    assert np.all(valid <= 320), "Temperature too high (K)"
                else:  # Likely Celsius
                    assert np.all(valid >= -100), "Temperature too low (°C)"
                    assert np.all(valid <= 50), "Temperature too high (°C)"
        
        if 'height' in data:
            height = data['height'].values
            valid = height[~np.isnan(height)]
            if len(valid) > 0:
                assert np.all(valid >= 0), "Negative height found"
        
        return True
    
    return validator


@pytest.fixture
def compare_datasets():
    """
    Helper fixture for comparing two datasets.
    
    Usage:
        def test_equality(compare_datasets):
            ds1 = read(source1)
            ds2 = read(source2)
            compare_datasets(ds1, ds2, rtol=1e-5)
    """
    def comparator(ds1, ds2, rtol=1e-5, check_attrs=False):
        """Compare two xsnowDatasets or xarray.Datasets."""
        # Unwrap if xsnowDatasets
        data1 = ds1.data if isinstance(ds1, xsnowDataset) else ds1
        data2 = ds2.data if isinstance(ds2, xsnowDataset) else ds2
        
        # Check dimensions
        assert data1.sizes == data2.sizes, f"Dataset dimensions differ: {data1.sizes} vs {data2.sizes}"
        
        # Check data variables
        common_vars = set(data1.data_vars) & set(data2.data_vars)
        for var in common_vars:
            np.testing.assert_allclose(
                data1[var].values,
                data2[var].values,
                rtol=rtol,
                equal_nan=True,
                err_msg=f"Variable '{var}' differs"
            )
        
        if check_attrs:
            assert data1.attrs == data2.attrs, "Dataset attributes differ"
        
        return True
    
    return comparator


# ============================================================================
# PERFORMANCE MONITORING FIXTURES
# ============================================================================

@pytest.fixture
def timer():
    """
    Simple timer fixture for performance tests.
    
    Usage:
        def test_performance(timer):
            timer.start('operation')
            # ... do something ...
            timer.stop('operation')
            timer.report()
            assert timer.get('operation') < 5.0  # < 5 seconds
    """
    import time
    
    class Timer:
        def __init__(self):
            self.times = {}
        
        def start(self, label: str):
            self.times[label] = {'start': time.time()}
        
        def stop(self, label: str):
            if label in self.times and 'start' in self.times[label]:
                self.times[label]['elapsed'] = time.time() - self.times[label]['start']
        
        def get(self, label: str) -> float:
            """Get elapsed time for a label, or None if not recorded."""
            return self.times.get(label, {}).get('elapsed')
        
        def report(self):
            """Print all recorded times."""
            print("\n" + "="*50)
            print("TIMING REPORT")
            print("="*50)
            for label, data in sorted(self.times.items()):
                if 'elapsed' in data:
                    print(f"  {label:30s}: {data['elapsed']:7.3f}s")
            print("="*50)
    
    return Timer()


@pytest.fixture
def memory_monitor():
    """
    Monitor memory usage during tests.
    
    Usage:
        def test_memory(memory_monitor):
            memory_monitor.start()
            # ... do something ...
            increase = memory_monitor.increase()
            assert increase < 1000  # < 1 GB
    """
    import psutil
    
    class MemoryMonitor:
        def __init__(self):
            self.process = psutil.Process()
            self.baseline = None
        
        def start(self):
            """Record baseline memory usage."""
            self.baseline = self.process.memory_info().rss / 1024 / 1024  # MB
            return self.baseline
        
        def current(self) -> float:
            """Get current memory usage in MB."""
            return self.process.memory_info().rss / 1024 / 1024
        
        def increase(self) -> float:
            """Get memory increase from baseline in MB."""
            if self.baseline is None:
                raise RuntimeError("Call start() before checking increase()")
            return self.current() - self.baseline
        
        def report(self):
            """Print memory usage report."""
            current = self.current()
            if self.baseline:
                increase = current - self.baseline
                print(f"\nMemory: {self.baseline:.1f} MB → {current:.1f} MB (+{increase:.1f} MB)")
            else:
                print(f"\nMemory: {current:.1f} MB")
    
    return MemoryMonitor()


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "real_data: mark test as using real SNOWPACK data from test repo"
    )
    config.addinivalue_line(
        "markers", "synthetic: mark test as using synthetic/generated data"
    )
    config.addinivalue_line(
        "markers", "edge_case: mark test as testing edge cases or error conditions"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow (>5 seconds)"
    )
    config.addinivalue_line(
        "markers", "benchmark: mark test as performance benchmark"
    )


# ============================================================================
# COMMAND-LINE SCRIPT FOR SETUP
# ============================================================================

if __name__ == "__main__":
    """Run to manually ensure test data is cloned."""
    print("Setting up xsnow test data...")
    _ensure_testdata_repo()
    print("\n✓ Test data ready!")
    print(f"  Location: {TESTDATA_DIR}")
    
    # List available data
    print("\nAvailable test data:")
    for subdir in ["pros/gridded", "pros/snowobs", "pits", "smets"]:
        path = TESTDATA_DIR / subdir
        if path.exists():
            count = len(list(path.glob("*")))
            print(f"  {subdir:20s}: {count} files") 