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

import bz2
import gzip
import lzma
import subprocess
import pytest
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import shutil

from xsnow import xsnowDataset, read
import xsnow.plugin

# Snapshot the "core" xsnowDataset surface at conftest import time, i.e.
# before any test module has a chance to `import xsnow.extensions.*` and
# permanently monkeypatch `xsnowDataset` via `register_xsnowDataset_method`.
DEFAULT_XSNOWDATASET_ATTRS = frozenset(dir(xsnowDataset))
DEFAULT_REGISTERED_METHODS = frozenset(xsnow.plugin.list_registered_methods())

# ============================================================================
# TEST DATA REPOSITORY MANAGEMENT
# ============================================================================

HERE = Path(__file__).parent
TESTDATA_DIR = HERE / "data"
TESTDATA_REPO = "https://gitlab.com/avalanche-warning/universal/archive/test-data"

COMPRESSION_MODULES = {".gz": gzip, ".bz2": bz2, ".xz": lzma}
ALL_FILE_EXTS = [""] + list(COMPRESSION_MODULES.keys())

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
def snp_gridded_cache_dir(tmp_path_factory) -> Path:
    """
    A per-session temp dir to store cached NetCDF outputs.
    One per pytest invocation (per worker under xdist).
    """
    return tmp_path_factory.mktemp("snp_gridded_cache")


@pytest.fixture(scope="session")
def snp_gridded_netcdf(
    snp_gridded_cache_dir: Path,
    gridded_pros_dir: Path,
) -> Path:
    """
    Build the slow dataset ONCE from raw files, then cache as NetCDF.
    """
    nc_path = snp_gridded_cache_dir / "snp_gridded.nc"

    if not nc_path.exists():
        xs = read(gridded_pros_dir, lazy=False)
        xs.to_netcdf(nc_path)

    return nc_path


@pytest.fixture(scope="function")
def snp_gridded_xs(snp_gridded_netcdf: Path):
    """
    Fast reader: open the cached NetCDF for tests.
    """
    xs = read(snp_gridded_netcdf, lazy=False)
    return xs


@pytest.fixture(scope="session")
def snowobs_pros_dir() -> Path:
    """Directory containing snow observation SNOWPACK .pro files."""
    try:
        return _get_dir_path("pros/snowobs")
    except FileNotFoundError:
        pytest.skip("Snow observation data not available in test repo")


@pytest.fixture(scope="session")
def snowobs_smets_dir_nowcast() -> Path:
    """Directory containing snow observation SNOWPACK .smet files (nowcast forcing)."""
    try:
        return _get_dir_path("smets/snowobs/nowcast")
    except FileNotFoundError:
        pytest.skip("Snow observation data not available in test repo")


@pytest.fixture(scope="session")
def snowobs_pits_dir() -> Path:
    """Directory containing snow observation .pro files (converted from caaml)."""
    try:
        return _get_dir_path("caamls/as_pro")
    except FileNotFoundError:
        pytest.skip("Snow observation data not available in test repo")


@pytest.fixture(scope="session")
def snp_snowobs_cache_dir(tmp_path_factory) -> Path:
    """
    A per-session temp dir to store cached NetCDF outputs.
    One per pytest invocation (per worker under xdist).
    """
    return tmp_path_factory.mktemp("snp_snowobs_cache")


@pytest.fixture(scope="session")
def snp_snowobs_netcdf(
    snp_snowobs_cache_dir: Path,
    snowobs_pros_dir: Path,
    snowobs_pits_dir: Path,
) -> Path:
    """
    Build the slow dataset ONCE from raw files, then cache as NetCDF.
    """
    nc_path_sim = snp_snowobs_cache_dir / "snp_snowobs_sim.nc"
    nc_path_obs = snp_snowobs_cache_dir / "snp_snowobs_obs.nc"

    if not nc_path_sim.exists():
        xs = read(snowobs_pros_dir, lazy=False)
        xs.to_netcdf(nc_path_sim)

    if not nc_path_obs.exists():
        xs = read(snowobs_pits_dir, lazy=False)
        xs.to_netcdf(nc_path_obs)

    return nc_path_sim, nc_path_obs


@pytest.fixture(scope="function")
def snp_snowobs_sim_xs(snp_snowobs_netcdf: Path):
    """
    Fast reader: open the cached NetCDF for tests.
    """
    xs = read(snp_snowobs_netcdf[0], lazy=False)
    return xs

@pytest.fixture(scope="function")
def snp_snowobs_obs_xs(snp_snowobs_netcdf: Path):
    """
    Fast reader: open the cached NetCDF for tests.
    """
    xs = read(snp_snowobs_netcdf[1], lazy=False)
    return xs


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
        
        station_params = f"""[STATION_PARAMETERS]
StationName= {station_name}
Latitude= 46.5
Longitude= 10.5
Altitude= 2500
SlopeAngle= 35
SlopeAzi= 180

"""
        if n_layers > 0:
            data_sect = f"""
[DATA]
0500,{timestamp}
0501,{n_layers},{heights}
0502,{n_layers},{densities}
"""
        else:
            data_sect= f"""
[DATA]
0500,{timestamp}
0501,{n_layers}
0502,{n_layers}
"""
        content = station_params + data_sect
        filepath = tmp_path / f"{station_name}.pro"
        filepath.write_text(content)
        return filepath

    @staticmethod
    def aspect_file(
        path: Path,
        station_name: str | None = None,
        slope_angle: int = 0,
        slope_azi: int = 0,
        latitude: float = 46.5,
        longitude: float = 10.5,
        altitude: float = 2500,
        n_layers: int = 3,
    ) -> Path:
        """Generate a minimal PRO file for slope/aspect merging tests."""
        station_line = "" if station_name is None else f"StationName= {station_name}\n"
        heights = ",".join(str(0.1 * i) for i in range(1, n_layers + 1))
        densities = ",".join(str(100 * i) for i in range(1, n_layers + 1))

        path.write_text(
            f"""[STATION_PARAMETERS]
{station_line}Latitude= {latitude}
Longitude= {longitude}
Altitude= {altitude}
SlopeAngle= {slope_angle}
SlopeAzi= {slope_azi}

[DATA]
0500,01.01.2025 12:00:00
0501,{n_layers},{heights}
0502,{n_layers},{densities}
"""
        )
        return path

    @staticmethod
    def with_declared_units(
        path: Path,
        station_name: str = "UNITTEST",
        height_unit: str = "cm",
        height_values: tuple[float, float] = (10.0, 20.0),
    ) -> Path:
        """Generate a minimal PRO file with explicit header units."""
        path.write_text(
            f"""[STATION_PARAMETERS]
StationName= {station_name}
Latitude= 46.5
Longitude= 10.5
Altitude= 2500
SlopeAngle= 35
SlopeAzi= 180

[HEADER]
0501,nElems,height [> 0: top, < 0: bottom of elem.] ({height_unit})
0502,nElems,element density (kg m-3)
0506,nElems,liquid water content by volume (%)
0512,nElems,grain size (mm)
0513,nElems,grain type (Swiss Code F1F2F3)
0517,nElems,stress (kPa)

[DATA]
0500,01.01.2025 12:00:00
0501,2,{height_values[0]},{height_values[1]}
0502,2,100,200
0506,2,5,10
0512,2,0.5,1.0
0513,3,550,440,0
0517,2,1.2,2.4
"""
        )
        return path
    
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
0501,2,INVALID_DATA
"""
        elif corruption_type == "empty":
            content = ""
            
        elif corruption_type == "no_data_section":
            content = """[STATION_PARAMETERS]
StationName= NO_DATA_SECTION
Latitude= 46.5
"""
        elif corruption_type == "empty_data_section":
            content = """[STATION_PARAMETERS]
StationName= EMPTY_DATA_SECTION
Latitude= 46.5
[DATA]
"""
        elif corruption_type == "good_and_bad_data":
            content = """[STATION_PARAMETERS]
StationName= GOOD_AND_BAD_TIME
[DATA]
0500,01.01.2025 12:00:00
0501,2,0.1,0.2
0500,01.01.2025 13:00:00
0501,2,INVALID_DATA
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
def smet_units_xsnow_dataset():
    """Single-location xsnow dataset with variables in xsnow-internal units."""
    times = pd.date_range("2025-01-01", periods=2, freq="D")
    ds = xr.Dataset(
        data_vars={
            "TA": (("location", "time", "slope"), [[[0.0], [-10.0]]]),
            "RH": (("location", "time", "slope"), [[[0.5], [0.75]]]),
            "HS_mod": (("location", "time", "slope"), [[[0.25], [0.3]]]),
        },
        coords={
            "location": ["TEST"],
            "time": times,
            "slope": [0],
            "latitude": ("location", [46.5]),
            "longitude": ("location", [10.5]),
            "altitude": ("location", [2500.0]),
        },
    )
    ds["TA"].attrs["units"] = "degC"
    ds["RH"].attrs["units"] = "1"
    ds["HS_mod"].attrs["units"] = "m"
    return xsnowDataset(ds)


@pytest.fixture
def stored_units_xr_dataset():
    """Factory for minimal stored datasets with registered unit metadata."""
    def make(unit: str = "m") -> xr.Dataset:
        ds = xr.Dataset(
            data_vars={
                "height": (
                    ("location", "time", "slope", "realization", "layer"),
                    np.array([[[[[0.1, 0.2]]]]]),
                ),
            },
            coords={
                "location": ["TEST"],
                "time": [pd.Timestamp("2025-01-01")],
                "slope": [0],
                "realization": [0],
                "layer": [0, 1],
            },
        )
        ds["height"].attrs["units"] = unit
        ds["height"].attrs["custom_attr"] = "preserve-me"
        return ds

    return make


@pytest.fixture
def snow_profile_xr_dataset():
    """xarray.Dataset with realistic snow profile structure."""
    height = np.array([[[0.1, 0.2, 0.3], [0.15, 0.25, 0.35]]])
    density = np.array([[[100, 200, 300], [110, 210, 310]]])
    hs = height.max(axis=2)
    ds = xr.Dataset(
        {
            'height': (['location', 'time', 'layer'], height),
            'density': (['location', 'time', 'layer'], density),
            'profile_status': (['location', 'time', 'slope'], np.ones((1, 2, 1), dtype='i1')),
            'HS': (['location', 'time', 'slope'], hs[..., None]),
            'is_valid': (['location', 'time', 'slope'], hs[..., None] > 0),
        },
        coords={
            'location': ['TestStation'],  # ✨ CHANGED: String instead of [0]
            'time': pd.date_range('2025-01-01', periods=2, freq='D'),
            'layer': [0, 1, 2],
            'slope': [0],
            'latitude': ('location', [46.5]),
            'longitude': ('location', [10.5]),
            'altitude': ('location', [2500.0]),
            'azimuth': ('slope', [180.0]),
            'inclination': ('slope', [30.0]),
        }
    )
    ds['height'].attrs['units'] = 'm'
    ds['density'].attrs['units'] = 'kg m-3'
    ds['HS'].attrs['units'] = 'm'
    return ds


@pytest.fixture
def xsnow_dataset(snow_profile_xr_dataset):
    """xsnowDataset wrapper around snow profile data."""
    return xsnowDataset(snow_profile_xr_dataset)


@pytest.fixture(scope="session")
def default_xsnowdataset_attrs() -> frozenset[str]:
    return DEFAULT_XSNOWDATASET_ATTRS


@pytest.fixture(scope="session")
def default_registered_methods() -> frozenset[str]:
    return DEFAULT_REGISTERED_METHODS


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
    Fixture to compare two xsnowDataset objects.
    
    Handles both eager (numpy) and lazy (dask) datasets by computing
    dask arrays before comparison.
    """
    import dask.array as da
    
    def _compare(ds1, ds2, rtol=1e-5):
        # Helper to compute dataset if it contains dask arrays
        def compute_if_dask(ds):
            """Compute all dask arrays in a dataset."""
            has_dask = False
            
            # Check if any variables are dask arrays
            for var_name in ds.data.data_vars:
                var = ds.data[var_name]
                if isinstance(var.data, da.Array):
                    has_dask = True
                    break
            
            # If dataset has dask arrays, compute them
            if has_dask:
                # Use xarray's compute method to compute all dask arrays at once
                ds_computed = ds.data.compute()
                # Return new xsnowDataset with computed data
                from xsnow import xsnowDataset
                return xsnowDataset(ds_computed)
            
            return ds
        
        # Compute any dask arrays before comparison
        ds1 = compute_if_dask(ds1)
        ds2 = compute_if_dask(ds2)
        
        # Compare dimensions
        assert ds1.sizes == ds2.sizes, \
            f"Dimension mismatch: {ds1.sizes} != {ds2.sizes}"
        
        # Compare data variables
        for var in ds1.data.data_vars:
            assert var in ds2.data.data_vars, \
                f"Variable {var} not found in second dataset"
            
            var1_vals = ds1.data[var].values
            var2_vals = ds2.data[var].values
            
            # Handle non-numeric variables (datetime, strings) - use exact equality
            if np.issubdtype(var1_vals.dtype, np.datetime64) or np.issubdtype(var1_vals.dtype, np.str_):
                np.testing.assert_array_equal(
                    var1_vals,
                    var2_vals,
                    err_msg=f"Variable {var} differs"
                )
            else:
                # Numeric variables - use rtol
                np.testing.assert_allclose(
                    var1_vals,
                    var2_vals,
                    rtol=rtol,
                    err_msg=f"Variable {var} differs"
                )
                                         
        # Compare data variables
        for var in ds1.data.data_vars:
            assert var in ds2.data.data_vars, \
                f"Variable {var} not found in second dataset"
            
            # Now both are numpy arrays, safe to compare
            np.testing.assert_allclose(
                ds1.data[var].values,
                ds2.data[var].values,
                rtol=rtol,
                err_msg=f"Variable {var} differs"
            )
    
    return _compare

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


#: Below this, an RSS delta says nothing.
#:
#: Process RSS moves for reasons that have nothing to do with the code under
#: test: garbage collection timing, allocator arenas being reused or returned
#: to the OS, and page-cache reclamation caused by *earlier* tests in the same
#: session. Measured swings of about -8 MB have been observed between two
#: samples taken around a 30 KB allocation, so this floor is set well above
#: that. Assertions on smaller deltas are measuring noise.
MEMORY_NOISE_MB = 16.0


@pytest.fixture
def memory_monitor():
    """
    Monitor process memory usage during tests.

    RSS is **not monotonic**. In a long-running single-process pytest session
    it can fall between two samples, which previously produced nonsensical
    negative "increases" (a -48.4 MB reading made a ratio assertion flip).
    Two things guard against that:

    - every sample forces a ``gc.collect()`` first, so a reading reflects live
      objects rather than whenever the collector happened to run;
    - ``increase()`` is floored at zero, since a negative increase is always a
      measurement artifact and never a fact about the code under test.

    Use ``noise_mb`` for assertion thresholds rather than comparing two deltas
    to each other: a difference of two noisy samples is noisier than either.
    ``raw_increase()`` is available when a test genuinely wants the unclamped
    number for reporting.

    Usage:
        def test_memory(memory_monitor):
            memory_monitor.start()
            # ... do something ...
            increase = memory_monitor.increase()
            assert increase < 1000  # < 1 GB
    """
    import gc

    import psutil

    class MemoryMonitor:
        #: Deltas at or below this are indistinguishable from sampling noise.
        noise_mb = MEMORY_NOISE_MB

        def __init__(self):
            self.process = psutil.Process()
            self.baseline = None

        def _sample(self) -> float:
            """RSS in MB, after collecting so the reading reflects live data."""
            gc.collect()
            return self.process.memory_info().rss / 1024 / 1024

        def start(self):
            """Record baseline memory usage."""
            self.baseline = self._sample()
            return self.baseline

        def current(self) -> float:
            """Get current memory usage in MB."""
            return self._sample()

        def raw_increase(self) -> float:
            """
            Increase from baseline in MB, unclamped.

            May be negative when the OS reclaimed memory allocated earlier in
            the session. Useful for reporting, not for assertions.
            """
            if self.baseline is None:
                raise RuntimeError("Call start() before checking increase()")
            return self.current() - self.baseline

        def increase(self) -> float:
            """
            Get memory increase from baseline in MB, floored at zero.

            A negative increase means the process gave memory back between
            samples, not that the code under test freed anything.
            """
            return max(0.0, self.raw_increase())

        def is_measurable(self, value: float) -> bool:
            """Whether a delta is large enough to mean anything."""
            return value > self.noise_mb

        def report(self):
            """Print memory usage report."""
            current = self.current()
            if self.baseline:
                increase = current - self.baseline
                print(
                    f"\nMemory: {self.baseline:.1f} MB → {current:.1f} MB "
                    f"({increase:+.1f} MB, noise floor {self.noise_mb:.0f} MB)"
                )
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
