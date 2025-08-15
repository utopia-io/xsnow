# test_io.py
"""
Unit and integration tests for the xsnow_io module.

This module uses the pytest framework to test the functionality of the
read() and read_pro() functions, ensuring they correctly parse .pro files,
handle various input scenarios, and process data in parallel.

---
Setup and Execution Guide
---

This test suite assumes you are using a standard Python project structure
with a `src` layout and a `pyproject.toml` file for configuration.

**1. Directory Structure:**

Your project should be organized as follows. The key is that the `tests`
directory is at the root level, alongside `pyproject.toml` and `src`.

xsnow/
├── pyproject.toml
├── src/
│   └── xsnow/
│       ├── __init__.py
│       ├── dataset.py
│       ├── registry.py
│       └── xsnow_io.py
└── tests/
    └── test_io.py

**2. Environment Setup:**

To run these tests, you must first install the `xsnow` package in "editable"
mode along with its development dependencies (like pytest).

From the root directory of your project (`xsnow/`), run the following command
using `uv`:

    uv pip install -e ".[dev]"

This command reads your `pyproject.toml`, installs the core dependencies,
and also installs the packages listed under `[project.optional-dependencies].dev`.

**3. Running the Tests:**

All tests should be executed from the **root directory** of the project,
not from within the `tests/` directory.

Navigate to your project's root folder in your terminal and run:

    pytest

Pytest will automatically discover the `tests` directory and run all the
test functions found within this file.
"""

import pytest
from pathlib import Path
import xarray as xr
import pandas as pd

# Assuming the project is structured such that xsnow can be imported
from xsnow.xsnow_io import read, read_pro

# --- Test Data Fixtures ---

DUMMY_CODE_MAP = {
    "0501": "height",
    "0502": "density",
    "0500": "timestamp"
}

@pytest.fixture
def single_pro_file(tmp_path):
    """
    Pytest fixture to create a single, well-formed .pro file in a temporary
    directory.
    """
    content = """
[STATION_PARAMETERS]
StationName= TEST_STATION_1
Latitude= 45.0
Longitude= 10.0
Altitude= 2000
SlopeAngle= 25.0
SlopeAzi= 90.0
[DATA]
0500,01.01.2025 12:00:00
0501,2,0.1,0.2
0502,2,200,250
0500,02.01.2025 12:00:00
0501,2,0.15,0.25
0502,2,210,260
"""
    file_path = tmp_path / "station_1.pro"
    file_path.write_text(content)
    return file_path

@pytest.fixture
def multiple_pro_files(tmp_path):
    """
    Pytest fixture to create multiple .pro files in a temporary directory.
    """
    files = []
    for i in range(3):
        content = f"""
[STATION_PARAMETERS]
StationName= TEST_STATION_{i}
Latitude= {45.0 + i}
Longitude= {10.0 + i}
Altitude= {2000 + i*100}
SlopeAngle= 30.0
SlopeAzi= 180.0
[DATA]
0500,01.01.2025 12:00:00
0501,1,0.5
0502,1,300
"""
        file_path = tmp_path / f"station_{i}.pro"
        file_path.write_text(content)
        files.append(file_path)
    return files

@pytest.fixture
def pro_files_variable_layers(tmp_path):
    """Creates two .pro files with different numbers of layers."""
    content_5_layers = """
[STATION_PARAMETERS]
StationName= FIVE_LAYERS
[DATA]
0500,01.01.2025 12:00:00
0501,5,0.1,0.2,0.3,0.4,0.5
"""
    (tmp_path / "five_layers.pro").write_text(content_5_layers)
    
    content_10_layers = """
[STATION_PARAMETERS]
StationName= TEN_LAYERS
[DATA]
0500,01.01.2025 12:00:00
0501,10,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0
"""
    (tmp_path / "ten_layers.pro").write_text(content_10_layers)
    return [tmp_path / "five_layers.pro", tmp_path / "ten_layers.pro"]

# --- Tests for read_pro() ---

def test_read_pro_single_file(single_pro_file):
    """Tests reading a single valid .pro file."""
    ds = read_pro(single_pro_file, code_map=DUMMY_CODE_MAP)
    assert isinstance(ds, xr.Dataset)
    assert ds.sizes['location'] == 1
    assert ds.sizes['time'] == 2
    assert ds.coords['location'].values[0] == 'TEST_STATION_1'
    assert ds.coords['latitude'].values[0] == 45.0
    assert ds['density'].values[0, 0, 0, 0, 0] == 200

def test_read_pro_datetime_filter(single_pro_file):
    """Tests the datetime filtering functionality of read_pro."""
    ds = read_pro(single_pro_file, datetime_start="2025-01-02 00:00:00", code_map=DUMMY_CODE_MAP)
    assert ds.sizes['time'] == 1
    assert pd.to_datetime(ds.time.values[0]).day == 2

def test_read_pro_file_not_found():
    """Tests that read_pro returns None for a non-existent file."""
    ds = read_pro("non_existent_file.pro", code_map=DUMMY_CODE_MAP)
    assert ds is None

def test_read_pro_empty_file(tmp_path):
    """Tests that read_pro handles empty files gracefully."""
    empty_file = tmp_path / "empty.pro"
    empty_file.write_text("")
    ds = read_pro(empty_file, code_map=DUMMY_CODE_MAP)
    assert ds is None

# --- Tests for read() ---

def test_read_single_file_path(single_pro_file):
    """Tests the high-level read() function with a single file path."""
    ds = read(str(single_pro_file), _code_map=DUMMY_CODE_MAP)
    assert isinstance(ds, xr.Dataset)
    assert ds.sizes['location'] == 1
    assert ds.coords['location'].values[0] == 'TEST_STATION_1'

def test_read_list_of_files(multiple_pro_files):
    """
    Tests reading a list of files, checking for correct concatenation.
    This test implicitly checks both parallel and sequential execution paths.
    """
    # Test sequential
    ds_seq = read([str(f) for f in multiple_pro_files], parallel=False, _code_map=DUMMY_CODE_MAP)
    assert isinstance(ds_seq, xr.Dataset)
    assert ds_seq.sizes['location'] == 3
    assert 'TEST_STATION_0' in ds_seq.coords['location'].values
    assert 'TEST_STATION_2' in ds_seq.coords['location'].values
    assert ds_seq.coords['latitude'].values[2] == 47.0

    # Test parallel
    ds_par = read([str(f) for f in multiple_pro_files], parallel=True, _code_map=DUMMY_CODE_MAP)
    assert isinstance(ds_par, xr.Dataset)
    assert ds_par.sizes['location'] == 3
    # Use isel to check a value from the last station
    assert ds_par.isel(location=2).coords['latitude'].values == 47.0
    # Verify the datasets are identical regardless of execution path
    xr.testing.assert_identical(ds_seq, ds_par)


def test_read_directory(multiple_pro_files):
    """Tests reading all .pro files from a given directory."""
    # The fixture gives us the directory path via the parent attribute of a file
    directory = multiple_pro_files[0].parent
    ds = read(str(directory), _code_map=DUMMY_CODE_MAP)
    assert isinstance(ds, xr.Dataset)
    assert ds.sizes['location'] == 3
    assert 'TEST_STATION_1' in ds.coords['location'].values

def test_read_max_layers_padding(pro_files_variable_layers):
    """Tests the max_layers padding and truncation logic."""
    files = [str(f) for f in pro_files_variable_layers]
    
    # Test 'auto' mode: should pad to the largest profile (10 layers)
    ds_auto = read(files, max_layers='auto', _code_map=DUMMY_CODE_MAP)
    assert ds_auto.sizes['layer'] == 10
    
    # Test user-specified padding: should pad to 15
    ds_padded = read(files, max_layers=15, _code_map=DUMMY_CODE_MAP)
    assert ds_padded.sizes['layer'] == 15
    
    # Test user-specified truncation: should truncate to 7
    ds_truncated = read(files, max_layers=7, _code_map=DUMMY_CODE_MAP)
    assert ds_truncated.sizes['layer'] == 7
    # Check that the 10-layer profile was actually truncated
    ten_layer_profile = ds_truncated.sel(location='TEN_LAYERS')
    assert ten_layer_profile['height'].notnull().sum().item() == 7


def test_read_empty_directory(tmp_path):
    """Tests that read() handles an empty directory gracefully."""
    ds = read(str(tmp_path), _code_map=DUMMY_CODE_MAP)
    assert ds is None
