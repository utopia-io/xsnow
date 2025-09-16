import os
import subprocess
import pytest

HERE = os.path.dirname(__file__)
TESTDATA_DIR = os.path.join(HERE, "data")
TESTDATA_REPO = "https://gitlab.com/avalanche-warning/universal/archive/test-data"

def _ensure_testdata_repo():
    """
    Clone an external testdata repo into TESTDATA_DIR.
    Only downloads if TESTDATA_DIR does not already exist.
    """
    if not os.path.isdir(TESTDATA_DIR):
        print(f"Cloning testdata repo into {TESTDATA_DIR} …")
        subprocess.run(["git", "clone", "--depth", "1", TESTDATA_REPO, TESTDATA_DIR], check=True)
    else:
        print(f"Testdata directory already exists at {TESTDATA_DIR}")

def _get_file_path(relpath: str) -> str:
    """Ensure testdata is present, then return the full path to a file inside it."""
    _ensure_testdata_repo()
    full = os.path.join(TESTDATA_DIR, relpath)
    if not os.path.isfile(full):
        raise FileNotFoundError(f"Expected test file not found: {full}")
    return full


@pytest.fixture(scope="session")
def sample_pro_file() -> str:
    """Path to a sample SNOWPACK PRO file."""
    return _get_file_path("pros/gridded/VIR1A1.pro")


if __name__ == "__main__":
    _ensure_testdata_repo()
