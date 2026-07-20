"""
Unit tests for the Relative Threshold Area extension.

These tests verify that compute_rta attaches the expected variables and
returns a wrapped xsnowDataset with the layer-based coordinate system.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xsnow import xsnowDataset
from xsnow.extensions.calculate_rta import compute_rta  # register the method on xsnowDataset


@pytest.fixture
def rta_ds():
    """Provides a compact xsnowDataset suitable for compute_rta."""
    grain_type = xr.DataArray(
        np.array([[[33, 45, 9, 332]]], dtype=float),
        dims=("location", "time", "layer"),
        attrs={"units": "Swiss Code F1F2F3"},
    )
    data = {
        "height": (("location", "time", "layer"), np.array([[[0.10, 0.22, 0.37, 0.55]]], dtype=float)),
        "hand_hardness": (("location", "time", "layer"), np.array([[[1.0, 2.0, 3.0, 4.0]]], dtype=float)),
        "grain_size": (("location", "time", "layer"), np.array([[[0.20, 0.30, 0.60, 0.90]]], dtype=float)),
        "grain_type": grain_type,
        "hardness_difference": (("location", "time", "layer"), np.array([[[0.10, 0.15, 0.20, 0.25]]], dtype=float)),
        "gs_difference": (("location", "time", "layer"), np.array([[[0.02, 0.04, 0.06, 0.08]]], dtype=float)),
        "sk38": (("location", "time", "layer"), np.array([[[0.80, 1.10, 1.40, 1.70]]], dtype=float)),
    }
    coords = {
        "location": [0],
        "time": [pd.to_datetime("2025-01-15")],
        "layer": [0, 1, 2, 3],
    }
    return xsnowDataset(xr.Dataset(data, coords))


def test_compute_rta_attaches_expected_outputs(rta_ds):
    """Tests that compute_rta adds the expected outputs and preserves dims."""
    ds_with_rta = rta_ds.compute_rta(config=None, rta_threshold=0.5)
    ds_with_rta_cpu = ds_with_rta.to_cpu()

    assert isinstance(ds_with_rta_cpu, xsnowDataset)
    assert "rta" in ds_with_rta_cpu.data
    assert "rta_interface" in ds_with_rta_cpu.data
    assert "sk38_masked" in ds_with_rta_cpu.data

    assert ds_with_rta_cpu["rta"].dims == ("location", "time", "layer")
    assert ds_with_rta_cpu["rta_interface"].dims == ("location", "time", "layer")
    assert ds_with_rta_cpu["sk38_masked"].dims == ("location", "time", "layer")

    assert np.isclose(np.nanmax(ds_with_rta_cpu["rta"].values), 1.0)
    assert np.isclose(np.nanmax(ds_with_rta_cpu["rta_interface"].values), 1.0)
    assert np.isfinite(ds_with_rta_cpu["rta"].values).any()
    assert np.isfinite(ds_with_rta_cpu["rta_interface"].values).any()
