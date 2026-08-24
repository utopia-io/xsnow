"""
Unit tests for the new-snow amounts.
"""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

# importing the module registers all methods on xsnowDataset
import xsnow.extensions.new_snow  # noqa: F401
from xsnow import xsnowDataset


@pytest.fixture
def ds():
    """Five layers ground->surface with PP/DF grains in the top two."""
    def lvar(values, dtype=float):
        return (('location', 'time', 'layer'),
                np.asarray(values, dtype=dtype)[None, None, :])

    data = xr.Dataset(
        {
            'height': lvar([0.1, 0.2, 0.3, 0.4, 0.5]),
            'grain_type': lvar([550, 440, 770, 420, 120]),
        },
        coords={
            'location': ['LOC'],
            'time': [pd.to_datetime('2026-01-15')],
            'layer': np.arange(5),
        },
    )
    return xsnowDataset(data)


class TestHnPpdf:
    def test_golden(self, ds):
        out = ds.compute_hn_ppdf()
        # code 120 (PP primary) and 420 (DF secondary) count, 0.1 m each;
        # the new_snow datetag group stays primary-only (see reduction tests)
        assert float(out.data['hn_ppdf'].values[0, 0]) == pytest.approx(0.2)


class TestHn72:
    def test_lag_and_clip(self):
        times = pd.date_range('2026-01-01', periods=4, freq='72h')
        data = xr.Dataset(
            {'HS_mod': (('location', 'time'), [[1.0, 1.4, 1.2, 1.5]])},
            coords={'location': ['LOC'], 'time': times},
        )
        out = xsnowDataset(data).compute_hn72().data['hn72'].values[0]
        # first step has no -72h sample; drops clip to 0
        np.testing.assert_allclose(out, [np.nan, 0.4, 0.0, 0.3])
