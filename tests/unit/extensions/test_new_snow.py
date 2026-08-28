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
        # code 120 (PP primary) and 420 (DF secondary) count, 0.1 m each
        assert float(out.data['hn_ppdf'].values[0, 0]) == pytest.approx(0.2)


@pytest.fixture
def heights():
    times = pd.date_range('2026-01-01', periods=4, freq='72h')
    return xr.Dataset(
        {'HS': (('location', 'time'), [[1.0, 1.4, 1.2, 1.5]]),
         'HS_mod': (('location', 'time'), [[1.0, 1.5, 1.2, 1.5]])},
        coords={'location': ['LOC'], 'time': times},
    )


class TestHn:
    def test_lag_and_clip(self, heights):
        out = xsnowDataset(heights).compute_hn(72).data['hn72']
        # first step has no -72h sample; drops clip to 0
        np.testing.assert_allclose(out.values[0], [np.nan, 0.4, 0.0, 0.3])
        assert out.attrs == {'units': 'm', 'long_name': '72 h snow height increase'}

    def test_height_variable(self, heights):
        out = xsnowDataset(heights).compute_hn(72, height_var='HS_mod').data
        np.testing.assert_allclose(out['hn72'].values[0], [np.nan, 0.5, 0.0, 0.3])

    def test_window(self, heights):
        hourly = heights.assign_coords(
            time=pd.date_range('2026-01-01', periods=4, freq='24h'))
        out = xsnowDataset(hourly).compute_hn(24).data
        assert 'hn24' in out
        np.testing.assert_allclose(out['hn24'].values[0], [np.nan, 0.4, 0.0, 0.3])
        # no sample exactly 72 h earlier until the fourth step
        out = xsnowDataset(hourly).compute_hn(72).data
        np.testing.assert_allclose(out['hn72'].values[0], [np.nan] * 3 + [0.5])
        for bad in (0, 24.5):
            with pytest.raises(ValueError, match='whole number'):
                xsnowDataset(hourly).compute_hn(bad)

    def test_hs_is_derived_from_the_layers(self, ds):
        times = pd.date_range('2026-01-15', periods=2, freq='72h')
        data = xr.concat([ds.data, ds.data.assign(height=ds.data['height'] + 0.1)],
                         dim='time').assign_coords(time=times)
        out = xsnowDataset(data).compute_hn(72).data
        assert 'HS' in out
        np.testing.assert_allclose(out['hn72'].values[0], [np.nan, 0.1])
