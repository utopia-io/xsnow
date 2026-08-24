"""
Unit tests for the datetag assignment extension.
"""
import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

# importing the module registers all methods on xsnowDataset
import xsnow.extensions.datetags  # noqa: F401
from xsnow import xsnowDataset

TAGS = ['2025-10-04', '2025-11-01']


def _dt(*values):
    return np.array(
        [np.datetime64('NaT', 'ns') if v is None else np.datetime64(v, 'ns')
         for v in values]
    )


@pytest.fixture
def ds():
    """Two locations x six layers of deposition dates."""
    dob = np.stack([
        _dt('2025-10-04T21:45:00', '2025-10-05T02:00:00',
            '2025-11-01T00:00:00', '2025-11-02T12:00:00',
            None, '2025-09-01T06:00:00'),
        _dt('2025-10-03T00:00:00', '2025-10-22T09:00:00',
            '2025-12-24T18:00:00', None, None, None),
    ])
    data = xr.Dataset(
        {'deposition_date': (('location', 'time', 'layer'), dob[:, None, :])},
        coords={
            'location': ['VIR1A', 'VIR2A'],
            'time': [pd.to_datetime('2026-01-15')],
            'layer': np.arange(6),
            'region_id': ('location', ['AT-07-24', 'AT-07-25']),
            'band': ('location', ['above_treeline', 'below_treeline']),
        },
    )
    return xsnowDataset(data)


def _tags_of(ds_out, loc):
    return ds_out.data['datetag'].sel(location=loc).values[0]


class TestListInput:
    def test_golden_assignment(self, ds):
        out = ds.assign_datetags(TAGS)
        got = _tags_of(out, 'VIR1A')
        expected = _dt('2025-10-04',   # 21:45 floors onto the tag day
                       '2025-11-01',   # next tag up
                       '2025-11-01',   # exactly on the tag
                       None,           # after the last tag
                       None,           # no deposition date
                       '2025-10-04')   # before the first tag
        np.testing.assert_array_equal(got, expected)

    def test_unsorted_tags_are_sorted(self, ds):
        out = ds.assign_datetags(list(reversed(TAGS)))
        np.testing.assert_array_equal(
            _tags_of(out, 'VIR1A'), _tags_of(ds.assign_datetags(TAGS), 'VIR1A')
        )

    def test_dtype_and_attrs(self, ds):
        da = ds.assign_datetags(TAGS).data['datetag']
        assert da.dtype == np.dtype('datetime64[ns]')
        assert 'units' not in da.attrs
        assert da.attrs['standard_name'] == 'deposition_datetag'


class TestMappingInput:
    MAPPING = {
        'unique_datetags': TAGS + ['2025-10-22'],
        'meta': {'level1': 'region_id', 'level2': 'band'},
        'AT-07-24': {'above_treeline': TAGS},
        'AT-07-25': {'below_treeline': ['2025-10-22']},
    }

    def test_per_location_tags(self, ds):
        out = ds.assign_datetags(self.MAPPING)
        np.testing.assert_array_equal(
            _tags_of(out, 'VIR1A'),
            _dt('2025-10-04', '2025-11-01', '2025-11-01', None, None,
                '2025-10-04'),
        )
        np.testing.assert_array_equal(
            _tags_of(out, 'VIR2A'),
            _dt('2025-10-22', '2025-10-22', None, None, None, None),
        )

    def test_missing_pair_stays_untagged(self, ds):
        mapping = {'meta': self.MAPPING['meta'],
                   'AT-07-24': {'above_treeline': TAGS}}
        out = ds.assign_datetags(mapping)
        assert pd.isna(_tags_of(out, 'VIR2A')).all()

    def test_missing_coord_raises(self, ds):
        bad = {'meta': {'level1': 'region_id', 'level2': 'elev_band'}}
        with pytest.raises(ValueError, match='elev_band'):
            ds.assign_datetags(bad)

    def test_json_file_input(self, ds, tmp_path):
        path = tmp_path / 'datetags.json'
        path.write_text(json.dumps(self.MAPPING))
        out = ds.assign_datetags(str(path))
        np.testing.assert_array_equal(
            _tags_of(out, 'VIR2A'),
            _dt('2025-10-22', '2025-10-22', None, None, None, None),
        )
