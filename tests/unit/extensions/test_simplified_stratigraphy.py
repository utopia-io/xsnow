"""
Unit tests for the simplified-stratigraphy reduction.
"""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

# importing the module registers all methods on xsnowDataset
import xsnow.extensions.datetags  # noqa: F401
import xsnow.extensions.hazard_chart  # noqa: F401
from xsnow import xsnowDataset

T1, T2 = np.datetime64('2025-11-01', 'ns'), np.datetime64('2025-12-06', 'ns')
NAT = np.datetime64('NaT', 'ns')


@pytest.fixture
def ds():
    """Five layers ground->surface: two tagged groups, shallow, new snow."""
    def lvar(values, dtype=float):
        return (('location', 'time', 'layer'),
                np.asarray(values, dtype=dtype)[None, None, :])

    data = xr.Dataset(
        {
            'height': lvar([0.1, 0.2, 0.3, 0.4, 0.5]),
            'grain_type': lvar([550, 440, 770, 420, 120]),
            'datetag': lvar([T1, T1, T2, NAT, NAT], 'datetime64[ns]'),
            'punstable': lvar([0.9, 0.5, np.nan, 0.3, 0.2]),
            'sk38': lvar([2.0, 0.8, 3.0, 4.0, 5.0]),
            'HS': (('location', 'time'), [[0.5]]),
        },
        coords={
            'location': ['LOC'],
            'time': [pd.to_datetime('2026-01-15')],
            'layer': np.arange(5),
        },
    )
    return xsnowDataset(data)


def _flat(red, var):
    return red.data[var].values[0, 0]


class TestReduction:
    def test_groups_and_order(self, ds):
        red = ds.compute_simplified_stratigraphy()
        assert list(red.data['datetag'].values) == [
            '2025-11-01', '2025-12-06', 'shallow_old_snow', 'new_snow'
        ]
        np.testing.assert_array_equal(
            _flat(red, 'datetag_layer_count'), [2, 1, 1, 1]
        )

    def test_punstable_golden(self, ds):
        red = ds.compute_simplified_stratigraphy()
        # weakest 0.9 (layer 0) is unstable -> deepest unstable depth 0.4 m
        # t2 group: NaN value but one layer (null-triplet case)
        # shallow/new: stable -> weakest layer's own depth
        np.testing.assert_allclose(
            _flat(red, 'punstable'), [0.9, np.nan, 0.3, 0.2]
        )
        np.testing.assert_allclose(
            _flat(red, 'punstable_depth'), [0.4, np.nan, 0.1, 0.0]
        )
        np.testing.assert_allclose(
            _flat(red, 'punstable_gcode_weak'), [550, np.nan, 420, 120]
        )
        np.testing.assert_allclose(
            _flat(red, 'punstable_gcode_deep'), [550, np.nan, 420, 120]
        )

    def test_min_side_index(self, ds):
        red = ds.compute_simplified_stratigraphy()
        # sk38 weakest = min 0.8 (layer 1, tag t1), unstable <= 1.0
        np.testing.assert_allclose(_flat(red, 'sk38'), [0.8, 3.0, 4.0, 5.0])
        np.testing.assert_allclose(
            _flat(red, 'sk38_depth'), [0.3, 0.2, 0.1, 0.0]
        )
        np.testing.assert_allclose(
            _flat(red, 'sk38_gcode_weak'), [440, 770, 420, 120]
        )

    def test_new_snow_beats_shallow_for_pp_df(self, ds):
        red = ds.compute_simplified_stratigraphy()
        counts = _flat(red, 'datetag_layer_count')
        # untagged PP layer (index 4) is new_snow, not shallow_old_snow
        assert counts[2] == 1 and counts[3] == 1

    def test_non_layer_vars_carried(self, ds):
        red = ds.compute_simplified_stratigraphy()
        assert 'HS' in red.data.data_vars
        assert red.sizes['layer'] == 4

    def test_unknown_index_raises(self, ds):
        with pytest.raises(ValueError, match='foo'):
            ds.compute_simplified_stratigraphy(indices=['foo'])

    def test_class_override(self, ds):
        red = ds.compute_simplified_stratigraphy(
            indices=['punstable'],
            classes={'punstable': [('stable', 0.95), ('unstable', np.inf)]},
        )
        # no poor-or-worse class -> depth of the weakest layer itself
        np.testing.assert_allclose(
            _flat(red, 'punstable_depth'), [0.4, np.nan, 0.1, 0.0]
        )


class TestSnpCharacteristics:
    def test_punstable_summary_golden(self, ds):
        summ = ds.compute_simplified_stratigraphy().compute_hc_points()
        d = summ.data
        assert float(d['punstable'].values[0, 0]) == pytest.approx(0.9)
        assert float(d['punstable_depth'].values[0, 0]) == pytest.approx(0.4)
        assert str(d['punstable_instability_drivenby'].values[0, 0]) == \
            '2025-11-01'
        assert str(d['punstable_depth_drivenby'].values[0, 0]) == '2025-11-01'
        assert float(d['punstable_gcode'].values[0, 0]) == 550
        assert float(d['punstable_size_estimate'].values[0, 0]) == \
            pytest.approx(2 + 0.05 / 0.35)

    def test_min_side_summary(self, ds):
        summ = ds.compute_simplified_stratigraphy().compute_hc_points()
        d = summ.data
        assert float(d['sk38'].values[0, 0]) == pytest.approx(0.8)
        assert float(d['sk38_depth'].values[0, 0]) == pytest.approx(0.3)
        assert float(d['sk38_size_estimate'].values[0, 0]) == \
            pytest.approx(1 + 0.12 / 0.17)

    def test_datetag_filter(self, ds):
        summ = ds.compute_simplified_stratigraphy().compute_hc_points(
            indices=['punstable'],
            datetags=['shallow_old_snow', 'new_snow'],
        )
        d = summ.data
        # 0.3 in shallow_old_snow, nothing unstable -> its own depth
        assert float(d['punstable'].values[0, 0]) == pytest.approx(0.3)
        assert float(d['punstable_depth'].values[0, 0]) == pytest.approx(0.1)
        assert str(d['punstable_depth_drivenby'].values[0, 0]) == \
            'shallow_old_snow'

    def test_all_filtered_gives_nan(self, ds):
        summ = ds.compute_simplified_stratigraphy().compute_hc_points(
            indices=['punstable'], datetags=['2025-12-06'],
        )
        d = summ.data
        assert np.isnan(d['punstable'].values[0, 0])
        assert str(d['punstable_instability_drivenby'].values[0, 0]) == ''

    def test_full_layer_needs_a_depth_variable(self, ds):
        with pytest.raises(ValueError, match="depth variable 'z'"):
            ds.compute_hc_points()

    def test_full_layer_input_reports_layer_indices(self, ds):
        data = ds.data.assign(
            z=ds.data['HS'] - ds.data['height'],
        )
        out = xsnowDataset(data).compute_hc_points(
            indices=['punstable'], depth_var='z',
        ).data
        # weakest is layer 0 (max side, 0.9), the only poor-or-worse layer
        assert float(out['punstable'].values[0, 0]) == pytest.approx(0.9)
        assert float(out['punstable_instability_drivenby'].values[0, 0]) == 0
        assert float(out['punstable_depth'].values[0, 0]) == pytest.approx(0.4)

    def test_depth_counted_up_is_flipped(self, ds):
        """A 'positive: up' depth variable is converted to depth below
        the surface, so the size estimate stays meaningful."""
        z_up = (ds.data['height'] - ds.data['HS']).assign_attrs(
            {'units': 'm', 'positive': 'up'}
        )
        out = xsnowDataset(ds.data.assign(z=z_up)).compute_hc_points(
            indices=['punstable'], depth_var='z',
        ).data
        assert float(out['punstable_depth'].values[0, 0]) == pytest.approx(0.4)
        assert float(out['punstable_size_estimate'].values[0, 0]) > 0

    def test_techel_size(self, ds):
        summ = ds.compute_simplified_stratigraphy().compute_hc_points(
            indices=['punstable'], size_estimate='Techel2023',
        )
        assert float(summ.data['punstable_size_estimate'].values[0, 0]) == \
            pytest.approx(2.0)
