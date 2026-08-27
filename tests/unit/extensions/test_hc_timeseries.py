"""
Unit tests for the pooled hazard-chart timeseries statistics.
"""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

# importing the module registers all methods on xsnowDataset
from xsnow import xsnowDataset
from xsnow.extensions.hazard_chart import (
    XAXIS_ANCHORS,
    register_stability_index,
    stability_indices,
)


@pytest.fixture(autouse=True, scope='module')
def _punstable_classes():
    if 'punstable' not in stability_indices():
        register_stability_index(
            'punstable', weak='max', axlim=(0.0, 1.0), label='Punstable',
            classes=[('rather stable', 0.5), ('fair', 0.7), ('poor', 0.77),
                     ('very poor', np.inf)],
        )


@pytest.fixture
def summ():
    """Summarized dataset: 3 locations x 2 times x 1 slope."""
    values = [[[0.9], [0.5]], [[0.3], [np.nan]], [[0.8], [0.1]]]
    depths = [[[0.4], [0.3]], [[0.2], [np.nan]], [[0.6], [0.1]]]
    data = xr.Dataset(
        {
            'punstable': (('location', 'time', 'slope'), values),
            'punstable_depth': (('location', 'time', 'slope'), depths),
        },
        coords={
            'location': ['A', 'B', 'C'],
            'time': pd.to_datetime(['2026-02-01', '2026-02-02']),
            'slope': [0],
        },
    )
    return xsnowDataset(data)


def test_percentiles_and_depth(summ):
    out = summ.compute_hc_timeseries().data
    # t0 values [0.9, 0.3, 0.8]; unstable (poor or worse, > 0.7): 0.9, 0.8
    np.testing.assert_allclose(out['punstable_median'].values, [0.8, 0.3])
    np.testing.assert_allclose(
        out['punstable_p25'].values, [np.percentile([0.9, 0.3, 0.8], 25), 0.2]
    )
    np.testing.assert_allclose(out['punstable_depth'].values[0], 0.5)
    assert np.isnan(out['punstable_depth'].values[1])
    np.testing.assert_allclose(
        out['punstable_xaxis'].values[0],
        np.interp(0.5, XAXIS_ANCHORS['McClung2009'], np.arange(6.0)),
    )
    assert out['punstable_xaxis'].attrs['xaxis_scaling'] == 'McClung2009'


def test_class_override(summ):
    out = summ.compute_hc_timeseries(
        classes={'punstable': [('rather stable', 0.95),
                               ('poor', np.inf)]},
    ).data
    # nothing beyond 0.95 -> no unstable entries at all
    assert np.isnan(out['punstable_depth'].values).all()


def test_at_time_subset(summ):
    out = summ.compute_hc_timeseries(at_time='2026-02-02').data
    assert out.sizes['time'] == 1
    np.testing.assert_allclose(out['punstable_median'].values, [0.3])
    out = summ.compute_hc_timeseries(at_time=slice('2026-02-01', '2026-02-02')).data
    assert out.sizes['time'] == 2


def test_unknown_index(summ):
    with pytest.raises(ValueError, match='No summarized variables'):
        summ.compute_hc_timeseries(indices=['foo'])


@pytest.fixture
def reduced():
    """Reduced dataset: 2 locations x 1 time x 1 slope x 2 datetag entries."""
    dims = ('location', 'time', 'slope', 'layer')
    values = [[[[0.9, 0.3]]], [[[0.5, 0.2]]]]
    depths = [[[[0.4, 0.1]]], [[[0.3, 0.0]]]]
    data = xr.Dataset(
        {
            'punstable': (dims, values),
            'punstable_depth': (dims, depths),
            'punstable_gcode_weak': (dims, depths),
        },
        coords={
            'location': ['A', 'B'],
            'time': pd.to_datetime(['2026-02-01']),
            'slope': [0],
            'layer': np.arange(2),
            'datetag': ('layer', ['2026-01-28', 'new_snow']),
        },
    )
    return xsnowDataset(data)


def test_reduced_input_summarizes(reduced):
    out = reduced.compute_hc_timeseries().data
    # weakest per profile: A 0.9 (poor or worse, depth 0.4), B 0.5
    np.testing.assert_allclose(out['punstable_median'].values, [0.7])
    np.testing.assert_allclose(out['punstable_depth'].values, [0.4])


def test_reduced_datetag_filter(reduced):
    out = reduced.compute_hc_timeseries(datetags=['new_snow']).data
    # only the entries [0.3, 0.2] remain; nothing poor or worse
    np.testing.assert_allclose(out['punstable_median'].values, [0.25])
    assert np.isnan(out['punstable_depth'].values).all()


def test_datetags_on_summarized_raises(summ):
    with pytest.raises(ValueError, match='reduced layer dimension'):
        summ.compute_hc_timeseries(datetags=['new_snow'])


def test_full_stratigraphy_input():
    """A full-layer dataset is summarized through compute_hc_points with
    the given depth variable."""
    dims = ('location', 'time', 'layer')
    data = xr.Dataset(
        {
            'punstable': (dims, [[[0.9, 0.5, 0.2]], [[0.1, 0.8, 0.3]]]),
            'z': (dims, [[[0.4, 0.2, 0.0]], [[0.6, 0.3, 0.0]]]),
        },
        coords={'location': ['A', 'B'],
                'time': pd.to_datetime(['2026-02-01']), 'layer': np.arange(3)},
    )
    out = xsnowDataset(data).compute_hc_timeseries(indices=['punstable']).data
    # weakest per profile 0.9 (depth 0.4) and 0.8 (depth 0.3): both unstable
    np.testing.assert_allclose(out['punstable_median'].values, [0.85])
    np.testing.assert_allclose(out['punstable_depth'].values, [0.35])
    with pytest.raises(ValueError, match="depth variable 'depth'"):
        xsnowDataset(data).compute_hc_timeseries(depth_var='depth')
