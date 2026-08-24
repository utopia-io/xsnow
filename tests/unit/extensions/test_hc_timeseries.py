"""
Unit tests for the pooled hazard-chart timeseries statistics.
"""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

# importing the module registers all methods on xsnowDataset
import xsnow.extensions.hazard_chart  # noqa: F401
from xsnow import xsnowDataset


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
        out['punstable_size_estimate'].values[0],
        np.interp(0.5, (0.0, 0.18, 0.35, 0.70, 1.50, 3.00), np.arange(6.0)),
    )


def test_class_override(summ):
    out = summ.compute_hc_timeseries(
        classes={'punstable': [('rather stable', 0.95),
                               ('poor', np.inf)]},
    ).data
    # nothing beyond 0.95 -> no unstable entries at all
    assert np.isnan(out['punstable_depth'].values).all()


def test_unknown_index(summ):
    with pytest.raises(ValueError, match='class config'):
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
