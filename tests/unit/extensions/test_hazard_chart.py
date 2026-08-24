"""
Unit tests for the hazard-chart variant engine.
"""
import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import xsnow.extensions.datetags  # noqa: F401
from xsnow import xsnowDataset
from xsnow.extensions.hazard_chart import (
    HazardChart,
    _array_backend,
    _to_host,
)

T1, T2 = np.datetime64('2025-11-01', 'ns'), np.datetime64('2025-12-06', 'ns')
NAT = np.datetime64('NaT', 'ns')


def _lvar(values, dtype=float):
    return (('location', 'time', 'layer'),
            np.asarray(values, dtype=dtype)[None, None, :])


@pytest.fixture
def full():
    """Full-layer stratigraphy, five layers ground->surface."""
    data = xr.Dataset(
        {
            'height': _lvar([0.1, 0.2, 0.3, 0.4, 0.5]),
            'grain_type': _lvar([550, 440, 770, 420, 120]),
            'datetag': _lvar([T1, T1, T2, NAT, NAT], 'datetime64[ns]'),
            'punstable': _lvar([0.9, 0.5, np.nan, 0.3, 0.2]),
            'sk38': _lvar([2.0, 0.8, 3.0, 4.0, 5.0]),
            'z': _lvar([0.4, 0.3, 0.2, 0.1, 0.0]),
            'HS': (('location', 'time'), [[0.5]]),
        },
        coords={
            'location': ['LOC'],
            'time': [pd.to_datetime('2026-01-15')],
            'layer': np.arange(5),
        },
    )
    return xsnowDataset(data)


@pytest.fixture
def reduced(full):
    return full.compute_simplified_stratigraphy()


def _flat(ds, var, variant=0):
    return ds[var].isel(hc_variant=variant).values.ravel()[0]


class TestFullLayerInput:
    def test_layer_index_provenance(self, full):
        out = HazardChart(full).compute_hazard_chart(
            stability_index='punstable', depth_var='z',
        ).data
        # weakest (max side) is layer 0 at 0.9; it is the only poor-or-worse
        # layer, so it drives the depth too
        assert _flat(out, 'hc_instability') == pytest.approx(0.9)
        assert _flat(out, 'hc_instabilityDrivenBy') == 0
        assert _flat(out, 'hc_depthDrivenBy') == 0
        assert _flat(out, 'hc_depth') == pytest.approx(0.4)

    def test_size_estimate(self, full):
        out = HazardChart(full).compute_hazard_chart(
            stability_index='punstable', depth_var='z',
            size_estimate='Techel2023',
        ).data
        assert _flat(out, 'hc_size_estimate') == pytest.approx(
            np.interp(0.4, (0.0, 0.30, 0.40, 0.50, 0.80, 1.00), np.arange(6.0))
        )

    def test_datetags_need_reduced(self, full):
        with pytest.raises(ValueError, match='reduced layer dimension'):
            HazardChart(full).compute_hazard_chart(
                stability_index='punstable', depth_var='z',
                datetags=['new_snow'],
            )


class TestReducedInput:
    def test_datetag_provenance_and_per_index_depth(self, reduced):
        out = HazardChart(reduced).compute_hazard_chart(
            indices=['punstable'],
        ).data
        # the reduced entries name their driver instead of a layer index
        assert _flat(out, 'hc_instabilityDrivenBy') == '2025-11-01'
        assert out['hc_depth'].attrs['depth_var'] == 'punstable_depth'
        assert not np.isnan(_flat(out, 'hc_gcode'))

    def test_datetag_filter(self, reduced):
        out = HazardChart(reduced).compute_hazard_chart(
            indices=['punstable'], datetags=['new_snow'],
        ).data
        assert _flat(out, 'hc_instabilityDrivenBy') == 'new_snow'

    def test_empty_profile_fills_empty_string(self, reduced):
        out = HazardChart(reduced).compute_hazard_chart(
            indices=['punstable'], datetags=['2099-01-01'],
        ).data
        assert _flat(out, 'hc_instabilityDrivenBy') == ''
        assert np.isnan(_flat(out, 'hc_instability'))


class TestVariants:
    def test_one_variant_per_index(self, reduced):
        out = HazardChart(reduced).compute_hazard_chart(
            indices=['punstable', 'sk38'],
        ).data
        assert out.sizes['hc_variant'] == 2
        assert _flat(out, 'hc_instability', 0) == pytest.approx(0.9)
        assert _flat(out, 'hc_instability', 1) == pytest.approx(0.8)

    def test_variants_accumulate_across_calls(self, reduced):
        hc = HazardChart(reduced).compute_hazard_chart(
            indices=['punstable'],
        )
        out = hc.compute_hazard_chart(indices=['sk38']).data
        assert out.sizes['hc_variant'] == 2
        infos = HazardChart(xsnowDataset(out)).list_hc_variants()
        assert [i.stability_index for i in infos] == ['punstable', 'sk38']
        assert [i.idx for i in infos] == [0, 1]

    def test_index_and_indices_conflict(self, reduced):
        with pytest.raises(ValueError, match='not both'):
            HazardChart(reduced).compute_hazard_chart(
                stability_index='sk38', indices=['punstable'],
            )

    def test_unknown_index(self, reduced):
        with pytest.raises(KeyError, match='Unknown stability index'):
            HazardChart(reduced).compute_hazard_chart(indices=['foo'])


class TestGrouping:
    @pytest.fixture
    def grouped(self, full):
        data = full.data.assign_coords(region=('location', ['R1']))
        return xsnowDataset(data)

    def test_group_id_and_mapping(self, grouped):
        out = HazardChart(grouped).compute_hazard_chart(
            stability_index='punstable', depth_var='z', groupby='region',
        ).data
        assert _flat(out, 'hc_group_id') == 0
        mapping = json.loads(out['hc_group_id'].attrs['mapping'])
        assert mapping['0']['labels'] == {'0': 'R1'}
        assert mapping['0']['keys'] == ['region']

    def test_mapping_survives_multiple_indices(self, grouped):
        out = HazardChart(grouped).compute_hazard_chart(
            indices=['punstable', 'sk38'], depth_var='z', groupby='region',
        ).data
        assert out.sizes['hc_variant'] == 2
        mapping = json.loads(out['hc_group_id'].attrs['mapping'])
        assert set(mapping) == {'0', '1'}

    def test_subset_by_group_label(self, grouped):
        hc = HazardChart(grouped).compute_hazard_chart(
            stability_index='punstable', depth_var='z', groupby='region',
        )
        sub = hc.subset_hazard_points(hc_variant=0, group='R1')
        assert 'hc_size_estimate' in sub
        assert sub['hc_instability'].values.ravel()[0] == pytest.approx(0.9)

    def test_subset_without_grouping_raises(self, full):
        hc = HazardChart(full).compute_hazard_chart(
            stability_index='punstable', depth_var='z',
        )
        with pytest.raises(ValueError, match='without .groupby.'):
            hc.subset_hazard_points(hc_variant=0, group='R1')


class TestArrayBackend:
    def test_numpy_arrays_stay_on_numpy(self):
        assert _array_backend(np.zeros(3)) is np
        assert _array_backend(0.5) is np

    def test_cupy_arrays_dispatch_to_cupy(self):
        cupy = pytest.importorskip('cupy')

        class FakeDeviceArray:
            pass

        FakeDeviceArray.__module__ = 'cupy._core.core'
        assert _array_backend(FakeDeviceArray()) is cupy

    def test_to_host_returns_numpy(self):
        values = np.arange(4)
        np.testing.assert_array_equal(_to_host(values), values)

        class FakeDeviceArray:
            def get(self):
                return np.arange(4)

        np.testing.assert_array_equal(_to_host(FakeDeviceArray()), values)


def test_backend_stays_within_cupy_api(reduced, full, monkeypatch):
    """Execute on numpy restricted to the cupy API surface (no GPU
    required; runs only where cupy is installed)."""
    cupy = pytest.importorskip('cupy')

    class CupySurface:
        def __getattr__(self, name):
            if not hasattr(cupy, name):
                raise AttributeError(f"cupy has no attribute {name!r}")
            return getattr(np, name)

    from xsnow.extensions import hazard_chart

    monkeypatch.setattr(hazard_chart, 'get_backend', lambda ds: CupySurface())
    monkeypatch.setattr(hazard_chart, '_array_backend',
                        lambda values: CupySurface())

    out = HazardChart(full).compute_hazard_chart(
        stability_index='punstable', depth_var='z',
    ).data
    assert np.isfinite(_flat(out, 'hc_instability'))
    assert np.isfinite(_flat(out, 'hc_size_estimate'))

    points = reduced.compute_hc_points(indices=['punstable']).data
    assert np.isfinite(points['punstable'].values).any()
    assert points['punstable_instability_drivenby'].values.ravel()[0] != ''

    stats = reduced.compute_hc_timeseries(indices=['punstable']).data
    assert np.isfinite(stats['punstable_median'].values).any()
