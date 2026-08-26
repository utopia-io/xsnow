"""
Unit tests for the hazard-chart variant engine.
"""
import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import xsnow.extensions.datetags  # noqa: F401
from dask.callbacks import Callback
from xsnow import xsnowDataset
from xsnow.extensions import get_array_backend
from xsnow.extensions.hazard_chart import HazardChart, _to_host

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
        assert get_array_backend(np.zeros(3)) is np
        assert get_array_backend(0.5) is np

    def test_cupy_arrays_dispatch_to_cupy(self):
        cupy = pytest.importorskip('cupy')

        class FakeDeviceArray:
            pass

        FakeDeviceArray.__module__ = 'cupy._core.core'
        assert get_array_backend(FakeDeviceArray()) is cupy

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

    monkeypatch.setattr(hazard_chart, 'get_array_backend',
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


class TestAxlim:
    def test_every_index_has_a_range(self):
        from xsnow.extensions.hazard_chart import _STABILITY_INDEX_INFO
        for name, cfg in _STABILITY_INDEX_INFO.items():
            lo, hi = cfg['axlim']
            assert lo < hi, name

    def test_plot_applies_the_range(self, full):
        import matplotlib
        matplotlib.use('Agg')
        hc = HazardChart(full).compute_hazard_chart(
            stability_index='punstable', depth_var='z',
        )
        ax = hc.plot_hazard_chart(hc_variant=0)
        assert ax.get_ylim() == (0.0, 1.0)

    def test_rc_slope_stays_an_intermediate(self):
        from xsnow.extensions.hazard_chart import _STABILITY_INDEX_INFO
        assert 'rc_slope' not in _STABILITY_INDEX_INFO
        assert 'rc_flat' in _STABILITY_INDEX_INFO


class TestTwoPanel:
    def test_scatter_panel_is_inherited_and_ranges_match(self, reduced):
        import matplotlib
        matplotlib.use('Agg')
        hc = HazardChart(reduced).compute_hazard_chart(indices=['punstable'])
        ax_hc, ax_ts = hc.plot_hazard_chart_timeseries(hc_variant=0)
        assert ax_hc.get_ylim() == (0.0, 1.0) == ax_ts.get_ylim()
        assert ax_hc.get_ylabel() == 'Punstable / -' == ax_ts.get_ylabel()
        assert ax_hc.get_title() == '' == ax_ts.get_title()
        assert len(ax_ts.collections) == 2  # 10-90 % and 25-75 % bands
        assert ax_ts.lines, 'median line missing'

    def test_paper_layout(self, reduced):
        import matplotlib
        matplotlib.use('Agg')
        hc = HazardChart(reduced).compute_hazard_chart(indices=['punstable'])
        ax_hc, ax_ts = hc.plot_hazard_chart_timeseries(
            hc_variant=0, time='2026-01-15',
        )
        # size-estimate ticks on the anchors, labelled 0-5
        assert [t.get_text() for t in ax_hc.get_xticklabels()] == \
            ['0', '1', '2', '3', '4', '5']
        assert ax_ts.yaxis.get_label_position() == 'right'
        assert [t.get_text() for t in ax_hc.texts] == ['a']
        assert [t.get_text() for t in ax_ts.texts] == ['b']
        assert ax_ts.get_xlabel() == ''
        # dashed timestamp mark on the right, green diamond on the left
        assert any(ln.get_linestyle() == '--' for ln in ax_ts.lines)
        diamonds = [ln for ln in ax_hc.lines if ln.get_marker() == 'D']
        assert diamonds and diamonds[0].get_color() == 'green'
        # shared y-axis and class-boundary lines in GUI colours
        assert ax_ts in ax_hc.get_shared_y_axes().get_siblings(ax_hc)
        bounds = {ln.get_ydata()[0]: ln.get_color() for ln in ax_hc.lines
                  if ln.get_linestyle() == '--' and len(ln.get_ydata()) == 2
                  and ln.get_ydata()[0] == ln.get_ydata()[1]}
        assert bounds == {0.5: '#ffffbf', 0.7: '#fdae61', 0.77: '#d7191c'}

    def test_depth_ticks_on_top(self, reduced):
        import matplotlib
        matplotlib.use('Agg')
        hc = HazardChart(reduced).compute_hazard_chart(indices=['punstable'])
        ax = hc.plot_hazard_chart(hc_variant=0, size_estimate='Techel2023')
        sec = [child for child in ax.get_children()
               if 'Secondary' in type(child).__name__]
        assert sec and sec[0].get_xlabel() == 'Depth / m'
        assert [t.get_text() for t in sec[0].get_xticklabels()] == \
            ['0.3', '0.4', '0.5', '0.8', '1']
        with pytest.raises(ValueError, match='size estimator'):
            hc.plot_hazard_chart(hc_variant=0, size_estimate='foo')

    def test_precomputed_timeseries_is_used(self, reduced):
        import matplotlib
        matplotlib.use('Agg')
        hc = HazardChart(reduced).compute_hazard_chart(indices=['punstable'])
        stats = reduced.compute_hc_timeseries(indices=['punstable'])
        _, ax_ts = hc.plot_hazard_chart_timeseries(hc_variant=0,
                                                   timeseries=stats)
        y = ax_ts.lines[0].get_ydata()
        np.testing.assert_allclose(
            y, stats.data['punstable_median'].values
        )


def test_points_coloured_by_grain_type(reduced):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.colors import to_hex
    hc = HazardChart(reduced).compute_hazard_chart(indices=['punstable'])
    ax = hc.plot_hazard_chart(hc_variant=0)
    face = ax.collections[0].get_facecolor()
    # one point per profile; its driving layer is the DH layer (550)
    assert [to_hex(c) for c in face] == ['#0078ff']


def _chunk_data_vars(xs, **chunks):
    """Chunk the data variables only, as xsnow.read leaves the small
    coordinates in numpy."""
    data = xs.data.copy()
    for name, var in xs.data.data_vars.items():
        data[name] = var.chunk({d: c for d, c in chunks.items()
                                if d in var.dims})
    return xsnowDataset(data)


class TestLazyInput:
    """The chart is built from blocks, so a chunked dataset stays lazy."""

    def test_stays_lazy_and_matches_the_eager_result(self, full):
        chunked = _chunk_data_vars(full, location=1, layer=-1)
        started = []
        with Callback(pretask=lambda key, *_: started.append(key)):
            lazy = HazardChart(chunked).compute_hazard_chart(
                stability_index='sk38', depth_var='z')
        assert started == []
        assert lazy.data['hc_instability'].chunks is not None
        eager = HazardChart(full).compute_hazard_chart(
            stability_index='sk38', depth_var='z')
        for name in ('hc_instability', 'hc_depth', 'hc_size_estimate'):
            np.testing.assert_allclose(
                lazy.data[name].values, eager.data[name].values, equal_nan=True)
        for name in ('hc_instabilityDrivenBy', 'hc_depthDrivenBy'):
            np.testing.assert_array_equal(
                lazy.data[name].values, eager.data[name].values)

    def test_datetag_provenance_stays_lazy(self, reduced):
        chunked = _chunk_data_vars(reduced, location=1, layer=-1)
        started = []
        with Callback(pretask=lambda key, *_: started.append(key)):
            lazy = HazardChart(chunked).compute_hazard_chart(
                stability_index='sk38')
        assert started == []
        assert lazy.data['hc_instabilityDrivenBy'].chunks is not None
        eager = HazardChart(reduced).compute_hazard_chart(stability_index='sk38')
        np.testing.assert_array_equal(
            lazy.data['hc_instabilityDrivenBy'].values,
            eager.data['hc_instabilityDrivenBy'].values)

    def test_grouping_on_a_lazy_dataset(self, reduced):
        data = reduced.data.assign_coords(
            region_id=('location', ['AT-07-01']))
        chunked = xsnowDataset(data.chunk({'location': 1, 'layer': -1}))
        chart = HazardChart(chunked).compute_hazard_chart(
            stability_index='sk38', groupby='region_id')
        assert 'hc_group_id' in chart.data
        assert np.isfinite(chart.data['hc_group_id'].values).any()
