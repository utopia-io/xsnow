"""
Unit tests for the hazard-chart variant engine.
"""
import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from dask.callbacks import Callback

from xsnow import xsnowDataset
from xsnow.extensions import get_backend
from xsnow.extensions.hazard_chart import (
    XAXIS_ANCHORS,
    HazardChart,
    _compute_variant_timeseries,
    _to_host,
    register_stability_index,
    stability_index_info,
    stability_indices,
)

T1, T2 = np.datetime64('2025-11-01', 'ns'), np.datetime64('2025-12-06', 'ns')
NAT = np.datetime64('NaT', 'ns')
PUNSTABLE_CLASSES = [
    ('rather stable', 0.5), ('fair', 0.7), ('poor', 0.77), ('very poor', np.inf),
]


@pytest.fixture(autouse=True, scope='module')
def _punstable_classes():
    """The punstable package registers its classes on import; the tests
    must not depend on it being installed."""
    if 'punstable' not in stability_indices():
        register_stability_index('punstable', weak='max',
                                 classes=PUNSTABLE_CLASSES, axlim=(0.0, 1.0),
                                 label='Punstable')


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
def reduced():
    """The datetag form of ``full``: one entry per layer group (the two
    tags, shallow old snow, new snow) with the per-index depth and grain
    codes, as the ``datetags`` extension reduces it."""
    data = xr.Dataset(
        {
            'punstable': _lvar([0.9, np.nan, 0.3, 0.2]),
            'punstable_depth': _lvar([0.4, np.nan, 0.1, 0.0]),
            'punstable_gcode_weak': _lvar([550, np.nan, 420, 120]),
            'sk38': _lvar([0.8, 3.0, 4.0, 5.0]),
            'sk38_depth': _lvar([0.3, 0.2, 0.1, 0.0]),
            'sk38_gcode_weak': _lvar([440, 770, 420, 120]),
            'datetag_layer_count': _lvar([2, 1, 1, 1], 'int16'),
            'HS': (('location', 'time'), [[0.5]]),
        },
        coords={
            'location': ['LOC'],
            'time': [pd.to_datetime('2026-01-15')],
            'layer': np.arange(4),
            'datetag': ('layer', ['2025-11-01', '2025-12-06',
                                  'shallow_old_snow', 'new_snow']),
        },
    )
    return xsnowDataset(data)


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

    def test_xaxis_scaling(self, full):
        out = HazardChart(full).compute_hazard_chart(
            stability_index='punstable', depth_var='z',
            xaxis_scaling='Techel2023',
        ).data
        assert _flat(out, 'hc_xaxis') == pytest.approx(
            np.interp(0.4, XAXIS_ANCHORS['Techel2023'], np.arange(6.0))
        )
        assert out['hc_xaxis'].attrs['xaxis_scaling'] == 'Techel2023'

    def test_variant_records_its_timestamps(self, full):
        hc = HazardChart(full).compute_hazard_chart(
            stability_index='punstable', depth_var='z',
        )
        assert hc.list_hc_variants()[0].time is None
        hc = hc.compute_hazard_chart(
            stability_index='sk38', depth_var='z', at_time='2026-01-15',
        )
        infos = hc.list_hc_variants()
        assert infos[0].time is None
        assert infos[1].time == ('2026-01-15T00:00:00',)


class TestReducedInput:
    def test_per_index_depth_and_layer_provenance(self, reduced):
        out = HazardChart(reduced).compute_hazard_chart(
            indices=['punstable'],
        ).data
        assert _flat(out, 'hc_instabilityDrivenBy') == 0
        assert out['hc_depth'].attrs['depth_var'] == 'punstable_depth'
        assert _flat(out, 'hc_gcode') == 550



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
        assert [i.xaxis_scaling for i in infos] == ['McClung2009'] * 2

    def test_index_and_indices_conflict(self, reduced):
        with pytest.raises(ValueError, match='not both'):
            HazardChart(reduced).compute_hazard_chart(
                stability_index='sk38', indices=['punstable'],
            )

    def test_unknown_index(self, reduced):
        with pytest.raises(KeyError, match='Unknown stability index'):
            HazardChart(reduced).compute_hazard_chart(indices=['foo'])


class TestLayerContext:
    def test_requested_values_create_one_variant_each(self, full):
        hc = HazardChart(full).compute_hazard_chart(
            indices=['sk38'], depth_var='z',
            layer_context={'datetag': ['2025-11-01', '2025-12-06']},
        )

        assert hc.data.sizes['hc_variant'] == 2
        assert [_flat(hc.data, 'hc_instability', i) for i in range(2)] == \
            pytest.approx([0.8, 3.0])
        assert [info.layer_context['value']
                for info in hc.list_hc_variants()] == [
                    '2025-11-01T00:00:00', '2025-12-06T00:00:00',
                ]

    def test_variable_name_enumerates_unique_nonmissing_values(self, full):
        hc = HazardChart(full).compute_hazard_chart(
            indices=['sk38'], depth_var='z', layer_context='datetag',
        )
        assert hc.data.sizes['hc_variant'] == 2
        assert [info.layer_context['value']
                for info in hc.list_hc_variants()] == [
                    '2025-11-01T00:00:00', '2025-12-06T00:00:00',
                ]

    def test_contexts_expand_each_index_in_order(self, full):
        hc = HazardChart(full).compute_hazard_chart(
            indices=['punstable', 'sk38'], depth_var='z',
            layer_context={'datetag': ['2025-11-01', '2025-12-06']},
        )
        infos = hc.list_hc_variants()
        assert [info.stability_index for info in infos] == [
            'punstable', 'punstable', 'sk38', 'sk38',
        ]

    def test_unmatched_value_is_an_empty_variant(self, full):
        hc = HazardChart(full).compute_hazard_chart(
            indices=['sk38'], depth_var='z',
            layer_context={'datetag': ['2099-01-01']},
        )
        assert np.isnan(_flat(hc.data, 'hc_instability'))

    def test_boolean_array_is_one_mask(self, full):
        mask = xr.DataArray(
            [False, False, True, False, False], dims=['layer'],
            name='persistent_layers',
        )
        hc = HazardChart(full).compute_hazard_chart(
            indices=['sk38'], depth_var='z', layer_context=mask,
        )
        assert _flat(hc.data, 'hc_instability') == pytest.approx(3.0)
        assert hc.list_hc_variants()[0].layer_context == {
            'kind': 'mask', 'variable': 'persistent_layers',
        }

    def test_categorical_array_creates_variants(self, full):
        context = xr.DataArray(
            ['old', 'old', 'persistent', None, None],
            dims=['layer'], name='custom_problem',
        )
        hc = HazardChart(full).compute_hazard_chart(
            indices=['sk38'], depth_var='z', layer_context=context,
        )
        assert [_flat(hc.data, 'hc_instability', i) for i in range(2)] == \
            pytest.approx([0.8, 3.0])

    def test_datetime_context_keeps_subsecond_values_distinct(self, full):
        context = xr.DataArray(
            np.array([
                '2026-01-01T00:00:00.100',
                '2026-01-01T00:00:00.200',
                'NaT', 'NaT', 'NaT',
            ], dtype='datetime64[ns]'),
            dims=['layer'], name='event_time',
        )
        hc = HazardChart(full).compute_hazard_chart(
            indices=['sk38'], depth_var='z', layer_context=context,
        )
        assert [info.layer_context['value']
                for info in hc.list_hc_variants()] == [
                    '2026-01-01T00:00:00.1',
                    '2026-01-01T00:00:00.2',
                ]

    def test_only_one_context_variable_is_supported(self, full):
        with pytest.raises(ValueError, match='exactly one context variable'):
            HazardChart(full).compute_hazard_chart(
                indices=['sk38'], depth_var='z',
                layer_context={'datetag': [T1], 'other': ['x']},
            )

    def test_variant_timeseries_uses_selected_context_and_classes(self, full):
        custom = [
            ('very poor', 0.1), ('poor', 0.5),
            ('fair', 1.0), ('rather stable', np.inf),
        ]
        hc = HazardChart(full).compute_hazard_chart(
            indices=['sk38'], depth_var='z', classes={'sk38': custom},
            layer_context={'datetag': ['2025-11-01', '2025-12-06']},
        )
        infos = hc.list_hc_variants()
        assert infos[0].classes == tuple(custom)
        medians = [
            _compute_variant_timeseries(hc.data, info).data['sk38_median'].item()
            for info in infos
        ]
        assert medians == pytest.approx([0.8, 3.0])
        assert np.isnan(
            _compute_variant_timeseries(hc.data, infos[0])
            .data['sk38_depth'].item()
        )
        raw = hc.data.attrs['hc_variants']
        assert 'Infinity' not in raw
        json.loads(
            raw,
            parse_constant=lambda value: pytest.fail(
                f'non-standard JSON constant: {value}'
            ),
        )


class TestGrouping:
    @pytest.fixture
    def grouped(self, full):
        data = full.data.assign_coords(region=('location', ['R1']))
        return xsnowDataset(data)

    @pytest.fixture
    def grouped_pair(self, reduced):
        weaker = reduced.data.assign(
            sk38=xr.full_like(reduced.data['sk38'], 0.4),
        )
        data = xr.concat([reduced.data, weaker], dim='location')
        data = data.assign_coords(
            location=['L1', 'L2'], region=('location', ['R1', 'R2']),
        )
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
        assert 'hc_xaxis' in sub
        assert sub['hc_instability'].values.ravel()[0] == pytest.approx(0.9)

    def test_subset_without_grouping_raises(self, full):
        hc = HazardChart(full).compute_hazard_chart(
            stability_index='punstable', depth_var='z',
        )
        with pytest.raises(ValueError, match='without .groupby.'):
            hc.subset_hazard_points(hc_variant=0, group='R1')

    def test_ungrouped_variant_stays_ungrouped_after_grouped_variant(self, grouped):
        hc = HazardChart(grouped).compute_hazard_chart(
            indices=['punstable'], depth_var='z', groupby='region',
        ).compute_hazard_chart(indices=['sk38'], depth_var='z')

        assert 'hc_group_id' in hc.data
        with pytest.raises(ValueError, match='without .groupby.'):
            hc.subset_hazard_points(hc_variant=1, group='R1')

    def test_variant_timeseries_aggregates_the_selected_group(self, grouped_pair):
        hc = HazardChart(grouped_pair).compute_hazard_chart(
            indices=['sk38'], groupby='region',
        )
        info = hc.list_hc_variants()[0]

        all_points = _compute_variant_timeseries(hc.data, info).data
        r1 = _compute_variant_timeseries(hc.data, info, group='R1').data
        r2 = _compute_variant_timeseries(hc.data, info, group='R2').data
        assert all_points['sk38_median'].item() == pytest.approx(0.6)
        assert r1['sk38_median'].item() == pytest.approx(0.8)
        assert r2['sk38_median'].item() == pytest.approx(0.4)

    def test_slope_and_realization_remain_point_dimensions(self, full):
        data = full.data.expand_dims(
            slope=['north', 'south'], realization=['r1', 'r2', 'r3'],
        )
        hc = HazardChart(xsnowDataset(data)).compute_hazard_chart(
            indices=['sk38'], depth_var='z',
            groupby=['slope', 'realization'],
        )
        point = hc.data['hc_instability'].isel(hc_variant=0)
        assert point.sizes['slope'] == 2
        assert point.sizes['realization'] == 3


class TestRegistry:
    def test_builtin_indices(self):
        names = stability_indices()
        for name in ('sk38', 'sk38_rta', 'sk38_ccl', 'sk38_ccl_rta', 'rta',
                     'rc_flat', 'lwc'):
            assert name in names
        assert 'rc_slope' not in names      # an sk38_ccl intermediate

    def test_classes_come_from_the_index_modules(self):
        from xsnow.extensions import critical_crack_length, rta, sk38
        assert stability_index_info('sk38')['classes'] == sk38.SK38_CLASSES
        assert stability_index_info('sk38_ccl')['weak'] == sk38.SK38_WEAK
        assert stability_index_info('rta')['classes'] == rta.RTA_CLASSES
        assert stability_index_info('rta')['classes'][0][1] == rta.RTA_THRESHOLD
        assert stability_index_info('rc_flat')['classes'] == \
            critical_crack_length.RC_CLASSES

    def test_every_index_has_a_range_and_ordered_classes(self):
        for name in stability_indices():
            cfg = stability_index_info(name)
            lo, hi = cfg['axlim']
            assert lo < hi, name
            bounds = [b for _, b in cfg['classes']]
            assert bounds == sorted(bounds) and bounds[-1] == np.inf, name

    def test_register_validates(self):
        with pytest.raises(ValueError, match="'min' or 'max'"):
            register_stability_index('bad', weak='low', classes=[('poor', np.inf)],
                                     axlim=(0, 1))
        with pytest.raises(ValueError, match='Unknown class labels'):
            register_stability_index('bad', weak='min',
                                     classes=[('unstable', np.inf)], axlim=(0, 1))
        assert 'bad' not in stability_indices()

    def test_unknown_index(self):
        with pytest.raises(KeyError, match='Unknown stability index'):
            stability_index_info('no_such_index')

    def test_external_package_registers_on_import(self, tmp_path, monkeypatch):
        (tmp_path / 'extidx.py').write_text(
            "from xsnow.extensions.hazard_chart import register_stability_index\n"
            "register_stability_index('extidx', weak='min',\n"
            "                         classes=[('poor', 1.0), ('rather stable', float('inf'))],\n"
            "                         axlim=(0.0, 2.0))\n"
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        assert stability_index_info('extidx')['weak'] == 'min'


class TestArrayBackend:
    def test_numpy_arrays_stay_on_numpy(self):
        assert get_backend(np.zeros(3)) is np
        assert get_backend(0.5) is np

    def test_cupy_arrays_dispatch_to_cupy(self):
        cupy = pytest.importorskip('cupy')

        class FakeDeviceArray:
            pass

        FakeDeviceArray.__module__ = 'cupy._core.core'
        assert get_backend(FakeDeviceArray()) is cupy

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

    monkeypatch.setattr(hazard_chart, 'get_backend',
                        lambda values: CupySurface())

    out = HazardChart(full).compute_hazard_chart(
        stability_index='punstable', depth_var='z',
    ).data
    assert np.isfinite(_flat(out, 'hc_instability'))
    assert np.isfinite(_flat(out, 'hc_xaxis'))

    points = reduced.compute_hc_points(indices=['punstable']).data
    assert np.isfinite(points['punstable'].values).any()
    assert points['punstable_instability_drivenby'].values.ravel()[0] != ''

    stats = reduced.compute_hc_timeseries(indices=['punstable']).data
    assert np.isfinite(stats['punstable_median'].values).any()


class TestPoints:
    def test_reduced_golden(self, reduced):
        d = reduced.compute_hc_points().data
        assert 'layer' not in d.dims
        assert float(d['punstable'].values[0, 0]) == pytest.approx(0.9)
        assert float(d['punstable_depth'].values[0, 0]) == pytest.approx(0.4)
        assert str(d['punstable_instability_drivenby'].values[0, 0]) == '2025-11-01'
        assert float(d['punstable_gcode'].values[0, 0]) == 550
        assert float(d['punstable_xaxis'].values[0, 0]) == \
            pytest.approx(2 + 0.05 / 0.35)
        assert float(d['sk38'].values[0, 0]) == pytest.approx(0.8)
        assert float(d['sk38_xaxis'].values[0, 0]) == pytest.approx(1 + 0.12 / 0.17)

    def test_datetag_filter(self, reduced):
        d = reduced.compute_hc_points(
            indices=['punstable'], datetags=['new_snow'],
        ).data
        assert str(d['punstable_instability_drivenby'].values[0, 0]) == 'new_snow'
        assert float(d['punstable'].values[0, 0]) == pytest.approx(0.2)

    def test_all_filtered_gives_an_empty_label(self, reduced):
        d = reduced.compute_hc_points(
            indices=['punstable'], datetags=['2099-01-01'],
        ).data
        assert str(d['punstable_instability_drivenby'].values[0, 0]) == ''
        assert np.isnan(d['punstable'].values[0, 0])

    def test_datetags_need_reduced(self, full):
        with pytest.raises(ValueError, match='reduced layer dimension'):
            full.compute_hc_points(
                indices=['punstable'], depth_var='z', datetags=['new_snow'],
            )

    def test_full_layer_reports_layer_indices(self, full):
        d = full.compute_hc_points(indices=['punstable'], depth_var='z').data
        assert float(d['punstable_instability_drivenby'].values[0, 0]) == 0
        assert float(d['punstable_depth'].values[0, 0]) == pytest.approx(0.4)

    def test_full_layer_needs_a_depth_variable(self, full):
        with pytest.raises(ValueError, match="depth variable 'foo'"):
            full.compute_hc_points(depth_var='foo')


class TestPlots:
    @pytest.fixture(autouse=True)
    def _agg(self):
        import matplotlib
        matplotlib.use('Agg')

    def test_plot_applies_the_range(self, full):
        hc = HazardChart(full).compute_hazard_chart(
            stability_index='punstable', depth_var='z',
        )
        ax = hc.plot_hazard_chart(hc_variant=0)
        assert ax.get_ylim() == (0.0, 1.0)

    def test_depth_axis_without_size_labels(self, reduced):
        hc = HazardChart(reduced).compute_hazard_chart(indices=['punstable'])
        ax = hc.plot_hazard_chart(hc_variant=0)
        # the evenly spaced ticks are labelled with the anchor depths
        assert [t.get_text() for t in ax.get_xticklabels()] == \
            ['0', '18', '35', '70', '150', '300']
        assert ax.get_xlabel() == 'Depth / cm'
        assert not [c for c in ax.get_children()
                    if 'Secondary' in type(c).__name__]
        assert 'size' not in (ax.get_xlabel() + ax.get_ylabel()).lower()
        ax = hc.plot_hazard_chart(hc_variant=0, xaxis_scaling='Techel2023')
        assert [t.get_text() for t in ax.get_xticklabels()] == \
            ['0', '30', '40', '50', '80', '100']
        with pytest.raises(ValueError, match='x-axis scaling'):
            hc.plot_hazard_chart(hc_variant=0, xaxis_scaling='foo')

    def test_points_coloured_by_grain_type(self, reduced):
        from matplotlib.colors import to_hex
        hc = HazardChart(reduced).compute_hazard_chart(indices=['punstable'])
        ax = hc.plot_hazard_chart(hc_variant=0)
        face = ax.collections[0].get_facecolor()
        # one point per profile; its driving layer is the DH layer (550)
        assert [to_hex(c) for c in face] == ['#0078ff']

    def test_single_chart_is_square(self, reduced):
        hc = HazardChart(reduced).compute_hazard_chart(indices=['punstable'])
        ax = hc.plot_hazard_chart(hc_variant=0)
        assert ax.get_box_aspect() == pytest.approx(1.0)
        w, h = ax.figure.get_size_inches()
        assert w == pytest.approx(h)

    def test_timeseries_panel_alone(self, reduced):
        hc = HazardChart(reduced).compute_hazard_chart(indices=['punstable'])
        ax = hc.plot_hc_timeseries(hc_variant=0)
        w, h = ax.figure.get_size_inches()
        assert w / h > 1.6          # wide: a timeseries
        assert len(ax.collections) == 2  # 10-90 % and 25-75 % bands
        assert ax.lines, 'median line missing'
        assert ax.yaxis.get_label_position() == 'left'
        assert ax.get_ylabel() == 'Punstable / -'
        assert ax.get_ylim() == (0.0, 1.0)
        ax = hc.plot_hc_timeseries(hc_variant=0, label_side='right')
        assert ax.yaxis.get_label_position() == 'right'
        with pytest.raises(ValueError, match='label_side'):
            hc.plot_hc_timeseries(hc_variant=0, label_side='top')

    def test_two_panel_layout(self, reduced):
        hc = HazardChart(reduced).compute_hazard_chart(indices=['punstable'])
        ax_hc, ax_ts = hc.plot_hc_with_timeseries(
            hc_variant=0, time='2026-01-15',
        )
        assert ax_hc.get_ylim() == (0.0, 1.0) == ax_ts.get_ylim()
        assert ax_hc.get_ylabel() == 'Punstable / -' == ax_ts.get_ylabel()
        assert ax_hc.get_title() == '' == ax_ts.get_title()
        assert [t.get_text() for t in ax_hc.get_xticklabels()] == \
            ['0', '18', '35', '70', '150', '300']
        assert ax_ts.yaxis.get_label_position() == 'right'
        assert [t.get_text() for t in ax_hc.texts] == ['a']
        assert [t.get_text() for t in ax_ts.texts] == ['b']
        assert ax_ts.get_xlabel() == ''
        # dashed timestamp mark on the right, green diamond on the left
        assert any(ln.get_linestyle() == '--' for ln in ax_ts.lines)
        diamonds = [ln for ln in ax_hc.lines if ln.get_marker() == 'D']
        assert diamonds and diamonds[0].get_color() == 'green'
        # shared y-axis and class-boundary lines in the class colours
        assert ax_ts in ax_hc.get_shared_y_axes().get_siblings(ax_hc)
        bounds = {ln.get_ydata()[0]: ln.get_color() for ln in ax_hc.lines
                  if ln.get_linestyle() == '--' and len(ln.get_ydata()) == 2
                  and ln.get_ydata()[0] == ln.get_ydata()[1]}
        assert bounds == {0.5: '#ffffbf', 0.7: '#fdae61', 0.77: '#d7191c'}

    def test_marker_pools_with_the_variant_depth_variable(self, full):
        depth = (full.data['HS'] - full.data['height']).assign_attrs(units='m')
        ds = xsnowDataset(full.data.assign(below=depth))
        hc = HazardChart(ds).compute_hazard_chart(
            stability_index='punstable', depth_var='below',
        )
        ax_hc, ax_ts = hc.plot_hc_with_timeseries(
            hc_variant=0, time='2026-01-15',
        )
        diamonds = [ln for ln in ax_hc.lines if ln.get_marker() == 'D']
        assert diamonds and np.isfinite(diamonds[0].get_xdata()).all()

    def test_precomputed_timeseries_is_used(self, reduced):
        hc = HazardChart(reduced).compute_hazard_chart(indices=['punstable'])
        stats = reduced.compute_hc_timeseries(indices=['punstable'])
        _, ax_ts = hc.plot_hc_with_timeseries(hc_variant=0, timeseries=stats)
        y = ax_ts.lines[0].get_ydata()
        np.testing.assert_allclose(
            y, stats.data['punstable_median'].values
        )

    def test_group_applies_to_timeseries_and_two_panel_plot(self, reduced):
        weaker = reduced.data.assign(
            sk38=xr.full_like(reduced.data['sk38'], 0.4),
        )
        data = xr.concat([reduced.data, weaker], dim='location').assign_coords(
            location=['L1', 'L2'], region=('location', ['R1', 'R2']),
        )
        hc = HazardChart(xsnowDataset(data)).compute_hazard_chart(
            indices=['sk38'], groupby='region',
        )

        ax = hc.plot_hc_timeseries(hc_variant=0, group='R2')
        np.testing.assert_allclose(ax.lines[0].get_ydata(), [0.4])
        _, ax_ts = hc.plot_hc_with_timeseries(hc_variant=0, group='R1')
        np.testing.assert_allclose(ax_ts.lines[0].get_ydata(), [0.8])


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
        for name in ('hc_instability', 'hc_depth', 'hc_xaxis'):
            np.testing.assert_allclose(
                lazy.data[name].values, eager.data[name].values, equal_nan=True)
        for name in ('hc_instabilityDrivenBy', 'hc_depthDrivenBy'):
            np.testing.assert_array_equal(
                lazy.data[name].values, eager.data[name].values)

    def test_layer_provenance_stays_lazy(self, reduced):
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
            region_id=('location', ['AT-07-01']), band=('location', ['alpine']))
        chunked = xsnowDataset(data.chunk({'location': 1, 'layer': -1}))
        chart = HazardChart(chunked).compute_hazard_chart(
            stability_index='sk38', groupby=['region_id', 'band'])
        assert 'hc_group_id' in chart.data
        assert np.isfinite(chart.data['hc_group_id'].values).any()
        mapping = json.loads(chart.data['hc_group_id'].attrs['mapping'])
        assert mapping['0']['labels'] == {'0': 'AT-07-01|alpine'}
        sub = chart.subset_hazard_points(hc_variant=0, group='AT-07-01|alpine')
        assert sub['hc_instability'].values.ravel()[0] == pytest.approx(0.8)

    def test_requested_layer_context_stays_lazy(self, full):
        chunked = _chunk_data_vars(full, location=1, layer=-1)
        started = []
        with Callback(pretask=lambda key, *_: started.append(key)):
            chart = HazardChart(chunked).compute_hazard_chart(
                indices=['sk38'], depth_var='z',
                layer_context={'datetag': ['2025-11-01']},
            )
        assert started == []
        assert chart.data['hc_instability'].chunks is not None


class TestAtTime:
    @pytest.fixture
    def two_times(self, reduced):
        data = xr.concat([reduced.data, reduced.data], dim='time')
        data = data.assign_coords(time=pd.to_datetime(['2026-01-15', '2026-01-16']))
        return xsnowDataset(data)

    def test_subset_keeps_the_time_axis(self, two_times):
        hc = HazardChart(two_times).compute_hazard_chart(
            indices=['punstable'], at_time='2026-01-16',
        )
        drv = hc.data['hc_instabilityDrivenBy'].isel(hc_variant=0).values.ravel()
        assert np.isnan(drv[0]) and drv[1] == 0
        assert np.isnan(hc.data['hc_instability'].values[0, 0, 0])
        assert hc.list_hc_variants()[0].time == ('2026-01-16T00:00:00',)
        # a full-time variant concatenates with it
        both = hc.compute_hazard_chart(indices=['sk38'])
        assert both.data.sizes['hc_variant'] == 2
        assert (both.data['hc_instabilityDrivenBy'].isel(hc_variant=1).values
                == 0).all()

    def test_default_subset_uses_only_times_contained_in_variant(self, two_times):
        hc = HazardChart(two_times).compute_hazard_chart(
            indices=['punstable'], at_time='2026-01-16',
        )
        sub = hc.subset_hazard_points(hc_variant=0)
        assert sub.sizes['time'] == 1
        assert sub['time'].values[0] == np.datetime64('2026-01-16')

    def test_scalar_slice_and_array_time_selection(self, two_times):
        hc = HazardChart(two_times).compute_hazard_chart(indices=['punstable'])
        scalar = hc.subset_hazard_points(
            hc_variant=0, time=np.datetime64('2026-01-15'))
        sliced = hc.subset_hazard_points(
            hc_variant=0, time=slice('2026-01-15', '2026-01-16'))
        array = hc.subset_hazard_points(
            hc_variant=0, time=['2026-01-16', '2026-01-15'])
        assert 'time' not in scalar.dims
        assert sliced.sizes['time'] == 2
        assert list(array['time'].dt.day.values) == [16, 15]

    def test_marker_aggregation_can_limit_work_to_selected_time(self, two_times):
        hc = HazardChart(two_times).compute_hazard_chart(indices=['punstable'])
        info = hc.list_hc_variants()[0]
        stats = _compute_variant_timeseries(
            hc.data, info, time='2026-01-16',
        ).data
        assert stats.sizes['time'] == 1
        assert stats['time'].values[0] == np.datetime64('2026-01-16')

    def test_subsecond_at_time_round_trips_through_metadata(self, reduced):
        data = xr.concat([reduced.data, reduced.data], dim='time')
        data = data.assign_coords(time=np.array([
            '2026-01-15T00:00:00.100',
            '2026-01-15T00:00:00.200',
        ], dtype='datetime64[ns]'))
        hc = HazardChart(xsnowDataset(data)).compute_hazard_chart(
            indices=['punstable'], at_time='2026-01-15T00:00:00.100',
        )
        info = hc.list_hc_variants()[0]
        assert info.time == ('2026-01-15T00:00:00.1',)
        sub = hc.subset_hazard_points(hc_variant=0)
        assert sub.sizes['time'] == 1
