"""Datetag derivation from the weather timeline.

The parity test runs the qmah implementation the derivation was ported from
side by side with xsnow's, on the same weather, and compares the tags.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import xsnow.extensions.datetags as dt
from xsnow import xsnowDataset

QMAH_SOURCE = Path(
    '/opt/awsome/code/snow-cover/postprocessing/qmah/qmah/datetags/datetags.py'
)


def _qmah_module():
    """qmah's datetags module, imported past its package __init__."""
    spec = importlib.util.spec_from_file_location('qmah_datetags', QMAH_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _season(seed=0, days=200, start='2025-10-01'):
    """A season of daily snow and rain: storms, dry spells, drizzle days."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days, freq='D')
    hn24 = np.zeros(days)
    rain24 = np.zeros(days)
    for onset in (12, 40, 47, 96, 150, 175):          # storms
        for offset, amount in enumerate((14.0, 22.0, 6.0)):
            if onset + offset < days:
                hn24[onset + offset] = amount
    for day in range(60, 92, 3):                       # a long drizzle gap
        hn24[day] = rng.uniform(3.0, 7.0)
    for onset in (30, 120):                            # rain events
        rain24[onset] = 9.0
        rain24[onset + 1] = 7.0
    hn24 += rng.uniform(0.0, 1.0, days)                # trace amounts
    frame = pd.DataFrame({'date': dates, 'hn24': hn24, 'rain24': rain24})
    frame['hn72'] = frame['hn24'].rolling(3).sum()
    frame['rain72'] = frame['rain24'].rolling(3).sum()
    return frame


@pytest.mark.skipif(not QMAH_SOURCE.is_file(), reason='no qmah checkout')
@pytest.mark.parametrize('seed', [0, 1, 2, 3, 4])
def test_parity_with_qmah(seed):
    qmah = _qmah_module()
    weather = _season(seed=seed)
    expected = qmah.derive_datetags(weather.copy())
    actual = dt.derive_datetags(weather.copy())
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.dtype('datetime64[D]')


@pytest.mark.skipif(not QMAH_SOURCE.is_file(), reason='no qmah checkout')
def test_parity_with_qmah_and_human_tags():
    qmah = _qmah_module()
    weather = _season(seed=7)
    human = ['2025-11-20', '2026-01-02']
    np.testing.assert_array_equal(
        dt.derive_datetags(weather.copy(), human_datetags=human),
        qmah.derive_datetags(weather.copy(), human_datetags=human),
    )


def test_storm_onsets_are_tagged():
    weather = _season(seed=0)
    tags = pd.to_datetime(dt.derive_datetags(weather))
    storms = pd.Timestamp('2025-10-01') + pd.to_timedelta(
        [11, 39, 95, 149, 174], unit='D')
    for storm in storms:                # the day before each storm's onset
        assert (abs(tags - storm) <= pd.Timedelta(days=1)).any(), storm
    assert (tags.to_series().diff().dropna() > pd.Timedelta(days=0)).all()


def test_thresholds_are_honoured():
    weather = _season(seed=0)
    few = dt.derive_datetags(weather, snow_thresholds={'hn72_thresh': 200.0},
                             rain_thresholds={'rain72_thresh': 200.0})
    assert len(few) < len(dt.derive_datetags(weather))


def test_empty_weather():
    empty = pd.DataFrame(
        {'date': pd.to_datetime([]), 'hn24': [], 'rain24': [], 'hn72': [],
         'rain72': []}
    )
    assert dt.derive_datetags(empty).size == 0


def _forcing(regions=('AT-07-01', 'AT-07-02'), bands=('montane', 'alpine'),
             per_unit=3, hours=200 * 24):
    """Hourly forcing on (location, time) with region/band coordinates."""
    rng = np.random.default_rng(5)
    times = pd.date_range('2025-10-01', periods=hours, freq='h')
    daily = _season()
    names, region_ids, band_ids, psum, temperature = [], [], [], [], []
    for r_index, region in enumerate(regions):
        for b_index, band in enumerate(bands):
            for member in range(per_unit):
                names.append(f'{region}_{band}_{member}')
                region_ids.append(region)
                band_ids.append(band)
                # one storm-free unit, the others follow the season
                scale = 0.0 if (r_index, b_index) == (1, 1) else 1.0
                hourly = np.repeat(
                    daily['hn24'].to_numpy() * scale / 24.0, 24)[:hours]
                wet = np.repeat(
                    daily['rain24'].to_numpy() * scale / 24.0, 24)[:hours]
                psum.append(hourly + wet)
                temperature.append(
                    np.where(wet > 0, 5.0, -5.0) + rng.normal(0, 0.1, hours))
    data = xr.Dataset(
        {
            'PSUM': (('location', 'time'), np.array(psum, dtype='float32')),
            'TA': (('location', 'time'),
                   np.array(temperature, dtype='float32'), {'units': 'degC'}),
        },
        coords={
            'location': names, 'time': times,
            'region_id': ('location', region_ids),
            'band': ('location', band_ids),
        },
    )
    return xsnowDataset(data)


def test_weather_datetags_shape_and_units():
    tags = _forcing().weather_datetags()
    assert tags['meta'] == {'level1': 'region_id', 'level2': 'band'}
    assert set(tags) == {'unique_datetags', 'meta', 'AT-07-01', 'AT-07-02'}
    assert set(tags['AT-07-01']) == {'montane', 'alpine'}
    assert tags['unique_datetags'] == sorted(tags['unique_datetags'])
    every = {tag for level in ('AT-07-01', 'AT-07-02')
             for tags_of_band in tags[level].values() for tag in tags_of_band}
    assert every == set(tags['unique_datetags'])
    assert all(len(tag) == 10 for tag in tags['unique_datetags'])
    # the unit without precipitation gets no tags, the others do
    assert tags['AT-07-02']['alpine'] == []
    assert len(tags['AT-07-01']['montane']) > 3


def test_weather_datetags_matches_the_direct_derivation():
    forcing = _forcing()
    tags = forcing.weather_datetags()
    daily = dt.daily_precipitation(forcing).compute()
    unit = daily.isel(location=[0, 1, 2])          # AT-07-01 / montane
    frame = pd.DataFrame({
        'date': pd.to_datetime(daily['time'].values),
        **{name: unit[name].quantile(0.5, dim='location').values
           for name in ('hn24', 'rain24', 'hn72', 'rain72')},
    })
    expected = [str(tag) for tag in dt.derive_datetags(frame)]
    assert tags['AT-07-01']['montane'] == expected


def test_weather_datetags_without_grouping_coords():
    forcing = _forcing()
    plain = xsnowDataset(forcing.data.drop_vars(['region_id', 'band']))
    tags = plain.weather_datetags()
    assert list(tags['all']) == ['all']
    assert tags['unique_datetags']


def test_datetags_feed_assign_datetags():
    forcing = _forcing()
    tags = forcing.weather_datetags()
    profiles = xsnowDataset(xr.Dataset(
        {
            'deposition_date': (
                ('location', 'time', 'layer'),
                np.array([[[np.datetime64('2025-11-05', 'ns'),
                            np.datetime64('2026-01-20', 'ns')]]] * 2),
            ),
        },
        coords={
            'location': ['AT-07-01_montane_0', 'AT-07-02_alpine_0'],
            'time': [pd.Timestamp('2026-02-01')], 'layer': [0, 1],
            'region_id': ('location', ['AT-07-01', 'AT-07-02']),
            'band': ('location', ['montane', 'alpine']),
        },
    ))
    tagged = profiles.assign_datetags(tags)
    assigned = tagged.data['datetag'].values
    assert not np.isnat(assigned[0]).all()       # tagged unit
    assert np.isnat(assigned[1]).all()           # unit without tags


def test_precipitation_phase_variable_wins():
    forcing = _forcing()
    warm = forcing.data.assign(
        PSUM_PH=xr.ones_like(forcing.data['PSUM']))    # everything is rain
    daily = dt.daily_precipitation(xsnowDataset(warm)).compute()
    assert float(daily['hn24'].max()) == pytest.approx(0.0)
    assert float(daily['rain24'].max()) > 0.0


def test_scale_and_explicit_variables():
    forcing = _forcing()
    data = forcing.data.rename({'PSUM': 'MS_Snow'})
    data['MS_Rain'] = xr.zeros_like(data['MS_Snow'])
    daily = dt.daily_precipitation(
        xsnowDataset(data), snowfall_var='MS_Snow', rainfall_var='MS_Rain',
        scale=2.0,
    ).compute()
    plain = dt.daily_precipitation(forcing).compute()
    np.testing.assert_allclose(
        daily['hn24'].values,
        2.0 * (plain['hn24'] + plain['rain24']).values, rtol=1e-5,
    )
