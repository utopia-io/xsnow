"""Tests for reading SNOWPACK ZarrIO archives."""
import json
import os
from pathlib import Path

import dask.array as da
import numpy as np
import pytest
import xarray as xr
import zarr
from dask.callbacks import Callback

import xsnow
from xsnow.io.snowpack_zarr import (
    ZarrIOLayoutError,
    adapt_snowpack_zarrio,
    is_snowpack_zarrio,
    open_snowpack_zarrio,
    read_snowpack_archives,
)
from xsnow.units import validate_xsnow_unit_attrs

REAL_ARCHIVE = Path('/home/snowpack/zarrtest/output/x1.zarr')

L, T, K = 4, 6, 5
IDS = ['WFJ', 'WFJ1', 'KLO', 'KLO1']
NAMES = ['Weissfluhjoch', 'Weissfluhjoch', 'Klosters', 'Klosters']


def _write_archive(path, *, pinned=False, checkpoint=True, chunk_times=T,
                   precip_rates=True, empty_slot=False, ids=None,
                   labels=True):
    """A ZarrIO-shaped Zarr v2 store (no .zmetadata, |S64 labels, NaN fill,
    storage units, optional checkpoint group)."""
    ids = ids or IDS
    g = zarr.open_group(str(path), mode='w', zarr_format=2)
    g.attrs.update({
        'experiment': 'res', 'snowpack_version': '3.7.1',
        'created_by': 'u@h', 'history': 'x', 'Conventions': 'CF-1.8',
    })

    def arr(name, dims, data, units, long_name, dtype='<f4', chunks=None):
        a = g.create_array(
            name, shape=data.shape, chunks=chunks or data.shape,
            dtype=dtype, fill_value=np.nan, compressors=None,
        )
        a[:] = data
        a.attrs.update({'_ARRAY_DIMENSIONS': list(dims), 'units': units,
                        'long_name': long_name})

    h = np.full((L, T, K), np.nan, np.float32)
    n = np.full((L, T), np.nan, np.float32)
    for loc in range(L):
        if empty_slot and loc == L - 1:
            continue
        for t in range(T - 1):          # last slot never written
            k = min(K, t + 1)
            h[loc, t, :k] = np.cumsum(np.full(k, 10.0))   # cm
            n[loc, t] = k
    layered = ('location', 'time', 'layer')
    scalar = ('location', 'time')
    lchunks = (L, chunk_times, K)
    arr('height', layered, h, 'cm', 'height', chunks=lchunks)
    arr('density', layered, np.where(np.isnan(h), np.nan, 250.0),
        'kg m-3', 'density', chunks=lchunks)
    arr('lwc', layered, np.where(np.isnan(h), np.nan, 2.5), '%', 'lwc',
        chunks=lchunks)
    arr('stress', layered, np.where(np.isnan(h), np.nan, 1.5), 'kPa',
        'stress', chunks=lchunks)
    arr('critical_cut_length', layered,
        np.where(np.isnan(h), np.nan, 0.3), 'm', 'ccl', chunks=lchunks)
    arr('rta', layered, np.where(np.isnan(h), np.nan, 0.5), '1', 'rta',
        chunks=lchunks)
    arr('deposition_date', layered,
        np.where(np.isnan(h), np.nan, 2460000.5).astype(np.float64),
        'Julian day (UTC)', 'deposition date', dtype='<f8', chunks=lchunks)
    arr('n_layers', scalar, n, '1', 'number of layers')
    hs = np.where(np.isnan(n), np.nan,
                  np.nanmax(np.nan_to_num(h, nan=0), axis=2))
    arr('HS', scalar, hs.astype(np.float32), 'cm', 'snow height')
    arr('HS_mod', scalar, hs.astype(np.float32), 'cm', 'modelled snow height')
    arr('TA', scalar, np.full((L, T), -3.0, np.float32), 'degC',
        'air temperature')
    arr('MS_Snow', scalar, np.full((L, T), 3.6, np.float32),
        'kg m-2 h-1' if precip_rates else 'kg m-2', 'solid precipitation')

    tv = np.array([1.7e9 + i * 3600 for i in range(T)], dtype=np.float64)
    if not pinned:
        tv[-1] = np.nan                      # append mode: unused slot
    ta = g.create_array('time', shape=(T,), chunks=(T,), dtype='<f8',
                        fill_value=np.nan, compressors=None)
    ta[:] = tv
    ta.attrs.update({'_ARRAY_DIMENSIONS': ['time'],
                     'units': 'seconds since 1970-01-01 00:00:00',
                     'calendar': 'proleptic_gregorian'})
    labels_arrays = labels
    labels = list(ids)
    if empty_slot:
        labels[-1] = ''
    for name, values in (('station_id', labels), ('station_name', NAMES)):
        if not labels_arrays:
            break
        a = g.create_array(name, shape=(L,), chunks=(L,), dtype='S64',
                           fill_value=None, compressors=None)
        a[:] = np.array([v.encode() for v in values], dtype='S64')
        a.attrs.update({'_ARRAY_DIMENSIONS': ['location']})
    if checkpoint:
        cp = g.create_group('checkpoint')
        for nm, val, dt in (
            ('slope_angle', [0, 38, 0, 38], '<f4'),
            ('slope_azi', [0, 180, 0, 90], '<f4'),
            ('latitude', [46.8, 46.8, 46.9, 46.9], '<f8'),
            ('longitude', [9.8, 9.8, 9.9, 9.9], '<f8'),
            ('altitude', [2540.0, 2540.0, 1200.0, 1200.0], '<f8'),
        ):
            a = cp.create_array(nm, shape=(L,), chunks=(L,), dtype=dt,
                                fill_value=np.nan, compressors=None)
            a[:] = np.asarray(val)
            a.attrs.update({'_ARRAY_DIMENSIONS': ['location']})
    with open(Path(path) / 'station_index.json', 'w') as f:
        json.dump({s: i for i, s in enumerate(ids)}, f)
    return Path(path)


@pytest.fixture
def archive(tmp_path):
    return _write_archive(tmp_path / 'station.zarr')


def _no_tasks(fn):
    started = []
    with Callback(pretask=lambda key, *_: started.append(key)):
        out = fn()
    return out, started


def _assert_layered_untouched(ds, started):
    """Opening may reduce the 2-D ``n_layers`` array, never a layered one."""
    layered = set()
    for name, var in ds.data_vars.items():
        if 'layer' in var.dims and isinstance(var.data, da.Array):
            layered |= set(var.data.dask.keys())
    assert not (set(started) & layered)


def test_detection_and_read_dispatch(archive):
    raw = open_snowpack_zarrio(archive)
    assert is_snowpack_zarrio(raw)
    xs, started = _no_tasks(lambda: xsnow.read(archive, lazy=True))
    _assert_layered_untouched(xs.data, started)
    assert isinstance(xs, xsnow.xsnowDataset)
    assert tuple(xs.data['height'].dims) == (
        'location', 'time', 'slope', 'realization', 'layer'
    )
    assert isinstance(xs.data['height'].data, da.Array)


def _slope_coord(ds, name, location):
    """Slope coordinate of one station; xsnow folds identical per-location
    slope signatures onto a plain ``(slope,)`` coordinate."""
    coord = ds[name]
    if 'location' in coord.dims:
        coord = coord.sel(location=location)
    return coord.values


def test_layout_labels_slopes_and_terrain(archive):
    xs = xsnow.read(archive, lazy=True)
    ds = xs.data
    assert list(ds['location'].values) == ['KLO', 'WFJ']
    assert list(ds['slope'].values) == [0, 1]
    assert list(ds['realization'].values) == ['res']
    np.testing.assert_allclose(
        _slope_coord(ds, 'inclination', 'WFJ'), [0.0, 38.0]
    )
    np.testing.assert_allclose(_slope_coord(ds, 'azimuth', 'KLO'), [0.0, 90.0])
    np.testing.assert_allclose(
        _slope_coord(ds, 'azimuth', 'WFJ'), [0.0, 180.0]
    )
    np.testing.assert_allclose(ds['altitude'].values, [1200.0, 2540.0])
    assert ds['time'].attrs['timezone'] == 'UTC'
    # append mode: the unused NaT slot is dropped
    assert ds.sizes['time'] == T - 1


def test_units_names_and_dtypes(archive):
    ds = xsnow.read(archive, lazy=True).data
    assert validate_xsnow_unit_attrs(ds, unit_validation='strict')
    assert 'rc_flat' in ds and 'critical_cut_length' not in ds
    assert ds['height'].attrs['units'] == 'm'
    assert ds['deposition_date'].dtype == np.dtype('datetime64[ns]')
    assert ds['n_layers'].dtype == np.dtype('int16')
    assert ds['profile_status'].dtype == np.dtype('int8')
    assert 'z' in ds.coords
    out = ds.compute()
    h = out['height'].sel(location='WFJ', slope=1).isel(
        time=2, realization=0
    ).values
    np.testing.assert_allclose(h[:3], [0.1, 0.2, 0.3])
    assert np.isnan(h[3:]).all()
    np.testing.assert_allclose(
        out['lwc'].isel(location=0, time=1, slope=0,
                        realization=0).values[:2], [0.025, 0.025]
    )
    np.testing.assert_allclose(
        out['stress'].isel(location=0, time=1, slope=0,
                           realization=0).values[:1], [1500.0]
    )
    np.testing.assert_allclose(
        out['MS_Snow'].isel(location=0, time=0, slope=0,
                            realization=0).values, 0.001
    )
    assert (out['deposition_date'].isel(
        location=0, time=1, slope=0, realization=0
    ).values[0] == np.datetime64('2023-02-25T00:00:00'))
    assert (out['profile_status'].values == 1).all()
    z = out['z'].sel(location='WFJ', slope=0).isel(time=2, realization=0)
    np.testing.assert_allclose(z.values[:3], [-0.2, -0.1, 0.0])


def test_layers_pad_and_trim(archive):
    ds = xsnow.read(archive, lazy=True).data
    assert ds.sizes['layer'] == K      # deepest profile has K layers
    padded = xsnow.read(archive, lazy=True, layers=8).data
    assert padded.sizes['layer'] == 8
    assert list(padded['layer'].values) == list(range(8))
    assert np.isnan(padded['height'].isel(layer=slice(K, None)).values).all()
    trimmed = xsnow.read(archive, lazy=True, layers=3).data
    assert trimmed.sizes['layer'] == 3


def test_pinned_time_grid_uses_data_presence(tmp_path):
    path = _write_archive(tmp_path / 'pinned.zarr', pinned=True)
    raw = open_snowpack_zarrio(path)
    assert not np.isnat(raw['time'].values).any()
    ds = xsnow.read(path, lazy=True).data
    assert ds.sizes['time'] == T - 1


def test_unwritten_slot_is_dropped(tmp_path):
    path = _write_archive(tmp_path / 'partial.zarr', empty_slot=True)
    ds = xsnow.read(path, lazy=True).data
    assert list(ds['location'].values) == ['KLO', 'WFJ']
    assert np.isnan(
        ds['height'].sel(location='KLO', slope=1).values
    ).all()


def test_terrain_from_caller_without_checkpoint(tmp_path):
    path = _write_archive(tmp_path / 'nocp.zarr', checkpoint=False)
    ds = xsnow.read(
        path, lazy=True,
        slope_metadata={1: (30.0, 270.0)},
        geo={'WFJ': (46.83, 9.81, 2540.0)},
    ).data
    np.testing.assert_allclose(
        _slope_coord(ds, 'inclination', 'WFJ'), [0.0, 30.0]
    )
    np.testing.assert_allclose(
        _slope_coord(ds, 'azimuth', 'KLO'), [0.0, 270.0]
    )
    assert ds['altitude'].sel(location='WFJ').item() == 2540.0
    assert np.isnan(ds['altitude'].sel(location='KLO').item())


def test_unconvertible_unit_raises_loudly(tmp_path):
    path = _write_archive(tmp_path / 'sums.zarr', precip_rates=False)
    with pytest.raises(ZarrIOLayoutError, match='MS_Snow'):
        xsnow.read(path, lazy=True)


def test_selectors_and_eager_read(archive):
    xs = xsnow.read(archive, lazy=False, location=['WFJ'],
                    time=slice('2023-11-14T22:00', None))
    assert not isinstance(xs.data['height'].data, da.Array)
    assert list(xs.data['location'].values) == ['WFJ']
    assert xs.data['height'].dtype == np.dtype('float32')


def test_time_chunks_are_coarsened(tmp_path):
    path = _write_archive(tmp_path / 'chunky.zarr', chunk_times=1)
    ds = xsnow.read(path, lazy=True, time_chunk=4).data
    assert ds['height'].chunks[1] == (4, 1)
    assert ds['height'].chunks[-1] == (K,)


def test_stale_layer_mask(archive):
    ds = xsnow.read(archive, lazy=True, mask_stale_layers=True).data
    out = ds['height'].compute()
    n = ds['n_layers'].compute()
    for t in range(ds.sizes['time']):
        k = int(
            n.sel(location='WFJ', slope=0).isel(time=t, realization=0)
        )
        assert np.isnan(
            out.sel(location='WFJ', slope=0).isel(
                time=t, realization=0
            ).values[k:]
        ).all()


def test_read_snowpack_archives_concat_and_exclude(tmp_path):
    a = _write_archive(tmp_path / 'a.zarr', pinned=True,
                       ids=['A', 'A1', 'B', 'B1'])
    b = _write_archive(tmp_path / 'b.zarr', pinned=True,
                       ids=['C', 'C1', 'D', 'D1'])
    xs, started = _no_tasks(
        lambda: read_snowpack_archives([a, b], exclude=['D'])
    )
    _assert_layered_untouched(xs.data, started)
    assert list(xs.data['location'].values) == ['A', 'B', 'C']
    assert isinstance(xs.data['density'].data, da.Array)
    assert xs.data.sizes['time'] == T - 1


def test_xsnow_layout_archive_is_untouched(tmp_path, archive):
    ds = xsnow.read(archive, lazy=False).data
    target = tmp_path / 'xsnow.zarr'
    ds.to_zarr(target, mode='w')
    # string realization labels are stored as variable-length strings,
    # which the eager memory guard cannot size: open lazily
    back = xsnow.read(target, lazy=True)
    assert not is_snowpack_zarrio(xr.open_zarr(target))
    xr.testing.assert_allclose(back.data['height'].compute(), ds['height'])
    with pytest.raises(ZarrIOLayoutError, match='ZarrIO reader options'):
        xsnow.read(target, layers=3)


def test_adapted_dataset_writes_to_zarr(tmp_path, archive):
    ds = xsnow.read(archive, lazy=True).data
    assert all(not v.encoding for v in ds.variables.values())
    target = tmp_path / 'derived.zarr'
    ds[['height', 'HS', 'n_layers', 'profile_status']].to_zarr(
        target, mode='w'
    )
    back = xr.open_zarr(target)
    xr.testing.assert_allclose(back['height'].compute(),
                               ds['height'].compute())


def test_station_index_file_replaces_missing_labels(tmp_path):
    path = _write_archive(tmp_path / 'nolabels.zarr', labels=False,
                          checkpoint=False)
    with pytest.raises(ZarrIOLayoutError, match='station_index'):
        xsnow.read(path, lazy=True)
    index = tmp_path / 'nolabels.zarr.stations.txt'
    index.write_text('WFJ 0\nWFJ1 1\nKLO 2\nKLO1 3\n')
    ds = xsnow.read(path, lazy=True, station_index=index).data
    assert list(ds['location'].values) == ['KLO', 'WFJ']
    assert list(ds['slope'].values) == [0, 1]
    # unknown terrain is left out rather than written as NaN coordinates
    assert 'inclination' not in ds.coords and 'altitude' not in ds.coords
    # profsnow's side files next to the archive are picked up automatically
    (tmp_path / 'nolabels.zarr.layout.json').write_text(
        '{"slope_metadata": {"0": [0, 0], "1": [38, 180]}}'
    )
    auto = xsnow.read(path, lazy=True).data
    assert list(auto['location'].values) == ['KLO', 'WFJ']
    np.testing.assert_allclose(_slope_coord(auto, 'azimuth', 'WFJ'),
                               [0.0, 180.0])


def test_adapt_rejects_foreign_layout():
    ds = xr.Dataset({'x': ('location', np.zeros(2))})
    with pytest.raises(ZarrIOLayoutError):
        adapt_snowpack_zarrio(ds)


@pytest.mark.skipif(not REAL_ARCHIVE.exists(), reason='no real archive')
def test_real_archive_opens_lazily():
    xs, started = _no_tasks(lambda: xsnow.read(REAL_ARCHIVE, lazy=True))
    _assert_layered_untouched(xs.data, started)
    ds = xs.data
    assert list(ds['location'].values) == ['MST96']
    assert list(ds['slope'].values) == [0, 1, 2, 3, 4]
    assert validate_xsnow_unit_attrs(ds, unit_validation='strict')
    assert ds.sizes['layer'] >= 400
    assert ds['height'].chunks[-1] == (ds.sizes['layer'],)
    tchunks = ds['height'].chunks[1]
    assert len(set(tchunks[:-1])) == 1 and tchunks[-1] <= tchunks[0]
    column = ds['height'].isel(location=0, slope=1, realization=0)
    assert os.path.exists(REAL_ARCHIVE / 'height')
    hs = ds['HS'].isel(location=0, slope=1, realization=0).compute()
    assert float(hs.max()) > 1.5           # metres
    top = column.isel(time=int(hs.argmax())).compute()
    assert np.nanmax(top.values) == pytest.approx(float(hs.max()), rel=1e-5)
