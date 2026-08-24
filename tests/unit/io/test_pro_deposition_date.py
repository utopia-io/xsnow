"""
Tests for the element deposition date (PRO code 0505).
"""
import numpy as np
import pandas as pd
import pytest

from xsnow import read
from xsnow.io.base import CODE_TO_NAME_MAP


def _layer_dates(ds, time):
    da = ds.data['deposition_date'].sel(time=time)
    return da.isel(location=0, slope=0, realization=0).values


class TestCodeMapping:
    def test_registry_codes(self):
        assert CODE_TO_NAME_MAP['0505'] == 'deposition_date'
        assert CODE_TO_NAME_MAP['0540'] == 'bulk_salinity'


@pytest.mark.real_data
class TestReadDepositionDates:
    def test_eager_golden_values(self, sample_pro_file):
        ds = read(sample_pro_file, lazy=False)
        da = ds.data['deposition_date']

        assert da.dtype == np.dtype('datetime64[ns]')
        # CF encoding owns 'units' for datetime vars; attrs must not carry one
        assert 'units' not in da.attrs
        assert da.attrs['standard_name'] == 'deposition_date_of_snow_element'

        # first snowy block of VIR1A1.pro: one layer deposited
        # 2024-01-17T23:45:00
        dates = _layer_dates(ds, '2024-01-18T00:00:00')
        assert dates[0] == np.datetime64('2024-01-17T23:45:00')
        assert pd.isna(dates[1:]).all()

    def test_dates_nondecreasing_with_height(self, sample_pro_file):
        ds = read(sample_pro_file, lazy=False)
        da = ds.data['deposition_date'].isel(
            location=0, slope=0, realization=0)
        for t in [-1, len(da.time) // 2]:
            dates = da.isel(time=t).values
            valid = dates[~pd.isna(dates)]
            assert len(valid) > 0
            assert (np.diff(valid.astype('int64')) >= 0).all()

    def test_lazy_matches_eager(self, sample_pro_file):
        eager_da = read(sample_pro_file, lazy=False).data['deposition_date']
        eager = eager_da.values
        lazy_da = read(sample_pro_file, lazy=True).data['deposition_date']
        lazy = lazy_da.compute().values
        assert np.array_equal(pd.isna(eager), pd.isna(lazy))
        assert np.array_equal(eager[~pd.isna(eager)], lazy[~pd.isna(lazy)])


@pytest.mark.synthetic
class TestAgeVariant:
    def test_age_rows_are_skipped(self, tmp_path):
        pro = tmp_path / 'AGETEST.pro'
        pro.write_text(
            "[STATION_PARAMETERS]\n"
            "StationName= AGETEST\n"
            "Latitude= 47.0\n"
            "Longitude= 11.0\n"
            "Altitude= 2000\n"
            "SlopeAngle= 0.00\n"
            "SlopeAzi= 0.00\n"
            "\n"
            "[HEADER]\n"
            "0500,Date\n"
            "0501,nElems,height [> 0: top, < 0: bottom of elem.] (cm)\n"
            "0502,nElems,element density (kg m-3)\n"
            "0505,nElems,element age (days)\n"
            "0512,nElems,grain size (mm)\n"
            "0513,nElems,grain type (Swiss Code F1F2F3)\n"
            "\n"
            "[DATA]\n"
            "0500,01.12.2024 12:00:00\n"
            "0501,2,10.0,20.0\n"
            "0502,2,150.0,200.0\n"
            "0505,2,12.5,3.25\n"
            "0512,2,1.0,1.0\n"
            "0513,2,772,772\n"
        )
        ds = read(pro, lazy=False)
        assert 'deposition_date' not in ds.data.data_vars
        assert 'density' in ds.data.data_vars


@pytest.mark.real_data
class TestRoundTrips:
    def test_to_pro_round_trip(self, sample_pro_file, tmp_path):
        ds = read(sample_pro_file, lazy=False).trim_padding()
        out = tmp_path / 'rt.pro'
        ds.to_pro(out)

        lines = [
            line for line in out.read_text().splitlines()
            if line.startswith('0505')
        ]
        assert lines[0] == '0505,nElems,element deposition date (ISO)'

        back = read(out, lazy=False)
        a = back.data['deposition_date'].values
        b = ds.data['deposition_date'].values
        n = min(a.shape[-1], b.shape[-1])
        a, b = a[..., :n], b[..., :n]
        assert np.array_equal(pd.isna(a), pd.isna(b))
        assert np.array_equal(a[~pd.isna(a)], b[~pd.isna(b)])

    def test_netcdf_round_trip(self, sample_pro_file, tmp_path):
        ds = read(sample_pro_file, lazy=False).trim_padding()
        out = tmp_path / 'rt.nc'
        ds.to_netcdf(out)
        back = read(out)
        a = back.data['deposition_date'].values
        b = ds.data['deposition_date'].values
        assert np.array_equal(pd.isna(a), pd.isna(b))
        assert np.array_equal(a[~pd.isna(a)], b[~pd.isna(b)])
