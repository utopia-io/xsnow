import numpy as np
import xarray as xr

from xsnow.extensions import (
    apply_profile_block,
    float_result_dtype,
    get_backend,
)


def test_public_profile_block_helpers_support_a_custom_layer_dimension():
    values = xr.DataArray(
        np.arange(12, dtype=np.float32).reshape(2, 2, 3),
        dims=('time', 'location', 'snow_layer'),
    )

    def subtract_profile_minimum(block):
        xp = get_backend(block)
        return block - xp.min(block, axis=-1)[..., None]

    result = apply_profile_block(
        subtract_profile_minimum,
        values,
        output_dtype=float_result_dtype(values),
        layer_dim='snow_layer',
    )

    expected = values - values.min(dim='snow_layer')
    xr.testing.assert_equal(result, expected)
    assert result.dtype == np.dtype('float32')


def test_get_backend_accepts_arrays_xarray_and_xsnow_datasets():
    values = xr.DataArray(np.arange(3), dims='layer')

    assert get_backend(values.data) is np
    assert get_backend(values) is np
    assert get_backend(values.to_dataset(name='values')) is np

    from xsnow import xsnowDataset

    assert get_backend(xsnowDataset(values.to_dataset(name='values'))) is np


def test_get_backend_inspects_dask_metadata_without_computing():
    values = xr.DataArray(np.arange(6), dims='layer').chunk({'layer': 2})

    assert get_backend(values) is np
