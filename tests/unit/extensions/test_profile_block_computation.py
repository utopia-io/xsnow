import numpy as np
import xarray as xr

from xsnow.extensions import (
    apply_profile_block,
    float_result_dtype,
    get_array_backend,
)


def test_public_profile_block_helpers_support_a_custom_layer_dimension():
    values = xr.DataArray(
        np.arange(12, dtype=np.float32).reshape(2, 2, 3),
        dims=('time', 'location', 'snow_layer'),
    )

    def subtract_profile_minimum(block):
        xp = get_array_backend(block)
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
