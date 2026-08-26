import dask.array as da
import numpy as np
import pytest
import xarray as xr
from dask.callbacks import Callback

from xsnow.extensions import (
    apply_profile_block,
    float_result_dtype,
    get_backend,
)


def test_get_backend_accepts_arrays_xarray_and_xsnow_datasets():
    values = xr.DataArray(np.arange(3), dims='layer')

    assert get_backend(values.data) is np
    assert get_backend(values) is np
    assert get_backend(values.to_dataset(name='values')) is np

    from xsnow import xsnowDataset

    assert get_backend(xsnowDataset(values.to_dataset(name='values'))) is np


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


@pytest.fixture
def profiles():
    """(time, location, layer) values plus a per-profile scalar."""
    values = xr.DataArray(
        np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        dims=('time', 'location', 'layer'),
    )
    surface = xr.DataArray(
        np.arange(6, dtype=np.float32).reshape(2, 3),
        dims=('time', 'location'),
    )
    return values, surface


def _chunked(da_, **chunks):
    return da_.chunk(chunks)


def test_output_dtype_defaults_to_the_promoted_float_dtype(profiles):
    values, _ = profiles
    integers = values.astype('int16').chunk({'time': 1, 'layer': -1})

    result = apply_profile_block(
        lambda block: block.astype('float32'), integers
    )
    assert float_result_dtype(integers) == np.dtype('float32')
    assert result.dtype == np.dtype('float32')
    xr.testing.assert_allclose(result.compute(), values.astype('float32'))


def test_an_input_without_the_profile_axis_is_broadcast(profiles):
    values, surface = profiles

    def depth_below_surface(block, top):
        return top[..., None] - block

    result = apply_profile_block(
        depth_below_surface,
        values,
        surface,
        input_core_dims=[['layer'], []],
    )
    xr.testing.assert_allclose(result.transpose(*values.dims), surface - values)


def test_a_reduction_drops_the_profile_axis(profiles):
    values, _ = profiles
    result = apply_profile_block(
        lambda block: np.nanmax(block, axis=-1),
        values,
        output_core_dims=[[]],
    )
    assert 'layer' not in result.dims
    xr.testing.assert_allclose(result, values.max('layer'))


def test_several_outputs_are_returned_as_a_tuple(profiles):
    values, _ = profiles

    def spread(block):
        return (np.nanmin(block, axis=-1).astype('float32'),
                np.nanmax(block, axis=-1).astype('float64'))

    low, high = apply_profile_block(
        spread,
        values.chunk({'time': 1, 'layer': -1}),
        output_dtype=['float32', 'float64'],
        output_core_dims=[[], []],
    )
    assert (low.dtype, high.dtype) == (np.dtype('float32'), np.dtype('float64'))
    xr.testing.assert_allclose(low.compute(), values.min('layer'))
    xr.testing.assert_allclose(high.compute(), values.max('layer'))


def test_the_declared_dtype_is_what_the_lazy_graph_advertises(profiles):
    """Nothing checks the promise: dask trusts output_dtype while the graph
    is built, so a block function that returns another dtype yields a lazy
    array whose advertised dtype differs from the computed one."""
    values, _ = profiles
    result = apply_profile_block(
        lambda block: block.astype('float32'),
        values.chunk({'time': 1, 'layer': -1}),
        output_dtype='float64',
    )
    assert result.dtype == np.dtype('float64')
    assert result.compute().dtype == np.dtype('float32')


def test_a_new_output_axis_is_sized_by_output_sizes(profiles):
    values, _ = profiles
    result = apply_profile_block(
        lambda block: np.sort(block, axis=-1)[..., -2:],
        values,
        output_core_dims=[['top']],
        output_sizes={'top': 2},
    )
    assert result.sizes['top'] == 2
    np.testing.assert_allclose(
        result.transpose('time', 'location', 'top').values,
        np.sort(values.values, axis=-1)[..., -2:],
    )


def test_one_dtype_per_output_is_required(profiles):
    values, _ = profiles
    with pytest.raises(ValueError, match='One output_dtype is required'):
        apply_profile_block(
            lambda block: (block, block),
            values,
            output_dtype='float32',
            output_core_dims=[['layer'], ['layer']],
        )


def test_a_missing_core_dimension_is_reported(profiles):
    _, surface = profiles
    with pytest.raises(ValueError, match="requires a 'layer' dimension"):
        apply_profile_block(lambda block: block, surface)


def test_the_graph_is_built_without_running_the_block_function(profiles):
    values, surface = profiles
    calls = []

    def spread(block, top):
        calls.append(1)
        return np.nanmin(block, axis=-1), top - np.nanmax(block, axis=-1)

    started = []
    with Callback(pretask=lambda key, *_: started.append(key)):
        low, gap = apply_profile_block(
            spread,
            _chunked(values, time=1, layer=-1),
            _chunked(surface, time=1),
            output_dtype=['float32', 'float32'],
            input_core_dims=[['layer'], []],
            output_core_dims=[[], []],
        )

    assert started == []
    assert calls == []
    assert isinstance(low.data, da.Array)
    xr.testing.assert_allclose(low.compute(), values.min('layer'))
    xr.testing.assert_allclose(
        gap.compute().transpose(*surface.dims), surface - values.max('layer'))


def test_only_the_core_dimension_is_rechunked(profiles):
    values, _ = profiles
    split = _chunked(values, time=1, location=1, layer=2)
    result = apply_profile_block(lambda block: block * 2.0, split)
    assert result.chunks[result.get_axis_num('layer')] == (4,)
    assert result.chunks[result.get_axis_num('time')] == (1, 1)
    assert result.chunks[result.get_axis_num('location')] == (1, 1, 1)
