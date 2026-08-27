"""Utilities for blockwise calculations on complete snow profiles.

Dask evaluates large xarray datasets in manageable blocks.  A profile-wide
calculation may split outer dimensions such as ``location`` or ``time``, but
each block must contain every ``layer`` of its profiles.

:func:`apply_profile_block` aligns the inputs, combines chunks along dimensions
needed in full, and calls an eager NumPy- or CuPy-compatible function on each
block.  Its ``input_core_dims`` and ``output_core_dims`` use
:func:`xarray.apply_ufunc` terminology: a *core dimension* is an axis consumed
or produced as a complete vector by this particular function, not an xsnow
dataset-dimension category.  Undeclared axes are *outer dimensions* that
xarray broadcasts and Dask may split.

Normally, ``layer`` is the only core dimension.  A block function then receives
an eager array shaped ``(..., n_layers)``: arbitrary leading axes contain one
or more profiles and the complete layer axis is last.  The function must
vectorize over those leading axes.

This example expresses each layer's density as a percentage of the maximum
density in its profile::

    from xsnow.extensions import (
        apply_profile_block,
        float_result_dtype,
        get_backend,
    )

    def _percentage_of_profile_max_block(density_block):
        # Shape: (..., n_layers); the complete layer axis is last.
        xp = get_backend(density_block)
        density_block = xp.asarray(density_block)
        profile_maximum = xp.nanmax(
            density_block, axis=-1, keepdims=True
        )
        return 100 * density_block / profile_maximum

    density = ds.data["density"]
    result = apply_profile_block(
        _percentage_of_profile_max_block,
        density,
        output_dtype=float_result_dtype(density),
    )

For a block shaped ``(2, 8, 150)``, reducing the last axis gives
``(2, 8, 1)``.  Division broadcasts each profile maximum over its 150 layers,
returning percentages with shape ``(2, 8, 150)``.  The result retains xarray
labels and, when the input is Dask-backed, stays lazy until computed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from types import ModuleType
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    import cupy
    import dask.array as da

    from ..core import xsnowDataset

__all__ = [
    'apply_profile_block',
    'float_result_dtype',
    'get_backend',
]


def get_backend(
    array_like: (
        np.ndarray
        | cupy.ndarray
        | da.Array
        | xr.DataArray
        | xr.Dataset
        | xsnowDataset
    ),
) -> ModuleType:
    """Return the NumPy or CuPy numerical backend for an array-like object.

    Parameters
    ----------
    array_like : array-like, xarray object, or xsnowDataset
        An eager array block, Dask array, xarray DataArray or Dataset, or
        xsnowDataset. CuPy arrays, including the metadata arrays of
        CuPy-backed Dask collections, select CuPy; all other inputs select
        NumPy. For a Dataset, the first data variable determines the backend.

    Returns
    -------
    module
        :mod:`numpy` or :mod:`cupy`.

    Notes
    -----
    Importing CuPy is deferred until a CuPy array is encountered, so CPU-only
    installations do not need CuPy installed. Inspecting a Dask collection's
    empty metadata array does not trigger computation.
    """
    # Import locally to avoid making the public block helpers part of the
    # core module's import graph.
    from ..core import xsnowDataset

    if isinstance(array_like, xsnowDataset):
        if array_like.data is None:
            return np
        array_like = array_like.data

    if isinstance(array_like, xr.Dataset):
        first_name = next(iter(array_like.data_vars), None)
        if first_name is None:
            return np
        array = array_like[first_name].data
    elif isinstance(array_like, xr.DataArray):
        array = array_like.data
    else:
        array = array_like

    # Dask exposes a zero-length array representative of its chunk backend.
    # Follow nested metadata defensively without importing Dask itself.
    seen = set()
    while hasattr(array, '_meta') and id(array) not in seen:
        seen.add(id(array))
        array = array._meta

    if type(array).__module__.split('.')[0] == 'cupy':
        import cupy

        return cupy
    return np


def float_result_dtype(*arrays: xr.DataArray) -> np.dtype:
    """Return a floating result dtype derived from xarray inputs.

    The result follows NumPy's type-promotion rules and is at least
    ``float32``.  Supplying this dtype to :func:`apply_profile_block` avoids
    evaluating a lazy block merely to infer its output dtype.
    """
    return np.result_type(*(array.dtype for array in arrays), np.float32)


def apply_profile_block(
    block_function: Callable[..., Any],
    *arrays: xr.DataArray,
    output_dtype: Any = None,
    kwargs: dict[str, Any] | None = None,
    layer_dim: str = 'layer',
    input_core_dims: Sequence[Sequence[str]] | None = None,
    output_core_dims: Sequence[Sequence[str]] | None = None,
    output_sizes: Mapping[str, int] | None = None,
) -> xr.DataArray | tuple[xr.DataArray, ...]:
    """Apply an eager array function to blocks of complete snow profiles.

    Parameters
    ----------
    block_function : callable
        Function accepting eager arrays plus ``kwargs`` and vectorizing over
        their leading outer axes.  Trailing input and output axes follow
        ``input_core_dims`` and ``output_core_dims``, respectively.
    *arrays : xarray.DataArray
        Inputs to align and broadcast.  By default, each must contain
        ``layer_dim``; explicit ``input_core_dims`` may declare otherwise.
    output_dtype : dtype-like or sequence of dtype-like, optional
        Declared dtype, one per output.  Defaults to
        :func:`float_result_dtype` over ``arrays`` for one output.  Dask trusts
        this declaration without running ``block_function``, so it must match
        the function's actual result.
    kwargs : dict, optional
        Keyword arguments forwarded unchanged to ``block_function``.
    layer_dim : str, default ``"layer"``
        Profile dimension used by the default core-dimension declarations.
    input_core_dims : sequence of sequences of str, optional
        Complete trailing axes passed to ``block_function``, one sequence per
        input.  The default is ``[[layer_dim], ...]``.  Use an empty sequence
        for an input that only broadcasts over outer dimensions.
    output_core_dims : sequence of sequences of str, optional
        Trailing axes returned per output.  The default ``[[layer_dim]]``
        gives one layer-resolved output; ``[[]]`` gives one profile reduction.
        A new axis such as ``[['group']]`` also requires ``output_sizes``.
    output_sizes : mapping, optional
        Sizes of output core dimensions not present on an input.

    Returns
    -------
    xarray.DataArray or tuple of xarray.DataArray
        Results with the broadcast outer dimensions and the output core
        dimensions.  They are lazy if any input is Dask-backed.

    Notes
    -----
    Each Dask input core dimension is rechunked as one complete axis; outer
    chunks remain parallel.  For example, ``[['layer'], []]`` supplies a layer
    vector and a broadcasting value.  Outputs ``[[]]`` reduce each profile,
    while ``[['group']]`` creates a vector sized through ``output_sizes``.
    """
    if output_dtype is None:
        output_dtype = float_result_dtype(*arrays)
    if input_core_dims is None:
        input_core_dims = [[layer_dim]] * len(arrays)
    if output_core_dims is None:
        output_core_dims = [[layer_dim]]
    dtypes = (
        [output_dtype]
        if isinstance(output_dtype, (str, np.dtype, type))
        or not isinstance(output_dtype, Sequence)
        else list(output_dtype)
    )
    if len(dtypes) != len(output_core_dims):
        raise ValueError(
            "One output_dtype is required per output_core_dims entry."
        )

    prepared = []
    for array, core in zip(arrays, input_core_dims):
        for dim in core:
            if dim not in array.dims:
                raise ValueError(
                    f"Profile calculation requires a {dim!r} dimension."
                )
            axis = array.get_axis_num(dim)
            if array.chunks is not None and len(array.chunks[axis]) > 1:
                array = array.chunk({dim: -1})
        prepared.append(array)

    gufunc_kwargs = {}
    if output_sizes:
        gufunc_kwargs['output_sizes'] = dict(output_sizes)
    return xr.apply_ufunc(
        block_function,
        *prepared,
        input_core_dims=[list(core) for core in input_core_dims],
        output_core_dims=[list(core) for core in output_core_dims],
        dask='parallelized',
        vectorize=False,
        output_dtypes=dtypes,
        dask_gufunc_kwargs=gufunc_kwargs,
        kwargs=kwargs or {},
    )
