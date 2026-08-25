"""Utilities for efficient blockwise snow-profile computations.

Large, lazy xarray datasets are commonly backed by Dask.  Evaluating such a
dataset as one NumPy or CuPy array would materialize the entire dataset in
memory.  Instead, Dask builds a lazy task graph and, when the result is
requested, evaluates manageable array blocks eagerly.  For a profile-wide
calculation, each block must contain the complete layer axis, but it can hold
only a subset of the dataset's times, locations, slopes, or other outer
dimensions.

:func:`apply_profile_block` sets up this execution model: it aligns the xarray
inputs, keeps only the profile axis in one chunk, and applies an eager NumPy-
or CuPy-compatible function to each block.  The tools in this module therefore
let built-in and third-party extensions process blocks of complete profiles
efficiently and in parallel.

A profile block function must:

* accept eager array blocks, not xarray or Dask objects;
* treat its last axis as the complete layer axis; and
* return an array with the same outer axes and complete layer axis.

With eager input, one call may contain every profile in the dataset.  With
Dask-backed input, one call receives one block containing one or more complete
profiles.  Block functions should therefore be vectorized over all axes except
the last rather than assuming that a call represents exactly one profile.

For example, an external extension can implement a block function and retain
lazy execution as follows::

    from xsnow.extensions import (
        apply_profile_block,
        float_result_dtype,
        get_array_backend,
    )

    def _distance_from_profile_min_block(values):
        xp = get_array_backend(values)
        values = xp.asarray(values)
        minimum = xp.nanmin(values, axis=-1)
        return values - minimum[..., None]

    result = apply_profile_block(
        _distance_from_profile_min_block,
        ds.data["grain_size"],
        output_dtype=float_result_dtype(ds.data["grain_size"]),
    )

The returned :class:`xarray.DataArray` is eager for eager input and remains
lazy for Dask-backed input.  Exceptions from a lazy block function are raised
when the returned array is computed.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import xarray as xr

__all__ = [
    'apply_profile_block',
    'float_result_dtype',
    'get_array_backend',
]


def get_array_backend(array: Any) -> Any:
    """Return the NumPy or CuPy module for an eager array block.

    Parameters
    ----------
    array : array-like
        An eager array passed to a profile block function.  CuPy arrays select
        CuPy;
        all other inputs select NumPy.

    Returns
    -------
    module
        :mod:`numpy` or :mod:`cupy`.

    Notes
    -----
    Importing CuPy is deferred until a CuPy array is encountered, so CPU-only
    installations do not need CuPy installed.
    """
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
    output_dtype: Any,
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
        Function accepting one eager array per input plus ``kwargs``.  The
        complete profile axis is last in every array supplied to the function.
        The return value must have the same outer axes and one complete
        profile axis (or the core axes named in ``output_core_dims``).
    *arrays : xarray.DataArray
        Inputs containing ``layer_dim``.  Xarray aligns and broadcasts their
        labelled outer dimensions before invoking ``block_function``.
    output_dtype : dtype-like or sequence of dtype-like
        Block result dtype, one per output.  It must be supplied explicitly
        so Dask does not execute the function while constructing the lazy
        graph.
    kwargs : dict, optional
        Keyword arguments forwarded unchanged to ``block_function``.
    layer_dim : str, default ``"layer"``
        Name of the complete profile dimension.
    input_core_dims : sequence of sequences of str, optional
        Core dimensions per input, for inputs that carry the profile axis
        under another name or no profile axis at all (an empty sequence).
        Default: ``layer_dim`` for every input.  Every named core dimension
        is kept in one chunk.
    output_core_dims : sequence of sequences of str, optional
        Core dimensions per output.  Default: ``[[layer_dim]]``, one output
        with the input profile axis.  A reduction over the profile returns
        ``[[]]``; a new axis of a different length is named here and sized
        in ``output_sizes``.
    output_sizes : mapping, optional
        Length of every output core dimension that does not exist on the
        inputs.

    Returns
    -------
    xarray.DataArray or tuple of xarray.DataArray
        Results with the broadcast outer dimensions and the output core
        dimensions.  They are lazy if any input is Dask-backed.

    Notes
    -----
    If a Dask input splits a core dimension into multiple chunks, only that
    core dimension is rechunked.  Outer chunks remain available for
    parallelism.  Each scheduled block call can therefore contain multiple
    profiles but always contains every layer belonging to them.  Inputs
    without an outer dimension of another input reach the block function
    with a size-1 axis there (NumPy broadcasting rules).
    """
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
