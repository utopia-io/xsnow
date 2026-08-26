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
        get_backend,
    )

    def _distance_from_profile_min_block(values):
        xp = get_backend(values)
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
    'get_array_backend',
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


#: The name third-party extensions import this under. punstable was written
#: against it, and renaming across repos in step would need them to land
#: together; keep both until the platform settles on one.
get_array_backend = get_backend


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
        Function accepting one eager array per input plus ``kwargs``.  The
        complete profile axis is last in every array supplied to the function.
        The return value must have the same outer axes and one complete
        profile axis (or the core axes named in ``output_core_dims``).
    *arrays : xarray.DataArray
        Inputs containing ``layer_dim``.  Xarray aligns and broadcasts their
        labelled outer dimensions before invoking ``block_function``.
    output_dtype : dtype-like or sequence of dtype-like, optional
        Block result dtype, one per output.  Default:
        :func:`float_result_dtype` over ``arrays``, a single floating
        output.  A dtype is never inferred by calling ``block_function``,
        which would execute a lazy graph while it is being built: on the
        Dask path it is the dtype the graph advertises, unverified, so a
        block function returning another dtype makes the lazy and the
        computed dtype disagree.  Eager input keeps the dtype the block
        function actually returns.
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
