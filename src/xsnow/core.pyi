from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Concatenate, Final, overload

import xarray as xr

from xsnow.combiners import (
    combine_first as _combine_first,
    fill_gaps_with as _fill_gaps_with,
    overwrite_with as _overwrite_with,
    update as _update,
)
from xsnow.io.memory import check_memory_budget as _check_memory_budget
from xsnow.io.netcdf import to_netcdf as _to_netcdf
from xsnow.io.pro import to_pro as _to_pro
from xsnow.io.smet import to_smet as _to_smet
from xsnow.io.zarr import to_zarr as _to_zarr
from xsnow.utils import print as _print, reorder_dims as _reorder_dims
from xsnow.xsnow_io import append_latest as _append_latest, trim_padding as _trim_padding

# Alias an xr.Dataset method as one returning xsnowDataset, with self
# rewrapped from xr.Dataset to xsnowDataset and the same parameter list.
def _rewraps[**_XArrayParams](
    m: Callable[Concatenate[xr.Dataset, _XArrayParams], Any],
) -> Callable[Concatenate[xsnowDataset, _XArrayParams], xsnowDataset]: ...


# Alias an xr.Dataset method as-is, with self rewrapped from xr.Dataset to
# xsnowDataset, the same parameter list, and the same return type.
def _forwards[**_XArrayParams, _R](
    m: Callable[Concatenate[xr.Dataset, _XArrayParams], _R],
) -> Callable[Concatenate[xsnowDataset, _XArrayParams], _R]: ...


# Alias one of xr.Dataset's equals/identical/broadcast_equals, with self
# rewrapped from xr.Dataset to xsnowDataset and `other` widened to accept
# either xsnowDataset or xr.Dataset.
def _forward_equals(
    m: Callable[[xr.Dataset, xr.Dataset], bool],
) -> Callable[[xsnowDataset, xsnowDataset | xr.Dataset], bool]: ...


# Dummy xr.Dataset for getting property type signatures
_ds: xr.Dataset


class xsnowDataset:
    """Typed public surface for xsnow's xarray-backed dataset wrapper.

    The runtime class intentionally uses composition: an ``xsnowDataset`` stores
    the real ``xarray.Dataset`` in ``.data`` and forwards unknown attributes to
    that object. This stub mirrors that design instead of pretending that
    ``xsnowDataset`` inherits from ``xr.Dataset``.

    The explicit method list is deliberately curated. It covers xsnow-native
    methods and the common xarray operations used in normal xsnow workflows,
    while keeping IDE autocomplete focused on the xsnow API. For less common
    xarray operations, use ``.to_xarray()`` when precise static typing matters.
    Dynamic coordinates and data variables, such as ``xs.time`` or
    ``xs.density``, are accepted through ``__getattr__`` and therefore typed as
    ``Any``.
    """

    data: xr.Dataset

    def __init__(self, source: xr.Dataset | xsnowDataset) -> None: ...

    # Mirrors xarray.Dataset.__getitem__'s typing. pyright's reportOverlappingOverload
    # warns (possibly incorrectly?) so suppress the warning.
    @overload
    def __getitem__(self, key: Hashable) -> xr.DataArray: ...
    @overload
    def __getitem__(self, key: Iterable[Hashable]) -> xr.Dataset: ...  # pyright: ignore[reportOverlappingOverload]

    def __setitem__(
        self,
        key: Hashable | list[Hashable] | Mapping[Any, Any],
        value: Any,
    ) -> None: ...
    def __contains__(self, key: object) -> bool: ...
    def __dir__(self) -> Sequence[str]: ...
    def __repr__(self, *, force_details: bool = False) -> str: ...

    # Escape hatch for forwarded xarray attributes, dynamic coordinates, and
    # data variables. This keeps valid runtime use type-checkable, but the
    # returned object is intentionally Any.
    def __getattr__(self, name: str) -> Any: ...

    # Explicitly defined xsnowDataset public methods.
    def copy(self, deep: bool = False) -> xsnowDataset: ...
    def merge_smet(
        self,
        source: str | list[str] | Path | list[Path],
        var_prefix: str | None = None,
        **kwargs: Any,
    ) -> xsnowDataset: ...
    def to_xarray(self) -> xr.Dataset: ...
    def compute_hs(self) -> xsnowDataset: ...
    def compute_z_depth(self) -> xsnowDataset: ...
    def update_profile_status(self) -> xsnowDataset: ...
    def profile_status_summary(self) -> None: ...
    def to_gpu(self) -> xsnowDataset: ...
    def to_cpu(self) -> xsnowDataset: ...

    # Methods registered from I/O and helper modules. These are registered by
    # default on xsnow import. Each is bound directly to its implementing free 
    # function, so the stub signature and docstring are identical to the runtime
    # signature and docstring.
    append_latest = _append_latest
    trim_padding = _trim_padding
    reorder_dims = _reorder_dims
    print = _print
    to_pro = _to_pro
    to_smet = _to_smet
    to_netcdf = _to_netcdf
    to_zarr = _to_zarr
    check_memory_budget = _check_memory_budget
    fill_gaps_with = _fill_gaps_with
    overwrite_with = _overwrite_with
    update = _update
    combine_first = _combine_first

    # Basic dataset-state properties forwarded from xarray. Each derives its type
    # from instance access on _ds, which yields a plain value type with no link to
    # xarray's property docstring, so the boilerplate docstrings below are added explicitly.
    attrs: Final = _ds.attrs
    """Forwarded to ``xarray.Dataset.attrs``; see the xarray documentation for details."""
    coords: Final = _ds.coords
    """Forwarded to ``xarray.Dataset.coords``; see the xarray documentation for details."""
    data_vars: Final = _ds.data_vars
    """Forwarded to ``xarray.Dataset.data_vars``; see the xarray documentation for details."""
    dims: Final = _ds.dims
    """Forwarded to ``xarray.Dataset.dims``; see the xarray documentation for details."""
    encoding: Final = _ds.encoding
    """Forwarded to ``xarray.Dataset.encoding``; see the xarray documentation for details."""
    indexes: Final = _ds.indexes
    """Forwarded to ``xarray.Dataset.indexes``; see the xarray documentation for details."""
    sizes: Final = _ds.sizes
    """Forwarded to ``xarray.Dataset.sizes``; see the xarray documentation for details."""
    variables: Final = _ds.variables
    """Forwarded to ``xarray.Dataset.variables``; see the xarray documentation for details."""

    # Forwarded xarray methods that rewrap the return value as xsnowDataset. The
    # ParamSpec captures xarray's signature from the source method, which also
    # carries xarray's docstring through to hover.
    sel = _rewraps(xr.Dataset.sel)
    isel = _rewraps(xr.Dataset.isel)
    where = _rewraps(xr.Dataset.where)
    squeeze = _rewraps(xr.Dataset.squeeze)
    assign = _rewraps(xr.Dataset.assign)
    assign_coords = _rewraps(xr.Dataset.assign_coords)
    rename = _rewraps(xr.Dataset.rename)
    swap_dims = _rewraps(xr.Dataset.swap_dims)
    expand_dims = _rewraps(xr.Dataset.expand_dims)
    drop_vars = _rewraps(xr.Dataset.drop_vars)
    drop_sel = _rewraps(xr.Dataset.drop_sel)
    drop_dims = _rewraps(xr.Dataset.drop_dims)
    transpose = _rewraps(xr.Dataset.transpose)
    sortby = _rewraps(xr.Dataset.sortby)
    fillna = _rewraps(xr.Dataset.fillna)
    dropna = _rewraps(xr.Dataset.dropna)
    mean = _rewraps(xr.Dataset.mean)
    sum = _rewraps(xr.Dataset.sum)
    min = _rewraps(xr.Dataset.min)
    max = _rewraps(xr.Dataset.max)
    count = _rewraps(xr.Dataset.count)
    any = _rewraps(xr.Dataset.any)
    all = _rewraps(xr.Dataset.all)
    compute = _rewraps(xr.Dataset.compute)
    load = _rewraps(xr.Dataset.load)
    persist = _rewraps(xr.Dataset.persist)
    chunk = _rewraps(xr.Dataset.chunk)
    unify_chunks = _rewraps(xr.Dataset.unify_chunks)

    # Forwarded xarray methods that do not return xsnowDataset. As with _rewraps,
    # the ParamSpec capture forwards xarray's signature and docstring.
    to_array = _forwards(xr.Dataset.to_array)
    to_dataframe = _forwards(xr.Dataset.to_dataframe)

    # Forwarded xarray comparison methods, widened to accept xsnowDataset inputs.
    # Unlike _rewraps/_forwards this uses a concrete signature (no ParamSpec), so
    # xarray's docstring is not forwarded, so we include boilerplate docstrings.
    equals = _forward_equals(xr.Dataset.equals)
    """Forwarded to ``xarray.Dataset.equals``; see the xarray documentation for details."""
    identical = _forward_equals(xr.Dataset.identical)
    """Forwarded to ``xarray.Dataset.identical``; see the xarray documentation for details."""
    broadcast_equals = _forward_equals(xr.Dataset.broadcast_equals)
    """Forwarded to ``xarray.Dataset.broadcast_equals``; see the xarray documentation for details."""


class DatasetDecorator(xsnowDataset):
    def __init__(self, base: xsnowDataset, *, data: xr.Dataset | xsnowDataset | None = None) -> None: ...
    @property
    def base(self) -> xsnowDataset: ...
