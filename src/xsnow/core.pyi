from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, overload

import pandas as pd
import xarray as xr

from xsnow.combiners import (
    combine_first as _combine_first,
    fill_gaps_with as _fill_gaps_with,
    overwrite_with as _overwrite_with,
    update as _update,
)
from xsnow.io.netcdf import to_netcdf as _to_netcdf
from xsnow.io.pro import to_pro as _to_pro
from xsnow.io.smet import to_smet as _to_smet
from xsnow.io.zarr import to_zarr as _to_zarr
from xsnow.utils import print as _print, reorder_dims as _reorder_dims
from xsnow.xsnow_io import append_latest as _append_latest, trim_padding as _trim_padding

# Mirrors xarray.core.types.ErrorOptions / ErrorOptionsWithWarn (private module).
ErrorOptions = Literal["raise", "ignore"]
ErrorOptionsWithWarn = Literal["raise", "warn", "ignore"]


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
    # function, so the stub signature is identical to the runtime signature.
    append_latest = _append_latest
    trim_padding = _trim_padding
    reorder_dims = _reorder_dims
    print = _print
    to_pro = _to_pro
    to_smet = _to_smet
    to_netcdf = _to_netcdf
    to_zarr = _to_zarr
    fill_gaps_with = _fill_gaps_with
    overwrite_with = _overwrite_with
    update = _update
    combine_first = _combine_first

    # Basic dataset-state properties from xarray 
    @property
    def attrs(self) -> dict[Any, Any]:
        """Forwarded to ``xarray.Dataset.attrs``; see the xarray documentation for details."""
        ...
    @property
    def coords(self) -> Any:
        """Forwarded to ``xarray.Dataset.coords``; see the xarray documentation for details."""
        ...
    @property
    def data_vars(self) -> Any:
        """Forwarded to ``xarray.Dataset.data_vars``; see the xarray documentation for details."""
        ...
    @property
    def dims(self) -> Mapping[Hashable, int]:
        """Forwarded to ``xarray.Dataset.dims``; see the xarray documentation for details."""
        ...
    @property
    def encoding(self) -> dict[Any, Any]:
        """Forwarded to ``xarray.Dataset.encoding``; see the xarray documentation for details."""
        ...
    @property
    def indexes(self) -> Any:
        """Forwarded to ``xarray.Dataset.indexes``; see the xarray documentation for details."""
        ...
    @property
    def sizes(self) -> Mapping[Hashable, int]:
        """Forwarded to ``xarray.Dataset.sizes``; see the xarray documentation for details."""
        ...
    @property
    def variables(self) -> Mapping[Hashable, Any]:
        """Forwarded to ``xarray.Dataset.variables``; see the xarray documentation for details."""
        ...

    # Curated methods that return a Dataset at runtime and are rewrapped
    # by xsnowDataset.__getattr__.
    def sel(
        self,
        indexers: Mapping[Any, Any] | None = None,
        method: str | None = None,
        tolerance: int | float | Iterable[int | float] | None = None,
        drop: bool = False,
        **indexers_kwargs: Any,
    ) -> xsnowDataset:
        """Forwarded to ``xarray.Dataset.sel``; see the xarray documentation for details."""
        ...
    def isel(
        self,
        indexers: Mapping[Any, Any] | None = None,
        drop: bool = False,
        missing_dims: ErrorOptionsWithWarn = "raise",
        **indexers_kwargs: Any,
    ) -> xsnowDataset:
        """Forwarded to ``xarray.Dataset.isel``; see the xarray documentation for details."""
        ...
    def where(self, cond: Any, other: Any = ..., drop: bool = False) -> xsnowDataset:
        """Forwarded to ``xarray.Dataset.where``; see the xarray documentation for details."""
        ...
    def squeeze(
        self,
        dim: Hashable | Iterable[Hashable] | None = None,
        drop: bool = False,
        axis: int | Iterable[int] | None = None,
    ) -> xsnowDataset:
        """Forwarded to ``xarray.Dataset.squeeze``; see the xarray documentation for details."""
        ...
    def assign(self, variables: Mapping[Any, Any] | None = None, **variables_kwargs: Any) -> xsnowDataset:
        """Forwarded to ``xarray.Dataset.assign``; see the xarray documentation for details."""
        ...
    def assign_coords(self, coords: Mapping[Any, Any] | None = None, **coords_kwargs: Any) -> xsnowDataset:
        """Forwarded to ``xarray.Dataset.assign_coords``; see the xarray documentation for details."""
        ...
    def rename(self, name_dict: Mapping[Any, Hashable] | None = None, **names: Hashable) -> xsnowDataset:
        """Forwarded to ``xarray.Dataset.rename``; see the xarray documentation for details."""
        ...
    def swap_dims(self, dims_dict: Mapping[Any, Hashable] | None = None, **dims_kwargs: Hashable) -> xsnowDataset:
        """Forwarded to ``xarray.Dataset.swap_dims``; see the xarray documentation for details."""
        ...
    def expand_dims(
        self,
        dim: Hashable | Sequence[Hashable] | Mapping[Any, Any] | None = None,
        axis: int | Sequence[int] | None = None,
        create_index_for_new_dim: bool = True,
        **dim_kwargs: Any,
    ) -> xsnowDataset:
        """Forwarded to ``xarray.Dataset.expand_dims``; see the xarray documentation for details."""
        ...
    def drop_vars(
        self,
        names: str | Iterable[Hashable] | Callable[..., str | Iterable[Hashable]],
        *,
        errors: ErrorOptions = "raise",
    ) -> xsnowDataset:
        """Forwarded to ``xarray.Dataset.drop_vars``; see the xarray documentation for details."""
        ...
    def drop_sel(
        self,
        labels: Mapping[Any, Any] | None = None,
        *,
        errors: ErrorOptions = "raise",
        **labels_kwargs: Any,
    ) -> xsnowDataset:
        """Forwarded to ``xarray.Dataset.drop_sel``; see the xarray documentation for details."""
        ...
    def drop_dims(self, drop_dims: str | Iterable[Hashable], *, errors: ErrorOptions = "raise") -> xsnowDataset:
        """Forwarded to ``xarray.Dataset.drop_dims``; see the xarray documentation for details."""
        ...
    def transpose(self, *dim: Hashable, missing_dims: ErrorOptionsWithWarn = "raise") -> xsnowDataset:
        """Forwarded to ``xarray.Dataset.transpose``; see the xarray documentation for details."""
        ...
    def sortby(
        self,
        variables: Hashable | xr.DataArray | Sequence[Hashable | xr.DataArray] | Callable[..., Any],
        ascending: bool = True,
    ) -> xsnowDataset:
        """Forwarded to ``xarray.Dataset.sortby``; see the xarray documentation for details."""
        ...
    def fillna(self, value: Any) -> xsnowDataset:
        """Forwarded to ``xarray.Dataset.fillna``; see the xarray documentation for details."""
        ...
    def dropna(
        self,
        dim: Hashable,
        *,
        how: Literal["any", "all"] = "any",
        thresh: int | None = None,
        subset: Iterable[Hashable] | None = None,
    ) -> xsnowDataset:
        """Forwarded to ``xarray.Dataset.dropna``; see the xarray documentation for details."""
        ...
    def mean(
        self,
        dim: str | Iterable[Hashable] | None = None,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        **kwargs: Any,
    ) -> xsnowDataset:
        """Forwarded to ``xarray.Dataset.mean``; see the xarray documentation for details."""
        ...
    def sum(
        self,
        dim: str | Iterable[Hashable] | None = None,
        *,
        skipna: bool | None = None,
        min_count: int | None = None,
        keep_attrs: bool | None = None,
        **kwargs: Any,
    ) -> xsnowDataset:
        """Forwarded to ``xarray.Dataset.sum``; see the xarray documentation for details."""
        ...
    def min(
        self,
        dim: str | Iterable[Hashable] | None = None,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        **kwargs: Any,
    ) -> xsnowDataset:
        """Forwarded to ``xarray.Dataset.min``; see the xarray documentation for details."""
        ...
    def max(
        self,
        dim: str | Iterable[Hashable] | None = None,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        **kwargs: Any,
    ) -> xsnowDataset:
        """Forwarded to ``xarray.Dataset.max``; see the xarray documentation for details."""
        ...
    def count(
        self,
        dim: str | Iterable[Hashable] | None = None,
        *,
        keep_attrs: bool | None = None,
        **kwargs: Any,
    ) -> xsnowDataset:
        """Forwarded to ``xarray.Dataset.count``; see the xarray documentation for details."""
        ...
    def any(
        self,
        dim: str | Iterable[Hashable] | None = None,
        *,
        keep_attrs: bool | None = None,
        **kwargs: Any,
    ) -> xsnowDataset:
        """Forwarded to ``xarray.Dataset.any``; see the xarray documentation for details."""
        ...
    def all(
        self,
        dim: str | Iterable[Hashable] | None = None,
        *,
        keep_attrs: bool | None = None,
        **kwargs: Any,
    ) -> xsnowDataset:
        """Forwarded to ``xarray.Dataset.all``; see the xarray documentation for details."""
        ...
    def compute(self, **kwargs: Any) -> xsnowDataset:
        """Forwarded to ``xarray.Dataset.compute``; see the xarray documentation for details."""
        ...
    def load(self, **kwargs: Any) -> xsnowDataset:
        """Forwarded to ``xarray.Dataset.load``; see the xarray documentation for details."""
        ...
    def persist(self, **kwargs: Any) -> xsnowDataset:
        """Forwarded to ``xarray.Dataset.persist``; see the xarray documentation for details."""
        ...
    # chunked_array_type and **chunks_kwargs reference xarray-internal types
    # (xarray.namedarray.parallelcompat / xarray.core.types); left untyped.
    def chunk(self, *args: Any, **kwargs: Any) -> xsnowDataset:
        """Forwarded to ``xarray.Dataset.chunk``; see the xarray documentation for details."""
        ...
    def unify_chunks(self) -> xsnowDataset:
        """Forwarded to ``xarray.Dataset.unify_chunks``; see the xarray documentation for details."""
        ...

    # Common conversions and comparisons from xarray that do not return xsnowDataset.
    def to_array(self, dim: Hashable = "variable", name: Hashable | None = None) -> xr.DataArray:
        """Forwarded to ``xarray.Dataset.to_array``; see the xarray documentation for details."""
        ...
    def to_dataframe(self, dim_order: Sequence[Hashable] | None = None) -> pd.DataFrame:
        """Forwarded to ``xarray.Dataset.to_dataframe``; see the xarray documentation for details."""
        ...
    def equals(self, other: xsnowDataset | xr.Dataset) -> bool:
        """Forwarded to ``xarray.Dataset.equals``; see the xarray documentation for details."""
        ...
    def identical(self, other: xsnowDataset | xr.Dataset) -> bool:
        """Forwarded to ``xarray.Dataset.identical``; see the xarray documentation for details."""
        ...
    def broadcast_equals(self, other: xsnowDataset | xr.Dataset) -> bool:
        """Forwarded to ``xarray.Dataset.broadcast_equals``; see the xarray documentation for details."""
        ...


class DatasetDecorator(xsnowDataset):
    def __init__(self, base: xsnowDataset, *, data: xr.Dataset | xsnowDataset | None = None) -> None: ...
    @property
    def base(self) -> xsnowDataset: ...
