from __future__ import annotations

from typing import Callable, Mapping, Optional, Literal
import xarray as xr
import numpy as np

from ..core import xsnowDataset, DatasetDecorator


class Classification(DatasetDecorator):
    """
    Generic classification extension.

    This decorator computes a classification over the input dataset (or a
    subset/variable thereof) using a user-supplied function, then writes the
    result into the dataset as either a *coordinate* or a *data variable*.

    Parameters
    ----------
    base : xsnowDataset
        The base dataset wrapper to extend.
    func : Callable[[xr.Dataset], xr.DataArray] | Callable[[xr.DataArray], xr.DataArray]
        A callable that takes either the full xr.Dataset (self.data) or one of
        its variables (if you pass `input_var`), and returns an xr.DataArray of
        classification labels. The returned DataArray must align with existing
        dimensions (a subset or full set of dims) and be broadcastable within
        the dataset.
    output_name : str, default "class_id"
        Name of the resulting coordinate or data variable.
    output_kind : {"coord","data"}, default "coord"
        Whether to store the classification as a coordinate (assign_coords) or
        as a data variable (assign).
    input_var : Optional[str], default None
        If provided, the classifier `func` will be called with `self.data[input_var]`
        instead of the full dataset.
    attrs : Optional[Mapping[str, str]], default None
        Optional metadata to attach to the resulting array (e.g., units, description,
        mapping dict for label lookups, etc.).
    data : Optional[xr.Dataset], default None
        Internal use: re-wrapping path when chained operations return a new xr.Dataset.

    Notes
    -----
    - Keep the callable side-effect free; it should only compute a classification array.
    - The returned DataArray’s `dims` should be a subset of the dataset’s dims
      (e.g., ("location","time") or ("location","time","slope","realization","layer")),
      or be broadcastable to them.
    """

    def __init__(
        self,
        base: xsnowDataset,
        *,
        func: Callable,
        output_name: str = "class_id",
        output_kind: Literal["coord", "data"] = "coord",
        input_var: Optional[str] = None,
        attrs: Optional[Mapping[str, str]] = None,
        data: Optional[xr.Dataset] = None,
    ):
        super().__init__(base, data=data)
        self._decorator_args = (base,)
        self._decorator_kwargs = dict(
            func=func,
            output_name=output_name,
            output_kind=output_kind,
            input_var=input_var,
            attrs=attrs,
        )

        # First-time construction: compute & attach classification
        if data is None:
            new_ds = self._compute_and_attach(
                func=func, output_name=output_name, output_kind=output_kind,
                input_var=input_var, attrs=attrs
            )
            # Replace this instance's underlying data with the updated dataset
            # (and keep behaving as a Classification)
            self.__dict__.update(self._rewrap(new_ds).__dict__)

    def _compute_and_attach(
        self,
        *,
        func: Callable,
        output_name: str,
        output_kind: Literal["coord", "data"],
        input_var: Optional[str],
        attrs: Optional[Mapping[str, str]],
    ) -> xr.Dataset:
        xds = self.data
        target = xds[input_var] if input_var else xds
        out = func(target)
        if not isinstance(out, xr.DataArray):
            raise TypeError(f"classifier must return xr.DataArray, got {type(out).__name__}")
        if out.name is None:
            out = out.rename(output_name)
        if attrs:
            out = out.copy()
            out.attrs.update(attrs)

        if output_kind == "coord":
            return xds.assign_coords({output_name: out})
        elif output_kind == "data":
            return xds.assign({output_name: out})
        else:
            raise ValueError("output_kind must be 'coord' or 'data'.")
