from __future__ import annotations

import re
from typing import Callable, Dict, Literal, Mapping, Optional

import numpy as np
import xarray as xr

from ..core import xsnowDataset
from ..plugin import register_xsnowDataset_method


# ---------------------- internal helpers (criteria parsing) -------------------

_OP_MAP = {
    ">":  np.greater,
    "<":  np.less,
    ">=": np.greater_equal,
    "<=": np.less_equal,
    "==": np.equal,
    "!=": np.not_equal,
}
# Note: keep '==' ahead of '=' in the regex alternation. We accept common ops.
_COND_RE = re.compile(r"^\s*(\w+)\s*([<>=!]=?|==)\s*(-?\d+\.?\d*)\s*$")


def _parse_single_condition(ds_xr: xr.Dataset, cond: str) -> xr.DataArray:
    """
    Parse an atomic condition like 'density > 300' into a boolean mask DataArray.
    Aligns/broadcasts by named dims automatically (xarray semantics).
    """
    m = _COND_RE.match(cond)
    if not m:
        raise ValueError(f"Could not parse condition: {cond!r}")
    var, op, val_str = m.groups()

    if var not in ds_xr:
        raise ValueError(f"Variable {var!r} not found in dataset.")
    if op not in _OP_MAP:
        raise ValueError(f"Unsupported operator: {op!r}")

    val = float(val_str)
    da = ds_xr[var]
    op_func = _OP_MAP[op]

    mask = op_func(da, val).astype(bool)
    mask.name = f"mask_{var}_{op}_{val_str}"
    return mask


def _combine_masks(m1: xr.DataArray, logical: str, m2: xr.DataArray) -> xr.DataArray:
    """Combine boolean masks via '&' or '|' with xarray alignment."""
    if logical == "&":
        return (m1 & m2).astype(bool)
    elif logical == "|":
        return (m1 | m2).astype(bool)
    raise ValueError(f"Unsupported logical operator: {logical!r}")


def _boolean_criteria_to_mask(ds_xr: xr.Dataset, criteria: str) -> xr.DataArray:
    """
    Turn 'density > 300 & grain_size < 0.5' into a boolean mask.
    Left-to-right evaluation of '&' / '|' (no parentheses/precedence support).
    """
    parts = re.split(r"\s*([&|])\s*", criteria.strip())
    if not parts or not parts[0] or parts[0] in ("&", "|"):
        raise ValueError(f"Invalid criteria string: {criteria!r}")

    mask = None
    i = 0
    while i < len(parts):
        piece = parts[i].strip()
        this_mask = _parse_single_condition(ds_xr, piece)
        if mask is None:
            mask = this_mask
        else:
            logical = parts[i - 1].strip()
            mask = _combine_masks(mask, logical, this_mask)
        i += 2

    if mask is None:
        raise ValueError(f"Failed to build mask from criteria: {criteria!r}")
    return mask


def _mask_dataset_like(ds_xr: xr.Dataset, mask: xr.DataArray) -> xr.Dataset:
    """
    Apply where(mask) to variables that include all mask dims; leave others unchanged.
    """
    masked: Dict[str, xr.DataArray] = {}
    mask_dims = set(mask.dims)
    for name, var in ds_xr.data_vars.items():
        if mask_dims.issubset(set(var.dims)):
            masked[name] = var.where(mask)
        else:
            masked[name] = var
    return xr.Dataset(masked, coords=ds_xr.coords, attrs=ds_xr.attrs)


# ---------------------- 1) general function-based classification --------------

@register_xsnowDataset_method
def classify(
    self: xsnowDataset,
    func: Callable[[xr.Dataset] | [xr.DataArray], xr.DataArray],
    *,
    output_name: str = "class_id",
    output_kind: Literal["coord", "data"] = "coord",
    input_var: Optional[str] = None,
    attrs: Optional[dict] = None,
    **func_kwargs
) -> xsnowDataset:
    """
    General classification: run a user function to compute a classification array,
    then attach it as a coordinate or data variable.

    Parameters
    ----------
    func : callable
        A custom classification function:
        Either f(xr.Dataset) -> xr.DataArray or f(xr.DataArray) -> xr.DataArray.
        The returned DataArray must align/broadcast within the dataset.
    output_name : str, default "class_id"
        Name of the resulting coord or data var.
    output_kind : {"coord","data"}, default "coord"
        How to store the classification result.
        * `"coord"`: attach as a coordinate (good for fast selection, e.g. `ds.sel(hs_class=2)` 
                     or boolean masks).
        * `"data"`: attach as a data variable (good if you’ll plot it, compute stats on it, or 
                    don’t need to use it as a selector).
    input_var : str, optional
        If set, call func(self.data[input_var]); else func(self.data).
    attrs : dict, optional
        Metadata to attach to the resulting DataArray.
    **func_kwargs :
        Extra keyword arguments forwarded to `func`.

    Returns
    -------
    xsnowDataset
        A new dataset wrapper with the classification attached.

    See Also
    --------
    classify_citeria : A generic, boolean classification function.

    Examples
    --------
    >>> def by_hs_tertiles(hs: xr.DataArray) -> xr.DataArray:
    ...     c = xr.zeros_like(hs, dtype=np.int64)
    ...     c = c.where(hs <= 0.5, 1)     # > 0.5 --> at least 1
    ...     c = c.where(hs <= 1.5, 2)     # > 1.5 --> 2
    ...     c.attrs["mapping"] = {0: "shallow", 1: "medium", 2: "deep"}
    ...     return c
    >>> ds2 = ds.classify(by_hs, output_name="hs_class", output_kind="coord", input_var="HS")
    >>> ds2.sel(hs_class=2)

    >>> def classify_by_bins(da: xr.DataArray, *, bins: np.ndarray, right: bool = True) -> xr.DataArray:
            '''
            Bin `da` into integer classes using np.digitize.
            - `bins` are bin *edges* (length B) → classes are 0..B (B+1 classes).
            - If `right=True`, intervals are (-inf, b0], (b0, b1], ..., (b_{B-1}, +inf).
            Returns an int64 DataArray with same shape/coords as `da`.
            '''
            def _digitize(x, *, bins, right=False):
                return np.digitize(x, bins, right=right)
            classes = xr.apply_ufunc(
                _digitize,
                da,
                dask="allowed",
                kwargs={"bins": np.asarray(bins), "right": right},
                output_dtypes=[np.int64],
            )
            return classes
    >>> hs_bins = np.array([0.25, 0.50, 1.00, 1.50, 2.00])
    >>> hs_mapping = {
            0: "(-inf, 0.25]",
            1: "(0.25, 0.50]"
            2: "(0.50, 1.00]",
            3: "(1.00, 1.50]",
            4: "(1.50, 2.00]",
            5: "(2.00, +inf)"
        }
    >>> class_attrs = {
            "mapping": hs_mapping,
            "long_name": "Snow depth class (6 bins)",
            "method": "np.digitize",
            "bin_edges": hs_bins.tolist(),
            "right_closed": True,
        }
    >>> ds2 = ds.classify(
            classify_by_bins,
            output_name="HS_class",
            output_kind="coord",
            input_var="HS",          # classifier gets ds["HS"]
            attrs=class_attrs,
            bins=hs_bins,            # forwarded to classify_by_bins()
            right=True,
        )
    >>> ds_extremely_deep = ds2.sel(HS_class=5)
    """
    if self.data is None:
        raise ValueError("Cannot classify an empty dataset.")

    xds = self.data
    target = xds[input_var] if input_var else xds
    out = func(target, **func_kwargs)
    if not isinstance(out, xr.DataArray):
        raise TypeError(f"classifier must return xr.DataArray, got {type(out).__name__}")
    if out.name is None:
        out = out.rename(output_name)
    if attrs:
        out = out.copy()
        out.attrs.update(attrs)

    if output_kind == "coord":
        new_xr = xds.assign_coords({output_name: out})
    elif output_kind == "data":
        new_xr = xds.assign({output_name: out})
    else:
        raise ValueError("output_kind must be 'coord' or 'data'.")

    return self.__class__(new_xr)


# ---------------------- 2) criteria-based classification ---------------------

@register_xsnowDataset_method
def classify_criteria(
    self: xsnowDataset,
    criteria: str,
    *,
    name: str = "classification_mask",
    kind: Literal["mask", "coord", "data"] = "mask",
    attrs: Optional[dict] = None,
) -> xr.DataArray | xsnowDataset:
    """
    Criteria-based boolean classification: build a boolean mask from a string expression.
    The mask can be returned, or attached as a coordinate/data variable.

    Parameters
    ----------
    criteria : str
        Boolean conditions chained with '&' and/or '|', e.g.
        "density > 300 & grain_size < 0.5".
        (Minimal parser: left-to-right; no parentheses/precedence.)
    name : str, default "classification_mask"
        Name for the attached output.
    kind : {"mask","coord","data"}, default "mask"
        - "mask": return the mask (xr.DataArray)
        - "coord":  attach as a coordinate and return xsnowDataset
        - "data":   attach as a data var and return xsnowDataset
    attrs : dict, optional
        Attributes to add to the mask array.

    Returns
    -------
    xr.DataArray | xsnowDataset

    See Also
    --------
    classify : Generic, customizable classification into multiple classes

    Examples
    --------
    >>> mask = ds.classify_criteria("density < 200 & grain_size > 0.8", kind="mask")
    >>> ds_sel = ds.sel(layer=mask)

    >>> ds2 = ds.classify_criteria("HS > 1.5", name="deep_snow", kind="coord")
    >>> ds2.sel(deep_snow=True)
    """
    if self.data is None:
        raise ValueError("Cannot classify an empty dataset.")

    # If you support GPU, you might want: xds = self.to_cpu().data
    xds = self.data
    mask = _boolean_criteria_to_mask(xds, criteria)
    if attrs:
        mask = mask.copy()
        mask.attrs.update(attrs)

    if kind == "mask":
        return mask
    elif kind == "coord":
        new_xr = xds.assign_coords({name: mask})
        return self.__class__(new_xr)
    elif kind == "data":
        new_xr = xds.assign({name: mask})
        return self.__class__(new_xr)
    else:
        raise ValueError("kind must be one of {'mask','coord','data'}.")


# ---------------------- optional convenience: mask and keep shape ------------

@register_xsnowDataset_method
def mask_by_criteria(self: xsnowDataset, criteria: str) -> xsnowDataset:
    """
    Convenience wrapper around `classify_criteria(..., kind='mask')`:
    masks non-matching layers (i.e., NaN) while keeping shape/dimensions identical. 
    If the criteria are layer-based, scalars are untouched, and vice versa.

    Returns
    -------
    Masked xsnowDataset with identical shape/dimensions as the input, but NaNs occur where 
    the criteria are not True.
    """
    if self.data is None:
        raise ValueError("Cannot filter an empty dataset.")
    mask = self.classify_criteria(criteria, kind="mask")
    return self.__class__(_mask_dataset_like(self.data, mask))
