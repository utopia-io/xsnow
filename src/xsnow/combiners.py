from __future__ import annotations

from typing import Iterable, NoReturn, Optional, Literal, Union, Sequence, Dict, Any, Mapping
import numpy as np
import pandas as pd
import xarray as xr

from .core import xsnowDataset
from .io.base import _promote_slope_metadata_coords
from .plugin import register_xsnowDataset_method


JoinMode = Literal["left", "right", "outer", "inner", "union"]


def _normalize_join_mode(mode: JoinMode | None, default: str = "outer") -> str:
    """
    Normalize join mode; accept 'union' as alias for 'outer'.
    """
    if mode is None:
        mode = default  # type: ignore[assignment]
    if mode == "union":
        mode = "outer"
    if mode not in {"left", "right", "outer", "inner"}:
        raise ValueError(f"Invalid join mode: {mode!r}")
    return mode


def _as_xr_dataset(obj: Union["xsnowDataset", xr.Dataset]) -> xr.Dataset:
    """
    Normalize inputs: xsnowDataset -> underlying xr.Dataset; pass-through xr.Dataset.
    """
    if isinstance(obj, xr.Dataset):
        return _promote_slope_metadata_coords(obj)
    return _promote_slope_metadata_coords(obj.data)


def _ensure_profile_status(ds: xr.Dataset) -> xr.Dataset:
    """
    Ensure that `profile_status` exists on the Dataset.

    Uses xsnowDataset.update_profile_status() if missing.
    Assumes xsnowDataset(ds) is valid.
    """
    if "profile_status" in ds:
        return ds
    wrapped = xsnowDataset(ds)
    ds_with = wrapped.update_profile_status().data
    return ds_with


def _compute_target_coords_for_two(
    left: xr.Dataset,
    right: xr.Dataset,
    *,
    join: str,
    join_location: Optional[str],
    join_time: Optional[str],
    join_slope: Optional[str],
    join_realization: Optional[str],
) -> Dict[str, Any]:
    """
    Compute target coordinate values per dimension for reindexing left & right.
    join* modes are already normalized to {'left','right','outer','inner'}.
    """
    # Core dims we care about
    core_dims = ("location", "time", "slope", "realization")

    # Map per-dim override or fallback to global join
    dim_join_mode = {
        "location": _normalize_join_mode(join_location, default=join),
        "time": _normalize_join_mode(join_time, default=join),
        "slope": _normalize_join_mode(join_slope, default=join),
        "realization": _normalize_join_mode(join_realization, default=join),
    }

    target_coords: Dict[str, Any] = {}

    for dim in core_dims:
        mode = dim_join_mode[dim]

        left_has = dim in left.coords
        right_has = dim in right.coords

        if not left_has and not right_has:
            continue

        if left_has:
            left_vals = left[dim].values
        else:
            left_vals = None

        if right_has:
            right_vals = right[dim].values
        else:
            right_vals = None

        if mode == "outer":
            if left_vals is not None and right_vals is not None:
                vals = np.union1d(left_vals, right_vals)
            elif left_vals is not None:
                vals = left_vals
            else:
                vals = right_vals
        elif mode == "inner":
            if left_vals is not None and right_vals is not None:
                vals = np.intersect1d(left_vals, right_vals)
            elif left_vals is not None:
                # If dim only exists on left, treat as left (reasonable fallback)
                vals = left_vals
            else:
                vals = right_vals
        elif mode == "left":
            vals = left_vals if left_vals is not None else right_vals
        elif mode == "right":
            vals = right_vals if right_vals is not None else left_vals
        else:
            raise ValueError(f"Unexpected join mode: {mode!r}")

        target_coords[dim] = vals

    return target_coords


def _nan_like(ref: xr.DataArray) -> xr.DataArray:
    """Return an all-missing array with the same shape/coords as ref."""
    if np.issubdtype(ref.dtype, np.datetime64):
        return xr.full_like(ref, np.datetime64('NaT', 'ns'))
    dtype = ref.dtype if ref.dtype.kind == "f" else "float64"
    arr = xr.zeros_like(ref, dtype=dtype)
    return arr * np.nan


def _combine_two(
    left: xr.Dataset,
    right: xr.Dataset,
    *,
    join: str,
    join_location: Optional[str],
    join_time: Optional[str],
    join_slope: Optional[str],
    join_realization: Optional[str],
    core_dims: Sequence[str] = ("location", "time", "slope", "realization"),
    layer_dim: str = "layer",
    compat: str = "no_conflicts",
    combine_attrs: str = "override",
) -> xr.Dataset:
    """
    Combine two xarray.Datasets with profile-aware logic:

    - Reindex both to a common coordinate domain according to join / join_*.
    - Ensure profile_status exists on both; align it.
    - For variables that are truly "profile-resolved" (layer variables), 
      choose profiles based on profile_status:

         * If left has a valid profile (>=1) -> use left profile
         * Else, if right has a valid profile (>=1) -> use right profile
         * Else -> keep left (usually NaNs / error codes)

      This ensures we never mix layers of a single profile between datasets.

    - Non-layer variables follow the same profile-status selection where either
      side has a valid profile. Where neither side has one (e.g. plain forcing
      timelines with profile_status 0 everywhere), they fall back to element-wise
      prefer-left semantics via `combine_first` (left wins where non-null, right
      fills gaps).
    """
    attr_conflicts: list[str] = []
    compat_conflicts: list[str] = []

    # Stash coord attrs before any reindexing (reindex can drop attrs)
    coord_attrs_pre: dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
    for cname in set(left.coords) | set(right.coords):
        attrs_l = dict(left[cname].attrs) if cname in left.coords else {}
        attrs_r = dict(right[cname].attrs) if cname in right.coords else {}
        coord_attrs_pre[cname] = (attrs_l, attrs_r)

    # Stash data_var attrs before alignment
    var_attrs_pre: dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
    for vname in set(left.data_vars) | set(right.data_vars):
        attrs_l = dict(left[vname].attrs) if vname in left.data_vars else {}
        attrs_r = dict(right[vname].attrs) if vname in right.data_vars else {}
        var_attrs_pre[vname] = (attrs_l, attrs_r)

    # 1) Reindex to target coords
    target_coords = _compute_target_coords_for_two(
        left,
        right,
        join=join,
        join_location=join_location,
        join_time=join_time,
        join_slope=join_slope,
        join_realization=join_realization,
    )

    if target_coords:
        left = left.reindex(
            {d: v for d, v in target_coords.items() if d in left.coords},
            copy=False,
        )
        right = right.reindex(
            {d: v for d, v in target_coords.items() if d in right.coords},
            copy=False,
        )

    # 2) Ensure profile_status on both; align
    left = _ensure_profile_status(left)
    right = _ensure_profile_status(right)

    status_left, status_right = xr.align(
        left["profile_status"], right["profile_status"], join="outer"
    )
    status_left = status_left.astype("float32").fillna(0).astype("i1")
    status_right = status_right.astype("float32").fillna(0).astype("i1")
               
    # Valid profile mask on core dims (status >= 1 means valid profile)
    has_left = status_left >= 1
    has_right = status_right >= 1

    # When both have valid: prefer left
    use_left_core = has_left
    use_right_core = (~has_left) & has_right

    # 3) Prepare output as a copy of left (coords + attrs)
    out = left.copy()
    
    # Build combined profile_status explicitly
    combined_status = xr.where(
        use_left_core,
        status_left,
        xr.where(use_right_core, status_right, status_left),
    )
    out["profile_status"] = combined_status

    # We'll iterate over the union of data_vars
    all_vars = set(left.data_vars) | set(right.data_vars)

    # Align all three objects in the same coordinate/variable space when needed
    for name in all_vars:
        if name == "profile_status":
            continue  # already handled specially above

        in_left = name in left.data_vars
        in_right = name in right.data_vars

        if in_left:
            var_left = left[name]
        else:
            ref = right[name] if in_right else xr.DataArray(np.array(np.nan))
            var_left = _nan_like(ref)

        if in_right:
            var_right = right[name]
        else:
            var_right = _nan_like(var_left)

        # Align on union of coords for this variable plus the profile masks
        var_left, var_right, mask_l, mask_r = xr.align(
            var_left,
            var_right,
            use_left_core,
            use_right_core,
            join="outer",
        )

        # Compatibility checks (only if requested and both have data)
        if compat != "override":
            both_valid = var_left.notnull() & var_right.notnull()
            if both_valid.any():
                diff = (var_left != var_right).where(both_valid, False)
                if bool(diff.any()):
                    compat_conflicts.append(f"'{name}'")

        if layer_dim in var_left.dims or layer_dim in var_right.dims:
            # Layer variables: whole-profile selection by status, so the layers of
            # one profile are never mixed between datasets. Where neither side has
            # a valid profile, keep left (usually NaNs / error codes).
            merged = xr.where(mask_l, var_left, xr.where(mask_r, var_right, var_left))
        else:
            # Non-layer variables follow the profile choice where either side has
            # a valid profile. Where neither has one (e.g. merging plain forcing
            # timelines, profile_status 0 everywhere), profile selection cannot
            # decide - fall back to element-wise prefer-left so no data is lost:
            # left wins where non-null, right fills the gaps.
            fallback = var_left.combine_first(var_right)
            merged = xr.where(mask_l, var_left, xr.where(mask_r, var_right, fallback))

        out[name] = merged

    # 4) Merge coordinates from right that are missing in out
    for cname in right.coords:
        if cname in out.coords:
            continue
        out = out.assign_coords({cname: right.coords[cname]})

    # 5) Merge attrs: dataset, coords, data_vars
    conflicts: list[str] = []

    # Dataset-level attrs (including crs)
    merged_ds_attrs = _merge_attrs(
        left.attrs,
        right.attrs,
        combine_attrs,
        ensure_no_conflicts_in=("crs",),
        conflicts=attr_conflicts,
    )
    out.attrs = merged_ds_attrs

    # Coord attrs (including timezone)
    merged_coords = {}
    for cname, coord in out.coords.items():
        attrs_left, attrs_right = coord_attrs_pre.get(cname, ({}, {}))
        ensure_keys = ["timezone"] if cname == "time" else None

        merged_attrs = _merge_attrs(
            attrs_left,
            attrs_right,
            combine_attrs,
            ensure_no_conflicts_in=ensure_keys,
            conflicts=attr_conflicts,
        )
        coord_out = coord.copy()
        coord_out.attrs = merged_attrs
        merged_coords[cname] = coord_out

    if merged_coords:
        out = out.assign_coords(merged_coords)

    # Data variable attrs
    for vname in out.data_vars:
        attrs_left, attrs_right = var_attrs_pre.get(vname, ({}, {}))
        merged_attrs = _merge_attrs(
            attrs_left,
            attrs_right,
            combine_attrs,
            conflicts=attr_conflicts,
        )
        out[vname].attrs = merged_attrs

    if attr_conflicts or compat_conflicts:
        parts = []
        if compat_conflicts:
            parts.append(
                f"Incompatible values under compat='{compat}': ["
                + ", ".join(sorted(set(compat_conflicts)))
                + "]"
            )
        if attr_conflicts:
            parts.append(
                f"Attribute conflicts under combine_attrs='{combine_attrs}': {'{'}"
                + "; ".join(attr_conflicts)
                + "}"
            )
        raise ValueError(" & ".join(parts))

    return out


def _values_equal(a: Any, b: Any) -> bool:
    try:
        return bool(np.array_equal(a, b))
    except Exception:
        try:
            return bool(a == b)
        except Exception:
            return False


def _merge_attrs(
    attrs_left: Mapping[str, Any],
    attrs_right: Mapping[str, Any],
    mode: str,
    ensure_no_conflicts_in: Iterable[str] | None = None,
    conflicts: list[str] | None = None,
) -> dict[str, Any]:
    if mode not in {"drop", "no_conflicts", "drop_conflicts", "override"}:
        raise ValueError(f"Invalid combine_attrs mode: {mode!r}")

    if mode == "drop":
        return {}

    out = dict(attrs_left)
    ensure_keys = set(ensure_no_conflicts_in or [])
    conflicts_list = conflicts if conflicts is not None else []

    for key, right_val in attrs_right.items():
        if key not in out:
            out[key] = right_val
            continue

        left_val = out[key]

        # For override we skip equality checks except for ensure_keys handled below
        if mode != "override":
            equal = _values_equal(left_val, right_val)
        else:
            equal = True  # treat as equal for normal keys

        if equal:
            continue  # keep existing (left) value

        if mode == "override":
            continue  # keep left
        elif mode == "no_conflicts":
            conflicts_list.append(f"'{key}': {left_val!r} vs {right_val!r}")
        elif mode == "drop_conflicts":
            out.pop(key, None)

    # Explicitly enforce conflict checking for critical keys even under override
    for key in ensure_keys:
        if key in attrs_left and key in attrs_right:
            if not _values_equal(attrs_left[key], attrs_right[key]):
                conflicts_list.append(f"Attribute conflict for key '{key}': {attrs_left[key]!r} vs {attrs_right[key]!r}")

    return out


def combine(
    objects: Iterable[Union[xsnowDataset, xr.Dataset]],
    join: Literal["left", "right", "outer", "inner"] = "outer",
    join_location: Optional[Literal["left", "right", "outer", "inner"]] = None,
    join_time: Optional[Literal["left", "right", "outer", "inner"]] = None,
    join_slope: Optional[Literal["left", "right", "outer", "inner"]] = None,
    join_realization: Optional[Literal["left", "right", "outer", "inner"]] = None,
    compat: Literal["broadcast_equals", "no_conflicts", "override"] = "override",
    combine_attrs: Literal["drop", "no_conflicts", "drop_conflicts", "override"] = "override",
) -> xsnowDataset:
    """
    Combine multiple xsnowDataset/xarray.Dataset objects into a single dataset
    using a profile-aware, dimension-controlled merging strategy.

    This function generalizes xarray merging and concatenation to support:
    - profile-wise merging of variables (no layer mixing)
    - explicit domain selection per core dimension
    - deterministic conflict resolution (“prefer left” semantics)
    - preservation of variables from all inputs.
    - metadata handling (dataset, coords, and data_var attrs) with configurable
      conflict policies.

    Parameters
    ----------
    objects : Iterable[xsnowDataset or xr.Dataset]
        Two or more datasets to combine. Datasets are processed from
        left → right, and in conflicts the left dataset takes precedence.

        (This corresponds to the “prefer left” model; the user can reorder
        inputs to control conflict resolution.)

    join : {"left", "right", "outer", "inner"}, optional
        Default join strategy for determining the *target coordinate domain*
        along all core dimensions:

            location, time, slope, realization

        Individual core dimensions may override this via join_* arguments.

    join_location, join_time, join_slope, join_realization : optional
        Per-dimension domain selection overrides. If provided, these replace
        the global `join` setting **only for their respective dimension**.

        Examples:
        - join="outer", join_time="left"   → union for location/slope/realization, 
                                             but time domain from the left dataset
        - join="inner" and no overrides    → intersection on all core dims

    compat : {"broadcast_equals", "no_conflicts", "override"}, optional
        Conflict policy for overlapping non-null values of the same variable.
        "override" (default) keeps the left values without checking;
        "no_conflicts" and "broadcast_equals" raise if both sides have
        different non-null values.

    combine_attrs : {"drop", "no_conflicts", "drop_conflicts", "override"}, optional
        How to reconcile attrs across dataset, coords, and data_vars.
        "override" keeps left attrs (only critical keys enforced), "drop"
        removes all attrs, "no_conflicts" raises on any differing value,
        "drop_conflicts" drops keys that differ. Conflicting `time.timezone`
        and dataset-level `crs` always raise (except when combine_attrs="drop")
        to prevent silent mismatches. Conflicts are accumulated and reported
        together.

    Returns
    -------
    xsnowDataset
        A new dataset containing all variables and coordinates from the
        supplied inputs, merged according to the specified rules.

    Notes
    -----
    **1. Variables that exist only in one dataset are preserved.**
        If a variable appears only in one input dataset, it is copied directly
        to the output without modification. Only variables present in *both*
        datasets go through profile-aware conflict resolution.

    **2. Extra dimensions beyond the core dims are handled automatically.**
        Dimensions other than {'location', 'time', 'slope', 'realization'}:

        - are *not* controlled by the join/join_* settings.
        - are aligned using xarray's default `align(..., join="outer")`.
        - typically broadcast correctly if possible (e.g. ensemble members).
        - may raise xarray broadcasting/alignment errors if incompatible.

        This means additional dimensions such as:

            "run", "scenario", "alternative_model_dim", ...

        are supported naturally without special-case logic.

        Users who require explicit domain control for a new dimension may
        request that feature in a gitlab issue.

    **3. Conflict resolution prefers the leftmost dataset.**
        In overlapping coordinates:
        - Missing data (NaNs) are filled from the right.
        - Non-missing conflicting values always favor the left dataset.

        This provides deterministic behavior and matches xarray's
        `combine_first` semantics but with controlled join domains.

    **4. Layer variables are never mixed.**
        Layered variables are merged profile-by-profile, not layer-by-layer,
        avoiding physically invalid layer stacking.

    Examples
    --------
    >>> # union on all core dims, prefer left on conflicts
    >>> out = combine([nowcast, forecast])

    >>> # inner join along time, union elsewhere
    >>> out = combine([nowcast, forecast], join="outer", join_time="inner")

    >>> # left-only domain for location, right-only for time
    >>> out = combine([dsA, dsB], join_location="left", join_time="right")
    """

    iterator = iter(objects)
    try:
        first = next(iterator)
    except StopIteration:
        raise ValueError("combine() requires at least one dataset")

    # Normalize first
    base_ds = _as_xr_dataset(first)
    base_wrapper = first if isinstance(first, xsnowDataset) else xsnowDataset(base_ds)

    # Normalize join modes
    join_global = _normalize_join_mode(join)
    jl = join_location
    jt = join_time
    js = join_slope
    jr = join_realization
    if compat not in {"broadcast_equals", "no_conflicts", "override"}:
        raise ValueError(f"Invalid compat mode: {compat!r}")
    if combine_attrs not in {"drop", "no_conflicts", "drop_conflicts", "override"}:
        raise ValueError(f"Invalid combine_attrs mode: {combine_attrs!r}")

    for obj in iterator:
        ds_next = _as_xr_dataset(obj)

        base_ds = _combine_two(
            base_ds,
            ds_next,
            join=join_global,
            join_location=jl,
            join_time=jt,
            join_slope=js,
            join_realization=jr,
            compat=compat,
            combine_attrs=combine_attrs,
        )
    # Wrap result in the same xsnowDataset class as the first wrapper
    return base_wrapper.__class__(base_ds)



def concat(
    objs: Iterable[Union[xsnowDataset, xr.Dataset]],
    dim: Union[str, xr.DataArray, pd.Index],
    join: Union[JoinMode, Literal["exact"]] = "outer",
    compat: str | callable = "no_conflicts",
    combine_attrs: Union[str, callable] = "no_conflicts_rlx",
    rlx_attrs: Iterable[str] = ("history",),
    **kwargs,
):
    """
    xsnow-aware wrapper around xarray.concat.

    - Accepts an iterable of `xsnowDataset` and/or `xarray.Dataset`.
    - For join in {"outer", "exact"}:
        Behaves exactly like `xarray.concat`, except that when
        `combine_attrs="no_conflicts_rlx"` is used, selected Dataset-level
        attributes (see `rlx_attrs`) are treated specially.
    - For join in {"inner", "left", "right"}:
        Applies the requested join semantics to all coordinates **except
        `layer`**. The `layer` dimension is always promoted to the union
        across inputs to avoid accidentally dropping layers.

    Parameters
    ----------
    objs : iterable of xsnowDataset or xarray.Dataset
        Datasets to concatenate.
    dim : str or DataArray or Index
        Dimension along which to concatenate (as in `xarray.concat`).
    join : {"outer", "inner", "left", "right", "exact"}, default "outer"
        Join behavior for non-concatenation coordinates.
        Note:
        - For "outer" and "exact": no xsnow-specific behavior is applied.
        - For "inner", "left", "right": 'layer' is treated specially and
          uses the union of layer coordinates from all inputs.
    compat : {"identical", "equals", "broadcast_equals", "no_conflicts", "override"} or callable
        How to compare non-concatenated variables of the same name for potential
        conflicts. Passed through to `xarray.concat`.
    combine_attrs : {"drop", "identical", "no_conflicts", "drop_conflicts",
                     "override", "no_conflicts_rlx"} or callable
        How to combine attrs of the objects being concatenated.

        - Standard xarray values are passed through untouched.
        - The xsnow-specific value `"no_conflicts_rlx"` behaves like
          `"no_conflicts"` *except* that attributes listed in `rlx_attrs`
          are ignored for conflict checking:

              * original inputs are left unchanged,
              * internally, shallow copies with those attrs removed are passed
                to `xarray.concat(..., combine_attrs="no_conflicts")`,
              * after concat, each relaxed attribute is rebuilt on the result:

                  - for `"history"`: all non-empty histories are concatenated
                    into a single multi-line string with a small header and
                    `[input i]` tags,
                  - for any other relaxed attr: the first non-empty / non-None
                    value among the inputs is used (if any).

        This means `crs`, timezone and other critical attrs still trigger
        errors when they differ, while differing `history` (and other
        relaxed attrs) do not.
    rlx_attrs : iterable of str, default ("history",)
        Dataset-level attribute names to *relax* under
        `combine_attrs="no_conflicts_rlx"`. All other attrs are still
        subject to strict `"no_conflicts"` checking.
    **kwargs :
        Additional keyword arguments forwarded to `xarray.concat`.

    Returns
    -------
    xsnowDataset or xarray.Dataset
        Concatenated dataset. If any input was an `xsnowDataset`, the result is
        wrapped as `xsnowDataset`, otherwise a plain `xarray.Dataset` is returned.
    """
    objs = list(objs)
    if not objs:
        raise ValueError("xsnow.concat() received no datasets to concatenate.")

    # Unwrap xsnowDataset -> xarray.Dataset
    xr_objs: list[xr.Dataset] = []
    saw_xsnow = False
    for obj in objs:
        if isinstance(obj, xsnowDataset):
            xr_objs.append(_promote_slope_metadata_coords(obj.data))
            saw_xsnow = True
        elif isinstance(obj, xr.Dataset):
            xr_objs.append(_promote_slope_metadata_coords(obj))
        else:
            raise TypeError(
                f"xsnow.concat only supports xsnowDataset or xarray.Dataset, got {type(obj)}"
            )

    if len(xr_objs) == 1:
        # trivial passthrough
        out = xr_objs[0]
        return xsnowDataset(out) if saw_xsnow else out

    # ---------------------------------------------------------
    # Handle special combine_attrs="no_conflicts_rlx"
    # ---------------------------------------------------------
    rlx_attrs = tuple(rlx_attrs)
    relaxed_values: dict[str, list[Any]] | None = None
    xr_combine_attrs: Union[str, callable] = combine_attrs

    if isinstance(combine_attrs, str) and combine_attrs == "no_conflicts_rlx":
        relaxed_values = {name: [] for name in rlx_attrs}
        xr_objs_for_concat: list[xr.Dataset] = []

        for ds in xr_objs:
            ds_view = ds.copy(deep=False)
            # ensure attrs is our own dict so we can safely pop
            ds_view.attrs = dict(ds_view.attrs)

            for name in rlx_attrs:
                val = ds_view.attrs.pop(name, None)
                relaxed_values[name].append(val)

            xr_objs_for_concat.append(ds_view)

        xr_objs = xr_objs_for_concat
        xr_combine_attrs = "no_conflicts"  # strict for all non-relaxed attrs

    # ---------------------------------------------------------
    # Helper to finalize: wrap and re-attach relaxed attrs
    # ---------------------------------------------------------
    def _wrap_and_attach_relaxed(ds_out: xr.Dataset) -> Union[xsnowDataset, xr.Dataset]:
        if relaxed_values is not None:
            ds_out = ds_out.copy(deep=False)
            ds_out.attrs = dict(ds_out.attrs)

            for name, values in relaxed_values.items():
                non_empty = [
                    v
                    for v in values
                    if v is not None and (not isinstance(v, str) or v.strip())
                ]
                if not non_empty:
                    # nothing meaningful to set
                    continue

                if name == "history":
                    lines = [
                        f"xsnow.concat: combined {len(values)} input histories."
                    ]
                    # Only include the ones that actually had history text
                    idx_non_empty = [
                        (idx, v) for idx, v in enumerate(values, start=1)
                        if v is not None and (not isinstance(v, str) or v.strip())
                    ]
                    for idx, h in idx_non_empty:
                        lines.append(f"[input {idx}] {h}")
                    ds_out.attrs[name] = "\n".join(lines)
                else:
                    # For other relaxed attrs, prefer the first non-empty value.
                    ds_out.attrs[name] = non_empty[0]

        return xsnowDataset(ds_out) if saw_xsnow else ds_out

    # ---------------------------------------------------------
    # Fast path: for outer / exact, just delegate to xarray
    # ---------------------------------------------------------
    if join in ("outer", "exact"):
        out = xr.concat(
            xr_objs,
            dim=dim,
            join=join,
            compat=compat,
            combine_attrs=xr_combine_attrs,
            **kwargs,
        )
        return _wrap_and_attach_relaxed(out)

    # ---------------------------------------------------------
    # For inner / left / right:
    # - apply requested join semantics to all dims
    # - but ensure `layer` uses the union of all its coordinates
    # ---------------------------------------------------------
    layer_dim = "layer"
    has_layer = any(layer_dim in ds.dims for ds in xr_objs)
    if not has_layer:
        out = xr.concat(
            xr_objs,
            dim=dim,
            join=join,
            compat=compat,
            combine_attrs=xr_combine_attrs,
            **kwargs,
        )
        return _wrap_and_attach_relaxed(out)

    # 1. Collect union of layer coords
    all_layers = None
    for ds in xr_objs:
        if layer_dim in ds.dims:
            if layer_dim in ds.coords:
                vals = ds[layer_dim].values
            else:
                vals = np.arange(ds.sizes[layer_dim])
            all_layers = vals if all_layers is None else np.union1d(all_layers, vals)

    # Safety: shouldn't happen if has_layer is True, but guard anyway
    if all_layers is None:
        out = xr.concat(
            xr_objs,
            dim=dim,
            join=join,
            compat=compat,
            combine_attrs=xr_combine_attrs,
            **kwargs,
        )
        return _wrap_and_attach_relaxed(out)

    # 2. Reindex just the layer_dim to the union; let xarray handle other dims
    reindexed = []
    for ds in xr_objs:
        if layer_dim in ds.dims:
            reindexed.append(ds.reindex({layer_dim: all_layers}))
        else:
            reindexed.append(ds)

    # 3. Final concat (join applies only to non-concat dims)
    out = xr.concat(
        reindexed,
        dim=dim,
        join=join,
        compat=compat,
        combine_attrs=xr_combine_attrs,
        **kwargs,
    )

    return _wrap_and_attach_relaxed(out)


@register_xsnowDataset_method
def fill_gaps_with(
    self: xsnowDataset,
    other: Union[xsnowDataset, xr.Dataset],
    *,
    compat: Literal["broadcast_equals", "no_conflicts", "override"] = "override",
    combine_attrs: Literal["drop", "no_conflicts", "drop_conflicts", "override"] = "override",
) -> xsnowDataset:
    """
    Fill gaps in this dataset using another dataset, without introducing
    new coordinates (except 'layers').

    Semantics
    ---------
    - Domain: **left join** on the core dims
              (location, time, slope, realization) → i.e., only the
              coordinates present in `self` are considered.
    - Preference: **prefer self** (leftmost) whenever both datasets have
      a valid profile at a given (location, time, slope, realization).
    - Gap filling:
        * If `self` has a valid profile (profile_status >= 1) at an index,
          its profile is kept entirely (no mixing of layers).
        * If `self` has no valid profile and `other` does, the full profile
          from `other` is used there.
        * If neither has a valid profile, `self` is kept (usually NaNs).

    Extra variables & dimensions
    ----------------------------
    - Variables that exist only in `other` are *ignored* where `self` has
      no corresponding coordinates (because we restrict the domain to `self`).
    - Non-core dimensions (anything beyond location/time/slope/realization)
      follow xarray's default outer alignment for the variables that are
      actually being merged.

    Parameters
    ----------
    other : xsnowDataset or xarray.Dataset
        Dataset from which gaps in `self` will be filled.
    compat : {"broadcast_equals", "no_conflicts", "override"}, optional
        Compatibility check for overlapping non-null values. (see `xsnow.combine?`)
    combine_attrs : {"drop", "no_conflicts", "drop_conflicts", "override"}, optional
        How to reconcile attrs across inputs. (see `xsnow.combine?`)

    Returns
    -------
    xsnowDataset
        A new dataset of the same wrapper type as `self`.
    """
    combined = combine(
        objects=[self, other],
        join="left",
        compat=compat,
        combine_attrs=combine_attrs,
    )
    # Ensure we preserve the concrete xsnowDataset subclass of `self`
    return self.__class__(combined.data)


@register_xsnowDataset_method
def overwrite_with(
    self: xsnowDataset,
    other: Union[xsnowDataset, xr.Dataset],
    *,
    compat: Literal["broadcast_equals", "no_conflicts", "override"] = "override",
    combine_attrs: Literal["drop", "no_conflicts", "drop_conflicts", "override"] = "override",
) -> xsnowDataset:
    """
    Overwrite this dataset with another dataset on the union of coordinates.

    Semantics
    ---------
    - Domain: **outer join** (a.k.a union) on the core dims
              (location, time, slope, realization).
      New coordinates from `other` that don't exist in `self` are added.
    - Preference: **prefer `other`** where both datasets have valid profiles.
      This is implemented by calling `combine([other, self], join='outer')`,
      which uses the leftmost dataset (“other”) as the preferred source.
    - Profile-wise overwrite:
        * Wherever `other` has a valid profile (profile_status >= 1), its
          profile is used in full (all layers) at that index.
        * Where only `self` has a valid profile, `self` is used.
        * Where neither has a valid profile, `self` is kept.

    Extra variables & dimensions
    ----------------------------
    - Variables that exist only in `other` are added to the output dataset
      (on the union of coordinates).
    - Non-core dimensions (beyond location/time/slope/realization) are
      aligned using xarray's standard outer-alignment behavior on a
      per-variable basis.

    Parameters
    ----------
    other : xsnowDataset or xarray.Dataset
        Dataset whose information should overwrite or extend `self`.
        It is treated as “more recent / more authoritative” in overlaps.
    compat : {"broadcast_equals", "no_conflicts", "override"}, optional
        Compatibility check for overlapping non-null values. (see `xsnow.combine?`)
    combine_attrs : {"drop", "no_conflicts", "drop_conflicts", "override"}, optional
        How to reconcile attrs across inputs. (see `xsnow.combine?`)

    Returns
    -------
    xsnowDataset
        A new dataset of the same wrapper type as `self`.
    """
    combined = combine(
        objects=[other, self],  # 'other' is leftmost --> preferred in combine()
        join="outer",
        compat=compat,
        combine_attrs=combine_attrs,
    )
    # Rewrap in the same xsnowDataset subclass as self
    return self.__class__(combined.data)


@register_xsnowDataset_method
def update(
    self: xsnowDataset,
    *args: Any,
    **kwargs: Any,
) -> NoReturn:
    """
    Placeholder to prevent accidental use of xarray's Dataset.update on xsnowDataset.

    xsnowDataset provides purpose-built helpers for incremental workflows:
      - `append_latest(...)` for appending newer data or reloading from a cutoff.
      - `overwrite_with(...)` for preferring another dataset on overlaps.

    Raises:
        NotImplementedError: always; directs users to the xsnow-specific methods.
    """
    raise NotImplementedError(
        "xsnowDataset.update is not supported. Use xsnowDataset.append_latest(...) "
        "for incremental appends or xsnowDataset.overwrite_with(...) to prefer "
        "another dataset on overlaps."
    )


@register_xsnowDataset_method
def combine_first(
    self: xsnowDataset,
    *args: Any,
    **kwargs: Any,
) -> NoReturn:
    """
    Placeholder to prevent accidental use of xarray's Dataset.combine_first on xsnowDataset.

    xsnowDataset provides `fill_gaps_with(...)` for profile-aware gap filling
    without introducing new coordinates (except layers).

    Raises:
        NotImplementedError: always; directs users to xsnowDataset.fill_gaps_with().
    """
    raise NotImplementedError(
        "xsnowDataset.combine_first is not supported. Use xsnowDataset.fill_gaps_with(...) "
        "for profile-aware gap filling."
    )
