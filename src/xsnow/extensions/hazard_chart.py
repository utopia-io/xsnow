from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import xarray as xr

from xsnow import DatasetDecorator

"""
TODO:
 * fix z minus issues
 * create axes limits per stability index
 * add timeseries plot
"""


# ---------------------------------------------------------------------
# Stability index configuration
# ---------------------------------------------------------------------
_STABILITY_INDEX_INFO: Dict[str, Dict[str, Any]] = {
    "sk38": dict(
        var="sk38",  # variable name holding the index (per-layer)
        weak="min",  # weakest = min value
        thresholds=[  # (label, upper_bound]
            ("very poor", 0.5),
            ("poor", 0.95),
            ("fair", 1.05),
            ("rather stable", np.inf),
        ],
        poor_or_worse=("very poor", "poor"),
    ),
    # add more indices here
}


@dataclass(frozen=True)
class HCVariantInfo:
    """
    Metadata describing one hazard-chart variant stored in the dataset.
    Note: time is *not* part of the variant. Variants represent configuration
    (stability index, grouping keys, chosen depth var).
    """
    idx: int
    stability_index: str
    groupby: Tuple[str, ...]   # grouping variables actually used (no 'time' added)
    depth_var: str             # depth variable used


class HazardChart(DatasetDecorator):
    """
    Hazard chart decorator that can accumulate multiple computed variants
    along a new `hc_variant` dimension. It never mutates the base dataset;
    instead, each method returns a *new* wrapped instance via `_rewrap()`.

    name/priority:
      - name helps introspection/registries.
      - priority can be used by your pipeline to order decorators if needed.
    """
    name = "hazard_chart"
    priority = 20

    # ------------------------------ public API ------------------------------

    def compute_hazard_chart_data(
        self,
        *,
        stability_index: str = "sk38",
        groupby: Optional[Union[str, Sequence[str]]] = None,
        layer_dim: str = "layer",
        depth_var: str = "z",
        at_time: Optional[Any] = None,
    ) -> "HazardChart":
        """
        Compute hazard-chart summary variables across time (or at a specific time
        if `at_time` is provided) and append them as a new entry on the `hc_variant`
        dimension.

        Notes
        -----
        Depth selection is dynamic per point:
          - depth @ weakest layer (min/max of stability index), and
          - deepest depth with qualitative class ∈ poor-or-worse
        are both computed; the *deeper* is chosen. Provenance for the chosen
        depth's driver layer is stored in `hc_depthDrivenBy`.

        The output variables have dims (hc_variant, <points dims incl. time>).
        """
        base = self.data

        # --- resolve index config ---
        if stability_index not in _STABILITY_INDEX_INFO:
            raise KeyError(f"Unknown stability_index '{stability_index}'. "
                           f"Known: {list(_STABILITY_INDEX_INFO)}")
        cfg = _STABILITY_INDEX_INFO[stability_index]
        index_var = cfg["var"]
        weak_is_min = cfg["weak"] == "min"
        thresholds: List[Tuple[str, float]] = cfg["thresholds"]
        poor_or_worse: Tuple[str, ...] = cfg.get("poor_or_worse", ("very poor", "poor"))

        if index_var not in base:
            raise KeyError(f"Expected index variable '{index_var}' not found in dataset.")
        if depth_var not in base:
            raise KeyError(f"Expected depth variable '{depth_var}' not found in dataset.")
        if layer_dim not in base.dims:
            raise KeyError(f"Expected layer dimension '{layer_dim}' not found in dataset.")

        # --- restrict to time if requested (but keep 'time' as a dim) ---
        ds = base if at_time is None else base.sel(time=at_time)

        # --- inputs (per-layer) ---
        si = ds[index_var]           # dims: (..., layer)
        depth_da = ds[depth_var]     # dims include layer
        if layer_dim not in si.dims or layer_dim not in depth_da.dims:
            raise ValueError(f"Both '{index_var}' and '{depth_var}' must have layer_dim='{layer_dim}'")

        # we operate on points grid = all dims except 'layer'
        points_dims = tuple(d for d in si.dims if d != layer_dim)

        # at least one finite layer per point?
        valid_point = si.count(dim=layer_dim) > 0

        # --- weakest stability & driver (guarded) ---
        if weak_is_min:
            hc_instability = si.min(dim=layer_dim, skipna=True)
            hc_instabilityDrivenBy = _safe_argreduce(si, dim=layer_dim, func="min")
        else:
            hc_instability = si.max(dim=layer_dim, skipna=True)
            hc_instabilityDrivenBy = _safe_argreduce(si, dim=layer_dim, func="max")

        # depth at weakest layer (NaN-safe)
        weakest_depth = _isel_with_nan_index(depth_da, layer_dim, hc_instabilityDrivenBy)

        # --- deepest depth among "poor or worse" layers ---
        class_bins = _classify_layers(si, thresholds)           # per-layer string labels
        poor_mask = _is_any_of(class_bins, poor_or_worse)       # bool like si
        depth_masked = xr.where(poor_mask, depth_da, -np.inf)   # invalid -> -inf
        deepest_poor = depth_masked.max(dim=layer_dim, skipna=True)
        arg_poor = depth_masked.argmax(dim=layer_dim, skipna=True)
        valid_poor = np.isfinite(deepest_poor)
        arg_poor = arg_poor.where(valid_poor)

        # compare depths using a metric that increases with depth (sign-agnostic)
        z_metric = _depth_metric(depth_da, layer_dim)
        weakest_depth_metric = _isel_with_nan_index(z_metric, layer_dim, hc_instabilityDrivenBy)
        deepest_poor_metric = deepest_poor.where(valid_poor, -np.inf)

        use_poor = valid_poor & (deepest_poor_metric > weakest_depth_metric)
        hc_depth = xr.where(use_poor, deepest_poor, weakest_depth)
        hc_depthDrivenBy = xr.where(use_poor, arg_poor, hc_instabilityDrivenBy)

        # mask outputs where profile is entirely invalid
        hc_instability = hc_instability.where(valid_point)
        hc_instabilityDrivenBy = hc_instabilityDrivenBy.where(valid_point)
        hc_depth = hc_depth.where(valid_point)
        hc_depthDrivenBy = hc_depthDrivenBy.where(valid_point)

        # --- grouping (NO time in the label; time remains a dim on outputs) ---
        groupby_list = _normalize_groupby(groupby)  # may be empty/None
        # Build labels against the proto grid of hc_instability (no layer, includes time if present)
        hc_group_labels = _compose_group_labels(ds, groupby_list, proto_like=hc_instability)
        hc_group_id, mapping = _unique_labels_to_ids(hc_group_labels)
        hc_group_id = hc_group_id.where(valid_point)  # NaN out invalid points

        # --- append as a new variant over existing dataset (keep time dim) ---
        out = self.data
        existing_variants = int(out.sizes.get("hc_variant", 0))
        new_idx = existing_variants  # next integer

        def _with_variant(arr: xr.DataArray) -> xr.DataArray:
            return arr if "hc_variant" in arr.dims else arr.expand_dims(hc_variant=[new_idx])

        new_vars = {
            "hc_instability": _with_variant(hc_instability).assign_attrs(
                dict(description="Weakest stability index value", index=stability_index)
            ),
            "hc_depth": _with_variant(hc_depth).assign_attrs(
                dict(description="Relevant depth (deeper of weakest vs deepest-poor)", units="m")
            ),
            "hc_instabilityDrivenBy": _with_variant(hc_instabilityDrivenBy).assign_attrs(
                dict(description="Layer index that drove hc_instability", layer_dim=layer_dim)
            ),
            "hc_depthDrivenBy": _with_variant(hc_depthDrivenBy).assign_attrs(
                dict(description="Layer index that drove hc_depth", layer_dim=layer_dim)
            ),
            "hc_group_id": _with_variant(hc_group_id).assign_attrs(
                dict(description="Integer group id; see coordinate attrs for mapping")
            ),
        }

        for name, arr in new_vars.items():
            if name in out:
                out[name] = xr.concat([out[name], arr], dim="hc_variant")
            else:
                out[name] = arr

        if "hc_variant" not in out.coords:
            out = out.assign_coords(hc_variant=np.arange(out.sizes["hc_variant"], dtype=int))

        # --- store/update mapping in the hc_group_id variable attrs (per-variant) ---
        out = _update_group_mapping_attr(out, variant_idx=new_idx, keys=groupby_list, labels_map=mapping)

        # --- store/append variant metadata ---
        info = HCVariantInfo(
            idx=new_idx,
            stability_index=stability_index,
            groupby=groupby_list,
            depth_var=depth_var,
        )
        _append_variant_info(out, info)

        # provenance
        _append_history(out, f"hazard_chart:variant={new_idx},index={stability_index}")

        return self._rewrap(out)

    def list_hc_variants(self) -> List[HCVariantInfo]:
        """Return metadata for all stored hazard-chart variants."""
        return _read_variant_info(self.data)

    def subset_hazard_points(
        self,
        *,
        hc_variant: int,
        group: Optional[Union[int, str]] = None,
        time: Optional[Any] = None,
    ) -> xr.Dataset:
        """
        Return per-point hazard variables for a single variant, optionally restricted
        to one group (by id or label) and/or time (scalar value).
        """
        ds = self.data
        if "hc_variant" not in ds.dims:
            raise ValueError("No hazard-chart data found (hc_variant missing).")

        sub = ds.sel(hc_variant=hc_variant)

        # optional time selection (preserve other dims)
        if time is not None and "time" in sub.dims:
            sub = sub.sel(time=time)

        if group is not None:
            if isinstance(group, str):
                gid = sub["hc_group_id"]
                import json
                maps = json.loads(gid.attrs.get("mapping", "{}"))
                mapping = maps.get(str(hc_variant), {}).get("labels", {})
                rev = {v: int(k) for k, v in mapping.items()}
                if group not in rev:
                    raise KeyError(f"group label '{group}' not found for variant {hc_variant}")
                group = rev[group]
            mask = sub["hc_group_id"] == int(group)
            sub = sub.where(mask, drop=True)

        keep = ["hc_instability", "hc_depth", "hc_instabilityDrivenBy", "hc_depthDrivenBy", "hc_group_id"]
        keep = [k for k in keep if k in sub]
        return sub[keep]

    def plot_hazard_chart(
        self,
        *,
        hc_variant: int,
        group: Optional[Union[int, str]] = None,
        time: Optional[Any] = None,
        ax=None,
        s: float = 18.0,
        alpha: float = 0.7,
    ):
        """
        Scatter-plot for one variant, optionally for a single group and/or time.
        X = hc_depth, Y = hc_instability.
        """
        import matplotlib.pyplot as plt

        sub = self.subset_hazard_points(hc_variant=hc_variant, group=group, time=time)
        x = sub["hc_depth"].values.ravel()
        y = sub["hc_instability"].values.ravel()

        if ax is None:
            _, ax = plt.subplots()

        ax.scatter(x, y, s=s, alpha=alpha)
        ax.set_xlabel("Depth")
        ax.set_ylabel("Instability index (weakest)")
        # Decide whether to invert axis depending on your depth convention
        # ax.invert_xaxis()

        # Title with basic info
        infos = self.list_hc_variants()
        title = f"Hazard chart (variant {hc_variant})"
        if hc_variant < len(infos):
            meta = infos[hc_variant]
            title += f" — {meta.stability_index}"
            if meta.groupby:
                title += f" grouped by {','.join(meta.groupby)}"
            if time is not None:
                title += f" @ time={_to_timestr(time)}"
        ax.set_title(title)
        return ax


# ------------------------------ helpers ------------------------------

def _normalize_groupby(groupby: Optional[Union[str, Sequence[str]]]) -> Tuple[str, ...]:
    if groupby is None:
        return tuple()
    if isinstance(groupby, str):
        return (groupby,)
    return tuple(groupby)


def _compose_group_labels(ds: xr.Dataset, keys: Sequence[str], *, proto_like: xr.DataArray) -> xr.DataArray:
    """
    Create a string label per point for the dims of `proto_like`
    (no 'layer', may include 'time' if present; we *don't* inject 'time').
    """
    if not keys:
        return xr.full_like(proto_like, "all", dtype=object).rename("hc_group")

    proto = xr.full_like(proto_like, "", dtype=object)

    pieces: List[xr.DataArray] = []
    for key in keys:
        if key in ds.coords:
            val = ds.coords[key]
        elif key in ds:
            val = ds[key]
        else:
            raise KeyError(f"groupby key '{key}' not found as coord or data_var.")
        if "layer" in getattr(val, "dims", ()):
            raise ValueError(f"groupby key '{key}' must not depend on 'layer'.")
        pieces.append(val.astype(str).broadcast_like(proto))

    label = pieces[0]
    for p in pieces[1:]:
        label = (label.astype(object) + "|" + p.astype(object))
    return label.rename("hc_group")


def _classify_layers(si: xr.DataArray, thresholds: Sequence[Tuple[str, float]]) -> xr.DataArray:
    """Map numeric stability index to qualitative class labels per layer."""
    out = xr.full_like(si, "", dtype=object)
    prev = -np.inf
    for label, upper in thresholds:
        mask = (si > prev) & (si <= upper)
        out = xr.where(mask, label, out)
        prev = upper
    return out


def _is_any_of(arr: xr.DataArray, labels: Sequence[str]) -> xr.DataArray:
    mask = False
    for lab in labels:
        mask = mask | (arr == lab)
    return mask


def _append_history(ds: xr.Dataset, line: str) -> None:
    hist = ds.attrs.get("history", "")
    if line not in hist.splitlines():
        ds.attrs["history"] = (hist + "\n" if hist else "") + line


def _append_variant_info(ds: xr.Dataset, info: HCVariantInfo) -> None:
    key = "hc_variants"
    import json

    try:
        lst = json.loads(ds.attrs.get(key, "[]"))
    except Exception:
        lst = []
    lst = [x for x in lst if int(x.get("idx", -1)) != info.idx]
    lst.append(asdict(info))
    ds.attrs[key] = json.dumps(lst, default=str)


def _read_variant_info(ds: xr.Dataset) -> List[HCVariantInfo]:
    import json

    key = "hc_variants"
    try:
        raw = json.loads(ds.attrs.get(key, "[]"))
    except Exception:
        return []
    out: List[HCVariantInfo] = []
    for r in raw:
        out.append(
            HCVariantInfo(
                idx=int(r["idx"]),
                stability_index=str(r["stability_index"]),
                groupby=tuple(r["groupby"]),
                depth_var=str(r["depth_var"]),
            )
        )
    return out


def _safe_argreduce(arr: xr.DataArray, dim: str, func: str = "min") -> xr.DataArray:
    """
    Like argmin/argmax, but returns NaN where all-NaN slices would fail.
    Keeps original dims except `dim`.
    """
    f = np.nanargmin if func == "min" else np.nanargmax
    axis = arr.get_axis_num(dim)

    data = arr.values
    nan_mask = np.isnan(data)

    # Move target axis to end for consistent indexing
    data_swapped = np.moveaxis(data, axis, -1)
    nan_swapped = np.moveaxis(nan_mask, axis, -1)

    flat = data_swapped.reshape(-1, data_swapped.shape[-1])
    flat_nan = nan_swapped.reshape(-1, nan_swapped.shape[-1])

    idx = np.full(flat.shape[0], np.nan, dtype=float)
    for i in range(flat.shape[0]):
        if np.all(flat_nan[i]):
            continue
        idx[i] = f(flat[i])

    idx = idx.reshape(data_swapped.shape[:-1])

    # Wrap back into DataArray, dropping the reduced dim
    out_dims = [d for d in arr.dims if d != dim]
    coords = {d: arr.coords[d] for d in out_dims if d in arr.coords}
    return xr.DataArray(idx, dims=out_dims, coords=coords)


def _ensure_int_indexer(indexer: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Make an integer indexer for isel, returning (idx_int, valid_mask)."""
    valid = indexer.notnull()
    idx_int = indexer.fillna(0).astype(int)
    return idx_int, valid


def _isel_with_nan_index(arr: xr.DataArray, dim: str, indexer: xr.DataArray) -> xr.DataArray:
    """
    isel with a possibly-NaN vectorized indexer: fill invalid with 0 for indexing,
    then mask invalid positions back to NaN afterwards.
    """
    idx_int, valid = _ensure_int_indexer(indexer)
    out = arr.isel({dim: idx_int})
    return out.where(valid)


def _depth_metric(z_da: xr.DataArray, layer_dim: str) -> xr.DataArray:
    """
    Return a DataArray with same values as z but transformed so that
    'larger' means 'deeper', regardless of sign convention.
    """
    positive_down = (z_da.attrs.get("positive", "").lower() == "down")
    med = z_da.median(dim=layer_dim, skipna=True)
    if positive_down or (np.nanmedian(med.values) > 0):
        return z_da  # larger is deeper
    else:
        return -z_da  # more negative is deeper → flip sign for comparisons


def _unique_labels_to_ids(labels_da: xr.DataArray) -> tuple[xr.DataArray, dict[int, str]]:
    """
    Map an object/string labels DataArray to integer ids with the same shape.
    Returns (ids_da, mapping).
    """
    flat = labels_da.values.reshape(-1).astype(object)
    uniq, inv = np.unique(flat, return_inverse=True)
    ids = inv.reshape(labels_da.shape).astype(np.int64)
    mapping = {int(i): str(uniq[i]) for i in range(len(uniq))}
    ids_da = xr.DataArray(ids, dims=labels_da.dims, coords={d: labels_da.coords[d] for d in labels_da.dims})
    return ids_da, mapping


def _update_group_mapping_attr(
    ds: xr.Dataset, *, variant_idx: int, keys: Sequence[str], labels_map: Dict[int, str]
) -> xr.Dataset:
    """
    Store/update a per-variant mapping in hc_group_id variable attrs:
    {
      "<variant_idx>": { "keys": [...], "labels": { "0": "P1|E", "1": "P2|W", ... } }
    }
    """
    import json
    var = ds["hc_group_id"]
    attrs = dict(var.attrs)
    all_maps = {}
    if "mapping" in attrs:
        try:
            all_maps = json.loads(attrs["mapping"])
        except Exception:
            all_maps = {}
    all_maps[str(variant_idx)] = {"keys": list(keys), "labels": {str(k): v for k, v in labels_map.items()}}
    attrs["mapping"] = json.dumps(all_maps)
    ds["hc_group_id"].attrs = attrs
    return ds


def _to_timestr(time_values: Any) -> str:
    try:
        val = np.array(time_values).item()
        if isinstance(val, (np.datetime64,)):
            return np.datetime_as_string(val, unit="s")
        return str(val)
    except Exception:
        return str(time_values)
