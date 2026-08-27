"""
Compute and plot hazard charts from xsnowDataset

A hazard chart places every profile at the value of its weakest layer
(y axis) and at the depth of the deepest layer in a poor-or-worse
stability class (x axis). The module holds one selection engine and
three ways to use it:

- :class:`HazardChart` (``compute_hazard_chart``) stacks one chart variant
  per stability index on a new ``hc_variant`` dimension and plots it.
- ``compute_hc_points`` returns the same selection as flat per-index
  variables, the compact form that geojson writers and zarr stores consume.
- ``compute_hc_timeseries`` pools those points per timestamp into a median,
  percentiles and the median unstable depth.

Both entry points run on a full-layer stratigraphy (depth from a shared
``depth_var``, driving layers reported by index) and on a dataset whose
layer dimension carries a string ``datetag`` coordinate, one entry per
layer group with per-index ``<idx>_depth`` (driving entries reported by
label; e.g. the simplified stratigraphy of the ``datetags`` extension).
Depths always count downwards from the snow surface, also for depth
variables stored counting up.

Stability classes are defined by the index extensions (``sk38``, ``rta``,
``critical_crack_length``) and imported here; an external index package
registers its classes with :func:`register_stability_index`, and an index
missing from the table is looked up as a package of that name. ``lwc`` is
a native SNOWPACK variable and therefore configured in this module.

The x axis is the failure depth on a non-linear scale: the anchor depths
of :data:`XAXIS_ANCHORS` are spaced evenly, so the axis resolves the
shallow depths where most profiles sit (the axis of the QMAH hazard
chart, Herla et al. 2025). The anchors are typical failure depths per
avalanche size class - the midpoints of the slab-depth ranges McClung
(2009, table 2) derives per size, or the median observed failure depths
per size of Mayer et al. (2023) - which is why the scaled depth
(``hc_xaxis``) reads as a first-order size estimate. Approximating
avalanche size by failure depth alone is far too coarse to report as a
size, so the figures label the axis with the depths, not with sizes.

This extension doubles as a **template for developing an extension** to
xsnow that adds a new dimension to the xsnowDataset class (**application
scenario 2** of the extension framework): the class subclasses the xsnow
``DatasetDecorator``, which handles the class configuration entirely,
while the module-level functions are attached with
``register_xsnowDataset_method``.

References
----------
- Herla, F., Haegeli, P., Horton, S., & Mair, P. (2025). A quantitative
  module of avalanche hazard - comparing forecaster assessments of storm
  and persistent slab avalanche problems with information derived from
  distributed snowpack simulations. Nat. Hazards Earth Syst. Sci., 25,
  625-646. https://doi.org/10.5194/nhess-25-625-2025
- McClung, D. M. (2009). Dimensions of dry snow slab avalanches from
  field measurements. J. Geophys. Res., 114, F01006.
  https://doi.org/10.1029/2007JF000941
- Mayer, S., Techel, F., Schweizer, J., & van Herwijnen, A. (2023).
  Prediction of natural dry-snow avalanche activity using physics-based
  snowpack simulations. Nat. Hazards Earth Syst. Sci., 23, 3445-3465.
  https://doi.org/10.5194/nhess-23-3445-2023
"""
# Author(s): Florian Herla, Michi Binder
# License: https://gitlab.com/avacollabra/postprocessing/xsnow/-/raw/main/LICENSE?ref_type=heads

from __future__ import annotations

import importlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import xarray as xr

from xsnow import DatasetDecorator

from ..core import xsnowDataset
from ..graincode_converter import GRAIN_COLORS, slf_graincode_to_ICSSG_primary
from ..plugin import register_xsnowDataset_method
from ..utils import latex_style
from .critical_crack_length import RC_AXLIM, RC_CLASSES, RC_WEAK
from .profile_block_computation import (
    apply_profile_block,
    float_result_dtype,
    get_backend,
)
from .rta import RTA_AXLIM, RTA_CLASSES, RTA_WEAK
from .sk38 import SK38_AXLIM, SK38_CLASSES, SK38_WEAK

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
#: The shared class labels, from unstable to stable; an index with a single
#: threshold uses only 'poor' and 'rather stable'.
CLASS_LABELS = ("very poor", "poor", "fair", "rather stable")
POOR_OR_WORSE = ("very poor", "poor")
CLASS_COLORS = {
    "very poor": "#d7191c",
    "poor": "#fdae61",
    "fair": "#ffffbf",
    "rather stable": "#a6d96a",
}
#: Liquid water content is a native SNOWPACK variable [volume fraction];
#: 3 % is the wet-snow instability threshold.
LWC_WEAK = "max"
LWC_CLASSES = [("rather stable", 0.03), ("poor", np.inf)]
LWC_AXLIM = (0.0, 0.1)

#: Per index: the weak side ('min': low values are unstable, 'max': high
#: values are), the classes as ascending ``(label, upper_bound]`` steps in
#: xsnow units, the y-axis range and the display label.
_STABILITY_INDEX_INFO: dict[str, dict[str, Any]] = {}


def register_stability_index(
    name: str,
    *,
    weak: str,
    classes: Sequence[tuple[str, float]],
    axlim: tuple[float, float],
    label: str | None = None,
) -> None:
    """
    Make a stability index known to the hazard chart.

    Built-in indices are registered on import; an external index package
    calls this when it is imported (the ``punstable`` package does), so
    the classes live next to the index they describe.

    Parameters
    ----------
    name
        Index name as used in ``indices=``, also the dataset variable.
    weak
        ``'min'`` if low values are unstable, ``'max'`` if high values are.
    classes
        Ascending ``(label, upper_bound]`` steps in xsnow units, the last
        bound being ``inf``, with labels from :data:`CLASS_LABELS`.
    axlim
        Y-axis range of the plots.
    label
        Display name, by default ``name``.
    """
    if weak not in ("min", "max"):
        raise ValueError(f"weak must be 'min' or 'max', got {weak!r}")
    unknown = [lbl for lbl, _ in classes if lbl not in CLASS_LABELS]
    if unknown:
        raise ValueError(f"Unknown class labels {unknown}; use {CLASS_LABELS}")
    _STABILITY_INDEX_INFO[name] = {
        "weak": weak, "classes": list(classes),
        "axlim": tuple(axlim), "label": label or name,
    }


for _name in ("sk38", "sk38_rta", "sk38_ccl", "sk38_ccl_rta"):
    register_stability_index(_name, weak=SK38_WEAK, classes=SK38_CLASSES,
                             axlim=SK38_AXLIM)
register_stability_index("rta", weak=RTA_WEAK, classes=RTA_CLASSES, axlim=RTA_AXLIM)
register_stability_index("rc_flat", weak=RC_WEAK, classes=RC_CLASSES, axlim=RC_AXLIM)
register_stability_index("lwc", weak=LWC_WEAK, classes=LWC_CLASSES, axlim=LWC_AXLIM)


def stability_index_info(name: str) -> dict[str, Any]:
    """
    Configuration of one stability index (see :func:`register_stability_index`).

    A name missing from the table is imported as a package of that name
    first, which lets an external index register itself (e.g.
    ``punstable``); only the absence of such a package is tolerated.

    Returns
    -------
    dict
        ``weak``, ``classes``, ``axlim`` and ``label`` of the index; KeyError
        for an unknown name.
    """
    if name not in _STABILITY_INDEX_INFO:
        try:
            importlib.import_module(name)
        except ModuleNotFoundError as exc:
            if exc.name != name:
                raise
    if name not in _STABILITY_INDEX_INFO:
        raise KeyError(
            f"Unknown stability index {name!r}. Known: {sorted(_STABILITY_INDEX_INFO)}"
        )
    return _STABILITY_INDEX_INFO[name]


def stability_indices() -> list[str]:
    """Names of the registered stability indices. An external index joins
    the list when its package is imported or once it is named explicitly
    (see :func:`stability_index_info`)."""
    return list(_STABILITY_INDEX_INFO)


def _index_config(indices: Sequence[str], classes: dict | None) -> dict[str, tuple]:
    """``{index: (weak, classes)}`` with the caller's class overrides."""
    return {
        name: (
            stability_index_info(name)["weak"],
            (classes or {}).get(name, stability_index_info(name)["classes"]),
        )
        for name in indices
    }


#: Anchor depths [m] of the x axis, spaced evenly: typical failure depths
#: of avalanche sizes 0-5 after McClung (2009, midpoints of the slab-depth
#: ranges of table 2) and Mayer et al. (2023, medians per size; the key
#: names the co-first author). Size 0 is the axis origin.
XAXIS_ANCHORS = {
    'McClung2009': (0.0, 0.18, 0.35, 0.70, 1.50, 3.00),
    'Techel2023': (0.0, 0.30, 0.40, 0.50, 0.80, 1.00),
}


# ---------------------------------------------------------------------
# Stability classes and x-axis scaling
# ---------------------------------------------------------------------
def classify_stability(values: np.ndarray, classes) -> np.ndarray:
    """Class index per value from the ascending (label, upper_bound]
    steps; NaN maps to ``len(classes)``."""
    xp = get_backend(values)
    bounds = xp.asarray([b for _, b in classes])
    return xp.searchsorted(bounds, values, side='left')


def classify_poor_or_worse(values: np.ndarray, classes) -> np.ndarray:
    """Per-entry ``POOR_OR_WORSE`` flag; independent of any grouping."""
    xp = get_backend(values)
    poor_ids = [i for i, (label, _) in enumerate(classes)
                if label in POOR_OR_WORSE]
    return xp.isin(classify_stability(values, classes), xp.asarray(poor_ids))


def scale_depth(depth, scaling: str = 'McClung2009'):
    """
    Depth [m] on the hazard-chart x axis: piecewise linear between the
    anchors of :data:`XAXIS_ANCHORS`, which sit at 0, 1, ..., 5.
    """
    if scaling not in XAXIS_ANCHORS:
        raise ValueError(f"Unknown x-axis scaling: {scaling!r}")
    xp = get_backend(depth)
    return xp.interp(depth, xp.asarray(XAXIS_ANCHORS[scaling]), xp.arange(6.0))


# ---------------------------------------------------------------------
# Hazard chart variants
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class HCVariantInfo:
    """
    Metadata of one hazard-chart variant stored in the dataset: its
    configuration and, when it was computed for selected timestamps,
    those timestamps (``None`` means every timestamp of the dataset).
    """
    idx: int
    stability_index: str
    groupby: tuple[str, ...]
    depth_var: str
    xaxis_scaling: str = "McClung2009"
    time: tuple[str, ...] | None = None


class HazardChart(DatasetDecorator):
    """
    Hazard-chart decorator that accumulates chart variants on a dataset.

    A *variant* is one configuration of the chart: a stability index, a
    grouping and a depth variable. ``compute_hazard_chart`` appends one entry
    per index along a new ``hc_variant`` dimension, so several indices - or
    the same index under different groupings - sit side by side in one
    dataset and keep their own ``hc_instability``/``hc_depth``/... slices;
    ``list_hc_variants`` reports what each entry holds. The base dataset is
    never mutated: every method returns a new wrapped instance.

    ``name`` aids introspection/registries, ``priority`` can order decorators
    in a pipeline.
    """
    name = "hazard_chart"
    priority = 20

    # ------------------------------ public API ------------------------------

    def compute_hazard_chart(
        self,
        *,
        stability_index: str | None = None,
        indices: Sequence[str] | None = None,
        groupby: str | Sequence[str] | None = None,
        layer_dim: str = "layer",
        depth_var: str = "z",
        datetags: Sequence[str] | None = None,
        classes: dict[str, Sequence] | None = None,
        xaxis_scaling: str = "McClung2009",
        at_time: Any | None = None,
    ) -> HazardChart:
        """
        Reduce every profile to one hazard-chart point per stability index
        and append the result as new entries of the ``hc_variant`` dimension.

        Per index the weakest layer sets ``hc_instability``; ``hc_depth`` is
        the depth of the deepest layer in the poor-or-worse classes, falling
        back to the weakest layer's depth. Both drivers are recorded in
        ``hc_instabilityDrivenBy`` / ``hc_depthDrivenBy``.

        Parameters
        ----------
        stability_index, indices
            One index, or several (one variant each). Names are resolved
            with :func:`stability_index_info`.
        groupby
            Coordinates or variables without the layer dimension that label
            the points, e.g. ``"region"`` or ``["region", "band"]``; their
            values are joined into one label per point and stored as integer
            ids in ``hc_group_id`` (id-to-label mapping in its ``mapping``
            attribute). Nothing is aggregated: the ids let
            ``subset_hazard_points`` and the plots pick one group.
        layer_dim
            Name of the layer dimension.
        depth_var
            Depth variable for full-layer input [m]. A variable marked
            ``positive: 'up'`` (such as ``z``) is flipped, so ``hc_depth``
            always counts downwards from the surface. Input with a string
            ``datetag`` coordinate on the layer dimension uses the per-index
            ``<idx>_depth`` instead.
        datetags
            Subset of datetag labels to consider; needs the datetag input.
        classes
            Per-index overrides of the registered stability classes
            (ascending ``(label, upper_bound]`` steps).
        xaxis_scaling
            Anchor set of :data:`XAXIS_ANCHORS` behind ``hc_xaxis``.
        at_time
            Timestamp(s) to compute for; every timestamp by default.

        Returns
        -------
        HazardChart
            The dataset with, per new variant on ``hc_variant`` and per point
            (all dimensions but the layer): ``hc_instability``, ``hc_depth``
            [m below surface], ``hc_xaxis`` (the depth on the chart's x
            axis), ``hc_instabilityDrivenBy`` / ``hc_depthDrivenBy`` (datetag
            label, ``''`` without a value or outside ``at_time``; or layer
            index, NaN without a value), ``hc_gcode`` (SLF grain code of the
            weakest layer, from
            ``<idx>_gcode_weak`` or ``grain_type``) and, with ``groupby``,
            ``hc_group_id``. The variant configuration is listed by
            ``list_hc_variants``.

        Examples
        --------
        >>> hc = HazardChart(ds).compute_hazard_chart(
        ...     indices=['sk38', 'punstable'], depth_var='z',
        ...     groupby=['region_id', 'band'])
        >>> hc.plot_hazard_chart(hc_variant=1, group='AT-07-01|alpine')
        """
        base = self.data
        if layer_dim not in base.dims:
            raise KeyError(f"Expected layer dimension '{layer_dim}' not found in dataset.")
        if indices is not None and stability_index is not None:
            raise ValueError("Pass either 'stability_index' or 'indices', not both.")
        if indices is None:
            indices = [stability_index if stability_index is not None else "sk38"]
        cfg = _index_config(indices, classes)

        # the time dimension survives a selection, so variants of different
        # timestamps concatenate; points outside the selection are NaN
        ds, times = base, None
        if at_time is not None:
            if not isinstance(at_time, slice):
                at_time = np.atleast_1d(at_time)
            ds = base.sel(time=at_time)
            times = tuple(_to_timestr(t) for t in ds["time"].values)

        # the datetag entries name the drivers; a full-layer dataset falls
        # back to layer indices
        labels = None
        if "datetag" in ds.coords and layer_dim in ds["datetag"].dims:
            labels = np.array([str(x) for x in ds["datetag"].values])
        keep = None
        if datetags is not None:
            if labels is None:
                raise ValueError("datetag filtering needs the reduced layer dimension.")
            keep = np.isin(labels, list(datetags))

        out = base.copy()
        groupby_list = _normalize_groupby(groupby)
        start_idx = int(out.sizes.get("hc_variant", 0))
        # collect the variants and concatenate once: assigning a longer
        # hc_variant into an indexed dataset would align it away
        collected: dict[str, list[xr.DataArray]] = {}
        group_maps: list[tuple[int, dict]] = []
        if "hc_variant" in out.dims:
            collected = {n: [out[n]] for n in out.data_vars if n.startswith("hc_")}
            out = out.drop_dims("hc_variant")

        for offset, index_var in enumerate(indices):
            side, cls_steps = cfg[index_var]
            if index_var not in ds:
                raise KeyError(f"Expected index variable '{index_var}' not found in dataset.")
            dvar = f"{index_var}_depth"
            if labels is None or dvar not in ds:
                dvar = depth_var
            if dvar not in ds:
                raise KeyError(f"Expected depth variable '{dvar}' not found in dataset.")

            si = ds[index_var]
            if layer_dim not in si.dims or layer_dim not in ds[dvar].dims:
                raise ValueError(
                    f"Both '{index_var}' and '{dvar}' must have layer_dim='{layer_dim}'"
                )
            gvar = f"{index_var}_gcode_weak"
            if gvar not in ds and "grain_type" in ds:
                gvar = "grain_type"
            with_gcode = gvar in ds

            inputs = [si, ds[dvar]] + ([ds[gvar]] if with_gcode else [])
            label_dtype = labels.dtype if labels is not None else float_result_dtype(si)
            dtypes = [float_result_dtype(si)] * 3 + [label_dtype] * 2
            if with_gcode:
                dtypes.append(float_result_dtype(si))
            results = apply_profile_block(
                _hazard_chart_block, *inputs,
                output_dtype=dtypes,
                layer_dim=layer_dim,
                output_core_dims=[[]] * len(dtypes),
                kwargs={
                    "side": side, "cls_steps": cls_steps, "labels": labels,
                    "keep": keep, "xaxis_scaling": xaxis_scaling,
                    "flip_depth": str(ds[dvar].attrs.get("positive", "")).lower() == "up",
                    "with_gcode": with_gcode,
                },
            )
            names = ["hc_instability", "hc_depth", "hc_xaxis",
                     "hc_instabilityDrivenBy", "hc_depthDrivenBy"]
            if with_gcode:
                names.append("hc_gcode")
            attributes = {
                "hc_instability": {
                    "description": "Weakest stability index value", "index": index_var,
                },
                "hc_depth": {
                    "description": "Depth of the deepest poor-or-worse layer",
                    "units": ds[dvar].attrs.get("units", "m"),
                    "positive": "down",
                    "depth_var": dvar,
                },
                "hc_xaxis": {
                    "units": "1",
                    "long_name": f"hc_depth on the {xaxis_scaling} x-axis scale",
                    "xaxis_scaling": xaxis_scaling,
                },
                "hc_instabilityDrivenBy": {
                    "description": "Datetag or layer index that drove hc_instability",
                },
                "hc_depthDrivenBy": {
                    "description": "Datetag or layer index that drove hc_depth",
                },
                "hc_gcode": {"units": "Swiss Code F1F2F3",
                             "description": "Grain code of the weakest layer"},
            }

            new_idx = start_idx + offset
            arrays = {
                name: result.assign_attrs(attributes[name])
                .expand_dims(hc_variant=[new_idx])
                for name, result in zip(names, results)
            }
            mapping = {}
            if groupby_list:
                proto = arrays["hc_instability"].isel(hc_variant=0)
                gid, mapping = _unique_labels_to_ids(
                    _compose_group_labels(ds, groupby_list, proto_like=proto)
                )
                gid = gid.where(proto.notnull())
                arrays["hc_group_id"] = gid.assign_attrs(
                    {"description": "Integer group id; see coordinate attrs for mapping"}
                ).expand_dims(hc_variant=[new_idx])

            for name, arr in arrays.items():
                collected.setdefault(name, []).append(arr)

            if groupby_list:
                group_maps.append((new_idx, mapping))
            _append_variant_info(out, HCVariantInfo(
                idx=new_idx,
                stability_index=index_var,
                groupby=groupby_list,
                depth_var=dvar,
                xaxis_scaling=xaxis_scaling,
                time=times,
            ))
            _append_history(out, f"hazard_chart:variant={new_idx},index={index_var}")

        for name, parts in collected.items():
            out[name] = parts[0] if len(parts) == 1 else xr.concat(parts, dim="hc_variant")
            if out[name].dtype.kind in "OU":
                # timestamps outside an at_time selection hold no value
                out[name] = out[name].fillna("")

        # only now: assigning hc_group_id above replaces the variable attrs
        for variant_idx, mapping in group_maps:
            out = _update_group_mapping_attr(
                out, variant_idx=variant_idx, keys=groupby_list, labels_map=mapping
            )

        return self._rewrap(out)

    def list_hc_variants(self) -> list[HCVariantInfo]:
        """Metadata of every stored variant, in ``hc_variant`` order."""
        return _read_variant_info(self.data)

    def subset_hazard_points(
        self,
        *,
        hc_variant: int,
        group: int | str | None = None,
        time: Any | None = None,
    ) -> xr.Dataset:
        """
        The points of one variant, optionally of one group and/or timestamp.

        A variant holds every profile of the dataset; this changes the view
        onto it without recomputing anything: zoom into one region or
        elevation band (``group``, by id or by its label from the
        ``hc_group_id`` mapping) or onto one timestamp. The plots use it to
        pick what they draw.

        Returns
        -------
        xarray.Dataset
            The ``hc_*`` variables of the variant, with the points outside
            the selection dropped.
        """
        ds = self.data
        if "hc_variant" not in ds.dims:
            raise ValueError("No hazard-chart data found (hc_variant missing).")

        sub = ds.sel(hc_variant=hc_variant)

        # optional time selection (preserve other dims)
        if time is not None and "time" in sub.dims:
            sub = sub.sel(time=time)

        if group is not None:
            if "hc_group_id" not in sub:
                raise ValueError(
                    f"Variant {hc_variant} was computed without 'groupby'; "
                    "there are no groups to select."
                )
            if isinstance(group, str):
                gid = sub["hc_group_id"]
                maps = json.loads(gid.attrs.get("mapping", "{}"))
                mapping = maps.get(str(hc_variant), {}).get("labels", {})
                rev = {v: int(k) for k, v in mapping.items()}
                if group not in rev:
                    raise KeyError(f"group label '{group}' not found for variant {hc_variant}")
                group = rev[group]
            mask = sub["hc_group_id"] == int(group)
            sub = sub.where(mask, drop=True)

        keep = ["hc_instability", "hc_depth", "hc_xaxis", "hc_gcode",
                "hc_instabilityDrivenBy", "hc_depthDrivenBy", "hc_group_id"]
        keep = [k for k in keep if k in sub]
        return sub[keep]

    def plot_hazard_chart(
        self,
        *,
        hc_variant: int,
        group: int | str | None = None,
        time: Any | None = None,
        ax=None,
        s: float = 7.0,
        alpha: float = 0.7,
        xaxis_scaling: str | None = None,
        pooled: xsnowDataset | None = None,
    ):
        """
        Scatter plot of one variant: depth of the driving layer against the
        weakest index value, one point per profile.

        Parameters
        ----------
        hc_variant, group, time
            What to draw, see ``subset_hazard_points``.
        ax
            Axes to draw into; a new figure otherwise.
        s, alpha
            Marker size and opacity.
        xaxis_scaling
            Anchor set of :data:`XAXIS_ANCHORS` spacing the depth axis;
            defaults to the variant's.
        pooled
            Precomputed ``compute_hc_timeseries`` result used for the
            marker at ``time`` (e.g. one pooled over a subset); without
            it the whole variant is pooled.

        Returns
        -------
        matplotlib.axes.Axes
            Points coloured by the grain type of the weakest layer
            (``GRAIN_COLORS['SARP']``), dashed class boundaries, and at a
            given ``time`` a green diamond at the pooled median value and
            median unstable depth of that timestamp.
        """
        import matplotlib.pyplot as plt

        meta = self.list_hc_variants()[hc_variant]
        index = meta.stability_index
        cfg = stability_index_info(index)
        scaling = xaxis_scaling or meta.xaxis_scaling
        if scaling not in XAXIS_ANCHORS:
            raise ValueError(f"Unknown x-axis scaling: {scaling!r}")

        sub = self.subset_hazard_points(hc_variant=hc_variant, group=group, time=time)
        x = _to_host(sub["hc_depth"].data).ravel()
        y = _to_host(sub["hc_instability"].data).ravel()
        colors = "black"
        if "hc_gcode" in sub:
            grains = slf_graincode_to_ICSSG_primary(
                _to_host(sub["hc_gcode"].data).ravel()
            ).values
            palette = GRAIN_COLORS["SARP"]
            colors = [palette.get(g, "black") for g in grains]

        with plt.style.context(latex_style()):
            if ax is None:
                _, ax = plt.subplots()

            ax.scatter(scale_depth(x, scaling), y, s=s, alpha=alpha, c=colors,
                       edgecolors=(0, 0, 0, 0.3), linewidths=0.3)
            _depth_axis(ax, scaling)
            ax.set_ylabel(_index_label(index))
            ax.set_ylim(*cfg["axlim"])
            ax.grid(True)
            _draw_class_lines(ax, index)

            if time is not None:
                stats = (pooled if pooled is not None
                         else _pool_variant(self.data, meta)).data
                at = stats.sel(time=time)
                mark_x = scale_depth(float(at[f"{index}_depth"]), scaling)
                ax.plot(float(mark_x), float(at[f"{index}_median"]),
                        "D", color="green", mec="black", mew=0.6,
                        ms=9, zorder=5)
        return ax

    def plot_hc_timeseries(
        self,
        *,
        hc_variant: int,
        time: Any | None = None,
        ax=None,
        timeseries: xsnowDataset | None = None,
        label_side: str = "left",
    ):
        """
        Pooled timeseries of one variant's index: median line, 25-75 %
        band (darker) and 10-90 % band (lighter) across all profiles per
        timestamp, with the class boundaries and a dashed mark at ``time``.

        Parameters
        ----------
        hc_variant
            Variant whose index is pooled.
        time
            Timestamp to mark; none by default.
        ax
            Axes to draw into; a new figure otherwise.
        timeseries
            Precomputed ``compute_hc_timeseries`` result (e.g. pooled over a
            subset); without it the whole variant is pooled.
        label_side
            ``'left'`` or ``'right'``: where the y-axis label and tick
            labels sit.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt

        meta = self.list_hc_variants()[hc_variant]
        index = meta.stability_index
        stats = (timeseries if timeseries is not None
                 else _pool_variant(self.data, meta)).data
        if label_side not in ("left", "right"):
            raise ValueError("label_side must be 'left' or 'right'")

        with plt.style.context(latex_style()):
            if ax is None:
                _, ax = plt.subplots()
            t = stats['time'].values
            ax.fill_between(t, stats[f'{index}_p10'].values,
                            stats[f'{index}_p90'].values,
                            color='#ddd', label='10-90 %')
            ax.fill_between(t, stats[f'{index}_p25'].values,
                            stats[f'{index}_p75'].values,
                            color='#bbb', label='25-75 %')
            ax.plot(t, stats[f'{index}_median'].values, color='green',
                    label='median')
            if time is not None:
                ax.axvline(self.data['time'].sel(time=time).values,
                           color='k', lw=0.8, ls='--')
            if len(t) > 1:
                ax.set_xlim(t[0], t[-1])
                locator = mdates.AutoDateLocator(maxticks=6)
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(
                    mdates.ConciseDateFormatter(locator)
                )
            right = label_side == "right"
            ax.tick_params(axis='y', which='both', left=True, right=True,
                           labelleft=not right, labelright=right)
            ax.yaxis.set_label_position(label_side)
            ax.set_ylabel(_index_label(index))
            ax.set_ylim(*stability_index_info(index)["axlim"])
            ax.grid(True)
            _draw_class_lines(ax, index)
            ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0),
                      ncols=3, frameon=False, fontsize="small",
                      columnspacing=1.2, handlelength=1.6)
        return ax

    def plot_hazard_chart_timeseries(
        self,
        *,
        hc_variant: int,
        group: int | str | None = None,
        time: Any | None = None,
        axes=None,
        s: float = 7.0,
        alpha: float = 0.7,
        xaxis_scaling: str | None = None,
        timeseries: xsnowDataset | None = None,
    ):
        """
        Two-panel figure: the hazard chart (``plot_hazard_chart``, panel a)
        next to the pooled timeseries (``plot_hc_timeseries``, panel b),
        sharing the y axis; the right panel keeps its labels on the right
        so the panels sit close together.

        ``group`` only subsets the scatter panel; ``timeseries`` feeds both
        the timeseries panel and the marker in the scatter panel. The
        remaining parameters are those of the two plotters.

        Returns
        -------
        tuple[matplotlib.axes.Axes, matplotlib.axes.Axes]
            The scatter and the timeseries axes.
        """
        import matplotlib.pyplot as plt

        meta = self.list_hc_variants()[hc_variant]
        stats = (timeseries if timeseries is not None
                 else _pool_variant(self.data, meta))

        with plt.style.context(latex_style()):
            if axes is None:
                fig, axes = plt.subplots(
                    1, 2, figsize=(7.5, 3.4), layout="constrained",
                    sharey=True,
                    gridspec_kw={"width_ratios": [1.0, 1.3]},
                )
                # constrained layout ignores gridspec wspace; tighten the
                # engine's inter-panel spacing instead
                fig.get_layout_engine().set(wspace=0.0, w_pad=0.02)
            ax_hc, ax_ts = axes
            self.plot_hazard_chart(hc_variant=hc_variant, group=group,
                                   time=time, ax=ax_hc, s=s, alpha=alpha,
                                   xaxis_scaling=xaxis_scaling,
                                   pooled=stats)
            self.plot_hc_timeseries(hc_variant=hc_variant, time=time,
                                    ax=ax_ts, timeseries=stats,
                                    label_side="right")
            for panel, label, lx in ((ax_hc, "a", 0.95), (ax_ts, "b", 0.96)):
                panel.text(lx, 0.95, label, transform=panel.transAxes,
                           ha="right", va="top", fontweight="bold",
                           bbox={"boxstyle": "circle", "facecolor": "white",
                                 "edgecolor": "black", "linewidth": 0.8})
        return ax_hc, ax_ts


# ---------------------------------------------------------------------
# Hazard-chart points and pooled timeseries
# ---------------------------------------------------------------------
@register_xsnowDataset_method
def compute_hc_points(
    ds: xsnowDataset,
    indices: Sequence[str] | None = None,
    datetags: Sequence[str] | None = None,
    xaxis_scaling: str = 'McClung2009',
    classes: dict[str, Sequence] | None = None,
    depth_var: str = 'z',
) -> xsnowDataset:
    """
    Reduce a stratigraphy to one hazard-chart point per profile and index.

    The compact form of the hazard chart: the layer dimension is dropped
    and every index gets a handful of flat variables, which is what geojson
    writers and zarr stores consume and what ``compute_hc_timeseries``
    pools. Runs on a full-layer stratigraphy and on a dataset whose layer
    dimension carries a string ``datetag`` coordinate (one entry per layer
    group with per-index ``<idx>_depth``, e.g. the simplified stratigraphy
    of the ``datetags`` extension).

    Per index this picks the weakest value across the entries and the
    depth of the deepest entry in the poor-or-worse classes (falling back
    to the weakest entry's depth), and records what drives each of the two.

    Parameters
    ----------
    indices
        Indices to summarize; defaults to the registered indices present
        (an external index such as ``punstable`` is registered once its
        package is imported, or when named here).
    datetags
        Subset of datetag labels to consider; needs the datetag input.
    xaxis_scaling
        Anchor set of :data:`XAXIS_ANCHORS` behind ``<idx>_xaxis``.
    classes
        Per-index overrides of the registered stability classes
        (ascending ``(label, upper_bound]`` steps).
    depth_var
        Depth variable for full-layer input, counted downwards from the
        surface (flipped when stored counting up); datetag input uses its
        per-index ``<idx>_depth`` instead.

    Returns
    -------
    xsnowDataset
        Layer-free dataset with, per index ``<idx>``: the weakest value
        ``<idx>``, ``<idx>_depth`` [m below the surface], ``<idx>_xaxis``
        (that depth on the chart's x axis), ``<idx>_gcode`` (SLF code of
        the weakest layer, from ``<idx>_gcode_weak`` on datetag input and
        from ``grain_type`` on full-layer input) and
        ``<idx>_instability_drivenby`` / ``<idx>_depth_drivenby``, which
        name the driving datetag on datetag input ('' where no value
        exists) and carry layer indices on full-layer input (NaN where no
        value exists). Non-layer variables are carried over.

    Notes
    -----
    Thin adapter: the selection runs in ``compute_hazard_chart``; this
    flattens its variants to the per-index variable names. A Dask-backed
    dataset stays lazy.
    """
    xs = ds if isinstance(ds, xsnowDataset) else xsnowDataset(ds)
    if 'layer' not in xs.data.dims:
        raise ValueError("compute_hc_points needs a 'layer' dimension.")
    reduced = ('datetag' in xs.data.coords
               and 'layer' in xs.data['datetag'].dims)
    if not reduced and depth_var not in xs.data:
        raise ValueError(
            f"Full-layer input needs the depth variable '{depth_var}'."
        )

    def _usable(name):
        return name in xs.data and (not reduced or f'{name}_depth' in xs.data)

    if indices is None:
        indices = [v for v in stability_indices() if _usable(v)]
    missing = [i for i in indices if not _usable(i)]
    if missing:
        raise ValueError(f"No variables for: {missing}")
    _index_config(indices, classes)      # unknown indices raise here

    # the engine appends its variants after any the input already carries
    start = int(xs.data.sizes.get('hc_variant', 0))
    hc = HazardChart(xs).compute_hazard_chart(
        indices=indices, datetags=datetags, classes=classes,
        xaxis_scaling=xaxis_scaling, depth_var=depth_var,
    ).data

    out = xs.data.drop_dims('layer')
    for offset, idx in enumerate(indices):
        var = hc.isel(hc_variant=start + offset).drop_vars('hc_variant')
        renames = {
            idx: ('hc_instability', dict(xs.data[idx].attrs)),
            f'{idx}_depth': (
                'hc_depth',
                {'units': 'm',
                 'long_name': f'{idx} driving depth below surface'},
            ),
            f'{idx}_xaxis': ('hc_xaxis', None),
            f'{idx}_gcode': ('hc_gcode', {'units': 'Swiss Code F1F2F3'}),
            f'{idx}_instability_drivenby': ('hc_instabilityDrivenBy', {}),
            f'{idx}_depth_drivenby': ('hc_depthDrivenBy', {}),
        }
        for name, (source, attrs) in renames.items():
            if source not in var:
                continue
            da = var[source]
            out[name] = da if attrs is None else da.assign_attrs(**attrs)
            if attrs is not None:
                out[name].attrs = dict(attrs)
    return xs.__class__(out)


@register_xsnowDataset_method
def compute_hc_timeseries(
    ds: xsnowDataset,
    indices: Sequence[str] | None = None,
    datetags: Sequence[str] | None = None,
    classes: dict[str, Sequence] | None = None,
    xaxis_scaling: str = 'McClung2009',
    depth_var: str = 'z',
    at_time: Any | None = None,
) -> xsnowDataset:
    """
    Pooled per-timestamp statistics of stability indices.

    Runs on a full-layer stratigraphy or a datetag dataset (both summarized
    internally via ``compute_hc_points``) or on an already summarized
    dataset. Per index and timestamp this pools every other dimension
    (locations, slopes, realizations) and derives the median, the 25/75 and
    10/90 percentiles, the median depth of the entries in the
    ``POOR_OR_WORSE`` classes, and that depth on the chart's x axis.

    Parameters
    ----------
    indices
        Indices to pool; defaults to the registered indices present (an
        external index such as ``punstable`` is registered once its package
        is imported, or when named here).
    datetags
        Subset of datetag labels to consider; needs the datetag input.
    classes
        Per-index overrides of the registered stability classes
        (ascending ``(label, upper_bound]`` steps).
    xaxis_scaling
        Anchor set of :data:`XAXIS_ANCHORS` behind ``<idx>_xaxis``.
    depth_var
        Depth variable for full-layer input, see ``compute_hc_points``.
    at_time
        Timestamp(s) to pool (a value, list or slice for ``sel``); every
        timestamp by default.

    Returns
    -------
    xsnowDataset
        Time-indexed dataset with, per index ``<idx>``:
        ``<idx>_median``, ``<idx>_p25``, ``<idx>_p75``, ``<idx>_p10``,
        ``<idx>_p90``, ``<idx>_depth`` [m] and ``<idx>_xaxis``.
    """
    xs = ds if isinstance(ds, xsnowDataset) else xsnowDataset(ds)
    if 'time' not in xs.data.dims:
        raise ValueError(
            "compute_hc_timeseries needs a 'time' dimension."
        )
    if at_time is not None:
        if not isinstance(at_time, slice):
            at_time = np.atleast_1d(at_time)
        xs = xsnowDataset(xs.data.sel(time=at_time))
    if 'layer' in xs.data.dims:
        xs = compute_hc_points(
            xs, indices=indices, datetags=datetags,
            xaxis_scaling=xaxis_scaling, classes=classes,
            depth_var=depth_var,
        )
    elif datetags is not None:
        raise ValueError(
            "datetag filtering needs the reduced layer dimension."
        )
    if indices is None:
        indices = [
            v for v in stability_indices()
            if v in xs.data and f'{v}_depth' in xs.data
        ]
    missing = [
        i for i in indices if i not in xs.data or f'{i}_depth' not in xs.data
    ]
    if missing:
        raise ValueError(f"No summarized variables for: {missing}")
    cfg = _index_config(indices, classes)

    out = xr.Dataset(coords={'time': xs.data['time']})
    n_time = xs.data.sizes['time']
    for idx in indices:
        _, cls_steps = cfg[idx]
        poor = [
            i for i, (label, _) in enumerate(cls_steps)
            if label in POOR_OR_WORSE
        ]
        # pooled percentiles run on the host: cupy has no nanpercentile
        vals = _to_host(
            xs.data[idx].transpose('time', ...).data
        ).reshape(n_time, -1)
        deps = _to_host(
            xs.data[f'{idx}_depth'].transpose('time', ...).data
        ).reshape(n_time, -1)
        unstable = np.isin(classify_stability(vals, cls_steps), poor)
        stats = np.full((5, n_time), np.nan)
        depth = np.full(n_time, np.nan)
        with np.errstate(all='ignore'):
            for t in range(n_time):
                if np.isfinite(vals[t]).any():
                    stats[:, t] = np.nanpercentile(
                        vals[t], (50, 25, 75, 10, 90)
                    )
                if unstable[t].any():
                    depth[t] = np.nanmedian(deps[t][unstable[t]])
        attrs = dict(xs.data[idx].attrs)
        for name, row in zip(('median', 'p25', 'p75', 'p10', 'p90'), stats):
            out[f'{idx}_{name}'] = (('time',), row, dict(attrs))
        out[f'{idx}_depth'] = (
            ('time',), depth,
            {'units': 'm', 'long_name': f'median unstable {idx} depth'},
        )
        out[f'{idx}_xaxis'] = (
            ('time',), scale_depth(depth, xaxis_scaling),
            {'units': '1',
             'long_name': f'{idx}_depth on the {xaxis_scaling} x-axis scale',
             'xaxis_scaling': xaxis_scaling},
        )
    return xs.__class__(out)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _normalize_groupby(groupby: str | Sequence[str] | None) -> tuple[str, ...]:
    if groupby is None:
        return ()
    if isinstance(groupby, str):
        return (groupby,)
    return tuple(groupby)


def _compose_group_labels(
    ds: xr.Dataset, keys: Sequence[str], *, proto_like: xr.DataArray
) -> xr.DataArray:
    """One string label per point of ``proto_like`` (no layer; time is
    kept if present, never injected)."""
    if not keys:
        return xr.full_like(proto_like, "all", dtype=object).rename("hc_group")

    proto = xr.full_like(proto_like, "", dtype=object)

    # the keys are point-level coordinates, small enough to compose on the host
    pieces: list[np.ndarray] = []
    for key in keys:
        if key in ds.coords:
            val = ds.coords[key]
        elif key in ds:
            val = ds[key]
        else:
            raise KeyError(f"groupby key '{key}' not found as coord or data_var.")
        if "layer" in getattr(val, "dims", ()):
            raise ValueError(f"groupby key '{key}' must not depend on 'layer'.")
        pieces.append(
            np.asarray(val.astype(str).broadcast_like(proto).values, dtype=object)
        )

    label = pieces[0]
    for piece in pieces[1:]:
        label = label + "|" + piece
    return xr.DataArray(label, dims=proto.dims, coords=proto.coords, name="hc_group")


def _append_history(ds: xr.Dataset, line: str) -> None:
    hist = ds.attrs.get("history", "")
    if line not in hist.splitlines():
        ds.attrs["history"] = (hist + "\n" if hist else "") + line


def _append_variant_info(ds: xr.Dataset, info: HCVariantInfo) -> None:
    key = "hc_variants"
    try:
        lst = json.loads(ds.attrs.get(key, "[]"))
    except Exception:
        lst = []
    lst = [x for x in lst if int(x.get("idx", -1)) != info.idx]
    lst.append(asdict(info))
    ds.attrs[key] = json.dumps(lst, default=str)


def _read_variant_info(ds: xr.Dataset) -> list[HCVariantInfo]:
    key = "hc_variants"
    try:
        raw = json.loads(ds.attrs.get(key, "[]"))
    except Exception:
        return []
    out: list[HCVariantInfo] = []
    for r in raw:
        time = r.get("time")
        out.append(
            HCVariantInfo(
                idx=int(r["idx"]),
                stability_index=str(r["stability_index"]),
                groupby=tuple(r["groupby"]),
                depth_var=str(r["depth_var"]),
                xaxis_scaling=str(r.get("xaxis_scaling", "McClung2009")),
                time=None if time is None else tuple(str(t) for t in time),
            )
        )
    return out


def _unique_labels_to_ids(labels_da: xr.DataArray) -> tuple[xr.DataArray, dict[int, str]]:
    """Integer ids of an object/string label array, same shape, plus the
    id-to-label mapping."""
    flat = labels_da.values.reshape(-1).astype(object)
    uniq, inv = np.unique(flat, return_inverse=True)
    ids = inv.reshape(labels_da.shape).astype(np.int64)
    mapping = {int(i): str(uniq[i]) for i in range(len(uniq))}
    ids_da = xr.DataArray(ids, dims=labels_da.dims,
                          coords={d: labels_da.coords[d] for d in labels_da.dims})
    return ids_da, mapping


def _update_group_mapping_attr(
    ds: xr.Dataset, *, variant_idx: int, keys: Sequence[str], labels_map: dict[int, str]
) -> xr.Dataset:
    """Per-variant mapping in the ``hc_group_id`` attrs:
    ``{"<variant_idx>": {"keys": [...], "labels": {"0": "P1|E", ...}}}``."""
    var = ds["hc_group_id"]
    attrs = dict(var.attrs)
    all_maps = {}
    if "mapping" in attrs:
        try:
            all_maps = json.loads(attrs["mapping"])
        except Exception:
            all_maps = {}
    all_maps[str(variant_idx)] = {
        "keys": list(keys), "labels": {str(k): v for k, v in labels_map.items()},
    }
    attrs["mapping"] = json.dumps(all_maps)
    ds["hc_group_id"].attrs = attrs
    return ds


def _to_timestr(value: Any) -> str:
    """ISO string of one timestamp, at second resolution."""
    return str(np.datetime_as_string(np.asarray(value).astype("datetime64[s]")))


def _to_host(values) -> np.ndarray:
    """Copy back to the CPU for string, pandas or numpy-only work."""
    return values.get() if hasattr(values, 'get') else np.asarray(values)


def _depth_axis(ax, scaling: str) -> None:
    """Depth ticks at the evenly spaced anchors of ``scaling``."""
    anchors = XAXIS_ANCHORS[scaling]
    ax.set_xticks(range(len(anchors)), labels=[f"{d:g}" for d in anchors])
    ax.set_xlim(0, len(anchors) - 1)
    ax.set_xlabel("Depth / m")


def _pool_variant(data: xr.Dataset, meta: HCVariantInfo) -> xsnowDataset:
    """The pooled timeseries of a variant's index, with its depth variable."""
    return compute_hc_timeseries(
        xsnowDataset(data), indices=[meta.stability_index],
        depth_var=meta.depth_var, xaxis_scaling=meta.xaxis_scaling,
    )


def _draw_class_lines(ax, index: str) -> None:
    """Dashed lines on the class boundaries, coloured like the more
    unstable of the two classes each boundary separates."""
    cfg = stability_index_info(index)
    classes = cfg["classes"]
    for i, (label, bound) in enumerate(classes[:-1]):
        worse = classes[i + 1][0] if cfg["weak"] == "max" else label
        ax.axhline(bound, ls="--", lw=1.2, zorder=1.5,
                   color=CLASS_COLORS.get(worse, "grey"))


def _index_label(index: str) -> str:
    """Axis label of an index: its display name over the '-' unit."""
    return f"{stability_index_info(index)['label']} / -"


def _argreduce_layer(values: np.ndarray, side: str):
    """NaN-safe argmax/argmin over the last axis with a validity flag."""
    xp = get_backend(values)
    fill = -np.inf if side == 'max' else np.inf
    filled = xp.where(xp.isnan(values), fill, values)
    idx = filled.argmax(-1) if side == 'max' else filled.argmin(-1)
    return idx, xp.isfinite(filled).any(-1)


def take_layer(values: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """The entry ``idx`` of every profile along the last axis."""
    xp = get_backend(values)
    return xp.take_along_axis(values, idx[..., None], -1)[..., 0]


def _hazard_chart_block(values, depths, gcodes=None, *, side, cls_steps,
                        labels=None, keep=None, xaxis_scaling='McClung2009',
                        flip_depth=False, with_gcode=False):
    """Reduce one block of complete profiles to the hazard-chart summary.

    The layer axis is last; every output loses it.
    """
    xp = get_backend(values)
    if keep is not None:
        values = xp.where(xp.asarray(keep), values, np.nan)
    if flip_depth:
        depths = -depths          # e.g. 'z' counts up from the surface
    weak_idx, valid, deep_idx = select_weak_and_deep(values, side, cls_steps)
    depth = xp.where(valid, take_layer(depths, deep_idx), np.nan)
    outputs = [
        xp.where(valid, take_layer(values, weak_idx), np.nan),
        depth,
        xp.where(valid, scale_depth(depth, xaxis_scaling), np.nan),
    ]
    if labels is not None:
        host = (_to_host(valid), _to_host(weak_idx), _to_host(deep_idx))
        outputs += [np.where(host[0], labels[host[1]], ""),
                    np.where(host[0], labels[host[2]], "")]
    else:
        outputs += [xp.where(valid, weak_idx, np.nan),
                    xp.where(valid, deep_idx, np.nan)]
    if with_gcode:
        outputs.append(xp.where(valid, take_layer(gcodes, weak_idx), np.nan))
    return tuple(outputs)


def select_weak_and_deep(values: np.ndarray, side: str, cls_steps,
                         poor: np.ndarray | None = None):
    """
    Weakest entry and depth-driving entry along the last axis.

    Entry 0 is the deepest, so the first poor-or-worse entry is the deepest
    one; without any such entry the weakest entry drives the depth as well.
    ``poor`` skips the classification when the caller already has it (see
    ``classify_poor_or_worse``). Returns ``(weak_idx, valid, depth_idx)``.
    """
    xp = get_backend(values)
    weak_idx, valid = _argreduce_layer(values, side)
    if poor is None:
        poor = classify_poor_or_worse(values, cls_steps)
    depth_idx = xp.where(poor.any(-1), poor.argmax(-1), weak_idx)
    return weak_idx, valid, depth_idx
