"""
Compute and plot hazard charts from xsnowDataset

The module holds one selection engine and three ways to use it. Per stability
index of ``_STABILITY_INDEX_INFO`` it picks the weakest layer of a profile and
the depth that drives the instability (the deepest layer of a poor-or-worse
stability class), and derives an avalanche size estimate from that depth:

- :class:`HazardChart` (``compute_hazard_chart``) stacks one chart variant per
  index on a new ``hc_variant`` dimension, and plots it.
- ``compute_hc_points`` returns the same selection as flat per-index variables,
  which is the form geojson writers and zarr stores consume.
- ``compute_hc_timeseries`` pools those summaries per timestamp into a median,
  percentiles, an unstable depth and a size estimate.

``compute_hazard_chart`` and ``compute_hc_points`` run on a full-layer
stratigraphy as well as on the reduced dataset of
``compute_simplified_stratigraphy``; ``compute_hc_timeseries`` takes the
reduced or an already summarized one. Reduced input uses the per-index
``<idx>_depth`` and names the driving datetag, full-layer input reads a shared
depth variable (``depth_var``, default ``'z'``) and reports driving layer
indices. Depths are always returned counting downwards from the snow surface,
including for depth variables that are stored counting up.

This extension doubles as a **template for developing an extension** to xsnow
that adds a new dimension to the xsnowDataset class (**application scenario 2**
of the extension framework): the class subclasses the xsnow
``DatasetDecorator``, which handles the class configuration entirely, while the
module-level functions are attached with ``register_xsnowDataset_method``.
"""
# Author(s): Florian Herla
# License: https://gitlab.com/avacollabra/postprocessing/xsnow/-/raw/main/LICENSE?ref_type=heads

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import xarray as xr

from xsnow import DatasetDecorator

from ..core import xsnowDataset
from ..graincode_converter import GRAIN_COLORS, slf_graincode_to_ICSSG_primary
from ..plugin import register_xsnowDataset_method
from .profile_block_computation import (
    apply_profile_block,
    float_result_dtype,
    get_backend,
)
from ..utils import latex_style

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
# per index: the weak side, the stability classes as ascending
# (label, upper_bound] steps in xsnow units, and the hazard-chart y-axis
# range axlim; the shared class labels are 'very poor' < 'poor' < 'fair'
# < 'rather stable' (single-threshold indices use only 'poor' and
# 'rather stable')
_SK38_CLASSES = [
    ("very poor", 0.5),
    ("poor", 0.95),
    ("fair", 1.05),
    ("rather stable", np.inf),
]
_RC_CLASSES = [
    ("very poor", 0.28),
    ("poor", 0.4),
    ("fair", 0.7),
    ("rather stable", np.inf),
]
POOR_OR_WORSE = ("very poor", "poor")
_SK38_AXLIM = (0.0, 1.5)
# the albina admin-GUI class palette (RdYlGn)
CLASS_COLORS = {
    "very poor": "#d7191c",
    "poor": "#fdae61",
    "fair": "#ffffbf",
    "rather stable": "#a6d96a",
}

_STABILITY_INDEX_INFO: dict[str, dict[str, Any]] = {
    "punstable": {
        "var": "punstable", "weak": "max",
        "classes": [
            ("rather stable", 0.5),
            ("fair", 0.7),
            ("poor", 0.77),
            ("very poor", np.inf),
        ],
        "axlim": (0.0, 1.0),
        "label": "Punstable",
    },
    "rta": {"var": "rta", "weak": "max",
            "classes": [("rather stable", 0.8), ("poor", np.inf)],
            "axlim": (0.0, 1.0)},
    "sk38": {"var": "sk38", "weak": "min", "classes": _SK38_CLASSES,
             "axlim": _SK38_AXLIM},
    "sk38_rta": {"var": "sk38_rta", "weak": "min", "classes": _SK38_CLASSES,
                 "axlim": _SK38_AXLIM},
    "sk38_ccl": {"var": "sk38_ccl", "weak": "min", "classes": _SK38_CLASSES,
                 "axlim": _SK38_AXLIM},
    "sk38_ccl_rta": {"var": "sk38_ccl_rta", "weak": "min",
                     "classes": _SK38_CLASSES, "axlim": _SK38_AXLIM},
    "rc_flat": {"var": "rc_flat", "weak": "min", "classes": _RC_CLASSES,
                "axlim": _SK38_AXLIM},
    "lwc": {"var": "lwc", "weak": "max",
            "classes": [("rather stable", 0.03), ("poor", np.inf)],
            "axlim": (0.0, 0.1)},
}


# median failure depths [m] per avalanche size 0..5
AVA_SIZE_ANCHORS = {
    'McClung2009': (0.0, 0.18, 0.35, 0.70, 1.50, 3.00),
    'Techel2023': (0.0, 0.30, 0.40, 0.50, 0.80, 1.00),
}


# ---------------------------------------------------------------------
# Stability classes and avalanche size
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


def estimate_avalanche_size(depth, kind: str = 'McClung2009'):
    """Avalanche size 0..5 interpolated from the failure depth [m]."""
    if kind not in AVA_SIZE_ANCHORS:
        raise ValueError(f"Unknown avalanche size estimator: {kind!r}")
    xp = get_backend(depth)
    return xp.interp(depth, xp.asarray(AVA_SIZE_ANCHORS[kind]), xp.arange(6.0))


# ---------------------------------------------------------------------
# Hazard chart variants
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class HCVariantInfo:
    """
    Metadata describing one hazard-chart variant stored in the dataset.
    Note: time is *not* part of the variant. Variants represent configuration
    (stability index, grouping keys, chosen depth var).
    """
    idx: int
    stability_index: str
    groupby: tuple[str, ...]   # grouping variables actually used (no 'time' added)
    depth_var: str             # depth variable used
    size_estimate: str = "McClung2009"


class HazardChart(DatasetDecorator):
    """
    Hazard-chart decorator that accumulates chart variants on a dataset.

    A *variant* is one configuration of the chart: a stability index, a
    grouping and a depth variable. ``compute_hazard_chart`` appends one entry
    per index along a new ``hc_variant`` dimension, so several indices - or the
    same index under different groupings - sit side by side in one dataset and
    keep their own ``hc_instability``/``hc_depth``/... slices;
    ``list_hc_variants`` reports what each entry holds. The base dataset is
    never mutated: every method returns a new wrapped instance.

    ``groupby`` names coordinates that do not depend on ``layer`` (e.g.
    ``"region"``, or ``["region", "band"]`` to combine them). Their values are
    joined into one label per point and stored as integer ids in
    ``hc_group_id``, with the id-to-label mapping in its ``mapping`` attribute.
    Nothing is aggregated - the ids only let ``subset_hazard_points`` and
    ``plot_hazard_chart`` pick a single group by its label.

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
        size_estimate: str = "McClung2009",
        at_time: Any | None = None,
    ) -> HazardChart:
        """
        Compute hazard-chart summary variables across time (or at a specific time
        if `at_time` is provided) and append one entry per stability index on the
        `hc_variant` dimension.

        Runs on a full-layer stratigraphy or on the reduced dataset from
        ``compute_simplified_stratigraphy``. Reduced input uses the per-index
        ``<idx>_depth`` and names the driving datetag; full-layer input uses
        ``depth_var`` and reports driving layer indices. A depth variable
        marked ``positive: 'up'`` (such as ``z``) is flipped, so ``hc_depth``
        always counts downwards from the surface.

        Parameters
        ----------
        stability_index, indices
            One index, or several (one variant each).
        groupby
            Grouping variables for ``hc_group_id``; ungrouped if omitted.
        depth_var
            Depth variable for full-layer input; flipped when it is stored
            counting up.
        datetags
            Optional subset of datetag labels; needs reduced input.
        classes
            Per-index overrides for the ``_STABILITY_INDEX_INFO`` stability
            classes (ascending ``(label, upper_bound]`` steps).
        size_estimate
            Avalanche size estimator, see ``AVA_SIZE_ANCHORS``.

        Notes
        -----
        Per index the weakest layer sets ``hc_instability``, and ``hc_depth`` is
        the depth of the deepest layer in the poor-or-worse classes, falling
        back to the weakest layer's depth. Provenance for both is stored in
        ``hc_instabilityDrivenBy`` / ``hc_depthDrivenBy`` using datetags, fallback
        are layer indices ('' where no value exists, NaN for layer indices).

        The output variables have dims (hc_variant, <points dims incl. time>).
        """
        base = self.data
        if layer_dim not in base.dims:
            raise KeyError(f"Expected layer dimension '{layer_dim}' not found in dataset.")
        if indices is not None and stability_index is not None:
            raise ValueError("Pass either 'stability_index' or 'indices', not both.")
        if indices is None:
            indices = [stability_index if stability_index is not None else "sk38"]
        cfg = {
            k: (v["weak"], (classes or {}).get(k, v["classes"]))
            for k, v in _STABILITY_INDEX_INFO.items()
        }
        unknown = [i for i in indices if i not in cfg]
        if unknown:
            raise KeyError(f"Unknown stability index/indices {unknown}. Known: {list(cfg)}")

        # --- restrict to time if requested (but keep 'time' as a dim) ---
        ds = base if at_time is None else base.sel(time=at_time)

        # the datetag entries of a reduced dataset name the drivers; a
        # full-layer dataset falls back to layer indices
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
                    "keep": keep, "size_estimate": size_estimate,
                    "flip_depth": str(ds[dvar].attrs.get("positive", "")).lower() == "up",
                    "with_gcode": with_gcode,
                },
            )
            names = ["hc_instability", "hc_depth", "hc_size_estimate",
                "hc_instabilityDrivenBy", "hc_depthDrivenBy"]
            if with_gcode:
                names.append("hc_gcode")
            attributes = {
                "hc_instability":
                    {"description": "Weakest stability index value", "index": index_var},
                "hc_depth": {
                    "description": "Depth of the deepest poor-or-worse layer",
                    "units": ds[dvar].attrs.get("units", "m"),
                    "positive": "down",
                    "depth_var": dvar,
                },
                "hc_size_estimate":
                    {"units": "1", "long_name": f"avalanche size ({size_estimate})"},
                "hc_instabilityDrivenBy":
                    {"description": "Datetag or layer index that drove hc_instability"},
                "hc_depthDrivenBy":
                    {"description": "Datetag or layer index that drove hc_depth"},
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
                size_estimate=size_estimate,
            ))
            _append_history(out, f"hazard_chart:variant={new_idx},index={index_var}")

        for name, parts in collected.items():
            out[name] = parts[0] if len(parts) == 1 else xr.concat(parts, dim="hc_variant")

        # only now: assigning hc_group_id above replaces the variable attrs
        for variant_idx, mapping in group_maps:
            out = _update_group_mapping_attr(
                out, variant_idx=variant_idx, keys=groupby_list, labels_map=mapping
            )

        return self._rewrap(out)

    def list_hc_variants(self) -> list[HCVariantInfo]:
        """Return metadata for all stored hazard-chart variants."""
        return _read_variant_info(self.data)

    def subset_hazard_points(
        self,
        *,
        hc_variant: int,
        group: int | str | None = None,
        time: Any | None = None,
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
            if "hc_group_id" not in sub:
                raise ValueError(
                    f"Variant {hc_variant} was computed without 'groupby'; "
                    "there are no groups to select."
                )
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

        keep = ["hc_instability", "hc_depth", "hc_size_estimate", "hc_gcode",
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
        size_estimate: str | None = None,
        pooled: xsnowDataset | None = None,
    ):
        """
        Scatter-plot for one variant, optionally for a single group and/or
        time. The points sit at the avalanche size their depth maps to, on
        a linear 0-5 axis like the dashboard chart; the ticks on top name
        the depths of sizes 1-5. ``size_estimate`` picks the anchor set
        (``'McClung2009'``/``'Techel2023'``), defaulting to the variant's.
        At a given ``time`` the pooled median and unstable depth are marked
        with a green diamond (``pooled`` takes a precomputed timeseries,
        e.g. a filtered one; without it the whole dataset is pooled).
        """
        import matplotlib.pyplot as plt

        meta = self.list_hc_variants()[hc_variant]
        index = meta.stability_index
        cfg = _STABILITY_INDEX_INFO.get(index, {})
        kind = size_estimate or meta.size_estimate
        if kind not in AVA_SIZE_ANCHORS:
            raise ValueError(f"Unknown avalanche size estimator: {kind!r}")

        sub = self.subset_hazard_points(hc_variant=hc_variant, group=group, time=time)
        x = _to_host(sub["hc_depth"].data).ravel()
        y = _to_host(sub["hc_instability"].data).ravel()
        # colour each point by the driving layer's grain type, GUI-style
        colors = "black"
        if "hc_gcode" in sub:
            grains = slf_graincode_to_ICSSG_primary(
                _to_host(sub["hc_gcode"].data).ravel()
            ).values
            palette = GRAIN_COLORS["SARP"]  # the dashboard scheme
            colors = [palette.get(g, "black") for g in grains]

        with plt.style.context(latex_style()):
            if ax is None:
                _, ax = plt.subplots()

            x = estimate_avalanche_size(x, kind)
            ax.set_xticks(range(6))
            ax.set_xlim(0, 5)
            ax.set_xlabel("Size estimate / -")
            # the top axis names the depths behind sizes 1-5
            ax.tick_params(top=False, which="both")
            sec = ax.secondary_xaxis("top")
            sec.set_xticks(range(1, 6),
                           labels=[f"{d:g}" for d in AVA_SIZE_ANCHORS[kind][1:]])
            sec.set_xlabel("Depth / m")
            ax.scatter(x, y, s=s, alpha=alpha, c=colors,
                       edgecolors=(0, 0, 0, 0.3), linewidths=0.3)
            ax.set_ylabel(_index_label(index))
            if "axlim" in cfg:
                ax.set_ylim(*cfg["axlim"])
            ax.grid(True)
            _draw_class_lines(ax, index)

            if time is not None:
                stats = (pooled if pooled is not None
                         else compute_hc_timeseries(xsnowDataset(self.data),
                                                    indices=[index])).data
                at = stats.sel(time=time)
                mark_x = estimate_avalanche_size(
                    float(at[f"{index}_depth"]), kind
                )
                ax.plot(float(mark_x), float(at[f"{index}_median"]),
                        "D", color="green", mec="black", mew=0.6,
                        ms=9, zorder=5)
        # ax.figure.savefig('hazard_chart.pdf', dpi=300)
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
        size_estimate: str | None = None,
        timeseries: xsnowDataset | None = None,
    ):
        """
        Two-panel figure: the hazard chart on the left (drawn by
        ``plot_hazard_chart``, square) and the pooled timeseries of the
        variant's index on the right, styled like the dashboard chart:
        median line, 25-75 % band (darker), 10-90 % band (lighter), and a
        dashed vertical mark at ``time``. The right panel keeps its y-axis
        on the right so the panels sit close together.

        The timeseries pools the whole dataset via ``compute_hc_timeseries``;
        pass ``timeseries`` to show a precomputed (e.g. filtered) one
        instead. ``group`` only subsets the scatter panel.
        """
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt

        index = self.list_hc_variants()[hc_variant].stability_index
        stats = (timeseries if timeseries is not None
                 else compute_hc_timeseries(xsnowDataset(self.data),
                                            indices=[index])).data

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
                                   size_estimate=size_estimate,
                                   pooled=xsnowDataset(stats))

            t = stats['time'].values
            ax_ts.fill_between(t, stats[f'{index}_p10'].values,
                               stats[f'{index}_p90'].values,
                               color='#ddd', label='10-90 %')
            ax_ts.fill_between(t, stats[f'{index}_p25'].values,
                               stats[f'{index}_p75'].values,
                               color='#bbb', label='25-75 %')
            ax_ts.plot(t, stats[f'{index}_median'].values, color='green',
                       label='median')
            if time is not None:
                ax_ts.axvline(self.data['time'].sel(time=time).values,
                              color='k', lw=0.8, ls='--')
            if len(t) > 1:
                ax_ts.set_xlim(t[0], t[-1])
                locator = mdates.AutoDateLocator(maxticks=6)
                ax_ts.xaxis.set_major_locator(locator)
                ax_ts.xaxis.set_major_formatter(
                    mdates.ConciseDateFormatter(locator)
                )

            # tick marks on both sides, labels only on the right
            # (sharey hides them by default)
            ax_ts.tick_params(axis='y', which='both', left=True, right=True,
                              labelleft=False, labelright=True)
            ax_ts.yaxis.set_label_position('right')
            ax_ts.set_ylabel(_index_label(index))
            cfg = _STABILITY_INDEX_INFO.get(index, {})
            if "axlim" in cfg:
                ax_ts.set_ylim(*cfg["axlim"])
            ax_ts.grid(True)
            _draw_class_lines(ax_ts, index)
            ax_ts.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0),
                         ncols=3, frameon=False, fontsize="small",
                         columnspacing=1.2, handlelength=1.6)
            for panel, label, lx in ((ax_hc, "a", 0.95), (ax_ts, "b", 0.96)):
                panel.text(lx, 0.95, label, transform=panel.transAxes,
                           ha="right", va="top", fontweight="bold",
                           bbox={"boxstyle": "circle", "facecolor": "white",
                                 "edgecolor": "black", "linewidth": 0.8})
        # ax_hc.figure.savefig('hazard_chart_timeseries.pdf', dpi=300)
        return ax_hc, ax_ts


# ---------------------------------------------------------------------
# Snowpack characteristics and pooled timeseries
# ---------------------------------------------------------------------
@register_xsnowDataset_method
def compute_hc_points(
    ds: xsnowDataset,
    indices: Sequence[str] | None = None,
    datetags: Sequence[str] | None = None,
    size_estimate: str = 'McClung2009',
    classes: dict[str, Sequence] | None = None,
    depth_var: str = 'z',
) -> xsnowDataset:
    """
    Reduce a stratigraphy to one hazard-chart point per profile.

    Runs on the reduced dataset from ``compute_simplified_stratigraphy``,
    where the entries are datetags, or on a full-layer stratigraphy.
    Per stability index this picks the weakest value across the entries
    and the depth of the deepest entry in the poor-or-worse classes
    (falling back to the weakest entry's depth), and records what drives
    each of the two.

    Parameters
    ----------
    indices
        Indices to summarize; defaults to all index variables present.
    datetags
        Optional subset of datetag labels; needs the reduced dataset.
    size_estimate
        Avalanche size estimator, see ``AVA_SIZE_ANCHORS``.
    classes
        Per-index overrides for the ``_STABILITY_INDEX_INFO`` stability
        classes (ascending ``(label, upper_bound]`` steps).
    depth_var
        Depth variable for full-layer input, counted downwards from the
        surface (flipped when stored counting up); the reduced dataset uses
        its per-index ``<idx>_depth`` instead.

    Returns
    -------
    xsnowDataset
        Layer-free dataset with, per index ``<idx>``: the weakest value
        ``<idx>``, ``<idx>_depth`` [m below the surface],
        ``<idx>_size_estimate``, ``<idx>_gcode`` (SLF code of the weakest
        layer, from ``<idx>_gcode_weak`` on reduced input and from
        ``grain_type`` on full-layer input) and
        ``<idx>_instability_drivenby`` / ``<idx>_depth_drivenby``,
        which name the driving datetag on reduced input ('' where no value
        exists) and carry layer indices on full-layer input (NaN where no
        value exists). Non-layer variables are carried over.

    Notes
    -----
    Thin adapter: the selection runs in ``compute_hazard_chart``; this
    flattens its variants to the per-index variable names.
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
    cfg = {
        k: (v['weak'], (classes or {}).get(k, v['classes']))
        for k, v in _STABILITY_INDEX_INFO.items()
    }

    def _usable(name):
        return name in xs.data and (not reduced or f'{name}_depth' in xs.data)

    if indices is None:
        indices = [v for v in cfg if _usable(v)]
    missing = [i for i in indices if i not in cfg or not _usable(i)]
    if missing:
        raise ValueError(f"No variables or class config for: {missing}")

    # the engine appends its variants after any the input already carries
    start = int(xs.data.sizes.get('hc_variant', 0))
    hc = HazardChart(xs).compute_hazard_chart(
        indices=indices, datetags=datetags, classes=classes,
        size_estimate=size_estimate, depth_var=depth_var,
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
            f'{idx}_size_estimate': ('hc_size_estimate', None),
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
    size_estimate: str = 'McClung2009',
) -> xsnowDataset:
    """
    Pooled per-timestamp statistics of stability indices.

    Runs on the reduced dataset from ``compute_simplified_stratigraphy``
    (summarized internally via ``compute_hc_points``) or on an
    already summarized dataset. Per index and timestamp this pools all
    other dimensions and derives the median, the 25/75 and 10/90
    percentiles, the median depth of the entries in the
    ``POOR_OR_WORSE`` classes, and the avalanche size estimate of that
    depth.

    Parameters
    ----------
    indices
        Indices to pool; defaults to all summarized index variables.
    datetags
        Optional subset of datetag labels to consider; needs the
        reduced dataset.
    classes
        Per-index overrides for the ``_STABILITY_INDEX_INFO`` stability
        classes (ascending ``(label, upper_bound]`` steps).
    size_estimate
        Avalanche size estimator, see ``AVA_SIZE_ANCHORS``.

    Returns
    -------
    xsnowDataset
        Time-indexed dataset with, per index ``<idx>``:
        ``<idx>_median``, ``<idx>_p25``, ``<idx>_p75``, ``<idx>_p10``,
        ``<idx>_p90``, ``<idx>_depth`` [m] and ``<idx>_size_estimate``.
    """
    xs = ds if isinstance(ds, xsnowDataset) else xsnowDataset(ds)
    if 'time' not in xs.data.dims:
        raise ValueError(
            "compute_hc_timeseries needs a 'time' dimension."
        )
    if 'layer' in xs.data.dims:
        xs = compute_hc_points(
            xs, indices=indices, datetags=datetags,
            size_estimate=size_estimate, classes=classes,
        )
    elif datetags is not None:
        raise ValueError(
            "datetag filtering needs the reduced layer dimension."
        )
    cfg = {
        k: (v['weak'], (classes or {}).get(k, v['classes']))
        for k, v in _STABILITY_INDEX_INFO.items()
    }
    if indices is None:
        indices = [
            v for v in cfg if v in xs.data and f'{v}_depth' in xs.data
        ]
    missing = [
        i for i in indices
        if i not in cfg or i not in xs.data or f'{i}_depth' not in xs.data
    ]
    if missing:
        raise ValueError(
            f"No summarized variables or class config for: {missing}"
        )

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
        out[f'{idx}_size_estimate'] = (
            ('time',), estimate_avalanche_size(depth, size_estimate),
            {'units': '1', 'long_name': f'avalanche size ({size_estimate})'},
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


def _compose_group_labels(ds: xr.Dataset, keys: Sequence[str], *, proto_like: xr.DataArray) -> xr.DataArray:
    """
    Create a string label per point for the dims of `proto_like`
    (no 'layer', may include 'time' if present; we *don't* inject 'time').
    """
    if not keys:
        return xr.full_like(proto_like, "all", dtype=object).rename("hc_group")

    proto = xr.full_like(proto_like, "", dtype=object)

    pieces: list[xr.DataArray] = []
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


def _read_variant_info(ds: xr.Dataset) -> list[HCVariantInfo]:
    import json

    key = "hc_variants"
    try:
        raw = json.loads(ds.attrs.get(key, "[]"))
    except Exception:
        return []
    out: list[HCVariantInfo] = []
    for r in raw:
        out.append(
            HCVariantInfo(
                idx=int(r["idx"]),
                stability_index=str(r["stability_index"]),
                groupby=tuple(r["groupby"]),
                depth_var=str(r["depth_var"]),
                size_estimate=str(r.get("size_estimate", "McClung2009")),
            )
        )
    return out


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
    ds: xr.Dataset, *, variant_idx: int, keys: Sequence[str], labels_map: dict[int, str]
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


def _to_host(values) -> np.ndarray:
    """Copy back to the CPU for string, pandas or numpy-only work."""
    return values.get() if hasattr(values, 'get') else np.asarray(values)


def _draw_class_lines(ax, index: str) -> None:
    """Dashed lines on the class boundaries, coloured like the more
    unstable of the two classes each boundary separates."""
    cfg = _STABILITY_INDEX_INFO.get(index, {})
    classes = cfg.get("classes", [])
    for i, (label, bound) in enumerate(classes[:-1]):
        worse = classes[i + 1][0] if cfg["weak"] == "max" else label
        ax.axhline(bound, ls="--", lw=1.2, zorder=1.5,
                   color=CLASS_COLORS.get(worse, "grey"))


def _index_label(index: str) -> str:
    """Axis label of an index: its display name over the '-' unit."""
    label = _STABILITY_INDEX_INFO.get(index, {}).get("label", index)
    return f"{label} / -"


def _argreduce_layer(values: np.ndarray, side: str):
    """NaN-safe argmax/argmin over the last axis with a validity flag."""
    xp = get_backend(values)
    fill = -np.inf if side == 'max' else np.inf
    filled = xp.where(xp.isnan(values), fill, values)
    idx = filled.argmax(-1) if side == 'max' else filled.argmin(-1)
    return idx, xp.isfinite(filled).any(-1)


def _take_layer(values: np.ndarray, idx: np.ndarray) -> np.ndarray:
    xp = get_backend(values)
    return xp.take_along_axis(values, idx[..., None], -1)[..., 0]


def _hazard_chart_block(values, depths, gcodes=None, *, side, cls_steps,
                        labels=None, keep=None, size_estimate='McClung2009',
                        flip_depth=False, with_gcode=False):
    """Reduce one block of complete profiles to the hazard-chart summary.

    The layer axis is last; every output loses it.
    """
    xp = get_backend(values)
    if keep is not None:
        values = xp.where(xp.asarray(keep), values, np.nan)
    if flip_depth:
        depths = -depths          # e.g. 'z' counts up from the surface
    weak_idx, valid, deep_idx = _select_weak_and_deep(values, side, cls_steps)
    depth = xp.where(valid, _take_layer(depths, deep_idx), np.nan)
    outputs = [
        xp.where(valid, _take_layer(values, weak_idx), np.nan),
        depth,
        xp.where(valid, estimate_avalanche_size(depth, size_estimate), np.nan),
    ]
    if labels is not None:
        host = (_to_host(valid), _to_host(weak_idx), _to_host(deep_idx))
        outputs += [np.where(host[0], labels[host[1]], ""),
                    np.where(host[0], labels[host[2]], "")]
    else:
        outputs += [xp.where(valid, weak_idx, np.nan),
                    xp.where(valid, deep_idx, np.nan)]
    if with_gcode:
        outputs.append(xp.where(valid, _take_layer(gcodes, weak_idx), np.nan))
    return tuple(outputs)


def _select_weak_and_deep(values: np.ndarray, side: str, cls_steps,
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
