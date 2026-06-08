"""
Seasonal snow-profile plot extension (interactive, Plotly).

Adds an interactive, stacked season plot to xsnowDataset. Each panel is a
``time x height`` heatmap of one parameter (grain type, LWC, hand hardness,
density, ...). Categorical variables (grain type) get a discrete, labelled
colorbar; continuous variables get a normal colorbar.

Registering
-----------
Importing this module registers ``plot_season`` as a method on xsnowDataset::

    import xsnow
    import xsnow.extensions.season_plot          # registers ds.plot_season(...)

    ds = xsnow.read("archive.zarr")
    fig = ds.plot_season(["grain_type", "lwc", "hand_hardness"],
                         station="WEISSFLUHJOCH", aspect="NE",
                         path="season.html")

Selection semantics
-------------------
* ``station`` selects the location by name. Omit it only if the dataset has a
  single location, otherwise a ValueError lists the available stations.
* ``aspect`` selects which slope of that station to plot. Accepts a compass
  point ("N", "NE", ... "NW") or degrees (e.g. 225). Azimuth/inclination are
  stored per (location, slope) in xsnow, so the aspect is resolved *within* the
  chosen station. Omit it only if the station has a single slope.
* Unknown station / aspect / parameter raise with the available options.

Dependencies
------------
Requires plotly (the ``viz`` extra: ``pip install xsnow[viz]``). plotly is
imported lazily, so importing this module without plotly still succeeds and
registers the method; the error is raised only when you call it.

Key caveats
-----------
* y-axis uses the SNOWPACK ``height`` parameter (code 0501). Layers are
  irregular and change every timestep, so each column is regridded onto a
  regular height axis with a piecewise-constant (per-layer) fill.
* ``grain_type`` is the Swiss F1F2F3 code; it is decoded to the *primary* grain
  shape (first digit) for coloring. Crust markers (F3) are not distinguished.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import xarray as xr

from ..plugin import register_xsnowDataset_method

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Lazy plotly import (optional `viz` dependency)
# --------------------------------------------------------------------------
def _import_plotly():
    """Import plotly on demand with an actionable error if it is missing."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        return go, make_subplots
    except ImportError as exc:  # pragma: no cover - dependency hint
        raise ImportError(
            "plot_season requires plotly. Install the viz extra with "
            "`pip install 'xsnow[viz]'` (or `uv pip install plotly`)."
        ) from exc


# --------------------------------------------------------------------------
# Grain-type (Swiss F1F2F3) primary-shape decoding + colors
# --------------------------------------------------------------------------
# Primary grain shape = first digit of the 3-digit code (e.g. 660 -> 6 = SH).
# Soil / non-snow codes (e.g. 81, 82) decode to 0 and are dropped.
GRAIN_SHAPE_NAMES: Dict[int, str] = {
    1: "PP",    # Precipitation particles
    2: "DF",    # Decomposing / fragmented
    3: "RG",    # Rounded grains
    4: "FC",    # Faceted crystals
    5: "DH",    # Depth hoar
    6: "SH",    # Surface hoar
    7: "MF",    # Melt forms
    8: "IF",    # Ice formations
    9: "FCxr",  # Rounding faceted particles
}

# Approximate IACS/niViz-style palette. Tweak hex values to match your house
# style; the structure (one color per primary shape) is what matters.
GRAIN_SHAPE_COLORS: Dict[int, str] = {
    1: "#00ff00",  # PP
    2: "#228b22",  # DF
    3: "#ffb6c1",  # RG
    4: "#add8e6",  # FC
    5: "#0000ff",  # DH
    6: "#ff00ff",  # SH
    7: "#ff0000",  # MF
    8: "#00ffff",  # IF
    9: "#e6a8d7",  # FCxr
}

# Sensible continuous colorscales per variable (fallbacks to Viridis).
_CONTINUOUS_COLORSCALES: Dict[str, str] = {
    "lwc": "Blues",
    "hand_hardness": "YlOrBr",
    "density": "Greys",
    "temperature": "RdBu_r",
    "grain_size": "Plasma",
}

def _to_celsius(values: np.ndarray) -> np.ndarray:
    """
    Return snow temperatures in °C.

    xsnow's `temperature` units attr says "K", but .pro data is actually stored
    in °C; some other sources are genuinely in Kelvin. Snow temperatures in K
    are ~200-300 and in °C are well below 100, so convert only when the values
    look like Kelvin.
    """
    arr = np.asarray(values, dtype="float64")
    mx = np.nanmax(arr) if np.isfinite(np.nanmax(arr)) else 0.0
    return arr - 273.15 if mx > 100 else arr


# Per-variable display transforms applied before plotting. hand_hardness is
# stored negative in .pro files; humans read it as positive index steps.
_VALUE_TRANSFORMS: Dict[str, Any] = {
    "hand_hardness": np.abs,
    "temperature": _to_celsius,
}

# Override the colorbar unit label where the dataset's `units` attr is wrong or
# unhelpful (e.g. temperature is mislabelled "K" but displayed in °C).
_DISPLAY_UNITS: Dict[str, str] = {
    "temperature": "°C",
}

_COMPASS_8 = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

DEFAULT_PARAMETERS = ("grain_type", "lwc", "hand_hardness")


# --------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------
def _get_xr(ds: Any) -> xr.Dataset:
    """Accept an xsnowDataset (has `.data`) or a plain xarray Dataset."""
    data = getattr(ds, "data", ds)
    if not isinstance(data, xr.Dataset):
        raise TypeError(f"Expected an xsnowDataset or xarray.Dataset, got {type(ds)!r}")
    return data


def _compass(azimuth_deg: float) -> str:
    """Azimuth in degrees -> 8-point compass label (aspect)."""
    return _COMPASS_8[int((azimuth_deg % 360) / 45 + 0.5) % 8]


def _aspect_to_label(aspect: Any) -> str:
    """Normalize a requested aspect (compass string or degrees) to a label."""
    if isinstance(aspect, (int, float)) and not isinstance(aspect, bool):
        return _compass(float(aspect))
    s = str(aspect).strip().upper()
    if s in _COMPASS_8:
        return s
    try:
        return _compass(float(s))
    except ValueError:
        raise ValueError(
            f"Unrecognized aspect {aspect!r}. Use a compass point "
            f"({_COMPASS_8}) or degrees."
        )


def _scalar_coord(data: xr.Dataset, name: str) -> Optional[Any]:
    """Return a scalar coord value, or None if absent/non-scalar/non-finite."""
    if name not in data.coords:
        return None
    val = data.coords[name].values
    try:
        val = val.item()
    except (ValueError, AttributeError):
        return None
    if isinstance(val, float) and not np.isfinite(val):
        return None
    return val


# --------------------------------------------------------------------------
# Station / aspect resolution (raises on missing)
# --------------------------------------------------------------------------
def _resolve_station(data: xr.Dataset, station: Optional[str]) -> xr.Dataset:
    """Select a single station by name, or raise listing what's available."""
    if "location" not in data.dims:
        return data  # already a single (scalar) location
    locs = [str(v) for v in np.atleast_1d(data["location"].values)]
    if station is None:
        if len(locs) == 1:
            return data.sel(location=locs[0])
        raise ValueError(
            f"Dataset has {len(locs)} stations; pass station=. Available: {locs}"
        )
    if station not in locs:
        raise ValueError(
            f"Station {station!r} not in dataset. Available stations: {locs}"
        )
    return data.sel(location=station)


def _aspect_options(data: xr.Dataset) -> list:
    """List (slope_position, azimuth, label) for the (already station-selected) data."""
    n = data.sizes.get("slope", 1)
    az = (
        np.atleast_1d(data["azimuth"].values).astype("float64")
        if "azimuth" in data.coords else np.full(n, np.nan)
    )
    incl = (
        np.atleast_1d(data["inclination"].values).astype("float64")
        if "inclination" in data.coords else np.full(n, np.nan)
    )
    opts = []
    for i in range(n):
        a = az[i] if i < az.size else np.nan
        slope_angle = incl[i] if i < incl.size else np.nan
        # A near-flat field has no meaningful aspect; label it "flat" so it is
        # distinguishable from a real N-facing slope (both have azimuth ~0).
        if np.isfinite(slope_angle) and slope_angle < 1.0:
            label = "flat"
        elif np.isfinite(a):
            label = _compass(a)
        else:
            label = "flat/NA"
        opts.append((i, a, label))
    return opts


def _describe_aspects(opts: list) -> str:
    return ", ".join(
        f"{lbl} ({a:.0f}°)" if np.isfinite(a) and lbl not in ("flat", "flat/NA") else lbl
        for _, a, lbl in opts
    )


def _resolve_aspect(data: xr.Dataset, aspect: Any) -> xr.Dataset:
    """Select a single slope by aspect, or raise listing what's available."""
    if "slope" not in data.dims:
        if aspect is not None:
            raise ValueError("Dataset has no 'slope' dimension; cannot select by aspect.")
        return data

    opts = _aspect_options(data)
    avail = _describe_aspects(opts)

    if aspect is None:
        if data.sizes["slope"] == 1:
            return data.isel(slope=0)
        raise ValueError(
            f"Station has {data.sizes['slope']} slopes; pass aspect=. "
            f"Available aspects: {avail}"
        )

    # Prefer an exact degree match when aspect is numeric.
    if isinstance(aspect, (int, float)) and not isinstance(aspect, bool):
        target = float(aspect) % 360
        exact = [
            i for i, a, _ in opts
            if np.isfinite(a) and abs((a - target + 180) % 360 - 180) <= 1.0
        ]
        if exact:
            return data.isel(slope=exact[0])

    req_label = _aspect_to_label(aspect)
    matches = [i for i, _, lbl in opts if lbl == req_label]
    if not matches:
        raise ValueError(
            f"Aspect {aspect!r} not found for this station. Available aspects: {avail}"
        )
    if len(matches) > 1:
        logger.warning("Multiple slopes match aspect %r; using the first.", aspect)
    return data.isel(slope=matches[0])


def _reduce_extra_dims(data: xr.Dataset, time_dim: str, layer_dim: str) -> xr.Dataset:
    """Squeeze size-1 dims (keeping metadata as scalar coords) and collapse leftovers."""
    data = data.squeeze(drop=False)
    for d in [d for d in data.dims if d not in (time_dim, layer_dim)]:
        logger.warning(
            "Multiple %s present (size %d); using the first.", d, data.sizes[d]
        )
        data = data.isel({d: 0}, drop=False)
    return data


def _select_profile(
    data: xr.Dataset,
    *,
    station: Optional[str],
    aspect: Any,
    time_dim: str = "time",
    layer_dim: str = "layer",
) -> xr.Dataset:
    """Reduce to a single (time, layer) profile for one station + aspect."""
    data = _resolve_station(data, station)
    data = _resolve_aspect(data, aspect)
    data = _reduce_extra_dims(data, time_dim, layer_dim)
    return data


# --------------------------------------------------------------------------
# Plotting internals
# --------------------------------------------------------------------------
def _is_categorical(da: xr.DataArray, var: str) -> bool:
    """xsnow stores categoricals with a `mapping` attr; grain_type is special."""
    return bool(da.attrs.get("mapping")) or var == "grain_type"


def _decode_grain_type(values: np.ndarray) -> Tuple[np.ndarray, Dict[int, str]]:
    """Map raw Swiss F1F2F3 codes -> primary grain shape id (1-9)."""
    arr = np.asarray(values, dtype="float64")
    primary = np.full(arr.shape, np.nan)
    valid = ~np.isnan(arr)
    primary[valid] = (arr[valid] // 100).astype(int)
    known = np.isin(primary, list(GRAIN_SHAPE_NAMES))
    primary[~known] = np.nan
    present = sorted(int(v) for v in np.unique(primary[~np.isnan(primary)]))
    mapping = {sid: GRAIN_SHAPE_NAMES[sid] for sid in present}
    return primary, mapping


def _regrid_to_height(
    values: np.ndarray, heights: np.ndarray, height_grid: np.ndarray
) -> np.ndarray:
    """
    Piecewise-constant regrid of (time, layer) data onto a regular height axis.

    Layer i occupies (h[i-1], h[i]] where h is cumulative element-top height
    from the ground (SNOWPACK 0501). A grid cell at height g takes the value of
    the lowest layer whose top >= g; cells above the surface become NaN.
    Returns shape (n_height, n_time) for go.Heatmap(z=...).
    """
    n_time = values.shape[0]
    out = np.full((height_grid.size, n_time), np.nan)
    for t in range(n_time):
        h = heights[t]
        v = values[t]
        mask = ~np.isnan(h)
        if not mask.any():
            continue
        h_valid, v_valid = h[mask], v[mask]
        order = np.argsort(h_valid)
        h_valid, v_valid = h_valid[order], v_valid[order]
        surface = h_valid[-1]
        idx = np.searchsorted(h_valid, height_grid, side="left")
        in_pack = (height_grid <= surface) & (idx < h_valid.size)
        out[in_pack, t] = v_valid[idx[in_pack]]
    return out


def _discrete_colorscale(colors: Sequence[str]) -> list:
    """Build a plotly step colorscale (one flat band per category)."""
    k = len(colors)
    scale = []
    for i, c in enumerate(colors):
        scale.append([i / k, c])
        scale.append([(i + 1) / k, c])
    return scale


def _align_colorbars(fig) -> None:
    """Shrink/center each trace's colorbar onto its own subplot row."""
    for tr in fig.data:
        if not hasattr(tr, "colorbar"):
            continue
        yref = tr.yaxis or "y"
        domain = getattr(fig.layout["yaxis" + yref[1:]], "domain", None)
        if not domain:
            continue
        bottom, top = float(domain[0]), float(domain[1])
        tr.colorbar.update(len=top - bottom, y=(top + bottom) / 2,
                           yanchor="middle", thickness=14)


def _profile_title(data: xr.Dataset, *, fallback_location: Optional[str] = None) -> str:
    """Station name on the main line; geo + aspect/inclination as a subtitle."""
    name = _scalar_coord(data, "location") or fallback_location or "unknown station"
    bits = []
    lat, lon = _scalar_coord(data, "latitude"), _scalar_coord(data, "longitude")
    if lat is not None and lon is not None:
        bits.append(f"{lat:.4f}°, {lon:.4f}°")
    alt = _scalar_coord(data, "altitude")
    if alt is not None:
        bits.append(f"{alt:.0f} m")
    azi = _scalar_coord(data, "azimuth")
    if azi is not None:
        bits.append(f"aspect {_compass(azi)} ({azi:.0f}°)")
    incl = _scalar_coord(data, "inclination")
    if incl is not None:
        bits.append(f"{incl:.0f}° slope")
    title = f"Seasonal snow profile — {name}"
    if bits:
        title += "<br><sup>" + "  ·  ".join(bits) + "</sup>"
    return title


# --------------------------------------------------------------------------
# Public: single panel (operates on an already-resolved profile)
# --------------------------------------------------------------------------
def add_season_panel(
    ds: Any,
    var: str,
    *,
    fig=None,
    row: int = 1,
    station: Optional[str] = None,
    aspect: Any = None,
    height_var: str = "height",
    layer_dim: str = "layer",
    time_dim: str = "time",
    n_height: int = 200,
    height_grid: Optional[np.ndarray] = None,
    colorscale: Optional[str] = None,
    showscale: bool = True,
):
    """
    Draw one parameter's season heatmap (time x height) into (fig, row).

    Accepts an xsnowDataset, a plain xarray Dataset, or an already-resolved
    single-profile dataset (station/aspect default to None and are idempotent
    on a reduced profile). Returns the plotly Figure.
    """
    go, make_subplots = _import_plotly()

    data = _select_profile(
        _get_xr(ds), station=station, aspect=aspect,
        time_dim=time_dim, layer_dim=layer_dim,
    )
    if var not in data:
        raise KeyError(
            f"Variable {var!r} not in dataset. Available: {sorted(data.data_vars)}"
        )
    if height_var not in data:
        raise KeyError(f"Height variable {height_var!r} not in dataset.")

    da = data[var].transpose(time_dim, layer_dim)
    h = data[height_var].transpose(time_dim, layer_dim)
    times = data[time_dim].values
    values = da.values
    heights = h.values

    if height_grid is None:
        hmax = np.nanmax(heights)
        hmax = float(hmax) if np.isfinite(hmax) else 1.0
        height_grid = np.linspace(0.0, hmax, n_height)

    if fig is None:
        fig = make_subplots(rows=1, cols=1)

    if _is_categorical(da, var):
        if var == "grain_type":
            decoded, mapping = _decode_grain_type(values)
        else:
            mapping = {int(k): str(v) for k, v in da.attrs["mapping"].items()}
            decoded = values.astype("float64")

        cats = sorted(mapping)
        cat_to_idx = {c: i for i, c in enumerate(cats)}
        labels = [mapping[c] for c in cats]
        if var == "grain_type":
            colors = [GRAIN_SHAPE_COLORS.get(c, "#999999") for c in cats]
        else:
            import plotly.express as px
            palette = px.colors.qualitative.Safe
            colors = [palette[i % len(palette)] for i in range(len(cats))]

        idx_values = np.full(decoded.shape, np.nan)
        for c, i in cat_to_idx.items():
            idx_values[decoded == c] = i
        z = _regrid_to_height(idx_values, heights, height_grid)

        text = np.empty_like(z, dtype=object)
        for c, i in cat_to_idx.items():
            text[z == i] = mapping[c]

        fig.add_trace(
            go.Heatmap(
                x=times, y=height_grid, z=z,
                colorscale=_discrete_colorscale(colors),
                zmin=-0.5, zmax=len(cats) - 0.5,
                colorbar=dict(title=var, tickmode="array",
                              tickvals=list(range(len(cats))), ticktext=labels),
                showscale=showscale, text=text,
                hovertemplate="%{x}<br>height=%{y:.2f}<br>" + var + "=%{text}<extra></extra>",
            ),
            row=row, col=1,
        )
    else:
        vals = values.astype("float64")
        transform = _VALUE_TRANSFORMS.get(var)
        if transform is not None:
            vals = transform(vals)
        z = _regrid_to_height(vals, heights, height_grid)
        units = _DISPLAY_UNITS.get(var, da.attrs.get("units", ""))
        title = f"{var} [{units}]" if units else var
        fig.add_trace(
            go.Heatmap(
                x=times, y=height_grid, z=z,
                colorscale=colorscale or _CONTINUOUS_COLORSCALES.get(var, "Viridis"),
                colorbar=dict(title=title), showscale=showscale,
                hovertemplate="%{x}<br>height=%{y:.2f}<br>" + var + "=%{z:.3f}<extra></extra>",
            ),
            row=row, col=1,
        )

    fig.update_yaxes(
        title_text=f"{height_var} [{data[height_var].attrs.get('units', '')}]",
        row=row, col=1,
    )
    return fig


# --------------------------------------------------------------------------
# Public: stacked season figure (registered as ds.plot_season)
# --------------------------------------------------------------------------
@register_xsnowDataset_method
def plot_season(
    self,
    parameters: Sequence[str] = DEFAULT_PARAMETERS,
    *,
    station: Optional[str] = None,
    aspect: Any = None,
    height_var: str = "height",
    n_height: int = 200,
    path: Optional[str] = None,
    row_height: int = 300,
    open_browser: bool = False,
):
    """
    Interactive stacked season plot, one panel per parameter.

    Parameters
    ----------
    parameters : sequence of str
        Variables to stack, top to bottom (e.g. ["grain_type", "lwc",
        "hand_hardness"]). Categorical vs continuous is auto-detected.
    station : str, optional
        Location/station name. Required if the dataset has >1 location.
    aspect : str or float, optional
        Slope aspect to plot: a compass point ("N".."NW") or degrees. Required
        if the chosen station has >1 slope.
    height_var : str
        Variable used for the y-axis (default "height", SNOWPACK 0501).
    n_height : int
        Resolution of the regular height axis.
    path : str, optional
        If given, also write a standalone HTML file (with a metadata <title>).
    row_height : int
        Pixel height per panel.
    open_browser : bool
        If True and `path` is given, open the written HTML in a browser.

    Returns
    -------
    plotly.graph_objects.Figure

    Raises
    ------
    ValueError
        If `station`/`aspect` is unknown or ambiguous (the message lists the
        available options).
    KeyError
        If a requested parameter (or `height_var`) is not in the dataset.
    """
    go, make_subplots = _import_plotly()

    parameters = list(parameters)
    if not parameters:
        raise ValueError("`parameters` must contain at least one variable.")

    # Resolve the single profile once; panels reuse it (warns once, one profile).
    data = _select_profile(_get_xr(self), station=station, aspect=aspect)

    missing = [p for p in parameters if p not in data.data_vars]
    if missing:
        raise KeyError(
            f"Parameter(s) {missing} not in dataset. "
            f"Available: {sorted(data.data_vars)}"
        )
    if height_var not in data:
        raise KeyError(f"Height variable {height_var!r} not in dataset.")

    fig = make_subplots(
        rows=len(parameters), cols=1, shared_xaxes=True,
        vertical_spacing=0.06, subplot_titles=parameters,
    )
    for i, var in enumerate(parameters, start=1):
        add_season_panel(data, var, fig=fig, row=i,
                         height_var=height_var, n_height=n_height)

    fig.update_layout(
        height=row_height * len(parameters),
        title=_profile_title(data, fallback_location=station),
        margin=dict(t=110),
    )
    fig.update_xaxes(title_text="time", row=len(parameters), col=1)
    _align_colorbars(fig)

    if path:
        page_title = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", fig.layout.title.text)).strip()
        html = fig.to_html(full_html=True, include_plotlyjs=True)
        html = html.replace("<head>", f"<head><title>{page_title}</title>", 1)
        Path(path).write_text(html, encoding="utf-8")
        logger.info("Wrote %s", path)
        if open_browser:
            import webbrowser
            webbrowser.open(Path(path).resolve().as_uri())
    return fig


# --------------------------------------------------------------------------
# CLI: python -m xsnow.extensions.season_plot SOURCE [...]
# --------------------------------------------------------------------------
def _main(argv: Optional[Sequence[str]] = None) -> None:
    import argparse
    p = argparse.ArgumentParser(description="Interactive seasonal snow-profile plot.")
    p.add_argument("source", help="Path readable by xsnow.read (dir, .zarr, .nc).")
    p.add_argument("--station", default=None, help="Station/location name.")
    p.add_argument("--aspect", default=None, help='Aspect: compass ("NE") or degrees.')
    p.add_argument("--params", nargs="+", default=list(DEFAULT_PARAMETERS),
                   help="Parameters to stack (top to bottom).")
    p.add_argument("--out", default="season.html", help="Output HTML path.")
    p.add_argument("--n-height", type=int, default=200, help="Height-grid resolution.")
    p.add_argument("--open", action="store_true", help="Open the HTML when done.")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    import xsnow
    ds = xsnow.read(args.source)
    if ds is None:
        raise SystemExit(f"xsnow.read returned None for {args.source!r}")

    ds.plot_season(
        args.params, station=args.station, aspect=args.aspect,
        n_height=args.n_height, path=args.out, open_browser=args.open,
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    _main()
